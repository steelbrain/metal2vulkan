//! Merges' kept images (the runtime's `texture/kept_image.rs`), on the device the cfg selects,
//! under the validation layer. On a device whose merged-back shadows keep their texels in an image
//! (one that takes the copy back's dispatch, `DriverFact::ShadowCopyBackDispatchFaster`, and not
//! the copy in's, `DriverFact::ShadowCopyInDispatchFaster`), a queue's keeps take the images its
//! earlier keeps made once the encoder that took one ended, whatever the texture, without waiting
//! for the command buffers that used it:
//!
//! - passes into one texture over a `Shared` buffer in command buffers committed together, more
//!   than the two images a shadow kept before, make one image, the later passes taking it ahead of
//!   the earlier command buffers' completion (`WorkCounts::shadow_merge_kept_images_followed`),
//!   under either completion policy (a group of held batches keeps its queue's order);
//! - a texture made each frame over the same bytes takes the last frame's image;
//! - an image the queue lets go of while a command buffer waiting for an event still uses it goes
//!   only once that command buffer completed (`WorkCounts::shadow_merge_kept_images_live`), and
//!   the next pass makes another.
//!
//! The buffer's bytes hold each pass's blend inside its scissor, as a private texture blended
//! alike does, and every other byte as it was. Elsewhere no merge keeps its texels in an image,
//! and the bytes are the same.

use super::no_copy_textures::{blending, floats, staged};
use super::{
    layer_reported_nothing_at_all, validated_device, validated_device_with, Messages, ScratchCache,
    WAIT,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{CompletionPolicy, DriverFact};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, Device, EventRef, LoadAction, Origin,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};

/// A tile of `BGRA8Unorm` texels: its width, height and bytes per row (padded to 64 bytes).
#[derive(Clone, Copy)]
struct Tile {
    w: u64,
    h: u64,
    bpr: u64,
}

/// The tile the passes blend into, and a smaller one of another extent, both of padded rows: a
/// shadow on every device.
const TILE: Tile = Tile {
    w: 300,
    h: 200,
    bpr: 1216,
};
const SMALL: Tile = Tile {
    w: 30,
    h: 32,
    bpr: 128,
};

/// The command buffers of a round, committed together, each with one pass.
const COMMAND_BUFFERS: usize = 6;
const ROUNDS: usize = 3;

/// The passes into the small tile while the first command buffer waits: more than the 256 takes
/// the queue keeps an image no encoder holds for. Each is a pass alone, which takes a kept image:
/// an empty blit encoder after it ends its run (`pass_runs.rs`), whose later passes would take
/// none.
const OTHER_PASSES: usize = 300;

impl Tile {
    fn len(self) -> u64 {
        self.bpr * self.h
    }

    fn row(self) -> u64 {
        self.w * 4
    }
}

/// What the passes draw with: the blend pipeline, the rectangle's corners and the color.
struct Drawing {
    blend: RenderPipelineState,
    corners: Buffer,
    color: Buffer,
}

impl Drawing {
    fn new(device: &Device) -> Drawing {
        Drawing {
            blend: blending(device, PixelFormat::BGRA8Unorm),
            corners: staged(device, &floats([-1.0, -1.0, 1.0, 1.0])),
            color: staged(device, &floats([0.25, 0.5, 0.75, 0.5])),
        }
    }

    /// A pass of `command_buffer` that loads `target` and blends the color inside `scissor`.
    fn pass(&self, command_buffer: &CommandBuffer, target: &Texture, scissor: ScissorRect) {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.blend).unwrap();
        encoder.set_scissor_rect(scissor).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.corners), 0, 0)
            .unwrap();
        encoder
            .set_fragment_buffer(Some(&self.color), 0, 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
    }
}

/// The scissor of the `k`th pass: rectangles that overlap, so each pass blends over the last.
fn scissor(k: usize) -> ScissorRect {
    let k = k as u64 % 5;
    ScissorRect::new(20 + 30 * k, 10 + 20 * k, 150, 100)
}

/// A `Shared` buffer of `tile` holding a pattern seeded by `seed`, with its bytes.
fn shared(device: &Device, tile: Tile, seed: u8) -> (Buffer, Vec<u8>) {
    let len = tile.len();
    let buffer = device.new_buffer(len, ResourceOptions::shared()).unwrap();
    let bytes: Vec<u8> = (0..len as usize)
        .map(|i| (i as u8).wrapping_mul(29).wrapping_add(seed))
        .collect();
    let contents = buffer.contents().unwrap();
    // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    (buffer, bytes)
}

/// A render target texture over `buffer`'s bytes of `tile`: a shadow, merged back.
fn over(buffer: &Buffer, tile: Tile) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, tile.w, tile.h, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&d, 0, tile.bpr).unwrap();
    // SAFETY: the handle is only compared, while the texture lives.
    assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
    texture
}

/// The texel rows of `bytes`, laid out at `tile`'s rows.
fn rows_of(tile: Tile, bytes: &[u8]) -> Vec<u8> {
    (0..tile.h)
        .flat_map(|y| {
            let at = (y * tile.bpr) as usize;
            bytes[at..at + tile.row() as usize].to_vec()
        })
        .collect()
}

/// A private texture of `tile` holding `rows` (tight), the reference the passes blend alike.
fn private_with(queue: &CommandQueue, tile: Tile, rows: &[u8]) -> Texture {
    let device = queue.device();
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, tile.w, tile.h, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = device.new_texture(&d).unwrap();
    let source = staged(device, rows);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        tile.row(),
        rows.len() as u64,
        Size::new(tile.w, tile.h, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    texture
}

/// `texture`'s texels (tight rows).
fn blit_out(queue: &CommandQueue, tile: Tile, texture: &Texture) -> Vec<u8> {
    let len = tile.row() * tile.h;
    let out = queue
        .device()
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(tile.w, tile.h, 1),
        &out,
        0,
        tile.row(),
        len,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    bytes_of(&out)
}

fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Commit `command_buffer` and wait for it, which must complete without an error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// Wait for each of `command_buffers`, which must complete without an error.
fn wait_all(command_buffers: &[CommandBuffer]) {
    for command_buffer in command_buffers {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
}

/// Whether `device`'s merged-back shadows keep their texels in an image (the module docs).
fn keeps_in_images(device: &Device) -> bool {
    let facts = &device.properties().driver_facts;
    facts.contains(&DriverFact::ShadowCopyBackDispatchFaster)
        && !facts.contains(&DriverFact::ShadowCopyInDispatchFaster)
}

/// The kept-image counts: merges kept in images, images made, taken ahead of a holder's
/// completion, and live.
fn kept(counts: &WorkCounts) -> [u64; 4] {
    [
        counts.shadow_merges_kept_in_images,
        counts.shadow_merge_kept_images_made,
        counts.shadow_merge_kept_images_followed,
        counts.shadow_merge_kept_images_live,
    ]
}

/// The buffer's bytes are the reference's texels in their rows, and its padding as it was.
fn assert_bytes(queue: &CommandQueue, buffer: &Buffer, reference: &Texture, before: &[u8]) {
    let now = bytes_of(buffer);
    let expected_rows = blit_out(queue, TILE, reference);
    assert!(
        rows_of(TILE, &now) == expected_rows,
        "the texels differ from the private texture's blend"
    );
    for y in 0..TILE.h {
        let (from, to) = (
            (y * TILE.bpr + TILE.row()) as usize,
            ((y + 1) * TILE.bpr) as usize,
        );
        assert!(
            now[from..to] == before[from..to],
            "row {y}'s padding changed"
        );
    }
}

/// Run `test` on a validated device of the lane under the completion policy `completion`, then
/// fail on any validation message.
fn on_device(completion: Option<CompletionPolicy>, test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = match completion {
            Some(completion) => {
                validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                    options.policy.completion = completion;
                })
            }
            None => validated_device(TEST_DEVICE, &messages, cache.path()),
        }
        .unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The module docs' first case: [`COMMAND_BUFFERS`] passes into one texture a round, each in a
/// command buffer of its own, committed together, under each completion policy.
#[test]
fn passes_in_command_buffers_committed_together_share_one_kept_image() {
    for completion in [CompletionPolicy::Exact, CompletionPolicy::Grouped] {
        on_device(Some(completion), |device| {
            let queue = device.new_command_queue().unwrap();
            let drawing = Drawing::new(device);
            let (buffer, before) = shared(device, TILE, 7);
            let texture = over(&buffer, TILE);
            let reference = private_with(&queue, TILE, &rows_of(TILE, &before));
            let images = keeps_in_images(device);
            for round in 0..ROUNDS {
                let start = kept(&device.work_counts());
                let command_buffers: Vec<CommandBuffer> = (0..COMMAND_BUFFERS)
                    .map(|k| {
                        let command_buffer = queue.command_buffer().unwrap();
                        drawing.pass(&command_buffer, &texture, scissor(k));
                        command_buffer
                    })
                    .collect();
                for command_buffer in &command_buffers {
                    command_buffer.commit().unwrap();
                }
                wait_all(&command_buffers);
                drop(command_buffers);
                for k in 0..COMMAND_BUFFERS {
                    let command_buffer = queue.command_buffer().unwrap();
                    drawing.pass(&command_buffer, &reference, scissor(k));
                    run(&command_buffer);
                }
                let now = kept(&device.work_counts());
                let round_counts: [u64; 3] = [0, 1, 2].map(|i| now[i] - start[i]);
                println!(
                    "KEPT-IMAGES on {} ({completion:?}): round {round}, {} merges kept in images, \
                     {} images made, {} taken ahead of a holder's completion, {} live",
                    device.name(),
                    round_counts[0],
                    round_counts[1],
                    round_counts[2],
                    now[3],
                );
                let n = COMMAND_BUFFERS as u64;
                let expected = match (images, round) {
                    (false, _) => [0, 0, 0],
                    (true, 0) => [n, 1, n - 1],
                    (true, _) => [n, 0, n - 1],
                };
                assert_eq!(round_counts, expected, "round {round}");
                assert_eq!(now[3], u64::from(images), "round {round}'s live images");
            }
            assert_bytes(&queue, &buffer, &reference, &before);
        });
    }
}

/// The module docs' second case: a texture made each frame over the same bytes.
#[test]
fn a_texture_made_each_frame_takes_the_last_frame_s_kept_image() {
    on_device(None, |device| {
        let queue = device.new_command_queue().unwrap();
        let drawing = Drawing::new(device);
        let (buffer, before) = shared(device, TILE, 11);
        let reference = private_with(&queue, TILE, &rows_of(TILE, &before));
        let start = kept(&device.work_counts());
        const FRAMES: usize = 4;
        for frame in 0..FRAMES {
            let texture = over(&buffer, TILE);
            let command_buffer = queue.command_buffer().unwrap();
            drawing.pass(&command_buffer, &texture, scissor(frame));
            run(&command_buffer);
            let command_buffer = queue.command_buffer().unwrap();
            drawing.pass(&command_buffer, &reference, scissor(frame));
            run(&command_buffer);
        }
        let now = kept(&device.work_counts());
        let counts: [u64; 3] = [0, 1, 2].map(|i| now[i] - start[i]);
        println!(
            "KEPT-IMAGES on {}: {FRAMES} frames' textures, {} merges kept in images, {} images \
             made, {} live",
            device.name(),
            counts[0],
            counts[1],
            now[3],
        );
        let images = keeps_in_images(device);
        let expected = match images {
            true => [FRAMES as u64, 1, 0],
            false => [0, 0, 0],
        };
        assert_eq!(counts, expected);
        assert_eq!(now[3], u64::from(images), "live images");
        assert_bytes(&queue, &buffer, &reference, &before);
    });
}

/// The module docs' third case: the first command buffer waits for an event while
/// [`OTHER_PASSES`] passes alone into a tile of another extent take the queue's images past its
/// first one's last take; that image goes once the first command buffer completed, and the next
/// pass into its texture makes one.
#[test]
fn a_kept_image_let_go_of_while_its_command_buffer_waits_goes_after_it() {
    on_device(None, |device| {
        let queue = device.new_command_queue().unwrap();
        let drawing = Drawing::new(device);
        let (buffer, before) = shared(device, TILE, 13);
        let texture = over(&buffer, TILE);
        let reference = private_with(&queue, TILE, &rows_of(TILE, &before));
        let (small_buffer, _) = shared(device, SMALL, 17);
        let small = over(&small_buffer, SMALL);
        let images = keeps_in_images(device);
        let start = kept(&device.work_counts());
        let gate = device.new_shared_event().unwrap();
        let waiting = queue.command_buffer().unwrap();
        waiting
            .encode_wait_for_event(EventRef::from(&gate), 1)
            .unwrap();
        drawing.pass(&waiting, &texture, scissor(0));
        waiting.commit().unwrap();
        let others = queue.command_buffer().unwrap();
        for k in 0..OTHER_PASSES {
            drawing.pass(&others, &small, ScissorRect::new(k as u64 % 14, 0, 16, 32));
            others
                .blit_command_encoder()
                .unwrap()
                .end_encoding()
                .unwrap();
        }
        others.commit().unwrap();
        // The tile's image left the queue's pool, but the waiting command buffer holds it.
        let held = kept(&device.work_counts());
        gate.set_signaled_value(1);
        wait_all(&[waiting, others]);
        let done = kept(&device.work_counts());
        let command_buffer = queue.command_buffer().unwrap();
        drawing.pass(&command_buffer, &texture, scissor(1));
        run(&command_buffer);
        let again = kept(&device.work_counts());
        for k in 0..2 {
            let command_buffer = queue.command_buffer().unwrap();
            drawing.pass(&command_buffer, &reference, scissor(k));
            run(&command_buffer);
        }
        println!(
            "KEPT-IMAGES on {}: while the first command buffer waited, {} images made and {} live; \
             {} live once it completed; the next pass made {}",
            device.name(),
            held[1] - start[1],
            held[3],
            done[3],
            again[1] - done[1],
        );
        if images {
            assert_eq!(held[1] - start[1], 2, "one image of each extent made");
            assert_eq!(
                held[3], 2,
                "the tile's image held by the waiting command buffer"
            );
            assert_eq!(done[3], 1, "the tile's image gone with it");
            assert_eq!(
                again[1] - done[1],
                1,
                "the next pass into the tile makes one"
            );
        } else {
            assert_eq!([held[1], held[3], again[1]], [start[1], 0, start[1]]);
        }
        assert_bytes(&queue, &buffer, &reference, &before);
    });
}
