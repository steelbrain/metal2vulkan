//! Device tests of a draw that reads, at its own pixels, a texture its own render pass instance
//! renders as color 0, and writes nothing of that color: it writes what it read into color 1
//! alone (`ipr_read_second` of the owned fixture `ipr_reads.metal`), under the validation layer.
//! Where the device enabled `attachmentFeedbackLoopLayout` and the color's image took feedback
//! loop usage, the draw reads the attachment in place, with a memory barrier before it or not
//! (`render/snapshot.rs`, in place; `WorkCounts::reading_draws_in_place`): no copy, snapshot or
//! split, and it reads what the draw before it wrote, the x86 Macs' answer (`docs/API.md`'s
//! accepted differences). With the feature withheld it reads the copy taken before the pass with
//! no barrier, the texels from before the pass (the M3 Max's answer), and after one what the draw
//! before the barrier wrote.
//!
//! The pass: color 0 the target, a [`SIDE`]-square `RGBA8Unorm` texture that an earlier pass
//! clears to red [`CLEARED`], loaded and stored; color 1 a private texture of its size, cleared to
//! 0 and stored. Draw F fills the block with red [`FILL`] (into color 0 alone); then the reader
//! over the block, a barrier over the render targets before it or not; then draw L fills the block
//! with red [`LATE`], after the reader's reads (the barrier after a draw in place).
//!
//! - [`a_reader_that_writes_nothing_of_its_color_reads_in_place`]: a private target, and a surface
//!   over host pages (a texture of an IOSurface plane over a no-copy buffer, as a compositor's
//!   guest makes its layers), each with the barrier and without: in place wherever the device
//!   enabled the feature and the image the pass renders took feedback loop usage (the counts
//!   printed say).
//! - [`with_the_feedback_loop_withheld_it_reads_the_copy`]: the private target with
//!   `attachmentFeedbackLoopLayout` withheld: the copy, [`CLEARED`] without the barrier.

use super::attachment_reads::{floats, function};
use super::degrade_notices::on_device_with;
use super::host_memory::{no_copy, Mapping, GUEST_PAGE};
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Device, DeviceOptions, IoSurfacePlane,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The target's side, in pixels.
const SIDE: u64 = 64;

/// The bytes of a row of the target, tight.
const ROW: u64 = SIDE * 4;

/// The target's red before the pass.
const CLEARED: u8 = 64;

/// The red draw F writes over the block.
const FILL: u8 = 40;

/// The red draw L writes over the block, after the reader.
const LATE: u8 = 200;

/// The block F, the reader and L draw over.
const BLOCK: [u64; 4] = [8, 8, 24, 24];

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    floats([x(x0), y(y1), x(x1), y(y0)])
}

/// The `tob_rect` state with the fragment function `name` of `fixture`, into two unblended
/// `RGBA8Unorm` colors: a function with no output at a color writes nothing of it.
fn state(device: &Device, name: &str, fixture: &str) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
    d.fragment = Some(function(device, name, fixture));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// What the pass moved of the counts: reads in place, through snapshots, splits for attachment
/// reads, and reads through copies.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    in_place: u64,
    snapshotted: u64,
    splits: u64,
    copies: u64,
}

fn moved(before: &WorkCounts, after: &WorkCounts) -> Moved {
    Moved {
        in_place: after.reading_draws_in_place - before.reading_draws_in_place,
        snapshotted: after.reading_draws_snapshotted - before.reading_draws_snapshotted,
        splits: after.attachment_read_splits - before.attachment_read_splits,
        copies: after.attachment_reads_through_copies - before.attachment_reads_through_copies,
    }
}

/// The bytes of `buffer`, which no command uses now.
fn bytes(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// What the pass of the module docs left: color 1's bytes and the target's, tight rows, what the
/// pass moved of the counts, and whether the image the pass renders took feedback loop usage
/// (for a surface, `WorkCounts::feedback_loop_alias_images` or `feedback_loop_shadow_images`; a
/// private target's does wherever the device enabled the feature).
#[derive(Debug)]
struct Left {
    second: Vec<u8>,
    target: Vec<u8>,
    moved: Moved,
    feedback: bool,
}

/// The pass of the module docs over a new target, a surface over host pages where `surface`,
/// with the barrier before the reader where `barrier`.
fn read_pass(device: &Device, queue: &CommandQueue, surface: bool, barrier: bool) -> Left {
    let fill = state(device, "tob_solid", "tob_renders.ll");
    let read = state(device, "ipr_read_second", "ipr_reads.ll");
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let second = device.new_texture(&d).unwrap();
    let len = ((SIDE * ROW) as usize).next_multiple_of(GUEST_PAGE);
    let pages = Mapping::anonymous(len);
    let made = device.work_counts();
    let (target, _held): (Texture, Option<Buffer>) = match surface {
        false => (device.new_texture(&d).unwrap(), None),
        true => {
            // SAFETY: the pass completes, and the texture and the buffer go, before the pages.
            let buffer = unsafe {
                no_copy(
                    device,
                    pages.pointer(),
                    len,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap();
            d.resource_options = buffer.resource_options();
            let plane = IoSurfacePlane::new(0, ROW, false);
            let texture = buffer.new_texture_with_iosurface_plane(&d, plane).unwrap();
            (texture, Some(buffer))
        }
    };
    let counts = device.work_counts();
    let feedback = match surface {
        false => (device.enabled().features).contains(&VkFeature::AttachmentFeedbackLoopLayout),
        true => {
            counts.feedback_loop_alias_images > made.feedback_loop_alias_images
                || counts.feedback_loop_shadow_images > made.feedback_loop_shadow_images
        }
    };
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(f64::from(CLEARED) / 255.0, 0.0, 0.0, 1.0);
    color.store_action = StoreAction::Store;
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    pass.color_attachments[0].load_action = LoadAction::Load;
    let color = &mut pass.color_attachments[1];
    color.texture = Some(second.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    let counted = device.work_counts();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_vertex_bytes(&corners(BLOCK), 0).unwrap();
    let solid = |red: u8| floats([f32::from(red) / 255.0, 0.0, 0.0, 1.0]);
    encoder.set_render_pipeline_state(&fill).unwrap();
    encoder.set_fragment_bytes(&solid(FILL), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    if barrier {
        encoder.memory_barrier_with_scope(
            BarrierScope::RENDER_TARGETS,
            RenderStages::FRAGMENT,
            RenderStages::FRAGMENT,
        );
    }
    encoder.set_render_pipeline_state(&read).unwrap();
    encoder.set_fragment_texture(Some(&target), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.set_render_pipeline_state(&fill).unwrap();
    encoder.set_fragment_bytes(&solid(LATE), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let back = || {
        device
            .new_buffer(SIDE * ROW, ResourceOptions::shared())
            .unwrap()
    };
    let backs = [back(), back()];
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for (texture, back) in [&second, &target].into_iter().zip(&backs) {
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            back,
            0,
            ROW,
            SIDE * ROW,
            BlitOption::None,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Left {
        second: bytes(&backs[0]),
        target: bytes(&backs[1]),
        moved: moved(&counted, &device.work_counts()),
        feedback,
    }
}

/// Hold `bytes` (tight `RGBA8Unorm` rows of [`SIDE`] pixels) to red `block` at every pixel of
/// the block and `outside` elsewhere.
fn holds(what: &str, bytes: &[u8], block: u8, outside: u8) {
    let [x0, y0, x1, y1] = BLOCK;
    for y in 0..SIDE {
        for x in 0..SIDE {
            let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
            let want = if inside { block } else { outside };
            let red = bytes[((y * SIDE + x) * 4) as usize];
            assert_eq!(red, want, "{what}: pixel ({x}, {y})");
        }
    }
}

/// The lane's device under the default policy, the consumer's, with `adjust` applied too.
fn on_default_policy(adjust: impl FnOnce(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    on_device_with(
        |options| {
            options.policy = DeviceOptions::default().policy;
            adjust(options);
        },
        |device, _messages| test(device),
    );
}

#[test]
fn a_reader_that_writes_nothing_of_its_color_reads_in_place() {
    on_default_policy(
        |_| {},
        |device| {
            let queue = device.new_command_queue().unwrap();
            for surface in [false, true] {
                for barrier in [false, true] {
                    let what = format!("surface {surface}, barrier {barrier}");
                    let left = read_pass(device, &queue, surface, barrier);
                    let moved = &left.moved;
                    eprintln!(
                        "in-place reads, {what}, feedback loop {}: {moved:?}",
                        left.feedback
                    );
                    if left.feedback {
                        let want = Moved {
                            in_place: 1,
                            snapshotted: 0,
                            splits: 0,
                            copies: 0,
                        };
                        assert_eq!(*moved, want, "{what}: the reader in place");
                    }
                    // In place, or after the barrier, what F wrote; else the copy's texels from
                    // before the pass.
                    let read = match moved.in_place == 1 || barrier {
                        true => FILL,
                        false => CLEARED,
                    };
                    holds(&format!("{what}: color 1"), &left.second, read, 0);
                    holds(&format!("{what}: the target"), &left.target, LATE, CLEARED);
                }
            }
        },
    );
}

#[test]
fn with_the_feedback_loop_withheld_it_reads_the_copy() {
    on_default_policy(
        |options| {
            (options.withheld_features).insert(VkFeature::AttachmentFeedbackLoopLayout);
        },
        |device| {
            let queue = device.new_command_queue().unwrap();
            for barrier in [false, true] {
                let what = format!("withheld, barrier {barrier}");
                let left = read_pass(device, &queue, false, barrier);
                let moved = &left.moved;
                eprintln!("in-place reads, {what}: {moved:?}");
                assert_eq!(moved.in_place, 0, "{what}: reads in place");
                if !barrier {
                    assert_eq!(moved.copies, 1, "{what}: the reader through the copy");
                }
                let read = if barrier { FILL } else { CLEARED };
                holds(&format!("{what}: color 1"), &left.second, read, 0);
                holds(&format!("{what}: the target"), &left.target, LATE, CLEARED);
            }
        },
    );
}
