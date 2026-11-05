//! Device tests of a draw that samples its own render pass's color attachment's texture, on the
//! device the cfg selects, under the validation layer: the draw reads a copy of the texture taken
//! before its render pass instance, never the attachment (`texture/attached_reads.rs`; Vulkan's
//! `VUID-vkCmdDraw-None-10795`, which the layer cannot check inside the runtime's secondary
//! command buffers).
//!
//! The target is a [`SIDE`]-square `RGBA8Unorm` texture holding red [`BEFORE`] before the pass:
//! one of its own image, and one over a buffer's bytes. Draw A fills its block `[8, 16)²` with red
//! [`DRAWN`]. Where the pass renders a target's linear image through a copied stand-in
//! (`linear_attachment.rs`), its image is no attachment, and the draw reads it in place
//! (`WorkCounts::linear_passes_through_copies` moved): the same answers, no copy of the read's.
//!
//! - [`a_draw_after_a_barrier_samples_what_the_draws_before_wrote`]: after a memory barrier over
//!   the render targets, after the fragment stage and before the fragment or the vertex stage,
//!   draw C (`a4_fragment_select_sample` over `[4, 8)²`, the target bound as both of its
//!   textures) samples, at pixel `(x, y)`, the target's texel `(2x + 1, 2y + 1)` (the pixel's
//!   center over 16, through a nearest sampler): other pixels than it writes, inside A's block.
//!   It draws A's red, as a Mac2 GPU orders the barrier; the draw records into a new instance
//!   (`attachment_read_splits`), reads the copy (`attachment_reads_through_copies`), and an
//!   encoder copies it in (`shadow_copies`). After one from the vertex stage to the vertex stage,
//!   which orders no render target, it draws the red from before the pass, with no split.
//! - [`a_draw_without_a_barrier_samples_the_texels_from_before_the_pass`]: with no barrier, draw C
//!   (`tob_sample` over A's block, at its own pixel) draws the red from before the pass, Metal's
//!   answer at a draw's own pixel (probe `selfsample`, an M3 Max under macOS 27.0.1, 80 of 80
//!   runs), with no split. A device that read the attachment in A's instance read A's red there.
//! - [`a_chain_of_barriers_copies_in_only_what_each_instance_drew`]: [`CHAIN`] steps after A, each
//!   a fill of one pixel through a one-pixel scissor, a memory barrier, and draw C through a
//!   one-pixel scissor at the pixel that samples it: every C draws its fill's red, every other
//!   pixel keeps its red, the answers of a copy of every texel before each instance. Each copy
//!   after the first takes only the rectangle bounding the pixels the instance before it drew
//!   (`bounded_shadow_copies`; `shadow_texels_copied_in` over a buffer,
//!   `shadow_texels_copied_in_images` for the texture of its own image).
//! - [`own_pixel_reads_copy_in_only_around_their_scissors`]: [`OWN_STEPS`] steps after A, each a
//!   fill of one pixel through a one-pixel scissor (corners and edges among them), a memory
//!   barrier, and draw C (`tob_read`, `t.read(uint2(in.position.xy))`) through a one-pixel
//!   scissor at that pixel, then one more at the first step's pixel: every C draws its fill's red.
//!   Each C reads at its own pixel alone (`own_pixel_reads`), and each copy in moves at most the
//!   3x3 texels around its pixel. A C the device reads through a snapshot
//!   inside its instance (`snapshot_reads.rs`, `reading_draws_snapshotted`) takes neither a split
//!   nor a copy, and answers the same.
//! - [`an_own_pixel_read_without_a_barrier_reads_the_texels_from_before_the_pass`]: with no
//!   barrier, `tob_read` over A's block draws the red from before the pass (probe `selfsample`),
//!   its copy the texels around the block alone.
//! - [`a_pass_that_samples_no_attachment_copies_nothing`]: A, then `tob_read` of another texture:
//!   no read through a copy, split, own-pixel read or shadow copy.
//! - [`a_draw_after_a_barrier_at_the_pass_start_reads_the_clear`]: a pass that clears the target
//!   to [`CLEAR`] and begins with a memory barrier over the render targets (the scope alone, with
//!   the textures, or the target named), then one draw through [`READ_AREA`] that reads the target
//!   at its own pixel (`psb_read`) or samples it at its pixel's center in pixel coordinates
//!   (`psb_sample`): the draw reads the clear, through its copy cleared to it in the pass's first
//!   instance (`attachment_read_clears`), in an instance after the pass's first, which runs the
//!   clear alone (`attachment_read_splits`), or through a snapshot.
//! - [`a_draw_after_one_that_wrote_behind_a_pass_start_barrier_reads_what_it_wrote`]: the same
//!   with draw A between the barrier and the read reads A's red inside A's block, the clear
//!   around it, and takes no cleared copy.
//! - [`a_draw_after_a_clear_with_no_barrier_reads_the_texels_from_before_the_pass`]: the same with
//!   no barrier reads the red from before the pass, with no split.
//! - [`a_barrier_at_the_pass_start_costs_nothing_without_a_read_of_a_cleared_color`]: a pass that
//!   loads the target and reads it after the barrier, or clears it and reads another texture,
//!   splits nothing and begins as many instances and copies as the same pass with no barrier.

use super::degrade_notices::{assert_once, counts, on_device};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Degradation, Device, LoadAction, Origin,
    PipelineOptions, Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ResourceRef, ScissorRect, Size, StoreAction, Texture, TextureDescriptor,
    WorkCounts,
};
use std::path::Path;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The target's side, in pixels.
pub(super) const SIDE: u64 = 32;

/// The target's red before the pass.
const BEFORE: u8 = 64;

/// The red draw A writes.
const DRAWN: u8 = 128;

/// A function of the fixture `fixture` under `fixtures/public/engine`.
pub(super) fn function(device: &Device, name: &str, fixture: &str) -> StageSource {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public/engine")
        .join(fixture);
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    StageSource::Function(
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor(),
    )
}

/// The `tob_rect` state with the fragment function `name` of `fixture`, into an unblended
/// `RGBA8Unorm` color.
fn state(device: &Device, name: &str, fixture: &str) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
    d.fragment = Some(function(device, name, fixture));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// The bytes of `f32`s.
pub(super) fn floats(v: [f32; 4]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
pub(super) fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    floats([x(x0), y(y1), x(x1), y(y0)])
}

/// Every texel red [`BEFORE`], opaque.
fn before() -> Vec<u8> {
    [BEFORE, 0, 0, 255].repeat((SIDE * SIDE) as usize)
}

/// The targets: a `Shared` texture of its own image, and one over a `Shared` buffer's bytes (the
/// buffer kept alive beside it), each holding [`before`].
fn targets(device: &Device) -> Vec<(&'static str, Texture, Option<Buffer>)> {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let own = device.new_texture(&d).unwrap();
    own.replace_region(
        Region::new_2d(0, 0, SIDE, SIDE),
        0,
        0,
        &before(),
        SIDE * 4,
        0,
    )
    .unwrap();
    // Tight rows: 128 bytes, a multiple of every linear alignment Metal answers (16).
    let row = SIDE * 4;
    let buffer = device
        .new_buffer(row * SIDE, ResourceOptions::shared())
        .unwrap();
    let over = buffer.new_texture(&d, 0, row).unwrap();
    over.replace_region(
        Region::new_2d(0, 0, SIDE, SIDE),
        0,
        0,
        &before(),
        SIDE * 4,
        0,
    )
    .unwrap();
    vec![
        ("its own image", own, None),
        ("over a buffer", over, Some(buffer)),
    ]
}

/// One pass over `target`, loaded and stored: draw A, then `then` (draw C), and the target's
/// bytes after it, tight rows, read back.
fn pass(
    queue: &CommandQueue,
    target: &Texture,
    fill: &RenderPipelineState,
    then: impl FnOnce(&mut RenderCommandEncoder),
) -> Vec<u8> {
    let device = queue.device();
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
    encoder.set_render_pipeline_state(fill).unwrap();
    encoder
        .set_vertex_bytes(&corners([8, 8, 16, 16]), 0)
        .unwrap();
    let red = f32::from(DRAWN) / 255.0;
    encoder
        .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    then(&mut encoder);
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = back.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), back.length() as usize) }.to_vec()
}

/// The red of pixel `(x, y)` of `bytes`, tight `RGBA8Unorm` rows.
pub(super) fn red(bytes: &[u8], x: u64, y: u64) -> u8 {
    bytes[((y * SIDE + x) * 4) as usize]
}

/// What the pass moved of the counts: its draws' splits for attachment reads, their reads of
/// copies cleared to the pass's clear in place of a split, their reads through snapshots inside
/// their instances, their reads through copies, the encoders' shadow
/// copies, of those the copies in of what changed within a rectangle alone, the texels the copies
/// in of buffers' textures moved and those of images' shadows, the texels the encoder's first
/// copy in moved of a truth that is an image and of one in a mapped buffer, and whether the pass
/// rendered the target through a copied stand-in.
struct Moved {
    splits: u64,
    cleared: u64,
    snapshotted: u64,
    reads: u64,
    copies: u64,
    bounded: u64,
    texels_in: u64,
    texels_in_images: u64,
    first: [u64; 2],
    apart: bool,
}

fn moved(before: &WorkCounts, after: &WorkCounts) -> Moved {
    Moved {
        splits: after.attachment_read_splits - before.attachment_read_splits,
        cleared: after.attachment_read_clears - before.attachment_read_clears,
        snapshotted: after.reading_draws_snapshotted - before.reading_draws_snapshotted,
        reads: after.attachment_reads_through_copies - before.attachment_reads_through_copies,
        copies: after.shadow_copies - before.shadow_copies,
        bounded: after.bounded_shadow_copies - before.bounded_shadow_copies,
        texels_in: after.shadow_texels_copied_in - before.shadow_texels_copied_in,
        texels_in_images: after.shadow_texels_copied_in_images
            - before.shadow_texels_copied_in_images,
        first: [
            after.shadow_texels_first_copied_in_images
                - before.shadow_texels_first_copied_in_images,
            after.shadow_texels_first_copied_in_mapped_buffers
                - before.shadow_texels_first_copied_in_mapped_buffers,
        ],
        apart: after.linear_passes_through_copies > before.linear_passes_through_copies,
    }
}

/// Hold `moved` to a pass whose draws C sample the target `name`, after `barriers` barriers, one
/// before each, or none: through a copy, taken after a split where a barrier came first or
/// cleared to the pass's clear where one came before any draw of a pass that clears the target,
/// or a draw after a barrier that reads at its own pixel alone through a snapshot inside its
/// instance (`snapshot_reads.rs`), where the pass renders the target itself; in place where it
/// renders through a stand-in.
fn check(name: &str, moved: &Moved, barriers: u64) {
    eprintln!(
        "attachment reads, {name}, barriers {barriers}: splits {}, cleared copies {}, snapshotted \
         {}, reads through copies {}, shadow copies {} (bounded {}, texels in {} and {} of \
         images, first in {:?}), through a stand-in {}",
        moved.splits,
        moved.cleared,
        moved.snapshotted,
        moved.reads,
        moved.copies,
        moved.bounded,
        moved.texels_in,
        moved.texels_in_images,
        moved.first,
        moved.apart
    );
    if moved.apart {
        assert_eq!(
            moved.reads, 0,
            "{name}: the stand-in rendered, the image read in place"
        );
        return;
    }
    assert_eq!(
        moved.splits + moved.cleared + moved.snapshotted,
        barriers,
        "{name}: a split, a cleared copy or a snapshot for each read after a barrier"
    );
    if moved.snapshotted < barriers.max(1) {
        assert!(moved.reads >= 1, "{name}: the draw read no copy");
    }
    if moved.snapshotted + moved.cleared < barriers.max(1) {
        assert!(moved.copies >= 1, "{name}: no copy was taken");
        // The encoder's first copy in, counted by what holds the target's texels: its own image,
        // or the shared buffer's bytes.
        let [images, mapped] = moved.first;
        let own = name.starts_with("its own image");
        assert_eq!(
            (images > 0, mapped > 0),
            (own, !own),
            "{name}: the first copy's texels counted by the truth's backing"
        );
    }
}

/// The steps of [`own_pixel_reads_copy_in_only_around_their_scissors`].
const OWN_STEPS: u64 = 12;

/// Step `i`'s pixel there: the corners, the middles of the edges, and pixels inside, none in
/// A's block.
fn own_step(i: u64) -> [u64; 2] {
    const PIXELS: [[u64; 2]; OWN_STEPS as usize] = [
        [0, 0],
        [SIDE - 1, 0],
        [0, SIDE - 1],
        [SIDE - 1, SIDE - 1],
        [SIDE / 2, 0],
        [0, SIDE / 2],
        [SIDE - 1, SIDE / 2],
        [SIDE / 2, SIDE - 1],
        [3, 4],
        [4, 3],
        [20, 25],
        [21, 25],
    ];
    PIXELS[i as usize]
}

/// The red step `i`'s fill writes: none of [`BEFORE`], [`DRAWN`] or another step's.
fn own_red(i: u64) -> u8 {
    (150 + 5 * i) as u8
}

/// Set a one-pixel scissor at `(x, y)` and the corners of that pixel.
fn one_pixel(encoder: &mut RenderCommandEncoder, [x, y]: [u64; 2]) {
    let rect = ScissorRect {
        x,
        y,
        width: 1,
        height: 1,
    };
    encoder.set_scissor_rect(rect).unwrap();
    encoder
        .set_vertex_bytes(&corners([x, y, x + 1, y + 1]), 0)
        .unwrap();
}

#[test]
fn own_pixel_reads_copy_in_only_around_their_scissors() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let fill = state(device, "tob_solid", "tob_renders.ll");
        let read = state(device, "tob_read", "tob_renders.ll");
        // Each step, then the first step's pixel again.
        let steps: Vec<(u64, [u64; 2])> = (0..OWN_STEPS)
            .map(|i| (i, own_step(i)))
            .chain([(OWN_STEPS, own_step(0))])
            .collect();
        for (name, target, _buffer) in targets(device) {
            let counted = device.work_counts();
            let bytes = pass(&queue, &target, &fill, |encoder| {
                for &(i, pixel) in &steps {
                    encoder.set_render_pipeline_state(&fill).unwrap();
                    one_pixel(encoder, pixel);
                    let red = f32::from(own_red(i)) / 255.0;
                    encoder
                        .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
                        .unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                    encoder.memory_barrier_with_scope(
                        BarrierScope::RENDER_TARGETS,
                        RenderStages::FRAGMENT,
                        RenderStages::FRAGMENT,
                    );
                    encoder.set_render_pipeline_state(&read).unwrap();
                    one_pixel(encoder, pixel);
                    encoder.set_fragment_texture(Some(&target), 0).unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                }
            });
            // Each C its fill's red (the last fill's at the first pixel), A's block A's red,
            // every other pixel the red from before the pass.
            let mut expected = vec![BEFORE; (SIDE * SIDE) as usize];
            for y in 8..16 {
                for x in 8..16 {
                    expected[(y * SIDE + x) as usize] = DRAWN;
                }
            }
            for &(i, [x, y]) in &steps {
                expected[(y * SIDE + x) as usize] = own_red(i);
            }
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let want = expected[(y * SIDE + x) as usize];
                    assert_eq!(red(&bytes, x, y), want, "{name}: pixel ({x}, {y})");
                }
            }
            let after = device.work_counts();
            let moved = moved(&counted, &after);
            let count = steps.len() as u64;
            check(name, &moved, count);
            let own = after.own_pixel_reads - counted.own_pixel_reads;
            eprintln!("own-pixel reads, {name}: {own}");
            if moved.apart {
                continue;
            }
            // Each C read the copy, or read through a snapshot inside its instance.
            assert_eq!(
                moved.reads + moved.snapshotted,
                count,
                "{name}: each C read the copy or a snapshot"
            );
            // Each copy in of at most the 3x3 texels around a C's pixel (`ShadowCopies::cuts`),
            // and over a buffer a shadowed attachment's own copy in at most every texel more.
            assert_eq!(own, moved.reads, "{name}: own-pixel reads");
            let (texels, most) = match name == "over a buffer" {
                true => (moved.texels_in, SIDE * SIDE + 9 * count),
                false => (moved.texels_in_images, 9 * count),
            };
            assert!(
                texels <= most,
                "{name}: {texels} texels copied in, past {most}"
            );
        }
    });
}

#[test]
fn an_own_pixel_read_without_a_barrier_reads_the_texels_from_before_the_pass() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let fill = state(device, "tob_solid", "tob_renders.ll");
        let read = state(device, "tob_read", "tob_renders.ll");
        for (name, target, _buffer) in targets(device) {
            let counted = device.work_counts();
            let bytes = pass(&queue, &target, &fill, |encoder| {
                encoder.set_render_pipeline_state(&read).unwrap();
                encoder
                    .set_vertex_bytes(&corners([8, 8, 16, 16]), 0)
                    .unwrap();
                encoder.set_fragment_texture(Some(&target), 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            });
            for y in 0..SIDE {
                for x in 0..SIDE {
                    assert_eq!(red(&bytes, x, y), BEFORE, "{name}: pixel ({x}, {y})");
                }
            }
            let after = device.work_counts();
            let moved = moved(&counted, &after);
            check(name, &moved, 0);
            let own = after.own_pixel_reads - counted.own_pixel_reads;
            eprintln!("own-pixel reads, {name}: {own}");
            if !moved.apart {
                assert_eq!(own, 1, "{name}: own-pixel reads");
                // The full scissor, the whole target: its copy every texel at most, and over a
                // buffer a shadowed attachment's own every texel more.
                let most = 2 * SIDE * SIDE;
                let texels = moved.texels_in + moved.texels_in_images;
                assert!(texels <= most, "{name}: {texels}");
            }
        }
    });
}

#[test]
fn a_pass_that_samples_no_attachment_copies_nothing() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let fill = state(device, "tob_solid", "tob_renders.ll");
        let read = state(device, "tob_read", "tob_renders.ll");
        let (name, target, _) = targets(device).swap_remove(0);
        let (_, other, _) = targets(device).swap_remove(0);
        let counted = device.work_counts();
        let bytes = pass(&queue, &target, &fill, |encoder| {
            encoder.set_render_pipeline_state(&read).unwrap();
            encoder.set_vertex_bytes(&corners([4, 4, 8, 8]), 0).unwrap();
            encoder.set_fragment_texture(Some(&other), 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        });
        assert_eq!(red(&bytes, 4, 4), BEFORE, "{name}: the other texture's red");
        assert_eq!(red(&bytes, 8, 8), DRAWN, "{name}: A's red");
        let after = device.work_counts();
        let moved = moved(&counted, &after);
        let own = after.own_pixel_reads - counted.own_pixel_reads;
        eprintln!(
            "no attachment read, {name}: reads {}, splits {}, own-pixel {own}, shadow copies {}, \
             command buffers begun {}, draw states recorded {}",
            moved.reads,
            moved.splits,
            moved.copies,
            after.command_buffers_begun - counted.command_buffers_begun,
            after.draw_states_recorded - counted.draw_states_recorded
        );
        assert_eq!(
            (moved.reads, moved.splits, own, moved.copies),
            (0, 0, 0, 0),
            "{name}: reads, splits, own-pixel reads, shadow copies"
        );
    });
}

#[test]
fn a_draw_after_a_barrier_samples_what_the_draws_before_wrote() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let fill = state(device, "tob_solid", "tob_renders.ll");
        let sample = state(
            device,
            "a4_fragment_select_sample",
            "a4_fragment_select_sample.ll",
        );
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let (fragment, vertex) = (RenderStages::FRAGMENT, RenderStages::VERTEX);
        let barriers = [
            (fragment, fragment, true),
            (fragment, vertex, true),
            (vertex, vertex, false),
        ];
        for ((name, target, _buffer), (after, before, ordered)) in barriers
            .into_iter()
            .flat_map(|barrier| targets(device).into_iter().map(move |t| (t, barrier)))
        {
            let counted = device.work_counts();
            let bytes = pass(&queue, &target, &fill, |encoder| {
                encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, after, before);
                encoder.set_render_pipeline_state(&sample).unwrap();
                encoder.set_vertex_bytes(&corners([4, 4, 8, 8]), 0).unwrap();
                encoder.set_fragment_texture(Some(&target), 0).unwrap();
                encoder.set_fragment_texture(Some(&target), 1).unwrap();
                encoder
                    .set_fragment_sampler_state(Some(&sampler), 0)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            });
            let want = if ordered { DRAWN } else { BEFORE };
            let name = format!("{name}, {after:?} to {before:?}");
            for y in 4..8 {
                for x in 4..8 {
                    assert_eq!(red(&bytes, x, y), want, "{name}: pixel ({x}, {y})");
                }
            }
            assert_eq!(red(&bytes, 0, 0), BEFORE, "{name}: outside the draws");
            check(
                &name,
                &moved(&counted, &device.work_counts()),
                u64::from(ordered),
            );
        }
    });
}

#[test]
fn a_draw_without_a_barrier_samples_the_texels_from_before_the_pass() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let fill = state(device, "tob_solid", "tob_renders.ll");
        let sample = state(device, "tob_sample", "tob_renders.ll");
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        for (name, target, _buffer) in targets(device) {
            let counted = device.work_counts();
            let bytes = pass(&queue, &target, &fill, |encoder| {
                encoder.set_render_pipeline_state(&sample).unwrap();
                encoder
                    .set_vertex_bytes(&corners([8, 8, 16, 16]), 0)
                    .unwrap();
                encoder.set_fragment_texture(Some(&target), 0).unwrap();
                encoder
                    .set_fragment_sampler_state(Some(&sampler), 0)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            });
            for y in 8..16 {
                for x in 8..16 {
                    assert_eq!(red(&bytes, x, y), BEFORE, "{name}: pixel ({x}, {y})");
                }
            }
            check(name, &moved(&counted, &device.work_counts()), 0);
        }
    });
}

/// The kind a draw's use of an attachment's memory other than as the attachment takes
/// ([`a_read_of_the_attachments_bytes_is_served_or_announced`]).
pub(super) const IN_PLACE: [Degradation; 1] = [Degradation::AttachmentReadInPlace];

/// Draws that read, in the pass over the target over a buffer, its bytes other than as the
/// attachment. Draw C (`tob_sample` over A's block) samples first the target itself, then a second
/// texture of the target's descriptor over the same bytes: each, where it and the target are
/// aliasing images over the buffer's memory (neither shadowed, `WorkCounts::shadowed_textures`,
/// and the target rendered through no copied stand-in), is held as the attachment's own texture
/// (an aliasing image's identity is its buffer's) and read through its copy; elsewhere it is
/// another image than the pass renders. Neither is announced. Then the fill (`tob_solid`) reads its
/// color from the buffer's first bytes through a buffer binding, which no copy serves
/// (`texture/attached_reads.rs`, not served): where the pass renders the buffer's memory itself
/// (the target not shadowed, through no stand-in), the draw reads the attachment's memory in
/// place, announced once as `attachment-read-in-place`, and what it reads is Vulkan's undefined
/// answer, unchecked; elsewhere it reads other memory than the pass renders, and nothing is
/// announced.
#[test]
fn a_read_of_the_attachments_bytes_is_served_or_announced() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let fill = state(device, "tob_solid", "tob_renders.ll");
        let sample = state(device, "tob_sample", "tob_renders.ll");
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let made = device.work_counts();
        let (name, target, buffer) = targets(device).pop().unwrap();
        let buffer = buffer.expect("the target over a buffer");
        let held = device.work_counts();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let other = buffer.new_texture(&d, 0, SIDE * 4).unwrap();
        let target_shadowed = held.shadowed_textures - made.shadowed_textures;
        let other_shadowed = device.work_counts().shadowed_textures - held.shadowed_textures;
        let draw_c = |encoder: &mut RenderCommandEncoder, texture: &Texture| {
            encoder.set_render_pipeline_state(&sample).unwrap();
            encoder
                .set_vertex_bytes(&corners([8, 8, 16, 16]), 0)
                .unwrap();
            encoder.set_fragment_texture(Some(texture), 0).unwrap();
            encoder
                .set_fragment_sampler_state(Some(&sampler), 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        };
        messages.take_notices();
        for (what, texture, shadowed) in [
            ("the target", &target, target_shadowed),
            (
                "a second texture over its bytes",
                &other,
                target_shadowed + other_shadowed,
            ),
        ] {
            let counted = device.work_counts();
            pass(&queue, &target, &fill, |encoder| draw_c(encoder, texture));
            let seen = moved(&counted, &device.work_counts());
            let notices = counts(&messages.take_notices());
            eprintln!(
                "{name}: {what}: shadowed textures {shadowed}, reads through copies {}, through a \
                 stand-in {}",
                seen.reads, seen.apart
            );
            assert!(
                !notices.contains_key(&IN_PLACE[0]),
                "{name}: {what}: {notices:?}"
            );
            if shadowed == 0 && !seen.apart {
                assert_eq!(seen.reads, 1, "{name}: {what}: read through its copy");
            }
        }
        let counted = device.work_counts();
        pass(&queue, &target, &fill, |encoder| {
            encoder
                .set_vertex_bytes(&corners([8, 8, 16, 16]), 0)
                .unwrap();
            encoder.set_fragment_buffer(Some(&buffer), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        });
        let apart = moved(&counted, &device.work_counts()).apart;
        let notices = messages.take_notices();
        eprintln!(
            "{name}: its buffer's bytes: shadowed textures {target_shadowed}, through a stand-in \
             {apart}"
        );
        if target_shadowed == 0 && !apart {
            assert_once(&notices, &IN_PLACE);
        } else {
            assert!(!counts(&notices).contains_key(&IN_PLACE[0]), "{name}");
        }
    });
}

/// The steps of [`a_chain_of_barriers_copies_in_only_what_each_instance_drew`]: as many
/// render-target barriers in one pass as the most a pass of the owner's RTX 5080 trace has.
const CHAIN: u64 = 29;

/// Step `i`'s pixel `(x, y)`, where its draw C writes, sampling texel `(2x + 1, 2y + 1)`, its
/// fill's pixel: even columns, so no C writes a pixel a fill writes or a C samples, and outside
/// A's block.
fn chained(i: u64) -> [u64; 2] {
    [2 * (i % 8), i / 8]
}

/// The red step `i`'s fill writes: none of [`BEFORE`], [`DRAWN`] or another step's.
fn chain_red(i: u64) -> u8 {
    (140 + 3 * i) as u8
}

/// Of the target's `SIDE` by `SIDE` texels, those the rectangles bounding the pixels each
/// instance of the chain after the first draws hold (draw C of the step before, then the fill):
/// what a copy in of only those moves before each later instance.
fn chain_texels() -> u64 {
    (1..CHAIN)
        .map(|i| {
            let ([cx, cy], [x, y]) = (chained(i - 1), chained(i));
            let (fx, fy) = (2 * x + 1, 2 * y + 1);
            (cx.abs_diff(fx) + 1) * (cy.abs_diff(fy) + 1)
        })
        .sum()
}

#[test]
fn a_chain_of_barriers_copies_in_only_what_each_instance_drew() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let fill = state(device, "tob_solid", "tob_renders.ll");
        let sample = state(
            device,
            "a4_fragment_select_sample",
            "a4_fragment_select_sample.ll",
        );
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let one = |encoder: &mut RenderCommandEncoder, [x, y]: [u64; 2]| {
            let rect = ScissorRect {
                x,
                y,
                width: 1,
                height: 1,
            };
            encoder.set_scissor_rect(rect).unwrap();
            encoder
                .set_vertex_bytes(&corners([x, y, x + 1, y + 1]), 0)
                .unwrap();
        };
        for (name, target, _buffer) in targets(device) {
            let counted = device.work_counts();
            let bytes = pass(&queue, &target, &fill, |encoder| {
                for i in 0..CHAIN {
                    let [x, y] = chained(i);
                    encoder.set_render_pipeline_state(&fill).unwrap();
                    one(encoder, [2 * x + 1, 2 * y + 1]);
                    let red = f32::from(chain_red(i)) / 255.0;
                    encoder
                        .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
                        .unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                    encoder.memory_barrier_with_scope(
                        BarrierScope::RENDER_TARGETS,
                        RenderStages::FRAGMENT,
                        RenderStages::FRAGMENT,
                    );
                    encoder.set_render_pipeline_state(&sample).unwrap();
                    one(encoder, [x, y]);
                    encoder.set_fragment_texture(Some(&target), 0).unwrap();
                    encoder.set_fragment_texture(Some(&target), 1).unwrap();
                    encoder
                        .set_fragment_sampler_state(Some(&sampler), 0)
                        .unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                        .unwrap();
                }
            });
            // What a copy of every texel before each instance answers: each C its fill's red,
            // A's block A's red, every other pixel the red from before the pass.
            let mut expected = vec![BEFORE; (SIDE * SIDE) as usize];
            for y in 8..16 {
                for x in 8..16 {
                    expected[(y * SIDE + x) as usize] = DRAWN;
                }
            }
            for i in 0..CHAIN {
                let [x, y] = chained(i);
                expected[(y * SIDE + x) as usize] = chain_red(i);
                expected[((2 * y + 1) * SIDE + 2 * x + 1) as usize] = chain_red(i);
            }
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let want = expected[(y * SIDE + x) as usize];
                    assert_eq!(red(&bytes, x, y), want, "{name}: pixel ({x}, {y})");
                }
            }
            let moved = moved(&counted, &device.work_counts());
            check(name, &moved, CHAIN);
            if moved.apart {
                continue;
            }
            assert_eq!(moved.reads, CHAIN, "{name}: each C read the copy");
            // The first copy of every texel, then only what each instance drew
            // (`ShadowCopies::cuts`); over a buffer, a shadowed attachment's own copy in takes at
            // most every texel more.
            assert_eq!(moved.bounded, CHAIN - 1, "{name}: bounded copies");
            let (texels, most) = match name == "over a buffer" {
                true => (moved.texels_in, 2 * SIDE * SIDE + chain_texels()),
                false => (moved.texels_in_images, SIDE * SIDE + chain_texels()),
            };
            assert!(
                texels <= most,
                "{name}: {texels} texels copied in, past {most}"
            );
        }
    });
}

/// The clear a pass of [`started_pass`] that clears its target clears it to: codes none of
/// [`BEFORE`]'s texel's, each a whole code so the clear stores it exactly.
const CLEAR: [u8; 4] = [200, 40, 120, 160];

/// The scissor of the draw of [`started_pass`]: inside the target, off its edges.
const READ_AREA: ScissorRect = ScissorRect {
    x: 4,
    y: 4,
    width: 20,
    height: 12,
};

/// What a pass of [`started_pass`] encodes before its draw.
#[derive(Clone, Copy, Debug)]
enum Start {
    /// Nothing.
    Nothing,
    /// `memoryBarrierWithScope:` of this scope, after the fragment stage and before it.
    Scope(BarrierScope),
    /// `memoryBarrierWithResources:` over the target, after the fragment stage and before it.
    Resources,
}

/// The `psb_rect` state with the fragment function `fragment` of `psb_reads`, into an unblended
/// `RGBA8Unorm` color.
fn psb_state(device: &Device, fragment: &str) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "psb_rect", "psb_reads.ll"));
    d.fragment = Some(function(device, fragment, "psb_reads.ll"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// One pass over `target`, loaded by `load` (to [`CLEAR`] where it clears) and stored: `start`
/// first, then draw A of `fill` where given (red [`DRAWN`] over `[8, 16)²`), then one draw of
/// `state` over the whole target through [`READ_AREA`], `read` at its fragment texture 0; and the
/// target's bytes, tight rows, read back.
fn started_pass(
    queue: &CommandQueue,
    target: &Texture,
    (load, start, fill): (LoadAction, Start, Option<&RenderPipelineState>),
    (state, read): (&RenderPipelineState, &Texture),
) -> Vec<u8> {
    let device = queue.device();
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = load;
    color.store_action = StoreAction::Store;
    let code = |at: usize| f64::from(CLEAR[at]) / 255.0;
    color.clear_color = ClearColor::new(code(0), code(1), code(2), code(3));
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
    let fragment = RenderStages::FRAGMENT;
    match start {
        Start::Nothing => {}
        Start::Scope(scope) => encoder.memory_barrier_with_scope(scope, fragment, fragment),
        Start::Resources => encoder
            .memory_barrier_with_resources(&[ResourceRef::Texture(target)], fragment, fragment)
            .unwrap(),
    }
    if let Some(fill) = fill {
        encoder.set_render_pipeline_state(fill).unwrap();
        encoder
            .set_vertex_bytes(&corners([8, 8, 16, 16]), 0)
            .unwrap();
        let red = f32::from(DRAWN) / 255.0;
        encoder
            .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }
    encoder.set_render_pipeline_state(state).unwrap();
    encoder
        .set_vertex_bytes(&corners([0, 0, SIDE, SIDE]), 0)
        .unwrap();
    encoder.set_scissor_rect(READ_AREA).unwrap();
    encoder.set_fragment_texture(Some(read), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = back.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), back.length() as usize) }.to_vec()
}

/// What `psb_read` and `psb_sample` write for the texel `seen`: its channels as `(b, r, g)`,
/// opaque.
fn reordered(seen: [u8; 4]) -> [u8; 4] {
    [seen[2], seen[0], seen[1], 255]
}

/// Hold `bytes`, a pass of [`started_pass`]'s, to `inside` in [`READ_AREA`] and `outside` around
/// it.
fn hold_area(name: &str, bytes: &[u8], inside: [u8; 4], outside: [u8; 4]) {
    let area = |x: u64, y: u64| {
        let (x0, y0) = (READ_AREA.x, READ_AREA.y);
        (x0..x0 + READ_AREA.width).contains(&x) && (y0..y0 + READ_AREA.height).contains(&y)
    };
    for y in 0..SIDE {
        for x in 0..SIDE {
            let at = ((y * SIDE + x) * 4) as usize;
            let want = if area(x, y) { inside } else { outside };
            assert_eq!(bytes[at..at + 4], want, "{name}: pixel ({x}, {y})");
        }
    }
}

/// The instance splits and the render pass instances a pass moved, by every cause.
fn instances(before: &WorkCounts, after: &WorkCounts) -> [u64; 2] {
    let splits = |c: &WorkCounts| {
        c.instance_splits_barrier
            + c.instance_splits_tile_dispatch
            + c.instance_splits_tile_memory
            + c.instance_splits_attached_read
            + c.instance_splits_feedback
            + c.instance_splits_memoryless_write
            + c.instance_splits_pending_pipeline
            + c.instance_splits_execution
            + c.instance_splits_visibility
            + c.instance_splits_shape
    };
    let begun = |c: &WorkCounts| {
        c.render_passes_begun_instances
            + c.render_passes_begun_inline
            + c.render_passes_begun_inline_splits
            + c.render_passes_begun_resolves
    };
    [splits(after) - splits(before), begun(after) - begun(before)]
}

/// The barriers [`a_draw_after_a_barrier_at_the_pass_start_reads_the_clear`] begins its passes
/// with: over the render targets alone (inside an instance where draws came before it), over
/// them and the textures (a split where draws came before it), and over the target itself.
const STARTS: [Start; 3] = [
    Start::Scope(BarrierScope::RENDER_TARGETS),
    Start::Scope(BarrierScope::RENDER_TARGETS.union(BarrierScope::TEXTURES)),
    Start::Resources,
];

/// A pass that clears its target and begins with a memory barrier over the render targets, after
/// the fragment stage and before it: its draw reads, at its own pixel (`psb_read`) or sampling at
/// its pixel's center in pixel coordinates (`psb_sample`), the clear, as a Mac2 GPU orders the
/// barrier (the pass's load actions are behind it; an M3 Max, whose GPU does not support the
/// scope, reads the texels from before the pass, `fence.rs`). Before any draw of the pass, it
/// reads its copy cleared to the clear in the pass's first instance (`attachment_read_clears`),
/// which the texture of its own image always takes, a whole one-level, one-slice texture the
/// pass clears in place to whole codes; or it records into an instance after the pass's first,
/// which runs the clear alone (`attachment_read_splits`, `instance_splits_attached_read`), or
/// reads through a snapshot inside the first (`reading_draws_snapshotted`).
#[test]
fn a_draw_after_a_barrier_at_the_pass_start_reads_the_clear() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        // Whether a pass split, and whether the device announced a split: once, at its first.
        let (mut split_any, mut announced) = (false, false);
        for fragment in ["psb_read", "psb_sample"] {
            let state = psb_state(device, fragment);
            for start in STARTS {
                for (name, target, _buffer) in targets(device) {
                    let name = format!("{name}, {fragment}, {start:?}");
                    messages.take_notices();
                    let counted = device.work_counts();
                    let bytes = started_pass(
                        &queue,
                        &target,
                        (LoadAction::Clear, start, None),
                        (&state, &target),
                    );
                    let after = device.work_counts();
                    // The split's notice names the read behind the barrier as its cause.
                    let notice = messages
                        .take_notices()
                        .into_iter()
                        .filter_map(|message| message.degradation)
                        .find(|notice| notice.degradation == Degradation::RenderPassSplit);
                    if let Some(split) = &notice {
                        assert!(
                            split.reason.contains(
                                "draws that read a color attachment's texture after a memory \
                                 barrier read what its load actions"
                            ),
                            "{name}: {}",
                            split.reason
                        );
                    }
                    hold_area(&name, &bytes, reordered(CLEAR), CLEAR);
                    let moved = moved(&counted, &after);
                    let [splits, begun] = instances(&counted, &after);
                    let attached =
                        after.instance_splits_attached_read - counted.instance_splits_attached_read;
                    eprintln!(
                        "{name}: instance splits {splits} ({attached} attached), begun {begun}"
                    );
                    check(&name, &moved, 1);
                    if !moved.apart {
                        assert_eq!(attached, moved.splits, "{name}: splits for the read");
                        assert_eq!(splits, moved.splits, "{name}: no other split");
                    }
                    if name.starts_with("its own image") && moved.snapshotted == 0 {
                        assert_eq!(
                            (moved.cleared, splits, begun),
                            (1, 0, 1),
                            "{name}: the copy cleared, in the pass's one instance"
                        );
                    }
                    split_any |= splits > 0;
                    announced |= notice.is_some();
                }
            }
        }
        assert_eq!(announced, split_any, "the split's notice");
    });
}

/// With no barrier, the draw of a pass that clears its target reads the texels from before the
/// pass at its own pixel, Metal's answer there (probe `selfsample`; the M3 Max reads them after a
/// barrier too), with no split.
#[test]
fn a_draw_after_a_clear_with_no_barrier_reads_the_texels_from_before_the_pass() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        for fragment in ["psb_read", "psb_sample"] {
            let state = psb_state(device, fragment);
            for (name, target, _buffer) in targets(device) {
                let name = format!("{name}, {fragment}");
                let counted = device.work_counts();
                let bytes = started_pass(
                    &queue,
                    &target,
                    (LoadAction::Clear, Start::Nothing, None),
                    (&state, &target),
                );
                let after = device.work_counts();
                hold_area(&name, &bytes, reordered([BEFORE, 0, 0, 255]), CLEAR);
                let moved = moved(&counted, &after);
                check(&name, &moved, 0);
                assert_eq!(instances(&counted, &after)[0], 0, "{name}: a split");
            }
        }
    });
}

/// After a memory barrier at the start of a pass that clears its target, draw A fills its block,
/// then the draw reads the target through [`READ_AREA`]: it reads A's red inside A's block and the
/// clear around it, as the copy after the instance A drew in holds (a Mac2 GPU orders A's write
/// before the read nowhere; the copy holds what the draws between the barrier and the read
/// wrote, `texture/attached_reads.rs`). A drew before it, so its copy is no copy cleared to the
/// clear (`attachment_read_clears`), but a split's or a snapshot.
#[test]
fn a_draw_after_one_that_wrote_behind_a_pass_start_barrier_reads_what_it_wrote() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let fill = state(device, "tob_solid", "tob_renders.ll");
        for fragment in ["psb_read", "psb_sample"] {
            let state = psb_state(device, fragment);
            for (name, target, _buffer) in targets(device) {
                let name = format!("{name}, {fragment}");
                let counted = device.work_counts();
                let start = Start::Scope(BarrierScope::RENDER_TARGETS);
                let bytes = started_pass(
                    &queue,
                    &target,
                    (LoadAction::Clear, start, Some(&fill)),
                    (&state, &target),
                );
                let after = device.work_counts();
                let block = |x: u64, y: u64| (8..16).contains(&x) && (8..16).contains(&y);
                let area = |x: u64, y: u64| {
                    let (x0, y0) = (READ_AREA.x, READ_AREA.y);
                    (x0..x0 + READ_AREA.width).contains(&x)
                        && (y0..y0 + READ_AREA.height).contains(&y)
                };
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let want = match (area(x, y), block(x, y)) {
                            (true, true) => reordered([DRAWN, 0, 0, 255]),
                            (true, false) => reordered(CLEAR),
                            (false, _) => CLEAR,
                        };
                        let at = ((y * SIDE + x) * 4) as usize;
                        assert_eq!(bytes[at..at + 4], want, "{name}: pixel ({x}, {y})");
                    }
                }
                let moved = moved(&counted, &after);
                assert_eq!(moved.cleared, 0, "{name}: a copy cleared after A drew");
                check(&name, &moved, 1);
            }
        }
    });
}

/// A barrier at the start of a pass costs nothing where no draw after it samples a color the pass
/// clears: a pass that loads its target, whose draw reads at its own pixel the texels from before
/// the pass (what the load holds), and one that clears it, whose draw reads another texture. Each
/// splits no instance, begins as many render pass instances and takes as many copies as the same
/// pass with no barrier.
#[test]
fn a_barrier_at_the_pass_start_costs_nothing_without_a_read_of_a_cleared_color() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let read = psb_state(device, "psb_read");
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let other = device.new_texture(&d).unwrap();
        other
            .replace_region(
                Region::new_2d(0, 0, SIDE, SIDE),
                0,
                0,
                &before(),
                SIDE * 4,
                0,
            )
            .unwrap();
        let rt = Start::Scope(BarrierScope::RENDER_TARGETS);
        for (name, target, _buffer) in targets(device) {
            for (what, load, texture) in [
                ("loaded, reading itself", LoadAction::Load, &target),
                (
                    "cleared, reading another texture",
                    LoadAction::Clear,
                    &other,
                ),
            ] {
                let name = format!("{name}, {what}");
                let outside = match load {
                    LoadAction::Clear => CLEAR,
                    _ => [BEFORE, 0, 0, 255],
                };
                let mut costs = Vec::new();
                for start in [Start::Nothing, rt] {
                    if load == LoadAction::Load {
                        target
                            .replace_region(
                                Region::new_2d(0, 0, SIDE, SIDE),
                                0,
                                0,
                                &before(),
                                SIDE * 4,
                                0,
                            )
                            .unwrap();
                    }
                    let counted = device.work_counts();
                    let bytes =
                        started_pass(&queue, &target, (load, start, None), (&read, texture));
                    let after = device.work_counts();
                    hold_area(&name, &bytes, reordered([BEFORE, 0, 0, 255]), outside);
                    let moved = moved(&counted, &after);
                    let [splits, begun] = instances(&counted, &after);
                    eprintln!(
                        "{name}, {start:?}: instance splits {splits}, begun {begun}, shadow \
                         copies {}, snapshotted {}",
                        moved.copies, moved.snapshotted
                    );
                    assert_eq!(splits, 0, "{name}, {start:?}: a split");
                    assert_eq!(moved.snapshotted, 0, "{name}, {start:?}: a snapshot");
                    costs.push((begun, moved.copies));
                }
                assert_eq!(costs[0], costs[1], "{name}: the barrier cost something");
            }
        }
    });
}
