//! Device tests of a draw that reads, at its own pixel, a texture its own render pass instance
//! renders, under the validation layer. After a memory barrier over the render targets, on a
//! device that enabled `attachmentFeedbackLoopLayout`, `extendedDynamicState3ColorWriteMask`,
//! `fragmentShaderPixelInterlock` and `fragmentStoresAndAtomics`, and whose passes fetch colors in
//! the interlock rather than as input attachments (lavapipe fetches those), the draw stays in its
//! instance:
//! drawn twice in its subpass, it copies its fragments' texels into a snapshot of the texture and
//! then reads the snapshot (`render/snapshot.rs`, `WorkCounts::reading_draws_snapshotted`; a draw
//! whose instance recorded inline records into a new one, `snapshot_inline_splits`). Elsewhere,
//! and with any of those features withheld, it records into a new instance, which reads a copy
//! taken after the draws before it (`attachment_read_splits`). Both answer as a Mac2 GPU orders
//! the barrier: what every draw before it wrote, and nothing the draw itself writes.
//!
//! The target is a private [`SIDE`]-square texture, `RGBA8Unorm` but where a test names another
//! format, that an earlier pass clears to red [`CLEARED`]. In the pass, which loads it, draw F
//! fills the block `[8, 16)²` with red [`FILL`]. The reader is `tob_read`
//! (`t.read(uint2(in.position.xy))`) over the block, blended `One + One`, so each fragment adds
//! what it read to the pixel, through the whole target's scissor but where a test names another:
//! the area rule (`render/snapshot.rs`) takes the snapshot through scissors of a 4-byte color
//! this large whatever the reader covers, the split's copy of the target's 1,048,576 texels
//! weighing more than the snapshot draw's 20 bytes at each pixel would at most, each beside a
//! reader's fixed cost; and the snapshot of an `RGBA8Unorm` texture holds its codes, 4 bytes a
//! pixel. The tests
//! run under the default policy, the consumer's, and the first under the lane's exact policy too,
//! whose exact texel conversion the translator follows into its call
//! (`StageModule::snapshot_textures`).
//!
//! - [`overlapping_primitives_read_what_the_draws_before_the_barrier_wrote`]: F, a barrier, the
//!   reader in two instances (every pixel covered by two primitives of the one draw), a barrier,
//!   the reader once more: `3 * FILL`, then `6 * FILL`. A read of the attachment itself would
//!   race with the draw's own color writes, the second primitive reading the first's (`4 * FILL`).
//! - [`a_reader_that_discards_by_its_texel_takes_the_snapshot`]: the first test's draws with a
//!   reader that discards its fragment where the texel it read is redder than 0.4
//!   (`sd_read_discarding` of `snapshot_discard.metal`): the first reader adds `FILL` twice
//!   (`3 * FILL`), the second reads that and discards (`3 * FILL`). Its branch only decides
//!   whether the invocation demotes itself, which a helper invocation already is, so the
//!   translator lets a snapshot serve its slot (`StageModule::snapshot_textures`): a snapshot for
//!   each reader where the device takes the path, else a split for each, and never a split for an
//!   own-pixel read no snapshot serves (`WorkCounts::reading_draws_split_unreadable_own_pixel`).
//! - [`a_reader_that_ends_itself_by_its_texel_takes_the_split_for_its_branch`]: the same draws on
//!   a device that withholds `shaderDemoteToHelperInvocation`, where the discard ends the
//!   invocation: the same answers, through a split for each reader on every device, each an
//!   own-pixel read no snapshot serves, counted under its branch alone
//!   (`WorkCounts::reading_draws_split_unreadable_own_pixel_branch`; `SnapshotRefusal::Branch`).
//! - [`readers_take_the_snapshot_by_the_area_rule`]: the same draws through each scissor,
//!   whatever the snapshot's form (`render/snapshot.rs`'s area rule, each form beside a reader's
//!   fixed cost of a mebibyte): over `RGBA8Unorm` through the whole target's, a snapshot for each
//!   where the device takes the path; through [`QUARTER`]'s (256 pixels of 20 bytes against the
//!   split's copy of the 18x18 texels around them, 4 bytes each), a split for each; over
//!   `RGBA16Float` through [`QUARTER`]'s (the copy's texels of 8 bytes), a snapshot for each where
//!   the device takes the path.
//! - [`a_surfaces_readers_take_the_snapshot_at_any_area`]: the same draws over a surface over
//!   host pages, an `RGBA8Unorm` texture of an IOSurface plane over a no-copy buffer, through the
//!   whole target's scissor: a snapshot for each where the device takes the path and the image
//!   the pass renders took feedback loop usage, the texture's shadow (a surface over host pages
//!   that renders is one where the device's check did not pass for it, `in_place.rs`;
//!   `WorkCounts::feedback_loop_shadow_images`) or a linear image aliasing the bytes
//!   (`WorkCounts::feedback_loop_alias_images`), with the storage usage the snapshot's fetch reads
//!   through, which one over host pages takes where the check passed for a shader's writes there
//!   (`Pointer writes: Passed`), as
//!   `RenderCommandEncoder::snapshot_cheaper` weighs no area for either; else a split.
//! - [`each_form_reads_what_the_split_reads`]: the first test's draws over a target of each of
//!   `RGBA8Unorm`, `BGRA8Unorm`, `R8Unorm`, `RG8Unorm` (8-bit codes), `RGBA16Float` (half-float
//!   codes) and `RGB10A2Unorm` (bits): the bytes the split leaves, byte for byte, with a snapshot
//!   for each reader where the device takes the path, the rule does (not `R8Unorm`'s or
//!   `RG8Unorm`'s, whose copies of 1 and 2 bytes a texel weigh less) and the device keeps the
//!   color in place (an `RGBA8Unorm` one's on every such device; the counts printed say which
//!   others did).
//! - [`without_a_barrier_the_readers_read_the_texels_from_before_the_pass`]: the same draws with
//!   no barrier: every reader reads [`CLEARED`], the texel from before the pass at its own pixel,
//!   through the copy taken before the pass: `FILL + 3 * CLEARED`, no snapshot. The M3 Max's
//!   answer (probe `overlap.m` under macOS 27.0.1: 232 at every block pixel at sides 32, 256
//!   and 1024, with the barriers or without, as Apple GPUs do not support that scope).
//! - [`with_a_feature_withheld_the_readers_take_the_split`]: the first test's draws, with each of
//!   the three features withheld in turn: the same answers, through a split for each reader.
//! - [`a_reader_in_a_pass_with_depth_or_stencil_takes_the_split`]: F, a barrier and the reader in
//!   two instances, in a pass that also loads a `Depth32Float_Stencil8` attachment cleared to depth
//!   1 and stencil 0 (both draws at depth 0): a split for the reader, as the binder makes no
//!   snapshot variant for a pass with a depth or stencil attachment (`render/snapshot.rs`). Under a
//!   `Less` depth with writes, and under a stencil test that blends each pixel once (`Equal` to
//!   reference 0, pass operation `IncrementClamp`), only the first instance's fragment draws at
//!   each pixel (`2 * FILL`); a snapshot draw that kept the depth/stencil state would write the
//!   depth or stencil first and leave the drawing draw nothing (`FILL`). Under a `Less` depth
//!   without writes both instances draw (`3 * FILL`).

use super::attachment_reads::{floats, function};
use super::degrade_notices::on_device_with;
use super::host_memory::{no_copy, Mapping, GUEST_PAGE};
use super::in_place_rows;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BarrierScope, BlendFactor, CompareFunction, PixelFormat, PrimitiveType, RenderStages,
    StencilOperation, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Device, DeviceOptions, IoSurfacePlane,
    LoadAction, Origin, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ScissorRect, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The target's red before the pass.
const CLEARED: u8 = 64;

/// The red draw F writes over the block.
const FILL: u8 = 40;

/// The target's side, in pixels: large enough that a reader of a 4-byte color through the whole
/// target's scissor takes the snapshot by the area rule (`render/snapshot.rs`).
const SIDE: u64 = 1024;

/// The block F and the readers draw over.
const BLOCK: [u64; 4] = [8, 8, 16, 16];

/// A scissor of a quarter of the target, over the block.
const QUARTER: [u64; 4] = [8, 8, 24, 24];

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    floats([x(x0), y(y1), x(x1), y(y0)])
}

/// The red of pixel `(x, y)` of `bytes`, tight `RGBA8Unorm` rows of [`SIDE`] pixels.
fn red(bytes: &[u8], x: u64, y: u64) -> u8 {
    bytes[((y * SIDE + x) * 4) as usize]
}

/// A fragment function: its name, and the fixture of `fixtures/public/engine` that holds it.
type Fragment = (&'static str, &'static str);

/// F's: the solid color of its fragment bytes.
const SOLID: Fragment = ("tob_solid", "tob_renders.ll");

/// The reader: `t.read(uint2(in.position.xy))`.
const READ: Fragment = ("tob_read", "tob_renders.ll");

/// The reader that discards its fragment where the texel it read is redder than 0.4.
const READ_DISCARDING: Fragment = ("sd_read_discarding", "snapshot_discard.ll");

/// The features the snapshot path takes.
const FEATURES: [VkFeature; 4] = [
    VkFeature::AttachmentFeedbackLoopLayout,
    VkFeature::ExtendedDynamicState3ColorWriteMask,
    VkFeature::FragmentShaderPixelInterlock,
    VkFeature::FragmentStoresAndAtomics,
];

/// The `tob_rect` state with the fragment function `fragment`, into a color of `format`, blended
/// `One + One` where `adds`.
fn state(
    device: &Device,
    fragment: Fragment,
    adds: bool,
    format: PixelFormat,
) -> RenderPipelineState {
    state_in(device, fragment, adds, format, None)
}

/// [`state`]'s, in a pass whose depth and stencil attachments are of `depth_stencil` where given.
fn state_in(
    device: &Device,
    (name, fixture): Fragment,
    adds: bool,
    format: PixelFormat,
    depth_stencil: Option<PixelFormat>,
) -> RenderPipelineState {
    let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
    d.depth_format = depth_stencil;
    d.stencil_format = depth_stencil;
    d.fragment = Some(function(device, name, fixture));
    let mut color = ColorAttachmentDescriptor::new(format);
    if adds {
        color.blending_enabled = true;
        color.source_rgb_blend_factor = BlendFactor::One;
        color.destination_rgb_blend_factor = BlendFactor::One;
        color.source_alpha_blend_factor = BlendFactor::One;
        color.destination_alpha_blend_factor = BlendFactor::One;
    }
    d.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// What the pass moved of the counts: reads through snapshots, of them those recorded into a
/// new instance as their own recorded inline, splits for attachment reads, and reads through
/// copies.
#[derive(Debug, PartialEq, Eq)]
struct Moved {
    snapshotted: u64,
    inline_splits: u64,
    splits: u64,
    copies: u64,
}

fn moved(before: &WorkCounts, after: &WorkCounts) -> Moved {
    Moved {
        snapshotted: after.reading_draws_snapshotted - before.reading_draws_snapshotted,
        inline_splits: after.snapshot_inline_splits - before.snapshot_inline_splits,
        splits: after.attachment_read_splits - before.attachment_read_splits,
        copies: after.attachment_reads_through_copies - before.attachment_reads_through_copies,
    }
}

/// The bytes of a texel of `format`, of the formats the tests render.
fn texel_bytes(format: PixelFormat) -> u64 {
    match format {
        PixelFormat::R8Unorm => 1,
        PixelFormat::RG8Unorm => 2,
        PixelFormat::RGBA16Float => 8,
        _ => 4,
    }
}

/// A clear of a new private target of `format` to [`CLEARED`], then one pass that loads it: F,
/// then the reader in two instances and once more, a barrier before each where `barriers`,
/// through `scissor` (`[x0, y0, x1, y1]`; else the whole target's). The target's bytes after it,
/// tight rows, and what the pass moved of the counts.
fn readers(
    device: &Device,
    queue: &CommandQueue,
    format: PixelFormat,
    barriers: bool,
    scissor: Option<[u64; 4]>,
) -> (Vec<u8>, Moved) {
    let (bytes, moved, _) = readers_over(device, queue, format, barriers, scissor, false, READ);
    (bytes, moved)
}

/// How [`readers_over`]' target was made: whether the image its passes render took feedback loop
/// usage (`WorkCounts::feedback_loop_alias_images` for an image aliasing its bytes, or
/// `feedback_loop_shadow_images` for a shadow), and whether a snapshot's fetch reads it there: a
/// shadow's, or an image over host pages in place where the device's check passed for a shader's
/// writes over such pages (`Pointer writes: Passed`), with which it takes the storage usage the
/// fetch reads through (`texture/in_place_image.rs`).
#[derive(Debug)]
struct Made {
    feedback: bool,
    fetched: bool,
}

/// [`readers`]' pass over a new target of `format`: private, or where `surface` a surface over
/// host pages (a texture of an IOSurface plane over a no-copy buffer, tight rows), as the
/// consumer's guest makes its framebuffer, with the readers drawn by `reader`. Also how the
/// target was made.
fn readers_over(
    device: &Device,
    queue: &CommandQueue,
    format: PixelFormat,
    barriers: bool,
    scissor: Option<[u64; 4]>,
    surface: bool,
    reader: Fragment,
) -> (Vec<u8>, Moved, Made) {
    let fill = state(device, SOLID, false, format);
    let read = state(device, reader, true, format);
    let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let row = SIDE * texel_bytes(format);
    let len = ((SIDE * row) as usize).next_multiple_of(GUEST_PAGE);
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
            let plane = IoSurfacePlane::new(0, row, false);
            let texture = buffer.new_texture_with_iosurface_plane(&d, plane).unwrap();
            (texture, Some(buffer))
        }
    };
    let counts = device.work_counts();
    let alias = counts.feedback_loop_alias_images > made.feedback_loop_alias_images;
    let shadow = counts.feedback_loop_shadow_images > made.feedback_loop_shadow_images;
    // An image in place over host pages takes the storage usage the snapshot's fetch reads
    // through where the device's check passed for a shader's writes over such pages, for texels
    // that lie in lines as the surface's do: its rows one after another from its pages' start.
    let on_lines = in_place_rows::on_lines(device, 0, row, row, SIDE);
    let writes = in_place_rows::passed(device, "Pointer writes", on_lines);
    let made = Made {
        feedback: alias || shadow,
        fetched: shadow || (alias && (!surface || writes)),
    };
    let back = device
        .new_buffer(SIDE * row, ResourceOptions::shared())
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(f64::from(CLEARED) / 255.0, 0.0, 0.0, 1.0);
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    pass.color_attachments[0].load_action = LoadAction::Load;
    let counted = device.work_counts();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&fill).unwrap();
    encoder.set_vertex_bytes(&corners(BLOCK), 0).unwrap();
    let red = f32::from(FILL) / 255.0;
    encoder
        .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.set_render_pipeline_state(&read).unwrap();
    encoder.set_fragment_texture(Some(&target), 0).unwrap();
    if let Some([x0, y0, x1, y1]) = scissor {
        let rect = ScissorRect {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        };
        encoder.set_scissor_rect(rect).unwrap();
    }
    for instances in [2, 1] {
        if barriers {
            encoder.memory_barrier_with_scope(
                BarrierScope::RENDER_TARGETS,
                RenderStages::FRAGMENT,
                RenderStages::FRAGMENT,
            );
        }
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, instances, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        row,
        SIDE * row,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let moved = moved(&counted, &device.work_counts());
    let contents = back.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    let bytes =
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), back.length() as usize) }.to_vec();
    (bytes, moved, made)
}

/// [`readers`]' first test's pass over a private `RGBA8Unorm` target, through the whole
/// target's scissor, beside a `Depth32Float_Stencil8` attachment that an earlier pass clears to depth 1 and
/// stencil 0: F, under Metal's default depth/stencil state, then a barrier and the reader in two
/// instances under a state of `reader` (stencil reference 0). The target's bytes, tight rows, and
/// what the pass moved of the counts.
fn depth_stencil_readers(
    device: &Device,
    queue: &CommandQueue,
    reader: &DepthStencilDescriptor,
) -> (Vec<u8>, Moved) {
    let format = PixelFormat::RGBA8Unorm;
    let depth_format = PixelFormat::Depth32FloatStencil8;
    let fill = state_in(device, SOLID, false, format, Some(depth_format));
    let read = state_in(device, READ, true, format, Some(depth_format));
    let reader = device.new_depth_stencil_state(reader).unwrap();
    let texture = |format: PixelFormat, usage: TextureUsage| {
        let mut d = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        device.new_texture(&d).unwrap()
    };
    let target = texture(
        format,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    );
    let depth = texture(depth_format, TextureUsage::RENDER_TARGET);
    let row = SIDE * texel_bytes(format);
    let back = device
        .new_buffer(SIDE * row, ResourceOptions::shared())
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(f64::from(CLEARED) / 255.0, 0.0, 0.0, 1.0);
    color.store_action = StoreAction::Store;
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = StoreAction::Store;
    pass.stencil_attachment.texture = Some(depth);
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.clear_stencil = 0;
    pass.stencil_attachment.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    pass.color_attachments[0].load_action = LoadAction::Load;
    pass.depth_attachment.load_action = LoadAction::Load;
    pass.stencil_attachment.load_action = LoadAction::Load;
    let counted = device.work_counts();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&fill).unwrap();
    encoder.set_vertex_bytes(&corners(BLOCK), 0).unwrap();
    let red = f32::from(FILL) / 255.0;
    encoder
        .set_fragment_bytes(&floats([red, 0.0, 0.0, 1.0]), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.set_render_pipeline_state(&read).unwrap();
    encoder.set_fragment_texture(Some(&target), 0).unwrap();
    encoder.set_depth_stencil_state(&reader).unwrap();
    encoder.set_stencil_reference_value(0);
    encoder.memory_barrier_with_scope(
        BarrierScope::RENDER_TARGETS,
        RenderStages::FRAGMENT,
        RenderStages::FRAGMENT,
    );
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 2, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        row,
        SIDE * row,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let moved = moved(&counted, &device.work_counts());
    let contents = back.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    let bytes =
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), back.length() as usize) }.to_vec();
    (bytes, moved)
}

/// Hold `bytes` to `block` red at every pixel of the block and [`CLEARED`] elsewhere.
fn holds(what: &str, bytes: &[u8], block: u8) {
    let [x0, y0, x1, y1] = BLOCK;
    for y in 0..SIDE {
        for x in 0..SIDE {
            let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
            let want = if inside { block } else { CLEARED };
            assert_eq!(red(bytes, x, y), want, "{what}: pixel ({x}, {y})");
        }
    }
}

/// Whether `device` takes the snapshot path for the readers: it enabled every feature the path
/// takes, and its passes fetch color attachments in the interlock, not as input attachments
/// (`rasterizationOrderColorAttachmentAccess`), whose subpass would name the color in two layouts.
fn snapshots(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    FEATURES
        .iter()
        .all(|feature| device.enabled().features.contains(feature))
        && !profile.fragment().rasterization_order_color_access
}

/// The lane's device under the default policy, where the translator reads `tob_read`'s slot
/// through a snapshot (`StageModule::snapshot_textures`), with `adjust` applied too.
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
fn overlapping_primitives_read_what_the_draws_before_the_barrier_wrote() {
    for exact in [false, true] {
        on_device_with(
            |options| {
                if !exact {
                    options.policy = DeviceOptions::default().policy;
                }
            },
            |device, _messages| {
                let queue = device.new_command_queue().unwrap();
                let (bytes, moved) = readers(device, &queue, PixelFormat::RGBA8Unorm, true, None);
                let takes = snapshots(device);
                eprintln!("snapshot reads, features {takes}, exact {exact}: {moved:?}");
                holds("after barriers", &bytes, 6 * FILL);
                if takes {
                    // Each reader through a snapshot, the second inside the first's instance.
                    assert_eq!(
                        moved.snapshotted, 2,
                        "exact {exact}: readers through snapshots"
                    );
                    assert_eq!(moved.splits, 0, "exact {exact}: splits");
                    assert!(moved.inline_splits <= 1, "exact {exact}: inline splits");
                    assert_eq!(moved.copies, 0, "exact {exact}: reads through copies");
                } else {
                    assert_eq!(
                        moved.snapshotted, 0,
                        "exact {exact}: readers through snapshots"
                    );
                    assert_eq!(moved.splits, 2, "exact {exact}: a split for each reader");
                }
            },
        );
    }
}

#[test]
fn a_reader_that_discards_by_its_texel_takes_the_snapshot() {
    on_default_policy(
        |_| {},
        |device| {
            let queue = device.new_command_queue().unwrap();
            let format = PixelFormat::RGBA8Unorm;
            let before = device.work_counts();
            let (bytes, moved, _) =
                readers_over(device, &queue, format, true, None, false, READ_DISCARDING);
            let unreadable = device
                .work_counts()
                .reading_draws_split_unreadable_own_pixel
                - before.reading_draws_split_unreadable_own_pixel;
            let takes = snapshots(device);
            eprintln!(
                "snapshot reads, a discarding reader, features {takes}: {moved:?}, \
                 unreadable own-pixel splits {unreadable}"
            );
            holds("a discarding reader", &bytes, 3 * FILL);
            assert_eq!(
                unreadable, 0,
                "splits for an own-pixel read no snapshot serves"
            );
            let (snapshotted, splits) = if takes { (2, 0) } else { (0, 2) };
            assert_eq!(
                (moved.snapshotted, moved.splits),
                (snapshotted, splits),
                "readers through snapshots, splits"
            );
        },
    );
}

/// A clause of the unreadable own-pixel splits: its name and its count.
type Clause = (&'static str, fn(&WorkCounts) -> u64);

/// What `before` to `after` moved of the unreadable own-pixel splits' clauses
/// (`WorkCounts::reading_draws_split_unreadable_own_pixel_*`), by name.
fn unreadable_by(before: &WorkCounts, after: &WorkCounts) -> Vec<(&'static str, u64)> {
    let clauses: [Clause; 12] = [
        ("branch", |c| {
            c.reading_draws_split_unreadable_own_pixel_branch
        }),
        ("derivative", |c| {
            c.reading_draws_split_unreadable_own_pixel_derivative
        }),
        ("sample", |c| {
            c.reading_draws_split_unreadable_own_pixel_sample
        }),
        ("group_operation", |c| {
            c.reading_draws_split_unreadable_own_pixel_group_operation
        }),
        ("store", |c| {
            c.reading_draws_split_unreadable_own_pixel_store
        }),
        ("atomic", |c| {
            c.reading_draws_split_unreadable_own_pixel_atomic
        }),
        ("image_write", |c| {
            c.reading_draws_split_unreadable_own_pixel_image_write
        }),
        ("undefined_operand", |c| {
            c.reading_draws_split_unreadable_own_pixel_undefined_operand
        }),
        ("writes_memory", |c| {
            c.reading_draws_split_unreadable_own_pixel_writes_memory
        }),
        ("interlock", |c| {
            c.reading_draws_split_unreadable_own_pixel_interlock
        }),
        ("coherent", |c| {
            c.reading_draws_split_unreadable_own_pixel_coherent
        }),
        ("other", |c| {
            c.reading_draws_split_unreadable_own_pixel_other
        }),
    ];
    (clauses.iter())
        .map(|&(name, count)| (name, count(after) - count(before)))
        .collect()
}

#[test]
fn a_reader_that_ends_itself_by_its_texel_takes_the_split_for_its_branch() {
    on_default_policy(
        |options| {
            (options.withheld_features).insert(VkFeature::ShaderDemoteToHelperInvocation);
        },
        |device| {
            let queue = device.new_command_queue().unwrap();
            let format = PixelFormat::RGBA8Unorm;
            let before = device.work_counts();
            let (bytes, moved, _) =
                readers_over(device, &queue, format, true, None, false, READ_DISCARDING);
            let after = device.work_counts();
            let unreadable = after.reading_draws_split_unreadable_own_pixel
                - before.reading_draws_split_unreadable_own_pixel;
            let clauses = unreadable_by(&before, &after);
            eprintln!(
                "snapshot reads, a reader that ends itself: {moved:?}, unreadable own-pixel \
                 splits {unreadable}, by clause {clauses:?}"
            );
            holds("a reader that ends itself", &bytes, 3 * FILL);
            assert_eq!(
                (moved.snapshotted, moved.splits),
                (0, 2),
                "readers through snapshots, splits"
            );
            assert_eq!(
                unreadable, 2,
                "splits for an own-pixel read no snapshot serves"
            );
            let named: Vec<_> = clauses.into_iter().filter(|&(_, n)| n != 0).collect();
            assert_eq!(named, [("branch", 2)], "the clauses that left the slot out");
        },
    );
}

#[test]
fn readers_take_the_snapshot_by_the_area_rule() {
    on_default_policy(
        |_| {},
        |device| {
            let queue = device.new_command_queue().unwrap();
            let narrow = PixelFormat::RGBA8Unorm;
            let (snapshotted, splits) = if snapshots(device) { (2, 0) } else { (0, 2) };
            let (bytes, moved) = readers(device, &queue, narrow, true, None);
            eprintln!("snapshot reads, whole scissor, RGBA8Unorm: {moved:?}");
            holds("whole scissor", &bytes, 6 * FILL);
            assert_eq!(
                (moved.snapshotted, moved.splits),
                (snapshotted, splits),
                "RGBA8Unorm over the target: snapshots, splits"
            );
            let (bytes, moved) = readers(device, &queue, narrow, true, Some(QUARTER));
            eprintln!("snapshot reads, quarter scissor, RGBA8Unorm: {moved:?}");
            holds("quarter scissor", &bytes, 6 * FILL);
            assert_eq!(
                (moved.snapshotted, moved.splits),
                (0, 2),
                "RGBA8Unorm over a quarter: snapshots, splits"
            );
            let wide = PixelFormat::RGBA16Float;
            let (_, moved) = readers(device, &queue, wide, true, Some(QUARTER));
            eprintln!("snapshot reads, quarter scissor, RGBA16Float: {moved:?}");
            assert_eq!(
                (moved.snapshotted, moved.splits),
                (snapshotted, splits),
                "RGBA16Float over a quarter: snapshots, splits"
            );
        },
    );
}

#[test]
fn a_surfaces_readers_take_the_snapshot_at_any_area() {
    on_default_policy(
        |_| {},
        |device| {
            let queue = device.new_command_queue().unwrap();
            let format = PixelFormat::RGBA8Unorm;
            let (bytes, moved, made) = readers_over(device, &queue, format, true, None, true, READ);
            let takes = snapshots(device);
            eprintln!("snapshot reads, surface, whole scissor: {made:?}, {moved:?}");
            holds("surface, whole scissor", &bytes, 6 * FILL);
            let expected = if takes && made.feedback && made.fetched {
                (2, 0)
            } else {
                (0, 2)
            };
            assert_eq!(
                (moved.snapshotted, moved.splits),
                expected,
                "a surface's readers over the target: snapshots, splits"
            );
        },
    );
}

#[test]
fn each_form_reads_what_the_split_reads() {
    for format in [
        PixelFormat::RGBA8Unorm,
        PixelFormat::BGRA8Unorm,
        PixelFormat::R8Unorm,
        PixelFormat::RG8Unorm,
        PixelFormat::RGBA16Float,
        PixelFormat::RGB10A2Unorm,
    ] {
        let mut split = None;
        on_default_policy(
            |options| {
                options
                    .withheld_features
                    .insert(VkFeature::AttachmentFeedbackLoopLayout);
            },
            |device| {
                let queue = device.new_command_queue().unwrap();
                let (bytes, moved) = readers(device, &queue, format, true, None);
                assert_eq!(moved.splits, 2, "{format:?}: a split for each reader");
                split = Some(bytes);
            },
        );
        on_default_policy(
            |_| {},
            |device| {
                let queue = device.new_command_queue().unwrap();
                let (bytes, moved) = readers(device, &queue, format, true, None);
                eprintln!("snapshot reads, {format:?}: {moved:?}");
                // Each reader through a snapshot where the device keeps the color in place (its
                // image has storage usage), as `RGBA8Unorm`'s on every device that takes the path.
                assert_eq!(
                    moved.snapshotted + moved.splits,
                    2,
                    "{format:?}: snapshots and splits"
                );
                if format == PixelFormat::RGBA8Unorm && snapshots(device) {
                    assert_eq!(
                        moved.snapshotted, 2,
                        "{format:?}: readers through snapshots"
                    );
                }
                assert!(
                    split.as_deref() == Some(&bytes[..]),
                    "{format:?}: the split's bytes"
                );
            },
        );
    }
}

#[test]
fn without_a_barrier_the_readers_read_the_texels_from_before_the_pass() {
    on_default_policy(
        |_| {},
        |device| {
            let queue = device.new_command_queue().unwrap();
            let (bytes, moved) =
                readers(device, &queue, PixelFormat::RGBA8Unorm, false, Some(BLOCK));
            eprintln!("snapshot reads, no barrier: {moved:?}");
            holds("without barriers", &bytes, FILL + 3 * CLEARED);
            assert_eq!(
                (moved.snapshotted, moved.splits),
                (0, 0),
                "snapshots, splits"
            );
            assert_eq!(moved.copies, 2, "each reader through the copy");
        },
    );
}

#[test]
fn with_a_feature_withheld_the_readers_take_the_split() {
    for withheld in &FEATURES[..3] {
        on_default_policy(
            |options| {
                options.withheld_features.insert(*withheld);
            },
            |device| {
                let queue = device.new_command_queue().unwrap();
                let (bytes, moved) = readers(device, &queue, PixelFormat::RGBA8Unorm, true, None);
                eprintln!("snapshot reads, {withheld:?} withheld: {moved:?}");
                holds(&format!("{withheld:?} withheld"), &bytes, 6 * FILL);
                assert_eq!(
                    moved.snapshotted, 0,
                    "{withheld:?}: readers through snapshots"
                );
                assert_eq!(moved.splits, 2, "{withheld:?}: a split for each reader");
            },
        );
    }
}

#[test]
fn a_reader_in_a_pass_with_depth_or_stencil_takes_the_split() {
    let mut depth_written = DepthStencilDescriptor::default();
    depth_written.depth_compare_function = CompareFunction::Less;
    depth_written.depth_write_enabled = true;
    let mut stencil_once = DepthStencilDescriptor::default();
    for face in [
        &mut stencil_once.front_face_stencil,
        &mut stencil_once.back_face_stencil,
    ] {
        face.stencil_compare_function = CompareFunction::Equal;
        face.depth_stencil_pass_operation = StencilOperation::IncrementClamp;
    }
    let mut depth_tested = DepthStencilDescriptor::default();
    depth_tested.depth_compare_function = CompareFunction::Less;
    on_default_policy(
        |_| {},
        |device| {
            let queue = device.new_command_queue().unwrap();
            for (what, reader, block) in [
                ("depth Less, written", &depth_written, 2 * FILL),
                ("stencil once", &stencil_once, 2 * FILL),
                ("depth Less, not written", &depth_tested, 3 * FILL),
            ] {
                let (bytes, moved) = depth_stencil_readers(device, &queue, reader);
                eprintln!("snapshot reads, {what}: {moved:?}");
                holds(what, &bytes, block);
                assert_eq!(
                    (moved.snapshotted, moved.splits),
                    (0, 1),
                    "{what}: snapshots, splits"
                );
            }
        },
    );
}
