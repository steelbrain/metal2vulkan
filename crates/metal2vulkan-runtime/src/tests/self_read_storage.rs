//! A private render target read at its own pixels after a memory barrier, whose image lacks
//! `STORAGE` (`DeviceShared::storage_costs_forced`: the device lays it out otherwise with it, as
//! ANV lays out an optimal `BGRA8Unorm` image), takes its storage form at the commit of its first
//! such pass (`render/snapshot.rs`, `texture/storage_form.rs`): that pass's reader splits it, as a
//! later fetch would render it through a stand-in its snapshot does not see, and every later
//! pass's reader takes the snapshot, on a device that takes the path. Elsewhere every reader
//! splits and nothing swaps. Each pass, a command buffer of its own over the `BGRA8Unorm` target,
//! fills the block [`BLOCK`] with a red of its own, then after a barrier draws the block through
//! its scissor with `tob_read`, each fragment writing the texel it read: the pass's own red at
//! every pixel of the block, as Metal orders the barrier, where a read of the copy from before
//! the pass would write the red before it.
//!
//! The fill's and the reader's pipelines are made first, by the same pass over an untracked
//! target (which never takes its storage form): a draw whose `VkPipeline` the device's pool is
//! still making is recorded by a host step, and its command buffer takes no switch.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::framebuffer_fetch::{storage_candidate, through_storage};
use crate::{
    BlitOption, CommandQueue, CpuCacheMode, Device, HazardTrackingMode, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size,
    StorageMode, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::ImageUsageFlags;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// The target's side, in pixels.
const SIDE: u64 = 32;

/// The target's format, a guest framebuffer's.
const FORMAT: PixelFormat = PixelFormat::BGRA8Unorm;

/// The block each pass fills and reads, `[x0, y0, x1, y1]`: its 64 pixels of the snapshot draw's
/// 20 bytes cost less than the split's copy of the target's 1024 texels of 4 bytes
/// (`RenderCommandEncoder::snapshot_cheaper`).
const BLOCK: [u64; 4] = [8, 8, 16, 16];

/// The red the first pass clears the target to.
const CLEARED: u8 = 20;

/// The passes over the target.
const PASSES: u8 = 3;

/// The red pass `pass` fills the block with.
fn red_of(pass: u8) -> u8 {
    60 + 50 * pass
}

/// The fixture's state of `tob_rect` and `fragment`, into a color of [`FORMAT`].
fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let function = |name: &str| {
        let function = library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(FORMAT));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A private [`FORMAT`] render target that shaders read, its hazards tracked where `tracked`.
fn target(device: &Device, tracked: bool) -> Texture {
    let mut d = TextureDescriptor::texture_2d(FORMAT, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = match tracked {
        true => ResourceOptions::private(),
        false => ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Untracked,
        ),
    };
    device.new_texture(&d).unwrap()
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / SIDE as f32 * 2.0;
    bytes_of(&[x(x0), y(y1), x(x1), y(y0)], f32::to_le_bytes)
}

/// Pass `pass` over `target` in a command buffer of its own, committed and completed: the
/// target's red at each pixel after it, and the device's work counts as it completed.
fn pass_over(
    queue: &CommandQueue,
    target: &Texture,
    (fill, read): (&RenderPipelineState, &RenderPipelineState),
    pass: u8,
) -> (Vec<u8>, WorkCounts) {
    let device = queue.device();
    let row = SIDE * 4;
    let back = device
        .new_buffer(SIDE * row, ResourceOptions::shared())
        .unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = match pass {
        0 => LoadAction::Clear,
        _ => LoadAction::Load,
    };
    color.clear_color = crate::ClearColor::new(f64::from(CLEARED) / 255.0, 0.0, 0.0, 1.0);
    color.store_action = StoreAction::Store;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    encoder.set_render_pipeline_state(fill).unwrap();
    encoder.set_vertex_bytes(&corners(BLOCK), 0).unwrap();
    let red = f32::from(red_of(pass)) / 255.0;
    let rgba = bytes_of(&[red, 0.0, 0.0, 1.0], f32::to_le_bytes);
    encoder.set_fragment_bytes(&rgba, 0).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    let fragment = RenderStages::FRAGMENT;
    encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
    let [x0, y0, x1, y1] = BLOCK;
    let scissor = ScissorRect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    };
    encoder.set_scissor_rect(scissor).unwrap();
    encoder.set_render_pipeline_state(read).unwrap();
    encoder.set_fragment_texture(Some(target), 0).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        row,
        SIDE * row,
        BlitOption::None,
    ))
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let counts = device.work_counts();
    let texels = super::merge_scratch::read(&back, SIDE * row);
    // `BGRA8Unorm`'s red is its third byte.
    (texels.chunks(4).map(|texel| texel[2]).collect(), counts)
}

/// Wait until the device's pipeline pool makes nothing.
fn settled(device: &Device) {
    let until = Instant::now() + WAIT;
    while device.work_counts().pipeline_pool_busy != 0 {
        assert!(Instant::now() < until, "the pipeline pool still busy");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_private_targets_readers_take_the_snapshot_once_it_took_its_storage_form() {
    let cache = Scratch::new("self-read-storage");
    let device = device(&cache.0);
    device
        .shared()
        .storage_costs_forced
        .store(true, Ordering::Relaxed);
    let queue = device.new_command_queue().unwrap();
    let states = (&state(&device, "tob_solid"), &state(&device, "tob_read"));
    let warm = target(&device, false);
    for pass in 0..2 {
        pass_over(&queue, &warm, states, pass);
    }
    settled(&device);
    let target = target(&device, true);
    let profile = device.compile_device().profile();
    let plan = target.image_plan().unwrap().clone();
    assert!(
        !plan.usage.contains(ImageUsageFlags::STORAGE),
        "the target's image lacks STORAGE"
    );
    // A device that takes the snapshot path (`render/snapshot.rs`) for an image of the target's
    // that has `STORAGE`, as the storage form gives it where the device makes one.
    let takes = device.shared().color_write_masks.is_some()
        && through_storage(profile)
        && storage_candidate(&plan, profile).is_some();
    let feedback = plan
        .usage
        .contains(ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT);
    eprintln!(
        "SELF-READ-STORAGE {}: takes {takes}, feedback {feedback}",
        device.name()
    );
    let mut before = device.work_counts();
    for pass in 0..PASSES {
        let pooled = before.pipelines_pooled;
        let (reds, after) = pass_over(&queue, &target, states, pass);
        let moved = (
            after.attachment_read_splits - before.attachment_read_splits,
            after.reading_draws_snapshotted - before.reading_draws_snapshotted,
            after.storage_forms_taken - before.storage_forms_taken,
        );
        eprintln!(
            "SELF-READ-STORAGE pass {pass}: (splits, snapshotted, storage forms) {moved:?}, \
             pipelines pooled {}",
            after.pipelines_pooled - pooled
        );
        let [x0, y0, x1, y1] = BLOCK;
        for (i, &red) in reds.iter().enumerate() {
            let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
            let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
            let want = if inside { red_of(pass) } else { CLEARED };
            assert_eq!(red, want, "pass {pass}: pixel ({x}, {y})");
        }
        let want = match (takes, pass) {
            (true, 0) => (1, 0, 1),
            (true, _) => (0, 1, 0),
            (false, _) => (1, 0, 0),
        };
        assert_eq!(
            moved,
            want,
            "pass {pass} on {}: splits, readers through snapshots, storage forms taken",
            device.name()
        );
        before = after;
    }
    let image = target.image_plan().unwrap();
    assert_eq!(
        image.usage.contains(ImageUsageFlags::STORAGE),
        takes,
        "the target's image has STORAGE after its passes"
    );
}
