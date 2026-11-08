//! A private render target read at its own pixels after a memory barrier, whose image lacks
//! `STORAGE` (`DeviceShared::storage_costs_forced`: the device lays it out otherwise with it, as
//! ANV lays out an optimal `BGRA8Unorm` image), on a device that takes the snapshot path
//! (`render/snapshot.rs`): its first such pass's reader takes the snapshot pinned to the target's
//! own image, which a later fetch would otherwise render through a stand-in the snapshot does not
//! see, and the target takes its storage form at that pass's commit where the device makes one
//! (`texture/storage_form.rs`), so every later pass's reader takes the snapshot unpinned. Elsewhere
//! every reader splits, and the target swaps only where a pass fetches it through the pixel
//! interlock into a scratch for want of `STORAGE` (MoltenVK's fetch). Each pass, a command buffer of its own over the
//! `BGRA8Unorm` target, fills the block [`BLOCK`] with a red of its own, then after a barrier draws
//! the block through the whole target's scissor with `tob_read`, each fragment writing the texel
//! it read: the pass's own red at every pixel of the block, as Metal orders the barrier, where a
//! read of the copy from before the pass would write the red before it.
//!
//! A pass whose reader pinned the target and that then fetches the target (`fetch_ordered` over
//! [`FETCHED`], halving its red) renders it through a stand-in from the fetch on: the stand-in
//! takes the target's texels after the reader's instance (`WorkCounts::stand_ins_after_pins`),
//! so the reader's pixels outside the fetch keep the pass's red, and those inside it hold half of
//! it, where a stand-in for the whole pass would leave the reader's snapshot the red before the
//! pass.
//!
//! The pipelines are made first, by the same passes over an untracked target (which never takes
//! its storage form): a draw whose `VkPipeline` the device's pool is still making is recorded by
//! a host step, and its command buffer takes no switch.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll`, and `fetch_ordered` of
//! `validation/fixtures/public/engine/fragment_fetch.ll` (`llvm-as`).

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

/// The target's side, in pixels: large enough that a reader through the whole target's scissor
/// takes the snapshot by the area rule (`render/snapshot.rs`).
const SIDE: u64 = 1024;

/// The target's format, a guest framebuffer's.
const FORMAT: PixelFormat = PixelFormat::BGRA8Unorm;

/// The block each pass fills and reads, `[x0, y0, x1, y1]`.
const BLOCK: [u64; 4] = [8, 8, 16, 16];

/// The block a fetching draw halves the red of, `[x0, y0, x1, y1]`: over a quarter of
/// [`BLOCK`] and past it.
const FETCHED: [u64; 4] = [12, 4, 24, 12];

/// The red the first pass clears the target to.
const CLEARED: u8 = 20;

/// The passes over the target.
const PASSES: u8 = 3;

/// The red pass `pass` fills the block with.
fn red_of(pass: u8) -> u8 {
    60 + 50 * pass
}

/// The state of `tob_rect` and the fixture `fixture`'s `fragment`, into a color of [`FORMAT`],
/// where the device makes it.
fn state_of(
    device: &Device,
    fixture: &str,
    fragment: &str,
) -> Result<RenderPipelineState, crate::Error> {
    let function = |path: &str, name: &str| {
        let function = library(device, path).new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let vertex = function("engine/tob_renders.ll", "tob_rect");
    let mut descriptor = RenderPipelineDescriptor::new(vertex);
    descriptor.fragment = Some(function(fixture, fragment));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(FORMAT));
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// The state of `tob_rect` and `tob_renders.ll`'s `fragment`, into a color of [`FORMAT`].
fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    state_of(device, "engine/tob_renders.ll", fragment).unwrap()
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

/// The scissor rectangle of the target's pixels `[x0, x1) x [y0, y1)`.
fn scissor([x0, y0, x1, y1]: [u64; 4]) -> ScissorRect {
    ScissorRect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

/// Pass `pass` over `target` in a command buffer of its own, committed and completed, a draw of
/// `fetch` over [`FETCHED`] after the reader where it is given: the target's red at each pixel
/// after it, and the device's work counts as it completed.
fn pass_over(
    queue: &CommandQueue,
    target: &Texture,
    (fill, read, fetch): (
        &RenderPipelineState,
        &RenderPipelineState,
        Option<&RenderPipelineState>,
    ),
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
    encoder.set_render_pipeline_state(read).unwrap();
    encoder.set_fragment_texture(Some(target), 0).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    if let Some(fetch) = fetch {
        encoder.set_scissor_rect(scissor(FETCHED)).unwrap();
        encoder.set_render_pipeline_state(fetch).unwrap();
        encoder.set_vertex_bytes(&corners(FETCHED), 0).unwrap();
        (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    }
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

/// The device's answers for a target of [`target`]'s whose image lacks `STORAGE`: whether it
/// takes the snapshot path (`render/snapshot.rs`: the snapshot draws' features, and a fetch in the
/// pixel interlock, as a pass that fetches through input attachments takes no snapshot), and
/// whether the target then takes its storage form (`texture/storage_form.rs`).
fn snapshots(device: &Device, target: &Texture) -> (bool, bool) {
    let profile = device.compile_device().profile();
    let plan = target.image_plan().unwrap().clone();
    assert!(
        !plan.usage.contains(ImageUsageFlags::STORAGE),
        "the target's image lacks STORAGE"
    );
    let snapshots = device.shared().color_write_masks.is_some() && through_storage(profile);
    let forms = snapshots && storage_candidate(&plan, profile).is_some();
    let feedback = plan
        .usage
        .contains(ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT);
    eprintln!(
        "SELF-READ-STORAGE {}: snapshots {snapshots}, storage forms {forms}, feedback {feedback}",
        device.name()
    );
    (snapshots, forms)
}

/// What pass `pass` moved of the counts between `before` and `after`: its splits, readers through
/// snapshots, those pinned, passes whose stand-ins arrived after a pin, and storage forms taken.
fn counts_moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    [
        after.attachment_read_splits - before.attachment_read_splits,
        after.reading_draws_snapshotted - before.reading_draws_snapshotted,
        after.reading_draws_pinned - before.reading_draws_pinned,
        after.stand_ins_after_pins - before.stand_ins_after_pins,
        after.storage_forms_taken - before.storage_forms_taken,
    ]
}

/// The red `fetched` passes leave at `(x, y)` of a target cleared to [`CLEARED`]: the last pass's
/// red inside [`BLOCK`], halved inside [`FETCHED`] by each pass that fetched.
fn red_after(pass: u8, fetched: bool, x: u64, y: u64) -> u8 {
    let within = |[x0, y0, x1, y1]: [u64; 4]| (x0..x1).contains(&x) && (y0..y1).contains(&y);
    let mut red = CLEARED;
    for at in 0..=pass {
        if within(BLOCK) {
            red = red_of(at);
        }
        if fetched && within(FETCHED) {
            red /= 2;
        }
    }
    red
}

#[test]
fn a_private_targets_readers_take_the_snapshot_pinned_until_it_takes_its_storage_form() {
    let cache = Scratch::new("self-read-storage");
    let device = device(&cache.0);
    device
        .shared()
        .storage_costs_forced
        .store(true, Ordering::Relaxed);
    let queue = device.new_command_queue().unwrap();
    let states = (
        &state(&device, "tob_solid"),
        &state(&device, "tob_read"),
        None,
    );
    let warm = target(&device, false);
    for pass in 0..2 {
        pass_over(&queue, &warm, states, pass);
    }
    settled(&device);
    let target = target(&device, true);
    let (snapshots, forms) = snapshots(&device, &target);
    let mut before = device.work_counts();
    for pass in 0..PASSES {
        let pooled = before.pipelines_pooled;
        let (reds, after) = pass_over(&queue, &target, states, pass);
        let moved = counts_moved(&before, &after);
        eprintln!(
            "SELF-READ-STORAGE pass {pass}: (splits, snapshotted, pinned, stand-ins after pins, \
             storage forms) {moved:?}, pipelines pooled {}, inline passes {}",
            after.pipelines_pooled - pooled,
            after.inline_passes - before.inline_passes
        );
        for (i, &red) in reds.iter().enumerate() {
            let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
            assert_eq!(
                red,
                red_after(pass, false, x, y),
                "pass {pass}: pixel ({x}, {y})"
            );
        }
        // The first pass's reader pins the target, which takes its storage form at its commit
        // where the device makes one; a later pass's reader pins it where it took none.
        let want = match (snapshots, pass) {
            (true, 0) => [0, 1, 1, 0, u64::from(forms)],
            (true, _) => [0, 1, u64::from(!forms), 0, 0],
            (false, _) => [1, 0, 0, 0, 0],
        };
        assert_eq!(
            moved,
            want,
            "pass {pass} on {}: splits, readers through snapshots, pinned, stand-ins after \
             pins, storage forms taken",
            device.name()
        );
        before = after;
    }
    let image = target.image_plan().unwrap();
    assert_eq!(
        image.usage.contains(ImageUsageFlags::STORAGE),
        forms,
        "the target's image has STORAGE after its passes"
    );
}

#[test]
fn a_fetch_after_a_pinned_reader_renders_through_a_stand_in_from_the_fetch_on() {
    let cache = Scratch::new("self-read-pinned-fetch");
    let device = device(&cache.0);
    device
        .shared()
        .storage_costs_forced
        .store(true, Ordering::Relaxed);
    let queue = device.new_command_queue().unwrap();
    let fetch = match state_of(&device, "engine/fragment_fetch.ll", "fetch_ordered") {
        Ok(fetch) => fetch,
        Err(error) => {
            eprintln!(
                "SELF-READ-PINNED-FETCH {}: no framebuffer fetch on this device: {error}",
                device.name()
            );
            return;
        }
    };
    let states = (
        &state(&device, "tob_solid"),
        &state(&device, "tob_read"),
        Some(&fetch),
    );
    let warm = target(&device, false);
    for pass in 0..2 {
        pass_over(&queue, &warm, states, pass);
    }
    settled(&device);
    let target = target(&device, true);
    let (snapshots, forms) = snapshots(&device, &target);
    // The fetch notes the target for its storage form wherever it renders it into a scratch for
    // want of `STORAGE`, a device without the snapshot path included (`render.rs`, the fetch's
    // stand-in): where the device's fetch reads the image through the pixel interlock, which is
    // where the plan has a storage candidate.
    let plan = target.image_plan().unwrap().clone();
    let fetch_forms = storage_candidate(&plan, device.compile_device().profile()).is_some();
    let mut before = device.work_counts();
    for pass in 0..2 {
        let (reds, after) = pass_over(&queue, &target, states, pass);
        let moved = counts_moved(&before, &after);
        eprintln!(
            "SELF-READ-PINNED-FETCH pass {pass} (fetch forms {fetch_forms}): (splits, snapshotted, pinned, stand-ins after \
             pins, storage forms) {moved:?}, inline passes {}",
            after.inline_passes - before.inline_passes
        );
        for (i, &red) in reds.iter().enumerate() {
            let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
            assert_eq!(
                red,
                red_after(pass, true, x, y),
                "pass {pass}: pixel ({x}, {y})"
            );
        }
        // The first pass's reader pins the target and its fetch takes the stand-in after the
        // reader's instance; a later pass's fetch renders the target in place where it took its
        // storage form. Without the snapshot path every reader splits, and the first pass's fetch
        // still takes the storage form where the device fetches through the interlock.
        let pinned = pass == 0 || !forms;
        let want = match snapshots {
            true => [
                0,
                1,
                u64::from(pinned),
                u64::from(pinned),
                u64::from(pass == 0 && forms),
            ],
            false => [1, 0, 0, 0, u64::from(pass == 0 && fetch_forms)],
        };
        assert_eq!(
            moved,
            want,
            "pass {pass} on {}: splits, readers through snapshots, pinned, stand-ins after \
             pins, storage forms taken",
            device.name()
        );
        before = after;
    }
}
