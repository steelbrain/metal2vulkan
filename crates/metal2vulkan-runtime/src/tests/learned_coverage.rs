//! A reader of its pass's private color through no scissor, weighed by what it covered in the
//! pass before (`render/coverage.rs`), on a device that takes the snapshot path
//! (`render/snapshot.rs`) and measures readers (`occlusionQueryPrecise`, queries reset on the
//! host). Each pass, a command buffer of its own over a [`SIDE`]-square `BGRA8Unorm` target, fills
//! a rectangle with a red of its own, then after a barrier draws the rectangle with `tob_read`,
//! each fragment writing the texel it read: the pass's red at every pixel of the rectangle, as
//! Metal orders the barrier, whichever form the reader took. The rectangle is [`SPARSE`] for two
//! passes, the whole target for three, then [`SPARSE`] for two more.
//!
//! - The first pass has no count: the area rule takes the snapshot through the whole target's
//!   scissor. Each later pass takes the form the pass before's count chose: the snapshot over
//!   [`SPARSE`]'s pixels, the split over the whole target's. So the first pass after each change
//!   takes the form the count before it chose, one pass, and the next the new count's.
//! - A device without the snapshot path splits every reader and records no query; one that takes
//!   it but measures nothing weighs every reader by the area rule: the snapshot.
//!
//! An app's visibility query around a reader (`Counting`) keeps its count, the fill's samples and
//! the reader's: the reader takes the split, as the snapshot path refuses it while a visibility
//! query is active, and records no query of its own.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are `tob_rect`,
//! `tob_solid` and `tob_read` of the owned fixture `validation/fixtures/public/engine/tob_renders.ll`
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{bytes_of, library};
use crate::framebuffer_fetch::{through_inputs, through_storage};
use crate::{
    BlitOption, CommandQueue, CpuCacheMode, Device, HazardTrackingMode, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StorageMode,
    StoreAction, Texture, TextureDescriptor, VisibilityResultMode, WorkCounts,
};
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::time::{Duration, Instant};

/// The target's side, in pixels: through the whole target's scissor, a reader of a 4-byte color
/// with no count takes the snapshot by the area rule.
const SIDE: u64 = 1024;

/// The target's format.
const FORMAT: PixelFormat = PixelFormat::BGRA8Unorm;

/// The sparse reader's rectangle, `[x0, y0, x1, y1]`: 4096 pixels, whose snapshot draw costs
/// less than the split's copy of the target.
const SPARSE: [u64; 4] = [100, 200, 164, 264];

/// The whole target.
const WHOLE: [u64; 4] = [0, 0, SIDE, SIDE];

/// The red the first pass clears the target to.
const CLEARED: u8 = 20;

/// The red pass `pass` fills its rectangle with.
fn red_of(pass: usize) -> u8 {
    40 + 25 * pass as u8
}

/// The state of `tob_rect` and `tob_renders.ll`'s `fragment`, into a color of [`FORMAT`].
fn state(device: &Device, fragment: &str) -> RenderPipelineState {
    let function = |name: &str| {
        let function = library(device, "engine/tob_renders.ll")
            .new_function_with_name(name)
            .unwrap();
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

/// Pass `pass` over `target`, a command buffer of its own, committed and completed: a fill of
/// `rect` with the pass's red, a barrier, and the reader over `rect` through the whole target's
/// scissor, all inside a `Counting` visibility query at offset 0 of `visibility` where given.
/// The target's red at each pixel after it.
fn pass_over(
    queue: &CommandQueue,
    target: &Texture,
    (fill, read): (&RenderPipelineState, &RenderPipelineState),
    (pass, rect): (usize, [u64; 4]),
    visibility: Option<&crate::Buffer>,
) -> Vec<u8> {
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
    descriptor.visibility_result_buffer = visibility.cloned();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    if visibility.is_some() {
        (encoder.set_visibility_result_mode(VisibilityResultMode::Counting, 0)).unwrap();
    }
    encoder.set_render_pipeline_state(fill).unwrap();
    encoder.set_vertex_bytes(&corners(rect), 0).unwrap();
    let red = f32::from(red_of(pass)) / 255.0;
    let rgba = bytes_of(&[red, 0.0, 0.0, 1.0], f32::to_le_bytes);
    encoder.set_fragment_bytes(&rgba, 0).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    let fragment = RenderStages::FRAGMENT;
    encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, fragment, fragment);
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
    let texels = super::merge_scratch::read(&back, SIDE * row);
    // `BGRA8Unorm`'s red is its third byte.
    texels.chunks(4).map(|texel| texel[2]).collect()
}

/// Wait until the device's pipeline pool makes nothing.
fn settled(device: &Device) {
    let until = Instant::now() + WAIT;
    while device.work_counts().pipeline_pool_busy != 0 {
        assert!(Instant::now() < until, "the pipeline pool still busy");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The device's answers: whether it takes the snapshot path for a private target's reader
/// (`render/snapshot.rs`: the snapshot draws' features, and a fetch in the pixel interlock), and
/// whether it measures readers (`render/coverage.rs`: `occlusionQueryPrecise`, and queries reset
/// on the host).
fn answers(device: &Device) -> (bool, bool) {
    let profile = device.compile_device().profile();
    let snapshots = device.shared().color_write_masks.is_some() && through_storage(profile);
    let precise = device
        .properties()
        .description
        .has(VkFeature::OcclusionQueryPrecise);
    let host_reset = device.shared().timing.resets_on_host();
    eprintln!(
        "LEARNED-COVERAGE {}: snapshots {snapshots}, occlusionQueryPrecise {precise}, host \
         reset {host_reset}",
        device.name()
    );
    (snapshots, snapshots && precise && host_reset)
}

/// Why a reader on `device` takes the split where the device takes no snapshot path: the device
/// lacks the path's color write masks (`SplitReason::DeviceLacks`), else it fetches through input
/// attachments (`through_inputs`), a pass the path refuses by its shape
/// (`SplitReason::PassShape`).
fn lacks(device: &Device) -> [u64; 2] {
    let masks = device.shared().color_write_masks.is_some();
    let inputs = through_inputs(device.compile_device().profile());
    [u64::from(!masks), u64::from(masks && inputs)]
}

/// What a pass moved of the counts between `before` and `after`: its splits, readers through
/// snapshots, coverage queries, and readers weighed by a count.
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.attachment_read_splits - before.attachment_read_splits,
        after.reading_draws_snapshotted - before.reading_draws_snapshotted,
        after.coverage_queries - before.coverage_queries,
        after.reading_draws_by_coverage - before.reading_draws_by_coverage,
    ]
}

/// Why a pass's reader took the split, of the counts between `before` and `after`
/// (`WorkCounts::reading_draws_split_*`): the device lacks the snapshot path, the pass's shape,
/// the coverage rule, an app's visibility query, and every other reason.
fn split_by(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    let delta = |count: fn(&WorkCounts) -> u64| count(after) - count(before);
    let named = [
        delta(|c| c.reading_draws_split_device_lacks),
        delta(|c| c.reading_draws_split_pass_shape),
        delta(|c| c.reading_draws_split_coverage),
        delta(|c| c.reading_draws_split_visibility_query),
    ];
    let every = [
        delta(|c| c.reading_draws_split_unreadable_own_pixel),
        delta(|c| c.reading_draws_split_not_in_place),
        delta(|c| c.reading_draws_split_device_lacks),
        delta(|c| c.reading_draws_split_visibility_query),
        delta(|c| c.reading_draws_split_pass_shape),
        delta(|c| c.reading_draws_split_other_format),
        delta(|c| c.reading_draws_split_stand_in),
        delta(|c| c.reading_draws_split_one_image),
        delta(|c| c.reading_draws_split_coverage),
        delta(|c| c.reading_draws_split_area),
        delta(|c| c.reading_draws_split_variant_refused),
        delta(|c| c.reading_draws_split_slot_left_out),
        delta(|c| c.reading_draws_split_views),
        delta(|c| c.reading_draws_split_storage_binding),
        delta(|c| c.reading_draws_split_other_attachment),
    ];
    let others = every.iter().sum::<u64>() - named.iter().sum::<u64>();
    [named[0], named[1], named[2], named[3], others]
}

/// The device made, its states and its pipelines made by two passes over an untracked target.
fn prepared(cache: &Scratch) -> (Device, CommandQueue, [RenderPipelineState; 2]) {
    let device = device(&cache.0);
    let queue = device.new_command_queue().unwrap();
    let states = [state(&device, "tob_solid"), state(&device, "tob_read")];
    let warm = target(&device, false);
    for pass in 0..2 {
        pass_over(
            &queue,
            &warm,
            (&states[0], &states[1]),
            (pass, SPARSE),
            None,
        );
    }
    settled(&device);
    (device, queue, states)
}

#[test]
fn a_readers_form_follows_what_it_covered_in_the_pass_before() {
    let cache = Scratch::new("learned-coverage");
    let (device, queue, [fill, read]) = prepared(&cache);
    let (snapshots, measures) = answers(&device);
    let lacks = lacks(&device);
    let target = target(&device, true);
    let rects = [SPARSE, SPARSE, WHOLE, WHOLE, WHOLE, SPARSE, SPARSE];
    let mut want_reds = vec![CLEARED; (SIDE * SIDE) as usize];
    let mut learned: Option<[u64; 4]> = None;
    let pools_before = device.work_counts().coverage_pools_made;
    for (pass, &rect) in rects.iter().enumerate() {
        let before = device.work_counts();
        let reds = pass_over(&queue, &target, (&fill, &read), (pass, rect), None);
        let after = device.work_counts();
        let moved = moved(&before, &after);
        let split_by = split_by(&before, &after);
        eprintln!(
            "LEARNED-COVERAGE pass {pass} over {rect:?}: (splits, snapshotted, queries, by \
             coverage) {moved:?}"
        );
        let [x0, y0, x1, y1] = rect;
        for y in y0..y1 {
            for x in x0..x1 {
                want_reds[(y * SIDE + x) as usize] = red_of(pass);
            }
        }
        // Both forms read what the fill wrote: Metal's answer, byte for byte.
        if let Some(i) = (0..reds.len()).find(|&i| reds[i] != want_reds[i]) {
            let (x, y) = (i as u64 % SIDE, i as u64 / SIDE);
            panic!(
                "pass {pass}: pixel ({x}, {y}) holds red {}, not {}",
                reds[i], want_reds[i]
            );
        }
        // The pass before's count chooses, the whole target's the split; with none, the area
        // rule the snapshot.
        let split = measures && learned == Some(WHOLE);
        let want = match (snapshots, measures) {
            (false, _) => [1, 0, 0, 0],
            (true, false) => [0, 1, 0, 0],
            (true, true) => [
                u64::from(split),
                u64::from(!split),
                1,
                u64::from(learned.is_some()),
            ],
        };
        assert_eq!(
            moved,
            want,
            "pass {pass} over {rect:?} on {}: splits, snapshotted, queries, by coverage",
            device.name()
        );
        // Each split counted by its reason: the device's lack or its fetch's pass shape, else
        // the count's.
        let want_by = match snapshots {
            false => [lacks[0], lacks[1], 0, 0, 0],
            true => [0, 0, u64::from(split), 0, 0],
        };
        assert_eq!(
            split_by, want_by,
            "pass {pass} over {rect:?}: split by the device, the pass's shape, coverage, a query, \
             others"
        );
        learned = Some(rect);
    }
    // Each pass takes the pool an earlier one gave back as its command buffer completed: a pass
    // makes one only where none has come back yet (the give-back may follow the wait's return).
    let pools = device.work_counts().coverage_pools_made - pools_before;
    eprintln!(
        "LEARNED-COVERAGE {}: pools made over {} passes: {pools}",
        device.name(),
        rects.len()
    );
    assert!(pools <= 2, "{pools} pools made over {} passes", rects.len());
}

#[test]
fn an_apps_visibility_query_keeps_its_count_with_a_reader_inside_it() {
    let cache = Scratch::new("learned-coverage-visibility");
    let (device, queue, [fill, read]) = prepared(&cache);
    let (snapshots, measures) = answers(&device);
    let precise = device
        .properties()
        .description
        .has(VkFeature::OcclusionQueryPrecise);
    if !precise {
        eprintln!(
            "LEARNED-COVERAGE {}: no counting visibility without occlusionQueryPrecise",
            device.name()
        );
        return;
    }
    let target = target(&device, true);
    let results = device.new_buffer(8, ResourceOptions::shared()).unwrap();
    let [x0, y0, x1, y1] = SPARSE;
    let pixels = (x1 - x0) * (y1 - y0);
    // Without the app's query, the reader records one of its own where the device measures.
    let before = device.work_counts();
    pass_over(&queue, &target, (&fill, &read), (0, SPARSE), None);
    let alone = moved(&before, &device.work_counts());
    assert_eq!(alone[2], u64::from(measures), "{alone:?}: queries alone");
    // Inside the app's query: the split, no query of its own, and the app's count is the fill's
    // samples and the reader's.
    let before = device.work_counts();
    let reds = pass_over(&queue, &target, (&fill, &read), (1, SPARSE), Some(&results));
    let after = device.work_counts();
    let inside = moved(&before, &after);
    // The device's lack is checked first, then the query, then the pass's shape.
    let lacks = lacks(&device)[0];
    let want_by = [lacks, 0, 0, 1 - lacks, 0];
    assert_eq!(
        split_by(&before, &after),
        want_by,
        "inside the app's query: split by the device, the pass's shape, coverage, a query, others"
    );
    eprintln!(
        "LEARNED-COVERAGE {} inside a visibility query: (splits, snapshotted, queries, by \
         coverage) {inside:?}",
        device.name()
    );
    assert_eq!(
        reds[(y0 * SIDE + x0) as usize],
        red_of(1),
        "the reader read the fill"
    );
    assert_eq!(inside[2], 0, "{inside:?}: no query inside the app's");
    assert_eq!(
        inside[..2],
        [1, 0],
        "{inside:?}: the split (snapshots {snapshots})"
    );
    let count = u64::from_le_bytes(super::merge_scratch::read(&results, 8).try_into().unwrap());
    assert_eq!(
        count,
        2 * pixels,
        "the app's count: the fill's samples and the reader's"
    );
}
