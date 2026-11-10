//! A render encoder's fences and memory barriers (`runtime/fence.rs`) on the device the cfg
//! selects, under the validation layer with synchronization validation on (the executor's
//! settings, `executor_layer_settings`). The owned fixture `rt6_attachmentless` draws the
//! fullscreen triangle over a 2x2 pass without attachments; each fragment ORs its sample into its
//! pixel's word and counts itself in word 4, so every draw writes the same five words. The
//! synchronization validation does not see a race between two draws of one render pass instance
//! (layer 1.4.341 on lavapipe reported none between two such draws without a barrier), so the
//! checks here are the counts, the split and the layer's other reports, not a hazard report.
//!
//! - A barrier that orders nothing (an empty scope, no stage before or after, a scope of the
//!   render targets alone after no fragment stage or before none at or before it, no resource)
//!   splits nothing.
//!   A memory barrier over buffers between two draws orders the second after the first: the
//!   layer reports nothing, the count is both draws', and the pass's split is announced once
//!   (`render-pass-split`), its reason naming the barriers. A barrier before the encoder's first
//!   draw or after its last splits nothing.
//! - A fence one render encoder updates and the next waits for orders their draws over an
//!   untracked buffer, and a wait after the same encoder's update is refused, as the compute and
//!   blit encoders' are.
//!
//! The rest draw into color attachments under the plain validation layer: with the owned fixture
//! `rt6_resolves` (`resolves::lane`), and with `rt6_amplified` into two views.
//!
//! - A barrier over the render targets alone, from the fragment stage to the fragment or the
//!   vertex stage, records inside the render pass instance and splits nothing; so it does in a
//!   multiview instance of amplified draws, with the view-local flag.
//! - A split keeps the samples of a multisampled attachment Metal stores nowhere, memoryless or
//!   resolved without a store, between its instances.
//! - So do an execution's commands split into runs by a command's barrier, of a memoryless color or
//!   depth attachment.

use super::degrade_notices::assert_once;
use super::memoryless::{memoryless, memoryless_msaa};
use super::resolves::{color_pass, lane, quiet, repeat, FULL, MASKED, ONE, SAMPLES, SIZE, WHITE};
use super::{
    at_plan, buffer_with, buffer_with_options, layer_reported_nothing_at_all, read_buffer,
    validated_device, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_case::executor_layer_settings;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BarrierScope, CompareFunction, IndirectCommandType, PixelFormat, PrimitiveTopologyClass,
    PrimitiveType, RenderStages, ResourceUsage, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CpuCacheMode, Degradation,
    DegradeNotice, Device, Error, HazardTrackingMode, IndirectCommandBufferDescriptor, LoadAction,
    Origin, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StorageMode, StoreAction, TextureDescriptor,
    VertexAmplificationViewMapping,
};
use std::path::Path;

/// The words after one draw: every pixel's sample 0, then the four invocations.
const ONE_DRAW: [u32; 5] = [1, 1, 1, 1, 4];

/// The words after two draws: the samples as one draw's, the invocations twice as many.
const TWO_DRAWS: [u32; 5] = [1, 1, 1, 1, 8];

/// A device whose validation layer checks synchronization, reporting to a fresh message list;
/// `test` runs on it with the fixture's state and the list, which is returned once the device is
/// destroyed.
fn on_device(test: impl FnOnce(&Device, &RenderPipelineState, &Messages)) -> Messages {
    on_device_with(|_| {}, test)
}

/// [`on_device`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn on_device_with(
    adjust: fn(&mut DeviceOptions),
    test: impl FnOnce(&Device, &RenderPipelineState, &Messages),
) -> Messages {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layer_settings = executor_layer_settings();
            adjust(options)
        })
        .unwrap();
        let state = state(&device);
        test(&device, &state, &messages);
    }
    messages
}

/// The fixture's fullscreen triangle and sample-rate fragment, at one sample, without
/// attachments.
fn state(device: &Device) -> RenderPipelineState {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/rt6_attachmentless.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("rt6_cover"));
    descriptor.fragment = Some(function("rt6_mark_sample"));
    descriptor.raster_sample_count = 1;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Five zero words, of `hazard` tracking.
fn words(device: &Device, hazard: HazardTrackingMode) -> Buffer {
    let options = ResourceOptions::new(StorageMode::Shared, CpuCacheMode::DefaultCache, hazard);
    buffer_with_options(device, &[0u8; 20], options).unwrap()
}

/// `buffer`'s five words.
fn read(buffer: &Buffer) -> Vec<u32> {
    read_buffer(buffer, 0, 20)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// A render encoder of a 2x2 pass without attachments on `command_buffer`, drawing with `state`
/// into `words`.
fn encoder(
    command_buffer: &CommandBuffer,
    state: &RenderPipelineState,
    words: &Buffer,
) -> RenderCommandEncoder {
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 2;
    pass.render_target_height = 2;
    pass.default_raster_sample_count = 1;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_fragment_buffer(Some(words), 0, 0).unwrap();
    encoder
}

/// Draw the fullscreen triangle.
fn draw(encoder: &mut RenderCommandEncoder) {
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
}

/// Commit `command_buffer` and wait for it to complete.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The words after one pass on `device` whose encoder `encode` fills, its own command buffer.
fn pass(
    device: &Device,
    state: &RenderPipelineState,
    encode: impl FnOnce(&mut RenderCommandEncoder),
) -> Vec<u32> {
    let words = words(device, HazardTrackingMode::Tracked);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = encoder(&command_buffer, state, &words);
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    read(&words)
}

const BUFFERS: BarrierScope = BarrierScope::BUFFERS;
const FRAGMENT: RenderStages = RenderStages::FRAGMENT;

/// Barriers of an empty scope, of no stage before or after, or of no resource, between two draws
/// order nothing and split nothing; both draws draw. (The draws race, as Metal leaves them, so
/// what the layer says of them is not checked.)
#[test]
fn barriers_that_order_nothing_split_nothing() {
    let messages = on_device(|device, state, _| {
        let words = pass(device, state, |encoder| {
            draw(encoder);
            encoder.memory_barrier_with_scope(BarrierScope::empty(), FRAGMENT, FRAGMENT);
            encoder.memory_barrier_with_scope(BUFFERS, RenderStages::empty(), FRAGMENT);
            encoder.memory_barrier_with_scope(BUFFERS, FRAGMENT, RenderStages::empty());
            encoder.memory_barrier_with_scope(
                BarrierScope::RENDER_TARGETS,
                RenderStages::VERTEX,
                RenderStages::VERTEX.union(FRAGMENT),
            );
            encoder.memory_barrier_with_scope(
                BarrierScope::RENDER_TARGETS,
                FRAGMENT,
                RenderStages::TILE,
            );
            encoder
                .memory_barrier_with_resources(&[], FRAGMENT, FRAGMENT)
                .unwrap();
            draw(encoder);
        });
        assert_eq!(words, TWO_DRAWS);
    });
    assert_eq!(
        split(&messages),
        None,
        "barriers that order nothing split the pass"
    );
}

/// The split notice among the notices `messages` delivered so far, which it removes.
fn split(messages: &Messages) -> Option<DegradeNotice> {
    messages
        .take_notices()
        .into_iter()
        .filter_map(|message| message.degradation)
        .find(|notice| notice.degradation == Degradation::RenderPassSplit)
}

/// A memory barrier between two draws orders the second after the first, through a split pass
/// announced once, its reason naming the barriers; barriers that order nothing split nothing.
#[test]
fn a_memory_barrier_orders_a_render_encoders_draws() {
    let messages = on_device_with(at_plan, |device, state, messages| {
        let words = pass(device, state, |encoder| {
            encoder.memory_barrier_with_scope(BUFFERS, FRAGMENT, FRAGMENT);
            draw(encoder);
            encoder.memory_barrier_with_scope(BUFFERS, FRAGMENT, FRAGMENT);
        });
        assert_eq!(
            words, ONE_DRAW,
            "barriers before the first draw and after the last"
        );
        assert_eq!(
            split(messages),
            None,
            "barriers before the first draw and after the last"
        );
        let words = pass(device, state, |encoder| {
            draw(encoder);
            encoder.memory_barrier_with_scope(BUFFERS, FRAGMENT, FRAGMENT);
            draw(encoder);
        });
        assert_eq!(words, TWO_DRAWS, "the scope barrier's pass");
        let buffer = super::buffer_with(device, &[0u8; 4]).unwrap();
        let words = pass(device, state, |encoder| {
            draw(encoder);
            encoder
                .memory_barrier_with_resources(&[(&buffer).into()], FRAGMENT, FRAGMENT)
                .unwrap();
            draw(encoder);
        });
        assert_eq!(words, TWO_DRAWS, "the resource barrier's pass");
    });
    layer_reported_nothing_at_all(&messages);
    let notices = messages.take_notices();
    assert_once(&notices, &[Degradation::RenderPassSplit]);
    let split = notices
        .into_iter()
        .filter_map(|message| message.degradation)
        .find(|notice| notice.degradation == Degradation::RenderPassSplit)
        .unwrap();
    assert!(
        split
            .reason
            .contains("memory barriers between its draws end 1 of"),
        "{}",
        split.reason
    );
}

/// A fence one render encoder updates after its draw and the next waits for before its own
/// orders the two draws over an untracked buffer; the first encoder's wait for the fence it
/// updated waits for the update before it, as plain Metal's does: none here.
#[test]
fn a_fence_orders_render_encoders() {
    let messages = on_device(|device, state, _| {
        let words = words(device, HazardTrackingMode::Untracked);
        let fence = device.new_fence();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut first = encoder(&command_buffer, state, &words);
        draw(&mut first);
        first.update_fence(&fence, FRAGMENT).unwrap();
        first.wait_for_fence(&fence, RenderStages::VERTEX).unwrap();
        first.end_encoding().unwrap();
        let mut second = encoder(&command_buffer, state, &words);
        second.wait_for_fence(&fence, FRAGMENT).unwrap();
        draw(&mut second);
        second.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(read(&words), TWO_DRAWS);
    });
    layer_reported_nothing_at_all(&messages);
}

/// A barrier over the render targets alone, from the fragment stage to the fragment or the
/// vertex stage, between two draws into a color attachment (the owned fixture `rt6_resolves`):
/// it records inside the render pass instance, within its subpass's dependency on itself, so the
/// layer reports nothing and the pass does not split; the second draw's white stands.
#[test]
fn a_render_target_barrier_stays_inside_the_pass() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let samples = lane.pipeline((FULL, SAMPLES), &[rgba8], (None, None), 1);
    let white = lane.pipeline((FULL, WHITE), &[rgba8], (None, None), 1);
    for before in [FRAGMENT, RenderStages::VERTEX] {
        let target = lane.single(rgba8);
        lane.run(
            &color_pass(&target, None, StoreAction::Store, (0, 0)),
            |l, e, h| {
                l.draw(e, h, &samples, 0.5);
                e.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, FRAGMENT, before);
                l.draw(e, h, &white, 0.5);
            },
        )
        .unwrap();
        assert_eq!(
            lane.read(&target, (0, 0), 4, BlitOption::None),
            repeat(&[0xff; 4], (SIZE * SIZE) as usize),
            "before {before:?}"
        );
        assert_eq!(
            split(&messages),
            None,
            "a render target barrier before {before:?} split the pass"
        );
    }
    quiet(lane, &messages);
}

/// A split keeps a multisampled attachment's samples between its render pass instances where
/// Metal stores none of them: a memoryless one (which a split pass renders into a stand-in) and a
/// texture whose store action is `MultisampleResolve`. Each is drawn with the per-sample colors
/// (the owned fixture `rt6_resolves`), then, after a barrier over buffers that splits the pass,
/// white on sample 0 alone; each pixel resolves to the mean of the four samples, white and the
/// first draw's samples 1 to 3, within a unit per channel, and the split is announced.
#[test]
fn a_split_keeps_samples_metal_does_not_store() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let samples = lane.pipeline((FULL, SAMPLES), &[rgba8], (None, None), 4);
    let masked = lane.pipeline((FULL, MASKED), &[rgba8], (None, None), 4);
    let mask = buffer_with(&lane.device, &1u32.to_le_bytes()).unwrap();
    // Sample 0 white; samples 1 to 3 red s/3 (85 and 170 of 255, then 255), green 255 on
    // sample 3 alone, blue 0.5 (127.5 of 255, either neighbour), alpha 1.
    let mean = [191.25, 127.5, 159.75, 255.0];
    let targets = [
        ("memoryless", memoryless_msaa(&lane, rgba8)),
        ("unstored", lane.msaa(rgba8)),
    ];
    for (name, msaa) in &targets {
        let resolve = lane.single(rgba8);
        lane.run(
            &color_pass(
                msaa,
                Some(&resolve),
                StoreAction::MultisampleResolve,
                (0, 0),
            ),
            |l, e, h| {
                l.draw(e, h, &samples, 0.5);
                e.memory_barrier_with_scope(BUFFERS, FRAGMENT, FRAGMENT);
                e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
                l.draw(e, h, &masked, 0.5);
            },
        )
        .unwrap();
        let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
        for (pixel, texel) in resolved.as_chunks::<4>().0.iter().enumerate() {
            for (channel, (&value, mean)) in texel.iter().zip(mean).enumerate() {
                assert!(
                    (f64::from(value) - mean).abs() <= 1.0,
                    "{name}: pixel {pixel} channel {channel} resolved to {value}, not {mean}"
                );
            }
        }
    }
    assert!(split(&messages).is_some(), "the split was not announced");
    quiet(lane, &messages);
}

/// An execution's commands split into runs by a command's barrier keep a memoryless attachment's
/// texels from run to run, as Metal's pass holds them: the two draws above as two commands of an
/// indirect command buffer, the second's barrier set before the encoder ends, into a memoryless
/// 4x attachment resolved as above (each pixel the mean of white and the first command's samples
/// 1 to 3); and into a memoryless `Depth32Float` attachment cleared to 1.0 beside a stored
/// color, white at depth 0.5 under `Less` with depth writes, then the per-sample colors at 0.75,
/// which the first command's depth fails, so every texel stays white. Plain Metal orders a
/// command after its barrier, whose draws are then the encoder's own two
/// draws of one pass. The runs load and store such an attachment between them, or, where its
/// image is lazily allocated (MoltenVK), render it into a backing, announced
/// (`memoryless-backed`).
#[test]
fn executed_runs_keep_a_memoryless_attachment() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let pipeline = |fragment, depth, samples| {
        lane.pipeline_of((FULL, fragment), &[rgba8], (depth, None), (samples, true))
    };
    let mask = buffer_with(&lane.device, &1u32.to_le_bytes()).unwrap();
    let z = |z: f32| buffer_with(&lane.device, &z.to_le_bytes()).unwrap();
    let (half, three_quarters) = (z(0.5), z(0.75));
    // Two commands, each a pipeline at a depth over the mask, the second's barrier set.
    let commands = |pipelines: [&RenderPipelineState; 2], depths: [&Buffer; 2]| {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.max_vertex_buffer_bind_count = 1;
        descriptor.max_fragment_buffer_bind_count = 1;
        let icb = lane
            .device
            .new_indirect_command_buffer(&descriptor, 2, ResourceOptions::shared())
            .unwrap();
        for (index, (pipeline, depth)) in pipelines.into_iter().zip(depths).enumerate() {
            let command = icb.indirect_render_command_at(index as u64).unwrap();
            command.set_render_pipeline_state(pipeline).unwrap();
            command.set_vertex_buffer(Some(depth), 0, 0).unwrap();
            command.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
            command
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
        icb.indirect_render_command_at(1)
            .unwrap()
            .set_barrier()
            .unwrap();
        icb
    };
    let execute = |encoder: &mut RenderCommandEncoder, icb| {
        for buffer in [&mask, &half, &three_quarters] {
            encoder
                .use_resource(
                    buffer.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX | RenderStages::FRAGMENT,
                )
                .unwrap();
        }
        encoder.execute_commands_in_buffer(icb, 0..2).unwrap();
    };
    // A memoryless 4x color, resolved.
    let (samples, masked) = (pipeline(SAMPLES, None, 4), pipeline(MASKED, None, 4));
    let icb = commands([&samples, &masked], [&half, &half]);
    let msaa = memoryless_msaa(&lane, rgba8);
    let resolve = lane.single(rgba8);
    let pass = color_pass(
        &msaa,
        Some(&resolve),
        StoreAction::MultisampleResolve,
        (0, 0),
    );
    lane.run(&pass, |_, e, _| execute(e, &icb)).unwrap();
    let mean = [191.25, 127.5, 159.75, 255.0];
    let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
    for (pixel, texel) in resolved.as_chunks::<4>().0.iter().enumerate() {
        for (channel, (&value, mean)) in texel.iter().zip(mean).enumerate() {
            assert!(
                (f64::from(value) - mean).abs() <= 1.0,
                "memoryless color: pixel {pixel} channel {channel} resolved to {value}, not {mean}"
            );
        }
    }
    // A memoryless depth beside a stored color.
    let depth32 = Some(PixelFormat::Depth32Float);
    let (white, samples) = (pipeline(WHITE, depth32, 1), pipeline(SAMPLES, depth32, 1));
    let icb = commands([&white, &samples], [&half, &three_quarters]);
    let color = lane.single(rgba8);
    let depth = memoryless(
        &lane,
        TextureType::Type2D,
        PixelFormat::Depth32Float,
        ONE,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    let mut pass = color_pass(&color, None, StoreAction::Store, (0, 0));
    pass.depth_attachment.texture = Some(depth);
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = StoreAction::DontCare;
    let mut less = DepthStencilDescriptor::default();
    less.depth_compare_function = CompareFunction::Less;
    less.depth_write_enabled = true;
    let less = lane.device.new_depth_stencil_state(&less).unwrap();
    lane.run(&pass, |_, e, _| {
        e.set_depth_stencil_state(&less).unwrap();
        execute(e, &icb);
    })
    .unwrap();
    assert_eq!(
        lane.read(&color, (0, 0), 4, BlitOption::None),
        repeat(&[0xff; 4], (SIZE * SIZE) as usize),
        "memoryless depth"
    );
    let backed = messages
        .take_notices()
        .into_iter()
        .filter_map(|message| message.degradation)
        .any(|notice| notice.degradation == Degradation::MemorylessBacked);
    eprintln!(
        "executed runs over memoryless attachments on {}: Metal's (backed: {backed})",
        lane.device.name()
    );
    quiet(lane, &messages);
}

/// [`executed_runs_keep_a_memoryless_attachment`]'s memoryless 4x color, its two commands (the
/// second's barrier set) copied by a blit of an earlier command buffer into another buffer, which
/// the pass executes: the GPU wrote its records, so a host step records the pass, on the
/// scheduler's thread, where it makes no render pass, framebuffer or image. Three passes over the
/// same memoryless texture each resolve to Metal's mean. The barrier splits the commands into
/// runs; where the texture's image is lazily allocated (MoltenVK) the runs render it into a
/// backing, which the commit takes from the texture and with which it makes the runs' passes and
/// framebuffer ahead, so the step makes none. The
/// texture keeps one backing for the three passes (`WorkCounts::kept_backings_made`), made where
/// the pass announced the backing (`memoryless-backed`) and nowhere else. The commits' make-ahead
/// work is printed, as its cost.
#[test]
fn a_host_step_s_runs_keep_a_memoryless_attachment_in_a_backing_made_ahead() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let pipeline = |fragment| lane.pipeline_of((FULL, fragment), &[rgba8], (None, None), (4, true));
    let mask = buffer_with(&lane.device, &1u32.to_le_bytes()).unwrap();
    let half = buffer_with(&lane.device, &0.5f32.to_le_bytes()).unwrap();
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 1;
    descriptor.max_fragment_buffer_bind_count = 1;
    let icb = || {
        lane.device
            .new_indirect_command_buffer(&descriptor, 2, ResourceOptions::shared())
            .unwrap()
    };
    let (samples, masked) = (pipeline(SAMPLES), pipeline(MASKED));
    let encoded = icb();
    for (index, pipeline) in [&samples, &masked].into_iter().enumerate() {
        let command = encoded.indirect_render_command_at(index as u64).unwrap();
        command.set_render_pipeline_state(pipeline).unwrap();
        command.set_vertex_buffer(Some(&half), 0, 0).unwrap();
        command.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
        command
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    encoded
        .indirect_render_command_at(1)
        .unwrap()
        .set_barrier()
        .unwrap();
    let written = icb();
    let copying = lane.queue.command_buffer().unwrap();
    let mut blit = copying.blit_command_encoder().unwrap();
    blit.copy_indirect_command_buffer(&encoded, 0..2, &written, 0)
        .unwrap();
    blit.end_encoding().unwrap();
    copying.commit().unwrap();
    copying.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(copying.status(), CommandBufferStatus::Completed);
    let msaa = memoryless_msaa(&lane, rgba8);
    let mean = [191.25, 127.5, 159.75, 255.0];
    let before = lane.device.work_counts();
    const PASSES: u64 = 3;
    for round in 0..PASSES {
        let resolve = lane.single(rgba8);
        let pass = color_pass(
            &msaa,
            Some(&resolve),
            StoreAction::MultisampleResolve,
            (0, 0),
        );
        lane.run(&pass, |_, e, _| {
            for buffer in [&mask, &half] {
                e.use_resource(
                    buffer.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX | RenderStages::FRAGMENT,
                )
                .unwrap();
            }
            e.execute_commands_in_buffer(&written, 0..2).unwrap();
        })
        .unwrap();
        let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
        for (pixel, texel) in resolved.as_chunks::<4>().0.iter().enumerate() {
            for (channel, (&value, mean)) in texel.iter().zip(mean).enumerate() {
                assert!(
                    (f64::from(value) - mean).abs() <= 1.0,
                    "pass {round}: pixel {pixel} channel {channel} resolved to {value}, not {mean}"
                );
            }
        }
    }
    let after = lane.device.work_counts();
    let backed = messages
        .take_notices()
        .into_iter()
        .filter_map(|message| message.degradation)
        .any(|notice| notice.degradation == Degradation::MemorylessBacked);
    let kept = after.kept_backings_made - before.kept_backings_made;
    assert_eq!(
        kept,
        u64::from(backed),
        "one backing for {PASSES} passes, where backed"
    );
    let passes = after.ahead_render_passes_made - before.ahead_render_passes_made;
    let framebuffers = after.ahead_framebuffers_made - before.ahead_framebuffers_made;
    let nanos = after.ahead_nanos - before.ahead_nanos;
    eprintln!(
        "AHEAD host step over a memoryless attachment on {}: backed {backed}, per commit {} \
         render passes, {} framebuffers, {} us made ahead; {kept} backing kept for {PASSES} \
         passes",
        lane.device.name(),
        passes / PASSES,
        framebuffers / PASSES,
        nanos / PASSES / 1000,
    );
    drop((samples, masked));
    quiet(lane, &messages);
}

/// A barrier over the render targets between two draws amplified into two views (the owned
/// fixture `rt6_amplified`, whose fragment writes its instance's tag above its amplification, in
/// a pass of two layers of an `R32Uint` array): it records inside the multiview render pass
/// instance, with the view-local flag its subpass's dependency on itself has there, so the
/// validation layer reports nothing and the pass does not split; each layer holds the second
/// draw's last instance's tag, `0x200` above its view. A device that makes no amplified pipeline
/// (no `multiview`) prints its refusal. A fragment imageblock's barrier takes the same flags
/// (`render_target::self_dependency_flags`), but the binder refuses an imageblock draw under
/// amplification (`NotImplemented`), so none runs here.
#[test]
fn a_render_target_barrier_stays_inside_a_multiview_pass() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let ll =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/rt6_amplified.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("rt6_amplified_vertex"));
        descriptor.fragment = Some(function("rt6_amplified_fragment"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
        descriptor.input_primitive_topology = PrimitiveTopologyClass::Triangle;
        descriptor.max_vertex_amplification_count = 2;
        let pipeline =
            match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
                Ok(pipeline) => Some(pipeline),
                Err(Error::Unsupported(u)) => {
                    eprintln!("refused: amplified pipelines on {}: {u:?}", device.name());
                    None
                }
                Err(other) => panic!("an amplified pipeline on {}: {other}", device.name()),
            };
        if let Some(pipeline) = pipeline {
            let mut texture = TextureDescriptor::texture_2d(PixelFormat::R32Uint, 2, 2, false);
            texture.texture_type = TextureType::Type2DArray;
            texture.array_length = 2;
            texture.usage = TextureUsage::RENDER_TARGET;
            texture.resource_options = ResourceOptions::private();
            let target = device.new_texture(&texture).unwrap();
            let mut pass = RenderPassDescriptor::new();
            pass.render_target_array_length = 2;
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::Store;
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            let views = [
                VertexAmplificationViewMapping::new(0, 0),
                VertexAmplificationViewMapping::new(0, 1),
            ];
            encoder
                .set_vertex_amplification_count(2, Some(&views))
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder.memory_barrier_with_scope(BarrierScope::RENDER_TARGETS, FRAGMENT, FRAGMENT);
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 2, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            let out = buffer_with(&device, &[0u8; 32]).unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            for slice in 0..2 {
                blit.copy_from_texture_to_buffer(
                    &target,
                    slice,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(2, 2, 1),
                    &out,
                    slice * 16,
                    8,
                    16,
                    BlitOption::None,
                )
                .unwrap();
            }
            blit.end_encoding().unwrap();
            run(&command_buffer);
            let words: Vec<u32> = read_buffer(&out, 0, 32)
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            assert_eq!(words, [[0x200; 4], [0x201; 4]].concat());
            assert_eq!(
                split(&messages),
                None,
                "a render target barrier split the pass"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
