//! Device tests of render passes without attachments on the device the cfg selects, under the
//! validation layer. Such a pass rasterizes at its default raster sample count
//! (`defaultRasterSampleCount`), with no attachment to carry it: the owned fixture
//! `rt6_attachmentless` draws the fullscreen triangle over a 2x2 render area, and each fragment
//! invocation ORs into its pixel's word the samples it ran for and counts itself, as a Metal probe
//! on an M4 drew it. Each such pass records, makes its pipelines against, and begins the pass its
//! device keeps for its sample count (`runtime/attachmentless.rs`); MoltenVK rasterized one sample
//! in a pass made per encoder.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::PrimitiveType;
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    CommandBufferError, CommandBufferStatus, CommandQueue, Device, Error, EventRef, Library,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ScissorRect,
};

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let ll = metal2vulkan_devtools::engine_dir().join("rt6_attachmentless.ll");
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// Metal's words after one draw at `samples` (a Metal probe on an M4 with macOS 27.0.1, both
/// functions, with and without Metal's API validation): every pixel ran every sample, once each (a
/// function that writes memory runs once per covered sample).
fn metal_words(samples: u32) -> [u32; 5] {
    let every = (1u32 << samples) - 1;
    [every, every, every, every, 4 * samples]
}

/// The words after `pipeline` draws the fullscreen triangle once in a 2x2 pass without
/// attachments at a default raster sample count of `samples`, its own command buffer on `queue`.
fn drawn(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    samples: u32,
) -> Vec<u32> {
    let words = buffer_with(device, &[0u8; 20]).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 2;
    pass.render_target_height = 2;
    pass.default_raster_sample_count = samples.into();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_fragment_buffer(Some(&words), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&words, 0, 20)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// A 2x2 pass without attachments at a default raster sample count of 1, 2 and 4, drawn by the
/// sample-rate function and by the coverage-mask one, each pipeline of that sample count and no
/// attachments: each pixel's word holds every sample, and the invocations are the pixels times the
/// samples, Metal's answer. Each pipeline draws in two passes, one after the other, the second
/// beginning the kept pass again with the pipeline made against it. A pipeline the device does not
/// make is reported as its refusal.
#[test]
fn a_pass_without_attachments_rasterizes_its_default_sample_count() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let library = library(&device);
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let queue = device.new_command_queue().unwrap();
        let mut answers = Vec::new();
        for fragment in ["rt6_mark_sample", "rt6_mark_coverage"] {
            for samples in [1u32, 2, 4] {
                let mut descriptor = RenderPipelineDescriptor::new(function("rt6_cover"));
                descriptor.fragment = Some(function(fragment));
                descriptor.raster_sample_count = samples;
                let pipeline = match device
                    .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                {
                    Ok(pipeline) => pipeline,
                    Err(Error::Unsupported(u)) => {
                        answers.push(format!("{fragment} {samples}x refused, {u:?}"));
                        continue;
                    }
                    Err(error) => panic!("{fragment} {samples}x: {error}"),
                };
                for pass in ["first", "second"] {
                    assert_eq!(
                        drawn(&device, &queue, &pipeline, samples),
                        metal_words(samples),
                        "{fragment} at {samples}x on {name}, its {pass} pass: each pixel's \
                         samples, then the invocations"
                    );
                }
                answers.push(format!("{fragment} {samples}x Metal's"));
            }
        }
        eprintln!("attachmentless passes on {name}: {}", answers.join("; "));
    }
    layer_reported_nothing_at_all(&messages);
}

/// What one command buffer on `queue` leaves: a blit filling mark 0 with `0x11`, a signal of
/// event `before`, a pass without attachments of `size` at a default raster sample count of
/// `samples`, in which `pipeline` (where given) draws the fullscreen triangle scissored to the
/// 2x2 corner, a blit filling mark 1 with `0x22`, and a signal of event `after`; then a second
/// command buffer filling mark 2 with `0x33`, which completes. The first's status and error, the
/// three marks, the invocations counted, and each event's signaled value.
type Failing = (
    CommandBufferStatus,
    Option<CommandBufferError>,
    Vec<u32>,
    u32,
    [u64; 2],
);

/// [`Failing`] for a pass of `size` at `samples`, drawn by `pipeline` where given.
fn failing_pass(
    device: &Device,
    queue: &CommandQueue,
    pipeline: Option<&RenderPipelineState>,
    size: [u64; 2],
    samples: u64,
) -> Failing {
    let marks = buffer_with(device, &[0u8; 12]).unwrap();
    let counts = buffer_with(device, &[0u8; 20]).unwrap();
    let before = device.new_shared_event().unwrap();
    let after = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&marks, 0..4, 0x11).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
        .encode_signal_event(EventRef::from(&before), 1)
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = size[0];
    pass.render_target_height = size[1];
    pass.default_raster_sample_count = samples;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    if let Some(pipeline) = pipeline {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder
            .set_scissor_rect(ScissorRect::new(0, 0, 2, 2))
            .unwrap();
        encoder.set_fragment_buffer(Some(&counts), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&marks, 4..8, 0x22).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
        .encode_signal_event(EventRef::from(&after), 1)
        .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let next = queue.command_buffer().unwrap();
    let mut blit = next.blit_command_encoder().unwrap();
    blit.fill_buffer(&marks, 8..12, 0x33).unwrap();
    blit.end_encoding().unwrap();
    next.commit().unwrap();
    next.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        next.status(),
        CommandBufferStatus::Completed,
        "the next one"
    );
    let read_words = |buffer, length| -> Vec<u32> {
        read_buffer(buffer, 0, length)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    };
    (
        command_buffer.status(),
        command_buffer.error(),
        read_words(&marks, 12),
        read_words(&counts, 20)[4],
        [before.signaled_value(), after.signaled_value()],
    )
}

/// A pass without attachments whose target width or height is past 16384 is encoded, and Metal
/// fails its command buffer from the pass on (an M4, and the M3 Max, alike, with and without API
/// validation): the blit before it runs and the event signaled before it fires, while the pass's
/// draw, the blit after it and the event signaled after it do not; the command buffer completes
/// with an error, `MTLCommandBufferErrorInternal` (1); the queue's next command buffer runs. A pass
/// of 16384 a side runs whole.
#[test]
fn a_pass_without_attachments_past_16384_fails_its_command_buffer() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let library = library(&device);
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
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let (status, error, marks, invocations, events) =
            failing_pass(&device, &queue, Some(&pipeline), [16384, 16], 1);
        assert_eq!(
            (status, error),
            (CommandBufferStatus::Completed, None),
            "16384 wide"
        );
        assert_eq!(marks, [0x1111_1111, 0x2222_2222, 0x3333_3333]);
        assert_eq!(invocations, 4, "the 2x2 corner, once each");
        assert_eq!(events, [1, 1]);
        for size in [[16385, 16], [16, 16385], [16385, 16385]] {
            let (status, error, marks, invocations, events) =
                failing_pass(&device, &queue, Some(&pipeline), size, 1);
            assert_eq!(status, CommandBufferStatus::Error, "{size:?}");
            assert_eq!(
                error,
                Some(CommandBufferError::RenderTargetSize),
                "{size:?}"
            );
            assert_eq!(error.map(CommandBufferError::metal_code), Some(1));
            assert_eq!(
                marks,
                [0x1111_1111, 0, 0x3333_3333],
                "{size:?}: the blit before the pass alone, then the next command buffer"
            );
            assert_eq!(invocations, 0, "{size:?}: the pass ran nothing");
            assert_eq!(events, [1, 0], "{size:?}: the signal before the pass alone");
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A pass without attachments at a default raster sample count other than 1, 2 and 4 is Metal's
/// on every device, whatever samples the driver rasterizes, by the count's low 8 bits (plain Metal
/// on an Apple M4 under macOS 27.0.1, and our probe with the consumer's run of its probe on the
/// M3 Max under macOS 26.7 alike): at 3, 5, 6, 7 and 257 the encoder is made and the command buffer
/// runs whole; at 16, 17, 64, 127 and 300 the encoder is made and Metal fails the command buffer
/// from the pass on with `MTLCommandBufferErrorInternal`, as past 16384 a side (the blit before the
/// pass and the signal before it alone, then the next command buffer); at 0 (the descriptor's
/// default), 8, 9, 15, 128, 255, 512 and `u64::MAX` Metal ends the process making the encoder
/// (`Invalid`). No pipeline is drawn: no render pipeline of those counts is made
/// (`runtime_lane::raster_sample_counts`).
#[test]
fn a_pass_without_attachments_takes_metal_s_default_sample_counts() {
    use metal2vulkan_runtime::{Invalid, Problem, RenderPassProblem};
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        for samples in [3, 5, 6, 7, 257] {
            let (status, error, marks, _, events) =
                failing_pass(&device, &queue, None, [16, 16], samples);
            assert_eq!(
                (status, error),
                (CommandBufferStatus::Completed, None),
                "{samples}"
            );
            assert_eq!(marks, [0x1111_1111, 0x2222_2222, 0x3333_3333], "{samples}");
            assert_eq!(events, [1, 1], "{samples}");
        }
        for samples in [16, 17, 64, 127, 300] {
            let (status, error, marks, _, events) =
                failing_pass(&device, &queue, None, [16, 16], samples);
            assert_eq!(status, CommandBufferStatus::Error, "{samples}");
            assert_eq!(
                error,
                Some(CommandBufferError::DefaultRasterSampleCount),
                "{samples}"
            );
            assert_eq!(error.map(CommandBufferError::metal_code), Some(1));
            assert_eq!(
                marks,
                [0x1111_1111, 0, 0x3333_3333],
                "{samples}: the blit before the pass alone, then the next command buffer"
            );
            assert_eq!(
                events,
                [1, 0],
                "{samples}: the signal before the pass alone"
            );
        }
        for samples in [0, 8, 9, 15, 128, 255, 512, u64::MAX] {
            let mut pass = RenderPassDescriptor::new();
            pass.render_target_width = 16;
            pass.render_target_height = 16;
            pass.default_raster_sample_count = samples;
            let command_buffer = queue.command_buffer().unwrap();
            let made = command_buffer.render_command_encoder(&pass).map(drop);
            assert_eq!(
                made,
                Err(Error::Invalid(Invalid::new(
                    "render_command_encoder",
                    Problem::RenderPass(RenderPassProblem::DefaultRasterSampleCount {
                        count: samples
                    })
                ))),
                "{samples}"
            );
        }
        eprintln!(
            "attachmentless default sample counts on {}: Metal's at 3, 5, 6, 7, 257 (run), 16, \
             17, 64, 127, 300 (fail the command buffer), 0, 8, 9, 15, 128, 255, 512, \
             u64::MAX (stopped)",
            device.name()
        );
    }
    layer_reported_nothing_at_all(&messages);
}
