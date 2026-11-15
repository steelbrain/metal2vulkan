//! Device tests of the loop over the covered samples on the device the cfg selects, under the
//! validation layer, over the owned fixture `coverage_loop`: fragment functions that read
//! `[[sample_mask]]` and that Metal runs once per covered sample.
//!
//! - Each function the loop does not fit is refused at 4 samples with its typed reason, through
//!   `Device::new_render_pipeline_state` as a Metal program makes the pipeline: early fragment
//!   tests as not representable; a discard, a derivative, a simdgroup operation and a color
//!   attachment read as not implemented. A device whose driver splits a clipped triangle's
//!   pixel-rate fragments (lavapipe) may refuse the coverage read for that first, and a device
//!   lacking a feature the function needs anyway, for that; none is ever made.
//! - The two the loop fits draw the fullscreen triangle over a 2x2 pass without attachments, as
//!   `runtime_lane::attachmentless` draws: at 4 samples each pixel's union of the coverages its
//!   runs read, the union of their samples, its runs, and no run outside its coverage (what any
//!   split of the pixel leaves alone, Metal's answer); at 2 samples, where Apple's rasterizer
//!   splits no pixel of this pass, each covered sample's run reading the pixel's coverage, once.
//! - The sample-qualified varying (`cov_loop_varying`) draws the same triangle over a 1x1 pass,
//!   whose quad holds three helper invocations, and over the 2x2 one, whose quad is four covered
//!   pixels, at 2 and 4 samples where the device takes the count without attachments: each
//!   covered sample's run reads the varying at the sample's standard position, which every lane
//!   device reports and Metal's answers at 2 and 4 samples are. A driver whose interpolation at a
//!   sample needs the whole quad (RADV) read garbage in the 1x1 pass until the entry read the
//!   varyings before the loop. At 8 samples, which an Apple GPU does not rasterize, its pipeline is
//!   Metal's error on every device.

use super::raster_sample_counts::refusal;
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverWorkaround;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    CommandBufferStatus, CommandQueue, Device, Error, Library, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState,
};

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let ll = metal2vulkan_devtools::engine_dir().join("coverage_loop.ll");
    device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap()
}

/// The fixture's functions the loop refuses, with what each is refused as.
const REFUSED: [(&str, Needs); 5] = [
    ("cov_loop_early", Needs::NotRepresentable),
    ("cov_loop_discard", Needs::NotImplemented),
    ("cov_loop_derivative", Needs::NotImplemented),
    ("cov_loop_simdgroup", Needs::NotImplemented),
    ("cov_loop_fetch", Needs::NotImplemented),
];

/// The 32 words after `pipeline` draws the fullscreen triangle once in a `side` x `side` pass
/// without attachments at a default raster sample count of `samples`, its own command buffer on
/// `queue`.
fn drawn(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    samples: u32,
    side: u32,
) -> Vec<u32> {
    let words = buffer_with(device, &[0u8; 128]).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = side.into();
    pass.render_target_height = side.into();
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
    read_buffer(&words, 0, 128)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// Metal's words for `fragment` at `samples` over the 2x2 pass: the union's four words per pixel,
/// which any split leaves alone, and at 2 samples, where this pass splits no pixel, each covered
/// sample's record and count.
fn metal_words(fragment: &str, samples: u32) -> Vec<u32> {
    let every = (1u32 << samples) - 1;
    let mut words = vec![0u32; 32];
    for pixel in 0..4 {
        if fragment == "cov_loop_union" {
            words[pixel * 4] = every;
            words[pixel * 4 + 1] = every;
            words[pixel * 4 + 2] = samples;
        } else {
            for sample in 0..samples as usize {
                words[pixel * 4 + sample] = every | 0x100;
                words[16 + pixel * 4 + sample] = 1;
            }
        }
    }
    words
}

/// The refusals at 4 samples, and the fitting functions' answers where Metal's do not depend on
/// Apple's split. A pipeline of a fitting function the device does not make is reported as its
/// refusal; lavapipe's is the driver's split.
#[test]
fn the_loop_runs_what_fits_and_refuses_the_rest_by_its_reason() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let splits = device
            .compile_device()
            .profile()
            .workarounds()
            .contains(&DriverWorkaround::ClippedTrianglesSplitPixelFragments);
        let library = library(&device);
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let descriptor = |fragment: &str, samples: u32| {
            let mut descriptor = RenderPipelineDescriptor::new(function("cov_cover"));
            descriptor.fragment = Some(function(fragment));
            descriptor.raster_sample_count = samples;
            if fragment == "cov_loop_fetch" {
                descriptor.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
            }
            descriptor
        };
        let mut answers = Vec::new();
        for (fragment, needs) in REFUSED {
            let made = device
                .new_render_pipeline_state(&descriptor(fragment, 4), &PipelineOptions::default());
            let Err(Error::Unsupported(u)) = made else {
                panic!(
                    "{fragment} at 4x on {name}: {:?}, not a refusal",
                    made.map(|_| ())
                );
            };
            let coverage = u.feature == Feature::PixelCoverageAtSampleRate && u.needs == needs;
            let driver_split = splits
                && u.feature == Feature::PixelCoverage
                && u.needs
                    == Needs::DriverDefect(DriverWorkaround::ClippedTrianglesSplitPixelFragments);
            let lacks = u.feature != Feature::PixelCoverageAtSampleRate
                && matches!(u.needs, Needs::AnyOf(_));
            assert!(
                coverage || driver_split || lacks,
                "{fragment} at 4x on {name}: {u:?}"
            );
            answers.push(format!(
                "{fragment} refused {}",
                if coverage {
                    "by its reason"
                } else {
                    "for what the device lacks"
                }
            ));
        }
        let queue = device.new_command_queue().unwrap();
        for (fragment, samples) in [("cov_loop_union", 4u32), ("cov_loop_record", 2)] {
            let pipeline = match device.new_render_pipeline_state(
                &descriptor(fragment, samples),
                &PipelineOptions::default(),
            ) {
                Ok(pipeline) => pipeline,
                Err(Error::Unsupported(u)) => {
                    answers.push(format!("{fragment} {samples}x refused, {u:?}"));
                    continue;
                }
                Err(error) => panic!("{fragment} {samples}x: {error}"),
            };
            assert_eq!(
                drawn(&device, &queue, &pipeline, samples, 2),
                metal_words(fragment, samples),
                "{fragment} at {samples}x on {name}"
            );
            answers.push(format!("{fragment} {samples}x Metal's"));
        }
        eprintln!("coverage loop on {name}: {}", answers.join("; "));
    }
    layer_reported_nothing_at_all(&messages);
}

/// The standard sample positions at 2 and 4 samples (Vulkan's "Standard sample locations",
/// which every lane device reports as `standardSampleLocations`; Metal's measured answers are the
/// same).
const STANDARD: [(u32, &[(f32, f32)]); 2] = [
    (2, &[(0.75, 0.75), (0.25, 0.25)]),
    (
        4,
        &[
            (0.375, 0.125),
            (0.875, 0.375),
            (0.125, 0.625),
            (0.625, 0.875),
        ],
    ),
];

/// What `words` of `cov_loop_varying` drawn over a `side` x `side` pass at `positions` differ in
/// from each covered sample's run reading the varying at its sample: per pixel `p` and sample `s`,
/// word `4p + s` holds `uv * 16` rounded in its two low bytes, bit `s` in the coverage byte it
/// read, and the run's mark; a word past the pass's samples is 0. The fixture's companion makes
/// `uv` the sample's position over the pass, its y counted from the pass's bottom: Metal's
/// window y runs down from the top, where clip-space y is 1.
fn varying_differences(words: &[u32], positions: &[(f32, f32)], side: u32) -> Vec<String> {
    let samples = positions.len();
    let mut differences = Vec::new();
    for pixel in 0..(side * side) as usize {
        let (px, py) = ((pixel % 2) as f32, (pixel / 2) as f32);
        for (s, (x, y)) in positions.iter().enumerate() {
            let word = words[pixel * 4 + s];
            let scale = 16.0 / side as f32;
            let up = side as f32 - (py + y);
            let at = ((px + x) * scale) as u32 | (((up * scale) as u32) << 8);
            let mask = (word >> 16) & 0xff;
            if word & 0xffff != at || (mask >> s) & 1 == 0 || word >> 24 != 1 {
                differences.push(format!(
                    "pixel {pixel} sample {s}: {word:#010x}, uv * 16 {at:#06x}"
                ));
            }
        }
        if side == 1 {
            for (s, word) in words.iter().enumerate().skip(samples) {
                if *word != 0 {
                    differences.push(format!("word {s}: {word:#010x}, past the samples"));
                }
            }
        }
    }
    differences
}

/// `cov_loop_varying` over a 1x1 pass (three helper invocations in the pixel's quad) and over a
/// 2x2 pass (a quad of four covered pixels) at 2 and 4 samples, where the device takes the count
/// without attachments: each covered sample's run reads the varying at the sample's standard
/// position. A pipeline the device does not make is reported as its typed refusal: MoltenVK's
/// per-sample runs, lavapipe's split, or what the device lacks. At 8 samples the pipeline is
/// Metal's error on every device, whatever the device rasterizes.
#[test]
fn the_loop_reads_each_sample_qualified_varying_at_its_sample() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let profile = device.compile_device().profile();
        let attachmentless = profile.limits().framebuffer_no_attachments_sample_counts;
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
        let mut eight = RenderPipelineDescriptor::new(function("cov_cover_uv"));
        eight.fragment = Some(function("cov_loop_varying"));
        eight.raster_sample_count = 8;
        let made = device.new_render_pipeline_state(&eight, &PipelineOptions::default());
        assert_eq!(
            made.map(drop),
            Err(refusal("new_render_pipeline_state", 8)),
            "cov_loop_varying at 8x on {name}"
        );
        answers.push("8x Metal's error".to_owned());
        for (samples, positions) in STANDARD {
            if attachmentless.as_raw() & samples == 0 {
                answers.push(format!("{samples}x not taken without attachments"));
                continue;
            }
            let mut descriptor = RenderPipelineDescriptor::new(function("cov_cover_uv"));
            descriptor.fragment = Some(function("cov_loop_varying"));
            descriptor.raster_sample_count = samples;
            let pipeline =
                match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
                    Ok(pipeline) => pipeline,
                    Err(Error::Unsupported(u)) => {
                        let typed = matches!(u.needs, Needs::DriverDefect(_) | Needs::AnyOf(_));
                        assert!(typed, "cov_loop_varying at {samples}x on {name}: {u:?}");
                        answers.push(format!("{samples}x refused, {:?}", u.needs));
                        continue;
                    }
                    Err(error) => panic!("cov_loop_varying at {samples}x on {name}: {error}"),
                };
            for side in [1, 2] {
                let words = drawn(&device, &queue, &pipeline, samples, side);
                let differences = varying_differences(&words, positions, side);
                assert!(
                    differences.is_empty(),
                    "cov_loop_varying at {samples}x over {side}x{side} on {name}: {differences:#?}"
                );
                answers.push(format!("{samples}x {side}x{side} at each sample"));
            }
        }
        eprintln!("sample-qualified varying on {name}: {}", answers.join("; "));
    }
    layer_reported_nothing_at_all(&messages);
}
