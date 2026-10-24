//! Fragment functions that read `[[sample_mask]]`, write memory and return a color, on the device
//! the cfg selects, under the validation layer, over the owned fixture `coverage_outputs`.
//!
//! Metal runs such a function once per covered sample, lowest sample first, each run reading the
//! pixel's coverage, and writes each covered sample's color from that sample's own run, keeping
//! the sample only where the run's `[[sample_mask]]` output has its bit (our probe `co` on an
//! Apple M3 Max under macOS 27.0.1, at 2 and 4 samples, whole and partly covered pixels).
//!
//! Each pass draws the fixture's triangle, the lower left half of a 16 x 16 target inside the
//! viewport, into an `RGBA8Uint` target cleared to 0 at 2 and 4 samples, each run returning the
//! count of the pixel's runs before it (red) and the coverage it read (green), and reads each
//! sample back. Every device writes exactly the samples Metal writes and runs the function once
//! per covered sample, each run reading the pixel's coverage. A device that runs the function per
//! sample itself (MoltenVK on Apple's GPU, as written) writes each sample its own run's color, as
//! Metal; elsewhere the function runs once per pixel and its body once per covered sample, and
//! every written sample takes the last pass's color, announced as `looped-sample-outputs`. A
//! device that refuses the pipeline for what it lacks, or for its rasterizer's split of a clipped
//! triangle's fragments (lavapipe), says so and draws nothing.

use super::degrade_notices::counts;
use super::{buffer_with, read_buffer, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, Error, Library, LoadAction,
    PipelineOptions, RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The target's side.
const SIDE: u64 = 16;

/// The pixels of the target.
const PIXELS: usize = (SIDE * SIDE) as usize;

/// A sample nothing wrote, as `cvo_read` reads it.
const UNWRITTEN: u32 = 0xffff;

struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

/// What a draw left: each pixel's samples as `cvo_read` reads them, each pixel's runs, and
/// whether the pipeline announced `looped-sample-outputs`, which the device announces for each
/// pipeline here (`Lane::new`) and not once as it would by default.
struct Drawn {
    samples: Vec<u32>,
    runs: Vec<u32>,
    looped: bool,
}

impl Lane {
    fn new(messages: &Messages, cache: &ScratchCache) -> Lane {
        // Each pipeline's own notice: a device announces a kind once, and every pipeline here is
        // read for whether it loops (`DeviceOptions::announce_each`).
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
            options.announce_each = [Degradation::LoopedSampleOutputs].into();
        })
        .unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("coverage_outputs.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        }
    }

    fn function(&self, name: &str) -> StageSource {
        StageSource::Function(
            self.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    }

    /// The draw of `fragment` at `samples`, its notices taken from `messages`; `None` where the
    /// device refuses the pipeline for what it lacks or for its rasterizer's split of a clipped
    /// triangle's fragments (lavapipe), never for the coverage at sample rate.
    fn draw(&self, messages: &Messages, fragment: &str, samples: u32) -> Option<Drawn> {
        let what = format!("{fragment} at {samples}");
        let mut d = RenderPipelineDescriptor::new(self.function("cvo_vertex"));
        d.fragment = Some(self.function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Uint));
        d.raster_sample_count = samples;
        let pipeline = match self
            .device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
        {
            Ok(p) => p,
            Err(Error::Unsupported(u)) if u.feature != Feature::PixelCoverageAtSampleRate => {
                println!("{}: {what} refused: {u:?}", self.device.name());
                return None;
            }
            Err(e) => panic!("{what}: {e}"),
        };
        let mut t = TextureDescriptor::texture_2d(PixelFormat::RGBA8Uint, SIDE, SIDE, false);
        t.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        t.resource_options = ResourceOptions::private();
        t.texture_type = TextureType::Type2DMultisample;
        t.sample_count = u64::from(samples);
        let target = self.device.new_texture(&t).unwrap();
        let runs = buffer_with(&self.device, &[0; 4 * PIXELS]).unwrap();
        let width = buffer_with(&self.device, &(SIDE as u32).to_le_bytes()).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(target.clone());
        a.load_action = LoadAction::Clear;
        a.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        a.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_fragment_buffer(Some(&runs), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&width), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{what}");
        // The pipeline's notices, announced where it, or its variant for the pass, was made.
        let looped =
            counts(&messages.take_notices()).contains_key(&Degradation::LoopedSampleOutputs);
        let words = |bytes: Vec<u8>| -> Vec<u32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect()
        };
        Some(Drawn {
            samples: words(self.read(&target, samples)),
            runs: words(read_buffer(&runs, 0, 4 * PIXELS as u64).unwrap()),
            looped,
        })
    }

    /// Each sample of `target` as `cvo_read` reads it, pixel by pixel.
    fn read(&self, target: &Texture, samples: u32) -> Vec<u8> {
        let state = self
            .device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(self.function("cvo_read")),
                &PipelineOptions::default(),
            )
            .unwrap();
        let length = 4 * PIXELS as u64 * u64::from(samples);
        let out = buffer_with(&self.device, &vec![0xee; length as usize]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        encoder.set_texture(Some(target), 0).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        read_buffer(&out, 0, length).unwrap()
    }
}

/// The rank of sample `s` among the set bits of `mask`: the count of its runs before its own,
/// lowest sample first.
fn rank(mask: u32, s: u32) -> u32 {
    (mask & ((1 << s) - 1)).count_ones()
}

/// `cvo_color` and `cvo_masked` at 2 and 4 samples write the samples Metal writes, run once per
/// covered sample reading the pixel's coverage, and write each sample its own run's color where
/// the device runs them per sample (no `looped-sample-outputs`), else the last pass's.
#[test]
fn coverage_reading_color_writers_write_the_samples_metal_writes() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::new(&messages, &cache);
    let name = lane.device.name().to_owned();
    let (mut whole, mut partial) = (0, 0);
    for samples in [2u32, 4] {
        let all = (1u32 << samples) - 1;
        for (fragment, masked) in [("cvo_color", false), ("cvo_masked", true)] {
            let made = lane.draw(&messages, fragment, samples);
            // MoltenVK on Apple's GPU runs a memory writer per sample itself, and makes both.
            let apple = name.starts_with("Apple");
            let Some(drawn) = made else {
                assert!(!apple, "{name}: {fragment} at {samples} refused");
                continue;
            };
            if apple {
                assert!(!drawn.looped, "{name}: {fragment} at {samples} looped");
            }
            println!(
                "{name}: {fragment} at {samples}: {}",
                if drawn.looped {
                    "looped, the last pass's color"
                } else {
                    "run per sample"
                }
            );
            for px in 0..PIXELS {
                let what = format!("{name}: {fragment} at {samples}, pixel {px}");
                let texels = &drawn.samples[px * samples as usize..][..samples as usize];
                let runs = drawn.runs[px];
                // The coverage the runs read, from any written sample (every run reads the same).
                let written: Vec<u32> = (0..samples)
                    .filter(|&s| texels[s as usize] != UNWRITTEN)
                    .collect();
                let Some(&first) = written.first() else {
                    // Uncovered, or covered with every sample's mask output dropping its bit.
                    if !masked {
                        assert_eq!(runs, 0, "{what}: runs without a written sample");
                    }
                    continue;
                };
                let coverage = texels[first as usize] >> 8;
                assert_ne!(coverage, 0, "{what}");
                assert_eq!(coverage & !all, 0, "{what}: coverage {coverage:#x}");
                assert_eq!(runs, coverage.count_ones(), "{what}: runs");
                let kept: Vec<u32> = (0..samples)
                    .filter(|&s| coverage & (1 << s) != 0)
                    .filter(|&s| !masked || rank(coverage, s) == s)
                    .collect();
                assert_eq!(
                    written, kept,
                    "{what}: written samples, coverage {coverage:#x}"
                );
                for &s in &written {
                    let texel = texels[s as usize];
                    let run = if drawn.looped {
                        coverage.count_ones() - 1
                    } else {
                        rank(coverage, s)
                    };
                    assert_eq!(texel, run | coverage << 8, "{what}: sample {s}");
                }
                if coverage == all {
                    whole += 1;
                } else {
                    partial += 1;
                }
            }
        }
    }
    println!("{name}: {whole} whole pixels, {partial} partly covered");
    if whole + partial > 0 {
        assert!(whole > 0 && partial > 0, "{whole} whole, {partial} partial");
    }
    drop(lane);
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
