//! The private sample coverage of OpenGL on Metal (`PrivateRenderState::sample_coverage`), on the
//! device the cfg selects, under the validation layer, over the owned fixture
//! `psc_sample_coverage`.
//!
//! Metal's rule, measured with our own probes `sc` to `sc4` on an Apple M3 Max under macOS 27.0.1
//! (8 x 8 and 16 x 16 targets at 1, 2 and 4 samples; every 8-bit code, both sides of each level's
//! edge, NaN and the infinities; viewport offsets; the sample mask and a `[[sample_mask]]` output;
//! depth, stencil, visibility counts, early tests, no fragment function): the value is clamped to
//! `[0, 1]`, a NaN taken as 0, and converted to an 8-bit unorm code `c`. At one sample a pixel is
//! covered where the value is above 0.5. At 2 and 4 samples each 2x2 quad of window pixels keeps
//! `floor((c + s / 2) / s)` samples, `s = 64 / n`, handed out one at a time to its pixels in the
//! order `(0, 0)`, `(1, 1)`, `(1, 0)`, `(0, 1)` of their coordinates modulo 2, each pixel's filled
//! in the sample order 0, 3, 1, 2 at 4 samples and 0, 1 at 2. The coverage is ANDed with the
//! sample mask. The fragment function runs for every rasterized sample and reads the rasterized
//! coverage; the colors, depth and visibility count keep the covered samples, under early tests
//! only the colors; a pipeline without a fragment function ignores it.
//!
//! Each draw here, a fullscreen triangle at depth 0.5 into a 16 x 16 target (`RGBA8Unorm` color
//! and `Depth32Float` depth cleared to 0 and 1, the depth written under `Always`), is checked
//! against that rule sample by sample. A draw the device cannot run is skipped with a printed
//! line, and the rest are checked: a sample count the device lacks (2 is optional in Vulkan:
//! lavapipe offers 1 and 4), refused as `Unsupported` with `Feature::RasterSampleCount`, and a
//! coverage read its driver answers wrong (lavapipe splits a clipped triangle's pixel into
//! fragments of their own), refused with `Feature::PixelCoverage`. One sample is never refused,
//! so every device draws that count's cases.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device, Error, Library, LoadAction,
    PipelineOptions, RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor, Viewport, VisibilityResultMode,
};

/// The kind the early-tests draws of [`the_sample_coverage_applies_after_the_fragment_function`]
/// take on a device that does not state `earlyFragmentMultisampleCoverageAfterSampleCounting`.
pub(super) const PAST_EARLY_MASKS: [Degradation; 1] = [Degradation::VisibilityPastEarlyMask];

/// The target's side.
const SIDE: u64 = 16;

/// The pixels of the target.
const PIXELS: usize = (SIDE * SIDE) as usize;

/// Metal's coverage of `coverage` at `samples` samples (the module docs), by the window pixel's
/// place in its quad, `(x & 1) | (y & 1) << 1`.
fn metal_masks(coverage: f32, samples: u32) -> [u32; 4] {
    if samples == 1 {
        return [u32::from(coverage > 0.5); 4];
    }
    let value = if coverage.is_nan() {
        0.0
    } else {
        coverage.clamp(0.0, 1.0)
    };
    let code = (f64::from(value) * 255.0).round() as u32;
    let step = 64 / samples;
    let kept = (code + step / 2) / step;
    let order: &[u32] = if samples == 4 { &[0, 3, 1, 2] } else { &[0, 1] };
    // The places in the order they take samples: (0, 0), (1, 1), (1, 0), (0, 1).
    let turns = [0usize, 3, 1, 2];
    let mut masks = [0u32; 4];
    for taken in 0..kept {
        let place = turns[(taken % 4) as usize];
        let filled = masks[place].count_ones() as usize;
        masks[place] |= 1 << order[filled];
    }
    masks
}

/// The mask of pixel `i` of the target under `masks`, with the draw's viewport (and so the
/// covered pixels) starting at `(x0, y0)`.
fn want_at(masks: &[u32; 4], i: usize, (x0, y0): (usize, usize)) -> u32 {
    let (x, y) = (i % SIDE as usize, i / SIDE as usize);
    match x >= x0 && y >= y0 {
        true => masks[(x & 1) | (y & 1) << 1],
        false => 0,
    }
}

/// A draw's answers: each pixel's color and depth samples written, the visibility count, and the
/// counter's runs and the coverage they read (by pixel, where the fragment function counts).
#[derive(Debug, PartialEq, Eq)]
struct Drawn {
    colors: Vec<u32>,
    depths: Vec<u32>,
    count: u64,
    runs: Vec<[u32; 2]>,
}

struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

/// What a draw sets.
#[derive(Clone, Copy, Debug)]
struct Case {
    samples: u32,
    coverage: f32,
    sample_mask: u64,
    /// The fragment function (`None`: no fragment function).
    fragment: Option<&'static str>,
    /// The viewport's origin.
    origin: (usize, usize),
}

impl Case {
    fn new(samples: u32, coverage: f32) -> Case {
        Case {
            samples,
            coverage,
            sample_mask: u64::MAX,
            fragment: Some("psc_white"),
            origin: (0, 0),
        }
    }
}

impl Lane {
    fn new(messages: &Messages, cache: &ScratchCache) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |_| {}).unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("psc_sample_coverage.ll")).unwrap();
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

    /// A private texture of `pf`, `samples` deep, that renders and reads.
    fn texture(&self, pf: PixelFormat, samples: u32) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = u64::from(samples);
        }
        self.device.new_texture(&d).unwrap()
    }

    /// Each pixel's mask of the samples of `t` that `kernel` finds written.
    fn read(&self, kernel: &str, t: &Texture) -> Vec<u32> {
        let state = self
            .device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(self.function(kernel)),
                &PipelineOptions::default(),
            )
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
        let out = buffer_with(&self.device, &[0xee; 4 * PIXELS]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        encoder.set_texture(Some(t), 0).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        read_buffer(&out, 0, 4 * PIXELS as u64)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }

    /// The draw of `case`; `None`, with a printed line, where the device refuses its sample count
    /// as one it lacks (the module docs).
    fn draw(&self, case: Case) -> Option<Drawn> {
        let n = case.samples;
        let mut d = RenderPipelineDescriptor::new(self.function("psc_vertex"));
        d.fragment = case.fragment.map(|f| self.function(f));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.depth_format = Some(PixelFormat::Depth32Float);
        d.raster_sample_count = n;
        d.private.sample_coverage = case.coverage;
        d.private.sample_mask = case.sample_mask;
        let pipeline = match self
            .device
            .new_render_pipeline_state(&d, &PipelineOptions::default())
        {
            Ok(pipeline) => pipeline,
            Err(Error::Unsupported(u))
                if matches!(
                    u.feature,
                    Feature::RasterSampleCount | Feature::PixelCoverage
                ) =>
            {
                println!("{}: {n} samples refused: {u:?}", self.device.name());
                return None;
            }
            Err(e) => panic!("{case:?}: {e}"),
        };
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Always;
        state.depth_write_enabled = true;
        let state = self.device.new_depth_stencil_state(&state).unwrap();
        let color = self.texture(PixelFormat::RGBA8Unorm, n);
        let depth = self.texture(PixelFormat::Depth32Float, n);
        let vis = buffer_with(&self.device, &[0xaa; 8]).unwrap();
        let runs = buffer_with(&self.device, &[0; 8 * PIXELS]).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let a = &mut pass.color_attachments[0];
        a.texture = Some(color.clone());
        a.load_action = LoadAction::Clear;
        a.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        a.store_action = StoreAction::Store;
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.visibility_result_buffer = Some(vis.clone());
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_depth_stencil_state(&state).unwrap();
        e.set_fragment_buffer(Some(&runs), 0, 0).unwrap();
        e.set_visibility_result_mode(VisibilityResultMode::Counting, 0)
            .unwrap();
        let (x0, y0) = case.origin;
        if (x0, y0) != (0, 0) {
            let (w, h) = (SIDE as f64 - x0 as f64, SIDE as f64 - y0 as f64);
            e.set_viewport(Viewport::new(x0 as f64, y0 as f64, w, h, 0.0, 1.0))
                .unwrap();
        }
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{case:?}");
        let (colors, depths) = match n {
            1 => (
                self.read("psc_read", &color),
                self.read("psc_read_depth", &depth),
            ),
            _ => (
                self.read("psc_read_samples", &color),
                self.read("psc_read_depth_samples", &depth),
            ),
        };
        let count = u64::from_le_bytes(read_buffer(&vis, 0, 8).unwrap().try_into().unwrap());
        let runs = read_buffer(&runs, 0, 8 * PIXELS as u64)
            .unwrap()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| {
                let word = |i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
                [word(0), word(1)]
            })
            .collect();
        Some(Drawn {
            colors,
            depths,
            count,
            runs,
        })
    }
}

/// The coverages drawn at each sample count: inside `[0, 1]`, both sides of 0.5 and of some
/// levels' edges, past `[0, 1]`, NaN and infinity.
const COVERAGES: [f32; 16] = [
    0.0,
    0.029,
    0.0295,
    0.0607,
    0.0608,
    0.3,
    0.5,
    0.500_000_06,
    0.6,
    0.75,
    0.9705,
    0.99,
    -0.5,
    2.0,
    f32::NAN,
    f32::INFINITY,
];

/// Each coverage of [`COVERAGES`] at 1, 2 and 4 samples, alone, with a sample mask, under a
/// viewport off the window's origin: the color and depth samples each pixel writes and the
/// visibility count are Metal's (the module docs).
#[test]
fn the_sample_coverage_keeps_the_samples_metal_keeps() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::new(&messages, &cache);
    let name = lane.device.name().to_owned();
    let mut cases = Vec::new();
    for samples in [1, 2, 4] {
        for coverage in COVERAGES {
            cases.push(Case::new(samples, coverage));
        }
        let mut masked = Case::new(samples, 0.75);
        masked.sample_mask = 0b0110;
        cases.push(masked);
        let mut offset = Case::new(samples, 0.3);
        offset.origin = (1, 3);
        cases.push(offset);
        let mut offset = Case::new(samples, 0.6);
        offset.origin = (2, 1);
        cases.push(offset);
    }
    let mut wrong = Vec::new();
    let mut drawn = 0;
    for case in &cases {
        let all = (1u32 << case.samples) - 1;
        let masks =
            metal_masks(case.coverage, case.samples).map(|m| m & case.sample_mask as u32 & all);
        let want: Vec<u32> = (0..PIXELS)
            .map(|i| want_at(&masks, i, case.origin))
            .collect();
        let count: u64 = want.iter().map(|m| u64::from(m.count_ones())).sum();
        let Some(got) = lane.draw(*case) else {
            continue;
        };
        drawn += 1;
        if got.colors != want || got.depths != want || got.count != count {
            wrong.push(format!(
                "{case:?}: count {} (Metal's {count}), colors {:x?}, depths {:x?}, Metal's {want:x?}",
                got.count, got.colors, got.depths
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{name}: {} of {drawn} draws differ from Metal's:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    // One sample is never refused: its cases are the first, every coverage and three more.
    assert!(drawn >= COVERAGES.len() + 3, "{name}: {drawn} draws");
    eprintln!(
        "{name}: {drawn} sample-coverage draws keep Metal's samples, {} refused",
        cases.len() - drawn
    );
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// Under the coverage the fragment function still runs for every rasterized sample and reads the
/// rasterized coverage, and the depth and the count keep the covered samples (`psc_counter`, which
/// writes memory, reads `[[sample_mask]]` and writes no color, so Metal runs it once per sample
/// and the colors keep the clear: probe `sc4`'s `seenvoid`); under early tests the depth and the
/// count keep every sample; and a pipeline without a fragment function ignores the coverage. On a
/// device that does not state `earlyFragmentMultisampleCoverageAfterSampleCounting` (ANV on the
/// Iris Xe of RPL-P), Vulkan lets the early-tests function's mask come before the query counts:
/// the draw is counted in `WorkCounts::draws_counted_past_early_masks` and announced
/// (`visibility-past-early-mask`), its count at most Metal's (a `SAMPLE-COVERAGE-PAST-MASK` line),
/// its colors and depths still Metal's.
#[test]
fn the_sample_coverage_applies_after_the_fragment_function() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let lane = Lane::new(&messages, &cache);
    let name = lane.device.name().to_owned();
    // Whether the device states that it counts before an early-tests function's mask
    // (`Degradation::VisibilityPastEarlyMask` where it does not).
    let counts_after = (lane.device.compile_device().profile().limits())
        .early_fragment_multisample_coverage_after_sample_counting;
    let mut wrong = Vec::new();
    let mut drawn = 0;
    for samples in [1, 2, 4] {
        let all = (1u32 << samples) - 1;
        let masks = metal_masks(0.3, samples);
        let covered: Vec<u32> = (0..PIXELS).map(|i| want_at(&masks, i, (0, 0))).collect();
        let count: u64 = covered.iter().map(|m| u64::from(m.count_ones())).sum();
        let every = vec![all; PIXELS];
        let total = u64::from(all.count_ones()) * PIXELS as u64;
        let mut counter = Case::new(samples, 0.3);
        counter.fragment = Some("psc_counter");
        // Each of the three draws is refused on its own: a device that answers a coverage read
        // wrong takes the other two, which read none.
        if let Some(got) = lane.draw(counter) {
            drawn += 1;
            let runs = vec![[samples, all]; PIXELS];
            let cleared = vec![0; PIXELS];
            if got.colors != cleared
                || got.depths != covered
                || got.count != count
                || got.runs != runs
            {
                wrong.push(format!("{samples} samples, counter: {got:x?}"));
            }
        }
        let mut early = Case::new(samples, 0.3);
        early.fragment = Some("psc_early");
        let before = lane.device.work_counts().draws_counted_past_early_masks;
        if let Some(got) = lane.draw(early) {
            drawn += 1;
            let past = lane.device.work_counts().draws_counted_past_early_masks - before;
            // Where the device may count after the function's mask, the draw is counted and
            // announced, and its count may leave out the samples the mask drops.
            let count = match counts_after {
                true => got.count == total,
                false => got.count <= total,
            };
            if got.colors != covered || got.depths != every || !count {
                wrong.push(format!("{samples} samples, early tests: {got:x?}"));
            }
            if past != u64::from(!counts_after) {
                wrong.push(format!(
                    "{samples} samples, early tests: {past} draws counted past early masks"
                ));
            }
            if !counts_after {
                println!(
                    "SAMPLE-COVERAGE-PAST-MASK {name}: {samples} samples, early tests: counted {} \
                     of the {total} samples Metal counts",
                    got.count
                );
            }
        }
        let mut bare = Case::new(samples, 0.3);
        bare.fragment = None;
        if let Some(got) = lane.draw(bare) {
            drawn += 1;
            if got.depths != every || got.count != total {
                wrong.push(format!("{samples} samples, no fragment function: {got:x?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{name}: {} of {drawn} draws differ from Metal's:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    // One sample is never refused: its three draws.
    assert!(drawn >= 3, "{name}: {drawn} draws");
    let [past_mask] = PAST_EARLY_MASKS;
    let announced = (messages.take_notices().iter())
        .filter_map(|m| m.degradation.as_ref())
        .any(|n| n.degradation == past_mask);
    assert_eq!(
        announced, !counts_after,
        "{name}: visibility-past-early-mask announced"
    );
    eprintln!("{name}: {drawn} of 9 coverage draws are Metal's");
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}
