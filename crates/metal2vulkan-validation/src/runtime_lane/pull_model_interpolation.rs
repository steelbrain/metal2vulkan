//! Metal's pull-model interpolation (`interpolant<T, P>` fragment inputs read through
//! `interpolate_at_center`, `_centroid`, `_sample` and `_offset`) on the device the cfg selects,
//! under the validation layer, sample by sample against Metal's own reading of the same draws
//! (`pull_model_interpolation_m3_max.txt`).
//!
//! Metal's reading: the probe `metal2vulkan-conformance/probes/pull_model_interpolation.m` (case
//! `lane`) draws the owned fixture `pull_model_interpolants` this lane's way through plain Metal on
//! an Apple M3 Max under macOS 27.0.1: one triangle ([`POSITIONS`], its vertices of `w` 1, 2 and 4,
//! each handing [`VALUES`] as every varying) over a 6x6 target of `RGBA32Float` attachments cleared
//! to -1000 ([`CLEAR`]), at 1 and 4 samples, through each fragment function of [`draws`]: a
//! perspective and a no-perspective `float4` interpolant read at the center, the centroid, each
//! sample by a buffer's index and by `[[sample_id]]`, and five offsets ([`OFFSETS`]); and the
//! `float2`, `float` and `half` interpolants (`pm_narrow`) and the `half4` one (`pm_half`, into
//! `RGBA16Float` attachments) read every way. Each attachment's every sample is read back: a
//! single-sample one from its storage, a multisampled one through the fixture's `pm_read_samples`.
//!
//! What the probe measured, there and in its other cases:
//!
//! - `interpolate_at_offset` reads at the offset from the pixel's upper-left corner, `(0.5, 0.5)`
//!   its center, as given: an offset of 0.999 at 0.999, one of `1e9` extrapolated, every step of a
//!   1/512-pixel sweep a step of its own, a NaN or infinite one NaN.
//! - `interpolate_at_sample(i)` reads sample `i mod n` at Vulkan's standard positions; at one
//!   sample every index reads the center, bit for bit.
//! - `interpolate_at_centroid` reads the center of a pixel the triangle covers whole, else its
//!   lowest covered sample's position.
//! - Only `pm_sample_id` runs once per sample; the rest run once per pixel.
//!
//! Where the profile takes `DriverDeviation::MetalInterpolation` (MoltenVK, which hands each read to
//! Metal's own interpolation functions) every sample of every draw is Metal's, word for word.
//! Elsewhere each value lies within the device's interpolation of Metal's
//! (`PrecisionNote::DeviceInterpolation`, [`device_bound`]), and further where Vulkan leaves the
//! location the device's: a centroid of a multisampled pixel anywhere in its covered part
//! (`PrecisionNote::MultisamplePosition`), an offset on the device's grid of at least sixteenths
//! (`PrecisionNote::DeviceInterpolationOffset`), each bounded by the value's change from pixel to
//! pixel in Metal's reading ([`Slopes`]). Every sample covers where Metal's covers, on every device.
//! The lane's offsets lie inside the range every Vulkan device takes, so no clamp moves them. On a
//! driver that shades a triangle it clips as several
//! (`DriverWorkaround::ClippedTrianglesSplitPixelFragments`: lavapipe) a centroid read of a
//! multisampled pixel is refused as a centroid varying is, and those draws are not drawn.

use super::extended_range_targets::half_value;
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverWorkaround, VkProperty};
use metal2vulkan::error::{DriverDeviation, Error as CompileError, Feature, Needs};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan::Budget;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandQueue, Device, Error, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, ResourceOptions, RuntimeGap, SamplePosition, Size, StoreAction, Texture,
    TextureDescriptor,
};
use std::collections::BTreeMap;

/// Metal's reading of the lane's draws: `DRAW <what> samples <n>`, then each covered pixel's
/// `x y m<covered samples, hex>` and its attachments' words where its covered samples agree, else
/// one line `x y s<sample>` per covered sample; after `#` comment lines.
const METAL: &str = include_str!("pull_model_interpolation_m3_max.txt");

/// The target's side.
const SIDE: u64 = 6;

/// The value every attachment is cleared to, which no interpolation of [`VALUES`] reaches.
const CLEAR: f32 = -1000.0;

/// The triangle's clip-space positions: `w` 1, 2 and 4.
const POSITIONS: [[f32; 4]; 3] = [
    [-0.9, -0.85, 0.0, 1.0],
    [0.95 * 2.0, -0.6 * 2.0, 0.0, 2.0],
    [-0.3 * 4.0, 0.9 * 4.0, 0.0, 4.0],
];

/// The value each vertex hands as every varying.
const VALUES: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 3.0],
    [0.0, 1.0, 0.0, -2.0],
    [0.0, 0.0, 1.0, 0.5],
];

/// The offsets `pm_offset` reads at.
const OFFSETS: [[f32; 2]; 5] = [
    [0.0, 0.0],
    [0.5, 0.5],
    [0.25, 0.8125],
    [0.9375, 0.0625],
    [0.40625, 0.59375],
];

/// The sample and offset `pm_narrow` and `pm_half` read at.
const NARROW_SAMPLE: u32 = 0;
const NARROW_OFFSET: [f32; 2] = [0.25, 0.75];

/// Where an attachment's value is read.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Read {
    Center,
    Centroid,
    Sample,
    Offset,
}

/// Which varyings an attachment's lanes interpolate: the `float4` pair's perspective one or
/// no-perspective one (the `PmPair` functions' attachments 0 and 1); `pm_narrow`'s `float2`
/// (perspective), `float` and `half` (no perspective); `pm_half`'s `half4` (perspective).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lanes {
    Perspective,
    NoPerspective,
    Narrow,
    Half,
}

/// One draw of the lane: the name Metal's reading gives it, the fragment function, its
/// attachments' format, how each attachment is read and which lanes it holds, and the fragment
/// bytes at buffers 0 and 1.
struct Draw {
    what: String,
    function: &'static str,
    format: PixelFormat,
    reads: Vec<(Read, Lanes)>,
    bytes: Vec<Vec<u8>>,
}

/// `words`' little-endian bytes.
fn bytes_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// `values`' little-endian bytes.
fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The lane's draws at `n` samples, in Metal's reading's order.
fn draws(n: u32) -> Vec<Draw> {
    let pair = |read| vec![(read, Lanes::Perspective), (read, Lanes::NoPerspective)];
    let plain = |what: &str, function, read| Draw {
        what: what.to_owned(),
        function,
        format: PixelFormat::RGBA32Float,
        reads: pair(read),
        bytes: Vec::new(),
    };
    let mut out = vec![
        plain("center", "pm_center", Read::Center),
        plain("centroid", "pm_centroid", Read::Centroid),
    ];
    for i in 0..n {
        out.push(Draw {
            bytes: vec![bytes_of(&[i])],
            ..plain(&format!("sample {i}"), "pm_sample", Read::Sample)
        });
    }
    out.push(plain("sample_id", "pm_sample_id", Read::Sample));
    for [x, y] in OFFSETS {
        let what = format!("offset {:08x} {:08x}", x.to_bits(), y.to_bits());
        out.push(Draw {
            bytes: vec![bytes_of(&[x.to_bits(), y.to_bits()])],
            ..plain(&what, "pm_offset", Read::Offset)
        });
    }
    let every = |lanes| {
        [Read::Center, Read::Centroid, Read::Sample, Read::Offset]
            .into_iter()
            .map(|read| (read, lanes))
            .collect()
    };
    let narrow = vec![
        bytes_of(&[NARROW_SAMPLE]),
        bytes_of(&NARROW_OFFSET.map(f32::to_bits)),
    ];
    out.push(Draw {
        what: "narrow".to_owned(),
        function: "pm_narrow",
        format: PixelFormat::RGBA32Float,
        reads: every(Lanes::Narrow),
        bytes: narrow.clone(),
    });
    out.push(Draw {
        what: "half".to_owned(),
        function: "pm_half",
        format: PixelFormat::RGBA16Float,
        reads: every(Lanes::Half),
        bytes: narrow,
    });
    out
}

/// A draw's every sample: by pixel `(x, y)` and sample, each covered sample's attachments' four
/// words.
type Samples = BTreeMap<(u64, u64, u32), Vec<[u32; 4]>>;

/// Metal's reading by `(what, samples)` ([`METAL`]), each line expanded to its covered samples.
fn metal_reading() -> BTreeMap<(String, u32), Samples> {
    let mut out: BTreeMap<(String, u32), Samples> = BTreeMap::new();
    let mut open: Option<(String, u32)> = None;
    for line in METAL.lines().filter(|line| !line.starts_with('#')) {
        if let Some(rest) = line.strip_prefix("DRAW ") {
            let (what, n) = rest.rsplit_once(" samples ").unwrap();
            let key = (what.to_owned(), n.parse().unwrap());
            out.insert(key.clone(), BTreeMap::new());
            open = Some(key);
            continue;
        }
        let draw = out.get_mut(open.as_ref().unwrap()).unwrap();
        let fields: Vec<&str> = line.split(' ').collect();
        let (x, y): (u64, u64) = (fields[0].parse().unwrap(), fields[1].parse().unwrap());
        let words: Vec<u32> = fields[3..]
            .iter()
            .map(|w| u32::from_str_radix(w, 16).unwrap())
            .collect();
        let texels: Vec<[u32; 4]> = words.as_chunks::<4>().0.to_vec();
        let samples: Vec<u32> = match fields[2].split_at(1) {
            ("m", mask) => {
                let mask = u32::from_str_radix(mask, 16).unwrap();
                (0..32).filter(|s| mask >> s & 1 != 0).collect()
            }
            ("s", s) => vec![s.parse().unwrap()],
            _ => panic!("{line}"),
        };
        for s in samples {
            draw.insert((x, y, s), texels.clone());
        }
    }
    out
}

/// How much each lane of the `float4` pair changes at most from a pixel to its neighbor, right or
/// down, in Metal's reading of the center at one sample: by perspective (0) and not (1), lane.
struct Slopes([[f32; 4]; 2]);

impl Slopes {
    fn of(center: &Samples) -> Slopes {
        let mut most = [[0f32; 4]; 2];
        for (&(x, y, _), here) in center {
            for next in [(x + 1, y, 0), (x, y + 1, 0)] {
                let Some(there) = center.get(&next) else {
                    continue;
                };
                for (a, row) in most.iter_mut().enumerate() {
                    for (c, m) in row.iter_mut().enumerate() {
                        let d = f32::from_bits(here[a][c]) - f32::from_bits(there[a][c]);
                        *m = m.max(d.abs());
                    }
                }
            }
        }
        Slopes(most)
    }

    /// The slope of lane `c` of an attachment holding `lanes`.
    fn of_lane(&self, lanes: Lanes, c: usize) -> f32 {
        let [perspective, linear] = &self.0;
        match lanes {
            Lanes::Perspective | Lanes::Half => perspective[c],
            Lanes::NoPerspective => linear[c],
            // `p2.xy` (perspective), `s` (the no-perspective `z`), `hn` (its `w`).
            Lanes::Narrow if c < 2 => perspective[c],
            Lanes::Narrow => linear[c],
        }
    }
}

/// How far a device not measured to interpolate as Metal does may be from Metal's `metal`: the
/// device's interpolation (`PrecisionNote::DeviceInterpolation`, about `1.5e-4` measured, `3e-4`
/// on lavapipe, held at `1e-3` of the value's magnitude or 1), plus where Vulkan leaves the
/// location the device's: a multisampled pixel's centroid anywhere in it (twice the slope), an
/// offset on a grid of sixteenths (half a step each way: an eighth of the slope, doubled), and a
/// `half` result's own rounding.
fn device_bound(metal: f32, read: Read, lanes: Lanes, slope: f32, n: u32) -> f32 {
    let mut bound = 1e-3 * metal.abs().max(1.0);
    bound += match read {
        Read::Centroid if n > 1 => 2.0 * slope,
        Read::Offset => slope / 8.0,
        _ => 0.0,
    };
    if lanes == Lanes::Half || (lanes == Lanes::Narrow && read == Read::Center) {
        bound += metal.abs() * 2f32.powi(-10) + 2f32.powi(-14);
    }
    bound
}

struct Lane {
    device: Device,
    queue: CommandQueue,
    /// The owned fixture `pull_model_interpolants`.
    library: Library,
}

impl Lane {
    fn new(messages: &Messages, cache: &ScratchCache) -> Lane {
        let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |_| {}).unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("pull_model_interpolants.ll")).unwrap();
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

    /// `draw`'s pipeline at `n` samples.
    fn descriptor(&self, draw: &Draw, n: u32) -> RenderPipelineDescriptor {
        let mut d = RenderPipelineDescriptor::new(self.function("pm_place"));
        d.fragment = Some(self.function(draw.function));
        for i in 0..draw.reads.len() {
            d.color_attachments[i] = Some(ColorAttachmentDescriptor::new(draw.format));
        }
        d.raster_sample_count = n;
        d
    }

    /// An attachment of `draw` at `n` samples: shared at one sample, private and read by the
    /// fixture's kernel at more.
    fn target(&self, format: PixelFormat, n: u32) -> Texture {
        let mut t = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
        t.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        if n > 1 {
            t.texture_type = TextureType::Type2DMultisample;
            t.sample_count = u64::from(n);
            t.resource_options = ResourceOptions::private();
        } else {
            t.resource_options = ResourceOptions::shared();
        }
        self.device.new_texture(&t).unwrap()
    }

    /// The pass drawing into `targets`, each cleared to [`CLEAR`], with sample `positions`.
    fn pass(targets: &[Texture], positions: Vec<SamplePosition>) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::new();
        for (i, target) in targets.iter().enumerate() {
            let color = &mut pass.color_attachments[i];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            let clear = f64::from(CLEAR);
            color.clear_color = ClearColor::new(clear, clear, clear, clear);
            color.store_action = StoreAction::Store;
        }
        pass.sample_positions = positions;
        pass
    }

    /// Encode one draw of the triangle through `pipeline` with `bytes` at the fragment buffers,
    /// into `pass`, and answer the encoder's draw call.
    fn encode_draw(
        &self,
        command_buffer: &metal2vulkan_runtime::CommandBuffer,
        pass: &RenderPassDescriptor,
        pipeline: &metal2vulkan_runtime::RenderPipelineState,
        bytes: &[Vec<u8>],
    ) -> Result<(), Error> {
        let at = buffer_with(&self.device, &float_bytes(POSITIONS.as_flattened())).unwrap();
        let values = buffer_with(&self.device, &float_bytes(VALUES.as_flattened())).unwrap();
        let mut e = command_buffer.render_command_encoder(pass).unwrap();
        e.set_render_pipeline_state(pipeline).unwrap();
        e.set_vertex_buffer(Some(&at), 0, 0).unwrap();
        e.set_vertex_buffer(Some(&values), 0, 1).unwrap();
        for (index, b) in bytes.iter().enumerate() {
            e.set_fragment_bytes(b, index as u32).unwrap();
        }
        let drawn = e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
        e.end_encoding().unwrap();
        drawn
    }

    /// `draw` at `n` samples: every covered sample's attachments' words, a `half` attachment's
    /// values widened to `float`.
    fn draw(&self, draw: &Draw, n: u32) -> Samples {
        let pipeline = self
            .device
            .new_render_pipeline_state(&self.descriptor(draw, n), &PipelineOptions::default())
            .unwrap_or_else(|e| panic!("{} at {n} samples: {e}", draw.what));
        let targets: Vec<Texture> = draw
            .reads
            .iter()
            .map(|_| self.target(draw.format, n))
            .collect();
        let command_buffer = self.queue.command_buffer().unwrap();
        self.encode_draw(
            &command_buffer,
            &Self::pass(&targets, Vec::new()),
            &pipeline,
            &draw.bytes,
        )
        .unwrap_or_else(|e| panic!("{} at {n} samples: {e}", draw.what));
        let per = (SIDE * SIDE) as usize * n as usize;
        let words = (n > 1).then(|| self.encode_reads(&command_buffer, &targets, n));
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{} at {n} samples", draw.what);
        // Each attachment's texels: attachment-major, then row, column and sample.
        let texels: Vec<[f32; 4]> = match words {
            Some(words) => {
                let bytes = read_buffer(&words, 0, words.length()).unwrap();
                bytes
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(|t| {
                        let (w, _) = t.as_chunks::<4>();
                        std::array::from_fn(|i| f32::from_le_bytes(w[i]))
                    })
                    .collect()
            }
            None => targets
                .iter()
                .flat_map(|target| read_single(target, draw.format))
                .collect(),
        };
        let mut out = Samples::new();
        for y in 0..SIDE {
            for x in 0..SIDE {
                for s in 0..n {
                    let at = |a: usize| {
                        texels[a * per + ((y * SIDE + x) as usize) * n as usize + s as usize]
                    };
                    if at(0)[0] == CLEAR {
                        continue;
                    }
                    let words = (0..draw.reads.len())
                        .map(|a| at(a).map(f32::to_bits))
                        .collect();
                    out.insert((x, y, s), words);
                }
            }
        }
        out
    }

    /// Encode the fixture's `pm_read_samples` over each of `targets`, of `n` samples, into one
    /// buffer: attachment-major, then row, column and sample, a `float4` each.
    fn encode_reads(
        &self,
        command_buffer: &metal2vulkan_runtime::CommandBuffer,
        targets: &[Texture],
        n: u32,
    ) -> metal2vulkan_runtime::Buffer {
        let function = self
            .library
            .new_function_with_name("pm_read_samples")
            .unwrap();
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let per = SIDE * SIDE * u64::from(n) * 16;
        let words = self
            .device
            .new_buffer(per * targets.len() as u64, ResourceOptions::shared())
            .unwrap();
        for (i, target) in targets.iter().enumerate() {
            let mut e = command_buffer.compute_command_encoder().unwrap();
            e.set_compute_pipeline_state(&pipeline).unwrap();
            e.set_texture(Some(target), 0).unwrap();
            e.set_buffer(Some(&words), per * i as u64, 0).unwrap();
            e.dispatch_threadgroups(Size::new(SIDE, SIDE, u64::from(n)), Size::new(1, 1, 1))
                .unwrap();
            e.end_encoding().unwrap();
        }
        words
    }
}

/// The `float` texels of `target`, a single-sample attachment of `format`.
fn read_single(target: &Texture, format: PixelFormat) -> Vec<[f32; 4]> {
    let texel = match format {
        PixelFormat::RGBA16Float => 8,
        _ => 16,
    };
    let row = SIDE * texel;
    let mut out = vec![0u8; (row * SIDE) as usize];
    target
        .get_bytes(
            &mut out,
            row,
            row * SIDE,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    match texel {
        8 => out
            .as_chunks::<8>()
            .0
            .iter()
            .map(|t| {
                let (h, _) = t.as_chunks::<2>();
                std::array::from_fn(|i| half_value(u16::from_le_bytes(h[i])))
            })
            .collect(),
        _ => out
            .as_chunks::<16>()
            .0
            .iter()
            .map(|t| {
                let (w, _) = t.as_chunks::<4>();
                std::array::from_fn(|i| f32::from_le_bytes(w[i]))
            })
            .collect(),
    }
}

/// Every draw of the lane, at 1 sample and at 4 where the device renders them, covers Metal's
/// samples, and reads Metal's words where the profile takes `MetalInterpolation`, else values
/// within [`device_bound`] of Metal's (the module docs). The pipelines that read at an offset, and
/// those alone, record `PrecisionNote::DeviceInterpolationOffset` and require the device's offset
/// origin: `InterpolateAtOffsetFromSevenSixteenths` on MoltenVK, `InterpolateAtOffsetCentered`
/// elsewhere.
#[test]
fn interpolants_read_metal_s_values() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let lane = Lane::new(&messages, &cache);
        let name = lane.device.name().to_owned();
        let profile = lane.device.compile_device().profile();
        let exact = profile.takes_driver_deviation(DriverDeviation::MetalInterpolation);
        let origin = match profile
            .enabled()
            .properties
            .contains(&VkProperty::InterpolateAtOffsetCentered)
        {
            true => VkProperty::InterpolateAtOffsetCentered,
            false => VkProperty::InterpolateAtOffsetFromSevenSixteenths,
        };
        println!("PM {name}: MetalInterpolation {exact}, offsets from {origin:?}");
        let reading = metal_reading();
        let slopes = Slopes::of(&reading[&("center".to_owned(), 1)]);
        let counts: Vec<u32> = [1, 4]
            .into_iter()
            .filter(|&n| {
                lane.device
                    .properties()
                    .supports_texture_sample_count(n.into())
                    == Ok(true)
            })
            .collect();
        assert!(counts.contains(&1), "{name}: {counts:?}");
        let mut wrong = Vec::new();
        let split = profile
            .workarounds()
            .contains(&DriverWorkaround::ClippedTrianglesSplitPixelFragments);
        let (mut drawn, mut refused) = (0, 0);
        for &n in &counts {
            for draw in draws(n) {
                let what = format!("{} at {n} samples", draw.what);
                let compiled = lane
                    .device
                    .compile_device()
                    .compile_render_pipeline(&lane.descriptor(&draw, n), &Budget::standard());
                // A centroid read of a multisampled pixel, refused as a centroid varying is on a
                // driver that shades a triangle it clips as several.
                let centroid = draw.reads.iter().any(|(read, _)| *read == Read::Centroid);
                if centroid && n > 1 && split {
                    let e = compiled.expect_err(&what);
                    let defect =
                        Needs::DriverDefect(DriverWorkaround::ClippedTrianglesSplitPixelFragments);
                    assert!(
                        matches!(&e, CompileError::Unsupported(u)
                            if u.feature == Feature::CentroidVarying && u.needs == defect),
                        "{name} {what}: {e:?}"
                    );
                    println!("PM {name} {what}: refused as a centroid varying is");
                    refused += 1;
                    continue;
                }
                let compiled = compiled.unwrap_or_else(|e| panic!("{what}: {e}"));
                let noted = compiled.stages.iter().any(|s| {
                    s.precision_notes
                        .contains(&PrecisionNote::DeviceInterpolationOffset)
                });
                let offsets = draw.reads.iter().any(|(read, _)| *read == Read::Offset);
                assert_eq!(noted, offsets, "{name} {what}: the offset note");
                assert_eq!(
                    compiled.requirements.properties.contains(&origin),
                    offsets,
                    "{name} {what}: the offset origin {origin:?}"
                );
                let theirs = &reading[&(draw.what.clone(), n)];
                let ours = lane.draw(&draw, n);
                let covered = |s: &Samples| s.keys().copied().collect::<Vec<_>>();
                assert_eq!(
                    covered(&ours),
                    covered(theirs),
                    "{name} {what}: the samples covered"
                );
                let mut most = 0f32;
                for (at, texels) in &ours {
                    let metal = &theirs[at];
                    for (a, (&(read, lanes), (ours, metal))) in
                        draw.reads.iter().zip(texels.iter().zip(metal)).enumerate()
                    {
                        for c in 0..4 {
                            let (o, m) = (f32::from_bits(ours[c]), f32::from_bits(metal[c]));
                            most = most.max((o - m).abs());
                            let agrees = match exact {
                                true => ours[c] == metal[c],
                                false => {
                                    let slope = slopes.of_lane(lanes, c);
                                    (o - m).abs() <= device_bound(m, read, lanes, slope, n)
                                }
                            };
                            if !agrees {
                                wrong.push(format!(
                                    "{what}, pixel {:?} attachment {a} lane {c}: {o:e} \
                                     ({:08x}), Metal's {m:e} ({:08x})",
                                    at, ours[c], metal[c]
                                ));
                            }
                        }
                    }
                }
                println!(
                    "PM {name} {what}: {} samples, most apart {most:e}",
                    ours.len()
                );
                drawn += 1;
            }
        }
        let draws_read = reading.keys().filter(|(_, n)| counts.contains(n)).count();
        assert_eq!(
            drawn + refused,
            draws_read,
            "{name}: every draw Metal's reading holds"
        );
        assert!(
            wrong.is_empty(),
            "{name}: {} values off:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// A draw at custom sample positions through a pipeline that reads an interpolant at a sample it
/// names is refused, `NotImplemented(InterpolateAtSampleWithPositions)`: Vulkan takes no
/// `InterpolateAtSample` in a pipeline made with custom sample locations. One that reads at an
/// offset draws there. On a device without programmable positions, nothing to check.
#[test]
fn interpolation_at_a_sample_is_refused_at_custom_positions() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let lane = Lane::new(&messages, &cache);
        let name = lane.device.name().to_owned();
        let programmable = lane
            .device
            .properties()
            .are_programmable_sample_positions_supported()
            == Ok(true);
        let four = lane.device.properties().supports_texture_sample_count(4) == Ok(true);
        if !(programmable && four) {
            println!("PM {name}: no programmable positions at 4 samples, nothing to check");
            return;
        }
        let positions = vec![SamplePosition::new(0.25, 0.25); 4];
        let all = draws(4);
        let named = |function: &str| all.iter().find(|d| d.function == function).unwrap();
        for (draw, refused) in [(named("pm_sample"), true), (named("pm_offset"), false)] {
            let pipeline = lane
                .device
                .new_render_pipeline_state(&lane.descriptor(draw, 4), &PipelineOptions::default())
                .unwrap();
            let targets: Vec<Texture> = draw
                .reads
                .iter()
                .map(|_| lane.target(draw.format, 4))
                .collect();
            let command_buffer = lane.queue.command_buffer().unwrap();
            let drawn = lane.encode_draw(
                &command_buffer,
                &Lane::pass(&targets, positions.clone()),
                &pipeline,
                &draw.bytes,
            );
            match refused {
                true => assert_eq!(
                    drawn,
                    Err(Error::NotImplemented(
                        RuntimeGap::InterpolateAtSampleWithPositions
                    )),
                    "{name}: {}",
                    draw.function
                ),
                false => {
                    drawn.unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(command_buffer.error(), None, "{name}: {}", draw.function);
                }
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}
