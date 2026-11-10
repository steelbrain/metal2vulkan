//! Samples through a runtime sampler whose min or mag filter is `MTLSamplerMinMagFilterCubic`
//! (raw value 2) on the device the cfg selects, under the validation layer.
//!
//! Metal's answer (our probe, on an M4 under macOS 27.0.1, and our probes on the
//! M3 Max under macOS 26.7, the chips alike on every row): without a mip filter, a sample through
//! such a sampler is bit for bit the sample through a constexpr `filter::bicubic` sampler of the
//! same address modes, border color and coordinates, whatever the other filter (nearest, linear or
//! cubic) and the level the sample asks; a `clamp_to_zero` axis reads the sampler's border color,
//! as every runtime sampler's does. The runtime binds the slot under its pipeline's cubic-sampler
//! variant, whose function emulates the constexpr bicubic sampler's filter, so here the two samples
//! of each lane must be the same bits. A cubic sampler with a mip filter, whose level Metal's
//! answers do not settle, is refused typed (`NotImplemented(SamplerBicubicFilter)`) at the
//! dispatch.
//!
//! The owned fixture `gsm2_cubic_samplers` writes, per lane, the runtime sampler's sample and
//! then the constexpr sampler's that `which` names (`gsm2_cubic_samples`), or, through `half`,
//! the runtime sampler's sample at level 2.5 and then the constexpr edge sampler's implicit one
//! (`gsm2_cubic_half_samples`). Here 12x10 textures of five formats (8-bit unorm, half, float,
//! one-channel unorm and the packed small floats) of random finite texels, denormals included,
//! and a 16x16 grid of lanes from 0.6 of the texture below it to 0.6 past it on both axes.

use super::degrade_notices::on_device;
use super::degrade_notices::on_device_with;
use super::{at_plan, buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::error::{BindError, Error as CompileError, Feature};
use metal2vulkan::metal::{
    PixelFormat, ResourceUsage, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    SamplerMipFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferError, CommandBufferStatus, CommandQueue, ComputePipelineState, Device, Error,
    PipelineOptions, Region, ResourceOptions, SamplerState, Size, Texture, TextureDescriptor,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The lanes of a threadgroup.
const GROUP: u64 = 32;

/// The lanes on each axis of the grid.
const SIDE: u32 = 16;

/// The textures' size: at least four texels each way, so the bicubic filter is not the small
/// level's linear one.
const WIDTH: u32 = 12;
const HEIGHT: u32 = 10;

/// The constexpr samplers the fixture's `which` names: the address mode of every axis, the
/// border color, and pixel coordinates.
const STATES: [(&str, SamplerAddressMode, SamplerBorderColor, bool); 7] = [
    (
        "edge",
        SamplerAddressMode::ClampToEdge,
        SamplerBorderColor::TransparentBlack,
        false,
    ),
    (
        "repeat",
        SamplerAddressMode::Repeat,
        SamplerBorderColor::TransparentBlack,
        false,
    ),
    (
        "mirror",
        SamplerAddressMode::MirrorRepeat,
        SamplerBorderColor::TransparentBlack,
        false,
    ),
    (
        "zero",
        SamplerAddressMode::ClampToZero,
        SamplerBorderColor::TransparentBlack,
        false,
    ),
    (
        "white border",
        SamplerAddressMode::ClampToBorderColor,
        SamplerBorderColor::OpaqueWhite,
        false,
    ),
    (
        "black border",
        SamplerAddressMode::ClampToBorderColor,
        SamplerBorderColor::OpaqueBlack,
        false,
    ),
    (
        "pixel edge",
        SamplerAddressMode::ClampToEdge,
        SamplerBorderColor::TransparentBlack,
        true,
    ),
];

/// The min and mag filters measured: a cubic one beside each filter, both ways.
const FILTERS: [(SamplerMinMagFilter, SamplerMinMagFilter); 5] = [
    (SamplerMinMagFilter::Cubic, SamplerMinMagFilter::Cubic),
    (SamplerMinMagFilter::Cubic, SamplerMinMagFilter::Linear),
    (SamplerMinMagFilter::Linear, SamplerMinMagFilter::Cubic),
    (SamplerMinMagFilter::Cubic, SamplerMinMagFilter::Nearest),
    (SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Cubic),
];

/// The formats sampled, one of each family the bicubic filter treats apart.
const FORMATS: [PixelFormat; 5] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA32Float,
    PixelFormat::R8Unorm,
    PixelFormat::RG11B10Float,
];

/// A deterministic stream of random words (xorshift64*).
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A word below `n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The bytes of a texel of `format`.
fn texel_size(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::R8Unorm => 1,
        PixelFormat::RGBA16Float => 8,
        PixelFormat::RGBA32Float => 16,
        _ => 4,
    }
}

/// `count` random finite texels of `format`, denormals among them.
fn texel_bytes(format: PixelFormat, count: u32, seed: u64) -> Vec<u8> {
    let mut r = Random(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = Vec::new();
    for _ in 0..count {
        match format {
            PixelFormat::RGBA16Float => {
                for _ in 0..4 {
                    // Exponents 0 (denormal) to 17 (below 8).
                    let h = (r.below(2) << 15) | (r.below(18) << 10) | r.below(1 << 10);
                    out.extend((h as u16).to_le_bytes());
                }
            }
            PixelFormat::RGBA32Float => {
                for _ in 0..4 {
                    // Exponents 0 (denormal) and 100 to 129 (below 8).
                    let e = match r.below(31) {
                        0 => 0,
                        e => 99 + e,
                    };
                    let f = (r.below(2) << 31) | (e << 23) | r.below(1 << 23);
                    out.extend((f as u32).to_le_bytes());
                }
            }
            PixelFormat::RG11B10Float => {
                // Each channel's exponent below 31 (finite), its mantissa random.
                let red = (r.below(31) << 6) | r.below(1 << 6);
                let green = (r.below(31) << 6) | r.below(1 << 6);
                let blue = (r.below(31) << 5) | r.below(1 << 5);
                let packed = red | (green << 11) | (blue << 22);
                out.extend((packed as u32).to_le_bytes());
            }
            _ => {
                for _ in 0..texel_size(format) {
                    out.push(r.below(256) as u8);
                }
            }
        }
    }
    out
}

/// A `WIDTH` x `HEIGHT` texture of `format` with random texels, its level `l` (of `levels`)
/// other random texels, so a sample that read a level past the base would differ.
fn texture(device: &Device, format: PixelFormat, levels: bool, seed: u64) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(format, u64::from(WIDTH), u64::from(HEIGHT), levels);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let count = if levels {
        descriptor.mipmap_level_count
    } else {
        1
    };
    for level in 0..count {
        let (w, h) = ((WIDTH >> level).max(1), (HEIGHT >> level).max(1));
        let bytes = texel_bytes(format, w * h, seed * 16 + level);
        texture
            .replace_region(
                Region::new_2d(0, 0, u64::from(w), u64::from(h)),
                level,
                0,
                &bytes,
                u64::from(w * texel_size(format)),
                0,
            )
            .unwrap();
    }
    texture
}

/// The lanes' normalized coordinates: a `SIDE` x `SIDE` grid from -0.6 to 1.6 on both axes, off
/// the texels' centres and edges.
fn coordinates() -> Vec<[f32; 2]> {
    let at = |k: u32| -0.6 + 2.2 * (k as f32 + 0.37) / SIDE as f32;
    (0..SIDE * SIDE)
        .map(|lane| [at(lane % SIDE), at(lane / SIDE)])
        .collect()
}

/// The coordinates' bytes, in texels where `pixel`.
fn coordinate_bytes(pixel: bool) -> Vec<u8> {
    let scale = if pixel {
        [WIDTH as f32, HEIGHT as f32]
    } else {
        [1.0, 1.0]
    };
    coordinates()
        .into_iter()
        .flat_map(|[u, v]| [u * scale[0], v * scale[1]])
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("gsm2_cubic_samplers.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A sampler of `filters` addressing every axis by `mode`, with `border` and pixel coordinates
/// where `pixel`, and mip filter `mip`.
fn cubic_sampler(
    device: &Device,
    (min, mag): (SamplerMinMagFilter, SamplerMinMagFilter),
    (mode, border, pixel): (SamplerAddressMode, SamplerBorderColor, bool),
    mip: SamplerMipFilter,
) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = min;
    descriptor.mag_filter = mag;
    descriptor.mip_filter = mip;
    descriptor.s_address_mode = mode;
    descriptor.t_address_mode = mode;
    descriptor.r_address_mode = mode;
    descriptor.border_color = border;
    descriptor.normalized_coordinates = !pixel;
    device.new_sampler_state(&descriptor).unwrap()
}

/// One dispatch of `state` over the grid with `texture` and `sampler` at slot 0, the
/// coordinates at buffer 0 and `which` (where given) at buffer 2: buffer 1's `bytes` per result,
/// two results per lane.
fn dispatch(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    (texture, sampler): (&Texture, &SamplerState),
    coords: &[u8],
    which: Option<u32>,
    bytes: u64,
) -> Vec<u8> {
    let device = queue.device();
    let lanes = u64::from(SIDE * SIDE);
    let coords = buffer_with(device, coords).unwrap();
    let out = buffer_with(device, &vec![0xab; (lanes * 2 * bytes) as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(sampler), 0).unwrap();
    encoder.set_buffer(Some(&coords), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    if let Some(which) = which {
        encoder.set_bytes(&which.to_le_bytes(), 2).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(lanes, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    read_buffer(&out, 0, lanes * 2 * bytes).unwrap()
}

/// The lanes whose two results (each `bytes` long) differ in any bit.
fn differing(out: &[u8], bytes: usize) -> Vec<usize> {
    out.chunks_exact(2 * bytes)
        .enumerate()
        .filter(|(_, pair)| pair[..bytes] != pair[bytes..])
        .map(|(lane, _)| lane)
        .collect()
}

/// The grid is whole threadgroups and reaches below and past the texture on both axes; the
/// random texels hold denormals and no infinity or NaN.
#[test]
fn the_lanes_reach_past_the_texture_and_the_texels_are_finite() {
    let points = coordinates();
    assert_eq!(points.len() as u64 % GROUP, 0, "whole threadgroups");
    for axis in 0..2 {
        assert!(points.iter().any(|p| p[axis] < -0.5));
        assert!(points.iter().any(|p| p[axis] > 1.5));
    }
    let halves = texel_bytes(PixelFormat::RGBA16Float, 1024, 1);
    let halves: Vec<u16> = halves
        .as_chunks::<2>()
        .0
        .iter()
        .map(|h| u16::from_le_bytes([h[0], h[1]]))
        .collect();
    assert!(halves.iter().all(|h| h & 0x7c00 != 0x7c00));
    assert!(halves.iter().any(|h| h & 0x7c00 == 0 && h & 0x3ff != 0));
    let floats = texel_bytes(PixelFormat::RGBA32Float, 1024, 1);
    let floats: Vec<f32> = floats
        .as_chunks::<4>()
        .0
        .iter()
        .map(|f| f32::from_le_bytes([f[0], f[1], f[2], f[3]]))
        .collect();
    assert!(floats.iter().all(|f| f.is_finite()));
    assert!(floats.iter().any(|f| f.is_subnormal()));
}

/// Every lane's runtime cubic sample is the constexpr bicubic sampler's, bit for bit, for each
/// format, address mode, border color, coordinate kind and min/mag pairing. Prints, per case,
/// the lanes checked and those that differ.
#[test]
fn cubic_samplers_sample_as_constexpr_bicubic_ones() {
    on_device(|device, _| {
        let pipeline = state(device, "gsm2_cubic_samples");
        let queue = device.new_command_queue().unwrap();
        let mut wrong = Vec::new();
        for (seed, format) in (1u64..).zip(FORMATS) {
            let texture = texture(device, format, false, seed);
            for (which, (name, mode, border, pixel)) in (0u32..).zip(STATES) {
                let coords = coordinate_bytes(pixel);
                for filters in FILTERS {
                    let sampler = cubic_sampler(
                        device,
                        filters,
                        (mode, border, pixel),
                        SamplerMipFilter::NotMipmapped,
                    );
                    let out = dispatch(
                        &queue,
                        &pipeline,
                        (&texture, &sampler),
                        &coords,
                        Some(which),
                        16,
                    );
                    let lanes = differing(&out, 16);
                    println!(
                        "CUBIC-SAMPLERS {} {format:?} {name} {filters:?}: {} lanes, {} differ \
                         from the constexpr bicubic sample",
                        device.name(),
                        SIDE * SIDE,
                        lanes.len()
                    );
                    for lane in lanes.iter().take(4) {
                        let pair = &out[lane * 32..lane * 32 + 32];
                        wrong.push(format!(
                            "{format:?} {name} {filters:?} lane {lane}: runtime {:02x?}, \
                             constexpr {:02x?}",
                            &pair[..16],
                            &pair[16..]
                        ));
                    }
                }
            }
        }
        for line in wrong.iter().take(80) {
            println!("CUBIC-SAMPLERS-DIFFER {}: {line}", device.name());
        }
        assert!(wrong.is_empty(), "{} lanes differ", wrong.len());
    });
}

/// A cubic sampler's compare function leaves its plain sample alone, as Metal does (an M4 under
/// macOS 27.0.1: four color formats and two depth formats, three compare functions, 2,048 lanes
/// each, every lane the sampler's without the function and the constexpr bicubic sampler's; Metal's
/// debug layer stops making such a sampler): every lane of each format and address mode, through
/// each compare function, is the constexpr bicubic sampler's, bit for bit.
#[test]
fn a_cubic_sampler_s_compare_function_leaves_its_sample_alone() {
    use metal2vulkan::metal::CompareFunction;
    on_device(|device, _| {
        let pipeline = state(device, "gsm2_cubic_samples");
        let queue = device.new_command_queue().unwrap();
        let mut differ = 0;
        for (seed, format) in (21u64..).zip(FORMATS) {
            let texture = texture(device, format, false, seed);
            for (which, (name, mode, border, pixel)) in (0u32..).zip(STATES) {
                let coords = coordinate_bytes(pixel);
                for compare in [
                    CompareFunction::Less,
                    CompareFunction::Greater,
                    CompareFunction::Always,
                ] {
                    let mut descriptor = SamplerDescriptor::new();
                    descriptor.min_filter = SamplerMinMagFilter::Cubic;
                    descriptor.mag_filter = SamplerMinMagFilter::Cubic;
                    descriptor.s_address_mode = mode;
                    descriptor.t_address_mode = mode;
                    descriptor.r_address_mode = mode;
                    descriptor.border_color = border;
                    descriptor.normalized_coordinates = !pixel;
                    descriptor.compare_function = compare;
                    let sampler = device.new_sampler_state(&descriptor).unwrap();
                    let out = dispatch(
                        &queue,
                        &pipeline,
                        (&texture, &sampler),
                        &coords,
                        Some(which),
                        16,
                    );
                    let lanes = differing(&out, 16).len();
                    println!(
                        "CUBIC-COMPARE {} {format:?} {name} {compare:?}: {} lanes, {lanes} differ \
                         from the constexpr bicubic sample",
                        device.name(),
                        SIDE * SIDE,
                    );
                    differ += lanes;
                }
            }
        }
        assert_eq!(differ, 0);
    });
}

/// A runtime cubic sample at level 2.5 reads the base level, as the constexpr bicubic sampler's
/// implicit one does: every lane of a mipmapped `half` and 8-bit unorm texture, whose other
/// levels hold other texels, is the same bits, for each min/mag pairing.
#[test]
fn a_cubic_sample_reads_the_base_level_whatever_its_level() {
    on_device(|device, _| {
        let pipeline = state(device, "gsm2_cubic_half_samples");
        let queue = device.new_command_queue().unwrap();
        let coords = coordinate_bytes(false);
        let (_, mode, border, pixel) = STATES[0];
        let mut differ = 0;
        for (seed, format) in (11u64..).zip([PixelFormat::RGBA16Float, PixelFormat::RGBA8Unorm]) {
            let texture = texture(device, format, true, seed);
            for filters in FILTERS {
                let sampler = cubic_sampler(
                    device,
                    filters,
                    (mode, border, pixel),
                    SamplerMipFilter::NotMipmapped,
                );
                let out = dispatch(&queue, &pipeline, (&texture, &sampler), &coords, None, 8);
                let lanes = differing(&out, 8);
                println!(
                    "CUBIC-LEVEL {} {format:?} {filters:?}: {} lanes, {} differ from the base \
                     level's constexpr bicubic sample",
                    device.name(),
                    SIDE * SIDE,
                    lanes.len()
                );
                differ += lanes.len();
            }
        }
        assert_eq!(differ, 0);
    });
}

/// A cubic sampler with a mip filter, whose level Metal's answers do not settle, is refused at
/// the dispatch, typed `NotImplemented(SamplerBicubicFilter)`, never sampled some other way.
#[test]
fn a_cubic_sampler_with_a_mip_filter_is_refused_at_the_dispatch() {
    on_device(|device, _| {
        let pipeline = state(device, "gsm2_cubic_samples");
        let queue = device.new_command_queue().unwrap();
        let texture = texture(device, PixelFormat::RGBA8Unorm, true, 21);
        let coords = buffer_with(device, &coordinate_bytes(false)).unwrap();
        let lanes = u64::from(SIDE * SIDE);
        let out = buffer_with(device, &vec![0; (lanes * 32) as usize]).unwrap();
        let (_, mode, border, pixel) = STATES[0];
        for mip in [SamplerMipFilter::Nearest, SamplerMipFilter::Linear] {
            for filters in FILTERS {
                let sampler = cubic_sampler(device, filters, (mode, border, pixel), mip);
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encoder.set_compute_pipeline_state(&pipeline).unwrap();
                encoder.set_texture(Some(&texture), 0).unwrap();
                encoder.set_sampler_state(Some(&sampler), 0).unwrap();
                encoder.set_buffer(Some(&coords), 0, 0).unwrap();
                encoder.set_buffer(Some(&out), 0, 1).unwrap();
                encoder.set_bytes(&0u32.to_le_bytes(), 2).unwrap();
                let refused = match encoder
                    .dispatch_threads(Size::new(lanes, 1, 1), Size::new(GROUP, 1, 1))
                {
                    Err(Error::Compile(CompileError::Unsupported(u)))
                    | Err(Error::Unsupported(u))
                    | Err(Error::Bind(BindError::Unsupported(u))) => {
                        u.feature == Feature::SamplerBicubicFilter
                    }
                    _ => false,
                };
                assert!(
                    refused,
                    "{mip:?} {filters:?}: not refused as a bicubic filter"
                );
                encoder.end_encoding().unwrap();
            }
        }
    });
}

/// One dispatch of `state` (an `argument_linear_samples::kernel` of a heap texture sampled
/// through its argument buffer's sampler) over `jobs`, the argument buffer naming `texture` and
/// `sampler`: the command buffer's status and error, and the `float4` results.
fn argument_dispatch(
    device: &Device,
    state: &ComputePipelineState,
    (texture, sampler): (&Texture, &SamplerState),
    jobs: &[[f32; 4]],
) -> (CommandBufferStatus, Option<CommandBufferError>, Vec<f32>) {
    let lanes = jobs.len() as u64;
    let bytes: Vec<u8> = jobs
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let input = buffer_with(device, &bytes).unwrap();
    let output = buffer_with(device, &vec![0; (lanes * 16) as usize]).unwrap();
    let ids = [texture.gpu_resource_id().0, sampler.gpu_resource_id().0];
    let words: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    let argument = buffer_with(device, &words).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder
        .use_resource(texture.as_resource(), ResourceUsage::READ)
        .unwrap();
    encoder.set_buffer(Some(&argument), 0, 2).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(lanes, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let out = read_buffer(&output, 0, lanes * 16)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (command_buffer.status(), command_buffer.error(), out)
}

/// While a cubic sampler state is live, a kernel that samples a texture through the sampler its
/// argument buffer holds runs. Through a linear sampler it answers every lane as the same texture
/// at a slot through that sampler (within `argument_linear_samples::DEVICE_FILTER` where the
/// device's own sampler filters, `argument_linear_samples::device_filters`), and its command buffer
/// completes; through the cubic
/// state, whose bicubic filter the runtime does not emulate through an argument buffer, its command
/// buffer fails `CubicArgumentSampler`, the reach decided where the function ran.
#[test]
fn an_argument_buffer_reaches_a_cubic_sampler_only_where_it_names_one() {
    use super::argument_linear_samples::{alike, device_filters, kernel};
    use super::depth_compares::{pipeline_of, run_held, Held, Through};
    use metal2vulkan_devtools::llvm_assemble_text;
    on_device_with(at_plan, |device, _| {
        let make = |heap: bool, through: Through| {
            let text = kernel(heap, through, false, false);
            pipeline_of(device, &llvm_assemble_text(&text).unwrap(), "k")
        };
        let heap = make(true, Through::Argument);
        let slot = make(false, Through::Runtime);
        let texture = texture(device, PixelFormat::RGBA8Unorm, false, 632);
        let edge = (
            SamplerAddressMode::ClampToEdge,
            SamplerBorderColor::TransparentBlack,
            false,
        );
        let cubic = cubic_sampler(
            device,
            (SamplerMinMagFilter::Cubic, SamplerMinMagFilter::Cubic),
            edge,
            SamplerMipFilter::NotMipmapped,
        );
        let linear = cubic_sampler(
            device,
            (SamplerMinMagFilter::Linear, SamplerMinMagFilter::Linear),
            edge,
            SamplerMipFilter::NotMipmapped,
        );
        let jobs: Vec<[f32; 4]> = (0..GROUP)
            .map(|i| {
                let t = i as f32 / GROUP as f32;
                [1.2 * t - 0.1, 0.9 - 0.7 * t, 0.0, 0.0]
            })
            .collect();
        let (status, error, got) = argument_dispatch(device, &heap, (&texture, &linear), &jobs);
        assert_eq!((status, error), (CommandBufferStatus::Completed, None));
        let want = run_held(
            device,
            (&texture, Held::Slot, Some(&linear)),
            &slot,
            &jobs,
            1,
        );
        let within = device_filters(device);
        let differ = (0..got.len())
            .filter(|&i| !alike(got[i], want[i], within))
            .count();
        println!(
            "CUBIC-ARGUMENT {}: a linear argument sampler beside a live cubic one: {} lanes, {} \
             differ from the slot's",
            device.name(),
            jobs.len(),
            differ
        );
        assert_eq!(differ, 0);
        let (status, error, _) = argument_dispatch(device, &heap, (&texture, &cubic), &jobs);
        println!(
            "CUBIC-ARGUMENT {}: the cubic argument sampler: {status:?}, {error:?}",
            device.name()
        );
        assert_eq!(
            (status, error),
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::CubicArgumentSampler)
            )
        );
    });
}
