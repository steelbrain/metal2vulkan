//! Metal's anisotropic filter against the answers of an Apple M4, on the device the cfg selects,
//! under the validation layer.
//!
//! The inputs are the own synthetic set A (its probe rounds 8 and 9, `probes/ani/gen8.py` and
//! `gen9.py` at probe refs 9efc27f1b and b09ea437a, and round 10, `gen10.py` at 67558f616): 1268
//! interior footprints of a 64x64 texture, of axis ratios 1 to 32 at six angles ([`GEOMETRY`]: per
//! footprint the normalized coordinate and both gradients), sampled through a runtime sampler of
//! linear or nearest filters, no, nearest or linear mip filter, `maxAnisotropy` 2, 3, 8 or 16 and
//! clamp-to-zero or clamp-to-edge addressing, from a random `rgba8Unorm` texture of seven levels
//! ([`texture_bytes`], seed `0x5108`). The answers ([`METAL`]) are the M4's (macOS 27.0.1; round 9
//! repeated round 8's nearest sampler bit for bit, and round 10 its linear one under a nearest mip
//! filter at 8; the footprints are interior, so clamp to edge answered as clamp to zero, bit for
//! bit), each channel stored as `round(v · 255 · 256)`, within `0.002` of a 256th of a code of the
//! `float` Metal wrote.
//!
//! The samplers run with `samplerAnisotropy` withheld: every Vulkan sampler is then isotropic and
//! every sample takes the product's emulation of Metal's filter (the module
//! `lower::sampler::anisotropic`) on every device, MoltenVK's too, and the device announces
//! `anisotropic-filter`. Each sampler's answers stay within its bound in codes ([`Bound`]): for
//! linear filters the footprint in the device's arithmetic (its logarithm, powers and level
//! blend), the device's own bilinear filter (under clamp to edge, lavapipe's 8-bit path); for
//! nearest ones a tap a texel off Metal's where it sits within the spacing's residue of a texel
//! edge. The test prints each
//! sampler's largest and mean difference in codes and the share of channels a code or more off.
//! (With `samplerAnisotropy`, the device's own anisotropic filter answers, which no bound here
//! measures.)

use super::degrade_notices::counts;
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter, SamplerMipFilter};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ComputePipelineState, Degradation, Device, PipelineOptions, Region, ResourceOptions,
    SamplerState, Size, Texture, TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// The footprints: per footprint two `float4`s, `(u, v, 0, 0)` and `(dx.x, dx.y, dy.x, dy.y)`,
/// normalized.
const GEOMETRY: &[u8] = include_bytes!("../../fixtures/public/anisotropic/geometry.bin");

/// The M4's answers, [`SAMPLERS`] in order, each footprint's four channels as `u16`s of
/// `round(v · 65280)`.
const METAL: &[u8] = include_bytes!("../../fixtures/public/anisotropic/metal_m4.bin");

/// SHA-256 of the probe's geometry, texture and the packed answers.
const GEOMETRY_SHA: &str = "670ff56cdf6392a2441a03e01d52d1e7501d2d5f9267c2f765abaded9627b1b2";
const TEXTURE_SHA: &str = "e84ababf5d020816ed6b9e2c827ca356070ad337176036ab8321ca34675e9a1a";
const METAL_SHA: &str = "aff7c6eecefbae4cb56b40f06c0e29aee5d9f4329e5b8a8f94b872f81407b153";

/// The footprints [`GEOMETRY`] holds.
const FOOTPRINTS: usize = 1268;

/// The texture's side and level count.
const SIDE: u64 = 64;
const LEVELS: u64 = 7;

/// The most a sampler's answers may differ from the M4's, in codes (`1/255`).
#[derive(Clone, Copy, Debug)]
enum Bound {
    /// Every channel within `max`, the mean within `mean`.
    Linear { max: f64, mean: f64 },
    /// The mean within `mean`, and at most `share` of the channels a code or more off.
    Nearest { mean: f64, share: f64 },
}

/// The bound of the linear filters: the footprint in the device's arithmetic (its `Log2` and
/// `Exp2`, the spacing and level unrounded, a linear mip filter's levels the device's blend), whose
/// tap count at a threshold and tap positions differ from Metal's by its logarithm's and rounding's
/// residue. On the M3 Max (MoltenVK, `samplerAnisotropy` withheld) at `maxAnisotropy` 3 to 16:
/// 10.375 codes at most (one footprint), the mean 0.21 at most.
const LINEAR: Bound = Bound::Linear {
    max: 12.0,
    mean: 0.25,
};

/// The bound of the nearest filters: a tap a texel off Metal's where it sits within the
/// spacing's residue of a texel edge, the spacing and the level in the device's arithmetic. On the
/// M3 Max at `maxAnisotropy` 2 to 16: the mean 0.82 codes at most, at most 338 of 5,072 channels
/// (6.7%) a code or more off.
const NEAREST: Bound = Bound::Nearest {
    mean: 1.0,
    share: 0.08,
};

/// The bound of the linear filters under clamp to edge: lavapipe filters `rgba8Unorm` under a
/// wrap mode it takes as simple (clamp to edge, repeat) by its own 8-bit path, whose bilinear
/// answers at these interior footprints differ from its clamp-to-zero ones by up to 1.62 codes
/// (mean 0.34) without anisotropy, beside the linear filters' residue ([`LINEAR`]); the M3 Max
/// was 6.531 codes off the M4 at most (mean 0.071) at `maxAnisotropy` 8 and 16.
const LINEAR_EDGE: Bound = Bound::Linear {
    max: 12.0,
    mean: 0.3,
};

/// The samplers, in [`METAL`]'s order: (name, min/mag filter, mip filter, `maxAnisotropy`,
/// address mode, bound). The last two are the consumer's anisotropic sampler (linear filters, a
/// nearest mip filter, clamp to edge: word `0x20000E`) at `maxAnisotropy` 8, the degree its
/// guests set, and 16.
const SAMPLERS: [(
    &str,
    SamplerMinMagFilter,
    SamplerMipFilter,
    u32,
    SamplerAddressMode,
    Bound,
); 15] = [
    (
        "linear, linear mip, 8",
        SamplerMinMagFilter::Linear,
        SamplerMipFilter::Linear,
        8,
        SamplerAddressMode::ClampToZero,
        LINEAR,
    ),
    (
        "linear, nearest mip, 8",
        SamplerMinMagFilter::Linear,
        SamplerMipFilter::Nearest,
        8,
        SamplerAddressMode::ClampToZero,
        LINEAR,
    ),
    (
        "linear, no mip, 8",
        SamplerMinMagFilter::Linear,
        SamplerMipFilter::NotMipmapped,
        8,
        SamplerAddressMode::ClampToZero,
        LINEAR,
    ),
    (
        "nearest, nearest mip, 8",
        SamplerMinMagFilter::Nearest,
        SamplerMipFilter::Nearest,
        8,
        SamplerAddressMode::ClampToZero,
        NEAREST,
    ),
    (
        "linear, linear mip, 16",
        SamplerMinMagFilter::Linear,
        SamplerMipFilter::Linear,
        16,
        SamplerAddressMode::ClampToZero,
        LINEAR,
    ),
    (
        "linear, linear mip, 3",
        SamplerMinMagFilter::Linear,
        SamplerMipFilter::Linear,
        3,
        SamplerAddressMode::ClampToZero,
        LINEAR,
    ),
    (
        "nearest, linear mip, 8",
        SamplerMinMagFilter::Nearest,
        SamplerMipFilter::Linear,
        8,
        SamplerAddressMode::ClampToZero,
        NEAREST,
    ),
    (
        "nearest, no mip, 8",
        SamplerMinMagFilter::Nearest,
        SamplerMipFilter::NotMipmapped,
        8,
        SamplerAddressMode::ClampToZero,
        NEAREST,
    ),
    (
        "nearest, nearest mip, 16",
        SamplerMinMagFilter::Nearest,
        SamplerMipFilter::Nearest,
        16,
        SamplerAddressMode::ClampToZero,
        NEAREST,
    ),
    (
        "nearest, nearest mip, 3",
        SamplerMinMagFilter::Nearest,
        SamplerMipFilter::Nearest,
        3,
        SamplerAddressMode::ClampToZero,
        NEAREST,
    ),
    (
        "nearest, nearest mip, 2",
        SamplerMinMagFilter::Nearest,
        SamplerMipFilter::Nearest,
        2,
        SamplerAddressMode::ClampToZero,
        NEAREST,
    ),
    (
        "nearest, linear mip, 16",
        SamplerMinMagFilter::Nearest,
        SamplerMipFilter::Linear,
        16,
        SamplerAddressMode::ClampToZero,
        NEAREST,
    ),
    (
        "linear, nearest mip, 16",
        SamplerMinMagFilter::Linear,
        SamplerMipFilter::Nearest,
        16,
        SamplerAddressMode::ClampToZero,
        LINEAR,
    ),
    (
        "linear, nearest mip, 8, clamp to edge",
        SamplerMinMagFilter::Linear,
        SamplerMipFilter::Nearest,
        8,
        SamplerAddressMode::ClampToEdge,
        LINEAR_EDGE,
    ),
    (
        "linear, nearest mip, 16, clamp to edge",
        SamplerMinMagFilter::Linear,
        SamplerMipFilter::Nearest,
        16,
        SamplerAddressMode::ClampToEdge,
        LINEAR_EDGE,
    ),
];

/// The kernel: thread `g` samples `t` through runtime sampler slot 0 at footprint `g` of buffer
/// 0 with its explicit gradients, and writes the texel to buffer 1.
const KERNEL: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @anisotropic_samples(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %cfg, ptr addrspace(1) %o, i32 %gid) {
  %i = shl i32 %gid, 1
  %pa = getelementptr <4 x float>, ptr addrspace(1) %cfg, i32 %i
  %a = load <4 x float>, ptr addrspace(1) %pa, align 16
  %j = or i32 %i, 1
  %pb = getelementptr <4 x float>, ptr addrspace(1) %cfg, i32 %j
  %b = load <4 x float>, ptr addrspace(1) %pb, align 16
  %uv = shufflevector <4 x float> %a, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %dx = shufflevector <4 x float> %b, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %dy = shufflevector <4 x float> %b, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %r = call { <4 x float>, i8 } @air.sample_texture_2d_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, <2 x float> %dx, <2 x float> %dy, float 0.000000e+00, i1 true, <2 x i32> zeroinitializer, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %p = getelementptr <4 x float>, ptr addrspace(1) %o, i32 %gid
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, <2 x float>, <2 x float>, float, i1, <2 x i32>, i32)

!air.kernel = !{!0}
!0 = !{ptr @anisotropic_samples, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"cfg"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"o"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// The probe's random `rgba8Unorm` texels, level 0 first: each byte the next of a 64-bit LCG
/// (seed `0x5108`, Knuth's multiplier and increment, its top 53 bits) modulo 256.
fn texture_bytes() -> Vec<u8> {
    let mut state: u64 = 0x5108;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) % 256) as u8
    };
    (0..LEVELS)
        .flat_map(|l| {
            let side = (SIDE >> l).max(1);
            (0..4 * side * side).map(|_| next()).collect::<Vec<_>>()
        })
        .collect()
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The texture of [`texture_bytes`], every level uploaded.
fn texture(device: &Device, bytes: &[u8]) -> Texture {
    let descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, true);
    assert_eq!(descriptor.mipmap_level_count, LEVELS);
    let texture = device.new_texture(&descriptor).unwrap();
    let mut at = 0;
    for level in 0..LEVELS {
        let side = (SIDE >> level).max(1);
        let length = (4 * side * side) as usize;
        texture
            .replace_region(
                Region::new_2d(0, 0, side, side),
                level,
                0,
                &bytes[at..at + length],
                4 * side,
                4 * side * side,
            )
            .unwrap();
        at += length;
    }
    assert_eq!(at, bytes.len());
    texture
}

/// A sampler of `filter` both ways, `mip`, `anisotropy`, and `address` on every axis.
fn sampler(
    device: &Device,
    filter: SamplerMinMagFilter,
    mip: SamplerMipFilter,
    anisotropy: u32,
    address: SamplerAddressMode,
) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.mip_filter = mip;
    descriptor.max_anisotropy = anisotropy;
    descriptor.s_address_mode = address;
    descriptor.t_address_mode = address;
    descriptor.r_address_mode = address;
    device.new_sampler_state(&descriptor).unwrap()
}

fn pipeline(device: &Device) -> ComputePipelineState {
    let bitcode = llvm_assemble_text(KERNEL).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library
        .new_function_with_name("anisotropic_samples")
        .unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// Each footprint's texel through `sampler`, as `f32` channels.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<f32> {
    let geometry = buffer_with(device, GEOMETRY).unwrap();
    let length = (FOOTPRINTS * 16) as u64;
    let output = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(sampler), 0).unwrap();
    encoder.set_buffer(Some(&geometry), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    // 1268 is 4 x 317: whole threadgroups of 4.
    encoder
        .dispatch_threads(Size::new(FOOTPRINTS as u64, 1, 1), Size::new(4, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&output, 0, length)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The M4's answers of sampler `n` of [`SAMPLERS`], in codes.
fn metal_codes(n: usize) -> Vec<f64> {
    let per = FOOTPRINTS * 4 * 2;
    METAL[n * per..(n + 1) * per]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f64::from(u16::from_le_bytes([c[0], c[1]])) / 256.0)
        .collect()
}

/// One sampler's differences from the M4, in codes.
struct Differences {
    largest: f64,
    mean: f64,
    /// The channels a code or more off.
    off: usize,
    /// The footprint and channel of the largest.
    at: (usize, usize),
}

fn differences(got: &[f32], want: &[f64]) -> Differences {
    assert_eq!(got.len(), want.len());
    let mut d = Differences {
        largest: 0.0,
        mean: 0.0,
        off: 0,
        at: (0, 0),
    };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let x = (f64::from(g) * 255.0 - w).abs();
        // A NaN is as far off as can be.
        let x = if x.is_nan() { f64::INFINITY } else { x };
        if x > d.largest {
            d.largest = x;
            d.at = (i / 4, i % 4);
        }
        d.mean += x;
        d.off += usize::from(x >= 1.0);
    }
    d.mean /= want.len() as f64;
    d
}

/// The inputs are the probe's: the geometry, the regenerated texture and the packed answers hash
/// to what our probes sampled and wrote.
#[test]
fn the_inputs_are_the_probes() {
    assert_eq!(GEOMETRY.len(), FOOTPRINTS * 32);
    assert_eq!(METAL.len(), SAMPLERS.len() * FOOTPRINTS * 4 * 2);
    assert_eq!(sha(GEOMETRY), GEOMETRY_SHA);
    assert_eq!(sha(&texture_bytes()), TEXTURE_SHA);
    assert_eq!(sha(METAL), METAL_SHA);
}

/// With `samplerAnisotropy` withheld every sampler state is made, its Vulkan sampler isotropic,
/// every sample takes the emulation on every device (`anisotropic-filter` announced, through
/// Metal's own sampler too), and each sampler's answers stay within its bound of the M4's.
#[test]
fn without_sampler_anisotropy_the_emulation_answers_within_its_bounds() {
    use metal2vulkan::device::VkFeature;
    within_bounds("samplerAnisotropy withheld", |options| {
        options
            .withheld_features
            .insert(VkFeature::SamplerAnisotropy);
    });
}

/// Every sampler's answers on the test device made with `adjust` beside the lane's options stay
/// within its bound of the M4's, and the device announces `anisotropic-filter`: the emulation ran;
/// never `device-anisotropic-filter`. Prints each sampler's differences under `setting`.
fn within_bounds(setting: &str, adjust: impl FnOnce(&mut metal2vulkan_runtime::DeviceOptions)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (name, outcomes) = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        let name = format!("{} {setting}", device.name());
        let texture = texture(&device, &texture_bytes());
        let state = pipeline(&device);
        let outcomes: Vec<(&str, Bound, Differences)> = SAMPLERS
            .iter()
            .enumerate()
            .map(|(n, &(what, filter, mip, anisotropy, address, bound))| {
                let s = sampler(&device, filter, mip, anisotropy, address);
                let got = run(&device, &state, &texture, &s);
                (what, bound, differences(&got, &metal_codes(n)))
            })
            .collect();
        (name, outcomes)
    };
    layer_reported_nothing_at_all(&messages);
    let announced: BTreeSet<Degradation> = counts(&messages.take_notices()).into_keys().collect();
    println!("ANISOTROPIC {name}: announced {announced:?}");
    let mut wrong = Vec::new();
    for (what, bound, d) in &outcomes {
        let share = d.off as f64 / (FOOTPRINTS * 4) as f64;
        println!(
            "ANISOTROPIC {name}: {what}: largest {:.3} codes (footprint {}, channel {}), mean \
             {:.4}, {} of {} channels a code or more off",
            d.largest,
            d.at.0,
            d.at.1,
            d.mean,
            d.off,
            FOOTPRINTS * 4
        );
        let within = match *bound {
            Bound::Linear { max, mean } => d.largest <= max && d.mean <= mean,
            Bound::Nearest { mean, share: most } => d.mean <= mean && share <= most,
        };
        if !within {
            wrong.push(format!("{what}: past {bound:?}"));
        }
    }
    assert!(
        announced.contains(&Degradation::AnisotropicFilter),
        "{name}: anisotropic-filter announced"
    );
    assert!(
        !announced.contains(&Degradation::DeviceAnisotropicFilter),
        "{name}: every sample is within the emulation's reach"
    );
    assert!(wrong.is_empty(), "{name}: {wrong:#?}");
}
