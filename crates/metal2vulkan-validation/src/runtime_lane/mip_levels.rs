//! A device test of the level a sample with an explicit level or a kernel's bias reads against
//! Metal's measured rule on the device the cfg selects, under the validation layer: Metal's own
//! sampler through MoltenVK (`VkProperty::MetalSampler`), which must answer it exactly; elsewhere
//! the device's own level selection, which the stages record (`PrecisionNote::DeviceSampling`) and
//! the lane counts against the rule (`LEVEL-DIFFERS` lines) without failing.
//!
//! Metal rounds the level `x` to `half` (a NaN reads 0), clamps it into the sampler's level clamp
//! and the texture's levels, and with `L = floor(x)` and `q = floor(64 (x - L)) / 64` reads level
//! `L + 1` under a nearest mip filter where `q > 1/2`, else `L`, blends `L` and `L + 1` with
//! weight `q` under a linear one, and magnifies where `L + q` is 0 (Apple M4 on macOS 27.0.1).
//!
//! The kernels are `msf_mip_levels`'s (`validation/fixtures/public/engine`), over `r32Float`
//! textures of five levels whose level `L` holds `L` in every texel (a 2D array's slice 1 holds
//! `100 + L`; a cube's six faces hold `L`), so a sample states the level and weight it read: the
//! 2D kernel's samplers at `level(x)` and at `bias(x)`, and the 3D, cube, and 2D-array kernels' at
//! `level(x)`. A 2D texture whose base level holds `i + 16 j` at texel `(i, j)` (level `L >= 1`
//! holding `1000 L`) shows whether the samplers minifying nearest and magnifying linearly
//! magnified: there the linear filter reads about 120 at texel position (5.3, 7.7), where the
//! nearest reads 117. A linear filter's own sum is the sampler's: its sample may sit [`FILTERED`]
//! from the level and weight (lavapipe's cube filter at a 2x2 face reads `3.0000002` for a level
//! holding 3), which no level error comes near (a weight's step is 1/64).
//!
//! The sampler clamping the level into `[0.3, 2.7]` under a linear mip filter is not checked: its
//! `lod_min_clamp` lies off the 1/64 grid, so the device's `minLod` raises Metal's level and the
//! translation keeps the device's.
//!
//! Each level is one of the probes' 2,230: every sixty-fourth from -1 to 5, the neighbourhoods of
//! `L + 1/2` in steps of `1/4096`, every `1/1024` from 1 to 2, the `half` ties around `L + k/64`
//! with their `f32` neighbours, and past the last level, the infinities and a NaN.

use super::degrade_notices::counts;
use super::pixel_samplers::metal_sampler;
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ComputePipelineState, Degradation, Device, PipelineOptions, Region, ResourceOptions, Size,
    Texture, TextureDescriptor,
};

/// The base level's side, and the levels below it with it.
pub(super) const SIDE: u64 = 16;
pub(super) const LEVELS: u64 = 5;

/// The threads of a threadgroup; each grid is padded to whole threadgroups.
const GROUP: usize = 64;

/// The level clamp of the samplers that clamp, as their `half`s hold it.
const CLAMP: (f32, f32) = (0.300_048_83, 2.699_218_8);

/// No level clamp.
const OPEN: (f32, f32) = (0.0, f32::INFINITY);

/// How far a linear filter's sum may sit from the level and weight it read: two `f32` ulps of the
/// largest level's value, 104 (module docs).
pub(super) const FILTERED: f32 = 1.0 / 65536.0;

/// What the linear filter reads at texel position (5.3, 7.7) of the base level `i + 16 j`, and
/// how far a device's sub-texel precision may move it.
const MAGNIFIED: (f32, f32) = (120.0, 1.5);

/// `x` rounded to `half`, to nearest even, as an `f32` (past `65504` to an infinity).
pub(super) fn half(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let a = f64::from(x).abs();
    let ulp = if a < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(a.log2().floor() as i32 - 10)
    };
    let r = (a / ulp).round_ties_even() * ulp;
    let r = if r > 65504.0 { f64::INFINITY } else { r };
    r.copysign(f64::from(x)) as f32
}

/// Metal's `(L, q)` for level `x` through a sampler clamping it into `clamp`.
fn metal(x: f32, (lo, hi): (f32, f32)) -> (f32, f32) {
    let x = if x.is_nan() { 0.0 } else { half(x) };
    let x = x.max(lo).min(hi).clamp(0.0, (LEVELS - 1) as f32);
    let l = x.floor();
    (l, ((x - l) * 64.0).floor() / 64.0)
}

/// The level a nearest mip filter reads at `x`.
fn nearest(x: f32, clamp: (f32, f32)) -> f32 {
    let (l, q) = metal(x, clamp);
    if q > 0.5 {
        l + 1.0
    } else {
        l
    }
}

/// The level and weight a linear mip filter reads at `x`, as the sample of a texture whose
/// level `L` holds `L`.
fn linear(x: f32, clamp: (f32, f32)) -> f32 {
    let (l, q) = metal(x, clamp);
    l + q
}

/// Whether a sample at `x` magnifies.
fn magnifies(x: f32) -> bool {
    let (l, q) = metal(x, OPEN);
    l + q == 0.0
}

/// The probes' levels.
pub(super) fn levels() -> Vec<f32> {
    let mut out: Vec<f32> = (0..=6 * 64).map(|m| -1.0 + m as f32 / 64.0).collect();
    for l in 0..4 {
        out.extend((-48..=48).map(|m| l as f32 + 0.5 + m as f32 / 4096.0));
    }
    out.extend((0..1024).map(|m| 1.0 + m as f32 / 1024.0));
    for l in 0..4 {
        for k in [0, 1, 31, 32, 33, 34, 63] {
            for d in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0] {
                let x = (f64::from(l) + f64::from(k) / 64.0 + d / 4096.0) as f32;
                if x > 0.0 {
                    out.extend([x, x.next_up(), x.next_down()]);
                } else {
                    out.push(x);
                }
            }
        }
    }
    out.extend([
        -0.0,
        4.0,
        4.25,
        4.5,
        4.9,
        5.0,
        7.5,
        100.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(1),
        f32::from_bits(0x3eff_ffff),
        f32::from_bits(0x3f00_0001),
        65504.0,
        65520.0,
    ]);
    out
}

/// A texture of `texture_type` of five levels whose level `L`, of slice `s`, holds `texel(L, s,
/// i, j)` at texel `(i, j)` of each of its layers.
pub(super) fn texture(
    device: &Device,
    texture_type: TextureType,
    texel: impl Fn(u64, u64, u64, u64) -> f32,
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, SIDE, SIDE, true);
    descriptor.texture_type = texture_type;
    let slices = match texture_type {
        TextureType::Type3D => {
            descriptor.depth = SIDE;
            1
        }
        TextureType::Cube => 6,
        TextureType::Type2DArray => {
            descriptor.array_length = 2;
            2
        }
        _ => 1,
    };
    assert_eq!(descriptor.mipmap_level_count, LEVELS);
    let texture = device.new_texture(&descriptor).unwrap();
    for level in 0..LEVELS {
        let n = SIDE >> level;
        let depth = if texture_type == TextureType::Type3D {
            n
        } else {
            1
        };
        for slice in 0..slices {
            let bytes: Vec<u8> = (0..depth)
                .flat_map(|_| (0..n).flat_map(move |j| (0..n).map(move |i| (i, j))))
                .flat_map(|(i, j)| texel(level, slice, i, j).to_le_bytes())
                .collect();
            texture
                .replace_region(
                    Region::new_3d(0, 0, 0, n, n, depth),
                    level,
                    slice,
                    &bytes,
                    4 * n,
                    4 * n * n,
                )
                .unwrap();
        }
    }
    texture
}

/// The kernel `name` of the fixture's library on `device`.
fn pipeline(device: &Device, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("msf_mip_levels.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// The floats kernel `name` writes reading `texture` at each of `levels`, `per` `float4`s per
/// level.
fn run(device: &Device, texture: &Texture, name: &str, levels: &[f32], per: usize) -> Vec<f32> {
    let state = pipeline(device, name);
    let padded = levels.len().div_ceil(GROUP) * GROUP;
    let mut bytes: Vec<u8> = levels.iter().flat_map(|v| v.to_le_bytes()).collect();
    bytes.resize(padded * 4, 0);
    let input = buffer_with(device, &bytes).unwrap();
    let length = (padded * per * 16) as u64;
    let output = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    encoder
        .dispatch_threads(
            Size::new(padded as u64, 1, 1),
            Size::new(GROUP as u64, 1, 1),
        )
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
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The four samples through (nearest, nearest mip), (nearest, linear mip), (linear, nearest
/// mip), and (linear, linear mip) at `x`, plus `base`.
fn four(x: f32, base: f32) -> [f32; 4] {
    let (n, l) = (nearest(x, OPEN), linear(x, OPEN));
    [n + base, l + base, n + base, l + base]
}

/// The samplers of the four, in the kernels' order.
const FOUR: [&str; 4] = [
    "nearest, nearest mip",
    "nearest, linear mip",
    "linear, nearest mip",
    "linear, linear mip",
];

/// The mismatches of `got` against `want` for the samplers `names` at level `x`: exact for a
/// nearest filter, within [`FILTERED`] for a linear one (a name from `linear`).
fn check(kernel: &str, x: f32, names: &[&str], got: &[f32], want: &[f32], wrong: &mut Vec<String>) {
    for ((name, g), w) in names.iter().zip(got).zip(want) {
        let differs = match name.starts_with("linear") {
            true => (g - w).abs() > FILTERED,
            false => g.to_bits() != w.to_bits(),
        };
        if differs {
            wrong.push(format!(
                "{kernel} {name} at {x:e} ({:#010x}): {g:e}, want {w:e}",
                x.to_bits()
            ));
        }
    }
}

/// On a device that samples through Metal's own sampler, every sample with an explicit level, or a
/// kernel's bias, reads Metal's levels with Metal's weights: through nearest and linear filters
/// under nearest and linear mip filters of 2D, 3D, cube, and 2D-array textures, through a nearest
/// mip filter clamping the level, and where a sampler minifies nearest and magnifies linearly.
/// Elsewhere the levels are the device's: the stages announce `device-sampling`, and the samples
/// that differ from Metal's rule are printed.
#[test]
fn explicit_levels_read_metals_levels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let xs = levels();
        let mut wrong = Vec::new();
        // A cube's faces hold the same texels, which its filter may read across a face's edge.
        let uniform = |t, slice: u64| texture(&device, t, move |l, s, _, _| (l + slice * s) as f32);
        let flat = uniform(TextureType::Type2D, 0);

        let out = run(&device, &flat, "msf_lvl2d", &xs, 2);
        for (k, &x) in xs.iter().enumerate() {
            let o = &out[8 * k..8 * k + 8];
            check("msf_lvl2d", x, &FOUR, &o[..4], &four(x, 0.0), &mut wrong);
            let (n, l) = (nearest(x, OPEN), linear(x, OPEN));
            check(
                "msf_lvl2d",
                x,
                &["clamped nearest, nearest mip", "mag_n", "mag_l"],
                &o[5..],
                &[nearest(x, CLAMP), n, l],
                &mut wrong,
            );
        }
        let out = run(&device, &flat, "msf_bias2d", &xs, 1);
        for (k, &x) in xs.iter().enumerate() {
            check(
                "msf_bias2d",
                x,
                &FOUR,
                &out[4 * k..4 * k + 4],
                &four(x, 0.0),
                &mut wrong,
            );
        }
        for (kernel, t, slice) in [
            ("msf_lvl3d", TextureType::Type3D, 0),
            ("msf_lvlcube", TextureType::Cube, 0),
            ("msf_lvlarray", TextureType::Type2DArray, 100),
        ] {
            let base = slice as f32;
            let out = run(&device, &uniform(t, slice), kernel, &xs, 1);
            for (k, &x) in xs.iter().enumerate() {
                check(
                    kernel,
                    x,
                    &FOUR,
                    &out[4 * k..4 * k + 4],
                    &four(x, base),
                    &mut wrong,
                );
            }
        }

        // The base level `i + 16 j`, level `L >= 1` holding `1000 L`.
        let value = |l: f32| if l == 0.0 { 117.0 } else { 1000.0 * l };
        let distinct = texture(&device, TextureType::Type2D, |l, _, i, j| match l {
            0 => (i + 16 * j) as f32,
            l => 1000.0 * l as f32,
        });
        let out = run(&device, &distinct, "msf_lvl2d", &xs, 2);
        for (k, &x) in xs.iter().enumerate() {
            let o = &out[8 * k + 6..8 * k + 8];
            if magnifies(x) {
                let (centre, within) = MAGNIFIED;
                for (s, g) in ["mag_n", "mag_l"].iter().zip(o) {
                    if (g - centre).abs() > within {
                        wrong.push(format!(
                            "msf_lvl2d {s} at {x:e} ({:#010x}): {g:e}, want a magnified \
                             {centre:e}",
                            x.to_bits()
                        ));
                    }
                }
                continue;
            }
            let (l, q) = metal(x, OPEN);
            let blend = value(l) * (1.0 - q) + value(l + 1.0) * q;
            check(
                "msf_lvl2d",
                x,
                &["mag_n minified", "mag_l minified"],
                o,
                &[value(nearest(x, OPEN)), blend],
                &mut wrong,
            );
        }
        eprintln!("{name}: {} of the samples differ", wrong.len());
        let own = !metal_sampler(&device);
        let announced = counts(&messages.take_notices());
        assert_eq!(
            announced.contains_key(&Degradation::DeviceSampling),
            own,
            "{name}: device-sampling exactly where the device's sampler is not Metal's"
        );
        if own {
            for w in wrong.iter().take(60) {
                eprintln!("LEVEL-DIFFERS {name}: {w}");
            }
        } else {
            assert!(
                wrong.is_empty(),
                "{name}: {} samples differ:\n{}",
                wrong.len(),
                wrong.join("\n")
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
