//! Samples at an explicit level, or a kernel's bias, through a runtime sampler on the device the
//! cfg selects, under the validation layer: the level and weight they read, and a nearest sample's
//! texel at the level it reads. Metal's own sampler through MoltenVK (`VkProperty::MetalSampler`)
//! must answer Metal's rule exactly; elsewhere the level and the texel are the device's sampler's,
//! which the stages record (`PrecisionNote::DeviceSampling`), and the lane prints what differs from
//! the rule (`LEVELS-DIFFERS`, `LEVELED-DIFFERS` lines) without failing.
//!
//! Metal's level rule holds for a runtime sampler with its level clamp as `f32`s, where a constexpr
//! sampler's are `half`s (Apple M4 on macOS 27.0.1, all 37,080 lanes): Metal
//! rounds the level `x` to `half` (a NaN reads 0), clamps it into the sampler's clamp and the
//! texture's levels, and with `L = floor(x)` and `q = floor(64 (x - L)) / 64` reads level `L + 1`
//! under a nearest mip filter where `q > 1/2`, else `L`, and blends `L` and `L + 1` with weight `q`
//! under a linear one. The function does not know a runtime sampler's clamp; the runtime's
//! `VkSampler` holds it on Metal's grid and the function hands the device the unclamped level on
//! that grid by the sampler's word (SEMANTICS 1095). Before that the device read its own level of
//! the clamped `x`, which differs wherever `q` lies at or near a half under a nearest mip filter,
//! off the grid under a linear one, or at a clamp bound off the grid.
//!
//! The kernels are `unv_runtime_levels`'s (`validation/fixtures/public/engine`):
//!
//! - `unv_rlvl2d` samples an `r32Float` 2D texture of five levels whose level `L` holds `L`, at
//!   its centre at `level(x)` and `bias(x)`, through the runtime sampler at slot 0 and through an
//!   argument buffer's sampler beside the same texels, at each level `x` `mip_levels` reads
//!   (2,230 of them), through each of the eighteen clamps the probe read under nearest and linear
//!   filters and mip filters, and without a mip filter (which reads level 0). A nearest filter's
//!   sample is the level exactly; a linear filter's may sit [`FILTERED`] from it.
//! - `unv_rpos2d` samples a 13x11 `r32Float` texture of four levels (13x11, 6x5, 3x2, 1x1) whose
//!   texel `(i, j)` of level `L` holds `i + 16 j + 1000 L` through a nearest sampler that repeats,
//!   then one that mirrors, both axes, at thirteen levels, without a mip filter, with a nearest
//!   one, and with a nearest one clamping the level into `[0.6, 2.4]` (levels 1 to 2): at each
//!   texel edge of each level's width shifted by -3 to 3 periods and 1 to 3 ulps either side, on
//!   a row Metal's position and the device's agree on. Metal reads texel `floor(X)` of the level
//!   it reads at its position `X` (`nearest_positions`); a device that multiplies first reads the
//!   next texel on some of these lanes (the deciding ones).

use super::mip_levels::{half, levels, texture, FILTERED, LEVELS};
use super::nearest_positions::{metal_texel, multiplied_first};
use super::pixel_samplers::own_sampler;
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    PixelFormat, ResourceUsage, SamplerAddressMode, SamplerMinMagFilter, SamplerMipFilter,
    TextureType,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions, SamplerState,
    SamplerStateOptions, Size, Texture, TextureDescriptor,
};

/// The threads of a threadgroup; each grid is padded to whole threadgroups.
const GROUP: usize = 64;

/// The positions texture's base level: neither axis a power of two.
const WIDTH: u32 = 13;
const HEIGHT: u32 = 11;

/// The positions texture's levels: 13x11, 6x5, 3x2, 1x1.
const POSITION_LEVELS: u32 = 4;

/// The `v` of every positions lane: on level `L` of height `H`, `0.3 H` lies inside a texel.
const ROW: f32 = 0.3;

/// The level clamps our probe read through runtime samplers, as `f32`s: no clamp,
/// clamps whose bounds lie on and off Metal's sixty-fourths and at and just below a nearest mip
/// filter's half-way points, an empty-width clamp, and clamps past the texture's levels.
fn clamps() -> [(f32, f32); 18] {
    let below = |x: f32| x.next_down();
    [
        (0.0, f32::MAX),
        (0.75, 2.5),
        (0.3, 2.7),
        (0.5, 9.0),
        (below(33.0 / 64.0), 9.0),
        (33.0 / 64.0, 9.0),
        (0.507_812_5, 9.0),
        (below(21.0 / 64.0), 9.0),
        (1.5, 9.0),
        (1.0 + below(33.0 / 64.0), 9.0),
        (0.0, 1.5),
        (0.0, 1.0 + below(33.0 / 64.0)),
        (0.0, 1.0 + 33.0 / 64.0),
        (0.0, 2.4999),
        (0.0, 1.0 + below(21.0 / 64.0)),
        (2.2, 2.2),
        (6.0, 9.0),
        (0.0001, 9.0),
    ]
}

/// Metal's `(L, q)` for level `x` through a runtime sampler clamping it into `[lo, hi]` (`f32`s)
/// in a texture of `levels` levels.
fn metal(x: f32, (lo, hi): (f32, f32), levels: u32) -> (f32, f32) {
    let x = if x.is_nan() { 0.0 } else { half(x) };
    let x = x.max(lo).min(hi).max(0.0).min((levels - 1) as f32);
    let l = x.floor();
    (l, ((x - l) * 64.0).floor() / 64.0)
}

/// The level Metal's nearest mip filter reads at `x`.
fn nearest(x: f32, clamp: (f32, f32), levels: u32) -> f32 {
    let (l, q) = metal(x, clamp, levels);
    if q > 0.5 {
        l + 1.0
    } else {
        l
    }
}

/// The kernel `name` of the fixture's library on `device`.
fn pipeline(device: &Device, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("unv_runtime_levels.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A sampler state of `filter` both ways, `mip`, `clamp`, and `address` on every axis, which an
/// argument buffer may hold.
fn sampler(
    device: &Device,
    filter: SamplerMinMagFilter,
    mip: SamplerMipFilter,
    (lo, hi): (f32, f32),
    address: SamplerAddressMode,
) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.mip_filter = mip;
    descriptor.lod_min_clamp = lo;
    descriptor.lod_max_clamp = hi;
    descriptor.s_address_mode = address;
    descriptor.t_address_mode = address;
    descriptor.r_address_mode = address;
    let mut options = SamplerStateOptions::default();
    options.support_argument_buffers = true;
    device
        .new_sampler_state_with_options(&descriptor, options)
        .unwrap()
}

/// The floats `state` writes for each of `inputs` (`stride` bytes each), `per` floats per lane,
/// `texture` and `sampler` at slot 0, and with `argument` an argument buffer holding both at
/// buffer 0 (the inputs then at buffer 1).
fn run(
    device: &Device,
    state: &ComputePipelineState,
    (texture, sampler): (&Texture, &SamplerState),
    argument: bool,
    (inputs, stride): (&[u8], usize),
    per: usize,
) -> Vec<f32> {
    let lanes = inputs.len() / stride;
    let padded = lanes.div_ceil(GROUP) * GROUP;
    let mut bytes = inputs.to_vec();
    bytes.resize(padded * stride, 0);
    let input = buffer_with(device, &bytes).unwrap();
    let length = (padded * per * 4) as u64;
    let output = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(sampler), 0).unwrap();
    let ids: Vec<u8> = [texture.gpu_resource_id().0, sampler.gpu_resource_id().0]
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect();
    let material = argument.then(|| buffer_with(device, &ids).unwrap());
    let first = if let Some(material) = &material {
        encoder.set_buffer(Some(material), 0, 0).unwrap();
        encoder
            .use_resource(texture.as_resource(), ResourceUsage::READ)
            .unwrap();
        1
    } else {
        0
    };
    encoder.set_buffer(Some(&input), 0, first).unwrap();
    encoder.set_buffer(Some(&output), 0, first + 1).unwrap();
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
    let mut out: Vec<f32> = read_buffer(&output, 0, length)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    out.truncate(lanes * per);
    out
}

/// On a device that samples through Metal's own sampler, every sample at a level, or a kernel's
/// bias, through a runtime sampler or an argument buffer's reads Metal's level and weight: through
/// each clamp under nearest and linear filters and mip filters, and level 0 without a mip filter.
/// Elsewhere the device's sampler reads its own, announced `device-sampling`. Prints the samples
/// checked and those that differ.
#[test]
fn runtime_samplers_levels_read_metals_levels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let state = pipeline(&device, "unv_rlvl2d");
        let flat = texture(&device, TextureType::Type2D, |l, _, _, _| l as f32);
        let xs = levels();
        let inputs: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        let levels = LEVELS as u32;
        let open = clamps()[0];
        let mut wrong = Vec::new();
        let mut checked = 0usize;
        let mut configs = Vec::new();
        for clamp in clamps() {
            for mip in [SamplerMipFilter::Nearest, SamplerMipFilter::Linear] {
                configs.push((clamp, mip));
            }
        }
        configs.push((open, SamplerMipFilter::NotMipmapped));
        for (clamp, mip) in configs {
            for filter in [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear] {
                let s = sampler(&device, filter, mip, clamp, SamplerAddressMode::ClampToEdge);
                let out = run(&device, &state, (&flat, &s), true, (&inputs, 4), 4);
                for (k, &x) in xs.iter().enumerate() {
                    let want = if mip == SamplerMipFilter::Nearest {
                        nearest(x, clamp, levels)
                    } else if mip == SamplerMipFilter::Linear {
                        let (l, q) = metal(x, clamp, levels);
                        l + q
                    } else {
                        0.0
                    };
                    let reads = ["slot level", "slot bias", "argument level", "argument bias"];
                    for (read, &got) in reads.iter().zip(&out[4 * k..4 * k + 4]) {
                        let differs = if filter == SamplerMinMagFilter::Linear {
                            (got - want).abs() > FILTERED
                        } else {
                            got.to_bits() != want.to_bits()
                        };
                        if differs {
                            wrong.push(format!(
                                "{read} {filter:?} {mip:?} [{:e}, {:e}] at {x:e} ({:#010x}): \
                                 {got:e}, want {want:e}",
                                clamp.0,
                                clamp.1,
                                x.to_bits()
                            ));
                        }
                        checked += 1;
                    }
                }
            }
        }
        println!(
            "LEVELS {name}: {checked} samples checked, {} differ from Metal",
            wrong.len()
        );
        for line in wrong.iter().take(60) {
            println!("LEVELS-DIFFERS {name}: {line}");
        }
        if !own_sampler(&device, &messages) {
            assert!(wrong.is_empty(), "{name}: {} samples differ", wrong.len());
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// The positions texture: level `L` of `max(13 >> L, 1)` by `max(11 >> L, 1)` texels, texel
/// `(i, j)` holding `i + 16 j + 1000 L`.
fn positions_texture(device: &Device) -> Texture {
    let descriptor =
        TextureDescriptor::texture_2d(PixelFormat::R32Float, WIDTH.into(), HEIGHT.into(), true);
    assert_eq!(descriptor.mipmap_level_count, u64::from(POSITION_LEVELS));
    let texture = device.new_texture(&descriptor).unwrap();
    for level in 0..POSITION_LEVELS {
        let (w, h) = ((WIDTH >> level).max(1), (HEIGHT >> level).max(1));
        let bytes: Vec<u8> = (0..h)
            .flat_map(|j| (0..w).map(move |i| (i, j)))
            .flat_map(|(i, j)| ((i + 16 * j + 1000 * level) as f32).to_le_bytes())
            .collect();
        texture
            .replace_region(
                Region::new_2d(0, 0, w.into(), h.into()),
                level.into(),
                0,
                &bytes,
                u64::from(w) * 4,
                0,
            )
            .unwrap();
    }
    texture
}

/// The positions lanes' levels: below the first, whole, on either side of a nearest mip filter's
/// half-way point, at the clamp's bounds, and past the last.
const POSITION_ZS: [f32; 13] = [
    -1.0, 0.0, 0.4, 0.6, 1.0, 1.4, 1.55, 2.0, 2.4, 2.6, 3.0, 4.0, 9.0,
];

/// The positions lanes' `u`: each texel edge of each level's width, shifted by -3 to 3 periods (0
/// excluded: inside `[0, 1)` both rules agree), and 1 to 3 ulps either side.
fn position_us() -> Vec<f32> {
    let mut out = Vec::new();
    for level in 0..POSITION_LEVELS {
        let n = (WIDTH >> level).max(1) as i32;
        for k in [-3i32, -2, -1, 1, 2, 3] {
            for e in 0..=n {
                let edge = (k * n + e) as f32 / n as f32;
                let (mut up, mut down) = (edge, edge);
                out.push(edge);
                for _ in 0..3 {
                    up = up.next_up();
                    down = down.next_down();
                    out.extend([up, down]);
                }
            }
        }
    }
    out
}

/// The lanes decide off the base level: at every level wider than one texel, Metal's position and
/// multiplying first part on some lane under both address modes; and a nearest mip filter's
/// clamp into `[0.6, 2.4]` reads levels 1 and 2 only.
#[test]
fn the_position_lanes_part_at_every_level() {
    let us = position_us();
    for level in 0..POSITION_LEVELS - 1 {
        let w = WIDTH >> level;
        for mirrored in [false, true] {
            let deciding = us
                .iter()
                .filter(|&&u| metal_texel(u, w, mirrored) != multiplied_first(u, w, mirrored))
                .count();
            assert!(
                deciding > 0,
                "level {level}, mirrored {mirrored}: no deciding lane"
            );
        }
    }
    let read: Vec<f32> = POSITION_ZS
        .iter()
        .map(|&z| nearest(z, (0.6, 2.4), POSITION_LEVELS))
        .collect();
    assert!(read.iter().all(|&l| l == 1.0 || l == 2.0), "{read:?}");
    assert!(read.contains(&1.0) && read.contains(&2.0), "{read:?}");
}

/// On a device that samples through Metal's own sampler, every nearest sample at a level through a
/// runtime sampler reads Metal's texel of the level Metal reads, without a mip filter, with a
/// nearest one, and with one clamping the level, on repeating and mirrored axes; elsewhere the
/// device's sampler reads its own, announced `device-sampling`. Prints, per sampler, the lanes
/// checked, those where multiplying first would read another texel (the deciding ones), and each
/// lane that differs.
#[test]
fn runtime_nearest_samples_read_metals_texel_at_their_level() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let state = pipeline(&device, "unv_rpos2d");
        let positions = positions_texture(&device);
        let lanes: Vec<(f32, f32)> = POSITION_ZS
            .iter()
            .flat_map(|&z| position_us().into_iter().map(move |u| (u, z)))
            .collect();
        let inputs: Vec<u8> = lanes
            .iter()
            .flat_map(|&(u, z)| [u, ROW, z, 0.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut wrong = Vec::new();
        for (address, mirrored) in [
            (SamplerAddressMode::Repeat, false),
            (SamplerAddressMode::MirrorRepeat, true),
        ] {
            for (mip, clamp) in [
                (SamplerMipFilter::NotMipmapped, (0.0, f32::MAX)),
                (SamplerMipFilter::Nearest, (0.0, f32::MAX)),
                (SamplerMipFilter::Nearest, (0.6, 2.4)),
            ] {
                let s = sampler(&device, SamplerMinMagFilter::Nearest, mip, clamp, address);
                let out = run(&device, &state, (&positions, &s), false, (&inputs, 16), 1);
                let (mut deciding, mut differ) = (0usize, 0usize);
                for (&(u, z), &got) in lanes.iter().zip(&out) {
                    let level = if mip == SamplerMipFilter::NotMipmapped {
                        0
                    } else {
                        nearest(z, clamp, POSITION_LEVELS) as u32
                    };
                    let (w, h) = ((WIDTH >> level).max(1), (HEIGHT >> level).max(1));
                    let (x, first) = (
                        metal_texel(u, w, mirrored),
                        multiplied_first(u, w, mirrored),
                    );
                    let y = metal_texel(ROW, h, mirrored);
                    deciding += usize::from(x != first);
                    let want = (x + 16 * y + 1000 * level) as f32;
                    if got.to_bits() != want.to_bits() {
                        differ += 1;
                        wrong.push(format!(
                            "{address:?} {mip:?} [{:e}, {:e}] u {u:e} ({:08x}) at {z}: read \
                             {got}, Metal {want} (multiplied first {})",
                            clamp.0,
                            clamp.1,
                            u.to_bits(),
                            (first + 16 * y + 1000 * level) as f32
                        ));
                    }
                }
                println!(
                    "LEVELED {name} {address:?} {mip:?} [{:e}, {:e}]: {} lanes, {deciding} \
                     deciding, {differ} differ from Metal",
                    clamp.0,
                    clamp.1,
                    lanes.len()
                );
            }
        }
        for line in wrong.iter().take(60) {
            println!("LEVELED-DIFFERS {name}: {line}");
        }
        if !own_sampler(&device, &messages) {
            assert!(wrong.is_empty(), "{name}: {} lanes differ", wrong.len());
        }
    }
    layer_reported_nothing_at_all(&messages);
}
