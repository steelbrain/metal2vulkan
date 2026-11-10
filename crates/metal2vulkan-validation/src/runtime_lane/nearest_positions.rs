//! Nearest samples through a runtime sampler on repeating and mirrored axes on the device the cfg
//! selects, under the validation layer.
//!
//! Metal's position (measured on an M3 Max and an M4): per axis of size `N`, the normalized
//! coordinate `u` is first wrapped into one period, `u - floor(u)` under `repeat` and `u - 2
//! floor(u / 2)` under `mirrored_repeat`, the subtraction rounded toward positive infinity; then
//! scaled, `X = u' N` rounded to `f32` with ties away from zero; a nearest filter reads texel
//! `floor(X)`, wrapped or mirrored. A Vulkan device multiplies first, `u N`, and wraps the texel
//! after, which reads the next texel wherever `u` lies outside `[0, 1)` on a size that is not a
//! power of two and `u N` lands within its rounding of a texel edge. The translation snaps a
//! constexpr sampler's nearest sample onto Metal's texel by its state, and a runtime sampler's by
//! its word where the word reads one texel of the base level (SEMANTICS 1091); before that,
//! lavapipe, RADV and NVIDIA each read another texel on some lanes.
//!
//! The owned `a2_clamp_to_zero` kernel samples its float texture through the runtime sampler at
//! `[[sampler(0)]]` at level 0 (its third result); here a 13x11 `R32Float` texture whose texel
//! `(x, y)` holds `x + 16 y`, through a nearest sampler that repeats, then one that mirrors, both
//! axes. Lanes lie at each texel edge of the 13-texel axis shifted by -3 to 3 periods, and 1 to 3
//! ulps either side, on the centre row of the 11-texel one.

use super::degrade_notices::on_device_with;
use super::pixel_samplers::own_sampler;
use super::{at_plan, buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions,
    Size, Texture, TextureDescriptor,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The texture's size: neither axis a power of two.
const WIDTH: u32 = 13;
const HEIGHT: u32 = 11;

/// `x`, exact in `f64`, rounded to `f32` toward positive infinity.
fn rounded_up(x: f64) -> f32 {
    let f = x as f32;
    if f64::from(f) < x {
        f.next_up()
    } else {
        f
    }
}

/// `x`, exact in `f64`, rounded to the nearest `f32`, ties away from zero.
fn rounded_away(x: f64) -> f32 {
    let f = x as f32;
    let d = f64::from(f) - x;
    if d == 0.0 {
        return f;
    }
    let other = if d > 0.0 { f.next_down() } else { f.next_up() };
    if (f64::from(other) - x).abs() == d.abs() && other.abs() > f.abs() {
        other
    } else {
        f
    }
}

/// Texel `i` of an axis of `n` wrapped (`mirrored` false) or mirrored into the level.
fn wrapped(i: i64, n: u32, mirrored: bool) -> u32 {
    let n = i64::from(n);
    let t = if mirrored {
        let m = i.rem_euclid(2 * n);
        if m < n {
            m
        } else {
            2 * n - 1 - m
        }
    } else {
        i.rem_euclid(n)
    };
    u32::try_from(t).unwrap()
}

/// The texel Metal's nearest filter reads at `u` on an axis of `n`: wrapped into one period
/// first, rounded up, then scaled with ties away from zero.
pub(super) fn metal_texel(u: f32, n: u32, mirrored: bool) -> u32 {
    let u = f64::from(u);
    let start = if mirrored {
        2.0 * (u / 2.0).floor()
    } else {
        u.floor()
    };
    let x = rounded_away(f64::from(rounded_up(u - start)) * f64::from(n));
    wrapped(x.floor() as i64, n, mirrored)
}

/// The texel a device that multiplies first reads at `u`: `u N` rounded to nearest even, then
/// wrapped.
pub(super) fn multiplied_first(u: f32, n: u32, mirrored: bool) -> u32 {
    wrapped((u * n as f32).floor() as i64, n, mirrored)
}

/// The lanes' `u`: each texel edge of the 13-texel axis, shifted by -3 to 3 periods (0 excluded:
/// inside `[0, 1)` both rules agree), and 1 to 3 ulps either side.
fn lanes() -> Vec<f32> {
    let mut out = Vec::new();
    for k in [-3i32, -2, -1, 1, 2, 3] {
        for e in 0..=WIDTH as i32 {
            let edge = (k * WIDTH as i32 + e) as f32 / WIDTH as f32;
            let (mut up, mut down) = (edge, edge);
            out.push(edge);
            for _ in 0..3 {
                up = up.next_up();
                down = down.next_down();
                out.push(up);
                out.push(down);
            }
        }
    }
    out
}

fn state(device: &Device) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("a2_clamp_to_zero.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("a2_clamp_to_zero").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A `width` x `height` texture of `format` whose texels are `texel(x, y)`'s bytes.
fn texture(
    device: &Device,
    format: PixelFormat,
    (width, height): (u32, u32),
    texel: impl Fn(u32, u32) -> [u8; 4],
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, width.into(), height.into(), false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| texel(x, y))
        .collect();
    texture
        .replace_region(
            Region::new_2d(0, 0, width.into(), height.into()),
            0,
            0,
            &bytes,
            u64::from(width) * 4,
            0,
        )
        .unwrap();
    texture
}

/// Each lane's red channel of the runtime sampler's sample, `float` bound with a nearest sampler
/// of `address` on every axis, at `(u, 0.5)` for each `u` of `us`.
fn sampled(device: &Device, float: &Texture, address: SamplerAddressMode, us: &[f32]) -> Vec<f32> {
    reds(device, float, address, us)
        .into_iter()
        .map(|lane| lane[2])
        .collect()
}

/// Each lane's red channel of all five of the kernel's results, `float` bound with a nearest
/// sampler of `address` on every axis, at `(u, 0.5)` for each `u` of `us`: the two constexpr
/// clamp-to-zero samplers' (0 and 1), the runtime sampler's (2), and the integer textures' (3
/// and 4).
fn reds(
    device: &Device,
    float: &Texture,
    address: SamplerAddressMode,
    us: &[f32],
) -> Vec<[f32; 5]> {
    let state = state(device);
    let queue = device.new_command_queue().unwrap();
    let small = |format| texture(device, format, (4, 4), |_, _| 1u32.to_le_bytes());
    let (uint, int) = (small(PixelFormat::R32Uint), small(PixelFormat::R32Sint));
    let coords: Vec<u8> = us
        .iter()
        .flat_map(|&u| [u, 0.5])
        .flat_map(f32::to_le_bytes)
        .collect();
    let coords = buffer_with(device, &coords).unwrap();
    let mut descriptor = SamplerDescriptor::new();
    descriptor.s_address_mode = address;
    descriptor.t_address_mode = address;
    descriptor.r_address_mode = address;
    let sampler = device.new_sampler_state(&descriptor).unwrap();
    let lanes = us.len() as u64;
    let out = buffer_with(device, &vec![0xab; (lanes * 5 * 16) as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(float), 0).unwrap();
    encoder.set_texture(Some(&uint), 1).unwrap();
    encoder.set_texture(Some(&int), 2).unwrap();
    encoder.set_sampler_state(Some(&sampler), 0).unwrap();
    encoder.set_buffer(Some(&coords), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(lanes, 1, 1), Size::new(lanes / 4, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let bytes = read_buffer(&out, 0, lanes * 5 * 16).unwrap();
    bytes
        .as_chunks::<80>()
        .0
        .iter()
        .map(|lane| {
            std::array::from_fn(|result| {
                let at = 16 * result;
                f32::from_le_bytes([lane[at], lane[at + 1], lane[at + 2], lane[at + 3]])
            })
        })
        .collect()
}

/// The lanes decide: at 13 texels Metal's rule and multiplying first part where `u N`'s
/// rounding crosses an edge, on both address modes; and the roundings are the rule's.
#[test]
fn the_lanes_reach_where_the_rules_part() {
    let us = lanes();
    assert_eq!(us.len() % 4, 0, "four whole threadgroups");
    for mirrored in [false, true] {
        let deciding = us
            .iter()
            .filter(|&&u| metal_texel(u, WIDTH, mirrored) != multiplied_first(u, WIDTH, mirrored))
            .count();
        assert!(deciding > 0, "mirrored {mirrored}: no deciding lane");
    }
    // A tie at `f32`'s precision: halfway between 1 and the next `f32`.
    let tie = 1.0 + f64::powi(2.0, -24);
    assert_eq!(rounded_away(tie), 1.0f32.next_up());
    assert_eq!(rounded_away(-tie), -(1.0f32.next_up()));
    assert_eq!(tie as f32, 1.0, "nearest even takes the other neighbour");
    assert_eq!(rounded_up(1.0 + f64::EPSILON), 1.0f32.next_up());
}

/// Every lane reads Metal's texel where the device samples through Metal's own sampler; elsewhere
/// it samples by its own (announced `device-sampling`), whose differences are printed. Prints, per
/// address mode, the lanes checked, those where multiplying first would read another texel (the
/// deciding ones), and each lane that differs.
#[test]
fn runtime_nearest_samples_read_metal_s_texel_off_the_level() {
    on_device_with(at_plan, |device, messages| {
        let float = texture(device, PixelFormat::R32Float, (WIDTH, HEIGHT), |x, y| {
            ((x + 16 * y) as f32).to_le_bytes()
        });
        let us = lanes();
        let row = (0.5 * HEIGHT as f32).floor();
        let mut wrong = Vec::new();
        for (name, address, mirrored) in [
            ("repeat", SamplerAddressMode::Repeat, false),
            ("mirrored", SamplerAddressMode::MirrorRepeat, true),
        ] {
            let read = sampled(device, &float, address, &us);
            let mut deciding = 0usize;
            let mut differ = 0usize;
            for (&u, &got) in us.iter().zip(&read) {
                let metal = metal_texel(u, WIDTH, mirrored);
                let first = multiplied_first(u, WIDTH, mirrored);
                deciding += usize::from(metal != first);
                let expected = metal as f32 + 16.0 * row;
                if got.to_bits() != expected.to_bits() {
                    differ += 1;
                    wrong.push(format!(
                        "{name} u {u:e} ({:08x}): read {got}, Metal {expected} (multiplied first \
                         {})",
                        u.to_bits(),
                        first as f32 + 16.0 * row
                    ));
                }
            }
            println!(
                "NEAREST {} {name}: {} lanes, {deciding} deciding, {differ} differ from Metal",
                device.name(),
                us.len()
            );
        }
        for line in wrong.iter().take(60) {
            println!("NEAREST-DIFFERS {}: {line}", device.name());
        }
        if own_sampler(device, messages) {
            println!(
                "NEAREST {}: the device's own sampler, {} lanes differ from Metal",
                device.name(),
                wrong.len()
            );
            return;
        }
        assert!(wrong.is_empty(), "{} lanes differ from Metal", wrong.len());
    });
}

/// The three lanes one, two and three units in the last place below `+0.0`: the kernel's lane
/// index of each, and the `f32` bits the lane set gives it.
const BELOW_ZERO: [(usize, u32); 3] = [(289, 0x8000_0001), (291, 0x8000_0002), (293, 0x8000_0003)];

/// What Metal's two constexpr clamp-to-zero samplers read there, measured on an Apple M3 Max
/// (macOS 27.0.1) through MoltenVK, which samples through Metal's own texture unit: Metal reads a
/// subnormal coordinate as the zero of its sign before the address mode decides, so the nearest
/// sampler reads texel (0, 5) and the linear one averages it with the zero border. A device that
/// keeps the subnormal would read the border, 0, through both.
const METAL_BELOW_ZERO: [f32; 2] = [80.0, 40.0];

/// Both constexpr clamp-to-zero samplers one, two and three units in the last place below `+0.0`
/// read what the device's sampler reads there: the module hands the coordinate as given. Through
/// Metal's own texture unit (MoltenVK) that is Metal's value, which flushes the subnormal; a
/// device of its own (`device-sampling` announced) reads it as it does, Metal's value or the
/// border.
#[test]
fn clamp_to_zero_samples_just_below_zero_read_the_devices_texels() {
    let us = lanes();
    for &(lane, bits) in &BELOW_ZERO {
        assert_eq!(us[lane].to_bits(), bits, "lane {lane}");
    }
    super::degrade_notices::on_device(|device, messages| {
        let float = texture(device, PixelFormat::R32Float, (WIDTH, HEIGHT), |x, y| {
            ((x + 16 * y) as f32).to_le_bytes()
        });
        let read = reds(device, &float, SamplerAddressMode::Repeat, &us);
        let mut wrong = Vec::new();
        for &(lane, bits) in &BELOW_ZERO {
            for (column, metal) in METAL_BELOW_ZERO.into_iter().enumerate() {
                let got = read[lane][column];
                println!(
                    "BELOW-ZERO {} lane {lane} ({bits:08x}) column {column}: read {got}, Metal \
                     {metal}",
                    device.name()
                );
                if got.to_bits() != metal.to_bits() {
                    wrong.push(format!(
                        "lane {lane} ({bits:08x}) column {column}: read {got}, Metal {metal}"
                    ));
                }
            }
        }
        if own_sampler(device, messages) {
            println!(
                "BELOW-ZERO {}: the device's own sampler, {} lanes differ from Metal",
                device.name(),
                wrong.len()
            );
            return;
        }
        assert!(wrong.is_empty(), "{}: {wrong:#?}", device.name());
    });
}
