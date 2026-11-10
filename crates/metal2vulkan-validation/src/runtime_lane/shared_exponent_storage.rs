//! A kernel's storage reads and writes of `RGB9E5Float` on the device the cfg selects, under the
//! validation layer, against Metal's answers (an Apple M3 Max under macOS 27.0.1, our own probe
//! `srgb8`: every texel of 65,536 `float` writes, both accesses, and 65,536 `half` writes, and
//! 16,384 words read). Every device writes and reads it through its `R32_UINT` alias
//! (`StorageImageClass::SharedExponentBits`), where the shader encodes Metal's word: a device's own
//! conversion rounds each mantissa half up, where Metal truncates it.
//!
//! Metal's word for red, green and blue: each channel clamped to `[0, 65408]` (a NaN, a negative
//! value and `-0` 0), the exponent `e = max(⌊log2 m⌋, -16) + 16` of the largest clamped channel
//! `m`, each mantissa the channel times `2^(24 - e)` truncated. A read gives each channel `mantissa
//! · 2^(e - 24)`, exact in `float` and in `half`, and alpha 1. The kernels are the owned fixture
//! `fpv_srgb_storage`'s, which take any `float` storage texture.

use super::degrade_notices::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions,
    Size, Texture, TextureDescriptor,
};

/// The texture's width; its height holds every value.
const WIDTH: u64 = 256;

/// Metal's word for red, green and blue `rgb` (the module docs).
fn word(rgb: [f32; 3]) -> u32 {
    let c = rgb.map(|v| match v.is_nan() || v.is_sign_negative() {
        true => 0.0,
        false => v.min(65408.0),
    });
    let m = c[0].max(c[1]).max(c[2]);
    let e = (((m.to_bits() >> 23) as i32).max(111) - 111) as u32;
    let scale = 2f64.powi(24 - e as i32);
    let mantissa = c.map(|v| ((f64::from(v) * scale).floor() as u32).min(511));
    mantissa[0] | mantissa[1] << 9 | mantissa[2] << 18 | e << 27
}

/// The value of `half` bits `h`, exactly (a NaN for a NaN).
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024.0 + m) * 2f32.powi(e - 25),
    }
}

/// The texels a write takes, four lanes each: specials, mantissa-step edges of every exponent,
/// and spread random values from `2^-30` to past the clamp.
fn floats() -> Vec<[f32; 4]> {
    let specials: [f32; 14] = [
        0.0,
        -0.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        65408.0,
        65472.0,
        65536.0,
        2e-5,
        1.0,
        0.5,
        f32::from_bits(1),
        2f32.powi(-16),
        2f32.powi(-24),
    ];
    let mut out: Vec<[f32; 4]> = specials
        .iter()
        .flat_map(|&a| {
            specials
                .iter()
                .map(move |&b| [a, b, 0.5 * a + 0.25 * b, 1.0])
        })
        .collect();
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    };
    for _ in 0..16384 {
        let e = (next() % 48) as i32 - 30;
        let big = 2f32.powi(e) * (1.0 + (next() % 4096) as f32 / 4096.0);
        let frac = |r: u64| (r % 65536) as f32 / 65536.0;
        out.push([
            big,
            big * frac(next()),
            big * frac(next()) * frac(next()),
            0.5,
        ]);
        let unit = 2f32.powi(e.clamp(-24, 7));
        let edge = ((next() % 511) as f32 + 0.5) * unit;
        out.push([
            edge,
            f32::from_bits(edge.to_bits() + 1),
            f32::from_bits(edge.to_bits() - 1),
            0.0,
        ]);
    }
    while !out.len().is_multiple_of(WIDTH as usize) {
        out.push([0.0; 4]);
    }
    out
}

fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("fpv_srgb_storage.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A shared `RGB9E5Float` texture, [`WIDTH`] texels wide and `height` high, that kernels read and
/// write, holding the words `words`.
fn texture(device: &Device, height: u64, words: &[u32]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGB9E5Float, WIDTH, height, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    texture
        .replace_region(
            Region::new_2d(0, 0, WIDTH, height),
            0,
            0,
            &bytes,
            WIDTH * 4,
            0,
        )
        .unwrap();
    texture
}

/// `texture`'s words, `height` rows.
fn words_of(texture: &Texture, height: u64) -> Vec<u32> {
    let mut out = vec![0u8; (WIDTH * height * 4) as usize];
    texture
        .get_bytes(
            &mut out,
            WIDTH * 4,
            0,
            Region::new_2d(0, 0, WIDTH, height),
            0,
            0,
        )
        .unwrap();
    out.as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect()
}

/// Run `state` over every texel of `texture` (`height` rows) with `data` at buffer 0, and return
/// buffer 0's bytes.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    height: u64,
    data: &[u8],
) -> Vec<u8> {
    let data_buffer = buffer_with(device, data).unwrap();
    let width = buffer_with(device, &(WIDTH as u32).to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&data_buffer), 0, 0).unwrap();
    encoder.set_buffer(Some(&width), 0, 1).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(WIDTH / 16, height, 1), Size::new(16, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    read_buffer(&data_buffer, 0, data.len() as u64).unwrap()
}

/// Every kernel write of the floats, and of every half (each lane its own bit pattern), stores
/// Metal's word, over words of `0xA5A5A5A5`.
#[test]
fn shared_exponent_storage_writes_store_metals_word() {
    on_device_with(at_plan, |device, _| {
        let name = device.name().to_owned();
        let floats = floats();
        let halves: Vec<[u16; 4]> = (0..=u16::MAX)
            .map(|h| {
                [
                    h,
                    h.wrapping_mul(7).wrapping_add(3),
                    h.wrapping_mul(13).wrapping_add(5),
                    0x3c00,
                ]
            })
            .collect();
        let mut differ = Vec::new();
        for (kernel, half) in [
            ("fpv_srgb_write", false),
            ("fpv_srgb_update", false),
            ("fpv_srgb_write_half", true),
            ("fpv_srgb_update_half", true),
        ] {
            let (texels, data): (Vec<[f32; 3]>, Vec<u8>) = match half {
                false => (
                    floats.iter().map(|t| [t[0], t[1], t[2]]).collect(),
                    floats
                        .iter()
                        .flatten()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect(),
                ),
                true => (
                    halves
                        .iter()
                        .map(|t| [0, 1, 2].map(|c| half_value(t[c])))
                        .collect(),
                    halves
                        .iter()
                        .flatten()
                        .flat_map(|h| h.to_le_bytes())
                        .collect(),
                ),
            };
            let height = texels.len() as u64 / WIDTH;
            let state = state(device, kernel);
            let target = texture(device, height, &vec![0xA5A5_A5A5; texels.len()]);
            run(device, &state, &target, height, &data);
            let got = words_of(&target, height);
            let wrong: Vec<_> = texels
                .iter()
                .zip(&got)
                .filter(|(rgb, w)| word(**rgb) != **w)
                .collect();
            eprintln!(
                "RGB9E5 write by {kernel} on {name}: {} of {} texels differ from Metal's{}",
                wrong.len(),
                texels.len(),
                wrong.first().map_or(String::new(), |(rgb, w)| format!(
                    ", e.g. {rgb:?}: {w:#010x}, Metal {:#010x}",
                    word(**rgb)
                ))
            );
            if !wrong.is_empty() {
                differ.push(format!("{kernel}: {} of {}", wrong.len(), texels.len()));
            }
        }
        assert!(
            differ.is_empty(),
            "RGB9E5 storage writes not Metal's: {differ:#?}"
        );
    });
}

/// Every `read_write` read of words of every exponent: each channel exact, alpha 1, `float` and
/// `half`.
#[test]
fn shared_exponent_storage_reads_decode_exactly() {
    on_device_with(at_plan, |device, _| {
        let name = device.name().to_owned();
        let words: Vec<u32> = (0..WIDTH as u32 * 4)
            .map(|n| (n % 32) << 27 | (n * 37 % 512) | (n * 101 % 512) << 9 | (n * 7 % 512) << 18)
            .collect();
        let height = words.len() as u64 / WIDTH;
        let source = texture(device, height, &words);
        let mut differ = Vec::new();
        for (kernel, half) in [
            ("fpv_srgb_read_write", false),
            ("fpv_srgb_read_write_half", true),
        ] {
            let size = if half { 8 } else { 16 };
            let state = state(device, kernel);
            let got = run(
                device,
                &state,
                &source,
                height,
                &vec![0u8; words.len() * size],
            );
            let mut wrong = 0;
            for (n, &w) in words.iter().enumerate() {
                for c in 0..4usize {
                    let v = match c {
                        3 => 1.0,
                        _ => ((w >> (9 * c)) & 0x1ff) as f32 * 2f32.powi((w >> 27) as i32 - 24),
                    };
                    let lane = &got[(n * 4 + c) * size / 4..];
                    let ok = match half {
                        true => {
                            half_value(u16::from_le_bytes([lane[0], lane[1]])).to_bits()
                                == v.to_bits()
                        }
                        false => {
                            u32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]) == v.to_bits()
                        }
                    };
                    wrong += usize::from(!ok);
                }
            }
            eprintln!(
                "RGB9E5 read by {kernel} on {name}: {wrong} of {} lanes differ from Metal's",
                words.len() * 4
            );
            if wrong > 0 {
                differ.push(format!("{kernel}: {wrong}"));
            }
        }
        assert!(
            differ.is_empty(),
            "RGB9E5 storage reads not Metal's: {differ:#?}"
        );
    });
}
