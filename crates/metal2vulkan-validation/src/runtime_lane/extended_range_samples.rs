//! 2D linear samples of an extended-range texture on the device the cfg selects, under the
//! validation layer.
//!
//! The owned fixture `xr_samples` samples its texture through the runtime sampler at
//! `[[sampler(0)]]` (here linear, clamp to edge, normalized coordinates) and through a constexpr
//! sampler of that state, with `float` and `half` results, one thread per coordinate. Each
//! format's texture is 32 x 2 texels: sample `k` reads texels `2k` and `2k + 1` of both rows at
//! the fractions Metal sampled.
//!
//! Metal's answer: `xr_samples.tsv` beside the fixture, 16 linear samples per format and result
//! type an Apple M3 Max (macOS 26.7) took, byte-identical on an Apple M4 (macOS 27.0.1), through a
//! runtime sampler of this state. A differing sample is printed on an `XR-SAMPLES-DIFFERS` line.
//!
//! A runtime sampler stating `MirrorClampToEdge` on `s` and `t` reads each tap mirrored once
//! about the level's start and then clamped, so its sample at a texel-space
//! position `x` off the texel boundaries and off the 1/512 ties is the clamp-to-edge sampler's at
//! `|x|` on each axis: the same taps at the same weights (a nearest one's `floor(-x) = -1 -
//! floor(x)`, a linear one's pair reversed). The lane compares the two on the device, nearest and
//! linear, normalized and pixel coordinates, `float` and `half` results; a differing sample is
//! printed on an `XR-MIRROR-CLAMP-DIFFERS` line.

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, xr_bicubic};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device, PipelineOptions,
    Region, ResourceOptions, SamplerState, Size, Texture, TextureDescriptor,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Each texture's width: two texels per sample.
const WIDTH: u64 = 32;

/// One of Metal's samples: the raw format, a `half` result (else `float`), the fractions in
/// 1/256 steps, the codes (red, green, blue, alpha) of the taps `(i, j)`, `(i + 1, j)`,
/// `(i, j + 1)` and `(i + 1, j + 1)`, and the result's lanes as bits.
struct Sample {
    raw: u32,
    half: bool,
    kx: u32,
    ky: u32,
    taps: [[u32; 4]; 4],
    out: [u32; 4],
}

/// Metal's samples, from `xr_samples.tsv`.
fn samples() -> Vec<Sample> {
    let path = engine_dir().join("xr_samples.tsv");
    let text = std::fs::read_to_string(&path).unwrap();
    let four = |s: &str, radix: u32| -> [u32; 4] {
        let v: Vec<u32> = s
            .split(',')
            .map(|x| u32::from_str_radix(x, radix).unwrap())
            .collect();
        [v[0], v[1], v[2], v[3]]
    };
    text.lines()
        .map(|line| {
            let f: Vec<&str> = line.split(' ').collect();
            Sample {
                raw: f[0].parse().unwrap(),
                half: f[1] == "h",
                kx: f[2].parse().unwrap(),
                ky: f[3].parse().unwrap(),
                taps: [
                    four(f[4], 10),
                    four(f[5], 10),
                    four(f[6], 10),
                    four(f[7], 10),
                ],
                out: four(f[8], 16),
            }
        })
        .collect()
}

/// The pixel format of raw format `raw`.
pub(super) fn format(raw: u32) -> PixelFormat {
    match raw {
        552 => PixelFormat::BGRA10XR,
        553 => PixelFormat::BGRA10XRSRGB,
        554 => PixelFormat::BGR10XR,
        555 => PixelFormat::BGR10XRSRGB,
        _ => panic!("raw format {raw}"),
    }
}

/// The bytes of a texel of codes `[r, g, b, a]` in raw format `raw`: `BGRA10_XR`'s four 16-bit
/// fields (blue, green, red, alpha), each code in bits 6 to 15; `BGR10_XR`'s word, blue in bits 0
/// to 9, green 10 to 19, red 20 to 29.
pub(super) fn texel(raw: u32, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
    match raw {
        552 | 553 => [b, g, r, a]
            .into_iter()
            .flat_map(|c| ((c << 6) as u16).to_le_bytes())
            .collect(),
        _ => (r << 20 | g << 10 | b).to_le_bytes().to_vec(),
    }
}

/// The `WIDTH` x 2 texture of `rows`' format holding their taps: sample `k`'s at texels `2k` and
/// `2k + 1` of both rows.
fn texture(device: &Device, rows: &[&Sample]) -> Texture {
    let raw = rows[0].raw;
    let bytes_per_texel = if raw <= 553 { 8 } else { 4 };
    let mut bytes = vec![0u8; (WIDTH * 2 * bytes_per_texel) as usize];
    for (k, s) in rows.iter().enumerate() {
        for (t, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let at = (dy * WIDTH as usize + 2 * k + dx) * bytes_per_texel as usize;
            let texel = texel(raw, s.taps[t]);
            bytes[at..at + texel.len()].copy_from_slice(&texel);
        }
    }
    let mut descriptor = TextureDescriptor::texture_2d(format(raw), WIDTH, 2, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    texture
        .replace_region(
            Region::new_2d(0, 0, WIDTH, 2),
            0,
            0,
            &bytes,
            WIDTH * bytes_per_texel,
            0,
        )
        .unwrap();
    texture
}

/// The normalized coordinates of `rows`: sample `k` at `((2k + 1/2 + kx/256) / WIDTH, (1/2 +
/// ky/256) / 2)`, exact in `f32`.
fn coords(device: &Device, rows: &[&Sample]) -> Buffer {
    let bytes: Vec<u8> = rows
        .iter()
        .enumerate()
        .flat_map(|(k, s)| {
            let u = (512 * k as u32 + 128 + s.kx) as f32 / (256.0 * WIDTH as f32);
            let v = (128 + s.ky) as f32 / 512.0;
            [u, v].into_iter().flat_map(f32::to_le_bytes)
        })
        .collect();
    buffer_with(device, &bytes).unwrap()
}

/// The fixture's kernel `name`.
pub(super) fn state(device: &Device, name: &str) -> ComputePipelineState {
    state_of(device, "xr_samples", name)
}

/// The kernel `name` of the owned fixture `fixture`.
pub(super) fn state_of(device: &Device, fixture: &str, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{fixture}.ll"))).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// What `state` samples of `texture` at `coords` (`lanes` of them), through `sampler` where the
/// kernel takes one: each result's lanes as bits (`half` lanes where `half`).
fn run(
    device: &Device,
    queue: &CommandQueue,
    (state, sampler): (&ComputePipelineState, Option<&SamplerState>),
    (texture, coords, lanes): (&Texture, &Buffer, u64),
    half: bool,
) -> Vec<[u32; 4]> {
    let size = if half { 8 } else { 16 };
    let out = buffer_with(device, &vec![0xab; (lanes * size) as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    if let Some(sampler) = sampler {
        encoder.set_sampler_state(Some(sampler), 0).unwrap();
    }
    encoder.set_buffer(Some(coords), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(lanes, 1, 1), Size::new(lanes, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let bytes = read_buffer(&out, 0, lanes * size).unwrap();
    bytes
        .chunks_exact(size as usize)
        .map(|v| {
            std::array::from_fn(|c| match half {
                true => u32::from(u16::from_le_bytes([v[2 * c], v[2 * c + 1]])),
                false => u32::from_le_bytes([v[4 * c], v[4 * c + 1], v[4 * c + 2], v[4 * c + 3]]),
            })
        })
        .collect()
}

/// The file holds 16 samples of each format and result type.
#[test]
fn metal_s_samples_cover_every_format_and_result() {
    let all = samples();
    for raw in 552..=555 {
        for half in [false, true] {
            let n = all
                .iter()
                .filter(|s| s.raw == raw && s.half == half)
                .count();
            assert_eq!(n, 16, "{raw} half {half}");
        }
    }
    assert_eq!(all.len(), 128);
}

/// Each of Metal's linear samples of each extended-range format, through the runtime sampler
/// and through the constexpr one, with a `float` and a `half` result, is Metal's within the
/// device's filter arithmetic (`bicubic-filter-arithmetic`, `xr_bicubic::filtered_close`): the
/// decoded taps blended by the device's `FMix`, where Metal sums them exactly on the format's grid.
#[test]
fn linear_samples_of_extended_range_textures_are_metal_s() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let mut linear = SamplerDescriptor::new();
        linear.min_filter = SamplerMinMagFilter::Linear;
        linear.mag_filter = SamplerMinMagFilter::Linear;
        let sampler = device.new_sampler_state(&linear).unwrap();
        let all = samples();
        let mut wrong = Vec::new();
        let mut checked = 0;
        for half in [false, true] {
            let suffix = if half { "_half" } else { "" };
            let kernels = [
                (format!("xr_sample_runtime{suffix}"), Some(&sampler)),
                (format!("xr_sample_constexpr{suffix}"), None),
            ];
            for (name, sampler) in &kernels {
                let state = state(device, name);
                for raw in 552..=555 {
                    let rows: Vec<&Sample> = all
                        .iter()
                        .filter(|s| s.raw == raw && s.half == half)
                        .collect();
                    let texture = texture(device, &rows);
                    let coords = coords(device, &rows);
                    let lanes = rows.len() as u64;
                    let got = run(
                        device,
                        &queue,
                        (&state, *sampler),
                        (&texture, &coords, lanes),
                        half,
                    );
                    for (k, (s, g)) in rows.iter().zip(got).enumerate() {
                        checked += 1;
                        if !xr_bicubic::filtered_close(&g, &s.out, half, 0.0) {
                            wrong.push(format!(
                                "{name} {raw} sample {k} kx {} ky {} taps {:?}: {g:x?}, Metal {:x?}",
                                s.kx, s.ky, s.taps, s.out
                            ));
                        }
                    }
                }
            }
        }
        for line in &wrong {
            println!("XR-SAMPLES-DIFFERS {}: {line}", device.name());
        }
        println!(
            "XR-SAMPLES {}: {checked} samples checked, {} differ from Metal",
            device.name(),
            wrong.len()
        );
        assert!(
            wrong.is_empty(),
            "{} samples differ from Metal",
            wrong.len()
        );
    });
}

/// A `w` x `h` texture of raw format `raw` holding codes below 1024 drawn from `seed`.
fn random_texture(device: &Device, raw: u32, (w, h): (u64, u64), seed: u64) -> Texture {
    let bytes_per_texel = if raw <= 553 { 8 } else { 4 };
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let bytes: Vec<u8> = (0..w * h)
        .flat_map(|_| texel(raw, std::array::from_fn(|_| next() % 1024)))
        .collect();
    let mut descriptor = TextureDescriptor::texture_2d(format(raw), w, h, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    texture
        .replace_region(
            Region::new_2d(0, 0, w, h),
            0,
            0,
            &bytes,
            w * bytes_per_texel,
            0,
        )
        .unwrap();
    texture
}

/// A buffer of the `float2` coordinates `at`.
fn coordinate_buffer(device: &Device, at: impl Iterator<Item = (f32, f32)>) -> Buffer {
    let bytes: Vec<u8> = at
        .flat_map(|(u, v)| [u, v].into_iter().flat_map(f32::to_le_bytes))
        .collect();
    buffer_with(device, &bytes).unwrap()
}

/// Each extended-range format's sample through a runtime sampler stating `MirrorClampToEdge` on
/// `s` and `t` is the clamp-to-edge sampler's at the reflected position (module docs), nearest
/// and linear, normalized and pixel coordinates, with a `float` and a `half` result, at 64
/// positions within three sizes either side of the 8 x 4 texture: a linear one within the
/// device's `FMix` of the reflected fraction (`xr_bicubic::filtered_close`).
#[test]
fn mirror_clamped_axes_sample_as_metal_addresses_them() {
    on_device(|device, _| {
        const W: u64 = 8;
        const H: u64 = 4;
        const LANES: u64 = 64;
        let queue = device.new_command_queue().unwrap();
        let mut seed = 0x5eed_0471_0004_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as u32
        };
        // Texel-space positions `(m + 1/4) / 256`: off every texel boundary and every tie.
        let mut at = |n: u64| {
            let span = 6 * 256 * n as u32;
            (f64::from(next() % span) - f64::from(span / 2) + 0.25) as f32 / 256.0
        };
        let positions: Vec<(f32, f32)> = (0..LANES).map(|_| (at(W), at(H))).collect();
        let mut wrong = Vec::new();
        let mut checked = 0;
        for raw in 552..=555 {
            let texture = random_texture(device, raw, (W, H), u64::from(raw));
            for half in [false, true] {
                let suffix = if half { "_half" } else { "" };
                let state = state(device, &format!("xr_sample_runtime{suffix}"));
                for (linear, pixel) in [(false, false), (true, false), (false, true), (true, true)]
                {
                    let sampler = |mode| {
                        let mut d = SamplerDescriptor::new();
                        if linear {
                            d.min_filter = SamplerMinMagFilter::Linear;
                            d.mag_filter = SamplerMinMagFilter::Linear;
                        }
                        d.normalized_coordinates = !pixel;
                        d.s_address_mode = mode;
                        d.t_address_mode = mode;
                        device.new_sampler_state(&d).unwrap()
                    };
                    let mirror = sampler(SamplerAddressMode::MirrorClampToEdge);
                    let edge = sampler(SamplerAddressMode::ClampToEdge);
                    let scale = |(x, y): (f32, f32)| match pixel {
                        true => (x, y),
                        false => (x / W as f32, y / H as f32),
                    };
                    let direct = coordinate_buffer(device, positions.iter().map(|p| scale(*p)));
                    let reflected = coordinate_buffer(
                        device,
                        positions.iter().map(|(x, y)| scale((x.abs(), y.abs()))),
                    );
                    let got = run(
                        device,
                        &queue,
                        (&state, Some(&mirror)),
                        (&texture, &direct, LANES),
                        half,
                    );
                    let want = run(
                        device,
                        &queue,
                        (&state, Some(&edge)),
                        (&texture, &reflected, LANES),
                        half,
                    );
                    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                        checked += 1;
                        if !xr_bicubic::filtered_close(g, w, half, 0.0) {
                            wrong.push(format!(
                                "{raw} half {half} linear {linear} pixel {pixel} at {:?}: {g:x?}, clamp-to-edge at the reflection {w:x?}",
                                positions[k]
                            ));
                        }
                    }
                }
            }
        }
        for line in &wrong {
            println!("XR-MIRROR-CLAMP-DIFFERS {}: {line}", device.name());
        }
        println!(
            "XR-MIRROR-CLAMP {}: {checked} samples checked, {} differ",
            device.name(),
            wrong.len()
        );
        assert!(wrong.is_empty(), "{} samples differ", wrong.len());
    });
}
