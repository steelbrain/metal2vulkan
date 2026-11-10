//! Samples, gathers and depth comparisons through a runtime sampler whose `s` axis is
//! `MirrorClampToEdge`, with pixel and with normalized coordinates, on the device the cfg selects,
//! under the validation layer.
//!
//! Metal's answer (an M4 under macOS 27.0.1, alike plain twice; a pixel sampler in the mode is
//! misuse, which Metal's validation stops at the descriptor, `validateMTLSamplerDescriptor:79:
//! failed assertion 'Sampler Descriptor Validation'`, and plain Metal samples): per axis of size
//! `N`, the position is the pixel coordinate as given, or the normalized one's `u N` rounded to
//! `f32` with ties away from zero, never wrapped. A nearest filter's tap is `floor(x)`; a linear
//! filter's, and a gather's pair, `q = floor(256 x - 127.5)`, the taps `q >> 8` and the next,
//! weighted `256 - (q & 255)` and `q & 255` in 256ths. A constant texel offset moves each tap, but
//! where the level's size drops it (3, 5, 6 or 7 texels at the base level, as on a repeating axis).
//! Each tap's index is then mirrored once, `-1 - i` for a negative `i`, and clamped to `N - 1`. A
//! comparison (`Less`) passes where the reference is below the texel.
//!
//! The owned fixture `rrp_mirror_clamp` samples a `texture2d<float>` (`rrp_mc_sample`,
//! `rrp_mc_gather`) or a `depth2d<float>` (`rrp_mc_compare`) through the runtime sampler at
//! `[[sampler(0)]]` at each lane's coordinate, without an offset and with the offsets -3, 2, -7
//! and 5 on `s`. Here textures 6 and 8 texels wide (the offset dropped and kept) and one high, an
//! `R32Float` one whose texel `j` holds `j + 1` and a `Depth32Float` one holding `(j + 1) / 16`;
//! lanes at `c + 1/2 + k/4` texels for `c` from `-2N` to `2N - 1` and `k` from 0 to 3, so every
//! linear weight is a quarter and every answer is exact; a comparison's reference at a lane's own
//! sixteenth plus a half.

use super::degrade_notices::on_device_with;
use super::pixel_samplers::{own_sampler, shifted, REACH};
use super::{at_plan, buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, SamplerAddressMode, SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions,
    Size, Texture, TextureDescriptor,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The fixture's texel offsets on `s`, one per result of a lane.
const OFFSETS: [i64; 5] = [0, -3, 2, -7, 5];

/// The lanes of a threadgroup.
const GROUP: u64 = 32;

/// What a lane reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Nearest,
    Linear,
    Gather,
    CompareNearest,
    CompareLinear,
}

impl Op {
    const ALL: [Op; 5] = [
        Op::Nearest,
        Op::Linear,
        Op::Gather,
        Op::CompareNearest,
        Op::CompareLinear,
    ];

    fn kernel(self) -> &'static str {
        match self {
            Op::Nearest | Op::Linear => "rrp_mc_sample",
            Op::Gather => "rrp_mc_gather",
            Op::CompareNearest | Op::CompareLinear => "rrp_mc_compare",
        }
    }

    fn filter(self) -> SamplerMinMagFilter {
        match self {
            Op::Linear | Op::CompareLinear => SamplerMinMagFilter::Linear,
            _ => SamplerMinMagFilter::Nearest,
        }
    }

    fn compares(self) -> bool {
        matches!(self, Op::CompareNearest | Op::CompareLinear)
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

/// Metal's position on an axis of `n` texels: the pixel coordinate as given, or `u n` rounded
/// to `f32` with ties away from zero.
fn position(u: f32, n: u32, pixel: bool) -> f64 {
    if pixel {
        f64::from(u)
    } else {
        f64::from(rounded_away(f64::from(u) * f64::from(n)))
    }
}

/// Tap `i` mirrored once and clamped into an axis of `n` texels.
fn mirror_clamped(i: i64, n: u32) -> usize {
    let m = if i < 0 { -1 - i } else { i };
    m.min(i64::from(n) - 1) as usize
}

/// The taps Metal reads at position `x` on an axis of `n` texels with offset `o`: each texel and
/// its weight in 256ths.
fn taps(op: Op, x: f64, n: u32, o: i64) -> Vec<(usize, i64)> {
    // The base level's sizes that drop a constant offset.
    let o = if matches!(n, 3 | 5 | 6 | 7) { 0 } else { o };
    match op {
        Op::Nearest | Op::CompareNearest => vec![(mirror_clamped(x.floor() as i64 + o, n), 256)],
        Op::Linear | Op::Gather | Op::CompareLinear => {
            let q = (x * 256.0 - 127.5).floor() as i64;
            let (i, k) = (q.div_euclid(256) + o, q.rem_euclid(256));
            vec![
                (mirror_clamped(i, n), 256 - k),
                (mirror_clamped(i + 1, n), k),
            ]
        }
    }
}

/// The lanes' coordinates on `s` in texels: `c + 1/2 + k/4` for `c` from `-2n` to `2n - 1`.
fn texel_lanes(n: u32) -> Vec<f64> {
    let n = i64::from(n);
    (-2 * n..2 * n)
        .flat_map(|c| (0..4).map(move |k| c as f64 + 0.5 + k as f64 / 4.0))
        .collect()
}

/// A lane's comparison reference: its own sixteenth plus a half, below, among and above the
/// texels.
fn reference(lane: usize) -> f32 {
    ((lane % 10) as f32 + 0.5) / 16.0
}

/// Metal's four results of lane `lane` at coordinate `u` (texels or normalized) for each offset.
fn metal(op: Op, u: f32, n: u32, pixel: bool, lane: usize) -> Vec<[f32; 4]> {
    let x = position(u, n, pixel);
    OFFSETS
        .iter()
        .map(|&o| {
            let taps = taps(op, x, n, o);
            match op {
                Op::Nearest | Op::Linear => {
                    let sum: i64 = taps.iter().map(|&(t, w)| (t as i64 + 1) * w).sum();
                    [sum as f32 / 256.0, 0.0, 0.0, 1.0]
                }
                Op::Gather => {
                    let (a, b) = ((taps[0].0 + 1) as f32, (taps[1].0 + 1) as f32);
                    [a, b, b, a]
                }
                Op::CompareNearest | Op::CompareLinear => {
                    let r = f64::from(reference(lane));
                    let passes = |t: usize| i64::from(r < (t as f64 + 1.0) / 16.0);
                    let sum: i64 = taps.iter().map(|&(t, w)| passes(t) * w).sum();
                    [sum as f32 / 256.0, 0.0, 0.0, 0.0]
                }
            }
        })
        .collect()
}

fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("rrp_mirror_clamp.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// An `n` x 1 texture of `format` holding `texel(j)`'s bytes at texel `j`.
fn texture(device: &Device, format: PixelFormat, n: u32, texel: impl Fn(u32) -> f32) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, n.into(), 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    if format == PixelFormat::R32Float {
        descriptor.resource_options = ResourceOptions::shared();
    }
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..n).flat_map(|j| texel(j).to_le_bytes()).collect();
    texture
        .replace_region(
            Region::new_2d(0, 0, n.into(), 1),
            0,
            0,
            &bytes,
            u64::from(n) * 4,
            0,
        )
        .unwrap();
    texture
}

/// Each lane's five results of `op` through a sampler mirror-clamping `s` (pixel coordinates or
/// normalized ones) on `texture`, at `(u, 1/2)` for each `u` of `us`.
fn sampled(device: &Device, texture: &Texture, op: Op, pixel: bool, us: &[f32]) -> Vec<[f32; 4]> {
    let state = state(device, op.kernel());
    let queue = device.new_command_queue().unwrap();
    // The one row's centre, in texels and normalized alike.
    let y = 0.5f32;
    let coords: Vec<u8> = us
        .iter()
        .enumerate()
        .flat_map(|(lane, &u)| [u, y, reference(lane), 0.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let coords = buffer_with(device, &coords).unwrap();
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = op.filter();
    descriptor.mag_filter = op.filter();
    descriptor.s_address_mode = SamplerAddressMode::MirrorClampToEdge;
    descriptor.normalized_coordinates = !pixel;
    if op.compares() {
        descriptor.compare_function = CompareFunction::Less;
    }
    let sampler = device.new_sampler_state(&descriptor).unwrap();
    let lanes = us.len() as u64;
    let out = buffer_with(device, &vec![0xab; (lanes * 5 * 16) as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(&sampler), 0).unwrap();
    encoder.set_buffer(Some(&coords), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(lanes, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let bytes = read_buffer(&out, 0, lanes * 5 * 16).unwrap();
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| {
            let f = |i: usize| f32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]);
            [f(0), f(4), f(8), f(12)]
        })
        .collect()
}

/// The lanes decide: each size reaches below the level and past it on every op, a linear
/// weight of every quarter, and the offsets move a tap at 8 texels where they are dropped at 6.
#[test]
fn the_lanes_reach_past_the_level_both_ways() {
    for n in [6u32, 8] {
        let xs = texel_lanes(n);
        assert_eq!(xs.len() as u64 % GROUP, 0, "whole threadgroups");
        assert!(
            xs.iter().any(|&x| x < -f64::from(n))
                && xs.iter().any(|&x| x > 2.0 * f64::from(n) - 1.0)
        );
        let weights: std::collections::BTreeSet<i64> =
            xs.iter().map(|&x| taps(Op::Linear, x, n, 0)[1].1).collect();
        assert_eq!(weights.into_iter().collect::<Vec<_>>(), [0, 64, 128, 192]);
        let moved = xs
            .iter()
            .any(|&x| taps(Op::Nearest, x, n, 2) != taps(Op::Nearest, x, n, 0));
        assert_eq!(moved, n == 8, "{n}");
    }
    // A mirror once, then the clamp: -1 reads 0, -3 reads 2, and past the level the last texel.
    assert_eq!(
        [-1, -3, -9, 7, 12].map(|i| mirror_clamped(i, 6)),
        [0, 2, 5, 5, 5]
    );
}

/// Every lane reads Metal's answer, on each op, pixel and normalized, 6 and 8 texels wide, where
/// the device samples through Metal's own sampler; but a pixel coordinate reaches the device
/// divided by the size (`pixel-sampler-scaling`), so a pixel lane may read instead Metal's
/// normalized answer at a coordinate within [`REACH`] ulps of the quotient (counted apart).
/// Elsewhere the device samples by its own sampler (announced `device-sampling`: its addressing,
/// offsets and filter), whose differences are printed. Prints, per case, the lanes checked and
/// those that differ, and each lane that differs.
#[test]
fn mirror_clamped_samples_read_metal_s_texels() {
    on_device_with(at_plan, |device, messages| {
        let mut wrong = Vec::new();
        for n in [6u32, 8] {
            let float = texture(device, PixelFormat::R32Float, n, |j| j as f32 + 1.0);
            let depth = texture(device, PixelFormat::Depth32Float, n, |j| {
                (j as f32 + 1.0) / 16.0
            });
            let xs = texel_lanes(n);
            for pixel in [true, false] {
                let us: Vec<f32> = xs
                    .iter()
                    .map(|&x| {
                        if pixel {
                            x as f32
                        } else {
                            (x / f64::from(n)) as f32
                        }
                    })
                    .collect();
                for op in Op::ALL {
                    let texture = if op.compares() { &depth } else { &float };
                    let got = sampled(device, texture, op, pixel, &us);
                    let (mut differ, mut scaled) = (0usize, 0usize);
                    for (lane, &u) in us.iter().enumerate() {
                        let want = metal(op, u, n, pixel, lane);
                        for (k, want) in want.iter().enumerate() {
                            let have = got[lane * 5 + k];
                            let same = |want: &[f32; 4]| {
                                (0..4).all(|c| have[c].to_bits() == want[c].to_bits())
                            };
                            let near = || {
                                (-REACH..=REACH).any(|ulps| {
                                    let v = shifted(u / n as f32, ulps);
                                    same(&metal(op, v, n, false, lane)[k])
                                })
                            };
                            if !same(want) && pixel && near() {
                                scaled += 1;
                            } else if !same(want) {
                                differ += 1;
                                wrong.push(format!(
                                    "{n} pixel {pixel} {op:?} u {u:e} offset {}: read {have:?}, \
                                     Metal {want:?}",
                                    OFFSETS[k]
                                ));
                            }
                        }
                    }
                    println!(
                        "MIRROR-CLAMP {} {n} pixel {pixel} {op:?}: {} results, {differ} differ \
                         from Metal, {scaled} Metal's at the scaled coordinate's neighbour",
                        device.name(),
                        us.len() * 5
                    );
                }
            }
        }
        for line in wrong.iter().take(80) {
            println!("MIRROR-CLAMP-DIFFERS {}: {line}", device.name());
        }
        if own_sampler(device, messages) {
            println!(
                "MIRROR-CLAMP {}: the device's own sampler, {} results differ from Metal",
                device.name(),
                wrong.len()
            );
            return;
        }
        assert!(
            wrong.is_empty(),
            "{} results differ from Metal",
            wrong.len()
        );
    });
}
