//! Samples and gathers with a constant texel offset on a repeating or mirrored axis of a view
//! whose first level lies above its parent's base, on the device the cfg selects, under the
//! validation layer.
//!
//! Metal's rule: a constant offset on a repeating or mirrored axis is dropped where the sampled
//! level's size lies in `[2, 7]` and the base-level size of the texture the view was made from is
//! not a power of two (a view of one level reads as its parent's level). A view of
//! level 2 of an 18-texel parent is 4 texels wide, a power of two: a rule read from the view's own
//! base would keep the offset Metal drops.
//!
//! The owned fixture `rrp_mirror_clamp` samples (`rrp_mc_sample`) or gathers (`rrp_mc_gather`)
//! through the runtime sampler at `[[sampler(0)]]` at each lane's coordinate, without an offset
//! and with the offsets -3, 2, -7 and 5 on `s`. Here square parents with full mip chains, each
//! level's texels distinct, and views of some of their levels ([`VIEWS`]): 4 and 2 texels wide
//! under 18 and 9 (the offset dropped by the parent's base where the view's own would keep it),
//! 4 under 16 (kept), and 8 under 17 with levels of 4 and 2 below it (kept: the sampled level is
//! 8 wide). The sampler repeats or mirror-repeats `s` and clamps `t` to the edge, normalized;
//! lanes at `c + 1/2 + k/4` texels on `s` for `c` from `-2n` to `2n - 1` and `k` from 0 to 3, on
//! row 0's centre, so every answer is exact.
//!
//! A nearest or linear sample and a gather of an `R32Float` view are held to Metal's taps (as in
//! `mirror_clamp`: the position `u n`, a nearest tap `floor(x)`, a linear pair and a gather's at
//! `q = floor(256 x - 127.5)`, weighted in 256ths; each tap moved by the offset the rule keeps and
//! then wrapped into the level). A device without Metal's own sampler emulates these accesses and
//! reads the rule's parent bit (`SpecKey::OffsetBaseAxes`); on MoltenVK the view is Metal's own
//! texture view of the image's texture and the sample Metal's. A bicubic sample (a runtime
//! sampler whose filters are `Cubic`, which the translation emulates on every device, MoltenVK
//! included) of an `RGBA8Unorm` view under `repeat` (an offset on a mirrored bicubic axis is
//! refused) is held to the device's own sample without an offset at the coordinate the rule places:
//! the lane's own where the offset is dropped, moved by the offset where it is kept.

use super::degrade_notices::on_device_with;
use super::pixel_samplers::own_sampler;
use super::{at_plan, buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerMinMagFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region, Size, Texture,
    TextureDescriptor,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The fixture's texel offsets on `s`, one per result of a lane.
const OFFSETS: [i64; 5] = [0, -3, 2, -7, 5];

/// The lanes of a threadgroup.
const GROUP: u64 = 32;

/// A view of `levels` levels from `level` on of a square parent `parent` texels wide with a full
/// mip chain.
#[derive(Clone, Copy, Debug)]
struct View {
    parent: u32,
    level: u32,
    levels: u32,
}

/// The views sampled: the offset dropped by the parent's base where the view's own base would keep
/// it (the first three), kept under a parent whose base is a power of two, and kept on a level 8
/// wide whose view holds levels of 4 and 2 (the binder's bit set, the rule's size not small).
const VIEWS: [View; 5] = [
    View {
        parent: 18,
        level: 2,
        levels: 1,
    },
    View {
        parent: 18,
        level: 2,
        levels: 3,
    },
    View {
        parent: 9,
        level: 2,
        levels: 1,
    },
    View {
        parent: 16,
        level: 2,
        levels: 1,
    },
    View {
        parent: 17,
        level: 1,
        levels: 3,
    },
];

impl View {
    /// The view's first level's size on each axis.
    fn n(self) -> u32 {
        (self.parent >> self.level).max(1)
    }

    /// Whether Metal drops an offset on a wrapped axis of the view's first level: the level is 2
    /// to 7 texels wide and the parent's base-level size is not a power of two.
    fn dropped(self) -> bool {
        (2..=7).contains(&self.n()) && !self.parent.is_power_of_two()
    }

    /// The rule read from the view's own base-level size instead, since replaced.
    fn dropped_by_own_base(self) -> bool {
        (2..=7).contains(&self.n()) && !self.n().is_power_of_two()
    }
}

/// What a lane reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Nearest,
    Linear,
    Gather,
    Bicubic,
}

impl Op {
    const ALL: [Op; 4] = [Op::Nearest, Op::Linear, Op::Gather, Op::Bicubic];

    fn kernel(self) -> &'static str {
        match self {
            Op::Gather => "rrp_mc_gather",
            _ => "rrp_mc_sample",
        }
    }

    fn filter(self) -> SamplerMinMagFilter {
        match self {
            Op::Nearest | Op::Gather => SamplerMinMagFilter::Nearest,
            Op::Linear => SamplerMinMagFilter::Linear,
            Op::Bicubic => SamplerMinMagFilter::Cubic,
        }
    }

    /// The address modes of `s` measured: an offset on a mirrored bicubic axis is refused.
    fn modes(self) -> &'static [SamplerAddressMode] {
        match self {
            Op::Bicubic => &[SamplerAddressMode::Repeat],
            _ => &[SamplerAddressMode::Repeat, SamplerAddressMode::MirrorRepeat],
        }
    }

    fn format(self) -> PixelFormat {
        match self {
            Op::Bicubic => PixelFormat::RGBA8Unorm,
            _ => PixelFormat::R32Float,
        }
    }
}

/// Tap `i` wrapped into an axis of `n` texels under `mode`.
fn wrapped(i: i64, n: u32, mode: SamplerAddressMode) -> usize {
    let n = i64::from(n);
    let i = match mode {
        SamplerAddressMode::Repeat => i.rem_euclid(n),
        _ => {
            let m = i.rem_euclid(2 * n);
            if m < n {
                m
            } else {
                2 * n - 1 - m
            }
        }
    };
    i as usize
}

/// The taps Metal reads on `s` at position `x` with the offset `o` placed: each texel and its
/// weight in 256ths.
fn taps(op: Op, x: f64, n: u32, o: i64, mode: SamplerAddressMode) -> Vec<(usize, i64)> {
    match op {
        Op::Nearest => vec![(wrapped(x.floor() as i64 + o, n, mode), 256)],
        _ => {
            let q = (x * 256.0 - 127.5).floor() as i64;
            let (i, k) = (q.div_euclid(256) + o, q.rem_euclid(256));
            vec![(wrapped(i, n, mode), 256 - k), (wrapped(i + 1, n, mode), k)]
        }
    }
}

/// The lanes' positions on `s` in texels: `c + 1/2 + k/4` for `c` from `-2n` to `2n - 1`.
fn texel_lanes(n: u32) -> Vec<f64> {
    let n = i64::from(n);
    (-2 * n..2 * n)
        .flat_map(|c| (0..4).map(move |k| c as f64 + 0.5 + k as f64 / 4.0))
        .collect()
}

/// The lane at `c + 1/2 + k/4` texels: lane `x`'s with `c` moved by `by` and wrapped by the
/// repeating axis's period, `n`.
fn moved_lane(lane: usize, by: i64, n: u32) -> usize {
    let n = i64::from(n);
    let (c, k) = ((lane / 4) as i64 - 2 * n, lane % 4);
    ((c + by).rem_euclid(n) + 2 * n) as usize * 4 + k
}

/// The `R32Float` texel `(j, r)` of the parent's level `level`.
fn value(level: u32, j: usize, r: usize) -> f32 {
    (100 * level as usize + 16 * r + j + 1) as f32
}

/// The `RGBA8Unorm` texel `(j, r)` of the parent's level `level`: varied bytes, every channel
/// its own.
fn rgba8(level: u32, j: usize, r: usize) -> [u8; 4] {
    let seed = 131 * level as usize + 29 * j + 71 * r;
    [0, 1, 2, 3].map(|ch| ((seed + 17 * ch) * 37 % 251) as u8)
}

/// Metal's five results of a lane at `x` texels on row 0 of `view` for each offset.
fn metal(op: Op, view: View, x: f64, mode: SamplerAddressMode) -> Vec<[f32; 4]> {
    let n = view.n();
    OFFSETS
        .iter()
        .map(|&o| {
            let o = if view.dropped() { 0 } else { o };
            let taps = taps(op, x, n, o, mode);
            match op {
                Op::Nearest | Op::Linear => {
                    let sum: f64 = taps
                        .iter()
                        .map(|&(t, w)| f64::from(value(view.level, t, 0)) * w as f64)
                        .sum();
                    [(sum / 256.0) as f32, 0.0, 0.0, 1.0]
                }
                Op::Gather => {
                    let (a, b) = (taps[0].0, taps[1].0);
                    let v = |j, r| value(view.level, j, r);
                    [v(a, 1), v(b, 1), v(b, 0), v(a, 0)]
                }
                Op::Bicubic => unreachable!("the bicubic sample is held to the device's own"),
            }
        })
        .collect()
}

/// A square parent `side` texels wide of `format` with a full mip chain, each level's texels
/// `texel(level, j, r)`.
fn parent(
    device: &Device,
    format: PixelFormat,
    side: u32,
    texel: impl Fn(u32, usize, usize) -> [u8; 4],
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, side.into(), side.into(), true);
    descriptor.usage = TextureUsage::SHADER_READ;
    let texture = device.new_texture(&descriptor).unwrap();
    for level in 0..texture.mipmap_level_count() as u32 {
        let w = (side >> level).max(1) as usize;
        let bytes: Vec<u8> = (0..w)
            .flat_map(|r| (0..w).map(move |j| (j, r)))
            .flat_map(|(j, r)| texel(level, j, r))
            .collect();
        texture
            .replace_region(
                Region::new_2d(0, 0, w as u64, w as u64),
                level.into(),
                0,
                &bytes,
                w as u64 * 4,
                0,
            )
            .unwrap();
    }
    texture
}

/// The view `view` of a parent of `op`'s format.
fn view_of(device: &Device, op: Op, view: View) -> Texture {
    let parent = match op {
        Op::Bicubic => parent(device, op.format(), view.parent, rgba8),
        _ => parent(device, op.format(), view.parent, |level, j, r| {
            value(level, j, r).to_le_bytes()
        }),
    };
    let levels = u64::from(view.level)..u64::from(view.level + view.levels);
    parent
        .new_texture_view_with_levels_slices(op.format(), TextureType::Type2D, levels, 0..1)
        .unwrap()
}

/// Each lane's five results of `op` through a sampler wrapping `s` by `mode` and clamping `t`
/// on `texture`, `n` texels each way, at `(x / n, 1 / (2 n))` for each `x` of `xs`.
fn sampled(
    state: &ComputePipelineState,
    texture: &Texture,
    (op, mode): (Op, SamplerAddressMode),
    n: u32,
    xs: &[f64],
) -> Vec<[f32; 4]> {
    let device = state.device();
    let queue = device.new_command_queue().unwrap();
    let y = 0.5 / n as f32;
    let coords: Vec<u8> = xs
        .iter()
        .flat_map(|&x| [(x / f64::from(n)) as f32, y, 0.0, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let coords = buffer_with(device, &coords).unwrap();
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = op.filter();
    descriptor.mag_filter = op.filter();
    descriptor.s_address_mode = mode;
    descriptor.t_address_mode = SamplerAddressMode::ClampToEdge;
    let sampler = device.new_sampler_state(&descriptor).unwrap();
    let lanes = xs.len() as u64;
    let out = buffer_with(device, &vec![0xab; (lanes * 5 * 16) as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
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

/// The lanes decide: the parent's rule and the view's own differ on the first three views and
/// agree on the others; on each view every op reads a tap the dropped and the kept offset place
/// apart, under both modes; and a moved lane is the lane `by` texels on, wrapped.
#[test]
fn the_lanes_tell_the_parent_s_rule_from_the_view_s() {
    let differs: Vec<bool> = VIEWS
        .iter()
        .map(|v| v.dropped() != v.dropped_by_own_base())
        .collect();
    assert_eq!(differs, [true, true, true, false, false]);
    assert_eq!(VIEWS.map(|v| v.dropped()), [true, true, true, false, false]);
    for view in VIEWS {
        let xs = texel_lanes(view.n());
        assert_eq!(xs.len() as u64 % GROUP, 0, "whole threadgroups");
        for op in [Op::Nearest, Op::Linear, Op::Gather] {
            for &mode in op.modes() {
                let apart = xs.iter().any(|&x| {
                    OFFSETS[1..]
                        .iter()
                        .any(|&o| taps(op, x, view.n(), o, mode) != taps(op, x, view.n(), 0, mode))
                });
                assert!(apart, "{view:?} {op:?} {mode:?}");
            }
        }
        let n = view.n();
        for (lane, &x) in xs.iter().enumerate() {
            for by in [-7, -3, 2, 5] {
                let moved = xs[moved_lane(lane, by, n)];
                assert_eq!((moved - x - by as f64).rem_euclid(f64::from(n)), 0.0);
            }
        }
    }
    // A mirror at the level's edges, a repeat by its size.
    assert_eq!(
        [-1, -4, 4, 7, 8].map(|i| wrapped(i, 4, SamplerAddressMode::MirrorRepeat)),
        [0, 3, 3, 0, 0]
    );
    assert_eq!(
        [-1, -4, 4, 7].map(|i| wrapped(i, 4, SamplerAddressMode::Repeat)),
        [3, 0, 0, 3]
    );
}

/// Every lane of every view reads Metal's answer by the parent's rule, on each op and mode, where
/// the device samples through Metal's own sampler. Elsewhere the device's own sampler answers its
/// samples and gathers (announced `device-sampling`), whose differences are printed, and the
/// module's bicubic samples, which it computes from fetched texels, still take the parent's rule.
/// Prints, per case, the results checked and those that differ, and each result that differs.
#[test]
fn a_view_s_offset_follows_its_parent_s_base() {
    on_device_with(at_plan, |device, messages| {
        let bitcode = llvm_assemble(&engine_dir().join("rrp_mirror_clamp.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let state = |kernel: &str| {
            let function = library.new_function_with_name(kernel).unwrap();
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(StageSource::Function(
                        function.function_descriptor(),
                    )),
                    &PipelineOptions::default(),
                )
                .unwrap()
        };
        let (sample, gather) = (state("rrp_mc_sample"), state("rrp_mc_gather"));
        let mut wrong = Vec::new();
        let mut bicubic = 0usize;
        for view in VIEWS {
            let n = view.n();
            let xs = texel_lanes(n);
            for op in Op::ALL {
                let texture = view_of(device, op, view);
                let state = if op.kernel() == "rrp_mc_gather" {
                    &gather
                } else {
                    &sample
                };
                for &mode in op.modes() {
                    let got = sampled(state, &texture, (op, mode), n, &xs);
                    let mut differ = 0usize;
                    let mut apart = false;
                    for (lane, &x) in xs.iter().enumerate() {
                        let want: Vec<[f32; 4]> = match op {
                            Op::Bicubic => OFFSETS
                                .iter()
                                .map(|&o| {
                                    let kept = got[moved_lane(lane, o, n) * 5];
                                    apart |=
                                        kept.map(f32::to_bits) != got[lane * 5].map(f32::to_bits);
                                    if view.dropped() {
                                        got[lane * 5]
                                    } else {
                                        kept
                                    }
                                })
                                .collect(),
                            _ => metal(op, view, x, mode),
                        };
                        for (k, want) in want.iter().enumerate() {
                            let have = got[lane * 5 + k];
                            if have.map(f32::to_bits) != want.map(f32::to_bits) {
                                differ += 1;
                                bicubic += usize::from(op == Op::Bicubic);
                                wrong.push(format!(
                                    "{view:?} {op:?} {mode:?} x {x} offset {}: read {have:?}, \
                                     Metal {want:?}",
                                    OFFSETS[k]
                                ));
                            }
                        }
                    }
                    if op == Op::Bicubic {
                        assert!(apart, "{view:?}: no lane tells the offsets apart");
                    }
                    println!(
                        "VIEW-BASE-OFFSET {} {view:?} {op:?} {mode:?} dropped {}: {} results, \
                         {differ} differ from Metal",
                        device.name(),
                        view.dropped(),
                        xs.len() * 5
                    );
                }
            }
        }
        for line in wrong.iter().take(80) {
            println!("VIEW-BASE-OFFSET-DIFFERS {}: {line}", device.name());
        }
        if own_sampler(device, messages) {
            println!(
                "VIEW-BASE-OFFSET {}: the device's own sampler, {} results differ from Metal",
                device.name(),
                wrong.len()
            );
            assert_eq!(bicubic, 0, "bicubic results differ from the parent's rule");
            return;
        }
        assert!(
            wrong.is_empty(),
            "{} results differ from Metal",
            wrong.len()
        );
    });
}
