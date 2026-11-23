//! The quad group operations end to end.
//!
//! The owned fixture `engine/gsm2_quad_ops` (Apple's compiler's AIR of our own MSL) goes
//! through the whole pipeline and its kernels run on the CPU interpreter in every kernel tier: the
//! `Direct` tier at 32 lanes on `moltenvk-m3`, the `Cluster32` tier at 64 lanes on a wave64-only
//! device, and the `Emulated` tier on `lavapipe` at 8, 16, 32 and 64 lanes, in both scheduling
//! orders. Every word is checked against Metal's answers as measured on the M3 Max and the M4:
//! - a quad is four aligned lanes; quad lane `q` of thread `t` is `t & 3`;
//! - a quad fill is the simdgroup fill within the quad for a modulo of 2 or 4, the only moduli
//!   the cases use (the others: `tests::engine_fill_moduli`); rotate up by `d` reads quad lane
//!   `(q - d) & 3`;
//! - the prefix scans run in lane order over the active lanes below (and at) the lane; an
//!   exclusive float sum with none is -0, an exclusive product 1;
//! - a reduction combines `(x0 op x1) op (x2 op x3)`, an inactive or missing lane the identity
//!   (checked with inputs whose float sum each other order rounds differently);
//! - `quad_broadcast_first` reads the quad's first active lane. The fixture's arm that Metal's
//!   backend speculates (`quad_first_in_a_speculated_arm`) runs in every lane of its branch's
//!   block, as Metal runs it (`air::canon::speculate`): it reads the quad's lane 0 whether that
//!   lane takes the arm or not.
//!
//! The float inputs are 1, -1, 2, -2, 0.5 and -0.5, so every sum and product of any association
//! order is exact; integer products wrap. Fills and rotations run over full quads, the rest also
//! over a threadgroup of 42, whose last quad holds two threads. No GPU is used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::device::subgroup::Tier;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::lower::arith::legal::f64_to_f16;
use crate::lower::select::Plan;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, profile, typed_module, AssembleError};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

/// The float pattern of the fixture's cases: thread `t` reads entry `7t mod 32`.
const P: [f64; 32] = [
    2.0, -1.0, 0.5, 1.0, -2.0, 1.0, 1.0, -0.5, 2.0, 1.0, -1.0, 0.5, 1.0, 2.0, -1.0, 1.0, 0.5, -2.0,
    1.0, 1.0, 2.0, -0.5, 1.0, -1.0, 2.0, 1.0, 0.5, 1.0, -2.0, 1.0, 1.0, -1.0,
];

/// Buffer 2's (delta, modulo) pairs of the fixture's cases.
const DM_A: [u32; 8] = [1, 2, 3, 4, 5, 4, 1, 4];
const DM_B: [u32; 8] = [0, 2, 2, 4, 6, 2, 2, 2];

fn xf(t: u32) -> f64 {
    P[(t as usize * 7) % 32]
}

/// The integer inputs: odd, so no product is zero.
fn xi(t: u32) -> u32 {
    (2 * ((t as i32 * 37 + 11) % 257) - 255) as u32
}

/// The threads of `t`'s quad in a threadgroup of `local` threads.
fn quad(t: u32, local: u32) -> Range<u32> {
    let b = t & !3;
    b..(b + 4).min(local)
}

/// The quad lane a fill of quad lane `q` by `d` in segments of `modulo` (2 or 4) reads, and
/// whether it reads the data (not the filling data) there.
fn fill_source(up: bool, q: u32, d: u32, modulo: u32) -> (u32, bool) {
    assert!(modulo == 2 || modulo == 4, "modulo {modulo}");
    let l = q % modulo;
    let seg = q - l;
    if up {
        (seg + (l + modulo - d % modulo) % modulo, l >= d)
    } else {
        (seg + (l + d) % modulo, l + d < modulo)
    }
}

/// A fill of `data` with `filling` by `d` modulo `modulo` at thread `t`.
fn fill<T>(
    t: u32,
    up: bool,
    d: u32,
    modulo: u32,
    data: impl Fn(u32) -> T,
    filling: impl Fn(u32) -> T,
) -> T {
    let (src, inside) = fill_source(up, t & 3, d, modulo);
    let u = (t & !3) + src;
    if inside {
        data(u)
    } else {
        filling(u)
    }
}

/// Rotate up by `d` at thread `t`: quad lane `(q - d) & 3`.
fn rotated(t: u32, d: u32) -> u32 {
    (t & !3) + ((t & 3).wrapping_sub(d) & 3)
}

fn fsum(lanes: impl Iterator<Item = u32>, f: impl Fn(u32) -> f64) -> f64 {
    lanes.fold(-0.0, |a, u| a + f(u))
}

fn fprod(lanes: impl Iterator<Item = u32>, f: impl Fn(u32) -> f64) -> f64 {
    lanes.map(f).product()
}

fn isum(lanes: impl Iterator<Item = u32>, f: impl Fn(u32) -> u32) -> u32 {
    lanes.fold(0u32, |a, u| a.wrapping_add(f(u)))
}

fn iprod(lanes: impl Iterator<Item = u32>, f: impl Fn(u32) -> u32) -> u32 {
    lanes.fold(1u32, |a, u| a.wrapping_mul(f(u)))
}

fn f32s(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect()
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Whether thread `t` takes `quad_ops_divergent`'s arm: quad `k` runs the lanes of mask `k mod 16`.
fn masked(t: u32) -> bool {
    (((t / 4) % 16) >> (t & 3)) & 1 == 1
}

/// The expected output bytes of `entry` over `local` threads, with buffer 2's pairs `dm`.
fn expected(entry: &str, local: u32, dm: &[u32; 8]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in 0..local {
        let g = quad(t, local);
        let b = g.start;
        let filled = |u: u32| xf(u) + 1000.0;
        match entry {
            "quad_fills" => out.extend(f32s(&[
                fill(t, true, 1, 4, xf, filled),
                fill(t, false, 2, 4, xf, filled),
                fill(t, true, 1, 2, xf, filled),
                fill(t, false, 3, 4, xf, filled),
                fill(t, true, dm[0], dm[1], xf, filled),
                fill(t, false, dm[2], dm[3], xf, filled),
                fill(t, true, dm[4], dm[5], xf, filled),
                fill(t, false, dm[6], dm[7], xf, filled),
                fill(t, false, dm[0], 4, xf, filled),
                xf(rotated(t, 1)),
                xf(rotated(t, dm[2])),
                fill(t, true, 2, 4, filled, xf),
            ])),
            "quad_pair_fills" => out.extend(f32s(&[
                fill(t, true, 1, 2, xf, filled),
                fill(t, false, 1, 2, xf, filled),
                fill(t, true, 2, 2, xf, filled),
                fill(t, false, 2, 2, xf, filled),
            ])),
            "quad_typed_fills" => {
                let half = |v: f64| u32::from(f64_to_f16(v));
                let ix = |u: u32| xi(u) as i32;
                out.extend(u32s(&[
                    fill(t, true, dm[0], dm[1], ix, |u| ix(u) + 1000) as u32,
                    fill(t, false, 1, 4, |u| ix(u) as i16, |_| 7) as i32 as u32,
                    half(fill(t, true, 2, 4, |u| f64::from(ix(u)), |_| -1.0))
                        | (half(f64::from(ix(rotated(t, 3)))) << 16),
                    u32::from(fill(t, false, 1, 2, |u| xi(u) as u8, |_| 9)),
                ]));
            }
            "quad_bitwise" => out.extend(u32s(&[
                g.clone().fold(!0, |a, u| a & xi(u)),
                g.clone().fold(0, |a, u| a | xi(u)),
                g.clone().fold(0, |a, u| a ^ xi(u)),
                g.clone().fold(0, |a, u| a | xi(u)) & 0xFFFF,
                g.clone().fold(0, |a, u| a ^ xi(u)) & 0xFF,
                xi(b),
            ])),
            "quad_float_scans" => out.extend(f32s(&[
                fprod(g.clone(), xf),
                fsum(g, xf),
                fsum(b..t + 1, xf),
                fsum(b..t, xf),
                fprod(b..t + 1, xf),
                fprod(b..t, xf),
                xf(b),
                fsum(b..t, |u| -xf(u)),
            ])),
            "quad_half_scans" => {
                for v in [
                    fprod(g.clone(), xf),
                    fsum(b..t + 1, xf),
                    fsum(b..t, xf),
                    fprod(b..t + 1, xf),
                    fprod(b..t, xf),
                    fprod(g, |u| -xf(u)),
                ] {
                    out.extend(f64_to_f16(v).to_le_bytes());
                }
            }
            "quad_int_scans" => out.extend(u32s(&[
                iprod(g.clone(), xi),
                isum(b..t + 1, xi),
                isum(b..t, xi),
                iprod(b..t + 1, xi),
                iprod(b..t, |u| xi(u).wrapping_mul(3)),
                iprod(g, xi) & 0xFFFF,
                isum(b..t, xi) as u16 as i16 as i32 as u32,
            ])),
            "quad_ops_divergent" => {
                if masked(t) {
                    let on = || g.clone().filter(|&u| masked(u));
                    let first = on().next().expect("the calling lane is active");
                    out.extend(f32s(&[
                        xf(first),
                        fprod(on(), xf),
                        fsum((b..t).filter(|&u| masked(u)), xf),
                        fprod((b..t + 1).filter(|&u| masked(u)), xf),
                    ]));
                    out.extend(u32s(&[
                        on().fold(0, |a, u| a | (u << 8)) | 0x4000_0000,
                        on().fold(0, |a, u| a ^ u) | 0x4000_0000,
                    ]));
                } else {
                    out.extend(f32s(&[-7.0; 6]));
                }
            }
            "quad_first_in_a_speculated_arm" => {
                let on = |u: &u32| !u.is_multiple_of(3);
                out.extend(u32s(&[if on(&t) {
                    g.clone().next().expect("the calling lane's quad") * 10 + 1
                } else {
                    7777
                }]));
            }
            other => panic!("no expectation for {other}"),
        }
    }
    out
}

/// The input bytes of `entry` for `local` threads.
fn input(entry: &str, local: u32) -> Vec<u8> {
    match entry {
        "quad_typed_fills" | "quad_bitwise" | "quad_int_scans" => {
            (0..local).flat_map(|t| xi(t).to_le_bytes()).collect()
        }
        "quad_half_scans" => (0..local)
            .flat_map(|t| f64_to_f16(xf(t)).to_le_bytes())
            .collect(),
        _ => f32s(&(0..local).map(xf).collect::<Vec<_>>()),
    }
}

/// Binding 1 after one threadgroup of `local` threads ran `m` (planned as `plan`) at `lanes`,
/// identical in both scheduling orders. Binding 0 holds the entry's input, binding 2 the pairs
/// `dm`.
fn run_entry(
    entry: &str,
    plan: &Plan,
    m: &TypedModule,
    local: u32,
    lanes: SubgroupSize,
    dm: &[u32; 8],
) -> Vec<u8> {
    let bytes = [
        input(entry, local),
        vec![0xAB; expected(entry, local, dm).len()],
        u32s(dm),
    ];
    run_buffers(entry, plan, m, local, lanes, bytes)
}

/// Binding 1 after one threadgroup of `local` threads ran `m` (planned as `plan`) at `lanes` over
/// bindings 0, 1 and 2 holding `bytes`, identical in both scheduling orders.
fn run_buffers(
    entry: &str,
    plan: &Plan,
    m: &TypedModule,
    local: u32,
    lanes: SubgroupSize,
    bytes: [Vec<u8>; 3],
) -> Vec<u8> {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([local, 1, 1]));
    let res = Resources {
        buffers: bytes
            .into_iter()
            .enumerate()
            .map(|(i, bytes)| Buffer {
                slot: Some(ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Buffer,
                    index: i as u32,
                }),
                address: None,
                bytes,
            })
            .collect(),
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, 1, 1],
            subgroup: lanes,
            order,
            ..Config::default()
        };
        outs.push(run(m, &cfg, res.clone()).expect("run").buffers[1].clone());
    }
    assert_eq!(outs[0], outs[1], "{entry}: order independent");
    outs.swap_remove(0)
}

/// Each kernel of the fixture, the threadgroup sizes it runs at, and buffer 2's pairs: the fills
/// and rotations over full quads only (a shuffle from a lane a quad lacks is undefined in Metal),
/// the rest also over 42 threads.
const KERNELS: [(&str, &[u32], &[[u32; 8]]); 9] = [
    ("quad_fills", &[64, 96], &[DM_A, DM_B]),
    ("quad_pair_fills", &[64, 42], &[DM_A]),
    ("quad_typed_fills", &[64], &[DM_A]),
    ("quad_bitwise", &[64, 42], &[DM_A]),
    ("quad_float_scans", &[64, 42], &[DM_A]),
    ("quad_half_scans", &[64, 42], &[DM_A]),
    ("quad_int_scans", &[64, 42], &[DM_A]),
    ("quad_ops_divergent", &[64, 42], &[DM_A]),
    ("quad_first_in_a_speculated_arm", &[64], &[DM_A]),
];

/// The fixture's bitcode, or `None` (reported) where `llvm-as` is missing.
fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("gsm2_quad_ops.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    }
}

/// Every kernel of the fixture under `p`, planned in `tier`, at each of `widths`, and
/// [`reductions_in_metal_s_order`].
fn run_kernels(p: &DeviceProfile, tier: Tier, widths: &[SubgroupSize]) {
    let Some(bc) = fixture() else {
        return;
    };
    reductions_in_metal_s_order(&bc, p, tier, widths);
    for (entry, locals, pairs) in KERNELS {
        let (plan, m) = typed_module(&bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(plan.simd.1, tier, "{entry}");
        for &lanes in widths {
            for &local in locals {
                for dm in pairs {
                    let got = run_entry(entry, &plan, &m, local, lanes, dm);
                    assert_eq!(
                        got,
                        expected(entry, local, dm),
                        "{entry} {local} threads at {lanes:?}, pairs {dm:?}"
                    );
                }
            }
        }
    }
}

/// Thread `t`'s input of [`reductions_in_metal_s_order`]: each quad holds `1, 2^24, 1, -2^24`,
/// whose float sum is 1 as `(x0 + x1) + (x2 + x3)`, 0 in lane order and 2 as
/// `((x0 + x2) + x1) + x3`.
fn ordered(t: u32) -> f32 {
    [1.0, 16_777_216.0, 1.0, -16_777_216.0][(t & 3) as usize]
}

/// `(x0 op x1) op (x2 op x3)` over a quad's lanes, a missing lane the identity.
fn tree(xs: [Option<f32>; 4], op: fn(f32, f32) -> f32) -> f32 {
    let pair = |a: Option<f32>, b: Option<f32>| match (a, b) {
        (Some(a), Some(b)) => Some(op(a, b)),
        (a, None) => a,
        (None, b) => b,
    };
    pair(pair(xs[0], xs[1]), pair(xs[2], xs[3])).expect("the calling lane is in its quad")
}

/// `quad_float_scans`'s `quad_product` and `quad_sum` (words 0 and 1 of each thread) over
/// [`ordered`] inputs, over full quads and a threadgroup of 42 whose last quad holds two threads:
/// Metal's `(x0 + x1) + (x2 + x3)` (measured), bit for bit; so the module records no
/// `ReductionOrder`.
fn reductions_in_metal_s_order(bc: &[u8], p: &DeviceProfile, tier: Tier, widths: &[SubgroupSize]) {
    let entry = "quad_float_scans";
    let (plan, m) = typed_module(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    assert_eq!(plan.simd.1, tier, "{entry}");
    assert!(
        !plan.precision.contains(&PrecisionNote::ReductionOrder),
        "{entry}: {:?}",
        plan.precision
    );
    for &lanes in widths {
        for local in [64, 42] {
            let input = (0..local).flat_map(|t| ordered(t).to_le_bytes()).collect();
            let bytes = [input, vec![0xAB; 32 * local as usize], u32s(&DM_A)];
            let got = run_buffers(entry, &plan, &m, local, lanes, bytes);
            for t in 0..local {
                let xs = std::array::from_fn(|j| {
                    let u = (t & !3) + j as u32;
                    (u < local).then_some(ordered(u))
                });
                let word = |i: usize| {
                    let at = 32 * t as usize + 4 * i;
                    f32::from_le_bytes(got[at..at + 4].try_into().expect("a word"))
                };
                let want = [tree(xs, |a, b| a * b), tree(xs, |a, b| a + b)];
                for (i, w) in want.into_iter().enumerate() {
                    assert_eq!(
                        word(i).to_bits(),
                        w.to_bits(),
                        "{entry} word {i} of thread {t} of {local} at {lanes:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn quad_operations_run_direct_at_32_lanes() {
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    run_kernels(&p, Tier::Direct, &[SubgroupSize::W32]);
}

/// Each 64-lane subgroup holds two Metal simdgroups; a quad is four aligned lanes in either.
#[test]
fn quad_operations_run_cluster32_at_64_lanes() {
    run_kernels(
        &super::engine_simd::wave64(),
        Tier::Cluster32,
        &[SubgroupSize::W64],
    );
}

/// `lavapipe` cannot pin 32 lanes: the quads are emulated through workgroup memory.
#[test]
fn quad_operations_run_emulated_at_every_width() {
    let p = profile("lavapipe", DevicePolicy::RESIDENT_TABLES).expect("profile");
    run_kernels(
        &p,
        Tier::Emulated,
        &[
            SubgroupSize::W8,
            SubgroupSize::W16,
            SubgroupSize::W32,
            SubgroupSize::W64,
        ],
    );
}
