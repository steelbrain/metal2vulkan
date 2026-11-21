//! The shuffle-and-fill moduli Metal answers by whether they are constants, end to end.
//!
//! The owned fixture `engine/gsm2_fill_moduli` (Apple's compiler's AIR of our own MSL) goes
//! through the whole pipeline and its kernels run on the CPU interpreter in every kernel tier: the
//! `Direct` tier at 32 lanes on `moltenvk-m3`, the `Cluster32` tier at 64 lanes on a wave64-only
//! device, and the `Emulated` tier on `lavapipe` at 8, 16, 32 and 64 lanes, in both scheduling
//! orders. Every word is checked against Metal's rule, measured on the M3 Max and the M4 for every
//! delta below 9:
//! - a constant modulo fills the simdgroup in segments of the next power of two of it, at least 2
//!   (past 32, of the power whose exponent is that one's modulo 8), in either scope;
//! - a runtime modulo moves the lanes only when it is a power of two from 2 to the group's width
//!   (2 and 4 for a quad, up to 32 for a simdgroup), and otherwise leaves each lane its own data.
//!
//! The kernels run whole simdgroups: a segment of 8 to 32 lanes reads past the quad. No GPU is
//! used.
//!
//! `llvm-as` assembles the fixture; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::device::subgroup::Tier;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::lower::select::Plan;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, profile, typed_module, AssembleError};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

/// The float pattern of the fixture's cases: thread `t` reads entry `7t mod 32`.
const P: [f64; 32] = [
    2.0, -1.0, 0.5, 1.0, -2.0, 1.0, 1.0, -0.5, 2.0, 1.0, -1.0, 0.5, 1.0, 2.0, -1.0, 1.0, 0.5, -2.0,
    1.0, 1.0, 2.0, -0.5, 1.0, -1.0, 2.0, 1.0, 0.5, 1.0, -2.0, 1.0, 1.0, -1.0,
];

/// Buffer 2's (delta, modulo) pairs of the fixture's runtime cases.
const DM_A: [u32; 8] = [1, 3, 2, 8, 1, 0, 3, 16];
const DM_B: [u32; 8] = [2, 5, 1, 32, 4, 1, 2, 4];

fn xf(t: u32) -> f64 {
    P[(t as usize * 7) % 32]
}

/// Metal's segment width for the constant modulo `m`.
fn width(m: u32) -> u32 {
    let k = 32 - (m.wrapping_sub(1) & 0xFFFF).leading_zeros();
    match k % 8 {
        e @ 1..=5 => 1 << e,
        _ => 2,
    }
}

/// Metal's segment width for the runtime modulo `m` of a group of `group` lanes, or `None` where
/// each lane keeps its own data.
fn runtime(m: u32, group: u32) -> Option<u32> {
    (m.is_power_of_two() && (2..=group).contains(&m)).then_some(m)
}

/// The fill of thread `t` by `d` in segments of `w` lanes of its simdgroup (its own data where
/// `w` is `None`): the data `in[t]`, the filling data `in[t] + 1000`.
fn fill(t: u32, up: bool, d: u32, w: Option<u32>) -> f64 {
    let Some(w) = w else {
        return xf(t);
    };
    let (sg, ln) = (t & !31, t & 31);
    let l = ln % w;
    let seg = ln - l;
    let (inside, src) = if up {
        (
            l >= d,
            if l >= d {
                ln - d
            } else {
                seg + (l + w - d % w) % w
            },
        )
    } else {
        (
            l + d < w,
            if l + d < w { ln + d } else { seg + (l + d) % w },
        )
    };
    let u = sg + src;
    if inside {
        xf(u)
    } else {
        xf(u) + 1000.0
    }
}

fn f32s(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect()
}

/// The expected output bytes of `entry` over `local` threads, with buffer 2's pairs `dm`.
fn expected(entry: &str, local: u32, dm: &[u32; 8]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in 0..local {
        let c = |up, d, m| fill(t, up, d, Some(width(m)));
        match entry {
            "fill_constant_moduli" => out.extend(f32s(&[
                c(true, 1, 3),
                c(false, 2, 3),
                c(true, 3, 5),
                c(false, 1, 5),
                c(true, 1, 33),
                c(false, 1, 0),
                c(true, 1, 3),
                c(false, 2, 5),
                c(true, 5, 33),
                c(false, 3, 20),
                c(true, 2, 1000),
                c(false, 6, 12),
            ])),
            "fill_runtime_moduli" => {
                let quad = |up, d, m| fill(t, up, d, runtime(m, 4));
                let simd = |up, d, m| fill(t, up, d, runtime(m, 32));
                out.extend(f32s(&[
                    quad(true, dm[0], dm[1]),
                    simd(true, dm[0], dm[1]),
                    quad(false, dm[2], dm[3]),
                    simd(false, dm[2], dm[3]),
                    quad(true, dm[4], dm[5]),
                    simd(false, dm[4], dm[5]),
                    quad(false, dm[6], dm[7]),
                    simd(true, dm[6], dm[7]),
                ]));
            }
            other => panic!("no expectation for {other}"),
        }
    }
    out
}

/// Binding 1 after one threadgroup of `local` threads ran `m` (planned as `plan`) at `lanes`,
/// identical in both scheduling orders. Binding 0 holds the inputs, binding 2 the pairs `dm`.
fn run_entry(
    entry: &str,
    plan: &Plan,
    m: &TypedModule,
    local: u32,
    lanes: SubgroupSize,
    dm: &[u32; 8],
) -> Vec<u8> {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip([local, 1, 1]) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(local));
    let bytes = [
        f32s(&(0..local).map(xf).collect::<Vec<_>>()),
        vec![0xAB; expected(entry, local, dm).len()],
        dm.iter().flat_map(|w| w.to_le_bytes()).collect(),
    ];
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

/// Both kernels under `p`, planned in `tier`, at each of `widths`, over 64 and 96 threads.
fn run_kernels(p: &DeviceProfile, tier: Tier, widths: &[SubgroupSize]) {
    let bc = match llvm_assemble(&engine_dir().join("gsm2_fill_moduli.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    };
    let kernels: [(&str, &[[u32; 8]]); 2] = [
        ("fill_constant_moduli", &[DM_A]),
        ("fill_runtime_moduli", &[DM_A, DM_B]),
    ];
    for (entry, pairs) in kernels {
        let (plan, m) = typed_module(&bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(plan.simd.1, tier, "{entry}");
        for &lanes in widths {
            for local in [64, 96] {
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

#[test]
fn fill_moduli_run_direct_at_32_lanes() {
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    run_kernels(&p, Tier::Direct, &[SubgroupSize::W32]);
}

#[test]
fn fill_moduli_run_cluster32_at_64_lanes() {
    run_kernels(
        &super::engine_simd::wave64(),
        Tier::Cluster32,
        &[SubgroupSize::W64],
    );
}

#[test]
fn fill_moduli_run_emulated_at_every_width() {
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
