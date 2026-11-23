//! Simdgroup products, prefix products and rotate up end to end.
//!
//! The owned fixture `engine/gsm2_simd_products` (Apple's compiler's AIR of our own MSL) goes
//! through the whole pipeline and its kernels run on the CPU interpreter in every kernel tier: the
//! `Direct` tier at 32 lanes on `moltenvk-m3`, the `Cluster32` tier at 64 lanes on a wave64-only
//! device, and the `Emulated` tier on `lavapipe` at 8, 16, 32 and 64 lanes, in both scheduling
//! orders. Every word is checked against the closed form: the float inputs are 1, -1, 2, -2, 0.5
//! and -0.5, so every product of any association order is exact; integer products wrap; the
//! exclusive product of a simdgroup's first active lane is 1 (Metal's, measured); rotate up by `d`
//! reads lane `(l - d) mod 32`. The products run over full simdgroups, a partial one (40
//! threads), and the non-contiguous active lanes of divergent control flow; rotate up over full
//! simdgroups, as its cases do. The committed cases' Metal rows were qualified on the M3 Max and
//! the M4. No GPU is used.
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
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

/// The float pattern of the fixture's cases: thread `t` reads entry `7t mod 32`.
const P: [f64; 32] = [
    2.0, -1.0, 0.5, 1.0, -2.0, 1.0, 1.0, -0.5, 2.0, 1.0, -1.0, 0.5, 1.0, 2.0, -1.0, 1.0, 0.5, -2.0,
    1.0, 1.0, 2.0, -0.5, 1.0, -1.0, 2.0, 1.0, 0.5, 1.0, -2.0, 1.0, 1.0, -1.0,
];

fn xf(t: u32) -> f64 {
    P[(t as usize * 7) % 32]
}

/// The integer inputs: odd, so no product is zero.
fn xi(t: u32) -> u32 {
    (2 * ((t as i32 * 37 + 11) % 257) - 255) as u32
}

/// The threads of `t`'s simdgroup in a threadgroup of `local` threads.
fn simdgroup(t: u32, local: u32) -> Range<u32> {
    let b = t & !31;
    b..(b + 32).min(local)
}

fn fprod(lanes: impl Iterator<Item = u32>, f: impl Fn(u32) -> f64) -> f64 {
    lanes.map(f).product()
}

fn iprod(lanes: impl Iterator<Item = u32>, f: impl Fn(u32) -> u32) -> u32 {
    lanes.fold(1u32, |a, u| a.wrapping_mul(f(u)))
}

fn f32s(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect()
}

/// The expected output bytes of `entry` over `local` threads.
fn expected(entry: &str, local: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for t in 0..local {
        let g = simdgroup(t, local);
        let (lo, l, b) = (g.start, t & 31, t & !31);
        match entry {
            "simd_float_products" => out.extend(f32s(&[
                fprod(g.clone(), xf),
                fprod(lo..t + 1, xf),
                fprod(lo..t, xf),
                fprod(g, |u| -xf(u)),
            ])),
            "simd_half_products" => {
                for v in [
                    fprod(g, xf),
                    fprod(lo..t + 1, xf),
                    fprod(lo..t, xf),
                    fprod(lo..t + 1, |u| -xf(u)),
                ] {
                    out.extend(f64_to_f16(v).to_le_bytes());
                }
            }
            "simd_int_products" => {
                let words = [
                    iprod(g.clone(), xi),
                    iprod(lo..t + 1, xi),
                    iprod(lo..t, xi),
                    iprod(g.clone(), |u| xi(u).wrapping_mul(3)),
                    iprod(g, xi) & 0xFFFF,
                    iprod(lo..t + 1, xi) & 0xFF,
                    iprod(lo..t, xi) as u16 as i16 as i32 as u32,
                ];
                out.extend(words.iter().flat_map(|w| w.to_le_bytes()));
            }
            "simd_products_divergent" => {
                let active = |u: &u32| !u.is_multiple_of(3);
                out.extend(f32s(&if active(&t) {
                    [
                        fprod(g.filter(active), xf),
                        fprod((lo..t).filter(active), xf),
                    ]
                } else {
                    [-7.0, -7.0]
                }));
            }
            "simd_rotate_up" => {
                let from = |d: u32| b + ((l + 32 - d) & 31);
                out.extend(f32s(&[xf(from(3)), xf(from(3)), -xf(from(31))]));
                out.extend(from(1).to_le_bytes());
            }
            other => panic!("no expectation for {other}"),
        }
    }
    out
}

/// The input bytes of `entry` for `local` threads.
fn input(entry: &str, local: u32) -> Vec<u8> {
    match entry {
        "simd_int_products" => (0..local).flat_map(|t| xi(t).to_le_bytes()).collect(),
        "simd_half_products" => (0..local)
            .flat_map(|t| f64_to_f16(xf(t)).to_le_bytes())
            .collect(),
        _ => f32s(&(0..local).map(xf).collect::<Vec<_>>()),
    }
}

/// Binding 1 after one threadgroup of `local` threads ran `m` (planned as `plan`) at `lanes`,
/// identical in both scheduling orders. Binding 0 holds the entry's input, binding 2 (rotate up's)
/// the delta 35.
fn run_entry(
    entry: &str,
    plan: &Plan,
    m: &TypedModule,
    local: u32,
    lanes: SubgroupSize,
) -> Vec<u8> {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([local, 1, 1]));
    let mut bytes = vec![
        input(entry, local),
        vec![0xAB; expected(entry, local).len()],
    ];
    if entry == "simd_rotate_up" {
        bytes.push(35u32.to_le_bytes().to_vec());
    }
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

const PRODUCTS: [&str; 4] = [
    "simd_float_products",
    "simd_half_products",
    "simd_int_products",
    "simd_products_divergent",
];

/// Every kernel of the fixture under `p`, planned in `tier`, at each of `widths`: the products
/// over 64, 96 and 40 threads, rotate up over 64 and 96.
fn run_kernels(p: &DeviceProfile, tier: Tier, widths: &[SubgroupSize]) {
    let bc = match llvm_assemble(&engine_dir().join("gsm2_simd_products.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    };
    let entries = PRODUCTS
        .iter()
        .map(|&e| (e, &[64, 96, 40][..]))
        .chain([("simd_rotate_up", &[64, 96][..])]);
    for (entry, locals) in entries {
        let (plan, m) = typed_module(&bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(plan.simd.1, tier, "{entry}");
        for &lanes in widths {
            for &local in locals {
                let got = run_entry(entry, &plan, &m, local, lanes);
                assert_eq!(
                    got,
                    expected(entry, local),
                    "{entry} {local} threads at {lanes:?}"
                );
            }
        }
    }
}

#[test]
fn simd_products_and_rotate_up_run_direct_at_32_lanes() {
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    run_kernels(&p, Tier::Direct, &[SubgroupSize::W32]);
}

/// Each 64-lane subgroup holds two Metal simdgroups: every product and rotation stays in its own
/// 32-lane cluster.
#[test]
fn simd_products_and_rotate_up_run_cluster32_at_64_lanes() {
    run_kernels(
        &super::engine_simd::wave64(),
        Tier::Cluster32,
        &[SubgroupSize::W64],
    );
}

/// `lavapipe` cannot pin 32 lanes: the simdgroups are emulated through workgroup memory, a tree
/// fold and scan with 1 as the identity.
#[test]
fn simd_products_and_rotate_up_run_emulated_at_every_width() {
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
