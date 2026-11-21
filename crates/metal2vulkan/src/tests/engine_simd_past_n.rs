//! A simdgroup or quadgroup operation that moves a value from a lane past the threadgroup's
//! thread count `N` reads all-zero bits, Metal's answer (amending the calling-lane rule).
//!
//! The owned fixtures, compiled without fast math, run every operation in workgroup-uniform
//! control flow on a threadgroup of `N` threads:
//!
//! - `engine/dsm2_shuffle_past_n` writes `simd_shuffle` (a source lane per lane),
//!   `simd_shuffle_down` (delta 8), and `simd_shuffle_xor` (mask 16) of a distinct word per
//!   thread. Metal's readings are zero for every source past `N`, whatever
//!   the inputs, the threadgroup, or the dispatch's shape; [`the_model_is_metals_reading`] checks
//!   [`model`] against two of them word for word.
//! - `engine/dsm2_shuffle_past_n_more` writes `simd_broadcast`, `simd_shuffle_rotate_down`,
//!   both `simd_shuffle_and_fill` forms, `quad_shuffle_xor`, `quad_broadcast`,
//!   `quad_shuffle_down`, `quad_shuffle_rotate_down`, and `float` and `half` shuffles: the same
//!   rule, which [`model_more`] states and the fixture's cases ask Metal.
//!
//! Both run on the emulating profiles (lavapipe, vk12-floor) and on `moltenvk-m3`'s
//! emulated-simdgroups variant, which a dispatch with a partial simdgroup runs, at 6 to 48 threads,
//! on the CPU interpreter at 8, 16, and 32 subgroup lanes in both scheduling orders; every word
//! equals the model. `moltenvk-m3`'s `Direct` tier runs the whole-simdgroup control. No GPU is
//! used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::device::subgroup::Tier;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::lower::select::Plan;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile_kept, engine_dir, link_and_validate, llvm_assemble, profile, typed_module_with,
    AssembleError,
};
use crate::pipeline::LoweringInputs;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

/// The committed profiles whose kernels' simdgroups are emulated.
const PROFILES: [&str; 2] = ["lavapipe", "runtime-floor"];

const WIDTHS: [SubgroupSize; 3] = [SubgroupSize::W8, SubgroupSize::W16, SubgroupSize::W32];

/// The poison every output word starts as.
const POISON: u32 = 0xABAB_ABAB;

fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    }
}

/// The source lanes `src[0..32]`: `(L + 16) & 31` for even `L`, `L ^ 2` for odd `L`.
fn sources() -> Vec<u32> {
    (0..32)
        .map(|l| if l % 2 == 0 { (l + 16) & 31 } else { l ^ 2 })
        .collect()
}

/// The fixtures' `src` buffer: [`sources`], then `dsm2_shuffle_past_n`'s delta 8 and mask 16, or
/// `dsm2_shuffle_past_n_more`'s broadcast lane 20, delta 8, modulo 32, quad mask 2, quad lane 3,
/// and quad delta 1.
fn src_buffer(more: bool) -> Vec<u32> {
    let mut s = sources();
    if more {
        s.extend([20, 8, 32, 2, 3, 1]);
    } else {
        s.extend([8, 16]);
    }
    s
}

/// The inputs: a distinct word per thread (input A, or input B with `other`).
fn inputs(n: u32, other: bool) -> Vec<u32> {
    (1..=n)
        .map(|k| {
            if other {
                0x3C00_0000 + 0x0002_0203 * k
            } else {
                0xA500_0000 + 0x0001_0101 * k
            }
        })
        .collect()
}

/// The word of the lane `lane` of thread `t`'s simdgroup, `x`, in a threadgroup of `n` threads,
/// or all-zero bits for a lane past `n`.
fn lane_word(x: &[u32], n: u32, t: u32, lane: u32) -> u32 {
    let u = (t & !31) + lane;
    if u < n {
        x[u as usize]
    } else {
        0
    }
}

/// `dsm2_shuffle_past_n`'s three words at thread `t` of a threadgroup of `n` threads over
/// inputs `x`: Metal's reading.
fn model(x: &[u32], n: u32, t: u32) -> [u32; 3] {
    let l = t & 31;
    let rd = |lane: u32| lane_word(x, n, t, lane);
    let src = sources();
    [
        rd(src[l as usize]),
        // A source past the simdgroup keeps the caller's value.
        if l + 8 < 32 { rd(l + 8) } else { x[t as usize] },
        rd(l ^ 16),
    ]
}

/// `dsm2_shuffle_past_n_more`'s ten words at thread `t` of a threadgroup of `n` threads over
/// inputs `x`, under [`model`]'s rule: a source past `n` reads zero bits, data and filling value
/// alike.
fn model_more(x: &[u32], n: u32, t: u32) -> [u32; 10] {
    let l = t & 31;
    let (q, qb) = (l & 3, l & !3);
    let rd = |lane: u32| lane_word(x, n, t, lane);
    let fill = |lane: u32| {
        if (t & !31) + lane < n {
            !rd(lane)
        } else {
            0
        }
    };
    let src = sources()[l as usize];
    [
        rd(20),
        rd((l + 8) & 31),
        if l >= 8 { rd(l - 8) } else { fill(l + 24) },
        if l + 8 < 32 { rd(l + 8) } else { fill(l - 24) },
        rd(qb | (q ^ 2)),
        rd(qb | 3),
        // A source past the quad keeps the caller's value.
        if q + 1 < 4 { rd(l + 1) } else { x[t as usize] },
        rd(qb | ((q + 1) & 3)),
        rd(src),
        rd(src) >> 16,
    ]
}

/// Our probe's Metal readings (apple-m3-max-macos26.7, both passes) of
/// `dsm2-shuffle-past-the-thread-count-sixteen` (16 threads, input A) and
/// `dsm2-shuffle-past-the-thread-count-eight` (8 threads, input A): thread `t`'s three words.
const SIXTEEN: [[u32; 3]; 16] = [
    [0x0000_0000, 0xa509_0909, 0x0000_0000],
    [0xa504_0404, 0xa50a_0a0a, 0x0000_0000],
    [0x0000_0000, 0xa50b_0b0b, 0x0000_0000],
    [0xa502_0202, 0xa50c_0c0c, 0x0000_0000],
    [0x0000_0000, 0xa50d_0d0d, 0x0000_0000],
    [0xa508_0808, 0xa50e_0e0e, 0x0000_0000],
    [0x0000_0000, 0xa50f_0f0f, 0x0000_0000],
    [0xa506_0606, 0xa510_1010, 0x0000_0000],
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0xa50c_0c0c, 0x0000_0000, 0x0000_0000],
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0xa50a_0a0a, 0x0000_0000, 0x0000_0000],
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0xa510_1010, 0x0000_0000, 0x0000_0000],
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0xa50e_0e0e, 0x0000_0000, 0x0000_0000],
];
const EIGHT: [[u32; 3]; 8] = [
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0xa504_0404, 0x0000_0000, 0x0000_0000],
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0xa502_0202, 0x0000_0000, 0x0000_0000],
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0xa508_0808, 0x0000_0000, 0x0000_0000],
    [0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0xa506_0606, 0x0000_0000, 0x0000_0000],
];

#[test]
fn the_model_is_metals_reading() {
    for (n, rows) in [(16u32, &SIXTEEN[..]), (8, &EIGHT[..])] {
        let x = inputs(n, false);
        for (t, row) in rows.iter().enumerate() {
            assert_eq!(model(&x, n, t as u32), *row, "{n} threads, thread {t}");
        }
    }
    // The calling-lane rule once guessed gives other words: the even lanes' shuffle would read
    // their own word.
    let x = inputs(16, false);
    assert_ne!(SIXTEEN[0][0], x[0]);
}

/// The buffers after one workgroup of `n` threads ran `m` over `bufs` (bound at slots 0..),
/// identical in both scheduling orders.
fn run_all(
    plan: &Plan,
    m: &TypedModule,
    n: u32,
    lanes: SubgroupSize,
    bufs: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip([n, 1, 1]) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(n));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        assert!(matches!(f.loc, PushLoc::Push { .. }), "{f:?}");
        // Single-region dispatch: the region is not read. Zero residuals.
        assert!(
            matches!(
                f.field,
                PushField::SlotBindOffset(_) | PushField::DispatchRegion
            ),
            "unexpected push field {f:?}"
        );
    }
    let buffers = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    let res = Resources {
        buffers,
        push,
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
        outs.push(run(m, &cfg, res.clone()).expect("run").buffers);
    }
    assert_eq!(outs[0], outs[1], "order independent");
    outs.swap_remove(0)
}

fn bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

type Compiled<'a> = &'a dyn Fn(&[u8], &str) -> (Plan, TypedModule);

/// Run both fixtures at each threadgroup size of `sizes` under both inputs, at each subgroup
/// width of `widths`, and check every word against the models.
fn check_on(what: &str, sizes: &[u32], widths: &[SubgroupSize], compiled: Compiled<'_>) {
    for (stem, more) in [
        ("dsm2_shuffle_past_n", false),
        ("dsm2_shuffle_past_n_more", true),
    ] {
        let Some(bc) = fixture(stem) else { return };
        let (plan, m) = compiled(&bc, stem);
        let per = if more { 10 } else { 3 };
        for &n in sizes {
            for other in [false, true] {
                let x = inputs(n, other);
                let bufs = [
                    bytes(&x),
                    bytes(&src_buffer(more)),
                    vec![0xAB; (4 * per * n) as usize],
                ];
                for &lanes in widths {
                    let out = words(&run_all(&plan, &m, n, lanes, &bufs)[2]);
                    for t in 0..n {
                        let got = &out[(per * t) as usize..(per * (t + 1)) as usize];
                        let want: Vec<u32> = if more {
                            model_more(&x, n, t).to_vec()
                        } else {
                            model(&x, n, t).to_vec()
                        };
                        assert!(got.iter().all(|&w| w != POISON), "{what} {stem} written");
                        assert_eq!(
                            got,
                            want,
                            "{what} {stem}: {n} threads, input {}, {lanes:?}, thread {t}",
                            if other { "B" } else { "A" }
                        );
                    }
                }
            }
        }
    }
}

/// The partial and whole threadgroups: a quad of two lanes (6), simdgroups of 8 and 16 lanes, a
/// whole simdgroup and one of 10 (42) or 16 (48) lanes, and a whole one (32).
const SIZES: [u32; 6] = [6, 8, 16, 32, 42, 48];

#[test]
fn the_emulated_tier_reads_a_lane_past_the_thread_count_as_zero() {
    for stem in PROFILES {
        let p: DeviceProfile = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        check_on(stem, &SIZES, &WIDTHS, &|bc, entry| {
            let (a, plan, m) =
                compile_kept(bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
            assert_eq!(plan.simd.1, Tier::Emulated, "{stem} {entry}");
            (plan, m)
        });
    }
}

#[test]
fn the_emulated_variant_reads_a_lane_past_the_thread_count_as_zero() {
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let mut inputs = LoweringInputs::new();
    inputs.emulated_simdgroups = true;
    check_on("moltenvk-m3 variant", &SIZES, &WIDTHS, &|bc, entry| {
        let (plan, m) =
            typed_module_with(bc, entry, &p, &inputs).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(plan.simd.1, Tier::Emulated, "{entry}");
        assert!(
            plan.facts.contains(&RuntimeFact::EmulatedSimdgroups),
            "{entry}"
        );
        (plan, m)
    });
}

#[test]
fn the_direct_tier_gives_the_whole_simdgroup_control() {
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    check_on("moltenvk-m3", &[32], &[SubgroupSize::W32], &|bc, entry| {
        let (a, plan, m) = compile_kept(bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(plan.simd.1, Tier::Direct, "{entry}");
        (plan, m)
    });
}
