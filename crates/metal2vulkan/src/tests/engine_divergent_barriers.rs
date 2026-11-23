//! Workgroup barriers that part of the workgroup skips.
//!
//! A threadgroup barrier after a divergent early return, under a thread-dependent condition, or
//! in a loop with a per-thread trip count is a workgroup barrier in non-uniform control flow,
//! which Vulkan leaves undefined (Intel's ANV hangs on it). Every kernel with one is walked
//! predicated (the walk, in every tier): every invocation reaches each such barrier, the returned
//! or skipping ones inactive, so the others synchronize among themselves, Apple's reading.
//!
//! The kernels below (inline AIR text, assembled by `llvm-as`) and the owned fixture
//! `a3_early_return_barrier` compile on Direct-tier, subgroup-free, and emulating profiles. Each
//! module passes `spirv-val`, has no workgroup barrier in a block its own uniformity does not
//! prove uniform (`spirv::testing::barriers`), and runs on the CPU interpreter under Vulkan's
//! barrier rule (`BarrierRule::Vulkan`: a returned invocation while others wait at a barrier
//! traps), in both scheduling orders, at several threadgroup sizes and subgroup widths, giving
//! the words the predicated reading predicts. A kernel whose barriers are all uniform is not
//! predicated. An invocation that leaves a predicated loop before the others keeps the value it
//! reads after the loop: the walk runs the loop's later trips for it too, so `spirv::carry` ends
//! no value at their tops. No GPU is used.
//!
//! `llvm-as` assembles the kernels; a missing tool skips visibly.

use std::collections::BTreeMap;

use metal2vulkan_devtools::loop_kernels;

use crate::device::subgroup::Tier;
use crate::device::DevicePolicy;
use crate::lower::select::Plan;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
    Scratch,
};
use crate::spirv::interp::{run, BarrierRule, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::barriers::workgroup_barriers;

/// Profiles whose kernels run on subgroups (`moltenvk-m3`, Intel, AMD, NVIDIA) and whose
/// simdgroups are emulated (`lavapipe`, `roadmap-2022`).
const PROFILES: [&str; 6] = [
    "moltenvk-m3",
    "intel-irisxe-rplp-anv",
    "amd-780m-radv",
    "nvidia-rtx50-linux",
    "lavapipe",
    "runtime-floor",
];

/// The poison every output word starts as.
const POISON: u32 = 0xABAB_ABAB;

fn assemble_file(path: &std::path::Path, what: &str) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {what}: {why}"),
    }
}

/// `body` (the kernel `name`'s definition, parameters `%limits`, `%out`, `%t`) in a module with
/// the threadgroup arrays `@a` and `@b`, the barrier and simd declarations, and the kernel's AIR
/// metadata (`loop_kernels::module`), assembled.
pub(super) fn kernel(name: &str, body: &str) -> Option<Vec<u8>> {
    let text = loop_kernels::module(name, body);
    let scratch = Scratch::new(&format!("m2v-divergent-barriers-{name}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    assemble_file(&path, name)
}

/// `if (t < limits[0]) { a[t] = 3t + 1; barrier; out[t] = a[(t + 1) % limits[0]]; } else
/// out[t] = 0xE0000000 | t;`: a barrier under a thread-dependent condition.
const IF_BARRIER: &str = r#"entry:
  %n = load i32, ptr addrspace(1) %limits, align 4
  %tz = zext i32 %t to i64
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  %c = icmp ult i32 %t, %n
  br i1 %c, label %in, label %outside
in:
  %v = mul i32 %t, 3
  %v1 = add i32 %v, 1
  %pa = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %tz
  store i32 %v1, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %t1 = add i32 %t, 1
  %src = urem i32 %t1, %n
  %sz = zext i32 %src to i64
  %ps = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %sz
  %x = load i32, ptr addrspace(3) %ps, align 4
  store i32 %x, ptr addrspace(1) %po, align 4
  br label %done
outside:
  %m = or i32 %t, -536870912
  store i32 %m, ptr addrspace(1) %po, align 4
  br label %done
done:
  ret void"#;

/// `for (i = 0; i < (t & 3); i++) { a[t] = 16t + i; barrier; acc += a[t ^ 4]; barrier; }
/// out[t] = acc;`: barriers in a loop with a per-thread trip count. Thread `t ^ 4` has `t`'s trip
/// count, so it is in the same iteration.
const LOOP_BARRIER: &str = r#"entry:
  %k = and i32 %t, 3
  %tz = zext i32 %t to i64
  %z = icmp eq i32 %k, 0
  br i1 %z, label %done, label %body
body:
  %i = phi i32 [ 0, %entry ], [ %i1, %body ]
  %acc = phi i32 [ 0, %entry ], [ %acc1, %body ]
  %w0 = shl i32 %t, 4
  %w = add i32 %w0, %i
  %pa = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %tz
  store i32 %w, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %p = xor i32 %t, 4
  %pz = zext i32 %p to i64
  %pp = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %pz
  %x = load i32, ptr addrspace(3) %pp, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %acc1 = add i32 %acc, %x
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, %k
  br i1 %more, label %body, label %done
done:
  %r = phi i32 [ 0, %entry ], [ %acc1, %body ]
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  store i32 %r, ptr addrspace(1) %po, align 4
  ret void"#;

/// `for (i = 0; i < limits[0]; i++) { if (t & 1) { if ((t & 3) == 3) continue; acc += 7; }
/// a[t] = t + 1000 i; barrier; acc += a[t ^ 4]; barrier; } out[t] = acc;`: a loop every thread
/// iterates together, whose barriers the threads `t & 3 == 3` skip by a `continue` nested in a
/// branch.
const CONTINUE_BARRIER: &str = r#"entry:
  %n = load i32, ptr addrspace(1) %limits, align 4
  %tz = zext i32 %t to i64
  %odd = and i32 %t, 1
  %isodd = icmp ne i32 %odd, 0
  %lo = and i32 %t, 3
  %skip = icmp eq i32 %lo, 3
  %z = icmp eq i32 %n, 0
  br i1 %z, label %done, label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc2, %latch ]
  br i1 %isodd, label %oddb, label %even
oddb:
  br i1 %skip, label %latch, label %addf
addf:
  %accf = add i32 %acc, 7
  br label %even
even:
  %acce = phi i32 [ %acc, %header ], [ %accf, %addf ]
  %im = mul i32 %i, 1000
  %w = add i32 %im, %t
  %pa = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %tz
  store i32 %w, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %p = xor i32 %t, 4
  %pz = zext i32 %p to i64
  %pp = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %pz
  %x = load i32, ptr addrspace(3) %pp, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %acc1 = add i32 %acce, %x
  br label %latch
latch:
  %acc2 = phi i32 [ %acc, %oddb ], [ %acc1, %even ]
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, %n
  br i1 %more, label %header, label %done
done:
  %r = phi i32 [ 0, %entry ], [ %acc2, %latch ]
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  store i32 %r, ptr addrspace(1) %po, align 4
  ret void"#;

/// `s = simd_sum(1); if (t < limits[0]) { a[t] = s + t; barrier; out[t] = a[(t + 1) %
/// limits[0]]; } else out[t] = 0xE0000000 | t;`: a simdgroup operation every thread runs, then a
/// threadgroup barrier under a thread-dependent condition (on an emulating profile the kernel's
/// only synchronization outside uniform control flow is that barrier).
const SIMD_THEN_IF_BARRIER: &str = r#"entry:
  %s = tail call i32 @air.simd_sum.u.i32(i32 1) #1
  %n = load i32, ptr addrspace(1) %limits, align 4
  %tz = zext i32 %t to i64
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  %c = icmp ult i32 %t, %n
  br i1 %c, label %in, label %outside
in:
  %v = add i32 %s, %t
  %pa = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %tz
  store i32 %v, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %t1 = add i32 %t, 1
  %src = urem i32 %t1, %n
  %sz = zext i32 %src to i64
  %ps = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %sz
  %x = load i32, ptr addrspace(3) %ps, align 4
  store i32 %x, ptr addrspace(1) %po, align 4
  br label %done
outside:
  %m = or i32 %t, -536870912
  store i32 %m, ptr addrspace(1) %po, align 4
  br label %done
done:
  ret void"#;

/// `for (i = 0; i < 3; i++) { a[t] = t + i; barrier; acc += a[(t + 1) % limits[1]]; barrier; }
/// out[t] = acc;`: every barrier in workgroup-uniform control flow.
const UNIFORM_BARRIERS: &str = r#"entry:
  %pn = getelementptr inbounds i32, ptr addrspace(1) %limits, i64 1
  %nn = load i32, ptr addrspace(1) %pn, align 4
  %tz = zext i32 %t to i64
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %i1, %body ]
  %acc = phi i32 [ 0, %entry ], [ %acc1, %body ]
  %w = add i32 %t, %i
  %pa = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %tz
  store i32 %w, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %t1 = add i32 %t, 1
  %src = urem i32 %t1, %nn
  %sz = zext i32 %src to i64
  %ps = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %sz
  %x = load i32, ptr addrspace(3) %ps, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %acc1 = add i32 %acc, %x
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, 3
  br i1 %more, label %body, label %done
done:
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  store i32 %acc1, ptr addrspace(1) %po, align 4
  ret void"#;

/// The threads of a threadgroup of `n`, and the limits each case passes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Case {
    pub(super) n: u32,
    pub(super) limits: [u32; 2],
}

/// Thread `t`'s expected word of `kernel` in `c`, by the predicated reading: the threads that
/// reach a barrier synchronize among themselves.
fn expected(kernel: &str, c: Case, t: u32) -> u32 {
    let [l0, l1] = c.limits;
    match kernel {
        "if_barrier_k" => {
            if t < l0 {
                ((t + 1) % l0) * 3 + 1
            } else {
                0xE000_0000 | t
            }
        }
        "loop_barrier_k" => (0..(t & 3)).map(|i| ((t ^ 4) << 4) + i).sum(),
        "loop_leaver_k" => loop_kernels::loop_leaver(t),
        "continue_barrier_k" => {
            if t & 3 == 3 {
                0
            } else {
                let odd = if t & 1 == 1 { 7 } else { 0 };
                (0..l0).map(|i| odd + (t ^ 4) + 1000 * i).sum()
            }
        }
        "simd_then_if_barrier_k" => {
            if t < l0 {
                let u = (t + 1) % l0;
                let lanes = (c.n - 32 * (u / 32)).min(32);
                lanes + u
            } else {
                0xE000_0000 | t
            }
        }
        "uniform_barriers_k" => (0..3).map(|i| (t + 1) % l1 + i).sum(),
        // Threads `t >= limits[0]` return before the first barrier, `t >= limits[1]` between the
        // two.
        "a3_early_return_barrier" => {
            if t >= l0 {
                0xE000_0000 | t
            } else if t >= l1 {
                0xE100_0000 | (((t + 1) % l0) * 3 + 1)
            } else {
                let u = (t + 5) % l1;
                ((u + 1) % l0) * 3 + 1 + 1000
            }
        }
        other => panic!("no expectation for {other}"),
    }
}

/// The cases of `kernel` (threadgroup sizes whose threads `t ^ 4` exist, multiples of 32 for a
/// kernel with a simdgroup operation on a subgroup tier).
fn cases(kernel: &str, tier: Tier) -> Vec<Case> {
    let sizes: &[u32] = match (kernel, tier) {
        ("simd_then_if_barrier_k", Tier::Emulated) => &[32, 64, 40, 96],
        ("simd_then_if_barrier_k", _) => &[32, 64, 96],
        _ => &[8, 32, 40, 64, 96, 128],
    };
    let mut out = Vec::new();
    for &n in sizes {
        let limits: Vec<[u32; 2]> = match kernel {
            "continue_barrier_k" => vec![[0, 0], [1, 0], [3, 0]],
            "uniform_barriers_k" => vec![[0, n]],
            "loop_barrier_k" | "loop_leaver_k" => vec![[0, 0]],
            "a3_early_return_barrier" => {
                vec![[n, n], [n - n / 4, n / 3 + 1], [n, 1], [n / 2 + 1, n / 2]]
            }
            _ => vec![[n, 0], [n / 2 + 1, 0], [1, 0]],
        };
        out.extend(limits.into_iter().map(|limits| Case { n, limits }));
    }
    out
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

/// The output words after one workgroup of `c.n` threads ran `m` (planned as `plan`) with
/// `lanes`-wide subgroups under Vulkan's barrier rule, identical in both scheduling orders.
pub(super) fn run_case(plan: &Plan, m: &TypedModule, c: Case, lanes: SubgroupSize) -> Vec<u32> {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([c.n, 1, 1]));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        assert!(matches!(f.loc, PushLoc::Push { .. }), "{f:?}");
        assert!(
            matches!(
                f.field,
                PushField::SlotBindOffset(_) | PushField::DispatchRegion
            ),
            "unexpected push field {f:?}"
        );
    }
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: slot(0),
                address: None,
                bytes: bytes(&c.limits),
            },
            Buffer {
                slot: slot(1),
                address: None,
                bytes: bytes(&vec![POISON; c.n as usize]),
            },
        ],
        push,
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, 1, 1],
            subgroup: lanes,
            order,
            barriers: BarrierRule::Vulkan,
            ..Config::default()
        };
        let o = run(m, &cfg, res.clone()).unwrap_or_else(|e| panic!("{c:?} {lanes:?}: {e}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "{c:?} {lanes:?}: order independent");
    words(&outs[0][1])
}

/// Compile `kernel` on `stem`, validate it, check its barriers against the module's own
/// uniformity, and run every case at every width its tier allows.
fn check(stem: &str, bc: &[u8], kernel: &str, predicated: bool) {
    let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
    let (plan, m) = typed_module(bc, kernel, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: {e}"));
    assert_eq!(
        plan.families.simd.predication.is_some(),
        predicated,
        "{stem} {kernel}: predicated"
    );
    let report = workgroup_barriers(&m).expect("check");
    assert!(report.barriers > 0, "{stem} {kernel}: no barrier");
    assert_eq!(report.in_callees, 0, "{stem} {kernel}");
    assert!(
        report.nonuniform.is_empty(),
        "{stem} {kernel}: workgroup barriers in non-uniform blocks {:?}",
        report.nonuniform
    );
    let tier = plan.simd.1;
    let widths: &[SubgroupSize] = match tier {
        Tier::Direct | Tier::Cluster32 => &[SubgroupSize::W32],
        _ => &[SubgroupSize::W8, SubgroupSize::W16, SubgroupSize::W32],
    };
    for c in cases(kernel, tier) {
        for &lanes in widths {
            let out = run_case(&plan, &m, c, lanes);
            for (t, got) in out.iter().enumerate() {
                let want = expected(kernel, c, t as u32);
                assert_eq!(
                    *got, want,
                    "{stem} {kernel} {c:?} {lanes:?} ({tier:?}) thread {t}"
                );
            }
        }
    }
}

/// The predicated kernels: on every profile each runs predicated and gives the predicated
/// reading, with no workgroup barrier in non-uniform control flow.
#[test]
fn divergent_barriers_run_predicated_on_every_profile() {
    let kernels = [
        ("if_barrier_k", IF_BARRIER),
        ("loop_barrier_k", LOOP_BARRIER),
        ("continue_barrier_k", CONTINUE_BARRIER),
        ("simd_then_if_barrier_k", SIMD_THEN_IF_BARRIER),
    ];
    for (name, body) in kernels {
        let Some(bc) = kernel(name, body) else { return };
        for stem in PROFILES {
            check(stem, &bc, name, true);
        }
    }
}

/// A thread that leaves a predicated loop before the others reads the value of its own last trip
/// after the loop: no store at a later trip's top, which it runs not live, ends that value.
#[test]
fn a_thread_that_leaves_a_vote_loop_early_keeps_its_last_trips_value() {
    let Some(bc) = kernel("loop_leaver_k", loop_kernels::LOOP_LEAVER) else {
        return;
    };
    for stem in PROFILES {
        check(stem, &bc, "loop_leaver_k", true);
    }
}

/// The owned fixture: threads returning before the first and between the two barriers.
#[test]
fn early_returns_before_barriers_run_predicated_on_every_profile() {
    let Some(bc) = assemble_file(
        &engine_dir().join("a3_early_return_barrier.ll"),
        "a3_early_return_barrier",
    ) else {
        return;
    };
    for stem in PROFILES {
        check(stem, &bc, "a3_early_return_barrier", true);
    }
}

/// A kernel whose barriers are all in workgroup-uniform control flow is not predicated: no
/// vote word, no fact.
#[test]
fn uniform_barriers_are_not_predicated() {
    let Some(bc) = kernel("uniform_barriers_k", UNIFORM_BARRIERS) else {
        return;
    };
    for stem in PROFILES {
        check(stem, &bc, "uniform_barriers_k", false);
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let (plan, _) = typed_module(&bc, "uniform_barriers_k", &p).expect("plan");
        assert!(
            !plan.facts.contains(&RuntimeFact::PredicationVote),
            "{stem}: no vote word"
        );
    }
}

/// A predicated loop votes in a workgroup word outside the emulated tier: the kernel depends on
/// `RuntimeFact::PredicationVote` exactly when a barrier lies in a loop, and in the emulated tier
/// it votes in its scratch instead.
#[test]
fn only_a_barrier_in_a_loop_takes_the_vote_word() {
    for (name, body, in_loop) in [
        ("if_barrier_k", IF_BARRIER, false),
        ("loop_barrier_k", LOOP_BARRIER, true),
        ("continue_barrier_k", CONTINUE_BARRIER, true),
    ] {
        let Some(bc) = kernel(name, body) else { return };
        for stem in PROFILES {
            let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
            let (plan, _) = typed_module(&bc, name, &p).expect("plan");
            let vote = plan.facts.contains(&RuntimeFact::PredicationVote);
            let emulated = plan.simd.1 == Tier::Emulated;
            assert_eq!(vote, in_loop && !emulated, "{stem} {name}: vote word");
        }
    }
}
