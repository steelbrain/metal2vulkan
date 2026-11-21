//! A structurizer variable whose value no later trip reads is ended at each trip's top
//! (`spirv::carry`).
//!
//! The kernel below (inline AIR text, assembled by `llvm-as`) leaves its loop by three exits, and
//! the second one reads a value its trip defined. The exit dispatch after the loop reads that
//! value's variable in one arm, so a path from the loop's back edge reaches the read without
//! passing the store, and a driver would keep the variable around the loop. On every profile the
//! module ends the variable at the top of each trip, no variable is left that a later trip never
//! reads but the module keeps around a back edge, the module passes `spirv-val`, and it runs on
//! the CPU interpreter giving the words the source computes. No GPU is used.
//!
//! `llvm-as` assembles the kernel; a missing tool skips visibly.

use std::collections::BTreeSet;

use super::engine_divergent_barriers::{kernel, run_case, Case};
use crate::device::DevicePolicy;
use crate::pipeline::testing::{compile, link_and_validate, profile, typed_module};
use crate::spirv::carry::dead_carries;
use crate::spirv::interp::SubgroupSize;
use crate::spirv::ir::{Inst, TypedModule};

/// Profiles of every vendor, and the emulating ones.
const PROFILES: [&str; 6] = [
    "moltenvk-m3",
    "intel-irisxe-rplp-anv",
    "amd-rdna3-radv",
    "nvidia-rtx50-linux",
    "lavapipe",
    "runtime-floor",
];

/// `for (i = 0; i < limits[0]; i++) { x = (7i + t) & 15; if (x == 3) { out[t] = 100i + x;
/// return; } if (x > 12) { out[t] = i + 50; return; } } out[t] = 999;`: `x` is defined after the
/// loop's first exit and read after the loop by its second.
const TRIP_EXITS: &str = r#"entry:
  %n = load i32, ptr addrspace(1) %limits, align 4
  %tz = zext i32 %t to i64
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  %more = icmp ult i32 %i, %n
  br i1 %more, label %b1, label %none
b1:
  %i7 = mul i32 %i, 7
  %s = add i32 %i7, %t
  %x = and i32 %s, 15
  %hit = icmp eq i32 %x, 3
  br i1 %hit, label %found, label %b2
b2:
  %big = icmp ugt i32 %x, 12
  br i1 %big, label %large, label %latch
latch:
  %i1 = add i32 %i, 1
  br label %header
found:
  %ih = mul i32 %i, 100
  %r = add i32 %ih, %x
  store i32 %r, ptr addrspace(1) %po, align 4
  br label %done
large:
  %rl = add i32 %i, 50
  store i32 %rl, ptr addrspace(1) %po, align 4
  br label %done
none:
  store i32 999, ptr addrspace(1) %po, align 4
  br label %done
done:
  ret void"#;

/// Thread `t`'s word for a trip bound of `n`, as the source computes it.
fn expected(n: u32, t: u32) -> u32 {
    for i in 0..n {
        let x = (7 * i + t) & 15;
        if x == 3 {
            return 100 * i + x;
        }
        if x > 12 {
            return i + 50;
        }
    }
    999
}

/// The stores of an undefined value in `m`.
fn undefined_stores(m: &TypedModule) -> usize {
    let insts = || m.blocks().iter().flat_map(|b| b.insts.iter());
    let undefs: BTreeSet<_> = insts()
        .filter(|node| matches!(node.inst, Inst::Undef))
        .filter_map(|node| node.result)
        .collect();
    insts()
        .filter(|node| matches!(&node.inst, Inst::Store { value, .. } if undefs.contains(value)))
        .count()
}

/// The value the second exit reads is ended at each trip's top, and none is kept around the back
/// edge that no later trip reads; the kernel computes what its source does.
#[test]
fn a_value_no_later_trip_reads_is_ended_at_each_trips_top() {
    let Some(bc) = kernel("trip_exits_k", TRIP_EXITS) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a = compile(&bc, "trip_exits_k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (plan, m) =
            typed_module(&bc, "trip_exits_k", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            plan.families.simd.predication.is_none(),
            "{stem}: the walk is not predicated"
        );
        let ended = undefined_stores(&m);
        eprintln!("{stem}: {ended} values ended at a trip's top");
        assert!(ended > 0, "{stem}: no value is ended at a trip's top");
        assert_eq!(
            dead_carries(&m),
            0,
            "{stem}: a value no later trip reads is kept around the loop"
        );
        for n in [32, 64] {
            for bound in [0, 1, 2, 5, 16] {
                let c = Case {
                    n,
                    limits: [bound, 0],
                };
                let out = run_case(&plan, &m, c, SubgroupSize::W32);
                for (t, got) in out.iter().enumerate() {
                    assert_eq!(*got, expected(bound, t as u32), "{stem} {c:?} thread {t}");
                }
            }
        }
    }
}
