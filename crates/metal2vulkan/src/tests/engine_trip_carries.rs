//! A structurizer variable whose value no later trip reads is ended at each trip's top
//! (`spirv::carry`), and the exit dispatch after a loop runs only the arm of the exit taken.
//!
//! The kernels (`metal2vulkan_devtools::loop_kernels`, inline AIR text assembled by `llvm-as`)
//! leave their loops by several exits, each reading values defined on its own path: `trip_exits_k`
//! a loop of three exits, `nested_exits_k` an outer loop of four exits whose inner loop has two,
//! one of them leaving both loops. The exit dispatch reads each such value's variable in one arm,
//! so a path from a loop's back edge reaches the read without passing the store, and a driver
//! would keep the variable around the loop. On every profile the module ends such variables at
//! the top of each trip, no variable is left that a later trip never reads but the module keeps
//! around a back edge, the module passes `spirv-val`, and it runs on the CPU interpreter giving the
//! words the source computes at every bound of `loop_kernels::BOUNDS`, which reach every exit. No
//! GPU is used.
//!
//! `llvm-as` assembles the kernels; a missing tool skips visibly.

use std::collections::BTreeSet;

use metal2vulkan_devtools::loop_kernels::{self, BOUNDS, NESTED_EXITS, TRIP_EXITS};

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

/// The kernels' values read by one exit each are ended at each trip's top, none is kept around a
/// back edge that no later trip reads, and each kernel computes what its source does.
#[test]
fn a_value_no_later_trip_reads_is_ended_at_each_trips_top() {
    let kernels: [(&str, &str, fn([u32; 2], u32) -> u32); 2] = [
        ("trip_exits_k", TRIP_EXITS, |[n, _], t| {
            loop_kernels::trip_exits(n, t)
        }),
        ("nested_exits_k", NESTED_EXITS, |[n, m], t| {
            loop_kernels::nested_exits(n, m, t)
        }),
    ];
    for (name, body, answer) in kernels {
        let Some(bc) = kernel(name, body) else {
            return;
        };
        for stem in PROFILES {
            let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
            let a = compile(&bc, name, &p).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
            let (plan, m) =
                typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{stem} {name}: {e}"));
            assert!(
                plan.families.simd.predication.is_none(),
                "{stem} {name}: the walk is not predicated"
            );
            let ended = undefined_stores(&m);
            eprintln!("{stem} {name}: {ended} values ended at a trip's top");
            assert!(
                ended > 0,
                "{stem} {name}: no value is ended at a trip's top"
            );
            assert_eq!(
                dead_carries(&m),
                0,
                "{stem} {name}: a value no later trip reads is kept around the loop"
            );
            for n in [32, 64] {
                for limits in BOUNDS {
                    let c = Case { n, limits };
                    let out = run_case(&plan, &m, c, SubgroupSize::W32);
                    for (t, got) in out.iter().enumerate() {
                        assert_eq!(
                            *got,
                            answer(limits, t as u32),
                            "{stem} {name} {c:?} thread {t}"
                        );
                    }
                }
            }
        }
    }
}
