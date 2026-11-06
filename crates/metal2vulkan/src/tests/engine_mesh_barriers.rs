//! Object and mesh functions that wait at threadgroup barriers.
//!
//! The owned fixture `engine/mesh_partial_barrier` holds three object functions, whose barriers
//! stand in uniform control flow and in a loop, after a divergent early return, and under a
//! thread-dependent condition, and a mesh function that returns before its barrier. On every
//! committed profile, natively or emulated, each object function runs a `drawMeshThreads` partial
//! object threadgroup (`RuntimeFact::PartialObjectThreadgroups`) and is walked predicated from its
//! threadgroup's threads (`lower::simd::divergent`'s partial object threadgroups), the mesh function
//! is walked predicated for its divergent barrier, and no workgroup barrier of a body stands in a
//! block the body's own uniformity does not prove uniform (`spirv::testing::barriers`). On the
//! emulating profiles, whose object functions are compute entries, each object function runs on
//! the CPU interpreter under Vulkan's barrier rule (`BarrierRule::Vulkan`: an invocation that
//! returned while others wait at a barrier traps), in both scheduling orders, for draws that leave
//! partial threadgroups and for whole ones, giving the reading where the threads that reach a
//! barrier synchronize among themselves: Metal's threads alone count themselves and write, the
//! invocations past a partial threadgroup's threads writing nothing. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;

use super::engine_mesh::{device, emulating_profiles, fixture, mesh_profiles};
use crate::bind::{BindState, DrawCall};
use crate::budget::Budget;
use crate::library::Library;
use crate::lower::mesh::{work, OBJECT_RECORDS_SLOT, OBJECT_WORK_SLOT};
use crate::lower::select::Plan;
use crate::pipeline::descriptor::{FunctionDescriptor, MeshPipelineDescriptor, StageSource};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::typed_module;
use crate::spirv::interp::{run, BarrierRule, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::{Inst, TypedModule};
use crate::spirv::testing::barriers::workgroup_barriers_of;

/// The fixture's object functions.
const OBJECTS: [&str; 3] = [
    "mesh_partial_barrier_uniform",
    "mesh_partial_barrier_early_return",
    "mesh_partial_barrier_if",
];

/// The fixture's mesh function with a divergent barrier.
const MESH: &str = "mesh_partial_barrier_mesh_early_return";

/// The words of out a case's buffer starts as, but its count.
const POISON: u32 = 0xABAB_ABAB;

/// The control words: the loop's trip count and the threads that stay past the early return.
const CTL: [u32; 4] = [3, 3, 0, 0];

/// The function the entry of `m` calls: an object or mesh function's body, which its entry wrapper
/// calls between its own barriers.
fn body(m: &TypedModule) -> crate::spirv::ids::FuncId {
    let entry = m.entry().func;
    let f = m.functions().get(entry.index()).expect("the entry");
    f.blocks
        .iter()
        .filter_map(|b| m.blocks().get(b.index()))
        .flat_map(|blk| blk.insts.iter())
        .find_map(|n| match &n.inst {
            Inst::Call { func, .. } => Some(*func),
            _ => None,
        })
        .expect("the entry calls its body")
}

/// Every object function of the fixture and its mesh function, on every committed profile: each
/// is walked predicated, an object function from its threadgroup's threads (its barriers uniform or
/// not) and the mesh function for its divergent barrier; every workgroup barrier of each body stands
/// in a block the body's uniformity proves uniform; each object function records
/// `PartialObjectThreadgroups`; and each module passes L3.
#[test]
fn object_and_mesh_functions_wait_at_their_barriers_predicated() {
    let Some(bc) = fixture("mesh_partial_barrier") else {
        return;
    };
    let profiles: Vec<_> = mesh_profiles()
        .into_iter()
        .chain(emulating_profiles())
        .collect();
    for (stem, p) in &profiles {
        for entry in OBJECTS.into_iter().chain([MESH]) {
            let what = format!("{stem} {entry}");
            let object = entry != MESH;
            let (plan, m) = typed_module(&bc, entry, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                plan.stage,
                if object { Stage::Object } else { Stage::Mesh },
                "{what}"
            );
            let walk = plan
                .families
                .simd
                .predication
                .as_ref()
                .unwrap_or_else(|| panic!("{what}: not predicated"));
            assert_eq!(walk.partial_groups, object, "{what}");
            assert_eq!(plan.families.mesh.partial_groups, object, "{what}");
            let report = workgroup_barriers_of(&m, body(&m)).expect("check");
            assert!(report.barriers > 0, "{what}: no barrier in the body");
            assert!(
                report.nonuniform.is_empty(),
                "{what}: workgroup barriers in non-uniform blocks {:?}",
                report.nonuniform
            );
            let a = super::engine_mesh::compile(&bc, entry, p)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(
                a.facts.contains(&RuntimeFact::PartialObjectThreadgroups),
                object,
                "{what}"
            );
            crate::spirv::testing::check_words_valid(&a.module.words, &a.module.requirements, p)
                .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
}

/// An object function whose body waits at a threadgroup barrier draws a `drawMeshThreads` that
/// leaves a partial object threadgroup, emulated or not: its pipeline records
/// `PartialObjectThreadgroups`, and both whole and partial draws instantiate, a partial one at the
/// threadgroups its threads fill, rounded up. `icb_object_threadgroup_object`'s barriers stand
/// under its thread 0's condition, the fixture's in uniform control flow.
#[test]
fn a_partial_object_threadgroup_of_a_body_with_a_barrier_draws() {
    let (Some(icb), Some(partial)) = (
        fixture("icb_object_threadgroup"),
        fixture("mesh_partial_barrier"),
    ) else {
        return;
    };
    let icb = Library::from_air(&icb).expect("library");
    let partial = Library::from_air(&partial).expect("library");
    let function = |lib: &Library, name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let pipelines = [
        (
            &icb,
            "icb_object_threadgroup_object",
            "icb_object_threadgroup_mesh",
        ),
        (
            &partial,
            "mesh_partial_barrier_uniform",
            "mesh_partial_barrier_mesh",
        ),
    ];
    let draw = |threads| DrawCall::MeshThreads {
        threads,
        threads_per_object_group: [4, 1, 1],
        threads_per_mesh_group: [8, 1, 1],
    };
    for (stem, _) in mesh_profiles().into_iter().chain(emulating_profiles()) {
        let dev = device(&stem);
        for (lib, object, mesh) in pipelines {
            let what = format!("{stem} {object}");
            let mut d = MeshPipelineDescriptor::new(function(lib, mesh));
            d.object = Some(function(lib, object));
            d.rasterization_enabled = false;
            let made = dev
                .compile_mesh_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(
                made.dependencies
                    .iter()
                    .any(|d| d.fact == RuntimeFact::PartialObjectThreadgroups),
                "{what}"
            );
            let state = BindState::default();
            let plan = made
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: plan: {e}"));
            plan.instantiate_draw(&state, &draw([8, 1, 1]))
                .unwrap_or_else(|e| panic!("{what}: whole object threadgroups: {e}"));
            let cmds = plan
                .instantiate_draw(&state, &draw([6, 1, 1]))
                .unwrap_or_else(|e| panic!("{what}: a partial object threadgroup: {e}"));
            let groups = match (&cmds.mesh_dispatch, &cmds.draw) {
                (Some(d), _) => d.object_groups,
                (
                    None,
                    crate::vk::DrawCommand::DrawMeshTasks {
                        group_count_x,
                        group_count_y,
                        group_count_z,
                    },
                ) => Some([*group_count_x, *group_count_y, *group_count_z]),
                (None, other) => panic!("{what}: {other:?}"),
            };
            assert_eq!(groups, Some([2, 1, 1]), "{what}");
        }
    }
}

/// One draw: Metal's threads and its threads per object threadgroup.
#[derive(Clone, Copy, Debug)]
struct Draw {
    threads: [u32; 2],
    l: [u32; 2],
}

/// The draws each object function runs: partial threadgroups along x, along both axes, past a
/// simdgroup, and whole ones.
const DRAWS: [Draw; 5] = [
    Draw {
        threads: [6, 1],
        l: [4, 1],
    },
    Draw {
        threads: [5, 3],
        l: [4, 2],
    },
    Draw {
        threads: [37, 1],
        l: [32, 1],
    },
    Draw {
        threads: [2, 1],
        l: [4, 1],
    },
    Draw {
        threads: [8, 2],
        l: [4, 2],
    },
];

/// The words of out after `draw` of object function `entry`, by the reading where the threads
/// that reach a barrier synchronize among themselves: every Metal thread counted in word 0, each
/// writing its two words at its grid position, the rest of the buffer untouched.
fn expected(entry: &str, draw: Draw) -> Vec<u32> {
    let [tx, ty] = draw.threads;
    let [lx, ly] = draw.l;
    let mut out = vec![POISON; 4 + 2 * (tx * ty) as usize];
    out[0] = tx * ty;
    for gy in 0..ty.div_ceil(ly) {
        for gx in 0..tx.div_ceil(lx) {
            // The threads this threadgroup runs, per axis.
            let rx = (tx - gx * lx).min(lx);
            let ry = (ty - gy * ly).min(ly);
            let n = rx * ry;
            let stay = CTL[1].min(n);
            let mut words: Vec<u32> = (0..n).map(|t| t * 3 + 1).collect();
            let mut acc = vec![0u32; n as usize];
            for _ in 0..CTL[0] {
                let next: Vec<u32> = (0..n).map(|t| words[((t + 1) % n) as usize]).collect();
                for t in 0..n as usize {
                    acc[t] += next[t];
                    words[t] = next[t] + 100;
                }
            }
            for y in 0..ry {
                for x in 0..rx {
                    let tid = x + rx * y;
                    let (px, py) = (gx * lx + x, gy * ly + y);
                    let at = (4 + 2 * (px + tx * py)) as usize;
                    out[at] = match entry {
                        "mesh_partial_barrier_uniform" => {
                            acc[tid as usize] + words[((tid + 2) % n) as usize]
                        }
                        "mesh_partial_barrier_early_return" if tid >= stay => 0xE000_0000 | tid,
                        "mesh_partial_barrier_early_return" => ((tid + 1) % stay) * 3 + 1,
                        "mesh_partial_barrier_if" if tid < stay => ((tid + 1) % stay) * 5 + 2,
                        "mesh_partial_barrier_if" => 0xE100_0000 | tid,
                        other => panic!("no expectation for {other}"),
                    };
                    out[at + 1] = rx | (ry << 8) | (tid << 16);
                }
            }
        }
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

/// The out buffer after `draw` of the emulated object module `m` (planned as `plan`), every object
/// threadgroup its threads fill a workgroup of the whole size, the draw's region flagged
/// `lower::dispatch::NONUNIFORM` as the binder flags a `drawMeshThreads`, with `lanes`-wide
/// subgroups under Vulkan's barrier rule, identical in both scheduling orders.
fn run_draw(plan: &Plan, m: &TypedModule, draw: Draw, lanes: SubgroupSize) -> Vec<u32> {
    let [tx, ty] = draw.threads;
    let [lx, ly] = draw.l;
    let groups = [tx.div_ceil(lx), ty.div_ceil(ly), 1];
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip([lx, ly, 1]) {
        spec.insert(SpecKey::ObjectLocalSize(a), u64::from(v));
    }
    spec.insert(SpecKey::ObjectFlatLocalSize, u64::from(lx * ly));
    spec.insert(SpecKey::ObjectMultiRegion, 1);
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        let PushLoc::Push { offset } = f.loc else {
            panic!("a field outside the push block {f:?}");
        };
        match f.field {
            PushField::DispatchRegion => {
                let region = crate::lower::dispatch::partial_region_bytes([tx, ty, 1], [lx, ly, 1])
                    .expect("a region");
                let at = offset as usize;
                push[at..at + region.len()].copy_from_slice(&region);
            }
            PushField::SlotBindOffset(_) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Object,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let objects = u64::from(groups[0] * groups[1]);
    let slot_words = work::GRID + plan.families.mesh.payload_words;
    let mut header = vec![0u32; work::words(objects, slot_words) as usize];
    header[work::SLOT_WORDS as usize] = slot_words;
    header[work::OBJECT_GROUPS as usize] = u32::try_from(objects).expect("a word");
    let mut out = vec![POISON; 4 + 2 * (tx * ty) as usize];
    out[0] = 0;
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: slot(0),
                address: None,
                bytes: bytes(&out),
            },
            Buffer {
                slot: slot(1),
                address: None,
                bytes: bytes(&CTL),
            },
            Buffer {
                slot: Some(OBJECT_WORK_SLOT),
                address: None,
                bytes: bytes(&header),
            },
            Buffer {
                slot: Some(OBJECT_RECORDS_SLOT),
                address: None,
                bytes: bytes(&[0; 4]),
            },
        ],
        push,
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: groups,
            subgroup: lanes,
            order,
            barriers: BarrierRule::Vulkan,
            ..Config::default()
        };
        let o = run(m, &cfg, res.clone()).unwrap_or_else(|e| panic!("{draw:?} {lanes:?}: {e}"));
        outs.push(o.buffers);
    }
    assert_eq!(
        outs[0][0], outs[1][0],
        "{draw:?} {lanes:?}: order independent"
    );
    words(&outs[0][0])
}

/// Each object function of the fixture, compiled as a compute entry on every emulating profile,
/// runs each of [`DRAWS`] on the interpreter under Vulkan's barrier rule and answers the reading
/// where the threads that reach a barrier synchronize among themselves: the invocations past a
/// partial threadgroup's threads reach every barrier, count nothing and write nothing.
#[test]
fn partial_object_threadgroups_wait_at_barriers_as_metal_runs_them() {
    let Some(bc) = fixture("mesh_partial_barrier") else {
        return;
    };
    for (stem, p) in emulating_profiles() {
        for entry in OBJECTS {
            let what = format!("{stem} {entry}");
            let (plan, m) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(plan.families.mesh.emulated, "{what}: a compute entry");
            for draw in DRAWS {
                for lanes in [SubgroupSize::W8, SubgroupSize::W32] {
                    let got = run_draw(&plan, &m, draw, lanes);
                    assert_eq!(got, expected(entry, draw), "{what} {draw:?} {lanes:?}");
                }
            }
        }
    }
}
