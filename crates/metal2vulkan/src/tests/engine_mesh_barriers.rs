//! Object and mesh functions that wait at threadgroup barriers.
//!
//! The owned fixture `engine/mesh_partial_barrier` holds three object functions, whose barriers
//! stand in uniform control flow and in a loop, after a divergent early return, and under a
//! thread-dependent condition, and a mesh function that returns before its barrier. On every
//! committed profile with `VK_EXT_mesh_shader`, each object function runs a `drawMeshThreads`
//! partial object threadgroup (`RuntimeFact::PartialObjectThreadgroups`) and is walked predicated
//! from its threadgroup's threads (`lower::simd::divergent`'s partial object threadgroups), the mesh
//! function is walked predicated for its divergent barrier, and no workgroup barrier of a body
//! stands in a block the body's own uniformity does not prove uniform
//! (`spirv::testing::barriers`). No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use super::engine_mesh::{fixture, mesh_profiles};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::Stage;
use crate::pipeline::testing::typed_module;
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
    for (stem, p) in &mesh_profiles() {
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
