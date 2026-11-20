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

use super::engine_mesh::{device, fixture, mesh_profiles};
use crate::bind::{BindState, DrawCall};
use crate::budget::Budget;
use crate::library::Library;
use crate::pipeline::descriptor::{FunctionDescriptor, MeshPipelineDescriptor, StageSource};
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

/// An object function whose body waits at a threadgroup barrier draws a `drawMeshThreads` that
/// leaves a partial object threadgroup: its pipeline records
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
    for (stem, _) in mesh_profiles() {
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
            let groups = match &cmds.draw {
                crate::vk::DrawCommand::DrawMeshTasks {
                    group_count_x,
                    group_count_y,
                    group_count_z,
                } => [*group_count_x, *group_count_y, *group_count_z],
                other => panic!("{what}: {other:?}"),
            };
            assert_eq!(groups, [2, 1, 1], "{what}");
        }
    }
}
