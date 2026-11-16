//! The Metal executor's mesh draws over the owned fixtures `mesh_draws` and `mesh_barrier`, held
//! to what Metal's own probe of the same functions drew ([`crate::mesh_cases`]).

use super::*;
use crate::mesh_cases::{self, MeshFixtureCase};

/// `fixture`'s selected output on the Metal device, run as compiled over its checked case: the
/// committed public row, its reflection and interface, and the companion the checker derived.
fn executed(fixture: &MeshFixtureCase) -> Vec<u8> {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let checked = crate::check::check_case(&root, fixture.case.clone())
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    let shader = &checked.primary;
    let resources = LiteralResources::prepare(&shader.case).unwrap();
    let source = OracleSource {
        row: &shader.source,
        entry_ll: shader.entry_ll(),
        air: platform::shader_air(&shader.source).unwrap().bitcode.into(),
        bound: OracleBound::LoopFree,
        companion: shader.companion.as_ref(),
    };
    platform::execute(
        &shader.case,
        source,
        &resources,
        &shader.reflection,
        &shader.interface,
        &shader.linked_functions,
        &BTreeMap::new(),
    )
    .unwrap()
}

/// A mesh function without an object function, drawn as `drawMeshThreadgroups:` of 3 by 2
/// threadgroups of 4 threads ([`mesh_cases::grid`]).
#[test]
fn a_mesh_case_draws_metal_s_grid() {
    let fixture = mesh_cases::grid();
    assert_eq!(executed(&fixture), fixture.metal);
}

/// An object case through its mesh partner, its payload and grid ([`mesh_cases::object`]).
#[test]
fn an_object_case_draws_through_its_mesh_partner() {
    let fixture = mesh_cases::object();
    assert_eq!(executed(&fixture), fixture.metal);
}

/// A mesh case without rasterization: no fragment function, and the mesh function's device write
/// lands ([`mesh_cases::device_write`]).
#[test]
fn a_mesh_case_without_rasterization_observes_its_device_writes() {
    let fixture = mesh_cases::device_write();
    assert_eq!(executed(&fixture), fixture.metal);
}

/// A mesh case observed through the generated observer fragment in its fragment partner's place:
/// the primitive data the fixture's own fragment function writes to its first target
/// ([`mesh_cases::observed`]).
#[test]
fn a_mesh_case_draws_through_its_generated_observer() {
    let fixture = mesh_cases::observed();
    assert_eq!(executed(&fixture), fixture.metal);
}

/// An object case observed through its launch: the generated mesh function draws a row for each
/// mesh threadgroup the object function launches, with the grid and the payload
/// ([`mesh_cases::launched`]).
#[test]
fn an_object_case_draws_its_launch_through_the_generated_mesh_function() {
    let fixture = mesh_cases::launched();
    assert_eq!(executed(&fixture), fixture.metal);
}

/// Each payload-reading mesh function of the owned fixtures under the generated object function of
/// its case's object launch: the payload it writes and the grid it launches, through a fragment
/// partner or a generated observer ([`mesh_cases::payload`] and its siblings).
#[test]
fn a_mesh_case_draws_the_payload_its_generated_object_function_writes() {
    for fixture in [
        mesh_cases::payload(),
        mesh_cases::payload_barrier(),
        mesh_cases::payload_order(),
        mesh_cases::payload_outputs(),
    ] {
        assert_eq!(executed(&fixture), fixture.metal, "{}", fixture.case.name);
    }
}
