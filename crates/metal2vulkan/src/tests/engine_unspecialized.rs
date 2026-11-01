//! Whether Metal makes no pipeline state of a stage function made unspecialized
//! (`FunctionDescriptor::needs_specialization`), against what the probe
//! `undefined_function_constant_values.m` (the conformance crate's `probes/`) measured on an Apple
//! M3 Max under macOS 27.0.1: none of a function made by `newFunctionWithName:` that declares a
//! function constant, at every stage, and one of every function made with constant values, none
//! among them too, or by `newFunctionWithDescriptor:`.
//!
//! The owned fixture `engine/unspecialized_stage_functions`: a kernel reading its optional constant
//! only where it is defined, a kernel whose constant gates an argument, and vertex, fragment, tile,
//! mesh and object functions reading one, beside kernel, vertex, fragment and mesh functions that
//! declare none, in one module. `llvm-as` assembles it; a missing tool skips visibly.

use crate::budget::Budget;
use crate::library::{FunctionConstantValue, FunctionConstantValues, Library};
use crate::metal::DataType;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::{FunctionDescriptor, StageEntries};

/// The functions that declare a constant: the probe's `ko`, `kg`, `vertex`, `fragment`, `tile`,
/// `mesh` and `object` shapes.
const DECLARING: [&str; 7] = [
    "unspecialized_guarded",
    "unspecialized_gate",
    "unspecialized_vertex",
    "unspecialized_fragment",
    "unspecialized_tile",
    "unspecialized_mesh",
    "unspecialized_object",
];

/// The functions that declare none: the probe's `kn`, `plain` and `mesh-plain` shapes.
const FREE: [&str; 4] = [
    "unspecialized_free",
    "unspecialized_vertex_free",
    "unspecialized_fragment_free",
    "unspecialized_mesh_free",
];

/// The fixture's library, or `None` (skipped) without `llvm-as`.
fn library() -> Option<Library> {
    let path = engine_dir().join("unspecialized_stage_functions.ll");
    match llvm_assemble(&path) {
        Ok(bitcode) => Some(Library::from_air(&bitcode).expect("a library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fixture: {why}"),
    }
}

/// Whether `descriptor` needs specialization, read through `entries`.
fn needs(descriptor: &FunctionDescriptor, entries: &StageEntries) -> bool {
    descriptor
        .needs_specialization(&Budget::standard(), entries)
        .expect("an answer")
}

/// A function made by name that declares a constant needs specialization at every stage, whether
/// it reads its constant only where it is defined or the constant gates an argument; one that
/// declares none does not (the probe's `by-name` cases). Every answer reads the one module decode
/// the first made.
#[test]
fn a_function_made_by_name_that_declares_a_constant_needs_specialization() {
    let Some(library) = library() else { return };
    let entries = StageEntries::new();
    for name in DECLARING {
        let by_name = FunctionDescriptor::new(library.function(name).unwrap());
        assert!(needs(&by_name, &entries), "{name}");
    }
    for name in FREE {
        let by_name = FunctionDescriptor::new(library.function(name).unwrap());
        assert!(!needs(&by_name, &entries), "{name}");
    }
    assert_eq!(entries.decodes(), (1, 10));
}

/// No specialized function needs it: made with no constant value, by descriptor
/// (`specialized`), or with a value, whatever `specialized` says (the probe's `empty` and
/// `descriptor` cases). Decided without decoding anything.
#[test]
fn a_specialized_function_needs_no_specialization() {
    let Some(library) = library() else { return };
    let entries = StageEntries::new();
    let valued = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, 3u32.to_le_bytes()),
    );
    for name in DECLARING.iter().chain(&FREE) {
        let function = library.function(name).unwrap();
        let specialized = FunctionDescriptor::new(function.clone()).with_specialized(true);
        assert!(!needs(&specialized, &entries), "{name}");
        let mut with_values = FunctionDescriptor::new(function);
        with_values.constants = valued.clone();
        assert!(!needs(&with_values, &entries), "{name} with a value");
    }
    assert_eq!(entries.decodes(), (0, 0));
}
