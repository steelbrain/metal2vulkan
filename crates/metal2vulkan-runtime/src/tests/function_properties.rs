//! A function's `MTLFunction` properties are computed once, however many getters read them
//! (`library.rs`: `Function::properties`).
//!
//! # Why this is a count and not a timing
//!
//! `functionType`, `patchType`, `patchControlPointCount`, `vertexAttributes` and
//! `stageInputAttributes` each answer one field of a single `FunctionProperties`, and computing
//! that for a function whose library is a module decodes the function's AIR. Computing it per
//! getter decodes the same bitcode once per getter to answer one `MTLFunction`, and a specialized
//! function pays twice over, because the specialized form computes the unspecialized properties
//! first and specializes the AIR again. Metal answers each getter from the object it already made.
//!
//! The conformance suite measured the gap this left: over its stored whole-set pass on an M3 Max,
//! the `function_facts` call cost 48.4x what Metal's did and `function_facts_from_descriptor`
//! 33.0x, both of them entirely the caller's own CPU with nothing blocked. The one case of the
//! family whose function needs no decode -- a library's visible function, whose properties are a
//! constant -- read 0.2x over the same pass, which is what said the decode was the cost rather
//! than the call.
//!
//! A ratio like that is a time, and a time is not a fact a check can hold on a host running other
//! work. The count is: however many getters read the properties, they must be computed **once**,
//! which is deterministic on any host under any load. Measured with the memo bypassed, the ten
//! reads each test makes below computed them ten times -- and twenty AIR decodes for the
//! specialized function, which decodes once for the unspecialized properties and again to
//! specialize them. The suite's own `function_facts` reads each getter once and the constants
//! dictionary too, so its call decoded one function's AIR six times over.

use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues, Library};
use metal2vulkan::metal::DataType;
use std::path::Path;

/// The runtime library of the fixture `name`, loaded as the device API loads AIR: through the
/// translator, so every function is module-sourced and its properties come from its AIR, which is
/// the shape the conformance cases measured.
fn library(device: &crate::Device, name: &str) -> crate::Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{name}.ll"));
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    device.new_library_with_compile_library(Library::from_air(&bitcode).unwrap())
}

/// Read all five property getters and answer what each said, so a getter cannot be dropped from
/// the check by being forgotten: the assertion below is on the count, and this is what drives it.
fn read_every_property(function: &crate::Function) {
    let function_type = function.function_type().unwrap();
    let patch_type = function.patch_type().unwrap();
    let patch_control_point_count = function.patch_control_point_count().unwrap();
    let vertex_attributes = function.vertex_attributes().unwrap();
    let stage_input_attributes = function.stage_input_attributes().unwrap();
    // Read again: a memo that answered the first five from one computation and then recomputed
    // for a sixth reader would still be redundant work, and the counts below would not see it
    // without this.
    assert_eq!(function.function_type().unwrap(), function_type);
    assert_eq!(function.patch_type().unwrap(), patch_type);
    assert_eq!(
        function.patch_control_point_count().unwrap(),
        patch_control_point_count
    );
    assert_eq!(function.vertex_attributes().unwrap(), vertex_attributes);
    assert_eq!(
        function.stage_input_attributes().unwrap(),
        stage_input_attributes
    );
}

/// A vertex function with attributes, the shape `function-facts-vertex-attributes` reads.
#[test]
fn every_getter_of_a_function_shares_one_computation() {
    let device = crate::Device::system_default().unwrap();
    let library = library(&device, "ob3_vertex_input");
    let function = library.new_function_with_name("ob3_vertex_input").unwrap();
    assert_eq!(
        function.properties_computed(),
        0,
        "making a function computes no properties: Metal's does not either"
    );
    read_every_property(&function);
    assert_eq!(
        function.properties_computed(),
        1,
        "ten reads of the five properties decode the function's AIR once"
    );
}

/// The same function specialized, which is the costlier half: the specialized form computes the
/// unspecialized properties and then specializes the AIR, so an unmemoized read did two decodes a
/// getter. `has_uv` is `[[function_constant(5)]]` of the conformance fixture, which gates the `uv`
/// attribute; `function-facts-specialized-by-name` is the case that reads it.
#[test]
fn every_getter_of_a_specialized_function_shares_one_computation() {
    let device = crate::Device::system_default().unwrap();
    let library = library(&device, "function_constant_interface");
    let constants = FunctionConstantValues::defaults().with_name(
        "has_uv",
        FunctionConstantValue::new(DataType::Bool, vec![1]),
    );
    let function = library
        .new_function_with_name_constant_values("fc_vertex_input_to_output", &constants)
        .unwrap();
    assert_eq!(function.properties_computed(), 0);
    read_every_property(&function);
    assert_eq!(
        function.properties_computed(),
        1,
        "a specialized function specializes its AIR once, not once a getter"
    );
}

/// A function object is the unit the memo belongs to, as it is the unit Metal's `MTLFunction` is:
/// two function objects of one library each compute their own, and a clone shares the one it was
/// cloned from, because a clone is the same `MTLFunction`.
#[test]
fn the_memo_belongs_to_the_function_object() {
    let device = crate::Device::system_default().unwrap();
    let library = library(&device, "ob3_vertex_input");
    let first = library.new_function_with_name("ob3_vertex_input").unwrap();
    let second = library.new_function_with_name("ob3_vertex_input").unwrap();
    first.function_type().unwrap();
    assert_eq!(first.properties_computed(), 1);
    assert_eq!(
        second.properties_computed(),
        0,
        "a second function object has its own memo"
    );
    let clone = first.clone();
    clone.vertex_attributes().unwrap();
    assert_eq!(
        first.properties_computed(),
        1,
        "a clone is the same MTLFunction and answers from the memo it shares"
    );
}
