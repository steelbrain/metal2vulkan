//! Every getter in [`ROWS`] derives its answer **once**, however many getters read it and however
//! often (`derived.rs`, and the memos in `library.rs`).
//!
//! # Why this is a list, and why it is a count
//!
//! A getter on a Metal object answers from the object. Several of ours answer from a derivation
//! over the object's inputs, and the expensive ones decode the function's AIR: `functionType`,
//! `patchType`, `patchControlPointCount`, `vertexAttributes` and `stageInputAttributes` each
//! answer one field of a single `FunctionProperties`, `functionConstantsDictionary` answers from
//! the same module, `reflectionForFunctionWithName:` from the named function's, and
//! `newArgumentEncoderWithBufferIndex:` from the indexed argument buffer's. A getter that derives
//! per call decodes the same bitcode per call; Metal answers each from the object it already made.
//!
//! One such getter was fixed with one memo and one test, and the question that left was what stops
//! the next one. This is the answer that costs least: a table of getters, each with how many
//! derivations reading it may cost, so a getter added to an object joins it as a **row**, and a
//! getter that loses its memo turns that row red at once. It is what would have caught the five
//! property getters on the day they were written.
//!
//! The assertion is a **count**, never a time. However many getters read a derivation, the object
//! must derive it once: that is a fact of the structure and holds on any host under any load,
//! where a ratio against Metal is a measurement of two machines and moves with both. Each count
//! is of one object made by the test, so it depends on nothing that ran before it in the process,
//! needs no quiet host, no warm cache and no ordering between tests.
//!
//! # What the table does not cover
//!
//! It holds the getters listed, on the shapes listed, and nothing else: a getter nobody adds a row
//! for is unguarded, and a derivation the runtime makes somewhere other than a getter on one of
//! these objects is out of its reach. It is a regression guard, not a census.
//!
//! It also cannot see a derivation that was never routed through a memo, since an uncounted
//! derivation is invisible by construction. What it does see is a memo **removed** from a row it
//! already holds: the count then stops moving and the row fails on the low side, which is why the
//! check is held in both directions and not only above the number.

use crate::{Function, Library};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::DataType;
use std::path::Path;

/// The runtime library of the fixture `name`, loaded as the device API loads AIR: through the
/// translator, so every function is module-sourced and its properties, constants and reflection
/// come from its AIR, which is the shape that pays for a per-call derivation.
fn library(device: &crate::Device, name: &str) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{name}.ll"));
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    )
}

/// The constant values `function_constant_interface`'s gated interface is read under: `has_uv` is
/// its `[[function_constant(5)]]`, which gates the `uv` attribute.
fn uv_constants() -> FunctionConstantValues {
    FunctionConstantValues::defaults().with_name(
        "has_uv",
        FunctionConstantValue::new(DataType::Bool, vec![1]),
    )
}

/// What a row reads the derivation count from once its getters have been read twice over.
enum Counted {
    /// A function's `MTLFunction` properties, which its five property getters share.
    Properties,
    /// A function's `functionConstantsDictionary`.
    ConstantsDictionary,
    /// A function's argument-buffer layouts, one per buffer index.
    ArgumentLayouts,
    /// A library's functions' reflections, one per function name.
    Reflections,
}

/// One row: a getter (or a set of getters over one derivation), the fixture and entry it is read
/// on, what reads it twice over, and how many derivations that may cost.
struct Row {
    /// What the row guards, for the assertion's message.
    getters: &'static str,
    /// The fixture whose library holds the function.
    fixture: &'static str,
    /// The entry in it, and the constant values the function is made with.
    entry: &'static str,
    /// Whether the function is made with [`uv_constants`], which specializes it.
    specialized: bool,
    /// Read the row's getters twice over, so a memo that answered the first read and recomputed
    /// for the second is caught: the counts below would not see it without the second read.
    read: fn(&Library, &Function),
    /// Which count the row reads.
    counted: Counted,
    /// How many derivations reading the getters twice over may cost. Every row is 1: one answer
    /// per object, or per key for a keyed one.
    derivations: u32,
}

/// Read the five property getters, twice over.
fn read_properties(_: &Library, function: &Function) {
    for _ in 0..2 {
        function.function_type().unwrap();
        function.patch_type().unwrap();
        function.patch_control_point_count().unwrap();
        function.vertex_attributes().unwrap();
        function.stage_input_attributes().unwrap();
    }
}

/// Read `functionConstantsDictionary`, twice over.
fn read_constants_dictionary(_: &Library, function: &Function) {
    let first = function.function_constants_dictionary().unwrap();
    assert_eq!(function.function_constants_dictionary().unwrap(), first);
}

/// Make the encoder of the argument buffer at buffer index 0, twice over.
fn read_argument_encoder(_: &Library, function: &Function) {
    for _ in 0..2 {
        function.new_argument_encoder_with_buffer_index(0).unwrap();
    }
}

/// Read the library's reflection of the function by name, twice over.
fn read_reflection(library: &Library, function: &Function) {
    let name = function.name().to_string();
    let first = library.reflection_for_function_with_name(&name).unwrap();
    assert_eq!(
        library.reflection_for_function_with_name(&name).unwrap(),
        first
    );
}

/// The guarded getters. **Add a row when you add a getter that derives**, and give it the
/// derivation count reading it twice over may cost; a getter that may derive more than once a
/// call is a defect, not a larger number here.
const ROWS: &[Row] = &[
    Row {
        getters: "functionType, patchType, patchControlPointCount, vertexAttributes, \
                  stageInputAttributes",
        fixture: "ob3_vertex_input",
        entry: "ob3_vertex_input",
        specialized: false,
        read: read_properties,
        counted: Counted::Properties,
        derivations: 1,
    },
    // The specialized half is the costlier one: the specialized form derives the unspecialized
    // properties and then specializes the AIR, so an unmemoized read did two decodes a getter.
    Row {
        getters: "the five property getters, of a function made with constant values",
        fixture: "function_constant_interface",
        entry: "fc_vertex_input_to_output",
        specialized: true,
        read: read_properties,
        counted: Counted::Properties,
        derivations: 1,
    },
    Row {
        getters: "functionConstantsDictionary",
        fixture: "function_constant_interface",
        entry: "fc_vertex_input_to_output",
        specialized: false,
        read: read_constants_dictionary,
        counted: Counted::ConstantsDictionary,
        derivations: 1,
    },
    Row {
        getters: "newArgumentEncoderWithBufferIndex:",
        fixture: "a3_rog_argument_buffer",
        entry: "a3_rog_argument_buffer",
        specialized: false,
        read: read_argument_encoder,
        counted: Counted::ArgumentLayouts,
        derivations: 1,
    },
    Row {
        getters: "reflectionForFunctionWithName:",
        fixture: "function_constant_interface",
        entry: "fc_vertex_input_to_output",
        specialized: false,
        read: read_reflection,
        counted: Counted::Reflections,
        derivations: 1,
    },
];

/// Every row of [`ROWS`]: making the object derives nothing, and reading its getters twice over
/// derives once.
/// Every row of [`ROWS`] is read, and every row that fails is reported: a change that loses one
/// memo should name its getter, and a change that loses the shared machinery should name all of
/// them, rather than stopping the run at whichever row happens to come first.
#[test]
fn every_listed_getter_derives_once() {
    let device = crate::Device::system_default().unwrap();
    let mut wrong = Vec::new();
    for row in ROWS {
        let library = library(&device, row.fixture);
        let function = match row.specialized {
            true => library
                .new_function_with_name_constant_values(row.entry, &uv_constants())
                .unwrap(),
            false => library.new_function_with_name(row.entry).unwrap(),
        };
        let count = |library: &Library, function: &Function| match row.counted {
            Counted::Properties => function.properties_derived(),
            Counted::ConstantsDictionary => function.constants_dictionary_derived(),
            Counted::ArgumentLayouts => function.argument_layouts_derived(),
            Counted::Reflections => library.reflections_derived(),
        };
        let made = count(&library, &function);
        if made != 0 {
            wrong.push(format!(
                "making the object derived {} {made} times, where Metal's making derives none",
                row.getters
            ));
        }
        (row.read)(&library, &function);
        let read = count(&library, &function) - made;
        if read != row.derivations {
            // Held in BOTH directions, as the conformance suite's slow-call gate is. Over the
            // count is a memo that stopped answering: the getter derives per call again. UNDER it
            // is a derivation that no longer goes through the memo at all, so nothing counts it --
            // which is what deleting a memo outright reads as, and is why the low side fails too
            // rather than being taken for an improvement.
            let how = match read > row.derivations {
                true => "a memo that no longer answers every reader",
                false => "a derivation that no longer goes through its memo, so nothing counts it",
            };
            wrong.push(format!(
                "reading {} twice over counted {read} derivations, not {}: {how}",
                row.getters, row.derivations
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} rows derive more than once:\n  {}",
        wrong.len(),
        ROWS.len(),
        wrong.join("\n  ")
    );
}

/// A function object is the unit a function's memo belongs to, as it is the unit Metal's
/// `MTLFunction` is: two function objects of one library each derive their own, and a clone
/// shares the one it was cloned from, because a clone is the same `MTLFunction`.
#[test]
fn a_functions_memo_belongs_to_the_function_object() {
    let device = crate::Device::system_default().unwrap();
    let library = library(&device, "ob3_vertex_input");
    let first = library.new_function_with_name("ob3_vertex_input").unwrap();
    let second = library.new_function_with_name("ob3_vertex_input").unwrap();
    first.function_type().unwrap();
    first.function_constants_dictionary().unwrap();
    assert_eq!(first.properties_derived(), 1);
    assert_eq!(first.constants_dictionary_derived(), 1);
    assert_eq!(
        (
            second.properties_derived(),
            second.constants_dictionary_derived()
        ),
        (0, 0),
        "a second function object has its own memos"
    );
    let clone = first.clone();
    clone.vertex_attributes().unwrap();
    clone.function_constants_dictionary().unwrap();
    assert_eq!(
        (
            first.properties_derived(),
            first.constants_dictionary_derived()
        ),
        (1, 1),
        "a clone is the same MTLFunction and answers from the memos it shares"
    );
}

/// A library's reflection memo is keyed by name: reading two functions' reflections derives two,
/// and reading either again derives none.
#[test]
fn a_librarys_reflection_memo_is_keyed_by_name() {
    let device = crate::Device::system_default().unwrap();
    let library = library(&device, "function_constant_interface");
    for name in ["fc_vertex_input_to_output", "fc_vertex_gated_outputs"] {
        library.reflection_for_function_with_name(name).unwrap();
    }
    assert_eq!(library.reflections_derived(), 2, "one derivation per name");
    for name in ["fc_vertex_input_to_output", "fc_vertex_gated_outputs"] {
        library.reflection_for_function_with_name(name).unwrap();
    }
    assert_eq!(
        library.reflections_derived(),
        2,
        "a name read again answers from the memo"
    );
}
