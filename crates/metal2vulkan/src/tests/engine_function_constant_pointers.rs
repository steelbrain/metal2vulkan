//! A function constant reached through any pointer path is the same constant: omitted, it is
//! undefined and the function compiles; supplied, its value is what the shader reads.
//!
//! The old engine's `tests/function_constant_pointers.rs` (deleted with it in 0.2.0) pinned this
//! for the paths a front end and
//! the optimizer produce: a direct load, a bitcast alias, a byte `getelementptr`, a call that
//! takes the constant's address, and an address kept in a private slot. Before, a path the
//! specializer did not follow left the load in place and the module read an undefined constant.
//! The engine specializes in `air::canon`, whose own tests use a direct load
//! (`a_surviving_value_of_an_unsupplied_constant_is_zero`); these tests run each path
//! through the public API:
//!
//! - Without a value, each path compiles.
//! - With the values 0 and 29, each path compiles, and what the two compiles read differs. Four
//!   paths fold the value into the module's words. The address kept in a private slot does not
//!   fold: the module reads the constant through the constant blob, whose bytes carry the supplied
//!   value, so the words are equal and the blobs differ.
//! - A byte `getelementptr` to a lane and the typed vector `getelementptr` to the same lane give
//!   the same final words, for every lane of `uint2`, `uint3` and `uint4`.
//!
//! The old file's third test (a `bitcast` does not stand in for `addrspacecast`, `ptrtoint` or
//! `inttoptr`) is the IR verifier's rule in the engine, tested in `air::ir::verify`'s tests.
//!
//! `llvm-as` assembles the templates; without it the tests skip visibly.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Error;
use crate::library::{FunctionConstantValue, FunctionConstantValues, Library};
use crate::metal::DataType;
use crate::pipeline::testing::{link_standalone, llvm_assemble, AssembleError};
use crate::pipeline::{FunctionDescriptor, LoweringInputs};
use crate::Budget;

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"
"#;

const TRAILER: &str = r#"!air.version = !{!93}
!air.language_version = !{!94}
!93 = !{i32 2, i32 8, i32 0}
!94 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A fragment function returning `%value` to a `uint` render target, where `BODY` computes
/// `%value` from the `uintN` function constant 7 (`N` is `LANES`), and `HELPER` is any function
/// `BODY` calls.
const TEMPLATE: &str = r#"@value.MTL_FC_INIT_7 = internal addrspace(2) externally_initialized constant <LANES x i32> undef, section "air.fc_initializer", align 16

HELPER

define i32 @fragment() {
entry:
BODY
  ret i32 %value
}

!air.fragment = !{!0}
!air.function_constants = !{!4}
!0 = !{ptr @fragment, !1, !2}
!1 = !{!3}
!2 = !{}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{ptr addrspace(2) @value.MTL_FC_INIT_7, !"uintLANES", !"value", i32 7}
"#;

/// Each pointer path to the constant, as `(label, body, helper)`. Each reads lane 1.
const PATHS: [(&str, &str, &str); 5] = [
    (
        "a direct load",
        "  %value = load i32, ptr addrspace(2) @value.MTL_FC_INIT_7, align 4",
        "",
    ),
    (
        "a bitcast alias",
        "  %alias = bitcast ptr addrspace(2) @value.MTL_FC_INIT_7 to ptr addrspace(2)\n  \
         %value = load i32, ptr addrspace(2) %alias, align 4",
        "",
    ),
    (
        "a byte getelementptr",
        "  %alias = getelementptr i8, ptr addrspace(2) @value.MTL_FC_INIT_7, i64 4\n  \
         %value = load i32, ptr addrspace(2) %alias, align 4",
        "",
    ),
    (
        "a call taking the address",
        "  %value = call i32 @read(ptr addrspace(2) @value.MTL_FC_INIT_7)",
        "define internal i32 @read(ptr addrspace(2) %input) {\nentry:\n  \
         %p = getelementptr <2 x i32>, ptr addrspace(2) %input, i64 0, i64 1\n  \
         %v = load i32, ptr addrspace(2) %p, align 4\n  ret i32 %v\n}",
    ),
    (
        "an address kept in a private slot",
        "  %slot = alloca ptr addrspace(2), align 8\n  \
         store ptr addrspace(2) @value.MTL_FC_INIT_7, ptr %slot, align 8\n  \
         %alias = load ptr addrspace(2), ptr %slot, align 8\n  \
         %value = load i32, ptr addrspace(2) %alias, align 4",
        "",
    ),
];

fn source(lanes: u32, body: &str, helper: &str) -> String {
    let text = TEMPLATE
        .replace("LANES", &lanes.to_string())
        .replace("BODY", body)
        .replace("HELPER", helper);
    format!("{HEADER}\n{text}\n{TRAILER}")
}

/// `llvm-as` of `text`, through a scratch file that is removed at once. `None` when `llvm-as` is
/// not installed (the reason is printed).
fn assemble(text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v-function-constant-pointers-{}-{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let assembled = llvm_assemble(&path);
    std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match assembled {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejects an input:\n{why}\n{text}"),
    }
}

fn device() -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/nvidia-rtx5080-laptop.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let description =
        DeviceDescription::from_khronos_json(&text, None).expect("nvidia-rtx5080-laptop");
    Device::new(description, DevicePolicy::default()).expect("nvidia-rtx5080-laptop")
}

/// The `uintN` type of `lanes` lanes.
fn vector_type(lanes: u32) -> DataType {
    match lanes {
        2 => DataType::UInt2,
        3 => DataType::UInt3,
        4 => DataType::UInt4,
        other => panic!("no uint{other}"),
    }
}

/// What a compiled function reads: its final words, linked alone, and its constant blob, the bytes
/// a pointer into `constant` memory that does not fold reads through a storage view.
type Output = (Vec<u32>, Option<Vec<u8>>);

/// The fragment function of `bitcode` compiled with constant 7 set to `values` (or omitted).
fn compile(bitcode: &[u8], device: &Device, values: Option<&[u32]>) -> Result<Output, Error> {
    let library = Library::from_air(bitcode)?;
    let mut descriptor = FunctionDescriptor::new(library.function("fragment")?);
    if let Some(values) = values {
        let lanes = u32::try_from(values.len()).expect("a few lanes");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        descriptor.constants = FunctionConstantValues::defaults()
            .with_index(7, FunctionConstantValue::new(vector_type(lanes), bytes));
    }
    let compiled =
        device.compile_function(&descriptor, &LoweringInputs::new(), &Budget::default())?;
    let words = link_standalone(compiled.artifact(), device.profile())?.1;
    Ok((
        words,
        compiled.artifact().blob.as_deref().map(<[u8]>::to_vec),
    ))
}

#[test]
fn every_pointer_path_compiles_the_omitted_constant_and_reads_the_supplied_one() {
    let device = device();
    for (label, body, helper) in PATHS {
        let Some(bitcode) = assemble(&source(2, body, helper)) else {
            return;
        };
        if let Err(e) = compile(&bitcode, &device, None) {
            panic!("{label}: an omitted constant is undefined, not refused: {e:?}");
        }
        let outputs: Vec<Output> = [0u32, 29]
            .iter()
            .map(|&value| {
                compile(&bitcode, &device, Some(&[value, value]))
                    .unwrap_or_else(|e| panic!("{label}, value {value}: {e:?}"))
            })
            .collect();
        assert_ne!(
            outputs[0], outputs[1],
            "{label}: the supplied value must reach the result"
        );
    }
}

#[test]
fn a_byte_lane_address_is_the_typed_lane_address() {
    let device = device();
    for lanes in [2u32, 3, 4] {
        let values = &[11u32, 29, 47, 61][..lanes as usize];
        for lane in 0..lanes {
            let compiled = |path: &str, body: String| {
                let bitcode = assemble(&source(lanes, &body, ""))?;
                Some(
                    compile(&bitcode, &device, Some(values))
                        .unwrap_or_else(|e| panic!("uint{lanes} lane {lane}, {path}: {e:?}")),
                )
            };
            let Some(byte) = compiled(
                "byte",
                format!(
                    "  %alias = getelementptr i8, ptr addrspace(2) @value.MTL_FC_INIT_7, i64 {}\n  \
                     %value = load i32, ptr addrspace(2) %alias, align 4",
                    lane * 4
                ),
            ) else {
                return;
            };
            let typed = compiled(
                "typed",
                format!(
                    "  %alias = getelementptr <{lanes} x i32>, ptr addrspace(2) \
                     @value.MTL_FC_INIT_7, i64 0, i64 {lane}\n  \
                     %value = load i32, ptr addrspace(2) %alias, align 4"
                ),
            );
            assert_eq!(Some(byte), typed, "uint{lanes} lane {lane}");
        }
    }
}
