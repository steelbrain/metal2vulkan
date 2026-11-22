//! Visible-function tables read from argument buffers end to end.
//!
//! The owned fixture `engine/uni_argument_tables` holds kernels that call entry `i % size`
//! of a `visible_function_table<uint(uint, uint)>` loaded from an argument buffer, or answer
//! `0xfffffffd` for a null entry, `0xfffffffe` for an empty table, and `0xffffffff` for a null
//! one: through one member, through a member array indexed per thread, and through a per-thread
//! select of two members. The functions are `engine/visible_table_functions`'s, which the
//! pipeline links.
//!
//! - The kernels compile and pass L3 on every committed profile, reading the stage's
//!   function-table snapshot beside its directory. A call nothing linked can answer compiles too:
//!   with no handle to hold, every entry is null.
//! - Through the public API, the plan writes the snapshot (`FunctionTableSnapshot`) and the
//!   stage's directory, whose bytes per dispatch give each listed table's nonzero id its record's
//!   word offset by the listed sizes.
//! - On the CPU interpreter, fed the binder's directory bytes and the snapshot of the tables'
//!   words, every profile's kernels compute what
//!   Metal does: the entry's function (zero for a function of another type), a null entry, an empty
//!   table, and a null table for the null id and for an id no listed table has. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;

use crate::budget::SPIRV_WORDS;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::Error;
use crate::library::FunctionConstantValues;
use crate::lower::access::function_table::{directory_symbol, snapshot_symbol};
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::sym::Stage;
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, AssembleError,
};
use crate::spirv::testing::committed_profiles;

const CALLER: &str = "uni_argument_tables";
const FUNCTIONS: &str = "visible_table_functions";
/// One table member; `k` 3.
const ONE: &str = "argument_table_dispatch";
/// Member `tables[tid & 1]` after a word, entry `tid >> 1`; `k` 5.
const ARRAY: &str = "argument_table_array_dispatch";
/// Member `a` for odd threads, `b` for even, entry `tid >> 1`; `k` 7.
const SELECT: &str = "argument_table_select";
const KERNELS: [&str; 3] = [ONE, ARRAY, SELECT];
/// The table's type's functions, then one of another type, in the order the tests link them.
const LINKED: [&str; 4] = ["table_add", "table_mul", "table_sub", "table_half"];

/// The fixture `stem` assembled.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    }
}

fn fixtures() -> Option<(Vec<u8>, Vec<u8>)> {
    Some((fixture(CALLER)?, fixture(FUNCTIONS)?))
}

/// The four functions of the functions module, in [`LINKED`] order.
fn sources<'a>(functions: &'a [u8], none: &'a FunctionConstantValues) -> Vec<LinkedSource<'a>> {
    LINKED
        .iter()
        .map(|name| LinkedSource::new(name, functions, none))
        .collect()
}

fn compile(
    caller: &[u8],
    entry: &str,
    linked: &[LinkedSource<'_>],
    p: &DeviceProfile,
) -> Result<std::sync::Arc<crate::library::FunctionArtifact>, Error> {
    compile_bitcode_linked(
        caller,
        EntryRef::Name(entry),
        (&FunctionConstantValues::defaults(), linked),
        p,
        &LowerInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// The kernels compile and validate on every committed profile, declaring the stage's snapshot
/// beside the directory.
#[test]
fn argument_tables_compile_on_every_profile() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in KERNELS {
            let a = compile(&caller, entry, &linked, &p)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
            let declares = |slot| a.interface.descriptors.iter().any(|d| d.slot == slot);
            assert!(declares(directory_symbol(Stage::Kernel)), "{stem} {entry}");
            assert!(declares(snapshot_symbol(Stage::Kernel)), "{stem} {entry}");
            let census = &a.strategies.entries;
            assert!(
                census.contains_key("access.function_table.word.snapshot"),
                "{stem} {entry}"
            );
        }
    }
}

/// A call through a table loaded from an argument buffer that no linked function can answer
/// compiles and validates, as a table parameter's call does: with nothing of the call's type
/// linked no entry can hold a handle, so the switch is unreachable. A linked function of another
/// type is no answer either, and compiles the same way.
#[test]
fn an_argument_table_call_without_a_function_of_its_type_compiles() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let p = crate::pipeline::testing::profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES)
        .expect("profile");
    let none = FunctionConstantValues::defaults();
    let half = [LinkedSource::new("table_half", &functions, &none)];
    let empty: &[LinkedSource<'_>] = &[];
    for entry in KERNELS {
        for linked in [empty, &half] {
            let a = compile(&caller, entry, linked, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{entry}: L3: {e}"));
        }
    }
}

pub(super) fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}
