//! Visible-function tables and function pointers end to end.
//!
//! The owned fixtures `engine/visible_table_caller` (a kernel calling entry `tid % size` of a
//! `visible_function_table<uint(uint, uint)>`, and one calling visible functions through a helper
//! that takes a function pointer) and `engine/visible_table_functions` (the visible
//! functions `table_add`, `table_mul`, `table_sub` of the table's type and `table_half` of
//! another) are what Apple's compiler emits. Through the public API the caller compiles whether or
//! not the pipeline links a function a table can hold: an entry holds a function handle, and a
//! handle exists only for a linked function, so with none linked every entry is null and only the
//! shader's own guards run. The compiled kernels validate on every committed profile and compute,
//! on the CPU interpreter, what Metal does for every defined entry (a function, a null entry, a
//! null or empty table); an entry holding a function of another type yields zero Metal's undefined
//! call. No GPU is used.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::path::Path;

use crate::bind::FunctionTableWords;
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Error;
use crate::library::{FunctionConstantValues, Library};
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, profile, typed_module_linked,
    AssembleError,
};
use crate::pipeline::{
    BufferKind, ComputePipelineDescriptor, FunctionDescriptor, FunctionHandle, LinkedFunction,
    LinkedFunctions, SlotInterface, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;

const CALLER: &str = "visible_table_caller";
const TABLE: &str = "visible_table_dispatch";
const POINTER: &str = "visible_pointer_dispatch";
const FUNCTIONS: &str = "visible_table_functions";
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

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The four functions of the functions module, in `functions` in [`LINKED`] order.
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
    stem: &str,
) -> Result<std::sync::Arc<crate::library::FunctionArtifact>, Error> {
    let p = profile(stem, DevicePolicy::default()).expect("profile");
    compile_bitcode_linked(
        caller,
        EntryRef::Name(entry),
        (&FunctionConstantValues::defaults(), linked),
        &p,
        &LowerInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// The table word of the `n`-th linked function: the handles number `functions` from 2.
fn word(n: u32) -> u32 {
    n + 2
}

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn buffer(index: u32, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    }
}

/// Run `module` over `n` threads with `table` bound at buffer 1 (none: an unbound slot, which
/// the binder fills with a zero-filled neutral): the words it writes.
fn dispatch(module: &crate::spirv::ir::TypedModule, n: u32, table: Option<&[u8]>) -> Vec<u32> {
    let table = table.map_or_else(|| vec![0u8; 16], <[u8]>::to_vec);
    let res = Resources {
        buffers: vec![buffer(0, vec![0; n as usize * 4]), buffer(1, table)],
        push: vec![0; 128],
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [n, 1, 1],
        ..Config::default()
    };
    let out = run(module, &cfg, res).unwrap_or_else(|e| panic!("{e:?}"));
    words(&out.buffers[0])
}

/// A table call that no linked function can answer compiles: a handle exists only for a linked
/// function, so with nothing of the call's type linked every entry's word is null, the switch is
/// unreachable, and the shader's `f == nullptr` guard is what runs. The module validates, and on
/// the CPU interpreter a two-entry table of null entries answers `0xfffffffd` for every thread --
/// what this M3 Max (macOS 27.0.1) observed from Metal for the same pipeline and table. A linked
/// function of another type is no answer either, and compiles the same way.
#[test]
fn a_table_call_without_a_function_of_its_type_reads_null_entries() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let half = [LinkedSource::new("table_half", &functions, &none)];
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let two = FunctionTableWords::new(2).to_bytes();
    let empty: &[LinkedSource<'_>] = &[];
    for linked in [empty, &half] {
        let a = compile(&caller, TABLE, linked, "moltenvk-m3").expect("compiles");
        link_and_validate(&a, &p).expect("L3");
        let (_, m) = typed_module_linked(&caller, TABLE, &p, linked).expect("module");
        assert_eq!(dispatch(&m, 12, Some(&two)), [0xffff_fffd; 12]);
    }
}

/// With the functions linked, both kernels compile, link, and pass L3 on every committed
/// profile.
#[test]
fn table_and_pointer_calls_validate_on_every_profile() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in [TABLE, POINTER] {
            let a = compile_bitcode_linked(
                &caller,
                EntryRef::Name(entry),
                (&none, &linked),
                &p,
                &LowerInputs::new(),
                &mut meter(),
                SPIRV_WORDS,
            )
            .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
        }
    }
}

/// The interface says which buffer slot holds a table: the table the kernel reads as its words is a
/// `VisibleFunctionTable` slot, read-only, and its output a `Buffer` slot. A kernel without a table
/// states only buffers.
#[test]
fn the_interface_says_which_slot_holds_a_table() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    let kinds = |entry: &str| -> Vec<(u32, BufferKind, bool)> {
        let a = compile(&caller, entry, &linked, "moltenvk-m3").expect("compiles");
        a.interface
            .slots
            .iter()
            .filter_map(|(slot, s)| match s {
                SlotInterface::Buffer(b) => Some((slot.index, b.kind, b.access.write)),
                _ => None,
            })
            .collect()
    };
    assert_eq!(
        kinds(TABLE),
        [
            (0, BufferKind::Buffer, true),
            (1, BufferKind::VisibleFunctionTable, false),
        ]
    );
    assert_eq!(kinds(POINTER), [(0, BufferKind::Buffer, true)]);
}

/// Each thread calls entry `tid % 5` of a table holding `table_add`, `table_mul`, a null
/// entry, `table_sub`, and `table_half` (another type) with `(tid, 3)`.
#[test]
fn a_table_call_calls_the_entry_the_table_holds() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let (_, m) = typed_module_linked(&caller, TABLE, &p, &linked).expect("module");
    const N: u32 = 40;
    let mut table = FunctionTableWords::new(5);
    let raw = [word(0), word(1), 0, word(2), word(3)];
    let expect_words = [1, 5, word(0), word(1), 0, word(2), word(3)];
    // The words the handles give are the ones the switch cases test.
    let lib = Library::from_air(&functions).expect("library");
    let linked_api = LINKED.iter().fold(LinkedFunctions::new(), |l, name| {
        l.with_function(lib.function(name).expect("function"))
    });
    for (i, &w) in raw.iter().enumerate() {
        let handle = LINKED
            .iter()
            .enumerate()
            .find(|&(k, _)| word(k as u32) == w)
            .map(|(_, name)| {
                linked_api
                    .function_handle(&LinkedFunction::new(lib.function(name).expect("f")))
                    .expect("a handle")
            });
        assert_eq!(handle.map(FunctionHandle::word).unwrap_or(0), w);
        assert!(table.set(i as u32, handle));
    }
    assert_eq!(table.words(), expect_words);
    let out = dispatch(&m, N, Some(&table.to_bytes()));
    let expect = (0..N)
        .map(|tid| match tid % 5 {
            0 => tid + 3,
            1 => tid * 3,
            2 => 0xffff_fffd,
            3 => tid.wrapping_sub(3),
            // A function of another type: the default, zero.
            _ => 0,
        })
        .collect::<Vec<_>>();
    assert_eq!(out, expect);
}

/// A null table (nothing bound: the zero-filled neutral) and an empty table are observed as
/// Metal observes them.
#[test]
fn null_and_empty_tables_are_observed() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let (_, m) = typed_module_linked(&caller, TABLE, &p, &linked).expect("module");
    assert_eq!(dispatch(&m, 4, None), [0xffff_ffff; 4]);
    let empty = FunctionTableWords::new(0).to_bytes();
    assert_eq!(dispatch(&m, 4, Some(&empty)), [0xffff_fffe; 4]);
}

/// A helper called with a visible function's address calls that function once inlined:
/// `(tid + 5) + tid * 2`.
#[test]
fn a_function_pointer_argument_calls_its_function() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let (_, m) = typed_module_linked(&caller, POINTER, &p, &linked).expect("module");
    const N: u32 = 32;
    let res = Resources {
        buffers: vec![buffer(0, vec![0; N as usize * 4])],
        push: vec![0; 128],
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [N, 1, 1],
        ..Config::default()
    };
    let out = run(&m, &cfg, res).unwrap_or_else(|e| panic!("{e:?}"));
    let expect = (0..N).map(|tid| (tid + 5) + tid * 2).collect::<Vec<_>>();
    assert_eq!(words(&out.buffers[0]), expect);
}

/// Through the public API: `functions` and `binary_functions` hold what a table can call, a
/// function only in `private_functions` has no handle and is no candidate, and the compile
/// succeeds whether or not a function of the call's type is linked.
#[test]
fn the_public_api_links_table_functions() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let caller_lib = Library::from_air(&caller).expect("library");
    let lib = Library::from_air(&functions).expect("library");
    let entry = caller_lib.function(TABLE).expect("entry");
    let f = |name: &str| lib.function(name).expect("function");
    let compile = |linked: LinkedFunctions| {
        let d = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(entry.clone()).with_linked(linked),
        ));
        dev.compile_compute_pipeline(&d, &Budget::standard())
            .map(|_| ())
    };
    let private = LinkedFunctions::new().with_private_function(f("table_add"));
    assert_eq!(
        private.function_handle(&LinkedFunction::new(f("table_add"))),
        None
    );
    assert_eq!(compile(private), Ok(()));
    let mut binary = LinkedFunctions::new().with_function(f("table_half"));
    binary.binary_functions.push(f("table_mul").into());
    assert_eq!(
        binary
            .function_handle(&LinkedFunction::new(f("table_mul")))
            .map(FunctionHandle::word),
        Some(3)
    );
    assert_eq!(compile(binary), Ok(()));
    let all = LINKED
        .iter()
        .fold(LinkedFunctions::new(), |l, name| l.with_function(f(name)));
    assert_eq!(compile(all), Ok(()));
}
