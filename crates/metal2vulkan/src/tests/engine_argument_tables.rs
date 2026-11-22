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

use crate::bind::{
    ArgumentFunctionTable, BindState, DescriptorBytes, Dispatch, FunctionTableSnapshot,
    FunctionTableWords, HeapPlacement, WriteSource,
};
use crate::budget::{Budget, SPIRV_WORDS};
use crate::cache::VariantKey;
use crate::device::profile::BufferAccess;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{Error, Feature};
use crate::library::{FunctionConstantValues, Library};
use crate::lower::access::function_table::{directory_symbol, snapshot_symbol};
use crate::lower::select::LowerInputs;
use crate::memory::handles::TABLE_ENTRY_BYTES;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, typed_module_linked, AssembleError,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, FunctionHandle, LinkedFunction,
    LinkedFunctions, StageSource,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;
use crate::vk;

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

/// The listed tables' resource ids (`T`'s high word is nonzero), and one no table has.
const T: u64 = 0x1_0000_0010;
const U: u64 = 0x20;
const E: u64 = 0x30;
const UNKNOWN: u64 = 0x40;

/// The tables' entries, by function in [`LINKED`] order (`None` a null entry).
const T_ENTRIES: [Option<usize>; 5] = [Some(0), Some(1), None, Some(2), Some(3)];
const U_ENTRIES: [Option<usize>; 2] = [Some(2), Some(0)];

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

/// The pipeline of kernel `entry` with the four functions linked, through the public API.
fn pipeline(dev: &Device, caller: &[u8], functions: &[u8], entry: &str) -> ComputePipeline {
    let caller = Library::from_air(caller).expect("library");
    let lib = Library::from_air(functions).expect("library");
    let linked = LINKED.iter().fold(LinkedFunctions::new(), |l, name| {
        l.with_function(lib.function(name).expect("function"))
    });
    let d = ComputePipelineDescriptor::new(StageSource::Function(
        FunctionDescriptor::new(caller.function(entry).expect("entry")).with_linked(linked),
    ));
    dev.compile_compute_pipeline(&d, &Budget::standard())
        .expect("pipeline")
}

/// The handles of the four linked functions, in [`LINKED`] order, through the public API.
fn handles(functions: &[u8]) -> Vec<FunctionHandle> {
    let lib = Library::from_air(functions).expect("library");
    let linked = LINKED.iter().fold(LinkedFunctions::new(), |l, name| {
        l.with_function(lib.function(name).expect("function"))
    });
    LINKED
        .iter()
        .map(|name| {
            linked
                .function_handle(&LinkedFunction::new(lib.function(name).expect("function")))
                .expect("a handle")
        })
        .collect()
}

/// The words of a table holding `entries`.
fn table_bytes(handles: &[FunctionHandle], entries: &[Option<usize>]) -> Vec<u8> {
    let mut t = FunctionTableWords::new(entries.len() as u32);
    for (i, e) in entries.iter().enumerate() {
        assert!(t.set(i as u32, e.map(|f| handles[f])));
    }
    t.to_bytes()
}

/// The bind state listing `T`, `U`, and `E` (elements 1, 2, 3; snapshot offsets 1, 9, 14).
fn state() -> BindState {
    BindState {
        function_tables: [(T, T_ENTRIES.len()), (U, U_ENTRIES.len()), (E, 0)]
            .map(|(id, size)| ArgumentFunctionTable {
                resource_id: id,
                size: size as u32,
                address: 0,
            })
            .to_vec(),
        ..Default::default()
    }
}

pub(super) fn one_group() -> Dispatch {
    Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    }
}

/// The `(set, binding)` of the pipeline's kernel synthetic binding `binding`.
fn synthetic(p: &ComputePipeline, binding: SyntheticBinding) -> Option<(u32, u32)> {
    p.layout.sets[0]
        .bindings
        .iter()
        .find(|b| {
            matches!(
                b.source,
                BindingSource::Synthetic { binding: found, .. } if found == binding
            )
        })
        .map(|b| (0, b.binding))
}

/// The `(set, binding)` of the pipeline's kernel function-table directory.
pub(super) fn directory_binding(p: &ComputePipeline) -> (u32, u32) {
    synthetic(p, SyntheticBinding::FunctionTableDirectory).expect("a function-table directory")
}

/// The bytes at `(set, binding)` of `bytes`.
pub(super) fn bytes_at(bytes: &[DescriptorBytes], (set, binding): (u32, u32)) -> &[u8] {
    &bytes
        .iter()
        .find(|b| b.set == set && b.binding == binding)
        .expect("directory bytes")
        .bytes
}

/// The entries `(id, element, flags)` of a directory's bytes.
fn entries(bytes: &[u8]) -> Vec<(u64, u32, u32)> {
    bytes
        .chunks(TABLE_ENTRY_BYTES as usize)
        .map(|e| {
            let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().expect("word"));
            (u64::from(w(0)) | (u64::from(w(1)) << 32), w(2), w(3))
        })
        .collect()
}

/// On every device the kernel's snapshot is `FunctionTableSnapshot`, and its directory, a handle
/// table of the class's entries, gives each listed nonzero id its record's word offset in the
/// snapshot of the listed sizes, whatever the heap placement. Up to 15 listed tables keep the
/// class; a 16th leaves it, and a dispatch of the old plan over it is refused.
#[test]
fn the_plan_writes_the_snapshot_and_its_directory() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    for stem in ["runtime-floor", "moltenvk-m3"] {
        let dev = device(stem);
        let pipe = pipeline(&dev, &caller, &functions, ONE);
        let snapshot =
            synthetic(&pipe, SyntheticBinding::FunctionTableSnapshot).expect("a snapshot");
        let directory = directory_binding(&pipe);
        let mut state = state();
        state.function_tables.push(ArgumentFunctionTable {
            resource_id: 0,
            size: 4,
            address: 0,
        });
        let plan = pipe.plan(&dev, &state.shape()).expect("plan");
        let write = |at: (u32, u32)| {
            plan.descriptors
                .iter()
                .find(|d| (d.set, d.binding) == at)
                .map(|d| (d.ty, d.source.clone()))
                .expect("a write")
        };
        assert_eq!(
            write(snapshot),
            (
                vk::DescriptorType::STORAGE_BUFFER,
                WriteSource::FunctionTableSnapshot
            ),
            "{stem}"
        );
        assert_eq!(
            write(directory).1,
            WriteSource::HandleTable {
                stage: Stage::Kernel,
                bytes_len: 16 * TABLE_ENTRY_BYTES,
            },
            "{stem}"
        );
        // `T` (5 entries) at 1, `U` (2) at 1 + 8, `E` (0) at 9 + 5; the id-less table is placed
        // after them but named by nothing.
        let layout = FunctionTableSnapshot::of(state.function_tables.iter().map(|f| f.size))
            .expect("snapshot");
        assert_eq!(layout.offsets(), [1, 9, 14, 17], "{stem}");
        let mut want = vec![(0, 0, 0); 13];
        want.extend([(U, 9, 0), (E, 14, 0), (T, 1, 0)]);
        for placement in [HeapPlacement::Planned, HeapPlacement::ResourceId] {
            state.heap_placement = placement;
            let cmds = plan
                .instantiate_dispatch(&state, &one_group())
                .expect("dispatch");
            assert_eq!(
                entries(bytes_at(&cmds.descriptor_bytes, directory)),
                want,
                "{stem} {placement:?}"
            );
        }
        let mut sixteen = state.clone();
        sixteen
            .function_tables
            .extend((0..12).map(|i| ArgumentFunctionTable {
                resource_id: 0x100 + i,
                size: 1,
                address: 0,
            }));
        assert_ne!(sixteen.shape(), state.shape(), "{stem}");
        assert!(plan.instantiate_dispatch(&sixteen, &one_group()).is_err());
    }
}

/// Where a dispatch's tables' words are.
#[derive(Clone, Copy)]
enum Words<'a> {
    /// The stage's function-table snapshot holds these bytes.
    Snapshot(&'a [u8]),
    /// These words lie at these device addresses.
    Addressed(&'a [(u64, Vec<u8>)]),
}

/// Run `m` over `n` threads with `args` at buffer 1, the directory `directory`, and the tables'
/// `words`: the words it writes.
fn dispatch(
    m: &TypedModule,
    n: u32,
    args: Vec<u8>,
    directory: &[u8],
    words: Words<'_>,
) -> Vec<u32> {
    let buffer = |slot: ResourceSlot, bytes: Vec<u8>| Buffer {
        slot: Some(slot),
        address: None,
        bytes,
    };
    let slot = |index| ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    };
    let mut buffers = vec![
        buffer(slot(0), vec![0; n as usize * 4]),
        buffer(slot(1), args),
        buffer(directory_symbol(Stage::Kernel), directory.to_vec()),
    ];
    match words {
        Words::Snapshot(bytes) => {
            buffers.push(buffer(snapshot_symbol(Stage::Kernel), bytes.to_vec()));
        }
        Words::Addressed(tables) => {
            buffers.extend(tables.iter().map(|(address, bytes)| Buffer {
                slot: None,
                address: Some(*address),
                bytes: bytes.clone(),
            }));
        }
    }
    let res = Resources {
        buffers,
        push: vec![0; 128],
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [n, 1, 1],
        ..Config::default()
    };
    let out = run(m, &cfg, res).unwrap_or_else(|e| panic!("{e:?}"));
    out.buffers[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// What Metal answers for entry `i` of the table whose id is `id`, called with `(tid, k)`.
fn expect(id: u64, i: u32, tid: u32, k: u32) -> u32 {
    let entries: &[Option<usize>] = match id {
        T => &T_ENTRIES,
        U => &U_ENTRIES,
        E => &[],
        _ => return 0xffff_ffff,
    };
    if entries.is_empty() {
        return 0xffff_fffe;
    }
    match entries[i as usize % entries.len()] {
        None => 0xffff_fffd,
        Some(0) => tid.wrapping_add(k),
        Some(1) => tid.wrapping_mul(k),
        Some(2) => tid.wrapping_sub(k),
        // A function of another type: the default, zero.
        Some(_) => 0,
    }
}

fn ids(list: &[u64]) -> Vec<u8> {
    list.iter().flat_map(|id| id.to_le_bytes()).collect()
}

/// The directory bytes a dispatch of kernel [`ONE`] gets with [`state`]'s tables listed, planned
/// on the device of profile `stem`.
fn directory_of(stem: &str, caller: &[u8], functions: &[u8]) -> Vec<u8> {
    let dev = device(stem);
    let pipe = pipeline(&dev, caller, functions, ONE);
    let directory = directory_binding(&pipe);
    let state = state();
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let cmds = plan
        .instantiate_dispatch(&state, &one_group())
        .expect("dispatch");
    bytes_at(&cmds.descriptor_bytes, directory).to_vec()
}

/// On every committed profile, each kernel calls the entry the table its argument buffer names
/// holds, through the snapshot of the tables' words and the binder's directory of their offsets
/// there: a listed table by its id (the high word too), a null entry, an empty table, and a null
/// table for the null id and for an id no listed table has.
#[test]
fn an_argument_table_call_calls_the_entry_its_table_holds() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let h = handles(&functions);
    let tables = [
        table_bytes(&h, &T_ENTRIES),
        table_bytes(&h, &U_ENTRIES),
        table_bytes(&h, &[]),
    ];
    let snapshot = FunctionTableSnapshot::of(state().function_tables.iter().map(|f| f.size))
        .expect("snapshot")
        .bytes(tables.iter().map(Vec::as_slice))
        .expect("snapshot bytes");
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    const N: u32 = 40;
    let words = Words::Snapshot(&snapshot);
    for (stem, p) in committed_profiles().expect("profiles") {
        let directory = &directory_of(&stem, &caller, &functions);
        let module = |entry| {
            typed_module_linked(&caller, entry, &p, &linked)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"))
                .1
        };
        let one = module(ONE);
        for id in [T, U, E, 0, UNKNOWN] {
            let want: Vec<u32> = (0..N).map(|tid| expect(id, tid, tid, 3)).collect();
            assert_eq!(
                dispatch(&one, N, ids(&[id]), directory, words),
                want,
                "{stem} {ONE} {id:#x}"
            );
        }
        let array = module(ARRAY);
        for pair in [[T, U], [E, UNKNOWN]] {
            let mut args = vec![0; 8];
            args.extend(ids(&pair));
            let want: Vec<u32> = (0..N)
                .map(|tid| expect(pair[(tid & 1) as usize], tid >> 1, tid, 5))
                .collect();
            assert_eq!(
                dispatch(&array, N, args, directory, words),
                want,
                "{stem} {ARRAY} {pair:x?}"
            );
        }
        let select = module(SELECT);
        for [a, b] in [[U, T], [0, E]] {
            let want: Vec<u32> = (0..N)
                .map(|tid| {
                    let id = if tid & 1 != 0 { a } else { b };
                    expect(id, tid >> 1, tid, 7)
                })
                .collect();
            assert_eq!(
                dispatch(&select, N, ids(&[a, b]), directory, words),
                want,
                "{stem} {SELECT} {a:#x} {b:#x}"
            );
        }
    }
}

/// A state listing no table: the directory holds only null entries, so every id reads the null
/// table.
#[test]
fn without_listed_tables_every_id_reads_the_null_table() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let pipe = pipeline(&dev, &caller, &functions, ONE);
    let directory = directory_binding(&pipe);
    let state = BindState::default();
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let cmds = plan
        .instantiate_dispatch(&state, &one_group())
        .expect("dispatch");
    let directory = bytes_at(&cmds.descriptor_bytes, directory).to_vec();
    assert_eq!(entries(&directory), vec![(0, 0, 0); 16]);
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    let p = crate::pipeline::testing::profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES)
        .expect("profile");
    let (_, one) = typed_module_linked(&caller, ONE, &p, &linked).expect("module");
    let null = FunctionTableSnapshot::of([])
        .expect("snapshot")
        .bytes([])
        .expect("snapshot bytes");
    for id in [T, 0] {
        assert_eq!(
            dispatch(&one, 4, ids(&[id]), &directory, Words::Snapshot(&null)),
            [0xffff_ffff; 4],
            "{id:#x}"
        );
    }
}

/// The bind state listing `T`, `U` and `E`, and a table of `2^25 + 64` entries, whose words
/// (and their snapshot) pass 128 MiB, lavapipe's `maxStorageBufferRange` and Vulkan's least.
fn wide_state() -> BindState {
    let mut s = state();
    s.function_tables.push(ArgumentFunctionTable {
        resource_id: 0x50,
        size: (1 << 25) + 64,
        address: 0,
    });
    s
}

/// `state`'s tables, each at a device address whose high word is its list position plus one.
fn at_addresses(mut state: BindState) -> BindState {
    for (i, t) in state.function_tables.iter_mut().enumerate() {
        t.address = ((i as u64 + 1) << 32) | 0x40;
    }
    state
}

/// The lowering inputs of the variant that reads tables at their addresses.
fn addressed_inputs() -> LowerInputs {
    let mut key = VariantKey::new();
    key.addressed_function_tables = true;
    key.lowering_inputs(Stage::Kernel)
}

/// Kernel `entry` with the four functions linked, compiled to read its tables at their device
/// addresses (`VariantKey::addressed_function_tables`), through the public API.
fn addressed_pipeline(
    dev: &Device,
    caller: &[u8],
    functions: &[u8],
    entry: &str,
) -> Result<ComputePipeline, Error> {
    let caller = Library::from_air(caller).expect("library");
    let lib = Library::from_air(functions).expect("library");
    let linked = LINKED.iter().fold(LinkedFunctions::new(), |l, name| {
        l.with_function(lib.function(name).expect("function"))
    });
    let f = FunctionDescriptor::new(caller.function(entry).expect("entry")).with_linked(linked);
    let compiled = dev.compile_function(&f, &addressed_inputs(), &Budget::standard())?;
    let d = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    dev.compile_compute_pipeline(&d, &Budget::standard())
}

/// Whether `e` is `Unsupported{BufferRange}` naming device addresses.
fn needs_device_addresses(e: &Error) -> bool {
    matches!(e, Error::Unsupported(u) if u.feature == Feature::BufferRange)
}

/// A shape whose listed tables may pass the device's `maxStorageBufferRange` (lavapipe's 128 MiB)
/// asks for the variant that reads them at their device addresses, on a device with device
/// addresses; tables below 128 MiB, which every device's range reaches, ask for none, nor do they
/// where the range is 4 GiB. The variant declares the directory alone (no snapshot) and states `FunctionTableRange` with `addressed`; its plan keeps the key whatever the
/// shape, and writes each listed table's address into the directory, low word as the element and
/// high word as the flags. A listed table without an address is refused, as is a directory that
/// lists no table (its null entry's loads take a listed table's address). Without device addresses
/// (the Vulkan 1.2 floors) such a shape keeps the whole-bound words, as before the variant, whose
/// consumer refuses a table that passes the range where it binds it (the shape's class cannot tell
/// one that fits), and the variant does not compile.
#[test]
fn tables_past_the_storage_range_are_read_at_their_addresses() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let wide = wide_state();
    assert_eq!(state().shape().function_tables.bytes_class, 0);
    assert_eq!(wide.shape().function_tables.bytes_class, 28);
    for stem in ["gate-lavapipe", "moltenvk-m3", "runtime-floor"] {
        let dev = device(stem);
        let addressed = dev.profile().fields().buffer_access == BufferAccess::Addressed;
        let limit = dev.profile().fields().limits.max_storage_buffer_range;
        let passes = wide.shape().function_tables.may_pass(limit);
        assert_eq!(passes, limit < 1 << 28, "{stem}");
        let pipe = pipeline(&dev, &caller, &functions, ONE);
        let small = pipe.plan(&dev, &state().shape()).expect("plan");
        assert!(!small.variant.addressed_function_tables, "{stem}");
        let plan = pipe.plan(&dev, &wide.shape()).expect("plan");
        assert_eq!(
            plan.variant.addressed_function_tables,
            addressed && passes,
            "{stem}"
        );
        let compiled = addressed_pipeline(&dev, &caller, &functions, ONE);
        if !addressed {
            let e = compiled.map(|_| ()).expect_err("no device addresses");
            assert!(needs_device_addresses(&e), "{stem}: {e:?}");
            continue;
        }
        let compiled = compiled.unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(
            synthetic(&compiled, SyntheticBinding::FunctionTableSnapshot),
            None,
            "{stem}"
        );
        assert!(compiled
            .dependencies
            .iter()
            .any(|d| d.fact == RuntimeFact::FunctionTableRange { addressed: true }));
        let directory = directory_binding(&compiled);
        for listed in [state(), wide.clone()] {
            let plan = compiled.plan(&dev, &listed.shape()).expect("plan");
            assert!(plan.variant.addressed_function_tables, "{stem}");
            let cmds = plan
                .instantiate_dispatch(&at_addresses(listed.clone()), &one_group())
                .expect("dispatch");
            let mut want: Vec<(u64, u32, u32)> = vec![(0, 0, 0); 16 - listed.function_tables.len()];
            want.extend([(U, 0x40, 2), (E, 0x40, 3)]);
            if listed.function_tables.len() == 4 {
                want.push((0x50, 0x40, 4));
            }
            want.push((T, 0x40, 1));
            assert_eq!(
                entries(bytes_at(&cmds.descriptor_bytes, directory)),
                want,
                "{stem}"
            );
            assert!(plan.instantiate_dispatch(&listed, &one_group()).is_err());
        }
        // A directory of null entries alone: their loads would have no listed table's address.
        let none = BindState::default();
        let plan = compiled.plan(&dev, &none.shape()).expect("plan");
        assert!(plan.variant.addressed_function_tables, "{stem}");
        assert!(matches!(
            plan.instantiate_dispatch(&none, &one_group()),
            Err(crate::error::BindError::Unsupported(_))
        ));
    }
}

/// Under the variant that reads tables at their device addresses, each kernel calls the entry the
/// table its argument buffer names holds, with the binder's directory bytes and the tables' words
/// at their addresses: a listed table by its id (the high word too), a null entry, an empty table,
/// and a null table for the null id and for an id no listed table has; on every committed profile
/// with device addresses.
#[test]
fn an_addressed_table_call_calls_the_entry_its_table_holds() {
    let Some((caller, functions)) = fixtures() else {
        return;
    };
    let h = handles(&functions);
    let listed = at_addresses(state());
    let tables: Vec<(u64, Vec<u8>)> = [
        table_bytes(&h, &T_ENTRIES),
        table_bytes(&h, &U_ENTRIES),
        table_bytes(&h, &[]),
    ]
    .into_iter()
    .zip(&listed.function_tables)
    .map(|(bytes, t)| (t.address, bytes))
    .collect();
    let dev = device("moltenvk-m3");
    let pipe = addressed_pipeline(&dev, &caller, &functions, ONE).expect("pipeline");
    let plan = pipe.plan(&dev, &listed.shape()).expect("plan");
    let cmds = plan
        .instantiate_dispatch(&listed, &one_group())
        .expect("dispatch");
    let directory = bytes_at(&cmds.descriptor_bytes, directory_binding(&pipe)).to_vec();
    let none = FunctionConstantValues::defaults();
    let linked = sources(&functions, &none);
    let inputs = addressed_inputs();
    let words = Words::Addressed(&tables);
    const N: u32 = 40;
    let mut ran = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if p.fields().buffer_access != BufferAccess::Addressed {
            continue;
        }
        ran += 1;
        let module = |entry| {
            crate::pipeline::testing::typed_module_linked_with(&caller, entry, &p, &linked, &inputs)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"))
                .1
        };
        let one = module(ONE);
        for id in [T, U, E, 0, UNKNOWN] {
            let want: Vec<u32> = (0..N).map(|tid| expect(id, tid, tid, 3)).collect();
            assert_eq!(
                dispatch(&one, N, ids(&[id]), &directory, words),
                want,
                "{stem} {ONE} {id:#x}"
            );
        }
        let array = module(ARRAY);
        for pair in [[T, U], [E, UNKNOWN]] {
            let mut args = vec![0; 8];
            args.extend(ids(&pair));
            let want: Vec<u32> = (0..N)
                .map(|tid| expect(pair[(tid & 1) as usize], tid >> 1, tid, 5))
                .collect();
            assert_eq!(
                dispatch(&array, N, args, &directory, words),
                want,
                "{stem} {ARRAY} {pair:x?}"
            );
        }
        let select = module(SELECT);
        for [a, b] in [[U, T], [0, E]] {
            let want: Vec<u32> = (0..N)
                .map(|tid| {
                    let id = if tid & 1 != 0 { a } else { b };
                    expect(id, tid >> 1, tid, 7)
                })
                .collect();
            assert_eq!(
                dispatch(&select, N, ids(&[a, b]), &directory, words),
                want,
                "{stem} {SELECT} {a:#x} {b:#x}"
            );
        }
    }
    assert!(ran > 0, "a profile with device addresses");
}
