//! Function pointers chosen at run time end to end.
//!
//! The owned fixture `engine/fpc_chosen_pointers` holds kernels that call through pointers
//! chosen among functions' addresses, not a table's entries alone: a select of two, a phi of
//! three, one carried around a loop, a private array, comparisons, a threadgroup array, a 64-bit
//! integer written to device memory and read back by the neighbouring thread, a table's entry
//! merged with an address and stored beside one, and recursion through a pointer each function
//! chooses. The functions are `engine/fpc_pointer_functions`', which the pipeline links;
//! `fpc_own_*` choose among the caller's own functions and link nothing.
//!
//! - Every kernel compiles and passes L3 on every committed profile.
//! - On the CPU interpreter every kernel computes what Metal does (our probes ran the fixture's own
//!   AIR): the select, phi, loop, array, comparison and threadgroup shapes; the integer round trip,
//!   whose bits are each function's word, the same in every thread; the table shapes, an entry
//!   holding a function's handle equal to that function's address; and the recursion to depth
//!   `2k - 1` at `maxCallStackDepth` `k`, as plain Metal reaches (and to `2k`, whose last call
//!   answers the zero a call past the stack yields), and zero past the stack (the recorded
//!   near-equivalent).
//! - A visible function whose address a kernel takes and the pipeline does not link needs it
//!   (`NeedsInput{LinkedFunction}`), as Metal refuses the pipeline ("unresolved visible function
//!   reference"), and the recursion's stack is refused past `k` 8179, as Metal's is.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::bind::FunctionTableWords;
use crate::budget::SPIRV_WORDS;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{DiagName, Error, Limit, RuntimeInput};
use crate::library::FunctionConstantValues;
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, profile, typed_module_linked_with,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;
use crate::spirv::testing::refuses_threadgroup_memory;

const CALLER: &str = "fpc_chosen_pointers";
const FUNCTIONS: &str = "fpc_pointer_functions";
/// The functions every kernel but the `fpc_own_*` ones links, in this order: their handles are
/// words 2 to 6.
const LINKED: [&str; 5] = [
    "fpc_inc",
    "fpc_dbl",
    "fpc_sq",
    "fpc_rec_even",
    "fpc_rec_odd",
];
/// Threads of each dispatch, all in one threadgroup.
const N: u32 = 40;

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

/// The functions `kernel`'s pipeline links.
fn linked_names(kernel: &str) -> &'static [&'static str] {
    if kernel.starts_with("fpc_own_") {
        &[]
    } else {
        &LINKED
    }
}

fn sources<'a>(
    functions: &'a [u8],
    names: &[&'a str],
    none: &'a FunctionConstantValues,
) -> Vec<LinkedSource<'a>> {
    names
        .iter()
        .map(|name| LinkedSource::new(name, functions, none))
        .collect()
}

fn depth(k: u32) -> LowerInputs {
    LowerInputs {
        max_call_stack_depth: Some(k),
        ..LowerInputs::new()
    }
}

/// `kernel` compiled on `p` at depth `k`, the functions `names` linked.
fn compile(
    (caller, functions): &(Vec<u8>, Vec<u8>),
    kernel: &str,
    names: &[&str],
    k: u32,
    p: &DeviceProfile,
) -> Result<std::sync::Arc<crate::library::FunctionArtifact>, Error> {
    let none = FunctionConstantValues::defaults();
    compile_bitcode_linked(
        caller,
        EntryRef::Name(kernel),
        (&none, &sources(functions, names, &none)),
        p,
        &depth(k),
        &mut meter(),
        SPIRV_WORDS,
    )
}

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Run `kernel` at depth `k` on `p` over [`N`] threads in one threadgroup, buffer `i` of
/// `buffers` at its index: every buffer's bytes after the run.
fn dispatch(
    fx: &(Vec<u8>, Vec<u8>),
    kernel: &str,
    k: u32,
    p: &DeviceProfile,
    buffers: &[(u32, Vec<u8>)],
) -> Vec<Vec<u8>> {
    let none = FunctionConstantValues::defaults();
    let linked = sources(&fx.1, linked_names(kernel), &none);
    let (plan, m) = typed_module_linked_with(&fx.0, kernel, p, &linked, &depth(k))
        .unwrap_or_else(|e| panic!("{kernel}: {e}"));
    // Zero bind offsets, and a region the single-region dispatch does not read.
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            other => panic!("{kernel}: unexpected push field {other:?}"),
        }
    }
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let spec = BTreeMap::from([
        (SpecKey::LocalSize(Axis::X), u64::from(N)),
        (SpecKey::LocalSize(Axis::Y), 1),
        (SpecKey::LocalSize(Axis::Z), 1),
        (SpecKey::FlatLocalSize, u64::from(N)),
    ]);
    let res = Resources {
        buffers: buffers
            .iter()
            .map(|(i, bytes)| Buffer {
                slot: Some(slot(*i)),
                address: None,
                bytes: bytes.clone(),
            })
            .collect(),
        push,
        spec,
    };
    let cfg = Config {
        workgroups: [1, 1, 1],
        ..Config::default()
    };
    run(&m, &cfg, res)
        .unwrap_or_else(|e| panic!("{kernel}: {e:?}"))
        .buffers
}

fn zeros() -> Vec<u8> {
    vec![0; N as usize * 8]
}

fn inc(x: u32) -> u32 {
    x.wrapping_add(1)
}

fn dbl(x: u32) -> u32 {
    x.wrapping_mul(2)
}

fn sq(x: u32) -> u32 {
    x.wrapping_mul(x)
}

const OPS: [fn(u32) -> u32; 3] = [inc, dbl, sq];

/// The comparisons' word: of `[inc, dbl, sq, inc]`, `f` at `tid % 4` and `g` at
/// `(tid / 4) % 4`; bit 0 `f == g`, bit 1 `f` is `inc`, bit 2 `f` is not null, bit 3 `g` is `sq`.
fn compared(tid: u32) -> u32 {
    let of = [0, 1, 2, 0];
    let (f, g) = (of[(tid % 4) as usize], of[((tid / 4) % 4) as usize]);
    4 | u32::from(f == g) | (u32::from(f == 0) << 1) | (u32::from(g == 2) << 3)
}

fn looped(tid: u32) -> u32 {
    let (mut f, mut acc) = (0, tid);
    for _ in 0..4 {
        acc = OPS[f](acc);
        f = if acc & 1 != 0 { 0 } else { 1 };
    }
    acc
}

/// A kernel that reads only `tid`, with the word Metal answers for it.
type Shape = (&'static str, fn(u32) -> u32);

/// Each kernel that reads only `tid`.
const SHAPES: [Shape; 9] = [
    ("fpc_select", |t| if t & 1 != 0 { inc(t) } else { dbl(t) }),
    ("fpc_phi", |t| OPS[(t % 3) as usize](t + 5)),
    ("fpc_loop", looped),
    ("fpc_private_array", |t| OPS[(t % 3) as usize](t)),
    ("fpc_compare", compared),
    (
        "fpc_threadgroup",
        |t| if t & 1 != 0 { inc(t) } else { sq(t) },
    ),
    (
        "fpc_own_select",
        |t| if t & 1 != 0 { inc(t) } else { dbl(t) },
    ),
    ("fpc_own_private_array", |t| OPS[(t % 3) as usize](t)),
    ("fpc_own_compare", compared),
];

/// Every other kernel.
const OTHERS: [&str; 5] = [
    "fpc_int_back",
    "fpc_table_mix",
    "fpc_table_store",
    "fpc_recurse",
    "fpc_wide",
];

/// `fpc_rec_even(n)` (`odd` false) or `fpc_rec_odd(n)` with `calls` activations the stack holds,
/// the root's among them: Metal's answer when the recursion fits, the zero a call past the stack
/// yields otherwise.
fn recursed(odd: bool, n: u32, calls: u32) -> u32 {
    if n == 0 || calls == 0 {
        return 0;
    }
    let next = if n & 2 != 0 { !odd } else { odd };
    (if odd { 2 * n } else { n }).wrapping_add(recursed(next, n - 1, calls - 1))
}

/// Every kernel compiles and passes L3 on every committed profile, its functions linked.
#[test]
fn chosen_pointer_kernels_validate_on_every_profile() {
    let Some(fx) = fixtures() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for kernel in SHAPES.iter().map(|&(k, _)| k).chain(OTHERS) {
            let a = match compile(&fx, kernel, linked_names(kernel), 1, &p) {
                // A profile without zeroed workgroup memory takes no threadgroup memory.
                Err(e) if refuses_threadgroup_memory(&e, &p) => continue,
                a => a.unwrap_or_else(|e| panic!("{stem} {kernel}: {e}")),
            };
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {kernel}: L3: {e}"));
        }
    }
}

/// A select, a phi, a loop, a private array, comparisons and a threadgroup array of functions'
/// addresses call and compare as Metal's do, linked functions' and the caller's own alike; the
/// phi's branches write their side words.
#[test]
fn pointers_chosen_at_run_time_call_the_functions_they_hold() {
    let Some(fx) = fixtures() else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    for (kernel, want) in SHAPES {
        let out = dispatch(&fx, kernel, 1, &p, &[(0, zeros()), (1, zeros())]);
        let all: Vec<u32> = (0..N).map(want).collect();
        assert_eq!(words(&out[0])[..N as usize], all, "{kernel}");
        if kernel == "fpc_phi" {
            let sides: Vec<u32> = (0..N).map(|t| 10 + t % 3).collect();
            assert_eq!(words(&out[1])[..N as usize], sides, "{kernel} side");
        }
    }
}

/// A pointer converted to a 64-bit integer, written to device memory, read back by the
/// neighbouring thread and converted back calls its function. Its bits are its function's word
/// zero-extended, the same in every thread that holds the function: with no table read in the
/// stage, a word of the stage's own past the pipeline's handles (2 to 6), `fpc_inc`'s 7 and
/// `fpc_dbl`'s 8 in the order the stage takes their addresses. Metal's are its pipeline's own and
/// vary between pipelines, so only their sameness is Metal's.
#[test]
fn a_pointer_through_an_integer_is_its_word() {
    let Some(fx) = fixtures() else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let out = dispatch(&fx, "fpc_int_back", 1, &p, &[(0, zeros()), (1, zeros())]);
    let all: Vec<u32> = (0..N)
        .map(|t| if t & 1 != 0 { dbl(t) } else { inc(t) })
        .collect();
    assert_eq!(words(&out[0])[..N as usize], all);
    let bits: Vec<u64> = out[1]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    let want: Vec<u64> = (0..N).map(|t| if t & 1 != 0 { 7 } else { 8 }).collect();
    assert_eq!(bits, want);
}

/// A table's entry merged with a function's address, compared with one, and stored beside one
/// calls and compares as Metal's does: the table holds `fpc_inc`, `fpc_dbl` and `fpc_sq`, an
/// entry holding a function's handle equal to that function's address.
#[test]
fn table_entries_and_addresses_call_and_compare_alike() {
    let Some(fx) = fixtures() else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let mut table = FunctionTableWords::new(3);
    let lib = crate::library::Library::from_air(&fx.1).expect("library");
    let linked = LINKED
        .iter()
        .fold(crate::pipeline::LinkedFunctions::new(), |l, name| {
            l.with_function(lib.function(name).expect("function"))
        });
    for (i, name) in LINKED[..3].iter().enumerate() {
        let f = crate::pipeline::LinkedFunction::new(lib.function(name).expect("function"));
        let handle = linked.function_handle(&f).expect("a handle");
        assert_eq!(handle.word(), i as u32 + 2, "{name}");
        assert!(table.set(i as u32, Some(handle)));
    }
    for kernel in ["fpc_table_mix", "fpc_table_store"] {
        let out = dispatch(&fx, kernel, 1, &p, &[(0, zeros()), (1, table.to_bytes())]);
        let all: Vec<u32> = (0..N)
            .map(|t| {
                let e = (t % 3) as usize;
                if kernel == "fpc_table_mix" {
                    let f = if t & 4 != 0 { e } else { 2 };
                    OPS[f](t) + if e == 0 { 1000 } else { 0 } + if f == 2 { 2000 } else { 0 }
                } else {
                    OPS[if (t >> 1) & 1 != 0 { 2 } else { e }](t)
                }
            })
            .collect();
        assert_eq!(words(&out[0])[..N as usize], all, "{kernel}");
    }
}

/// Recursion through pointers each function chooses between the two reaches depth `2k - 1` at
/// `maxCallStackDepth` `k`, as plain Metal's does (at `k` 1, 2, 4 and
/// 8): the component's two 16-byte functions hold `2k` activations, the kernel's call and
/// `2k - 1` recursive calls. Thread `t` runs depth `t / 2` through `fpc_rec_odd` for odd `t`; past
/// the stack a call yields zero.
#[test]
fn recursion_through_chosen_pointers_reaches_metal_s_depth() {
    let Some(fx) = fixtures() else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    for k in [1, 2, 4, 8] {
        let (calls, reach) = (2 * k, 2 * k - 1);
        let depths: Vec<u8> = (0..N).flat_map(|t| (t / 2).to_le_bytes()).collect();
        let out = dispatch(&fx, "fpc_recurse", k, &p, &[(0, zeros()), (2, depths)]);
        let got = &words(&out[0])[..N as usize];
        let all: Vec<u32> = (0..N).map(|t| recursed(t & 1 != 0, t / 2, calls)).collect();
        assert_eq!(got, all, "k {k}");
        // Within the reach, the full recursion's answer.
        for t in 0..N.min(2 * reach + 2) {
            assert_eq!(
                got[t as usize],
                recursed(t & 1 != 0, t / 2, u32::MAX),
                "k {k} t {t}"
            );
        }
    }
}

/// A visible function whose address a kernel takes and the pipeline does not link needs it, as
/// Metal refuses the pipeline ("unresolved visible function reference: fpc_dbl").
#[test]
fn an_unlinked_function_s_address_needs_it() {
    let Some(fx) = fixtures() else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let needs = |symbol: &str| {
        Err(Error::NeedsInput(RuntimeInput::LinkedFunction {
            symbol: DiagName::new(symbol),
        }))
    };
    for (kernel, missing) in [
        ("fpc_select", "fpc_dbl"),
        ("fpc_threadgroup", "fpc_sq"),
        ("fpc_int_back", "fpc_dbl"),
    ] {
        let r = compile(&fx, kernel, &["fpc_inc"], 1, &p).map(|_| ());
        assert_eq!(r, needs(missing), "{kernel}");
    }
}

/// The recursion's stack is refused past `k` 8179, where Metal says "Compute pipeline exceeds
/// available stack space" (linked in `functions` or `private_functions` alike): two
/// 16-byte frames, 32 bytes a depth of Metal's 261728.
#[test]
fn the_recursion_s_stack_is_refused_where_metal_s_is() {
    let Some(fx) = fixtures() else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let refused = Err(Error::LimitExceeded(Limit::CallStackBytes));
    let names = &LINKED[3..];
    assert_eq!(
        compile(&fx, "fpc_recurse", names, 8180, &p).map(|_| ()),
        refused
    );
    assert_ne!(
        compile(&fx, "fpc_recurse", names, 8179, &p).map(|_| ()),
        refused
    );
}
