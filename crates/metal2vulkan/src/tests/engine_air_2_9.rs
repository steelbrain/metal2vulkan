//! AIR 2.9's operand forms translate as AIR 2.8's.
//!
//! At deployment target macOS 27 Apple's compiler writes AIR 2.9: every memory atomic
//! (`air.atomic.{global,local}.*`) takes its Metal `mem_flags` between its scope and its volatile
//! flag, and both barriers (`air.wg.barrier`, `air.simdgroup.barrier`) take a `memory_order`
//! between their flags and their scope. Metal's headers pass `mem_none` and `seq_cst` there in
//! every language before MSL 4.1, which is all AIR 2.8 can say, so an AIR 2.8 module and its AIR
//! 2.9 spelling with `mem_flags` 0 and order `seq_cst` (5) are the same program.
//!
//! Each public `.ll` fixture with such a call is respelled in AIR 2.9 here: those operands
//! inserted into every declaration and call, and the triple and `!air.version` moved to AIR 2.9.
//! A fixture older than AIR 2.0 is left alone: its atomics state no volatile flag and its
//! barriers no scope, so it has no AIR 2.9 spelling to compare against.
//! Both spellings compile, entry by entry, on every committed profile to the same SPIR-V words
//! (or the same refusal). The respelled modules keep their barrier declarations, so a barrier the
//! lockstep pass inserts is built through the module's AIR 2.9 declaration; at least one respelled
//! entry must take that path.
//!
//! Apple's compiler is the other half of that argument. The `a27_*` twins in
//! `validation/fixtures/public/engine/` are owned fixtures' MSL compiled unchanged as MSL 4.0
//! for macOS 27, so in AIR 2.9; each must compile, entry by entry, on every committed profile to
//! its original's words, so the original's authored cases and rows speak for it. The MSL 4.1
//! fixture `a27_memory_order` spells what only AIR 2.9 can (orders, memory flags, and barrier
//! memory scopes); its barriers' and atomics' scopes and semantics are checked here. The MSL 4.1
//! fixture `a27_thread_atomics` runs every atomic on thread memory (`air.atomic.private.*`); the
//! CPU interpreter computes its documented words.
//!
//! `llvm-as` assembles the text; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::{DevicePolicy, DeviceProfile};
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_and_validate, link_standalone, llvm_assemble, meter,
    profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order as RunOrder, Resources, SubgroupSize};
use crate::spirv::testing::{committed_profiles, spirv_dis};

/// The operand AIR 2.9 inserts before the last operand of a call to `name`: `mem_none` (0)
/// before an atomic's volatile flag, `seq_cst` (5) before a barrier's scope. `None` for any other
/// callee.
fn inserted(name: &str) -> Option<&'static str> {
    let atomic = ["air.atomic.global.", "air.atomic.local."]
        .iter()
        .any(|p| name.starts_with(p));
    if atomic {
        Some("i32 0")
    } else if name == "air.wg.barrier" || name == "air.simdgroup.barrier" {
        Some("i32 5")
    } else {
        None
    }
}

/// The index of the `)` closing the `(` at byte `open` of `s`.
fn closing(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s.char_indices().skip_while(|&(i, _)| i < open) {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return (c == ')').then_some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// The byte offset of the last top-level operand in the operand list `s[open + 1..close]`.
fn last_operand(s: &str, open: usize, close: usize) -> usize {
    let mut depth = 0usize;
    let mut last = open + 1;
    for (i, c) in s[open + 1..close].char_indices() {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => last = open + 1 + i + 1,
            _ => {}
        }
    }
    last
}

/// `line` in AIR 2.9's spelling: the inserted operand (a type in a declaration, a constant in a
/// call) before the last operand of a changed intrinsic, and whether the line changed.
fn respell_line(line: &str) -> (String, bool) {
    let Some(at) = line.find("@air.") else {
        return (line.to_owned(), false);
    };
    let rest = &line[at + 1..];
    let Some(name_len) = rest.find('(') else {
        return (line.to_owned(), false);
    };
    let Some(constant) = inserted(&rest[..name_len]) else {
        return (line.to_owned(), false);
    };
    let open = at + 1 + name_len;
    let close = closing(line, open).expect("a balanced operand list");
    let last = last_operand(line, open, close);
    let declaration = line.trim_start().starts_with("declare ");
    let operand = if declaration { "i32" } else { constant };
    let out = format!("{} {operand},{}", &line[..last], &line[last..]);
    (out, true)
}

/// Whether module text `text` is already AIR 2.9 (its triple's).
fn is_air_2_9(text: &str) -> bool {
    text.lines()
        .any(|l| l.starts_with("target triple = \"air64_v29-"))
}

/// Whether module text `text` is AIR 2.x (its triple's), the spelling AIR 2.9 extends. Before
/// AIR 2.0 a memory atomic states no volatile flag and a barrier no scope, so inserting an
/// operand before the last one would write a shape no Metal compiler emits (`mem_flags` before a
/// scope, an order before a flag set) rather than that module's AIR 2.9 spelling, which does not
/// exist: AIR 2.9 states both.
fn is_air_2(text: &str) -> bool {
    text.lines()
        .any(|l| l.starts_with("target triple = \"air64_v2"))
}

/// `text` in AIR 2.9's spelling, or `None` when it calls no changed intrinsic, already is AIR
/// 2.9, or is older than AIR 2.0 ([`is_air_2`]).
fn respell(text: &str) -> Option<String> {
    if is_air_2_9(text) || !is_air_2(text) {
        return None;
    }
    let mut changed = false;
    let mut version = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("!air.version = !{!") {
            version = rest.strip_suffix('}').map(|n| format!("!{n} = "));
        }
    }
    let mut out = String::with_capacity(text.len() + 1024);
    for line in text.lines() {
        let (mut line, c) = respell_line(line);
        changed |= c;
        if line.starts_with("target triple = \"air64_v2") {
            line = "target triple = \"air64_v29-apple-macosx27.0.0\"".to_owned();
        }
        if version.as_deref().is_some_and(|v| line.starts_with(v)) {
            line = line.replace("i32 2, i32 8, i32 0", "i32 2, i32 9, i32 0");
        }
        out.push_str(&line);
        out.push('\n');
    }
    changed.then_some(out)
}

/// Bitcode of `text` through `llvm-as`.
fn assemble(text: &str) -> Result<Vec<u8>, AssembleError> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v_air_2_9_{}_{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).expect("write the respelled text");
    let out = llvm_assemble(&path);
    let _ = std::fs::remove_file(&path);
    out
}

/// Every public `.ll` fixture (not the typed `.air.ll` companions).
fn fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in [fixtures_dir(), engine_dir()] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("fixture directory")
            .map(|e| e.expect("entry").path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                name.ends_with(".ll") && !name.ends_with(".air.ll")
            })
            .collect();
        files.sort();
        out.extend(files);
    }
    out
}

/// What entry `entry` of `bitcode` compiles to on `p`: its standalone words, or its refusal.
fn outcome(bitcode: &[u8], entry: &str, p: &DeviceProfile) -> (Result<Vec<u32>, String>, bool) {
    match compile(bitcode, entry, p) {
        Ok(a) => {
            let lockstep = a
                .strategies
                .entries
                .keys()
                .any(|k| k.starts_with("barrier.lockstep"));
            let words = link_standalone(&a, p)
                .map(|(_, w)| w)
                .map_err(|e| format!("link: {e}"));
            (words, lockstep)
        }
        Err(e) => (Err(e.to_string()), false),
    }
}

#[test]
fn respelling_inserts_the_operands() {
    let (call, c) = respell_line(
        "  %10 = tail call i32 @air.atomic.global.xor.u.i32(ptr addrspace(1) captures(none) %9, \
         i32 %8, i32 0, i32 2, i1 true) #4",
    );
    assert!(c);
    assert_eq!(
        call,
        "  %10 = tail call i32 @air.atomic.global.xor.u.i32(ptr addrspace(1) captures(none) %9, \
         i32 %8, i32 0, i32 2, i32 0, i1 true) #4"
    );
    let (decl, _) = respell_line("declare void @air.wg.barrier(i32, i32) local_unnamed_addr #2");
    assert_eq!(
        decl,
        "declare void @air.wg.barrier(i32, i32, i32) local_unnamed_addr #2"
    );
    let (gep, _) = respell_line(
        "  tail call void @air.atomic.local.store.i32(ptr addrspace(3) getelementptr (i8, ptr \
         addrspace(3) @g, i64 4), i32 0, i32 0, i32 1, i1 true)",
    );
    assert!(
        gep.ends_with("i32 0, i32 0, i32 1, i32 0, i1 true)"),
        "{gep}"
    );
    let (barrier, _) = respell_line("  tail call void @air.simdgroup.barrier(i32 2, i32 4) #5");
    assert_eq!(
        barrier,
        "  tail call void @air.simdgroup.barrier(i32 2, i32 5, i32 4) #5"
    );
    // Neither the fence, a texture atomic, nor any other intrinsic changes.
    for line in [
        "  tail call void @air.atomic.fence(i32 1, i32 5, i32 2)",
        "  %1 = call <4 x i32> @air.atomic_fetch_add_explicit_texture_2d.u.v4i32(ptr \
         addrspace(1) %t, <2 x i32> %c, <2 x i32> zeroinitializer, <4 x i32> %v, i32 0, i32 3)",
    ] {
        assert_eq!(respell_line(line), (line.to_owned(), false));
    }
}

#[test]
fn air_2_9_spelling_compiles_to_the_same_words() {
    let profiles = committed_profiles().expect("committed profiles");
    let (mut respelled_fixtures, mut compared, mut lockstep) = (0, 0, 0);
    for path in fixtures() {
        let name = path.display().to_string();
        let text = std::fs::read_to_string(&path).expect("fixture text");
        let Some(respelled) = respell(&text) else {
            continue;
        };
        let old = match llvm_assemble(Path::new(&path)) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED air_2_9_spelling_compiles_to_the_same_words: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("{name}: llvm-as: {why}"),
        };
        let new = assemble(&respelled).unwrap_or_else(|e| panic!("{name} respelled: {e:?}"));
        // A barrier the lockstep pass inserts goes through the module's own declaration.
        let declares_barrier = [
            "@air.wg.barrier(i32, i32, i32)",
            "@air.simdgroup.barrier(i32, i32, i32)",
        ]
        .iter()
        .any(|d| {
            respelled
                .lines()
                .any(|l| l.starts_with("declare") && l.contains(d))
        });
        let entries: Vec<String> = crate::air::decode(&old, &mut meter())
            .unwrap_or_else(|e| panic!("{name}: decode: {e}"))
            .abi
            .entries
            .iter()
            .map(|e| e.name.to_string())
            .collect();
        respelled_fixtures += 1;
        for entry in &entries {
            for (stem, p) in &profiles {
                let (a, _) = outcome(&old, entry, p);
                let (b, inserted) = outcome(&new, entry, p);
                assert_eq!(a, b, "{name} {entry} on {stem}");
                compared += 1;
                lockstep += usize::from(inserted && declares_barrier);
            }
        }
    }
    eprintln!(
        "{respelled_fixtures} fixtures, {compared} cells compared, {lockstep} inserting lockstep barriers \
         through an AIR 2.9 declaration"
    );
    assert!(
        respelled_fixtures >= 30,
        "only {respelled_fixtures} fixtures call a changed intrinsic"
    );
    assert!(
        lockstep > 0,
        "no respelled entry inserted a lockstep barrier through an AIR 2.9 declaration"
    );
}

/// The AIR 2.9 twins: each `a27_<original>` fixture and its original's module, a top-level public
/// fixture or an engine one.
const TWINS: &[(&str, bool)] = &[
    ("kernel_atomic_xor_and_subtract", true),
    ("kernel_imageblock_tile_to_texture_array", true),
    ("ob1_buffer_atomic", false),
    ("ob1_threadgroup", false),
    ("spse_write_barrier", false),
    ("a4_simdgroup_barrier", false),
    ("a4_atomic_load_only", false),
    ("a3_early_return_barrier", false),
    ("icb_replay_grid_template", false),
];

/// The text before the `)` closing the operand list of each changed intrinsic's declaration in
/// `text` (the declaration's type, without its attributes), sorted.
fn changed_declarations(text: &str) -> Vec<String> {
    let mut out: Vec<String> = text
        .lines()
        .filter(|l| l.starts_with("declare "))
        .filter_map(|l| {
            let at = l.find("@air.")?;
            let open = at + l[at..].find('(')?;
            inserted(&l[at + 1..open])?;
            Some(l[..=closing(l, open)?].to_owned())
        })
        .collect();
    out.sort();
    out
}

/// Each twin is AIR 2.9 (its declarations are its original's, respelled) and compiles on every
/// committed profile to its original's words or refusal.
#[test]
fn air_2_9_twins_compile_to_their_originals_words() {
    let profiles = committed_profiles().expect("committed profiles");
    let mut compared = 0;
    for &(stem, top_level) in TWINS {
        let original = if top_level {
            fixtures_dir()
        } else {
            engine_dir()
        }
        .join(format!("{stem}.ll"));
        let twin = engine_dir().join(format!("a27_{stem}.ll"));
        let (original_text, twin_text) = (
            std::fs::read_to_string(&original).expect("original text"),
            std::fs::read_to_string(&twin).expect("twin text"),
        );
        assert!(
            is_air_2_9(&twin_text) && twin_text.contains("= !{i32 2, i32 9, i32 0}"),
            "a27_{stem} is not AIR 2.9"
        );
        let respelled = respell(&original_text).expect("the original calls a changed intrinsic");
        assert_eq!(
            changed_declarations(&twin_text),
            changed_declarations(&respelled),
            "a27_{stem}: declarations"
        );
        let bitcode = |path: &Path| match llvm_assemble(path) {
            Ok(b) => Some(b),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED air_2_9_twins_compile_to_their_originals_words: {why}");
                None
            }
            Err(AssembleError::Rejected(why)) => panic!("{}: llvm-as: {why}", path.display()),
        };
        let (Some(old), Some(new)) = (bitcode(&original), bitcode(&twin)) else {
            return;
        };
        let entries: Vec<String> = crate::air::decode(&old, &mut meter())
            .unwrap_or_else(|e| panic!("{stem}: decode: {e}"))
            .abi
            .entries
            .iter()
            .map(|e| e.name.to_string())
            .collect();
        assert!(!entries.is_empty(), "{stem}: no entry");
        for entry in &entries {
            for (p_stem, p) in &profiles {
                let (a, _) = outcome(&old, entry, p);
                let (b, _) = outcome(&new, entry, p);
                assert_eq!(a, b, "a27_{stem} {entry} on {p_stem}");
                compared += 1;
            }
        }
    }
    eprintln!("{} twins, {compared} cells compared", TWINS.len());
}

/// The scope and semantics operands of each barrier and atomic in disassembly `text`, by opcode:
/// `OpControlBarrier`'s execution scope, memory scope, and semantics; an atomic's scope and
/// semantics (a compare-exchange's both semantics).
fn sync_operands(text: &str) -> Vec<(String, Vec<u32>)> {
    let mut constants = BTreeMap::new();
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if let [id, "=", "OpConstant", ty, value] = words[..] {
            if ty == "%uint" {
                if let Ok(v) = value.parse::<u32>() {
                    constants.insert(id.to_owned(), v);
                }
            }
        }
    }
    let value = |id: &str| -> u32 {
        *constants
            .get(id)
            .unwrap_or_else(|| panic!("{id} is no 32-bit constant"))
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        let (op, rest) = match words[..] {
            [_, "=", op, ..] => (op, &words[3..]),
            [op, ..] => (op, &words[1..]),
            [] => continue,
        };
        // The operands after the result type and the pointer; a store has no result type.
        let operands: Vec<u32> = match op {
            "OpControlBarrier" => rest.iter().map(|id| value(id)).collect(),
            "OpAtomicStore" => rest[1..3].iter().map(|id| value(id)).collect(),
            "OpAtomicCompareExchange" => rest[2..5].iter().map(|id| value(id)).collect(),
            o if o.starts_with("OpAtomic") => rest[2..4].iter().map(|id| value(id)).collect(),
            _ => continue,
        };
        out.push((op.to_owned(), operands));
    }
    out.sort();
    out
}

/// `semantics` under the module's memory model: the Vulkan model adds `MakeAvailable` to a
/// release and `MakeVisible` to an acquire that names memory.
fn under(vulkan: bool, semantics: u32) -> u32 {
    let storage = semantics & (0x40 | 0x100 | 0x800) != 0;
    let mut s = semantics;
    if vulkan && storage && semantics & (0x4 | 0x8) != 0 {
        s |= 0x2000;
    }
    if vulkan && storage && semantics & (0x2 | 0x8) != 0 {
        s |= 0x4000;
    }
    s
}

/// An opcode and its scope and semantics operands ([`sync_operands`]).
type Op = (&'static str, Vec<u32>);

/// MSL 4.1's orders, memory flags, and barrier memory scopes reach SPIR-V: a non-explicit atomic is
/// `AcquireRelease` (a load's acquire, a store's release half) over every memory the stage has, an
/// explicit one its order over the memories it names, and each barrier its order at its own memory
/// scope, on a device whose simdgroup is its subgroup.
#[test]
fn msl_4_1_orders_reach_spirv() {
    let path = engine_dir().join("a27_memory_order.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED msl_4_1_orders_reach_spirv: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let p = profile("nvidia-rtx50-linux", DevicePolicy::RESIDENT_TABLES).expect("profile");
    // Scopes: Device 1, Workgroup 2, Subgroup 3. Semantics: Acquire 0x2, Release 0x4,
    // AcquireRelease 0x8, UniformMemory 0x40, WorkgroupMemory 0x100, ImageMemory 0x800.
    let every = 0x40 | 0x100 | 0x800;
    let render = 0x40 | 0x800;
    let cases: [(&str, Vec<Op>); 3] = [
        (
            "a27_memory_order",
            vec![
                // Threadgroup `sum`: a seq_cst store, add, and an acquire load naming it.
                ("OpAtomicStore", vec![2, 0x4 | every]),
                ("OpAtomicIAdd", vec![2, 0x8 | every]),
                ("OpAtomicLoad", vec![2, 0x2 | 0x100]),
                // acq_rel and seq_cst threadgroup barriers.
                ("OpControlBarrier", vec![2, 2, 0x8 | 0x100]),
                ("OpControlBarrier", vec![2, 2, 0x8 | 0x100]),
                // `thread_scope_simdgroup` and `thread_scope_device` memory scopes.
                ("OpControlBarrier", vec![2, 3, 0x8 | 0x100]),
                ("OpControlBarrier", vec![2, 1, 0x8 | 0x40]),
                // The `seq_cst` barrier at `thread_scope_device` is a sequentially consistent
                // fence there: the status word's increment after it.
                ("OpAtomicIAdd", vec![1, 0x8 | 0x40]),
                // Device counters: release, seq_cst, relaxed (naming texture memory), relaxed
                // (`thread_scope_threadgroup`), acq_rel naming texture memory.
                ("OpAtomicIAdd", vec![1, 0x4 | 0x40]),
                ("OpAtomicIAdd", vec![1, 0x8 | every]),
                ("OpAtomicUMax", vec![1, 0]),
                ("OpAtomicIAdd", vec![1, 0]),
                ("OpAtomicOr", vec![1, 0x8 | 0x800]),
                ("OpAtomicCompareExchange", vec![1, 0x8 | 0x40, 0x2 | 0x40]),
                // release, acquire, relaxed, and `thread_scope_thread` barriers.
                ("OpControlBarrier", vec![2, 2, 0x4 | 0x40]),
                ("OpControlBarrier", vec![2, 2, 0x2 | 0x40]),
                ("OpControlBarrier", vec![2, 2, 0]),
                ("OpControlBarrier", vec![2, 2, 0]),
                // An acq_rel simdgroup barrier.
                ("OpControlBarrier", vec![3, 3, 0x8 | 0x100]),
                // The lockstep device pass's barriers, `seq_cst` with `mem_device`: it cannot prove
                // `out[96 + (t ^ 31)]` apart, within a simdgroup, from the words the stores after
                // it write, and the release, acquire, relaxed and thread-scope barriers between
                // them order no pair.
                ("OpControlBarrier", vec![2, 2, 0x8 | 0x40]),
                ("OpControlBarrier", vec![2, 2, 0x8 | 0x40]),
                ("OpControlBarrier", vec![2, 2, 0x8 | 0x40]),
                ("OpControlBarrier", vec![2, 2, 0x8 | 0x40]),
                ("OpControlBarrier", vec![2, 2, 0x8 | 0x40]),
            ],
        ),
        (
            "a27_wide_and_float",
            vec![
                ("OpAtomicUMax", vec![1, 0]),
                // The float add's compare-and-swap loop: its acquiring read, then the exchange.
                ("OpAtomicLoad", vec![1, 0x2 | 0x40]),
                ("OpAtomicCompareExchange", vec![1, 0x8 | every, 0]),
            ],
        ),
        (
            // A render stage has no threadgroup memory.
            "a27_fragment_atomics",
            vec![
                ("OpAtomicIAdd", vec![1, 0x8 | render]),
                ("OpAtomicLoad", vec![1, 0x2 | render]),
                ("OpAtomicStore", vec![1, 0x4 | render]),
            ],
        ),
    ];
    for (entry, want) in cases {
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
        let text = spirv_dis(&words);
        let vulkan = text.contains("OpMemoryModel Logical Vulkan");
        let mut want: Vec<(String, Vec<u32>)> = want
            .into_iter()
            .map(|(op, mut v)| {
                // Semantics follow the scopes: two for a barrier's scopes, one for an atomic's.
                let first = if op == "OpControlBarrier" { 2 } else { 1 };
                for s in &mut v[first..] {
                    *s = under(vulkan, *s);
                }
                (op.to_owned(), v)
            })
            .collect();
        want.sort();
        assert_eq!(sync_operands(&text), want, "{entry}\n{text}");
    }
}

/// A fragment function's ordered atomic naming `mem_threadgroup` orders its tile, the device
/// memory it is kept in, at its order: a release add on the tile is `Release | UniformMemory` at
/// `Device` scope, where the flag was once dropped and the add left relaxed.
#[test]
fn a_fragment_ordered_tile_atomic_keeps_its_order() {
    let text = "target triple = \"air64_v29-apple-macosx27.0.0\"\n\n\
        define <4 x float> @f(ptr addrspace(3) %0) {\n  %r = call i32 \
        @air.atomic.local.add.u.i32(ptr addrspace(3) %0, i32 1, i32 3, i32 1, i32 2, i1 true)\n  \
        ret <4 x float> zeroinitializer\n}\n\n\
        declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i32, i1)\n\n\
        !air.fragment = !{!0}\n!air.version = !{!5}\n!0 = !{ptr @f, !1, !3}\n!1 = !{!2}\n\
        !2 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\", \
        !\"air.arg_name\", !\"color\"}\n!3 = !{!4}\n\
        !4 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, \
        !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 4, \
        !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
        !\"air.arg_name\", !\"bits\"}\n!5 = !{i32 2, i32 9, i32 0}\n";
    let bc = match assemble(text) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED a_fragment_ordered_tile_atomic_keeps_its_order: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let p = profile("nvidia-rtx50-linux", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(&bc, "f", &p).unwrap_or_else(|e| panic!("f: {e}"));
    let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("f: {e}"));
    let text = spirv_dis(&words);
    let vulkan = text.contains("OpMemoryModel Logical Vulkan");
    assert_eq!(
        sync_operands(&text),
        vec![(
            "OpAtomicIAdd".to_owned(),
            vec![1, under(vulkan, 0x4 | 0x40)]
        )],
        "{text}"
    );
}

/// A compare-exchange whose failure acquires where its success does not keeps the acquire: the
/// success is raised to acquire too, since SPIR-V's failure semantics may not be stronger than
/// the success's. `release`/`acquire` takes `AcquireRelease` and `Acquire`; `monotonic`/`acquire`
/// takes `Acquire` both ways.
#[test]
fn a_compare_exchange_failing_acquiring_keeps_its_acquire() {
    let text = "target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
        define void @k(ptr addrspace(1) %0) {\n\
        \x20 %a = cmpxchg ptr addrspace(1) %0, i32 0, i32 1 release acquire, align 4\n\
        \x20 %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1\n\
        \x20 %b = cmpxchg ptr addrspace(1) %p, i32 0, i32 1 monotonic acquire, align 4\n\
        \x20 %x = extractvalue { i32, i1 } %a, 0\n\
        \x20 %y = extractvalue { i32, i1 } %b, 0\n\
        \x20 %s = add i32 %x, %y\n\
        \x20 %q = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2\n\
        \x20 store i32 %s, ptr addrspace(1) %q, align 4\n\
        \x20 ret void\n}\n\n\
        !air.kernel = !{!0}\n!air.version = !{!5}\n!0 = !{ptr @k, !1, !2}\n!1 = !{}\n\
        !2 = !{!3}\n!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, \
        !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, \
        !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
        !\"air.arg_name\", !\"out\"}\n!5 = !{i32 2, i32 8, i32 0}\n";
    let bc = match assemble(text) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED a_compare_exchange_failing_acquiring_keeps_its_acquire: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let p = profile("nvidia-rtx50-linux", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("k: {e}"));
    let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("k: {e}"));
    let text = spirv_dis(&words);
    let vulkan = text.contains("OpMemoryModel Logical Vulkan");
    let cx = |success: u32, failure: u32| {
        (
            "OpAtomicCompareExchange".to_owned(),
            vec![
                1,
                under(vulkan, success | 0x40),
                under(vulkan, failure | 0x40),
            ],
        )
    };
    let mut want = vec![cx(0x8, 0x2), cx(0x2, 0x2)];
    want.sort();
    assert_eq!(sync_operands(&text), want, "{text}");
}

/// The words thread `t` of `a27_thread_atomics` writes (its source documents them).
fn thread_atomic_words(t: u32) -> [u32; 24] {
    let x = t.wrapping_mul(0x0101_0101).wrapping_add(3);
    let a = x.wrapping_add(3) & 0xf0f0_f0f0;
    let b = a | 0x0101;
    let c = b ^ 0xffff;
    let d = c.min(0x1000);
    let s = (x as i32).wrapping_sub(100);
    let f = t as f32 * 0.5;
    [
        x,
        x.wrapping_add(5),
        x.wrapping_add(3),
        a,
        b,
        c,
        d,
        d.max(0x800),
        1,
        77,
        0,
        88,
        88,
        s as u32,
        s.min(-7) as u32,
        3,
        2,
        f.to_bits(),
        (f + 1.25).to_bits(),
        (f + 0.5).to_bits(),
        (-2.5f32).to_bits(),
        x & 1,
        1,
        88,
    ]
}

/// The resources of a one-buffer kernel of one threadgroup of `threads` threads: buffer 0 of
/// `words` zero words and the local size.
fn one_buffer(plan: &Plan, words: usize, threads: u32) -> Resources {
    let mut spec = BTreeMap::new();
    spec.extend(crate::pipeline::threadgroup_constants([threads, 1, 1]));
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            // The buffer is bound at its start.
            (PushField::SlotBindOffset(_), PushLoc::Push { offset }) => {
                push[offset as usize..offset as usize + 4].fill(0)
            }
            other => panic!("unexpected push field {other:?}"),
        }
    }
    Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; words * 4],
        }],
        push,
        spec,
    }
}

/// MSL 4.1's atomics on thread memory (`air.atomic.private.*`) are the plain read-modify-writes of
/// their thread's word: on the CPU interpreter, in both scheduling orders, each thread of
/// `a27_thread_atomics` writes exactly the words its source documents, and the module validates on
/// every profile that compiles it, with no SPIR-V atomic.
#[test]
fn thread_atomics_compute_their_documented_words() {
    let path = engine_dir().join("a27_thread_atomics.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED thread_atomics_compute_their_documented_words: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let entry = "a27_thread_atomics";
    let threads = 8;
    let want: Vec<u32> = (0..threads).flat_map(thread_atomic_words).collect();
    for stem in ["moltenvk-m3", "lavapipe", "runtime-floor"] {
        let p = profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile");
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let text = spirv_dis(&words);
        assert!(
            !text.contains("OpAtomic"),
            "{stem}: a thread atomic emitted a SPIR-V atomic\n{text}"
        );
        let (plan, m) = typed_module(&bc, entry, &p).expect("module");
        let res = one_buffer(&plan, want.len(), threads);
        for order in [RunOrder::Forward, RunOrder::Reverse] {
            let cfg = Config {
                subgroup: SubgroupSize::W32,
                order,
                ..Config::default()
            };
            let out = run(&m, &cfg, res.clone()).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            let got: Vec<u32> = out.buffers[0]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got, want, "{stem} {order:?}");
        }
    }
}
