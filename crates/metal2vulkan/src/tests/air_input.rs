//! The engine's AIR input layer over the committed public fixtures.
//!
//! Every public `.ll` fixture is assembled with `llvm-as` at test time (opaque pointers) and every
//! `engine/*.air.ll` companion with `xcrun metal-as` (typed pointers, as Apple's compiler
//! writes them). The tests check that each decodes into a verified module whose functions have the
//! block count and opcode histogram of the text they were assembled from, that decoding is
//! deterministic, that the diagnostic printer output of the engine fixtures matches committed
//! golden snapshots, and that the typed and opaque spellings of one fixture decode to the same
//! instructions (they differ only in pointees). A missing tool skips its test visibly; a tool that
//! rejects a committed fixture fails it. A legacy top-level fixture whose hand-written metadata is
//! not the AIR ABI grammar is refused by decode and skipped visibly (`src/tests/air_abi.rs` pins
//! which).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::air::bitcode::{decode, decode_with, DecodeOptions, Decoded};
use crate::air::error::{Error, InputErrorKind};
use crate::air::ir::print::print;
use crate::air::ir::{Callee, CastOp, Const, ConstExpr, ConstId, FuncId, InstKind, Type, TypeId};
use crate::budget::{Budget, Meter};
use crate::pipeline::testing::Scratch;

// ---- Tools and fixtures ----

/// `llvm-as`: the `METAL2VULKAN_LLVM_AS` override, else `PATH`, else the Homebrew LLVM prefixes.
fn llvm_as() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("METAL2VULKAN_LLVM_AS") {
        return Some(PathBuf::from(p));
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .chain(
            ["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"]
                .iter()
                .map(PathBuf::from),
        )
        .map(|d| d.join("llvm-as"))
        .find(|p| p.is_file())
}

/// The major version `llvm-as --version` reports.
fn llvm_major(tool: &Path) -> Option<String> {
    let out = Command::new(tool).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let at = text.find("LLVM version ")? + "LLVM version ".len();
    let version = text[at..].split_whitespace().next()?;
    Some(version.split('.').next()?.to_string())
}

/// The LLVM major the owned fixtures' `.ll` companions are written for: the `llvm:` line of
/// `engine/TOOLCHAIN` (`scripts/regen-air-fixtures`; Linux CI's and the Linux gate image's
/// LLVM).
fn pinned_llvm_major() -> String {
    let path = engine_dir().join("TOOLCHAIN");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    text.lines()
        .find_map(|line| line.strip_prefix("llvm: "))
        .and_then(|rest| rest.split_whitespace().next())
        .unwrap_or_else(|| panic!("{path:?} has no `llvm: <major>` line"))
        .to_string()
}

/// An `llvm-as` of LLVM `major`: the first of `METAL2VULKAN_LLVM_AS`, `llvm-as-<major>` and
/// `llvm-as` on `PATH`, and Homebrew's `llvm@<major>` whose `--version` reports that major.
fn llvm_as_of_major(major: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::var_os("METAL2VULKAN_LLVM_AS")
        .map(PathBuf::from)
        .into_iter()
        .chain(
            [format!("llvm-as-{major}"), "llvm-as".to_string()]
                .iter()
                .flat_map(|name| {
                    std::env::split_paths(&path)
                        .map(|dir| dir.join(name))
                        .collect::<Vec<_>>()
                }),
        )
        .chain(
            ["/opt/homebrew/opt", "/usr/local/opt"]
                .iter()
                .map(|prefix| Path::new(prefix).join(format!("llvm@{major}/bin/llvm-as"))),
        )
        .filter(|tool| tool.is_file())
        .find(|tool| llvm_major(tool).as_deref() == Some(major))
}

fn have_metal_as() -> bool {
    cfg!(target_os = "macos")
        && Command::new("xcrun")
            .args(["-f", "metal-as"])
            .output()
            .is_ok_and(|o| o.status.success())
}

fn assemble(tool: &Path, args: &[&std::ffi::OsStr], out: &Path) -> Option<Vec<u8>> {
    let status = Command::new(tool)
        .args(args)
        .arg("-o")
        .arg(out)
        .output()
        .ok()?;
    if !status.status.success() {
        return None;
    }
    std::fs::read(out).ok()
}

fn llvm_assemble(tool: &Path, ll: &Path, scratch: &Scratch) -> Vec<u8> {
    try_llvm_assemble(tool, ll, scratch).unwrap_or_else(|| panic!("llvm-as rejected {ll:?}"))
}

/// `None` when this `llvm-as` rejects the text: a few legacy top-level fixtures are not valid
/// LLVM for every version (their text, not bitcode, was the old engine's input). The
/// engine fixtures must always assemble.
fn try_llvm_assemble(tool: &Path, ll: &Path, scratch: &Scratch) -> Option<Vec<u8>> {
    let out = scratch.path().join("fixture.bc");
    let bytes = assemble(tool, &[ll.as_os_str()], &out);
    assert!(
        bytes.is_some() || !ll.starts_with(engine_dir()),
        "llvm-as rejected {ll:?}"
    );
    bytes
}

fn metal_assemble(air_ll: &Path, scratch: &Scratch) -> Vec<u8> {
    let out = scratch.path().join("fixture.air");
    let args = [std::ffi::OsStr::new("metal-as"), air_ll.as_os_str()];
    assemble(Path::new("xcrun"), &args, &out)
        .unwrap_or_else(|| panic!("xcrun metal-as rejected {air_ll:?}"))
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/fixtures/public")
}

fn engine_dir() -> PathBuf {
    fixtures_dir().join("engine")
}

/// Every committed opaque-pointer `.ll` fixture in `dir`, sorted.
fn opaque_fixtures(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            name.ends_with(".ll") && !name.ends_with(".air.ll")
        })
        .collect();
    out.sort();
    out
}

fn all_opaque_fixtures() -> Vec<PathBuf> {
    let mut out = opaque_fixtures(&fixtures_dir());
    out.extend(opaque_fixtures(&engine_dir()));
    out
}

fn decode_ok(bytes: &[u8], what: &Path) -> Decoded {
    let mut m = Meter::new(&Budget::default());
    decode(bytes, &mut m).unwrap_or_else(|e| panic!("{what:?}: {e}"))
}

/// Decode for a comparison across assemblers: a marker declaration no use survives is how an LLVM
/// major spells the marker, so it is removed (`DecodeOptions::drop_unused_markers`).
fn decode_compared(bytes: &[u8], what: &Path) -> Decoded {
    let mut m = Meter::new(&Budget::default());
    let opts = DecodeOptions {
        drop_unused_markers: true,
        ..DecodeOptions::default()
    };
    decode_with(bytes, opts, &mut m).unwrap_or_else(|e| panic!("{what:?}: {e}"))
}

/// Decode a public fixture; `None` when a legacy top-level fixture's hand-written metadata is not
/// the AIR ABI grammar, which decode refuses as an ABI error (the ABI is decoded with the module).
/// The old engine read that text its own way; the engine fixtures must decode, and any other
/// refusal fails.
fn decode_fixture(bytes: &[u8], what: &Path) -> Option<Decoded> {
    let mut m = Meter::new(&Budget::default());
    match decode(bytes, &mut m) {
        Ok(d) => Some(d),
        Err(Error::InvalidInput(e))
            if matches!(e.kind, InputErrorKind::Abi(_)) && !what.starts_with(engine_dir()) =>
        {
            None
        }
        Err(e) => panic!("{what:?}: {e}"),
    }
}

// ---- Structural shape: text versus decoded IR ----

/// Per defined function: its block count (one terminator per block) and opcode histogram.
type Shape = BTreeMap<String, (usize, BTreeMap<String, usize>)>;

const TERMINATORS: [&str; 4] = ["ret", "br", "switch", "unreachable"];

/// Callees whose calls decode drops, as `llvm-dis` spells them.
const DROPPED: [&str; 6] = [
    "@llvm.lifetime.start.",
    "@llvm.lifetime.end.",
    "@llvm.assume(",
    "@llvm.experimental.noalias.scope.decl(",
    "@llvm.agx3.yield(",
    "void asm sideeffect \"\", \"\"()",
];

/// The name of the function a `define`/`declare` line names (`@name(` or `@"name"(`).
fn text_fn_name(line: &str) -> Option<String> {
    let at = line.find('@')? + 1;
    let rest = &line[at..];
    if let Some(quoted) = rest.strip_prefix('"') {
        return Some(quoted[..quoted.find('"')?].to_string());
    }
    Some(rest[..rest.find('(')?].to_string())
}

/// The shape of every `define` in LLVM text as `llvm-dis` writes it.
fn text_shape(text: &str) -> Shape {
    let mut out = Shape::new();
    let mut current: Option<(String, usize, BTreeMap<String, usize>)> = None;
    let mut in_switch = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("define ") {
            let name = text_fn_name(line).expect("function name");
            current = Some((name, 0, BTreeMap::new()));
            continue;
        }
        let Some((name, blocks, ops)) = current.as_mut() else {
            continue;
        };
        if line == "}" {
            let (name, blocks, ops) = current.take().unwrap();
            out.insert(name, (blocks, ops));
            continue;
        }
        if in_switch {
            in_switch = line != "]";
            continue;
        }
        if line.is_empty() || line.starts_with(';') {
            continue;
        }
        // Block labels: `name:` optionally followed by a `; preds` comment.
        let head = line.split(';').next().unwrap().trim_end();
        if head.ends_with(':') && !head.contains(' ') {
            continue;
        }
        let body = match line.find(" = ") {
            Some(at) if line.starts_with('%') => &line[at + 3..],
            _ => line,
        };
        let mut words = body.split_whitespace();
        let mut op = words.next().unwrap();
        if matches!(op, "tail" | "musttail" | "notail") {
            op = words.next().unwrap();
        }
        if op == "call" && DROPPED.iter().any(|d| body.contains(d)) {
            continue;
        }
        if op == "switch" && body.ends_with('[') {
            in_switch = true;
        }
        if TERMINATORS.contains(&op) {
            *blocks += 1;
        }
        *ops.entry(op.to_string()).or_default() += 1;
        let _ = name;
    }
    out
}

/// The LLVM mnemonic of an instruction, as the text scanner sees it.
fn mnemonic(kind: &InstKind) -> &'static str {
    match kind {
        InstKind::BinOp { op, .. } => op.mnemonic(),
        InstKind::FNeg { .. } => "fneg",
        InstKind::Cast { op, .. } => op.mnemonic(),
        InstKind::ICmp { .. } => "icmp",
        InstKind::FCmp { .. } => "fcmp",
        InstKind::Select { .. } => "select",
        InstKind::Freeze(_) => "freeze",
        InstKind::ExtractElement { .. } => "extractelement",
        InstKind::InsertElement { .. } => "insertelement",
        InstKind::ShuffleVector { .. } => "shufflevector",
        InstKind::ExtractValue { .. } => "extractvalue",
        InstKind::InsertValue { .. } => "insertvalue",
        InstKind::Alloca { .. } => "alloca",
        InstKind::Load { .. } => "load",
        InstKind::Store { .. } => "store",
        InstKind::Gep { .. } => "getelementptr",
        InstKind::AtomicRmw { .. } => "atomicrmw",
        InstKind::CmpXchg { .. } => "cmpxchg",
        InstKind::Fence(_) => "fence",
        InstKind::Call(_) => "call",
        InstKind::Phi { .. } => "phi",
        InstKind::Ret(_) => "ret",
        InstKind::Br { .. } | InstKind::CondBr { .. } => "br",
        InstKind::Switch { .. } => "switch",
        InstKind::Unreachable => "unreachable",
    }
}

fn name(d: &Decoded, sym: crate::air::ir::Sym) -> String {
    String::from_utf8_lossy(d.module.symbols().get(sym).unwrap()).into_owned()
}

fn ir_shape(d: &Decoded) -> Shape {
    let mut out = Shape::new();
    for f in d.module.parts().funcs.values() {
        let Some(body) = &f.body else {
            continue;
        };
        let mut ops = BTreeMap::new();
        for inst in body.insts.values() {
            *ops.entry(mnemonic(&inst.kind).to_string()).or_default() += 1;
        }
        out.insert(name(d, f.name), (body.blocks.len(), ops));
    }
    out
}

#[test]
fn every_public_fixture_decodes_into_its_text_shape() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP every_public_fixture_decodes_into_its_text_shape: llvm-as not found");
        return;
    };
    let scratch = Scratch::new("m2v-air-input");
    let (mut decoded, mut skipped, mut refused) = (0, Vec::new(), Vec::new());
    for ll in all_opaque_fixtures() {
        let Some(bytes) = try_llvm_assemble(&tool, &ll, &scratch) else {
            skipped.push(ll);
            continue;
        };
        let Some(d) = decode_fixture(&bytes, &ll) else {
            refused.push(ll);
            continue;
        };
        let text = std::fs::read_to_string(&ll).unwrap();
        assert_eq!(ir_shape(&d), text_shape(&text), "{ll:?}");
        assert!(d.pointees.is_empty(), "{ll:?}: opaque bitcode has pointees");
        decoded += 1;
    }
    if !skipped.is_empty() {
        eprintln!("SKIP (llvm-as rejected the text): {skipped:?}");
    }
    if !refused.is_empty() {
        eprintln!("SKIP (legacy ABI metadata refused): {refused:?}");
    }
    assert!(decoded > 100, "only {decoded} fixtures decoded");
}

/// Every intrinsic call in the public fixtures decodes, or belongs to a family the decoder lists
/// as not decoded.
#[test]
fn every_public_intrinsic_call_decodes_or_is_listed() {
    use crate::air::intrinsics::undecoded_family;
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP every_public_intrinsic_call_decodes_or_is_listed: llvm-as not found");
        return;
    };
    let scratch = Scratch::new("m2v-air-input");
    let (mut decoded, mut unlisted) = (0, Vec::new());
    for ll in all_opaque_fixtures() {
        let Some(bytes) = try_llvm_assemble(&tool, &ll, &scratch) else {
            continue;
        };
        let Some(d) = decode_fixture(&bytes, &ll) else {
            continue;
        };
        for f in d.module.parts().funcs.values() {
            let Some(body) = &f.body else {
                continue;
            };
            for inst in body.insts.values() {
                let InstKind::Call(call) = &inst.kind else {
                    continue;
                };
                match call.callee {
                    Callee::Air { .. } | Callee::Llvm { .. } => decoded += 1,
                    Callee::UnknownIntrinsic(f) => {
                        let callee = callee_name(&d, f);
                        if undecoded_family(&callee).is_none() {
                            unlisted.push(format!("{}: {callee}", ll.display()));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(unlisted.is_empty(), "{unlisted:#?}");
    // The ray-intersection fixtures, once the listed family, decode since `air.intersect` did.
    assert!(decoded > 900, "only {decoded} intrinsic calls decoded");
}

#[test]
fn decoding_is_deterministic() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP decoding_is_deterministic: llvm-as not found");
        return;
    };
    let scratch = Scratch::new("m2v-air-input");
    for ll in all_opaque_fixtures() {
        let Some(bytes) = try_llvm_assemble(&tool, &ll, &scratch) else {
            continue;
        };
        let Some(a) = decode_fixture(&bytes, &ll) else {
            continue;
        };
        let b = decode_ok(&bytes, &ll);
        assert_eq!(print(&a.module), print(&b.module), "{ll:?}");
        assert_eq!(a, b, "{ll:?}");
    }
}

// ---- Golden printer snapshots ----

fn golden_path(ll: &Path) -> PathBuf {
    let stem = ll.file_stem().unwrap().to_string_lossy().into_owned();
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/air_input")
        .join(format!("{stem}.txt"))
}

/// The golden text: the printer output of the module decoded for comparison
/// ([`decode_compared`]). It is recorded under the pinned LLVM major ([`pinned_llvm_major`]), the
/// major the `.ll` companions are written for: `scripts/regen-air-fixtures` writes each with
/// `llvm-dis` of that major and refuses any other, and
/// `metal2vulkan-validation`'s `every_committed_ll_fixture_assembles_with_the_pinned_llvm_and_decodes`
/// holds every committed `.ll` to it. Every gate compares the golden, and no fixture is exempt.
fn golden_text(d: &Decoded) -> String {
    print(&d.module)
}

/// `text` with every type and constant id renumbered by order of first appearance, which is what
/// two printouts of one fixture under two LLVM majors may differ in and nothing else.
///
/// LLVM numbers a module's types and constants in the order its writer reaches them, and a marker
/// declaration's type is part of that order: LLVM 21 spells `llvm.lifetime.start(i64, ptr)` and
/// LLVM 22 and later `llvm.lifetime.start(ptr)`, so a type the marker's signature alone reached
/// first (LLVM 21 numbers `i64` there) is numbered later under a newer major and every type after
/// it moves down one. Dropping the unused marker (`DecodeOptions::drop_unused_markers`, which
/// [`decode_compared`] asks for) does not undo that: it renumbers what remains in its old order,
/// and the orders themselves differ (`air/src/bitcode/markers.rs`).
///
/// A quoted name is copied through, since a name may read like an id (`"c0"` is an MSL name in
/// these fixtures); only `%t<n>` and `c<n>` outside quotes are rewritten. Renaming cannot hide a
/// changed printout: a line the renumbering does not explain still differs, because the mapping is
/// by first appearance and every later use must agree with it.
fn ids_by_first_appearance(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut slots: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let bytes = text.as_bytes();
    let word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let (mut i, mut quoted) = (0, false);
    while i < bytes.len() {
        if bytes[i] == b'"' {
            quoted = !quoted;
            out.push('"');
            i += 1;
            continue;
        }
        // An id is `%t<n>` or `c<n>` with no word character on either side: `c` in `bitcast` has
        // one before it, and `%t1` in a longer `%t12` has one after.
        let prefix = (!quoted)
            .then(|| {
                ["%t", "c"].into_iter().find(|p| {
                    text[i..].starts_with(p)
                        && (i == 0 || !word(bytes[i - 1]))
                        && bytes[i + p.len()..].first().is_some_and(u8::is_ascii_digit)
                })
            })
            .flatten();
        let Some(prefix) = prefix else {
            let c = text[i..].chars().next().expect("char boundary");
            out.push(c);
            i += c.len_utf8();
            continue;
        };
        let start = i + prefix.len();
        let end = start
            + bytes[start..]
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .count();
        if bytes.get(end).is_some_and(|&b| word(b)) {
            out.push_str(prefix);
            i = start;
            continue;
        }
        let next = counts.entry(prefix).or_default();
        let slot = *slots.entry((prefix, &text[start..end])).or_insert_with(|| {
            *next += 1;
            *next
        });
        out.push_str(prefix);
        out.push('#');
        out.push_str(&slot.to_string());
        i = end;
    }
    out
}

/// What [`ids_by_first_appearance`] may and may not hide: a shift of every id after one point
/// compares equal, while a quoted name that reads like an id, an id inside a word, and an id
/// changed in only some of its uses do not.
#[test]
fn renumbering_ids_hides_a_shift_and_nothing_else() {
    let same = |a: &str, b: &str| ids_by_first_appearance(a) == ids_by_first_appearance(b);
    // The real case: the types after the marker's move down one.
    assert!(same(
        "%t23 = type void ()\n%t26 = type [2 x i32]\nc25 = %t26 [27, 0]",
        "%t23 = type void ()\n%t25 = type [2 x i32]\nc25 = %t25 [27, 0]",
    ));
    // A name may read like an id; these fixtures have `"c0"`. It is not renumbered, so two
    // printouts naming different ones differ.
    assert!(!same(r#"%a0="c0" c1 = i32 1"#, r#"%a0="c9" c1 = i32 1"#));
    assert_eq!(
        ids_by_first_appearance(r#"call @f3 "c0" (c7)"#),
        r#"call @f3 "c0" (c#1)"#
    );
    // An id in only some of its uses, and a changed value, are not a shift.
    assert!(!same(
        "%t5 = type i32\nc1 = %t5 0\nc2 = %t5 1",
        "%t5 = type i32\nc1 = %t9 0\nc2 = %t5 1"
    ));
    assert!(!same("c25 = %t26 [27, 0]", "c25 = %t26 [28, 0]"));
    // `c` inside a word is not an id, and `%t1` is not the start of `%t12`.
    assert_eq!(
        ids_by_first_appearance("bitcast %t12 c3"),
        "bitcast %t#1 c#1"
    );
    assert!(!same("%t1 = type i32", "%t12 = type i32\n%t1 = type i64"));
}

/// Why a golden printout's type and constant ids may differ off the pinned major
/// (`air/src/bitcode/markers.rs`).
const GOLDEN_RENUMBER_CAUSE: &str = "LLVM 21 spells llvm.lifetime.start(i64, ptr) and LLVM 22 and \
     later llvm.lifetime.start(ptr), so a type the marker's signature alone reached first is \
     numbered later and the types after it move down one";

/// Every owned fixture's golden printout, under the host's `llvm-as` and under one of the pinned
/// major ([`pinned_llvm_major`]) when the host's is another major.
///
/// Under the pinned major -- the major the goldens are recorded under, Linux CI's own `llvm-as` --
/// every golden must match to the byte. Under any other major the content must still match, but
/// type and constant ids may be renumbered, because LLVM's value enumeration is the assembler's
/// and a marker declaration's signature is part of the order it walks
/// ([`ids_by_first_appearance`]). Such a fixture is reported by name and counted, not excused by a
/// list: the comparison still runs, and a line the renumbering does not explain fails. Anything
/// else -- a changed type, constant, instruction, or metadata -- fails under every major.
///
/// Without an `llvm-as` of the pinned major that leg is skipped visibly, unless
/// `METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM=1`, which makes it a failure.
#[test]
fn engine_fixtures_match_their_golden_printout() {
    let test = "engine_fixtures_match_their_golden_printout";
    let mut tools = Vec::new();
    match llvm_as() {
        Some(tool) => tools.push(tool),
        None => eprintln!("SKIP {test}: llvm-as not found"),
    }
    let major = pinned_llvm_major();
    if !tools
        .iter()
        .any(|tool| llvm_major(tool).as_deref() == Some(major.as_str()))
    {
        match llvm_as_of_major(&major) {
            Some(tool) => tools.push(tool),
            None => {
                let why = format!("no llvm-as of the pinned LLVM {major} found");
                assert!(
                    std::env::var_os("METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM")
                        .is_none_or(|v| v != "1"),
                    "{test}: METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM=1: {why}"
                );
                eprintln!("SKIP {test} under LLVM {major}: {why}");
            }
        }
    }
    let scratch = Scratch::new("m2v-air-input");
    let (mut differ, mut renumbered) = (Vec::new(), Vec::new());
    for tool in &tools {
        let version = llvm_major(tool).unwrap_or_else(|| "?".to_string());
        let pinned = version == major;
        let mut compared = 0;
        for ll in opaque_fixtures(&engine_dir()) {
            let golden = std::fs::read_to_string(golden_path(&ll))
                .unwrap_or_else(|e| panic!("{:?}: {e}", golden_path(&ll)));
            let d = decode_compared(&llvm_assemble(tool, &ll, &scratch), &ll);
            let text = golden_text(&d);
            let name = ll.file_name().unwrap().to_string_lossy().into_owned();
            if text == golden {
                compared += 1;
                continue;
            }
            // Off the pinned major, ids renumbered by LLVM's value enumeration are the one
            // accepted difference, and only once the rest of the printout is shown to match.
            let (canon_text, canon_golden) = (
                ids_by_first_appearance(&text),
                ids_by_first_appearance(&golden),
            );
            let ids_only = !pinned && canon_text == canon_golden;
            // Which line to name: off the pinned major the ids differ by design, so the first
            // line that differs once they are renumbered is the one that changed.
            let first_differing = |want: &str, got: &str| {
                want.lines()
                    .chain(std::iter::repeat(""))
                    .zip(got.lines().chain(std::iter::repeat("")))
                    .position(|(want, got)| want != got)
                    .expect("texts that differ differ in a line")
            };
            let line = if pinned || ids_only {
                first_differing(&golden, &text)
            } else {
                first_differing(&canon_golden, &canon_text)
            };
            let at = |text: &str| text.lines().nth(line).unwrap_or("").to_string();
            let (want, got) = (at(&golden), at(&text));
            if ids_only {
                renumbered.push(format!(
                    "{name} (first at line {}: `{want}` -> `{got}`)",
                    line + 1
                ));
            } else {
                differ.push(format!(
                    "{name}: golden recorded under the pinned LLVM {major}, printed under LLVM \
                     {version} ({tool:?}): line {}: golden `{want}`, printed `{got}`{}",
                    line + 1,
                    if pinned {
                        ""
                    } else {
                        " (not a type/constant renumbering: the printout changed)"
                    }
                ));
            }
            compared += 1;
        }
        assert!(compared > 0);
    }
    if !renumbered.is_empty() {
        eprintln!(
            "{test}: {} fixture(s) print the same module with type/constant ids renumbered off the \
             pinned LLVM {major} ({GOLDEN_RENUMBER_CAUSE}): {}",
            renumbered.len(),
            renumbered.join(", ")
        );
    }
    assert!(
        differ.is_empty(),
        "{} golden printout(s) differ in content, not only in type/constant ids:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

/// Rewrite the golden snapshots: `cargo test --all-features --lib tests::air_input -- --ignored`.
///
/// Only an `llvm-as` of the pinned major ([`pinned_llvm_major`]) may record them: the goldens are
/// that major's printout, and recording them under the host's newer `llvm-as` would renumber the
/// types of a fixture holding a marker and fail on Linux CI, whose `llvm-as` is the pinned major.
#[test]
#[ignore]
fn bless_engine_goldens() {
    let major = pinned_llvm_major();
    let tool = llvm_as_of_major(&major).unwrap_or_else(|| {
        panic!("no llvm-as of the pinned LLVM {major}: the goldens are that major's printout")
    });
    let scratch = Scratch::new("m2v-air-input");
    for ll in opaque_fixtures(&engine_dir()) {
        let d = decode_compared(&llvm_assemble(&tool, &ll, &scratch), &ll);
        let path = golden_path(&ll);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, golden_text(&d)).unwrap();
    }
}

// ---- Typed versus opaque pointers ----

/// An id-free rendering of every instruction of every defined function: types and constants by
/// structure, globals and functions by name, arguments and instructions by index. A call to a
/// decoded intrinsic renders its decoded operation, and the intrinsic's declaration renders
/// without its name: an overloaded `llvm.*` name mangles its pointer operands' types
/// (`llvm.memcpy.p1i8.p1i8.i64` typed, `llvm.memcpy.p1.p1.i64` opaque), which decode erases.
/// Definitions are rendered in order and declarations as a set: the opaque text is `llvm-dis` of
/// the typed module, and LLVM's reader re-mangles a typed-pointer intrinsic into a new declaration
/// after the module's others, so a declaration's position is the reader's.
fn structural(d: &Decoded) -> Vec<String> {
    let m = &d.module;
    let ty = |t: TypeId| render_type(d, t);
    let value = |v: crate::air::ir::Value| match v {
        crate::air::ir::Value::Arg(a) => format!("%a{}", a.raw()),
        crate::air::ir::Value::Inst(i) => format!("%i{}", i.raw()),
        crate::air::ir::Value::Const(c) => render_const(d, c),
    };
    let decoded: std::collections::BTreeSet<FuncId> = m
        .parts()
        .funcs
        .values()
        .filter_map(|f| f.body.as_ref())
        .flat_map(|b| b.insts.values())
        .filter_map(|inst| match &inst.kind {
            InstKind::Call(call) => match call.callee {
                Callee::Air { decl, .. } | Callee::Llvm { decl, .. } => Some(decl),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let mut out = Vec::new();
    let mut declared = Vec::new();
    for (id, f) in m.parts().funcs.iter() {
        let line = if decoded.contains(&id) {
            format!("fn intrinsic {}", ty(f.ty))
        } else {
            format!("fn {} {}", name(d, f.name), ty(f.ty))
        };
        let Some(body) = &f.body else {
            declared.push(line);
            continue;
        };
        out.push(line);
        for (b, block) in body.blocks.iter() {
            out.push(format!(
                "bb{} [{}..{})",
                b.raw(),
                block.start.raw(),
                block.end.raw()
            ));
        }
        for inst in body.insts.values() {
            let mut operands = Vec::new();
            inst.kind.for_each_value(|v| operands.push(value(v)));
            let extra = match &inst.kind {
                InstKind::Call(call) => match call.callee {
                    Callee::Direct(f) | Callee::UnknownIntrinsic(f) => callee_name(d, f),
                    Callee::Indirect { fn_ty, .. } => format!("indirect {}", ty(fn_ty)),
                    Callee::Air { op, .. } => format!("{op:?}"),
                    Callee::Llvm { op, .. } => format!("{op:?}"),
                },
                InstKind::Gep { src_ty, .. } => ty(*src_ty),
                InstKind::Alloca { ty: t, .. } => ty(*t),
                _ => String::new(),
            };
            out.push(format!(
                "{} {} {extra} ({})",
                mnemonic(&inst.kind),
                ty(inst.ty),
                operands.join(", ")
            ));
        }
    }
    declared.sort();
    out.extend(declared);
    out
}

fn callee_name(d: &Decoded, f: FuncId) -> String {
    name(d, d.module.func(f).unwrap().name)
}

fn render_type(d: &Decoded, t: TypeId) -> String {
    let render = |t: TypeId| render_type(d, t);
    match d.module.types().get(t).unwrap() {
        Type::Vector { elem, len } => format!("<{len} x {}>", render(*elem)),
        Type::Array { elem, len } => format!("[{len} x {}]", render(*elem)),
        Type::Struct { fields, packed, .. } => {
            let f: Vec<_> = fields.iter().map(|&f| render(f)).collect();
            format!(
                "{}{{{}}}",
                if *packed { "packed" } else { "" },
                f.join(", ")
            )
        }
        Type::Function { ret, params, .. } => {
            let p: Vec<_> = params.iter().map(|&p| render(p)).collect();
            format!("{}({})", render(*ret), p.join(", "))
        }
        other => format!("{other:?}"),
    }
}

/// A constant, id-free. A constant `bitcast` whose operand already has the result type renders as
/// its operand: after pointee erasure that is a typed-pointer cast (a constexpr sampler global cast
/// to the sampler pointer type, say) which the opaque spelling never has and which canon erases
/// (`air::canon::erase`). So does a constant `getelementptr` with only zero indices and its base's
/// type (an array's first element as an intrinsic operand), which LLVM folds to the base when it
/// reads typed pointers as opaque ones and which canon erases too.
fn render_const(d: &Decoded, c: ConstId) -> String {
    let k = d.module.consts().get(c).unwrap();
    let consts = d.module.consts();
    match &k.value {
        Const::Expr(ConstExpr::Cast {
            op: CastOp::Bitcast,
            ops: [v],
        }) if consts.ty(*v) == Some(k.ty) => return render_const(d, *v),
        Const::Expr(ConstExpr::Gep { ops, .. })
            if ops.first().is_some_and(|&b| consts.ty(b) == Some(k.ty))
                && ops.iter().skip(1).all(|&i| {
                    matches!(
                        consts.get(i).map(|x| &x.value),
                        Some(Const::Int(0) | Const::Null)
                    )
                }) =>
        {
            return render_const(d, ops[0]);
        }
        _ => {}
    }
    let t = render_type(d, k.ty);
    let v = match &k.value {
        Const::Aggregate(elems) => {
            let e: Vec<_> = elems.iter().map(|&e| render_const(d, e)).collect();
            format!("[{}]", e.join(", "))
        }
        Const::GlobalAddr(g) => format!("@{}", name(d, d.module.global(*g).unwrap().name)),
        Const::FuncAddr(f) => format!("@{}", callee_name(d, *f)),
        Const::Expr(e) => {
            let ops = match e {
                ConstExpr::Cast { ops, .. } => ops.as_slice(),
                ConstExpr::BinOp { ops, .. } => ops.as_slice(),
                ConstExpr::Gep { ops, .. } => ops,
            };
            let o: Vec<_> = ops.iter().map(|&o| render_const(d, o)).collect();
            format!("expr({})", o.join(", "))
        }
        other => format!("{other:?}"),
    };
    format!("{t} {v}")
}

/// Whether some function's parameters or result include a pointer.
fn signature_has_pointer(d: &Decoded) -> bool {
    let types = d.module.types();
    let is_ptr = |t: TypeId| matches!(types.get(t), Some(Type::Ptr { .. }));
    d.module
        .parts()
        .funcs
        .values()
        .any(|f| match types.get(f.ty) {
            Some(Type::Function { ret, params, .. }) => {
                is_ptr(*ret) || params.iter().any(|&p| is_ptr(p))
            }
            _ => false,
        })
}

#[test]
fn typed_and_opaque_fixtures_decode_to_the_same_instructions() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP typed_and_opaque_fixtures_decode_to_the_same_instructions: no llvm-as");
        return;
    };
    if !have_metal_as() {
        eprintln!(
            "SKIP typed_and_opaque_fixtures_decode_to_the_same_instructions: no xcrun metal-as"
        );
        return;
    }
    let scratch = Scratch::new("m2v-air-input");
    let mut compared = 0;
    for ll in opaque_fixtures(&engine_dir()) {
        let air_ll = ll.with_extension("air.ll");
        let opaque = decode_compared(&llvm_assemble(&tool, &ll, &scratch), &ll);
        let typed = decode_compared(&metal_assemble(&air_ll, &scratch), &air_ll);
        assert_eq!(structural(&opaque), structural(&typed), "{ll:?}");
        assert!(opaque.pointees.is_empty(), "{ll:?}");
        // A typed spelling records a pointee wherever a signature has a pointer; a module
        // without one (an entry whose arguments are all values) has none to record.
        assert_eq!(
            !typed.pointees.is_empty(),
            signature_has_pointer(&opaque),
            "{ll:?}"
        );
        compared += 1;
    }
    assert!(compared > 0);
}

/// Assemble `text` with `llvm-as` and decode it; `None` when `llvm-as` is missing.
fn decode_text(text: &str) -> Option<Result<Decoded, Error>> {
    let tool = llvm_as()?;
    let scratch = Scratch::new("m2v-air-input");
    let ll = scratch.path().join("input.ll");
    std::fs::write(&ll, text).unwrap();
    let bytes = llvm_assemble(&tool, &ll, &scratch);
    let mut m = Meter::new(&Budget::default());
    Some(decode(&bytes, &mut m))
}

/// The calls of the function named `func`, in order.
fn calls<'d>(d: &'d Decoded, func: &str) -> Vec<&'d crate::air::ir::Call> {
    let f = d
        .module
        .parts()
        .funcs
        .values()
        .find(|f| name(d, f.name) == func)
        .unwrap();
    f.body
        .as_ref()
        .unwrap()
        .insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::Call(c) => Some(&**c),
            _ => None,
        })
        .collect()
}

/// Written in the LLVM 22 lifetime-marker syntax; see `markers_and_intrinsics`.
const MARKERS_AND_INTRINSICS: &str = r#"
define i32 @k(ptr addrspace(1) %p, i32 %x, i1 %c) {
entry:
  %a = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.experimental.noalias.scope.decl(metadata !0)
  call void @llvm.assume(i1 %c)
  %old = call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %p, i32 %x, i32 0, i32 2, i1 true), !range !3
  call void @llvm.lifetime.end.p0(ptr %a)
  %z = call i32 @air.clz.i32(i32 %old, i1 false), !range !4
  %f = call float @air.convert.f.f32.u.i32(i32 %z)
  %g = call fast float @llvm.fmuladd.f32(float %f, float %f, float %f)
  %u = call float @air.unmodeled_family.f32(float %g)
  br label %loop

loop:
  br i1 %c, label %loop, label %exit, !llvm.loop !5

exit:
  ret i32 %z
}

declare void @llvm.lifetime.start.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr)
declare void @llvm.experimental.noalias.scope.decl(metadata)
declare void @llvm.assume(i1)
declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.clz.i32(i32, i1)
declare float @air.convert.f.f32.u.i32(i32)
declare float @llvm.fmuladd.f32(float, float, float)
declare float @air.unmodeled_family.f32(float)

!0 = !{!1}
!1 = distinct !{!1, !2}
!2 = distinct !{!2}
!3 = !{i32 0, i32 100}
!4 = !{i32 0, i32 33}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
"#;

/// `MARKERS_AND_INTRINSICS` in the lifetime-marker syntax of `llvm-as` major `major`. LLVM 22
/// dropped the markers' leading `i64` size operand, and an `llvm-as` before 22 rejects the
/// one-operand text (Ubuntu 26.04, the Linux CI image, ships LLVM 21). The decoder takes both
/// shapes as lifetime markers, so either spelling exercises the same drop.
fn markers_and_intrinsics(major: u32) -> String {
    if major >= 22 {
        return MARKERS_AND_INTRINSICS.to_owned();
    }
    let text = MARKERS_AND_INTRINSICS
        .replace(".p0(ptr %a)", ".p0(i64 4, ptr %a)")
        .replace(".p0(ptr)\n", ".p0(i64, ptr)\n");
    assert_eq!(text.matches(".p0(i64 4, ptr %a)").count(), 2);
    assert_eq!(text.matches(".p0(i64, ptr)").count(), 2);
    text
}

#[test]
fn markers_are_dropped_and_intrinsics_decode_once() {
    use crate::air::intrinsics::{
        AirIntrinsic, Atomic, LlvmFloatOp, LlvmIntrinsic, Math, MathFn, MemoryOrder, ThreadScope,
    };
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP markers_are_dropped_and_intrinsics_decode_once: llvm-as not found");
        return;
    };
    let major = llvm_major(&tool)
        .and_then(|major| major.parse().ok())
        .expect("llvm-as --version");
    let d = decode_text(&markers_and_intrinsics(major))
        .expect("llvm-as was found above")
        .unwrap();
    let calls = calls(&d, "k");
    // The four markers (including the metadata-operand `noalias.scope.decl`) are gone.
    assert_eq!(calls.len(), 5);
    let Callee::Air { op, .. } = calls[0].callee else {
        panic!("{:?}", calls[0].callee);
    };
    let AirIntrinsic::Atomic(Atomic {
        order,
        scope,
        volatile,
        ..
    }) = op
    else {
        panic!("{op:?}");
    };
    assert_eq!(
        (order, scope, volatile),
        (MemoryOrder::Relaxed, ThreadScope::Device, true)
    );
    assert!(matches!(
        calls[1].callee,
        Callee::Air {
            op: AirIntrinsic::Math(Math {
                f: MathFn::Clz { zero_undef: false },
                ..
            }),
            ..
        }
    ));
    assert!(matches!(
        calls[2].callee,
        Callee::Air {
            op: AirIntrinsic::Convert(_),
            ..
        }
    ));
    assert!(matches!(
        calls[3].callee,
        Callee::Llvm {
            op: LlvmIntrinsic::Float {
                op: LlvmFloatOp::FMulAdd,
                ..
            },
            ..
        }
    ));
    assert!(matches!(calls[4].callee, Callee::UnknownIntrinsic(_)));
    // Attachments after a dropped marker still land on their own instruction.
    let range = |c: &crate::air::ir::Call| c.range.as_ref().map(|r| r.ranges.to_vec());
    assert_eq!(range(calls[0]), Some(vec![(0, 100)]));
    assert_eq!(range(calls[1]), Some(vec![(0, 33)]));
    assert_eq!(range(calls[2]), None);
    let body = d
        .module
        .parts()
        .funcs
        .values()
        .find_map(|f| f.body.as_ref())
        .unwrap();
    let hints = body.insts.values().find_map(|i| i.kind.loop_hints());
    assert!(hints.is_some_and(|h| h.must_progress));
}

/// A declaration typed unlike every form of its intrinsic's signature is reported by name, at the
/// function whose body calls it. Without the naming the report is a bitstream position alone,
/// which is what an outside report of an AIR 2.1 library arrived as.
#[test]
fn a_declaration_mismatch_names_the_declaration_and_its_function() {
    use crate::air::error::{DiagName, InputErrorKind, IntrinsicError};
    // A `texture2d<float>::read` that returns a residency status (AIR 2.2 and later) and states
    // no trailing access qualifier (AIR 1.x alone): a mixture of two emitter steps, which no
    // Metal compiler writes and no form of the operation is.
    let text = "declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, i32)\n\
         define void @sampleOne(ptr addrspace(1) %t, ptr addrspace(1) %o) {\n\
         \x20 %c = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> zeroinitializer, i32 0)\n\
         \x20 %v = extractvalue { <4 x float>, i8 } %c, 0\n\
         \x20 store <4 x float> %v, ptr addrspace(1) %o\n\
         \x20 ret void\n}\n";
    let Some(got) = decode_text(text) else {
        eprintln!(
            "SKIP a_declaration_mismatch_names_the_declaration_and_its_function: llvm-as not found"
        );
        return;
    };
    let Err(Error::InvalidInput(e)) = got else {
        panic!("{got:?}");
    };
    assert_eq!(
        e.kind,
        InputErrorKind::Intrinsic(IntrinsicError::DeclaredSignatureMismatch {
            name: DiagName::new("air.read_texture_2d.v4f32"),
        }),
    );
    assert_eq!(
        e.at.in_function().map(|(f, b, i)| (f.to_string(), b, i)),
        Some(("sampleOne".to_owned(), None, None)),
    );
    // Both names reach the message a consumer and the command line print.
    let shown = e.to_string();
    assert!(shown.contains("air.read_texture_2d.v4f32"), "{shown}");
    assert!(shown.contains("sampleOne"), "{shown}");
}

#[test]
fn intrinsic_call_errors_are_typed() {
    use crate::air::error::{DiagName, InputErrorKind, IntrinsicError};
    let cases = [
        (
            "declare float @air.fmax.f32(float)\n\
             define float @k(float %x) {\n  %r = call float @air.fmax.f32(float %x)\n  ret float %r\n}\n",
            Some(IntrinsicError::DeclaredSignatureMismatch {
                name: DiagName::new("air.fmax.f32"),
            }),
        ),
        (
            "declare i32 @air.clz.i32(i32, i1)\n\
             define i32 @k(i32 %x, i1 %f) {\n  %r = call i32 @air.clz.i32(i32 %x, i1 %f)\n  ret i32 %r\n}\n",
            Some(IntrinsicError::ImmediateNotConstant),
        ),
        // A declared but never called mismatch is not an error; an unmodeled scope keeps the
        // call unknown.
        (
            "declare float @air.fmax.f32(float)\n\
             declare i32 @air.atomic.local.load.i32(ptr addrspace(3), i32, i32, i1)\n\
             define i32 @k(ptr addrspace(3) %p) {\n  %r = call i32 @air.atomic.local.load.i32(ptr addrspace(3) %p, i32 0, i32 9, i1 true)\n  ret i32 %r\n}\n",
            None,
        ),
    ];
    for (text, want) in cases {
        let Some(got) = decode_text(text) else {
            eprintln!("SKIP intrinsic_call_errors_are_typed: llvm-as not found");
            return;
        };
        match (got, want) {
            (Err(Error::InvalidInput(e)), Some(want)) => {
                assert_eq!(e.kind, InputErrorKind::Intrinsic(want), "{text}")
            }
            (Ok(d), None) => {
                assert!(matches!(
                    calls(&d, "k")[0].callee,
                    Callee::UnknownIntrinsic(_)
                ));
            }
            (got, want) => panic!("{text}: {got:?}, want {want:?}"),
        }
    }
}
