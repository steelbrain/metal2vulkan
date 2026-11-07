//! Canonicalization (`air::specialize`) over the committed public fixtures.
//!
//! Every public `.ll` fixture is assembled with `llvm-as` (opaque pointers) and every
//! `engine/*.air.ll` companion with `xcrun metal-as` (typed pointers). The tests check that
//! every entry point of every fixture either specializes into a verified single-function module
//! or is refused with a typed, non-internal error; that specialization is deterministic; and that
//! the typed and opaque spellings of one program specialize to the same printed module. A missing
//! tool skips its test visibly; a tool that rejects an engine fixture fails it.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::air::error::Error;
use crate::air::ir::print::print;
use crate::air::{decode, specialize, ConstValue, Decoded, Entry, SpecRequest};
use crate::budget::{Budget, Meter};
use crate::metal::DataType;
use crate::pipeline::testing::Scratch;

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

/// Opaque bitcode of `ll`; `None` when this `llvm-as` rejects a legacy top-level fixture.
fn llvm_assemble(tool: &Path, ll: &Path, scratch: &Scratch) -> Option<Vec<u8>> {
    let bytes = assemble(tool, &[ll.as_os_str()], &scratch.path().join("fixture.bc"));
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

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

/// The names of every entry point of `d`, from the ABI decode already cross-checked.
fn entry_names(d: &Decoded) -> Vec<String> {
    d.abi
        .entries
        .iter()
        .map(|e| {
            let f = d.module.func(e.func).unwrap();
            d.module.symbols().as_str(f.name).unwrap().to_string()
        })
        .collect()
}

fn spec(bytes: &[u8], entry: &str) -> Result<Entry, Error> {
    let mut m = meter();
    let d = decode(bytes, &mut m).unwrap();
    let r = SpecRequest {
        call_stack_depth: 1,
        entry: entry.into(),
        ..SpecRequest::default()
    };
    specialize(d, &r, &mut m)
}

/// Every entry of every public fixture specializes, or is refused with a typed input-level
/// error. An internal error (a broken invariant) always fails.
#[test]
fn every_public_entry_specializes_or_is_refused_typed() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP every_public_entry_specializes_or_is_refused_typed: no llvm-as");
        return;
    };
    let scratch = Scratch::new("m2v-air-canon");
    let (mut ok, mut refused, mut abi_refused) = (0, 0, 0);
    let mut fixtures = opaque_fixtures(&fixtures_dir());
    fixtures.extend(opaque_fixtures(&engine_dir()));
    for ll in fixtures {
        let Some(bytes) = llvm_assemble(&tool, &ll, &scratch) else {
            continue;
        };
        // A hand-written legacy fixture whose ABI does not decode is `air::abi`'s concern
        // (src/tests/air_abi.rs pins the exact set). Decode refuses it with a typed error since the
        // ABI joined `Decoded`; an engine fixture must always decode.
        let names = match decode(&bytes, &mut meter()) {
            Ok(d) => entry_names(&d),
            Err(e) => {
                assert!(!matches!(e, Error::Internal(_)), "{ll:?}: {e}");
                assert!(
                    !ll.starts_with(engine_dir()),
                    "{ll:?}: engine fixture refused: {e}"
                );
                abi_refused += 1;
                continue;
            }
        };
        for name in names {
            match spec(&bytes, &name) {
                Ok(e) => {
                    let parts = e.module.parts();
                    let defined = parts.funcs.values().filter(|f| f.body.is_some()).count();
                    assert_eq!(defined, 1, "{ll:?} {name}");
                    assert!(parts.funcs.get(e.func).unwrap().body.is_some());
                    assert_eq!(e.ep.func, e.func);
                    assert!(e.ep.params.iter().all(|p| p.predicate.is_none()));
                    ok += 1;
                }
                Err(Error::Internal(e)) => panic!("{ll:?} {name}: internal error {e}"),
                Err(Error::Unsupported(_) | Error::NeedsInput(_) | Error::InvalidInput(_)) => {
                    refused += 1
                }
                Err(e) => panic!("{ll:?} {name}: unexpected {e}"),
            }
        }
    }
    eprintln!("specialized {ok} entries, refused {refused}; {abi_refused} fixtures without an ABI");
    assert!(ok > 0);
}

#[test]
fn specialization_is_deterministic() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP specialization_is_deterministic: no llvm-as");
        return;
    };
    let scratch = Scratch::new("m2v-air-canon");
    for ll in opaque_fixtures(&engine_dir()) {
        let bytes = llvm_assemble(&tool, &ll, &scratch).unwrap();
        for name in entry_names(&decode(&bytes, &mut meter()).unwrap()) {
            // A refusal (an entry that needs a function constant) must be deterministic too.
            let a = spec(&bytes, &name).map(|e| print(&e.module));
            let b = spec(&bytes, &name).map(|e| print(&e.module));
            assert_eq!(a, b, "{ll:?} {name}");
        }
    }
}

/// `text` (a module printout) with each quoted `llvm.*` name's typed-pointer overload tokens
/// (`p<space><pointee>`, as in `llvm.memcpy.p1i8.p1i8.i64`) spelled opaque (`p<space>`): the
/// pointee is the only part of such a name the typed spelling adds. A pointee whose mangling holds
/// a `.` is not modelled.
fn opaque_intrinsic_names(text: &str) -> String {
    let token = |t: &str| -> String {
        let digits = t.strip_prefix('p').map_or(0, |r| {
            r.len() - r.trim_start_matches(|c: char| c.is_ascii_digit()).len()
        });
        if digits > 0 {
            t[..1 + digits].to_owned()
        } else {
            t.to_owned()
        }
    };
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("\"llvm.") {
        out.push_str(&rest[..=at]);
        let name = &rest[at + 1..];
        let end = name.find('"').unwrap_or(name.len());
        let parts: Vec<String> = name[..end].split('.').map(token).collect();
        out.push_str(&parts.join("."));
        rest = &name[end..];
    }
    out.push_str(rest);
    out
}

/// Typed- and opaque-pointer bitcode of one program specialize to the same module, up to the
/// pointee manglings in overloaded `llvm.*` names.
#[test]
fn typed_and_opaque_fixtures_specialize_identically() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP typed_and_opaque_fixtures_specialize_identically: no llvm-as");
        return;
    };
    if !have_metal_as() {
        eprintln!("SKIP typed_and_opaque_fixtures_specialize_identically: no xcrun metal-as");
        return;
    }
    let scratch = Scratch::new("m2v-air-canon");
    let mut compared = 0;
    for ll in opaque_fixtures(&engine_dir()) {
        let opaque = llvm_assemble(&tool, &ll, &scratch).unwrap();
        let typed = metal_assemble(&ll.with_extension("air.ll"), &scratch);
        for name in entry_names(&decode(&opaque, &mut meter()).unwrap()) {
            match (spec(&opaque, &name), spec(&typed, &name)) {
                (Ok(o), Ok(t)) => {
                    assert_eq!(
                        opaque_intrinsic_names(&print(&o.module)),
                        opaque_intrinsic_names(&print(&t.module)),
                        "{ll:?} {name}"
                    );
                    assert_eq!(o.ep, t.ep, "{ll:?} {name}");
                    compared += 1;
                }
                // An entry that needs a function constant is refused alike in both spellings.
                (o, t) => assert_eq!(o.err(), t.err(), "{ll:?} {name}"),
            }
        }
    }
    assert!(compared > 0);
}

/// A module's order of declarations reaches no specialized module: LLVM's upgrade of a
/// typed-pointer intrinsic's name declares it anew at the module's end, so the opaque spelling of
/// a program can list its declarations in another order than the typed one (`zws_compositor`'s
/// `zws_box`: `llvm.memset` before `llvm.fmuladd` typed, after it opaque). The fixture is
/// specialized as committed and with its `llvm.memset` declaration moved before every function.
#[test]
fn the_order_of_declarations_reaches_no_specialized_module() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP the_order_of_declarations_reaches_no_specialized_module: no llvm-as");
        return;
    };
    let scratch = Scratch::new("m2v-air-canon");
    let ll = engine_dir().join("zws_compositor.ll");
    let text = std::fs::read_to_string(&ll).unwrap();
    let memset = text
        .lines()
        .find(|l| l.starts_with("declare void @llvm.memset."))
        .unwrap();
    let first = text.find("\ndefine ").unwrap() + 1;
    let moved = format!(
        "{}{memset}\n{}",
        &text[..first],
        text[first..].replacen(&format!("{memset}\n"), "", 1)
    );
    assert_ne!(moved, text);
    let reordered = scratch.path().join("reordered.ll");
    std::fs::write(&reordered, &moved).unwrap();
    let a = llvm_assemble(&tool, &ll, &scratch).unwrap();
    let b = llvm_assemble(&tool, &reordered, &scratch).unwrap();
    let a = spec(&a, "zws_box").unwrap();
    let b = spec(&b, "zws_box").unwrap();
    assert_eq!(print(&a.module), print(&b.module));
    assert_eq!(a.ep, b.ep);
}

/// The frontend's function-constant shape copies the constant into an internal global in
/// `air.static_init`. With no value, the copy is undefined and the kernel's branch on it takes its
/// false way; with one, the static initializer bakes the copy and SCCP folds every branch on it.
#[test]
fn a_static_initializer_fixture_specializes_with_its_constant() {
    let Some(tool) = llvm_as() else {
        eprintln!("SKIP a_static_initializer_fixture_specializes_with_its_constant: no llvm-as");
        return;
    };
    let scratch = Scratch::new("m2v-air-canon");
    let ll = fixtures_dir().join("kernel_function_constant_gate_fold.ll");
    let bytes = llvm_assemble(&tool, &ll, &scratch).unwrap();
    let undefined = print(&spec(&bytes, "fcfold").unwrap().module);
    assert!(!undefined.contains("switch"), "{undefined}");
    let mut m = meter();
    let r = SpecRequest {
        call_stack_depth: 1,
        open_cl_mode: false,
        entry: "fcfold".into(),
        constants: [(
            0,
            ConstValue {
                ty: DataType::Int,
                bytes: 2i32.to_le_bytes().into(),
            },
        )]
        .into_iter()
        .collect(),
    };
    let e = specialize(decode(&bytes, &mut m).unwrap(), &r, &mut m).unwrap();
    let text = print(&e.module);
    assert!(!text.contains("switch"), "{text}");
    assert!(!text.contains("load i32"), "{text}");
    assert!(e.module.parts().globals.is_empty(), "{text}");
}

/// A `uint4` function constant in the frontend's shape (the compiled
/// `engine/kernel_vector_function_constant`): an `air.fc_initializer` global that
/// `!air.function_constants` names, copied into a program-scope constant by `air.static_init`.
/// With no value the copy is undefined and the entry specializes; with one, both spellings bake
/// the copy and fold the sum of its lanes into the store.
#[test]
fn a_vector_function_constant_bakes_through_the_static_initializer() {
    let Some(tool) = llvm_as() else {
        eprintln!(
            "SKIP a_vector_function_constant_bakes_through_the_static_initializer: no llvm-as"
        );
        return;
    };
    let scratch = Scratch::new("m2v-air-canon");
    let ll = engine_dir().join("kernel_vector_function_constant.ll");
    let mut spellings = vec![llvm_assemble(&tool, &ll, &scratch).unwrap()];
    if have_metal_as() {
        spellings.push(metal_assemble(&ll.with_extension("air.ll"), &scratch));
    }
    let entry = "kernel_vector_function_constant";
    let lanes: Vec<u8> = [1u32, 2, 3, 4]
        .iter()
        .flat_map(|l| l.to_le_bytes())
        .collect();
    let mut printed = Vec::new();
    for bytes in &spellings {
        spec(bytes, entry).unwrap();
        let mut m = meter();
        let r = SpecRequest {
            call_stack_depth: 1,
            open_cl_mode: false,
            entry: entry.into(),
            constants: [(
                0,
                ConstValue {
                    ty: DataType::UInt4,
                    bytes: lanes.clone().into(),
                },
            )]
            .into_iter()
            .collect(),
        };
        let e = specialize(decode(bytes, &mut m).unwrap(), &r, &mut m).unwrap();
        let text = print(&e.module);
        assert!(e.module.parts().globals.is_empty(), "{text}");
        assert!(text.contains("store i32 10,"), "{text}");
        printed.push(text);
    }
    printed.dedup();
    assert_eq!(
        printed.len(),
        1,
        "typed and opaque spellings differ: {printed:?}"
    );
}
