//! A module whose function was renamed to its qualified name translates as the signature it wraps.
//!
//! Apple's compiler at macOS 14 writes a specialized function's module with
//! `!air.signature = !{!s}`, `!s = !{!"air.rename_qualified_function_signature", !l, !"<name>"}`,
//! where `!l` is the `air.linked_constants_signature` of the constants the function was
//! specialized with: the module declares no function constants and its own code is what runs.
//!
//! Every fourth public `.ll` fixture (in path order) that declares no function constants and
//! carries no signature is respelled here with such a signature, and both spellings compile,
//! entry by entry, to the same SPIR-V words (or the same refusal). Every fixture that declares
//! function constants is refused under the renamed signature exactly as under the
//! linked-constants signature it wraps.
//!
//! `llvm-as` assembles the text; without it the tests skip visibly.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::{DevicePolicy, DeviceProfile};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_standalone, llvm_assemble, meter, profile,
    AssembleError,
};

/// `text` with an `!air.signature` appended: the linked-constants signature of one constant
/// (index 3, `bool` false), wrapped in a rename to `name` when `renamed`. The metadata ids are
/// past any a fixture uses.
fn signed(text: &str, name: &str, renamed: bool) -> String {
    let linked = "!990002 = !{!\"air.linked_constants_signature\", !990003, !990004}\n\
                  !990003 = !{!\"air.module_signature\", \
                  !\"00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF\"}\n\
                  !990004 = !{i8 0, !\"bool\", i32 3}\n";
    let mut out = String::with_capacity(text.len() + 512);
    out.push_str(text);
    if !text.ends_with('\n') {
        out.push('\n');
    }
    if renamed {
        out.push_str("!air.signature = !{!990001}\n");
        out.push_str(&format!(
            "!990001 = !{{!\"air.rename_qualified_function_signature\", !990002, !\"{name}\"}}\n"
        ));
    } else {
        out.push_str("!air.signature = !{!990002}\n");
    }
    out.push_str(linked);
    out
}

/// Bitcode of `text` through `llvm-as`.
fn assemble(text: &str) -> Result<Vec<u8>, AssembleError> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v_qualified_signature_{}_{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).expect("write the signed text");
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
fn outcome(bitcode: &[u8], entry: &str, p: &DeviceProfile) -> Result<Vec<u32>, String> {
    let a = compile(bitcode, entry, p).map_err(|e| e.to_string())?;
    link_standalone(&a, p)
        .map(|(_, w)| w)
        .map_err(|e| format!("link: {e}"))
}

/// Whether module text `text` declares function constants or already carries a signature.
fn declares(text: &str, list: &str) -> bool {
    text.lines().any(|l| l.starts_with(list))
}

#[test]
fn a_renamed_function_compiles_as_its_signature_s_module() {
    let p = profile("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).expect("profile");
    let (mut signed_fixtures, mut compared, mut compiled, mut refused_alike) = (0, 0, 0, 0);
    let mut plain = 0usize;
    for path in fixtures() {
        let name = path.display().to_string();
        let text = std::fs::read_to_string(&path).expect("fixture text");
        if declares(&text, "!air.signature") {
            continue;
        }
        let constants = declares(&text, "!air.function_constants");
        if !constants {
            plain += 1;
            if plain % 4 != 1 {
                continue;
            }
        }
        let old = match llvm_assemble(Path::new(&path)) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED a_renamed_function_compiles_as_its_signature_s_module: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("{name}: llvm-as: {why}"),
        };
        let entries: Vec<String> = crate::air::decode(&old, &mut meter())
            .unwrap_or_else(|e| panic!("{name}: decode: {e}"))
            .abi
            .entries
            .iter()
            .map(|e| e.name.to_string())
            .collect();
        let Some(first) = entries.first() else {
            continue;
        };
        let renamed = assemble(&signed(&text, first, true))
            .unwrap_or_else(|e| panic!("{name} renamed: {e:?}"));
        signed_fixtures += 1;
        if constants {
            // Refused as the linked-constants signature it wraps is.
            let linked = assemble(&signed(&text, first, false))
                .unwrap_or_else(|e| panic!("{name} linked: {e:?}"));
            for entry in &entries {
                let a = outcome(&linked, entry, &p);
                let b = outcome(&renamed, entry, &p);
                assert!(
                    a.is_err(),
                    "{name} {entry}: a linked module declaring constants"
                );
                assert_eq!(a, b, "{name} {entry}");
                refused_alike += 1;
            }
            continue;
        }
        for entry in &entries {
            let a = outcome(&old, entry, &p);
            let b = outcome(&renamed, entry, &p);
            compiled += usize::from(a.is_ok());
            assert_eq!(a, b, "{name} {entry}");
            compared += 1;
        }
    }
    eprintln!(
        "{signed_fixtures} fixtures signed, {compared} entries compared ({compiled} compiled), \
         {refused_alike} refused alike"
    );
    assert!(compiled >= 40, "only {compiled} renamed entries compiled");
    assert!(
        refused_alike > 0,
        "no fixture declaring constants was signed"
    );
}
