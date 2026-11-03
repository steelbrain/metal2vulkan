//! Inline assembly: the owned fixture `engine/kernel_inline_asm`.
//!
//! An MSL `asm volatile("")` compiles to a call of `asm sideeffect "", ""`, whose callee is a
//! function-local INLINEASM constant (CONSTANTS_BLOCK record 30). The module decodes whatever its
//! entries call: `asm_free`, which reaches no assembly although its module holds two calls of it,
//! translates; `asm_in_loop` (a call in a loop body) and `asm_in_helper` (in a helper the entry
//! inlines) are refused as `InlineAsm`, naming the expression. Both spellings are checked: the
//! opaque-pointer `.ll` (`llvm-as`) and the typed-pointer `.air.ll` (`xcrun metal-as`, which
//! writes the record as Apple's compiler does). A missing tool skips visibly. No GPU is used.

use crate::air::error::DiagName;
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, Needs};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, metal_assemble, profile, AssembleError,
};
use crate::spirv::testing::committed_profiles;

const STEM: &str = "kernel_inline_asm";

/// The fixture under each available assembler: `(spelling, bitcode)`.
fn spellings() -> Vec<(&'static str, Vec<u8>)> {
    let mut out = Vec::new();
    for (what, assembled) in [
        (
            "llvm-as",
            llvm_assemble(&engine_dir().join(format!("{STEM}.ll"))),
        ),
        (
            "metal-as",
            metal_assemble(&engine_dir().join(format!("{STEM}.air.ll"))),
        ),
    ] {
        match assembled {
            Ok(bc) => out.push((what, bc)),
            Err(AssembleError::Missing(why)) => eprintln!("SKIPPED {what}: {why}"),
            Err(AssembleError::Rejected(why)) => panic!("{what} rejected {STEM}: {why}"),
        }
    }
    out
}

/// An entry that reaches no assembly translates, and links and validates, on every committed
/// profile, in both spellings: the module's other calls of `asm sideeffect "", ""` cost it
/// nothing.
#[test]
fn an_entry_that_reaches_no_inline_asm_translates() {
    let profiles = committed_profiles().expect("profiles");
    for (what, bc) in spellings() {
        for (stem, p) in &profiles {
            let a = compile(&bc, "asm_free", p)
                .unwrap_or_else(|e| panic!("{what} asm_free on {stem}: {e:?}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what} asm_free on {stem}: {e}"));
        }
    }
}

/// A call of inline assembly that the entry reaches, in its own body or in a helper it inlines,
/// is refused as `InlineAsm`, not implemented, naming the expression.
#[test]
fn a_reached_inline_asm_call_is_refused_by_name() {
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    for (what, bc) in spellings() {
        for entry in ["asm_in_loop", "asm_in_helper"] {
            match compile(&bc, entry, &p) {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::InlineAsm, "{what} {entry}: {u:?}");
                    assert_eq!(u.needs, Needs::NotImplemented, "{what} {entry}");
                    assert_eq!(
                        u.symbol,
                        Some(DiagName::new(r#"asm sideeffect "", """#)),
                        "{what} {entry}"
                    );
                }
                other => panic!("{what} {entry}: {other:?}"),
            }
        }
    }
}
