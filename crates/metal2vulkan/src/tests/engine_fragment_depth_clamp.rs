//! A fragment function's depth output stored as Metal stores it (`StageIoPlan::depth_clamp`), over
//! the owned fixture `engine/rfb_viewports` (`rfb_depth` writes a depth; `rfb_full` and
//! `rfb_position` write none):
//!
//! - On every profile, `rfb_depth` kills the invocation of a positive NaN, which fails Metal's
//!   depth test: one `if` whose arm kills. Vulkan clamps the depth to `[0, 1]` itself, as no
//!   runtime enables `VK_EXT_depth_range_unrestricted`.
//! - Every module passes L3 (`spirv-val` and requirements within the profile).
//!
//! - Under `LoweringInputs::depth_clamp_range` `rfb_depth` states `RuntimeFact::DepthClampRange`,
//!   reads `PushField::DepthClampRange` (8 bytes of the push window) and clamps the depth to it
//!   after selecting +0 or 1 by the depth's bits: one ordered less-than and one ordered
//!   greater-than more than without the input, and its kill. The functions that write no depth
//!   are the same module under it.
//!
//! What the depth stores is Metal probes 3 and 3b's. `llvm-as` assembles the fixture; a missing
//! tool skips visibly. No GPU is used.

use std::collections::BTreeMap;

use crate::device::{DevicePolicy, DeviceProfile};
use crate::library::FunctionArtifact;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::testing::{compile_with, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;

/// The bitcode of `engine/rfb_viewports.ll`, or `None` (printed) where `llvm-as` is missing.
fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("rfb_viewports.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rfb_viewports.ll: {why}"),
    }
}

/// Link `a` alone, relocate it, and run L3; its relocated words.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) -> Vec<u32> {
    let linked = link(&[a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("link: {e}"));
    let words = relocate(a, &linked.table).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
    words
}

/// How many instructions of each opcode `words` (a SPIR-V module) holds.
fn opcodes(words: &[u32]) -> BTreeMap<u32, usize> {
    let mut counts = BTreeMap::new();
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        assert!(count > 0, "a zero-length instruction at word {i}");
        *counts.entry(words[i] & 0xffff).or_insert(0) += 1;
        i += count;
    }
    counts
}

/// `OpKill` and `OpTerminateInvocation`, one of which the kill is.
const KILLS: [u32; 2] = [252, 4416];

/// How many instructions of `op` `c` holds.
fn count(c: &BTreeMap<u32, usize>, op: u32) -> usize {
    c.get(&op).copied().unwrap_or(0)
}

/// How many kills `c` holds.
fn kills(c: &BTreeMap<u32, usize>) -> usize {
    KILLS.iter().map(|op| count(c, *op)).sum()
}

/// A positive NaN depth kills the invocation on every profile (ANV stores a NaN as 0 and writes
/// its color, where Metal's depth test fails it):
/// one kill in `rfb_depth`, none in the functions that write no depth.
#[test]
fn a_positive_nan_depth_kills_on_every_profile() {
    let Some(bc) = fixture() else {
        return;
    };
    let inputs = LoweringInputs::new();
    for stem in [
        "amd-780m-radv",
        "lavapipe",
        "nvidia-rtx50-linux",
        "moltenvk-m3",
        "intel-irisxe-rplp-anv",
    ] {
        let p = crate::pipeline::testing::profile(stem, DevicePolicy::RESIDENT_TABLES).expect(stem);
        for (entry, expected) in [("rfb_depth", 1), ("rfb_full", 0), ("rfb_position", 0)] {
            let a = compile_with(&bc, entry, &p, &inputs)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert_eq!(
                kills(&opcodes(&validate(&a, &p))),
                expected,
                "{stem} {entry}"
            );
        }
    }
}

/// `OpFOrdLessThan` and `OpFOrdGreaterThan`, the range's two clamps.
const RANGE_COMPARES: [u32; 2] = [184, 186];

/// Under `depth_clamp_range`, `rfb_depth` states the fact, reads the push field and clamps its
/// depth to the range on every profile; the functions that write no depth are unchanged.
#[test]
fn a_depth_clamp_range_variant_clamps_the_depth_to_the_draws_range() {
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::interface::PushLoc;
    use crate::pipeline::sym::PushField;
    let Some(bc) = fixture() else {
        return;
    };
    let plain = LoweringInputs::new();
    let mut ranged = LoweringInputs::new();
    ranged.depth_clamp_range = true;
    for stem in [
        "amd-780m-radv",
        "lavapipe",
        "nvidia-rtx50-linux",
        "moltenvk-m3",
        "intel-irisxe-rplp-anv",
    ] {
        let p = crate::pipeline::testing::profile(stem, DevicePolicy::RESIDENT_TABLES).expect(stem);
        for entry in ["rfb_full", "rfb_position"] {
            let with = compile_with(&bc, entry, &p, &ranged)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            let without = compile_with(&bc, entry, &p, &plain)
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert!(
                !with.facts.contains(&RuntimeFact::DepthClampRange),
                "{stem} {entry}"
            );
            assert_eq!(
                validate(&with, &p),
                validate(&without, &p),
                "{stem} {entry}"
            );
        }
        let with =
            compile_with(&bc, "rfb_depth", &p, &ranged).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let without =
            compile_with(&bc, "rfb_depth", &p, &plain).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            with.facts.contains(&RuntimeFact::DepthClampRange),
            "{stem}: {:?}",
            with.facts
        );
        let field = with
            .interface
            .push
            .fields
            .iter()
            .find(|f| f.field == PushField::DepthClampRange);
        assert!(
            field.is_some_and(|f| matches!(f.loc, PushLoc::Push { .. }) && f.size == 8),
            "{stem}: {field:?}"
        );
        let (a, b) = (
            opcodes(&validate(&with, &p)),
            opcodes(&validate(&without, &p)),
        );
        for op in RANGE_COMPARES {
            assert_eq!(count(&a, op), count(&b, op) + 1, "{stem}: opcode {op}");
        }
        // The `[0, 1]` selects run under the input.
        assert!(count(&a, 169) >= 5, "{stem}: {} selects", count(&a, 169));
        assert_eq!((kills(&a), kills(&b)), (1, 1), "{stem}: the kill");
    }
}
