//! A buffer argument held to less than 4 bytes (`air.arg_type_align_size` 1 or 2) whose every
//! access is one narrow unit no wider than that alignment is reached under the identity domains
//! from its address as it is (`lower::access::AccessPlan::reach`, `psb::prepare_direct`), as one
//! held to 4 is: no access rounds its address to the word below it, which ands a 64-bit address
//! with 3. `buffer_end_byte_load` and `buffer_end_byte_store` reach a `uchar` buffer by `i8`
//! accesses, and `buffer_end_half_load` a `ushort` one by `i16` loads. On a profile without that
//! unit's storage access the access is a word's, and the slot keeps the word form. Each module
//! passes L3, and the narrow access claims its unit's alignment.
//!
//! `llvm-as` assembles the fixtures; without it the test skips visibly.

use std::collections::BTreeSet;

use super::engine_word_phases::{device_alignments, identity_profiles};
use crate::memory::leaf::Units;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::spirv_binary::{parse_bytes, Operand};
use crate::spv::Op;

/// The `OpBitwiseAnd`s of 64-bit integers in serialized module `words`.
fn wide_ands(words: &[u32]) -> usize {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let module = parse_bytes(&bytes).expect("parse");
    let wide: BTreeSet<u32> = (module.instructions.iter())
        .filter(|i| i.opcode == Op::TypeInt && i.operands.first() == Some(&Operand::Word(64)))
        .filter_map(|i| i.result_id)
        .collect();
    (module.instructions.iter())
        .filter(|i| i.opcode == Op::BitwiseAnd && i.result_type.is_some_and(|t| wide.contains(&t)))
        .count()
}

#[test]
fn a_slot_held_to_its_narrow_unit_is_reached_from_its_address_as_it_is() {
    for (fixture, unit) in [
        ("buffer_end_byte_load", 1),
        ("buffer_end_byte_store", 1),
        ("buffer_end_half_load", 2),
    ] {
        let bc = match llvm_assemble(&engine_dir().join(format!("{fixture}.ll"))) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(e) => panic!("{fixture}: {e:?}"),
        };
        for (stem, p) in &identity_profiles() {
            let what = format!("{fixture} on {stem}");
            let a = compile(&bc, fixture, p).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let ands = wide_ands(&a.module.words);
            match Units::storage_on(p).has(unit) {
                true => {
                    assert_eq!(ands, 0, "{what}: an address rounded to the word below it");
                    let claimed = device_alignments(&a.module.words);
                    assert!(
                        claimed.contains(&unit),
                        "{what}: claimed {claimed:?}, no {unit}-byte unit"
                    );
                }
                false => assert!(ands > 0, "{what}: no word form without the unit"),
            }
        }
    }
}
