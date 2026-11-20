//! The bound texture values a function reads as data, end to end over owned fixtures.
//!
//! The class of the format bound at a texture slot reaches the function from its own push field,
//! `PushField::Specialized`, and is no specialization constant: a new bound format makes no new
//! pipeline. Every other value the binder computes from what a draw or dispatch binds (a slot's
//! presence, a sampler's word, ...) stays a specialization constant. `link_and_validate` refuses an
//! artifact that leaves a texel format key a specialization constant, so every test that validates
//! such a module checks it too.

use crate::pipeline::sym::{PushField, SpecKey};
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::spirv::testing::committed_profiles;

/// Owned fixtures and entries that read texel formats and other bound values.
const ENTRIES: [(&str, &str); 6] = [
    ("texture_ops", "queries"),
    ("texture_ops", "texel_buffers"),
    ("sps2_residency_probes", "sps2_residency_probes"),
    ("sampling", "sample_grad"),
    ("sampling", "sample_level"),
    ("sampling_completion", "lod_query_kernel"),
];

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// On every committed profile where an entry compiles, each texel format key it reads is a
/// placed `PushField::Specialized` field and no specialization constant, every other bound value
/// it reads stays one, and the module links and validates; across the profiles the entries
/// specialize a sampler's word and a slot's presence.
#[test]
fn texel_formats_are_push_fields_and_other_bound_values_specializations() {
    let mut bitcode = Vec::new();
    for (file, entry) in ENTRIES {
        let Some(bc) = fixture(file) else {
            return;
        };
        bitcode.push((file, entry, bc));
    }
    let (mut sampler, mut presence) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        for (file, entry, bc) in &bitcode {
            let what = format!("{stem}: {file}/{entry}");
            let Ok(a) = compile(bc, entry, &p) else {
                continue;
            };
            for c in &a.interface.spec {
                assert!(
                    !matches!(c.key, SpecKey::TexelFormat(_)),
                    "{what}: {:?} is a specialization constant",
                    c.key
                );
                match c.key {
                    SpecKey::SamplerState { .. } => sampler += 1,
                    SpecKey::Presence(_) => presence += 1,
                    _ => {}
                }
            }
            for f in &a.interface.push.fields {
                if let PushField::Specialized(key) = f.field {
                    assert!(
                        matches!(key, SpecKey::TexelFormat(_)),
                        "{what}: a pushed {key:?}"
                    );
                }
            }
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
    }
    assert!(sampler > 0, "no entry specializes a sampler's word");
    assert!(presence > 0, "no entry specializes a slot's presence");
}
