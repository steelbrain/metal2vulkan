//! A visible function called through a declaration of another type than its definition's, end to
//! end against what the probe `linked_function_signatures.m` (the conformance crate's `probes/`)
//! measured on Metal: each case of `metal2vulkan_devtools::linked_signatures` links its function
//! into its kernel of the owned fixture `engine/linked_signature_caller`, the arguments passed by
//! position, and stores Metal's word 0 where Metal defined one and the translator's own where
//! Metal's is undefined; the shapes no adapter passes are refused as not implemented. The
//! functions are `engine/linked_signature_functions`'s, the stitched one
//! `metal2vulkan_devtools::stitched`'s module of `STITCHED_PAIR`. The kernels compute on the CPU
//! interpreter; no GPU is used.
//!
//! `llvm-as` assembles the modules; a missing tool skips visibly.

use metal2vulkan_devtools::linked_signatures::{Answer, CASES, STITCHED_PAIR};
use metal2vulkan_devtools::stitched;

use crate::budget::SPIRV_WORDS;
use crate::device::DevicePolicy;
use crate::error::{DiagName, Error, Feature};
use crate::library::FunctionConstantValues;
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, llvm_assemble_text, meter, profile,
    typed_module_linked, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles;

/// The bitcode `assembled` gives, or `None` (skipped) without `llvm-as`.
fn assembled(what: &str, assembled: Result<Vec<u8>, AssembleError>) -> Option<Vec<u8>> {
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {what}: {why}"),
    }
}

/// The caller fixture, the functions fixture and the stitched module.
fn fixtures() -> Option<[Vec<u8>; 3]> {
    let fixture = |stem: &str| {
        assembled(
            stem,
            llvm_assemble(&engine_dir().join(format!("{stem}.ll"))),
        )
    };
    Some([
        fixture("linked_signature_caller")?,
        fixture("linked_signature_functions")?,
        assembled(
            STITCHED_PAIR.name,
            llvm_assemble_text(&stitched::module(&STITCHED_PAIR)),
        )?,
    ])
}

/// What each case's kernel links: its function, of the functions fixture or the stitched module.
fn linked<'a>(
    [_, functions, stitched]: &'a [Vec<u8>; 3],
    case: &metal2vulkan_devtools::linked_signatures::Case,
    none: &'a FunctionConstantValues,
) -> LinkedSource<'a> {
    let module = if case.stitched { stitched } else { functions };
    LinkedSource::new(case.function, module, none)
}

/// Words 0 and 1 of `out` the case's kernel stores, one thread on `in = {2, 3, 4, 5}`, or why its
/// compile was refused.
fn words(
    modules: &[Vec<u8>; 3],
    case: &metal2vulkan_devtools::linked_signatures::Case,
) -> Result<[u32; 2], Error> {
    let none = FunctionConstantValues::defaults();
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let kernel = format!("k_{}", case.name);
    let (_, m) = typed_module_linked(&modules[0], &kernel, &p, &[linked(modules, case, &none)])?;
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let input: Vec<u8> = [2.0f32, 3.0, 4.0, 5.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: slot(0),
                address: None,
                bytes: vec![0xff; 8],
            },
            Buffer {
                slot: slot(1),
                address: None,
                bytes: input,
            },
        ],
        push: vec![0; 128],
        spec: Default::default(),
    };
    let out = run(&m, &Config::default(), res).unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
    let word = |i: usize| u32::from_le_bytes(out.buffers[0][4 * i..4 * i + 4].try_into().unwrap());
    Ok([word(0), word(1)])
}

/// Each case stores Metal's word where Metal defined one and the translator's where Metal's is
/// undefined, then the sentinel; the shapes no adapter passes are refused as not implemented,
/// naming the function, where Metal makes the pipeline.
#[test]
fn a_reference_of_another_type_computes_by_position() {
    let Some(modules) = fixtures() else {
        return;
    };
    for case in &CASES {
        let got = words(&modules, case);
        match case.answer {
            Answer::Metal(w) | Answer::Undefined(w) => {
                assert_eq!(got, Ok([w, 0x5e47]), "{}", case.name);
            }
            Answer::Refused => match got {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::Linking, "{}: {u}", case.name);
                    assert_eq!(
                        u.symbol,
                        Some(DiagName::new(case.function)),
                        "{}: {u}",
                        case.name
                    );
                }
                other => panic!("{}: not refused: {other:?}", case.name),
            },
        }
    }
}

/// Each case that links passes L3 on every committed profile, or is refused there for what the
/// profile lacks: never for its link.
#[test]
fn a_reference_of_another_type_validates_on_every_profile() {
    let Some(modules) = fixtures() else {
        return;
    };
    let none = FunctionConstantValues::defaults();
    for (stem, p) in committed_profiles().expect("profiles") {
        for case in CASES.iter().filter(|c| c.answer != Answer::Refused) {
            let kernel = format!("k_{}", case.name);
            let made = compile_bitcode_linked(
                &modules[0],
                EntryRef::Name(&kernel),
                (&none, &[linked(&modules, case, &none)]),
                &p,
                &LowerInputs::new(),
                &mut meter(),
                SPIRV_WORDS,
            );
            match made {
                Ok(a) => {
                    link_and_validate(&a, &p)
                        .unwrap_or_else(|e| panic!("{stem}: {}: L3: {e}", case.name));
                }
                Err(Error::Unsupported(u)) if u.feature != Feature::Linking => {
                    eprintln!("{stem}: {}: refused for the profile: {u}", case.name);
                }
                Err(e) => panic!("{stem}: {}: {e}", case.name),
            }
        }
    }
}
