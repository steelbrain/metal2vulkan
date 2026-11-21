//! The meter charges by its input alone.
//!
//! Work units and ledger bytes decide whether an input near a cap translates or is refused with
//! `LimitExceeded`, a refusal permanent for the input. So, like the output, they must be a function
//! of the bitcode, the entry, and the profile: never of hash seeds, addresses, or what the process
//! compiled before. Rust seeds each `HashMap`'s hasher differently, two maps in one process
//! included, so compiling one input twice in one process exposes a charge that follows iteration
//! order or an address, as `deterministic_output_tests` does for output. The second compile also
//! runs after the first has warmed whatever the process keeps.
//!
//! Every entry of every public fixture is compiled twice on two committed profiles (the floor and
//! the most capable), and both compiles must charge the same work units and ledger peak and give
//! the same artifact or the same refusal. `llvm-as` assembles the fixtures (a typed-pointer
//! `.air.ll` goes through `xcrun metal-as`); a fixture whose assembler is missing or rejects it is
//! skipped visibly.

use std::path::{Path, PathBuf};

use crate::budget::SPIRV_WORDS;
use crate::device::DevicePolicy;
use crate::library::FunctionConstantValues;
use crate::lower::select::LowerInputs;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::testing::{
    engine_dir, fixtures_dir, llvm_assemble, metal_assemble, meter, profile,
};

/// `.ll` files directly in `dir`, sorted by name.
fn ll_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("read fixture directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "ll"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn a_second_compile_of_every_public_fixture_is_charged_the_same() {
    let profiles = ["runtime-floor", "moltenvk-m3"].map(|stem| {
        (
            stem,
            profile(stem, DevicePolicy::RESIDENT_TABLES).expect(stem),
        )
    });
    let (mut compared, mut skipped) = (0usize, 0usize);
    for path in ll_files(&fixtures_dir())
        .into_iter()
        .chain(ll_files(&engine_dir()))
    {
        let name = path.display();
        let assembled = if path.to_str().is_some_and(|p| p.ends_with(".air.ll")) {
            metal_assemble(&path)
        } else {
            llvm_assemble(&path)
        };
        let Ok(bitcode) = assembled else {
            eprintln!("SKIPPED {name}: {assembled:?}");
            skipped += 1;
            continue;
        };
        // A fixture whose ABI does not decode has no entries to compile (the inventory test
        // records its refusal).
        let Ok(decoded) = crate::air::decode(&bitcode, &mut meter()) else {
            continue;
        };
        for entry in decoded.abi.entries.iter().map(|e| e.name.to_string()) {
            for (stem, p) in &profiles {
                let charged = || {
                    let mut m = meter();
                    let compiled = compile_bitcode(
                        &bitcode,
                        EntryRef::Name(&entry),
                        &FunctionConstantValues::defaults(),
                        p,
                        &LowerInputs::new(),
                        &mut m,
                        SPIRV_WORDS,
                    )
                    .map_err(|e| format!("{e:?}"));
                    (m.work(), m.ledger_peak(), compiled)
                };
                let (first, second) = (charged(), charged());
                assert_eq!(
                    (first.0, first.1),
                    (second.0, second.1),
                    "{name} {entry} on {stem}: two compiles charged different work units or \
                     ledger bytes; some metered loop follows iteration order, an address, or \
                     process state rather than the input"
                );
                assert!(
                    first.2 == second.2,
                    "{name} {entry} on {stem}: two compiles gave different results"
                );
                compared += 1;
            }
        }
    }
    if skipped > 0 {
        eprintln!("SKIPPED {skipped} fixture(s) without an assembler that accepts them");
    }
    assert!(compared > 0 || skipped > 0, "no fixture entry was compiled");
}
