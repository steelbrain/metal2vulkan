//! The byte ledger accounts for what a compile allocates.
//!
//! Every owned engine fixture entry is compiled for every committed profile, and each compile
//! the allocator measured must have a ledger peak of at least 7/10 of its allocator peak: the L7
//! gate `corpus-triage --audit budgets` applies to the corpus, applied here to inputs the tests
//! own, so an uncharged growth site fails without the private corpus.
//!
//! The fixtures run in [`SHARDS`] shards, test binary `ledger_coverage_<k>` running the fixtures
//! whose index is `k` modulo [`SHARDS`], so nextest runs them side by side (as one test this was
//! the validation suite's longest by far: 684 s of a 709-s run in gate 9c95feea). Each binary
//! installs the worker's counting allocator and runs one test, so no other test thread allocates
//! while a compile is measured and the allocator peak is exactly the compile's.

use metal2vulkan::Budget;
use metal2vulkan_air::budget::Meter;
use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
use metal2vulkan_validation::budgets::ledger_covers;
use metal2vulkan_validation::engine::{committed_profiles, load_profile, translate_new};

/// The number of shards: binaries `tests/ledger_coverage_0.rs` to `ledger_coverage_5.rs`.
pub const SHARDS: usize = 6;

/// Shard `shard` of the check: the sorted owned fixtures whose index is `shard` modulo
/// [`SHARDS`]. Every shard's binary must exist, so no fixture goes unchecked.
pub fn ledger_covers_the_allocator_peak_of_every_owned_fixture_compile(shard: usize) {
    let tests = metal2vulkan_devtools::validation_dir().join("tests");
    for k in 0..=SHARDS {
        let binary = tests.join(format!("ledger_coverage_{k}.rs"));
        assert_eq!(
            binary.is_file(),
            k < SHARDS,
            "{}: shards are binaries 0 to {}",
            binary.display(),
            SHARDS - 1
        );
    }
    let mut fixtures = std::fs::read_dir(engine_dir())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "ll")
                && !path.to_string_lossy().ends_with(".air.ll")
        })
        .collect::<Vec<_>>();
    fixtures.sort();
    let profiles = committed_profiles()
        .unwrap()
        .into_iter()
        .map(|stem| {
            let profile = load_profile(&stem).unwrap();
            (stem, profile)
        })
        .collect::<Vec<_>>();
    let (mut measured, mut shortfalls) = (0usize, Vec::new());
    for path in fixtures.iter().skip(shard).step_by(SHARDS) {
        let bitcode = match llvm_assemble(path) {
            Ok(bytes) => bytes,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            // Owned fixtures llvm-as rejects are the inventory test's to report.
            Err(AssembleError::Rejected(why)) => {
                eprintln!("SKIPPED {}: {why}", path.display());
                continue;
            }
        };
        let entries = match metal2vulkan_air::decode(&bitcode, &mut Meter::new(&Budget::default()))
        {
            Ok(decoded) => decoded
                .abi
                .entries
                .iter()
                .map(|entry| entry.name.to_string())
                .collect::<Vec<_>>(),
            Err(_) => continue,
        };
        for entry in &entries {
            for (stem, profile) in &profiles {
                let attempt = translate_new(&bitcode, entry, profile, None);
                let Some(cost) = attempt.cost else { continue };
                let Some(allocator) = cost.allocator_peak.filter(|peak| *peak > 0) else {
                    continue;
                };
                measured += 1;
                if !ledger_covers(cost.ledger_peak, allocator) {
                    shortfalls.push(format!(
                        "{} {entry} on {stem}: ledger {} < 7/10 of allocator {allocator} ({})",
                        path.file_name().unwrap().to_string_lossy(),
                        cost.ledger_peak,
                        attempt.outcome.label(),
                    ));
                }
            }
        }
    }
    assert!(measured > 0, "shard {shard}: no compile was measured");
    assert!(
        shortfalls.is_empty(),
        "shard {shard}: {} of {measured} compiles:\n{}",
        shortfalls.len(),
        shortfalls.join("\n")
    );
}
