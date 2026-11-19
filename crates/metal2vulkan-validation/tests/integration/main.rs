//! The validation crate's integration tests, one test binary for all of them.
//!
//! Each module below was its own test binary. Every binary of this crate's tests links the whole
//! translator, the runtime, and validation, so one binary instead of sixteen saves fifteen links
//! (and their codegen and disk) on every build of the suite. nextest still runs each test in its
//! own process. A test that installs its own global allocator to measure a compile alone
//! (`ledger_call_payloads.rs`, the `ledger_coverage_<k>.rs` shards) keeps a binary of its own:
//! a binary has one global allocator, and those tests need no other test thread allocating while
//! they measure.

/// The path of this crate's binary `$name` (a string literal), read when the test runs: the
/// `NEXTEST_BIN_EXE_<name>` that cargo-nextest sets (documented from 0.9.15 under the hyphenated
/// name, with an underscored copy from 0.9.113), which points into the directory a reused archive
/// was extracted to; else the path Cargo baked in at build time, where `cargo test` finds it.
macro_rules! bin_exe {
    ($name:literal) => {
        std::env::var_os(concat!("NEXTEST_BIN_EXE_", $name))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(env!(concat!("CARGO_BIN_EXE_", $name))))
    };
}

mod audit_worker;
mod authored_compile_census;
mod bins;
mod committed_public_cases_check;
mod committed_shards_keep_the_store_layout;
mod compile_worker;
mod device_worker;
mod drm_shim;
mod engine_linked_cases;
mod fast_trig_within_apples_bound;
mod fixtures_assemble_with_the_pinned_llvm;
mod linux_gate_policy;
mod metal_coverage;
mod no_conflict_markers;
mod pipeline_selection;
mod private_corpus_root;
mod public_entry_rows;
mod published_crates;
mod replay_completion_mark;
mod replay_range;
mod swept_scratch;
mod translation_l3_stage;
