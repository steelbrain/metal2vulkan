//! The API parity harness: Metal programs written once, against [`api::Api`], run on this Mac's
//! Metal to record its answers and on the runtime on every C3 venue to compare with them. It covers
//! what a Metal program sees of the API itself: copies, fills, hazards and their order, statuses
//! and refusals. Shader semantics are the authored corpus's; a scenario runs a shader only from a
//! committed public fixture.
//!
//! - **Scenarios** ([`SCENARIOS`]): each a set of cases, each case a short program that ends at
//!   its first refused call, on Metal (whose API validation stops the process there) as on the
//!   runtime. They cover valid usage, and misuse the runtime refuses as Metal's validation stops
//!   it. Misuse the runtime answers as plain Metal does has no case here: Metal's records are made
//!   under API validation, and that answer's truth is the lane tests pinned to the plain probes on
//!   the cloud minis, one case per process. No second, plain run of a case is recorded here. A
//!   scenario recorded again holds no misuse at all, since the M3 Max records run on a Mac where no
//!   misuse runs: `buffer_texture_copies` holds valid usage only, and `buffer_copies`' refusals,
//!   recorded before, leave it before it is recorded again there.
//! - **Records** ([`record`]): each call taken or refused, bytes read back, and command buffer
//!   completions, by name, in order; one line per Metal environment in
//!   `corpus/observations/parity/<scenario>.jsonl`. Metal records them under API validation, each
//!   case in a child process of its own (`lane::record_metal`), with the device's hang tripwire.
//! - **Fingerprint** ([`fingerprint`]): the scenario's source and the harness sources Metal's
//!   records depend on, the scenario's version, its fixtures' AIR and the Metal environment. A
//!   compare against a line whose fingerprint differs fails as stale, and so does
//!   `tests::committed_records_are_current` in the ordinary suite.
//! - **Compare** (`lane`): each scenario on the runtime, on the device the cfg selects, under the
//!   validation layer, against the records of `runtime_lane::METAL_ENVIRONMENT`: exact, record by
//!   record. A runtime gap never matches.
//! - **Sequences** (`sequences`): seeded programs a generator makes from the commands the consumer
//!   reaches, every one race-free and ending in a readback of every resource, with records and a
//!   fingerprint of their own beside the scenarios', so the harness's sources do not depend on
//!   them; a CPU model of each checks Metal's records and shrinks a runtime difference.
//! - **Time** is reported, never compared: wall time per case beside its GPU time, where the API
//!   tells it (`GPUStartTime` to `GPUEndTime`, summed over the case's command buffers). Scenario
//!   `command_buffer_times` records facts of a command buffer's times (each 0 before its commit,
//!   each in order with the host's time around it), never the times.

use crate::hash::sha256_bytes;
use crate::store::CorpusStore;
use record::{Fingerprint, FixtureDigest};
use std::path::{Path, PathBuf};

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod api;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod lane;
#[cfg(all(m2v_gpu, target_os = "macos"))]
mod metal;
mod record;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod runtime;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod scenarios;
pub(crate) mod sequences;
mod tests;

/// A scenario: its name, version, source, the committed cases whose AIR it loads, and its cases.
pub(crate) struct Scenario {
    pub(crate) name: &'static str,
    /// Raised when the scenario's meaning changes without its source changing.
    pub(crate) version: u32,
    pub(crate) source: &'static str,
    pub(crate) fixtures: &'static [&'static str],
    pub(crate) cases: &'static [&'static str],
}

/// The committed case `hazards` dispatches: `output[0] = input[0]`, `input` at buffer 0 and
/// `output` at buffer 1.
const COPY_WORD: &str = "single-thread-copy-word";

pub(crate) const SCENARIOS: [Scenario; 4] = [
    Scenario {
        name: "buffer_copies",
        version: 1,
        source: include_str!("parity/scenarios/buffer_copies.rs"),
        fixtures: &[],
        cases: &[
            "copies_between_buffers",
            "copies_within_one_buffer",
            "fills_of_odd_ranges",
            "a_copy_past_the_source",
            "a_copy_past_the_destination",
            "a_fill_past_the_buffer",
        ],
    },
    Scenario {
        name: "buffer_texture_copies",
        version: 1,
        source: include_str!("parity/scenarios/buffer_texture_copies.rs"),
        fixtures: &[],
        cases: &[
            "pitches_r8unorm",
            "pitches_rg8unorm",
            "pitches_rgba8unorm",
            "pitches_rgba8unorm_srgb",
            "pitches_rgb10a2unorm",
            "pitches_rg11b10float",
            "pitches_rgba16float",
            "pitches_rgba32float",
            "cbt.offset4",
            "cbt.bpr0.1row",
            "cbt.bpi16.2d",
            "cbt.3d",
            "cbt.bc1.6x6",
            "cbt.source16",
        ],
    },
    Scenario {
        name: "hazards",
        version: 1,
        source: include_str!("parity/scenarios/hazards.rs"),
        fixtures: &[COPY_WORD],
        cases: &[
            "one_command_buffer",
            "a_command_buffer_each",
            "two_queues_and_an_event",
            "untracked_with_fences",
        ],
    },
    Scenario {
        name: "command_buffer_times",
        version: 1,
        source: include_str!("parity/scenarios/command_buffer_times.rs"),
        fixtures: &[],
        cases: &["no_work", "one_fill", "fills_around_an_event"],
    },
];

/// The harness sources every scenario's Metal records depend on: the API, Metal's side of it,
/// the steps the scenarios share, and the records. A scenario's own source is its
/// [`Scenario::source`]; no harness source names a scenario
/// (`tests::no_harness_source_names_a_scenario`), so adding or changing one stales its records
/// alone.
const HARNESS: [&str; 4] = [
    include_str!("parity/api.rs"),
    include_str!("parity/metal.rs"),
    include_str!("parity/scenarios/steps.rs"),
    include_str!("parity/record.rs"),
];

/// The SHA-256 of `scenario`'s source and [`HARNESS`], each with its length before it.
fn sources_sha256(scenario: &Scenario) -> String {
    let mut bytes = Vec::new();
    for source in std::iter::once(scenario.source).chain(HARNESS) {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source.as_bytes());
    }
    sha256_bytes(&bytes)
}

/// What `scenario`'s records in `environment` depend on now.
pub(crate) fn fingerprint(scenario: &Scenario, environment: &str) -> Result<Fingerprint, String> {
    Ok(Fingerprint {
        metal_environment: environment.to_owned(),
        scenario: scenario.name.to_owned(),
        scenario_version: scenario.version,
        sources_sha256: sources_sha256(scenario),
        fixtures: fixture_digests(scenario.fixtures)?,
    })
}

/// Each committed case of `names`: its AIR's SHA-256, and that of the file Metal compiles its
/// public fixture from. One pass over the case shards.
fn fixture_digests(names: &[&str]) -> Result<Vec<FixtureDigest>, String> {
    if names.is_empty() {
        return Ok(Vec::new());
    }
    let store = CorpusStore::new(corpus());
    let mut found: Vec<Option<(String, String)>> = vec![None; names.len()];
    for shard in 0..64 {
        for case in store.read_case_shard(shard)? {
            if let Some(index) = names.iter().position(|name| *name == case.name) {
                if found[index]
                    .replace((case.air_sha256, case.entry))
                    .is_some()
                {
                    return Err(format!("several committed cases are named {}", case.name));
                }
            }
        }
    }
    let rows = crate::source::public_sources()?;
    names
        .iter()
        .zip(found)
        .map(|(name, case)| {
            let (air_sha256, entry) =
                case.ok_or_else(|| format!("no committed case is named {name}"))?;
            let row = rows
                .iter()
                .find(|row| row.air_sha256 == air_sha256 && row.entry == entry)
                .ok_or_else(|| format!("{name}: its AIR is no public fixture"))?;
            let (input, _) = crate::source::public_metal_input(&row.label)
                .ok_or_else(|| format!("{name}: {} is no public module", row.label))?;
            let bytes = std::fs::read(&input)
                .map_err(|error| format!("{name}: read {}: {error}", input.display()))?;
            Ok(FixtureDigest {
                case: (*name).to_owned(),
                air_sha256,
                metal_input_sha256: sha256_bytes(&bytes),
            })
        })
        .collect()
}

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus")
}

/// The committed records file of the scenario named `scenario`.
pub(crate) fn records_path(scenario: &str) -> PathBuf {
    corpus()
        .join("observations")
        .join("parity")
        .join(format!("{scenario}.jsonl"))
}
