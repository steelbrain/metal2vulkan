//! Seeded differential sequences: Metal programs a seed generates, run on Metal to record its
//! answers and on the runtime on every venue to compare with them, the way the harness's scenarios
//! are, to find a barrier the runtime's hazard tracking leaves out.
//!
//! - **Sequences** ([`generate`]): buffers (`Shared` and `Private`, tracked and untracked, each
//!   with a role that bounds what its words hold; of their own, over a host page without a copy,
//!   or placed in a placement heap where neighbouring places overlap) and textures (render
//!   targets kernels read and write, of one slice or two, alone or in a heap of their own, seen
//!   through views in another format, of one slice, or swizzled), on one to three queues;
//!   command buffers of blit encoders (fills, copies, copies between buffers and textures),
//!   compute encoders (one-thread dispatches of a word copy or an in-place mix, direct and
//!   indirect, and grids that read texels into words or write words as texels; serial and
//!   concurrent, with memory barriers of buffers or of textures too), and render passes (a
//!   cleared or loaded target slice, instanced draws of whole-pixel rectangles from a buffer in
//!   colours from another, direct and indirect). Every access is ordered after each earlier conflicting
//!   one, by hazard tracking where Metal's order holds and else by a barrier, fence or event the
//!   generator adds ([`order`]): a sequence never races, so every byte it ends with is defined.
//! - **Runs** ([`run`]): a sequence through [`api::SeqApi`], each call recorded as the harness
//!   records it, then every resource read back. On Metal, each seed in a child process under API
//!   validation (`lane`'s recorder); on the runtime, each seed on a device of its own under the
//!   validation layer and the lane's checks, whose synchronization hazards between commands that
//!   share no byte the lane explains ([`hazards`]).
//! - **Expectations**: Metal's records, one line per Metal environment in
//!   `corpus/observations/parity/sequences.jsonl`, compared exactly; and the CPU [`model`] of
//!   each sequence, which Metal's records check (`tests`) and which gives a runtime difference
//!   its expected bytes. A seed whose runtime bytes differ from the model's is shrunk to a
//!   smaller sequence that still differs (`minimize`), reported with its seed.
//! - **Parallel encodes** (`parallel`, device tests): the sequences, every command buffer on one
//!   queue, encoded on 2, 4 and 8 threads under seeded schedules and committed in order, against
//!   the model and a serial run's notices: a runner of their own, so Metal's records depend on
//!   none of it.
//! - **Fingerprint** ([`fingerprint`]): the sources Metal's records depend on (these, not the
//!   harness's [`HARNESS`](super::HARNESS) beyond the records and the observation, so no other
//!   scenario's records go stale with them), the version, and the fixture `seq_programs`.

use super::record::{Fingerprint, FixtureDigest, Refusal, Value};
use crate::hash::sha256_bytes;

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod api;
mod generate;
pub(crate) mod hazards;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod lane;
#[cfg(all(m2v_gpu, target_os = "macos"))]
mod metal;
mod minimize;
mod model;
mod order;
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod parallel;
mod program;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod run;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod runtime;
mod tests;

/// The records' scenario name.
pub(crate) const NAME: &str = "sequences";
/// Raised when the sequences' meaning changes without their sources changing.
pub(crate) const VERSION: u32 = 1;
/// The seeds run, `0..SEEDS`.
pub(crate) const SEEDS: u64 = 64;
/// The fixture whose functions the sequences run, in `validation/fixtures/public/engine/`.
pub(crate) const FIXTURE: &str = "seq_programs";

/// The name, after a seed's `case_name/`, of the one record a Metal line holds for a seed Metal
/// could not run: its recording child died of a signal (a driver crash) before it finished, so
/// the line records the seed as a Metal-side gap, a [`Refusal::Gap`] whose detail is the crash's
/// signature (the call, the faulting frame, the command), instead of answers no one has.
pub(crate) const METAL_GAP: &str = "metal-gap";

/// The detail of `seed`'s Metal-side gap ([`METAL_GAP`]) in a Metal line's `records`: `Some` only
/// when the seed's one record is the gap, a gap refusal with a detail. A seed with answers, or
/// with a gap beside other records, has none.
pub(crate) fn metal_gap(records: &[(String, Value)], seed: u64) -> Option<&str> {
    let prefix = format!("{}/", program::case_name(seed));
    let mut of_seed = records.iter().filter(|(name, _)| name.starts_with(&prefix));
    match (of_seed.next(), of_seed.next()) {
        (
            Some((
                name,
                Value::Refused {
                    refusal: Refusal::Gap,
                    detail,
                },
            )),
            None,
        ) if name[prefix.len()..] == *METAL_GAP && !detail.is_empty() => Some(detail),
        _ => None,
    }
}

/// The sources Metal's records of the sequences depend on: the sequence, its generator and
/// order, the runner, its API and Metal's side of it, and the harness's observation and records.
const SOURCES: [&str; 8] = [
    include_str!("sequences/program.rs"),
    include_str!("sequences/generate.rs"),
    include_str!("sequences/order.rs"),
    include_str!("sequences/run.rs"),
    include_str!("sequences/api.rs"),
    include_str!("sequences/metal.rs"),
    include_str!("api.rs"),
    include_str!("record.rs"),
];

/// What the sequences' records in `environment` depend on now. The fixture's digest is of its
/// `.ll` text (the AIR the runtime assembles) and its typed `.air.ll` text (what Metal's side
/// assembles).
pub(crate) fn fingerprint(environment: &str) -> Result<Fingerprint, String> {
    let mut bytes = Vec::new();
    for source in SOURCES {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source.as_bytes());
    }
    let read = |extension: &str| {
        let path = metal2vulkan_devtools::engine_dir().join(format!("{FIXTURE}.{extension}"));
        std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))
    };
    Ok(Fingerprint {
        metal_environment: environment.to_owned(),
        scenario: NAME.to_owned(),
        scenario_version: VERSION,
        sources_sha256: sha256_bytes(&bytes),
        fixtures: vec![FixtureDigest {
            case: FIXTURE.to_owned(),
            air_sha256: sha256_bytes(&read("ll")?),
            metal_input_sha256: sha256_bytes(&read("air.ll")?),
        }],
    })
}
