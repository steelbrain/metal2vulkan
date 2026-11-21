//! Two kernels whose threadgroup words only a simdgroup's lockstep orders.
//!
//! The owned fixture `rtl_lockstep` holds two kernels over one 64-thread threadgroup (two
//! simdgroups) whose loops each thread leaves after its own count, with no barrier inside: thread
//! `t` reads another thread's word of its own simdgroup and writes its own from it. Nothing but the
//! simdgroup running its lanes in lockstep decides what those words come to. The Metal Shading
//! Language calls such accesses a data race, and the runtime gives the threads of a SIMD-group no
//! order between barriers (docs/API.md, accepted differences, "The threads of a SIMD-group run in
//! no order between barriers"). So each pipeline carries the static note
//! `PrecisionNote::SimdgroupAccessOrder`, which the runtime's side announces as
//! `simdgroup-access-order`.
//!
//! The cases therefore record no raced word. Each records the dispatch's status and the notices,
//! and the divergent loop, which writes nothing past its 64 words, records that the second half of
//! its output buffer is still the `0xee` the buffer was made with: no order of the threads changes
//! which words a thread writes, so that half compares exactly.
//!
//! Nothing here can fault a GPU: one compute pass a row, one 64-thread threadgroup, two shared
//! buffers, no texture, and each loop bounded by a buffer word's count or by 8.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Negative, Observed, Sites, Value};

/// The fixture both kernels live in.
const FIXTURE: &str = "rtl_lockstep";

/// The threads the one threadgroup holds: two 32-lane simdgroups.
const THREADS: u32 = 64;

/// The words the output buffer holds: the kernels' `out[t]` and `out[64 + t]`.
const WORDS: u32 = 2 * THREADS;

/// The byte the output buffer is made with, so what a kernel leaves alone is a recorded value
/// rather than whatever the allocator left.
const FILL: u8 = 0xee;

/// One row: a kernel of the fixture and the words its input buffer holds.
struct Row {
    /// The case's name.
    case: &'static str,
    /// The kernel, by its name in the library.
    entry: &'static str,
    /// The input words.
    input: fn() -> Vec<u32>,
    /// Whether the kernel writes the second half of the output buffer. Where it does not, the case
    /// records that the dispatch changed nothing there.
    writes_counts: bool,
}

/// `rtl_lockstep_divergent_loop`'s iteration counts, 1 to 5.
fn counts() -> Vec<u32> {
    (0..THREADS).map(|t| 1 + (t * 7 + 3) % 5).collect()
}

/// `rtl_lockstep_until`'s limits.
fn limits() -> Vec<u32> {
    (0..THREADS).map(|t| 16 + (t * 37) % 97).collect()
}

/// The two rows.
const ROWS: &[Row] = &[
    Row {
        case: "simdgroup-lockstep-divergent-loop",
        entry: "rtl_lockstep_divergent_loop",
        input: counts,
        writes_counts: false,
    },
    Row {
        case: "simdgroup-lockstep-until",
        entry: "rtl_lockstep_until",
        input: limits,
        writes_counts: true,
    },
];

/// The input words as the bytes a buffer is made with.
fn input_bytes(row: &Row) -> Vec<u8> {
    (row.input)()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// A case of the family: one kernel.
pub struct Lockstep {
    row: &'static Row,
}

impl Lockstep {
    /// The fixture under this row's entry point.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: self.row.entry,
        }
    }
}

impl Case for Lockstep {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The access-order notice both pipelines announce (the module documentation). And one
        // declaration, loosening the notices site alone: both kernels hold simdgroup-scoped
        // accesses inside divergent loops, so on a device whose answer lacks maximal reconvergence
        // (`shaderMaximalReconvergence`, the SIMD groups' query form) the stage carries its static
        // note, `DeviceReconvergence`, which the sides announce as `device-reconvergence`. The
        // device's answer decides it: a device with maximal reconvergence announces nothing, and
        // this declaration then holds nowhere.
        vec![
            crate::replay::notice_allowance(
                crate::degradation::Degradation::SimdgroupAccessOrder.code(),
            ),
            crate::replay::notice_allowance(
                crate::degradation::Degradation::DeviceReconvergence.code(),
            ),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: self.row.entry.to_owned(),
                source_sha256: crate::sha256_hex(self.fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(self.fixture().text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(self.fixture().text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:0", &input_bytes(self.row)),
            Input::bytes("buffer:1", &vec![FILL; (WORDS * 4) as usize]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, self.row.entry),
        )?;
        let input = made(sites, "buffer/0", side.buffer(&input_bytes(self.row)))?;
        let made_with = vec![FILL; (WORDS * 4) as usize];
        let output = made(sites, "buffer/1", side.buffer(&made_with))?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch(
                pipeline,
                &Bindings::buffers(&[input, output]),
                [1, 1, 1],
                [THREADS, 1, 1],
            ),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(output, 0, made_with.len()),
            |_| Value::Absent,
        )?;
        // The kernel that writes no count leaves the second half of the buffer as it was made,
        // which is a fact about what the dispatch did **not** write and is stated as one.
        let counts_half = if self.row.writes_counts {
            Observed::absent()
        } else if bytes[(THREADS * 4) as usize..] == made_with[(THREADS * 4) as usize..] {
            Observed::value(Value::Nothing(Negative::ChangedNothing))
        } else {
            Observed::value(Value::Bool(true))
        };
        sites.leaf("counts-half", counts_half);
        Ok(())
    }
}

/// The case of `rtl_lockstep_divergent_loop`.
pub fn divergent_loop() -> Lockstep {
    Lockstep { row: &ROWS[0] }
}

/// The case of `rtl_lockstep_until`.
pub fn until() -> Lockstep {
    Lockstep { row: &ROWS[1] }
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(divergent_loop()), Box::new(until())]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each row's inputs are 64 words, one a thread, and every count is inside the bound the
    /// fixture's loops carry: a word outside them would hand the device a loop the fixture's
    /// comment does not bound.
    #[test]
    fn every_input_word_is_inside_the_fixtures_bounds() {
        for row in ROWS {
            let input = (row.input)();
            assert_eq!(input.len(), THREADS as usize, "{}", row.case);
            for (t, &word) in input.iter().enumerate() {
                // The first kernel runs `counts[t]` iterations, the second at most 8 whatever its
                // limit, so only the first kernel's words bound a loop.
                if !row.writes_counts {
                    assert!(
                        (1..=5).contains(&word),
                        "{}: thread {t} runs {word}",
                        row.case
                    );
                }
            }
        }
    }
}
