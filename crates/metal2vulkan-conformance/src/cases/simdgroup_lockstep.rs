//! Threadgroup words that only a simdgroup's lockstep orders.
//!
//! Ported from the two lockstep rows of the runtime lane `runtime_lane/driver_deviations.rs`. The
//! owned fixture `rtl_lockstep` holds two kernels over one 64-thread threadgroup — two simdgroups —
//! whose loops each thread leaves after its own count, with no barrier inside: thread `t` reads
//! another thread's word of its own simdgroup and writes its own from it. Nothing but the
//! simdgroup running its lanes in lockstep decides what those words come to, so the recorded words
//! **are** Metal's lockstep, in the one shape where it is observable.
//!
//! That fact is load-bearing for the translator. `DriverDeviation::SimdgroupLockstep` is the
//! measured driver answer that lets a kernel keep these accesses with no barrier where the device
//! runs a simdgroup in lockstep itself, and take emulated simdgroups everywhere else. The lane
//! checks which of the two a profile took and compares the words against an expectation it computes
//! in Rust — a simulation of lockstep (this crate's own `tests::in_lockstep`, kept here as the
//! readable model) whose only tie to Metal is a comment saying it was measured on an Apple M3 Max
//! and an Apple M4. Record mode now runs the kernels on the oracle, so the recording holds Metal's
//! own words and the simulation is checked against them.
//!
//! Each case records all 512 bytes of its output buffer under a layout of two rows of 64 words:
//! for `until` the second row is each thread's iteration count, and for `divergent_loop` the kernel
//! writes nothing there, so the row is the `0xee` the buffer was made with and the case also
//! records that the dispatch wrote nothing past its words.
//!
//! **The rest of `driver_deviations` is not portable, and the two reasons are worth writing down.**
//! Its float-denormal, fused-pair, half-edge, `dot`, half-arithmetic, minimum-and-maximum and
//! float-function rows all reach their Metal values through `public_case` and `metal_row`, which
//! read the **harvested corpus store** and its stored Metal rows; the conformance suite may not
//! touch any of that, so those values cannot come here however much Metal has an answer for them.
//! Its remaining rows — which profile took which deviation, and which `VkProperty` a compiled
//! kernel then requires — are facts about *our* compile decisions on *this* driver release, which
//! Metal has no counterpart for at all. Both kinds stay lanes. (The cube-gradient rows are a third
//! case: Metal has an answer, but it too is a corpus case's stored row, and the fragment leg needs
//! a render pass this stream does not build.)
//!
//! Nothing here can fault a GPU: one compute pass a row, one 64-thread threadgroup, two shared
//! buffers, no texture, and each loop bounded by a buffer word's count or by 8.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{
    ByteLayout, Digest, Element, Gaps, Negative, Observed, Region, Sites, Value,
};
use metal2vulkan_runtime::Degradation;

/// The fixture both kernels live in.
const FIXTURE: &str = "rtl_lockstep";

/// The threads the one threadgroup holds: two 32-lane simdgroups.
const THREADS: u32 = 64;

/// The words the output buffer holds: the kernels' `out[t]` and `out[64 + t]`.
const WORDS: u32 = 2 * THREADS;

/// The byte the output buffer is made with, so what a kernel leaves alone is a recorded value
/// rather than whatever the allocator left. The lane's own byte.
const FILL: u8 = 0xee;

/// One row: a kernel of the fixture and the words its input buffer holds.
struct Row {
    /// The case's name.
    case: &'static str,
    /// The kernel, by its name in the library.
    entry: &'static str,
    /// The input words, as the lane picks them.
    input: fn() -> Vec<u32>,
    /// Whether the kernel writes the second half of the output buffer. Where it does not, the case
    /// records that the dispatch changed nothing there.
    writes_counts: bool,
}

/// `rtl_lockstep_divergent_loop`'s iteration counts, 1 to 5. The lane's own.
fn counts() -> Vec<u32> {
    (0..THREADS).map(|t| 1 + (t * 7 + 3) % 5).collect()
}

/// `rtl_lockstep_until`'s limits. The lane's own.
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

/// The layout the output buffer compares under: two rows of 64 words, so a diff names the thread
/// whose word moved and says whether it was its word or its count.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(WORDS) * 4,
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: THREADS,
        row_pitch: THREADS * 4,
        rows: 2,
        image_pitch: WORDS * 4,
        images: 1,
        gaps: Gaps::Compared,
    }
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

/// The degrade notices one of these cases may announce. A list rather than a device query,
/// because a case never branches on the device: it runs and reads what came back, and a venue that
/// announces a kind not here fails the case until the kind is added.
///
/// Declaring a kind states what our side announced at the notices site and loosens no value:
/// every lane of every record still compares exactly.
const PASS_NOTICES: &[Degradation] = &[
    // The buffer each dispatch writes its per-lane record into is `Shared`.
    // A device with no unified memory keeps a `Shared` allocation in system memory because that is
    // the only memory both sides can reach, so this notice belongs to the venue's memory model
    // rather than to anything these cases do. A device whose driver is Metal announces nothing
    // for it.
    Degradation::SharedInSystemMemory,
];

impl Case for Lockstep {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        PASS_NOTICES
            .iter()
            .map(|kind| crate::replay::notice_allowance(kind.code()))
            .collect()
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
        sites.leaf("output", Observed::bytes(bytes, layout()));
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

    /// The lane's own model of lockstep, kept exactly as
    /// `runtime_lane::driver_deviations::in_lockstep` states it: each simdgroup's threads run each
    /// iteration together, so the threads still in the loop all read, then all write. `step` gives
    /// a thread's iteration from the words as the iteration found them: `None` where the thread
    /// leaves the loop, else its new word.
    pub(super) fn in_lockstep(
        words: &mut [u32],
        mut step: impl FnMut(usize, u32, &[u32]) -> Option<u32>,
    ) {
        let mut active = vec![true; words.len()];
        let mut iteration = 0;
        while active.iter().any(|&a| a) {
            let before = words.to_vec();
            for t in 0..words.len() {
                if active[t] {
                    match step(t, iteration, &before) {
                        Some(word) => words[t] = word,
                        None => active[t] = false,
                    }
                }
            }
            iteration += 1;
        }
    }

    /// The lane's expectation of `rtl_lockstep_divergent_loop`.
    fn lanes_divergent_loop() -> Vec<u32> {
        let counts = counts();
        let mut words: Vec<u32> = (0..THREADS).collect();
        in_lockstep(&mut words, |t, i, before| {
            (i < counts[t]).then(|| {
                before[t ^ 16]
                    .wrapping_mul(3)
                    .wrapping_add(i)
                    .wrapping_add(1)
            })
        });
        let mut out = words;
        out.resize(WORDS as usize, u32::from_le_bytes([FILL; 4]));
        out
    }

    /// The lane's expectation of `rtl_lockstep_until`.
    fn lanes_until() -> Vec<u32> {
        let limits = limits();
        let mut words: Vec<u32> = (1..=THREADS).collect();
        let mut steps = vec![0u32; THREADS as usize];
        in_lockstep(&mut words, |t, _, before| {
            if before[t] >= limits[t] || steps[t] >= 8 {
                return None;
            }
            steps[t] += 1;
            let partner = ((t + 8) & 31) | (t & 32);
            Some(before[t].wrapping_add(before[partner]))
        });
        words.extend(steps);
        words
    }

    /// The words a case's committed recording holds, as 32-bit words.
    fn recorded(case: &dyn Case) -> Vec<u32> {
        let recording = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let outcome = &recording.answers[0].outcomes[0];
        let node = outcome
            .transcript
            .root
            .child("output")
            .expect("the recording holds the output");
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("the recording holds no bytes");
        };
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }

    /// **The cross-check the port owes the lane.** Metal's own words, as the committed recordings
    /// hold them, are the words the lane's simulation of simdgroup lockstep predicts, for both
    /// kernels and every thread. The lane's tie to Metal was a comment; this is the evidence.
    #[test]
    fn metals_words_are_the_lanes_lockstep_model() {
        assert_eq!(recorded(&divergent_loop()), lanes_divergent_loop());
        assert_eq!(recorded(&until()), lanes_until());
    }

    /// The layout covers the output buffer exactly, and each row names a distinct case and entry
    /// point: a layout that reached further would read past the buffer.
    #[test]
    fn the_layout_covers_the_buffer_and_the_rows_are_distinct() {
        assert_eq!(layout().extent(), (WORDS * 4) as usize);
        assert_ne!(ROWS[0].case, ROWS[1].case);
        assert_ne!(ROWS[0].entry, ROWS[1].entry);
    }

    /// Each row's inputs are 64 words, one a thread, and every count and limit is inside the
    /// bound the fixture's loops carry: a word outside them would hand the device a loop the
    /// fixture's comment does not bound.
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

    /// The kernel that writes no count records that fact, and the one that does records its
    /// counts: a row that claimed the wrong one would compare a negative fact against a value.
    #[test]
    fn only_the_divergent_loop_leaves_the_counts_half_alone() {
        assert!(!ROWS[0].writes_counts);
        assert!(ROWS[1].writes_counts);
        let counts = lanes_until();
        assert!(
            counts[THREADS as usize..].iter().any(|&steps| steps > 0),
            "the `until` kernel's counts are not all zero"
        );
        assert!(
            lanes_divergent_loop()[THREADS as usize..]
                .iter()
                .all(|&word| word == u32::from_le_bytes([FILL; 4])),
            "the divergent loop writes nothing past its words"
        );
    }
}
