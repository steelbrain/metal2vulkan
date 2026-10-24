//! What a blit encoder's buffer copies and fills leave behind, measured on Metal.
//!
//! Ported from the runtime lane `runtime_lane/blits.rs`, whose doc comment says "the expected bytes
//! are Metal's: copies and fills of any range, overlapping copies within one buffer as `memmove`,
//! and every refusal, as measured on Metal". Three of those claims are now run on Metal rather than
//! restated: the lane computes each expectation in Rust — a `Vec` filled and `copy_within`'d — and
//! compares it against our own runtime, so nothing in it was ever tied to a Metal measurement.
//! Record mode runs each case below on the oracle, and the recording holds the bytes Metal's own
//! blit encoder produced.
//!
//! Four cases, one row each, kept apart rather than folded into one: a refused call stops a case's
//! run at that call, so a row that might be refused must not be able to take the others' sites with
//! it.
//!
//! - [`any_range`]: one blit encoder holding a fill of unaligned ends, a copy into its middle at
//!   odd offsets, a fill of two bytes inside one word and a fill of a lone tail, in that order,
//!   into a **private** buffer. It pins both that each command landed where Metal puts it and that
//!   they landed in encoded order, since three of the four overlap.
//! - [`overlapping_copies`]: `copyFromBuffer:` between two ranges of **one** buffer, forwards and
//!   backwards, in shared memory and in private, and one of them applied twice. Metal moves these
//!   as `memmove` moves them, which is the lane's claim and is what the recording now holds.
//! - [`empty_ranges`]: a copy of no bytes and a fill of no bytes, each at an offset inside the
//!   buffer and at one past its end. Plain Metal skips an empty range wherever it lies and runs the
//!   encoder's other commands, so the row's answer is that nothing changed and the command buffer
//!   completed.
//! - [`every_fill_size`]: five fills of one 600 KiB private buffer, sized to cross 64 KiB, 128 KiB
//!   and 256 KiB, and overlapping each other. This is the row that exercises the form a driver
//!   measured to compile its fill per command pool takes (MoltenVK: the first 64 KiB updated and
//!   doubled by copies), against the bytes Metal writes.
//!
//! **A case states what its buffers start from.** Metal does not promise what a fresh buffer holds,
//! so no row here reads a byte it did not write: a private destination is filled with a known byte
//! before the probe's own commands, which is also why [`any_range`] can compare the two bytes the
//! probe's first fill leaves alone. The lane reads those two as zero, which is a fresh Private
//! buffer's contents on both implementations today and nothing Metal guarantees.
//!
//! **Not ported, and why.** The lane's other tests are about our own hazard tracking, our fences,
//! `MTLStorageModeManaged` and where `Shared` storage lies on a device without unified memory —
//! each an expectation about the venue's memory model rather than a value Metal produced, and the
//! managed and system-memory rows have no answer at all on a Metal-less host. Its out-of-range
//! refusals are not portable either: our side returns a typed error, and finding out what plain
//! Metal does with a copy that runs past a buffer means handing the GPU an out-of-bounds transfer,
//! which this round does not do. Those stay lanes.
//!
//! Nothing here can fault a GPU: every command is a buffer copy or a buffer fill inside its
//! buffer, there is no shader, no texture and no loop, and every wait is bounded.

use crate::allowance::Allowance;
use crate::case::{call, Abort, Case, Input, MetalSource, Side};
use crate::transcript::{ByteLayout, Element, Negative, Node, Observed, Sites, Value};
use crate::work::{Resource, Step, Storage, Work};
use metal2vulkan_runtime::Degradation;

/// The byte a private destination is filled with before a row's own commands, so every byte the
/// row compares is a byte something wrote.
const START: u8 = 0xcd;

/// `length` bytes, byte `i` holding `i % 251`: a period no power of two divides, so a copy that
/// lands at the wrong offset shows. The lane's own pattern.
fn counting(length: usize) -> Vec<u8> {
    (0..length).map(|index| (index % 251) as u8).collect()
}

/// A buffer's bytes as a site: read byte by byte, so a diff names the byte that moved.
fn bytes_site(name: &str, bytes: Vec<u8>) -> Node {
    let length = bytes.len() as u64;
    Node::leaf(
        name,
        Observed::bytes(bytes, ByteLayout::buffer(0, length, Element::U8)),
    )
}

/// The command buffers' statuses, in commit order, as the site a row records them at. A fill or a
/// copy that failed its command buffer shows here rather than only in the bytes.
fn statuses_site(statuses: Vec<String>) -> Node {
    Node::leaf("statuses", Observed::value(Value::Lines(statuses)))
}

/// Add a copy of `length` bytes from `from` at `from_offset` to `to` at `to_offset`.
fn copy(work: &mut Work, from: usize, from_offset: u64, to: usize, to_offset: u64, length: u64) {
    work.step(Step::CopyBufferToBuffer {
        from,
        from_offset,
        to,
        to_offset,
        length,
    });
}

/// Add a fill of `[offset, offset + length)` of `buffer` with `value`.
fn fill(work: &mut Work, buffer: usize, offset: u64, length: u64, value: u8) {
    work.step(Step::FillBuffer {
        buffer,
        offset,
        length,
        value,
    });
}

/// The runs of equal bytes `bytes` holds: each a first byte, a length and the byte itself.
///
/// It is a **lossless** projection of the whole buffer, which is why a 600 KiB row can record this
/// instead of its bytes and still compare exactly: a fill's answer is piecewise constant, so the
/// runs say which range took which byte, and a fill that moved by one byte moves a run's offset
/// where a reader can see it rather than printing six hundred thousand numbers.
fn runs_of(bytes: &[u8]) -> Value {
    let mut runs = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        let byte = bytes[at];
        let mut end = at;
        while end < bytes.len() && bytes[end] == byte {
            end += 1;
        }
        runs.push(Value::fields([
            ("at", Value::Uint(at as u64)),
            ("length", Value::Uint((end - at) as u64)),
            ("byte", Value::Uint(u64::from(byte))),
        ]));
        at = end;
    }
    Value::List(runs)
}

/// One row of the family: its name, the program it runs, and the sites it emits from what the
/// program read.
struct Row {
    /// The case's name: its recording's file name, and its row in the report.
    case: &'static str,
    /// The program the row runs.
    program: fn() -> Work,
    /// The sites the row emits from the program's reads and statuses.
    sites: fn(&crate::work::Done) -> Vec<Node>,
    /// What the row sets that is not a buffer's bytes, as its input fingerprint.
    parameters: fn() -> Vec<u8>,
    /// The degrade notices this row's commands announce on a device whose copy or fill cannot take
    /// the form Metal's does. Metal announces none, so the notices site is the only field of these
    /// rows that may differ at all — every byte compares exactly — and each kind here is what
    /// justifies that one difference. A row that announced a kind not listed fails until the kind
    /// is added, which is the model working: an announced difference nobody named is not an
    /// announced difference.
    notices: &'static [Degradation],
}

/// A case of the family: one row.
pub struct BufferBlits {
    row: &'static Row,
}

impl Case for BufferBlits {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 4 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        std::iter::once(crate::replay::shared_memory_notice())
            .chain(
                self.row
                    .notices
                    .iter()
                    .map(|kind| crate::replay::notice_allowance(kind.code())),
            )
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // Every byte the program feeds its run, in resource order, and the geometry of every step:
        // either moving makes the recording's answer another program's.
        let work = (self.row.program)();
        let mut inputs = Vec::new();
        for (at, resource) in work.resources.iter().enumerate() {
            match resource {
                Resource::Buffer { bytes } => {
                    inputs.push(Input::bytes(format!("buffer:{at}"), bytes));
                }
                Resource::UnwrittenBuffer { length, storage } => {
                    let mut words = length.to_le_bytes().to_vec();
                    // The storage word is written **only where it is not private**, which is the
                    // storage the resource spelled before it carried the mode at all. So a case
                    // that means a private buffer fingerprints to the bytes it always has, and
                    // naming the mode here did not move three committed recordings.
                    if *storage != Storage::Private {
                        words.extend(storage.word().as_bytes());
                    }
                    inputs.push(Input::bytes(format!("private-buffer:{at}"), &words));
                }
                other => {
                    return Err(format!(
                        "{}: resource {at} is {other:?}, which this family does not state",
                        self.row.case
                    ))
                }
            }
        }
        inputs.push(Input::bytes("steps", &(self.row.parameters)()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let work = (self.row.program)();
        let done = call(sites, "work", side.work(&work), |_| Value::Absent)?;
        for node in (self.row.sites)(&done) {
            sites.push(node);
        }
        Ok(())
    }
}

/// A fill of unaligned ends, a copy into its middle at odd offsets, a fill inside one word and a
/// fill of a lone tail, in one blit encoder in that order.
static ANY_RANGE: Row = Row {
    case: "buffer-blit-any-range",
    program: || {
        let mut work = Work::new();
        let source = work.buffer(counting(64));
        let destination = work.add(Resource::UnwrittenBuffer {
            length: 64,
            storage: Storage::Private,
        });
        let read_back = work.buffer(vec![0; 64]);
        // The destination's starting byte, in a command buffer of its own, so what the probe's
        // own commands left alone is a byte this case wrote.
        fill(&mut work, destination, 0, 64, START);
        work.step(Step::Commit);
        // The probe's four commands, one encoder, in order.
        fill(&mut work, destination, 1, 62, 0x5a);
        copy(&mut work, source, 3, destination, 21, 10);
        fill(&mut work, destination, 41, 2, 0xee);
        fill(&mut work, destination, 48, 5, 0x11);
        work.step(Step::Commit);
        copy(&mut work, destination, 0, read_back, 0, 64);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: read_back,
            offset: 0,
            length: 64,
        });
        work
    },
    sites: |done| {
        vec![
            bytes_site("output", done.reads.first().cloned().unwrap_or_default()),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || b"fill 1+62 5a; copy 3->21+10; fill 41+2 ee; fill 48+5 11".to_vec(),
    // Three of the four fills cover bytes that are not whole 4-byte words at multiples of 4,
    // which is the whole point of the row.
    notices: &[Degradation::SplitBufferFill],
};

/// Overlapping ranges of one buffer, forwards and backwards, in shared memory and in private.
static OVERLAPPING: Row = Row {
    case: "buffer-blit-overlapping-copies",
    program: || {
        let mut work = Work::new();
        let forward = work.buffer(counting(4099));
        let backward = work.buffer(counting(4099));
        let private = work.add(Resource::UnwrittenBuffer {
            length: 4099,
            storage: Storage::Private,
        });
        let read_back = work.buffer(vec![0; 4099]);
        copy(&mut work, forward, 1, forward, 7, 4000);
        copy(&mut work, backward, 9, backward, 2, 4000);
        // The private copy takes the forward buffer as the encoder left it, then moves inside
        // itself: two overlapping moves, the second over the first's answer.
        copy(&mut work, forward, 0, private, 0, 4099);
        copy(&mut work, private, 5, private, 6, 4093);
        work.step(Step::Commit);
        copy(&mut work, private, 0, read_back, 0, 4099);
        work.step(Step::Commit);
        for buffer in [forward, backward, read_back] {
            work.step(Step::ReadBuffer {
                buffer,
                offset: 0,
                length: 4099,
            });
        }
        work
    },
    sites: |done| {
        let read = |at: usize| done.reads.get(at).cloned().unwrap_or_default();
        vec![
            bytes_site("forward", read(0)),
            bytes_site("backward", read(1)),
            bytes_site("private-twice", read(2)),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || b"copy 1->7+4000; copy 9->2+4000; copy private 5->6+4093".to_vec(),
    // Every copy of the row is between two overlapping ranges of one buffer, which Vulkan's copy
    // forbids outright.
    notices: &[Degradation::OverlappingBufferCopy],
};

/// A copy of no bytes and a fill of no bytes, inside the buffer and past its end.
static EMPTY_RANGES: Row = Row {
    case: "buffer-blit-empty-ranges",
    program: || {
        let mut work = Work::new();
        let source = work.buffer(counting(32));
        let destination = work.buffer(vec![0; 16]);
        copy(&mut work, source, 0, destination, 0, 0);
        copy(&mut work, source, 0, destination, 64, 0);
        fill(&mut work, destination, 4, 0, 1);
        fill(&mut work, destination, 64, 0, 1);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: destination,
            offset: 0,
            length: 16,
        });
        work
    },
    sites: |done| {
        let after = done.reads.first().cloned().unwrap_or_default();
        // The row's answer is a negative fact, so it is stated as one rather than as bytes that
        // happen to be zero: a side that moved a byte emits its bytes instead, and the two differ.
        let changed = if after == vec![0; 16] {
            Node::leaf(
                "changed",
                Observed::value(Value::Nothing(Negative::CopiedNoBytes)),
            )
        } else {
            bytes_site("changed", after.clone())
        };
        vec![
            changed,
            bytes_site("output", after),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || b"copy 0->0+0; copy 0->64+0; fill 4+0; fill 64+0".to_vec(),
    // Nothing is copied and nothing is filled, so nothing takes another form.
    notices: &[],
};

/// The 600 KiB [`EVERY_FILL_SIZE`] fills.
const BIG: u64 = 600 * 1024;

/// The fills of [`EVERY_FILL_SIZE`]: the lane's own, sized to cross 64 KiB, 128 KiB and 256 KiB
/// and to overlap each other, so the answer also states which fill ran last where two meet.
const FILLS: [(u64, u64, u8); 5] = [
    (0, BIG, 0x11),
    (3, 65_537, 0x22),
    (70_001, 3 * 65_536 + 6, 0x33),
    (300_000, 256 * 1024 + 4, 0x44),
    (BIG - 9, 9, 0x55),
];

/// Five fills of one private buffer, sized to cross the thresholds a transfer-compiled fill takes
/// its steps at.
static EVERY_FILL_SIZE: Row = Row {
    case: "buffer-fills-of-every-size",
    program: || {
        let mut work = Work::new();
        let destination = work.add(Resource::UnwrittenBuffer {
            length: BIG,
            storage: Storage::Private,
        });
        let read_back = work.buffer(vec![0; BIG as usize]);
        for (offset, length, value) in FILLS {
            fill(&mut work, destination, offset, length, value);
        }
        work.step(Step::Commit);
        copy(&mut work, destination, 0, read_back, 0, BIG);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: read_back,
            offset: 0,
            length: BIG as usize,
        });
        work
    },
    sites: |done| {
        let after = done.reads.first().cloned().unwrap_or_default();
        vec![
            // The runs are lossless and small; the digest pins the bytes themselves without
            // committing 600 KiB of them.
            Node::leaf("runs", Observed::value(runs_of(&after))),
            Node::leaf(
                "bytes",
                Observed::absent().digested(crate::transcript::Digest::sha256(&after)),
            ),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || {
        FILLS
            .iter()
            .flat_map(|(offset, length, value)| {
                let mut word = offset.to_le_bytes().to_vec();
                word.extend(length.to_le_bytes());
                word.push(*value);
                word
            })
            .collect()
    },
    // Four of the five fills start or end inside a word.
    notices: &[Degradation::SplitBufferFill],
};

/// The case of one blit encoder's fills and copies of any range.
pub fn any_range() -> BufferBlits {
    BufferBlits { row: &ANY_RANGE }
}

/// The case of overlapping copies within one buffer.
pub fn overlapping_copies() -> BufferBlits {
    BufferBlits { row: &OVERLAPPING }
}

/// The case of a copy and a fill of no bytes.
pub fn empty_ranges() -> BufferBlits {
    BufferBlits { row: &EMPTY_RANGES }
}

/// The case of five fills sized across a transfer-compiled fill's thresholds.
pub fn every_fill_size() -> BufferBlits {
    BufferBlits {
        row: &EVERY_FILL_SIZE,
    }
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(any_range()),
        Box::new(overlapping_copies()),
        Box::new(empty_ranges()),
        Box::new(every_fill_size()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The rows of the family, for the guards that read every one.
    const EVERY_ROW: [&Row; 4] = [&ANY_RANGE, &OVERLAPPING, &EMPTY_RANGES, &EVERY_FILL_SIZE];

    /// The bytes the private destination of [`ANY_RANGE`] holds once its four commands have run, as a
    /// reader of the lane's own expectation would write it: every byte stated, nothing read that
    /// nothing wrote.
    fn any_range_expected() -> Vec<u8> {
        let mut expected = vec![0x5a; 64];
        expected[0] = START;
        expected[63] = START;
        expected[21..31].copy_from_slice(&counting(64)[3..13]);
        expected[41..43].fill(0xee);
        expected[48..53].fill(0x11);
        expected
    }

    /// The bytes of `counting(4099)` with the forward overlapping copy applied.
    fn moved_forward() -> Vec<u8> {
        let mut moved = counting(4099);
        moved.copy_within(1..4001, 7);
        moved
    }

    /// The bytes a recorded site holds.
    fn recorded(case: &dyn Case, site: &str) -> Vec<u8> {
        let recording = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let answer = recording
            .answers
            .first()
            .expect("the recording holds an answer");
        let outcome = answer
            .outcomes
            .first()
            .expect("the answer holds an outcome");
        let node = outcome
            .transcript
            .root
            .child(site)
            .unwrap_or_else(|| panic!("the recording holds no {site}"));
        match &node.observed.value {
            Value::Bytes(bytes) => bytes.clone(),
            other => panic!("{site}: the recording holds {}", other.kind()),
        }
    }

    /// A row loosens the notices site and nothing else: a byte of these cases is never loosened,
    /// whatever form a device's copy or fill takes.
    #[test]
    fn a_row_loosens_the_notices_site_alone() {
        for case in all() {
            for allowance in case.allowances() {
                assert_eq!(
                    allowance.path,
                    format!("case/{}/*", crate::replay::NOTICES_SITE),
                    "{}",
                    case.name()
                );
            }
        }
        // The empty-range row copies and fills nothing, so it declares no notice of its own --
        // only the venue's, which every row of the family declares.
        assert_eq!(
            empty_ranges().allowances().len(),
            1,
            "the empty-range row copies and fills nothing, so it loosens nothing of its own"
        );
    }

    /// The cases name distinct rows and distinct sites.
    #[test]
    fn the_cases_are_distinct() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name()).collect();
        assert_eq!(named.len(), all().len());
        let rows: BTreeSet<&str> = EVERY_ROW.iter().map(|row| row.case).collect();
        assert_eq!(rows.len(), EVERY_ROW.len());
    }

    /// Every case states its inputs and asks a program with at least one read, so no case can
    /// record an answer nothing read.
    #[test]
    fn every_case_states_its_inputs_and_reads_something() {
        for case in all() {
            let inputs = case
                .inputs()
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            assert!(!inputs.is_empty(), "{}", case.name());
        }
        for row in EVERY_ROW {
            let work = (row.program)();
            assert!(work.reads() > 0, "{}", row.case);
        }
    }

    /// Every fill and copy of every program lies inside its buffer, so no program can hand a
    /// device an out-of-bounds transfer. The empty-range row's two commands past the destination
    /// are the exception the row is about, and they move no byte.
    #[test]
    fn every_command_lies_inside_its_buffer() {
        for row in EVERY_ROW {
            let work = (row.program)();
            let length = |at: usize| match &work.resources[at] {
                Resource::Buffer { bytes } => bytes.len() as u64,
                Resource::UnwrittenBuffer { length, .. } => *length,
                other => panic!("{}: resource {at} is {other:?}", row.case),
            };
            for (at, step) in work.steps.iter().enumerate() {
                let (held, offset, size) = match step {
                    Step::FillBuffer {
                        buffer,
                        offset,
                        length: size,
                        ..
                    } => (length(*buffer), *offset, *size),
                    Step::CopyBufferToBuffer {
                        to,
                        to_offset,
                        length: size,
                        ..
                    } => (length(*to), *to_offset, *size),
                    _ => continue,
                };
                assert!(
                    size == 0 || offset + size <= held,
                    "{}: step {at} reaches {} of {held} bytes",
                    row.case,
                    offset + size
                );
            }
        }
    }

    /// **The cross-check the port owes the lane.** Metal's own bytes, as the committed recordings
    /// hold them, are the bytes the lane computes in Rust: the fills and the copy of any range, and
    /// the overlapping copies as `memmove` moves them. The lane never ran Metal, so this is what
    /// ties its expectation to the oracle.
    #[test]
    fn metals_bytes_are_the_lanes_expectation() {
        assert_eq!(recorded(&any_range(), "output"), any_range_expected());

        let overlapping = overlapping_copies();
        assert_eq!(recorded(&overlapping, "forward"), moved_forward());
        let mut backward = counting(4099);
        backward.copy_within(9..4009, 2);
        assert_eq!(recorded(&overlapping, "backward"), backward);
        let mut twice = moved_forward();
        twice.copy_within(5..4098, 6);
        assert_eq!(recorded(&overlapping, "private-twice"), twice);
    }

    /// The runs of the 600 KiB row are the ranges the fills name, in the order the fills ran, so
    /// the recording's projection and the lane's own expectation agree.
    #[test]
    fn metals_fill_runs_are_the_lanes_ranges() {
        let mut expected = vec![0u8; BIG as usize];
        for (offset, length, value) in FILLS {
            expected[offset as usize..(offset + length) as usize].fill(value);
        }
        let recording =
            crate::recording::read(every_fill_size().name()).expect("the committed recording");
        let outcome = &recording.answers[0].outcomes[0];
        let runs = outcome
            .transcript
            .root
            .child("runs")
            .expect("the recording holds the runs");
        assert_eq!(runs.observed.value, runs_of(&expected));
    }

    /// The runs are a lossless projection: a buffer rebuilt from them is the buffer they were taken
    /// from. Without this the 600 KiB row would be recording a summary nobody checked.
    #[test]
    fn the_runs_rebuild_the_bytes_they_came_from() {
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i / 7 % 5) as u8).collect();
        let Value::List(runs) = runs_of(&bytes) else {
            panic!("the runs are a list");
        };
        let mut rebuilt = Vec::new();
        for run in &runs {
            let Value::Fields(fields) = run else {
                panic!("a run is fields");
            };
            let word = |name: &str| match fields[name] {
                Value::Uint(word) => word,
                _ => panic!("{name} is a word"),
            };
            assert_eq!(word("at"), rebuilt.len() as u64);
            rebuilt.extend(std::iter::repeat_n(
                word("byte") as u8,
                word("length") as usize,
            ));
        }
        assert_eq!(rebuilt, bytes);
        // And it really is a projection that can move: two neighbouring runs never share a byte.
        for pair in runs.windows(2) {
            let byte = |value: &Value| match value {
                Value::Fields(fields) => fields["byte"].clone(),
                _ => panic!("a run is fields"),
            };
            assert_ne!(byte(&pair[0]), byte(&pair[1]));
        }
    }
}
