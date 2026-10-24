//! `frexp` and `ilogb`, the maths Apple's compiler emits as calls of runtime routines its backend
//! defines at pipeline creation.
//!
//! Two cases, one per width, over the owned fixture `kernel_frexp_ilogb`: each thread reads one
//! input word and writes eight — the scalar `frexp`'s mantissa and exponent, `ilogb`,
//! `precise::ilogb`, and lanes of the four-lane forms beside other lanes' constants, the last word
//! also folding in a 64-bit population count. The inputs sweep what the routines are made of:
//! both zeros, normals, denormals, the largest finite value, both infinities, and quiet and
//! signalling NaNs with payloads.
//!
//! Why these two: `runtime_lane::math_routines` states Metal's answer as a **rule** it computes on
//! the host, not as a table Metal wrote. The rule was fitted to a probe whose numbers are gone, so
//! the lane holds no Metal value at all and nothing checks the fit. These cases record what Metal
//! actually writes for every input, word by word, so the measurement stops being a reconstruction.
//!
//! Each word is a column of its own, because the eight words are different answers: a diff names
//! `frexp-mantissa` or `ilogb` of one input rather than a row of eight numbers. Every column
//! compares exactly; there is nothing here a device may round differently, and no value is
//! loosened. The only allowance is the declaration of the notice a device without unified memory
//! announces for the pass's `Shared` buffers ([`PASS_NOTICES`]).
//!
//! Nothing here can fault a GPU: one compute pass a case, two buffers, no texture, no loop, and
//! each thread reads its own word.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};

/// The fixture both entry points live in.
const FIXTURE: &str = "kernel_frexp_ilogb";

/// The words each thread writes.
const WORDS: u32 = 8;

/// What each of a thread's eight words is, in the order the fixture writes them. The names are the
/// diff's: a failure says which answer moved, not which offset.
const COLUMNS: [&str; WORDS as usize] = [
    "frexp-mantissa",
    "frexp-exponent",
    "ilogb",
    "precise-ilogb",
    "frexp4-mantissa-lane1",
    "frexp4-exponent-lane3",
    "ilogb4-lane1",
    "folded-lanes-and-popcount",
];

/// The `float` inputs, as their bits: both zeros, normals, denormals, the extremes, both
/// infinities, and quiet and signalling NaNs with payloads.
const FLOATS: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f40_0000,
    0x4060_0000,
    0x0000_0001,
    0x807f_ffff,
    0x0040_0000,
    0x0080_0000,
    0x80c0_0000,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0001,
    0xffc1_2345,
    0x7f80_0001,
    0x42f6_e979,
    0x7149_f2ca,
    0x0d80_0000,
];

/// The `half` inputs, in the low 16 bits of each word: both zeros, normals, denormals, the
/// extremes, both infinities and NaNs.
const HALVES: &[u32] = &[
    0x0000, 0x8000, 0x3c00, 0xbc00, 0x0001, 0x8001, 0x03ff, 0x0200, 0x0400, 0x7bff, 0x7c00, 0xfc00,
    0x7e01, 0xfe55, 0x7c01, 0x3555, 0xc8f0, 0x5640,
];

/// A case: one entry point of the fixture over one sweep of inputs.
pub struct MathRoutines {
    name: &'static str,
    entry: &'static str,
    /// The input words, one thread each.
    inputs: &'static [u32],
}

/// Both cases of the family.
pub fn all() -> Vec<MathRoutines> {
    vec![
        MathRoutines {
            name: "math-frexp-ilogb-float",
            entry: "frexp_ilogb_float",
            inputs: FLOATS,
        },
        MathRoutines {
            name: "math-frexp-ilogb-half",
            entry: "frexp_ilogb_half",
            inputs: HALVES,
        },
    ]
}

impl MathRoutines {
    /// The fixture under the case's entry point.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: self.entry,
        }
    }

    /// The input buffer's bytes: one word a thread.
    fn input_bytes(&self) -> Vec<u8> {
        self.inputs
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    /// The bytes the pass writes: eight words a thread.
    fn output_bytes(&self) -> usize {
        self.inputs.len() * WORDS as usize * 4
    }

    /// The layout one column's words are read under: one row per input, so a diff names the input
    /// whose answer moved and reads one word of its row. The value is the whole read-back under
    /// each column's own layout, so every byte offset a diff prints is the buffer's own.
    fn layout(&self, column: u32) -> ByteLayout {
        let pitch = WORDS * 4;
        ByteLayout {
            region: Region::Buffer {
                offset: u64::from(column) * 4,
                length: self.output_bytes() as u64,
            },
            element: Element::U32,
            row_offset: column * 4,
            row_elements: 1,
            row_pitch: pitch,
            rows: self.inputs.len() as u32,
            image_pitch: pitch * self.inputs.len() as u32,
            images: 1,
            // The columns are read apart, so each leaves the other seven words outside its own
            // elements; the sites together cover every byte, so there is nothing to compare twice.
            gaps: Gaps::Ignored,
        }
    }

    /// The `output` node of `bytes`: one child per column.
    fn output_node(&self, bytes: Vec<u8>) -> Node {
        Node::branch(
            "output",
            (0..WORDS)
                .map(|column| {
                    Node::leaf(
                        COLUMNS[column as usize],
                        Observed::bytes(bytes.clone(), self.layout(column)),
                    )
                })
                .collect(),
        )
    }
}

impl Case for MathRoutines {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration and no tolerance: both of this pass's buffers is `Shared`, which is
        // what [`crate::replay::shared_memory_notice`] is for. It loosens the notices site alone,
        // so every value this case reads still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: self.entry.to_owned(),
                    source_sha256: crate::sha256_hex(self.fixture().text(extension)?.as_bytes()),
                })
            })
            .collect()
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
            Input::bytes("buffer:0", &self.input_bytes()),
            // The output buffer starts zeroed on both sides, so a word the kernel did not write
            // is part of the answer rather than whatever was there.
            Input::bytes("buffer:1", &vec![0u8; self.output_bytes()]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self
            .fixture()
            .loaded()
            .map_err(|why| stop("library", why))?;
        let threads = self.inputs.len() as u32;

        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, self.entry),
        )?;
        let input = made(sites, "buffer/0", side.buffer(&self.input_bytes()))?;
        let output = made(sites, "buffer/1", side.zeroed_buffer(self.output_bytes()))?;
        // One thread an input, as a grid of threads rather than of threadgroups: the sweeps are
        // not multiples of a threadgroup, and Metal's edge threadgroup is part of the answer.
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                &Bindings::buffers(&[input, output]),
                &[],
                [threads, 1, 1],
                [threads, 1, 1],
            ),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(output, 0, self.output_bytes()),
            |_| Value::Absent,
        )?;
        sites.push(self.output_node(bytes));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Each case's inputs are distinct, so every thread is a different question and a recording of
    /// one input cannot stand for another.
    #[test]
    fn each_sweeps_inputs_are_distinct() {
        for case in all() {
            let distinct: BTreeSet<u32> = case.inputs.iter().copied().collect();
            assert_eq!(distinct.len(), case.inputs.len(), "{}", case.name);
        }
    }

    /// The columns together cover every word a thread writes, each exactly once: a column left out
    /// would be a word Metal wrote that nothing compares.
    #[test]
    fn the_columns_cover_every_word_once() {
        assert_eq!(COLUMNS.len(), WORDS as usize);
        let named: BTreeSet<&str> = COLUMNS.into_iter().collect();
        assert_eq!(named.len(), COLUMNS.len());
        for case in all() {
            let mut covered = vec![0u32; case.output_bytes() / 4];
            for column in 0..WORDS {
                let layout = case.layout(column);
                for row in 0..layout.rows {
                    let at = layout.at(0, row, 0);
                    assert_eq!(at % 4, 0, "{}", case.name);
                    covered[at / 4] += 1;
                }
            }
            assert!(
                covered.iter().all(|times| *times == 1),
                "{}: a word is covered {:?} times",
                case.name,
                covered.iter().collect::<BTreeSet<_>>()
            );
        }
    }
}
