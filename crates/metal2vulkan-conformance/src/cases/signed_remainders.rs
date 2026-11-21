//! `%` on signed integers: the remainder with the dividend's sign, in `int`, `short`, `char`,
//! `long` and `int2`, by run-time divisors and by constants, over operands of both signs with the
//! most negative values among them.
//!
//! One case over the owned fixture `kernel_signed_remainders`: each thread reads a 64-bit dividend
//! and divisor as four words and writes eight. Vulkan makes `OpSRem` and `OpSMod` poison for a
//! negative operand on a device that does not enable `maintenance8`, so the translator never
//! emits either; this case records what Metal writes for each operand pair, word by word, and
//! holds every device's answer to it. A remainder MSL leaves undefined (by zero, or of the most
//! negative value by -1) is never taken: the fixture writes a marker there instead, so every word
//! is a defined answer.
//!
//! Each word is a column of its own, so a diff names the width whose remainder moved. Every column
//! compares exactly: integer remainders have one answer, and no value is loosened.
//!
//! Nothing here can fault a GPU: one compute pass, two buffers, no texture, no loop, and each
//! thread reads its own four words.

use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};

/// The fixture and its entry point.
const FIXTURE: Fixture = Fixture {
    name: "kernel_signed_remainders",
    entry: "signed_remainders",
};

/// The words each thread writes.
const WORDS: u32 = 8;

/// What each of a thread's eight words is, in the order the fixture writes them.
const COLUMNS: [&str; WORDS as usize] = [
    "int",
    "short",
    "char",
    "long-low",
    "long-high",
    "int2-lanes-xor",
    "int-by-3",
    "int-by-minus-8",
];

/// The dividends: both signs, the most negative and largest `long`, low words that are the most
/// negative `int`, `short` and `char`, and values past 32 bits.
const DIVIDENDS: [i64; 17] = [
    i64::MIN,
    i64::MIN + 1,
    -((1 << 40) + 3),
    i32::MIN as i64,
    -100,
    -7,
    -1,
    0,
    1,
    7,
    100,
    i32::MAX as i64,
    (1 << 40) + 3,
    i64::MAX,
    0x0000_0001_8000_0000,
    0x0000_8000_FFFF_8080,
    0x0000_0000_0000_8000,
];

/// The divisors: both signs, the most negative and largest `long` and `int`, and low words whose
/// `short` and `char` are -1 or -128 (or zero, which the fixture does not divide by).
const DIVISORS: [i64; 14] = [
    i64::MIN,
    -((1 << 33) + 1),
    i32::MIN as i64,
    -7,
    -3,
    -1,
    1,
    3,
    7,
    (1 << 33) + 1,
    i32::MAX as i64,
    i64::MAX,
    0x0000_0001_FFFF_FFFF,
    0xFFFF_FFFF_0000_FF80u64 as i64,
];

/// The case.
pub struct SignedRemainders;

/// The case of the family.
pub fn case() -> SignedRemainders {
    SignedRemainders
}

impl SignedRemainders {
    /// Every operand pair, one thread each: each dividend by each divisor.
    fn pairs() -> Vec<(i64, i64)> {
        DIVIDENDS
            .iter()
            .flat_map(|x| DIVISORS.iter().map(move |y| (*x, *y)))
            .collect()
    }

    /// The input buffer's bytes: the dividend's low and high words, then the divisor's.
    fn input_bytes() -> Vec<u8> {
        Self::pairs()
            .into_iter()
            .flat_map(|(x, y)| {
                [x as u32, (x >> 32) as u32, y as u32, (y >> 32) as u32]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
            })
            .collect()
    }

    /// The bytes the pass writes: eight words a thread.
    fn output_bytes() -> usize {
        Self::pairs().len() * WORDS as usize * 4
    }

    /// The layout one column's words are read under: one row per operand pair.
    fn layout(column: u32) -> ByteLayout {
        let pitch = WORDS * 4;
        let rows = Self::pairs().len() as u32;
        ByteLayout {
            region: Region::Buffer {
                offset: u64::from(column) * 4,
                length: Self::output_bytes() as u64,
            },
            element: Element::U32,
            row_offset: column * 4,
            row_elements: 1,
            row_pitch: pitch,
            rows,
            image_pitch: pitch * rows,
            images: 1,
            // The columns are read apart, so each leaves the other seven words outside its own
            // elements; the sites together cover every byte.
            gaps: Gaps::Ignored,
        }
    }

    /// The `output` node of `bytes`: one child per column.
    fn output_node(bytes: Vec<u8>) -> Node {
        Node::branch(
            "output",
            (0..WORDS)
                .map(|column| {
                    Node::leaf(
                        COLUMNS[column as usize],
                        Observed::bytes(bytes.clone(), Self::layout(column)),
                    )
                })
                .collect(),
        )
    }
}

impl Case for SignedRemainders {
    fn name(&self) -> &'static str {
        "signed-remainders"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", FIXTURE.name),
                    entry: FIXTURE.entry.to_owned(),
                    source_sha256: crate::sha256_hex(FIXTURE.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(FIXTURE.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(FIXTURE.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:0", &Self::input_bytes()),
            // The output buffer starts zeroed on both sides, so a word the kernel did not write
            // is part of the answer rather than whatever was there.
            Input::bytes("buffer:1", &vec![0u8; Self::output_bytes()]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = FIXTURE.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let threads = Self::pairs().len() as u32;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, FIXTURE.entry),
        )?;
        let input = made(sites, "buffer/0", side.buffer(&Self::input_bytes()))?;
        let output = made(sites, "buffer/1", side.zeroed_buffer(Self::output_bytes()))?;
        // One thread a pair, as a grid of threads: the pairs are not a multiple of a threadgroup.
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
            side.read_buffer(output, 0, Self::output_bytes()),
            |_| Value::Absent,
        )?;
        sites.push(Self::output_node(bytes));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every operand pair is distinct, so each thread is a different question.
    #[test]
    fn the_operand_pairs_are_distinct() {
        let pairs = SignedRemainders::pairs();
        let distinct: BTreeSet<(i64, i64)> = pairs.iter().copied().collect();
        assert_eq!(distinct.len(), pairs.len());
    }

    /// The sweep holds what the case is for: a negative dividend by a positive divisor and the
    /// reverse at every width the fixture reads, the most negative `int`, `short`, `char` and
    /// `long` as dividends, and the pairs MSL leaves undefined, which the fixture marks.
    #[test]
    fn the_sweep_holds_each_sign_and_the_most_negative_values() {
        let pairs = SignedRemainders::pairs();
        let low = |v: i64| v as i32;
        let has = |f: &dyn Fn(i64, i64) -> bool| pairs.iter().any(|(x, y)| f(*x, *y));
        assert!(has(&|x, y| x < 0 && y > 0) && has(&|x, y| x > 0 && y < 0));
        assert!(has(&|x, y| low(x) < 0 && low(y) > 0) && has(&|x, y| low(x) > 0 && low(y) < 0));
        assert!(has(&|x, _| x == i64::MIN));
        assert!(has(&|x, _| low(x) == i32::MIN));
        assert!(has(&|x, _| x as i16 == i16::MIN));
        assert!(has(&|x, _| x as i8 == i8::MIN));
        assert!(has(&|x, y| x == i64::MIN && y == -1));
        assert!(has(&|x, y| low(x) == i32::MIN && low(y) == -1));
        assert!(has(&|_, y| low(y) == 0));
    }

    /// The columns together cover every word a thread writes, each exactly once.
    #[test]
    fn the_columns_cover_every_word_once() {
        let named: BTreeSet<&str> = COLUMNS.into_iter().collect();
        assert_eq!(named.len(), COLUMNS.len());
        let mut covered = vec![0u32; SignedRemainders::output_bytes() / 4];
        for column in 0..WORDS {
            let layout = SignedRemainders::layout(column);
            for row in 0..layout.rows {
                let at = layout.at(0, row, 0);
                assert_eq!(at % 4, 0);
                covered[at / 4] += 1;
            }
        }
        assert!(covered.iter().all(|times| *times == 1));
    }
}
