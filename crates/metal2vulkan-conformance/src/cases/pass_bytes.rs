//! The bytes a compute pass wrote.
//!
//! One owned fixture, one compute pipeline, one output buffer, one dispatch, one read-back. Every
//! word of the output compares against Metal's, and the diff names the thread's row and the word
//! inside it rather than a byte offset.
//!
//! Each case of the family is exhaustive over a small space — every byte value, every sixteenth
//! half bit pattern, the edge of the float-to-half conversion — which is where a rounding
//! difference hides and where one word moving is a readable failure. None reads a texture or an
//! input buffer, each binds one buffer, and none can fault a GPU.

use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Transcript, Value,
};

/// The threads in one threadgroup.
const THREADS_PER_GROUP: u32 = 64;

/// A column of the words each thread writes: the words of one answer, compared apart from the
/// others, so a diff names the answer that moved and not its neighbours.
#[derive(Clone, Copy, Debug)]
pub struct Column {
    /// The site's name under `output`.
    pub name: &'static str,
    /// The first of the thread's words this column holds.
    pub first_word: u32,
    /// The words it holds.
    pub words: u32,
}

impl Column {
    /// A column of `words` from `first_word`, compared exactly.
    const fn exact(name: &'static str, first_word: u32, words: u32) -> Self {
        Self {
            name,
            first_word,
            words,
        }
    }
}

/// The whole of a thread's words as one column, for a case that compares them as one.
const EVERY_WORD: &str = "words";

/// The columns of `a1_half_tanh_regimes`'s two words. A device that does not compute Metal's own
/// transcendental functions or arithmetic announces it (`device-transcendentals`,
/// `device-arithmetic`): the device's own answers, which the suite declares for every case
/// (`crate::replay::NATIVE_NOTES`) and which loosen no word, so both columns compare exactly and a
/// word the device's maths moves is a `known-gaps-rt.tsv` row of the venue.
const TANH_COLUMNS: &[Column] = &[
    Column::exact("fast-float-tanh", 0, 1),
    Column::exact("half-tanh", 1, 1),
];

/// A case that compares the bytes a compute pass wrote with Metal's.
pub struct PassBytes {
    name: &'static str,
    /// The owned fixture the case runs.
    fixture: Fixture,
    /// Threads in the grid, and words each writes.
    threads: u32,
    words_per_thread: u32,
    /// The columns of a thread's words, which compare as sites of their own. Empty means the
    /// whole row is one column, compared exactly.
    columns: &'static [Column],
}

/// Every case of the family: an owned fixture that writes one buffer and takes its input from the
/// thread position alone, swept over its whole small space.
///
/// Each is a self-contained compute pass that reads nothing, binds one buffer and cannot fault a
/// GPU, and each is exhaustive over the edge of a conversion where Metal's rounding is the only
/// answer there is.
pub fn all() -> Vec<PassBytes> {
    vec![
        // Metal's sRGB decode of every one of the 256 byte values, in float and in half: six
        // words a thread.
        PassBytes::new(
            "pass-bytes-srgb-decode-all-bytes",
            "a1_srgb_decode_all_bytes",
            256,
            6,
        ),
        // Metal's sRGB encode over the bottom of the float range and the normal half bit
        // patterns, packed to bytes: two words a thread.
        PassBytes::new(
            "pass-bytes-srgb-encode-sweep",
            "a1_srgb_encode_sweep",
            4096,
            2,
        ),
        // `tanh` of every sixteenth half bit pattern, in half, in a half vector and through the
        // fast float form: two words a thread.
        // Word 0 is `fast::tanh` of a float, a fast-math transcendental: a device that does not
        // compute Metal's own announces it, and on lavapipe its answer is thousands of units in
        // the last place away, so the column is unchecked where the notice fired. Word 1 is the
        // precise half `tanh`, in a scalar and in a vector, and compares exactly everywhere.
        PassBytes::new(
            "pass-bytes-half-tanh-regimes",
            "a1_half_tanh_regimes",
            4096,
            2,
        )
        .in_columns(TANH_COLUMNS),
        // The float-to-half conversion at and past the largest finite half, in both signs, and
        // half addition overflowing there: two words a thread.
        PassBytes::new(
            "pass-bytes-quantize-f16-overflow",
            "a1_quantize_f16_overflow",
            2048,
            2,
        ),
    ]
}

/// The first case of the family, which the suite's own tests use.
pub fn srgb_decode_all_bytes() -> PassBytes {
    all().swap_remove(0)
}

impl PassBytes {
    /// A case of `fixture`, whose entry point is the fixture's own name, over `threads` threads
    /// each writing `words_per_thread` words.
    fn new(name: &'static str, fixture: &'static str, threads: u32, words_per_thread: u32) -> Self {
        Self {
            name,
            fixture: Fixture::new(fixture),
            threads,
            words_per_thread,
            columns: &[],
        }
    }

    /// The case with its words compared as `columns` rather than as one.
    const fn in_columns(mut self, columns: &'static [Column]) -> Self {
        self.columns = columns;
        self
    }

    /// The columns the case compares: its own, or the whole row as one.
    fn columns(&self) -> Vec<Column> {
        if self.columns.is_empty() {
            return vec![Column::exact(EVERY_WORD, 0, self.words_per_thread)];
        }
        self.columns.to_vec()
    }

    /// The bytes the pass writes.
    fn output_bytes(&self) -> u64 {
        u64::from(self.threads) * u64::from(self.words_per_thread) * 4
    }

    /// The threadgroups and the threads in each.
    fn dispatch(&self) -> ([u32; 3], [u32; 3]) {
        (
            [self.threads / THREADS_PER_GROUP, 1, 1],
            [THREADS_PER_GROUP, 1, 1],
        )
    }

    /// The layout one column's words are read under: one row per thread, so a diff names the
    /// input whose answer moved and the word inside its row. The rows are packed, and a column
    /// reads only its own words, so there are no gaps to compare.
    fn layout(&self, column: &Column) -> ByteLayout {
        let pitch = self.words_per_thread * 4;
        ByteLayout {
            region: Region::Buffer {
                offset: u64::from(column.first_word) * 4,
                length: self.output_bytes(),
            },
            element: Element::U32,
            row_offset: column.first_word * 4,
            row_elements: column.words,
            row_pitch: pitch,
            rows: self.threads,
            image_pitch: pitch * self.threads,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }

    /// The `output` node of `bytes`: one child per column, each holding the whole read-back under
    /// its own layout, so every byte offset a diff prints is the buffer's own.
    fn output_node(&self, bytes: Vec<u8>) -> Node {
        Node::branch(
            "output",
            self.columns()
                .iter()
                .map(|column| {
                    Node::leaf(
                        column.name,
                        Observed::bytes(bytes.clone(), self.layout(column)),
                    )
                })
                .collect(),
        )
    }

    /// The sites the case emits, with `output` the bytes the pass wrote and `status` the command
    /// buffer's status: the shape both sides produce, so a carried or hand-built answer is built
    /// through the same function the run uses.
    pub fn transcript(&self, output: Vec<u8>, status: &str) -> Result<Transcript, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.value("library", Value::ok(Value::Absent));
        sites.value("pipeline", Value::ok(Value::Absent));
        sites.value("buffer/0", Value::ok(Value::Absent));
        sites.value("dispatch", Value::ok(Value::Absent));
        sites.value("status", Value::ok(Value::unit_variant(status)));
        sites.value("read-back", Value::ok(Value::Absent));
        sites.push(self.output_node(output));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(sites.finish())
    }
}

impl Case for PassBytes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", self.fixture.name),
                    entry: self.fixture.entry.to_owned(),
                    source_sha256: crate::sha256_hex(self.fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let (groups, threads) = self.dispatch();
        let mut words = Vec::new();
        for axis in groups.iter().chain(&threads) {
            words.extend(axis.to_le_bytes());
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(self.fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(self.fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("dispatch", &words),
            // The output buffer starts zeroed on both sides, so the bytes the pass did not write
            // are part of the answer rather than whatever was there.
            Input::bytes("buffer:0", &vec![0u8; self.output_bytes() as usize]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self.fixture.loaded().map_err(|why| stop("library", why))?;
        let (groups, threads) = self.dispatch();
        let length = self.output_bytes() as usize;

        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, self.fixture.entry),
        )?;
        let output = made(sites, "buffer/0", side.zeroed_buffer(length))?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch(pipeline, &Bindings::buffers(&[output]), groups, threads),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(output, 0, length),
            |_| Value::Absent,
        )?;
        sites.push(self.output_node(bytes));
        Ok(())
    }
}
