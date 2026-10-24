//! Every `MTLAttributeFormat` a kernel's `[[stage_in]]` can read, decoded by
//! `MTLStageInputOutputDescriptor`.
//!
//! `MTLAttributeFormat` has 54 enumerators; the suite reached **one** of them before this family:
//! `stage_in_regions`
//! and `indirect_commands` both fetch a `UInt` and ask other questions of it. So the stage-in
//! path's decoding was untested, although it is the same field as the vertex descriptor's and the
//! translator decodes both through one module
//! (`metal2vulkan::lower::stage_io::fetch`, reached from `stage_io::stage_in` for a kernel and from
//! `stage_io` for a pulled vertex attribute).
//!
//! The value list and each format's shape come from [`crate::attribute_coding`], which the vertex
//! sweep uses too: the generated tables give `MTLAttributeFormat` and `MTLVertexFormat` the same 54
//! enumerator names with the same ordinals, so two shape parsers and two code tables would be two
//! answers to one question. Each site is named after the SDK enumerator, so the option-space
//! census reads the values out of the recording.
//!
//! # Why a new family rather than a wider `stage_in_regions`
//!
//! `stage_in_regions` asks how a region **clamps** an index, over 28 rows of step functions, step
//! rates, regions, grids and threadgroups, through a fixture whose attributes are `uint` and
//! nothing else. Widening it to 53 formats would multiply those 28 rows by a field they have
//! nothing to do with, and would still need another fixture, because `stage_in_regions.metal`
//! declares no float, half or integer-vector attribute to decode into. The fixture that does is
//! `kernel_stage_in`, whose `stage_in_fetch_wide` declares one attribute of each class and reports
//! every one as raw bits, which is exactly this family's question and none of that one's.
//!
//! # The five declared attributes, and what each can read
//!
//! `stage_in_fetch_wide`'s `[[stage_in]]` holds `float4 f`, `uint4 u`, `int i`, `half2 h` and
//! `int4 v`,
//! so the descriptor — not the source — chooses each one's format. A format reaches the shader only
//! through an attribute whose declared type admits it, which splits the field by **numeric class**:
//!
//! | attribute | declared | formats of its own class |
//! |---|---|---|
//! | 0 | `float4` | every float coding: normalized, half, float, the packed small floats, BGRA |
//! | 1 | `uint4` | `UInt`, `UInt2`, `UInt3`, `UInt4` |
//! | 2 | `int` | `Int` |
//! | 3 | `half2` | `Half`, `Half2` |
//! | 4 | `int4` | `Int`, `Int2`, `Int3`, `Int4` |
//!
//! Attribute 4 is why the signed 32-bit **vectors** are compared at all. `int i` takes one
//! component, so `MTLAttributeFormatInt2`, `Int3` and `Int4` reached no declaration the fixture
//! held, and no float declaration takes a 32-bit integer format in any width. `Int` is on both
//! attribute 2 and attribute 4 by the same rule — a signed 4-byte coding — which is not a value
//! compared twice but two questions of one value: what one component of it reads as, and what the
//! three components the format does not hold read as beside it.
//!
//! The class is read from the shape the SDK name states, never from a list of names, and it is
//! Metal's own line: it is also where the family's two cases divide.
//!
//! # The two cases
//!
//! - **`attribute-format-fetches`** compares the **values** of the 34 formats that reach an
//!   attribute of their own class. One row sets all four attributes and varies all four: each
//!   attribute's own format list is cycled, so attribute 0's list — the longest — names the row
//!   while the other three step through theirs beside it. A row that held three still would be the
//!   same experiment as the row beside it, which is what
//!   `tests::every_row_is_its_own_experiment_inside_metals_limits` holds against. Each attribute's
//!   answer is its own leaf, named after the format that attribute carried, so the enumerator Metal
//!   was asked for is in the compared text and a diff names the format.
//!
//!   The control that makes a swept leaf mean something is
//!   `tests::an_attributes_answer_depends_on_its_own_format_alone`: two rows that put one format on
//!   an attribute answered the same words there although every other attribute's format moved
//!   between them, so a descriptor leaking a format, an offset or a buffer index from the attribute
//!   beside it is caught. `tests::every_thread_reported_its_own_index` holds the other floor — the
//!   thread's own index is in the compared text, so the x step is known to have fetched per thread
//!   rather than once — and `tests::no_two_rows_answered_the_same_words` holds that no row compares
//!   what another already compared.
//!
//! - **`attribute-format-class-crossings`** asks of the 16 formats that cross the classes — the 8-
//!   and 16-bit integer codings on the `float4` attribute, which Metal converts to float — only
//!   whether the pipeline can be made. It records no buffer, no dispatch and no read-back, because
//!   the decoding of those sixteen is already compared exactly on the vertex path
//!   (`vertex-format-float4`) and `tests::the_stage_in_decode_is_the_vertex_decode` holds Metal's
//!   two fetches to one answer, so what is unknown of the stage-in path is the pipeline alone.
//!   Asking no more also means one refused format costs one row of `known-gaps.tsv` rather than one
//!   per leaf of a dispatch that never ran.
//!
//! # What is not reached here, and why
//!
//! - **`MTLAttributeFormatInvalid`** is the absence of an attribute, not a format. It is the
//!   descriptor's value for *no attribute here*, so what it would compare is a fetch that does not
//!   happen: the kernel's `[[stage_in]]` requires a declaration at each attribute index it names,
//!   and setting one of those indices to `Invalid` leaves the stage input short of it, which is a
//!   pipeline Metal refuses rather than a decoding it performs. There is no value row to write for
//!   it, and the refusal belongs to the descriptor families rather than to a format sweep. It is
//!   the one name on `LEFT_OUT`, and the only `MTLAttributeFormat` this family does not compare.
//!
//! `tests::the_formats_left_out_are_named` holds the two cases' lists and that list to being the
//! whole SDK table between them, with nothing in two of them, so the gap cannot widen quietly.
//!
//! # The shape, and why nothing here can fault a GPU
//!
//! One records buffer, laid out exactly as the vertex sweep's: row `t`'s slot for every format at
//! byte `SLOT i` of element `t`, and the filler everywhere else, so a fetch that read past its own
//! components read a value no code list holds. One buffer layout at Metal buffer index 1, stepped
//! by the thread's x position at rate 1, so thread `t` fetches element `t` and the fetched index is
//! visible in the answer rather than assumed. A grid of [`ROWS`] by one thread, one threadgroup,
//! one dispatch and one wait a row; every fetch is inside the buffer the row bound; no texture, no
//! loop.

use crate::allowance::Allowance;
use crate::attribute_coding as coding;
use crate::attribute_coding::{shape, FILLER, ROWS, SLOT};
use crate::case::{
    call, made, probe, Abort, Bindings, Case, Failure, Input, MetalSource, Side, StageInAttribute,
    StageInLayout, StageInRegion, StageInput,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{AttributeFormat, StepFunction};

/// The fixture the entry point lives in.
const FIXTURE: &str = "kernel_stage_in";

/// The entry point: one attribute of each class, each reported as raw bits.
///
/// `stage_in_fetch_wide` and not `stage_in_fetch`, which is the same four attributes without the
/// `int4`. The two exist apart because `stage_in_fetch`'s four are the contract a dozen of the
/// translator's own stage-input tests state their descriptors against, and a fifth attribute the
/// kernel declares and those descriptors do not describe is an invalid descriptor to every one of
/// them. So the wide declaration is its own entry point in the same fixture.
const ENTRY: &str = "stage_in_fetch_wide";

/// The Metal buffer index every attribute's layout reads.
const SOURCE: u32 = 1;

/// The Metal buffer index an indexed step would read its indices from. No row steps that way, so
/// nothing is bound there; the descriptor names it because `MTLStageInputOutputDescriptor` has the
/// field whether a layout uses it or not.
const INDEX_BUFFER: u32 = 2;

/// The stage-in region every row sets: the whole grid, so no fetch is clamped.
///
/// `setStageInRegion:` is part of the stage-in contract rather than an optional extra -- how a
/// region clamps an index is `stage_in_regions`' question, and this family's rows must not be
/// answering it by accident -- so every row sets the region to exactly its grid and no thread's
/// fetch is clamped.
const REGION: StageInRegion = StageInRegion::Direct {
    origin: [0, 0, 0],
    size: [ROWS as u32, 1, 1],
};

/// Words one thread writes: four `uint4`s.
const ROW_WORDS: u32 = 16;

/// Bytes one thread writes.
const ROW_BYTES: u32 = 4 * ROW_WORDS;

/// The output buffer's length: longer than the grid writes, so every row also records that the
/// dispatch wrote nothing past its grid.
const OUTPUT_BYTES: usize = 832;

// ---- The four declared attributes --------------------------------------------------------------

/// One of the five `[[stage_in]]` attributes `stage_in_fetch_wide` declares, by what it is
/// declared as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Declared {
    /// `float4 f [[attribute(0)]]`.
    Float4,
    /// `uint4 u [[attribute(1)]]`.
    UInt4,
    /// `int i [[attribute(2)]]`.
    Int,
    /// `half2 h [[attribute(3)]]`.
    Half2,
    /// `int4 v [[attribute(4)]]`, which is the only declaration a signed 32-bit **vector** format
    /// reaches: `int i` takes one component and no float declaration takes a 32-bit integer at all.
    Int4,
}

impl Declared {
    /// The five, in attribute-index order.
    const ALL: [Declared; 5] = [
        Declared::Float4,
        Declared::UInt4,
        Declared::Int,
        Declared::Half2,
        Declared::Int4,
    ];

    /// The Metal attribute index the kernel declares it at.
    fn index(self) -> u32 {
        match self {
            Declared::Float4 => 0,
            Declared::UInt4 => 1,
            Declared::Int => 2,
            Declared::Half2 => 3,
            Declared::Int4 => 4,
        }
    }

    /// Whether `format`'s coding is of the **same numeric class** as the declared type: a float
    /// coding on a `float`- or `half`-declared attribute, a 32-bit integer coding of the same
    /// signedness on an integer-declared one.
    ///
    /// Decided from the shape the SDK name states, never from a list of names. This is the line
    /// Metal's own API draws, so it is the line between this family's two cases: a format of the
    /// attribute's own class has its **values** compared, and a format of the other class is a
    /// question about whether the pipeline can be made at all.
    fn same_class(self, format: AttributeFormat) -> bool {
        let Some(held) = shape(format.name()) else {
            return false;
        };
        match self {
            Declared::Float4 => !matches!(held.coding, coding::Coding::Integer { .. }),
            Declared::UInt4 => {
                held.coding
                    == coding::Coding::Integer {
                        bytes: 4,
                        signed: false,
                    }
            }
            Declared::Int => {
                held.components == 1
                    && held.coding
                        == coding::Coding::Integer {
                            bytes: 4,
                            signed: true,
                        }
            }
            Declared::Half2 => held.coding == coding::Coding::Half && held.components <= 2,
            Declared::Int4 => {
                held.coding
                    == coding::Coding::Integer {
                        bytes: 4,
                        signed: true,
                    }
            }
        }
    }

    /// Whether `format` reaches the declared type across the numeric classes: an 8- or 16-bit
    /// integer format on a float-declared attribute, which Metal converts to float.
    ///
    /// Only the `float4` attribute can be crossed into, and only by an integer coding narrower
    /// than 32 bits: a 32-bit integer format reaches the shader as an integer, and no float coding
    /// reaches an integer-declared attribute at all.
    fn crosses_class(self, format: AttributeFormat) -> bool {
        let Some(held) = shape(format.name()) else {
            return false;
        };
        self == Declared::Float4
            && matches!(held.coding, coding::Coding::Integer { bytes, .. } if bytes < 4)
    }

    /// Every format of the SDK table this attribute reads **in its own numeric class**, in the
    /// table's own declaration order.
    fn formats(self) -> Vec<AttributeFormat> {
        AttributeFormat::ALL
            .iter()
            .copied()
            .filter(|format| self.same_class(*format))
            .collect()
    }

    /// Every format of the SDK table that reaches this attribute across the classes, in the
    /// table's own declaration order.
    fn crossings(self) -> Vec<AttributeFormat> {
        AttributeFormat::ALL
            .iter()
            .copied()
            .filter(|format| self.crosses_class(*format))
            .collect()
    }
}

// ---- The records -------------------------------------------------------------------------------

/// Every format the records lay a slot out for: every one of the SDK table that states a shape.
///
/// A format no attribute reads still gets a slot, because the slot index is the format's place in
/// this list and a list that depended on which attribute swept it would move every other slot when
/// an attribute gained a format.
fn laid_out() -> Vec<AttributeFormat> {
    AttributeFormat::ALL
        .iter()
        .copied()
        .filter(|format| shape(format.name()).is_some())
        .collect()
}

/// Each laid-out format's SDK name, in slot order.
fn names() -> Vec<&'static str> {
    laid_out().into_iter().map(|format| format.name()).collect()
}

/// The slot `format` sits at in a record.
fn slot_of(format: AttributeFormat) -> u32 {
    let at = laid_out()
        .iter()
        .position(|held| *held == format)
        .expect("a format a row names is laid out");
    SLOT * at as u32
}

/// Bytes one element of the source buffer takes.
fn stride() -> u32 {
    coding::stride(&names())
}

/// The source buffer's bytes: element `t` is row `t`'s slot for every laid-out format.
fn records() -> Vec<u8> {
    coding::records(&names())
}

/// Where one declared attribute's words sit in a thread's sixty-four bytes, and how many words
/// they are.
///
/// Thread `t` writes `out[4t] = bits(f)`, `out[4t + 1] = u`,
/// `out[4t + 2] = (bits(i), bits(h.x), bits(h.y), t)` and `out[4t + 3] = bits(v)`, so each
/// attribute's answer is a known slice of the sixteen words and word 11 is the thread's own index.
impl Declared {
    /// The byte offset of the attribute's words inside a thread's row, and the words it holds.
    fn slice(self) -> (u32, u32) {
        match self {
            Declared::Float4 => (0, 4),
            Declared::UInt4 => (16, 4),
            Declared::Int => (32, 1),
            Declared::Half2 => (36, 2),
            Declared::Int4 => (48, 4),
        }
    }

    /// The layout the attribute's own words are compared under: one row of the layout per thread
    /// of the grid, so a diff names the thread and the component inside the fetch.
    fn layout(self) -> ByteLayout {
        let (offset, words) = self.slice();
        grid_layout(offset, words)
    }
}

/// The layout of the last word of each thread's row: the thread's own index, which says the x step
/// fetched per thread rather than once.
fn index_layout() -> ByteLayout {
    grid_layout(44, 1)
}

/// A layout over `words` words at byte `offset` of each of the grid's threads.
///
/// `Gaps::Ignored`, because the five attributes' slices, the thread index and
/// [`tail_layout`] cover every byte of the buffer exactly once between them: a layout that also
/// compared its gaps would compare most of the buffer six times and name the wrong leaf for a
/// difference.
fn grid_layout(offset: u32, words: u32) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: OUTPUT_BYTES as u64,
        },
        element: Element::U32,
        row_offset: offset,
        row_elements: words,
        row_pitch: ROW_BYTES,
        rows: ROWS as u32,
        image_pitch: OUTPUT_BYTES as u32,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The layout of the bytes past the grid: the filler the row started from, which no thread wrote.
///
/// So "the dispatch wrote nothing past its grid" is a fact the comparison checks rather than one
/// nobody looked at.
fn tail_layout() -> ByteLayout {
    let first = ROWS as u32 * ROW_BYTES;
    grid_tail(first, (OUTPUT_BYTES as u32 - first) / 4)
}

/// One row of `words` words at byte `offset` of the bytes read back, whose gaps are compared:
/// every byte of the read-back it does not name is one of them.
///
/// The region stays the whole buffer, because a layout addresses the bytes it is **given** from
/// `row_offset` and the region only says where in the resource they came from. A region offset
/// here would read the tail's words off the front of the buffer, which is data the grid wrote.
fn grid_tail(offset: u32, words: u32) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: OUTPUT_BYTES as u64,
        },
        element: Element::U32,
        row_offset: offset,
        row_elements: words,
        row_pitch: words * 4,
        rows: 1,
        image_pitch: OUTPUT_BYTES as u32,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

// ---- The cases ---------------------------------------------------------------------------------

/// One row: the format each of the five declared attributes carries.
///
/// A row sets **all five**, because the kernel declares all five, and it varies all five, because
/// a row that held four still would be the same experiment as the row beside it: each attribute's
/// own format list is cycled, so attribute 0's format — the longest list, and so unique to the row
/// — names the row while the other four step through their own values beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    /// The format on each attribute, in [`Declared::ALL`] order.
    formats: [AttributeFormat; 5],
}

impl Row {
    /// Row `at` of the family: each attribute's own list at `at` modulo its length.
    fn new(at: usize) -> Row {
        Row {
            formats: Declared::ALL.map(|declared| {
                let held = declared.formats();
                held[at % held.len()]
            }),
        }
    }

    /// The format on `declared`.
    fn format(&self, declared: Declared) -> AttributeFormat {
        self.formats[declared.index() as usize]
    }

    /// The site the row's answer is emitted at: attribute 0's format, which is the SDK enumerator's
    /// own name and unique to the row.
    fn name(&self) -> &'static str {
        self.format(Declared::Float4).name()
    }

    /// The row's stage-input descriptor: each attribute at its own format, reading its own slot of
    /// the one source buffer, whose layout is stepped by the thread's x position at rate 1.
    fn stage_input(&self) -> StageInput {
        let attributes = Declared::ALL
            .into_iter()
            .map(|declared| {
                let format = self.format(declared);
                StageInAttribute {
                    index: declared.index(),
                    format,
                    offset: slot_of(format),
                    buffer_index: SOURCE,
                }
            })
            .collect();
        let layouts = vec![StageInLayout {
            buffer_index: SOURCE,
            stride: Some(stride()),
            step_function: StepFunction::ThreadPositionInGridX,
            step_rate: 1,
        }];
        StageInput::new(attributes, layouts, INDEX_BUFFER)
    }

    /// Everything the row states that is not a buffer's bytes. Its digest is one of the case's
    /// inputs, so a row whose descriptor moved is `Unrecorded` at that row rather than a quiet
    /// pass.
    fn parameters(&self) -> Vec<u8> {
        let held = self.stage_input();
        let mut bytes = ENTRY.as_bytes().to_vec();
        for attribute in &held.attributes {
            bytes.extend(attribute.index.to_le_bytes());
            bytes.extend(attribute.format.as_raw().to_le_bytes());
            bytes.extend(attribute.offset.to_le_bytes());
            bytes.extend(attribute.buffer_index.to_le_bytes());
        }
        for layout in &held.layouts {
            bytes.extend(layout.buffer_index.to_le_bytes());
            bytes.extend(layout.stride.unwrap_or(0).to_le_bytes());
            bytes.extend(layout.step_function.as_raw().to_le_bytes());
            bytes.extend(layout.step_rate.to_le_bytes());
        }
        bytes
    }
}

/// Every row of the family: as many as the longest attribute's own format list, so every value of
/// every attribute is reached and attribute 0's format names the row.
fn rows() -> Vec<Row> {
    let longest = Declared::ALL
        .into_iter()
        .map(|declared| declared.formats().len())
        .max()
        .expect("five attributes");
    (0..longest).map(Row::new).collect()
}

/// The case.
pub struct AttributeFormats;

/// The case the suite holds.
pub fn all() -> Vec<AttributeFormats> {
    vec![AttributeFormats]
}

impl AttributeFormats {
    /// The fixture under the case's entry point.
    pub(super) fn fixture() -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }
}

impl Case for AttributeFormats {
    fn name(&self) -> &'static str {
        "attribute-format-fetches"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value is loosened. Every decoding is exact arithmetic on the codes the record holds
        // and every site is the raw bits the kernel reported, so a word that differs is a
        // difference in the decode and not a rounding.
        //
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // this case, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: ENTRY.to_owned(),
                source_sha256: crate::sha256_hex(Self::fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The records are the whole input table, and the output buffer every row starts from,
            // so the words no thread wrote are part of the answer rather than whatever was there.
            Input::bytes("records", &records()),
            Input::bytes("output", &vec![FILLER; OUTPUT_BYTES]),
        ];
        for row in rows() {
            inputs.push(Input::bytes(
                format!("row:{}", row.name()),
                &row.parameters(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let source = made(sites, "buffer/records", side.buffer(&records()))?;

        for row in rows() {
            let at = |site: &str| format!("{}/{site}", row.name());
            // A pipeline either side refuses is recorded at its own site and the sweep carries on,
            // so one refused row compares as a refusal there while every other row still compares.
            // Its output then stays at the filler on that side, which is the same shape as every
            // other row's: a refusal never arrives as a missing site.
            let pipeline = probe(
                sites,
                &at("pipeline"),
                side.compute_pipeline_with_stage_input(library, ENTRY, &row.stage_input()),
                |_| Value::Absent,
            );
            let output = made(
                sites,
                &at("buffer/output"),
                side.buffer(&vec![FILLER; OUTPUT_BYTES]),
            )?;
            if let Some(pipeline) = pipeline {
                let command_buffer = made(
                    sites,
                    &at("dispatch"),
                    side.dispatch_threads(
                        pipeline,
                        &Bindings::buffers(&[output, source]),
                        &[REGION],
                        [ROWS as u32, 1, 1],
                        [ROWS as u32, 1, 1],
                    ),
                )?;
                call(
                    sites,
                    &at("status"),
                    side.wait(command_buffer),
                    Clone::clone,
                )?;
            }
            let bytes = call(
                sites,
                &at("read-back"),
                side.read_buffer(output, 0, OUTPUT_BYTES),
                |_| Value::Absent,
            )?;
            // One leaf an attribute, each named after the format that attribute carried, so the
            // enumerator Metal was asked for is in the compared text and a diff names the format
            // rather than a byte offset. The five slices and the tail cover every byte of the
            // buffer exactly once, and the tail is where "the dispatch wrote nothing past its
            // grid" is checked.
            let mut node = Node::branch(row.name(), Vec::new());
            for declared in Declared::ALL {
                node = node.with(
                    Node::branch(format!("attribute{}", declared.index()), Vec::new()).with(
                        Node::leaf(
                            row.format(declared).name(),
                            Observed::bytes(bytes.clone(), declared.layout()),
                        ),
                    ),
                );
            }
            node = node.with(Node::leaf(
                "thread-index",
                Observed::bytes(bytes.clone(), index_layout()),
            ));
            node = node.with(Node::leaf(
                "past-the-grid",
                Observed::bytes(bytes, tail_layout()),
            ));
            sites.push(node);
        }
        Ok(())
    }
}

/// The second case: whether a pipeline can be made at all for a format that reaches the `float4`
/// attribute across the numeric classes.
///
/// It creates one pipeline per crossing format and records **nothing else** — no buffer, no
/// dispatch, no read-back. The reason is the shape of the question rather than economy: the
/// decoding of those sixteen formats is already compared, exactly, on the vertex path
/// (`vertex-format-float4`, where Metal's own fixed-function fetch converts each to float), and
/// `attribute_formats::tests::the_stage_in_decode_is_the_vertex_decode` holds Metal's two fetches
/// to one answer. What is not known of the stage-in path is whether the pipeline exists. So the
/// case asks that and no more, and one refused format costs one row of the ledger rather than one
/// per leaf of a dispatch that never ran.
pub struct ClassCrossings;

impl ClassCrossings {
    /// Every crossing format, with the attribute it crosses into.
    fn rows() -> Vec<(Declared, AttributeFormat)> {
        Declared::ALL
            .into_iter()
            .flat_map(|declared| {
                declared
                    .crossings()
                    .into_iter()
                    .map(move |format| (declared, format))
            })
            .collect()
    }

    /// The descriptor for `format` on `declared`: the crossing format on its attribute, and each
    /// other attribute at the first format of its own class, so nothing but the crossing is in
    /// question.
    fn stage_input(declared: Declared, format: AttributeFormat) -> StageInput {
        let attributes = Declared::ALL
            .into_iter()
            .map(|held| {
                let set = if held == declared {
                    format
                } else {
                    held.formats()[0]
                };
                StageInAttribute {
                    index: held.index(),
                    format: set,
                    offset: slot_of(set),
                    buffer_index: SOURCE,
                }
            })
            .collect();
        let layouts = vec![StageInLayout {
            buffer_index: SOURCE,
            stride: Some(stride()),
            step_function: StepFunction::ThreadPositionInGridX,
            step_rate: 1,
        }];
        StageInput::new(attributes, layouts, INDEX_BUFFER)
    }
}

impl Case for ClassCrossings {
    fn name(&self) -> &'static str {
        "attribute-format-class-crossings"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened: whether a pipeline was made is not a number that can round, and
        // the case reads no buffer back, so the notice a card without unified memory announces for
        // a `Shared` buffer cannot reach it. Measured exact on the RTX 5080 Laptop (615.71.09)
        // and on llvmpipe, which is why it declares nothing.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        AttributeFormats.shaders()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = AttributeFormats::fixture();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        for (declared, format) in Self::rows() {
            let held = Self::stage_input(declared, format);
            let mut bytes = Vec::new();
            for attribute in &held.attributes {
                bytes.extend(attribute.index.to_le_bytes());
                bytes.extend(attribute.format.as_raw().to_le_bytes());
                bytes.extend(attribute.offset.to_le_bytes());
            }
            inputs.push(Input::bytes(format!("row:{}", format.name()), &bytes));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = AttributeFormats::fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        for (declared, format) in Self::rows() {
            probe(
                sites,
                format.name(),
                side.compute_pipeline_with_stage_input(
                    library,
                    ENTRY,
                    &Self::stage_input(declared, format),
                ),
                |_| Value::Absent,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// The formats no declared attribute of the fixture reads, which this family leaves out.
    ///
    /// Stated here so that [`the_formats_left_out_are_named`] can hold the swept list and this one
    /// to being the whole SDK table between them. It is a statement of what is missing, not a value
    /// list a sweep reads: every row is derived from `AttributeFormat::ALL` and never from this.
    const LEFT_OUT: [&str; 1] = ["MTLAttributeFormatInvalid"];

    /// Every format the family reaches, over both its cases: the value rows of
    /// `attribute-format-fetches` and the pipeline rows of `attribute-format-class-crossings`.
    fn swept() -> BTreeSet<&'static str> {
        rows()
            .iter()
            .flat_map(|row| row.formats)
            .chain(ClassCrossings::rows().into_iter().map(|(_, format)| format))
            .map(|format| format.name())
            .collect()
    }

    /// The swept formats and [`LEFT_OUT`] are the whole SDK table between them, with nothing in
    /// both.
    ///
    /// The family's depth claim, held by execution: a format that stopped being swept has to join
    /// the named list, and a name on that list that became readable has to leave it.
    #[test]
    fn the_formats_left_out_are_named() {
        let swept = swept();
        let left: BTreeSet<&str> = LEFT_OUT.into_iter().collect();
        assert!(
            swept.is_disjoint(&left),
            "a format is both swept and named as left out: {:?}",
            swept.intersection(&left).collect::<Vec<_>>()
        );
        let whole: BTreeSet<&str> = AttributeFormat::ALL
            .iter()
            .map(|format| format.name())
            .collect();
        assert_eq!(
            swept.union(&left).copied().collect::<BTreeSet<&str>>(),
            whole,
            "the swept formats and the named omissions are not the SDK table"
        );
        assert_eq!(swept.len(), AttributeFormat::ALL.len() - LEFT_OUT.len());
        // The two cases' halves, said apart: the values and the pipelines are different claims.
        let valued: BTreeSet<&str> = rows()
            .iter()
            .flat_map(|row| row.formats)
            .map(|format| format.name())
            .collect();
        let crossed: BTreeSet<&str> = ClassCrossings::rows()
            .into_iter()
            .map(|(_, format)| format.name())
            .collect();
        assert!(
            valued.is_disjoint(&crossed),
            "a format is both a value row and a pipeline row: {:?}",
            valued.intersection(&crossed).collect::<Vec<_>>()
        );
        assert_eq!(
            (valued.len(), crossed.len()),
            (37, 16),
            "the two halves of the field moved: {} valued, {} crossed",
            valued.len(),
            crossed.len()
        );
    }

    /// Every format the SDK table holds either states a shape or is named in [`LEFT_OUT`], every
    /// format reaches an attribute whose declared type admits it, and the cycling reaches **every**
    /// value of **every** attribute.
    ///
    /// A format the SDK adds fails here rather than being dropped from the sweep in silence, and a
    /// row count that stopped covering an attribute's list fails here rather than leaving its last
    /// value unexercised.
    #[test]
    fn every_attribute_format_of_the_sdk_table_is_classified_and_reached() {
        let mut unshaped = Vec::new();
        for &format in AttributeFormat::ALL {
            if shape(format.name()).is_none() && !LEFT_OUT.contains(&format.name()) {
                unshaped.push(format.name());
            }
        }
        assert!(
            unshaped.is_empty(),
            "the SDK table holds {} format(s) this family neither shapes nor names: {}",
            unshaped.len(),
            unshaped.join(", ")
        );
        for declared in Declared::ALL {
            let wanted: BTreeSet<&str> = declared
                .formats()
                .into_iter()
                .map(|format| format.name())
                .collect();
            assert!(
                !wanted.is_empty(),
                "attribute {} reads nothing",
                declared.index()
            );
            let reached: BTreeSet<&str> = rows()
                .iter()
                .map(|row| row.format(declared).name())
                .collect();
            assert_eq!(
                reached,
                wanted,
                "attribute {} does not reach every format it reads",
                declared.index()
            );
            for row in rows() {
                assert!(
                    declared.same_class(row.format(declared)),
                    "{} reached attribute {} and is not of its numeric class",
                    row.format(declared).name(),
                    declared.index()
                );
            }
        }
    }

    /// The rows name distinct sites, digest distinctly, and each sets all four attributes at an
    /// offset aligned to its format's components and wholly inside the element.
    ///
    /// Two rows that fed one digest would let a recording of one stand for the other, and two rows
    /// with one descriptor would be the same experiment twice: attribute 0's list is the longest,
    /// so its format is what makes every row's descriptor its own.
    ///
    /// Metal's own limits on a `MTLStageInputOutputDescriptor` are the vertex descriptor's: at most
    /// 31 attributes, a stride of at most 65535 bytes and a multiple of 4, and an offset aligned to
    /// the format's component width. Metal exposes no device query for any of them, so the numbers
    /// are the API's and this holds the table to them.
    #[test]
    fn every_row_is_its_own_experiment_inside_metals_limits() {
        const MAX_ATTRIBUTES: usize = 31;
        const MAX_STRIDE: u32 = 65535;
        let rows = rows();
        let named: BTreeSet<&str> = rows.iter().map(Row::name).collect();
        assert_eq!(named.len(), rows.len(), "two rows name one site");
        let digests: BTreeSet<String> = rows
            .iter()
            .map(|row| crate::sha256_hex(&row.parameters()))
            .collect();
        assert_eq!(digests.len(), rows.len(), "two rows feed one digest");
        let descriptors: BTreeSet<[u32; 5]> = rows
            .iter()
            .map(|row| row.formats.map(AttributeFormat::as_raw))
            .collect();
        assert_eq!(descriptors.len(), rows.len(), "two rows set one descriptor");
        let stride = stride();
        assert!(
            stride <= MAX_STRIDE && stride.is_multiple_of(4),
            "the records' stride is {stride} bytes"
        );
        for row in &rows {
            let held = row.stage_input();
            assert_eq!(held.attributes.len(), Declared::ALL.len());
            assert!(held.attributes.len() <= MAX_ATTRIBUTES);
            for attribute in &held.attributes {
                let shaped = shape(attribute.format.name()).expect("a set format has a shape");
                assert_eq!(
                    attribute.offset % shaped.coding.width() as u32,
                    0,
                    "{}: offset {} is not aligned to its {}-byte components",
                    attribute.format.name(),
                    attribute.offset,
                    shaped.coding.width()
                );
                assert!(attribute.offset + shaped.size as u32 <= stride);
            }
        }
    }

    /// The four attributes' slices, the thread index and the tail cover every byte of the output
    /// buffer exactly once, and the grid writes less than the buffer holds.
    ///
    /// Without this the leaves could overlap -- comparing some bytes five times and naming the
    /// wrong format for a difference -- or leave a gap nothing compares.
    #[test]
    fn the_leaves_cover_every_byte_of_the_output_exactly_once() {
        let mut covered = vec![0u32; OUTPUT_BYTES];
        let mut mark = |layout: &ByteLayout| {
            let Region::Buffer { offset, .. } = layout.region else {
                panic!("a leaf of this family reads a buffer");
            };
            for row in 0..layout.rows {
                for element in 0..layout.row_elements {
                    let first = offset as usize
                        + (row * layout.row_pitch + layout.row_offset) as usize
                        + (element * layout.element.size()) as usize;
                    for byte in &mut covered[first..first + layout.element.size() as usize] {
                        *byte += 1;
                    }
                }
            }
        };
        for declared in Declared::ALL {
            mark(&declared.layout());
        }
        mark(&index_layout());
        mark(&tail_layout());
        let uncovered = covered.iter().filter(|held| **held == 0).count();
        let twice = covered.iter().filter(|held| **held > 1).count();
        assert_eq!(
            (uncovered, twice),
            (0, 0),
            "{uncovered} byte(s) compared by no leaf and {twice} by more than one"
        );
        assert!(ROWS as u32 * ROW_BYTES < OUTPUT_BYTES as u32);
    }

    /// One row's answer, as the recording holds it: the words each attribute reported, by
    /// attribute.
    fn recorded(row: &Row) -> Option<BTreeMap<u32, Vec<Vec<u32>>>> {
        let recording = crate::recording::read(all()[0].name()).ok()?;
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let Some(node) = outcome.transcript.root.child(row.name()) else {
                    continue;
                };
                let mut out = BTreeMap::new();
                for declared in Declared::ALL {
                    let Some(Value::Bytes(bytes)) = node
                        .child(&format!("attribute{}", declared.index()))
                        .and_then(|held| held.child(row.format(declared).name()))
                        .map(|held| &held.observed.value)
                    else {
                        return None;
                    };
                    let (offset, words) = declared.slice();
                    out.insert(
                        declared.index(),
                        (0..ROWS)
                            .map(|thread| {
                                let first = thread * ROW_BYTES as usize + offset as usize;
                                (0..words as usize)
                                    .map(|word| {
                                        let at = first + 4 * word;
                                        u32::from_le_bytes([
                                            bytes[at],
                                            bytes[at + 1],
                                            bytes[at + 2],
                                            bytes[at + 3],
                                        ])
                                    })
                                    .collect()
                            })
                            .collect(),
                    );
                }
                return Some(out);
            }
        }
        None
    }

    /// An attribute's answer depends on **its own** format and nothing else: two rows that put the
    /// same format on an attribute answered the same words there, although every other attribute's
    /// format moved between them.
    ///
    /// The control that makes a swept leaf mean something. The descriptor sets four attributes and
    /// a row varies all four, so an answer that moved on an attribute whose format did not is a
    /// descriptor leaking a format, an offset or a buffer index from the attribute beside it --
    /// which is what a sweep over one shared descriptor is most likely to get wrong. Attribute 2
    /// carries one format in every row and attributes 1 and 3 repeat theirs many times, so the
    /// check has rows to compare.
    #[test]
    fn an_attributes_answer_depends_on_its_own_format_alone() {
        let mut checked = 0usize;
        let mut differ = Vec::new();
        for declared in Declared::ALL {
            let mut seen: BTreeMap<&str, Vec<Vec<u32>>> = BTreeMap::new();
            for row in rows() {
                let Some(answer) = recorded(&row) else {
                    continue;
                };
                let words = answer[&declared.index()].clone();
                let name = row.format(declared).name();
                match seen.get(name) {
                    None => {
                        seen.insert(name, words);
                    }
                    Some(held) => {
                        checked += 1;
                        if *held != words {
                            differ.push(format!(
                                "{}: attribute {} moved although it still carried {name}",
                                row.name(),
                                declared.index()
                            ));
                        }
                    }
                }
            }
        }
        if checked == 0 {
            println!("no recording of this family in the tree yet");
            return;
        }
        assert!(
            differ.is_empty(),
            "{} reading(s) of {checked} moved on an attribute whose format did not: {}",
            differ.len(),
            differ.join("; ")
        );
    }

    /// Every thread reported its own index, so the x step fetched per thread rather than once.
    ///
    /// The step function's fetched index is in the compared text rather than assumed: a layout
    /// stepped constantly would have every thread answer element 0 and this says it did not.
    #[test]
    fn every_thread_reported_its_own_index() {
        let mut checked = 0usize;
        for row in rows() {
            let Some(answer) = recorded(&row) else {
                continue;
            };
            // The thread index is the last word of the row, which `index_layout` compares; the
            // `half2` slice ends one word before it, so reading it here means reading the leaf's
            // own bytes rather than re-deriving the layout.
            checked += 1;
            let words = &answer[&Declared::Float4.index()];
            assert_eq!(words.len(), ROWS, "{}: {} threads", row.name(), words.len());
            // Attribute 0's own answer must differ between at least two threads, since each thread
            // fetches a different element: an answer equal across the grid would mean the step
            // fetched one element for all of them.
            assert!(
                words.iter().any(|held| *held != words[0]),
                "{}: every thread answered the same words, so the step fetched one element",
                row.name()
            );
        }
        if checked == 0 {
            println!("no recording of this family in the tree yet");
        }
    }

    /// No two rows answered the same words on attribute 0, in Metal's own recording.
    ///
    /// The family's non-vacuity proof: a row whose leaf answered exactly what another's did
    /// compares nothing of its own. Any pair that cannot be told apart has to be named here with
    /// its reason, so it is a statement someone wrote rather than a hole.
    #[test]
    fn no_two_rows_answered_the_same_words() {
        let held: Vec<(Row, Vec<Vec<u32>>)> = rows()
            .into_iter()
            .filter_map(|row| {
                recorded(&row).map(|answer| (row, answer[&Declared::Float4.index()].clone()))
            })
            .collect();
        if held.is_empty() {
            println!("no recording of this family in the tree yet");
            return;
        }
        let mut same = Vec::new();
        for (at, (one, left)) in held.iter().enumerate() {
            for (other, right) in &held[at + 1..] {
                if left == right {
                    same.push(format!("{} and {}", one.name(), other.name()));
                }
            }
        }
        assert!(
            same.is_empty(),
            "{} pair(s) of rows answered identically on attribute 0, so neither compares anything \
             the other does not: {}",
            same.len(),
            same.join("; ")
        );
        assert_eq!(
            held.len(),
            rows().len(),
            "the recording holds {} of {} rows",
            held.len(),
            rows().len()
        );
    }

    /// The stage-in path decodes a format exactly as the vertex path does, where both reach it.
    ///
    /// The two are one field of Metal and one module of ours
    /// (`metal2vulkan::lower::stage_io::fetch`), so a format that decodes one way through a vertex
    /// descriptor and another through a stage-input descriptor is a bug in whichever path is wrong.
    /// Read from the two families' recordings, so it runs where there is no GPU, and compared on
    /// attribute 0, which both families declare as a `float4` over the same twelve code rows.
    #[test]
    fn the_stage_in_decode_is_the_vertex_decode() {
        use crate::cases::vertex_formats;
        let mut compared = 0usize;
        let mut differ = Vec::new();
        for row in rows() {
            let Some(mine) = recorded(&row) else { continue };
            let name = row.format(Declared::Float4).name();
            let Some(theirs) = vertex_formats::recorded_float4_bits(name) else {
                continue;
            };
            compared += 1;
            for (thread, words) in mine[&Declared::Float4.index()].iter().enumerate() {
                if words[..] != theirs[thread][..] {
                    differ.push(format!(
                        "{name} thread {thread}: stage-in {words:08x?}, vertex {:08x?}",
                        theirs[thread]
                    ));
                }
            }
        }
        if compared == 0 {
            println!("no recording of both families in the tree yet");
            return;
        }
        for line in differ.iter().take(20) {
            println!("STAGE-IN-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} reading(s) over {compared} format(s) differ between the two paths",
            differ.len()
        );
    }
}
