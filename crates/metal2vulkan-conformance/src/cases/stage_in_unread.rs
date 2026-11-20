//! A compute stage-input descriptor that states **more attributes than its kernel reads**, and
//! what the attributes the kernel does read answer while the others are swept.
//!
//! `MTLStageInputOutputDescriptor` has 31 attribute slots and 31 buffer layouts, and a kernel's
//! `[[stage_in]]` struct declares whichever of them it wants. Metal's own rules tie the two
//! together only through the layout agreement: every attribute that is **set** (its format not
//! `MTLAttributeFormatInvalid`) needs a layout for the buffer it names, and every layout of a
//! nonzero or dynamic stride needs a set attribute naming its buffer. Nothing says a set attribute
//! has to reach a declaration. So a descriptor may carry attributes the kernel never looks at, and
//! a Metal implementation must fetch the ones it does look at as if the others were not there.
//!
//! **Nothing in the suite asked that before this family.** `attribute-format-fetches` sets exactly
//! the five attributes `stage_in_fetch_wide` declares and varies all five;
//! `attribute-stride-*` and `stage_in_regions` set one. Every case of the stage-input path
//! therefore stated a descriptor whose attributes were exactly the kernel's, which is the one shape
//! in which an implementation cannot tell an attribute it must fetch from one it must ignore.
//!
//! # What a row varies, and what it must not move
//!
//! The **read set** is fixed in every row: the five attributes `stage_in_fetch_wide` declares,
//! each at its own format and offset inside one 64-byte element of `SOURCE`, whose layout steps
//! by the thread's x position at rate 1. Those five, that layout, the grid, the stage-in region
//! and the source bytes are the same in all rows, so a row's answer is a function of the swept
//! part alone.
//!
//! The **swept part** is the attributes past the kernel's five, their buffers' layouts, and the
//! descriptor's index fields. A row states nought to three of them and the layouts they name:
//!
//! | axis | values |
//! |---|---|
//! | format | `UInt`, `Float4`, `UChar4Normalized`, `Half2`, `Int4`, `Short2`, `UInt1010102Normalized`, `Invalid` |
//! | offset in the element | 0, 4, 32, 48 |
//! | buffer | the read attributes' own (`SOURCE`), or one of two of its own (`SPARE_A`, `SPARE_B`) |
//! | its layout's step | constant at rate 0, thread position in grid x at rate 1 and at rate 3, in grid y at rate 1, x indexed, y indexed |
//! | its layout's stride | a fixed 64, a fixed 16, `MTLBufferLayoutStrideDynamic` |
//! | the descriptor's index fields | `MTLIndexTypeUInt16` and `UInt32`, at two index buffer indices |
//!
//! and the rows that cross two axes at once are the ones the single-axis rows cannot ask:
//!
//! - the widest format at the furthest offset its layout's stride admits (`Int4` at 48 of 64);
//! - an unread attribute **inside the buffer the read attributes read**, at an offset none of them
//!   sits at, so an implementation that confused one attribute's offset for another's moves a read
//!   attribute's answer;
//! - two spare layouts of **different steps** at once (a constant step beside a y step), which is
//!   the shape where a fetch that read the wrong layout would still read a valid one;
//! - an indexed step beside a dynamic stride, which are the two layout forms that make a kernel
//!   read something besides its element — the index buffer, and the stride the encoder bound — on
//!   **unread** attributes, where neither may be read at all;
//! - three unread attributes over three different buffers, which is the eight-attribute descriptor
//!   over a five-attribute kernel;
//! - an unset attribute (`Invalid`) beside a set one, and an unset one whose buffer carries a
//!   layout of stride 0, which is the one layout Metal admits with no set attribute naming it;
//! - an unread attribute at a far attribute index (15), since the indices need not be contiguous.
//!
//! # The claim, and why it can catch another implementation's bug
//!
//! Every row must answer the bytes the baseline row answers — the row that states the five
//! attributes and nothing else. The comparison is against Metal's recording field by field, so the
//! family states that **twice over**: each row against Metal, and (in this module's own tests)
//! every row's descriptor against every other's, so the sweep is known to have moved something.
//!
//! An implementation that keys a kernel's fetch code on its whole descriptor rather than on the
//! attributes the kernel reads answers this family correctly and pays for it. One that lets an
//! unread attribute's format, offset or buffer reach a read attribute's fetch fails the row that
//! puts an unread attribute in the read buffer. One that refuses a descriptor whose attributes are
//! not exactly its kernel's fails every row but the baseline, as a refusal. One that reads the
//! index buffer or the bound stride for an unread attribute's layout answers the indexed and
//! dynamic rows differently from the baseline. None of those is visible to a case whose descriptor
//! states exactly what its kernel declares.
//!
//! # Safety
//!
//! One dispatch a row, of `THREADS` threads in one threadgroup — eight, inside the smallest
//! `maxTotalThreadsPerThreadgroup` any venue reports and inside one simdgroup — and no grid is
//! derived from a device limit. The stage-in region is the grid's own size, so no fetch is clamped.
//! Every **read** attribute's bytes lie inside its element and every element inside `SOURCE`
//! (`tests::every_read_fetch_lies_inside_its_buffer`); every attribute a row sets, read or not,
//! lies inside its layout's stride where that stride is fixed, every buffer index a row names past
//! the two the kernel declares is one no run binds, and every attribute and buffer index is inside
//! the 31 slots Metal's descriptor holds (`tests::every_row_is_a_descriptor_metal_admits`). The fixture is loop-free. Nothing here
//! can hang, fault or trip a watchdog.

use crate::allowance::Allowance;
use crate::case::{
    call, made, probe, Abort, Bindings, Case, Failure, Input, MetalSource, Side, StageInAttribute,
    StageInLayout, StageInRegion, StageInput,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{AttributeFormat, IndexType, StepFunction};

/// The fixture the entry point lives in.
const FIXTURE: &str = "kernel_stage_in";

/// The entry point: the five attributes of [`Read::ALL`], each reported as raw bits.
///
/// `stage_in_fetch_wide` and not `stage_in_fetch`, because five declared attributes leave the
/// descriptor's remaining slots free for a sweep that cannot be mistaken for one of the kernel's.
const ENTRY: &str = "stage_in_fetch_wide";

/// The Metal buffer index every **read** attribute's layout reads.
const SOURCE: u32 = 1;

/// Metal buffer indices the rows name and **no run binds**: the index fields' two buffers, and two
/// buffers of their own for an unread attribute's layout.
///
/// Nothing is bound there, and nothing may be: Metal's API validation layer refuses a compute
/// encoder holding a binding the kernel does not use ("unused binding in compute encoder at Buffer
/// index 2"), which stops the recording's validated leg outright. The fetches these indices would
/// feed are all of attributes the kernel never reads, so plain Metal reads none of them and a bound
/// buffer would be a binding nothing uses. `attribute-format-fetches` names its index buffer index
/// the same way and binds nothing there.
const INDEX_A: u32 = 2;
const INDEX_B: u32 = 3;
const SPARE_A: u32 = 4;
const SPARE_B: u32 = 5;

/// Metal buffer indices a run binds: the output at 0 and the source at 1, which the kernel
/// declares and reads. Read by the guard that holds every row to a descriptor Metal admits.
#[cfg(test)]
const BOUND: u32 = 2;

/// The attribute and buffer indices `MTLStageInputOutputDescriptor` holds. Read by the same guard.
#[cfg(test)]
const SLOTS: u32 = 31;

/// Threads in the row's one threadgroup: inside one simdgroup on every Apple GPU, and inside the
/// smallest threadgroup limit any venue reports.
const THREADS: u32 = 8;

/// Bytes one element of `SOURCE` takes, which is its layout's stride: the five read attributes
/// and eight bytes between the `half2` and the `int4` that no fetch of the read set reads.
const ELEMENT: u32 = 64;

/// Elements `SOURCE` holds: more than the grid walks, so no fetch of any stride under test
/// leaves it.
const SOURCE_ELEMENTS: u32 = 64;

/// Words one thread of the fixture writes: four `uint4`s.
const ROW_WORDS: u32 = 16;

/// Bytes one thread writes.
const ROW_BYTES: u32 = 4 * ROW_WORDS;

/// The output buffer's length: the grid's rows and a tail no thread writes, so every row also
/// records that its dispatch wrote nothing past its grid.
const OUTPUT_BYTES: usize = (THREADS * ROW_BYTES) as usize + 64;

/// The byte every output buffer is made holding, so the bytes no dispatch wrote are a value the
/// case put there rather than whatever the allocator left.
const FILLER: u8 = 0x7e;

/// The stage-in region every row sets: the grid's own size, so no thread's element is clamped and
/// what a row answers is its descriptor alone.
const REGION: StageInRegion = StageInRegion::Direct {
    origin: [0, 0, 0],
    size: [THREADS, 1, 1],
};

// ---- The read set ------------------------------------------------------------------------------

/// One of the five `[[stage_in]]` attributes `stage_in_fetch_wide` declares **and reads**, which
/// every row of this family states identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// `float4 f [[attribute(0)]]`.
    Float4,
    /// `uint4 u [[attribute(1)]]`.
    UInt4,
    /// `int i [[attribute(2)]]`.
    Int,
    /// `half2 h [[attribute(3)]]`.
    Half2,
    /// `int4 v [[attribute(4)]]`.
    Int4,
}

impl Read {
    /// The five, in attribute-index order.
    const ALL: [Read; 5] = [
        Read::Float4,
        Read::UInt4,
        Read::Int,
        Read::Half2,
        Read::Int4,
    ];

    /// The Metal attribute index the kernel declares it at.
    fn index(self) -> u32 {
        match self {
            Read::Float4 => 0,
            Read::UInt4 => 1,
            Read::Int => 2,
            Read::Half2 => 3,
            Read::Int4 => 4,
        }
    }

    /// The format the descriptor states for it, of its own numeric class, in every row.
    fn format(self) -> AttributeFormat {
        match self {
            Read::Float4 => AttributeFormat::Float4,
            Read::UInt4 => AttributeFormat::UInt4,
            Read::Int => AttributeFormat::Int,
            Read::Half2 => AttributeFormat::Half2,
            Read::Int4 => AttributeFormat::Int4,
        }
    }

    /// The byte offset of its bytes inside one element of `SOURCE`, in every row.
    fn offset(self) -> u32 {
        match self {
            Read::Float4 => 0,
            Read::UInt4 => 16,
            Read::Int => 32,
            Read::Half2 => 36,
            Read::Int4 => 48,
        }
    }

    /// The byte offset of its words inside a thread's sixty-four bytes of the output, and how many
    /// words it holds.
    ///
    /// Thread `t` writes `out[4t] = bits(f)`, `out[4t + 1] = u`,
    /// `out[4t + 2] = (bits(i), bits(h.x), bits(h.y), t)` and `out[4t + 3] = bits(v)`, so each
    /// attribute's answer is a known slice of the sixteen words and word 11 is the thread's index.
    fn slice(self) -> (u32, u32) {
        match self {
            Read::Float4 => (0, 4),
            Read::UInt4 => (16, 4),
            Read::Int => (32, 1),
            Read::Half2 => (36, 2),
            Read::Int4 => (48, 4),
        }
    }

    /// The layout its own words are compared under: one row of the layout per thread of the grid,
    /// so a diff names the thread and the component inside the fetch.
    fn layout(self) -> ByteLayout {
        let (offset, words) = self.slice();
        grid_layout(offset, words)
    }
}

/// The five read attributes, which every row states.
fn read_attributes() -> Vec<StageInAttribute> {
    Read::ALL
        .into_iter()
        .map(|held| StageInAttribute {
            index: held.index(),
            format: held.format(),
            offset: held.offset(),
            buffer_index: SOURCE,
        })
        .collect()
}

/// The layout the read attributes read, which every row states: one element a thread, stepped by
/// the thread's x position at rate 1.
fn source_layout() -> StageInLayout {
    StageInLayout {
        buffer_index: SOURCE,
        stride: Some(ELEMENT),
        step_function: StepFunction::ThreadPositionInGridX,
        step_rate: 1,
    }
}

// ---- The buffers -------------------------------------------------------------------------------

/// Element `e` of `SOURCE`, as sixteen 32-bit words.
///
/// Each read attribute's bytes are a value of its own that moves with the element, so a fetch that
/// read the wrong element, the wrong offset or the wrong attribute shows in every word of the
/// answer rather than in one. The `float4`'s words are finite floats near 2, and the `half2`'s are
/// finite halves near 1, so no fetch of this family hands a side a NaN whose bits it may canonicalize
/// ([`crate::cases::half_read_nans`] is where that question lives). The eight bytes between the
/// `half2` and the `int4` are the element's own filler, which no read attribute reads.
fn element(e: u32) -> [u32; 16] {
    let mut words = [0u32; 16];
    // `float4 f` at 0: finite floats a little above 2.
    for (k, word) in words[..4].iter_mut().enumerate() {
        *word = 0x4000_0000 + 0x10 * e + k as u32;
    }
    // `uint4 u` at 16.
    for (k, word) in words[4..8].iter_mut().enumerate() {
        *word = 0x0001_0000 + 0x10 * e + k as u32;
    }
    // `int i` at 32: a negative int.
    words[8] = 0xffff_0000 | (0x10 * e + 8);
    // `half2 h` at 36: two finite halves a little above 1, x in the low half-word.
    let x = 0x3c00 + 2 * e;
    let y = 0x3c00 + 2 * e + 1;
    words[9] = x | (y << 16);
    // The element's own filler at 40: eight bytes no read attribute reads.
    words[10] = 0xeeee_eeee;
    words[11] = 0xeeee_eeee;
    // `int4 v` at 48.
    for (k, word) in words[12..].iter_mut().enumerate() {
        *word = 0x1111_0000 + 0x10 * e + k as u32;
    }
    words
}

/// `SOURCE`'s bytes: [`SOURCE_ELEMENTS`] elements of [`ELEMENT`] bytes.
fn source() -> Vec<u8> {
    (0..SOURCE_ELEMENTS)
        .flat_map(|e| element(e).into_iter().flat_map(u32::to_le_bytes))
        .collect()
}

/// The output buffer every row starts from.
fn output() -> Vec<u8> {
    vec![FILLER; OUTPUT_BYTES]
}

/// The buffers a row binds, at Metal buffer index 0 up: the output (buffer 0) and the source
/// (buffer 1), which are the two the kernel declares and the only two any row's fetch reads.
fn buffers() -> Vec<Vec<u8>> {
    vec![output(), source()]
}

// ---- The sweep ---------------------------------------------------------------------------------

/// One attribute a row states past the kernel's five: the descriptor sets it, the kernel declares
/// nothing at its index, and nothing fetches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Unread {
    index: u32,
    format: AttributeFormat,
    offset: u32,
    buffer: u32,
}

/// One buffer layout a row states past the read attributes' own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spare {
    buffer: u32,
    /// `None` is `MTLBufferLayoutStrideDynamic`.
    stride: Option<u32>,
    step: StepFunction,
    rate: u32,
}

/// One row: the swept part of a stage-input descriptor whose read part is fixed.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    /// The site the row's answer is emitted at, unique to the row.
    name: String,
    unread: Vec<Unread>,
    spares: Vec<Spare>,
    index_type: IndexType,
    index_buffer: u32,
}

impl Row {
    /// A row of `name` whose unread attributes are `unread` and whose spare layouts are `spares`,
    /// with the descriptor's index fields at `MTLIndexTypeUInt32` and [`INDEX_A`].
    fn new(name: impl Into<String>, unread: Vec<Unread>, spares: Vec<Spare>) -> Row {
        Row {
            name: name.into(),
            unread,
            spares,
            index_type: IndexType::UInt32,
            index_buffer: INDEX_A,
        }
    }

    /// The row with the descriptor's index fields at `index_type` and `index_buffer`.
    fn indexing(mut self, index_type: IndexType, index_buffer: u32) -> Row {
        self.index_type = index_type;
        self.index_buffer = index_buffer;
        self
    }

    /// The row's whole stage-input descriptor: the five read attributes and the read layout, then
    /// the row's own attributes and layouts, and its index fields.
    fn stage_input(&self) -> StageInput {
        let mut attributes = read_attributes();
        for held in &self.unread {
            attributes.push(StageInAttribute {
                index: held.index,
                format: held.format,
                offset: held.offset,
                buffer_index: held.buffer,
            });
        }
        let mut layouts = vec![source_layout()];
        for held in &self.spares {
            layouts.push(StageInLayout {
                buffer_index: held.buffer,
                stride: held.stride,
                step_function: held.step,
                step_rate: held.rate,
            });
        }
        let mut held = StageInput::new(attributes, layouts, self.index_buffer);
        held.index_type = self.index_type;
        held
    }

    /// Everything the row states that is not a buffer's bytes, as words. Its digest is one of the
    /// case's inputs, so a row whose descriptor moved is `Unrecorded` at that row rather than a
    /// quiet pass.
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
            // `u32::MAX` stands for the dynamic stride, which has no number of its own; no layout
            // of this family states that stride, so the two cannot collide.
            bytes.extend(layout.stride.unwrap_or(u32::MAX).to_le_bytes());
            bytes.extend(layout.step_function.as_raw().to_le_bytes());
            bytes.extend(layout.step_rate.to_le_bytes());
        }
        bytes.extend(held.index_type.as_raw().to_le_bytes());
        bytes.extend(held.index_buffer_index.to_le_bytes());
        bytes
    }
}

/// The formats the sweep puts on an unread attribute: one of each shape class the field has, and
/// `Invalid`, which is the descriptor's value for *no attribute here*.
const FORMATS: [AttributeFormat; 8] = [
    AttributeFormat::UInt,
    AttributeFormat::Float4,
    AttributeFormat::UChar4Normalized,
    AttributeFormat::Half2,
    AttributeFormat::Int4,
    AttributeFormat::Short2,
    AttributeFormat::UInt1010102Normalized,
    AttributeFormat::Invalid,
];

/// The offsets the sweep puts an unread attribute at inside its element: the element's first byte,
/// a byte no read attribute starts at, the byte the read `int` starts at, and the furthest offset
/// a 64-byte stride admits for the format the offset rows carry.
const OFFSETS: [u32; 4] = [0, 4, 32, 48];

/// The steps the sweep puts on a spare layout: every step function a compute layout takes, with
/// the rates Metal admits for each (0 for a constant step, and both 1 and a rate of its own for a
/// thread-position step).
const STEPS: [(&str, StepFunction, u32); 6] = [
    ("constant", StepFunction::Constant, 0),
    ("grid-x", StepFunction::ThreadPositionInGridX, 1),
    ("grid-x-rate-3", StepFunction::ThreadPositionInGridX, 3),
    ("grid-y", StepFunction::ThreadPositionInGridY, 1),
    (
        "grid-x-indexed",
        StepFunction::ThreadPositionInGridXIndexed,
        1,
    ),
    (
        "grid-y-indexed",
        StepFunction::ThreadPositionInGridYIndexed,
        1,
    ),
];

/// The strides the sweep puts on a spare layout: the read layout's own, a smaller fixed one, and
/// `MTLBufferLayoutStrideDynamic`.
const STRIDES: [(&str, Option<u32>); 3] = [
    ("fixed-64", Some(ELEMENT)),
    ("fixed-16", Some(16)),
    ("dynamic", None),
];

/// A spare layout at `SPARE_A` of a fixed 64-byte stride stepped by the thread's x position at
/// rate 1: the layout the single-axis rows state where the axis they sweep is not the layout's.
fn plain_spare(buffer: u32) -> Spare {
    Spare {
        buffer,
        stride: Some(ELEMENT),
        step: StepFunction::ThreadPositionInGridX,
        rate: 1,
    }
}

/// One unread attribute at index 5.
fn at5(format: AttributeFormat, offset: u32, buffer: u32) -> Unread {
    Unread {
        index: 5,
        format,
        offset,
        buffer,
    }
}

/// Every row of the family.
///
/// The first is the baseline — the five read attributes and nothing else — and every other row's
/// answer must be its answer. Each single-axis group carries a format of its own, so no two rows
/// of the family state the same descriptor
/// (`tests::no_two_rows_state_the_same_descriptor`).
fn rows() -> Vec<Row> {
    let mut rows = vec![Row::new("no-unread-attributes", Vec::new(), Vec::new())];
    // The format axis, at offset 0 of a spare layout. `Invalid` is an unset attribute, so it names
    // the read attributes' own buffer and states no spare layout: a spare layout of a nonzero
    // stride that no set attribute names is a descriptor Metal stops at.
    for format in FORMATS {
        let (unread, spares) = match format {
            AttributeFormat::Invalid => (at5(format, 0, SOURCE), Vec::new()),
            _ => (at5(format, 0, SPARE_A), vec![plain_spare(SPARE_A)]),
        };
        rows.push(Row::new(
            format!("format-{}", slug(format.name())),
            vec![unread],
            spares,
        ));
    }
    // The offset axis, over a format no other group carries (eight bytes, so 48 of a 64-byte
    // stride is inside it).
    for offset in OFFSETS {
        rows.push(Row::new(
            format!("offset-{offset}"),
            vec![at5(AttributeFormat::UShort4, offset, SPARE_A)],
            vec![plain_spare(SPARE_A)],
        ));
    }
    // The buffer axis: the read attributes' own buffer, and one of its own either way.
    for (name, buffer) in [
        ("source", SOURCE),
        ("spare-a", SPARE_A),
        ("spare-b", SPARE_B),
    ] {
        let attribute = at5(AttributeFormat::Short2, 4, buffer);
        let spares = match buffer {
            SOURCE => Vec::new(),
            held => vec![plain_spare(held)],
        };
        rows.push(Row::new(format!("buffer-{name}"), vec![attribute], spares));
    }
    // The step axis, on a spare layout of the read layout's own stride.
    for (name, step, rate) in STEPS {
        rows.push(Row::new(
            format!("step-{name}"),
            vec![at5(AttributeFormat::UShort2, 0, SPARE_A)],
            vec![Spare {
                buffer: SPARE_A,
                stride: Some(ELEMENT),
                step,
                rate,
            }],
        ));
    }
    // The stride axis, on a spare layout stepped by the thread's x position.
    for (name, stride) in STRIDES {
        rows.push(Row::new(
            format!("stride-{name}"),
            vec![at5(AttributeFormat::Char4, 0, SPARE_A)],
            vec![Spare {
                buffer: SPARE_A,
                stride,
                step: StepFunction::ThreadPositionInGridX,
                rate: 1,
            }],
        ));
    }
    // The index fields, which only an indexed step reads at all: both index types, and an index
    // buffer index of its own.
    for (name, index_type, index_buffer) in [
        ("index-type-uint16", IndexType::UInt16, INDEX_A),
        ("index-type-uint32", IndexType::UInt32, INDEX_A),
        ("index-buffer-3", IndexType::UInt32, INDEX_B),
    ] {
        rows.push(
            Row::new(
                name,
                vec![at5(AttributeFormat::UInt, 0, SPARE_A)],
                vec![Spare {
                    buffer: SPARE_A,
                    stride: Some(ELEMENT),
                    step: StepFunction::ThreadPositionInGridXIndexed,
                    rate: 1,
                }],
            )
            .indexing(index_type, index_buffer),
        );
    }
    // The crossings: what the single-axis rows cannot ask.
    //
    // The widest format at the furthest offset its stride admits.
    rows.push(Row::new(
        "widest-format-at-the-furthest-offset",
        vec![at5(AttributeFormat::Int4, 48, SPARE_A)],
        vec![plain_spare(SPARE_A)],
    ));
    // An unread attribute inside the read attributes' own buffer, at a byte none of them starts
    // at: an implementation that took one attribute's offset for another's moves a read answer.
    rows.push(Row::new(
        "inside-the-read-buffer",
        vec![at5(AttributeFormat::Half2, 4, SOURCE)],
        Vec::new(),
    ));
    // Two spare layouts of different steps at once.
    rows.push(Row::new(
        "a-constant-step-beside-a-y-step",
        vec![
            at5(AttributeFormat::UInt, 0, SPARE_A),
            Unread {
                index: 6,
                format: AttributeFormat::UInt4,
                offset: 16,
                buffer: SPARE_B,
            },
        ],
        vec![
            Spare {
                buffer: SPARE_A,
                stride: Some(ELEMENT),
                step: StepFunction::Constant,
                rate: 0,
            },
            Spare {
                buffer: SPARE_B,
                stride: Some(ELEMENT),
                step: StepFunction::ThreadPositionInGridY,
                rate: 1,
            },
        ],
    ));
    // The two layout forms that make a kernel read something besides its element, on attributes
    // nothing reads: an indexed step, and a dynamic stride.
    rows.push(Row::new(
        "an-indexed-step-beside-a-dynamic-stride",
        vec![
            at5(AttributeFormat::UInt, 0, SPARE_A),
            Unread {
                index: 6,
                format: AttributeFormat::UInt,
                offset: 0,
                buffer: SPARE_B,
            },
        ],
        vec![
            Spare {
                buffer: SPARE_A,
                stride: Some(ELEMENT),
                step: StepFunction::ThreadPositionInGridXIndexed,
                rate: 1,
            },
            Spare {
                buffer: SPARE_B,
                stride: None,
                step: StepFunction::ThreadPositionInGridX,
                rate: 1,
            },
        ],
    ));
    // Three unread attributes over three buffers: the eight-attribute descriptor over a
    // five-attribute kernel.
    rows.push(Row::new(
        "three-unread-over-three-buffers",
        vec![
            at5(AttributeFormat::Short2, 4, SOURCE),
            Unread {
                index: 6,
                format: AttributeFormat::Int4,
                offset: 0,
                buffer: SPARE_A,
            },
            Unread {
                index: 7,
                format: AttributeFormat::UChar4Normalized,
                offset: 12,
                buffer: SPARE_B,
            },
        ],
        vec![
            Spare {
                buffer: SPARE_A,
                stride: Some(ELEMENT),
                step: StepFunction::ThreadPositionInGridX,
                rate: 3,
            },
            Spare {
                buffer: SPARE_B,
                stride: Some(16),
                step: StepFunction::Constant,
                rate: 0,
            },
        ],
    ));
    // An unset attribute beside a set one.
    rows.push(Row::new(
        "an-unset-attribute-beside-a-set-one",
        vec![
            at5(AttributeFormat::Invalid, 0, SOURCE),
            Unread {
                index: 6,
                format: AttributeFormat::UInt,
                offset: 0,
                buffer: SPARE_A,
            },
        ],
        vec![plain_spare(SPARE_A)],
    ));
    // An unset attribute whose buffer carries a layout of stride 0, the one layout Metal admits
    // with no set attribute naming its buffer.
    rows.push(Row::new(
        "an-unset-attribute-on-a-zero-stride-layout",
        vec![at5(AttributeFormat::Invalid, 0, SPARE_A)],
        vec![Spare {
            buffer: SPARE_A,
            stride: Some(0),
            step: StepFunction::ThreadPositionInGridX,
            rate: 1,
        }],
    ));
    // A far attribute index: the indices need not be contiguous with the kernel's.
    rows.push(Row::new(
        "unread-at-attribute-15",
        vec![Unread {
            index: 15,
            format: AttributeFormat::UInt,
            offset: 0,
            buffer: SPARE_A,
        }],
        vec![plain_spare(SPARE_A)],
    ));
    rows
}

/// An SDK enumerator's name as a site name: its tail, in lower case with words parted by dashes.
fn slug(name: &str) -> String {
    let tail = name
        .strip_prefix("MTLAttributeFormat")
        .unwrap_or(name)
        .to_owned();
    let mut out = String::new();
    for (at, c) in tail.char_indices() {
        if c.is_ascii_uppercase() && at > 0 {
            out.push('-');
        }
        out.extend(c.to_lowercase());
    }
    out
}

// ---- The layouts the answer is compared under --------------------------------------------------

/// A layout over `words` words at byte `offset` of each of the grid's threads.
///
/// `Gaps::Ignored`, because the five attributes' slices, the thread index and [`tail_layout`]
/// cover every byte of the buffer exactly once between them: a layout that also compared its gaps
/// would compare most of the buffer seven times and name the wrong leaf for a difference.
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
        rows: THREADS,
        image_pitch: OUTPUT_BYTES as u32,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The layout of the last word of each thread's row: the thread's own index, which says the x step
/// fetched per thread rather than once.
fn index_layout() -> ByteLayout {
    grid_layout(44, 1)
}

/// The layout of the bytes past the grid: the filler the row started from, which no thread wrote.
fn tail_layout() -> ByteLayout {
    let first = THREADS * ROW_BYTES;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: OUTPUT_BYTES as u64,
        },
        element: Element::U32,
        row_offset: first,
        row_elements: (OUTPUT_BYTES as u32 - first) / 4,
        row_pitch: OUTPUT_BYTES as u32 - first,
        rows: 1,
        image_pitch: OUTPUT_BYTES as u32,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

// ---- The case ----------------------------------------------------------------------------------

/// The case.
pub struct StageInUnread;

/// The case the suite holds.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![Box::new(StageInUnread)]
}

impl StageInUnread {
    /// The fixture under the case's entry point.
    fn fixture() -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }
}

impl Case for StageInUnread {
    fn name(&self) -> &'static str {
        "stage-in-unread-attributes"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing here is loosened: every byte of every row compares exactly, the guard past the
        // grid included.
        //
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone, so every value of this family still compares against Metal's exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Self::fixture();
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: ENTRY.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
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
        ];
        for (at, bytes) in buffers().into_iter().enumerate() {
            inputs.push(Input::bytes(format!("buffer:{at}"), &bytes));
        }
        for row in rows() {
            inputs.push(Input::bytes(format!("row:{}", row.name), &row.parameters()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        // The output is made again for each row; the source is made once and bound by every row,
        // so what a row answers cannot depend on which row ran before it.
        let source = made(sites, "buffer/source", side.buffer(&source()))?;

        for row in rows() {
            let at = |site: &str| format!("{}/{site}", row.name);
            // A pipeline either side refuses is recorded at its own site and the sweep carries on,
            // so one refused row compares as a refusal there while every other row still compares.
            // The row stops at its refusal: the sites after it are the other side's alone, which
            // the comparison folds under the refusal (`compare`'s walk), so a refused row is one
            // refusal and not a divergence for each leaf of a dispatch that never ran.
            let Some(pipeline) = probe(
                sites,
                &at("pipeline"),
                side.compute_pipeline_with_stage_input(library, ENTRY, &row.stage_input()),
                |_| Value::Absent,
            ) else {
                continue;
            };
            let out = made(sites, &at("buffer/out"), side.buffer(&output()))?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::buffers(&[out, source]),
                    &[REGION],
                    [THREADS, 1, 1],
                    [THREADS, 1, 1],
                ),
            )?;
            call(
                sites,
                &at("status"),
                side.wait(command_buffer),
                Clone::clone,
            )?;
            let bytes = call(
                sites,
                &at("read-back"),
                side.read_buffer(out, 0, OUTPUT_BYTES),
                |_| Value::Absent,
            )?;
            // One leaf a read attribute, named after what the kernel declares it as, so a diff
            // names the attribute rather than a byte offset. The five slices, the thread index and
            // the tail cover every byte of the read-back exactly once.
            let mut node = Node::branch(row.name.clone(), Vec::new());
            for read in Read::ALL {
                node = node.with(Node::leaf(
                    format!("attribute{}", read.index()),
                    Observed::bytes(bytes.clone(), read.layout()),
                ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute_coding::shape;
    use std::collections::BTreeSet;

    /// The bytes `SOURCE` holds, for the guard that bounds every read fetch.
    const SOURCE_BYTES: usize = (SOURCE_ELEMENTS * ELEMENT) as usize;

    /// The bytes a format occupies in one element, read from the SDK enumerator's own name.
    fn size_of(format: AttributeFormat) -> Option<usize> {
        shape(format.name()).map(|held| held.size)
    }

    /// Every row's name is its own, so no two sites of the report stand for two rows.
    #[test]
    fn every_row_names_itself() {
        let rows = rows();
        let names: BTreeSet<&str> = rows.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(names.len(), rows.len(), "{names:?}");
    }

    /// No two rows state the same stage-input descriptor.
    ///
    /// This is the control that makes the family's claim non-vacuous. Every row must answer the
    /// baseline's bytes, so a row that accidentally stated the baseline's descriptor — or another
    /// row's — would pass while comparing nothing, and the sweep's depth would be a count of
    /// duplicates. The descriptors are compared as the words a row fingerprints, which is what the
    /// recording keys on.
    #[test]
    fn no_two_rows_state_the_same_descriptor() {
        let rows = rows();
        let mut seen: Vec<(String, Vec<u8>)> = Vec::new();
        for row in &rows {
            let words = row.parameters();
            if let Some((other, _)) = seen.iter().find(|(_, held)| *held == words) {
                panic!(
                    "rows {other} and {} state the same descriptor, so one of them compares \
                     nothing",
                    row.name
                );
            }
            seen.push((row.name.clone(), words));
        }
        assert_eq!(seen.len(), rows.len());
    }

    /// Every row states the five read attributes, identically, and the layout they read.
    ///
    /// The family's claim is that what the kernel reads does not move, so the read part of every
    /// descriptor has to be the same part: a row that moved a read attribute would be asking the
    /// format sweep's question instead.
    #[test]
    fn every_row_states_the_same_read_set() {
        for row in rows() {
            let held = row.stage_input();
            let read: Vec<StageInAttribute> = held
                .attributes
                .iter()
                .copied()
                .filter(|a| Read::ALL.iter().any(|r| r.index() == a.index))
                .collect();
            assert_eq!(read, read_attributes(), "{}", row.name);
            assert!(
                held.layouts.contains(&source_layout()),
                "{}: the read attributes' layout is not the one every row states",
                row.name
            );
            // And nothing the row adds is at an index the kernel declares.
            for unread in &row.unread {
                assert!(
                    unread.index > 4,
                    "{}: attribute {} is one the kernel declares",
                    row.name,
                    unread.index
                );
            }
        }
    }

    /// Every read attribute's fetch lies inside `SOURCE`, over the whole grid.
    ///
    /// This is the case for the module's safety claim, computed from the read set rather than
    /// asserted beside it.
    #[test]
    fn every_read_fetch_lies_inside_its_buffer() {
        for read in Read::ALL {
            let size = size_of(read.format()).expect("the SDK name states a shape");
            let past = read.offset() as usize + size;
            assert!(
                past <= ELEMENT as usize,
                "{read:?} reads {past} bytes of a {ELEMENT}-byte element"
            );
            let furthest = (THREADS - 1) as usize * ELEMENT as usize + past;
            assert!(
                furthest <= SOURCE_BYTES,
                "{read:?} reads to byte {furthest} of a buffer of {SOURCE_BYTES}"
            );
        }
    }

    /// Every row states a descriptor Metal admits, by Metal's own two rules and the rules its
    /// pipeline creation stops at.
    ///
    /// Computed from the rows, so a row added without a layout for a buffer it names, with a
    /// vertex step function, with a constant step at a nonzero rate, with a thread-position step
    /// at a rate of 0, with an attribute past its layout's fixed stride, or naming a buffer index
    /// no run binds, fails here rather than on a Mac. The last of those is why the test is in this
    /// module and not a reading of Metal's: an assertion inside Metal's pipeline creation ends the
    /// recording's process, and this family's rows are exactly the descriptors nobody has stated
    /// before.
    #[test]
    fn every_row_is_a_descriptor_metal_admits() {
        assert_eq!(buffers().len() as u32, BOUND);
        for row in rows() {
            let held = row.stage_input();
            let set: Vec<&StageInAttribute> = held
                .attributes
                .iter()
                .filter(|a| a.format != AttributeFormat::Invalid)
                .collect();
            // Every set attribute's buffer has a layout.
            for attribute in &set {
                assert!(
                    held.layouts
                        .iter()
                        .any(|l| l.buffer_index == attribute.buffer_index),
                    "{}: attribute {} names buffer {} and no layout states it",
                    row.name,
                    attribute.index,
                    attribute.buffer_index
                );
            }
            for layout in &held.layouts {
                // Every layout of a nonzero or dynamic stride is named by a set attribute.
                if layout.stride != Some(0) {
                    assert!(
                        set.iter().any(|a| a.buffer_index == layout.buffer_index),
                        "{}: layout {} has stride {:?} and no set attribute names it",
                        row.name,
                        layout.buffer_index,
                        layout.stride
                    );
                }
                // No compute layout takes a vertex stage's step function.
                assert!(
                    !matches!(
                        layout.step_function,
                        StepFunction::PerVertex
                            | StepFunction::PerInstance
                            | StepFunction::PerPatch
                            | StepFunction::PerPatchControlPoint
                    ),
                    "{}: layout {} steps by {:?}",
                    row.name,
                    layout.buffer_index,
                    layout.step_function
                );
                // A constant step takes a rate of 0 and every other step a rate of its own.
                match layout.step_function {
                    StepFunction::Constant => assert_eq!(
                        layout.step_rate, 0,
                        "{}: a constant step at rate {}",
                        row.name, layout.step_rate
                    ),
                    _ => assert!(
                        layout.step_rate > 0,
                        "{}: layout {} steps at rate 0",
                        row.name,
                        layout.buffer_index
                    ),
                }
                // Every attribute fits inside a fixed stride, so no row asks Metal's validation
                // about an attribute that reaches past its element.
                if let Some(stride) = layout.stride.filter(|&s| s > 0) {
                    for attribute in set.iter().filter(|a| a.buffer_index == layout.buffer_index) {
                        let size =
                            size_of(attribute.format).expect("the SDK name states a shape") as u32;
                        assert!(
                            attribute.offset + size <= stride,
                            "{}: attribute {} reads {} bytes of a {stride}-byte element",
                            row.name,
                            attribute.index,
                            attribute.offset + size
                        );
                    }
                }
                // A spare layout's buffer index is one no run binds, so no row leaves the
                // compute encoder holding a binding the kernel does not use; the read layout's is
                // the source the kernel declares.
                assert!(
                    layout.buffer_index == SOURCE || layout.buffer_index >= BOUND,
                    "{}: layout {} is a bound buffer index the kernel does not read",
                    row.name,
                    layout.buffer_index
                );
                assert!(
                    layout.buffer_index < SLOTS,
                    "{}: layout {} is past Metal's {SLOTS} buffer layouts",
                    row.name,
                    layout.buffer_index
                );
            }
            assert!(
                (BOUND..SLOTS).contains(&held.index_buffer_index),
                "{}: the index buffer index {} is either bound or past Metal's {SLOTS} buffers",
                row.name,
                held.index_buffer_index
            );
            for attribute in &held.attributes {
                assert!(
                    attribute.index < SLOTS,
                    "{}: attribute {} is past Metal's {SLOTS} attribute slots",
                    row.name,
                    attribute.index
                );
            }
        }
    }

    /// The sweep reaches every value of every axis the module's table names, and the crossings
    /// state at least two unread attributes or two spare layouts.
    ///
    /// Without this a value could be dropped from an axis and the module's prose would go on
    /// claiming it was compared.
    #[test]
    fn the_sweep_reaches_every_value_of_every_axis() {
        let rows = rows();
        let unread: Vec<Unread> = rows.iter().flat_map(|row| row.unread.clone()).collect();
        let spares: Vec<Spare> = rows.iter().flat_map(|row| row.spares.clone()).collect();
        for format in FORMATS {
            assert!(
                unread.iter().any(|held| held.format == format),
                "no row puts {format:?} on an unread attribute"
            );
        }
        for offset in OFFSETS {
            assert!(
                unread.iter().any(|held| held.offset == offset),
                "no row puts an unread attribute at offset {offset}"
            );
        }
        for buffer in [SOURCE, SPARE_A, SPARE_B] {
            assert!(
                unread.iter().any(|held| held.buffer == buffer),
                "no row puts an unread attribute in buffer {buffer}"
            );
        }
        for (name, step, rate) in STEPS {
            assert!(
                spares
                    .iter()
                    .any(|held| held.step == step && held.rate == rate),
                "no row states a spare layout stepping {name}"
            );
        }
        for (name, stride) in STRIDES {
            assert!(
                spares.iter().any(|held| held.stride == stride),
                "no row states a spare layout of stride {name}"
            );
        }
        for index_type in [IndexType::UInt16, IndexType::UInt32] {
            assert!(
                rows.iter().any(|row| row.index_type == index_type
                    && row.spares.iter().any(|held| matches!(
                        held.step,
                        StepFunction::ThreadPositionInGridXIndexed
                            | StepFunction::ThreadPositionInGridYIndexed
                    ))),
                "no row states {index_type:?} beside an indexed step"
            );
        }
        for index_buffer in [INDEX_A, INDEX_B] {
            assert!(
                rows.iter().any(|row| row.index_buffer == index_buffer),
                "no row names index buffer {index_buffer}"
            );
        }
        // And the crossings: rows that state more than one of the swept items at once, which is
        // what a single-axis sweep cannot reach.
        let crossed = rows
            .iter()
            .filter(|row| row.unread.len() > 1 || row.spares.len() > 1)
            .count();
        assert!(
            crossed >= 4,
            "the family states {crossed} rows that cross two axes at once"
        );
    }

    /// The baseline row states the read set alone, and it is the first row, so the report reads it
    /// before the rows that must answer it.
    #[test]
    fn the_baseline_states_the_read_set_alone() {
        let rows = rows();
        let first = rows.first().expect("a row");
        assert_eq!(first.name, "no-unread-attributes");
        assert!(first.unread.is_empty() && first.spares.is_empty());
        assert_eq!(first.stage_input().attributes, read_attributes());
        assert_eq!(first.stage_input().layouts, vec![source_layout()]);
    }

    /// Each element of `SOURCE` holds a value of its own for every read attribute, and the
    /// element's own filler nowhere a read attribute reads.
    ///
    /// So a row's answer names the element its fetch walked, and a fetch that read the gap between
    /// the `half2` and the `int4` is unmistakable.
    #[test]
    fn each_element_holds_a_value_of_its_own() {
        let bytes = source();
        assert_eq!(bytes.len(), SOURCE_BYTES);
        let mut seen: BTreeSet<[u32; 16]> = BTreeSet::new();
        for e in 0..SOURCE_ELEMENTS {
            let held = element(e);
            assert!(seen.insert(held), "element {e} repeats another's words");
            // The filler sits at bytes 40..48 alone, which no read attribute's slice covers.
            for (at, word) in held.iter().enumerate() {
                let filler = *word == 0xeeee_eeee;
                assert_eq!(
                    filler,
                    at == 10 || at == 11,
                    "element {e} word {at} is {word:#x}"
                );
            }
            // And the half words are finite: an exponent short of the all-ones one.
            for half in [held[9] & 0xffff, held[9] >> 16] {
                assert_ne!(
                    half & 0x7c00,
                    0x7c00,
                    "element {e} holds a half of {half:#x}"
                );
            }
            // As are the float words.
            for word in &held[..4] {
                assert_ne!(
                    word & 0x7f80_0000,
                    0x7f80_0000,
                    "element {e} holds a float of {word:#x}"
                );
            }
        }
    }
}
