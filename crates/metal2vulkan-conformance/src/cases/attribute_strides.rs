//! The compute encoder's four `attributeStride` setters, swept over the strides Metal treats
//! apart.
//!
//! `MTLComputeCommandEncoder` binds a buffer five ways. Four of them carry an `attributeStride`
//! — `setBuffer:offset:attributeStride:atIndex:`, `setBuffers:offsets:attributeStrides:withRange:`,
//! `setBufferOffset:attributeStride:atIndex:` and `setBytes:length:attributeStride:atIndex:` — and
//! **only a `[[stage_in]]` layout of `MTLBufferLayoutStrideDynamic` reads that argument**. Against
//! a layout of a fixed stride the four are the plain setters with one more number that reaches
//! nothing, so every case here binds a pipeline whose stage-input descriptor states a dynamic
//! layout, and the stride is then the distance between the elements a thread's fetch walks.
//!
//! # The sweep, and why these values
//!
//! The stride is a 64-bit number, so the sweep is over the boundaries the behaviour is documented
//! at rather than over a range: 0, a stride of its own, the named constant that sets none, and a
//! stride past 32 bits, which Metal takes **modulo `2^32`**. `STRIDES` holds nine values across
//! those four behaviours, and what Metal in fact answered for each (an Apple M3 Max under macOS
//! 27.0.1, `attribute-stride-buffer`) is this:
//!
//! | stride | what each thread fetched |
//! |---|---|
//! | `0` | element 0, every thread: `0x110` eight times |
//! | `1` | one byte a thread, so a thread's four bytes straddle two of the buffer's words: `0x110`, `0x11000100`, `0x00110100`, … |
//! | `4` | the attribute's own size, element `t` for thread `t`: `0x110`, `0x111`, `0x112`, … |
//! | `8` | two words, element `2t`: `0x110`, `0x112`, `0x114`, … |
//! | `MTLAttributeStrideStatic` (`u64::MAX`) | **element 0 for thread 0, and zero for every thread past it**: `0x110`, then `0x100` seven times |
//! | `2^32` | as `0`, by the truncation |
//! | `2^32 + 8` | as `8`, by the truncation |
//! | `2^33 + 4` | as `4`, by the truncation |
//! | `2^63 + 8` | as `8`, by the truncation |
//!
//! **The constant's row is measured, not predicted, and it is not what the runtime's prose said.**
//! A dynamic layout bound with `MTLAttributeStrideStatic` was expected to read at stride 0, the way
//! a buffer bound by the plain setters does. Metal does something else: it fetches the element at
//! the bind offset for the first thread and hands every later thread zeros. The runtime documents
//! exactly that for the **inline-bytes** form (`set_bytes_with_stride`, measured on an M4) and
//! stride 0 for the buffer forms; this oracle answers the inline-bytes behaviour for all four
//! setters. The recordings are the claim, and a reader who trusted the buffer forms' note would
//! have written the wrong expectation.
//!
//! **Which values are indistinguishable by construction, and why.** Nine values gave **five**
//! distinct answers, so four values duplicate another's and the groups are named rather than left
//! to be noticed: `{0, 2^32}`, `{4, 2^33 + 4}` and `{8, 2^32 + 8, 2^63 + 8}`. Each group is one
//! stride reached two or three ways, and that is the claim rather than a sweep comparing the same
//! bytes twice for nothing: an implementation that **saturated** a stride past 32 bits instead of
//! truncating it parts every one of those groups at once, and an implementation that clamped only
//! at `2^32` keeps `{0, 2^32}` together while splitting `2^63 + 8` off `8`. `1` and
//! `MTLAttributeStrideStatic` share their answer with nothing.
//!
//! All four setters tell the constant from `0`. `BUFFER_OFFSET`'s row asks the one thing the
//! other three cannot: `setBufferOffset:attributeStride:atIndex:` reaches an index that **already
//! holds a stride**, so the row binds its buffer at `PRIOR_STRIDE` first and a later call
//! passing the constant has a stride to clear. Metal's answer for its last slot — the constant at
//! a bind offset of 8 — is `0x112` and then zeros, which says two things at once: the prior stride
//! of 8 is gone, and the bind offset the same call carried did reach the fetch.
//!
//! # How a case is read
//!
//! One output buffer, one slot a stride, `THREADS` words a slot, and every byte compared
//! including the guard past the last slot. The fixture's `stage_in_region_word` writes
//! `attribute + 0x100` at the thread's position, and the source buffer holds `0x10 + e` at element
//! `e`, so a slot's words name the elements its stride walked:
//! stride 4 reads `0x110, 0x111, …` and stride 8 reads `0x110, 0x112, …`.
//!
//! # Safety
//!
//! Every dispatch is one threadgroup of `THREADS` threads — eight, inside the smallest
//! `maxTotalThreadsPerThreadgroup` any venue reports and inside one simdgroup — and no grid is
//! derived from a device limit. The stage-in region is the grid's own size, so no fetch is
//! clamped. **Every fetch of every stride lies inside its buffer**: the largest stride in effect
//! is 8 bytes, the furthest thread is `THREADS` − 1, and the largest offset a row binds is
//! `FURTHEST_OFFSET`, so the furthest byte read is well under `SOURCE_BYTES`
//! (`every_fetch_lies_inside_its_buffer`). The fixture is loop-free. Nothing here can hang, fault
//! or trip a watchdog.

use crate::allowance::Allowance;
use crate::case::{
    made, Abort, Case, Failure, Input, MetalSource, Side, StageInAttribute, StageInLayout,
    StageInput,
};
use crate::compute_pass::{ComputePass, ComputeStep, Dispatching, ATTRIBUTE_STRIDE_STATIC};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites};
use metal2vulkan::metal::{AttributeFormat, StepFunction};

/// The fixture whose `stage_in_region_word` writes its one `[[stage_in]]` attribute plus `0x100`
/// at the thread's position in the grid.
const FIXTURE: &str = "stage_in_regions";
const ENTRY: &str = "stage_in_region_word";

/// The byte every output buffer is made holding, so the bytes no dispatch wrote are a value the
/// case put there rather than whatever the allocator left.
const GUARD: u8 = 0x7e;

/// Threads in each dispatch's one threadgroup: inside one simdgroup on every Apple GPU, and inside
/// the smallest threadgroup limit any venue reports.
const THREADS: u32 = 8;

/// Bytes one slot of an output buffer holds: one 32-bit word a thread.
const SLOT: usize = THREADS as usize * 4;

/// Bytes of guard past the last slot, so a dispatch that wrote more words than the case asked for
/// shows.
const TAIL: usize = 64;

/// The Metal buffer index the output is bound at, which the fixture declares.
const OUT_INDEX: u32 = 0;

/// The Metal buffer index the swept attribute's layout reads, and the index every stride setter
/// under test is aimed at.
const SOURCE_INDEX: u32 = 1;

/// The Metal buffer index an indexed step would read its indices from. No row steps by index, so
/// nothing is bound there; the descriptor names it because Metal's descriptor has the field.
const INDEX_INDEX: u32 = 3;

/// Elements the source buffer holds, as 32-bit words.
const SOURCE_ELEMENTS: u32 = 128;

/// Bytes the source buffer holds. Read by the guard that bounds every row's fetch.
#[cfg(test)]
const SOURCE_BYTES: usize = SOURCE_ELEMENTS as usize * 4;

/// The largest bind offset any row states, which the fetch's furthest byte is measured from.
const FURTHEST_OFFSET: u64 = 8;

/// The strides every row sweeps, in the order its slots hold them: the four documented behaviours
/// and the values that reach each of them (the module's own table).
const STRIDES: [u64; 9] = [
    0,
    1,
    4,
    8,
    ATTRIBUTE_STRIDE_STATIC,
    1 << 32,
    (1 << 32) + 8,
    (1 << 33) + 4,
    (1 << 63) + 8,
];

/// What a bind of `stride` in fact fetched, as the recordings show it.
///
/// This is the module's own reading of **measured** behaviour, used by the guard that bounds every
/// row's fetch and by the guard that counts the sweep's distinct answers. **No case compares
/// against it**: a case's answer is Metal's recorded bytes, so this is a test's item and not a
/// product path.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Fetched {
    /// Element `t` at `t · stride` from the bind offset, for every thread.
    Stride(u32),
    /// `MTLAttributeStrideStatic`: the element at the bind offset for thread 0, and zero for every
    /// thread past it.
    FirstThenZeros,
}

/// What `stride` fetched: the constant's own behaviour, or the stride's low 32 bits.
#[cfg(test)]
const fn fetched(stride: u64) -> Fetched {
    if stride == ATTRIBUTE_STRIDE_STATIC {
        Fetched::FirstThenZeros
    } else {
        // Truncation to 32 bits is the measured answer.
        Fetched::Stride(stride as u32)
    }
}

/// The furthest byte past the bind offset a bind of `stride` reads, over [`THREADS`] threads.
///
/// The constant reads the first element alone, so its reach is the attribute's own four bytes; a
/// stride reads the last thread's element.
#[cfg(test)]
const fn reach(stride: u64) -> u64 {
    match fetched(stride) {
        Fetched::FirstThenZeros => 4,
        Fetched::Stride(by) => (THREADS - 1) as u64 * by as u64 + 4,
    }
}

/// The source buffer's bytes: `0x10 + e` as a 32-bit word at element `e`, so a slot's words name
/// the elements its stride walked and a fetch that read the wrong element shows in every word.
fn source() -> Vec<u8> {
    (0..SOURCE_ELEMENTS)
        .flat_map(|element| (0x10 + element).to_le_bytes())
        .collect()
}

/// An output buffer of `slots` slots, with [`TAIL`] of guard past the last.
fn output(slots: usize) -> Vec<u8> {
    vec![GUARD; slots * SLOT + TAIL]
}

/// A buffer's bytes as a site: every byte compared, guard included, as `Element::U32` so a diff
/// names the word that moved.
fn words_site(name: &str, bytes: Vec<u8>) -> Node {
    let length = bytes.len() as u64;
    Node::leaf(
        name,
        Observed::bytes(bytes, ByteLayout::buffer(0, length, Element::U32)),
    )
}

/// The stage-input descriptor every case binds: one `UInt` attribute at offset 0 of
/// [`SOURCE_INDEX`], whose layout's stride is `MTLBufferLayoutStrideDynamic` — the one stride that
/// makes an `attributeStride` reach anything — stepping by the thread's x position at rate 1.
fn stage_input() -> StageInput {
    StageInput::new(
        vec![StageInAttribute {
            index: 0,
            format: AttributeFormat::UInt,
            offset: 0,
            buffer_index: SOURCE_INDEX,
        }],
        vec![StageInLayout {
            buffer_index: SOURCE_INDEX,
            // `None` is `MTLBufferLayoutStrideDynamic`.
            stride: None,
            step_function: StepFunction::ThreadPositionInGridX,
            step_rate: 1,
        }],
        INDEX_INDEX,
    )
}

/// The descriptor's own words, so a recording stops being the answer to a program whose
/// stage input moved.
fn stage_input_words() -> Vec<u8> {
    let input = stage_input();
    let mut words = Vec::new();
    for attribute in &input.attributes {
        words.extend(attribute.index.to_le_bytes());
        words.extend(attribute.format.as_raw().to_le_bytes());
        words.extend(attribute.offset.to_le_bytes());
        words.extend(attribute.buffer_index.to_le_bytes());
    }
    for layout in &input.layouts {
        words.extend(layout.buffer_index.to_le_bytes());
        // `u32::MAX` stands for the dynamic stride here, which is `None` and has no number of its
        // own; no layout states that stride, so the two cannot collide.
        words.extend(layout.stride.unwrap_or(u32::MAX).to_le_bytes());
        words.extend(layout.step_function.as_raw().to_le_bytes());
        words.extend(layout.step_rate.to_le_bytes());
    }
    words.extend(input.index_type.as_raw().to_le_bytes());
    words.extend(input.index_buffer_index.to_le_bytes());
    words
}

/// The region every dispatch fetches from: the grid's own size, so no thread's element is clamped
/// and what a slot's words hold is the stride alone.
fn region() -> ComputeStep {
    ComputeStep::StageInRegion {
        origin: [0, 0, 0],
        size: [THREADS, 1, 1],
    }
}

/// One dispatch of [`THREADS`] threads in one threadgroup.
fn dispatch() -> ComputeStep {
    ComputeStep::DispatchThreadgroups {
        groups: [1, 1, 1],
        threads: [THREADS, 1, 1],
    }
}

/// The output bound so the next dispatch writes into slot `slot`.
fn out_slot(buffer: crate::case::Handle, slot: usize) -> ComputeStep {
    ComputeStep::Buffer {
        buffer,
        offset: (slot * SLOT) as u64,
        index: OUT_INDEX,
    }
}

/// One bind of a row's sweep: the bind offset the call states, and the stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bind {
    offset: u64,
    stride: u64,
}

/// A case of the family: one of the four setters, with the binds its slots hold.
struct Row {
    /// The case's name: its recording's file name, and its row in the report.
    case: &'static str,
    /// The site the row's output buffer is emitted at.
    site: &'static str,
    /// Which setter the row sweeps.
    setter: Setter,
    /// The binds the row sweeps, one slot each.
    binds: fn() -> Vec<Bind>,
}

/// Which of the four `attributeStride` setters a row sweeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Setter {
    /// `setBuffer:offset:attributeStride:atIndex:`.
    Buffer,
    /// `setBuffers:offsets:attributeStrides:withRange:`, over a range of one.
    PluralBuffers,
    /// `setBufferOffset:attributeStride:atIndex:`, after a bind that left a stride of its own.
    BufferOffset,
    /// `setBytes:length:attributeStride:atIndex:`, with no buffer of the case's own behind it.
    Bytes,
}

impl Setter {
    /// The call that binds `bind` at [`SOURCE_INDEX`], out of the row's source buffer.
    fn bind(self, source: crate::case::Handle, bind: Bind) -> ComputeStep {
        match self {
            Setter::Buffer => ComputeStep::BufferWithStride {
                buffer: source,
                offset: bind.offset,
                stride: bind.stride,
                index: SOURCE_INDEX,
            },
            Setter::PluralBuffers => ComputeStep::BuffersWithStrides {
                buffers: vec![(source, bind.offset, bind.stride)],
                first: SOURCE_INDEX,
            },
            Setter::BufferOffset => ComputeStep::BufferOffsetWithStride {
                offset: bind.offset,
                stride: bind.stride,
                index: SOURCE_INDEX,
            },
            Setter::Bytes => ComputeStep::BytesWithStride {
                // The bind offset is the row's own, applied to the bytes the call copies: a slot
                // of a `setBytes` row at offset `n` hands the source's bytes from `n`.
                bytes: source_from(bind.offset),
                stride: bind.stride,
                index: SOURCE_INDEX,
            },
        }
    }

    /// The call every row makes **before** its sweep, where the setter needs one: the buffer-offset
    /// form reaches an index that already holds a buffer, and the row binds that buffer at a stride
    /// of its own ([`PRIOR_STRIDE`]) so a later call that cleared the stride reads apart from one
    /// that set it.
    fn prior(self, source: crate::case::Handle) -> Option<ComputeStep> {
        match self {
            Setter::BufferOffset => Some(ComputeStep::BufferWithStride {
                buffer: source,
                offset: 0,
                stride: PRIOR_STRIDE,
                index: SOURCE_INDEX,
            }),
            // The three other setters state the whole binding themselves, so a prior one would
            // only hide what they set.
            Setter::Buffer | Setter::PluralBuffers | Setter::Bytes => None,
        }
    }
}

/// The stride [`Setter::BufferOffset`]'s row binds before its sweep, so
/// `MTLAttributeStrideStatic` has a stride to clear. Two words, which no slot of that row's sweep
/// reads by accident: the sweep's own 8 reads the same elements, and that is the point — the two
/// slots that differ are the constant's and that one's.
const PRIOR_STRIDE: u64 = 8;

/// The source buffer's bytes from `offset`, which a `setBytes` row hands the call.
fn source_from(offset: u64) -> Vec<u8> {
    source().split_off(offset as usize)
}

/// The nine strides at bind offset 0: every row's sweep.
fn swept() -> Vec<Bind> {
    STRIDES
        .iter()
        .map(|&stride| Bind { offset: 0, stride })
        .collect()
}

/// The sweep, and then two binds at a nonzero offset: the buffer-offset form takes an offset
/// beside its stride, and a row that swept only the stride would never say whether the offset
/// still reached the fetch.
fn swept_and_moved() -> Vec<Bind> {
    let mut binds = swept();
    binds.push(Bind {
        offset: 4,
        stride: 4,
    });
    binds.push(Bind {
        offset: FURTHEST_OFFSET,
        stride: ATTRIBUTE_STRIDE_STATIC,
    });
    binds
}

/// `setBuffer:offset:attributeStride:atIndex:` over the sweep.
static BUFFER: Row = Row {
    case: "attribute-stride-buffer",
    site: "nine-strides",
    setter: Setter::Buffer,
    binds: swept,
};

/// `setBuffers:offsets:attributeStrides:withRange:` over the same sweep, as a range of one.
static PLURAL_BUFFERS: Row = Row {
    case: "attribute-stride-plural-buffers",
    site: "nine-strides",
    setter: Setter::PluralBuffers,
    binds: swept,
};

/// `setBufferOffset:attributeStride:atIndex:` over the sweep and two moved binds, after a bind
/// that left a stride of its own.
static BUFFER_OFFSET: Row = Row {
    case: "attribute-stride-buffer-offset",
    site: "nine-strides-and-two-offsets",
    setter: Setter::BufferOffset,
    binds: swept_and_moved,
};

/// `setBytes:length:attributeStride:atIndex:` over the sweep.
static BYTES: Row = Row {
    case: "attribute-stride-inline-bytes",
    site: "nine-strides",
    setter: Setter::Bytes,
    binds: swept,
};

/// The rows the suite holds.
static ROWS: [&Row; 4] = [&BUFFER, &PLURAL_BUFFERS, &BUFFER_OFFSET, &BYTES];

/// A case of the family.
pub struct AttributeStrides {
    row: &'static Row,
}

impl AttributeStrides {
    /// The fixture under its one entry point.
    fn fixture() -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }

    /// The row's pass: the pipeline, the region, then a bind and a dispatch into a slot apiece.
    fn pass(
        &self,
        pipeline: crate::case::Handle,
        held: &[crate::case::Handle],
    ) -> Vec<ComputeStep> {
        let out = held.first().copied().unwrap_or(crate::case::Handle(0));
        let source = held.get(1).copied().unwrap_or(crate::case::Handle(0));
        let mut steps = vec![ComputeStep::Pipeline(pipeline), region()];
        if let Some(prior) = self.row.setter.prior(source) {
            steps.push(prior);
        }
        for (slot, bind) in (self.row.binds)().into_iter().enumerate() {
            steps.push(out_slot(out, slot));
            steps.push(self.row.setter.bind(source, bind));
            steps.push(dispatch());
        }
        steps
    }

    /// The buffers the row makes, in the order its pass names them: the output, then the source.
    ///
    /// A `setBytes` row makes the source buffer too and never binds it: the bytes it hands the
    /// call come out of the same `source()`, and making the buffer keeps one shape for the case's
    /// inputs and its read-back.
    fn buffers(&self) -> Vec<Vec<u8>> {
        vec![output((self.row.binds)().len()), source()]
    }
}

impl Case for AttributeStrides {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing here is loosened: every byte of every slot compares exactly.
        //
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
            Input::bytes("stage-input", &stage_input_words()),
        ];
        for (at, bytes) in self.buffers().into_iter().enumerate() {
            inputs.push(Input::bytes(format!("buffer:{at}"), &bytes));
        }
        // The pass itself: every call, in order, with its own numbers, so a recording cannot stay
        // the answer to a program whose strides moved.
        let steps = self.pass(crate::case::Handle(0), &[]);
        inputs.push(Input::bytes(
            "pass",
            &ComputePass {
                dispatching: Dispatching::Serial,
                steps: &steps,
            }
            .words(),
        ));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Self::fixture();
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline_with_stage_input(library, ENTRY, &stage_input()),
        )?;
        let mut held = Vec::new();
        for (at, bytes) in self.buffers().into_iter().enumerate() {
            held.push(made(sites, &format!("buffer/{at}"), side.buffer(&bytes))?);
        }
        let steps = self.pass(pipeline, &held);
        let outcome = made(
            sites,
            "pass",
            side.compute_pass(&ComputePass {
                dispatching: Dispatching::Serial,
                steps: &steps,
            }),
        )?;
        sites.push(outcome.node("outcome", &steps));
        // A refused command stops the case here, after every command's own site is emitted: the
        // bytes a refused dispatch did not write would otherwise read as divergences of their own.
        if let Some((at, failure)) = outcome.first_refusal() {
            return Err(Abort {
                at: format!("outcome/step/{at}/{}", steps[at].call()),
                failure: failure.clone(),
            });
        }
        let read = made(
            sites,
            "read-back",
            side.read_buffer(held[0], 0, output((self.row.binds)().len()).len()),
        )?;
        sites.push(words_site(self.row.site, read));
        Ok(())
    }
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    ROWS.iter()
        .map(|row| Box::new(AttributeStrides { row }) as Box<dyn Case>)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every row's name is its own, so no two recordings share a file and no site of the report
    /// stands for two cases.
    #[test]
    fn every_row_names_itself() {
        let names: BTreeSet<&str> = ROWS.iter().map(|row| row.case).collect();
        assert_eq!(names.len(), ROWS.len(), "{names:?}");
    }

    /// Every fetch of every row lies inside the source buffer, by what the bind in fact fetches,
    /// the furthest thread and the bind's own offset.
    ///
    /// This is the case for the module's safety claim, computed from the rows rather than asserted
    /// beside them: a row that swept a stride whose fetch left its buffer would fail here instead
    /// of asking a GPU what lies past it.
    #[test]
    fn every_fetch_lies_inside_its_buffer() {
        for row in ROWS {
            for bind in (row.binds)() {
                let past = bind.offset + reach(bind.stride);
                assert!(
                    past <= SOURCE_BYTES as u64,
                    "{}: a bind at offset {} and stride {} reads {past} bytes of a buffer of {}",
                    row.case,
                    bind.offset,
                    bind.stride,
                    SOURCE_BYTES
                );
                assert!(
                    bind.offset <= FURTHEST_OFFSET,
                    "{}: a bind at offset {} is past the offset the safety note states",
                    row.case,
                    bind.offset
                );
            }
        }
    }

    /// The sweep holds every one of the four documented behaviours, and the module's reading of
    /// what Metal answered collapses onto exactly the five distinct answers its table names.
    ///
    /// Without this a value could be dropped from [`STRIDES`] and the module's prose would go on
    /// claiming the behaviour was compared.
    #[test]
    fn the_sweep_covers_the_four_documented_behaviours() {
        assert!(STRIDES.contains(&0), "the sweep states no stride of 0");
        assert!(
            STRIDES.contains(&ATTRIBUTE_STRIDE_STATIC),
            "the sweep states no MTLAttributeStrideStatic"
        );
        assert!(
            STRIDES.iter().any(|&stride| stride > 0
                && stride != ATTRIBUTE_STRIDE_STATIC
                && fetched(stride) == Fetched::Stride(stride as u32)
                && stride <= u64::from(u32::MAX)),
            "the sweep states no stride of its own inside 32 bits"
        );
        assert!(
            STRIDES
                .iter()
                .any(|&stride| stride > u64::from(u32::MAX) && stride != ATTRIBUTE_STRIDE_STATIC),
            "the sweep states no stride past 32 bits"
        );
        let answers: BTreeSet<Fetched> = STRIDES.iter().copied().map(fetched).collect();
        assert_eq!(
            answers,
            BTreeSet::from([
                Fetched::Stride(0),
                Fetched::Stride(1),
                Fetched::Stride(4),
                Fetched::Stride(8),
                Fetched::FirstThenZeros,
            ]),
            "the sweep's answers are not the five the module's table names"
        );
    }

    /// Four of the sweep's nine values share an answer with another value, which is what the
    /// module's three named groups are. The count is asserted so a value added to [`STRIDES`]
    /// without a line of the table fails here.
    #[test]
    fn the_sweep_states_which_values_are_indistinguishable() {
        let answers: BTreeSet<Fetched> = STRIDES.iter().copied().map(fetched).collect();
        assert_eq!(answers.len(), 5, "{answers:?}");
        assert_eq!(
            STRIDES.len() - answers.len(),
            4,
            "the sweep's nine values give {} distinct answers, so the module's groups are stale",
            answers.len()
        );
        // The constant and a stride of one are each alone: the module says so, and a change that
        // made either share an answer would make that sentence false.
        for stride in [ATTRIBUTE_STRIDE_STATIC, 1] {
            let shared = STRIDES
                .iter()
                .filter(|&&other| fetched(other) == fetched(stride))
                .count();
            assert_eq!(shared, 1, "{stride} no longer answers alone");
        }
    }

    /// Only the buffer-offset row binds a prior stride, and that prior stride is one the
    /// constant's own answer cannot be mistaken for.
    ///
    /// The row's point is that `MTLAttributeStrideStatic` **clears** a stride that was there, so
    /// the stride it clears has to be one whose answer differs from the constant's.
    #[test]
    fn only_the_buffer_offset_row_binds_a_prior_stride() {
        let with_prior: Vec<&str> = ROWS
            .iter()
            .filter(|row| row.setter.prior(crate::case::Handle(0)).is_some())
            .map(|row| row.case)
            .collect();
        assert_eq!(with_prior, [BUFFER_OFFSET.case]);
        assert_ne!(
            fetched(PRIOR_STRIDE),
            fetched(ATTRIBUTE_STRIDE_STATIC),
            "the prior stride answers as the constant does, so clearing it would read alike"
        );
        // And the row ends on the constant at a non-zero offset, which is what says the offset
        // reached the fetch beside the cleared stride.
        let last = (BUFFER_OFFSET.binds)()
            .pop()
            .expect("the buffer-offset row sweeps at least one bind");
        assert_eq!(last.stride, ATTRIBUTE_STRIDE_STATIC);
        assert!(last.offset > 0);
    }

    /// Every row's pass makes one dispatch a slot and binds the output at that slot's own offset,
    /// so a slot's words are the bind the row swept and no other.
    #[test]
    fn each_slot_holds_one_dispatch_of_its_own_bind() {
        for row in ROWS {
            let case = AttributeStrides { row };
            let steps = case.pass(crate::case::Handle(0), &[]);
            let dispatches = steps.iter().filter(|step| step.dispatches()).count();
            let binds = (row.binds)().len();
            assert_eq!(dispatches, binds, "{}", row.case);
            let offsets: Vec<u64> = steps
                .iter()
                .filter_map(|step| match step {
                    ComputeStep::Buffer { offset, index, .. } if *index == OUT_INDEX => {
                        Some(*offset)
                    }
                    _ => None,
                })
                .collect();
            let wanted: Vec<u64> = (0..binds).map(|slot| (slot * SLOT) as u64).collect();
            assert_eq!(offsets, wanted, "{}", row.case);
        }
    }

    /// Every row's dispatch is one threadgroup of [`THREADS`] threads, which is the module's
    /// safety bound, and the stage-in region is the grid's own size so nothing is clamped.
    #[test]
    fn every_dispatch_is_one_small_threadgroup_over_the_whole_region() {
        for row in ROWS {
            let case = AttributeStrides { row };
            for step in case.pass(crate::case::Handle(0), &[]) {
                match step {
                    ComputeStep::DispatchThreadgroups { groups, threads } => {
                        assert_eq!(groups, [1, 1, 1], "{}", row.case);
                        assert_eq!(threads, [THREADS, 1, 1], "{}", row.case);
                    }
                    ComputeStep::StageInRegion { origin, size } => {
                        assert_eq!(origin, [0, 0, 0], "{}", row.case);
                        assert_eq!(size, [THREADS, 1, 1], "{}", row.case);
                    }
                    _ => {}
                }
            }
        }
    }

    /// The source buffer's words are the sequence the module's prose names, so a slot's words can
    /// be read as the elements its stride walked.
    #[test]
    fn the_source_holds_the_words_the_module_names() {
        let bytes = source();
        assert_eq!(bytes.len(), SOURCE_BYTES);
        for element in 0..SOURCE_ELEMENTS {
            let at = element as usize * 4;
            let word = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
            assert_eq!(word, 0x10 + element);
        }
    }
}
