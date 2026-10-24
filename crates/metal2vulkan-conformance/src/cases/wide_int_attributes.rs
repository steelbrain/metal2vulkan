//! Every `MTLVertexFormat` only an **integer-declared** attribute can read, decoded by the vertex
//! descriptor.
//!
//! `vertex_formats` sweeps the 46 formats a `float4` or `half4` attribute can read. The eight it
//! cannot are the 32-bit integer ones — `MTLVertexFormatInt`, `Int2`, `Int3`, `Int4`,
//! `MTLVertexFormatUInt`, `UInt2`, `UInt3`, `UInt4` — because a 32-bit integer format reaches the
//! shader as the integer it holds rather than converted to float, so it needs an `int4`- or
//! `uint4`-declared attribute. No committed fixture declared one, which is why those eight were the
//! whole of `MTLVertexFormat`'s remaining depth. `dsx_wide_int_attributes` declares both, and these
//! two cases are the eight.
//!
//! The value list is **not written here.** It is read from `VertexFormat::ALL`, which
//! `scripts/regen-metal-enums/` generates from the SDK's own headers, filtered by the shape
//! [`crate::attribute_coding`] parses out of each enumerator's name: a format whose coding is a
//! 4-byte integer is one of these, and a format the SDK adds to that coding joins the sweep with no
//! edit here. The codes come from the same module as the float sweep's, so the two families put one
//! value table to Metal through two declarations rather than two tables through two.
//!
//! # Why the answer is unconverted bits
//!
//! The point of an integer attribute is that nothing converts. `MTLVertexFormatInt`'s code
//! `0x7FFF_FFFF` is 2147483647, which the widest `float` cannot hold exactly — a float-declared
//! attribute would answer 2147483648.0 and the sweep would be comparing a rounding rather than a
//! fetch. So the fixture writes `in.a` straight to `out[vertex_id]` as the `int4`/`uint4` it
//! declared, and the read-back reads it as [`Element::I32`] or [`Element::U32`]: every bit of the
//! fetch is in the compared text.
//!
//! # What each format must answer, and why no two are the same words
//!
//! Each format's component `c` takes the code at index `(row + 3 c) % 12` of its signedness's own
//! twelve-code list, so one fetch's components differ from each other and a decoder that crossed
//! two of them is caught by the values. The components the format does **not** hold come back as
//! Metal's own `(0, 0, 0, 1)` fill, so the four formats of one signedness differ from each other in
//! the trailing components as well as in how many codes they read: `MTLVertexFormatInt` answers
//! `(code, 0, 0, 1)` and `MTLVertexFormatInt4` four codes, with no fill at all.
//! `tests::no_two_formats_answer_the_same_words` holds that, so no row of either sweep compares
//! what the row beside it already compared.
//!
//! Signedness is the other half. The signed and unsigned code lists are different lists, and the
//! same 32 bits read as a signed and an unsigned component are different numbers above
//! `0x7FFF_FFFF`: `tests::the_signed_and_unsigned_sweeps_ask_different_questions` holds that the
//! two cases' twelve-code lists differ, so the `uint4` case is not the `int4` case spelled twice.
//!
//! # The twelve rows
//!
//! [`ROWS`] is the shared code count: 0, 1, the minimum code and the code above it, the maximum
//! code and the code below it, 2, the code below the maximum, 1000 and -1000, and two alternating
//! bit patterns. For a 32-bit integer format those are exactly the bounded-numeric edges — zero,
//! one, `INT_MIN`, `INT_MAX`, `UINT_MAX` and either side of each — which is the depth a 32-bit
//! integer fetch has to get right and a narrower one cannot be asked for.
//!
//! # What is held still, and why
//!
//! One step function, `MTLVertexStepFunctionPerVertex`. The step is the vertex descriptor's
//! **index** arithmetic and the format is its **decode**; they are separate fields of
//! `MTLVertexBufferLayoutDescriptor` and `MTLVertexAttributeDescriptor`, and the step's own sweep is
//! `vertex_formats`' (`vertex-format-constant-step`, `vertex-format-instance-step`) over the 46
//! formats it can declare. Sweeping it again here would multiply these eight by a field this
//! family's question has nothing to do with, and would compare the same decode under a second
//! index arithmetic rather than compare anything new.
//!
//! # The shape, and why nothing here can fault a GPU
//!
//! A `vertex void` function has no position to rasterize, so each pipeline disables rasterization
//! and carries no fragment stage, and each pass has no attachment of any kind and states its own
//! one-by-one extent. That is the one shape Metal accepts for such a function. Every format's draw
//! is eight — [`ROWS`] — points into its own 192-byte slice of one read-back buffer, so the case
//! costs one commit and one wait; every fetch is inside the record the case bound; no texture, no
//! loop, no rasterization.

use crate::allowance::Allowance;
use crate::attribute_coding as coding;
use crate::attribute_coding::{shape, FILLER, ROWS, SLOT};
use crate::case::{call, made, probe, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, VertexAttribute,
    VertexBufferLayout, VertexLayout,
};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PrimitiveType, VertexFormat};

/// The fixture both entry points live in.
const FIXTURE: &str = "dsx_wide_int_attributes";

/// Bytes one row of a case's answer takes: one four-component word.
const ROW_BYTES: u32 = 16;

/// Bytes one format's answer takes, which is its slice of the read-back buffer.
const ANSWER_BYTES: u32 = ROWS as u32 * ROW_BYTES;

/// Whether `format`'s coding is a 32-bit integer of the signedness `signed`, which is what makes it
/// one of this family's and not `vertex_formats`'.
fn is_wide_integer(format: VertexFormat, signed: bool) -> bool {
    shape(format.name())
        .is_some_and(|held| held.coding == coding::Coding::Integer { bytes: 4, signed })
}

/// Every format of the SDK table an integer-declared attribute reads, in the table's own
/// declaration order: the slot index a record lays a format out at is its place in this list.
///
/// Both signednesses, because one records buffer serves both cases and a list that held only one
/// case's formats would move the other's slots.
fn laid_out() -> Vec<VertexFormat> {
    VertexFormat::ALL
        .iter()
        .copied()
        .filter(|format| is_wide_integer(*format, true) || is_wide_integer(*format, false))
        .collect()
}

/// Each laid-out format's SDK name, in slot order.
fn names() -> Vec<&'static str> {
    laid_out().into_iter().map(|format| format.name()).collect()
}

/// The slot `format` sits at in a record.
fn slot_of(format: VertexFormat) -> u32 {
    let at = laid_out()
        .iter()
        .position(|held| *held == format)
        .expect("a format a sweep names is laid out");
    SLOT * at as u32
}

/// Bytes one record takes: one [`SLOT`] per laid-out format.
fn stride() -> u32 {
    coding::stride(&names())
}

/// The records the vertex buffer holds: row `r`'s slot for each laid-out format at byte `SLOT i`,
/// and [`FILLER`] everywhere else, so a fetch that read past its own components read a value no
/// code list holds.
fn records() -> Vec<u8> {
    coding::records(&names())
}

/// One case of the family: the declaration its entry point fetches into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Declared {
    /// `int4 a [[attribute(0)]]`, which reads the signed 32-bit formats.
    Int4,
    /// `uint4 a [[attribute(0)]]`, which reads the unsigned ones.
    UInt4,
}

impl Declared {
    /// The entry point whose `[[stage_in]]` holds the declaration.
    fn entry(self) -> &'static str {
        match self {
            Declared::Int4 => "wide_int_attribute_to_int4",
            Declared::UInt4 => "wide_uint_attribute_to_uint4",
        }
    }

    /// Whether the declaration reads a signed format.
    fn signed(self) -> bool {
        self == Declared::Int4
    }

    /// How the read-back reads one component: as the signed or unsigned 32-bit integer the
    /// declaration fetched it as, never as a float.
    fn element(self) -> Element {
        match self {
            Declared::Int4 => Element::I32,
            Declared::UInt4 => Element::U32,
        }
    }

    /// Every format of the SDK table this declaration reads, in the table's own declaration order.
    fn formats(self) -> Vec<VertexFormat> {
        VertexFormat::ALL
            .iter()
            .copied()
            .filter(|format| is_wide_integer(*format, self.signed()))
            .collect()
    }

    /// The layout one format's answer is read under: one row per row of the records, one element
    /// per component, so a diff names the row and the component inside it.
    fn answer_layout(self) -> ByteLayout {
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: u64::from(ANSWER_BYTES),
            },
            element: self.element(),
            row_offset: 0,
            row_elements: 4,
            row_pitch: ROW_BYTES,
            rows: ROWS as u32,
            image_pitch: ANSWER_BYTES,
            images: 1,
            gaps: Gaps::Compared,
        }
    }
}

/// The site one format's answer is emitted at: the SDK enumerator's **own name**.
///
/// Not a prettier spelling of it. The suite's depth in a Metal enum is derived by
/// `scripts/conformance-coverage/option_space.py`, one numerator of which is the enumerators a
/// committed recording's compared transcript holds by name, and a site named `int4` holds none:
/// this family states its values through `VertexFormat::ALL` rather than variant by variant, so the
/// transcript is the only place the SDK's name can appear.
fn site_name(format: VertexFormat) -> &'static str {
    format.name()
}

/// One case of the family.
pub struct WideIntAttributes {
    name: &'static str,
    declared: Declared,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<WideIntAttributes> {
    vec![
        WideIntAttributes {
            name: "vertex-format-int4",
            declared: Declared::Int4,
        },
        WideIntAttributes {
            name: "vertex-format-uint4",
            declared: Declared::UInt4,
        },
    ]
}

impl WideIntAttributes {
    /// The fixture under the case's entry point.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: self.declared.entry(),
        }
    }

    /// Attribute 0 of `format`: that format at its own slot of buffer 1.
    fn attributes(&self, format: VertexFormat) -> [VertexAttribute; 1] {
        [VertexAttribute::new(0, format, slot_of(format), 1)]
    }

    /// Buffer 1's layout: one record of [`stride`] bytes, stepped once per vertex.
    fn layouts(&self) -> [VertexBufferLayout; 1] {
        [VertexBufferLayout::per_vertex(1, stride())]
    }
}

impl Case for WideIntAttributes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened, and nothing here could be: an integer attribute converts nothing, so
        // every component the fetch answers is the code the record holds, bit for bit. A tolerance
        // would hide the only bug this sweep exists to find.
        //
        // One notice is declared, and it is the venue's and not this sweep's: the buffer each draw
        // reads its fetched rows back through is `Shared`, which a device with no unified memory
        // keeps in system memory because that is the only memory both sides can reach.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: self.declared.entry().to_owned(),
                source_sha256: crate::sha256_hex(self.fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let formats = self.declared.formats();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The records are the whole input table: every code every row fetches, and the filler
            // that says a fetch read past its own components.
            Input::bytes("records", &records()),
            // The buffer every answer starts from, so a format whose pipeline was not made has its
            // slice read as the filler rather than as whatever was there.
            Input::bytes(
                "answers",
                &vec![FILLER; formats.len() * ANSWER_BYTES as usize],
            ),
        ];
        // One fingerprint over every pipeline, which are generated from the format list and the
        // declaration: a format added or a slot moved moves the whole table.
        let mut pipelines = Vec::new();
        for format in &formats {
            pipelines.extend(site_name(*format).as_bytes());
            let attributes = self.attributes(*format);
            let layouts = self.layouts();
            pipelines.extend(
                RenderPipelineSpec::converting(
                    StageFunction::named(self.declared.entry()),
                    VertexLayout::new(&attributes, &layouts),
                )
                .words(),
            );
        }
        inputs.push(Input::bytes("pipelines", &pipelines));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let records = made(sites, "buffer/records", side.buffer(&records()))?;
        let formats = self.declared.formats();
        let out = made(
            sites,
            "buffer/answers",
            side.buffer(&vec![FILLER; formats.len() * ANSWER_BYTES as usize]),
        )?;

        // A pipeline Metal or we refuse is recorded at its own site and the sweep carries on, so
        // one refused format compares as a refusal there while every other format still compares.
        let mut pipelines: Vec<Option<Handle>> = Vec::with_capacity(formats.len());
        for &format in &formats {
            let attributes = self.attributes(format);
            let layouts = self.layouts();
            pipelines.push(probe(
                sites,
                &format!("{}/pipeline", site_name(format)),
                side.render_pipeline(
                    library,
                    &RenderPipelineSpec::converting(
                        StageFunction::named(self.declared.entry()),
                        VertexLayout::new(&attributes, &layouts),
                    ),
                ),
                |_| Value::Absent,
            ));
        }

        // One command buffer: a pass per pipeline, each writing its own slice of the one read-back
        // buffer, so the case costs one commit and one wait rather than one per format.
        let mut work = Vec::new();
        for (at, pipeline) in pipelines.iter().enumerate() {
            let Some(pipeline) = pipeline else { continue };
            let base = at as u64 * u64::from(ANSWER_BYTES);
            let steps = vec![
                RenderStep::Pipeline(*pipeline),
                RenderStep::VertexBuffer {
                    buffer: out,
                    offset: base,
                    index: 0,
                },
                RenderStep::VertexBuffer {
                    buffer: records,
                    offset: 0,
                    index: 1,
                },
                RenderStep::Draw {
                    primitive: PrimitiveType::Point,
                    first: 0,
                    count: ROWS as u32,
                    instances: 1,
                },
            ];
            work.push((
                at,
                RenderWork::Pass(RenderPass::sized([1, 1]).drawing(steps)),
            ));
        }
        let drawn: Vec<usize> = work.iter().map(|(at, _)| *at).collect();
        let passes: Vec<RenderWork> = work.into_iter().map(|(_, pass)| pass).collect();
        let outcome = call(sites, "render", side.render(&passes), |outcome| {
            outcome.status.clone()
        })?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(out, 0, formats.len() * ANSWER_BYTES as usize),
            |_| Value::Absent,
        )?;

        for (at, &format) in formats.iter().enumerate() {
            let first = at * ANSWER_BYTES as usize;
            let answer = bytes[first..first + ANSWER_BYTES as usize].to_vec();
            let mut row = Node::branch(site_name(format), Vec::new()).with(Node::leaf(
                "rows",
                Observed::bytes(answer, self.declared.answer_layout()),
            ));
            // What the format's pass answered, which is `None` on Metal's side: a pass of ours that
            // refuses compares against Metal encoding and reads as a refusal rather than a
            // divergence.
            if let Some(pass) = drawn.iter().position(|held| *held == at) {
                if let Some(Some(failure)) = outcome.refused.get(pass) {
                    row = row.with(Node::leaf("pass", Observed::value(failure.recorded())));
                }
            }
            sites.push(row);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute_coding::codes;
    use std::collections::BTreeSet;

    /// What `format` must answer in row `row`: its own components' codes, then Metal's
    /// `(0, 0, 0, 1)` fill for the components it does not hold.
    ///
    /// The expectation of the sweep, written from the shape the SDK name states and the shared code
    /// table — never from a recording, which is what it is checked against.
    fn expected(format: VertexFormat, row: usize) -> [u64; 4] {
        let held = shape(format.name()).expect("a swept format has a shape");
        let list = codes(held.coding);
        let mut out = [0, 0, 0, 1];
        for component in 0..held.components {
            out[component] = list[(row + 3 * component) % ROWS];
        }
        out
    }

    /// Every format the two cases sweep is exactly the SDK table's 4-byte integer codings, and the
    /// eight of them together are what `vertex_formats` cannot declare.
    ///
    /// The family's depth claim, held by execution: a format that left this sweep has to be one
    /// `vertex_formats` picked up, and the two lists stay the generated table's between them.
    #[test]
    fn the_two_sweeps_are_every_wide_integer_format_of_the_sdk_table() {
        let swept: BTreeSet<&str> = all()
            .iter()
            .flat_map(|case| case.declared.formats())
            .map(site_name)
            .collect();
        let wide: BTreeSet<&str> = VertexFormat::ALL
            .iter()
            .filter(|format| shape(format.name()).is_some_and(|held| !held.coding.float_readable()))
            .map(|format| format.name())
            .collect();
        assert_eq!(swept, wide, "the sweeps are not the wide integer codings");
        assert_eq!(
            swept.len(),
            8,
            "the SDK table holds eight of them: {swept:?}"
        );
        // Disjoint from the float sweep's list, so no format is compared twice and none is in
        // neither: the two families are one partition of the table.
        let float_readable: BTreeSet<&str> = VertexFormat::ALL
            .iter()
            .filter(|format| shape(format.name()).is_some_and(|held| held.coding.float_readable()))
            .map(|format| format.name())
            .collect();
        assert!(swept.is_disjoint(&float_readable));
        let classified: BTreeSet<&str> = VertexFormat::ALL
            .iter()
            .filter(|format| shape(format.name()).is_some())
            .map(|format| format.name())
            .collect();
        assert_eq!(
            swept
                .union(&float_readable)
                .copied()
                .collect::<BTreeSet<&str>>(),
            classified,
            "a classified format is in neither family"
        );
    }

    /// Each case's two declarations read different code lists, so the `uint4` sweep is not the
    /// `int4` sweep spelled twice.
    ///
    /// The non-vacuity floor for the signedness dimension: the same 32 bits read signed and
    /// unsigned are different numbers above `0x7FFF_FFFF`, and the two lists differ in the codes
    /// themselves as well as in their order.
    #[test]
    fn the_signed_and_unsigned_sweeps_ask_different_questions() {
        let signed = codes(coding::Coding::Integer {
            bytes: 4,
            signed: true,
        });
        let unsigned = codes(coding::Coding::Integer {
            bytes: 4,
            signed: false,
        });
        assert_ne!(signed, unsigned, "one list cannot stand for both");
        let only_signed: BTreeSet<u64> = signed.iter().copied().collect();
        let only_unsigned: BTreeSet<u64> = unsigned.iter().copied().collect();
        assert_ne!(
            only_signed, only_unsigned,
            "the two lists hold the same codes in another order, which no fetch can tell apart"
        );
    }

    /// No two formats of one case answer the same words, so every swept value compares something
    /// the value beside it did not.
    ///
    /// This is the standard a sweep has to meet: a format whose answer another format's answer
    /// already held would be comparing nothing a second time. The four of one signedness differ
    /// both in how many codes they read and in the `(0, 0, 0, 1)` fill behind them.
    #[test]
    fn no_two_formats_answer_the_same_words() {
        for case in all() {
            let mut seen: BTreeSet<Vec<[u64; 4]>> = BTreeSet::new();
            for format in case.declared.formats() {
                let answer: Vec<[u64; 4]> = (0..ROWS).map(|row| expected(format, row)).collect();
                assert!(
                    seen.insert(answer),
                    "{}: {} answers what another format already did",
                    case.name,
                    site_name(format)
                );
            }
            assert_eq!(seen.len(), 4, "{}: four formats a signedness", case.name);
        }
    }

    /// Every row of a format's answer differs from every other row of it, so the twelve rows are
    /// twelve experiments rather than one repeated.
    #[test]
    fn no_two_rows_of_a_format_answer_the_same_words() {
        for case in all() {
            for format in case.declared.formats() {
                let rows: BTreeSet<[u64; 4]> = (0..ROWS).map(|row| expected(format, row)).collect();
                assert_eq!(
                    rows.len(),
                    ROWS,
                    "{}: {} repeats a row",
                    case.name,
                    site_name(format)
                );
            }
        }
    }

    /// The codes a 32-bit integer sweep must hold: zero, one, both extremes of each signedness and
    /// the code either side of each, which is the bounded-numeric depth the brief asks of a
    /// 32-bit fetch.
    ///
    /// Written against the shared table so a code list edited elsewhere fails here rather than
    /// quietly narrowing what the sweep asks.
    #[test]
    fn the_codes_hold_every_edge_of_a_32_bit_integer() {
        for signed in [true, false] {
            let list: BTreeSet<u64> = codes(coding::Coding::Integer { bytes: 4, signed })
                .into_iter()
                .collect();
            for wanted in [
                0,
                1,
                2,
                0x7FFF_FFFF,
                0x8000_0000,
                0x8000_0001,
                0xFFFF_FFFE,
                0xFFFF_FFFF,
            ] {
                assert!(
                    list.contains(&wanted),
                    "signed={signed}: the list holds no {wanted:#x}"
                );
            }
        }
    }

    /// A format's slot never overlaps the one beside it, and no slot reads the filler where a
    /// component lies: the records say what each fetch is asked for.
    #[test]
    fn every_slot_holds_its_own_codes() {
        let names = names();
        let bytes = records();
        assert_eq!(bytes.len(), stride() as usize * ROWS);
        for (at, format) in laid_out().into_iter().enumerate() {
            let held = shape(format.name()).expect("a laid-out format has a shape");
            assert_eq!(slot_of(format), SLOT * at as u32);
            for row in 0..ROWS {
                let first = row * stride() as usize + slot_of(format) as usize;
                let held_bytes = &bytes[first..first + held.size];
                assert_eq!(
                    held_bytes,
                    coding::slot(names[at], row).as_slice(),
                    "{}: row {row} is not its own slot",
                    format.name()
                );
                // The bytes of the slot the format does not fill stay the filler, so a fetch that
                // read past its components read a value no code list holds.
                let rest = &bytes[first + held.size..first + SLOT as usize];
                assert!(
                    rest.iter().all(|byte| *byte == FILLER),
                    "{}: the rest of its slot is not the filler",
                    format.name()
                );
            }
        }
    }
}
