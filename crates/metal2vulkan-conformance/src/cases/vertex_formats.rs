//! Every `MTLVertexFormat` a float attribute can read, decoded by the vertex descriptor.
//!
//! `MTLVertexFormat` has 54 enumerators and the suite's other families between them reached
//! eighteen. A once-each suite passes an implementation that decodes seventeen of them correctly
//! and `MTLVertexFormatUChar4Normalized_BGRA` with the wrong channel order, or one that divides a
//! signed-normalized code by 128 instead of 127, so this family sweeps the **field**: every format
//! a float-class attribute can read, each at the codes its own encoding makes interesting, read
//! back as the numbers the shader saw.
//!
//! The value list is **not written here.** It is read from `VertexFormat::ALL`, which
//! `scripts/regen-metal-enums/regen_metal_enums.py` extracts from the SDK headers, and each
//! format's shape — how many components it holds, how wide they are and how they are coded — is
//! derived from the SDK enumerator's **own name**, which states exactly that. So a format the SDK
//! adds appears as an unclassified name in
//! `tests::every_vertex_format_of_the_sdk_table_is_classified` rather than as silence, and each
//! site is named after the enumerator, so a value renamed upstream renames its site loudly.
//!
//! # What is not swept here, and why
//!
//! - **`MTLVertexFormatInvalid`** is the absence of an attribute, not a format: a
//!   `MTLVertexAttributeDescriptor` left at it is the one the descriptor does not describe, and
//!   `metal2vulkan`'s own pipeline builder filters it out (`pipeline::descriptor`). It is a
//!   refusal-surface question — what Metal does with an attribute the function declares and the
//!   descriptor leaves `Invalid` — and not a value row, so no row here sets it.
//! - **`MTLVertexFormatInt`, `Int2`, `Int3`, `Int4`, `UInt`, `UInt2`, `UInt3`, `UInt4`**: a 32-bit
//!   integer format reaches the shader as an integer, so it needs an `int4`- or `uint4`-declared
//!   attribute and a `float4` or `half4` one cannot read it. Those eight are the integer half of the
//!   field and are swept by `cases::wide_int_attributes` (`vertex-format-int4`,
//!   `vertex-format-uint4`) through `dsx_wide_int_attributes`, which declares both. They are left
//!   out **here** and compared there, over the same twelve codes of
//!   [`crate::attribute_coding`]: `tests::the_formats_left_out_are_the_integer_ones_and_invalid`
//!   holds that this list and the swept one together are the whole SDK table, so this family's
//!   half cannot widen quietly, and
//!   `wide_int_attributes::tests::the_two_sweeps_are_every_wide_integer_format_of_the_sdk_table`
//!   holds that the other family is exactly the eight.
//!
//! The 8- and 16-bit **unnormalized** integer formats are swept, because Metal converts those to
//! float in a float attribute: `integer-vertex-attributes` already records Metal doing it for
//! sixteen of them.
//!
//! # The codes, and why these separate the decodings
//!
//! Twelve rows. Row `r` gives each format's component `c` the code at index `(r + 3 c) % 12` of
//! its encoding's own list, so the components of one fetch differ from each other and a decoder
//! that swapped two components is caught by the values rather than by luck. Each list is the
//! encoding's documented edges:
//!
//! | encoding | what the twelve codes hold |
//! |---|---|
//! | unsigned 8, 16, 32 | 0, 1, 2, the signed midpoint and the two codes astride it, the top three codes, and two alternating-bit patterns |
//! | signed 8, 16, 32 | 0, 1, -1, the maximum, the minimum **and the code above it** (both of which a normalized decode maps to -1.0), 2, -2 and two powers of two |
//! | `Half` | +0, -0, the smallest subnormal, the largest subnormal, the smallest normal, 1.0, the largest finite, +inf, -inf, a NaN payload and its negative, and a value needing every mantissa bit |
//! | `Float` | the same twelve, one width up |
//! | `*_2_10_10_10` | zero, each of the four fields at its maximum with the others zero, every field at its maximum, the 10-bit field's sign boundary (512, 513 and 511, the first two of which a signed normalized decode maps to -1.0), the 2-bit field at each of its signed values, and a mixed word |
//! | `FloatRG11B10`, `FloatRGB9E5` | zero, the lowest set bit, each field's boundary on both sides, every bit set (which puts an all-ones exponent in every field), and three mixed words |
//!
//! The signed-normalized pair is the one the table is built around: Metal's signed normalized
//! decode is `max(code / (2^(n-1) - 1), -1)`, so the minimum code and the code above it **both**
//! read -1.0, while an implementation that divided by `2^(n-1)` reads -1.0 only at the minimum.
//! `tests::minus_one_rows` names the two rows that put that pair in component 0 of each signed coding —
//! the index differs between the per-component lists and the packed word's — and
//! `tests::two_codes_of_every_signed_normalized_format_read_minus_one` holds Metal's recorded
//! answer to it. No row index is written in prose or in a test: both read that one function, so a
//! list reordered cannot leave the claim pointing at the wrong codes.
//!
//! `MTLVertexFormatUChar4Normalized_BGRA` holds the same four bytes as
//! `MTLVertexFormatUChar4Normalized` in every row, by construction: the code a component takes
//! depends only on its encoding, and the two formats share one. So the two sites are the same
//! bytes decoded two ways, which is the channel-order question with nothing else moving, and
//! `tests::the_bgra_format_reads_the_same_bytes_with_its_channels_swapped` holds Metal's answer
//! to it.
//!
//! # The three cases, and what each adds
//!
//! - **`vertex-format-float4`** — every swept format into the fixture's `float4` attribute,
//!   stepped once per vertex. The decoding itself.
//! - **`vertex-format-half4`** — the same formats into its `half4` attribute. Metal decodes the
//!   format and then narrows each component to `half`, so this pairs the format against the
//!   attribute's declared width: the 16-bit unsigned codes near 65535 reach `half`'s overflow,
//!   every subnormal and infinity has to survive the narrowing, and a 32-bit float code needs
//!   round-to-nearest-even. The shader widens the `half4` back to `float4` exactly, so nothing is
//!   lost between the fetch and the read-back.
//! - **`vertex-format-constant-step`** — every swept format into the `float4` attribute under
//!   `MTLVertexStepFunctionConstant`, whose fetch no fixed-function vertex input can express, so
//!   the shader pulls it. The decoding must not depend on which path fetched it, and
//!   `tests::the_two_step_functions_recorded_the_same_values` holds that.
//! - **`vertex-format-instance-step`** — every swept format under
//!   `MTLVertexStepFunctionPerInstance`, drawn as **one** instance of twelve vertices, so the step
//!   fetches element 0 for every one of them. That is the whole of what this fixture can observe of
//!   the instance step, and `tests::the_instance_step_fetched_one_element_for_every_vertex` holds
//!   it: the answer is twelve copies of what the per-vertex step answered at row 0, which tells it
//!   apart from the per-vertex step (whose twelve rows differ from each other) and not from the
//!   constant step.
//!
//! Format against attribute width and format against step are the two pairings that can change a
//! decode; the three of them together are not taken, because a step function cannot reach the
//! narrowing a width does.
//!
//! # The step functions this family does not set, and why
//!
//! - **`MTLVertexStepFunctionPerInstance`'s step rate** is not observable here. One instance
//!   fetches element `0 / rate` whatever the rate is, and separating two rates needs two instances
//!   with destinations of their own — which needs a fixture declaring `[[instance_id]]`, and no
//!   committed one does. The case sets rate 1 and claims nothing about any other.
//! - **`MTLVertexStepFunctionPerPatch` and `PerPatchControlPoint`** are tessellation's: they need a
//!   post-tessellation vertex function, a patch control-point buffer and a tessellation factor
//!   buffer, which is a family of its own rather than a row of this one. Named here because no case
//!   in the suite sets either, so the two values are uncompared.
//! - **`MTLStepFunctionPerVertex`, `PerInstance`, `PerPatch` and `PerPatchControlPoint`** — the
//!   vertex and tessellation steps of `MTLStageInputOutputDescriptor`, the compute descriptor —
//!   are **not set by any case and must not be**. A vertex or tessellation step in a compute layout
//!   is an invalid descriptor, and Metal's compute pipeline creation stops at an **assertion** for
//!   one (`metal2vulkan::lower::stage_io::stage_in`'s module docs record the measurement). An
//!   assertion ends the process, so a case that set one could not report anything and would take
//!   the run down with it. Those four values are therefore uncomparable rather than uncompared, and
//!   that is a property of Metal's API and not a gap of the suite's.
//!
//! # What the three cases separate, measured
//!
//! Metal's answers to all 45 formats are distinct, format for format
//! (`tests::no_two_swept_formats_answered_the_same_rows`), so no site of this family compares what
//! another already compared. Two of the answers are worth naming, because they are what an
//! implementation gets wrong:
//!
//! - **`MTLVertexFormatHalf`'s widening canonicalizes a NaN.** The codes `0x7E01` and `0xFE01`
//!   both reach the shader as the positive quiet NaN, payload and sign dropped, while
//!   `MTLVertexFormatFloat` over the same twelve specials is a straight copy that keeps both
//!   (`tests::halfs_specials_are_metals_own_widening`,
//!   `tests::floats_specials_reach_the_shader_as_the_bytes_they_were`). The float rows are the
//!   control that puts the canonicalization in the widening rather than in the buffer, the
//!   read-back or the transcript.
//! - **The constant step is where a pulled decode is told from a fixed-function one.** Metal
//!   answers the same values under both steps, format for format
//!   (`tests::the_two_step_functions_recorded_the_same_values`), so the step axis compares no new
//!   *Metal* value — and that is exactly what makes it load-bearing: an implementation that cannot
//!   express the fetch in fixed-function vertex input must decode it in the shader, and the two
//!   decodes have to agree. `known-gaps.tsv` holds five rows where ours does not.
//!
//! # The device's own bounds
//!
//! One attribute and one vertex buffer layout a pipeline, of stride `16 n` for the `n` formats
//! swept, every attribute at a 16-byte-aligned offset. Metal's own limits on a
//! `MTLVertexDescriptor` are 31 attributes, a stride of at most 65535 bytes that is a multiple of
//! 4, and an offset aligned to the format's component width;
//! `tests::the_descriptor_stays_inside_metals_own_limits` holds every row to all four, so a
//! format the SDK adds cannot push the table past them without saying so. Nothing here reads a
//! device limit, because Metal exposes none for these: the numbers are the API's.
//!
//! # The shape, and why nothing here can fault a GPU
//!
//! The fixture's functions are `vertex void`, so there is no position to rasterize: each pipeline
//! disables rasterization and carries no fragment stage, and each pass has no attachment of any
//! kind and states its own one-by-one extent. Every draw is of points, twelve or one at a time,
//! every fetch is inside a record the case bound, and the whole case is one command buffer, one
//! commit and one wait.

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
use metal2vulkan::metal::{PrimitiveType, VertexFormat, VertexStepFunction};
use metal2vulkan_runtime::Degradation;

/// The fixture both entry points live in, shared with `integer-vertex-attributes`: its two
/// `vertex void` functions fetch attribute 0 as a `float4` and as a `half4` and write it to
/// `out[vertex_id]`, which is every shape this family needs.
const FIXTURE: &str = "dsx_int_attributes";

/// The entry point whose `[[attribute(0)]]` is a `float4`.
const TO_FLOAT4: &str = "int_attribute_to_float4";

/// The entry point whose `[[attribute(0)]]` is a `half4`, widened to `float4` exactly on the way
/// out.
const TO_HALF4: &str = "int_attribute_to_half4";

/// Bytes one row of a case's answer takes: one `float4`.
const ROW_BYTES: u32 = 16;

/// Bytes one format's answer takes, which is its slice of the read-back buffer.
const ANSWER_BYTES: u32 = ROWS as u32 * ROW_BYTES;

// ---- The SDK's own value list, over this family's own path -----------------------------------

/// Every format of the SDK table a float-declared attribute can read, in the table's own
/// declaration order.
fn formats() -> Vec<VertexFormat> {
    VertexFormat::ALL
        .iter()
        .copied()
        .filter(|format| shape(format.name()).is_some_and(|held| held.coding.float_readable()))
        .collect()
}

/// The site one format's answer is emitted at: the SDK enumerator's **own name**.
///
/// Not a prettier spelling of it. The suite's depth in a Metal enum is derived by
/// `scripts/conformance-coverage/option_space.py`, one numerator of which is the enumerators a
/// committed recording's compared transcript holds by name, and a site named `uchar4-normalized`
/// holds none: this family states its values through `VertexFormat::ALL` rather than variant by
/// variant, so the transcript is the only place the SDK's name can appear. Naming the site after
/// the enumerator closes that, puts the value Metal was asked for in the compared text, and makes
/// a value the SDK renames rename its site.
fn site_name(format: VertexFormat) -> &'static str {
    format.name()
}

/// Every swept format's SDK name, in the order the records lay their slots out.
fn names() -> Vec<&'static str> {
    formats().into_iter().map(site_name).collect()
}

/// Bytes one record takes: one [`SLOT`] per swept format.
fn stride() -> u32 {
    coding::stride(&names())
}

/// The records the vertex buffer holds: row `r`'s slot for each format at byte `SLOT i`, and
/// [`FILLER`] everywhere else.
fn records() -> Vec<u8> {
    coding::records(&names())
}

/// The layout one format's answer is read under: one row of the layout per row of the records, one
/// element per component, so a diff names the row and the component inside it.
fn answer_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(ANSWER_BYTES),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: ROW_BYTES,
        rows: ROWS as u32,
        image_pitch: ANSWER_BYTES,
        images: 1,
        gaps: Gaps::Compared,
    }
}

// ---- The cases ---------------------------------------------------------------------------------

/// What one case of the family holds still and what it varies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sweep {
    /// Every swept format into the `float4` attribute, stepped once per vertex.
    Float4,
    /// Every swept format into the `half4` attribute, stepped once per vertex.
    Half4,
    /// Every swept format into the `float4` attribute under Metal's constant step.
    Constant,
    /// Every swept format into the `float4` attribute under Metal's instance step, drawn as one
    /// instance of [`ROWS`] vertices.
    PerInstance,
}

impl Sweep {
    /// The entry point the sweep draws with.
    fn entry(self) -> &'static str {
        match self {
            Sweep::Half4 => TO_HALF4,
            _ => TO_FLOAT4,
        }
    }

    /// The step function the sweep's one vertex buffer layout carries.
    fn step(self) -> VertexStepFunction {
        match self {
            Sweep::Constant => VertexStepFunction::Constant,
            Sweep::PerInstance => VertexStepFunction::PerInstance,
            _ => VertexStepFunction::PerVertex,
        }
    }
}

/// One case of the family.
pub struct VertexFormats {
    name: &'static str,
    sweep: Sweep,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<VertexFormats> {
    vec![
        VertexFormats {
            name: "vertex-format-float4",
            sweep: Sweep::Float4,
        },
        VertexFormats {
            name: "vertex-format-half4",
            sweep: Sweep::Half4,
        },
        VertexFormats {
            name: "vertex-format-constant-step",
            sweep: Sweep::Constant,
        },
        VertexFormats {
            name: "vertex-format-instance-step",
            sweep: Sweep::PerInstance,
        },
    ]
}

impl VertexFormats {
    /// The fixture under the case's entry point.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: self.sweep.entry(),
        }
    }

    /// Attribute 0 of the format at slot `at`: that format at byte `SLOT at` of buffer 1.
    fn attributes(&self, at: usize, format: VertexFormat) -> [VertexAttribute; 1] {
        [VertexAttribute::new(0, format, SLOT * at as u32, 1)]
    }

    /// Buffer 1's layout under the sweep's step: one record of [`stride`] bytes.
    fn layouts(&self) -> [VertexBufferLayout; 1] {
        [match self.sweep.step() {
            VertexStepFunction::Constant => VertexBufferLayout::constant(1, stride()),
            VertexStepFunction::PerInstance => VertexBufferLayout {
                index: 1,
                stride: stride(),
                step: VertexStepFunction::PerInstance,
                // The rate is not observable through this fixture, which has no `[[instance_id]]`:
                // one instance fetches element `0 / rate` whatever the rate is. 1 is the plain
                // value; the module docs say what a rate sweep would need.
                rate: 1,
            },
            _ => VertexBufferLayout::per_vertex(1, stride()),
        }]
    }
}

impl Case for VertexFormats {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened. Every decoding here is exact arithmetic on the codes the record
        // holds -- an integer widened, a code divided by its own maximum, a half or a float
        // widened, a packed field extracted -- so a component that differs is a difference in the
        // decode and not a rounding, and a tolerance written here would hide the bug the sweep
        // exists to find.
        //
        // One notice is declared, and it is the venue's and not this sweep's: the buffer each draw
        // reads its fetched rows back through is `Shared`, which a device with no unified memory
        // keeps in system memory because that is the only memory both sides can reach.
        vec![crate::replay::notice_allowance(
            Degradation::SharedInSystemMemory.code(),
        )]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: self.sweep.entry().to_owned(),
                source_sha256: crate::sha256_hex(self.fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
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
                &vec![FILLER; formats().len() * ANSWER_BYTES as usize],
            ),
        ];
        // One fingerprint over every pipeline, which are generated from the format list and the
        // sweep: a format added, a slot moved or a step changed moves the whole table.
        let mut pipelines = Vec::new();
        for (at, format) in formats().into_iter().enumerate() {
            pipelines.extend(site_name(format).as_bytes());
            let attributes = self.attributes(at, format);
            let layouts = self.layouts();
            pipelines.extend(
                RenderPipelineSpec::converting(
                    StageFunction::named(self.sweep.entry()),
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
        let formats = formats();
        let out = made(
            sites,
            "buffer/answers",
            side.buffer(&vec![FILLER; formats.len() * ANSWER_BYTES as usize]),
        )?;

        // A pipeline Metal or we refuse is recorded at its own site and the sweep carries on, so
        // one refused format compares as a refusal there while every other format still compares.
        // Its answer slice then stays at the filler on that side, which is the same shape as every
        // other slice: a refusal never arrives as a missing site.
        let mut pipelines: Vec<Option<Handle>> = Vec::with_capacity(formats.len());
        for (at, &format) in formats.iter().enumerate() {
            let attributes = self.attributes(at, format);
            let layouts = self.layouts();
            pipelines.push(probe(
                sites,
                &format!("{}/pipeline", site_name(format)),
                side.render_pipeline(
                    library,
                    &RenderPipelineSpec::converting(
                        StageFunction::named(self.sweep.entry()),
                        VertexLayout::new(&attributes, &layouts),
                    ),
                ),
                |_| Value::Absent,
            ));
        }

        // One command buffer: a pass per pipeline, each writing its own slice of the one read-back
        // buffer, so the case costs one commit and one wait rather than one per format.
        let stride = stride();
        let mut work = Vec::new();
        for (at, pipeline) in pipelines.iter().enumerate() {
            let Some(pipeline) = pipeline else { continue };
            let base = at as u64 * u64::from(ANSWER_BYTES);
            let mut steps = vec![RenderStep::Pipeline(*pipeline)];
            if self.sweep.step() == VertexStepFunction::Constant {
                // The constant step fetches the one element the binding's offset names, so each
                // row is its own draw of one point, bound at its own record.
                for row in 0..ROWS as u32 {
                    steps.push(RenderStep::VertexBuffer {
                        buffer: out,
                        offset: base + u64::from(row * ROW_BYTES),
                        index: 0,
                    });
                    steps.push(RenderStep::VertexBuffer {
                        buffer: records,
                        offset: u64::from(row * stride),
                        index: 1,
                    });
                    steps.push(RenderStep::Draw {
                        primitive: PrimitiveType::Point,
                        first: 0,
                        count: 1,
                        instances: 1,
                    });
                }
            } else {
                steps.push(RenderStep::VertexBuffer {
                    buffer: out,
                    offset: base,
                    index: 0,
                });
                steps.push(RenderStep::VertexBuffer {
                    buffer: records,
                    offset: 0,
                    index: 1,
                });
                steps.push(RenderStep::Draw {
                    primitive: PrimitiveType::Point,
                    first: 0,
                    count: ROWS as u32,
                    instances: 1,
                });
            }
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
            let mut row = Node::branch(site_name(format), Vec::new())
                .with(Node::leaf("rows", Observed::bytes(answer, answer_layout())));
            // What the format's pass answered, which is `None` on Metal's side: a pass of ours
            // that refuses compares against Metal encoding and reads as a refusal rather than a
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

/// The `float4` case's recorded answer for the format named `name`, as the bits of each row's four
/// components — or `None` where the recording holds no such site.
///
/// Read by `cases::attribute_formats`, which compares the stage-in path's decode of the same
/// format against this one: Metal's two fetches are one field and ours is one module, so the
/// cross-check belongs to whichever family runs second and needs the other's numbers.
#[cfg(test)]
pub fn recorded_float4_bits(name: &str) -> Option<Vec<[u32; 4]>> {
    let recording = crate::recording::read(all()[0].name()).ok()?;
    for answer in &recording.answers {
        for outcome in &answer.outcomes {
            let Some(Value::Bytes(bytes)) = outcome
                .transcript
                .root
                .child(name)
                .and_then(|node| node.child("rows"))
                .map(|node| &node.observed.value)
            else {
                continue;
            };
            return Some(
                bytes
                    .chunks(16)
                    .map(|row| {
                        [0, 1, 2, 3].map(|at| {
                            u32::from_le_bytes([
                                row[4 * at],
                                row[4 * at + 1],
                                row[4 * at + 2],
                                row[4 * at + 3],
                            ])
                        })
                    })
                    .collect(),
            );
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute_coding::{codes, Coding};
    use crate::transcript::Value;
    use std::collections::BTreeSet;

    /// The bytes format `format`'s slot holds in row `row`, as the records lay it out.
    fn slot(format: VertexFormat, row: usize) -> Vec<u8> {
        coding::slot(site_name(format), row)
    }

    /// The two rows whose code, in component 0 of a signed format, is the minimum code and the
    /// code above it — the pair a signed normalized decode maps to -1.0 together.
    ///
    /// `None` where the coding has no such pair. A packed coding's row is one word, so its pair is
    /// the two words with the 10-bit field at 512 and at 513, which sit elsewhere in its own list.
    ///
    /// Both the claim about Metal's answer and the check that these really are the minimum and the
    /// code above it read this one function, so a code list reordered cannot leave either pointing
    /// at the wrong rows.
    fn minus_one_rows(coding: Coding) -> Option<[usize; 2]> {
        match coding {
            Coding::Integer { signed: true, .. } | Coding::Normalized { signed: true, .. } => {
                Some([2, 3])
            }
            Coding::Packed1010102 { signed: true } => Some([6, 7]),
            _ => None,
        }
    }

    /// Whether `coding` scales a **signed** code into `[-1, 1]`, so two of its codes read -1.0.
    fn signed_normalized(coding: Coding) -> bool {
        matches!(
            coding,
            Coding::Normalized { signed: true, .. } | Coding::Packed1010102 { signed: true }
        )
    }

    /// The formats a float-declared attribute cannot read, which this family leaves out: every
    /// 32-bit integer format, and `MTLVertexFormatInvalid`, which is not a format at all.
    ///
    /// The eight integer ones are left out of **this** family and swept by
    /// `cases::wide_int_attributes`, which declares the `int4` and `uint4` attributes they need, so
    /// a name here is this family's omission rather than the suite's.
    ///
    /// Stated here so that [`the_formats_left_out_are_the_integer_ones_and_invalid`] can hold the
    /// swept list and this one to being the whole SDK table between them. It is a statement of
    /// what is missing, not a value list the sweep reads: `formats()` derives the sweep from
    /// `VertexFormat::ALL` and never from this.
    const LEFT_OUT: [&str; 9] = [
        "MTLVertexFormatInvalid",
        "MTLVertexFormatInt",
        "MTLVertexFormatInt2",
        "MTLVertexFormatInt3",
        "MTLVertexFormatInt4",
        "MTLVertexFormatUInt",
        "MTLVertexFormatUInt2",
        "MTLVertexFormatUInt3",
        "MTLVertexFormatUInt4",
    ];

    /// Every value of the SDK table either states a shape this family can read from its name, or
    /// is one of the names [`LEFT_OUT`] accounts for.
    ///
    /// A format the SDK adds fails here rather than being dropped from the sweep in silence, which
    /// is what keeps the denominator the generated table's.
    #[test]
    fn every_vertex_format_of_the_sdk_table_is_classified() {
        let mut unshaped = Vec::new();
        for &format in VertexFormat::ALL {
            if shape(format.name()).is_none() && !LEFT_OUT.contains(&format.name()) {
                unshaped.push(format.name());
            }
        }
        assert!(
            unshaped.is_empty(),
            "the SDK table holds {} format(s) this family neither shapes nor accounts for: {}",
            unshaped.len(),
            unshaped.join(", ")
        );
    }

    /// The swept list and [`LEFT_OUT`] are the whole SDK table between them, with nothing in both.
    ///
    /// This is the family's depth claim, held by execution: a format that stopped being swept has
    /// to join the named list, and a name on that list that became readable has to leave it.
    #[test]
    fn the_formats_left_out_are_the_integer_ones_and_invalid() {
        let swept: BTreeSet<&str> = formats().iter().map(|format| format.name()).collect();
        let left: BTreeSet<&str> = LEFT_OUT.into_iter().collect();
        assert!(
            swept.is_disjoint(&left),
            "a format is both swept and named as left out: {:?}",
            swept.intersection(&left).collect::<Vec<_>>()
        );
        let whole: BTreeSet<&str> = VertexFormat::ALL
            .iter()
            .map(|format| format.name())
            .collect();
        let covered: BTreeSet<&str> = swept.union(&left).copied().collect();
        assert_eq!(
            covered, whole,
            "the swept formats and the named omissions are not the SDK table"
        );
        assert_eq!(swept.len(), VertexFormat::ALL.len() - LEFT_OUT.len());
    }

    /// Each swept format's site names a distinct site, since the comparison pairs the sides' sites
    /// by name.
    #[test]
    fn every_format_names_a_distinct_site() {
        let named: BTreeSet<&str> = formats().iter().copied().map(site_name).collect();
        assert_eq!(named.len(), formats().len());
        // The names are the SDK's own, written one way: a format whose kebab form collided with
        // another's would have one of the two silently compared against the other's bytes.
        // The names are the SDK's own, so the option-space census reads them out of the
        // recording: a site named anything else earns this family no depth at all.
        assert_eq!(
            site_name(VertexFormat::UChar4NormalizedBGRA),
            "MTLVertexFormatUChar4Normalized_BGRA"
        );
        assert_eq!(
            site_name(VertexFormat::FloatRG11B10),
            "MTLVertexFormatFloatRG11B10"
        );
        for format in formats() {
            assert!(
                site_name(format).starts_with("MTLVertexFormat"),
                "{}: its site is not the SDK enumerator",
                format.name()
            );
        }
    }

    /// Every row of every case stays inside Metal's own limits on a `MTLVertexDescriptor`: at most
    /// 31 attributes, a stride of at most 65535 bytes and a multiple of 4, each attribute's offset
    /// aligned to its format's component width, and the format wholly inside the element.
    ///
    /// Metal exposes no device query for any of these, so the numbers are the API's and the test
    /// holds the table to them. A format the SDK adds that pushed the stride past 65535 fails here
    /// rather than at pipeline creation on one device.
    #[test]
    fn the_descriptor_stays_inside_metals_own_limits() {
        const MAX_ATTRIBUTES: usize = 31;
        const MAX_STRIDE: u32 = 65535;
        let stride = stride();
        assert!(
            stride <= MAX_STRIDE && stride.is_multiple_of(4),
            "the records' stride is {stride} bytes"
        );
        for case in all() {
            for (at, format) in formats().into_iter().enumerate() {
                let attributes = case.attributes(at, format);
                assert!(attributes.len() <= MAX_ATTRIBUTES);
                let held = shape(format.name()).expect("a swept format has a shape");
                let offset = attributes[0].offset;
                assert_eq!(
                    offset % held.coding.width() as u32,
                    0,
                    "{}: offset {offset} is not aligned to its {}-byte components",
                    format.name(),
                    held.coding.width()
                );
                assert!(
                    offset + held.size as u32 <= stride,
                    "{}: its {} bytes at offset {offset} reach past the {stride}-byte element",
                    format.name(),
                    held.size
                );
            }
        }
    }

    /// The records hold each format's codes where its attribute reads them, the filler everywhere
    /// else, and one answer slice is the layout's extent exactly.
    #[test]
    fn the_records_and_the_layout_agree_with_the_buffers() {
        let formats = formats();
        let stride = stride() as usize;
        let records = records();
        assert_eq!(records.len(), stride * ROWS);
        assert_eq!(answer_layout().extent(), ANSWER_BYTES as usize);
        for row in 0..ROWS {
            for (at, &format) in formats.iter().enumerate() {
                let held = shape(format.name()).expect("a swept format has a shape");
                assert!(
                    held.size <= SLOT as usize,
                    "{} outgrew a slot",
                    format.name()
                );
                let first = row * stride + SLOT as usize * at;
                assert_eq!(
                    &records[first..first + held.size],
                    &slot(format, row)[..],
                    "{} row {row}",
                    format.name()
                );
                // Every byte of the slot the format does not reach is the filler, so a fetch that
                // read past its components read a value no code list holds.
                assert!(records[first + held.size..first + SLOT as usize]
                    .iter()
                    .all(|byte| *byte == FILLER));
            }
        }
    }

    /// `MTLVertexFormatUChar4Normalized_BGRA` and `MTLVertexFormatUChar4Normalized` hold the
    /// **same four bytes** in every row, so the two sites differ only by the channel order.
    ///
    /// Held by construction here, because the recording test that reads Metal's answer to it is
    /// only a question about the channel order while this holds.
    #[test]
    fn the_bgra_format_holds_the_same_bytes_as_the_rgba_one() {
        for row in 0..ROWS {
            assert_eq!(
                slot(VertexFormat::UChar4Normalized, row),
                slot(VertexFormat::UChar4NormalizedBGRA, row),
                "row {row}"
            );
        }
    }

    /// [`minus_one_rows`] names, for every signed coding, the two rows whose code is the minimum
    /// and the code above it, and names none for a coding that has no such pair.
    ///
    /// The construction behind
    /// [`two_codes_of_every_signed_normalized_format_read_minus_one`]: without it that test could
    /// pass over two codes that are not the pair at all.
    #[test]
    fn the_signed_lists_hold_the_minimum_and_the_code_above_it() {
        for bytes in [1usize, 2, 4] {
            let coding = Coding::Normalized {
                bytes,
                signed: true,
            };
            let list = codes(coding);
            let [low, next] = minus_one_rows(coding).expect("a signed coding names its pair");
            let minimum = 1u64 << (8 * bytes as u32 - 1);
            assert_eq!(list[low], minimum, "{bytes}-byte signed: the minimum code");
            assert_eq!(
                list[next],
                minimum + 1,
                "{bytes}-byte signed: the code above it"
            );
        }
        // The packed signed field is ten bits wide inside its word, so its pair is the word with
        // that field at 512 and the word with it at 513.
        let packed = Coding::Packed1010102 { signed: true };
        let [low, next] = minus_one_rows(packed).expect("the signed packed coding names its pair");
        assert_eq!(codes(packed)[low], 512);
        assert_eq!(codes(packed)[next], 513);
        // And an unsigned coding names none, so the recording test skips it rather than asking a
        // question it has no codes for.
        for coding in [
            Coding::Normalized {
                bytes: 1,
                signed: false,
            },
            Coding::Half,
            Coding::Single,
            Coding::PackedRG11B10,
            Coding::PackedRGB9E5,
            Coding::Packed1010102 { signed: false },
        ] {
            assert_eq!(minus_one_rows(coding), None, "{coding:?}");
        }
    }

    /// One case's answer, as the recording holds it: the `ROWS` by four floats of `format`.
    fn recorded(case: &VertexFormats, format: VertexFormat) -> Option<Vec<[u32; 4]>> {
        let recording = crate::recording::read(case.name()).ok()?;
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let Some(Value::Bytes(bytes)) = outcome
                    .transcript
                    .root
                    .child(site_name(format))
                    .and_then(|node| node.child("rows"))
                    .map(|node| &node.observed.value)
                else {
                    continue;
                };
                return Some(
                    bytes
                        .chunks(16)
                        .map(|row| {
                            [0, 1, 2, 3].map(|at| {
                                u32::from_le_bytes([
                                    row[4 * at],
                                    row[4 * at + 1],
                                    row[4 * at + 2],
                                    row[4 * at + 3],
                                ])
                            })
                        })
                        .collect(),
                );
            }
        }
        None
    }

    /// Metal's recorded answer for `MTLVertexFormatUChar4Normalized_BGRA` is its answer for
    /// `MTLVertexFormatUChar4Normalized` with the first and third components swapped.
    ///
    /// The two formats hold the same bytes
    /// ([`the_bgra_format_holds_the_same_bytes_as_the_rgba_one`]), so this is the channel order and
    /// nothing else. An implementation that forwarded `_BGRA` as plain RGBA passes every other
    /// site of this family and fails here.
    #[test]
    fn the_bgra_format_reads_the_same_bytes_with_its_channels_swapped() {
        let case = &all()[0];
        let (Some(rgba), Some(bgra)) = (
            recorded(case, VertexFormat::UChar4Normalized),
            recorded(case, VertexFormat::UChar4NormalizedBGRA),
        ) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        assert_eq!(rgba.len(), ROWS);
        let mut differ = Vec::new();
        for (row, (held, swapped)) in rgba.iter().zip(&bgra).enumerate() {
            let want = [held[2], held[1], held[0], held[3]];
            if *swapped != want {
                differ.push(format!(
                    "row {row}: BGRA {swapped:08x?}, RGBA swapped {want:08x?}"
                ));
            }
        }
        assert!(
            differ.is_empty(),
            "the BGRA format is not the RGBA one's channels swapped: {}",
            differ.join("; ")
        );
        // And the swap is observable: at least one row's first and third components differ, so the
        // claim is not two equal numbers agreeing.
        assert!(
            rgba.iter().any(|held| held[0] != held[2]),
            "no row distinguishes the first component from the third"
        );
    }

    /// Metal's recorded answer maps **two** codes of every signed normalized format to -1.0: the
    /// minimum code and the code above it.
    ///
    /// [`minus_one_rows`] names the two rows holding that pair in component 0, and
    /// [`the_signed_lists_hold_the_minimum_and_the_code_above_it`] holds that they are it. An
    /// implementation that
    /// divided by `2^(n-1)` rather than `2^(n-1) - 1` reads -1.0 at row 3 and something above it at
    /// row 4, and fails here while passing every unsigned site.
    #[test]
    fn two_codes_of_every_signed_normalized_format_read_minus_one() {
        let case = &all()[0];
        let mut checked = 0usize;
        let mut differ = Vec::new();
        for format in formats() {
            let held = shape(format.name()).expect("a swept format has a shape");
            if !signed_normalized(held.coding) {
                continue;
            }
            let Some(rows) = recorded(case, format) else {
                continue;
            };
            let pair = minus_one_rows(held.coding).expect("a signed coding names its pair");
            for row in pair {
                checked += 1;
                if rows[row][0] != (-1.0f32).to_bits() {
                    differ.push(format!(
                        "{} row {row}: {}",
                        format.name(),
                        f32::from_bits(rows[row][0])
                    ));
                }
            }
        }
        if checked == 0 {
            println!("{}: no recording in the tree yet", case.name());
            return;
        }
        assert!(
            differ.is_empty(),
            "{} signed normalized reading(s) of {checked} are not -1.0: {}",
            differ.len(),
            differ.join("; ")
        );
    }

    /// No two swept formats answered the same twelve rows, in Metal's own recording.
    ///
    /// This is the family's non-vacuity proof: a format whose site answered exactly what another's
    /// did compares nothing of its own, and the sweep would be one comparison repeated. Any pair
    /// that is indistinguishable has to be named here with its reason, so "these two cannot be
    /// told apart" is a statement someone wrote rather than a hole.
    #[test]
    fn no_two_swept_formats_answered_the_same_rows() {
        let case = &all()[0];
        let mut held: Vec<(VertexFormat, Vec<[u32; 4]>)> = Vec::new();
        for format in formats() {
            if let Some(rows) = recorded(case, format) {
                held.push((format, rows));
            }
        }
        if held.is_empty() {
            println!("{}: no recording in the tree yet", case.name());
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
            "{} pair(s) of formats answered identically, so neither compares anything the other \
             does not: {}",
            same.len(),
            same.join("; ")
        );
        assert_eq!(
            held.len(),
            formats().len(),
            "the recording holds {} of {} formats",
            held.len(),
            formats().len()
        );
    }

    /// `MTLVertexFormatHalf`'s specials, as Metal's own fetch answers them: the zeroes keep their
    /// sign, the subnormals widen exactly, the infinities keep their sign, and a **NaN loses both
    /// its payload and its sign** — Metal answers the canonical quiet NaN for `0x7E01` and for
    /// `0xFE01` alike.
    ///
    /// That last one is the sharpest row of the family, and it is Metal's measured behaviour on
    /// this venue rather than anything IEEE 754 requires: widening binary16 to binary32 can carry
    /// a payload, since binary32 has the bits for it, and Metal does not. An implementation whose
    /// half fetch forwards the payload answers `0x7FC02000` here, and one that keeps the sign
    /// answers `0xFFC00000`; either passes every other site of this family and fails this one.
    /// Hard-coded because the recording is this venue's Metal: a release that changes it fails here
    /// loudly instead of quietly re-blessing.
    #[test]
    fn halfs_specials_are_metals_own_widening() {
        let case = &all()[0];
        let Some(rows) = recorded(case, VertexFormat::Half) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let list = codes(Coding::Half);
        let mut differ = Vec::new();
        let mut canonicalized = 0usize;
        for (row, held) in rows.iter().enumerate() {
            let code = list[row] as u16;
            let want = metal_widened(code);
            if held[0] != want {
                differ.push(format!(
                    "row {row}: code {code:#06x} read as {:#010x}, Metal's widening is {want:#010x}",
                    held[0]
                ));
            }
            if code & 0x7FFF > 0x7C00 {
                canonicalized += 1;
            }
            // Component 0 of a one-component format takes the row's own code; the three the format
            // does not hold are the fill Metal puts there, `(_, 0, 0, 1)`.
            if [held[1], held[2], held[3]] != [0, 0, 1.0f32.to_bits()] {
                differ.push(format!(
                    "row {row}: the components the format does not hold read {:08x?}",
                    &held[1..]
                ));
            }
        }
        assert!(
            differ.is_empty(),
            "{} half reading(s) of {ROWS} are not Metal's widening: {}",
            differ.len(),
            differ.join("; ")
        );
        assert_eq!(
            canonicalized, 2,
            "the codes no longer hold a NaN of each sign, so the canonicalization is unexercised"
        );
    }

    /// `MTLVertexFormatFloat`, over the same twelve specials, is a **copy**: every code reaches the
    /// shader as the bits the record holds, NaN payload, NaN sign and subnormal alike.
    ///
    /// The control for [`halfs_specials_are_metals_own_widening`], and the half of the pair that
    /// makes it a finding: the NaN payload and sign that the half fetch drops are not dropped by
    /// the read-back, the buffer, the shader or the transcript, because the float fetch of the same
    /// twelve codes carries both through. So the canonicalization belongs to the widening.
    #[test]
    fn floats_specials_reach_the_shader_as_the_bytes_they_were() {
        let case = &all()[0];
        let Some(rows) = recorded(case, VertexFormat::Float) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let list = codes(Coding::Single);
        let mut differ = Vec::new();
        for (row, held) in rows.iter().enumerate() {
            if u64::from(held[0]) != list[row] {
                differ.push(format!(
                    "row {row}: code {:#010x} read as {:#010x}",
                    list[row], held[0]
                ));
            }
        }
        assert!(
            differ.is_empty(),
            "{} float reading(s) of {ROWS} are not the bytes the record held: {}",
            differ.len(),
            differ.join("; ")
        );
        // And the two NaN codes of the list really are a payload and a sign the copy could lose.
        assert_eq!(list[9], 0x7FC0_0001);
        assert_eq!(list[10], 0xFFC0_0001);
    }

    /// The `f32` bits Metal's fetch answers for the IEEE binary16 `bits`: the exact widening, with
    /// a subnormal normalized and **a NaN canonicalized** to the positive quiet NaN, which is what
    /// `halfs_specials_are_metals_own_widening` measured Metal doing.
    fn metal_widened(bits: u16) -> u32 {
        let sign = u32::from(bits & 0x8000) << 16;
        let exponent = u32::from((bits >> 10) & 0x1F);
        let mantissa = u32::from(bits & 0x03FF);
        if exponent == 0x1F {
            // An infinity keeps its sign and an empty mantissa; a NaN keeps neither.
            return if mantissa == 0 {
                sign | 0x7F80_0000
            } else {
                0x7FC0_0000
            };
        }
        if exponent == 0 {
            if mantissa == 0 {
                return sign;
            }
            // A subnormal half is a normal float: the value is `mantissa * 2^-24`, so the exponent
            // is `127 - 24 + k` for the mantissa's highest set bit `k`, and the bits below it
            // become the float's mantissa.
            let top = 31 - mantissa.leading_zeros();
            let exponent = 127 - 24 + top;
            return sign | (exponent << 23) | ((mantissa - (1 << top)) << (23 - top));
        }
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    }

    /// The instance step fetched element 0 for every vertex: its answer is `ROWS` copies of what
    /// the per-vertex step answered at its row 0.
    ///
    /// Non-vacuous against the per-vertex step, whose twelve rows differ from each other, so an
    /// implementation that stepped per vertex where Metal steps per instance fails here. It says
    /// nothing against the **constant** step, which fetches the same element too: separating those
    /// two needs the instance index in the answer, and the module docs say what that would take.
    #[test]
    fn the_instance_step_fetched_one_element_for_every_vertex() {
        let cases = all();
        let (per_vertex, per_instance) = (&cases[0], &cases[3]);
        assert_eq!(per_instance.sweep, Sweep::PerInstance);
        let mut compared = 0usize;
        let mut differ = Vec::new();
        let mut varying = 0usize;
        for format in formats() {
            let (Some(one), Some(other)) =
                (recorded(per_vertex, format), recorded(per_instance, format))
            else {
                continue;
            };
            compared += 1;
            // The per-vertex answer varies across its rows, which is what makes the claim below a
            // claim rather than two constants agreeing.
            if one.iter().any(|held| *held != one[0]) {
                varying += 1;
            }
            for (row, held) in other.iter().enumerate() {
                if *held != one[0] {
                    differ.push(format!(
                        "{} row {row}: instance step {held:08x?}, per-vertex row 0 {:08x?}",
                        format.name(),
                        one[0]
                    ));
                }
            }
        }
        if compared == 0 {
            println!("no recording of both step functions in the tree yet");
            return;
        }
        for line in differ.iter().take(20) {
            println!("INSTANCE-STEP-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} reading(s) over {compared} format(s) are not the per-vertex step's first row",
            differ.len()
        );
        assert!(
            varying > 0,
            "no format's per-vertex answer varies across its rows, so the claim compares nothing"
        );
    }

    /// The constant step answered what the per-vertex step answered, format for format.
    ///
    /// One fetch is fixed-function vertex input on a device that has it and the other no
    /// fixed-function path can express, so the decode must not depend on which ran.
    #[test]
    fn the_two_step_functions_recorded_the_same_values() {
        let cases = all();
        let (per_vertex, constant) = (&cases[0], &cases[2]);
        assert_eq!(constant.sweep, Sweep::Constant);
        let mut compared = 0usize;
        let mut differ = Vec::new();
        for format in formats() {
            let (Some(one), Some(other)) =
                (recorded(per_vertex, format), recorded(constant, format))
            else {
                continue;
            };
            compared += 1;
            if one != other {
                differ.push(format.name());
            }
        }
        if compared == 0 {
            println!("no recording of both step functions in the tree yet");
            return;
        }
        assert!(
            differ.is_empty(),
            "{} format(s) of {compared} differ between the two step functions: {}",
            differ.len(),
            differ.join(", ")
        );
    }
}
