//! What Metal converts an unnormalized 8- or 16-bit integer vertex format to in a float attribute.
//!
//! The owned fixture `dsx_int_attributes` holds two `vertex void` functions,
//! `int_attribute_to_float4` and `int_attribute_to_half4`, each of which writes the one attribute
//! it fetched to `out[vertex_id]`: the first as the `float4` it declared, the second as its `half4`
//! widened exactly. Buffer 1 holds eight 256-byte records, and row `r`'s component `c` of the
//! format at byte `16 i` holds the probe's value `(r + 3 c) % 8` of that width and signedness; the
//! rest of each record is `0xEE`, so a fetch that reads past its components reads a value no table
//! holds.
//!
//! Sixteen formats, two functions and two step functions: 64 pipelines, each with a vertex
//! descriptor whose attribute 0 is the format at byte `16 i` of buffer 1. Metal's conversion is
//! the subject — each component reads as its integer value, zero- or sign-extended, as a float, the
//! components the format does not hold as `(0, 0, 0, 1)`, and a `half` attribute rounds to nearest
//! even with 65520 and above going to infinity — and the case takes it from a run rather than from
//! the lane's table.
//!
//! A `vertex void` function has no position to rasterize, so each pipeline disables rasterization
//! and carries no fragment stage, and each pass has no attachment of any kind and states its own
//! one-by-one extent. That is the one shape Metal accepts for such a function.
//!
//! The two step functions ask different questions of the same conversion. Per vertex, one draw of
//! eight points fetches record `r` for vertex `r`, which a device may do with fixed-function
//! vertex input. With Metal's constant step, eight draws of one point each fetch the one record the
//! draw's binding offset names, which no fixed-function path can express, so the shader pulls it.
//! Both must answer the same values, and the case compares them both.
//!
//! All 64 pipelines' draws go into one command buffer, each writing its own 128-byte slice of a
//! single read-back buffer, so the case costs one commit and one wait. A row's site is its slice
//! under a layout whose row is one `float4`, so a diff names the row and the component.
//!
//! Nothing here can fault a GPU: no rasterization, no attachment, eight points a draw, and every
//! fetch is inside a record the case bound.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, VertexAttribute,
    VertexBufferLayout, VertexLayout,
};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PrimitiveType, VertexFormat, VertexStepFunction};

/// The fixture every function of the case lives in.
const FIXTURE: &str = "dsx_int_attributes";

/// The two conversions: the entry point, and whether its attribute is a `half4`.
const CONVERSIONS: [(&str, bool); 2] = [
    ("int_attribute_to_float4", false),
    ("int_attribute_to_half4", true),
];

/// Each format with its component bytes, component count and signedness, in the order the records
/// lay them out.
const FORMATS: [(VertexFormat, usize, usize, bool); 16] = [
    (VertexFormat::UChar, 1, 1, false),
    (VertexFormat::UChar2, 1, 2, false),
    (VertexFormat::UChar3, 1, 3, false),
    (VertexFormat::UChar4, 1, 4, false),
    (VertexFormat::Char, 1, 1, true),
    (VertexFormat::Char2, 1, 2, true),
    (VertexFormat::Char3, 1, 3, true),
    (VertexFormat::Char4, 1, 4, true),
    (VertexFormat::UShort, 2, 1, false),
    (VertexFormat::UShort2, 2, 2, false),
    (VertexFormat::UShort3, 2, 3, false),
    (VertexFormat::UShort4, 2, 4, false),
    (VertexFormat::Short, 2, 1, true),
    (VertexFormat::Short2, 2, 2, true),
    (VertexFormat::Short3, 2, 3, true),
    (VertexFormat::Short4, 2, 4, true),
];

/// The rows each conversion writes, which is the records the buffer holds.
const ROWS: u32 = 8;

/// Bytes one record takes: one 16-byte slot per format.
const STRIDE: u32 = 16 * FORMATS.len() as u32;

/// Bytes one row of a conversion's answer takes: one `float4`.
const ROW_BYTES: u32 = 16;

/// Bytes one conversion's whole answer takes, which is its slice of the read-back buffer.
const ANSWER_BYTES: u32 = ROWS * ROW_BYTES;

/// The two step functions the case asks each conversion under.
const STEPS: [VertexStepFunction; 2] =
    [VertexStepFunction::PerVertex, VertexStepFunction::Constant];

/// The probe's values of each width and signedness, which the records hold.
fn values(bytes: usize, signed: bool) -> [i64; 8] {
    match (bytes, signed) {
        (1, false) => [0, 1, 2, 127, 128, 200, 254, 255],
        (1, true) => [0, 1, -1, 127, -128, -2, 100, -100],
        (_, false) => [0, 1, 2049, 2051, 65504, 65519, 65520, 65535],
        (_, true) => [0, -1, 32767, -32768, 2049, -2051, 1000, -1000],
    }
}

/// The records buffer 1 holds: row `r`'s component `c` of format `i` at byte `16 i + c * bytes`,
/// and `0xEE` everywhere else, so a fetch that reads past the format's components reads a value no
/// table holds.
fn records() -> Vec<u8> {
    let mut out = vec![0xEE; (STRIDE * ROWS) as usize];
    for row in 0..ROWS as usize {
        for (at, &(_, bytes, count, signed)) in FORMATS.iter().enumerate() {
            for component in 0..count {
                let value = values(bytes, signed)[(row + 3 * component) % 8];
                let first = row * STRIDE as usize + 16 * at + component * bytes;
                out[first..first + bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
            }
        }
    }
    out
}

/// The name of the step function, as the site spells it.
fn step_name(step: VertexStepFunction) -> &'static str {
    match step {
        VertexStepFunction::Constant => "constant",
        _ => "per-vertex",
    }
}

/// Every draw the case makes: each conversion of each format under each step function.
fn rows() -> Vec<(usize, usize, VertexStepFunction)> {
    let mut out = Vec::new();
    for conversion in 0..CONVERSIONS.len() {
        for format in 0..FORMATS.len() {
            for step in STEPS {
                out.push((conversion, format, step));
            }
        }
    }
    out
}

/// The site one draw's answer is emitted at.
fn row_name(conversion: usize, format: usize, step: VertexStepFunction) -> String {
    let (entry, _) = CONVERSIONS[conversion];
    let (held, ..) = FORMATS[format];
    format!("{entry}/{held:?}/{}", step_name(step))
}

/// Attribute 0 of format `at`: that format at byte `16 at` of buffer 1.
fn attributes_of(at: usize) -> [VertexAttribute; 1] {
    [VertexAttribute::new(0, FORMATS[at].0, 16 * at as u32, 1)]
}

/// Buffer 1's layout under `step`: one record of [`STRIDE`] bytes.
fn layouts_of(step: VertexStepFunction) -> [VertexBufferLayout; 1] {
    [match step {
        VertexStepFunction::Constant => VertexBufferLayout::constant(1, STRIDE),
        _ => VertexBufferLayout::per_vertex(1, STRIDE),
    }]
}

/// The layout one conversion's answer is read under: one row of the layout per row of the records,
/// one element per component, so a diff names the row and the component.
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
        rows: ROWS,
        image_pitch: ANSWER_BYTES,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The case.
pub struct IntAttributes;

/// The case the suite holds.
pub fn all() -> Vec<IntAttributes> {
    vec![IntAttributes]
}

impl IntAttributes {
    /// The fixture under the entry point `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }
}

impl Case for IntAttributes {
    fn name(&self) -> &'static str {
        "integer-vertex-attributes"
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // No value: Metal's conversion of an integer format to a float attribute is exact, and
        // both step functions must answer the same values on any device. A row that differs is a
        // bug in the fetch, not a tolerance.
        //
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for this case and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (entry, _) in CONVERSIONS {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Self::fixture(entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture(CONVERSIONS[0].0);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The records are the whole input table: every value every row fetches, and the 0xEE
            // that says a fetch read past its components.
            Input::bytes("records", &records()),
        ];
        // One fingerprint over all 64 pipelines, because they are generated from two constants: a
        // format, a step or an offset that moved moved the whole table.
        let mut pipelines = Vec::new();
        for (conversion, format, step) in rows() {
            pipelines.extend(row_name(conversion, format, step).into_bytes());
            let attributes = attributes_of(format);
            let layouts = layouts_of(step);
            pipelines.extend(
                RenderPipelineSpec::converting(
                    StageFunction::named(CONVERSIONS[conversion].0),
                    VertexLayout::new(&attributes, &layouts),
                )
                .words(),
            );
        }
        inputs.push(Input::bytes("pipelines", &pipelines));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture(CONVERSIONS[0].0)
            .loaded()
            .map_err(|why| Abort {
                at: "library".to_owned(),
                failure: Failure::named("FixtureNotRead", why),
            })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let records = made(sites, "buffer/records", side.buffer(&records()))?;
        let rows = rows();
        let out = made(
            sites,
            "buffer/answers",
            side.zeroed_buffer(rows.len() * ANSWER_BYTES as usize),
        )?;

        let mut pipelines: Vec<Handle> = Vec::with_capacity(rows.len());
        for &(conversion, format, step) in &rows {
            let name = row_name(conversion, format, step);
            let attributes = attributes_of(format);
            let layouts = layouts_of(step);
            pipelines.push(made(
                sites,
                &format!("{name}/pipeline"),
                side.render_pipeline(
                    library,
                    &RenderPipelineSpec::converting(
                        StageFunction::named(CONVERSIONS[conversion].0),
                        VertexLayout::new(&attributes, &layouts),
                    ),
                ),
            )?);
        }

        // One command buffer: a pass per pipeline, each writing its own slice of the one read-back
        // buffer, so the case costs one commit and one wait rather than 64.
        let mut work = Vec::with_capacity(rows.len());
        for (at, (&(_, _, step), pipeline)) in rows.iter().zip(&pipelines).enumerate() {
            let base = at as u64 * u64::from(ANSWER_BYTES);
            let mut steps = vec![RenderStep::Pipeline(*pipeline)];
            if step == VertexStepFunction::Constant {
                // The constant step fetches the one element the binding's offset names, so each
                // row is its own draw of one point, bound at its own record.
                for row in 0..ROWS {
                    steps.push(RenderStep::VertexBuffer {
                        buffer: out,
                        offset: base + u64::from(row * ROW_BYTES),
                        index: 0,
                    });
                    steps.push(RenderStep::VertexBuffer {
                        buffer: records,
                        offset: u64::from(row * STRIDE),
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
                    count: ROWS,
                    instances: 1,
                });
            }
            work.push(RenderWork::Pass(RenderPass::sized([1, 1]).drawing(steps)));
        }
        let outcome = call(sites, "render", side.render(&work), |outcome| {
            outcome.status.clone()
        })?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(out, 0, rows.len() * ANSWER_BYTES as usize),
            |_| Value::Absent,
        )?;

        for (at, &(conversion, format, step)) in rows.iter().enumerate() {
            let name = row_name(conversion, format, step);
            let first = at * ANSWER_BYTES as usize;
            let answer = bytes[first..first + ANSWER_BYTES as usize].to_vec();
            let mut row = Node::branch(name, Vec::new())
                .with(Node::leaf("rows", Observed::bytes(answer, answer_layout())));
            // What the row's pass answered, which is `None` on Metal's side: a pass of ours that
            // refuses compares against Metal encoding and reads as a refusal rather than a
            // divergence.
            if let Some(Some(failure)) = outcome.refused.get(at) {
                row = row.with(Node::leaf("pass", Observed::value(failure.recorded())));
            }
            sites.push(row);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Metal's `half` of the integer `value`: rounded to nearest even at 11 significant bits, and
    /// 65520 and above to infinity. Copied from `runtime_lane::int_attributes::metal_half` to be
    /// checked against the recording, never to stand in for it.
    fn metal_half(value: i64) -> f32 {
        let size = value.unsigned_abs();
        let magnitude = if size >= 65520 {
            f32::INFINITY
        } else {
            let bits = 64 - size.leading_zeros();
            let rounded = if bits <= 11 {
                size
            } else {
                let shift = bits - 11;
                let (quotient, rest, half) = (
                    size >> shift,
                    size & ((1 << shift) - 1),
                    1u64 << (shift - 1),
                );
                let quotient = if rest > half || (rest == half && quotient & 1 == 1) {
                    quotient + 1
                } else {
                    quotient
                };
                quotient << shift
            };
            rounded as f32
        };
        if value < 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    /// Metal's `float4` for row `row` of format `at`, read into a `float4` or a `half4` attribute:
    /// each component its integer value as a float, the ones the format does not hold `(0, 0, 0,
    /// 1)`. The lane's rule, which the recording is checked against.
    fn metal_row(at: usize, row: usize, half: bool) -> [f32; 4] {
        let (_, bytes, count, signed) = FORMATS[at];
        let mut out = [0.0f32, 0.0, 0.0, 1.0];
        for (component, held) in out.iter_mut().enumerate().take(count) {
            let value = values(bytes, signed)[(row + 3 * component) % 8];
            *held = if half {
                metal_half(value)
            } else {
                value as f32
            };
        }
        out
    }

    /// The case's bookkeeping holds: every row names a distinct site, the records cover every
    /// format's slot, and the layout covers one answer exactly.
    #[test]
    fn the_rows_and_the_layout_agree_with_the_buffers() {
        let rows = rows();
        assert_eq!(rows.len(), CONVERSIONS.len() * FORMATS.len() * STEPS.len());
        let named: BTreeSet<String> = rows
            .iter()
            .map(|&(conversion, format, step)| row_name(conversion, format, step))
            .collect();
        assert_eq!(named.len(), rows.len());
        assert_eq!(answer_layout().extent(), ANSWER_BYTES as usize);
        assert_eq!(records().len(), (STRIDE * ROWS) as usize);
        // Every format's slot is 16 bytes and its components fit inside it, so no two formats'
        // values overlap in a record.
        for &(_, bytes, count, _) in &FORMATS {
            assert!(bytes * count <= 16);
        }
    }

    /// Every answer Metal recorded holds exactly the values the lane's rule says Metal converts
    /// to, format for format, row for row, and under both step functions.
    ///
    /// This is the cross-check the port owes the lane: the lane carried the rule, the case carries
    /// Metal's own floats, and this says whether they agree. It reads the recording, so it runs
    /// where there is no GPU.
    #[test]
    fn metal_s_recorded_conversions_are_the_lane_s_values() {
        let case = IntAttributes;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let mut checked = 0usize;
        let mut differ = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                for (conversion, format, step) in rows() {
                    let name = row_name(conversion, format, step);
                    let Some(Value::Bytes(bytes)) = outcome
                        .transcript
                        .root
                        .child(&name)
                        .and_then(|node| node.child("rows"))
                        .map(|node| &node.observed.value)
                    else {
                        continue;
                    };
                    checked += 1;
                    for (row, held) in bytes.chunks(16).enumerate() {
                        let got = [0, 1, 2, 3].map(|component| {
                            let first = component * 4;
                            f32::from_le_bytes([
                                held[first],
                                held[first + 1],
                                held[first + 2],
                                held[first + 3],
                            ])
                        });
                        let want = metal_row(format, row, CONVERSIONS[conversion].1);
                        if got.map(f32::to_bits) != want.map(f32::to_bits) {
                            differ.push(format!(
                                "{name} row {row}: Metal {got:?}, the lane's rule {want:?}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(checked > 0, "the recording holds no row");
        for line in differ.iter().take(20) {
            println!("INT-ATTR-DIFFERS: {line}");
        }
        assert!(
            differ.is_empty(),
            "{} value(s) of {checked} recorded answers differ from the lane's rule",
            differ.len()
        );
    }

    /// The two step functions answer the same values, in the recording: the fetch's path may be
    /// fixed function one way and pulled the other, and the conversion must not depend on which.
    #[test]
    fn the_two_step_functions_recorded_the_same_values() {
        let case = IntAttributes;
        let Ok(recording) = crate::recording::read(case.name()) else {
            println!("{}: no recording in the tree yet", case.name());
            return;
        };
        let mut compared = 0usize;
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let held = |name: &str| match outcome
                    .transcript
                    .root
                    .child(name)
                    .and_then(|node| node.child("rows"))
                    .map(|node| &node.observed.value)
                {
                    Some(Value::Bytes(bytes)) => Some(bytes.clone()),
                    _ => None,
                };
                for conversion in 0..CONVERSIONS.len() {
                    for format in 0..FORMATS.len() {
                        let per_vertex =
                            held(&row_name(conversion, format, VertexStepFunction::PerVertex));
                        let constant =
                            held(&row_name(conversion, format, VertexStepFunction::Constant));
                        if let (Some(per_vertex), Some(constant)) = (per_vertex, constant) {
                            compared += 1;
                            assert_eq!(
                                per_vertex,
                                constant,
                                "{} differs between its step functions",
                                row_name(conversion, format, VertexStepFunction::PerVertex)
                            );
                        }
                    }
                }
            }
        }
        assert!(compared > 0, "the recording holds no pair");
    }
}
