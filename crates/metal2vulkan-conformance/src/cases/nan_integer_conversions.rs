//! MSL's float-to-integer conversions of a NaN, in a kernel and in a fragment function.
//!
//! Section 8.6 of the MSL specification: "When converting a floating-point type to an integer
//! type, if the floating-point value is NaN, the resulting integer is 0." Metal's compiler spells
//! every such conversion as an `air.convert` call, which the translator lowers saturating: a NaN to
//! 0, a value past the destination's range to its nearest bound. SPIR-V's own conversions
//! (`OpConvertFToS`, `OpConvertFToU`) leave both undefined, so a device answers Metal's only through
//! that lowering, and this case is what shows each one doing so.
//!
//! One case over two owned fixtures, the same program compiled two ways, because the lowering's
//! NaN test takes two forms: `nan_to_int`, without fast math, whose entries keep a float operation
//! with no fast-math flags, so its module declares `SignedZeroInfNanPreserve` where the device
//! supports it and the test reads the float; and `nan_to_int_fast`, under the default fast math
//! that Metal's programs are compiled with, whose module declares none, so the test reads the bits.
//! Each fixture runs a kernel, one thread an input, and a fragment function, one texel of a one-row
//! `RGBA32Uint` target an input, which a full-screen triangle covers. Every invocation answers four
//! words: `int(f)`, `uint(f)`, `int(h)` and `uint(h)` of the float and the half its input holds.
//!
//! The inputs are eight NaNs of each width (quiet and signalling, either sign, with payloads)
//! beside eight values that are not NaN (`1.5`, `-1.5`, both infinities, a value past `int`'s range
//! for a float and `half`'s largest for a half, zero, a fraction and `-1`), so a device that
//! answered 0 everywhere fails the case. Every word compares exactly.
//!
//! Nothing here can fault a GPU: per fixture one dispatch of sixteen threads and one draw of a
//! sixteen-texel target, each invocation reading its own two words and writing its own four.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, FragmentFill, Input, MetalSource, Side,
    TARGET_TEXEL,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;

/// The case's name.
pub const NAME: &str = "nan-to-integer-conversions";

/// The case.
pub struct NanIntegerConversions;

/// One of the two compiles of the program: the name its sites take, the fixture, and its entry
/// points.
struct Program {
    site: &'static str,
    fixture: &'static str,
    kernel: &'static str,
    vertex: &'static str,
    fragment: &'static str,
}

/// The two compiles, without fast math and with it.
const PROGRAMS: [Program; 2] = [
    Program {
        site: "precise",
        fixture: "nan_to_int",
        kernel: "nan_to_int_kernel",
        vertex: "nan_to_int_vertex",
        fragment: "nan_to_int_fragment",
    },
    Program {
        site: "fast",
        fixture: "nan_to_int_fast",
        kernel: "nan_to_int_fast_kernel",
        vertex: "nan_to_int_fast_vertex",
        fragment: "nan_to_int_fast_fragment",
    },
];

/// The `float` inputs, as their bits: eight NaNs, then eight values that are not.
const FLOATS: [u32; INPUTS] = [
    0x7fc0_0000, // quiet
    0xffc0_0000, // quiet, negative
    0x7f80_0001, // signalling, payload 1
    0xff80_0001, // signalling, negative
    0x7fff_ffff, // quiet, every payload bit
    0xffff_ffff, // the same, negative
    0x7fa0_0000, // signalling, the payload's top bit
    0x7fc1_2345, // quiet, a payload
    0x3fc0_0000, // 1.5
    0xbfc0_0000, // -1.5
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x4f80_0000, // 2^32, past both integers' ranges
    0x0000_0000, // 0
    0x42f6_e979, // 123.456
    0xbf80_0000, // -1
];

/// The `half` inputs, as their bits: eight NaNs, then eight values that are not.
const HALVES: [u16; INPUTS] = [
    0x7e00, // quiet
    0xfe00, // quiet, negative
    0x7c01, // signalling, payload 1
    0xfc01, // signalling, negative
    0x7fff, // quiet, every payload bit
    0xffff, // the same, negative
    0x7d00, // signalling, the payload's top bit
    0x7e55, // quiet, a payload
    0x3e00, // 1.5
    0xbe00, // -1.5
    0x7c00, // +inf
    0xfc00, // -inf
    0x7bff, // 65504, the largest half
    0x0000, // 0
    0x57b7, // 123.4375
    0xbc00, // -1
];

/// Inputs, so invocations of each stage.
const INPUTS: usize = 16;

/// The words an invocation answers, in the order the fixtures write them. The names are the
/// diff's: a failure says which conversion moved, not which offset.
const COLUMNS: [&str; 4] = [
    "int-of-float",
    "uint-of-float",
    "int-of-half",
    "uint-of-half",
];

/// The vertices the one draw of the full-screen triangle takes.
const VERTICES: u32 = 3;

/// The bytes of one stage's answer: four words an invocation.
const ANSWER_BYTES: usize = INPUTS * TARGET_TEXEL as usize;

/// The input buffer: invocation `i`'s float bits at word `2i` and its half in the low 16 bits of
/// word `2i + 1`.
fn input_bytes() -> Vec<u8> {
    FLOATS
        .iter()
        .zip(HALVES)
        .flat_map(|(float, half)| [*float, u32::from(half)])
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// The layout one column's words are read under, in a region of `region`: one row an
/// invocation, its four words at the target texel's pitch, so a diff names the input whose
/// answer moved. Each column's value is the whole answer, so every byte offset a diff prints is
/// the resource's own.
fn layout(region: Region, column: u32) -> ByteLayout {
    ByteLayout {
        region,
        element: Element::U32,
        row_offset: column * 4,
        row_elements: 1,
        row_pitch: TARGET_TEXEL,
        rows: INPUTS as u32,
        image_pitch: TARGET_TEXEL * INPUTS as u32,
        images: 1,
        // The columns are read apart, so each leaves the other three words outside its own
        // elements; the columns together cover every byte.
        gaps: Gaps::Ignored,
    }
}

/// The kernel's output buffer.
fn buffer_region() -> Region {
    Region::Buffer {
        offset: 0,
        length: ANSWER_BYTES as u64,
    }
}

/// The fragment function's target, one row of an input's texel each.
fn target_region() -> Region {
    Region::Texture {
        level: 0,
        slice: 0,
        aspect: Aspect::Color,
        sample: 0,
        origin: [0, 0, 0],
        size: [INPUTS as u32, 1, 1],
    }
}

/// The node `name` of one stage's `bytes`: one child per column.
fn output_node(name: String, bytes: &[u8], region: fn() -> Region) -> Node {
    Node::branch(
        name,
        (0..COLUMNS.len() as u32)
            .map(|column| {
                Node::leaf(
                    COLUMNS[column as usize],
                    Observed::bytes(bytes.to_vec(), layout(region(), column)),
                )
            })
            .collect(),
    )
}

impl Program {
    fn fixture(&self) -> Fixture {
        Fixture {
            name: self.fixture,
            entry: self.kernel,
        }
    }

    /// Run the kernel and the draw of this compile, emitting their sites under its name.
    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let at = |what: &str| format!("{}/{what}", self.site);
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: at("library"),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, &at("library"), side.library(&loaded.source()))?;
        let input = made(sites, &at("buffer/in"), side.buffer(&input_bytes()))?;

        let pipeline = made(
            sites,
            &at("kernel/pipeline"),
            side.compute_pipeline(library, self.kernel),
        )?;
        let output = made(
            sites,
            &at("kernel/buffer/out"),
            side.zeroed_buffer(ANSWER_BYTES),
        )?;
        // One thread an input, as a grid of threads in one threadgroup.
        let command_buffer = made(
            sites,
            &at("kernel/dispatch"),
            side.dispatch_threads(
                pipeline,
                &Bindings::buffers(&[input, output]),
                &[],
                [INPUTS as u32, 1, 1],
                [INPUTS as u32, 1, 1],
            ),
        )?;
        call(
            sites,
            &at("kernel/status"),
            side.wait(command_buffer),
            Clone::clone,
        )?;
        let words = call(
            sites,
            &at("kernel/read-back"),
            side.read_buffer(output, 0, ANSWER_BYTES),
            |_| Value::Absent,
        )?;
        sites.push(output_node(at("kernel/output"), &words, buffer_region));

        let filled = call(
            sites,
            &at("fragment/pass"),
            side.fragment_fill(&FragmentFill {
                library,
                vertex: self.vertex,
                fragment: self.fragment,
                format: PixelFormat::RGBA32Uint,
                size: [INPUTS as u32, 1],
                row_pitch: TARGET_TEXEL * INPUTS as u32,
                buffers: &[input],
                vertices: VERTICES,
            }),
            |filled| filled.status.clone(),
        )?;
        if filled.status != Value::unit_variant("Completed") {
            return Err(Abort {
                at: at("fragment/pass"),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&filled.status)),
            });
        }
        sites.push(output_node(
            at("fragment/output"),
            &filled.bytes,
            target_region,
        ));
        Ok(())
    }
}

impl Case for NanIntegerConversions {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every word compares exactly. Both answers come back through `Shared` buffers, so a
        // device without unified memory announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason).
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for program in &PROGRAMS {
            for entry in [program.kernel, program.vertex, program.fragment] {
                for extension in [RUNTIME_TEXT, METAL_TEXT] {
                    shaders.push(ShaderRef {
                        fixture: format!("{}{extension}", program.fixture),
                        entry: entry.to_owned(),
                        source_sha256: crate::sha256_hex(
                            program.fixture().text(extension)?.as_bytes(),
                        ),
                    });
                }
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = Vec::new();
        for program in &PROGRAMS {
            inputs.push(Input::bytes(
                format!("air:{}", program.fixture),
                program.fixture().text(RUNTIME_TEXT)?.as_bytes(),
            ));
            inputs.push(Input::bytes(
                format!("air-typed:{}", program.fixture),
                program.fixture().text(METAL_TEXT)?.as_bytes(),
            ));
        }
        inputs.push(Input::bytes("buffer:in", &input_bytes()));
        // The kernel's output starts zeroed on both sides, so a word it did not write is part of
        // the answer rather than whatever was there.
        inputs.push(Input::bytes("buffer:out", &[0u8; ANSWER_BYTES]));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for program in &PROGRAMS {
            program.run(side, sites)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSL 8.6's answer for one input: truncated toward zero, a NaN 0, a value past the
    /// destination's range its nearest bound, the signed one as its bits.
    fn rule(value: f64) -> [u32; 2] {
        if value.is_nan() {
            return [0, 0];
        }
        let t = value.trunc();
        let signed = t.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
        let unsigned = t.clamp(0.0, f64::from(u32::MAX)) as u32;
        [signed as u32, unsigned]
    }

    /// A half's value, from its bits.
    fn half(bits: u16) -> f64 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let mantissa = f64::from(bits & 0x3ff);
        sign * match exponent {
            0 => mantissa * 2f64.powi(-24),
            0x1f if mantissa == 0.0 => f64::INFINITY,
            0x1f => f64::NAN,
            _ => (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
        }
    }

    /// The four words every invocation answers, by the rule.
    fn want() -> Vec<u8> {
        FLOATS
            .iter()
            .zip(HALVES)
            .flat_map(|(float, bits)| {
                let [i, u] = rule(f64::from(f32::from_bits(*float)));
                let [hi, hu] = rule(half(bits));
                [i, u, hi, hu]
            })
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    /// The first eight inputs of each width are NaNs and the last eight are not, as the module's
    /// documentation says, and the inputs are distinct.
    #[test]
    fn the_inputs_are_eight_nans_then_eight_values() {
        for (index, (float, bits)) in FLOATS.iter().zip(HALVES).enumerate() {
            let nan = index < INPUTS / 2;
            assert_eq!(f32::from_bits(*float).is_nan(), nan, "float {float:#010x}");
            assert_eq!(half(bits).is_nan(), nan, "half {bits:#06x}");
        }
        let floats: std::collections::BTreeSet<u32> = FLOATS.into_iter().collect();
        let halves: std::collections::BTreeSet<u16> = HALVES.into_iter().collect();
        assert_eq!((floats.len(), halves.len()), (INPUTS, INPUTS));
        // The rule is not 0 everywhere: the values that are not NaN answer otherwise.
        assert!(want().iter().any(|byte| *byte != 0));
    }

    /// Metal's recorded answer, in both stages of both compiles, is MSL 8.6's: every NaN converts
    /// to 0, and every other input truncates and saturates.
    #[test]
    fn metal_converts_a_nan_to_zero() {
        let recording =
            crate::recording::read(NAME).unwrap_or_else(|e| panic!("{NAME}: no recording: {e}"));
        let want = want();
        let mut answers = 0;
        for metal in &recording.answers {
            for outcome in &metal.outcomes {
                for program in &PROGRAMS {
                    for stage in ["kernel", "fragment"] {
                        answers += 1;
                        let site = format!("{}/{stage}/output", program.site);
                        let got = outcome
                            .transcript
                            .root
                            .child(&site)
                            .and_then(|node| node.child(COLUMNS[0]))
                            .map(|node| &node.observed.value);
                        assert!(
                            matches!(got, Some(Value::Bytes(bytes)) if *bytes == want),
                            "{NAME}: Metal's {site} is {got:?}"
                        );
                    }
                }
            }
        }
        assert!(answers > 0, "{NAME}: no Metal answer");
    }
}
