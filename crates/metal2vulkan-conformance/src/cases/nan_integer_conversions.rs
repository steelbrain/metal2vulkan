//! MSL's float-to-integer conversions, of a NaN and of values that are not, from `float` and
//! `half` to every integer width, in a kernel and in a fragment function.
//!
//! Section 8.6 of the MSL specification: "When converting from a floating-point type to an
//! integer, the conversion uses round toward zero rounding mode", and "if the floating-point value
//! is NaN, the resulting integer is 0." Metal's compiler spells every such conversion as an
//! `air.convert` call, which the translator lowers saturating: a NaN to 0, a value past the
//! destination's range to its nearest bound. SPIR-V's own conversions (`OpConvertFToS`,
//! `OpConvertFToU`) leave both undefined, so a device answers Metal's only through that lowering,
//! and this case is what shows each one doing so.
//!
//! One case over two owned fixtures, the same program compiled two ways, because the lowering's
//! NaN test takes two forms: `nan_to_int`, without fast math, whose entries keep a float operation
//! with no fast-math flags, so its module declares `SignedZeroInfNanPreserve` where the device
//! supports it and the test reads the float; and `nan_to_int_fast`, under the default fast math
//! that Metal's programs are compiled with, whose module declares none, so the test reads the bits.
//! Each fixture runs a kernel and a fragment function, the latter one texel of a one-row
//! `RGBA32Uint` target an invocation, which a full-screen triangle covers. Every input takes five
//! invocations of each, which answer its twenty words: `char`, `uchar`, `short`, `ushort`, `int`,
//! `uint`, `long` and `ulong` of its `float`, then of its `half`.
//!
//! The inputs are eight NaNs of each width (quiet and signalling, either sign, with payloads) and
//! ten values that are not NaN: six that truncate toward zero inside the signed destinations'
//! ranges (`1.5`, `-1.5`, `-0.75`, `-2.75`, about `123.45` and about `-100.9`), both infinities,
//! and a value past `int`'s range and inside `long`'s for a float (`2^40` and its negative) or
//! past `short`'s for a half (`65504` and its negative). So a device that answered 0 everywhere
//! fails, and a value past a destination's range asks what Metal saturates it to, which MSL does
//! not state. Every word compares exactly.
//!
//! Nothing here can fault a GPU: per fixture one dispatch of 90 threads and one draw of a 90-texel
//! target, each invocation reading its input's two words and writing its own four.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, FragmentFill, Input, MetalSource, Side,
    TARGET_TEXEL,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::notice_allowance;
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

/// Inputs.
const INPUTS: usize = 18;

/// The `float` inputs, as their bits: eight NaNs, then ten values that are not.
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
    0xbf40_0000, // -0.75
    0xc030_0000, // -2.75
    0x42f6_e979, // 123.456
    0xc2c9_cccd, // -100.9
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x5380_0000, // 2^40
    0xd380_0000, // -2^40
];

/// The `half` inputs, as their bits: eight NaNs, then ten values that are not.
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
    0xba00, // -0.75
    0xc180, // -2.75
    0x57b7, // 123.4375
    0xd64e, // -100.875
    0x7c00, // +inf
    0xfc00, // -inf
    0x7bff, // 65504, the largest half
    0xfbff, // -65504
];

/// The invocations that answer one input's words.
const PARTS: usize = 5;

/// Invocations of each stage.
const INVOCATIONS: u32 = (INPUTS * PARTS) as u32;

/// The bytes of one input's answer: four words an invocation.
const ROW: u32 = PARTS as u32 * TARGET_TEXEL;

/// The bytes of one stage's answer.
const ANSWER_BYTES: usize = INPUTS * ROW as usize;

/// One conversion's answer in an input's row: its name in a diff, its destination, and where it
/// lies. The row holds the `float`'s ten words, then the `half`'s.
struct Column {
    name: &'static str,
    /// The destination's bits.
    bits: u32,
    signed: bool,
    /// Its byte offset in the row.
    offset: u32,
}

/// The twenty words of an input's row as the fixtures write them, one column a conversion: a
/// word each for the destinations of up to 32 bits (a signed one sign-extended), two for `long`
/// and `ulong`, the low first.
const COLUMNS: [Column; 16] = [
    column("char-of-float", 8, true, 0),
    column("uchar-of-float", 8, false, 4),
    column("short-of-float", 16, true, 8),
    column("ushort-of-float", 16, false, 12),
    column("int-of-float", 32, true, 16),
    column("uint-of-float", 32, false, 20),
    column("long-of-float", 64, true, 24),
    column("ulong-of-float", 64, false, 32),
    column("char-of-half", 8, true, 40),
    column("uchar-of-half", 8, false, 44),
    column("short-of-half", 16, true, 48),
    column("ushort-of-half", 16, false, 52),
    column("int-of-half", 32, true, 56),
    column("uint-of-half", 32, false, 60),
    column("long-of-half", 64, true, 64),
    column("ulong-of-half", 64, false, 72),
];

const fn column(name: &'static str, bits: u32, signed: bool, offset: u32) -> Column {
    Column {
        name,
        bits,
        signed,
        offset,
    }
}

impl Column {
    /// How a diff reads the column's element.
    fn element(&self) -> Element {
        match (self.bits, self.signed) {
            (64, _) => Element::U64,
            (_, true) => Element::I32,
            (_, false) => Element::U32,
        }
    }

    /// The layout the column is read under, in `region`: one row an input, so a diff names the
    /// input whose answer moved. Each column's value is the whole answer, so every byte offset a
    /// diff prints is the resource's own.
    fn layout(&self, region: Region) -> ByteLayout {
        ByteLayout {
            region,
            element: self.element(),
            row_offset: self.offset,
            row_elements: 1,
            row_pitch: ROW,
            rows: INPUTS as u32,
            image_pitch: ROW * INPUTS as u32,
            images: 1,
            // The columns are read apart, so each leaves the others' words outside its own
            // elements; the columns together cover every byte.
            gaps: Gaps::Ignored,
        }
    }
}

/// The input buffer: input `i`'s float bits at word `2i` and its half in the low 16 bits of word
/// `2i + 1`.
fn input_bytes() -> Vec<u8> {
    FLOATS
        .iter()
        .zip(HALVES)
        .flat_map(|(float, half)| [*float, u32::from(half)])
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// The kernel's output buffer.
fn buffer_region() -> Region {
    Region::Buffer {
        offset: 0,
        length: ANSWER_BYTES as u64,
    }
}

/// The fragment function's target, a texel an invocation in one row.
fn target_region() -> Region {
    Region::Texture {
        level: 0,
        slice: 0,
        aspect: Aspect::Color,
        sample: 0,
        origin: [0, 0, 0],
        size: [INVOCATIONS, 1, 1],
    }
}

/// The node `name` of one stage's `bytes`: one child per column.
fn output_node(name: String, bytes: &[u8], region: fn() -> Region) -> Node {
    Node::branch(
        name,
        COLUMNS
            .iter()
            .map(|column| {
                Node::leaf(
                    column.name,
                    Observed::bytes(bytes.to_vec(), column.layout(region())),
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
        // One thread an invocation, as a grid of threads in one threadgroup.
        let command_buffer = made(
            sites,
            &at("kernel/dispatch"),
            side.dispatch_threads(
                pipeline,
                &Bindings::buffers(&[input, output]),
                &[],
                [INVOCATIONS, 1, 1],
                [INVOCATIONS, 1, 1],
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
                size: [INVOCATIONS, 1],
                row_pitch: TARGET_TEXEL * INVOCATIONS,
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

/// The vertices the one draw of the full-screen triangle takes.
const VERTICES: u32 = 3;

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
        // (`crate::replay::shared_memory_notice` holds the whole reason). The fragment fill's pass
        // loads nothing into the tile, which MoltenVK announced on the M3 Max (job
        // m3-max-55436), as `rgb10a2_srgb`'s fills do.
        vec![
            crate::replay::shared_memory_notice(),
            notice_allowance(metal2vulkan_runtime::Degradation::DontCareLoadTiles.code()),
        ]
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

    /// The inputs that are NaNs: the first of each width's.
    const NANS: usize = 8;

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

    /// Input `input`'s source for `column`: its `half` for a column in the row's second half,
    /// else its `float`.
    fn source(column: &Column, input: usize) -> f64 {
        if column.offset >= ROW / 2 {
            half(HALVES[input])
        } else {
            f64::from(f32::from_bits(FLOATS[input]))
        }
    }

    /// MSL 8.6's answer of `column` for `value`, as the bits the fixture stores, where MSL states
    /// one: a NaN converts to 0, and a value whose truncation toward zero the destination holds
    /// converts to that. `None` for a value past the destination's range, whose answer MSL does
    /// not state.
    fn rule(column: &Column, value: f64) -> Option<u64> {
        if value.is_nan() {
            return Some(0);
        }
        let t = value.trunc();
        let bits = column.bits as i32;
        let (lo, hi) = if column.signed {
            (-(2f64.powi(bits - 1)), 2f64.powi(bits - 1))
        } else {
            (0.0, 2f64.powi(bits))
        };
        if t < lo || t >= hi {
            return None;
        }
        // Sign-extended to 64 bits, then cut to the column's element.
        let bits = if t < 0.0 { t as i64 as u64 } else { t as u64 };
        Some(match column.element() {
            Element::U64 => bits,
            _ => bits & 0xffff_ffff,
        })
    }

    /// Column `column` of input `input` in `bytes`.
    fn read(bytes: &[u8], column: &Column, input: usize) -> u64 {
        let at = input * ROW as usize + column.offset as usize;
        let width = if column.bits == 64 { 8 } else { 4 };
        let mut word = [0u8; 8];
        word[..width].copy_from_slice(&bytes[at..at + width]);
        u64::from_le_bytes(word)
    }

    /// The first eight inputs of each width are NaNs and the rest are not, as the module's
    /// documentation says; the inputs are distinct; and every column has rows MSL states an answer
    /// for that is not 0, so a device that answered 0 everywhere fails.
    #[test]
    fn the_inputs_are_eight_nans_then_ten_values() {
        for (index, (float, bits)) in FLOATS.iter().zip(HALVES).enumerate() {
            let nan = index < NANS;
            assert_eq!(f32::from_bits(*float).is_nan(), nan, "float {float:#010x}");
            assert_eq!(half(bits).is_nan(), nan, "half {bits:#06x}");
        }
        let floats: std::collections::BTreeSet<u32> = FLOATS.into_iter().collect();
        let halves: std::collections::BTreeSet<u16> = HALVES.into_iter().collect();
        assert_eq!((floats.len(), halves.len()), (INPUTS, INPUTS));
        for column in &COLUMNS {
            let nonzero = (NANS..INPUTS)
                .filter(|input| matches!(rule(column, source(column, *input)), Some(b) if b != 0))
                .count();
            assert!(nonzero > 0, "{}: every stated answer is 0", column.name);
        }
    }

    /// The columns tile an input's row, each word once, in the fixtures' order.
    #[test]
    fn the_columns_cover_every_word_of_a_row_once() {
        let mut covered = vec![0u32; ROW as usize / 4];
        for column in &COLUMNS {
            let words = if column.bits == 64 { 2 } else { 1 };
            for word in 0..words {
                covered[column.offset as usize / 4 + word] += 1;
            }
        }
        assert!(covered.iter().all(|times| *times == 1), "{covered:?}");
    }

    /// Metal's recorded answer, in both stages of both compiles, is MSL 8.6's wherever MSL states
    /// one: every NaN converts to 0 at every width, and every value a destination holds the
    /// truncation of converts to it. A value past a destination's range is held only to Metal's
    /// recorded answer, by the comparison.
    #[test]
    fn metal_converts_a_nan_to_zero_and_rounds_toward_zero() {
        let recording =
            crate::recording::read(NAME).unwrap_or_else(|e| panic!("{NAME}: no recording: {e}"));
        let mut checked = 0;
        for metal in &recording.answers {
            for outcome in &metal.outcomes {
                for program in &PROGRAMS {
                    for stage in ["kernel", "fragment"] {
                        let site = format!("{}/{stage}/output", program.site);
                        for column in &COLUMNS {
                            let got = outcome
                                .transcript
                                .root
                                .child(&site)
                                .and_then(|node| node.child(column.name))
                                .map(|node| &node.observed.value);
                            let Some(Value::Bytes(bytes)) = got else {
                                panic!("{NAME}: Metal's {site}/{} is {got:?}", column.name);
                            };
                            for input in 0..INPUTS {
                                let Some(want) = rule(column, source(column, input)) else {
                                    continue;
                                };
                                checked += 1;
                                assert_eq!(
                                    read(bytes, column, input),
                                    want,
                                    "{NAME}: Metal's {site}/{} of input {input}",
                                    column.name
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 0, "{NAME}: no Metal answer");
    }
}
