//! Two `ushort3` function constants read as `half3` through `as_type`.
//!
//! The owned fixture `fc_ushort3_half3` (our MSL) reads each constant through `as_type<half3>`,
//! which its compiler folds into a `<3 x half>` load of the constant's `<3 x i16>` global: a read
//! of the global through another type than its own, which the translator holds in the constant
//! blob. Under the runtime's identity domains the blob is the module's own words, with no
//! descriptor. The pipeline is made with both constants'
//! values (`newFunctionWithName:constantValues:error:`), and one dispatch of `THREADS` threads
//! stores, for thread `t`, lane `t % 3` of each constant as a float, their `half` sum, and `t`.
//! The values ([`A`], [`B`]) are halves whose sums are exact, so the answer is the program's.
//!
//! Every output byte holds `BLANK` before the dispatch, so a word the kernel did not write is a
//! recorded value. Nothing here can fault a GPU: one dispatch of 12 threads, each writing its own
//! 16 bytes of a 192-byte buffer.

use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::function_facts::{ConstantAt, ConstantValue};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites};
use metal2vulkan::metal::DataType;

/// The case.
pub struct FunctionConstantsHalf3;

/// The case's name.
pub const NAME: &str = "function-constants-read-as-half3";

/// The fixture, whose kernel has its name.
const FIXTURE: &str = "fc_ushort3_half3";

/// Threads of the one dispatch, one threadgroup.
const THREADS: u32 = 12;

/// The bytes one thread stores: a `float4`.
const STRIDE: u32 = 16;

/// Constant 0's lanes, as `half` bits: 1.0, 2.5 and -0.75.
pub const A: [u16; 3] = [0x3c00, 0x4100, 0xba00];

/// Constant 1's lanes: 0.5, 4.0 and 1.25.
pub const B: [u16; 3] = [0x3800, 0x4400, 0x3d00];

/// Each output byte before the dispatch.
const BLANK: u8 = 0xEE;

/// The two constants, each a `ushort3` at its index, its lanes packed as the translator takes
/// them.
fn constants() -> Vec<ConstantValue> {
    [(0, A), (1, B)]
        .into_iter()
        .map(|(index, lanes)| ConstantValue {
            at: ConstantAt::Index(index),
            data_type: DataType::UShort3,
            bytes: lanes.iter().flat_map(|lane| lane.to_le_bytes()).collect(),
        })
        .collect()
}

fn blank() -> Vec<u8> {
    vec![BLANK; (THREADS * STRIDE) as usize]
}

fn layout() -> ByteLayout {
    let length = THREADS * STRIDE;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(length),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: STRIDE / 4,
        row_pitch: STRIDE,
        rows: THREADS,
        image_pitch: length,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

impl Case for FunctionConstantsHalf3 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: FIXTURE.to_owned(),
                source_sha256: crate::sha256_hex(Fixture::new(FIXTURE).text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture::new(FIXTURE);
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:out", &blank()),
            Input::bytes("dispatch", &THREADS.to_le_bytes()),
        ];
        for (at, value) in constants().iter().enumerate() {
            inputs.push(Input::bytes(format!("constant:{at}"), &value.words()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline_with_constants(library, FIXTURE, &constants()),
        )?;
        let out = made(sites, "buffer/out", side.buffer(&blank()))?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                &Bindings::buffers(&[out]),
                &[],
                [THREADS, 1, 1],
                [THREADS, 1, 1],
            ),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(out, 0, blank().len()),
            |_| crate::transcript::Value::Absent,
        )?;
        sites.push(Node::leaf("words", Observed::bytes(bytes, layout())));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::Value;

    /// The program's answer: thread `t`'s lane of each constant as a float, their sum, and `t`.
    fn programs_floats() -> Vec<f32> {
        (0..THREADS)
            .flat_map(|t| {
                let lane = (t % 3) as usize;
                let (a, b) = (half_value(A[lane]), half_value(B[lane]));
                [a, b, a + b, t as f32]
            })
            .collect()
    }

    /// A normal `half`'s value (every lane here is one).
    fn half_value(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f) - 15;
        let mantissa = 1.0 + f32::from(bits & 0x3ff) / 1024.0;
        sign * mantissa * 2f32.powi(exponent)
    }

    /// The values are the floats they are stated as, and every sum is exact in `half`.
    #[test]
    fn the_constants_are_the_stated_halves_and_their_sums_exact() {
        assert_eq!(A.map(half_value), [1.0, 2.5, -0.75]);
        assert_eq!(B.map(half_value), [0.5, 4.0, 1.25]);
        assert_eq!(programs_floats().len(), (THREADS * 4) as usize);
    }

    /// Metal's recording stores the program's floats for every thread.
    #[test]
    fn metal_stores_the_programs_floats() {
        let recording =
            crate::recording::read(NAME).unwrap_or_else(|e| panic!("{NAME}: no recording: {e}"));
        let want: Vec<u8> = (programs_floats().iter())
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert!(!recording.answers.is_empty(), "{NAME}: no Metal answer");
        for metal in &recording.answers {
            assert!(!metal.outcomes.is_empty(), "{NAME}: no outcome");
            for outcome in &metal.outcomes {
                let node = (outcome.transcript.root.child("words"))
                    .unwrap_or_else(|| panic!("{NAME}: no words site"));
                let Value::Bytes(bytes) = &node.observed.value else {
                    panic!("{NAME}: the words are {:?}", node.observed.value);
                };
                assert_eq!(bytes, &want, "{NAME}: Metal's words");
            }
        }
    }
}
