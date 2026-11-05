//! Two-plane YCbCr textures sampled and gathered by one function: Metal's filter and Metal's taps
//! of the texels it reads.
//!
//! A function that both samples and gathers a `YCBCR8_420_2P` or `YCBCR10_420_2P` texture through
//! one runtime sampler is the shape of a probe that asks each lane for a sample or a gather: the
//! fixture `ycg_sample_gather`, whose lane `i % 3` is a sample, a gather of red or a gather of
//! blue at position `i / 3`. Where the device's memory is the host's, the runtime reads such a
//! texture's planes in place (`metal2vulkan`'s `lower::sampler::in_place`), and that form gathers
//! as well as samples, so the slot keeps it: a sample is Metal's filter of the texels Metal reads
//! and a gather is those texels at the footprint's four taps. Elsewhere the slot binds the
//! texture's image of converted texels, which a gather reads exactly and a filter filters as the
//! device's sampler does.
//!
//! Each case is one sampler state, linear (a gather follows no filter), addressed by one mode on
//! both axes: clamp to the edge, clamp to zero, the opaque-white border, and repeat (which a 4:2:0
//! texture samples as clamp to the edge). Each runs the `float` and the `half` kernel over a `420v`
//! surface and an `x420` one, 12 by 6 texels of noise codes, at positions a third of a texel apart
//! from a texel before the texture to a texel past it, the texel centres among them, where a
//! gather's footprint turns.
//!
//! # Why it cannot fault a GPU
//!
//! Two 576-byte surfaces, one sampler and four dispatches of 3,024 threads each, with no loop.

use crate::allowance::Allowance;
use crate::case::{
    made, Abort, Bindings, Case, Failure, Input, MetalSource, PlaneTexture, SamplerSpec, Side,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
};

/// The fixture whose kernels sample and gather, and their entries: `float4`s and `half4`s.
const FIXTURE: &str = "ycg_sample_gather";
const ENTRIES: [(&str, &str, Element); 2] = [
    ("ycg_float", "float", Element::F32),
    ("ycg_half", "half", Element::F16),
];

/// The texture's extent in texels: even, so both planes' rows hold whole pairs.
const W: u64 = 12;
const H: u64 = 6;
/// The bytes between rows of either plane.
const ROW: u64 = 64;
/// The chroma plane's first byte.
const CHROMA_AT: u64 = ROW * H;
/// The surface buffer's length: the chroma plane's rows end it.
const LENGTH: usize = (CHROMA_AT + ROW * H / 2) as usize;

/// The surfaces: their OSType, their format, and the seed of their codes.
const SURFACES: [([u8; 4], PixelFormat, u64); 2] = [
    (*b"420v", PixelFormat::YCBCR84202P, 3),
    (*b"x420", PixelFormat::YCBCR104202P, 4),
];

/// Positions a third of a texel apart on each axis, from a texel before the texture to a texel
/// past it.
const COLUMNS: u64 = 3 * (W + 2);
const ROWS: u64 = 3 * (H + 2);
const POSITIONS: u64 = COLUMNS * ROWS;

/// The lanes of one position: a sample, a gather of red, a gather of blue.
const LANES: u64 = 3;

/// The threads in a threadgroup of each dispatch.
const GROUP: u32 = 64;

/// The byte the read-back buffers hold before the dispatches.
const BLANK: u8 = 0xab;

/// One case: the sampler's address mode on both axes.
pub struct YCbCrGathers {
    name: &'static str,
    mode: SamplerAddressMode,
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    [
        ("ycbcr-gathers-edge", SamplerAddressMode::ClampToEdge),
        ("ycbcr-gathers-zero", SamplerAddressMode::ClampToZero),
        (
            "ycbcr-gathers-white",
            SamplerAddressMode::ClampToBorderColor,
        ),
        ("ycbcr-gathers-repeat", SamplerAddressMode::Repeat),
    ]
    .into_iter()
    .map(|(name, mode)| Box::new(YCbCrGathers { name, mode }) as Box<dyn Case>)
    .collect()
}

/// A 64-bit LCG, the generator of the surfaces' codes.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u16 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u16
    }
}

/// The bytes of a surface of `format` from generator `seed`: every luma code and chroma pair's
/// codes noise (a 10-bit code in the high bits of its little-endian word, its low bits noise too,
/// which Metal does not read), every byte past a plane's codes `0xa5`.
fn surface_bytes(format: PixelFormat, seed: u64) -> Vec<u8> {
    let unit = match format {
        PixelFormat::YCBCR104202P => 2,
        _ => 1,
    };
    let mut random = Random(seed);
    let mut bytes = vec![0xa5; LENGTH];
    let mut put = |at: u64, word: u16| {
        let at = at as usize;
        match unit {
            1 => bytes[at] = word as u8,
            _ => bytes[at..at + 2].copy_from_slice(&word.to_le_bytes()),
        }
    };
    for y in 0..H {
        for x in 0..W {
            put(y * ROW + x * unit, random.next());
        }
    }
    for y in 0..H / 2 {
        for x in 0..W {
            put(CHROMA_AT + y * ROW + x * unit, random.next());
        }
    }
    bytes
}

/// The positions, row by row, as the `float2`s the kernels read: normalized, at texel coordinates
/// `-1 + (k + 1/2) / 3` on each axis.
fn positions() -> Vec<u8> {
    let at = |k: u64, n: u64| ((-1.0 + (k as f64 + 0.5) / 3.0) / n as f64) as f32;
    let mut bytes = Vec::with_capacity(POSITIONS as usize * 8);
    for j in 0..ROWS {
        for i in 0..COLUMNS {
            bytes.extend(at(i, W).to_le_bytes());
            bytes.extend(at(j, H).to_le_bytes());
        }
    }
    bytes
}

/// The layout a kernel's results are read under: one row a position, its three lanes' four
/// channels, so a diff names the position and the lane.
fn layout(element: Element) -> ByteLayout {
    let row_pitch = LANES as u32 * 4 * element.size();
    let length = u64::from(row_pitch) * POSITIONS;
    ByteLayout {
        region: Region::Buffer { offset: 0, length },
        element,
        row_offset: 0,
        row_elements: LANES as u32 * 4,
        row_pitch,
        rows: POSITIONS as u32,
        image_pitch: length as u32,
        images: 1,
        // A position's lanes fill its row, so there is nothing between them.
        gaps: Gaps::Ignored,
    }
}

impl YCbCrGathers {
    /// The sampler: linear on both filters, the case's mode on both axes, the opaque-white border.
    fn sampler_spec(&self) -> SamplerSpec {
        SamplerSpec::filtered(SamplerMinMagFilter::Linear)
            .addressed(self.mode)
            .bordered(SamplerBorderColor::OpaqueWhite)
    }
}

impl Case for YCbCrGathers {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration and no tolerance: the buffers the kernels write into are `Shared`
        // ([`crate::replay::shared_memory_notice`]). It loosens the notices site alone, so every
        // value still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (entry, _, _) in ENTRIES {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                let fixture = Fixture {
                    name: FIXTURE,
                    entry,
                };
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture {
            name: FIXTURE,
            entry: ENTRIES[0].0,
        };
        let mut inputs = vec![
            Input {
                name: "air:kernels".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed:kernels".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
        ];
        for (ostype, format, seed) in SURFACES {
            let name = String::from_utf8_lossy(&ostype).into_owned();
            inputs.push(Input::bytes(
                format!("format:{name}"),
                &format.as_raw().to_le_bytes(),
            ));
            inputs.push(Input::bytes(
                format!("surface:{name}"),
                &surface_bytes(format, seed),
            ));
        }
        inputs.push(Input::bytes("sampler", &self.sampler_spec().words()));
        inputs.push(Input::bytes("positions", &positions()));
        inputs.push(Input::bytes("buffer:blank", &[BLANK]));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = Fixture {
            name: FIXTURE,
            entry: ENTRIES[0].0,
        };
        let source = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&source.source()))?;
        let mut pipelines = Vec::new();
        for (entry, kind, element) in ENTRIES {
            let at = format!("pipeline/{kind}");
            let pipeline = made(sites, &at, side.compute_pipeline(library, entry))?;
            pipelines.push((pipeline, kind, element));
        }
        let sampler = made(sites, "sampler", side.sampler(&self.sampler_spec()))?;
        let positions = made(sites, "positions", side.buffer(&positions()))?;
        for (ostype, format, seed) in SURFACES {
            let name = String::from_utf8_lossy(&ostype).into_owned();
            let surface = made(
                sites,
                &format!("surface/{name}"),
                side.buffer(&surface_bytes(format, seed)),
            )?;
            let texture = made(
                sites,
                &format!("texture/{name}"),
                side.plane_texture(
                    surface,
                    &PlaneTexture {
                        ostype: Some(ostype),
                        format: format.as_raw(),
                        size: [W as u32, H as u32],
                        plane: 0,
                        planes: [(0, ROW), (CHROMA_AT, ROW)],
                    },
                ),
            )?;
            for &(pipeline, kind, element) in &pipelines {
                let length = (LANES * POSITIONS) as usize * 4 * element.size() as usize;
                let at = format!("{name}/{kind}");
                let output = made(
                    sites,
                    &format!("output/{at}"),
                    side.buffer(&vec![BLANK; length]),
                )?;
                let textures = [texture];
                let buffers = [output, positions];
                let samplers = [sampler];
                let command_buffer = made(
                    sites,
                    &format!("dispatch/{at}"),
                    side.dispatch_threads(
                        pipeline,
                        &Bindings::new(&textures, &buffers).sampled(&samplers),
                        &[],
                        [(LANES * POSITIONS) as u32, 1, 1],
                        [GROUP, 1, 1],
                    ),
                )?;
                let status = made(sites, &format!("status/{at}"), side.wait(command_buffer))?;
                if status != Value::unit_variant("Completed") {
                    return Err(Abort {
                        at: format!("status/{at}"),
                        failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
                    });
                }
                let bytes = made(
                    sites,
                    &format!("read-back/{at}"),
                    side.read_buffer(output, 0, length),
                )?;
                let digest = Digest::sha256(&bytes);
                sites.push(Node::leaf(
                    format!("samples/{at}"),
                    Observed::bytes(bytes, layout(element)).digested(digest),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four cases ask four sampler states, each a distinct input, over surfaces whose rows
    /// hold their codes: a luma row's `W` codes, a chroma row's `W / 2` pairs.
    #[test]
    fn the_cases_ask_four_states_over_two_surfaces() {
        let cases = all();
        assert_eq!(cases.len(), 4);
        let mut samplers = std::collections::BTreeSet::new();
        for case in &cases {
            let inputs = case.inputs().expect("inputs");
            let sampler = inputs
                .iter()
                .find(|i| i.name == "sampler")
                .expect("sampler");
            assert!(
                samplers.insert(format!("{:?}", sampler.digest)),
                "{}",
                case.name()
            );
        }
        for (_, format, seed) in SURFACES {
            let unit = if format == PixelFormat::YCBCR104202P {
                2
            } else {
                1
            };
            assert!(W * unit <= ROW, "{format:?}: a row's codes");
            assert_eq!(surface_bytes(format, seed).len(), LENGTH);
        }
    }

    /// The positions start a texel before the texture and end a texel past it, and every third
    /// one on each axis is a texel's centre, where a gather's footprint turns.
    #[test]
    fn the_positions_cross_the_edges_and_the_centres() {
        let bytes = positions();
        assert_eq!(bytes.len() as u64, POSITIONS * 8);
        let x = |i: usize| f32::from_le_bytes(bytes[8 * i..8 * i + 4].try_into().unwrap());
        let texel = |i: usize| f64::from(x(i)) * W as f64;
        assert!(texel(0) < 0.0, "{}", texel(0));
        assert!(
            texel(COLUMNS as usize - 1) > W as f64,
            "{}",
            texel(COLUMNS as usize - 1)
        );
        let centre = texel(1) - 0.5;
        assert!((centre - centre.round()).abs() < 1e-5, "{centre}");
    }
}
