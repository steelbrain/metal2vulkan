//! Metal's bicubic filter through a constexpr sampler, over one texture of each class of texels
//! the filter treats apart.
//!
//! The owned fixture `bcu_formats` (our MSL) samples ten `texture2d` slots at each lane's
//! normalized coordinate through one constexpr `filter::bicubic, address::clamp_to_edge,
//! coord::normalized` sampler, an implicit sample (in a kernel, the base level): slots 0 to 7 are
//! `float` textures and slots 8 and 9 `half` ones, whose samples it widens to `float`. Each lane
//! stores its ten samples, slot by slot.
//!
//! What Metal's filter does with its sum depends on the bound texture's pixel format, so the case
//! binds one texture of each class it treats apart: `R32Float` (32-bit float channels),
//! `RGBA8Unorm`, `RGBA16Unorm` and `R8Unorm` (normalized unsigned channels, of two widths and of
//! one channel and four), `RGBA8Snorm` (normalized signed channels), `RGBA16Float` (half
//! channels), and `RGBA8Unorm_sRGB` (8-bit sRGB codes). Each class's row binds its texture at
//! every slot, so each texture is sampled with `float` and with `half` results; the `mixed` row
//! binds the first six at slots `s mod 6`, so one dispatch samples six classes at once; the sRGB
//! row is last.
//!
//! Every texture holds the same texels (`texel`), each the nearest code of its format: a step
//! from 0 to 1 along `x` in red (a texel earlier on odd rows), one along `y` in green, a checker of
//! 0.25 and 0.75 in blue, and two lone texels of 1 in alpha. A bicubic sum over a step rings below
//! 0 and above 1, which is where a clamp into `[0, 1]`, a clamp into `[-1, 1]` and no clamp part,
//! and the checker's inner values are where the formats' codes part. The coordinates
//! (`coords`) are off every texel's centre (a quarter of a texel past its left edge, five
//! eighths past its top edge), every edge and corner texel among them, and ten more near the
//! corners and outside `[0, 1]`, where the filter's taps are clamped to the edge.
//!
//! Every output byte holds `FILL` before a dispatch, so a word the kernel did not write is a
//! recorded value. Nothing here can fault a GPU: one compute pass a row, textures of 6 by 5 texels
//! of shader-read usage, two buffers a row, one thread a lane, no loop, and every lane writes its
//! own 160 bytes of the output.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, Side, TextureRole,
    TextureSpec,
};
use crate::degradation::Degradation;
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::notice_allowance;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;

/// The case.
pub struct BicubicTexelClasses;

/// The case's name.
pub const NAME: &str = "bicubic-texel-classes";

/// The fixture, whose kernel has its name.
const FIXTURE: &str = "bcu_formats";

/// The kernel's texture slots; the last two are `half` textures.
const SLOTS: usize = 10;

/// The textures' width and height, in texels: at least four on each axis, so a sample is the
/// filter's sixteen taps rather than the linear form Metal takes below that.
const SIZE: [u32; 2] = [6, 5];

/// Threads a threadgroup.
const GROUP: u32 = 8;

/// The bytes one lane stores: ten `float4`s.
const LANE: u32 = SLOTS as u32 * 16;

/// Each output byte before a dispatch.
const FILL: u8 = 0xEE;

/// The coordinates past the texel grid: near the corners, and outside `[0, 1]` on one axis and on
/// both, where the taps are clamped to the edge.
const OUTSIDE: [[f32; 2]; 10] = [
    [-0.25, 0.5],
    [1.25, 0.5],
    [0.5, -0.3125],
    [0.5, 1.3125],
    [-0.5, -0.5],
    [1.5, 1.5],
    [-0.0625, 1.0625],
    [1.0625, -0.0625],
    [0.03125, 0.96875],
    [0.96875, 0.03125],
];

/// A texture of one class: its row's name, its pixel format, its bytes a texel, and its texel's
/// bytes for the four channels' values.
struct Class {
    name: &'static str,
    format: PixelFormat,
    bytes: u32,
    encode: fn([f32; 4]) -> Vec<u8>,
}

/// The nearest 8-bit unsigned normalized code of `v`.
fn unorm8(v: f32) -> u8 {
    (v * 255.0).round() as u8
}

/// The `half` bits of `v`, a value a `half` holds exactly: zero, or a normal one.
fn half_bits(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((bits >> 16) & 0x8000) as u16 | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// The classes, each texture's row in this order; the sRGB one is last.
const CLASSES: [Class; 7] = [
    Class {
        name: "r32float",
        format: PixelFormat::R32Float,
        bytes: 4,
        encode: |v| v[0].to_le_bytes().to_vec(),
    },
    Class {
        name: "rgba8unorm",
        format: PixelFormat::RGBA8Unorm,
        bytes: 4,
        encode: |v| v.map(unorm8).to_vec(),
    },
    Class {
        name: "rgba8snorm",
        format: PixelFormat::RGBA8Snorm,
        bytes: 4,
        encode: |v| v.map(|c| (c * 127.0).round() as i8 as u8).to_vec(),
    },
    Class {
        name: "rgba16float",
        format: PixelFormat::RGBA16Float,
        bytes: 8,
        encode: |v| v.iter().flat_map(|&c| half_bits(c).to_le_bytes()).collect(),
    },
    Class {
        name: "rgba16unorm",
        format: PixelFormat::RGBA16Unorm,
        bytes: 8,
        encode: |v| {
            (v.iter())
                .flat_map(|&c| ((c * 65535.0).round() as u16).to_le_bytes())
                .collect()
        },
    },
    Class {
        name: "r8unorm",
        format: PixelFormat::R8Unorm,
        bytes: 1,
        encode: |v| vec![unorm8(v[0])],
    },
    Class {
        name: "rgba8unorm-srgb",
        format: PixelFormat::RGBA8UnormSRGB,
        bytes: 4,
        encode: |v| v.map(unorm8).to_vec(),
    },
];

/// The value of texel `(x, y)`'s four channels, before its format's coding.
fn texel(x: u32, y: u32) -> [f32; 4] {
    let step = |on: bool| if on { 1.0 } else { 0.0 };
    [
        step(x + y % 2 >= 3),
        step(y >= 2),
        if (x + y) % 2 == 0 { 0.75 } else { 0.25 },
        step(matches!((x, y), (1, 1) | (4, 3))),
    ]
}

/// A class's texels, row by row, at its width's pitch.
fn texels(class: &Class) -> Vec<u8> {
    (0..SIZE[1])
        .flat_map(|y| (0..SIZE[0]).flat_map(move |x| (class.encode)(texel(x, y))))
        .collect()
}

/// Each lane's coordinate: a quarter of a texel past each texel's left edge and five eighths past
/// its top edge, row by row, then [`OUTSIDE`].
fn coords() -> Vec<[f32; 2]> {
    let [w, h] = SIZE;
    let mut at: Vec<[f32; 2]> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| [(x as f32 + 0.25) / w as f32, (y as f32 + 0.625) / h as f32])
        })
        .collect();
    at.extend(OUTSIDE);
    at
}

/// The lanes of a dispatch.
fn lanes() -> u32 {
    coords().len() as u32
}

/// The coordinate buffer.
fn coord_bytes() -> Vec<u8> {
    coords()
        .iter()
        .flatten()
        .flat_map(|c| c.to_le_bytes())
        .collect()
}

/// A row's output buffer before its dispatch.
fn blank() -> Vec<u8> {
    vec![FILL; (lanes() * LANE) as usize]
}

/// Each row, by name, with the class bound at each slot.
fn rows() -> Vec<(&'static str, [usize; SLOTS])> {
    let mut rows: Vec<_> = (0..6).map(|c| (CLASSES[c].name, [c; SLOTS])).collect();
    rows.push(("mixed", std::array::from_fn(|slot| slot % 6)));
    rows.push((CLASSES[6].name, [6; SLOTS]));
    rows
}

/// The layout a row's output is read under: a lane's ten `float4`s a row.
fn layout() -> ByteLayout {
    let length = lanes() * LANE;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(length),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: LANE / 4,
        row_pitch: LANE,
        rows: lanes(),
        image_pitch: length,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

impl Case for BicubicTexelClasses {
    fn name(&self) -> &'static str {
        NAME
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The runtime emulates Metal's bicubic footprint and announces that its arithmetic is its
        // own (`bicubic-filter-arithmetic`): a declaration, which loosens no value. The output is
        // a `Shared` buffer, so a device without unified memory announces
        // `shared-in-system-memory` (`crate::replay::shared_memory_notice` holds the whole reason).
        vec![
            crate::replay::shared_memory_notice(),
            notice_allowance(Degradation::BicubicFilterArithmetic.code()),
        ]
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
            Input::bytes("buffer:coords", &coord_bytes()),
            Input::bytes("buffer:out", &blank()),
            Input::bytes("dispatch", &[lanes(), GROUP].map(u32::to_le_bytes).concat()),
        ];
        for class in &CLASSES {
            inputs.push(Input::bytes(
                format!("texture:{}", class.name),
                &texels(class),
            ));
        }
        for (name, slots) in rows() {
            let bound: Vec<u8> = slots.iter().map(|&c| c as u8).collect();
            inputs.push(Input::bytes(format!("bound:{name}"), &bound));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(sites, "pipeline", side.compute_pipeline(library, FIXTURE))?;
        let mut textures: Vec<Handle> = Vec::new();
        for class in &CLASSES {
            let spec =
                TextureSpec::new(class.format, SIZE, SIZE[0] * class.bytes, TextureRole::Read);
            let at = format!("texture/{}", class.name);
            textures.push(made(
                sites,
                &at,
                side.texture_2d_in_role(&spec, &texels(class)),
            )?);
        }
        let coordinates = made(sites, "buffer/coords", side.buffer(&coord_bytes()))?;
        for (name, slots) in rows() {
            let at = |site: &str| format!("{name}/{site}");
            let output = made(sites, &at("buffer/out"), side.buffer(&blank()))?;
            let bound: Vec<Handle> = slots.iter().map(|&c| textures[c]).collect();
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::new(&bound, &[coordinates, output]),
                    &[],
                    [lanes(), 1, 1],
                    [GROUP, 1, 1],
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
                side.read_buffer(output, 0, blank().len()),
                |_| Value::Absent,
            )?;
            sites.push(
                Node::branch(name.to_owned(), Vec::new())
                    .with(Node::leaf("sampled", Observed::bytes(bytes, layout()))),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each committed Metal answer's samples, by row: `samples[row][(lane * SLOTS + slot) * 4 +
    /// channel]`.
    fn recorded() -> Vec<std::collections::BTreeMap<String, Vec<f32>>> {
        let recording =
            crate::recording::read(NAME).unwrap_or_else(|e| panic!("{NAME}: no recording: {e}"));
        let mut answers = Vec::new();
        for metal in &recording.answers {
            assert!(!metal.outcomes.is_empty(), "{NAME}: no outcome");
            for outcome in &metal.outcomes {
                let mut rows = std::collections::BTreeMap::new();
                for (name, _) in rows_named() {
                    let node = (outcome.transcript.root.child(name))
                        .and_then(|row| row.child("sampled"))
                        .unwrap_or_else(|| panic!("{NAME}: no {name}/sampled site"));
                    let Value::Bytes(bytes) = &node.observed.value else {
                        panic!("{NAME}: {name}'s samples are {:?}", node.observed.value);
                    };
                    let floats: Vec<f32> = (bytes.as_chunks::<4>().0.iter())
                        .map(|&b| f32::from_le_bytes(b))
                        .collect();
                    assert_eq!(
                        floats.len(),
                        (lanes() * LANE / 4) as usize,
                        "{NAME}: {name}"
                    );
                    rows.insert(name.to_owned(), floats);
                }
                answers.push(rows);
            }
        }
        assert!(!answers.is_empty(), "{NAME}: no Metal answer");
        answers
    }

    /// The rows' names, each with its slots' classes.
    fn rows_named() -> Vec<(&'static str, [usize; SLOTS])> {
        rows()
    }

    /// Sample `(lane, slot, channel)` of a row's samples.
    fn sample(row: &[f32], lane: u32, slot: usize, channel: usize) -> f32 {
        row[(lane as usize * SLOTS + slot) * 4 + channel]
    }

    /// `v` rounded to the nearest `half`, ties to even (every value here is a normal `float` or
    /// zero, inside `half`'s range).
    fn to_half(v: f32) -> f32 {
        if v == 0.0 {
            return v;
        }
        let exponent = (((v.to_bits() >> 23) & 0xff) as i32 - 127).max(-14);
        let quantum = 2f32.powi(exponent - 10);
        (v / quantum).round_ties_even() * quantum
    }

    /// The texels are the stated codes: the steps' 0 and 1 exact in every format, and the checker's
    /// inner values each format's nearest code.
    #[test]
    fn the_texels_are_each_formats_codes() {
        assert_eq!(texel(2, 0), [0.0, 0.0, 0.75, 0.0]);
        assert_eq!(texel(2, 1), [1.0, 0.0, 0.25, 0.0]);
        assert_eq!(texel(4, 3), [1.0, 1.0, 0.25, 1.0]);
        let bytes = |name: &str| texels(CLASSES.iter().find(|c| c.name == name).unwrap());
        assert_eq!(&bytes("rgba8unorm")[8..12], &[0, 0, 191, 0]);
        assert_eq!(&bytes("rgba8snorm")[8..12], &[0, 0, 95, 0]);
        assert_eq!(&bytes("rgba16float")[16..24], &[0, 0, 0, 0, 0, 0x3a, 0, 0]);
        assert_eq!(
            &bytes("rgba16unorm")[16..24],
            &[0, 0, 0, 0, 0xff, 0xbf, 0, 0]
        );
        for class in &CLASSES {
            assert_eq!(
                texels(class).len() as u32,
                SIZE[0] * SIZE[1] * class.bytes,
                "{}",
                class.name
            );
        }
        assert_eq!(lanes(), 40);
    }

    /// Metal's answers take each class's own arm, read off the red step: where the `RGBA16Float`
    /// sum rings above 1 or below 0 (no clamp), each normalized unsigned format reads 1 and 0
    /// (the clamp into `[0, 1]`), `RGBA8Snorm` reads 1 and keeps the value below 0 (the clamp into
    /// `[-1, 1]`), and `R32Float`, whose filter is not the sixteen taps, stays inside `[0, 1]`
    /// and parts from the `RGBA16Float` answer.
    #[test]
    fn metals_classes_take_their_own_arms() {
        for rows in recorded() {
            let red = |row: &str, lane: u32| sample(&rows[row], lane, 0, 0);
            let over: Vec<u32> = (0..lanes())
                .filter(|&l| red("rgba16float", l) > 1.0)
                .collect();
            let under: Vec<u32> = (0..lanes())
                .filter(|&l| red("rgba16float", l) < 0.0)
                .collect();
            assert!(!over.is_empty() && !under.is_empty(), "{NAME}: no ringing");
            for row in ["rgba8unorm", "rgba16unorm", "r8unorm"] {
                assert!(
                    (0..lanes()).all(|l| (0.0..=1.0).contains(&red(row, l))),
                    "{row}"
                );
                assert!(over.iter().all(|&l| red(row, l) == 1.0), "{row}");
                assert!(under.iter().all(|&l| red(row, l) == 0.0), "{row}");
            }
            assert!((0..lanes()).all(|l| red("rgba8snorm", l) <= 1.0));
            assert!(over.iter().all(|&l| red("rgba8snorm", l) == 1.0));
            assert!(under.iter().all(|&l| red("rgba8snorm", l) < 0.0));
            assert!((0..lanes()).all(|l| (0.0..=1.0).contains(&red("r32float", l))));
            assert!((0..lanes()).any(|l| red("r32float", l) != red("rgba16float", l)));
        }
    }

    /// No two classes' rows give one answer, and the `mixed` row's every slot answers what its
    /// class's own row answered there: each slot's treatment is its own texture's.
    #[test]
    fn metals_rows_part_and_the_mixed_row_is_its_slots_classes() {
        for rows in recorded() {
            for (i, a) in CLASSES.iter().enumerate() {
                for b in &CLASSES[i + 1..] {
                    assert_ne!(rows[a.name], rows[b.name], "{} and {}", a.name, b.name);
                }
            }
            let (_, mixed) = rows_named()[6];
            for (slot, &class) in mixed.iter().enumerate() {
                for lane in 0..lanes() {
                    for c in 0..4 {
                        assert_eq!(
                            sample(&rows["mixed"], lane, slot, c).to_bits(),
                            sample(&rows[CLASSES[class].name], lane, slot, c).to_bits(),
                            "slot {slot}, lane {lane}, channel {c}"
                        );
                    }
                }
            }
        }
    }

    /// In every class's row the `half` slots answer the `float` slots' sample rounded to the
    /// nearest `half`, and the `float` slots answer alike.
    #[test]
    fn metals_half_slots_are_the_float_samples_rounded_to_half() {
        for rows in recorded() {
            for class in &CLASSES {
                let row = &rows[class.name];
                for lane in 0..lanes() {
                    for c in 0..4 {
                        let float = sample(row, lane, 0, c);
                        for slot in 1..8 {
                            assert_eq!(sample(row, lane, slot, c).to_bits(), float.to_bits());
                        }
                        for slot in 8..SLOTS {
                            assert_eq!(
                                sample(row, lane, slot, c).to_bits(),
                                to_half(float).to_bits(),
                                "{}: lane {lane}, channel {c}",
                                class.name
                            );
                        }
                    }
                }
            }
        }
    }
}
