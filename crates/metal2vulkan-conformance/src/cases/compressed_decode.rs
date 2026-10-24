//! How BC1 to BC5 texels decode, ported from `runtime_lane::compressed_decode`.
//!
//! The consumer's x86 guests hold BC textures, so what one texel of a block decodes to is a value a
//! guest shader reads. The lane's expectation is not a table but a **rule** fitted bit for bit on an
//! Apple M4 ([`model`]): each endpoint widened by replicating its bits, an index's weight of 64,
//! the sum `(64 - w) a + w b`, BC1 to BC3's channels rounded half up, BC4 and BC5's unrounded, an
//! sRGB channel through Metal's twelve-bit table.
//!
//! **The port re-measures it here.** Record mode runs the case on this M3 Max, so each recording
//! holds Metal's own 1,024 texels per format rather than a model's output, and the fitted rule
//! becomes a cross-check instead of the expectation
//! (the test `every_recorded_texel_is_the_lanes_fitted_rule`). That is the one thing the port adds:
//! if Apple's decode ever moves, the recording moves with it and the rule's test says so, where the
//! lane would have failed without saying which side was wrong.
//!
//! **One case per format**, so a venue that holds no texture of one answers its typed refusal there
//! and the suite counts that format as [`LacksFeature`](crate::verdict::Verdict::LacksFeature). The
//! lane walks the ten formats in one test, where a device missing one would take the whole lane
//! down.
//!
//! Each case is four 16 x 16 textures of pseudorandom blocks — the lane's own splitmix64 stream at
//! the lane's own seeds, so the bytes are the bytes the lane measured — read whole by the owned
//! fixture `format_scout`'s `scout_read_f`, one thread a texel. A signed format's first round
//! starts its first block with an endpoint of -128, which reads -1: the one hand-placed byte the
//! lane uses, kept.
//!
//! A device whose driver is not measured to decode a format as Metal does reads it through the
//! texture's decoded twin and announces `decoded-block-twin` once, which the case allows by naming
//! that code. It does **not** allow `compressed-texel-decode`: the lane asserts no BC1 to BC5
//! texture announces it, so a run that does is a divergence this case reports rather than excuses.
//! That still holds for **this** family's rows -- neither an RTX 5080 Laptop nor llvmpipe
//! announced it for any of the ten -- but it is a claim about a texture these cases read, not
//! about BC4 and BC5 everywhere: a **view** of a BC4 or BC5 carrier announces it on both of those
//! drivers, and `super::compressed_views` declares it for exactly those four rows.
//!
//! Nothing here can fault a GPU: one compute pass a round, one read-only texture, one buffer, one
//! thread a texel inside the texture, no loop and no sampling.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The fixture whose kernel reads a texture's texels as floats.
const FIXTURE: &str = "format_scout";

/// The entry point that reads a 16 x 16 texture whole, one thread a texel.
const ENTRY: &str = "scout_read_f";

/// Each texture's side, which the kernel's indexing is written for.
const SIDE: u32 = 16;

/// The texels of one texture.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The bytes one `float4` texel reads back as.
const TEXEL_BYTES: usize = 16;

/// The textures read per format, as the lane reads them.
const ROUNDS: u64 = 4;

/// The bound `docs/FAST-PATHS.md` states for `texel_conversion = TexelConversionPolicy::Device`:
/// one `f32` ulp for 8- and 16-bit codes. It is the ledger's number, copied here so a run that
/// exceeds it fails rather than being re-fitted to whatever a device answers.
const LEDGERS_TEXEL_CONVERSION_ULPS: u32 = 1;

/// How Metal decodes a format's block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// BC1: one color block.
    Bc1,
    /// BC2: explicit 4-bit alpha, then a four-color block.
    Bc2,
    /// BC3: an interpolated alpha block, then a four-color block.
    Bc3,
    /// BC4 and BC5: one or two interpolated channel blocks.
    Channels { count: usize, signed: bool },
}

impl Kind {
    /// The bytes one block takes.
    const fn block_bytes(self) -> usize {
        match self {
            Kind::Bc1
            | Kind::Channels {
                count: 1,
                signed: _,
            } => 8,
            _ => 16,
        }
    }
}

/// One format the lane reads: the case's name, the raw `MTLPixelFormat` and how Metal decodes its
/// block. Whether its color is sRGB follows from the format number ([`model::srgb`]).
struct Format {
    name: &'static str,
    /// The raw `MTLPixelFormat`, as `metal2vulkan::metal::PixelFormat` numbers it.
    /// Each row states it as `PixelFormat::<name>.as_raw()` rather than as a number, so
    /// the format a case puts to Metal is named in the case's own text and not only in a
    /// comment beside it.
    format: u32,
    kind: Kind,
}

/// Every BC1 to BC5 format, in the lane's order, because the seed of a round is the format's place
/// in that order and the bytes must stay the bytes the lane measured.
const FORMATS: [Format; 10] = [
    Format {
        name: "bc1-rgba",
        format: PixelFormat::BC1RGBA.as_raw(),
        kind: Kind::Bc1,
    },
    Format {
        name: "bc1-rgba-srgb",
        format: PixelFormat::BC1RGBASRGB.as_raw(),
        kind: Kind::Bc1,
    },
    Format {
        name: "bc2-rgba",
        format: PixelFormat::BC2RGBA.as_raw(),
        kind: Kind::Bc2,
    },
    Format {
        name: "bc2-rgba-srgb",
        format: PixelFormat::BC2RGBASRGB.as_raw(),
        kind: Kind::Bc2,
    },
    Format {
        name: "bc3-rgba",
        format: PixelFormat::BC3RGBA.as_raw(),
        kind: Kind::Bc3,
    },
    Format {
        name: "bc3-rgba-srgb",
        format: PixelFormat::BC3RGBASRGB.as_raw(),
        kind: Kind::Bc3,
    },
    Format {
        name: "bc4-r-unorm",
        format: PixelFormat::BC4RUnorm.as_raw(),
        kind: Kind::Channels {
            count: 1,
            signed: false,
        },
    },
    Format {
        name: "bc4-r-snorm",
        format: PixelFormat::BC4RSnorm.as_raw(),
        kind: Kind::Channels {
            count: 1,
            signed: true,
        },
    },
    Format {
        name: "bc5-rg-unorm",
        format: PixelFormat::BC5RGUnorm.as_raw(),
        kind: Kind::Channels {
            count: 2,
            signed: false,
        },
    },
    Format {
        name: "bc5-rg-snorm",
        format: PixelFormat::BC5RGSnorm.as_raw(),
        kind: Kind::Channels {
            count: 2,
            signed: true,
        },
    },
];

/// A splitmix64 stream of bytes: the lane's own, so a round's blocks are the lane's blocks.
struct Rng(u64);

impl Rng {
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u8
            })
            .collect()
    }
}

/// The blocks of round `round` of the format at place `at`: the lane's seed, the lane's stream, and
/// the lane's one hand-placed byte for a signed format's first round (an endpoint of -128, which
/// reads -1 and is the inference the lane's Metal venues check).
fn blocks(at: usize, round: u64, kind: Kind) -> Vec<u8> {
    let mut rng = Rng(0x5eed_bc00 + 0x100 * at as u64 + round);
    let mut blocks = rng.bytes(16 * kind.block_bytes());
    if let (0, Kind::Channels { signed: true, .. }) = (round, kind) {
        blocks[0] = 0x80;
    }
    blocks
}

/// The case for one format.
pub struct CompressedDecode {
    at: usize,
}

/// Every format's case, in the lane's order.
pub fn all() -> Vec<CompressedDecode> {
    (0..FORMATS.len())
        .map(|at| CompressedDecode { at })
        .collect()
}

impl CompressedDecode {
    /// The format the case reads.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The fixture the case runs.
    fn fixture() -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }

    /// The texture of one round: the format, 16 x 16 texels, read-only and shared, loaded with the
    /// round's blocks at four blocks a row.
    fn request<'a>(&self, uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
        TextureRequest::read_2d(self.format().format, SIDE, SIDE).loaded(uploads)
    }

    /// The bytes a row of blocks takes: four blocks of a 16-texel row.
    fn row_pitch(&self) -> u32 {
        4 * self.format().kind.block_bytes() as u32
    }

    /// The layout a round's texels read back under: one row of the layout per row of texels, four
    /// `float` channels a texel, so a diff names the texel and the channel rather than an offset.
    fn layout() -> ByteLayout {
        let pitch = SIDE * 4 * 4;
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [SIDE, SIDE, 1],
            },
            element: Element::F32,
            row_offset: 0,
            row_elements: SIDE * 4,
            row_pitch: pitch,
            rows: SIDE,
            image_pitch: pitch * SIDE,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for CompressedDecode {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A driver not measured to decode this format as Metal does reads it through the texture's
        // decoded twin, which the runtime announces once per device. The texels themselves still
        // compare exactly: the only site loosened is the set of notices, which Metal holds empty,
        // and the allowance fails the case if the notice did not fire. `compressed-texel-decode`
        // is deliberately absent — the lane asserts no BC1 to BC5 texture announces it, so a run
        // that does is a divergence to report rather than one to excuse.
        // Under the default policy a texel's `float` value is the device's conversion of the
        // twin's 8-bit codes, not Metal's exact one, which the runtime announces as
        // `device-texel-conversion`. `docs/FAST-PATHS.md`'s row for
        // `texel_conversion = TexelConversionPolicy::Device` bounds that at one `f32` ulp for 8-
        // and 16-bit codes, so the bound here is the ledger's number and not a measurement of
        // this chip: a run that exceeds one ulp is the ledger's claim failing, which is what this
        // allowance exists to catch. Under `TexelConversionPolicy::Exact` the notice does not
        // fire, the justification does not hold, and every texel is compared exactly.
        // A CPU copy that fills such a texture (`replaceRegion`) is decoded on the CPU into that
        // twin and copied in, announced as `cpu-block-decode`: the host-copy arm of the very
        // mechanism the line above names, measured on lavapipe for every format of this family. It
        // is a declaration and not a tolerance -- every texel the case reads still compares
        // exactly with Metal.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 10 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut all = vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::DecodedBlockTwin.code()),
            crate::replay::notice_allowance(Degradation::CpuBlockDecode.code()),
        ];
        for round in 0..ROUNDS {
            all.push(Allowance::new(
                format!("case/round/{round}/texels"),
                Loosening::Ulps(LEDGERS_TEXEL_CONVERSION_ULPS),
                Justification::Degradation(Degradation::DeviceTexelConversion.code().to_owned()),
            ));
        }
        all
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Self::fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: ENTRY.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
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
        // Each round's whole texture request: the descriptor and the blocks it is loaded with, so
        // a round whose format, extent, pitch or blocks move is `Unrecorded` rather than compared
        // against Metal's answer over other bytes.
        for round in 0..ROUNDS {
            let bytes = blocks(self.at, round, self.format().kind);
            let uploads = [Upload::level_0(
                [SIDE, SIDE],
                self.row_pitch(),
                SIDE / 4,
                &bytes,
            )];
            inputs.push(Input::bytes(
                format!("texture:{round}"),
                &self.request(&uploads).words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = Self::fixture()
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, loaded.entry),
        )?;
        for round in 0..ROUNDS {
            let at = |site: &str| format!("round/{round}/{site}");
            let bytes = blocks(self.at, round, self.format().kind);
            let uploads = [Upload::level_0(
                [SIDE, SIDE],
                self.row_pitch(),
                SIDE / 4,
                &bytes,
            )];
            let texture = made(sites, &at("texture"), side.texture(&self.request(&uploads)))?;
            let out = made(
                sites,
                &at("buffer"),
                side.zeroed_buffer(TEXELS * TEXEL_BYTES),
            )?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::new(&[texture], &[out]),
                    &[],
                    [SIDE, SIDE, 1],
                    [SIDE, SIDE, 1],
                ),
            )?;
            call(
                sites,
                &at("status"),
                side.wait(command_buffer),
                Clone::clone,
            )?;
            let read = call(
                sites,
                &at("read-back"),
                side.read_buffer(out, 0, TEXELS * TEXEL_BYTES),
                |_| Value::Absent,
            )?;
            sites.push(
                Node::branch(format!("round/{round}"), Vec::new())
                    .with(Node::leaf("texels", Observed::bytes(read, Self::layout()))),
            );
        }
        Ok(())
    }
}

/// Metal's decode of BC1 to BC5, as `runtime_lane::compressed_decode` fitted it on an Apple M4.
///
/// It is **not** what any case here expects: the recordings hold Metal's own texels, measured on
/// this M3 Max. The rule is kept so a test can hold the two against each other, which is the one
/// direction in which a difference tells us something — that Apple's decode moved, or that the
/// fitted rule was never Metal's.
pub mod model {
    use super::Kind;

    /// The weights of 64 of a 2-bit (thirds) and a 3-bit (sevenths) index.
    const THIRDS: [u32; 4] = [0, 21, 43, 64];
    const SEVENTHS: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
    /// The weights of the fifths of a six-value channel block.
    const FIFTHS: [u32; 6] = [0, 13, 26, 38, 51, 64];

    /// `bits` bits widened to 8 by replicating them.
    fn widen(value: u32, bits: u32) -> u32 {
        (value << (8 - bits)) | (value >> (2 * bits - 8))
    }

    /// A 565 color's three channels widened to 8 bits.
    fn rgb565(code: u32) -> [u32; 3] {
        [
            widen((code >> 11) & 31, 5),
            widen((code >> 5) & 63, 6),
            widen(code & 31, 5),
        ]
    }

    /// Whether a raw `MTLPixelFormat`'s color is sRGB, among the BC1 to BC5 family: the sRGB twin
    /// of BC1, BC2 and BC3.
    pub fn srgb(raw: u32) -> bool {
        matches!(raw, 131 | 133 | 135)
    }

    /// An 8-bit sRGB code through Metal's twelve-bit table.
    fn srgb_step(code: u32) -> u32 {
        let value = f64::from(code) / 255.0;
        let linear = if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        };
        (linear * 4095.0).round() as u32
    }

    /// Texel `j` (0 to 15, row-major in its block) of a color block: its three 8-bit codes and
    /// whether it is BC1's transparent black. `four` forces the four-color mode (BC2 and BC3).
    fn color(block: &[u8], j: usize, four: bool) -> ([u32; 3], bool) {
        let c0 = u32::from(u16::from_le_bytes([block[0], block[1]]));
        let c1 = u32::from(u16::from_le_bytes([block[2], block[3]]));
        let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let i = (indices >> (2 * j)) & 3;
        let (e0, e1) = (rgb565(c0), rgb565(c1));
        let mix = |w: u32| std::array::from_fn(|k| ((64 - w) * e0[k] + w * e1[k] + 32) >> 6);
        match (i, four || c0 > c1) {
            (0, _) => (e0, false),
            (1, _) => (e1, false),
            (2, true) => (mix(THIRDS[1]), false),
            (3, true) => (mix(THIRDS[2]), false),
            (2, false) => (mix(32), false),
            _ => ([0; 3], true),
        }
    }

    /// Texel `j` of an interpolated channel block (BC3's alpha, BC4's and BC5's channels): the sum
    /// of 64ths of its endpoints, or a six-value block's constant as `Err(high)`.
    fn channel(block: &[u8], j: usize, signed: bool) -> Result<i32, bool> {
        let end = |byte: u8| {
            if signed {
                i32::from(byte as i8)
            } else {
                i32::from(byte)
            }
        };
        let (a0, a1) = (end(block[0]), end(block[1]));
        let mut bits = [0u8; 8];
        bits[..6].copy_from_slice(&block[2..8]);
        let i = ((u64::from_le_bytes(bits) >> (3 * j)) & 7) as usize;
        let w = match (i, a0 > a1) {
            (0, _) => 0,
            (1, _) => 64,
            (_, true) => SEVENTHS[i - 1],
            (6, false) => return Err(false),
            (7, false) => return Err(true),
            (_, false) => FIFTHS[i - 1],
        } as i32;
        Ok((64 - w) * a0 + w * a1)
    }

    /// `n / d` correctly rounded to `f32` (`d` odd: no `f64` result lies on an `f32` tie).
    fn ratio(n: i32, d: u32) -> f32 {
        (f64::from(n) / f64::from(d)) as f32
    }

    /// Metal's value of texel `(x, y)` of a 16 x 16 texture of `kind` whose blocks are `blocks`
    /// (four across, row-major), `srgb` its color sRGB.
    pub fn texel(kind: Kind, srgb: bool, blocks: &[u8], x: usize, y: usize) -> [f32; 4] {
        let size = kind.block_bytes();
        let at = (y / 4) * 4 + x / 4;
        let block = &blocks[at * size..(at + 1) * size];
        let j = (y % 4) * 4 + x % 4;
        let code = |c: u32| ratio(c as i32, 255);
        let rgb = |codes: [u32; 3]| -> [f32; 3] {
            codes.map(|c| {
                if srgb {
                    ratio(srgb_step(c) as i32, 4095)
                } else {
                    code(c)
                }
            })
        };
        match kind {
            Kind::Bc1 => {
                let (codes, black) = color(block, j, false);
                let [r, g, b] = rgb(codes);
                [r, g, b, if black { 0.0 } else { 1.0 }]
            }
            Kind::Bc2 => {
                let (codes, _) = color(&block[8..], j, true);
                let alpha = (u64::from_le_bytes(block[..8].try_into().expect("eight bytes"))
                    >> (4 * j))
                    & 15;
                let [r, g, b] = rgb(codes);
                [r, g, b, code(alpha as u32 * 17)]
            }
            Kind::Bc3 => {
                let (codes, _) = color(&block[8..], j, true);
                let alpha = match channel(&block[..8], j, false) {
                    Ok(sum) => (sum as u32 + 32) >> 6,
                    Err(high) => u32::from(high) * 255,
                };
                let [r, g, b] = rgb(codes);
                [r, g, b, code(alpha)]
            }
            Kind::Channels { count, signed } => {
                let mut texel = [0.0, 0.0, 0.0, 1.0];
                for (c, value) in texel.iter_mut().take(count).enumerate() {
                    let (low, d) = if signed { (-1.0, 8128) } else { (0.0, 16320) };
                    *value = match channel(&block[8 * c..8 * c + 8], j, signed) {
                        // A signed endpoint of -128 reads -1, as its code does.
                        Ok(sum) => ratio(sum, d).max(low),
                        Err(high) => [low, 1.0][usize::from(high)],
                    };
                }
                texel
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The cases name distinct recordings, since a case's name is its recording's file name.
    #[test]
    fn the_formats_name_distinct_cases() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
    }

    /// A round's blocks fill the texture's blocks exactly: sixteen 4 x 4 blocks of a 16 x 16
    /// texture, at the pitch the upload states.
    #[test]
    fn every_rounds_blocks_fill_its_texture() {
        for (at, format) in FORMATS.iter().enumerate() {
            let case = CompressedDecode { at };
            for round in 0..ROUNDS {
                let bytes = blocks(at, round, format.kind);
                assert_eq!(
                    bytes.len(),
                    16 * format.kind.block_bytes(),
                    "{}",
                    format.name
                );
                assert_eq!(
                    bytes.len() as u32,
                    case.row_pitch() * (SIDE / 4),
                    "{}",
                    format.name
                );
            }
        }
    }

    /// The layout covers the whole read-back and no more, so the comparison reads every texel.
    #[test]
    fn the_layout_covers_the_read_back() {
        assert_eq!(CompressedDecode::layout().extent(), TEXELS * TEXEL_BYTES);
    }

    /// Every texel of every committed recording is the lane's fitted rule's.
    ///
    /// This is the cross-check the port exists for. The recordings hold what Metal answered on this
    /// M3 Max; the rule was fitted on an Apple M4. A difference means one of two things, and either
    /// is a finding rather than something to edit away: Apple's decode moved between the chips or
    /// the releases, or the rule was never Metal's.
    ///
    /// A case with no recording yet is skipped by name, so the test says which: a missing recording
    /// is record mode's work, not a silent pass.
    #[test]
    fn every_recorded_texel_is_the_lanes_fitted_rule() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for (at, format) in FORMATS.iter().enumerate() {
            let recording = match crate::recording::read(format.name) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(format.name);
                    continue;
                }
            };
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for round in 0..ROUNDS {
                        let blocks = blocks(at, round, format.kind);
                        let node = outcome
                            .transcript
                            .root
                            .child(&format!("round/{round}"))
                            .and_then(|node| node.child("texels"));
                        let Some(node) = node else {
                            differ.push(format!("{}: round {round} has no texels", format.name));
                            continue;
                        };
                        let crate::transcript::Value::Bytes(bytes) = &node.observed.value else {
                            differ.push(format!(
                                "{}: round {round}'s texels are no bytes",
                                format.name
                            ));
                            continue;
                        };
                        for texel in 0..TEXELS {
                            let want = model::texel(
                                format.kind,
                                model::srgb(format.format),
                                &blocks,
                                texel % SIDE as usize,
                                texel / SIDE as usize,
                            );
                            for (channel, wanted) in want.iter().enumerate() {
                                let at_byte = texel * TEXEL_BYTES + channel * 4;
                                let word = u32::from_le_bytes(
                                    bytes[at_byte..at_byte + 4].try_into().expect("four bytes"),
                                );
                                checked += 1;
                                if word != wanted.to_bits() && differ.len() < 24 {
                                    differ.push(format!(
                                        "{} round {round} texel {texel} channel {channel}: Metal \
                                         {} against the rule's {wanted}",
                                        format.name,
                                        f32::from_bits(word),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} channels checked; Metal's recorded texels differ from the lane's fitted \
             rule:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
