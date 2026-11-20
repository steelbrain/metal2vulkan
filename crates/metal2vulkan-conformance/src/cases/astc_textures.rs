//! How ASTC texels decode.
//!
//! No desktop GPU and no lavapipe holds an ASTC texture in its own format. Where a device holds a
//! format's block, the runtime keeps the blocks in an image of it and a guest's read is the
//! device's decode, the static note `PrecisionNote::CompressedTexelDecode`
//! (`compressed-texel-decode`). Where it holds none, the runtime makes no texture of the format
//! (`Gap::TextureFormat`), and the case reads as the device lacking it.
//!
//! **The recordings hold Metal's own texels**, recorded on this M3 Max.
//! `metal2vulkan::bind::AstcDecode`, the translator's model of Metal's decode, is a cross-check
//! against them (the test `every_recorded_texel_is_the_decoders`): a difference says either that
//! Apple's ASTC decode is not what the model implements, or that it moved.
//!
//! # The three profiles, and what each one compares
//!
//! Forty-two cases: the fourteen 2D footprints in each of ASTC's three profiles. Each compares
//! Apple's decode against the device's, with nothing of ours in between, on a device whose format
//! query holds the format's block (for the `_HDR` ones `ASTC_*_SFLOAT_BLOCK`,
//! `textureCompressionASTC_HDR`, Vulkan 1.3 core or `VK_EXT_texture_compression_astc_hdr`).
//!
//! ## Which footprints the sweep can tell apart, measured
//!
//! Of the two rounds, **`seeded-blocks` distinguishes all fourteen footprints**: the fourteen HDR
//! recordings hold fourteen distinct texel images, because the footprint decides how many blocks a
//! 16 x 16 level takes and therefore which of the seeded stream's bytes land where.
//!
//! **`zero-blocks` distinguishes none of them**, and that is by construction rather than by
//! oversight: a zero block is an error block in every profile and at every footprint, so every
//! texel of every one of the fourteen reads the same error colour and the fourteen recordings are
//! byte-identical. The round still compares something — Metal's error-block colour against the
//! device's, which is the one value a device answers for a malformed block — but it is not what
//! tells one footprint from another, and a sweep that had only this round would be comparing one
//! answer fourteen times.
//!
//! **One case per format**, so a venue that holds no texture of one answers its typed refusal there
//! and the suite counts that format as [`LacksFeature`](crate::verdict::Verdict::LacksFeature).
//!
//! The footprint itself comes from [`metal2vulkan::metal::format::block_extent`], the translator's
//! public query: the footprint is a fact about the `MTLPixelFormat` that every profile shares.
//!
//! Each case is two 16 x 16 textures, read whole by the owned fixture `format_scout`'s
//! `scout_read_f`, one thread a texel: one of **zero blocks**, which are error blocks and read
//! magenta (stated as an upload so the bytes are the case's own rather than whatever a fresh
//! texture holds), and one of seeded xorshift blocks — mostly error blocks, the rest decoding
//! through every mode, and every fourth an LDR **void-extent** block of a random colour. Both are
//! loaded by `replaceRegion:` as the texture is made. A 16-texel side over the fourteen footprints
//! gives sixteen blocks of a 4 x 4 format down to four of a 12 x 12 one, each level's blocks
//! covering texels past its edge.
//!
//! **Tolerances.** Only a linear (`_LDR`) format has one: its channel is the device's decode of the
//! block, whose `compressed-texel-decode` catalogue entry (docs/API.md) bounds an ASTC LDR channel
//! at **1/2048**. The sRGB and `_HDR` forms name none, and a value a device's decode moves there is
//! a `known-gaps-rt.tsv` row of its venue.
//!
//! Nothing here can fault a GPU: one compute pass a read, one read-only texture, one buffer, one
//! thread a texel inside the texture, no loop and no sampling.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::degradation::Degradation;
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::bind::{AstcDecode, AstcTexels};
use metal2vulkan::metal::{format::block_extent, PixelFormat};

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

/// What each round's texture is loaded with, and the name its site takes. A round is a whole
/// texture of its own, so a diff names which blocks moved.
const ROUNDS: [&str; 2] = ["zero-blocks", "seeded-blocks"];

/// The most a linear channel of the device's own decode of an ASTC LDR block may be off Metal's
/// value, as the `compressed-texel-decode` catalogue entry states it (docs/API.md): 1/2048, the
/// largest difference measured over the fourteen footprints' random blocks, plus the `float`
/// rounding of both values. It is the entry's number, so a run past it is the note's claim failing.
const DEVICE_DECODE_BOUND: f64 = 1.0 / 2048.0 + f32::EPSILON as f64 / 2.0;

/// An LDR void-extent block of red, green, blue and alpha `0x4000`, `0x8000`, `0xc000` and
/// `0xffff`: every texel of the block that colour. The seeded stream puts its first eight bytes in
/// every fourth block.
const VOID_EXTENT: [u8; 16] = [
    0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x40, 0x00, 0x80, 0x00, 0xc0, 0xff, 0xff,
];

/// One format the case reads: its name and the `MTLPixelFormat`. Its footprint follows
/// from the format ([`metal2vulkan::metal::format::block_extent`]) and its profile from whether
/// the decoder has an answer for it ([`AstcDecode::of`]), so there is no second spelling of either
/// to keep in step.
struct Format {
    name: &'static str,
    format: PixelFormat,
}

/// Every ASTC format, in a fixed order, because the seed of a round is the format's place in that
/// order and the bytes must stay the bytes the recordings hold: the fourteen linear footprints,
/// then the fourteen sRGB ones, then the fourteen HDR ones.
///
/// The HDR fourteen are appended rather than interleaved for exactly that reason — a place in this
/// list is a seed, so inserting a format anywhere but the end would change the blocks of every
/// format after it and invalidate twenty-eight committed recordings for no change in subject.
const FORMATS: [Format; 42] = [
    Format {
        name: "astc-4x4-ldr",
        format: PixelFormat::ASTC4x4LDR,
    },
    Format {
        name: "astc-5x4-ldr",
        format: PixelFormat::ASTC5x4LDR,
    },
    Format {
        name: "astc-5x5-ldr",
        format: PixelFormat::ASTC5x5LDR,
    },
    Format {
        name: "astc-6x5-ldr",
        format: PixelFormat::ASTC6x5LDR,
    },
    Format {
        name: "astc-6x6-ldr",
        format: PixelFormat::ASTC6x6LDR,
    },
    Format {
        name: "astc-8x5-ldr",
        format: PixelFormat::ASTC8x5LDR,
    },
    Format {
        name: "astc-8x6-ldr",
        format: PixelFormat::ASTC8x6LDR,
    },
    Format {
        name: "astc-8x8-ldr",
        format: PixelFormat::ASTC8x8LDR,
    },
    Format {
        name: "astc-10x5-ldr",
        format: PixelFormat::ASTC10x5LDR,
    },
    Format {
        name: "astc-10x6-ldr",
        format: PixelFormat::ASTC10x6LDR,
    },
    Format {
        name: "astc-10x8-ldr",
        format: PixelFormat::ASTC10x8LDR,
    },
    Format {
        name: "astc-10x10-ldr",
        format: PixelFormat::ASTC10x10LDR,
    },
    Format {
        name: "astc-12x10-ldr",
        format: PixelFormat::ASTC12x10LDR,
    },
    Format {
        name: "astc-12x12-ldr",
        format: PixelFormat::ASTC12x12LDR,
    },
    Format {
        name: "astc-4x4-srgb",
        format: PixelFormat::ASTC4x4SRGB,
    },
    Format {
        name: "astc-5x4-srgb",
        format: PixelFormat::ASTC5x4SRGB,
    },
    Format {
        name: "astc-5x5-srgb",
        format: PixelFormat::ASTC5x5SRGB,
    },
    Format {
        name: "astc-6x5-srgb",
        format: PixelFormat::ASTC6x5SRGB,
    },
    Format {
        name: "astc-6x6-srgb",
        format: PixelFormat::ASTC6x6SRGB,
    },
    Format {
        name: "astc-8x5-srgb",
        format: PixelFormat::ASTC8x5SRGB,
    },
    Format {
        name: "astc-8x6-srgb",
        format: PixelFormat::ASTC8x6SRGB,
    },
    Format {
        name: "astc-8x8-srgb",
        format: PixelFormat::ASTC8x8SRGB,
    },
    Format {
        name: "astc-10x5-srgb",
        format: PixelFormat::ASTC10x5SRGB,
    },
    Format {
        name: "astc-10x6-srgb",
        format: PixelFormat::ASTC10x6SRGB,
    },
    Format {
        name: "astc-10x8-srgb",
        format: PixelFormat::ASTC10x8SRGB,
    },
    Format {
        name: "astc-10x10-srgb",
        format: PixelFormat::ASTC10x10SRGB,
    },
    Format {
        name: "astc-12x10-srgb",
        format: PixelFormat::ASTC12x10SRGB,
    },
    Format {
        name: "astc-12x12-srgb",
        format: PixelFormat::ASTC12x12SRGB,
    },
    Format {
        name: "astc-4x4-hdr",
        format: PixelFormat::ASTC4x4HDR,
    },
    Format {
        name: "astc-5x4-hdr",
        format: PixelFormat::ASTC5x4HDR,
    },
    Format {
        name: "astc-5x5-hdr",
        format: PixelFormat::ASTC5x5HDR,
    },
    Format {
        name: "astc-6x5-hdr",
        format: PixelFormat::ASTC6x5HDR,
    },
    Format {
        name: "astc-6x6-hdr",
        format: PixelFormat::ASTC6x6HDR,
    },
    Format {
        name: "astc-8x5-hdr",
        format: PixelFormat::ASTC8x5HDR,
    },
    Format {
        name: "astc-8x6-hdr",
        format: PixelFormat::ASTC8x6HDR,
    },
    Format {
        name: "astc-8x8-hdr",
        format: PixelFormat::ASTC8x8HDR,
    },
    Format {
        name: "astc-10x5-hdr",
        format: PixelFormat::ASTC10x5HDR,
    },
    Format {
        name: "astc-10x6-hdr",
        format: PixelFormat::ASTC10x6HDR,
    },
    Format {
        name: "astc-10x8-hdr",
        format: PixelFormat::ASTC10x8HDR,
    },
    Format {
        name: "astc-10x10-hdr",
        format: PixelFormat::ASTC10x10HDR,
    },
    Format {
        name: "astc-12x10-hdr",
        format: PixelFormat::ASTC12x10HDR,
    },
    Format {
        name: "astc-12x12-hdr",
        format: PixelFormat::ASTC12x12HDR,
    },
];

/// `count` 16-byte blocks of a seeded xorshift, so a round's blocks are the same on every run:
/// random bits, of which most are error blocks and the rest decode through every mode,
/// every fourth an LDR void-extent block of a random colour.
fn seeded_blocks(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let mut blocks = Vec::with_capacity(16 * count);
    for index in 0..count {
        let mut block = [0; 16];
        block[..8].copy_from_slice(&next().to_le_bytes());
        block[8..].copy_from_slice(&next().to_le_bytes());
        if index % 4 == 3 {
            block[..8].copy_from_slice(&VOID_EXTENT[..8]);
        }
        blocks.extend_from_slice(&block);
    }
    blocks
}

/// The blocks across and down a 16 x 16 level of the footprint `extent`: a block covering texels
/// past the level's edge counts, as Metal counts it.
fn blocks_across_and_down(extent: [u32; 2]) -> [u32; 2] {
    extent.map(|side| SIDE.div_ceil(side))
}

/// The blocks of round `round` of the format at place `at`: zeros, which are error blocks, then
/// the seeded stream at the seed for that format's place.
fn blocks(at: usize, round: usize, extent: [u32; 2]) -> Vec<u8> {
    let [across, down] = blocks_across_and_down(extent);
    let count = (across * down) as usize;
    match round {
        0 => vec![0; count * AstcDecode::BLOCK_BYTES as usize],
        _ => seeded_blocks(0xa57c_0000 + 0x10 * at as u64, count),
    }
}

/// The `float4` Metal reads of each texel of a `SIDE` x `SIDE` level of `decode`'s `blocks`, as
/// the decoder under test computes it.
///
/// This is `metal_texels`, kept as a **cross-check against the recording** rather
/// than as any case's expectation: the recordings hold Metal's answer, and the test
/// `every_recorded_texel_is_the_decoders` holds the two against each other.
pub fn decoders_texels(decode: AstcDecode, blocks: &[u8]) -> Vec<[f32; 4]> {
    decode
        .decode(AstcTexels::Float, blocks, [SIDE, SIDE, 1])
        .as_chunks::<16>()
        .0
        .iter()
        .map(|texel| {
            std::array::from_fn(|c| {
                f32::from_le_bytes(texel[4 * c..4 * c + 4].try_into().expect("four bytes"))
            })
        })
        .collect()
}

/// The case for one format.
pub struct AstcTextures {
    at: usize,
}

/// Every format's case, in `FORMATS` order.
pub fn all() -> Vec<AstcTextures> {
    (0..FORMATS.len()).map(|at| AstcTextures { at }).collect()
}

impl AstcTextures {
    /// The format the case reads.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The format's footprint, which every ASTC format of every profile answers.
    fn footprint(&self) -> [u32; 2] {
        block_extent(self.format().format).expect("an ASTC format")
    }

    /// How the format's blocks decode on a device with no ASTC block: every case's format has an
    /// answer, LDR, sRGB or HDR.
    fn decode(&self) -> AstcDecode {
        AstcDecode::of(self.format().format).expect("an ASTC format of a Vulkan footprint")
    }

    /// The fixture the case runs.
    fn fixture() -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }

    /// The bytes a row of blocks takes: the blocks across a 16-texel row, 16 bytes each.
    fn row_pitch(&self) -> u32 {
        blocks_across_and_down(self.footprint())[0] * AstcDecode::BLOCK_BYTES
    }

    /// The block rows a 16-texel level holds, which is what an upload hands rather than its texel
    /// rows: the difference the compressed families are about.
    fn block_rows(&self) -> u32 {
        blocks_across_and_down(self.footprint())[1]
    }

    /// The texture of one round: the format, 16 x 16 texels, read-only and shared, loaded with the
    /// round's blocks at the footprint's own pitch.
    fn request<'a>(&self, uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
        TextureRequest::read_2d(self.format().format.as_raw(), SIDE, SIDE).loaded(uploads)
    }

    /// Whether this format's channels are linear LDR ones, the only ones the device's decode
    /// may move (`compressed-texel-decode`); an sRGB or HDR form's texels compare exactly.
    fn linear(&self) -> bool {
        let decode = self.decode();
        !decode.srgb && !decode.hdr
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

impl Case for AstcTextures {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Where the device holds the format's block itself, the runtime keeps the blocks in an
        // image of it and reads the device's decode of them, the static note
        // `compressed-texel-decode` of every block-compressed texture it makes. Here it is a
        // declaration; a linear form's texels also take the entry's bound (below), and an sRGB or
        // HDR form's compare exactly.
        let mut all = vec![crate::replay::notice_allowance(
            Degradation::CompressedTexelDecode.code(),
        )];
        // A linear channel is the device's decode of the block, the static note
        // `compressed-texel-decode`, whose catalogue entry bounds an ASTC LDR channel at 1/2048.
        if self.linear() {
            for round in ROUNDS {
                all.push(Allowance::new(
                    format!("case/{round}/texels"),
                    Loosening::Absolute(DEVICE_DECODE_BOUND),
                    Justification::notice(Degradation::CompressedTexelDecode),
                ));
            }
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
        for (round, name) in ROUNDS.iter().enumerate() {
            let bytes = blocks(self.at, round, self.footprint());
            let uploads = [Upload::level_0(
                [SIDE, SIDE],
                self.row_pitch(),
                self.block_rows(),
                &bytes,
            )];
            inputs.push(Input::bytes(
                format!("texture:{name}"),
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
        for (round, name) in ROUNDS.iter().enumerate() {
            let at = |site: &str| format!("{name}/{site}");
            let bytes = blocks(self.at, round, self.footprint());
            let uploads = [Upload::level_0(
                [SIDE, SIDE],
                self.row_pitch(),
                self.block_rows(),
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
                Node::branch((*name).to_owned(), Vec::new())
                    .with(Node::leaf("texels", Observed::bytes(read, Self::layout()))),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The profile a case's format decodes with, from the decoder's own answer.
    fn profile(decode: AstcDecode) -> &'static str {
        match (decode.srgb, decode.hdr) {
            (true, _) => "srgb",
            (_, true) => "hdr",
            _ => "ldr",
        }
    }

    /// The cases name distinct recordings, since a case's name is its recording's file name.
    #[test]
    fn the_formats_name_distinct_cases() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
    }

    /// Every case names a format with a footprint, and the forty-two cases cover every one of the
    /// fourteen footprints in all three profiles exactly once.
    ///
    /// The profile comes from the decoder's `srgb` and `hdr` flags. ASTC's other two footprints
    /// (4x2 and 8x4) have no decoder, and they are not here — [`AstcTextures::decode`] expects
    /// one of every case.
    #[test]
    fn the_cases_are_every_footprint_in_all_three_profiles() {
        let mut seen = BTreeSet::new();
        for (at, format) in FORMATS.iter().enumerate() {
            let case = AstcTextures { at };
            assert!(
                seen.insert((case.footprint(), profile(case.decode()))),
                "{}: a second case of its footprint and profile",
                format.name
            );
        }
        assert_eq!(seen.len(), 42);
        // Fourteen footprints, each in three profiles.
        assert_eq!(
            seen.iter()
                .map(|(extent, _)| *extent)
                .collect::<BTreeSet<_>>()
                .len(),
            14
        );
    }

    /// A case's name states the footprint and the profile its format decodes with, so a reader of
    /// a failure is not trusting the table's spelling.
    #[test]
    fn every_cases_name_states_its_own_footprint() {
        for (at, format) in FORMATS.iter().enumerate() {
            let case = AstcTextures { at };
            let [width, height] = case.footprint();
            let space = profile(case.decode());
            assert_eq!(format.name, format!("astc-{width}x{height}-{space}"));
        }
    }

    /// A round's blocks fill the level's blocks exactly, at the pitch the upload states.
    #[test]
    fn every_rounds_blocks_fill_its_texture() {
        for (at, format) in FORMATS.iter().enumerate() {
            let case = AstcTextures { at };
            let [across, down] = blocks_across_and_down(case.footprint());
            for round in 0..ROUNDS.len() {
                let bytes = blocks(at, round, case.footprint());
                assert_eq!(
                    bytes.len() as u32,
                    across * down * AstcDecode::BLOCK_BYTES,
                    "{}",
                    format.name
                );
                assert_eq!(
                    bytes.len() as u32,
                    case.row_pitch() * case.block_rows(),
                    "{}",
                    format.name
                );
            }
        }
    }

    /// The two rounds hand different blocks, so neither is the other's measurement under another
    /// name.
    #[test]
    fn the_rounds_hand_different_blocks() {
        for (at, format) in FORMATS.iter().enumerate() {
            let extent = AstcTextures { at }.footprint();
            assert_ne!(
                blocks(at, 0, extent),
                blocks(at, 1, extent),
                "{}",
                format.name
            );
        }
    }

    /// The seeded stream puts the void-extent block's first eight bytes in every fourth
    /// block, so a round's blocks are not error blocks throughout and the cross-check reads
    /// decoded colour as well as magenta.
    #[test]
    fn every_fourth_seeded_block_is_a_void_extent() {
        for (at, format) in FORMATS.iter().enumerate() {
            let extent = AstcTextures { at }.footprint();
            let bytes = blocks(at, 1, extent);
            let mut extents = 0;
            for (index, block) in bytes.as_chunks::<16>().0.iter().enumerate() {
                if index % 4 == 3 {
                    assert_eq!(block[..8], VOID_EXTENT[..8], "{}", format.name);
                    extents += 1;
                }
            }
            let [across, down] = blocks_across_and_down(extent);
            assert_eq!(extents, (across * down) as usize / 4, "{}", format.name);
        }
    }

    /// Only the linear formats name a tolerance, and each names the bound the
    /// `compressed-texel-decode` entry states.
    ///
    /// This is the ruling that a notice declaration is not a tolerance, held as a test: the
    /// `compressed-texel-decode` declaration is on every case and loosens no value, while the one
    /// allowance that loosens a texel is named only where the entry says a texel moves.
    #[test]
    fn only_the_linear_formats_loosen_a_texel() {
        let mut loosened = 0;
        for (at, format) in FORMATS.iter().enumerate() {
            let case = AstcTextures { at };
            let bounds: Vec<Loosening> = case
                .allowances()
                .into_iter()
                .filter(|allowance| allowance.path.ends_with("/texels"))
                .map(|allowance| allowance.loosening)
                .collect();
            // An sRGB or HDR form loosens no texel; only the fourteen linear ones do, each round
            // by the device decode's bound.
            if !case.linear() {
                assert!(bounds.is_empty(), "{}: {bounds:?}", format.name);
                continue;
            }
            let expected = vec![Loosening::Absolute(DEVICE_DECODE_BOUND); ROUNDS.len()];
            assert_eq!(bounds, expected, "{}", format.name);
            loosened += 1;
        }
        assert_eq!(loosened, 14);
    }

    /// The layout covers the whole read-back and no more, so the comparison reads every texel.
    #[test]
    fn the_layout_covers_the_read_back() {
        assert_eq!(AstcTextures::layout().extent(), TEXELS * TEXEL_BYTES);
    }

    /// Every texel of every committed recording is `metal2vulkan::bind::AstcDecode`'s.
    ///
    /// The recordings hold what Metal answered on an Apple M3 Max; the decoder is what every guest
    /// shader on a device without the ASTC block actually reads. A difference is a finding rather
    /// than something to edit away:
    /// either Apple's decode is not what we implement, or it moved.
    ///
    /// A case with no recording yet is named, so a missing recording is record mode's work rather
    /// than a silent pass.
    #[test]
    fn every_recorded_texel_is_the_decoders() {
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
            let decode = (AstcTextures { at }).decode();
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for (round, name) in ROUNDS.iter().enumerate() {
                        let blocks = blocks(at, round, decode.block_extent());
                        let node = outcome
                            .transcript
                            .root
                            .child(name)
                            .and_then(|node| node.child("texels"));
                        let Some(node) = node else {
                            differ.push(format!("{}: {name} has no texels", format.name));
                            continue;
                        };
                        let Value::Bytes(bytes) = &node.observed.value else {
                            differ.push(format!("{}: {name}'s texels are no bytes", format.name));
                            continue;
                        };
                        let want = decoders_texels(decode, &blocks);
                        for (texel, want) in want.iter().enumerate() {
                            for (channel, wanted) in want.iter().enumerate() {
                                let at_byte = texel * TEXEL_BYTES + channel * 4;
                                let word = u32::from_le_bytes(
                                    bytes[at_byte..at_byte + 4].try_into().expect("four bytes"),
                                );
                                checked += 1;
                                if word != wanted.to_bits() && differ.len() < 24 {
                                    differ.push(format!(
                                        "{} {name} texel {texel} channel {channel}: Metal {} \
                                         against AstcDecode's {wanted}",
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
            "{checked} channels checked; Metal's recorded texels differ from \
             metal2vulkan::bind::AstcDecode:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
