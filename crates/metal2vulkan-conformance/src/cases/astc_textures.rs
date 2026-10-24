//! How ASTC texels decode, ported from `runtime_lane::astc_textures`'s first lane.
//!
//! An ASTC texture is a texture no desktop GPU and no lavapipe holds in its own format, so the
//! runtime keeps its blocks in a carrier image and its reads go through a **decoded twin** it fills
//! itself, announcing `carried-decoded-twin` once per device. What a guest shader then reads is not
//! the device's decode but **ours**, which makes every texel of such a texture a value this
//! implementation computes rather than one a driver answers.
//!
//! **The port is the first independent check of that decoder.** `metal2vulkan::bind::AstcDecode` is
//! today checked against its own tests and against one hand-copied measurement in the lane: the
//! lane computes Metal's expected texels *from the decoder itself* (its own `metal_texels`) and
//! compares our read against that, so a decoder that is wrong in the same way on both sides passes.
//! Here record mode runs the case on this M3 Max and the recording holds **Metal's own texels**.
//! The decoder then becomes a cross-check against the oracle (the test
//! `every_recorded_texel_is_the_decoders`), which is the one direction in which a difference says
//! something: either Apple's ASTC decode is not what we implement, or it moved.
//!
//! # The three profiles, and what each one compares
//!
//! Forty-two cases: the fourteen 2D footprints in each of ASTC's three profiles. The `_LDR` and
//! `_sRGB` ones are the port described above, where the subject is **our** decoder, because no
//! desktop GPU and no lavapipe holds an ASTC block and the texels a guest reads are ours.
//!
//! The fourteen `_HDR` ones are a different comparison, and a stronger one where the venue has the
//! block. This crate has **no** ASTC HDR decode (`AstcDecode::of` holds the LDR and sRGB profiles
//! alone), so there is no decoded twin: a device either holds `ASTC_*_SFLOAT_BLOCK` natively —
//! which needs `textureCompressionASTC_HDR`, Vulkan 1.3 core or
//! `VK_EXT_texture_compression_astc_hdr` — or refuses the texture, and the site is then the
//! venue's own gap rather than a refusal of ours. Where it does hold it, the case compares
//! **Apple's ASTC HDR decode against the driver's**, with nothing of ours in between, which is a
//! thing no other case in this suite does. On MoltenVK over this M3 Max (the extension and the
//! feature both present) all fourteen are exact over both rounds, every texel and every channel.
//!
//! That is also why an HDR case names **no allowance at all**: the twin notices
//! (`carried-decoded-twin`, `cpu-block-decode`) and the compact tolerance
//! (`decoded-texel-precision`) are all facts about a twin, and an HDR texture has none.
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
//! byte-identical. The round still compares something — Metal's error-block colour against ours,
//! which is the one value a device answers for a malformed block — but it is not what tells one
//! footprint from another, and a sweep that had only this round would be comparing one answer
//! fourteen times.
//!
//! **One case per format**, so a venue that holds no texture of one answers its typed refusal there
//! and the suite counts that format as [`LacksFeature`](crate::verdict::Verdict::LacksFeature). The
//! lane walks all twenty-eight formats in one test, where a device missing one would take the whole
//! lane down.
//!
//! The footprint itself comes from [`metal2vulkan::metal::format::block_extent`], the translator's
//! public query, and not from the decoder: the decoder holds two of the three profiles, while the
//! footprint is a fact about the `MTLPixelFormat` that every profile shares. Until that query
//! existed this family could state only the formats `AstcDecode::of` answered for, which is what
//! kept the fourteen HDR formats out of it.
//!
//! Each case is two 16 x 16 textures, read whole by the owned fixture `format_scout`'s
//! `scout_read_f`, one thread a texel: one of **zero blocks**, which are error blocks and read
//! magenta (the lane's "fresh" read, stated as an upload so the bytes are the case's own rather
//! than whatever a fresh texture holds), and one of the lane's own seeded xorshift blocks at the
//! lane's own seed — mostly error blocks, the rest decoding through every mode, and every fourth an
//! LDR **void-extent** block of a random colour. Both are loaded by `replaceRegion:` as the texture
//! is made, which is the lane's CPU-copy write. A 16-texel side over the fourteen footprints gives
//! sixteen blocks of a 4 x 4 format down to four of a 12 x 12 one, each level's blocks covering
//! texels past its edge, which is the lane's shape.
//!
//! **What is not ported, and why.** The lane reads each format five times — fresh, after a CPU copy,
//! after a blit from a buffer, after a fill of one block, and after a copy from another texture —
//! and its subject there is that **our twin is refreshed by every kind of write**, counted in
//! `WorkCounts`. Metal has no twin and no counter: its answer is decided by the blocks alone, so
//! four of those five reads would record the same Metal fact over four draws of one random
//! distribution. The twin-refresh paths stay a validation lane, where the counters live; what comes
//! here is the part Metal can answer. The lane's three other tests are left where they are: the
//! block round trip at every level and slice is `compressed_moves`'s subject in this suite, and the
//! second-twin view and the refusals are facts about our own objects and our own typed errors
//! rather than about Metal's texels.
//!
//! **Tolerances.** Only a linear (`_LDR`) format has one. Under the default
//! `DecodedTexelsPolicy::Compact` its twin holds each channel as the 8-bit code nearest Metal's
//! 12-bit value, which `docs/FAST-PATHS.md` bounds at **1/510**, and the runtime announces
//! `decoded-texel-precision` where such a texture is made. That is the ledger's own number, not a
//! round one. The fourteen sRGB forms hold Metal's own 8-bit codes, which their slots decode through
//! Metal's table, so they are exact under either value and name **no tolerance at all**; and neither
//! does any format under `DecodedTexelsPolicy::Exact`, where the twin holds Metal's values as
//! `float`s. An `_HDR` format names none under any policy, having no twin.
//!
//! Nothing here can fault a GPU: one compute pass a read, one read-only texture, one buffer, one
//! thread a texel inside the texture, no loop and no sampling.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::bind::{AstcDecode, AstcTexels};
use metal2vulkan::metal::{format::block_extent, PixelFormat};
use metal2vulkan_runtime::Degradation;

/// The fixture whose kernel reads a texture's texels as floats.
const FIXTURE: &str = "format_scout";

/// The entry point that reads a 16 x 16 texture whole, one thread a texel.
const ENTRY: &str = "scout_read_f";

/// Each texture's side, which the kernel's indexing is written for, and the lane's own.
const SIDE: u32 = 16;

/// The texels of one texture.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The bytes one `float4` texel reads back as.
const TEXEL_BYTES: usize = 16;

/// What each round's texture is loaded with, and the name its site takes. A round is a whole
/// texture of its own, so a diff names which blocks moved.
const ROUNDS: [&str; 2] = ["zero-blocks", "seeded-blocks"];

/// The most a compact linear channel may be off Metal's value, as `docs/FAST-PATHS.md` states it
/// for `decoded_texels = DecodedTexelsPolicy::Compact`: the 8-bit code nearest Metal's 12-bit
/// value is at most half an 8-bit step away, which is 1/510. It is the ledger's number, so a run
/// that exceeds it fails rather than being re-fitted to whatever a device answers.
const LEDGERS_COMPACT_BOUND: f64 = 1.0 / 510.0;

/// An LDR void-extent block of red, green, blue and alpha `0x4000`, `0x8000`, `0xc000` and
/// `0xffff`: every texel of the block that colour. The lane's own, since the seeded stream puts its
/// first eight bytes in every fourth block.
const VOID_EXTENT: [u8; 16] = [
    0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x40, 0x00, 0x80, 0x00, 0xc0, 0xff, 0xff,
];

/// One format the lane reads: the case's name and the `MTLPixelFormat`. Its footprint follows
/// from the format ([`metal2vulkan::metal::format::block_extent`]) and its profile from whether
/// the decoder has an answer for it ([`AstcDecode::of`]), so there is no second spelling of either
/// to keep in step.
struct Format {
    name: &'static str,
    format: PixelFormat,
}

/// Every ASTC format, in the lane's order, because the seed of a round is the format's place in
/// that order and the bytes must stay the bytes the lane measured: the fourteen linear footprints,
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

/// `count` 16-byte blocks of the lane's own seeded xorshift, so a round's blocks are the lane's
/// blocks: random bits, of which most are error blocks and the rest decode through every mode,
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
/// the lane's own seeded stream at the lane's own seed for that format's place.
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
/// This is the lane's own `metal_texels`, kept as a **cross-check against the recording** rather
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

/// Every format's case, in the lane's order.
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

    /// How the format's blocks decode on a device with no ASTC block, where this crate has an
    /// answer: `None` for the HDR profile, whose textures have no decoded twin at all
    /// (`AstcDecode::of` holds the LDR and sRGB profiles alone). That `None` is the structural
    /// reason an HDR case names no tolerance and no twin notice: there is nothing of ours between
    /// the blocks and the texels, so a device either holds the format natively or refuses it.
    fn decode(&self) -> Option<AstcDecode> {
        AstcDecode::of(self.format().format)
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

    /// How far a compact texel of this format may be off Metal's value, where that is not zero:
    /// a linear format's half 8-bit step. `None` for an sRGB form, whose twin holds Metal's own
    /// codes and whose slots decode them through Metal's table, so it names no tolerance, and
    /// `None` for an HDR form, which has no twin to be compact about.
    fn compact_bound(&self) -> Option<f64> {
        self.decode()
            .filter(|decode| !decode.srgb)
            .map(|_| LEDGERS_COMPACT_BOUND)
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
        // A device with no ASTC block keeps the blocks in a carrier image and reads them through a
        // decoded twin, which the runtime announces once per device. Metal announces nothing, so
        // the notice has to be declared by its code or the comparison cannot be made at all. The
        // declaration loosens no value: the only site it covers is the set of notices, which Metal
        // holds empty, and it fails the case if the notice did not fire.
        // A CPU copy that fills such a texture (`replaceRegion`) is decoded on the CPU into that
        // twin and copied in, announced as `cpu-block-decode`: the host-copy arm of the very
        // mechanism the line above names, measured on lavapipe for every format of this family. It
        // is a declaration and not a tolerance -- every texel the case reads still compares
        // exactly with Metal.
        // Named only where this crate has a decode for the format, which is the LDR and sRGB
        // profiles. An HDR texture has no decoded twin: a device either holds the ASTC HDR block
        // natively, in which case nothing of ours sits between the blocks and the texels and no
        // notice fires, or it refuses the texture and the site is the venue's own gap.
        // Every row that runs reads its texels back through a `Shared` buffer, so on a device
        // without unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone, so every texel still compares against Metal's exactly.
        //
        // Named on the LDR and sRGB rows, which is where it was **measured**: on the RTX 5080
        // Laptop (615.71.09) all 28 of them announced it and were divergent on that field and
        // nothing else (replay-nvidia.log). The 14 HDR rows read `lacks the device feature` on
        // that card -- the venue holds no ASTC HDR block -- so no run has announced anything for
        // them and they declare nothing: a declaration on a row no venue has run is a claim
        // nobody has checked.
        let mut all = match self.decode().is_some() {
            true => vec![
                crate::replay::shared_memory_notice(),
                crate::replay::notice_allowance(Degradation::CarriedDecodedTwin.code()),
                crate::replay::notice_allowance(Degradation::CpuBlockDecode.code()),
            ],
            false => Vec::new(),
        };
        // An sRGB form's twin holds Metal's own 8-bit codes, which its slots decode through Metal's
        // table, so its texels compare exactly under either policy and it names nothing further. A
        // linear format's compact twin holds the 8-bit code nearest Metal's 12-bit value, which the
        // runtime announces as `decoded-texel-precision` and `docs/FAST-PATHS.md` bounds at 1/510.
        // The bound here is the ledger's number, so a run that exceeds it is the ledger's claim
        // failing. Under `DecodedTexelsPolicy::Exact` the twin holds Metal's values as `float`s,
        // the notice does not fire, the justification does not hold, and every texel is compared
        // exactly.
        if let Some(bound) = self.compact_bound() {
            all.push(crate::replay::notice_allowance(
                Degradation::DecodedTexelPrecision.code(),
            ));
            for round in ROUNDS {
                all.push(Allowance::new(
                    format!("case/{round}/texels"),
                    Loosening::Absolute(bound),
                    Justification::notice(Degradation::DecodedTexelPrecision),
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

    /// The profile a case's format decodes with, from the decoder's own answer: the LDR and sRGB
    /// profiles are the two this crate decodes, and a format it has no decode for is HDR.
    fn profile(decode: Option<AstcDecode>) -> &'static str {
        match decode {
            Some(decode) if decode.srgb => "srgb",
            Some(_) => "ldr",
            None => "hdr",
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
    /// The profile comes from the decoder's presence and its `srgb` flag, which within this list
    /// is exact: ASTC's other two footprints (4x2 and 8x4) also have no decoder, and they are not
    /// here — [`every_cases_name_states_its_own_footprint`] is what holds that, since a 4x2 case
    /// would have to be named `astc-4x2-hdr` to pass it and no such case exists.
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

    /// The seeded stream puts the lane's void-extent block's first eight bytes in every fourth
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

    /// Only the linear formats name a tolerance, and each names the bound `docs/FAST-PATHS.md`
    /// states.
    ///
    /// This is the ruling that a notice declaration is not a tolerance, held as a test: the
    /// `carried-decoded-twin` declaration is on every case and loosens no value, while the one
    /// allowance that loosens a texel is named only where the ledger says a texel moves.
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
            // An sRGB form's twin holds Metal's own codes and an HDR format has no twin, so
            // neither loosens a texel; only the fourteen linear ones do.
            if case.compact_bound().is_none() {
                assert!(bounds.is_empty(), "{}: {bounds:?}", format.name);
                continue;
            }
            assert_eq!(
                bounds,
                vec![Loosening::Absolute(LEDGERS_COMPACT_BOUND); ROUNDS.len()],
                "{}",
                format.name
            );
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
    /// **This is the cross-check the port exists for.** The recordings hold what Metal answered on
    /// this M3 Max; the decoder is what every guest shader on a device without the ASTC block
    /// actually reads. Until this test there was no comparison of the two that did not run through
    /// the decoder on both sides. A difference is a finding rather than something to edit away:
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
            // The HDR profile has no decode of ours to cross-check against: a device either
            // holds the ASTC HDR block natively or refuses the texture, so the recording is
            // Metal's answer and there is no second derivation to hold it to.
            let Some(decode) = (AstcTextures { at }).decode() else {
                continue;
            };
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
