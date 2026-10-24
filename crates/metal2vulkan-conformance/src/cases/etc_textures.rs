//! How ETC2 and EAC texels decode, ported from `runtime_lane::etc_textures`.
//!
//! An ETC2 or EAC texture is a texture no desktop GPU and no lavapipe holds in its own format, so
//! the runtime keeps its blocks in a carrier image and its reads go through a **decoded twin** it
//! fills itself, announcing `carried-decoded-twin` once per device. What a guest shader then reads
//! is not the device's decode but **ours**, which makes every texel of such a texture a value this
//! implementation computes rather than one a driver answers.
//!
//! **The port is the first independent check of that decoder.** `metal2vulkan::bind::EtcDecode` is
//! today checked against its own tests and against one hand-copied measurement in the lane: the
//! lane computes Metal's expected texels *from the decoder itself* (its own `metal_texels`) and
//! compares our read against that, so a decoder that is wrong in the same way on both sides passes. Here record mode runs the case on this M3
//! Max and the recording holds **Metal's own texels**. The decoder then becomes a cross-check
//! against the oracle (the test `every_recorded_texel_is_the_decoders`), which is the one direction
//! in which a difference says something: either Apple's ETC decode is not what we implement, or it
//! moved.
//!
//! **One case per format**, so a venue that holds no texture of one answers its typed refusal there
//! and the suite counts that format as [`LacksFeature`](crate::verdict::Verdict::LacksFeature). The
//! lane walks the ten formats in one test, where a device missing one would take the whole lane
//! down.
//!
//! Each case is two 16 x 16 textures, read whole by the owned fixture `format_scout`'s
//! `scout_read_f`, one thread a texel: one of **zero blocks** (the lane's "fresh" read, stated as
//! an upload so the bytes are the case's own rather than whatever a fresh texture holds), and one of
//! the lane's own seeded xorshift blocks at the lane's own seed, so those bytes are the bytes the
//! lane measured. Both are loaded by `replaceRegion:` as the texture is made, which is the lane's
//! CPU-copy write.
//!
//! **What is not ported, and why.** The lane reads each format five times — fresh, after a CPU copy,
//! after a blit from a buffer, after a fill of one block, and after a copy from another texture —
//! and its subject there is that **our twin is refreshed by every kind of write**, counted in
//! `WorkCounts`. Metal has no twin and no counter: its answer is decided by the blocks alone, so
//! four of those five reads would record the same Metal fact over four draws of one random
//! distribution. The twin-refresh paths stay a validation lane, where the counters live; what comes
//! here is the part Metal can answer. The lane's three further seeded draws (`seed + 1` to
//! `seed + 3`) are left for the same reason.
//!
//! **Tolerances.** Two, each the ledger's own number for a fast default, and each one that fails
//! where the notice it names did not fire.
//!
//! Under the default `DecodedTexelsPolicy::Compact` an 11-bit channel is read from a 16-bit
//! normalized twin, which `docs/FAST-PATHS.md` bounds at half a 16-bit step — **1/131070
//! unsigned, 1/65534 signed** — and the runtime announces `decoded-texel-precision` where such a
//! texture is made. Those are the ledger's own numbers, named per format, not a round number.
//!
//! Under the default `TexelConversionPolicy::Device` the `float` a function reads of a twin's
//! 8-bit code is the **device's** conversion of that code, not Metal's `c / 255` correctly
//! rounded, which `docs/FAST-PATHS.md` bounds at **one `f32` ulp** for 8- and 16-bit codes and the
//! runtime announces as `device-texel-conversion`. It reaches the three **linear** 8-bit formats
//! (`ETC2_RGB8`, `ETC2_RGB8A1`, `EAC_RGBA8`): their twin holds Metal's codes exactly and the
//! device converts them. It does not reach their sRGB forms, whose colour the function decodes
//! from the raw code through Metal's twelve-bit table itself and whose alpha it converts by the
//! format's class, so no device arithmetic stands between the twin's codes and the answer.
//!
//! No format names either tolerance under `DecodedTexelsPolicy::Exact` with
//! `TexelConversionPolicy::Exact`, where the twin holds Metal's values as `float`s and the
//! function converts a code by the bound format's class: neither notice fires, neither
//! justification holds, and every texel is compared exactly.
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
use metal2vulkan::bind::{EtcDecode, EtcTexels, PvrtcDecode};
use metal2vulkan::metal::format::srgb_raw_twin;
use metal2vulkan::metal::PixelFormat;
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

/// The blocks a 16 x 16 level of a 4 x 4 block holds, across and down.
const BLOCKS: u32 = SIDE / 4;

/// What each round's texture is loaded with, and the name its site takes. A round is a whole
/// texture of its own, so a diff names which blocks moved.
const ROUNDS: [&str; 2] = ["zero-blocks", "seeded-blocks"];

/// The most a compact unsigned 11-bit texel may be off Metal's value: half a 16-bit step, as
/// `docs/FAST-PATHS.md` states it for `decoded_texels = DecodedTexelsPolicy::Compact`. It is the
/// ledger's number, so a run that exceeds it fails rather than being re-fitted to whatever a
/// device answers.
const LEDGERS_UNSIGNED_BOUND: f64 = 1.0 / 131_070.0;

/// The same for a signed 11-bit texel, whose twin holds a 16-bit **signed** normalized code, so
/// half a step is twice as wide ([`LEDGERS_UNSIGNED_BOUND`]).
const LEDGERS_SIGNED_BOUND: f64 = 1.0 / 65_534.0;

/// The bound `docs/FAST-PATHS.md` states for `texel_conversion = TexelConversionPolicy::Device`:
/// one `f32` ulp for 8- and 16-bit codes. It is the ledger's number, copied here so a run that
/// exceeds it fails rather than being re-fitted to whatever a device answers, and it is the same
/// number [`crate::cases::compressed_decode`] names for the same read path over a BC twin's codes.
const LEDGERS_TEXEL_CONVERSION_ULPS: u32 = 1;

/// One format the lane reads: the case's name and the `MTLPixelFormat`. How its blocks decode and
/// whether its colour is sRGB follow from the format ([`EtcDecode::of`], [`srgb_raw_twin`]), so
/// there is no second spelling of either to keep in step.
struct Format {
    name: &'static str,
    format: PixelFormat,
}

/// Every ETC2 and EAC format, in the lane's order, because the seed of a round is the format's
/// place in that order and the bytes must stay the bytes the lane measured.
const FORMATS: [Format; 10] = [
    Format {
        name: "etc2-rgb8",
        format: PixelFormat::ETC2RGB8,
    },
    Format {
        name: "etc2-rgb8-srgb",
        format: PixelFormat::ETC2RGB8SRGB,
    },
    Format {
        name: "etc2-rgb8a1",
        format: PixelFormat::ETC2RGB8A1,
    },
    Format {
        name: "etc2-rgb8a1-srgb",
        format: PixelFormat::ETC2RGB8A1SRGB,
    },
    Format {
        name: "eac-rgba8",
        format: PixelFormat::EACRGBA8,
    },
    Format {
        name: "eac-rgba8-srgb",
        format: PixelFormat::EACRGBA8SRGB,
    },
    Format {
        name: "eac-r11-unorm",
        format: PixelFormat::EACR11Unorm,
    },
    Format {
        name: "eac-r11-snorm",
        format: PixelFormat::EACR11Snorm,
    },
    Format {
        name: "eac-rg11-unorm",
        format: PixelFormat::EACRG11Unorm,
    },
    Format {
        name: "eac-rg11-snorm",
        format: PixelFormat::EACRG11Snorm,
    },
];

/// `count` blocks of `bytes` each of the lane's own seeded xorshift, so a round's blocks are the
/// lane's blocks: random bits, which decode through every mode.
fn seeded_blocks(seed: u64, count: usize, bytes: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(count * bytes + 8);
    while out.len() < count * bytes {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        out.extend_from_slice(&state.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
    }
    out.truncate(count * bytes);
    out
}

/// The blocks of round `round` of the format at place `at`: zeros, then the lane's own seeded
/// stream at the lane's own seed for that format's place.
fn blocks(at: usize, round: usize, decode: EtcDecode) -> Vec<u8> {
    let bytes = decode.block_bytes() as usize;
    let count = (BLOCKS * BLOCKS) as usize;
    match round {
        0 => vec![0; count * bytes],
        _ => seeded_blocks(0xe7c2_0000 + 0x10 * at as u64, count, bytes),
    }
}

/// The `float4` Metal reads of each texel of a `SIDE` x `SIDE` level of `pf`'s `blocks`, as the
/// decoders under test compute it: an 8-bit code over 255 and an sRGB form's colour through
/// Metal's twelve-bit table, an 11-bit value over 2047 (signed 1023), and an `R11` or `RG11`
/// format's missing channels 0, alpha 1.
///
/// This is the lane's own `metal_texels`, kept as a **cross-check against the recording** rather
/// than as any case's expectation: the recordings hold Metal's answer, and the test
/// `every_recorded_texel_is_the_decoders` holds the two against each other.
pub fn decoders_texels(pf: PixelFormat, blocks: &[u8]) -> Vec<[f32; 4]> {
    let decode = EtcDecode::of(pf).expect("an ETC2 or EAC format");
    let size = [SIDE, SIDE, 1];
    let texels = decode.decode(EtcTexels::Float, blocks, size);
    if decode.eleven_bit() {
        let channels = decode.texel_bytes(EtcTexels::Float) as usize / 4;
        return texels
            .chunks_exact(4 * channels)
            .map(|texel| {
                let value = |k: usize| {
                    f32::from_le_bytes(texel[4 * k..4 * k + 4].try_into().expect("four bytes"))
                };
                let green = if channels == 2 { value(1) } else { 0.0 };
                [value(0), green, 0.0, 1.0]
            })
            .collect();
    }
    // An 8-bit format's twin holds four codes a texel; Metal's `float` of a code is the table its
    // sRGB-ness names, which is the table PVRTC's and BC's sRGB forms read through.
    let table = match srgb_raw_twin(pf).is_some() {
        true => PvrtcDecode::of(PixelFormat::PVRTCRGBA4BPPSRGB),
        false => PvrtcDecode::of(PixelFormat::PVRTCRGBA4BPP),
    }
    .expect("a PVRTC format");
    texels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|codes| std::array::from_fn(|c| table.metal_value(c, codes[c])))
        .collect()
}

/// The case for one format.
pub struct EtcTextures {
    at: usize,
}

/// Every format's case, in the lane's order.
pub fn all() -> Vec<EtcTextures> {
    (0..FORMATS.len()).map(|at| EtcTextures { at }).collect()
}

impl EtcTextures {
    /// The format the case reads.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// How the format's blocks decode.
    fn decode(&self) -> EtcDecode {
        EtcDecode::of(self.format().format).expect("an ETC2 or EAC format")
    }

    /// The fixture the case runs.
    fn fixture() -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: ENTRY,
        }
    }

    /// The bytes a row of blocks takes: four blocks of a 16-texel row.
    fn row_pitch(&self) -> u32 {
        BLOCKS * self.decode().block_bytes()
    }

    /// The texture of one round: the format, 16 x 16 texels, read-only and shared, loaded with the
    /// round's blocks at four blocks a row.
    fn request<'a>(&self, uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
        TextureRequest::read_2d(self.format().format.as_raw(), SIDE, SIDE).loaded(uploads)
    }

    /// How far a compact texel of this format may be off Metal's value, where that is not zero:
    /// an EAC 11-bit channel's half 16-bit step, by the format's own signedness. `None` for the
    /// 8-bit formats, whose twin holds its codes exactly under either value, so they name no
    /// tolerance.
    fn compact_bound(&self) -> Option<f64> {
        match self.format().format {
            PixelFormat::EACR11Snorm | PixelFormat::EACRG11Snorm => Some(LEDGERS_SIGNED_BOUND),
            PixelFormat::EACR11Unorm | PixelFormat::EACRG11Unorm => Some(LEDGERS_UNSIGNED_BOUND),
            _ => None,
        }
    }

    /// Whether a device's own arithmetic stands between this format's twin and the `float` the
    /// function answers, so that under `TexelConversionPolicy::Device` the texel is the device's
    /// conversion of an 8-bit code rather than Metal's `c / 255` correctly rounded.
    ///
    /// Decided from the format's own shape, never from the device or the case's name: the twin
    /// holds 8-bit codes ([`EtcDecode::eleven_bit`] is false — an 11-bit format's compact twin
    /// holds 16-bit codes, whose own, wider bound `compact_bound` already covers) and the format
    /// is not an sRGB form ([`srgb_raw_twin`]), whose colour the function decodes from the raw
    /// code through Metal's table and whose alpha it converts by the format's class, so no device
    /// conversion is involved either way. It is the same test the translator's read path makes
    /// (`lower::texture`: an sRGB slot's raw codes are converted exactly, so its access records
    /// no `PrecisionNote::DeviceTexelConversion`).
    fn device_converts_codes(&self) -> bool {
        !self.decode().eleven_bit() && srgb_raw_twin(self.format().format).is_none()
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

impl Case for EtcTextures {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A device with no ETC2 format keeps the blocks in a carrier image and reads them through
        // a decoded twin, which the runtime announces once per device. Metal announces nothing, so
        // the notice has to be declared by its code or the comparison cannot be made at all. The
        // declaration loosens no value: the only site it covers is the set of notices, which Metal
        // holds empty, and it fails the case if the notice did not fire.
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
            crate::replay::notice_allowance(Degradation::CarriedDecodedTwin.code()),
            crate::replay::notice_allowance(Degradation::CpuBlockDecode.code()),
        ];
        // An 8-bit format's twin holds Metal's own codes, but under the default
        // `TexelConversionPolicy::Device` the `float` the function answers is the **device's**
        // conversion of a code, not Metal's `c / 255` correctly rounded: lavapipe multiplies by
        // the reciprocal. The runtime announces that as `device-texel-conversion`, and
        // `docs/FAST-PATHS.md`'s row for the policy bounds it at one `f32` ulp for 8- and 16-bit
        // codes, so the bound here is the ledger's number and not a measurement of one driver: a
        // run that exceeds one ulp is the ledger's claim failing, which is what this allowance
        // exists to catch. It is the same allowance `cases::compressed_decode` names for the same
        // read path over a BC twin's codes. Under `TexelConversionPolicy::Exact` the function
        // recovers the code from the device's value by the bound format's class
        // (`PackedTexels::Argb8`) and converts it itself, the notice does not fire, the
        // justification does not hold, and every texel is compared exactly.
        if self.device_converts_codes() {
            for round in ROUNDS {
                all.push(Allowance::new(
                    format!("case/{round}/texels"),
                    Loosening::Ulps(LEDGERS_TEXEL_CONVERSION_ULPS),
                    Justification::notice(Degradation::DeviceTexelConversion),
                ));
            }
        }
        // An EAC 11-bit format's compact twin holds the 16-bit normalized code nearest Metal's
        // value, which the runtime announces as `decoded-texel-precision` and
        // `docs/FAST-PATHS.md` bounds at half a 16-bit step. The bound here is the ledger's
        // number for this format's signedness, so a run that exceeds it is the ledger's claim
        // failing. It is wider than the conversion ulp above and covers it, which is why an
        // 11-bit format names this one alone. Under `DecodedTexelsPolicy::Exact` the twin holds
        // Metal's values as `float`s, the notice does not fire, the justification does not hold,
        // and every texel is compared exactly.
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
            let bytes = blocks(self.at, round, self.decode());
            let uploads = [Upload::level_0(
                [SIDE, SIDE],
                self.row_pitch(),
                BLOCKS,
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
            let bytes = blocks(self.at, round, self.decode());
            let uploads = [Upload::level_0(
                [SIDE, SIDE],
                self.row_pitch(),
                BLOCKS,
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
    use crate::transcript::Transcript;
    use std::collections::BTreeSet;

    /// The cases name distinct recordings, since a case's name is its recording's file name.
    #[test]
    fn the_formats_name_distinct_cases() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
    }

    /// Every format the lane walks has a decode and a case, and no case names a format whose
    /// blocks nothing here can decode.
    #[test]
    fn every_case_names_an_etc_format() {
        for format in &FORMATS {
            assert!(
                EtcDecode::of(format.format).is_some(),
                "{}: no ETC decode",
                format.name
            );
        }
    }

    /// A round's blocks fill the texture's blocks exactly, at the pitch the upload states.
    #[test]
    fn every_rounds_blocks_fill_its_texture() {
        for (at, format) in FORMATS.iter().enumerate() {
            let case = EtcTextures { at };
            let decode = case.decode();
            for round in 0..ROUNDS.len() {
                let bytes = blocks(at, round, decode);
                assert_eq!(
                    bytes.len(),
                    (BLOCKS * BLOCKS) as usize * decode.block_bytes() as usize,
                    "{}",
                    format.name
                );
                assert_eq!(
                    bytes.len() as u32,
                    case.row_pitch() * BLOCKS,
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
            let decode = EtcTextures { at }.decode();
            assert_ne!(
                blocks(at, 0, decode),
                blocks(at, 1, decode),
                "{}",
                format.name
            );
        }
    }

    /// Each format loosens the texel its own `docs/FAST-PATHS.md` row names, and nothing else.
    ///
    /// This is the ruling that a notice declaration is not a tolerance, held as a test: the
    /// `carried-decoded-twin` declaration is on every case and loosens no value, while an
    /// allowance that loosens a texel is named only where a fast default puts a device's own
    /// arithmetic between the twin's codes and the answer. Three groups, each format named in its
    /// own, so a format moving between them fails here:
    ///
    /// - the three **linear** 8-bit formats, whose codes the device converts under
    ///   `TexelConversionPolicy::Device`: one `f32` ulp, both rounds;
    /// - their **sRGB** forms, whose colour the function decodes from the raw code through Metal's
    ///   twelve-bit table itself: **nothing at all**;
    /// - the four EAC **11-bit** formats, read from a compact 16-bit twin under
    ///   `DecodedTexelsPolicy::Compact`: half a 16-bit step by the format's signedness.
    #[test]
    fn each_format_loosens_the_texel_its_ledger_row_names() {
        let mut loosened = Vec::new();
        let mut ulps = Vec::new();
        let mut exact = Vec::new();
        for (at, format) in FORMATS.iter().enumerate() {
            let case = EtcTextures { at };
            let bounds: Vec<Loosening> = case
                .allowances()
                .into_iter()
                .filter(|allowance| allowance.path.ends_with("/texels"))
                .map(|allowance| allowance.loosening)
                .collect();
            let decode = case.decode();
            if !decode.eleven_bit() {
                match case.device_converts_codes() {
                    true => {
                        // The literal 1 is deliberate, not `LEDGERS_TEXEL_CONVERSION_ULPS`:
                        // asserting against the constant the allowance is built from makes the
                        // number unfalsifiable here, and widening it would pass. The ledger's
                        // figure is written out, so moving the constant fails this test as well
                        // as `the_texel_allowance_holds_one_ulp_only_where_the_notice_fired`.
                        assert_eq!(
                            bounds,
                            vec![Loosening::Ulps(1); ROUNDS.len()],
                            "{}",
                            format.name
                        );
                        ulps.push(format.name);
                    }
                    false => {
                        assert!(bounds.is_empty(), "{}: {bounds:?}", format.name);
                        exact.push(format.name);
                    }
                }
                continue;
            }
            let signed = matches!(decode, EtcDecode::R11Snorm | EtcDecode::Rg11Snorm);
            let bound = match signed {
                true => LEDGERS_SIGNED_BOUND,
                false => LEDGERS_UNSIGNED_BOUND,
            };
            assert_eq!(
                bounds,
                vec![Loosening::Absolute(bound); ROUNDS.len()],
                "{}",
                format.name
            );
            loosened.push(format.name);
        }
        assert_eq!(ulps, ["etc2-rgb8", "etc2-rgb8a1", "eac-rgba8"]);
        assert_eq!(
            exact,
            ["etc2-rgb8-srgb", "etc2-rgb8a1-srgb", "eac-rgba8-srgb"]
        );
        assert_eq!(
            loosened,
            [
                "eac-r11-unorm",
                "eac-r11-snorm",
                "eac-rg11-unorm",
                "eac-rg11-snorm"
            ]
        );
    }

    /// A round's texels as a transcript, every channel the `f32` word `word`.
    fn texels_transcript(round: &str, word: u32) -> Transcript {
        let mut bytes = Vec::with_capacity(TEXELS * TEXEL_BYTES);
        for _ in 0..TEXELS * TEXEL_BYTES / 4 {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        Transcript::new(vec![Node::branch(round.to_owned(), Vec::new()).with(
            Node::leaf("texels", Observed::bytes(bytes, EtcTextures::layout())),
        )])
    }

    /// The case at `at`, its seeded round's texels compared against Metal's `word` where our run
    /// answered `answered` and the device announced `announced`.
    fn compared(
        at: usize,
        word: u32,
        answered: u32,
        announced: &[String],
    ) -> crate::compare::Comparison {
        let round = ROUNDS[1];
        crate::compare::compare(
            &texels_transcript(round, word),
            &texels_transcript(round, answered),
            &EtcTextures { at }.allowances(),
            crate::allowance::Offered::notices(announced),
        )
    }

    /// The place of the first format whose codes the device converts, and of the first 8-bit sRGB
    /// one, taken from the cases' own test rather than from their names.
    fn linear_and_srgb() -> (usize, usize) {
        let linear = (0..FORMATS.len())
            .find(|&at| EtcTextures { at }.device_converts_codes())
            .expect("a linear 8-bit format");
        let srgb = (0..FORMATS.len())
            .find(|&at| {
                let case = EtcTextures { at };
                !case.decode().eleven_bit() && !case.device_converts_codes()
            })
            .expect("an 8-bit sRGB format");
        (linear, srgb)
    }

    /// The one-ulp texel allowance is a claim about what the runtime **said**, not a blanket
    /// tolerance, and the comparison is exercised in the four directions that decide that.
    ///
    /// Metal's word here is the one this M3 Max answered at `etc2-rgb8`'s first differing channel
    /// (`0x3ee4e4e5`) and ours the word arm64 lavapipe answered there under the default policy
    /// (`0x3ee4e4e6`, one ulp above). With no device in sight:
    ///
    /// 1. with `device-texel-conversion` announced the difference reads **announced**;
    /// 2. with **nothing** announced the same difference reads **divergent** and the allowance is
    ///    reported bad as `NoticeNotAnnounced` — so the day the read path stops announcing, this
    ///    allowance stops excusing anything;
    /// 3. **two** ulps apart fails even with the notice announced, so the ledger's bound binds and
    ///    the notice is no licence;
    /// 4. an 8-bit **sRGB** format, which names no texel allowance, is divergent at one ulp
    ///    whatever was announced, so the partition above is load-bearing.
    #[test]
    fn the_texel_allowance_holds_one_ulp_only_where_the_notice_fired() {
        let (linear, srgb) = linear_and_srgb();
        let metal = 0x3ee4_e4e5;
        let code = Degradation::DeviceTexelConversion.code().to_owned();
        let announced = [code.clone()];

        let found = compared(linear, metal, metal + 1, &announced);
        assert!(found.divergences.is_empty(), "{found:?}");
        assert!(found.bad_allowances.is_empty(), "{found:?}");
        assert_eq!(found.announced.len(), 1, "{found:?}");
        assert_eq!(
            found.announced[0].count,
            TEXELS * TEXEL_BYTES / 4,
            "{found:?}"
        );

        let found = compared(linear, metal, metal + 1, &[]);
        assert_eq!(found.divergences.len(), 1, "{found:?}");
        assert_eq!(found.bad_allowances.len(), 1, "{found:?}");
        assert_eq!(
            found.bad_allowances[0].why,
            crate::allowance::Unjustified::NoticeNotAnnounced(code)
        );

        let found = compared(linear, metal, metal + 2, &announced);
        assert_eq!(found.divergences.len(), 1, "{found:?}");
        assert!(found.announced.is_empty(), "{found:?}");

        let found = compared(srgb, metal, metal + 1, &announced);
        assert_eq!(found.divergences.len(), 1, "{found:?}");
        assert!(found.announced.is_empty(), "{found:?}");
    }

    /// The layout covers the whole read-back and no more, so the comparison reads every texel.
    #[test]
    fn the_layout_covers_the_read_back() {
        assert_eq!(EtcTextures::layout().extent(), TEXELS * TEXEL_BYTES);
    }

    /// Every texel of every committed recording is `metal2vulkan::bind::EtcDecode`'s.
    ///
    /// **This is the cross-check the port exists for.** The recordings hold what Metal answered on
    /// this M3 Max; the decoder is what every guest shader on a device without ETC2 actually reads.
    /// Until this test there was no comparison of the two that did not run through the decoder on
    /// both sides. A difference is a finding rather than something to edit away: either Apple's
    /// decode is not what we implement, or it moved.
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
            let decode = EtcTextures { at }.decode();
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for (round, name) in ROUNDS.iter().enumerate() {
                        let blocks = blocks(at, round, decode);
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
                        let want = decoders_texels(format.format, &blocks);
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
                                         against EtcDecode's {wanted}",
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
             metal2vulkan::bind::EtcDecode:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
