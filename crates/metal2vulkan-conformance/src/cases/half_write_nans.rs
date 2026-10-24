//! A `half4` written to a float storage texture: the bits each format stores.
//!
//! This is where Metal's odd float rules live, so the case records **bits** and compares them
//! exactly: a 16-bit float texel keeps the half's own bits, a signalling NaN's payload included
//! (`0x7d49` stays `0x7d49`); a 32-bit float texel keeps the half's exact widening, but
//! `0x7fc00000` for every NaN, so a signalling NaN's payload is gone and a quiet one's is
//! rewritten; and an `RG11B10Float` texel packs each NaN channel as the format's own canonical NaN.
//! A signed zero is its own code in each. None of that survives a comparison in floats, which is
//! why every site here reads as unsigned words.
//!
//! One case per format, so a format whose storage writes a venue honestly lacks is that venue's gap
//! ([`crate::verdict::Verdict::LacksFeature`]) at its own row rather than a hole in the others.
//!
//! The kernel is the owned fixture `xr_two_planes`'s `xr2p_write_half`, a
//! `texture2d<half, access::write>` that writes buffer 0's `half4` at the thread's texel, with the
//! row's width in buffer 1 — the same call shape as the lane's own `write` kernel, and a committed
//! fixture, so Metal's assembler and a host without Metal read the same shader. The lane's two
//! other accesses (a `texture2d<half, access::read_write>`, and a texture an argument buffer holds)
//! have no committed fixture that writes a `half4` from a buffer, so they are not ported; the
//! module docs of `runtime_lane::half_write_nans` record that plain Metal stored the same bits
//! through all three.
//!
//! Every texel of the texture holds one fill byte before the dispatch, so a texel the kernel did not
//! reach is a recorded value rather than whatever the allocator left.
//!
//! Nothing here can fault a GPU: one compute pass, one texture, one thread a texel inside it, no
//! loop and no sampling.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The fixture the kernel lives in.
const FIXTURE: &str = "xr_two_planes";

/// The kernel: `half4` from buffer 0 at the thread's texel, the row's width in buffer 1.
const WRITE: &str = "xr2p_write_half";

/// The halves written, in the lane's order: quiet NaNs, signalling NaNs, negative NaNs, then the
/// infinities, the zeros, the denormals and two normals.
///
/// A signalling NaN is in here on purpose: it is the one input that tells a 16-bit float store
/// (which keeps `0x7d49`) from a 32-bit float one (which rewrites every NaN to `0x7fc00000`).
const V: [u16; 25] = [
    0x7e00, 0x7e01, 0x7f30, 0x7fff, 0x7e55, // quiet
    0x7c01, 0x7d49, 0x7d30, 0x7dff, 0x7c80, // signalling
    0xfe00, 0xfe01, 0xffff, 0xfc01, 0xfd49, 0xfdff, // negative
    0x7c00, 0xfc00, 0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x3c00, 0x7bff, // controls
];

/// The texture's width: one texel per value.
const N: u32 = V.len() as u32;

/// The byte every texel holds before the kernel runs.
const FILL: u8 = 0xab;

/// Texel `x`'s four halves: each value at every channel position across the texels, so a store
/// that treated one channel differently from another shows as a texel rather than as a hash.
fn texel(x: usize) -> [u16; 4] {
    std::array::from_fn(|channel| V[(x + channel) % V.len()])
}

/// Buffer 0's bytes: one `half4` a texel.
fn halves() -> Vec<u8> {
    (0..V.len())
        .flat_map(texel)
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// Buffer 1's bytes: the row's width, as the kernel's `constant uint &w`.
fn width() -> Vec<u8> {
    N.to_le_bytes().to_vec()
}

/// A format the case writes: its name, the bytes of one texel, and what one stored word is.
struct Shape {
    /// The word a channel is read back as: a 16-bit float format's own half, a 32-bit float one's
    /// `float`, and a packed format's whole 32-bit texel.
    word: Element,
    /// The words one texel holds.
    words: u32,
}

/// The case: one pixel format's stored words.
pub struct HalfWriteNans {
    name: &'static str,
    format: PixelFormat,
    shape: Shape,
    /// Whether one word of the format holds every channel. A packed format's channel is narrower
    /// than the `half` written into it, so it has no room for a NaN's payload and what reaches the
    /// texel is the device's own encoding; a format whose word is one channel stores the bits it
    /// was handed.
    packed: bool,
}

/// Every format the lane writes: the three 16-bit float widths, the three 32-bit float widths, and
/// the packed one whose NaN is the format's own.
pub fn all() -> Vec<HalfWriteNans> {
    // One word is one channel for every float format, and the whole texel for the packed one.
    let channels = |name, format, word, words| HalfWriteNans {
        name,
        format,
        shape: Shape { word, words },
        packed: false,
    };
    let whole_texel = |name, format, word| HalfWriteNans {
        name,
        format,
        shape: Shape { word, words: 1 },
        packed: true,
    };
    vec![
        channels(
            "half-write-nans-r16f",
            PixelFormat::R16Float,
            Element::U16,
            1,
        ),
        channels(
            "half-write-nans-rg16f",
            PixelFormat::RG16Float,
            Element::U16,
            2,
        ),
        channels(
            "half-write-nans-rgba16f",
            PixelFormat::RGBA16Float,
            Element::U16,
            4,
        ),
        channels(
            "half-write-nans-r32f",
            PixelFormat::R32Float,
            Element::U32,
            1,
        ),
        channels(
            "half-write-nans-rg32f",
            PixelFormat::RG32Float,
            Element::U32,
            2,
        ),
        channels(
            "half-write-nans-rgba32f",
            PixelFormat::RGBA32Float,
            Element::U32,
            4,
        ),
        whole_texel(
            "half-write-nans-rg11b10f",
            PixelFormat::RG11B10Float,
            Element::U32,
        ),
    ]
}

impl HalfWriteNans {
    /// The fixture, under the kernel this case runs.
    fn fixture() -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: WRITE,
        }
    }

    /// The bytes one texel of the format takes.
    fn texel_bytes(&self) -> u32 {
        self.shape.words * self.shape.word.size()
    }

    /// The bytes of the whole level, which is also the bytes the case hands in.
    fn level_bytes(&self) -> usize {
        (N * self.texel_bytes()) as usize
    }

    /// The layout the stored texels are read under: one row of unsigned words, so a diff names the
    /// word that moved and prints it as the bits Metal stored rather than as a float.
    fn layout(&self) -> ByteLayout {
        let pitch = N * self.texel_bytes();
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [N, 1, 1],
            },
            element: self.shape.word,
            row_offset: 0,
            row_elements: N * self.shape.words,
            row_pitch: pitch,
            rows: 1,
            image_pitch: pitch,
            images: 1,
            // The row is the whole level and the words cover every byte of it, so there is no gap.
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for HalfWriteNans {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The kernel writes `half` into a storage texture, and only the notice set is loosened, so
        // every stored word still compares exactly whatever the device does.
        //
        // The loosening is carried by the formats that cannot hold a NaN's bits: a channel narrower
        // than the half it is given has no room for the payload, so what reaches the texel is the
        // device's own encoding and the runtime announces it. The float and half formats store the
        // bits they are handed, so a run of them needs no tolerance at all — and this round's runs
        // on both venues confirmed it, which is why they name none. A device that does encode its
        // own way there shows the word it stored beside Metal's rather than passing quietly.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for all 7 rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        let mut named = vec![crate::replay::shared_memory_notice()];
        if self.packed {
            named.push(crate::replay::notice_allowance(
                Degradation::DeviceNanEncoding.code(),
            ));
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: WRITE.to_owned(),
                source_sha256: crate::sha256_hex(Self::fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The format is the question the case asks, so a case re-pointed at another one is
            // `Unrecorded` rather than compared against the first's words.
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("buffer:0", &halves()),
            Input::bytes("buffer:1", &width()),
            // What the texture holds before the dispatch.
            Input::bytes("texture:blank", &vec![FILL; self.level_bytes()]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Self::fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, loaded.entry),
        )?;
        let values = made(sites, "buffer/0", side.buffer(&halves()))?;
        let shape = made(sites, "buffer/1", side.buffer(&width()))?;
        let pitch = N * self.texel_bytes();
        let texture = made(
            sites,
            "texture",
            side.texture_2d(self.format, [N, 1], pitch, &vec![FILL; self.level_bytes()]),
        )?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&[texture], &[values, shape]),
                &[],
                [N, 1, 1],
                [N, 1, 1],
            ),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let stored = call(
            sites,
            "read-back",
            side.read_texture(texture, [N, 1], pitch),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf("stored", Observed::bytes(stored, self.layout())));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Each case's layout covers exactly the bytes its texture holds: a layout that reached further
    /// would read past the level, and one that stopped short would leave stored words uncompared.
    #[test]
    fn every_layout_covers_its_whole_level() {
        for case in all() {
            assert_eq!(
                case.layout().extent(),
                case.level_bytes(),
                "{}: its layout reaches {} of {} bytes",
                case.name,
                case.layout().extent(),
                case.level_bytes()
            );
        }
    }

    /// The cases name distinct rows and distinct formats, so no two share a recording.
    #[test]
    fn the_cases_name_distinct_formats() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        let formats: BTreeSet<u32> = all().iter().map(|case| case.format.as_raw()).collect();
        assert_eq!(named.len(), all().len());
        assert_eq!(formats.len(), all().len());
    }

    /// The input the whole family turns on: a signalling NaN, whose payload a 16-bit float store
    /// keeps and a 32-bit one rewrites. A sweep that lost it would compare NaN handling without
    /// asking the one question that separates the formats.
    #[test]
    fn the_values_hold_a_signalling_nan_and_both_zeros() {
        let signalling = V
            .iter()
            .filter(|&&bits| bits & 0x7c00 == 0x7c00 && bits & 0x3ff != 0 && bits & 0x200 == 0)
            .count();
        assert!(signalling >= 2, "{signalling} signalling NaNs");
        assert!(V.contains(&0x0000) && V.contains(&0x8000), "both zeros");
    }

    /// Every texel holds its own four halves, so a store that mixed two texels up could not pass,
    /// and the buffer is one `half4` a texel.
    #[test]
    fn every_texel_is_its_own_word() {
        let distinct: BTreeSet<[u16; 4]> = (0..V.len()).map(texel).collect();
        assert_eq!(distinct.len(), V.len());
        assert_eq!(halves().len(), V.len() * 8);
    }
}
