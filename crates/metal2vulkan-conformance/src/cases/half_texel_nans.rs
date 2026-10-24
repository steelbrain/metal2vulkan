//! What a **`half`** result gets back from a float texel's NaNs, and what the shader's own widening
//! then does with it.
//!
//! [`crate::cases::half_read_nans`] asks the same texels with a `float` result. The two answers are
//! not the same rule and that is the point of having both: with a `half` result a **16-bit** float
//! texel hands back its own bits — sign, payload and signalling bit kept, so `0x7c01` stays
//! `0x7c01` — while a 32-bit float texel's NaN, a small float's (`RG11B10Float`) and every filtered
//! sample's come back as `0x7e00`. `RGB9E5Float` decodes every code to a finite value, so no lane of
//! it is a NaN at all. That is what `runtime_lane::half_texel_nans` measured on plain Metal, and
//! these recordings are what hold the implementation to it.
//!
//! One case per format, so a format a venue honestly lacks a readable or filterable texture of is
//! that venue's gap ([`crate::verdict::Verdict::LacksFeature`]) at its own row.
//!
//! **One texture, three accesses, four recorded columns.** The owned fixture `hrd_half_read` reads
//! texel `(i mod 64, i div 64)` of a 64 by 16 texture three ways and stores the results as
//! **unsigned words**, which is the whole reason it is the fixture for this family: `as_type` keeps
//! the half's bits where `float4(h)` would throw a NaN's payload away.
//!
//! | Column | What it is |
//! |---|---|
//! | `read-through-sample` | `t.read(texel)` of a `texture2d<half, access::sample>`, as four halves |
//! | `nearest-sample` | a nearest constexpr sample at the texel's centre, as four halves |
//! | `read` | `r.read(texel)` of a `texture2d<half, access::read>`, as four halves |
//! | `widened` | `float4` of the first column's `half4`: the **shader's** half-to-float widening, not a texture's |
//! | `zeros` | the two words the kernel writes zero, which is the control that the lane ran |
//!
//! The one texture is bound at **both** texture slots, so the two declared accesses read the same
//! bytes by construction and a difference between those two columns cannot be a difference of
//! input.
//!
//! **What recording the three accesses together found, which the lane could not say.** One source
//! value in the whole sweep makes them disagree: a 32-bit float texel holding the largest finite
//! `float` (`0x7f7fffff`) comes back as `0x7c00`, the infinity of its sign, through both `read()`s
//! and as `0x7bff`, the largest `half`, through the nearest sample. The lane allowed either value
//! for a lane past `half`'s range -- its own words are "infinity of its sign, or 65504 where the
//! device's conversion rounds toward zero" -- but it compared one access at a time, so it never
//! said that Metal itself answers both, on the same texel, in the same kernel. Nothing else in the
//! sweep separates a fetch from a nearest sample on this chip, which the cross-check below asserts
//! rather than observes.
//!
//! **Where our side differs from Metal, measured over every word.** On this round's MoltenVK
//! (Apple M3 Max, API 1.2) the three 16-bit float formats are `divergent` and the other five are
//! `announced`. Our side answers `0x7e00` for **every** NaN texel where Metal hands back the
//! texel's own code, so a signalling NaN, a payload NaN and a negative NaN all collapse to the one
//! quiet positive NaN; the runtime announces `device-nan-encoding` for it. Measured with a
//! temporary tolerance wide enough to pass, then read off the per-site counts and the tolerance
//! dropped: 515 of each site's 4,096 words for `R16Float`, 1,026 for `RG16Float` and 2,045 for
//! `RGBA16Float`, the same count at all three access sites, largest difference 33,279 as unsigned
//! words (Metal `0xffff` against our `0x7e00`). The differing words are exactly the NaN texels
//! whose code is not already `0x7e00` -- for `R16Float`, the twelve such values of the half table at their
//! 515 texels -- and no finite, zero, denormal or infinite texel differs anywhere.
//!
//! **Where the payload goes.** Our `half` access reads the image as a `vec4` of `f32` and converts
//! it (`lower::texture::half`, through `narrow::round_texel`), and this chip's decode of a 16-bit
//! float texel into `f32` canonicalizes the NaN, so the sign and the payload are gone before the
//! conversion sees them -- which is why all twelve codes, of both signs, collapse onto one value,
//! and why the `widened` column, where Metal canonicalizes too, is exact. Metal's `texture2d<half>`
//! read never widens, so the texel's bits survive by construction.
//!
//! This is **not** the gap that `vertex-format-constant-step` and `attribute-format-fetches` carry
//! for their half rows, and one fix cannot close both: those run the other way round -- Metal
//! canonicalizes the widen and we carry the payload and the sign -- so their exact form, a
//! canonicalizing select after the widen, leaves this family exactly as it stands, because here we
//! already answer the canonical NaN. What the two share is one device fact, that this chip's
//! half-to-float widening canonicalizes a NaN: there it is what Metal shows and we do not
//! reproduce, here it is what destroys the payload inside our own path.
//!
//! That stays `divergent` rather than being loosened: `device-nan-encoding`'s own definition says
//! validation derives no tolerance from it, so a tolerance here would be the suite overruling the
//! contract under test. `known-gaps.tsv` carries the nine rows.
//!
//! The lane's four filtered accesses — a linear sample at the centre, a linear sample a quarter
//! texel right of it, a gather, and the same through the runtime sampler at slot 0 — ran on the
//! lane's own generated AIR, and no committed fixture stores a `half` result's raw bits through one
//! of them. They are not ported; the module docs of `runtime_lane::half_texel_nans` record that
//! plain Metal answered `0x7e00` for every one of their NaN lanes.
//!
//! Every word of the output holds [`BLANK`] before the dispatch, so a word the kernel did not reach
//! is a recorded value rather than whatever the allocator left, and the five columns together cover
//! every byte of it.
//!
//! Nothing here can fault a GPU: one compute pass, one texture, one thread a texel inside it, no
//! loop and no coordinate outside the texture.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side, TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The fixture the kernel lives in, and its entry point.
const FIXTURE: &str = "hrd_half_read";

/// The texture's width, which the kernel's `tid mod 64` fixes.
const WIDTH: u32 = 64;

/// The texture's height, which the kernel's division by `float2(64, 16)` fixes: a shorter texture
/// would put the sample's coordinate inside the wrong row.
const HEIGHT: u32 = 16;

/// The texels of one level, which is also the threads the case dispatches.
const TEXELS: usize = (WIDTH * HEIGHT) as usize;

/// The threads a threadgroup takes. Sixty-four, not the grid: the case is about texels, not about
/// the device's threadgroup limit, so it asks for a size every venue holds.
const GROUP: u32 = 64;

/// The bytes the kernel writes a lane: three `uint4`s.
const LANE_BYTES: usize = 48;

/// The byte every output word holds before the dispatch.
pub const BLANK: u8 = 0xab;

/// The halves of the 16-bit float formats, in the lane's own order: quiet and signalling NaNs of
/// both signs and both payload edges, then the infinities, the denormals, the zeros and the
/// normals.
const V: [u16; 24] = [
    0x7e00, 0x7c01, 0x7dff, 0x7fff, 0x7e01, 0x7d55, //
    0xfe00, 0xfc01, 0xfdff, 0xffff, 0xfe01, 0x7f00, //
    0x7c00, 0xfc00, 0x0001, 0x8001, 0x03ff, 0x3c00, //
    0x0000, 0x8000, 0x7bff, 0xfbff, 0x3555, 0xff00,
];

/// The words of the 32-bit float formats: NaNs of every class, then the infinities, the denormals,
/// the zeros and the normals.
const F: [u32; 24] = [
    0x7fc00000, 0x7f800001, 0x7fbfffff, 0xffc00001, 0xff800001, 0x7fffffff, //
    0x7fc12345, 0xffffffff, 0x7f800000, 0x00000001, 0x80000001, 0x3f800000, //
    0x7fe00000, 0xffc00000, 0x7f802000, 0x00000000, 0x7fc02000, 0x7fa00000, //
    0xff800000, 0x80000000, 0x7f7fffff, 0x3eaaaaab, 0x7fd00000, 0xfff00000,
];

/// `RG11B10Float`'s 11-bit red and green codes: NaNs, an infinity, normals, denormals, zero.
const R11: [u32; 12] = [
    0x7c1, 0x7ff, 0x7e0, 0x7d5, 0x7c0, 0x3c0, 0x000, 0x001, 0x7bf, 0x400, 0x5aa, 0x7f0,
];

/// `RG11B10Float`'s 10-bit blue codes.
const B10: [u32; 12] = [
    0x3e1, 0x3ff, 0x3f0, 0x3ea, 0x3e0, 0x1e0, 0x000, 0x001, 0x3df, 0x200, 0x2aa, 0x3f8,
];

/// `RGB9E5Float`'s shared exponents, the largest among them.
const E5: [u32; 8] = [31, 30, 15, 0, 31, 16, 24, 31];

/// How far apart two channels of one texel are in [`V`] and [`F`], so each value lands at every
/// channel position across the texels.
const STRIDE: usize = 5;

/// How a format's texels are written, which is also what its NaN rule turns on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Channels of 16-bit floats, whose bits a one-texel access hands straight back.
    Half,
    /// Channels of 32-bit floats, whose NaN comes back as `0x7e00`.
    Float,
    /// `RG11B10Float`: three channels packed in one word, whose NaN comes back as `0x7e00`.
    SmallFloat,
    /// `RGB9E5Float`: three mantissas and one shared exponent, which decodes finite everywhere.
    SharedExponent,
}

/// The columns the case records, by site name, the byte a lane's column starts at and the elements
/// it holds.
///
/// They partition the lane's 48 bytes, so every byte the kernel could write is recorded exactly
/// once and a word it left as [`BLANK`] shows in whichever column holds it.
const COLUMNS: [(&str, usize, Element, u32); 5] = [
    ("read-through-sample", 0, Element::U16, 4),
    ("nearest-sample", 8, Element::U16, 4),
    ("read", 16, Element::U16, 4),
    ("zeros", 24, Element::U32, 2),
    ("widened", 32, Element::U32, 4),
];

/// The case: one pixel format's `half` words, read three ways.
pub struct HalfTexelNans {
    name: &'static str,
    format: PixelFormat,
    /// The channels the format's texel holds.
    channels: usize,
    kind: Kind,
}

/// Every format the lane reads: the three 16-bit float widths, the three 32-bit float widths, and
/// the two packed ones.
pub fn all() -> Vec<HalfTexelNans> {
    let case = |name, format, channels, kind| HalfTexelNans {
        name,
        format,
        channels,
        kind,
    };
    vec![
        case("half-texel-nans-r16f", PixelFormat::R16Float, 1, Kind::Half),
        case(
            "half-texel-nans-rg16f",
            PixelFormat::RG16Float,
            2,
            Kind::Half,
        ),
        case(
            "half-texel-nans-rgba16f",
            PixelFormat::RGBA16Float,
            4,
            Kind::Half,
        ),
        case(
            "half-texel-nans-r32f",
            PixelFormat::R32Float,
            1,
            Kind::Float,
        ),
        case(
            "half-texel-nans-rg32f",
            PixelFormat::RG32Float,
            2,
            Kind::Float,
        ),
        case(
            "half-texel-nans-rgba32f",
            PixelFormat::RGBA32Float,
            4,
            Kind::Float,
        ),
        case(
            "half-texel-nans-rg11b10f",
            PixelFormat::RG11B10Float,
            3,
            Kind::SmallFloat,
        ),
        case(
            "half-texel-nans-rgb9e5f",
            PixelFormat::RGB9E5Float,
            3,
            Kind::SharedExponent,
        ),
    ]
}

impl HalfTexelNans {
    /// The fixture under its one kernel.
    fn fixture() -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The bytes one texel takes.
    fn texel_bytes(&self) -> u32 {
        match self.kind {
            Kind::Half => 2 * self.channels as u32,
            Kind::Float => 4 * self.channels as u32,
            Kind::SmallFloat | Kind::SharedExponent => 4,
        }
    }

    /// The texture's bytes, row major: the format's own encoding of texel `index`.
    fn texels(&self) -> Vec<u8> {
        (0..TEXELS)
            .flat_map(|index| match self.kind {
                Kind::Half => (0..self.channels)
                    .flat_map(|channel| V[(index + STRIDE * channel) % V.len()].to_le_bytes())
                    .collect::<Vec<u8>>(),
                Kind::Float => (0..self.channels)
                    .flat_map(|channel| F[(index + STRIDE * channel) % F.len()].to_le_bytes())
                    .collect(),
                Kind::SmallFloat => small_float(index).to_le_bytes().to_vec(),
                Kind::SharedExponent => shared_exponent(index).to_le_bytes().to_vec(),
            })
            .collect()
    }

    /// The texture the case reads: 64 by 16 of its format, shader-read and nothing else, since
    /// every access the kernel makes only reads.
    fn spec(&self) -> TextureSpec {
        TextureSpec::new(
            self.format,
            [WIDTH, HEIGHT],
            WIDTH * self.texel_bytes(),
            TextureRole::Read,
        )
    }
}

/// `RG11B10Float`'s texel `index`, as the lane packs one.
fn small_float(index: usize) -> u32 {
    R11[index % 12] | R11[(index + 5) % 12] << 11 | B10[(index + 3) % 12] << 22
}

/// `RGB9E5Float`'s texel `index`, as the lane packs one.
fn shared_exponent(index: usize) -> u32 {
    let at = index as u32;
    let (red, green, blue) = ((at * 37) & 511, (at * 101 + 7) & 511, (at * 13 + 300) & 511);
    red | green << 9 | blue << 18 | E5[index % 8] << 27
}

/// The output buffer before the dispatch: three `uint4`s a lane, every byte [`BLANK`].
fn blank_output() -> Vec<u8> {
    vec![BLANK; TEXELS * LANE_BYTES]
}

/// One column's bytes gathered out of the whole output: `width` elements of `element` from
/// `offset` of each lane, the lanes in order.
///
/// Gathered rather than recorded as a column of the whole buffer, because a site holding the whole
/// buffer five times over would store the same 48 KiB five times. The five columns partition the
/// lane, so nothing is lost by it.
fn column(output: &[u8], offset: usize, element: Element, width: u32) -> Vec<u8> {
    let bytes = element.size() as usize * width as usize;
    (0..TEXELS)
        .flat_map(|lane| {
            let at = lane * LANE_BYTES + offset;
            output[at..at + bytes].to_vec()
        })
        .collect()
}

/// The layout a gathered column is compared under: one row a lane, so the element a diff names is
/// the texel and the channel inside it.
fn layout(element: Element, width: u32) -> ByteLayout {
    let pitch = element.size() * width;
    let length = pitch * TEXELS as u32;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(length),
        },
        element,
        row_offset: 0,
        row_elements: width,
        row_pitch: pitch,
        rows: TEXELS as u32,
        image_pitch: length,
        images: 1,
        // A row is its elements and the rows abut, so there is no byte outside one.
        gaps: Gaps::Ignored,
    }
}

impl Case for HalfTexelNans {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A **notice declaration**, not a tolerance: it says what our side announces, so the case
        // can be read beside Metal, which announces nothing, and it loosens **no recorded word**.
        // It is chosen from the case's own shape -- a `half` result of a float texel, which is the
        // access the runtime raises the kind for -- never from the device.
        //
        // Nothing loosens a stored half here, and nothing may: `device-nan-encoding`'s own
        // definition in `pipeline/artifact.rs` ends "Validation derives no tolerance from it", so a
        // tolerance written from it would be this suite overruling the contract it exists to test.
        // The three 16-bit float formats are therefore left **divergent** on this round's MoltenVK,
        // with the measured difference in the module docs above.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for all 8 rows of the family and each
        // was divergent on that field and nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::DeviceNanEncoding.code()),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: FIXTURE.to_owned(),
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
            // The format is the question the case asks.
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("texture:texels", &self.texels()),
            Input::bytes("buffer:blank", &blank_output()),
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
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(&self.spec(), &self.texels()),
        )?;
        let output = made(sites, "buffer", side.buffer(&blank_output()))?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                // The one texture at both slots: the `access::sample` declaration at 0 and the
                // `access::read` one at 1 read the same bytes.
                &Bindings::new(&[texture, texture], &[output]),
                &[],
                [TEXELS as u32, 1, 1],
                [GROUP, 1, 1],
            ),
        )?;
        let status = call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        if status != Value::unit_variant("Completed") {
            return Err(Abort {
                at: "status".to_owned(),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        let read = call(
            sites,
            "read-back",
            side.read_buffer(output, 0, TEXELS * LANE_BYTES),
            |_| Value::Absent,
        )?;
        let mut words = Node::branch("words", Vec::new());
        for (name, offset, element, width) in COLUMNS {
            words = words.with(Node::leaf(
                name,
                Observed::bytes(
                    column(&read, offset, element, width),
                    layout(element, width),
                ),
            ));
        }
        sites.push(words);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The exact `f32` bits of half `bits`, denormals included.
    fn widened(bits: u16) -> u32 {
        let sign = u32::from(bits >> 15) << 31;
        let exponent = u32::from((bits >> 10) & 0x1f);
        let mantissa = u32::from(bits & 0x3ff);
        match exponent {
            0x1f => sign | 0x7f80_0000 | mantissa << 13,
            0 => sign | (mantissa as f32 * 2f32.powi(-24)).to_bits(),
            raised => sign | (raised + 112) << 23 | mantissa << 13,
        }
    }

    /// What channel `channel` of texel `index` holds, as far as a NaN goes: the NaN's bits in the
    /// format's own encoding, or `None` where the channel is not a NaN.
    ///
    /// This is `runtime_lane::half_texel_nans`'s own `class`, restated so the cross-check below has
    /// the lane's rule to hold Metal's recording to.
    fn nan_bits(case: &HalfTexelNans, index: usize, channel: usize) -> Option<u32> {
        if channel >= case.channels {
            return None;
        }
        let (all_ones, mantissa) = match case.kind {
            Kind::Half => {
                let bits = u32::from(V[(index + STRIDE * channel) % V.len()]);
                ((bits >> 10) & 31 == 31, bits & 0x3ff)
            }
            Kind::Float => {
                let bits = F[(index + STRIDE * channel) % F.len()];
                ((bits >> 23) & 255 == 255, bits & 0x7f_ffff)
            }
            Kind::SmallFloat => {
                let code = [
                    R11[index % 12],
                    R11[(index + 5) % 12],
                    B10[(index + 3) % 12],
                ][channel];
                let width = if channel < 2 { 6 } else { 5 };
                (code >> width == 31, code & ((1 << width) - 1))
            }
            // Every code decodes to a finite value, so no channel of it is a NaN.
            Kind::SharedExponent => return None,
        };
        let bits = match case.kind {
            Kind::Half => u32::from(V[(index + STRIDE * channel) % V.len()]),
            Kind::Float => F[(index + STRIDE * channel) % F.len()],
            _ => 0,
        };
        (all_ones && mantissa != 0).then_some(bits)
    }

    /// The `half` word Metal's rule gives channel `channel` of texel `index` for a one-texel
    /// access, or `None` where the rule names no exact word.
    ///
    /// The rule is the lane's: a 16-bit float texel's NaN keeps its own bits; every other NaN is
    /// `0x7e00`; a missing green or blue reads 0 and a missing alpha 1. A finite channel of a
    /// format whose conversion to `half` the lane never pinned has no exact word here, and the
    /// cross-check only asks that it is not a NaN.
    fn metal_half(case: &HalfTexelNans, index: usize, channel: usize) -> Option<u16> {
        if channel >= case.channels {
            return Some(if channel == 3 { 0x3c00 } else { 0x0000 });
        }
        match (nan_bits(case, index, channel), case.kind) {
            (Some(bits), Kind::Half) => Some(bits as u16),
            (Some(_), _) => Some(0x7e00),
            (None, Kind::Half) => Some(V[(index + STRIDE * channel) % V.len()]),
            (None, _) => None,
        }
    }

    /// Whether `bits` is a half NaN.
    fn is_nan(bits: u16) -> bool {
        bits & 0x7c00 == 0x7c00 && bits & 0x3ff != 0
    }

    /// The sites the three one-texel accesses are recorded at, in the order the columns hold them.
    const ACCESS_SITES: [&str; 3] = ["read-through-sample", "nearest-sample", "read"];

    /// Whether channel `channel` of texel `index` holds a finite value whose magnitude is past
    /// `half`'s largest, so the conversion to `half` has to overflow.
    ///
    /// Only a 32-bit float texel can: a half channel is already a half, and both packed formats'
    /// largest finite value is inside `half`'s range.
    fn overflows(case: &HalfTexelNans, index: usize, channel: usize) -> bool {
        if case.kind != Kind::Float || channel >= case.channels {
            return false;
        }
        let value = f32::from_bits(F[(index + STRIDE * channel) % F.len()]);
        value.is_finite() && value.abs() > 65504.0
    }

    /// Every case states bytes for its whole level, and the output and the columns line up: a
    /// column that reached past a lane would read another lane's word.
    #[test]
    fn the_columns_partition_the_lane_and_cover_it() {
        for case in all() {
            assert_eq!(
                case.texels().len(),
                TEXELS * case.texel_bytes() as usize,
                "{}",
                case.name
            );
            assert_eq!(case.spec().row_pitch, WIDTH * case.texel_bytes());
        }
        let mut covered = vec![false; LANE_BYTES];
        for (_, offset, element, width) in COLUMNS {
            let bytes = element.size() as usize * width as usize;
            for (byte, held) in covered.iter_mut().enumerate().skip(offset).take(bytes) {
                assert!(!*held, "byte {byte} is in two columns");
                *held = true;
            }
            assert_eq!(layout(element, width).extent(), TEXELS * bytes);
        }
        assert!(covered.iter().all(|&held| held), "{covered:?}");
        assert_eq!(blank_output().len(), TEXELS * LANE_BYTES);
    }

    /// A column gathers the bytes it names and no other: each lane's own slice, in lane order.
    ///
    /// Written as a guard over distinct per-byte values, so a column taking another's offset
    /// cannot pass by coincidence.
    #[test]
    fn a_column_gathers_its_own_bytes() {
        let output: Vec<u8> = (0..TEXELS * LANE_BYTES)
            .map(|at| (at % 251) as u8)
            .collect();
        for (name, offset, element, width) in COLUMNS {
            let bytes = element.size() as usize * width as usize;
            let taken = column(&output, offset, element, width);
            assert_eq!(taken.len(), TEXELS * bytes, "{name}");
            for lane in 0..TEXELS {
                for byte in 0..bytes {
                    assert_eq!(
                        taken[lane * bytes + byte],
                        output[lane * LANE_BYTES + offset + byte],
                        "{name} lane {lane} byte {byte}"
                    );
                }
            }
        }
    }

    /// The cases name distinct rows and distinct formats.
    #[test]
    fn the_cases_name_distinct_formats() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        let formats: BTreeSet<u32> = all().iter().map(|case| case.format.as_raw()).collect();
        assert_eq!(named.len(), all().len());
        assert_eq!(formats.len(), all().len());
    }

    /// The inputs hold the NaNs the family turns on, in every encoding it reads: a signalling half,
    /// a negative half NaN, a signalling and a negative 32-bit NaN, and a NaN code in both of
    /// `RG11B10Float`'s channel widths. A sweep that lost one of these would compare NaN handling
    /// without asking the question that separates the encodings.
    #[test]
    fn the_values_hold_a_nan_in_every_encoding() {
        assert!(V
            .iter()
            .any(|&bits| is_nan(bits) && bits & 0x200 == 0 && bits >> 15 == 0));
        assert!(V.iter().any(|&bits| is_nan(bits) && bits >> 15 == 1));
        assert!(F.contains(&0x7f800001) && F.contains(&0xff800001));
        assert!(R11.iter().any(|&code| code >> 6 == 31 && code & 63 != 0));
        assert!(B10.iter().any(|&code| code >> 5 == 31 && code & 31 != 0));
        // And the exponent table holds the largest shared exponent, whose mantissas still decode
        // finite: the one input that would be a NaN in any other encoding.
        assert!(E5.contains(&31));
    }

    /// Metal's recorded `half` words are the rule the lane measured: a 16-bit float texel's NaN
    /// keeps its bits, every other NaN is `0x7e00`, and `RGB9E5Float` holds no NaN at all.
    ///
    /// This is the cross-check the port exists for. The lane holds its rule in a function and
    /// asserts a live device against it, which says nothing on a host with no Metal; here the rule
    /// is held against Metal's own recorded answer, so a release that changes Metal's conversion
    /// shows as the texel that moved.
    ///
    /// It also pins the one place the three accesses **disagree**, which recording them found:
    /// a 32-bit float texel beyond `half`'s range reads as the infinity of its sign through the
    /// two `read()`s and as 65504 through the nearest sample. The lane allowed either value
    /// without saying which access gave which, because it only ever compared one access at a time.
    /// So the accesses are compared here where the rule names a word and the disagreement is
    /// counted where it does not, rather than asserted away in either direction.
    #[test]
    fn metals_recorded_halves_are_the_rule_the_lane_measured() {
        use metal2vulkan_runtime::MetalLevel;
        for case in all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                // A case recorded in a later commit of this batch.
                continue;
            };
            let answer = recording
                .answer(MetalLevel::NEWEST)
                .unwrap_or_else(|why| panic!("{}: {why}", case.name));
            let taken = answer
                .deterministic()
                .unwrap_or_else(|| panic!("{}: one outcome", case.name));
            let words = taken
                .root
                .child("words")
                .unwrap_or_else(|| panic!("{}: no words site", case.name));
            let bytes_of = |name: &str| {
                let site = words
                    .child(name)
                    .unwrap_or_else(|| panic!("{}: no {name} site", case.name));
                match &site.observed.value {
                    Value::Bytes(bytes) => bytes.clone(),
                    other => panic!("{}: {name} holds {}", case.name, other.kind()),
                }
            };
            let halves: Vec<Vec<u16>> = ["read-through-sample", "nearest-sample", "read"]
                .into_iter()
                .map(|name| {
                    bytes_of(name)
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|pair| u16::from_le_bytes(*pair))
                        .collect()
                })
                .collect();
            for lanes in &halves {
                assert_eq!(lanes.len(), TEXELS * 4, "{}", case.name);
            }
            let mut disagreed = Vec::new();
            for index in 0..TEXELS {
                for channel in 0..4 {
                    let at = 4 * index + channel;
                    match metal_half(&case, index, channel) {
                        // Where the rule names a word, every one of the three one-texel accesses
                        // must answer it.
                        Some(want) => {
                            for (access, lanes) in ACCESS_SITES.iter().zip(&halves) {
                                assert_eq!(
                                    lanes[at], want,
                                    "{} {access} texel {index} channel {channel}: Metal recorded \
                                     {:#06x}, the lane's rule is {want:#06x}",
                                    case.name, lanes[at]
                                );
                            }
                        }
                        // A channel the lane pinned no word for: every access must still answer
                        // something that is not a NaN, and a value beyond `half`'s range must be
                        // the infinity of its sign or the largest half of its sign -- the two
                        // answers the lane recorded as the bound of a native conversion.
                        None => {
                            for (access, lanes) in ACCESS_SITES.iter().zip(&halves) {
                                let got = lanes[at];
                                assert!(
                                    !is_nan(got),
                                    "{} {access} texel {index} channel {channel}: {got:#06x} is a \
                                     NaN where the texel is finite",
                                    case.name
                                );
                                if overflows(&case, index, channel) {
                                    assert!(
                                        got & 0x7fff == 0x7c00 || got & 0x7fff == 0x7bff,
                                        "{} {access} texel {index} channel {channel}: \
                                         {got:#06x} is neither an infinity nor the largest half",
                                        case.name
                                    );
                                }
                            }
                            if halves.iter().any(|lanes| lanes[at] != halves[0][at]) {
                                disagreed.push((index, channel));
                            }
                        }
                    }
                }
            }
            // Where they disagree, it is the overflow lanes alone: nothing else in the sweep
            // separates a fetch from a nearest sample on this chip.
            for &(index, channel) in &disagreed {
                assert!(
                    overflows(&case, index, channel),
                    "{} texel {index} channel {channel}: the accesses differ ({:#06x}, {:#06x}, \
                     {:#06x}) on a channel that does not overflow `half`",
                    case.name,
                    halves[0][4 * index + channel],
                    halves[1][4 * index + channel],
                    halves[2][4 * index + channel]
                );
            }
            // The widening column is the shader's own `float4(half4)`, and it is exact for every
            // half that is not a NaN. A NaN's payload is the one thing a widening may rewrite, so
            // the recording is what says what Metal does there.
            let wide: Vec<u32> = bytes_of("widened")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            assert_eq!(wide.len(), TEXELS * 4, "{}", case.name);
            for (at, &word) in wide.iter().enumerate() {
                let half = halves[0][at];
                match is_nan(half) {
                    true => assert!(
                        f32::from_bits(word).is_nan(),
                        "{} lane {at}: {half:#06x} widened to {word:#010x}, which is no NaN",
                        case.name
                    ),
                    false => assert_eq!(
                        word,
                        widened(half),
                        "{} lane {at}: {half:#06x} widened to {word:#010x}",
                        case.name
                    ),
                }
            }
            // The control the kernel writes zero: the proof the lane ran rather than that the
            // buffer kept its fill.
            let zeros = bytes_of("zeros");
            assert_eq!(zeros.len(), TEXELS * 8, "{}", case.name);
            assert!(zeros.iter().all(|&byte| byte == 0), "{}", case.name);
        }
    }
}
