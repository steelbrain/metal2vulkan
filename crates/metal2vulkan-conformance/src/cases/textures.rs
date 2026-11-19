//! Every uncompressed colour format's texture, ported from
//! `runtime_lane::textures`'s `every_format_makes_a_texture_or_is_refused_typed`.
//!
//! One case a format. Each one makes an 8 by 8 `Shared` 2D texture of its format, read by shaders,
//! at a row pitch one texel wider than its rows, then asks three things of it:
//!
//! - what a **fresh** texture reads (`get_bytes`), which Metal answers as zero;
//! - what it reads back after `replace_region` wrote the case's pattern over its whole level 0;
//! - whether the **bytes between the rows** stayed as they were.
//!
//! The third is why both reads go through [`Side::read_texture_over`] rather than
//! [`Side::read_texture`]: the destination holds [`BEFORE`] before the copy, so a byte the copy did
//! not write is that byte rather than a zero nobody can tell from a written one, and the site's
//! layout compares the gaps ([`Gaps::Compared`]).
//!
//! **What the port adds to the lane.** The lane asserts *rules*: a fresh texture reads zero, the
//! bytes come back, the padding is untouched, a format Metal describes none of is stopped. It holds no Metal value at all, and it walks all 301 formats in one test, so a
//! format the venue has no texture of leaves the test green with nothing compared. Here Metal's own
//! bytes are recorded per format, and a format the venue honestly lacks stops that format's case
//! alone and counts as [`LacksFeature`](crate::verdict::Verdict::LacksFeature) — neither a pass nor
//! a failure, and visible in the matrix.
//!
//! **What is left in the lane.** Whether a texture of each of the 301 formats is made at all, which
//! the lane checks against two rules (the formats Metal describes, and the formats Metal answers
//! nil for). One case per format would record Metal's answer for each, but the suite's
//! `LacksFeature` outcome arrives only when a case stops, so a case covering many formats would
//! have to swallow a venue's gap to keep going. That is a verdict-model question, not something to
//! decide inside a case, so this family covers the formats whose texels are Metal's own bytes and
//! says so.
//!
//! **What the family can tell apart, and what it cannot.** The compared value is bytes moved by
//! `replace_region` and read back by `get_bytes`, so what the format decides here is the texel's
//! size — which sets the extent, the pitch and the padding the layout compares — and whether Metal
//! makes a texture of the format at all. Two formats of the same texel bytes that Metal both makes
//! and copies verbatim are **indistinguishable by construction** in this family: the written
//! pattern is seeded by the format's own ordinal (`TextureRoundTrip::written`), so their rows
//! hold different bytes and compare the same behaviour. Measured, not reasoned: with the runtime
//! leg patched to make every `RGB10A2Uint` texture as `RGB10A2Unorm` instead, this family's
//! `texture-round-trip-91` still read `exact` on an Apple M3 Max, while the same patch to
//! `R16Unorm` — a different texel size — diverged at 128 bytes. A format's *numeric* behaviour is
//! another family's question: `norm_storage_writes` sweeps the normalized packings, and
//! `shared_exponent_storage` `RGB9E5Float`'s word.
//!
//! Nothing here reaches the GPU: a texture is created and the host copies bytes into and out of it.

use crate::case::{Abort, Case, Input, MetalSource, Side, TextureRole, TextureSpec};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;

/// The texture's side in texels, which is the lane's extent.
const SIDE: u32 = 8;

/// The byte every read-back destination holds before the copy, so a byte the copy did not write is
/// a recorded value rather than a zero nobody can read.
pub const BEFORE: u8 = 0xee;

/// The formats this family covers: every format of [`PixelFormat::ALL`] whose texels are Metal's
/// own bytes — an uncompressed colour format of one texel a block — with the bytes one texel takes
/// and the name its case holds.
///
/// The table is the case names' home, because a case names itself with a static string. Nothing
/// here may drift: `tests::the_table_is_exactly_the_formats_whose_texels_are_metals_bytes` holds
/// it against the predicate over every format Metal describes, and
/// `tests::every_case_names_its_own_format` holds each name against its format's ordinal.
///
/// Each row names its format through [`PixelFormat`] rather than by ordinal. The two spellings
/// were both here — the ordinal in the row and the enumerator in a comment beside it — which is
/// the second spelling `crate::texture`'s raw `format` field exists to avoid, and a comment is
/// not something the compiler or `scripts/conformance-coverage/` can hold to the row.
const FORMATS: [(PixelFormat, u32, &str); 76] = [
    (PixelFormat::A8Unorm, 1, "texture-round-trip-1"),
    (PixelFormat::R8Unorm, 1, "texture-round-trip-10"),
    (PixelFormat::R8UnormSRGB, 1, "texture-round-trip-11"),
    (PixelFormat::R8Snorm, 1, "texture-round-trip-12"),
    (PixelFormat::R8Uint, 1, "texture-round-trip-13"),
    (PixelFormat::R8Sint, 1, "texture-round-trip-14"),
    (PixelFormat::R16Unorm, 2, "texture-round-trip-20"),
    (PixelFormat::R16Snorm, 2, "texture-round-trip-22"),
    (PixelFormat::R16Uint, 2, "texture-round-trip-23"),
    (PixelFormat::R16Sint, 2, "texture-round-trip-24"),
    (PixelFormat::R16Float, 2, "texture-round-trip-25"),
    (PixelFormat::RG8Unorm, 2, "texture-round-trip-30"),
    (PixelFormat::RG8UnormSRGB, 2, "texture-round-trip-31"),
    (PixelFormat::RG8Snorm, 2, "texture-round-trip-32"),
    (PixelFormat::RG8Uint, 2, "texture-round-trip-33"),
    (PixelFormat::RG8Sint, 2, "texture-round-trip-34"),
    (PixelFormat::B5G6R5Unorm, 2, "texture-round-trip-40"),
    (PixelFormat::A1BGR5Unorm, 2, "texture-round-trip-41"),
    (PixelFormat::ABGR4Unorm, 2, "texture-round-trip-42"),
    (PixelFormat::BGR5A1Unorm, 2, "texture-round-trip-43"),
    (PixelFormat::BGRA4Unorm, 2, "texture-round-trip-44"),
    (PixelFormat::R32Uint, 4, "texture-round-trip-53"),
    (PixelFormat::R32Sint, 4, "texture-round-trip-54"),
    (PixelFormat::R32Float, 4, "texture-round-trip-55"),
    (PixelFormat::RG16Unorm, 4, "texture-round-trip-60"),
    (PixelFormat::RG16Snorm, 4, "texture-round-trip-62"),
    (PixelFormat::RG16Uint, 4, "texture-round-trip-63"),
    (PixelFormat::RG16Sint, 4, "texture-round-trip-64"),
    (PixelFormat::RG16Float, 4, "texture-round-trip-65"),
    (PixelFormat::RGBA8Unorm, 4, "texture-round-trip-70"),
    (PixelFormat::RGBA8UnormSRGB, 4, "texture-round-trip-71"),
    (PixelFormat::RGBA8Snorm, 4, "texture-round-trip-72"),
    (PixelFormat::RGBA8Uint, 4, "texture-round-trip-73"),
    (PixelFormat::RGBA8Sint, 4, "texture-round-trip-74"),
    (PixelFormat::BGRA8Unorm, 4, "texture-round-trip-80"),
    (PixelFormat::BGRA8UnormSRGB, 4, "texture-round-trip-81"),
    (PixelFormat::RGB10A2Unorm, 4, "texture-round-trip-90"),
    (PixelFormat::RGB10A2Uint, 4, "texture-round-trip-91"),
    (PixelFormat::RG11B10Float, 4, "texture-round-trip-92"),
    (PixelFormat::RGB9E5Float, 4, "texture-round-trip-93"),
    (PixelFormat::BGR10A2Unorm, 4, "texture-round-trip-94"),
    (PixelFormat::BGR10XR, 4, "texture-round-trip-554"),
    (PixelFormat::BGR10XRSRGB, 4, "texture-round-trip-555"),
    (PixelFormat::RG32Uint, 8, "texture-round-trip-103"),
    (PixelFormat::RG32Sint, 8, "texture-round-trip-104"),
    (PixelFormat::RG32Float, 8, "texture-round-trip-105"),
    (PixelFormat::RGBA16Unorm, 8, "texture-round-trip-110"),
    (PixelFormat::RGBA16Snorm, 8, "texture-round-trip-112"),
    (PixelFormat::RGBA16Uint, 8, "texture-round-trip-113"),
    (PixelFormat::RGBA16Sint, 8, "texture-round-trip-114"),
    (PixelFormat::RGBA16Float, 8, "texture-round-trip-115"),
    (PixelFormat::BGRA10XR, 8, "texture-round-trip-552"),
    (PixelFormat::BGRA10XRSRGB, 8, "texture-round-trip-553"),
    (PixelFormat::RGBA32Uint, 16, "texture-round-trip-123"),
    (PixelFormat::RGBA32Sint, 16, "texture-round-trip-124"),
    (PixelFormat::RGBA32Float, 16, "texture-round-trip-125"),
    (PixelFormat::ARGB8Unorm, 4, "texture-round-trip-84"),
    (PixelFormat::R16Bfloat, 2, "texture-round-trip-26"),
    (PixelFormat::RG16Bfloat, 4, "texture-round-trip-66"),
    (PixelFormat::RGBA16Bfloat, 8, "texture-round-trip-116"),
    (PixelFormat::RGBX8Unorm, 4, "texture-round-trip-300"),
    (PixelFormat::RGBX8UnormSRGB, 4, "texture-round-trip-301"),
    (PixelFormat::RGBX8Snorm, 4, "texture-round-trip-302"),
    (PixelFormat::RGBX8Uint, 4, "texture-round-trip-303"),
    (PixelFormat::RGBX8Sint, 4, "texture-round-trip-304"),
    (PixelFormat::RGBX16Uint, 8, "texture-round-trip-313"),
    (PixelFormat::RGBX16Sint, 8, "texture-round-trip-314"),
    (PixelFormat::RGBX16Float, 8, "texture-round-trip-315"),
    (PixelFormat::RGBX32Uint, 16, "texture-round-trip-323"),
    (PixelFormat::RGBX32Sint, 16, "texture-round-trip-324"),
    (PixelFormat::RGBX32Float, 16, "texture-round-trip-325"),
    (PixelFormat::R10UnormX6, 2, "texture-round-trip-576"),
    (PixelFormat::R10UnormX6SRGB, 2, "texture-round-trip-577"),
    (PixelFormat::RG10UnormX12, 4, "texture-round-trip-578"),
    (PixelFormat::RG10UnormX12SRGB, 4, "texture-round-trip-579"),
    (PixelFormat::RGB10A2UnormSRGB, 4, "texture-round-trip-586"),
];

/// The bytes of one texel of `pf`, for the uncompressed colour formats this family round-trips:
/// none for a format whose image does not hold Metal's bytes, whose host copies the runtime
/// refuses. The lane's own predicate, over the translator's public format table. A format held
/// for copies alone (`metal::format::copies_only`) is
/// [`format_footprints`](crate::cases::format_footprints)'s, which records the same host round
/// trip of it beside the footprints and texture types Metal answers for it, so it is none here.
///
/// It is the table's guard and nothing else reads it: [`FORMATS`] holds the names a case must
/// answer with as static strings, so the list cannot be built from the predicate at run time, and
/// this is what keeps the two from parting.
#[cfg(test)]
fn texel_bytes(pf: PixelFormat) -> Option<u32> {
    if metal2vulkan::metal::format::copies_only(pf) {
        return None;
    }
    let block = metal2vulkan::metal::format::byte_format(pf)?.block()?;
    let compressed =
        block.texels > 1 || (block.packed_bits.is_none() && block.component_bits.is_none());
    (block.color.is_some() && !compressed).then_some(block.size)
}

/// `n` bytes of the lane's pattern, seeded by `seed`.
fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|at| (at as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// A case: one format's texture.
pub struct TextureRoundTrip {
    name: &'static str,
    format: PixelFormat,
    /// The bytes one texel takes.
    texel: u32,
}

/// Every case of the family: one a format.
pub fn all() -> Vec<TextureRoundTrip> {
    FORMATS
        .iter()
        .map(|&(format, texel, name)| TextureRoundTrip {
            name,
            format,
            texel,
        })
        .collect()
}

impl TextureRoundTrip {
    /// The bytes one row of texels takes.
    fn row(&self) -> u32 {
        SIDE * self.texel
    }

    /// The row pitch the case reads and writes at: its rows and one texel of padding after each,
    /// which is what makes the gaps a question.
    fn pitch(&self) -> u32 {
        self.row() + self.texel
    }

    /// The whole destination of a read-back, before the copy.
    fn before(&self) -> Vec<u8> {
        vec![BEFORE; (self.pitch() * SIDE) as usize]
    }

    /// The bytes `replace_region` writes: the lane's pattern over the whole destination, padding
    /// included, so what lands in the gaps is the copy's business and not the case's.
    fn written(&self) -> Vec<u8> {
        pattern((self.pitch() * SIDE) as usize, self.format.as_raw() as u8)
    }

    /// The texture the case makes: one level, `Shader_Read` usage and `Shared` storage, which is
    /// what `MTLTextureDescriptor`'s own defaults give and what the lane's descriptor holds.
    fn spec(&self) -> TextureSpec {
        TextureSpec::new(self.format, [SIDE, SIDE], self.pitch(), TextureRole::Read)
    }

    /// The layout a read-back's bytes are compared under: `SIDE` rows of the row's bytes at the
    /// case's pitch, byte by byte, with the bytes between the rows compared too.
    fn layout(&self) -> ByteLayout {
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [SIDE, SIDE, 1],
            },
            element: Element::U8,
            row_offset: 0,
            row_elements: self.row(),
            row_pitch: self.pitch(),
            rows: SIDE,
            image_pitch: self.pitch() * SIDE,
            images: 1,
            // The question the padding asks: a copy that wrote a byte between the rows differs
            // from Metal, and this is what says so.
            gaps: Gaps::Compared,
        }
    }
}

/// The site a creator's answer records at, with the side's own error kept out of it.
///
/// Metal has one way to refuse a texture — `newTextureWithDescriptor:` answers nil — and no typed
/// error at all, while the runtime refuses with one of several. So the comparable fact is whether a
/// texture of the format was made, and the record holds that and no more: a side that made none
/// says `Err(NotMade)` whatever it would have called the reason. Our *reason* has no Metal
/// counterpart, so it stays out, as the design says an answer with no Metal counterpart must.
///
/// This keeps the verdict that matters. A format Metal makes a texture of and we refuse is Metal
/// answering `Ok` against our `Err`, which the comparison reads as
/// [`Refused`](crate::verdict::Verdict::Refused) — never a pass, and never buried in a count.
///
/// What it costs: our typed error is not in the record, so a `known-gaps.tsv` row for such a
/// format reads `Err(NotMade)` and cannot tell one refusal of ours from another at the same site.
/// The reason is one call away in the runtime and the lane keeps it, so the cost is a reader's
/// convenience rather than a fact nobody holds — and it buys the alternative's absence: an error
/// payload in the record would read as a divergence on the day Metal refuses a format too, when
/// the two sides in fact agree.
fn made_value(made: bool) -> Value {
    if made {
        Value::ok(Value::Absent)
    } else {
        Value::error("NotMade", Value::Absent)
    }
}

impl Case for TextureRoundTrip {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes(
                "extent-and-pitch",
                &[SIDE, SIDE, self.pitch()]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            Input::bytes("before", &self.before()),
            Input::bytes("replace_region:bytes", &self.written()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        // Made with no bytes, because what a *fresh* texture reads is the first question.
        let texture = match side.texture_2d_in_role(&self.spec(), &[]) {
            Ok(texture) => {
                sites.value("texture", made_value(true));
                texture
            }
            Err(failure) => {
                sites.value("texture", made_value(false));
                // The run stops: a format with no texture has no bytes to read, and stopping is
                // what turns a venue's own gap into `LacksFeature` rather than a difference.
                return Err(Abort {
                    at: "texture".to_owned(),
                    failure,
                });
            }
        };
        let read = |side: &mut dyn Side, sites: &mut Sites, at: &str| -> Result<Vec<u8>, Abort> {
            match side.read_texture_over(texture, [SIDE, SIDE], self.pitch(), &self.before()) {
                Ok(bytes) => Ok(bytes),
                Err(failure) => {
                    sites.push(Node::leaf(at, Observed::value(failure.recorded())));
                    Err(Abort {
                        at: at.to_owned(),
                        failure,
                    })
                }
            }
        };
        let fresh = read(side, sites, "fresh")?;
        sites.leaf("fresh", Observed::bytes(fresh, self.layout()));
        match side.replace_region(texture, [SIDE, SIDE], self.pitch(), &self.written()) {
            Ok(()) => sites.value("replace_region", Value::ok(Value::Absent)),
            Err(failure) => {
                sites.push(Node::leaf(
                    "replace_region",
                    Observed::value(failure.recorded()),
                ));
                return Err(Abort {
                    at: "replace_region".to_owned(),
                    failure,
                });
            }
        }
        let back = read(side, sites, "round-trip")?;
        sites.leaf("round-trip", Observed::bytes(back, self.layout()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The table is exactly the formats whose texels are Metal's own bytes, with the bytes a texel
    /// each one takes. Without this the table would be a hand-kept list that a new format or a
    /// changed block silently falls out of, and the family's coverage would shrink unannounced.
    #[test]
    fn the_table_is_exactly_the_formats_whose_texels_are_metals_bytes() {
        let tabled: BTreeSet<PixelFormat> = FORMATS.iter().map(|&(format, _, _)| format).collect();
        assert_eq!(tabled.len(), FORMATS.len(), "the table repeats a format");
        let mut wanted = BTreeSet::new();
        for &pf in PixelFormat::ALL {
            let Some(bytes) = texel_bytes(pf) else {
                continue;
            };
            wanted.insert(pf);
            let row = FORMATS
                .iter()
                .find(|&&(format, _, _)| format == pf)
                .unwrap_or_else(|| panic!("{pf:?} is not in the table"));
            assert_eq!(row.1, bytes, "{pf:?} takes other bytes a texel");
        }
        let extra: Vec<&PixelFormat> = tabled.difference(&wanted).collect();
        assert!(
            extra.is_empty(),
            "the table holds {extra:?}, which no longer round-trip"
        );
    }

    /// Every case's name is its own, and names its format, so a recording cannot be read as
    /// another format's.
    #[test]
    fn every_case_names_its_own_format() {
        let cases = all();
        assert_eq!(cases.len(), FORMATS.len(), "a format made no case");
        let names: BTreeSet<&str> = cases.iter().map(|case| case.name).collect();
        assert_eq!(names.len(), cases.len());
        for case in &cases {
            assert!(
                case.name.ends_with(&format!("-{}", case.format.as_raw())),
                "{} does not name {:?}",
                case.name,
                case.format
            );
        }
    }

    /// The layout a read-back compares under covers every byte the read fills, names the byte as
    /// its element, and compares the padding. A layout that stopped at the rows would let a copy
    /// scribble between them unnoticed, which is the assertion the lane makes and this port must
    /// keep.
    #[test]
    fn the_layout_covers_the_padding() {
        for case in all() {
            let layout = case.layout();
            assert_eq!(layout.gaps, Gaps::Compared, "{}", case.name);
            assert_eq!(
                layout.extent(),
                case.before().len() - case.texel as usize,
                "{}",
                case.name
            );
            assert_eq!(layout.row_elements + case.texel, layout.row_pitch);
        }
    }

    /// The pattern the case writes differs from the byte the destination holds before a read, so a
    /// round trip that copied nothing reads as a difference rather than as a match.
    #[test]
    fn the_written_pattern_is_not_the_byte_before_it() {
        for case in all() {
            let written = case.written();
            assert!(written.iter().any(|&byte| byte != BEFORE), "{}", case.name);
        }
    }
}
