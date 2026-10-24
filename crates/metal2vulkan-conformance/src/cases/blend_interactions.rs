//! Pairs of a render pipeline's color state, where one field's answer depends on another's.
//!
//! [`super::blend_sweeps`] sweeps each field of `MTLRenderPipelineColorAttachmentDescriptor` **one
//! at a time**: every factor in each of the four slots, every operation, all sixteen write masks,
//! the blend color at its boundaries. That is depth along one axis and it is what
//! `docs/CONFORMANCE-COVERAGE.md` names as its own blind spot: "a type whose values are all
//! compared says nothing about the pairs and triples of them".
//!
//! This family is the pairs. Two of them, each chosen because **a cell of the cross answers a
//! question neither of its rows or columns can**:
//!
//! - **The write mask crossed with the blend** ([`Subject::MaskBlend`]). Metal's `writeMask` decides
//!   which channels the blended result replaces, **after** the blend has read the destination. The
//!   per-value sweeps cannot see that order: `blend-sweep-write-masks` crosses all sixteen masks
//!   with exactly one blend, `One`, `One`, `Add`, and that is the one blend under which a mask
//!   applied **before** the blend stores the same texel as a mask applied after — `0 * 1 + d * 1`
//!   is `d`, which is what the mask was supposed to keep. So the single cell the suite had is the
//!   single cell that is blind. This case crosses four masks with eight blends, and names which of
//!   the eight can see the order and which cannot.
//! - **The blend crossed with the attachment's format class** ([`Subject::Format`]). **No case of
//!   the suite blends into a normalized or an sRGB render target**: every blending family reaches
//!   for a float format (`blend_sweeps` `RGBA32Float`, `half_output_blends` the 16- and 32-bit
//!   floats, `extended_range_blends` Apple's XR formats) and the one 8-bit unorm blend is
//!   `rotated_targets`' IOSurface. A float target cannot ask where the clamp sits, because nothing
//!   clamps; a normalized one cannot ask what an unclamped product is. The pair can ask both, and
//!   that is what decides whether a blend color outside `[0, 1]` is clamped **as a constant**
//!   before it multiplies or **as a result** after.
//!
//! # What the pairs found that the per-value sweeps could not
//!
//! Each case's own section below states its cells. Across both, four results are the family's
//! reason to exist:
//!
//! 1. Of the five `MTLBlendOperation` values under factors `One` and `One`, only **`Min`** and
//!    **`Subtract`** can see a mask applied before the blend instead of after. `Add`, `Max` and
//!    `ReverseSubtract` store the destination either way in a channel the mask drops, so they are
//!    blind **by construction** and not by accident (the `BLIND_TO_MASK_ORDER` table of this module's tests is the list and the
//!    arithmetic).
//! 2. A write mask that drops the **alpha** channel must not change what an **RGB** factor reads
//!    from the destination's alpha. `da-zero` and `sas-zero` are the two cells that ask it, and
//!    neither a mask sweep nor a factor sweep can: the mask sweep's blend reads no other channel,
//!    and the factor sweep's mask keeps every channel.
//! 3. **Metal clamps the blend color to `[0, 1]` before a factor multiplies by it on a normalized
//!    attachment, and does not on a float one.** Measured, and the pair is what measures it: under a
//!    blend color of `(2, 3, 1.5, 2.5)` and a factor of `BlendColor`, the `RGBA8Unorm` case stores
//!    the source's own codes `(40, 90, 160, 200)` — the codes a factor of exactly 1 gives — while
//!    the `RGBA32Float` case stores `0.3137`, `1.0588`, `0.9412`, `1.9608`, the same source times
//!    the same unclamped color. The float answer alone is `blend_sweeps`' and reads as "the color is
//!    not clamped"; the normalized answer alone reads as "the result is clamped like any other
//!    overflow". Only the cross says which, and it says the factor was 1 and not 2.
//!
//!    Which cells could have said it is the `CLAMP_CELLS` table of this module's tests, derived
//!    rather than asserted: on the
//!    normalized attachment the two `BlendColor` rows separate the orders and the two
//!    `OneMinusBlendColor` rows cannot, and on the float attachment all four do. Whether the clamp
//!    sits on the **color** or on the **factor** is not decidable at all —
//!    `clamp(1 - c)` and `1 - clamp(c)` are one function — and the family says so rather than
//!    picking one.
//! 4. Blending into an `RGBA8Unorm_sRGB` attachment is compared here for the first time. Its
//!    **alpha** channel is linear and is held against the same model as the plain unorm case; its
//!    color channels are Metal's own decode and encode, which `super::srgb_fetch` measured to
//!    differ from the textbook round trip in 15 of 256 codes, so they are **recorded** rather than
//!    derived. The case says which half is which rather than claiming a model for both.
//!
//! # What this family deliberately does not re-author
//!
//! **The separate RGB and alpha halves of the blend are already compared non-vacuously**, so no
//! case here crosses an RGB factor with an alpha factor. `blend_sweeps`' four slot sweeps each put
//! a moving factor in one slot and a **different** constant in the slot beside it — the source-RGB
//! sweep is `(factor, Zero, Add)` for color against `(One, Zero, Add)` for alpha — so an
//! implementation that collapsed either half onto the other changes those rows. The one collapse
//! those four sweeps are blind to is of the **destination** slots, whose constant is `Zero` in
//! three of the four; `blend-sweep-destination-rgb` is the fourth and it does see it. A fifth case
//! crossing the halves would compare a cross whose answer is a function of each half alone, which
//! is the cross this family's own rule says not to author.
//!
//! # The inputs
//!
//! One source color for every pixel, read from a buffer rather than interpolated, and a
//! **destination that varies per pixel**. That is `blend_sweeps`' arrangement and it is here for
//! its reasons: a constant source makes a factor that varies across the image one that read the
//! destination, and a varying destination alpha is what gives `SourceAlphaSaturated` both branches
//! of its `min`.
//!
//! The fixture is the owned `rt6_samples`: its full-screen triangle and its `rt6_fill_rgba32float`,
//! which returns the `float4` of buffer 0 at the fragment's own texel. Nothing is interpolated, so
//! no answer here depends on a venue's rasterizer agreeing with Metal's in the last bit.
//!
//! # Nothing here can fault a GPU
//!
//! Single-sampled targets of the case's own memory, one full-screen triangle a pass, no loop, no
//! indexing outside a bound buffer, and every pass's attachment written and read back by the same
//! command buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, BlendColorChannels, ColorState, PassColor, Plane, ReadAt, RenderPass,
    RenderPipelineSpec, RenderStep, RenderWork, TargetSpec,
};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{
    BlendFactor as F, BlendOperation as O, ColorWriteMask as M, PixelFormat,
};

use super::extended_range_targets::{
    fixture, fixture_inputs, not_read, program, shader_refs, shape, FILLS, FILL_FLOAT, VERTEX,
};

// ---- The two subjects --------------------------------------------------------------------------

/// Which pair a case crosses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subject {
    /// `writeMask` crossed with the blend, in a 32-bit float attachment where every number is
    /// exact.
    MaskBlend,
    /// The blend crossed with the attachment's format class: the same twelve blends into a 32-bit
    /// float, an 8-bit normalized and an 8-bit sRGB attachment.
    Format(PixelFormat),
}

/// One case of the family.
pub struct BlendInteraction {
    name: &'static str,
    subject: Subject,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<BlendInteraction> {
    vec![
        BlendInteraction {
            name: "blend-interaction-mask-blend",
            subject: Subject::MaskBlend,
        },
        BlendInteraction {
            name: "blend-interaction-format-float32",
            subject: Subject::Format(PixelFormat::RGBA32Float),
        },
        BlendInteraction {
            name: "blend-interaction-format-unorm8",
            subject: Subject::Format(PixelFormat::RGBA8Unorm),
        },
        BlendInteraction {
            name: "blend-interaction-format-srgb8",
            subject: Subject::Format(PixelFormat::RGBA8UnormSRGB),
        },
    ]
}

// ---- The mask-and-blend cross ------------------------------------------------------------------

/// The mask cross's target: 64 pixels, each with its own destination texel.
const MASK_SIDE: u32 = 8;

/// The pixels one draw of the mask cross covers.
const MASK_PIXELS: usize = (MASK_SIDE * MASK_SIDE) as usize;

/// The unit the mask cross's inputs are multiples of.
const Q: f32 = 1.0 / 64.0;

/// The color the fragment function returns at every pixel of the mask cross: `(6, 10, 14, 22) / 64`.
///
/// The alpha is `22/64` so that `1 - destinationAlpha` crosses it inside the image: the destination
/// alpha sweeps `p/64` over the 64 pixels, so `min(sourceAlpha, 1 - destinationAlpha)` takes its
/// left branch above pixel 42 and its right branch below, which is what makes
/// `SourceAlphaSaturated` equal to neither of the factors beside it.
/// `tests::the_saturated_factor_takes_both_of_its_branches` holds that.
const MASK_SOURCE: [f32; 4] = [6.0 * Q, 10.0 * Q, 14.0 * Q, 22.0 * Q];

/// The destination texel of pixel `pixel` of the mask cross.
///
/// Each color channel steps by an odd multiplier, so each takes all 64 values over the image and no
/// two channels hold one sequence; the alpha channel is `p / 64`, which is what crosses the source's
/// alpha.
fn mask_destination(pixel: usize) -> [f32; 4] {
    let p = pixel as u32;
    [
        f32::from(((11 * p + 5) % 64) as u16) * Q,
        f32::from(((19 * p + 2) % 64) as u16) * Q,
        f32::from(((29 * p + 7) % 64) as u16) * Q,
        f32::from(p as u16) * Q,
    ]
}

/// The four write masks the cross sweeps, with the channels each keeps.
///
/// `all` is Metal's default and the cross's control column: those rows are what the per-value blend
/// sweeps already compare, and a falsification of the mask handling must leave them exact. The other
/// three each drop a different set — the alpha channel alone, every color channel, and all but one
/// color channel — so every one of the four channels is dropped by some row and kept by another.
fn mask_columns() -> Vec<(&'static str, M)> {
    vec![
        ("all", M::ALL),
        ("rgb", M::RED.union(M::GREEN).union(M::BLUE)),
        ("a", M::ALPHA),
        ("g", M::GREEN),
    ]
}

/// The eight blends the cross sweeps, each with the name its site takes.
///
/// `off` is the cross's control row: blending disabled is what `color_write_masks` already compares
/// against Metal under three masks, so a falsification of the blend handling must leave it exact.
/// The five `one-one` operations are there because the order of mask and blend is a question about
/// the **operation**, and `blend-sweep-write-masks` asked it under `Add` alone. `one-zero-add` is
/// the blend that reads no destination at all, and the last two are the cells where an **RGB**
/// factor reads the **destination's alpha**, which is the channel a mask of `rgb` drops.
fn blend_rows() -> Vec<(&'static str, Option<Blend>)> {
    vec![
        ("off", None),
        ("one-zero-add", Some(Blend::added(F::One, F::Zero))),
        ("one-one-add", Some(Blend::added(F::One, F::One))),
        (
            "one-one-min",
            Some(Blend::separate(
                (F::One, F::One, O::Min),
                (F::One, F::One, O::Min),
            )),
        ),
        (
            "one-one-max",
            Some(Blend::separate(
                (F::One, F::One, O::Max),
                (F::One, F::One, O::Max),
            )),
        ),
        (
            "one-one-subtract",
            Some(Blend::separate(
                (F::One, F::One, O::Subtract),
                (F::One, F::One, O::Subtract),
            )),
        ),
        (
            "one-one-reverse-subtract",
            Some(Blend::separate(
                (F::One, F::One, O::ReverseSubtract),
                (F::One, F::One, O::ReverseSubtract),
            )),
        ),
        (
            "da-zero-add",
            Some(Blend::separate(
                (F::DestinationAlpha, F::Zero, O::Add),
                (F::One, F::Zero, O::Add),
            )),
        ),
        (
            "sas-zero-add",
            Some(Blend::separate(
                (F::SourceAlphaSaturated, F::Zero, O::Add),
                (F::One, F::Zero, O::Add),
            )),
        ),
    ]
}

/// The blends of the cross under which a write mask applied **before** the blend stores the same
/// texel as one applied **after**, with the arithmetic that says so.
///
/// This is the family's first result and it is a statement about Metal's model rather than about
/// any device: in a channel the mask drops, the right answer is the destination `d`, and a mask
/// applied to the fragment's output instead would blend a zero there. Where
/// `blend(0, d) == d` the two orders are one texel and no cell of the cross can tell them apart.
///
/// `tests::the_named_blends_are_the_ones_blind_to_the_mask_order` derives this list from the
/// blends themselves rather than trusting it, so a blend added to [`blend_rows`] joins the list or
/// fails.
#[cfg(test)]
const BLIND_TO_MASK_ORDER: &[(&str, &str)] = &[
    (
        "one-one-add",
        "0 * 1 + d * 1 is d: the destination survives a zeroed source either way",
    ),
    (
        "one-one-max",
        "max(0, d) is d for a destination that is never negative",
    ),
    ("one-one-reverse-subtract", "d * 1 - 0 * 1 is d"),
];

/// The pairs of rows of the mask cross that give one answer, with the reason they cannot differ.
///
/// One pair, and it is Metal's own definition rather than a badly chosen input: blending **off**
/// stores the fragment's output, and so does blending on with a source factor of `One`, a
/// destination factor of `Zero` and `Add`. `blend-sweep-enabled` is the per-value case that
/// measures it; here it is the reason the cross's control column and its first blend row agree, and
/// `tests::the_mask_cross_separates_every_mask_and_every_blend` fails on any other pair that does.
#[cfg(test)]
const COLLAPSED: &[(&str, &str, &str)] = &[(
    "off",
    "one-zero-add",
    "blending off stores the source, and so does a source factor of one with a destination factor \
     of zero",
)];

// ---- The format-class cross --------------------------------------------------------------------

/// The format cross's target side: 256 pixels, so every one of an 8-bit channel's 256 codes is some
/// pixel's destination in every channel.
const FORMAT_SIDE: u32 = 16;

/// The pixels one draw of the format cross covers.
const FORMAT_PIXELS: usize = (FORMAT_SIDE * FORMAT_SIDE) as usize;

/// The color the fragment function returns at every pixel of the format cross, as the **codes** of
/// an 8-bit normalized channel: `(40, 90, 160, 200) / 255`.
///
/// Codes rather than a dyadic fraction, because the normalized attachment is the one whose
/// arithmetic has to land on a code for the comparison to be exact. In the 32-bit float attachment
/// the same four `f32` values are what the blend computes with, and the model computes in `f32`
/// beside it.
///
/// The four are far apart and none is half of another, so the four channels are told apart by
/// value, and they are spread across the range so that one blend color clamps some channels and
/// not others — which is what makes the clamp's **placement** readable off one row.
const FORMAT_SOURCE_CODES: [u32; 4] = [40, 90, 160, 200];

/// The destination code of channel `channel` at pixel `pixel` of the format cross.
///
/// Each channel is a different affine permutation of `0..256`, so every code occurs once in every
/// channel and no two channels hold one sequence. Every code occurring is what puts pixels on both
/// sides of each clamp in the same row.
fn format_destination_code(pixel: usize, channel: usize) -> u32 {
    let p = pixel as u32;
    match channel {
        0 => p,
        1 => (37 * p + 11) % 256,
        2 => (91 * p + 200) % 256,
        _ => (173 * p + 77) % 256,
    }
}

/// The blend colors the format cross sets, each with the name its rows take.
///
/// The channels are **integers and halves** so that a factor's product with a source code is itself
/// a code: a `BlendColor` factor of 3 against a source of 90 is the code 270, which a normalized
/// attachment clamps and a float one does not. A fractional blend color would make every product a
/// rounding and the comparison would need a tolerance for arithmetic Metal got exactly right.
const IN_RANGE: [f32; 4] = [0.5, 0.25, 0.75, 1.0];

/// A blend color above the unit range in every channel.
const ABOVE: [f32; 4] = [2.0, 3.0, 1.5, 2.5];

/// A blend color below it in every channel.
const BELOW: [f32; 4] = [-0.5, -1.0, -0.25, -2.0];

/// Which cell of the format cross answers the clamp-placement question, and which is blind to it.
///
/// The question is whether a normalized attachment clamps the blend **color** to `[0, 1]` before it
/// multiplies, or clamps only the **result** after. Reading the table: a row is *decisive* where the
/// two answers differ in some channel of some pixel, and *blind* where they are one texel.
///
/// `tests::the_clamp_cells_are_what_the_arithmetic_says` derives the table rather than trusting it.
#[cfg(test)]
const CLAMP_CELLS: &[(&str, &str, bool, &str)] = &[
    (
        "unorm8",
        "bc-above",
        true,
        "a factor of 2 stores the doubled code, a factor clamped to 1 stores the source's",
    ),
    (
        "unorm8",
        "bc-below",
        false,
        "a negative factor and a factor clamped to 0 both store 0 in a normalized channel",
    ),
    (
        "unorm8",
        "1mbc-above",
        false,
        "one minus an above-range color is negative and one minus a clamped 1 is zero, and both \
         store 0",
    ),
    (
        "unorm8",
        "1mbc-below",
        true,
        "one minus a below-range color is above 1 and one minus a clamped 0 is exactly 1",
    ),
    (
        "float32",
        "bc-above",
        true,
        "the unclamped product is stored as it is computed",
    ),
    (
        "float32",
        "bc-below",
        true,
        "a float attachment stores the negative product, so the clamped factor's 0 differs",
    ),
    (
        "float32",
        "1mbc-above",
        true,
        "a float attachment stores the negative product, so the clamped factor's 0 differs",
    ),
    (
        "float32",
        "1mbc-below",
        true,
        "the unclamped product is stored as it is computed",
    ),
];

/// One row of the format cross: the name its site takes, the blend its pipeline carries (`None` for
/// blending off) and the blend color its pass sets before the draw (`None` for a pass that sets
/// none).
#[derive(Clone, Copy, Debug)]
struct FormatRow {
    name: &'static str,
    blend: Option<Blend>,
    blend_color: Option<[f32; 4]>,
}

/// The twelve blends the format cross sweeps, each with its name, its color state's blend, and the
/// blend color its pass sets.
///
/// `off` is the control: the attachment's own conversion of the source and nothing else, which is
/// what says the format reached the pipeline at all. `one-zero-add` is the same answer through the
/// blend unit. The three `one-one` arithmetic operations are the rows a normalized attachment
/// clamps and a float one does not; `min` and `max` read no factor and are where the destination is
/// compared rather than combined. The five blend-color rows are the clamp-placement cells.
fn format_rows() -> Vec<FormatRow> {
    let direct = Blend::added(F::BlendColor, F::Zero);
    let complement = Blend::added(F::OneMinusBlendColor, F::Zero);
    let row = |name, blend, blend_color| FormatRow {
        name,
        blend,
        blend_color,
    };
    let both = |op| Some(Blend::separate((F::One, F::One, op), (F::One, F::One, op)));
    vec![
        row("off", None, None),
        row("one-zero-add", Some(Blend::added(F::One, F::Zero)), None),
        row("one-one-add", Some(Blend::added(F::One, F::One)), None),
        row("one-one-subtract", both(O::Subtract), None),
        row("one-one-reverse-subtract", both(O::ReverseSubtract), None),
        row("one-one-min", both(O::Min), None),
        row("one-one-max", both(O::Max), None),
        row("bc-in-range", Some(direct), Some(IN_RANGE)),
        row("bc-above", Some(direct), Some(ABOVE)),
        row("bc-below", Some(direct), Some(BELOW)),
        row("1mbc-above", Some(complement), Some(ABOVE)),
        row("1mbc-below", Some(complement), Some(BELOW)),
    ]
}

// ---- One row of a case -------------------------------------------------------------------------

/// One row of a case's cross: the site's name, the color-attachment state its pipeline takes, and
/// the `setBlendColor:green:blue:alpha:` call its pass makes before the draw.
#[derive(Clone, Debug)]
struct Variant {
    name: String,
    color: ColorState,
    blend_color: Option<[f32; 4]>,
}

impl Variant {
    /// The bytes a case's input fingerprint covers for this row: its name, its color state's whole
    /// six-field blend and mask, and the blend color its pass sets.
    fn words(&self) -> Vec<u8> {
        let mut bytes = self.name.as_bytes().to_vec();
        bytes.push(0);
        bytes.extend(self.color.words());
        if let Some(channels) = self.blend_color {
            bytes.push(0xff);
            bytes.extend(BlendColorChannels::new(channels).words());
        }
        bytes
    }
}

impl BlendInteraction {
    /// The attachment format the case renders into.
    fn format(&self) -> PixelFormat {
        match self.subject {
            Subject::MaskBlend => PixelFormat::RGBA32Float,
            Subject::Format(format) => format,
        }
    }

    /// The bytes one texel of the case's format takes.
    fn texel(&self) -> u32 {
        match self.format() {
            PixelFormat::RGBA32Float => 16,
            _ => 4,
        }
    }

    /// The target's side in pixels.
    fn side(&self) -> u32 {
        match self.subject {
            Subject::MaskBlend => MASK_SIDE,
            Subject::Format(_) => FORMAT_SIDE,
        }
    }

    /// The pixels one draw covers.
    fn pixels(&self) -> usize {
        match self.subject {
            Subject::MaskBlend => MASK_PIXELS,
            Subject::Format(_) => FORMAT_PIXELS,
        }
    }

    /// The bytes one target's level 0 takes.
    fn level(&self) -> usize {
        self.pixels() * self.texel() as usize
    }

    /// The target every pass blends into: of the case's own memory and of shared storage, because
    /// an arbitrary destination code reaches a texture only through `replaceRegion:`.
    fn target_spec(&self) -> TargetSpec {
        TargetSpec::plain(self.format(), [self.side(), self.side()]).shared()
    }

    /// Buffer 0's bytes: the case's source color at every pixel.
    ///
    /// One color rather than a sweep, deliberately: the source is then a constant that each factor
    /// multiplies, so a factor that varies across the image is reading the destination and a factor
    /// that does not is reading a constant, and the two cannot be confused.
    fn source_bytes(&self) -> Vec<u8> {
        let source = self.source();
        std::iter::repeat_n(source, self.pixels())
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect()
    }

    /// The `float4` the fragment function returns at every pixel.
    fn source(&self) -> [f32; 4] {
        match self.subject {
            Subject::MaskBlend => MASK_SOURCE,
            Subject::Format(_) => {
                let mut source = [0.0f32; 4];
                for (channel, held) in source.iter_mut().enumerate() {
                    *held = FORMAT_SOURCE_CODES[channel] as f32 / 255.0;
                }
                source
            }
        }
    }

    /// The destination texel of `pixel`, in the attachment's own channel values.
    fn destination(&self, pixel: usize) -> [f32; 4] {
        match self.subject {
            Subject::MaskBlend => mask_destination(pixel),
            Subject::Format(_) => {
                let mut texel = [0.0f32; 4];
                for (channel, held) in texel.iter_mut().enumerate() {
                    *held = format_destination_code(pixel, channel) as f32 / 255.0;
                }
                texel
            }
        }
    }

    /// The bytes of the whole destination level, as the host writes it before the pass.
    ///
    /// A 32-bit float target takes the four `f32`; an 8-bit one takes the four codes, which is why
    /// the format cross states its destination as codes rather than as fractions.
    fn destination_bytes(&self) -> Vec<u8> {
        match self.subject {
            Subject::MaskBlend => (0..self.pixels())
                .flat_map(mask_destination)
                .flat_map(f32::to_le_bytes)
                .collect(),
            Subject::Format(PixelFormat::RGBA32Float) => (0..self.pixels())
                .flat_map(|pixel| self.destination(pixel))
                .flat_map(f32::to_le_bytes)
                .collect(),
            Subject::Format(_) => (0..self.pixels())
                .flat_map(|pixel| {
                    [0usize, 1, 2, 3].map(|channel| format_destination_code(pixel, channel) as u8)
                })
                .collect(),
        }
    }

    /// Every row of the case's cross.
    fn variants(&self) -> Vec<Variant> {
        let plain = ColorState::new(self.format());
        match self.subject {
            Subject::MaskBlend => {
                let mut rows = Vec::new();
                for (mask_name, mask) in mask_columns() {
                    for (blend_name, blend) in blend_rows() {
                        let mut color = plain.masked(mask);
                        if let Some(blend) = blend {
                            color = color.blended(blend);
                        }
                        rows.push(Variant {
                            name: format!("{mask_name}-{blend_name}"),
                            color,
                            blend_color: None,
                        });
                    }
                }
                rows
            }
            Subject::Format(_) => format_rows()
                .into_iter()
                .map(|row| Variant {
                    name: row.name.to_owned(),
                    color: match row.blend {
                        Some(blend) => plain.blended(blend),
                        None => plain,
                    },
                    blend_color: row.blend_color,
                })
                .collect(),
        }
    }

    /// The layout a stored level is compared under: one row a pixel, a channel an element, so a
    /// diff names the pixel and the channel inside it rather than a byte offset.
    fn layout(&self) -> ByteLayout {
        let texel = self.texel();
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [self.side(), self.side(), 1],
            },
            element: match self.format() {
                PixelFormat::RGBA32Float => Element::F32,
                _ => Element::U8,
            },
            row_offset: 0,
            row_elements: 4,
            row_pitch: texel,
            rows: self.pixels() as u32,
            image_pitch: self.level() as u32,
            images: 1,
            // A row is one pixel's four channels and the pixels abut, so no byte lies outside one.
            gaps: Gaps::Ignored,
        }
    }
}

// ---- The blend model ---------------------------------------------------------------------------

/// The value of `factor` in channel `channel` (3 is alpha), given the source, the destination and
/// the blend color.
///
/// Only the factors these two crosses set are here: a factor the cases do not reach has no model
/// and `tests::every_factor_the_cases_set_has_a_model` is what says the two sets are one.
#[cfg(test)]
fn factor_value(
    factor: F,
    channel: usize,
    source: [f32; 4],
    destination: [f32; 4],
    blend_color: [f32; 4],
) -> f32 {
    match factor {
        F::Zero => 0.0,
        F::One => 1.0,
        F::DestinationAlpha => destination[3],
        F::SourceAlphaSaturated => {
            if channel == 3 {
                1.0
            } else {
                source[3].min(1.0 - destination[3])
            }
        }
        F::BlendColor => blend_color[channel],
        F::OneMinusBlendColor => 1.0 - blend_color[channel],
        other => panic!("no model for {}", other.name()),
    }
}

/// What the blend computes in channel `channel` before the write mask and before the attachment's
/// own conversion, for the color state `color` under the blend color `blend_color`.
///
/// `masked_source` is what the fragment's output is taken to be: the model calls it with the real
/// source for Metal's order (the mask applies to the **write**) and with a zero for the order a
/// mask applied **before** the blend would give, which is how
/// `tests::the_named_blends_are_the_ones_blind_to_the_mask_order` derives its own list.
#[cfg(test)]
fn blended(
    color: &ColorState,
    channel: usize,
    source: [f32; 4],
    destination: [f32; 4],
    blend_color: [f32; 4],
) -> f32 {
    let s = source[channel];
    let d = destination[channel];
    let Some(blend) = &color.blend else {
        return s;
    };
    let (source_factor, destination_factor, op) = match channel {
        3 => (blend.source_alpha, blend.destination_alpha, blend.alpha),
        _ => (blend.source_rgb, blend.destination_rgb, blend.rgb),
    };
    let factored = |factor| factor_value(factor, channel, source, destination, blend_color);
    let from_source = s * factored(source_factor);
    let from_destination = d * factored(destination_factor);
    match op {
        O::Add => from_source + from_destination,
        // The negative zero `blend_sweeps` measured is not reachable here: no row of either cross
        // puts both factored terms at zero under `Subtract`.
        O::Subtract => from_source - from_destination,
        O::ReverseSubtract => from_destination - from_source,
        O::Min => s.min(d),
        O::Max => s.max(d),
        other => panic!("no model for {}", other.name()),
    }
}

/// The channel bits in the order a texel's channels lie, which
/// `tests::the_channel_bits_are_in_the_texels_order` holds against the SDK's own table.
#[cfg(test)]
const CHANNEL_BITS: [M; 4] = [M::RED, M::GREEN, M::BLUE, M::ALPHA];

impl BlendInteraction {
    /// Whether the attachment's blend unit clamps the blend color to `[0, 1]` before a factor
    /// multiplies by it.
    ///
    /// **Measured on this M3 Max, macOS 27.0.1, and the finding this cross exists for.** A
    /// normalized attachment does; a 32-bit float one does not. `blend-interaction-format-unorm8`'s
    /// `bc-above` stores the source's own codes `(40, 90, 160, 200)` under a blend color of
    /// `(2, 3, 1.5, 2.5)` — the codes a factor of exactly 1 gives — while
    /// `blend-interaction-format-float32`'s stores `0.3137`, `1.0588`, `0.9412`, `1.9608`, which is
    /// the same source times the same unclamped color. Neither case alone says that: the float one
    /// is `blend_sweeps`' answer and reads as "the color is not clamped", and the normalized one
    /// without it reads as "the result is clamped like any other".
    ///
    /// Whether the clamp sits on the **color** or on the **factor** is not decidable and is not
    /// claimed: for `c` above 1, `1 - c` is negative and clamps to 0, which is `1 - clamp(c)`; for
    /// `c` below 0, `1 - c` is above 1 and clamps to 1, which is `1 - clamp(c)` again. The two
    /// orders are one function over the whole range, which
    /// `tests::clamping_the_color_and_clamping_the_factor_are_one_function` holds.
    #[cfg(test)]
    fn clamps_blend_color(&self) -> bool {
        !matches!(self.format(), PixelFormat::RGBA32Float)
    }

    /// The blend color `variant`'s blend unit reads, which is not always the one
    /// `setBlendColor:green:blue:alpha:` was given ([`BlendInteraction::clamps_blend_color`]).
    #[cfg(test)]
    fn blend_color(&self, variant: &Variant) -> [f32; 4] {
        let channels = variant.blend_color.unwrap_or([0.0; 4]);
        match self.clamps_blend_color() {
            true => channels.map(|c| c.clamp(0.0, 1.0)),
            false => channels,
        }
    }

    /// What the attachment holds in channel `channel` of `pixel` after one draw of `variant` under
    /// the blend color `blend_color`, **before** the format's own conversion: the blend's result
    /// where the write mask keeps the channel and the destination where it drops it.
    ///
    /// The blend color is a parameter rather than the variant's own, so a test can ask the model
    /// the question twice — once with the color the pass set and once with it clamped — which is how
    /// [`CLAMP_CELLS`] is derived rather than trusted.
    #[cfg(test)]
    fn model_channel_with(
        &self,
        variant: &Variant,
        pixel: usize,
        channel: usize,
        blend_color: [f32; 4],
    ) -> f32 {
        let source = self.source();
        let destination = self.destination(pixel);
        if !variant.color.write_mask.contains(CHANNEL_BITS[channel]) {
            return destination[channel];
        }
        blended(&variant.color, channel, source, destination, blend_color)
    }

    /// [`BlendInteraction::model_channel_with`] under the blend color the attachment's blend unit
    /// reads.
    #[cfg(test)]
    fn model_channel(&self, variant: &Variant, pixel: usize, channel: usize) -> f32 {
        self.model_channel_with(variant, pixel, channel, self.blend_color(variant))
    }

    /// The model's whole level for `variant`, in the bytes the site holds.
    ///
    /// `None` where the format's conversion is not derivable, which is the sRGB case's three color
    /// channels: Metal's own decode and encode are what `super::srgb_fetch` measured, and a model
    /// of them written here would be a second spelling of a measurement. Its **alpha** channel is
    /// linear and is modelled, which is what
    /// `tests::metals_srgb_alpha_channel_is_the_normalized_model` holds.
    #[cfg(test)]
    fn model_bytes(&self, variant: &Variant) -> Option<Vec<u8>> {
        self.model_bytes_with(variant, self.blend_color(variant))
    }

    /// [`BlendInteraction::model_bytes`] under a stated blend color.
    #[cfg(test)]
    fn model_bytes_with(&self, variant: &Variant, blend_color: [f32; 4]) -> Option<Vec<u8>> {
        let channel =
            |pixel, channel| self.model_channel_with(variant, pixel, channel, blend_color);
        match self.format() {
            PixelFormat::RGBA32Float => Some(
                (0..self.pixels())
                    .flat_map(|pixel| [0usize, 1, 2, 3].map(|held| channel(pixel, held)))
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            ),
            PixelFormat::RGBA8Unorm => Some(
                (0..self.pixels())
                    .flat_map(|pixel| {
                        [0usize, 1, 2, 3].map(|held| normalized_code(channel(pixel, held)))
                    })
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// `value` stored in an 8-bit normalized channel: clamped to `[0, 1]` and rounded to a code.
///
/// Every number either cross forms is `k / 255` for an integer `k`, so the rounding never decides
/// anything here and the clamp is the whole content of this function — which is the point of
/// choosing the inputs that way.
#[cfg(test)]
fn normalized_code(value: f32) -> u8 {
    let clamped = value.clamp(0.0, 1.0);
    (clamped * 255.0).round() as u8
}

// ---- The case ----------------------------------------------------------------------------------

impl Case for BlendInteraction {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened, and nothing may be. Every number the float cases form is exact in an
        // `f32` channel, and every number the 8-bit cases form is a code, so a texel that differs
        // is a difference in the blend and not in a rounding. The sRGB case's color channels are
        // Metal's own decode and encode, which are **compared** exactly even though they are not
        // modelled: a device whose sRGB blend is not Metal's is divergent, which is the honest
        // verdict rather than a tolerance that would hide the thing this cross exists to find.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(FILLS, &[VERTEX, FILL_FLOAT])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("program", FILLS)?;
        inputs.extend([
            Input::bytes("target", &self.target_spec().words()),
            Input::bytes("buffer:source", &self.source_bytes()),
            Input::bytes("buffer:shape", &shape(self.side(), 1)),
            // The destination every blend reads, which half of every answer depends on.
            Input::bytes("texture:destination", &self.destination_bytes()),
        ]);
        // Each row's own state, keyed by its name, so a case whose cross moved is unrecorded rather
        // than compared against the old cross's texels.
        for variant in self.variants() {
            inputs.push(Input::bytes(
                format!("variant:{}", variant.name),
                &variant.words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = fixture(FILLS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let sources = made(sites, "buffer/source", side.buffer(&self.source_bytes()))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(self.side(), 1)))?;
        let destination = self.destination_bytes();
        let level = self.level();
        let side_pixels = self.side();
        let pitch = side_pixels * self.texel();
        let variants = self.variants();
        let mut work = Vec::new();
        let mut reads = Vec::new();
        for variant in &variants {
            let colors = [Some(variant.color)];
            let pipeline = made(
                sites,
                &format!("pipeline/{}", variant.name),
                side.render_pipeline(
                    library,
                    &RenderPipelineSpec::new(VERTEX, FILL_FLOAT, &colors),
                ),
            )?;
            let target = made(
                sites,
                &format!("target/{}", variant.name),
                side.target(&self.target_spec()),
            )?;
            // The destination codes go in before the pass: a blend's answer depends on them, and
            // the pass loads rather than clears, so this is the only thing that puts them there.
            call(
                sites,
                &format!("replace-region/{}", variant.name),
                side.replace_region(target, [side_pixels, side_pixels], pitch, &destination),
                |_| Value::Absent,
            )?;
            let into = made(
                sites,
                &format!("buffer/texels/{}", variant.name),
                side.zeroed_buffer(level),
            )?;
            let mut steps = vec![
                RenderStep::Pipeline(pipeline),
                RenderStep::FragmentBuffer {
                    buffer: sources,
                    offset: 0,
                    index: 0,
                },
                RenderStep::FragmentBuffer {
                    buffer: extent,
                    offset: 0,
                    index: 1,
                },
            ];
            if let Some(channels) = variant.blend_color {
                steps.push(RenderStep::BlendColor(BlendColorChannels::new(channels)));
            }
            steps.push(RenderStep::triangle());
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(steps),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into,
                offset: 0,
                row_pitch: pitch,
                size: [side_pixels, side_pixels],
                plane: Plane::Whole,
                at: ReadAt::BASE,
            });
            reads.push((variant.name.clone(), into));
        }
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        program(sites, "program", &outcome, &work);

        let mut stored = Node::branch("blended", Vec::new());
        for (name, buffer) in reads {
            let bytes = call(
                sites,
                &format!("read-back/{name}"),
                side.read_buffer(buffer, 0, level),
                |_| Value::Absent,
            )?;
            stored = stored.with(Node::leaf(name, Observed::bytes(bytes, self.layout())));
        }
        sites.push(stored);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! What each cross separates, which cells can see the interaction it is there for, and Metal's
    //! recorded texels held against the model.

    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// The case whose subject is `subject`.
    fn case_of(subject: Subject) -> BlendInteraction {
        all()
            .into_iter()
            .find(|case| case.subject == subject)
            .expect("the family holds the case")
    }

    /// [`CHANNEL_BITS`] is the SDK's four single-channel bits in the order a texel's channels lie,
    /// which is the order [`BlendInteraction::model_channel`] reads it in.
    ///
    /// A reorder upstream fails here rather than masking the wrong channel in every row of the
    /// cross.
    #[test]
    fn the_channel_bits_are_in_the_texels_order() {
        let named: Vec<(M, &str)> = M::KNOWN
            .iter()
            .copied()
            .filter(|(mask, _)| mask.as_raw().count_ones() == 1 && M::ALL.contains(*mask))
            .collect();
        assert_eq!(
            named.iter().map(|(_, name)| *name).collect::<Vec<&str>>(),
            vec![
                "MTLColorWriteMaskRed",
                "MTLColorWriteMaskGreen",
                "MTLColorWriteMaskBlue",
                "MTLColorWriteMaskAlpha",
            ]
        );
        assert_eq!(
            named.iter().map(|(mask, _)| *mask).collect::<Vec<M>>(),
            CHANNEL_BITS.to_vec()
        );
    }

    /// Every factor and every operation any row of either cross sets has a model, so no site is
    /// compared against a `panic!` waiting to happen.
    #[test]
    fn every_factor_and_operation_the_cases_set_has_a_model() {
        let mut factors = BTreeSet::new();
        let mut operations = BTreeSet::new();
        for case in all() {
            for row in case.variants() {
                let Some(blend) = row.color.blend else {
                    continue;
                };
                for factor in [
                    blend.source_rgb,
                    blend.destination_rgb,
                    blend.source_alpha,
                    blend.destination_alpha,
                ] {
                    factors.insert(factor.name());
                    for channel in 0..4 {
                        // The call is the assertion: a factor with no model panics here.
                        let _ = factor_value(factor, channel, [0.5; 4], [0.25; 4], [0.75; 4]);
                    }
                }
                for op in [blend.rgb, blend.alpha] {
                    operations.insert(op.name());
                }
            }
            // And every row's whole level is modelled, except the sRGB case's, whose color
            // channels are Metal's own encode.
            for row in case.variants() {
                let held = case.model_bytes(&row);
                match case.format() {
                    PixelFormat::RGBA8UnormSRGB => assert!(held.is_none()),
                    _ => assert_eq!(
                        held.map(|bytes| bytes.len()),
                        Some(case.level()),
                        "{}: {} is not modelled over its whole level",
                        case.name(),
                        row.name
                    ),
                }
            }
        }
        // The cross reaches five of the twenty factors and all five operations. Stated as a tally
        // rather than as prose, so a row removed from a cross shows up here.
        assert_eq!(
            factors,
            BTreeSet::from([
                "MTLBlendFactorBlendColor",
                "MTLBlendFactorDestinationAlpha",
                "MTLBlendFactorOne",
                "MTLBlendFactorOneMinusBlendColor",
                "MTLBlendFactorSourceAlphaSaturated",
                "MTLBlendFactorZero",
            ])
        );
        assert_eq!(
            operations,
            BTreeSet::from([
                "MTLBlendOperationAdd",
                "MTLBlendOperationMax",
                "MTLBlendOperationMin",
                "MTLBlendOperationReverseSubtract",
                "MTLBlendOperationSubtract",
            ])
        );
    }

    /// `SourceAlphaSaturated` takes both branches of its minimum over the mask cross's destination
    /// sweep, so the `sas-zero-add` row is not a second spelling of a constant factor.
    #[test]
    fn the_saturated_factor_takes_both_of_its_branches() {
        let mut left = 0;
        let mut right = 0;
        for pixel in 0..MASK_PIXELS {
            let da = mask_destination(pixel)[3];
            let saturated = MASK_SOURCE[3].min(1.0 - da);
            if saturated == MASK_SOURCE[3] {
                left += 1;
            }
            if saturated == 1.0 - da {
                right += 1;
            }
        }
        assert!(
            left > 0 && right > 0,
            "the minimum takes one branch only: {left} pixels left, {right} right"
        );
    }

    /// Both of the mask cross's cross-channel rows read the destination's **alpha** from a
    /// **color** channel's factor, which is the whole reason they are in the cross: a mask that
    /// drops alpha must not change what they read.
    ///
    /// Held by construction rather than by reading the names: the row's RGB source factor is one of
    /// the two that read the alpha channel, and its answer in a color channel changes when the
    /// destination's alpha changes and nothing else does.
    #[test]
    fn the_cross_channel_rows_read_the_destinations_alpha_from_a_color_channel() {
        let case = case_of(Subject::MaskBlend);
        let rows = case.variants();
        for name in ["all-da-zero-add", "all-sas-zero-add"] {
            let row = rows
                .iter()
                .find(|row| row.name == name)
                .unwrap_or_else(|| panic!("no {name} row"));
            let source = case.source();
            let mut answers = BTreeSet::new();
            // The three destination alphas straddle the source's own alpha, which is what gives
            // `SourceAlphaSaturated` three values rather than two: its `min` takes the source's
            // alpha until `1 - destinationAlpha` falls below it.
            for alpha in [0.0f32, 0.75, 1.0] {
                let destination = [0.25, 0.5, 0.75, alpha];
                answers.insert(blended(&row.color, 0, source, destination, [0.0; 4]).to_bits());
            }
            assert_eq!(
                answers.len(),
                3,
                "{name}: the red channel's answer does not move with the destination's alpha, so \
                 the cell asks nothing a mask of rgb could break"
            );
        }
    }

    /// The blends of the mask cross that cannot tell a mask applied before the blend from one
    /// applied after are exactly [`BLIND_TO_MASK_ORDER`], derived from the blends themselves.
    ///
    /// This is the family's first result. In a channel the mask drops, Metal stores the destination;
    /// an implementation that applied the mask to the fragment's output instead would blend a zero
    /// there and store the result. Where the two are one texel at every pixel, no cell of the cross
    /// can see the order — and `blend-sweep-write-masks`' one blend, `One`, `One`, `Add`, is one of
    /// them, which is why the per-value sweep could not ask the question.
    #[test]
    fn the_named_blends_are_the_ones_blind_to_the_mask_order() {
        let case = case_of(Subject::MaskBlend);
        let source = case.source();
        let mut blind: Vec<&str> = Vec::new();
        for (name, blend) in blend_rows() {
            let color = match blend {
                Some(blend) => ColorState::new(case.format()).blended(blend),
                None => ColorState::new(case.format()),
            };
            let mut seen = false;
            for pixel in 0..MASK_PIXELS {
                let destination = case.destination(pixel);
                for channel in 0..4 {
                    // Metal's order: the channel the mask drops keeps the destination.
                    let after = destination[channel];
                    // The other order: the mask zeroes the fragment's output and the blend runs.
                    let mut zeroed = source;
                    zeroed[channel] = 0.0;
                    let before = blended(&color, channel, zeroed, destination, [0.0; 4]);
                    if after.to_bits() != before.to_bits() {
                        seen = true;
                    }
                }
            }
            if !seen {
                blind.push(name);
            }
        }
        let named: Vec<&str> = BLIND_TO_MASK_ORDER.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            blind, named,
            "the blends blind to the mask's order are not the ones the table explains"
        );
        // And the cross holds at least one blend that is **not** blind, or it would compare the
        // order nowhere.
        assert!(
            blend_rows().len() > named.len(),
            "every blend of the cross is blind to the mask's order"
        );
        // Every name in the table is a row of the cross, so a row renamed away from its reason
        // fails rather than being skipped.
        let held: BTreeSet<&str> = blend_rows().into_iter().map(|(name, _)| name).collect();
        for (name, _) in BLIND_TO_MASK_ORDER {
            assert!(held.contains(name), "the table names no row {name}");
        }
    }

    /// Each mask column and each blend row of the mask cross changes the answer: no value is swept
    /// that the cross cannot tell from another.
    ///
    /// The guard against a vacuous cross. A mask whose column is identical to another's, or a blend
    /// whose row is, would be a value compared many times and separated never.
    #[test]
    fn the_mask_cross_separates_every_mask_and_every_blend() {
        let case = case_of(Subject::MaskBlend);
        let rows = case.variants();
        let answer = |name: &str| -> Vec<u8> {
            let row = rows
                .iter()
                .find(|row| row.name == name)
                .unwrap_or_else(|| panic!("no {name} row"));
            case.model_bytes(row).expect("the float case is modelled")
        };
        // Along the mask axis: for each blend, the four masks give four answers, except where the
        // blend stores the destination in every channel it is asked about.
        for (blend_name, _) in blend_rows() {
            let by_mask: BTreeSet<Vec<u8>> = mask_columns()
                .into_iter()
                .map(|(mask_name, _)| answer(&format!("{mask_name}-{blend_name}")))
                .collect();
            assert_eq!(
                by_mask.len(),
                mask_columns().len(),
                "{blend_name}: two masks give one answer"
            );
        }
        // Along the blend axis: under a mask that keeps every channel the nine blends are nine
        // answers.
        let by_blend: BTreeMap<Vec<u8>, BTreeSet<&str>> =
            blend_rows()
                .into_iter()
                .fold(BTreeMap::new(), |mut held, (blend_name, _)| {
                    held.entry(answer(&format!("all-{blend_name}")))
                        .or_default()
                        .insert(blend_name);
                    held
                });
        let agreeing: BTreeSet<BTreeSet<&str>> = by_blend
            .into_values()
            .filter(|group| group.len() > 1)
            .collect();
        let named: BTreeSet<BTreeSet<&str>> = COLLAPSED
            .iter()
            .map(|(left, right, _)| BTreeSet::from([*left, *right]))
            .collect();
        assert_eq!(
            agreeing, named,
            "the blends that give one answer under a mask of every channel are not the ones the \
             collapse table explains"
        );
        // And a mask of one color channel reaches the one case the mask sweep's single blend could
        // not: the dropped **alpha** channel beside a color factor that reads it.
        assert_ne!(
            answer("all-da-zero-add"),
            answer("rgb-da-zero-add"),
            "dropping the alpha channel changes nothing in the cell that reads it"
        );
    }

    /// Which cell of the format cross can see where a normalized attachment clamps the blend color
    /// is exactly [`CLAMP_CELLS`], derived from the model rather than trusted.
    ///
    /// The family's third result. The question is whether the blend color is clamped to `[0, 1]`
    /// **as a constant** before it multiplies or **as a result** after, and it has no answer from
    /// one format: on the normalized attachment the two `BlendColor` rows separate the orders and
    /// the two `OneMinusBlendColor` rows do not, and on the float attachment all four do. So the
    /// pair resolves it cell by cell and neither row nor column does on its own.
    #[test]
    fn the_clamp_cells_are_what_the_arithmetic_says() {
        let mut derived: Vec<(&str, &str, bool)> = Vec::new();
        for (label, format) in [
            ("unorm8", PixelFormat::RGBA8Unorm),
            ("float32", PixelFormat::RGBA32Float),
        ] {
            let case = case_of(Subject::Format(format));
            for row in case.variants() {
                let Some(channels) = row.blend_color else {
                    continue;
                };
                if channels == IN_RANGE {
                    continue;
                }
                // The row under the two candidate orders: the blend color as the pass set it, and
                // the blend color clamped to the unit range before a factor reads it. The format's
                // own store clamps either way, so a row where the two answers are one texel cannot
                // say which order the attachment took.
                let held = case
                    .model_bytes_with(&row, channels)
                    .expect("a modelled format");
                let clamped = case
                    .model_bytes_with(&row, channels.map(|c| c.clamp(0.0, 1.0)))
                    .expect("a modelled format");
                derived.push((label, name_of(&row.name), held != clamped));
            }
        }
        let want: Vec<(&str, &str, bool)> = CLAMP_CELLS
            .iter()
            .map(|(format, row, decisive, _)| (*format, *row, *decisive))
            .collect();
        assert_eq!(
            derived, want,
            "the cells that resolve the clamp's placement are not the ones the table explains"
        );
        // The pair's whole claim, as one assertion: each of the two formats is blind to some cell
        // the other resolves, or resolves one the other does not.
        let unorm: BTreeSet<&str> = want
            .iter()
            .filter(|(format, _, decisive)| *format == "unorm8" && *decisive)
            .map(|(_, row, _)| *row)
            .collect();
        let float: BTreeSet<&str> = want
            .iter()
            .filter(|(format, _, decisive)| *format == "float32" && *decisive)
            .map(|(_, row, _)| *row)
            .collect();
        assert_ne!(
            unorm, float,
            "the two formats resolve the same cells, so the pair buys nothing"
        );
        assert!(
            !unorm.is_empty() && !float.is_empty(),
            "one of the two formats resolves no cell at all"
        );
    }

    /// Clamping the blend color to `[0, 1]` and clamping the **factor** that one minus it forms are
    /// one function, so the measured clamp's position is not something this family claims to know.
    ///
    /// Held over every `f32` of a wide sweep rather than argued: for a color channel `c`,
    /// `clamp(1 - c)` and `1 - clamp(c)` agree everywhere, so a cell that separated the two orders
    /// cannot exist for a `BlendColor` or `OneMinusBlendColor` factor.
    #[test]
    fn clamping_the_color_and_clamping_the_factor_are_one_function() {
        let clamp = |v: f32| v.clamp(0.0, 1.0);
        for step in -4096i32..=4096 {
            let c = step as f32 / 1024.0;
            assert_eq!(
                clamp(1.0 - c).to_bits(),
                (1.0 - clamp(c)).to_bits(),
                "the two clamp orders differ at a blend color of {c}"
            );
        }
    }

    /// A row's name as a `&'static str` of [`format_rows`], so the derived table reads like the
    /// written one.
    fn name_of(name: &str) -> &'static str {
        format_rows()
            .into_iter()
            .map(|row| row.name)
            .find(|held| *held == name)
            .unwrap_or_else(|| panic!("no row {name}"))
    }

    /// The format cross clamps on the normalized attachment and does not on the float one, in rows
    /// whose only difference is the attachment's format.
    ///
    /// The second half of the format pair: without this the cross could be twelve blends compared
    /// three times rather than a question about the format.
    #[test]
    fn the_normalized_attachment_clamps_where_the_float_one_does_not() {
        let unorm = case_of(Subject::Format(PixelFormat::RGBA8Unorm));
        let float = case_of(Subject::Format(PixelFormat::RGBA32Float));
        let mut clamped_rows = Vec::new();
        for row in unorm.variants() {
            let mut above = false;
            let mut below = false;
            for pixel in 0..unorm.pixels() {
                for channel in 0..4 {
                    let value = unorm.model_channel(&row, pixel, channel);
                    above |= value > 1.0;
                    below |= value < 0.0;
                }
            }
            if above || below {
                clamped_rows.push((row.name.clone(), above, below));
            }
        }
        // Both directions of the clamp are reached, by rows the table names.
        assert!(
            clamped_rows.iter().any(|(_, above, _)| *above),
            "no row of the format cross exceeds 1, so the upper clamp is untested"
        );
        assert!(
            clamped_rows.iter().any(|(_, _, below)| *below),
            "no row of the format cross falls below 0, so the lower clamp is untested"
        );
        // And the float case stores those same numbers unclamped, so the two cases differ for the
        // format's sake and not for the blend's.
        for (name, _, _) in &clamped_rows {
            let row = float
                .variants()
                .into_iter()
                .find(|held| held.name == *name)
                .expect("the float case holds every row");
            let outside = (0..float.pixels()).any(|pixel| {
                (0..4).any(|channel| {
                    let value = float.model_channel(&row, pixel, channel);
                    value > 1.0 || value < 0.0
                })
            });
            assert!(
                outside,
                "{name}: the float case's model stays inside the unit range, so the two formats \
                 are not being asked one question"
            );
        }
    }

    /// Each case states inputs that cover every row of its cross, names Metal as its source, and
    /// loosens nothing.
    #[test]
    fn every_case_states_its_rows_and_loosens_nothing() {
        let mut names = BTreeSet::new();
        for case in all() {
            assert!(names.insert(case.name()), "{} twice", case.name());
            assert_eq!(case.metal_source(), MetalSource::Run);
            assert!(
                case.allowances().is_empty(),
                "{}: this family loosens nothing",
                case.name()
            );
            case.shaders().unwrap();
            let inputs = case.inputs().unwrap();
            let stated: BTreeSet<&str> = inputs.iter().map(|input| input.name.as_str()).collect();
            assert_eq!(
                stated.len(),
                inputs.len(),
                "{}: an input twice",
                case.name()
            );
            let rows = case.variants();
            for row in &rows {
                assert!(
                    stated.contains(format!("variant:{}", row.name).as_str()),
                    "{}: {} is in no input",
                    case.name(),
                    row.name
                );
            }
            // Two rows that differ only in their blend color still fingerprint apart.
            let words: BTreeSet<Vec<u8>> = rows.iter().map(Variant::words).collect();
            assert_eq!(
                words.len(),
                rows.len(),
                "{}: two rows fingerprint alike",
                case.name()
            );
            assert_eq!(case.layout().extent(), case.level());
        }
    }

    /// The crosses are the size the module docs quote, each count derived from its own axes.
    #[test]
    fn the_crosses_are_the_size_their_prose_claims() {
        let want: Vec<(&str, usize)> = vec![
            (
                "blend-interaction-mask-blend",
                mask_columns().len() * blend_rows().len(),
            ),
            ("blend-interaction-format-float32", format_rows().len()),
            ("blend-interaction-format-unorm8", format_rows().len()),
            ("blend-interaction-format-srgb8", format_rows().len()),
        ];
        let held: Vec<(&str, usize)> = all()
            .iter()
            .map(|case| (case.name(), case.variants().len()))
            .collect();
        assert_eq!(held, want);
        assert_eq!(mask_columns().len(), 4);
        assert_eq!(blend_rows().len(), 9);
        assert_eq!(format_rows().len(), 12);
        let states: usize = held.iter().map(|(_, rows)| rows).sum();
        assert_eq!(states, 72, "the family holds {states} color states");
    }

    /// Metal's recorded texels are the model, at every pixel of every row of the modelled cases.
    ///
    /// The model's answer at a site depends on the source buffer, the destination the host wrote,
    /// the blend color the pass set, the six fields of the blend and the write mask, so a setup
    /// that failed to deliver any of them fails here rather than passing on bytes that happen to
    /// match.
    #[test]
    fn metals_recorded_texels_are_the_model_at_every_pixel() {
        let mut checked = 0usize;
        let mut unrecorded = Vec::new();
        for case in all() {
            let Some(blended) = recorded(&case) else {
                unrecorded.push(case.name());
                continue;
            };
            for row in case.variants() {
                let Some(want) = case.model_bytes(&row) else {
                    continue;
                };
                let bytes = site_bytes(&case, &blended, &row.name);
                assert_eq!(bytes.len(), case.level(), "{}: {}", case.name(), row.name);
                if bytes == want {
                    checked += 1;
                    continue;
                }
                // Name the first pixel and channel that moved, with both numbers, rather than
                // printing a kilobyte of bytes.
                let held = case.layout();
                for pixel in 0..case.pixels() {
                    for channel in 0..4usize {
                        let at = held.at(0, pixel as u32, channel as u32);
                        let size = held.element.size() as usize;
                        assert_eq!(
                            bytes[at..at + size],
                            want[at..at + size],
                            "{}: {} pixel {pixel} channel {channel}: Metal stored {:?}, the model \
                             says {:?} (source {:?}, destination {:?})",
                            case.name(),
                            row.name,
                            &bytes[at..at + size],
                            &want[at..at + size],
                            case.source(),
                            case.destination(pixel)
                        );
                    }
                }
                panic!(
                    "{}: {} differs from the model in a byte no channel covers",
                    case.name(),
                    row.name
                );
            }
        }
        assert!(
            checked > 0,
            "no recorded answer of this family to hold the model against (unrecorded: {})",
            unrecorded.join(", ")
        );
    }

    /// The sRGB case's **alpha** channel is the plain normalized case's model, at every pixel of
    /// every row.
    ///
    /// sRGB leaves the alpha channel linear, so the one half of that case a model can state is
    /// stated and held; its three color channels are Metal's own decode and encode, which
    /// `super::super::srgb_fetch` measured to differ from the textbook round trip, and are compared
    /// against Metal rather than against a second spelling of a measurement.
    #[test]
    fn metals_srgb_alpha_channel_is_the_normalized_model() {
        let srgb = case_of(Subject::Format(PixelFormat::RGBA8UnormSRGB));
        let Some(blended) = recorded(&srgb) else {
            return;
        };
        let unorm = case_of(Subject::Format(PixelFormat::RGBA8Unorm));
        let held = srgb.layout();
        let mut checked = 0usize;
        for row in srgb.variants() {
            let bytes = site_bytes(&srgb, &blended, &row.name);
            for pixel in 0..srgb.pixels() {
                let at = held.at(0, pixel as u32, 3);
                let modelled = normalized_code(unorm.model_channel(&row, pixel, 3));
                assert_eq!(
                    bytes[at],
                    modelled,
                    "{}: {} pixel {pixel} alpha: Metal stored {}, the linear model says {modelled}",
                    srgb.name(),
                    row.name,
                    bytes[at]
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "the sRGB case holds no alpha channel to compare"
        );
        // And the sRGB case's **color** channels are not the plain unorm case's, or the format
        // would be making no difference and the case would be a copy of its neighbour.
        let mut differs = false;
        for row in srgb.variants() {
            let bytes = site_bytes(&srgb, &blended, &row.name);
            for pixel in 0..srgb.pixels() {
                for channel in 0..3u32 {
                    let at = held.at(0, pixel as u32, channel);
                    let plain = normalized_code(unorm.model_channel(&row, pixel, channel as usize));
                    differs |= bytes[at] != plain;
                }
            }
        }
        assert!(
            differs,
            "the sRGB attachment stored the plain normalized codes in every channel of every row"
        );
    }

    /// The `blended` branch of `case`'s newest recorded Metal answer, or `None` where the case is
    /// unrecorded.
    fn recorded(case: &BlendInteraction) -> Option<Node> {
        use metal2vulkan_runtime::MetalLevel;
        let recording = crate::recording::read(case.name()).ok()?;
        let answer = recording
            .answer(MetalLevel::NEWEST)
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let taken = answer
            .deterministic()
            .unwrap_or_else(|| panic!("{}: more than one outcome", case.name()));
        Some(
            taken
                .root
                .child("blended")
                .unwrap_or_else(|| panic!("{}: no blended site", case.name()))
                .clone(),
        )
    }

    /// The bytes `blended`'s `name` site holds.
    fn site_bytes(case: &BlendInteraction, blended: &Node, name: &str) -> Vec<u8> {
        let site = blended
            .child(name)
            .unwrap_or_else(|| panic!("{}: no {name} site", case.name()));
        let Value::Bytes(bytes) = &site.observed.value else {
            panic!(
                "{}: {name} holds {}",
                case.name(),
                site.observed.value.kind()
            );
        };
        bytes.clone()
    }
}
