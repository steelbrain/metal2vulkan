//! Every value of Metal's blend state, swept one field at a time against native Metal.
//!
//! `MTLBlendFactor` has twenty enumerators and the suite's other blend families between them reach
//! eleven. A once-each suite passes an implementation that gets nineteen of them right and
//! `MTLBlendFactorSourceAlphaSaturated` wrong, so this family sweeps the **field** rather than
//! sampling it: every settable factor in each of the four factor slots, every operation against six
//! factor pairs and against every other operation, all sixteen write masks with the blend on and
//! off, blending enabled against disabled, and the blend colour at its boundaries including values
//! outside `[0, 1]`.
//!
//! The value lists are **not written here**. They are read from `BlendFactor::ALL`,
//! `BlendOperation::ALL` and `ColorWriteMask::KNOWN`, which
//! `scripts/regen-metal-enums/regen_metal_enums.py` extracts from the SDK headers, and each site's
//! name is the SDK enumerator's own name in kebab case. So a value the SDK adds appears as an
//! unclassified factor in `tests::every_blend_factor_of_the_sdk_table_is_classified` rather than
//! as silence, and a value renamed upstream renames its site loudly.
//!
//! # What is not swept, and why
//!
//! `MTLBlendFactorUnspecialized`, `MTLBlendOperationUnspecialized` and
//! `MTLColorWriteMaskUnspecialized` are Metal 4's pipeline-specialization placeholders, not a
//! factor, an operation or a channel. `MTLRenderPipelineColorAttachmentDescriptor` is the Metal 3
//! descriptor and takes none of them; our own translator refuses each as
//! `PlaceholderEnumValue` (`metal2vulkan::metal::state`). No case here sets one, and **no case
//! anywhere compares that refusal against Metal's** — stated as a gap rather than guessed at.
//!
//! # The inputs, and why these numbers separate the factors
//!
//! Every factor is a number the blend multiplies a colour by, so two factors are only told apart by
//! inputs that give them different products. The inputs here are chosen so that **every factor's
//! value is a different number**:
//!
//! | | red | green | blue | alpha |
//! |---|---|---|---|---|
//! | the source the fragment returns | 5/64 | 9/64 | 13/64 | 17/64 |
//! | the blend colour | 21/64 | 25/64 | 29/64 | 33/64 |
//! | the second output (`Source1`) | 37/64 | 41/64 | 45/64 | 49/64 |
//!
//! Their complements are 59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19 and 15 sixty-fourths. Those
//! twelve numerators and the twelve above are twenty-four **distinct** odd numerators, so no
//! constant factor equals another and none equals `Zero` or `One`. That is the whole reason the
//! sweep is not twenty comparisons of the same number.
//!
//! The **destination varies per pixel** — 64 pixels, each channel its own sequence over the
//! sixty-fourths and the alpha channel `p/64` — which separates the four factors that read the
//! destination from every constant one, and does one more job that no single destination can:
//! `SourceAlphaSaturated` is `min(sourceAlpha, 1 - destinationAlpha)`, which is *always* equal to
//! one of the two factors beside it. With the alpha destination sweeping across `1 - 17/64` the
//! minimum takes its left branch at the pixels below 48 and its right branch above, so over the
//! image it equals neither `SourceAlpha` nor `OneMinusDestinationAlpha` and an implementation that
//! lowered it to either is caught. `tests::the_saturated_factor_takes_both_of_its_branches` holds
//! that by construction.
//!
//! Each case isolates one slot by making the other three terms carry a constant: the source-RGB
//! sweep puts `Zero` in the destination-RGB slot, so the stored colour is exactly the factor times
//! the source and the alpha channel stays at the source's alpha as a control that the colour slot
//! did not reach it. The destination sweeps do the mirror, so their colour channels read the
//! destination back and say that `replaceRegion:` reached the blend at all.
//!
//! # The format, and why nothing here needs a tolerance
//!
//! `RGBA32Float`. Blending happens in the target's format, so the format decides whether a product
//! rounds: in an 8-bit normalized target `(5/64) * (21/64)` is not a code and the comparison would
//! need a tolerance for arithmetic Metal got exactly right. Every number above is `k/64` with
//! `k` at most 192, every product is `k*m/4096` and every sum of two is under 4 — all exact in
//! `f32`, so Metal's answer and ours are the same bits or one of them is wrong. **No site of this
//! family is loosened.**
//!
//! A float target earns its place twice more. It does not clamp, so `Subtract` stores the negative
//! result it computes rather than a floor, and a blend colour outside `[0, 1]` is observable at all
//! — in a normalized target both questions would have the same answer whatever the implementation
//! did.
//!
//! # The controls against a vacuous sweep
//!
//! A sweep that cannot distinguish its values compares nothing many times, so three things are
//! checked rather than assumed:
//!
//! - `tests::metals_recorded_texels_are_the_blend_model_at_every_pixel` holds **every** site of
//!   **every** case, pixel by pixel and channel by channel, against one written statement of
//!   Metal's blend equation. A destination that never reached the target, a source buffer that was
//!   never bound, a blend colour that went unset — each fails it, because the model's answer at
//!   each site depends on all of them.
//! - `tests::the_sweeps_separate_every_value_except_the_named_collapses` groups each case's
//!   sites by the answer the model gives and holds the groups against `COLLAPSED`, which names
//!   every pair that must agree and why. A pair that collapses for a reason not listed there fails,
//!   which is how a badly chosen input reads as a failure rather than as twenty passes.
//! - `blend-sweep-enabled` is the control that the blend state is read at all: blending off and
//!   blending on with `One`, `Zero`, `Add` must store the identical image, and blending on with
//!   `One`, `One` must not.
//!
//! # Where values agree by Metal's own definition
//!
//! `COLLAPSED` is that list, and it is short. In the **alpha** slots a factor named for a colour
//! reads the colour's alpha channel, so seven pairs are one value there and nothing can separate
//! them: `SourceColor`/`SourceAlpha`, `DestinationColor`/`DestinationAlpha`,
//! `BlendColor`/`BlendAlpha`, each with its `OneMinus` partner, and `SourceAlphaSaturated`, whose
//! alpha component is 1, with `One`. `Min` and `Max` read no factor, so the six factor pairs give
//! one answer under each. And `Add` and `Subtract` agree where the destination factor is `Zero`,
//! because there is nothing to subtract.
//!
//! Each of those is a fact about Metal that the recording **measures** rather than one the suite
//! asserts: the model says they agree, and the recorded texels say the model is Metal's.
//!
//! # What the sweep measured about Metal that nothing here knew before
//!
//! The eleven cases hold **179 blend states**, each over 64 pixels and four channels, so Metal's
//! answer is held against the written blend equation at **45,824** compared channels; the counts
//! are the subject of `tests::the_sweep_is_the_size_its_prose_claims` rather than of this
//! sentence. **45,820 of them agree** on this M3 Max, macOS 27.0.1. The four that do not are one
//! fact, and it is the sort a sweep finds and a once-each suite cannot:
//!
//! **`MTLBlendOperationSubtract` stores negative zero where both factored terms are zero.** The
//! four leaves are `dc-1mdc-subtract` at the one pixel in each channel where the destination is
//! zero, which makes `destinationColor * source` and `(1 - destinationColor) * destination` both
//! zero. IEEE subtraction gives `+0` for those operands, and so does every neighbour of the
//! measurement: `Add` and `ReverseSubtract` over the same two terms store `+0`, and `Subtract`
//! itself stores `+0` at the leaves of the same site where its two terms are equal and **non**-zero.
//! So the condition is exactly "both terms zero" and nothing wider, which is why
//! `tests::metals_subtract_stores_a_negative_zero_where_both_terms_are_zero` holds the boundary
//! as well as the fact. What the blend unit does to arrive there is not something this suite can
//! see, and it does not guess.
//!
//! It is a bit a Metal program can read -- a sign test, or a division -- so the model states it and
//! no site is loosened for it. A device that stores `+0` there is divergent from Metal, and that is
//! the honest verdict rather than a rounding to wave through.
//!
//! The sweep also put a fact in the recording's **second** oracle. Metal's own API validation layer
//! stops a pass that sets the blend colour and then sets it again, with `previous setBlendColorRed
//! was unused`, while plain Metal takes the second value and draws. Both halves are recorded:
//! `blend-sweep-blend-color-order` carries the layer's stop, and its `once` and `twice` rows store
//! the identical image because the last call is the one the draw reads. Nothing of ours is refused
//! by it, and nothing here is excused by it.
//!
//! Splitting that case off bought a second fact, which is the whole reason it is split: with the
//! redundant call in its own case, `blend-sweep-blend-color`'s validated leg **ran to the end**, so
//! Metal's validation layer accepts a blend colour of `(-0.5, -1, -1.5, -2)` and one of
//! `(1.5, 2, 2.5, 3)` without a word. Out of range is not out of bounds, and a recording now says
//! so instead of a stop three rows earlier hiding it.
//!
//! # Nothing here can fault a GPU
//!
//! Single-sampled 8 by 8 targets of the case's own memory, one full-screen triangle a pass, no
//! loop, no indexing outside a bound buffer, and every pass's attachment written and read back by
//! the same command buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, BlendColorChannels, ColorState, PassColor, Plane, RenderPass, RenderPipelineSpec,
    RenderStep, RenderWork, TargetSpec,
};
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{BlendFactor as F, BlendOperation as O, ColorWriteMask, PixelFormat};

use super::extended_range_targets::{
    fixture, fixture_inputs, not_read, program, shader_refs, shape, FILLS, FILL_FLOAT, VERTEX,
};

/// The owned fixture whose fragment function returns a second output at `[[color(0), index(1)]]`,
/// which is what a `Source1` factor blends. Built without fast math.
const DUAL: &str = "xrb_dual";
const DUAL_VERTEX: &str = "xrb_dual_vertex";
const DUAL_FRAGMENT: &str = "xrb_dual_float";

/// The target's extent: 64 pixels, each with its own destination texel.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;

/// The pixels one draw covers.
const PIXELS: usize = (WIDTH * HEIGHT) as usize;

/// The bytes one `RGBA32Float` texel takes.
const TEXEL: u32 = 16;

/// The bytes one target's level 0 takes.
const LEVEL: usize = PIXELS * TEXEL as usize;

/// The format every case blends into: exact for every product this family forms, unclamped, and
/// signed, so a subtraction's negative result and an out-of-range blend colour are both observable.
const FORMAT: PixelFormat = PixelFormat::RGBA32Float;

/// The unit every input is a multiple of.
const Q: f32 = 1.0 / 64.0;

/// The colour the fragment function returns at every pixel: `(5, 9, 13, 17) / 64`.
const SOURCE: [f32; 4] = [5.0 * Q, 9.0 * Q, 13.0 * Q, 17.0 * Q];

/// `setBlendColor:green:blue:alpha:`'s four channels for the cases that read it:
/// `(21, 25, 29, 33) / 64`.
const BLEND_COLOR: [f32; 4] = [21.0 * Q, 25.0 * Q, 29.0 * Q, 33.0 * Q];

/// The second output the dual-source case's fragment function returns: `(37, 41, 45, 49) / 64`.
const SECOND: [f32; 4] = [37.0 * Q, 41.0 * Q, 45.0 * Q, 49.0 * Q];

/// Metal's blend colour before anything sets it, which the `unset` site of `blend-sweep-blend-color`
/// is what measures: the model states it and the recording holds the model.
#[cfg(test)]
const UNSET_BLEND_COLOR: [f32; 4] = [0.0; 4];

/// The blend colour above the unit range: `(96, 128, 160, 192) / 64`.
const ABOVE: [f32; 4] = [96.0 * Q, 128.0 * Q, 160.0 * Q, 192.0 * Q];

/// The blend colour below it, which is `1 - ABOVE` channel for channel: `(-32, -64, -96, -128) / 64`.
///
/// Chosen so that `OneMinusBlendColor` of one is `BlendColor` of the other, which is why
/// `above-complement` and `below` are a `COLLAPSED` pair: they agree only where Metal computes
/// `1 - c` exactly and clamps neither side.
const BELOW: [f32; 4] = [-32.0 * Q, -64.0 * Q, -96.0 * Q, -128.0 * Q];

// ---- The destination ---------------------------------------------------------------------------

/// The destination texel of pixel `pixel`, as the host writes it before the pass.
///
/// Each colour channel steps by an odd multiplier, so each takes all 64 values of the sweep over the
/// image and no two channels hold one sequence; the alpha channel is `p / 64`, which is what makes
/// it cross `1 - 17/64` and give `SourceAlphaSaturated` both of its branches.
fn destination(pixel: usize) -> [f32; 4] {
    let p = pixel as u32;
    [
        f32::from(((13 * p + 1) % 64) as u16) * Q,
        f32::from(((29 * p + 2) % 64) as u16) * Q,
        f32::from(((37 * p + 3) % 64) as u16) * Q,
        f32::from(p as u16) * Q,
    ]
}

/// The bytes of the whole destination level, one `float4` a pixel.
fn destination_bytes() -> Vec<u8> {
    (0..PIXELS)
        .flat_map(destination)
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// Buffer 0's bytes: [`SOURCE`] at every pixel.
///
/// One colour rather than a sweep, deliberately: the source is then a constant that each factor
/// multiplies, so a factor that varies across the image is reading the destination and a factor that
/// does not is reading a constant, and the two cannot be confused.
fn source_bytes() -> Vec<u8> {
    std::iter::repeat_n(SOURCE, PIXELS)
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// Buffer 2's bytes for the dual-source case: [`SECOND`] at every pixel.
fn second_bytes() -> Vec<u8> {
    std::iter::repeat_n(SECOND, PIXELS)
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect()
}

// ---- The SDK's own value lists -----------------------------------------------------------------

/// Why a `MTLBlendFactor` is or is not in a slot sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// A factor the slot sweeps set, which needs nothing of the fragment function.
    Settable,
    /// A factor that reads the fragment's **second** output, so only the dual-source case can set
    /// it and only on a device with `dualSrcBlend`.
    SecondOutput,
    /// Metal 4's pipeline-specialization placeholder, which no Metal 3 descriptor takes.
    Placeholder,
    /// A value the SDK table holds that this family does not classify, which is a test failure
    /// rather than a silent omission.
    Unclassified,
}

/// Which family `factor` is in. The wildcard is what
/// `tests::every_blend_factor_of_the_sdk_table_is_classified` turns into a failure, since
/// `BlendFactor` is `#[non_exhaustive]` and no match outside its own crate can be exhaustive.
fn family(factor: F) -> Family {
    match factor {
        F::Zero
        | F::One
        | F::SourceColor
        | F::OneMinusSourceColor
        | F::SourceAlpha
        | F::OneMinusSourceAlpha
        | F::DestinationColor
        | F::OneMinusDestinationColor
        | F::DestinationAlpha
        | F::OneMinusDestinationAlpha
        | F::SourceAlphaSaturated
        | F::BlendColor
        | F::OneMinusBlendColor
        | F::BlendAlpha
        | F::OneMinusBlendAlpha => Family::Settable,
        F::Source1Color | F::OneMinusSource1Color | F::Source1Alpha | F::OneMinusSource1Alpha => {
            Family::SecondOutput
        }
        F::Unspecialized => Family::Placeholder,
        _ => Family::Unclassified,
    }
}

/// Every factor of the SDK table in `family`, in the table's own declaration order.
fn factors(wanted: Family) -> Vec<F> {
    F::ALL
        .iter()
        .copied()
        .filter(|factor| family(*factor) == wanted)
        .collect()
}

/// Every `MTLBlendOperation` of the SDK table that a Metal 3 descriptor takes: all of them but the
/// specialization placeholder, which `metal2vulkan::metal::state::blend_op` refuses.
fn operations() -> Vec<O> {
    O::ALL
        .iter()
        .copied()
        .filter(|op| *op != O::Unspecialized)
        .collect()
}

/// The single-channel bits of `MTLColorWriteMask`, from the generated table: every named value that
/// is one bit of `MTLColorWriteMaskAll`.
///
/// Reading them rather than listing them is what keeps `MTLColorWriteMaskUnspecialized` — one bit,
/// and outside the field — out of the sweep without naming it here.
fn channel_bits() -> Vec<(ColorWriteMask, &'static str)> {
    ColorWriteMask::KNOWN
        .iter()
        .copied()
        .filter(|(mask, _)| mask.as_raw().count_ones() == 1 && ColorWriteMask::ALL.contains(*mask))
        .collect()
}

/// `name` with the SDK's prefix dropped and its camel case written as a site name:
/// `MTLBlendFactorOneMinusSource1Color` becomes `one-minus-source1-color`.
fn kebab(name: &str, prefix: &str) -> String {
    let tail = name.strip_prefix(prefix).unwrap_or(name);
    let mut out = String::new();
    for letter in tail.chars() {
        if letter.is_ascii_uppercase() && !out.is_empty() {
            out.push('-');
        }
        out.extend(letter.to_lowercase());
    }
    out
}

/// A factor's site name.
fn factor_name(factor: F) -> String {
    kebab(factor.name(), "MTLBlendFactor")
}

/// An operation's site name.
fn operation_name(op: O) -> String {
    kebab(op.name(), "MTLBlendOperation")
}

// ---- The blend model ---------------------------------------------------------------------------

/// The value of `factor` in channel `channel` (3 is alpha), given the source, destination, blend
/// colour and second output.
///
/// This is the whole statement of what a Metal blend factor is, and it is what the recorded texels
/// are held against. The alpha channel of a factor named for a colour is that colour's alpha, which
/// is why the alpha slots collapse seven pairs; `SourceAlphaSaturated`'s alpha component is 1.
#[cfg(test)]
fn factor_value(
    factor: F,
    channel: usize,
    source: [f32; 4],
    destination: [f32; 4],
    blend_color: [f32; 4],
    second: [f32; 4],
) -> f32 {
    match factor {
        F::Zero => 0.0,
        F::One => 1.0,
        F::SourceColor => source[channel],
        F::OneMinusSourceColor => 1.0 - source[channel],
        F::SourceAlpha => source[3],
        F::OneMinusSourceAlpha => 1.0 - source[3],
        F::DestinationColor => destination[channel],
        F::OneMinusDestinationColor => 1.0 - destination[channel],
        F::DestinationAlpha => destination[3],
        F::OneMinusDestinationAlpha => 1.0 - destination[3],
        F::SourceAlphaSaturated => {
            if channel == 3 {
                1.0
            } else {
                source[3].min(1.0 - destination[3])
            }
        }
        F::BlendColor => blend_color[channel],
        F::OneMinusBlendColor => 1.0 - blend_color[channel],
        F::BlendAlpha => blend_color[3],
        F::OneMinusBlendAlpha => 1.0 - blend_color[3],
        F::Source1Color => second[channel],
        F::OneMinusSource1Color => 1.0 - second[channel],
        F::Source1Alpha => second[3],
        F::OneMinusSource1Alpha => 1.0 - second[3],
        other => panic!("no model for {}", other.name()),
    }
}

/// What the attachment holds after one draw of `variant` over the destination of `pixel`.
///
/// Blending off stores the source; blending on applies the operation to the two factored terms,
/// with `Min` and `Max` reading no factor at all. The write mask then decides, channel by channel,
/// whether the result or the destination stays.
#[cfg(test)]
fn model(variant: &Variant, pixel: usize) -> [f32; 4] {
    let source = SOURCE;
    let destination = destination(pixel);
    let blend_color = variant
        .blend_colors
        .last()
        .copied()
        .unwrap_or(UNSET_BLEND_COLOR);
    let mut texel = [0.0f32; 4];
    for (channel, held) in texel.iter_mut().enumerate() {
        let s = source[channel];
        let d = destination[channel];
        let written = match &variant.color.blend {
            None => s,
            Some(blend) => {
                let (source_factor, destination_factor, op) = if channel == 3 {
                    (blend.source_alpha, blend.destination_alpha, blend.alpha)
                } else {
                    (blend.source_rgb, blend.destination_rgb, blend.rgb)
                };
                let factored = |factor| {
                    factor_value(factor, channel, source, destination, blend_color, SECOND)
                };
                let from_source = s * factored(source_factor);
                let from_destination = d * factored(destination_factor);
                match op {
                    O::Add => from_source + from_destination,
                    // Measured, not derived: where **both** factored terms are zero, Metal's
                    // `Subtract` stores negative zero. Its sibling operations over the same two
                    // operands store positive zero, and so does `Subtract` where the two terms are
                    // equal and non-zero -- which IEEE subtraction would give in every one of those
                    // cases, so this is the one place the blend unit is not it. The four leaves of
                    // this family that show it, and the many that pin the boundary around them, are
                    // `tests::metals_subtract_stores_a_negative_zero_where_both_terms_are_zero`'s
                    // subject; the mechanism behind it is not something the suite can see, and the
                    // suite does not guess at one.
                    O::Subtract if from_source == 0.0 && from_destination == 0.0 => -0.0,
                    O::Subtract => from_source - from_destination,
                    O::ReverseSubtract => from_destination - from_source,
                    O::Min => s.min(d),
                    O::Max => s.max(d),
                    other => panic!("no model for {}", other.name()),
                }
            }
        };
        let bit = channel_bits()[channel].0;
        *held = if variant.color.write_mask.contains(bit) {
            written
        } else {
            d
        };
    }
    texel
}

/// The model's whole level for `variant`, in the bytes the site holds.
#[cfg(test)]
fn model_bytes(variant: &Variant) -> Vec<u8> {
    (0..PIXELS)
        .flat_map(|pixel| model(variant, pixel))
        .flat_map(f32::to_le_bytes)
        .collect()
}

// ---- The sweeps --------------------------------------------------------------------------------

/// Which of the four factor slots a sweep moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    SourceRgb,
    DestinationRgb,
    SourceAlpha,
    DestinationAlpha,
}

impl Slot {
    /// The six-field blend that puts `factor` in this slot and a constant in the other three, so the
    /// stored texel is the factor's own contribution and the channels the slot does not reach are a
    /// control.
    fn blend(self, factor: F) -> Blend {
        let add = O::Add;
        match self {
            // Colour: the factor times the source. Alpha: the source's alpha, unmoved.
            Slot::SourceRgb => Blend::separate((factor, F::Zero, add), (F::One, F::Zero, add)),
            // Colour: the factor times the destination, which also says the destination arrived.
            Slot::DestinationRgb => Blend::separate((F::Zero, factor, add), (F::Zero, F::One, add)),
            // Alpha: the factor times the source's alpha. Colour: the source, unmoved.
            Slot::SourceAlpha => Blend::separate((F::One, F::Zero, add), (factor, F::Zero, add)),
            // Alpha: the factor times the destination's alpha. Colour: the destination, unmoved.
            Slot::DestinationAlpha => {
                Blend::separate((F::Zero, F::One, add), (F::Zero, factor, add))
            }
        }
    }

    /// The slot's own name, for a site of the dual-source case.
    fn name(self) -> &'static str {
        match self {
            Slot::SourceRgb => "source-rgb",
            Slot::DestinationRgb => "destination-rgb",
            Slot::SourceAlpha => "source-alpha",
            Slot::DestinationAlpha => "destination-alpha",
        }
    }
}

/// The six factor pairs the operations are swept against.
///
/// Each is a pair whose two terms are different numbers, so `Add`, `Subtract` and `ReverseSubtract`
/// give three answers under it -- except `one-zero`, whose destination term is nothing to subtract,
/// which is a `COLLAPSED` row rather than a gap. Between them they cover a constant pair, the
/// consumer's alpha pair, two destination-reading factors, the blend colour and the saturated
/// factor, so "the operation ignores the factors" is a claim this case can make about six different
/// factor pairs rather than one.
fn operation_pairs() -> Vec<(&'static str, F, F)> {
    vec![
        ("one-zero", F::One, F::Zero),
        ("one-one", F::One, F::One),
        ("sa-1msa", F::SourceAlpha, F::OneMinusSourceAlpha),
        ("dc-1mdc", F::DestinationColor, F::OneMinusDestinationColor),
        ("bc-1mbc", F::BlendColor, F::OneMinusBlendColor),
        ("sas-da", F::SourceAlphaSaturated, F::DestinationAlpha),
    ]
}

/// What one case sweeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sweep {
    /// Every settable factor in one slot.
    Slot(Slot),
    /// Every operation against each of [`operation_pairs`].
    Operations,
    /// Every pair of a colour operation and an alpha operation, which is what says the two fields
    /// are read separately.
    OperationPairs,
    /// All sixteen write masks, with the blend on and with it off.
    WriteMasks,
    /// Blending off, the identity blend, and a blend that moves, with otherwise identical state.
    Enabled,
    /// The blend colour at its boundaries, outside the unit range, and unset.
    BlendColour,
    /// The same blend colour reached by one call and by two, and the two in the other order, which
    /// is what says the last `setBlendColor:green:blue:alpha:` of a pass is the one the draw reads.
    ///
    /// Its own case rather than three more rows of [`Sweep::BlendColour`], because a pass that sets
    /// the blend colour and then sets it again is something **Metal's own API validation layer
    /// stops on** (`previous setBlendColorRed was unused`). Recorded together, that stop cost the
    /// nine rows beside it their validated leg -- including the layer's opinion on a blend colour
    /// outside `[0, 1]`, which is a fact that case is there to collect. Split, each case keeps the
    /// second oracle's answer to its own question.
    BlendColourOrder,
    /// The four `Source1` factors in each slot, through a fragment function with a second output.
    SecondSource,
}

/// One row of a case's sweep: the site's name, the colour-attachment state its pipeline takes, and
/// the `setBlendColor:green:blue:alpha:` calls its pass makes before the draw.
#[derive(Clone, Debug)]
struct Variant {
    name: String,
    color: ColorState,
    blend_colors: Vec<[f32; 4]>,
}

impl Variant {
    /// A row whose pass sets the family's blend colour, which is what every case but the blend
    /// colour's own does: a factor that reads it then reads a stated value rather than whatever the
    /// encoder began with.
    fn new(name: String, color: ColorState) -> Self {
        Self {
            name,
            color,
            blend_colors: vec![BLEND_COLOR],
        }
    }

    /// The bytes a case's input fingerprint covers for this row: its name, its colour state's whole
    /// six-field blend and mask, and every blend colour its pass sets, in order.
    fn words(&self) -> Vec<u8> {
        let mut bytes = self.name.as_bytes().to_vec();
        bytes.push(0);
        bytes.extend(self.color.words());
        for channels in &self.blend_colors {
            bytes.push(0xff);
            bytes.extend(BlendColorChannels::new(*channels).words());
        }
        bytes
    }
}

/// One case of the family.
pub struct BlendSweep {
    name: &'static str,
    sweep: Sweep,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<BlendSweep> {
    let case = |name, sweep| BlendSweep { name, sweep };
    vec![
        case("blend-sweep-source-rgb", Sweep::Slot(Slot::SourceRgb)),
        case(
            "blend-sweep-destination-rgb",
            Sweep::Slot(Slot::DestinationRgb),
        ),
        case("blend-sweep-source-alpha", Sweep::Slot(Slot::SourceAlpha)),
        case(
            "blend-sweep-destination-alpha",
            Sweep::Slot(Slot::DestinationAlpha),
        ),
        case("blend-sweep-operations", Sweep::Operations),
        case("blend-sweep-operation-pairs", Sweep::OperationPairs),
        case("blend-sweep-write-masks", Sweep::WriteMasks),
        case("blend-sweep-enabled", Sweep::Enabled),
        case("blend-sweep-blend-color", Sweep::BlendColour),
        case("blend-sweep-blend-color-order", Sweep::BlendColourOrder),
        case("blend-sweep-second-source", Sweep::SecondSource),
    ]
}

/// The target every pass blends into: of the case's own memory and of shared storage, because an
/// arbitrary destination code reaches a texture only through `replaceRegion:`.
fn target_spec() -> TargetSpec {
    TargetSpec::plain(FORMAT, [WIDTH, HEIGHT]).shared()
}

/// The layout a stored level is compared under: one row a pixel, a channel an element, so a diff
/// names the pixel and the channel inside it rather than a byte offset.
fn layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [WIDTH, HEIGHT, 1],
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: TEXEL,
        rows: PIXELS as u32,
        image_pitch: LEVEL as u32,
        images: 1,
        // A row is one pixel's four channels and the pixels abut, so no byte lies outside one.
        gaps: Gaps::Ignored,
    }
}

impl BlendSweep {
    /// Whether the case's fragment function returns a second output, which only a `Source1` factor
    /// blends and which needs the device's dual-source blending.
    fn dual(&self) -> bool {
        self.sweep == Sweep::SecondSource
    }

    /// The fixture and the two entry points the case draws with.
    fn program(&self) -> (&'static str, &'static str, &'static str) {
        if self.dual() {
            (DUAL, DUAL_VERTEX, DUAL_FRAGMENT)
        } else {
            (FILLS, VERTEX, FILL_FLOAT)
        }
    }

    /// Every row of the case's sweep.
    fn variants(&self) -> Vec<Variant> {
        let plain = ColorState::new(FORMAT);
        match self.sweep {
            Sweep::Slot(slot) => factors(Family::Settable)
                .into_iter()
                .map(|factor| Variant::new(factor_name(factor), plain.blended(slot.blend(factor))))
                .collect(),
            Sweep::Operations => {
                let mut rows = Vec::new();
                for (pair, source, destination) in operation_pairs() {
                    for op in operations() {
                        rows.push(Variant::new(
                            format!("{pair}-{}", operation_name(op)),
                            plain.blended(Blend::separate(
                                (source, destination, op),
                                (source, destination, op),
                            )),
                        ));
                    }
                }
                rows
            }
            Sweep::OperationPairs => {
                let mut rows = Vec::new();
                for rgb in operations() {
                    for alpha in operations() {
                        rows.push(Variant::new(
                            format!("{}-{}", operation_name(rgb), operation_name(alpha)),
                            plain.blended(Blend::separate(
                                (F::One, F::One, rgb),
                                (F::One, F::One, alpha),
                            )),
                        ));
                    }
                }
                rows
            }
            Sweep::WriteMasks => {
                let mut rows = Vec::new();
                for (label, blend) in [
                    ("plain", None),
                    ("blend", Some(Blend::added(F::One, F::One))),
                ] {
                    for (mask, name) in masks() {
                        let mut color = plain.masked(mask);
                        if let Some(blend) = blend {
                            color = color.blended(blend);
                        }
                        rows.push(Variant::new(format!("{label}-{name}"), color));
                    }
                }
                rows
            }
            Sweep::Enabled => vec![
                Variant::new("off".to_owned(), plain),
                Variant::new(
                    "on-one-zero".to_owned(),
                    plain.blended(Blend::added(F::One, F::Zero)),
                ),
                Variant::new(
                    "on-one-one".to_owned(),
                    plain.blended(Blend::added(F::One, F::One)),
                ),
            ],
            Sweep::BlendColour => {
                let direct = plain.blended(Blend::separate(
                    (F::BlendColor, F::Zero, O::Add),
                    (F::BlendAlpha, F::Zero, O::Add),
                ));
                let complement = plain.blended(Blend::separate(
                    (F::OneMinusBlendColor, F::Zero, O::Add),
                    (F::OneMinusBlendAlpha, F::Zero, O::Add),
                ));
                let row = |name: &str, color: ColorState, blend_colors: Vec<[f32; 4]>| Variant {
                    name: name.to_owned(),
                    color,
                    blend_colors,
                };
                vec![
                    row("unset", direct, Vec::new()),
                    row("zeros", direct, vec![[0.0; 4]]),
                    row("ones", direct, vec![[1.0; 4]]),
                    row("mid", direct, vec![BLEND_COLOR]),
                    row("above", direct, vec![ABOVE]),
                    row("below", direct, vec![BELOW]),
                    row("mid-complement", complement, vec![BLEND_COLOR]),
                    row("above-complement", complement, vec![ABOVE]),
                    row("below-complement", complement, vec![BELOW]),
                ]
            }
            Sweep::BlendColourOrder => {
                let direct = plain.blended(Blend::separate(
                    (F::BlendColor, F::Zero, O::Add),
                    (F::BlendAlpha, F::Zero, O::Add),
                ));
                let row = |name: &str, blend_colors: Vec<[f32; 4]>| Variant {
                    name: name.to_owned(),
                    color: direct,
                    blend_colors,
                };
                vec![
                    row("once", vec![BLEND_COLOR]),
                    row("twice", vec![BELOW, BLEND_COLOR]),
                    row("reversed", vec![BLEND_COLOR, BELOW]),
                ]
            }
            Sweep::SecondSource => {
                let mut rows = vec![Variant::new(
                    "control-one".to_owned(),
                    plain.blended(Blend::added(F::One, F::Zero)),
                )];
                for slot in [
                    Slot::SourceRgb,
                    Slot::DestinationRgb,
                    Slot::SourceAlpha,
                    Slot::DestinationAlpha,
                ] {
                    for factor in factors(Family::SecondOutput) {
                        rows.push(Variant::new(
                            format!("{}-{}", slot.name(), factor_name(factor)),
                            plain.blended(slot.blend(factor)),
                        ));
                    }
                }
                rows
            }
        }
    }
}

/// All sixteen write masks of the four-bit field, each named by the channels it keeps in the SDK's
/// own declaration order, or `none`.
fn masks() -> Vec<(ColorWriteMask, String)> {
    let bits = channel_bits();
    (0..1u32 << bits.len())
        .map(|held| {
            let mut mask = ColorWriteMask::NONE;
            let mut name = String::new();
            for (index, (bit, metal)) in bits.iter().enumerate() {
                if held & (1 << index) != 0 {
                    mask = mask.union(*bit);
                    name.extend(
                        kebab(metal, "MTLColorWriteMask")
                            .chars()
                            .next()
                            .map(|letter| letter.to_string()),
                    );
                }
            }
            if name.is_empty() {
                name = "none".to_owned();
            }
            (mask, name)
        })
        .collect()
}

/// Every pair of sites whose answer is one answer, with the reason it cannot be otherwise.
///
/// A pair here is a fact about Metal that the recording measures, not permission for anything: the
/// model gives the two sites the same image, and
/// `tests::metals_recorded_texels_are_the_blend_model_at_every_pixel` holds the model against
/// Metal. `tests::the_sweeps_separate_every_value_except_the_named_collapses` fails on any other
/// pair that agrees, which is what makes a badly chosen input a failure rather than a pass.
#[cfg(test)]
const COLLAPSED: &[(&str, &str, &str, &str)] = &[
    // The alpha slots: a factor named for a colour reads the colour's alpha channel there, and
    // `SourceAlphaSaturated`'s alpha component is 1. Seven pairs a slot, by Metal's definition.
    (
        "blend-sweep-source-alpha",
        "one",
        "source-alpha-saturated",
        "the saturated factor's alpha component is 1",
    ),
    (
        "blend-sweep-source-alpha",
        "source-color",
        "source-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-source-alpha",
        "one-minus-source-color",
        "one-minus-source-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-source-alpha",
        "destination-color",
        "destination-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-source-alpha",
        "one-minus-destination-color",
        "one-minus-destination-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-source-alpha",
        "blend-color",
        "blend-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-source-alpha",
        "one-minus-blend-color",
        "one-minus-blend-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-destination-alpha",
        "one",
        "source-alpha-saturated",
        "the saturated factor's alpha component is 1",
    ),
    (
        "blend-sweep-destination-alpha",
        "source-color",
        "source-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-destination-alpha",
        "one-minus-source-color",
        "one-minus-source-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-destination-alpha",
        "destination-color",
        "destination-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-destination-alpha",
        "one-minus-destination-color",
        "one-minus-destination-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-destination-alpha",
        "blend-color",
        "blend-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-destination-alpha",
        "one-minus-blend-color",
        "one-minus-blend-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    // The operations: `Min` and `Max` read no factor, so the six factor pairs give one answer under
    // each -- which is the finding the pairwise sweep exists to make, stated as six sites that must
    // agree rather than one site nobody can check.
    (
        "blend-sweep-operations",
        "one-zero-min",
        "one-one-min",
        "min reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-min",
        "sa-1msa-min",
        "min reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-min",
        "dc-1mdc-min",
        "min reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-min",
        "bc-1mbc-min",
        "min reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-min",
        "sas-da-min",
        "min reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-max",
        "one-one-max",
        "max reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-max",
        "sa-1msa-max",
        "max reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-max",
        "dc-1mdc-max",
        "max reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-max",
        "bc-1mbc-max",
        "max reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-max",
        "sas-da-max",
        "max reads no factor",
    ),
    (
        "blend-sweep-operations",
        "one-zero-add",
        "one-zero-subtract",
        "a destination factor of zero leaves nothing to subtract",
    ),
    // A mask of no channel writes nothing, so whether the pipeline blended is unobservable through
    // it. That is the one pair of the mask sweep, and it is the reason the other fifteen masks are
    // asked twice.
    (
        "blend-sweep-write-masks",
        "plain-none",
        "blend-none",
        "a mask of no channel stores neither answer",
    ),
    // The control: blending off is the identity blend. The case's point is the other direction,
    // that `One, One` is *not* this image.
    (
        "blend-sweep-enabled",
        "off",
        "on-one-zero",
        "blending off is source times one plus destination times zero",
    ),
    // The blend colour: the default is zero, the second call wins, and `1 - c` of each out-of-range
    // colour is the other one.
    (
        "blend-sweep-blend-color",
        "unset",
        "zeros",
        "Metal's blend colour before anything sets it is (0, 0, 0, 0)",
    ),
    (
        "blend-sweep-blend-color-order",
        "once",
        "twice",
        "the last setBlendColor: of a pass is the one the draw reads",
    ),
    (
        "blend-sweep-blend-color",
        "below",
        "above-complement",
        "one minus the above-range colour is the below-range one, exactly and unclamped",
    ),
    (
        "blend-sweep-blend-color",
        "above",
        "below-complement",
        "one minus the below-range colour is the above-range one, exactly and unclamped",
    ),
    // The dual-source slots' alpha channels collapse exactly as the ordinary alpha slots do.
    (
        "blend-sweep-second-source",
        "source-alpha-source1-color",
        "source-alpha-source1-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-second-source",
        "source-alpha-one-minus-source1-color",
        "source-alpha-one-minus-source1-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-second-source",
        "destination-alpha-source1-color",
        "destination-alpha-source1-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
    (
        "blend-sweep-second-source",
        "destination-alpha-one-minus-source1-color",
        "destination-alpha-one-minus-source1-alpha",
        "a colour factor's alpha component is that colour's alpha",
    ),
];

impl Case for BlendSweep {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Nothing is loosened, and nothing may be. Every number this family forms is exact in the
        // target's `f32` channel (the module docs give the arithmetic), so a texel that differs is
        // a difference in the blend and not in a rounding, and a tolerance written here would hide
        // exactly the bug the sweep exists to find.
        Vec::new()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let (name, vertex, fragment) = self.program();
        shader_refs(name, &[vertex, fragment])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let (name, _, _) = self.program();
        let mut inputs = fixture_inputs("program", name)?;
        inputs.extend([
            Input::bytes("target", &target_spec().words()),
            Input::bytes("buffer:source", &source_bytes()),
            Input::bytes("buffer:shape", &shape(WIDTH, 1)),
            // The destination every blend reads, which half of every answer depends on.
            Input::bytes("texture:destination", &destination_bytes()),
        ]);
        if self.dual() {
            inputs.push(Input::bytes("buffer:second", &second_bytes()));
        }
        // Each row's own state, keyed by its name, so a case whose sweep moved is unrecorded rather
        // than compared against the old sweep's texels.
        for variant in self.variants() {
            inputs.push(Input::bytes(
                format!("variant:{}", variant.name),
                &variant.words(),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let (name, vertex, fragment) = self.program();
        let loaded = fixture(name, vertex)
            .loaded()
            .map_err(|why| not_read("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let sources = made(sites, "buffer/source", side.buffer(&source_bytes()))?;
        let extent = made(sites, "buffer/shape", side.buffer(&shape(WIDTH, 1)))?;
        let seconds = match self.dual() {
            true => Some(made(sites, "buffer/second", side.buffer(&second_bytes()))?),
            false => None,
        };
        let destination = destination_bytes();
        let variants = self.variants();
        let mut work = Vec::new();
        let mut reads = Vec::new();
        for variant in &variants {
            let colors = [Some(variant.color)];
            let pipeline = made(
                sites,
                &format!("pipeline/{}", variant.name),
                side.render_pipeline(library, &RenderPipelineSpec::new(vertex, fragment, &colors)),
            )?;
            let target = made(
                sites,
                &format!("target/{}", variant.name),
                side.target(&target_spec()),
            )?;
            // The destination codes go in before the pass: a blend's answer depends on them, and
            // the pass loads rather than clears, so this is the only thing that puts them there.
            call(
                sites,
                &format!("replace-region/{}", variant.name),
                side.replace_region(target, [WIDTH, HEIGHT], WIDTH * TEXEL, &destination),
                |_| Value::Absent,
            )?;
            let into = made(
                sites,
                &format!("buffer/texels/{}", variant.name),
                side.zeroed_buffer(LEVEL),
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
            if let Some(seconds) = seconds {
                steps.push(RenderStep::FragmentBuffer {
                    buffer: seconds,
                    offset: 0,
                    index: 2,
                });
            }
            for channels in &variant.blend_colors {
                steps.push(RenderStep::BlendColor(BlendColorChannels::new(*channels)));
            }
            steps.push(RenderStep::triangle());
            work.push(RenderWork::Pass(
                RenderPass::clearing(vec![PassColor::loaded(target)]).drawing(steps),
            ));
            work.push(RenderWork::ReadTexture {
                texture: target,
                into,
                offset: 0,
                row_pitch: WIDTH * TEXEL,
                size: [WIDTH, HEIGHT],
                plane: Plane::Whole,
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
                side.read_buffer(buffer, 0, LEVEL),
                |_| Value::Absent,
            )?;
            stored = stored.with(Node::leaf(name, Observed::bytes(bytes, layout())));
        }
        sites.push(stored);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! The sweep's own depth, held against the SDK's tables, and Metal's recorded texels held
    //! against one statement of the blend equation.

    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// Every value of `MTLBlendFactor` the SDK table holds is classified, so a factor a later SDK
    /// adds fails here rather than joining a list silently.
    ///
    /// `BlendFactor` is `#[non_exhaustive]`, so no match in this crate can be exhaustive and the
    /// compiler cannot hold this; the test is what does.
    #[test]
    fn every_blend_factor_of_the_sdk_table_is_classified() {
        let unclassified: Vec<&str> = F::ALL
            .iter()
            .copied()
            .filter(|factor| family(*factor) == Family::Unclassified)
            .map(F::name)
            .collect();
        assert!(
            unclassified.is_empty(),
            "the SDK's MTLBlendFactor table holds {} value(s) this family neither sweeps nor \
             explains: {unclassified:?}",
            unclassified.len()
        );
        // The depth of the sweep, as a tally rather than a claim: every value of the table is in
        // exactly one of the three families, and the two the slot cases cannot set are named for
        // their own reasons in the module docs.
        assert_eq!(
            factors(Family::Settable).len()
                + factors(Family::SecondOutput).len()
                + factors(Family::Placeholder).len(),
            F::ALL.len()
        );
        assert_eq!(factors(Family::Placeholder), vec![F::Unspecialized]);
        assert_eq!(factors(Family::SecondOutput).len(), 4);
    }

    /// Every value of `MTLBlendOperation` but the specialization placeholder is swept, and the
    /// placeholder is the only one left out.
    #[test]
    fn every_blend_operation_of_the_sdk_table_is_classified() {
        let swept = operations();
        let left: Vec<&str> = O::ALL
            .iter()
            .copied()
            .filter(|op| !swept.contains(op))
            .map(O::name)
            .collect();
        assert_eq!(
            left,
            vec!["MTLBlendOperationUnspecialized"],
            "the operations left out of the sweep are not the one placeholder"
        );
        assert_eq!(swept.len() + left.len(), O::ALL.len());
    }

    /// The mask sweep is the whole four-bit field: sixteen distinct masks built from the SDK's own
    /// single-channel bits, their union is `MTLColorWriteMaskAll`, and `None` and `All` are both
    /// among them.
    #[test]
    fn the_write_mask_sweep_is_the_whole_four_bit_field() {
        let bits = channel_bits();
        assert_eq!(bits.len(), 4, "the field is not four bits: {bits:?}");
        // The order matters beyond this test: `model` reads `channel_bits()[channel]` to decide
        // whether channel `channel` of a texel is written, so the table's order and the texel's
        // channel order must be the same one. A reorder upstream fails here rather than masking the
        // wrong channel in every row of the sweep.
        assert_eq!(
            bits.iter().map(|(_, name)| *name).collect::<Vec<&str>>(),
            vec![
                "MTLColorWriteMaskRed",
                "MTLColorWriteMaskGreen",
                "MTLColorWriteMaskBlue",
                "MTLColorWriteMaskAlpha",
            ]
        );
        let swept = masks();
        assert_eq!(swept.len(), 16);
        let raws: BTreeSet<u32> = swept.iter().map(|(mask, _)| mask.as_raw()).collect();
        assert_eq!(raws.len(), 16, "two rows of the sweep are one mask");
        assert_eq!(
            raws,
            (0..=ColorWriteMask::ALL.as_raw()).collect::<BTreeSet<u32>>(),
            "the sweep is not every subset of MTLColorWriteMaskAll"
        );
        let names: BTreeSet<&str> = swept.iter().map(|(_, name)| name.as_str()).collect();
        assert_eq!(names.len(), 16, "two rows of the sweep share a name");
        // The one named value of the table outside the field is Metal 4's placeholder, which no
        // Metal 3 descriptor takes and no row here sets.
        let outside: Vec<&str> = ColorWriteMask::KNOWN
            .iter()
            .filter(|(mask, _)| !ColorWriteMask::ALL.contains(*mask))
            .map(|(_, name)| *name)
            .collect();
        assert_eq!(outside, vec!["MTLColorWriteMaskUnspecialized"]);
    }

    /// Each slot case sets every settable factor in its own slot and leaves the other three slots
    /// constant, and the four cases between them cover the four slots.
    #[test]
    fn the_slot_sweeps_hold_every_settable_factor_in_every_slot() {
        let settable = factors(Family::Settable);
        let mut slots = BTreeSet::new();
        for case in all() {
            let Sweep::Slot(slot) = case.sweep else {
                continue;
            };
            slots.insert(slot.name());
            let rows = case.variants();
            assert_eq!(rows.len(), settable.len(), "{}", case.name);
            for (row, factor) in rows.iter().zip(&settable) {
                assert_eq!(row.name, factor_name(*factor), "{}", case.name);
                let blend = row.color.blend.expect("a slot sweep blends");
                let held = match slot {
                    Slot::SourceRgb => blend.source_rgb,
                    Slot::DestinationRgb => blend.destination_rgb,
                    Slot::SourceAlpha => blend.source_alpha,
                    Slot::DestinationAlpha => blend.destination_alpha,
                };
                assert_eq!(held, *factor, "{}: {} is in no slot", case.name, row.name);
                // The operation is `Add` in both equations, so the swept factor is the only thing
                // that moves between this case's rows.
                assert_eq!(blend.rgb, O::Add);
                assert_eq!(blend.alpha, O::Add);
            }
        }
        assert_eq!(
            slots,
            BTreeSet::from([
                "source-rgb",
                "destination-rgb",
                "source-alpha",
                "destination-alpha"
            ])
        );
    }

    /// The dual-source case sets each of the four second-output factors in each of the four slots,
    /// and carries one row whose blend reads no second output as the control that the pipeline draws
    /// at all.
    #[test]
    fn the_second_source_case_holds_every_second_output_factor_in_every_slot() {
        let case = all()
            .into_iter()
            .find(|case| case.sweep == Sweep::SecondSource)
            .expect("the family holds a dual-source case");
        let rows = case.variants();
        let second = factors(Family::SecondOutput);
        assert_eq!(rows.len(), 1 + 4 * second.len());
        assert_eq!(rows[0].name, "control-one");
        for slot in [
            "source-rgb",
            "destination-rgb",
            "source-alpha",
            "destination-alpha",
        ] {
            for factor in &second {
                let name = format!("{slot}-{}", factor_name(*factor));
                assert!(
                    rows.iter().any(|row| row.name == name),
                    "no {name} row of the dual-source case"
                );
            }
        }
        // And it is the only case that needs a second output, so every other case runs on a device
        // without dual-source blending.
        for other in all().into_iter().filter(|other| !other.dual()) {
            for row in other.variants() {
                if let Some(blend) = row.color.blend {
                    for factor in [
                        blend.source_rgb,
                        blend.destination_rgb,
                        blend.source_alpha,
                        blend.destination_alpha,
                    ] {
                        assert_ne!(
                            family(factor),
                            Family::SecondOutput,
                            "{} row {} reads the second output",
                            other.name,
                            row.name
                        );
                    }
                }
            }
        }
    }

    /// No case names a value the model has no arm for, and no row names a placeholder.
    #[test]
    fn no_row_of_the_family_sets_a_specialization_placeholder() {
        for case in all() {
            for row in case.variants() {
                if let Some(blend) = row.color.blend {
                    for factor in [
                        blend.source_rgb,
                        blend.destination_rgb,
                        blend.source_alpha,
                        blend.destination_alpha,
                    ] {
                        assert_ne!(family(factor), Family::Placeholder, "{}", case.name);
                    }
                    for op in [blend.rgb, blend.alpha] {
                        assert_ne!(op, O::Unspecialized, "{}", case.name);
                    }
                }
                assert!(
                    !row.color.write_mask.contains(ColorWriteMask::UNSPECIALIZED),
                    "{}",
                    case.name
                );
            }
        }
    }

    /// The groups of sites the model gives one answer to are exactly the pairs `COLLAPSED` names,
    /// and every other site of every case is its own answer.
    ///
    /// This is the family's whole guard against a vacuous sweep. A source colour or a destination
    /// that failed to separate two factors would show up here as a group nothing explains, which
    /// fails; and a pair named here that the model separates fails too, so a row cannot outlive its
    /// reason.
    #[test]
    fn the_sweeps_separate_every_value_except_the_named_collapses() {
        for case in all() {
            let rows = case.variants();
            let mut by_answer: BTreeMap<Vec<u8>, BTreeSet<String>> = BTreeMap::new();
            for row in &rows {
                by_answer
                    .entry(model_bytes(row))
                    .or_default()
                    .insert(row.name.clone());
            }
            // The sites that agree, as the model computes them.
            let found: BTreeSet<BTreeSet<String>> = by_answer
                .into_values()
                .filter(|group| group.len() > 1)
                .collect();
            // The sites that must agree, as the closure of the named pairs.
            let mut named: Vec<BTreeSet<String>> = Vec::new();
            for (which, left, right, _) in COLLAPSED {
                if *which != case.name {
                    continue;
                }
                for name in [left, right] {
                    assert!(
                        rows.iter().any(|row| row.name == *name),
                        "{}: the collapse table names a site {name} the case does not hold",
                        case.name
                    );
                }
                let mut joined = BTreeSet::from([(*left).to_owned(), (*right).to_owned()]);
                named.retain(|group| {
                    if group.is_disjoint(&joined) {
                        true
                    } else {
                        joined.extend(group.iter().cloned());
                        false
                    }
                });
                named.push(joined);
            }
            let named: BTreeSet<BTreeSet<String>> = named.into_iter().collect();
            assert_eq!(
                found, named,
                "{}: the sites that give one answer are not the ones the collapse table explains",
                case.name
            );
            // Every case moves something: a sweep whose rows all agreed would compare nothing.
            let answers: BTreeSet<Vec<u8>> = rows.iter().map(model_bytes).collect();
            assert!(
                answers.len() > 1,
                "{}: every row of the sweep gives one answer",
                case.name
            );
        }
    }

    /// `SourceAlphaSaturated` takes both branches of its minimum over the destination sweep, so it
    /// is distinguishable from the two factors it would otherwise equal.
    ///
    /// Without this the factor is the one the module docs name as the trap: `min(sa, 1 - da)` is
    /// always one of `sa` and `1 - da`, so a single destination compares nothing about it.
    #[test]
    fn the_saturated_factor_takes_both_of_its_branches() {
        let mut left = 0;
        let mut right = 0;
        for pixel in 0..PIXELS {
            let da = destination(pixel)[3];
            let saturated = SOURCE[3].min(1.0 - da);
            if saturated == SOURCE[3] {
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
        // And over the whole image its answer is neither of the two factors it equals pixel by
        // pixel. Asked of the two **colour** slots: in an alpha slot the factor is 1 and collapses
        // with `One` instead, which the collapse table names.
        for case in all().into_iter().filter(|case| {
            matches!(
                case.sweep,
                Sweep::Slot(Slot::SourceRgb | Slot::DestinationRgb)
            )
        }) {
            let rows = case.variants();
            let of = |name: &str| {
                rows.iter()
                    .find(|row| row.name == name)
                    .map(model_bytes)
                    .unwrap_or_else(|| panic!("{}: no {name} row", case.name))
            };
            let saturated = of("source-alpha-saturated");
            for other in ["source-alpha", "one-minus-destination-alpha"] {
                assert_ne!(
                    saturated,
                    of(other),
                    "{}: the saturated factor is indistinguishable from {other}",
                    case.name
                );
            }
        }
    }

    /// The inputs that separate the factors are the twenty-four distinct sixty-fourths the module
    /// docs claim, so no constant factor is another constant factor, `Zero` or `One`.
    #[test]
    fn the_constant_factors_are_twenty_four_distinct_numbers() {
        let mut held: BTreeSet<u32> = BTreeSet::new();
        for colour in [SOURCE, BLEND_COLOR, SECOND] {
            for channel in colour {
                for value in [channel, 1.0 - channel] {
                    assert_eq!(
                        value * 64.0,
                        (value * 64.0).round(),
                        "{value} is not a multiple of 1/64"
                    );
                    assert!(held.insert(value.to_bits()), "{value} appears twice");
                }
            }
        }
        assert_eq!(held.len(), 24);
        assert!(!held.contains(&0.0f32.to_bits()));
        assert!(!held.contains(&1.0f32.to_bits()));
    }

    /// Each case states inputs that cover every row of its sweep, names Metal as its source, and
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
            // Two rows that differ only in their blend colour still fingerprint apart, which is
            // what makes `mid` and `twice` two questions rather than one.
            let words: BTreeSet<Vec<u8>> = rows.iter().map(Variant::words).collect();
            assert_eq!(
                words.len(),
                rows.len(),
                "{}: two rows fingerprint alike",
                case.name()
            );
        }
    }

    /// The sweep is the size the module docs quote: 177 blend states over 64 pixels and four
    /// channels, and each case holds the rows its own rule gives.
    ///
    /// A number in prose is a number that rots, so the prose quotes these and this holds them. The
    /// per-case counts are each derived from the SDK table or from the shape of the sweep -- the
    /// settable factors, the operations squared, the four-bit field twice over -- so a row that
    /// quietly disappeared from a sweep fails here.
    #[test]
    fn the_sweep_is_the_size_its_prose_claims() {
        let settable = factors(Family::Settable).len();
        let second = factors(Family::SecondOutput).len();
        let ops = operations().len();
        let want: Vec<(&str, usize)> = vec![
            ("blend-sweep-source-rgb", settable),
            ("blend-sweep-destination-rgb", settable),
            ("blend-sweep-source-alpha", settable),
            ("blend-sweep-destination-alpha", settable),
            ("blend-sweep-operations", operation_pairs().len() * ops),
            ("blend-sweep-operation-pairs", ops * ops),
            ("blend-sweep-write-masks", 2 * masks().len()),
            ("blend-sweep-enabled", 3),
            ("blend-sweep-blend-color", 9),
            ("blend-sweep-blend-color-order", 3),
            ("blend-sweep-second-source", 1 + 4 * second),
        ];
        let held: Vec<(&str, usize)> = all()
            .iter()
            .map(|case| (case.name(), case.variants().len()))
            .collect();
        assert_eq!(held, want);
        let states: usize = held.iter().map(|(_, rows)| rows).sum();
        assert_eq!(states, 179, "the family holds {states} blend states");
        assert_eq!(states * PIXELS * 4, 45_824);
        // And every state is compared over the whole level, which is what makes the channel count
        // the comparison's own size rather than a multiplication in the prose.
        assert_eq!(layout().extent(), LEVEL);
        assert_eq!(LEVEL, PIXELS * 4 * 4);
    }

    /// Metal's `Subtract` stores negative zero where both factored terms are zero, and positive
    /// zero everywhere else a zero comes out — including under `Subtract` itself where the two
    /// terms are equal and non-zero.
    ///
    /// Read straight off the recording, not off the model, so the model's one measured exception
    /// cannot be the thing that proves itself. The boundary is what makes the exception a fact
    /// rather than a fudge: in the **same site**, under the **same operation**, a leaf whose two
    /// terms are equal and non-zero stores `+0` and a leaf whose two terms are both zero stores
    /// `-0`, so nothing wider than "both terms zero" is being claimed. `Add`, `ReverseSubtract`,
    /// `Min` and `Max` over the same two zero operands store `+0`, which is the other half.
    #[test]
    fn metals_subtract_stores_a_negative_zero_where_both_terms_are_zero() {
        use metal2vulkan_runtime::MetalLevel;
        let case = all()
            .into_iter()
            .find(|case| case.sweep == Sweep::Operations)
            .expect("the family holds an operations case");
        let Ok(recording) = crate::recording::read(case.name()) else {
            return;
        };
        let answer = recording.answer(MetalLevel::NEWEST).unwrap();
        let taken = answer.deterministic().expect("one outcome");
        let blended = taken.root.child("blended").expect("the blended site");
        let held = layout();
        let read = |site: &str, pixel: usize, channel: usize| -> f32 {
            let node = blended
                .child(site)
                .unwrap_or_else(|| panic!("no {site} site"));
            let Value::Bytes(bytes) = &node.observed.value else {
                panic!("{site} holds no bytes");
            };
            let at = held.at(0, pixel as u32, channel as u32);
            f32::from_bits(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
        };
        // `dc-1mdc` factors the source by the destination and the destination by its complement, so
        // both terms are zero exactly where the destination channel is zero.
        let mut both_zero = Vec::new();
        let mut equal_and_not_zero = Vec::new();
        for pixel in 0..PIXELS {
            for (channel, s) in SOURCE.into_iter().enumerate() {
                let d = destination(pixel)[channel];
                let from_source = s * d;
                let from_destination = d * (1.0 - d);
                if from_source == 0.0 && from_destination == 0.0 {
                    both_zero.push((pixel, channel));
                } else if from_source == from_destination {
                    equal_and_not_zero.push((pixel, channel));
                }
            }
        }
        assert_eq!(
            both_zero.len(),
            4,
            "the destination sweep reaches zero in {} leaves, not four: {both_zero:?}",
            both_zero.len()
        );
        assert!(
            !equal_and_not_zero.is_empty(),
            "no leaf of dc-1mdc has two equal non-zero terms, so the boundary is untested"
        );
        for (pixel, channel) in &both_zero {
            assert_eq!(
                read("dc-1mdc-subtract", *pixel, *channel).to_bits(),
                (-0.0f32).to_bits(),
                "dc-1mdc-subtract pixel {pixel} channel {channel}: both terms are zero, so Metal \
                 stores negative zero"
            );
            // `Add` and `ReverseSubtract` are the other two operations over these same two zero
            // terms, and both store positive zero. `Min` and `Max` are not about the terms at all
            // -- they read no factor, which the collapse table already holds -- so they store the
            // smaller and the larger of the source and the destination and have no zero here to
            // sign.
            for other in ["dc-1mdc-add", "dc-1mdc-reverse-subtract"] {
                assert_eq!(
                    read(other, *pixel, *channel).to_bits(),
                    0.0f32.to_bits(),
                    "{other} pixel {pixel} channel {channel}: only Subtract signs its zero"
                );
            }
            assert_eq!(
                read("dc-1mdc-max", *pixel, *channel),
                SOURCE[*channel].max(destination(*pixel)[*channel]),
                "dc-1mdc-max pixel {pixel} channel {channel}"
            );
        }
        for (pixel, channel) in &equal_and_not_zero {
            assert_eq!(
                read("dc-1mdc-subtract", *pixel, *channel).to_bits(),
                0.0f32.to_bits(),
                "dc-1mdc-subtract pixel {pixel} channel {channel}: the two terms are equal and \
                 non-zero, so the zero is positive and the exception is only about zero terms"
            );
        }
    }

    /// Metal's recorded texels are the blend model, at every pixel of every row of every case.
    ///
    /// The strongest thing the family can say, and the one that makes the rest of it mean anything:
    /// the model's answer at a site depends on the source buffer, the destination the host wrote,
    /// the blend colour the pass set and the six fields of the state, so a setup that failed to
    /// deliver any of them fails here rather than passing on bytes that happen to match.
    #[test]
    fn metals_recorded_texels_are_the_blend_model_at_every_pixel() {
        use metal2vulkan_runtime::MetalLevel;
        let mut checked = 0usize;
        let mut unrecorded = Vec::new();
        for case in all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                unrecorded.push(case.name());
                continue;
            };
            let answer = recording
                .answer(MetalLevel::NEWEST)
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            let taken = answer
                .deterministic()
                .unwrap_or_else(|| panic!("{}: more than one outcome", case.name()));
            let blended = taken
                .root
                .child("blended")
                .unwrap_or_else(|| panic!("{}: no blended site", case.name()));
            for row in case.variants() {
                let site = blended
                    .child(&row.name)
                    .unwrap_or_else(|| panic!("{}: no {} site", case.name(), row.name));
                let Value::Bytes(bytes) = &site.observed.value else {
                    panic!(
                        "{}: {} holds {}",
                        case.name(),
                        row.name,
                        site.observed.value.kind()
                    );
                };
                assert_eq!(bytes.len(), LEVEL, "{}: {}", case.name(), row.name);
                let want = model_bytes(&row);
                if *bytes == want {
                    checked += 1;
                    continue;
                }
                // Name the first pixel and channel that moved, with both numbers, rather than
                // printing a kilobyte of bytes.
                let held = layout();
                for pixel in 0..PIXELS {
                    for channel in 0..4u32 {
                        let at = held.at(0, pixel as u32, channel);
                        let read = f32::from_bits(u32::from_le_bytes(
                            bytes[at..at + 4].try_into().unwrap(),
                        ));
                        let modelled = model(&row, pixel)[channel as usize];
                        assert_eq!(
                            read.to_bits(),
                            modelled.to_bits(),
                            "{}: {} pixel {pixel} channel {channel}: Metal stored {read}, the \
                             blend model says {modelled} (source {:?}, destination {:?})",
                            case.name(),
                            row.name,
                            SOURCE,
                            destination(pixel)
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
}
