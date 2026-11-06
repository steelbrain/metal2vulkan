//! How a fragment function's float color outputs reach the attachments whose formats convert
//! them (`LoweringInputs::color_writes`).
//!
//! The device converts each output it is handed: its conversion is the contract. Metal stores a
//! float written to a normalized attachment as the exact product with the format's maximum code
//! rounded to nearest, to an 8-bit sRGB one through its twelve-bit encode, and to a half-float one
//! rounded toward zero (`metal::format::write_conversion`, probes on an Apple M3 Max and an M4).
//! ANV, RADV and lavapipe round the `f32` product with ties to even, NVIDIA's 615.71.09 on the
//! RTX 5080 Laptop GPU stores an 8-bit channel's lower code on a tie and up to two `f32` steps above
//! one (0.5 as `0x7f`, where Metal stores `0x80`), and each encodes sRGB by its own approximation,
//! so near a tie they may store another code (`runtime_lane::color_writes`). Their half-float and small-float channels round toward zero as
//! Metal's do where measured (`DriverDeviation::MetalSmallFloatColorWrites`: RADV on the Radeon
//! 780M, ANV on the Iris Xe and the UHD 770, NVIDIA on the RTX 3090, NaNs aside); only MoltenVK on
//! Apple's GPUs stores Metal's codes for every format (`VkProperty::MetalColorWrites`). The map names each channel's rule so the
//! fragment function can record where the device's conversion is not measured to be Metal's
//! (`PrecisionNote::DeviceColorConversion`); it rounds none of them. On MoltenVK the map keeps the
//! half-float channels alone, which record the device's NaN encoding of a `half` output.
//!
//! The map names each rule as the function's lowering on the profile tells it apart
//! ([`lowered`]): a rule whose channel the device's conversion stores and notes alike, Metal's
//! rule or not (an 8-bit unorm channel's on lavapipe, say), is [`ChannelWrite::Device`]. So a
//! fragment function linked into pipelines that differ there alone -- a sweep of blend states,
//! one of them of factors one and zero -- compiles under one map, once. Whether a pipeline's
//! attachment takes a rule of Metal's at all, which decides its blend of one and zero
//! (`link::graphics`), is the pipeline's own ([`takes_metal_rule`]).
//!
//! Under `TextureWriteRoundingMode::NearestEven` (Apple's `textureWriteRoundingMode` 2) Metal
//! rounds a render state's half-float channels to nearest even ([`HALF_NEAREST`]), the mode the
//! pipeline asks for, which MoltenVK's conversion (toward zero, Metal's default) does not give: the
//! function rounds each such channel itself on every profile, where the channel stores the output
//! as written.
//!
//! An attachment that blends with factors other than one and zero keeps the device's conversion
//! of the blend's result ([`ChannelWrite::Device`]), and so does the alpha of attachment 0 under
//! alpha-to-coverage at the default rounding mode, which the coverage reads as computed. Under
//! `NearestEven` that alpha rounds as the other channels do where the attachment stores it: Metal
//! stores the nearest-even half and reads the coverage from the alpha before rounding (probe
//! `a2c` on an Apple M3 Max, macOS 27.0.1: the coverage of 16,384 alphas is the same under either
//! mode), which no device separates, so the coverage reads the rounded alpha
//! (`PrecisionNote::CoverageReadsRoundedAlpha`, [`pipeline_writes`]). An alpha its write mask does
//! not store keeps the device's conversion, which stores nothing, so the coverage reads it as
//! computed, as Metal's does. A format whose conversion Metal was not measured with keeps the
//! device's conversion too, and so does a 32-bit float attachment that blends so, whose stored
//! value is the device's blend arithmetic.
//!
//! `Device::compile_render_pipeline` derives the map for a `Function` fragment stage from every
//! color attachment of the pipeline's descriptor ([`color_writes`]); the fragment function
//! records the entries of the attachments it writes (`StageIo::color_writes`), and link derives
//! them again from the pipeline it links into ([`check`]): a fragment compiled for another map,
//! a standalone one compiled without the descriptor included, is `NeedsInput(ColorWrites)`.

use std::collections::{BTreeMap, BTreeSet};

use crate::device::requirements::VkProperty;
use crate::device::DeviceProfile;
use crate::error::{DriverDeviation, Error, RuntimeInput};
use crate::library::FunctionArtifact;
use crate::metal;
use crate::metal::format::{color_output_components, write_conversion, WriteConversion};
use crate::metal::TextureWriteRoundingMode;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::graphics::ColorAttachmentState;
use crate::pipeline::lower_inputs::{
    ChannelWrite, ExtendedRangeColor, FactorLane, RotatedBlendForm,
};
use crate::pipeline::rotated_blend::{rotated_blend, rotated_form, RotatedBlend};
use crate::pipeline::stage_io::IoRole;

/// The color writes of a fragment function writing the attachments `written` (dual-source index
/// 0) of a pipeline whose color attachments are `colors` (alpha-to-coverage on where
/// `alpha_to_coverage`, its half-float channels rounding by `rounding`) on profile `p` (the
/// module docs): each channel's rule as the function's lowering on `p` tells it apart
/// ([`lowered`]), so pipelines whose rules differ only where the lowering does not compile the
/// function under one map. Where the device stores Metal's codes itself
/// (`VkProperty::MetalColorWrites`) the map keeps only the half-float channels
/// ([`metal_stored`]), whose `half` NaN the function's `float` output does not carry.
pub(crate) fn color_writes(
    written: impl IntoIterator<Item = u32>,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> BTreeMap<u32, [Option<ChannelWrite>; 4]> {
    written
        .into_iter()
        .filter_map(|attachment| {
            let rules = metal_rules(attachment, colors, rounding, p)?;
            Some((
                attachment,
                rules.map(|rule| rule.map(|rule| lowered(rule, p))),
            ))
        })
        .collect()
}

/// Whether attachment `attachment` of a pipeline whose color attachments are `colors`
/// (alpha-to-coverage on where `alpha_to_coverage`, rounding by `rounding`) on `p` takes a rule of
/// Metal's for some channel of the fragment function's output ([`metal_rules`]): its format
/// converts the output and the pipeline stores the output as written there (no blend, or one of
/// factors one and zero), or the profile keeps a half-float channel's rule. Link then gives a
/// blend of one and zero no Vulkan blend (`link::graphics`), whatever the function's map holds
/// once [`lowered`].
pub(crate) fn takes_metal_rule(
    attachment: u32,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> bool {
    metal_rules(attachment, colors, rounding, p)
        .is_some_and(|rules| rules.iter().flatten().any(|r| *r != ChannelWrite::Device))
}

/// Each channel's rule of Metal's for a fragment function's output to attachment `attachment`
/// of a pipeline whose color attachments are `colors` (alpha-to-coverage on where
/// `alpha_to_coverage`, rounding by `rounding`) on `p`, before [`lowered`]: `None` where the
/// attachment is absent, a word attachment (whose output is the word the function encodes,
/// `word_colors`), or takes no rule.
fn metal_rules(
    attachment: u32,
    colors: &[Option<ColorAttachmentDescriptor>],
    (alpha_to_coverage, rounding): (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> Option<[Option<ChannelWrite>; 4]> {
    let c = colors.get(attachment as usize).copied().flatten()?;
    if word_color(attachment, &c, alpha_to_coverage, p).is_some() {
        return None;
    }
    let coverage_alpha = attachment == 0 && alpha_to_coverage;
    let write = match metal_color_writes(p) {
        true => metal_stored(&c),
        false => attachment_write(&c, coverage_alpha),
    };
    match rounding {
        TextureWriteRoundingMode::NearestEven => write.map(|w| nearest_even(w, &c, coverage_alpha)),
        _ => write,
    }
}

/// `rule` as the fragment function's lowering on `p` tells it apart
/// (`lower::stage_io::color_write`): [`ChannelWrite::Device`] for a rule the device's conversion
/// stores whatever the output and however the function writes it, which records
/// `PrecisionNote::DeviceColorConversion` as `Device` does and emits the same words; the rule
/// itself where the lowering reads it. It reads a half-float channel's (whose `half` output is
/// exact, and whose NaN it notes), one rounded to nearest even or after a blend of other factors,
/// a small float's where the profile takes `DriverDeviation::MetalSmallFloatColorWrites`, and an
/// unblended 16-bit unorm or snorm one's where it takes `DriverDeviation::MetalNorm16ColorWrites`.
/// Every other rule (an 8- or 10-bit unorm or snorm, an sRGB, a blended 16-bit snorm, a small
/// float on a profile not measured to store it as Metal does) is the device's.
fn lowered(rule: ChannelWrite, p: &DeviceProfile) -> ChannelWrite {
    let takes = |d| p.takes_driver_deviation(d);
    match rule {
        _ if half_float(rule) => rule,
        ChannelWrite::FloatNearestEven { .. } | ChannelWrite::BlendedNearestEven { .. } => rule,
        ChannelWrite::FloatTowardZero { .. }
            if takes(DriverDeviation::MetalSmallFloatColorWrites) =>
        {
            rule
        }
        ChannelWrite::Unorm { bits: 16 }
        | ChannelWrite::Snorm {
            bits: 16,
            blended: false,
        } if takes(DriverDeviation::MetalNorm16ColorWrites) => rule,
        _ => ChannelWrite::Device,
    }
}

/// Whether profile `p`'s device stores Metal's codes itself (`VkProperty::MetalColorWrites`).
fn metal_color_writes(p: &DeviceProfile) -> bool {
    p.fields()
        .enabled
        .properties
        .contains(&VkProperty::MetalColorWrites)
}

/// The precision notes and the attachments it rounds in the shader of fragment function `a` in
/// a render pipeline whose color attachments are `colors` (`GraphicsState::color_attachments`):
/// the function's own (`FunctionArtifact::precision`, `StageIo::rounded_colors`),
/// `PrecisionNote::HalfOutputPrecision` where an attachment of `StageIo::half_blend_colors`
/// blends ([`blends_output`]), and `PrecisionNote::HalfOutputFastMathBlend` where one of
/// `StageIo::half_fast_colors` stores some channel ([`stores_output`]), and
/// `PrecisionNote::CoverageReadsRoundedAlpha` where the pipeline covers by alpha
/// (`alpha_to_coverage`, `GraphicsState::alpha_to_coverage`) and the function rounds that alpha to
/// nearest even ([`rounds_coverage_alpha`]), and `PrecisionNote::SrgbTexelDecode` where an
/// attachment of `StageIo::decoded_fetches` has an sRGB format: the device's sRGB decode of the
/// input attachment, where Metal decodes the bytes through its twelve-bit table. Each sorted, each
/// entry once.
pub(crate) fn pipeline_writes(
    a: &FunctionArtifact,
    colors: &[Option<ColorAttachmentState>],
    alpha_to_coverage: bool,
) -> (Vec<PrecisionNote>, Vec<u32>) {
    let mut notes = a.precision.clone();
    let rounded = a.interface.io.rounded_colors.clone();
    fn insert<T: Ord>(v: &mut Vec<T>, x: T) {
        if let Err(at) = v.binary_search(&x) {
            v.insert(at, x);
        }
    }
    if alpha_to_coverage && rounds_coverage_alpha(a) {
        insert(&mut notes, PrecisionNote::CoverageReadsRoundedAlpha);
    }
    let attachment = |c: &u32| {
        usize::try_from(*c)
            .ok()
            .and_then(|i| colors.get(i))
            .and_then(Option::as_ref)
    };
    let blended = |c: &u32| attachment(c).is_some_and(blends_output);
    if a.interface.io.half_blend_colors.iter().any(blended) {
        insert(&mut notes, PrecisionNote::HalfOutputPrecision);
    }
    // A `float` output of a module without fast math into a 16-bit float attachment's blend of one
    // and zero, which the pipeline blends ([`keeps_zero`]); its other blends record the device's
    // conversion already.
    let one_zero = |c: &u32| attachment(c).is_some_and(|s| blends_output(s) && one_zero_half(s));
    if a.interface.io.float_blend_colors.iter().any(one_zero) {
        insert(&mut notes, PrecisionNote::HalfOutputPrecision);
    }
    let stored = |c: &u32| attachment(c).is_some_and(stores_output);
    if a.interface.io.half_fast_colors.iter().any(stored) {
        insert(&mut notes, PrecisionNote::HalfOutputFastMathBlend);
    }
    let srgb = |c: &u32| {
        attachment(c).is_some_and(|s| {
            s.format
                .block()
                .is_some_and(|b| b.color == Some(crate::vk::NumericFormat::Srgb))
        })
    };
    if a.interface.io.decoded_fetches.iter().any(srgb) {
        insert(&mut notes, PrecisionNote::SrgbTexelDecode);
    }
    (notes, rounded)
}

/// Whether fragment function `a` rounds the alpha of its output to attachment 0 to nearest even in
/// the shader ([`HALF_NEAREST`], the attachment listed in `StageIo::rounded_colors`), which
/// alpha-to-coverage then reads where Metal's reads the alpha before rounding (the module docs).
fn rounds_coverage_alpha(a: &FunctionArtifact) -> bool {
    let io = &a.interface.io;
    io.rounded_colors.contains(&0)
        && io
            .color_writes
            .get(&0)
            .is_some_and(|rules| rules[3] == Some(HALF_NEAREST))
}

/// A half-float channel's rule ([`ChannelWrite::FloatTowardZero`] of ten fraction bits, signed).
pub(crate) const HALF: ChannelWrite = ChannelWrite::FloatTowardZero {
    mantissa: 10,
    signed: true,
};

/// A half-float channel's rule under `TextureWriteRoundingMode::NearestEven`
/// ([`ChannelWrite::FloatNearestEven`] of ten fraction bits, signed).
pub(crate) const HALF_NEAREST: ChannelWrite = ChannelWrite::FloatNearestEven {
    mantissa: 10,
    signed: true,
};

/// Whether `rule` is a half-float channel's ([`HALF`] or [`HALF_NEAREST`]).
pub(crate) fn half_float(rule: ChannelWrite) -> bool {
    rule == HALF || rule == HALF_NEAREST
}

/// A half-float channel's rule under `TextureWriteRoundingMode::NearestEven` after a blend of
/// factors other than one and zero ([`ChannelWrite::BlendedNearestEven`]).
pub(crate) const HALF_BLENDED: ChannelWrite = ChannelWrite::BlendedNearestEven {
    mantissa: 10,
    signed: true,
};

/// The channels `rules` of attachment `c` (alpha-to-coverage reading its alpha where
/// `coverage_alpha`) under `TextureWriteRoundingMode::NearestEven`: each channel the format rounds
/// as a half ([`HALF`], whatever the default map made of it) rounded to nearest even
/// ([`HALF_NEAREST`]) where it stores the output as written (no blend, or one of factors one and
/// zero), the alpha alpha-to-coverage reads included where the write mask stores it (the module
/// docs), and Metal's rounding of the blend's result ([`HALF_BLENDED`], which the device's blend
/// makes) after a blend of other factors. An alpha the coverage reads and the mask does not store
/// keeps its rule, the coverage reading it as computed.
fn nearest_even(
    mut rules: [Option<ChannelWrite>; 4],
    c: &ColorAttachmentDescriptor,
    coverage_alpha: bool,
) -> [Option<ChannelWrite>; 4] {
    let Some(WriteConversion::Rules(format)) = write_conversion(c.pixel_format) else {
        return rules;
    };
    let rounded = match !c.blending_enabled || replaces(c) {
        true => HALF_NEAREST,
        false => HALF_BLENDED,
    };
    let unstored_coverage = coverage_alpha && !c.write_mask.contains(metal::ColorWriteMask::ALPHA);
    for (i, (rule, own)) in rules.iter_mut().zip(format).enumerate() {
        if own == Some(HALF) && rule.is_some() && !(unstored_coverage && i == 3) {
            *rule = Some(rounded);
        }
    }
    rules
}

/// The channels of attachment `c` on a device that stores a float output as Metal does
/// (`VkProperty::MetalColorWrites`): its half-float channels, whatever its blend, and nothing
/// where it has none or writes none. The function rounds nothing there; a `half` output it writes
/// to such a channel is held widened to `float`, and the device stores that `float`'s NaN as Metal
/// stores a `float` NaN, not as the `half`'s own bits, so the function records
/// `PrecisionNote::DeviceNanEncoding` for it.
fn metal_stored(c: &ColorAttachmentDescriptor) -> Option<[Option<ChannelWrite>; 4]> {
    if c.write_mask.is_empty() {
        return None;
    }
    let WriteConversion::Rules(rules) = write_conversion(c.pixel_format)? else {
        return None;
    };
    let halves = rules.map(|r| r.filter(|&r| r == HALF));
    halves.iter().any(Option::is_some).then_some(halves)
}

/// Check the fragment stage `fragment` of a render pipeline whose color attachments are
/// `colors` (alpha-to-coverage on where `alpha_to_coverage`, rounding by `rounding`) on `p`: the
/// color writes it compiled under are the pipeline's for the attachments it writes
/// ([`writes_hold`]), and so are its extended-range attachments' writes
/// ([`wanted_extended_range`]), and every color it stores nowhere (`StageIo::unattached_colors`)
/// has no attachment and is not color 0 under alpha to coverage, and the second outputs it writes
/// for rotated attachments are the pipeline's ([`wanted_rotated_factors`]), and so are the forms of
/// their blends ([`wanted_rotated_blends`]), and the attachments holding Metal's alpha in red it
/// writes or reads are the pipeline's ([`wanted_alpha_in_red`]), and its texture writes
/// that Metal's mode 2 stores otherwise compiled under the pipeline's `textureWriteRoundingMode`
/// (`StageIo::texture_write_rounding`), else `NeedsInput(ColorWrites)`.
pub(crate) fn check(
    fragment: Option<&FunctionArtifact>,
    colors: &[Option<ColorAttachmentDescriptor>],
    (alpha_to_coverage, rounding): (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> Result<(), Error> {
    let Some(fragment) = fragment else {
        return Ok(());
    };
    let io = &fragment.interface.io;
    let unattached = io.unattached_colors.iter().all(|&c| {
        colors.get(c as usize).copied().flatten().is_none() && !(alpha_to_coverage && c == 0)
    });
    if writes_hold(fragment, colors, (alpha_to_coverage, rounding), p)
        && wanted_extended_range(fragment, colors, rounding) == io.extended_range_colors
        && wanted_word_colors(fragment, colors, alpha_to_coverage, p) == io.word_colors
        && wanted_rotated(fragment, colors, p) == io.rotated_colors
        && wanted_alpha_in_red(fragment, colors, p) == io.alpha_in_red_colors
        && wanted_rotated_factors(fragment, colors, p) == io.rotated_factors
        && wanted_rotated_blends(fragment, colors, p) == io.rotated_blends
        && io.texture_write_rounding.is_none_or(|m| m == rounding)
        && unattached
    {
        Ok(())
    } else {
        Err(Error::NeedsInput(RuntimeInput::ColorWrites))
    }
}

/// The extended-range color attachments of a render pipeline whose color attachments are
/// `colors` and whose `textureWriteRoundingMode` is `rounding`, by color index
/// (`LoweringInputs::extended_range_colors`): every attachment of one of Apple's extended-range
/// formats, blended where the pipeline enables blending there, a bfloat one rounding by the mode
/// (`ExtendedRangeColor::nearest_even`).
pub(crate) fn extended_range_colors(
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: TextureWriteRoundingMode,
) -> BTreeMap<u32, ExtendedRangeColor> {
    colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let c = c.as_ref()?;
            let form = metal::format::ExtendedRange::of(c.pixel_format)?;
            let write = ExtendedRangeColor::new(form, c.blending_enabled).with_rounding(rounding);
            Some((i as u32, write))
        })
        .collect()
}

/// Whether color attachment `c`, of an extended-range form that holds its three color codes in
/// one raw component (`BGR10_XR`, the two-plane forms' color word, and their `_sRGB` twins), is
/// written unblended under a mask of some of those codes: no mask of the raw components keeps one
/// code of a word and writes another, so the fragment function merges its codes into the word it
/// reads first, in the interlocked storage form (`lower::fragment::fetch`), where the device orders
/// fetch so. Over an input attachment (lavapipe) the merge is not built, and link refuses the mask
/// `NotImplemented(Format)`; on a device that orders fetch in neither form, it names the orderings
/// the merge needs (`lower::fragment::fetch::unordered_extended_fetch`).
pub(crate) fn merged_extended(c: &ColorAttachmentDescriptor) -> bool {
    use metal::format::ExtendedRange as X;
    use metal::ColorWriteMask as M;
    let word = matches!(
        X::of(c.pixel_format),
        Some(X::Bgr10 { .. } | X::Planes { .. })
    );
    let color = c.write_mask & M::RED.union(M::GREEN).union(M::BLUE);
    let rgb = M::RED.union(M::GREEN).union(M::BLUE);
    word && !c.blending_enabled && !color.is_empty() && color != rgb
}

/// The word view through which profile `p`'s device renders color attachment `index` of a
/// render pipeline, described by `c`, that covers by alpha where `alpha_to_coverage`
/// (`LoweringInputs::word_colors`): `metal::format::WordAttachment::of` the attachment's format,
/// where the pipeline writes it unblended (or blended with factors one and zero, added, which
/// Metal stores alike), under a write mask of all of the format's channels or none, or of some of
/// them where the function merges the word into the texel over its input attachment
/// ([`merged_word`]), and it is not attachment 0 under alpha-to-coverage (whose coverage reads the
/// output's alpha as computed). `None` for every other attachment: one the function stores in the
/// interlocked storage form ([`stored_word`]), or one that takes its format's own attachment, as
/// the device has it or not.
pub(crate) fn word_color(
    index: u32,
    c: &ColorAttachmentDescriptor,
    alpha_to_coverage: bool,
    p: &DeviceProfile,
) -> Option<metal::format::WordAttachment> {
    let word = metal::format::WordAttachment::of(c.pixel_format, p)?;
    let kept = c.write_mask & word_channels(word);
    let whole = kept.is_empty() || kept == word_channels(word);
    let merged = merges_over_input(c, word, p);
    let unblended = !c.blending_enabled || replaces(c);
    (unblended && (whole || merged) && !(index == 0 && alpha_to_coverage)).then_some(word)
}

/// The channels of `word`'s fields: Metal's red, green, blue and alpha, in its field order.
fn word_channels(word: metal::format::WordAttachment) -> metal::ColorWriteMask {
    use metal::ColorWriteMask as M;
    [M::RED, M::GREEN, M::BLUE, M::ALPHA]
        .into_iter()
        .zip(word.fields())
        .filter(|(_, f)| f.is_some())
        .fold(M::NONE, |m, (bit, _)| m | bit)
}

/// Whether the fragment function merges its word of attachment `c` (a `word` attachment) into the
/// texel over the attachment's input attachment: a write mask of some of the word's channels, on a
/// device that orders framebuffer fetch through rasterization-order input attachments
/// (`lower::fragment`'s `word_input`), for a format whose mask the function compiles as the
/// descriptor sets it (`lower::fragment::fetch::kept_at_draw` false: the 8-bit sRGB words, where a
/// packed 16-bit or `RGB9E5Float` word's mask stands for its class).
fn merges_over_input(
    c: &ColorAttachmentDescriptor,
    word: metal::format::WordAttachment,
    p: &DeviceProfile,
) -> bool {
    let channels = word_channels(word);
    let kept = c.write_mask & channels;
    let partial = !kept.is_empty() && kept != channels;
    partial
        && !crate::lower::fragment::fetch::kept_at_draw(c.pixel_format)
        && crate::lower::fragment::inputs_order(p)
}

/// The channels the fragment function keeps from the texel of color attachment `index` of a
/// render pipeline, described by `c`, that covers by alpha where `alpha_to_coverage`, where it
/// merges its word into the texel over the attachment's input attachment ([`word_color`],
/// `lower::fragment`'s `word_input`): those of the word's channels the write mask leaves out.
/// `None` for every other attachment.
pub(crate) fn merged_word(
    index: u32,
    c: &ColorAttachmentDescriptor,
    alpha_to_coverage: bool,
    p: &DeviceProfile,
) -> Option<metal::ColorWriteMask> {
    let word = word_color(index, c, alpha_to_coverage, p)?;
    merges_over_input(c, word, p).then(|| word_channels(word).difference(c.write_mask))
}

/// The word view of color attachment `c` of a render pipeline on profile `p` whose writes the
/// fragment function stores in the interlocked storage form (`lower::fragment::fetch`), reading
/// the word first inside the interlock: a packed 16-bit word attachment
/// (`metal::format::WordAttachment::of`) written under a blend other than one and zero, added,
/// whose factors read no second source (Metal's packed 16-bit blend, which the function computes
/// over the word, `extended_range::Blend::packed16`), or a word attachment other than
/// `RGB9E5Float`'s written unblended under a write mask of some of the format's channels (the
/// function merges the fields of the channels the mask keeps), on a device that orders fetch
/// through the pixel interlock alone.
/// `None` for every other attachment, and for a write mask of none of the format's channels.
pub(crate) fn stored_word(
    c: &ColorAttachmentDescriptor,
    p: &DeviceProfile,
) -> Option<metal::format::WordAttachment> {
    use metal::{BlendFactor as F, ColorWriteMask as M};
    let word = metal::format::WordAttachment::of(c.pixel_format, p)?;
    let channels = [M::RED, M::GREEN, M::BLUE, M::ALPHA]
        .into_iter()
        .zip(word.fields())
        .filter(|(_, f)| f.is_some())
        .fold(M::NONE, |m, (bit, _)| m | bit);
    let kept = c.write_mask & channels;
    let partial = !kept.is_empty() && kept != channels;
    let unblended = !c.blending_enabled || replaces(c);
    let second = [
        c.source_rgb_blend_factor,
        c.destination_rgb_blend_factor,
        c.source_alpha_blend_factor,
        c.destination_alpha_blend_factor,
    ]
    .iter()
    .any(|f| {
        matches!(
            f,
            F::Source1Color | F::OneMinusSource1Color | F::Source1Alpha | F::OneMinusSource1Alpha
        )
    });
    // Metal's blend of a packed 16-bit word is built; an 8-bit sRGB word's is not, nor is an
    // `RGB9E5Float` word's blend or partial mask (its mantissas share the exponent).
    let packed = matches!(word, metal::format::WordAttachment::Packed16 { .. });
    let fielded = word != metal::format::WordAttachment::Rgb9e5;
    let stored = match unblended {
        true => partial && fielded,
        false => packed && !kept.is_empty() && !second,
    };
    (stored && crate::lower::fragment::interlock_orders(p)).then_some(word)
}

/// The word attachments of a render pipeline whose color attachments are `colors` (covering by
/// alpha where `alpha_to_coverage`) on `p`, by color index ([`word_color`]).
pub(crate) fn word_colors(
    colors: &[Option<ColorAttachmentDescriptor>],
    alpha_to_coverage: bool,
    p: &DeviceProfile,
) -> BTreeMap<u32, metal::format::WordAttachment> {
    colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let word = word_color(i as u32, c.as_ref()?, alpha_to_coverage, p)?;
            Some((i as u32, word))
        })
        .collect()
}

/// The word attachments the fragment stage `fragment` writes through output variables of
/// dual-source index 0 in a render pipeline whose color attachments are `colors` (covering by
/// alpha where `alpha_to_coverage`) on `p`: [`word_colors`]'s, for those attachments.
pub(crate) fn wanted_word_colors(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    alpha_to_coverage: bool,
    p: &DeviceProfile,
) -> BTreeMap<u32, metal::format::WordAttachment> {
    let written: BTreeSet<u32> = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        })
        .collect();
    let mut out = word_colors(colors, alpha_to_coverage, p);
    out.retain(|a, _| written.contains(a));
    out
}

/// The color attachments of a render pipeline whose color attachments are `colors` held with
/// their channels rotated on `p`, by color index (`LoweringInputs::rotated_colors`): every
/// attachment of a format `metal::format::rotated_channels` names (`ARGB8Unorm`), and every one
/// whose image holds Metal's alpha in red (`metal::format::alpha_in_red`: `A8Unorm` without
/// `A8_UNORM`), which the rotation's red is.
pub(crate) fn rotated_colors(
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    colors
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.is_some_and(|c| {
                metal::format::rotated_channels(c.pixel_format)
                    || metal::format::alpha_in_red(c.pixel_format, p)
            })
        })
        .map(|(i, _)| i as u32)
        .collect()
}

/// The rotated attachments the fragment stage `fragment` writes in a render pipeline whose
/// color attachments are `colors`: [`rotated_colors`]'s, for the attachments it writes through
/// output variables of either dual-source index.
pub(crate) fn wanted_rotated(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    let written: BTreeSet<u32> = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color { attachment, .. } => Some(attachment),
            _ => None,
        })
        .collect();
    let mut out = rotated_colors(colors, p);
    out.retain(|a| written.contains(a));
    out
}

/// The color attachments of a render pipeline whose color attachments are `colors` whose image
/// holds Metal's alpha in red on `p` (`metal::format::alpha_in_red`: `A8Unorm` without
/// `A8_UNORM`), by color index (`LoweringInputs::alpha_in_red_colors`).
pub(crate) fn alpha_in_red_colors(
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    colors
        .iter()
        .enumerate()
        .filter(|(_, c)| c.is_some_and(|c| metal::format::alpha_in_red(c.pixel_format, p)))
        .map(|(i, _)| i as u32)
        .collect()
}

/// The attachments holding Metal's alpha in red that the fragment stage `fragment` writes or reads
/// in a render pipeline whose color attachments are `colors` on `p`: [`alpha_in_red_colors`]'s,
/// for the attachments it writes through output variables of either dual-source index, reads
/// through an input attachment, or reads in the interlocked storage form
/// (`RuntimeFact::InterlockedFramebufferFetch`), the set it must have compiled under
/// (`StageIo::alpha_in_red_colors`).
pub(crate) fn wanted_alpha_in_red(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    let written = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color { attachment, .. } => Some(attachment),
            _ => None,
        });
    let inputs = fragment
        .interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            crate::pipeline::sym::ResourceKind::InputAttachment(color) => Some(color),
            _ => None,
        });
    let interlocked = fragment.facts.iter().filter_map(|f| match *f {
        RuntimeFact::InterlockedFramebufferFetch { color, .. } => Some(color),
        _ => None,
    });
    let used: BTreeSet<u32> = written.chain(inputs).chain(interlocked).collect();
    let mut out = alpha_in_red_colors(colors, p);
    out.retain(|a| used.contains(a));
    out
}

/// The second outputs of the attachments of `colors` held with their channels rotated, on `p`
/// (`LoweringInputs::rotated_factors`): every such attachment whose blend Vulkan's factors read
/// from a second output the rotation needs (`rotated_blend::RotatedBlend::Synthesized`). An
/// attachment whose blend no Vulkan state gives has none; link refuses its pipeline.
pub(crate) fn rotated_factors(
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeMap<u32, [FactorLane; 4]> {
    colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let c = c.as_ref()?;
            if !metal::format::rotated_channels(c.pixel_format) {
                return None;
            }
            match rotated_blend(c, p) {
                Ok(Some(RotatedBlend::Synthesized(_, lanes))) => Some((i as u32, lanes)),
                _ => None,
            }
        })
        .collect()
}

/// The second outputs the fragment stage `fragment` writes for rotated attachments in a render
/// pipeline whose color attachments are `colors` on `p`: [`rotated_factors`]'s, for the
/// attachments it writes through a first (dual-source index 0) output.
pub(crate) fn wanted_rotated_factors(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeMap<u32, [FactorLane; 4]> {
    let written: BTreeSet<u32> = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        })
        .collect();
    let mut out = rotated_factors(colors, p);
    out.retain(|a, _| written.contains(a));
    out
}

/// The forms of the blends of the attachments of `colors` held with their channels rotated, on
/// `p` (`LoweringInputs::rotated_blends`): every such attachment whose blend the fragment
/// function shapes or announces (`rotated_blend::rotated_form`).
pub(crate) fn rotated_blends(
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeMap<u32, RotatedBlendForm> {
    colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let c = c.as_ref()?;
            if !metal::format::rotated_channels(c.pixel_format) {
                return None;
            }
            Some((i as u32, rotated_form(c, p)?))
        })
        .collect()
}

/// The forms of the blends of rotated attachments the fragment stage `fragment` compiles under in
/// a render pipeline whose color attachments are `colors` on `p`: [`rotated_blends`]'s, for the
/// attachments it writes through a first (dual-source index 0) output.
pub(crate) fn wanted_rotated_blends(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeMap<u32, RotatedBlendForm> {
    let written: BTreeSet<u32> = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        })
        .collect();
    let mut out = rotated_blends(colors, p);
    out.retain(|a, _| written.contains(a));
    out
}

/// The extended-range writes and reads the fragment stage `fragment` compiles under in a render
/// pipeline whose color attachments are `colors` and whose `textureWriteRoundingMode` is
/// `rounding`: [`extended_range_colors`]'s, for the attachments it writes through output
/// variables of dual-source index 0 or reads through input attachments.
pub(crate) fn wanted_extended_range(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: TextureWriteRoundingMode,
) -> BTreeMap<u32, ExtendedRangeColor> {
    let interface = &fragment.interface;
    let written: std::collections::BTreeSet<u32> = interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        })
        .chain(
            interface
                .descriptors
                .iter()
                .filter_map(|d| match d.slot.kind {
                    crate::pipeline::sym::ResourceKind::InputAttachment(color) => Some(color),
                    _ => None,
                }),
        )
        .collect();
    let mut out = extended_range_colors(colors, rounding);
    out.retain(|a, _| written.contains(a));
    out
}

/// The color writes the fragment stage `fragment` compiles under in a render pipeline whose
/// color attachments are `colors` (alpha-to-coverage on where `alpha_to_coverage`, rounding by
/// `rounding`) on `p`: the pipeline's, for the attachments it writes through output variables of
/// dual-source index 0 or in the storage form ([`written`]).
pub(crate) fn wanted(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> BTreeMap<u32, [Option<ChannelWrite>; 4]> {
    color_writes(written(fragment), colors, rounding, p)
}

/// The color attachments the fragment stage `fragment` writes through output variables of
/// dual-source index 0 or in the storage form.
fn written(fragment: &FunctionArtifact) -> impl Iterator<Item = u32> + '_ {
    let outputs = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        });
    let storage = fragment.facts.iter().filter_map(|f| match *f {
        RuntimeFact::InterlockedFramebufferFetch {
            color,
            writes: true,
            ..
        } => Some(color),
        _ => None,
    });
    outputs.chain(storage)
}

/// Whether the color writes the fragment stage `fragment` compiled under
/// (`StageIo::color_writes`) are the pipeline's ([`wanted`]): the same entry for each attachment,
/// and none for another.
fn writes_hold(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> bool {
    wanted(fragment, colors, rounding, p) == fragment.interface.io.color_writes
}

/// How attachment `c` takes each channel of a float output; `coverage_alpha` where
/// alpha-to-coverage reads its alpha. `None` where it stores the value as written or nothing at
/// all.
fn attachment_write(
    c: &ColorAttachmentDescriptor,
    coverage_alpha: bool,
) -> Option<[Option<ChannelWrite>; 4]> {
    if c.write_mask.is_empty() {
        return None;
    }
    let device = |channels: u8| {
        let mut out = [None; 4];
        for channel in out.iter_mut().take(usize::from(channels)) {
            *channel = Some(ChannelWrite::Device);
        }
        out
    };
    let replaced = !c.blending_enabled || replaces(c);
    let mut rules = match write_conversion(c.pixel_format)? {
        // An extended-range attachment's texels are the function's own encoding.
        WriteConversion::Exact | WriteConversion::Encoded => return None,
        // A 32-bit float channel stores the output as written; a blend of it is the device's.
        WriteConversion::Float if replaced => return None,
        WriteConversion::Float | WriteConversion::Unmeasured => {
            return color_output_components(c.pixel_format).map(device);
        }
        WriteConversion::Rules(rules) => rules,
    };
    for rule in rules.iter_mut().flatten() {
        match rule {
            _ if !replaced => *rule = ChannelWrite::Device,
            ChannelWrite::Snorm { blended, .. } => *blended = c.blending_enabled,
            _ => {}
        }
    }
    if coverage_alpha && rules[3].is_some() {
        rules[3] = Some(ChannelWrite::Device);
    }
    Some(rules)
}

/// Whether the pipeline's attachment `c` blends the fragment function's output: its blend is
/// enabled and it writes some channel. Where it does not, a `half` output of a module without
/// fast math stores Metal's value (`StageIo::half_blend_colors`). Factors one and zero, added,
/// blend too: lavapipe (Mesa 26.2.4) stores the source through the destination there, one
/// `float` unit in the last place at the destination's magnitude off it in up to 11% of a
/// 32-bit float attachment's channels.
pub(crate) fn blends_output(c: &ColorAttachmentState) -> bool {
    c.blend.is_some() && !c.write_mask.is_empty()
}

/// Whether the pipeline's attachment `c` stores some channel of the fragment function's output,
/// blended or not: its write mask names one of its format's color components (a mask of blue
/// alone stores nothing into a one-channel format). Where it stores none, a `half` output of a
/// module with fast math leaves nothing Metal's compiler could fold otherwise
/// (`StageIo::half_fast_colors`).
pub(crate) fn stores_output(c: &ColorAttachmentState) -> bool {
    let components = c
        .format
        .block()
        .map_or(crate::vk::ColorComponentFlags::known_bits(), |b| {
            b.color_components
        });
    c.write_mask.intersects(components)
}

/// Whether a 16-bit float attachment `c` at `color` blends the output of the fragment stage of
/// interface `io` by one and zero though [`replaces`] holds of it: an output of a module without
/// fast math (`StageIo::half_blend_colors`, `StageIo::float_blend_colors`), whose blend there
/// Metal computes keeping the destination's product by zero (a NaN of an infinite or NaN texel,
/// `+0` of a `-0` output over a texel of `+0` or above), which no write of the
/// output alone stores. The pipeline blends it and announces `PrecisionNote::HalfOutputPrecision`;
/// the storage form computes Metal's (`lower::fragment::fetch`).
pub(crate) fn keeps_zero(
    io: &crate::pipeline::stage_io::StageIo,
    color: u32,
    c: &ColorAttachmentDescriptor,
) -> bool {
    crate::lower::fragment::half_float_format(c.pixel_format)
        && (io.half_blend_colors.contains(&color) || io.float_blend_colors.contains(&color))
}

/// Whether the linked attachment `s` is a 16-bit float one blended by one and zero, added, for
/// color and alpha ([`keeps_zero`]'s).
fn one_zero_half(s: &ColorAttachmentState) -> bool {
    use crate::vk::{BlendFactor as F, BlendOp as O, Format};
    let half = matches!(
        s.format,
        Format::R16_SFLOAT | Format::R16G16_SFLOAT | Format::R16G16B16A16_SFLOAT
    );
    half && s.blend.as_ref().is_some_and(|b| {
        (b.src_color, b.dst_color, b.color_op) == (F::ONE, F::ZERO, O::ADD)
            && (b.src_alpha, b.dst_alpha, b.alpha_op) == (F::ONE, F::ZERO, O::ADD)
    })
}

/// Whether `c`'s blend stores the source as it is: factors one and zero, added, for color and
/// alpha.
pub(crate) fn replaces(c: &ColorAttachmentDescriptor) -> bool {
    use metal::{BlendFactor as F, BlendOperation as O};
    (
        c.source_rgb_blend_factor,
        c.destination_rgb_blend_factor,
        c.rgb_blend_operation,
    ) == (F::One, F::Zero, O::Add)
        && (
            c.source_alpha_blend_factor,
            c.destination_alpha_blend_factor,
            c.alpha_blend_operation,
        ) == (F::One, F::Zero, O::Add)
}

#[cfg(test)]
mod tests;
