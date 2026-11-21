//! Framebuffer fetch: what orders it, and the writes a fragment function computes over the texel
//! it reads.
//!
//! **Where.** Metal orders a fetch after every earlier fragment of the pixel, across draws. A color
//! input reads the attachment's input attachment (`color`), which Vulkan orders so only under
//! rasterization-order color attachment access ([`inputs_order`]). A device without that access
//! reads the input attachment ordered between draws alone, by the consumer's barrier of the
//! subpass's dependency on itself, and the read notes `PrecisionNote::UnorderedFetchWithinDraw`
//! ([`fetch_unordered`]); under the identity domains it reads the attachment's pushed texture at
//! the invocation's pixel instead (`PushField::FetchedColor`, [`pixel`]). The pixel interlock
//! orders no fetch: it serves raster order groups on memory the function itself reads and writes
//! (`fragment::rog`).
//!
//! **Writes computed over the input attachment.** On a device that orders fetch through
//! rasterization-order input attachments, the function computes over the texel it reads the writes
//! no Vulkan blend or write mask gives, from the pipeline's state (`LoweringInputs::framebuffer_fetch`,
//! which the render pipeline compile sets, `compile::fetch_inputs`): a padded extended-range
//! attachment's blend, and a write of some of the codes one word of such an attachment holds
//! (`input_blend`, [`extended_range_blend`]); an attachment held with its channels rotated whose
//! blend no Vulkan state on its image gives (`input_blend`'s `RotatedInputBlend`); a word
//! attachment written under a mask of some of the word's channels (`word_input`); and a private
//! logic operation the device lacks (`logic`). On a device that does not order fetch so, each is
//! refused naming rasterization-order access ([`unordered_extended_fetch`]).
//!
//! **Per sample**. Metal runs a function that reads a color attachment of a multisampled pass once
//! per covered sample, reading its own sample, in draw order per sample, with `[[position]]`'s x
//! and y at the pixel center (measured). Compiled for such a pipeline
//! (`LoweringInputs::multisampled`), the function reads the attachment's sample at its `SampleId`
//! ([`plan_sample`]), which runs it once per covered sample under Vulkan's sample shading, and needs
//! `sampleRateShading`.
//!
//! **A write mask** short of the format's channels: Metal keeps the channels the mask leaves out.
//! A write the function computes over its input attachment keeps them from the texel it read
//! ([`kept_channels`]); a mask that covers none of the format's channels writes nothing. The
//! function compiles a format's mask by its class ([`compiled_write_mask`]: none, every channel, or
//! some), but an extended-range or an sRGB format's mask as the descriptor sets it
//! ([`kept_at_draw`]); link holds the channels the stage keeps to the pipeline's own mask.
//!
//! **Apple's extended-range forms** (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their
//! `_sRGB` twins, [`codes_round_the_tile`]): Metal's tile memory holds such an attachment as `half`
//! through a render pass and rounds it to the codes only at the store, so a fetch or a blend onto
//! what an earlier draw of the pass wrote sees that draw's `half`, which the codes the attachment
//! holds do not keep. A fetch of one, and a blend into one the function writes, are not
//! implemented: refused on every device (`Feature::Format`, at the attachment; here, in
//! `color::select_param`, and in `stage_io::extended_color` where the device does not order
//! fetch). An unblended write, a masked one included, is built: a kept channel's code is the
//! encode of the `half` the tile keeps there. `metal2vulkan-runtime` makes no extended-range
//! texture (`Gap::TextureFormat`).
//!
//! A blocked attachment is refused with the blocker's reason (what the device lacks, or
//! `NotImplemented`): no ordered path exists there.

use std::collections::BTreeSet;

use crate::air::abi::{AbiInt, OutputRole};
use crate::air::ir::TypeId as AirTypeId;
use crate::device::requirements::{FormatUsage, VkFeature};
use crate::device::DeviceProfile;
use crate::error::{Attachment, Capability, Error, ExtendedRangeBlend, Feature, Unsupported};
use crate::lower::arith::{extended_range, FloatFmt, Form};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io;
use crate::metal::format::{color_output_components, ExtendedRange, WordAttachment};
use crate::metal::ColorWriteMask;
use crate::pipeline::lower_inputs::BlendEquation;
use crate::pipeline::sym::PushField;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{ConvOp, ImageOperands, MemAccess, Type as SType};
use crate::vk;

/// Whether `p` orders framebuffer fetch through rasterization-order input attachments (lavapipe):
/// rasterization-order color attachment access.
pub(crate) fn inputs_order(p: &DeviceProfile) -> bool {
    p.fields().fragment.rasterization_order_color_access
}

/// Whether `p` orders framebuffer fetch in no form: no rasterization-order color attachment
/// access. A fetch there reads the input attachment, which
/// the consumer's barrier of the subpass's dependency on itself orders after the earlier draws'
/// writes but not after an overlapping earlier fragment of the same draw
/// (`PrecisionNote::UnorderedFetchWithinDraw`, `color`); a blend or a merge the function computes
/// over its own read ([`unordered_extended_fetch`]) is refused.
pub(crate) fn fetch_unordered(p: &DeviceProfile) -> bool {
    !inputs_order(p)
}

/// Whether Metal's tile memory holds an attachment of `form` as `half` through a render pass and
/// rounds it to the codes only at the store, where the attachment holds the codes: Apple's
/// extended-range forms (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their `_sRGB` twins), not
/// a padded or bfloat one. A fetch of such an attachment, and a blend into one, are not
/// implemented (`Feature::Format`, module docs).
pub(crate) fn codes_round_the_tile(form: ExtendedRange) -> bool {
    !form.padded() && !form.bfloat()
}

/// The refusal of a blend into an extended-range attachment, or of a write of some of the codes
/// one of its words holds (`color_writes::merged_extended`), on a device that does not order
/// framebuffer fetch ([`fetch_unordered`]): the function computes the blend, or merges the codes,
/// over its own read of the attachment's codes, which no device's blend unit or write mask does,
/// and that read must be ordered after every earlier fragment's write of the pixel, through
/// rasterization-order color attachment access (over the input attachment).
pub(crate) fn unordered_extended_fetch() -> Unsupported {
    Unsupported::needs(
        Feature::FramebufferFetch,
        Capability::Feature(VkFeature::RasterizationOrderColorAttachmentAccess),
    )
}

/// Whether `pf` is a 16-bit float color format, whose blend of a `half` output of a module
/// without fast math Metal computes in `half` arithmetic, and of one with fast math in the fast
/// forms it folds into the function ([`extended_range::blend::BlendMath`]).
pub(crate) fn half_float_format(pf: crate::metal::PixelFormat) -> bool {
    use crate::metal::PixelFormat as P;
    matches!(pf, P::R16Float | P::RG16Float | P::RGBA16Float)
}

/// Whether `pf` is an 8-bit unorm color format, whose blend Metal computes in the arithmetic of
/// the output's type over the clamped output ([`extended_range::Blend::unorm8`]): one held
/// directly, or `ARGB8Unorm`, which Metal blends as `RGBA8Unorm` with its channels rotated.
pub(crate) fn unorm8_format(pf: crate::metal::PixelFormat) -> bool {
    use crate::metal::PixelFormat as P;
    matches!(
        pf,
        P::R8Unorm | P::RG8Unorm | P::RGBA8Unorm | P::BGRA8Unorm | P::ARGB8Unorm
    )
}

/// Whether `pf` is a packed 16-bit unorm format (`metal::format::WordAttachment::Packed16`), whose
/// blend Metal computes in `half` over its decoded codes ([`extended_range::Blend::packed16`]).
pub(crate) fn packed16_format(pf: crate::metal::PixelFormat) -> bool {
    matches!(
        WordAttachment::layout(pf),
        Some(WordAttachment::Packed16 { .. })
    )
}

/// The channels of `mask` a write of every channel of a color attachment of `pf` must keep: its
/// first `color_output_components(pf)` of red, green, blue, and alpha (all four for a format of
/// none).
fn channels(pf: crate::metal::PixelFormat) -> ColorWriteMask {
    let n = color_output_components(pf).unwrap_or(4);
    let order = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
        ColorWriteMask::ALPHA,
    ];
    order
        .iter()
        .take(usize::from(n))
        .fold(ColorWriteMask::NONE, |m, &c| m.union(c))
}

/// Whether the write masks of a class compile alike for an attachment of `pf`
/// ([`compiled_write_mask`]): every format but an extended-range one, whose kept channels Metal's
/// blend model and encode read (`extended_range::Blend`, `encode_keeping`), and an sRGB one.
pub(crate) fn kept_at_draw(pf: crate::metal::PixelFormat) -> bool {
    let srgb = crate::metal::format::direct_format(pf)
        .and_then(|f| f.block())
        .is_some_and(|b| b.color == Some(vk::NumericFormat::Srgb));
    ExtendedRange::of(pf).is_none() && !srgb
}

/// The channels a write of the attachment of `pf` under write mask `mask` keeps from its texel:
/// none where the function does not write it (`writes` false) or the mask covers every channel of
/// the format, all four (`R`, `G`, `B` and `A`) where it covers none of them (the write stores
/// nothing), else those of the format the mask leaves out.
pub(crate) fn kept_channels(
    pf: crate::metal::PixelFormat,
    mask: ColorWriteMask,
    writes: bool,
) -> ColorWriteMask {
    let all = channels(pf);
    if !writes {
        ColorWriteMask::NONE
    } else if all.intersection(mask).is_empty() {
        ColorWriteMask::ALL
    } else {
        all.difference(mask)
    }
}

/// The write mask a fragment function compiles its computed write of an attachment of `pf` under,
/// for the descriptor's `mask` (`FramebufferFetchInputs::of`), blended where `blended`: where the
/// masks of a class compile alike ([`kept_at_draw`]), the class of `mask` alone (none
/// where it covers none of the format's channels, all where it covers every one, and for a mask
/// covering some, every channel of the format but the first, the one mask that stands for them
/// all), so the write masks of a class share one key and one module; any other format's `mask` as
/// it is. A blended 8-bit unorm attachment with alpha, or a blended packed 16-bit one, splits the
/// partial class in three, as Metal's blend forms do (`extended_range::Blend::unorm8`,
/// `extended_range::Blend::packed16`): alpha alone (`ALPHA`), one color channel with or without
/// alpha (`RED` and `ALPHA`), and two or more color channels.
pub(crate) fn compiled_write_mask(
    pf: crate::metal::PixelFormat,
    mask: ColorWriteMask,
    blended: bool,
) -> ColorWriteMask {
    if !kept_at_draw(pf) || mask.contains(ColorWriteMask::UNSPECIALIZED) {
        return mask;
    }
    let all = channels(pf);
    let written = all.intersection(mask);
    let colors = [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
    ]
    .into_iter()
    .filter(|&c| written.contains(c))
    .count();
    // Metal's blend forms differ by the class (`extended_range::Blend::unorm8`, `packed16`).
    let unorm8 = blended
        && ((unorm8_format(pf) && all.contains(ColorWriteMask::ALPHA)) || packed16_format(pf));
    if written.is_empty() {
        ColorWriteMask::NONE
    } else if written == all {
        ColorWriteMask::ALL
    } else if unorm8 && colors == 0 {
        ColorWriteMask::ALPHA
    } else if unorm8 && colors == 1 {
        ColorWriteMask::RED.union(ColorWriteMask::ALPHA)
    } else {
        all.difference(ColorWriteMask::RED)
    }
}

/// Whether a render pipeline on `p` blends into a color attachment of `pf`, one of Apple's
/// extended-range formats (`metal::format::extended_range_blend`, its public door): the decision
/// `input_blend` makes for such an attachment, written whole by the function's
/// first output under a blend Metal's model holds (premultiplied source-over, one), in a pipeline
/// of one sample with no depth or stencil attachment. One of Apple's extended-range forms
/// ([`codes_round_the_tile`]) answers `NotImplemented(Format)` on every device. Another's: `Ok`
/// where the device orders fetch through rasterization-order input attachments and the function's
/// blend over the attachment's input attachment holds (`input_blend`); else what blocks it
/// ([`unordered_extended_fetch`]). A bfloat format, which
/// Metal blends nowhere (its pipeline validation asserts; link refuses such a blend
/// `InvalidDescriptor(ColorAttachment)`), answers `NotRepresentable(Format)`, as the runtime's
/// formats answer what Metal refuses.
pub(crate) fn extended_range_blend(
    pf: crate::metal::PixelFormat,
    p: &DeviceProfile,
) -> Result<(), Unsupported> {
    use crate::metal::{BlendFactor as F, BlendOperation as O};
    let Some(form) = ExtendedRange::of(pf) else {
        return Err(Unsupported::not_implemented(Feature::Format));
    };
    if form.bfloat() {
        return Err(Unsupported::not_representable(Feature::Format));
    }
    if codes_round_the_tile(form) {
        return Err(Unsupported::not_implemented(Feature::Format));
    }
    let equation = BlendEquation {
        source_rgb: F::One,
        destination_rgb: F::OneMinusSourceAlpha,
        rgb_operation: O::Add,
        source_alpha: F::One,
        destination_alpha: F::OneMinusSourceAlpha,
        alpha_operation: O::Add,
    };
    if fetch_unordered(p) {
        return Err(unordered_extended_fetch());
    }
    // The blend over the input attachment: the attachment's raw format a color attachment, and
    // the blend Metal's model's.
    let usages = BTreeSet::from([FormatUsage::ColorAttachment]);
    match crate::metal::format::vk_format(pf, &usages, p) {
        Ok(_) => {}
        Err(Error::Unsupported(gap)) => return Err(gap),
        Err(_) => return Err(Unsupported::not_representable(Feature::Format)),
    }
    match extended_range::blend::NativeBlend::of(&equation, channels(pf)) {
        Ok(_) => Ok(()),
        Err(Error::Unsupported(gap)) => Err(gap),
        Err(_) => Err(Unsupported::not_representable(Feature::Format)),
    }
}

/// The extended-range attachments the function writes through a first output where the
/// pipeline blends them (`LoweringInputs::extended_range_colors`), whose blend the function
/// computes over the input attachment (`input_blend`), and the AIR type of each output. A blend into one of Apple's extended-range forms
/// ([`codes_round_the_tile`]) is refused (`Feature::Format`, at the attachment).
pub(super) fn blended_extended_range(cx: &SelectCx<'_>) -> Result<Vec<(u32, AirTypeId)>, Error> {
    let mut out = Vec::new();
    for (o, ty) in cx
        .entry
        .ep
        .outputs
        .iter()
        .zip(stage_io::output_types(cx)?.1)
    {
        if let OutputRole::RenderTarget {
            attachment: AbiInt::Const(color),
            index: 0,
            ..
        } = o.role
        {
            match cx.inputs.extended_range_colors.get(&color) {
                Some(e) if e.blended && codes_round_the_tile(e.form) => {
                    let refused = Unsupported::not_implemented(Feature::Format);
                    return Err(refused.at_attachment(Attachment::Color(color)).into());
                }
                Some(e) if e.blended => out.push((color, ty)),
                _ => {}
            }
        }
    }
    Ok(out)
}

/// Plan the invocation's sample for a function compiled for a multisampled pipeline
/// (`LoweringInputs::multisampled`), which reads each color it fetches, in either form, at its
/// `SampleId`: Metal runs a function that reads a color attachment once per covered sample, reading
/// its own sample, and a `SampleId` input runs a Vulkan fragment shader once per covered sample.
/// Nothing for one sample.
pub(super) fn plan_sample(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if !cx.inputs.multisampled || cx.families.fragment.fetch_sample.is_some() {
        return Ok(());
    }
    if !cx.profile.fields().fragment.sample_rate_shading {
        return Err(Unsupported::needs(
            Feature::FramebufferFetch,
            Capability::Feature(VkFeature::SampleRateShading),
        )
        .into());
    }
    cx.families.fragment.fetch_sample = Some(stage_io::sample_id(cx));
    Ok(())
}

/// The function's second (index-1) output to attachment `color`: `None` where it writes none,
/// `Some(Some(fmt))` where it is four lanes of `fmt`, a `half` or a `float`, and `Some(None)` for
/// another.
pub(super) fn second_output4(
    cx: &SelectCx<'_>,
    color: u32,
) -> Result<Option<Option<FloatFmt>>, Error> {
    for (o, ty) in cx
        .entry
        .ep
        .outputs
        .iter()
        .zip(stage_io::output_types(cx)?.1)
    {
        if let OutputRole::RenderTarget {
            attachment: AbiInt::Const(c),
            index: 1,
            ..
        } = o.role
        {
            if c == color {
                let f = super::color::form(cx, ty)?;
                let fmt = f.float_fmt().filter(|_| f.lanes == 4);
                return Ok(Some(
                    fmt.filter(|f| matches!(f, FloatFmt::Half | FloatFmt::F32)),
                ));
            }
        }
    }
    Ok(None)
}

/// The image operands of a fetched color's read or write in a multisampled image (an input
/// attachment, or a pushed texture's fetch): the invocation's `SampleId` for a per-sample
/// function (`FragmentPlan::fetch_sample`), none for one sample.
pub(super) fn operands(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
) -> BResult<ImageOperands> {
    let Some(sample) = sample(cx, fb)? else {
        return Ok(ImageOperands::default());
    };
    Ok(ImageOperands {
        sample: Some(sample),
        ..ImageOperands::default()
    })
}

/// The invocation's `SampleId`, loaded, for a per-sample function; `None` for one sample.
pub(super) fn sample(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<Option<SValue>> {
    let Some(object) = cx.plan.families.fragment.fetch_sample else {
        return Ok(None);
    };
    let g = stage_io::object_global(cx, object)?;
    let p = fb.module().global_value(g)?;
    fb.load(p, MemAccess::default()).map(Some)
}

/// The invocation's pixel, a `uvec2` from the integer part of `FragCoord.xy` (the pixel's
/// center, single-sampled): the texel a fetch of the attachment's pushed texture reads.
pub(super) fn pixel(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let Some(coord) = cx.plan.families.fragment.fetch_coord else {
        return Err(cx.fail("a pushed fetch without its FragCoord"));
    };
    let frag = stage_io::object_global(cx, coord)?;
    let frag = fb.module().global_value(frag)?;
    let frag = fb.load(frag, MemAccess::default())?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = fb.module().ty(SType::Vector { elem: u32_, len: 2 })?;
    let fx = fb.composite_extract(frag, &[0])?;
    let fy = fb.composite_extract(frag, &[1])?;
    let x = fb.convert(ConvOp::ConvertFToU, fx, u32_)?;
    let y = fb.convert(ConvOp::ConvertFToU, fy, u32_)?;
    fb.composite_construct(uvec2, &[x, y])
}

/// The second source the blend `b` of an extended-range attachment reads, where it reads one:
/// the function's second (index-1) output `second`, as it computed it.
pub(super) fn blend_second(
    cx: &mut EmitCx<'_>,
    b: &extended_range::blend::NativeBlend,
    second: Option<(Form, SValue)>,
) -> BResult<Option<SValue>> {
    match (b.reads_second(), second) {
        (true, Some((_, s1))) => Ok(Some(s1)),
        (true, None) => Err(cx.fail("a blend of the second source without its output")),
        (false, _) => Ok(None),
    }
}

/// The blend color words the blend `b` reads (`PushField::HalfBlendColor`, the encoder's color's
/// `f32` bits), where it reads the blend color.
pub(super) fn blend_constant(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &extended_range::blend::NativeBlend,
) -> BResult<Option<SValue>> {
    if !b.reads_constant() {
        return Ok(None);
    }
    let field = cx.push_field(fb, PushField::HalfBlendColor)?;
    Ok(Some(fb.composite_extract(field, &[0])?))
}

/// `Ok` for an output of four `half` or `float` lanes (`output`, `None` for another), the outputs
/// an extended-range blend takes; else `Feature::ExtendedRangeBlend` `Output`.
pub(super) fn extended_range_output(output: Option<FloatFmt>) -> Result<(), Error> {
    match output {
        Some(FloatFmt::Half | FloatFmt::F32) => Ok(()),
        _ => Err(Unsupported::not_implemented(Feature::ExtendedRangeBlend(
            ExtendedRangeBlend::Output,
        ))
        .into()),
    }
}

/// The blend math of the function `cx` selects ([`extended_range::blend::BlendMath`]).
pub(super) fn blend_math(cx: &SelectCx<'_>) -> extended_range::blend::BlendMath {
    extended_range::blend::BlendMath::of(cx.entry.compile.fast_math)
}

/// The channels a write that keeps `kept` stores.
pub(super) fn written(kept: vk::ColorComponentFlags) -> ColorWriteMask {
    [
        (vk::ColorComponentFlags::R, ColorWriteMask::RED),
        (vk::ColorComponentFlags::G, ColorWriteMask::GREEN),
        (vk::ColorComponentFlags::B, ColorWriteMask::BLUE),
        (vk::ColorComponentFlags::A, ColorWriteMask::ALPHA),
    ]
    .into_iter()
    .filter(|(k, _)| !kept.contains(*k))
    .fold(ColorWriteMask::NONE, |m, (_, c)| m.union(c))
}

#[cfg(test)]
mod tests;
