//! Framebuffer fetch: what orders it, and the writes no form computes over the texel it reads.
//!
//! **Where.** Metal orders a fetch after every earlier fragment of the pixel, across draws. A color
//! input reads the attachment's input attachment (`color`), which Vulkan orders so only under
//! rasterization-order color attachment access, which no form here takes: the read is ordered
//! between draws alone, by the consumer's barrier of the subpass's dependency on itself, and notes
//! `PrecisionNote::UnorderedFetchWithinDraw`; under the identity domains it reads the attachment's
//! pushed texture at the invocation's pixel instead (`PushField::FetchedColor`, [`pixel`]). The
//! pixel interlock orders no fetch: it serves raster order groups on memory the function itself
//! reads and writes (`fragment::rog`).
//!
//! **Writes over the texel.** A write no Vulkan blend or write mask gives would be computed by the
//! function over an ordered read of the attachment, which is not built: a padded extended-range
//! attachment's blend, a write of some of the codes one word of such an attachment holds
//! (`color_writes::merged_extended`), and a blend no Vulkan state gives on an attachment held with
//! its channels rotated (`color_writes::rotated_blend_in_shader`) are refused on every device
//! ([`computed_write_refused`]); so is a private logic operation the device lacks
//! (`pipeline::private_state`).
//!
//! **Per sample**. Metal runs a function that reads a color attachment of a multisampled pass once
//! per covered sample, reading its own sample, in draw order per sample, with `[[position]]`'s x
//! and y at the pixel center (measured). Compiled for such a pipeline
//! (`LoweringInputs::multisampled`), the function reads the attachment's sample at its `SampleId`
//! ([`plan_sample`]), which runs it once per covered sample under Vulkan's sample shading, and needs
//! `sampleRateShading`.
//!
//! **Apple's extended-range forms** (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their
//! `_sRGB` twins, [`codes_round_the_tile`]): Metal's tile memory holds such an attachment as `half`
//! through a render pass and rounds it to the codes only at the store, so a fetch or a blend onto
//! what an earlier draw of the pass wrote sees that draw's `half`, which the codes the attachment
//! holds do not keep. A fetch of one, and a blend into one the function writes, are not
//! implemented: refused on every device (`Feature::Format`, at the attachment; here, in
//! `color::select_param`, and in `stage_io::extended_color`). An unblended write, a masked one
//! included, is built: a kept channel's code is the encode of the `half` the tile keeps there.
//! `metal2vulkan-runtime` makes no extended-range texture (`Gap::TextureFormat`).

use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io;
use crate::metal::format::ExtendedRange;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{ConvOp, ImageOperands, MemAccess, Type as SType};

/// Whether Metal's tile memory holds an attachment of `form` as `half` through a render pass and
/// rounds it to the codes only at the store, where the attachment holds the codes: Apple's
/// extended-range forms (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their `_sRGB` twins), not
/// a padded or bfloat one. A fetch of such an attachment, and a blend into one, are not
/// implemented (`Feature::Format`, module docs).
pub(crate) fn codes_round_the_tile(form: ExtendedRange) -> bool {
    !form.padded() && !form.bfloat()
}

/// The refusal of a write the fragment function would compute over its own ordered read of the
/// attachment, on every device (module docs): a padded extended-range attachment's blend, a write
/// of some of the codes one word of such an attachment holds, and a blend no Vulkan state gives on
/// an attachment held with its channels rotated. No device's blend unit or write mask gives them,
/// and their form over the read is not built.
pub(crate) fn computed_write_refused() -> Unsupported {
    Unsupported::not_implemented(Feature::FramebufferFetch)
}

/// Whether `pf` is a 16-bit float color format, whose blend of a `half` output of a module
/// without fast math Metal computes in `half` arithmetic.
pub(crate) fn half_float_format(pf: crate::metal::PixelFormat) -> bool {
    use crate::metal::PixelFormat as P;
    matches!(pf, P::R16Float | P::RG16Float | P::RGBA16Float)
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
