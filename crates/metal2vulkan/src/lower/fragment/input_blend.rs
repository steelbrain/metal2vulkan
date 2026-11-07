//! The blend into an extended-range attachment on a device that orders framebuffer fetch through
//! rasterization-order input attachments.
//!
//! Such an attachment holds Metal's raw texels in its raw unsigned format, which no device blends.
//! On a device that fetches in the pixel interlock the storage form computes the blend in the
//! shader (`fetch`); on one that fetches through input attachments with rasterization-order access
//! (lavapipe) the function computes it over the attachment's input attachment instead: its first
//! output's write reads the pixel's raw texel there (at the invocation's sample in a multisampled
//! pipeline), decodes it to Metal's tile value (`extended_range::tile_value`, as a fetch reads it),
//! blends the output over that by the device's arithmetic (`extended_range::blend::native`, the
//! second source and blend color as the storage form reads them, `fetch::blend_second`,
//! `fetch::blend_constant`),
//! and writes the kernel write's encode of the result as the raw texel through the output variable
//! (`extended_range::encode`), the channels a write mask keeps taken from the texel read
//! (`extended_range::encode_keeping`). The attachment then takes no Vulkan blend and every raw
//! component is written (link, from `RuntimeFact::ExtendedRangeInputBlend`). Rasterization-order
//! attachment access orders each fragment's read after every earlier fragment's write of the pixel,
//! so each blend reads the codes the one before stored, as a draw of the storage form without the
//! `half` shadow does (the announcement, `extended-range-blends-stored-codes`, stays the runtime's
//! there).
//!
//! Where the draw's instance holds the attachment's `half` shadow (`input_shadow`), the blend's
//! destination is the shadow's texel, Metal's tile value, and the write stores the blend's result
//! there too; a merge stores the written channels' `half`s there.
//!
//! There too an unblended write of some of the codes one word holds (`BGR10_XR`'s word, the
//! two-plane forms' color word, `color_writes::merged_extended`), which no mask of the raw
//! components gives, reads the texel's raw codes through the input attachment and writes the
//! kernel write's encode of its output with the codes the mask keeps taken from them
//! (`extended_range::encode_keeping`), as Metal's tile keeps a masked channel's code; it states
//! `RuntimeFact::WordInputMerge`, and link writes every raw component.
//!
//! An attachment held with its channels rotated (`ARGB8Unorm` in `R8G8B8A8_UNORM`) whose blend no
//! Vulkan state on its image gives (`color_writes::shader_rotated_blend`) takes Metal's blend there
//! too ([`RotatedInputBlend`]): Metal blends the format as `RGBA8Unorm` with its channels rotated, so
//! the first output's write reads the pixel's texel through the input attachment, rotates it into
//! Metal's order, computes Metal's 8-bit unorm blend of the clamped output over it
//! (`extended_range::Blend::unorm8`, `extended_range::blend::blend_unorm8`, as the storage form
//! computes it), and writes the result through the output variable, which the stage interface
//! rotates into the image's order as every output to such an attachment
//! (`stage_io::output_part`); the device's conversion stores its code
//! (`PrecisionNote::DeviceColorConversion`), and the descriptor's write mask, rotated, keeps the
//! channels it leaves out. It states `RuntimeFact::RotatedInputBlend`, and link gives the
//! attachment no Vulkan blend.
//!
//! The blend takes a `half4` or `float4` output under every factor, operation and write mask, as
//! the storage form does (another output is refused, `Feature::ExtendedRangeBlend` `Output`); its
//! second source is the function's index-1 output, read from the returned value and stored nowhere
//! else ([`select_second_output`]). A write of none of the format's channels stores nothing and
//! blends nothing. A padded or bfloat attachment's blend is not built, and stays refused
//! `NotImplemented(Format)` (`stage_io::extended_color`).

use std::collections::BTreeMap;

use crate::air::abi::{AbiInt, OutputRole};
use crate::air::ir::TypeId as AirTypeId;
use crate::error::{
    Attachment, DescriptorError, Error, ExtendedRangeBlend, Feature, RuntimeInput, Unsupported,
};
use crate::lower::arith::{extended_range, FloatFmt, Form};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::metal::format::ExtendedRange;
use crate::metal::{ColorWriteMask, PixelFormat};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::BlendEquation;
use crate::pipeline::sym::{PushField, Stage};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::vk;

use super::fetch;

/// An extended-range attachment whose blend the function computes over its input attachment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputBlend {
    /// The attachment's form.
    pub form: ExtendedRange,
    /// The pipeline's blend at the attachment; `None` for an unblended write of some of the codes
    /// one word holds, which the function merges into the texel (see the module docs).
    pub equation: Option<BlendEquation>,
    /// The blend of the write, as the shader computes it; `None` where the write stores none of
    /// the format's channels, and blends nothing.
    pub blend: Option<extended_range::blend::NativeBlend>,
    /// The format's channels the write keeps from the texel (those the descriptor's write mask
    /// leaves out); empty where it stores every one, all four where it stores none.
    pub kept: vk::ColorComponentFlags,
    /// The returned value's field holding the function's second (index-1) output to the
    /// attachment and its form, once the stage interface planned it, where the blend reads the
    /// second source ([`select_second_output`]).
    pub second: Option<(u32, Form)>,
}

impl InputBlend {
    /// Whether the write is unblended and keeps some of the format's codes from the texel it reads
    /// (`color_writes::merged_extended`).
    pub fn merges(&self) -> bool {
        self.equation.is_none()
    }

    /// Whether the write reads the attachment's input attachment: a blend, or a merge.
    pub fn reads(&self) -> bool {
        self.blend.is_some() || self.merges()
    }
}

/// An attachment held with its channels rotated (`ARGB8Unorm`) whose blend the function computes
/// over its input attachment (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RotatedInputBlend {
    /// The pipeline's blend at the attachment.
    pub equation: BlendEquation,
    /// Metal's channels the write stores: those the descriptor's write mask covers, none where it
    /// covers none (the write then reads and blends nothing).
    pub written: ColorWriteMask,
}

impl RotatedInputBlend {
    /// Whether the write reads the attachment's input attachment: it stores some channel.
    pub fn reads(&self) -> bool {
        !self.written.is_empty()
    }
}

/// An upper estimate of the words one blended write adds beside its blend's arithmetic: the
/// input attachment's read, its decode, the encode, and a merge of kept channels.
const WRITE_WORDS: u64 = 120;

/// Whether the device orders framebuffer fetch through rasterization-order input attachments,
/// where this module computes an extended-range blend (on one that orders it through the pixel
/// interlock, the storage form does, `fetch::interlock_orders`).
fn through_inputs(cx: &SelectCx<'_>) -> bool {
    fetch::inputs_order(cx.profile)
}

/// Plan the extended-range attachments the function writes through a first output where the
/// pipeline blends them (`LoweringInputs::extended_range_colors`), on a device that fetches
/// through input attachments, before the stage interface: each one whose write stores some of
/// its format's channels reads its input attachment, blends there, and states
/// `RuntimeFact::ExtendedRangeInputBlend` (see the module docs).
pub(super) fn plan(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage != Stage::Fragment || !through_inputs(cx) {
        return Ok(());
    }
    let blended = fetch::blended_extended_range(cx)?;
    let Some(inputs) = cx.inputs.framebuffer_fetch.clone() else {
        return match blended.is_empty() {
            true => Ok(()),
            false => Err(Error::NeedsInput(RuntimeInput::FramebufferFetch)),
        };
    };
    let mut planned = BTreeMap::new();
    for (color, ty) in blended {
        let tied = |e: Error| e.at_attachment(Attachment::Color(color));
        let Some(form) = cx
            .inputs
            .extended_range_colors
            .get(&color)
            .map(|e| e.form)
            .filter(|f| !f.padded() && !f.bfloat())
        else {
            // A padded or bfloat attachment's blend is not built (`extended_color`).
            continue;
        };
        // Compiled under the state of a pipeline for another function.
        let Some((attachment, eq)) = inputs
            .attachments
            .get(&color)
            .copied()
            .flatten()
            .and_then(|a| a.equation.map(|eq| (a, eq)))
        else {
            return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
        };
        let kept = fetch::kept_channels(attachment.pixel_format, attachment.write_mask, true);
        let stores = kept != ColorWriteMask::ALL;
        let kept = crate::metal::state::color_write_mask(kept).map_err(tied)?;
        cx.facts.insert(RuntimeFact::ExtendedRangeInputBlend {
            color,
            blend: eq,
            kept,
        });
        cx.families.arith.converts(form);
        // A write of none of the format's channels stores nothing: link gives it no channel.
        if !stores {
            let ib = InputBlend {
                form,
                equation: Some(eq),
                blend: None,
                kept,
                second: None,
            };
            planned.insert(color, ib);
            continue;
        }
        let f = super::color::form(cx, ty).map_err(tied)?;
        let output = (f.lanes == 4).then(|| f.float_fmt()).flatten();
        fetch::extended_range_output(output).map_err(tied)?;
        let blend =
            extended_range::blend::NativeBlend::of(&eq, fetch::written(kept)).map_err(tied)?;
        // A blend of the second source reads the function's index-1 output, four `half` or
        // `float` lanes as the first's. Without one, Metal refuses the pipeline ("Fragment shader
        // does not write to render target color(0), index(1) that is required for blending").
        if blend.reads_second() {
            match fetch::second_output4(cx, color).map_err(tied)? {
                Some(Some(_)) => {}
                Some(None) => {
                    let gap = Feature::ExtendedRangeBlend(ExtendedRangeBlend::Output);
                    return Err(tied(Unsupported::not_implemented(gap).into()));
                }
                None => {
                    let invalid = DescriptorError::ColorAttachment { index: color };
                    return Err(Error::InvalidDescriptor(invalid));
                }
            }
        }
        if blend.reads_constant() {
            cx.push.insert(PushField::HalfBlendColor);
        }
        // The attachment's input attachment, which a `[[color(n)]]` read of it shares (`color`).
        cx.descriptors.input_attachments = cx.descriptors.input_attachments.saturating_add(1);
        cx.facts.insert(RuntimeFact::FramebufferFetch { color });
        planned.insert(
            color,
            InputBlend {
                form,
                equation: Some(eq),
                blend: Some(blend),
                kept,
                second: None,
            },
        );
    }
    merged(cx, &inputs, &mut planned)?;
    let rotated = rotated(cx, &inputs)?;
    let reads =
        planned.values().any(InputBlend::reads) || rotated.values().any(RotatedInputBlend::reads);
    cx.families.fragment.input_blends = planned;
    cx.families.fragment.rotated_blends = rotated;
    match reads {
        true => fetch::plan_sample(cx),
        false => Ok(()),
    }
}

/// Plan, into `planned`, the extended-range attachments the function writes through a first output
/// unblended under a mask of some of the codes one word holds, as the pipeline's state `inputs`
/// names them (`color_writes::merged_extended`): each reads its input attachment, merges the
/// written codes into the texel it reads, and states `RuntimeFact::WordInputMerge` with the
/// channels it keeps (see the module docs).
fn merged(
    cx: &mut SelectCx<'_>,
    inputs: &crate::pipeline::lower_inputs::FramebufferFetchInputs,
    planned: &mut BTreeMap<u32, InputBlend>,
) -> Result<(), Error> {
    let written: Vec<u32> = cx
        .entry
        .ep
        .outputs
        .iter()
        .filter_map(|o| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } => Some(color),
            _ => None,
        })
        .collect();
    for color in written {
        let Some(a) = inputs.attachments.get(&color).copied().flatten() else {
            continue;
        };
        let Some(form) = cx
            .inputs
            .extended_range_colors
            .get(&color)
            .filter(|e| !e.blended)
            .map(|e| e.form)
        else {
            continue;
        };
        let merges = crate::pipeline::color_writes::merges_extended;
        if !merges(a.pixel_format, a.write_mask, a.blending_enabled) {
            continue;
        }
        let tied = |e: Error| e.at_attachment(Attachment::Color(color));
        let kept = fetch::kept_channels(a.pixel_format, a.write_mask, true);
        let kept = crate::metal::state::color_write_mask(kept).map_err(tied)?;
        cx.facts.insert(RuntimeFact::WordInputMerge { color, kept });
        cx.facts.insert(RuntimeFact::FramebufferFetch { color });
        cx.families.arith.converts(form);
        cx.descriptors.input_attachments = cx.descriptors.input_attachments.saturating_add(1);
        planned.insert(
            color,
            InputBlend {
                form,
                equation: None,
                blend: None,
                kept,
                second: None,
            },
        );
    }
    Ok(())
}

/// Plan the attachments held with their channels rotated that the function writes through a first
/// output and the pipeline's state `inputs` names (`color_writes::shader_rotated_blend`, the only
/// reason such an attachment is named on this device): each one whose write stores some of its
/// channels reads its input attachment and computes Metal's 8-bit unorm blend there, and each
/// states `RuntimeFact::RotatedInputBlend` (see the module docs).
fn rotated(
    cx: &mut SelectCx<'_>,
    inputs: &crate::pipeline::lower_inputs::FramebufferFetchInputs,
) -> Result<BTreeMap<u32, RotatedInputBlend>, Error> {
    let mut planned = BTreeMap::new();
    let outputs: Vec<(u32, AirTypeId)> = cx
        .entry
        .ep
        .outputs
        .iter()
        .zip(crate::lower::stage_io::output_types(cx)?.1)
        .filter_map(|(o, ty)| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } => Some((color, ty)),
            _ => None,
        })
        .collect();
    for (color, ty) in outputs {
        let Some(a) = inputs.attachments.get(&color).copied().flatten() else {
            continue;
        };
        if !crate::metal::format::rotated_channels(a.pixel_format)
            || !cx.inputs.rotated_colors.contains(&color)
        {
            continue;
        }
        let tied = |e: Error| e.at_attachment(Attachment::Color(color));
        let Some(eq) = a.equation.filter(|_| a.blending_enabled) else {
            // Compiled under the state of a pipeline for another function.
            return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
        };
        let kept = fetch::kept_channels(a.pixel_format, a.write_mask, true);
        let written = ColorWriteMask::ALL.difference(kept);
        let kept = crate::metal::state::color_write_mask(kept).map_err(tied)?;
        cx.facts.insert(RuntimeFact::RotatedInputBlend {
            color,
            blend: eq,
            kept,
        });
        let rb = RotatedInputBlend {
            equation: eq,
            written: match kept == vk::ColorComponentFlags::known_bits() {
                true => ColorWriteMask::NONE,
                false => written,
            },
        };
        if rb.reads() {
            let f = super::color::form(cx, ty).map_err(tied)?;
            let arith = match (f.lanes, f.float_fmt()) {
                (4, Some(FloatFmt::Half)) => extended_range::blend::BlendArith::Half,
                (4, Some(FloatFmt::F32)) => extended_range::blend::BlendArith::Float,
                // An output of fewer lanes to a rotated attachment (`stage_io`).
                _ => {
                    let refused = Unsupported::not_implemented(Feature::Format);
                    return Err(tied(refused.into()));
                }
            };
            let math = fetch::blend_math(cx);
            let b = extended_range::Blend::unorm8(
                &eq,
                PixelFormat::ARGB8Unorm,
                rb.written,
                (math, arith),
            )
            .map_err(tied)?;
            if b.reads_constant() {
                cx.push.insert(PushField::HalfBlendColor);
            }
            // The attachment's input attachment.
            cx.descriptors.input_attachments = cx.descriptors.input_attachments.saturating_add(1);
            cx.facts.insert(RuntimeFact::FramebufferFetch { color });
            // The result is the device's to convert to its code; the blend is Metal's where the
            // module states its math (Metal clamps the output first, so none of the output's
            // arithmetic folds into it), as the storage form's is.
            cx.precision.insert(PrecisionNote::DeviceColorConversion);
            if cx.entry.compile.fast_math.is_none() {
                let note = crate::lower::arith::output_precision::half_output_note(cx);
                cx.precision.insert(note);
            }
        }
        planned.insert(color, rb);
    }
    Ok(planned)
}

/// Whether the function's second (index-1) output to `color` is the second source of the blend
/// the function computes over the attachment's input attachment ([`select_second_output`]).
pub(super) fn blends_second(cx: &SelectCx<'_>, color: u32) -> bool {
    cx.families
        .fragment
        .input_blends
        .get(&color)
        .and_then(|ib| ib.blend.as_ref())
        .is_some_and(extended_range::blend::NativeBlend::reads_second)
}

/// Plan the second (index-1) output of AIR type `ty` held in field `field` of the returned value
/// to attachment `color`, whose blend the function computes over its input attachment and reads
/// the second source ([`blends_second`]): the blend reads it, clamped as the first output; no
/// interface variable holds it.
pub fn select_second_output(
    cx: &mut SelectCx<'_>,
    field: u32,
    color: u32,
    ty: AirTypeId,
) -> Result<(), Error> {
    let form = super::color::form(cx, ty)?;
    let Some(ib) = cx.families.fragment.input_blends.get_mut(&color) else {
        return Err(cx.invariant("a second output of an attachment no input blend writes"));
    };
    ib.second = Some((field, form));
    Ok(())
}

/// The raw texel, as a `uvec4`, a write of output `part` (of `form`; the second source `second`
/// where the blend reads it) to attachment `color` stores, whose blend the function computes over
/// its input attachment (`ib`; see the module docs). For a write that stores none of the
/// format's channels, the unblended output's, which link stores nowhere.
pub(crate) fn texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (color, ib): (u32, &InputBlend),
    (form, part): (Form, SValue),
    second: Option<(Form, SValue)>,
) -> BResult<SValue> {
    let kept = [
        vk::ColorComponentFlags::R,
        vk::ColorComponentFlags::G,
        vk::ColorComponentFlags::B,
        vk::ColorComponentFlags::A,
    ]
    .map(|c| ib.kept.contains(c));
    // The instance's `half` shadow of the attachment, which the write reads as its destination
    // and stores its value into (`input_shadow`).
    let shadowed = super::input_shadow::shadows(cx, color).is_some();
    let Some(b) = &ib.blend else {
        if !ib.merges() {
            return crate::lower::stage_io::extended_color::texel(fb, ib.form, form, part);
        }
        // The unblended write's `half`, encoded, the codes the mask keeps taken from the texel.
        let codes = super::color::input_texel(cx, fb, color)?;
        let v = crate::lower::stage_io::extended_color::held(fb, ib.form, form, part)?;
        if shadowed {
            super::input_shadow::store(cx, fb, color, v)?;
        }
        return extended_range::encode_keeping(fb, ib.form, v, (codes, kept));
    };
    let s1 = fetch::blend_second(cx, b, second)?;
    let (d, codes) = match shadowed {
        true => (super::input_shadow::read(cx, fb, color)?, None),
        false => {
            let codes = super::color::input_texel(cx, fb, color)?;
            (extended_range::tile_value(fb, ib.form, codes)?, Some(codes))
        }
    };
    let constant = fetch::blend_constant(cx, fb, b)?;
    let ranges = Some(extended_range::blend::extended_ranges(ib.form));
    let v = extended_range::blend::native(fb, b, ranges, (part, s1, d), constant)?;
    if shadowed {
        super::input_shadow::store(cx, fb, color, v)?;
    }
    if ib.kept.is_empty() {
        return extended_range::encode(fb, ib.form, v);
    }
    let codes = match codes {
        Some(codes) => codes,
        None => super::color::input_texel(cx, fb, color)?,
    };
    extended_range::encode_keeping(fb, ib.form, v, (codes, kept))
}

/// The Metal-ordered value a write of first output `part` (of `form`: four `half` or `float`
/// lanes held as `f32`s) to attachment `color`, held with its channels rotated, stores, whose blend
/// the function computes over its input attachment (`rb`; see the module docs): the texel read
/// there, rotated into Metal's order, and Metal's 8-bit unorm blend of the output over it
/// (`extended_range::blend::blend_unorm8`), the blend color from `PushField::HalfBlendColor` where
/// the blend reads it. For a write that stores none of the format's channels, the output.
pub(crate) fn rotated_texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (color, rb): (u32, &RotatedInputBlend),
    (form, part): (Form, SValue),
) -> BResult<SValue> {
    if !rb.reads() {
        return Ok(part);
    }
    let arith = match (form.lanes, form.float_fmt()) {
        (4, Some(FloatFmt::Half)) => extended_range::blend::BlendArith::Half,
        (4, Some(FloatFmt::F32)) => extended_range::blend::BlendArith::Float,
        _ => return Err(cx.fail("an 8-bit unorm blend of another output")),
    };
    let math = extended_range::blend::BlendMath::of(cx.entry.compile.fast_math);
    let pf = PixelFormat::ARGB8Unorm;
    let Ok(b) = extended_range::Blend::unorm8(&rb.equation, pf, rb.written, (math, arith)) else {
        return Err(cx.fail("an 8-bit unorm blend outside Metal's model"));
    };
    let texel = super::color::input_texel(cx, fb, color)?;
    // The image's red, green, blue and alpha hold Metal's alpha, red, green and blue.
    let texel = fb.vector_shuffle(texel, texel, &[1, 2, 3, 0])?;
    let constant = match b.reads_constant() {
        true => {
            let field = cx.push_field(fb, PushField::HalfBlendColor)?;
            Some(fb.composite_extract(field, &[0])?)
        }
        false => None,
    };
    extended_range::blend::blend_unorm8(fb, &b, arith, (part, texel), constant)
}

/// The words the planned blends add at `rets` returns: one input attachment each beside the
/// reads `color` counts, and at each return the read, decode, blend and encode.
pub(super) fn words(cx: &SelectCx<'_>, rets: u64) -> u64 {
    let fp = &cx.families.fragment;
    let extended = fp.input_blends.values().filter(|ib| ib.reads()).map(|ib| {
        let blend = ib.blend.as_ref().map_or(0, |b| b.words());
        let each = WRITE_WORDS
            .saturating_add(u64::from(blend))
            .saturating_add(u64::from(
                extended_range::DECODE_WORDS + extended_range::ENCODE_WORDS,
            ));
        each.saturating_mul(rets.max(1))
    });
    let math = fetch::blend_math(cx);
    let rotated = fp
        .rotated_blends
        .values()
        .filter(|rb| rb.reads())
        .map(|rb| {
            let half = extended_range::blend::BlendArith::Half;
            let pf = PixelFormat::ARGB8Unorm;
            let blend = extended_range::Blend::unorm8(&rb.equation, pf, rb.written, (math, half))
                .map_or(0, |b| b.words());
            WRITE_WORDS
                .saturating_add(u64::from(blend))
                .saturating_mul(rets.max(1))
        });
    extended.chain(rotated).fold(0u64, u64::saturating_add)
}
