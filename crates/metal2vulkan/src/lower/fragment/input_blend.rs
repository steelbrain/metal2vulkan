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
//! The blend takes a `half4` or `float4` output under every factor, operation and write mask, as
//! the storage form does (another output is refused, `Feature::ExtendedRangeBlend` `Output`); its
//! second source is the function's index-1 output, read from the returned value and stored nowhere
//! else ([`select_second_output`]). A write of none of the format's channels stores nothing and
//! blends nothing. A padded or bfloat attachment's blend is not built, and stays refused
//! `NotImplemented(Format)` (`stage_io::extended_color`).

use std::collections::BTreeMap;

use crate::air::ir::TypeId as AirTypeId;
use crate::error::{
    Attachment, DescriptorError, Error, ExtendedRangeBlend, Feature, RuntimeInput, Unsupported,
};
use crate::lower::arith::{extended_range, Form};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::metal::format::ExtendedRange;
use crate::metal::ColorWriteMask;
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
    /// The pipeline's blend at the attachment.
    pub equation: BlendEquation,
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
    if blended.is_empty() {
        return Ok(());
    }
    let Some(inputs) = cx.inputs.framebuffer_fetch.clone() else {
        return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
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
                equation: eq,
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
                equation: eq,
                blend: Some(blend),
                kept,
                second: None,
            },
        );
    }
    let reads = planned.values().any(|ib| ib.blend.is_some());
    cx.families.fragment.input_blends = planned;
    match reads {
        true => fetch::plan_sample(cx),
        false => Ok(()),
    }
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
    let Some(b) = &ib.blend else {
        return crate::lower::stage_io::extended_color::texel(cx, fb, ib.form, form, part);
    };
    let s1 = fetch::blend_second(cx, b, second)?;
    let table = crate::lower::texture::extended_table(cx, ib.form)?;
    let codes = super::color::input_texel(cx, fb, color)?;
    let d = extended_range::tile_value(fb, ib.form, codes, table)?;
    let constant = fetch::blend_constant(cx, fb, b)?;
    let ranges = Some(extended_range::blend::extended_ranges(ib.form));
    let v = extended_range::blend::native(fb, b, ranges, (part, s1, d), constant)?;
    if ib.kept.is_empty() {
        return extended_range::encode(fb, ib.form, v, table);
    }
    let kept = [
        vk::ColorComponentFlags::R,
        vk::ColorComponentFlags::G,
        vk::ColorComponentFlags::B,
        vk::ColorComponentFlags::A,
    ]
    .map(|c| ib.kept.contains(c));
    extended_range::encode_keeping(fb, ib.form, v, table, (codes, kept))
}

/// The words the planned blends add at `rets` returns: one input attachment each beside the
/// reads `color` counts, and at each return the read, decode, blend and encode.
pub(super) fn words(cx: &SelectCx<'_>, rets: u64) -> u64 {
    cx.families
        .fragment
        .input_blends
        .values()
        .filter_map(|ib| ib.blend.as_ref())
        .map(|b| {
            let each = WRITE_WORDS
                .saturating_add(u64::from(b.words()))
                .saturating_add(u64::from(
                    extended_range::DECODE_WORDS + extended_range::ENCODE_WORDS,
                ));
            each.saturating_mul(rets.max(1))
        })
        .fold(0u64, u64::saturating_add)
}
