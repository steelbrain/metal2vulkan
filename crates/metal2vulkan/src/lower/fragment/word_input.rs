//! Metal's write of some of a word attachment's channels on a device that orders framebuffer fetch
//! through rasterization-order input attachments.
//!
//! An attachment the device renders through a view of its word (`metal::format::WordAttachment`)
//! has one component, which the device writes whole, so the device's write mask cannot keep a
//! field of it. On a device that orders fetch through the pixel interlock the storage form merges
//! the word (`fetch`); on one that orders it through input attachments with rasterization-order
//! access (lavapipe) the function reads the texel's word through the attachment's input attachment
//! at its first output's write (at the invocation's sample in a multisampled pipeline) and stores
//! Metal's word of the output (`stage_io::word_color::encode`) with the fields of the channels the
//! descriptor's write mask leaves out taken from the word read, as Metal's tile keeps them. Link
//! writes the whole word (`RuntimeFact::WordInputMerge`). Rasterization-order attachment access
//! orders each fragment's read after every earlier fragment's write of the pixel, so each merge
//! keeps what the fragments before it stored.
//!
//! The function compiles under the descriptor's mask (`LoweringInputs::framebuffer_fetch`), which
//! is the descriptor's own for the formats this merges, the 8-bit sRGB words
//! (`color_writes::merged_word`); a stage linked into a pipeline of another mask there was
//! compiled for another pipeline, `NeedsInput(FramebufferFetch)` (`link::fetch::word_merges`).

use std::collections::BTreeMap;

use crate::air::abi::{AbiInt, OutputRole};
use crate::error::{Attachment, Error};
use crate::lower::arith::{ops, Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::metal::format::WordAttachment;
use crate::metal::ColorWriteMask;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::Stage;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::BinOp;

use super::fetch;

/// The bits of `word`'s fields of the channels `kept`.
fn kept_bits(word: WordAttachment, kept: ColorWriteMask) -> u32 {
    [
        ColorWriteMask::RED,
        ColorWriteMask::GREEN,
        ColorWriteMask::BLUE,
        ColorWriteMask::ALPHA,
    ]
    .into_iter()
    .zip(word.fields())
    .filter_map(|(channel, field)| field.filter(|_| kept.contains(channel)))
    .fold(0, |bits, f| bits | ((1u32 << f.bits) - 1) << f.shift)
}

/// Plan the word attachments the function writes through a first output under a write mask of
/// some of their channels, on a device that fetches through input attachments, before the stage
/// interface: each one the pipeline's state names (`FramebufferFetchInputs`) reads its input
/// attachment and keeps the fields of the channels the mask leaves out, and states
/// `RuntimeFact::WordInputMerge` (see the module docs).
pub(super) fn plan(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage != Stage::Fragment || !fetch::inputs_order(cx.profile) {
        return Ok(());
    }
    let Some(inputs) = cx.inputs.framebuffer_fetch.clone() else {
        return Ok(());
    };
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
    let mut planned = BTreeMap::new();
    for color in written {
        let Some(word) = cx.inputs.word_colors.get(&color).copied() else {
            continue;
        };
        let Some(a) = inputs.attachments.get(&color).copied().flatten() else {
            continue;
        };
        // A mask of every channel or none writes the word whole (`color_writes::word_color`).
        let kept = fetch::kept_channels(a.pixel_format, a.write_mask, true);
        if kept.is_empty() || kept == ColorWriteMask::ALL {
            continue;
        }
        let flags = crate::metal::state::color_write_mask(kept)
            .map_err(|e| e.at_attachment(Attachment::Color(color)))?;
        cx.descriptors.input_attachments = cx.descriptors.input_attachments.saturating_add(1);
        cx.facts.insert(RuntimeFact::FramebufferFetch { color });
        cx.facts
            .insert(RuntimeFact::WordInputMerge { color, kept: flags });
        planned.insert(color, kept_bits(word, kept));
    }
    let merges = !planned.is_empty();
    cx.families.fragment.word_merges = planned;
    match merges {
        true => fetch::plan_sample(cx),
        false => Ok(()),
    }
}

/// The word a write stores to attachment `color`, whose fields `kept` (bits) it takes from the
/// texel's word read through the attachment's input attachment, and the rest from `word`, Metal's
/// word of the output (see the module docs).
pub(crate) fn merge(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    kept: u32,
    word: SValue,
) -> BResult<SValue> {
    let u = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    };
    let texel = super::color::input_texel(cx, fb, color)?;
    let read = fb.composite_extract(texel, &[0])?;
    let keep = ops::splat(fb, u, u64::from(kept))?;
    let take = ops::splat(fb, u, u64::from(!kept))?;
    let read = fb.binary(BinOp::BitwiseAnd, read, keep)?;
    let word = fb.binary(BinOp::BitwiseAnd, word, take)?;
    fb.binary(BinOp::BitwiseOr, read, word)
}
