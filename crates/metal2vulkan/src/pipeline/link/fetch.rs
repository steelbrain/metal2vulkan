//! A render pipeline's framebuffer fetch computed over input attachments.
//!
//! On a device that orders framebuffer fetch through rasterization-order input attachments, the
//! function computes an extended-range attachment's blend over its input attachment
//! (`RuntimeFact::ExtendedRangeInputBlend`, [`input_blends`]): the attachment stays a color
//! attachment of the subpass, which `graphics_state` gives no Vulkan blend and a write of every raw
//! component, the function merging the channels the descriptor's mask keeps; a stage compiled under
//! another blend or mask there was compiled for another pipeline, `NeedsInput(FramebufferFetch)`.
//! There too the function computes the blend of an attachment held with its channels rotated that
//! no Vulkan state on its image gives (`RuntimeFact::RotatedInputBlend`, [`input_blends`]), which
//! `graphics_state` gives no Vulkan blend and the descriptor's write mask, rotated.
//! There too the function merges its word of a word attachment written under a mask of some of
//! the word's channels into the word its input attachment holds (`RuntimeFact::WordInputMerge`,
//! [`word_merges`]); `graphics_state` writes the whole word, and a stage compiled under another
//! mask there, or without the merge, was compiled for another pipeline. So does it merge the codes
//! of an extended-range attachment written unblended under a mask of some of the codes one word
//! holds (`color_writes::merged_extended`, the same fact): [`input_blends`] names it beside the
//! blended ones, so that `graphics_state` gives it no blend and a write of every raw component.

use std::collections::BTreeMap;

use crate::device::DeviceProfile;
use crate::error::{Error, InternalError, InternalKind, RuntimeInput};
use crate::library::FunctionArtifact;
use crate::lower::fragment::{inputs_order, kept_channels};
use crate::metal::format::ExtendedRange;
use crate::metal::state::color_write_mask;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::RenderLinkInputs;
use crate::pipeline::lower_inputs::BlendEquation;
use crate::pipeline::stage_io::IoRole;

/// The extended-range attachments whose blend fragment stage `fragment` computes over their
/// input attachments (`RuntimeFact::ExtendedRangeInputBlend`), or whose codes it merges there
/// (`RuntimeFact::WordInputMerge` of such an attachment), and the attachments held with their
/// channels rotated whose blend it computes there (`RuntimeFact::RotatedInputBlend`, which link
/// gives no Vulkan blend and the descriptor's write mask, rotated), on a device that orders fetch through
/// rasterization-order input attachments, against the attachments of render inputs `r`: by color
/// index, the channels each write keeps from its texel, where the pipeline's blend and write mask
/// there are those the function compiled under; `NeedsInput(FramebufferFetch)` where they are
/// another pipeline's (a merge's mask, [`word_merges`]).
pub fn input_blends(
    r: &RenderLinkInputs,
    fragment: Option<&FunctionArtifact>,
    p: &DeviceProfile,
) -> Result<BTreeMap<u32, crate::vk::ColorComponentFlags>, Error> {
    let mut out = BTreeMap::new();
    for fact in fragment.into_iter().flat_map(|f| &f.facts) {
        // An extended-range attachment's merge, which [`word_merges`] holds to the pipeline's
        // state, writes every raw component as a blend over the input attachment does.
        if let RuntimeFact::WordInputMerge { color, kept } = *fact {
            let extended = r
                .color_attachments
                .get(color as usize)
                .copied()
                .flatten()
                .is_some_and(|a| ExtendedRange::of(a.pixel_format).is_some());
            if extended {
                out.insert(color, kept);
            }
            continue;
        }
        // An attachment held with its channels rotated whose blend the function computes over its
        // input attachment (`color_writes::shader_rotated_blend`): its write keeps the channels the
        // descriptor's mask leaves out through the attachment's write mask.
        if let RuntimeFact::RotatedInputBlend { color, blend, kept } = *fact {
            if !inputs_order(p) {
                return Err(Error::Internal(InternalError::new(
                    InternalKind::Invariant,
                    "a rotated blend over an input attachment on a device that fetches otherwise",
                )));
            }
            let Some(a) = r.color_attachments.get(color as usize).copied().flatten() else {
                return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
            };
            let pipeline_kept =
                color_write_mask(kept_channels(a.pixel_format, a.write_mask, true))?;
            if !crate::pipeline::color_writes::shader_rotated_blend(&a, p)
                || BlendEquation::of(&a) != Some(blend)
                || pipeline_kept != kept
            {
                return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
            }
            out.insert(color, kept);
            continue;
        }
        let RuntimeFact::ExtendedRangeInputBlend { color, blend, kept } = *fact else {
            continue;
        };
        if !inputs_order(p) {
            return Err(Error::Internal(InternalError::new(
                InternalKind::Invariant,
                "an extended-range blend over an input attachment on a device that fetches \
                 otherwise",
            )));
        }
        let Some(a) = r.color_attachments.get(color as usize).copied().flatten() else {
            return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
        };
        let pipeline_kept = color_write_mask(kept_channels(a.pixel_format, a.write_mask, true))?;
        if BlendEquation::of(&a) != Some(blend) || pipeline_kept != kept {
            return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
        }
        out.insert(color, kept);
    }
    Ok(out)
}

/// The word merges the fragment stage `fragment` computes over input attachments
/// (`RuntimeFact::WordInputMerge`), against the attachments of render inputs `r` on `p`: each
/// attachment the stage writes through a first output whose word the pipeline's state merges
/// (`color_writes::merged_word`, and an extended-range one's codes, `color_writes::merged_extended`)
/// needs the stage's merge keeping the channels the mask leaves out,
/// and each merge the stage states needs that state. A stage compiled under another mask there,
/// or without the merge, is `NeedsInput(FramebufferFetch)`.
pub fn word_merges(
    r: &RenderLinkInputs,
    fragment: Option<&FunctionArtifact>,
    p: &DeviceProfile,
) -> Result<(), Error> {
    let Some(f) = fragment else {
        return Ok(());
    };
    let stated: BTreeMap<u32, crate::vk::ColorComponentFlags> = f
        .facts
        .iter()
        .filter_map(|fact| match *fact {
            RuntimeFact::WordInputMerge { color, kept } => Some((color, kept)),
            _ => None,
        })
        .collect();
    if !stated.is_empty() && !inputs_order(p) {
        return Err(Error::Internal(InternalError::new(
            InternalKind::Invariant,
            "a word merge over an input attachment on a device that fetches otherwise",
        )));
    }
    let mut wanted = BTreeMap::new();
    for o in &f.interface.io.outputs {
        let IoRole::Color {
            attachment: color,
            index: 0,
        } = o.role
        else {
            continue;
        };
        let Some(a) = r.color_attachments.get(color as usize).copied().flatten() else {
            continue;
        };
        if let Some(kept) =
            crate::pipeline::color_writes::merged_word(color, &a, r.alpha_to_coverage, p)
        {
            wanted.insert(color, color_write_mask(kept)?);
        } else if crate::pipeline::color_writes::merged_extended(&a) && inputs_order(p) {
            let kept = kept_channels(a.pixel_format, a.write_mask, true);
            wanted.insert(color, color_write_mask(kept)?);
        }
    }
    match stated == wanted {
        true => Ok(()),
        false => Err(Error::NeedsInput(RuntimeInput::FramebufferFetch)),
    }
}
