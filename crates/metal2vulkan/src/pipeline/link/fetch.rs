//! A render pipeline's interlocked framebuffer fetch: `GraphicsState::storage_attachments` from the
//! fragment stage's `RuntimeFact::InterlockedFramebufferFetch`.
//!
//! On a device that orders framebuffer fetch only through the ordered pixel interlock, the fragment
//! function reads each color attachment, and writes it, as a storage image inside the interlock
//! where the pipeline's state lets that form be exact (`lower::fragment`'s `fetch`). The function
//! compiled under its pipeline's state (`LoweringInputs::framebuffer_fetch`); link decides the form
//! again from the state of the pipeline it links into, with the same rule, and a stage whose form,
//! attachment format, kept channels (a write mask short of the format's channels), or in-shader
//! blend (an extended-range or 16-bit float attachment's) that state would not choose (a write
//! before a depth or stencil attachment's tests only where the function states
//! `RuntimeFact::TestedStorageWrites`, which each draw's plan checks) was compiled for another
//! pipeline: `NeedsInput(FramebufferFetch)`. A write mask that keeps some of a format's channels
//! compiles as its class (`FramebufferFetchInputs::of`), so the comparison holds across the masks
//! of a class, and the pipeline's own kept channels go in `StorageAttachmentUse::kept`, which the
//! binder writes into each draw's push bytes. An attachment the function reads in the storage form
//! is no color attachment of the subpass (`graphics_state` gives it no state), and the consumer
//! binds it as `GraphicsState::storage_attachments` states.
//!
//! In a multisampled pipeline the form is per sample. Where the device has multisampled storage
//! images, the image has the pipeline's samples, so the device's `storageImageSampleCounts` must
//! hold that count (`VkProperty::StorageImageSampleCount`), as for the shader-side stencil's
//! shadow; otherwise the fetch is refused naming it. Elsewhere the image holds the samples as
//! planes (`StorageAttachmentView::SamplePlanes`), a single-sample image of one layer per sample,
//! which needs no sample count of storage images.
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

use crate::device::requirements::VkProperty;
use crate::device::DeviceProfile;
use crate::error::{
    Capability, Error, Feature, InternalError, InternalKind, RuntimeInput, Unsupported,
};
use crate::library::FunctionArtifact;
use crate::lower::fragment::{
    inputs_order, interlock_orders, kept_channels, storage_form, ColorUse, StorageSamples,
};
use crate::metal::format::ExtendedRange;
use crate::metal::state::color_write_mask;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::graphics::{StorageAttachmentUse, StorageAttachmentView};
use crate::pipeline::link::RenderLinkInputs;
use crate::pipeline::lower_inputs::{BlendEquation, FramebufferFetchInputs};
use crate::pipeline::stage_io::IoRole;

/// The attachments the fragment stage `fragment` of render inputs `r` reads in the storage form
/// on `p`, by color index (see the module docs).
pub fn storage_attachments(
    r: &RenderLinkInputs,
    fragment: Option<&FunctionArtifact>,
    p: &DeviceProfile,
) -> Result<Vec<StorageAttachmentUse>, Error> {
    let Some(fragment) = fragment else {
        return Ok(Vec::new());
    };
    let reads: Vec<(
        u32,
        crate::vk::Format,
        bool,
        bool,
        crate::vk::ColorComponentFlags,
        (Option<crate::pipeline::lower_inputs::BlendEquation>, bool),
    )> = fragment
        .facts
        .iter()
        .filter_map(|fact| match *fact {
            RuntimeFact::InterlockedFramebufferFetch {
                color,
                format,
                writes,
                early_tests,
                kept,
                blend,
                rounded,
            } => Some((color, format, writes, early_tests, kept, (blend, rounded))),
            _ => None,
        })
        .collect();
    if reads.is_empty() {
        return Ok(Vec::new());
    }
    if !interlock_orders(p) {
        return Err(Error::Internal(InternalError::new(
            InternalKind::Invariant,
            "an interlocked framebuffer fetch on a device that orders fetch otherwise",
        )));
    }
    let inputs = FramebufferFetchInputs::of(
        reads.iter().map(|&(color, ..)| color),
        &r.color_attachments,
        r.depth_format.is_some() || r.stencil_format.is_some(),
        r.alpha_to_coverage,
        r.max_vertex_amplification_count,
    );
    let multisampled = r.raster_sample_count > 1 && r.rasterization_enabled;
    // A write before the tests, which each draw's plan checks.
    let draw_tested = fragment.facts.iter().any(|f| {
        matches!(
            f,
            RuntimeFact::TestedStorageWrites {
                early_tests: false,
                ..
            }
        )
    });
    let mut out = Vec::with_capacity(reads.len());
    let mut multisampled_image = false;
    for (color, format, writes, early_tests, kept, (blend, rounded)) in reads {
        let math = match rounded {
            true => crate::lower::arith::extended_range::blend::BlendMath::Rounded,
            false => crate::lower::arith::extended_range::blend::BlendMath::Fast,
        };
        let use_ = ColorUse {
            writes,
            early_tests,
            draw_tested,
            math,
            logic_op: fragment.interface.io.logic_operation,
        };
        match storage_form(color, &inputs, multisampled, use_, p)? {
            Ok(view) if view.format == format && view.kept == kept && view.blend == blend => {
                multisampled_image |= view.samples == StorageSamples::Multisampled;
                let view_of = match view.samples {
                    StorageSamples::One | StorageSamples::Multisampled => {
                        StorageAttachmentView::Attachment
                    }
                    StorageSamples::Planes => StorageAttachmentView::SamplePlanes {
                        layers: r.raster_sample_count,
                    },
                };
                // The pipeline's own kept channels, not the class the function compiled under.
                let kept = match r.color_attachments.get(color as usize).copied().flatten() {
                    Some(a) => {
                        color_write_mask(kept_channels(a.pixel_format, a.write_mask, writes))?
                    }
                    None => return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch)),
                };
                out.push(StorageAttachmentUse {
                    color_index: color,
                    format,
                    writes,
                    formatless: view.formatless(),
                    view: view_of,
                    view_alias: view.uint_view.or(view.rgba_view),
                    kept,
                });
            }
            _ => return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch)),
        }
    }
    // The per-sample form's multisampled image has the pipeline's samples.
    if multisampled_image {
        let property = VkProperty::StorageImageSampleCount(r.raster_sample_count);
        if !p.fields().enabled.properties.contains(&property) {
            return Err(Unsupported::needs(
                Feature::FramebufferFetch,
                Capability::Property(property),
            )
            .into());
        }
    }
    Ok(out)
}

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
