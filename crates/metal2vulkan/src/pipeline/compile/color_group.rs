//! Render pipelines whose color attachments a device draws in groups ([`ColorGroup`]).
//!
//! A subpass, or a dynamic rendering, holds at most `maxColorAttachments` color attachments
//! (`VUID-VkSubpassDescription-colorAttachmentCount-00845`,
//! `VUID-VkRenderingInfo-colorAttachmentCount-06106`), and a fragment shader's outputs lie below
//! `maxFragmentOutputAttachments` locations (`VUID-RuntimeSpirv-Location-06272`). Metal's
//! pipelines take eight color attachments; a device at Vulkan's floor holds four. The lesser of
//! the two limits is a group's width `w`.
//!
//! **Grouping**. A pipeline's attachments with a pixel format, in the order of their indices,
//! fall into groups of `w`: group `g` holds the `g`-th `w` of them. Metal requires the pass's
//! attachments to match the pipeline's, so the pass's attachments group alike. A pipeline whose
//! attachments are all below `w` keeps every attachment at its own index and takes no group (the
//! common path, unchanged); one with at most `w` attachments, one of them at an index at or past
//! `w` (`color(0)` and `color(5)` on a device of four), takes one group, its attachments
//! renumbered below `w`; one with more takes more groups. A variant of the pipeline draws one
//! group (`VariantKey::color_group`).
//!
//! **Projection**. Group `g`'s variant compiles as the pipeline whose descriptor holds the group's
//! attachments alone, at locations `0..` ([`descriptor`]), and whose fragment function's color
//! indices, the `[[color(n)]]` outputs' and inputs', are renumbered by [`ColorGroup::location`]
//! ([`entry`]): a color of another group, or one without an attachment, lands past the group's,
//! where the projected descriptor has no attachment, so the function stores it nowhere, as plain
//! Metal stores a color without an attachment. Every attachment-indexed input, table and check of
//! the compile and link then speaks the variant's locations, and the variant is an ordinary
//! pipeline of its group's attachments: its variant key's choices of colors (the zeroed, dropped,
//! interlocked and shadowed ones) name the group's locations, as a bind plan of the variant over
//! its group's part of the pass asks for them. The grouping is the pipeline's own, whatever the
//! pass holds.
//!
//! **Invariance**. A draw of a pipeline of several groups is a draw of each group's variant, each
//! in a pass of its group's attachments over its own copy of the depth and stencil the pass
//! loads. Each group's colors are Metal's where every variant generates the same fragments, with
//! the same depth, stencil reference, coverage and discard. Vulkan's invariance appendix gives
//! that for "identical pipelines" alone ("`VkPipeline` objects that have been created with
//! identical SPIR-V binaries and identical state", Rule 4), and its Rule 2 lists the state whose
//! change has no side effects ("Color and depth/stencil attachment contents", "Write masks",
//! "Blend parameters", among others), whence "Fragment generation is invariant with respect to the
//! state values listed in Rule 2" (Corollary 1). The variants differ beyond those: in their color
//! attachments' formats and count, and in their fragment modules, which write other colors. What
//! can be made invariant is:
//!
//! - The vertex stage: every variant's is the same module (its inputs do not name the group), and
//!   its `Position` is decorated `Invariant`, as in every pipeline.
//! - The fragment stage's per-fragment outputs, `FragDepth`, `SampleMask` and
//!   `FragStencilRefEXT`: each variant keeps them, computed by the same expressions, and decorates
//!   them `Invariant`, SPIR-V's "expressions computing its value be computed invariantly with
//!   respect to other shaders computing the same expressions". `NoContraction` adds nothing for
//!   them: an `Invariant` output's expressions are computed alike in every variant, contraction
//!   included, and decorating the arithmetic would round it apart where Metal fuses it.
//!
//! What cannot: a discard has no output to decorate, and the device's rasterization of the
//! fixed-function depth is not covered across pipelines of other fragment modules and attachments.
//! So the variants record `PrecisionNote::ColorAttachmentGroups`, whose residual is that groups
//! past the first may differ at discard or depth edges if the driver compiles the variants
//! differently, and nothing broader.
//!
//! **Refusals**. What runs once per covered sample in Metal and would run once per group here is
//! refused where the pipeline takes more than one group, with the pipeline's own limit refusal
//! (`Unsupported{InterfaceComponents}` at `MaxColorAttachments`, or `MaxFragmentOutputAttachments`
//! where that is the lesser): a vertex or fragment stage that writes memory, stores or atomics
//! (`StageInterface::writes_memory`, link); a tessellation pipeline (link); alpha to coverage,
//! which reads color 0's alpha in every group ([`group`]); a `[[color(n)]]` input of another
//! group that the function reads, imageblock data and raster order groups ([`entry`]). A color
//! input of another group that the specialized body never uses (its read folded away by the
//! function's constants) reads nothing there, and is not refused. Imageblocks are refused under any
//! renumbering: an implicit imageblock's members name the attachments they alias by index. A mesh
//! render pipeline and a tile pipeline take no groups and keep the limit refusals.

use crate::air;
use crate::air::abi::{AbiInt, OutputRole, ParamRole, Stage as AbiStage};
use crate::cache::VariantKey;
use crate::device::requirements::VkLimit;
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::descriptor::RenderPipelineDescriptor;
use crate::pipeline::lower_inputs::ColorGroup;

/// The color group variant `v` of render pipeline `d` draws on `p` (the module docs): `None`
/// where each of `d`'s attachments is below the group width. A pipeline of more than one group
/// under alpha to coverage, or one renumbering color 0, is the limit refusal; a variant of a
/// group the pipeline does not have is no plan's (an invariant).
pub(crate) fn group(
    d: &RenderPipelineDescriptor,
    v: &VariantKey,
    p: &DeviceProfile,
) -> Result<Option<ColorGroup>, Error> {
    let present: Vec<u32> = (0u32..)
        .zip(&d.color_attachments)
        .filter_map(|(i, a)| a.map(|_| i))
        .collect();
    let Some(g) = ColorGroup::of(present, width(p), v.color_group) else {
        return match v.color_group {
            0 => Ok(None),
            _ => Err(invariant("a color group of a pipeline that takes none")),
        };
    };
    if v.color_group >= g.groups() {
        return Err(invariant("a color group past the pipeline's last"));
    }
    // Metal's coverage reads color 0's alpha, attached or not, which another group's variant
    // does not write, and a renumbering moves.
    if d.alpha_to_coverage && (g.groups() > 1 || g.location(0) != 0) {
        return Err(refusal(&g, p));
    }
    Ok(Some(g))
}

/// `d` as group `g`'s variant compiles it: the group's attachments alone, attachment
/// `g.colors()[l]` at index `l`.
pub(crate) fn descriptor(d: &RenderPipelineDescriptor, g: &ColorGroup) -> RenderPipelineDescriptor {
    let mut out = d.clone();
    out.color_attachments = Default::default();
    for &color in g.colors() {
        let at = g.location(color) as usize;
        if let (Some(slot), Some(a)) = (
            out.color_attachments.get_mut(at),
            d.color_attachments.get(color as usize),
        ) {
            *slot = *a;
        }
    }
    out
}

/// The specialized fragment entry `spec` as group `g`'s variant compiles it: each color index its
/// outputs and `[[color(n)]]` inputs name renumbered by [`ColorGroup::location`]. Imageblock data
/// is the limit refusal on `p`, and so, where the pipeline takes more than one group, is a color
/// input of another group that the body uses and a raster order group. Any other entry as it is.
pub(crate) fn entry(
    mut spec: air::Entry,
    g: &ColorGroup,
    p: &DeviceProfile,
) -> Result<air::Entry, Error> {
    if !matches!(spec.ep.stage, AbiStage::Fragment { .. }) {
        return Ok(spec);
    }
    let split = g.groups() > 1;
    let refused = || refusal(g, p);
    // A color input the specialized body never uses reads no attachment, of whichever group.
    let used = super::params_used(&spec);
    for b in &mut spec.ep.params {
        let read = used.contains(&b.param);
        match &mut b.role {
            ParamRole::Imageblock(_) => return Err(refused()),
            ParamRole::ColorInput {
                index,
                raster_order_group,
            } => {
                if split && raster_order_group.is_some() {
                    return Err(refused());
                }
                if let AbiInt::Const(color) = *index {
                    if split && read && !g.colors().contains(&color) {
                        return Err(refused());
                    }
                    *index = AbiInt::Const(g.location(color));
                }
            }
            ParamRole::Buffer(a) | ParamRole::ArgumentBuffer(a) | ParamRole::Threadgroup(a)
                if split && a.raster_order_group.is_some() =>
            {
                return Err(refused());
            }
            ParamRole::Texture(t) if split && t.raster_order_group.is_some() => {
                return Err(refused());
            }
            _ => {}
        }
    }
    for o in &mut spec.ep.outputs {
        match &mut o.role {
            OutputRole::ImageblockData(_) => return Err(refused()),
            OutputRole::RenderTarget {
                attachment,
                raster_order_group,
                ..
            } => {
                if split && raster_order_group.is_some() {
                    return Err(refused());
                }
                if let AbiInt::Const(color) = *attachment {
                    *attachment = AbiInt::Const(g.location(color));
                }
            }
            _ => {}
        }
    }
    Ok(spec)
}

/// The attachments a group holds on `p` ([`ColorGroup::width_on`]).
fn width(p: &DeviceProfile) -> u32 {
    ColorGroup::width_on(p)
}

/// The refusal of a pipeline of color group `g` on `p` whose groups cannot keep Metal's answer:
/// the limit the pipeline's attachments pass, as a pipeline the device cannot draw in groups is
/// refused.
pub(crate) fn refusal(g: &ColorGroup, p: &DeviceProfile) -> Error {
    let l = &p.fields().limits;
    let limit = match l.max_fragment_output_attachments < l.max_color_attachments {
        true => VkLimit::MaxFragmentOutputAttachments,
        false => VkLimit::MaxColorAttachments,
    };
    let required = g.present.last().map_or(0, |&last| u64::from(last) + 1);
    Unsupported::needs(
        Feature::InterfaceComponents,
        Capability::Limit {
            limit,
            required,
            available: u64::from(width(p)),
        },
    )
    .into()
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        detail.to_owned(),
    ))
}
