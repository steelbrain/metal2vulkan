//! Raster order groups: a fragment function that names a raster order group on any parameter it
//! uses, output it reads, or argument-buffer member runs its whole body inside a fragment-interlock
//! critical section.
//!
//! Metal orders a raster order group's accesses between fragments that cover the same pixel (the
//! same sample at sample rate). AIR states the group only as `air.raster_order_group` metadata on
//! buffers, textures, color inputs and outputs, and imageblock members; the accesses themselves
//! are ordinary loads and stores. The lowering therefore orders the whole body, which is a
//! superset of every group's accesses (groups only reduce the ordering Metal must provide):
//!
//! ```text
//! main:  OpBeginInvocationInterlockEXT
//!        OpFunctionCall body
//!        OpEndInvocationInterlockEXT
//!        if discarded { OpKill | OpTerminateInvocation }      // only when a discard can set it
//!        OpReturn
//! ```
//!
//! No terminator may run between `Begin` and `End`, so inside the body a terminate-form discard
//! is `discarded ← true; OpReturn`, a predicated one only sets `discarded`, and the body's returns
//! do not kill; `main` kills after `End`. A demoted invocation stays in the body and ends it as
//! usual.
//!
//! **Color outputs alone**. A group's color output that the function does not read (no color input
//! of its attachment) asks for no ordering beyond what a render target already has: Vulkan performs
//! a fragment's color writes and blending per sample in rasterization order, as Metal does. Such an
//! output alone does not interlock the function, so a function whose groups hold nothing else
//! compiles on every device, the ones without an interlock included.
//!
//! **Unused inputs**. A group on a parameter the body never uses, such as an input whose reads the
//! function's constants fold away, orders no access, and neither does a member of a layout only
//! such a parameter reaches, nor a color output whose attachment only such a color input reads. None
//! of them interlocks the function: plain Metal makes and draws such a function where the read is
//! dead as where it is live (`metal2vulkan-conformance/probes/dead_ordered_and_unattached_reads.m`,
//! an M3 Max under macOS 27.0.1).
//!
//! **Mode** ([`mode`]). `PixelInterlockOrderedEXT` where the profile has
//! `fragmentShaderPixelInterlock`: it orders every invocation covering the pixel, which covers
//! Metal's pixel- and sample-rate ordering. `SampleInterlockOrderedEXT` orders only invocations
//! covering the same sample, which equals Metal's ordering only when the module itself runs at
//! sample rate (it reads `SampleId` or has a sample-interpolated input); a pixel-rate module on a
//! multisampled target would leave two fragments covering disjoint samples of one pixel unordered,
//! so there it is not used. Otherwise the function is refused
//! `Unsupported{RasterOrderGroups, AnyOf[[FragmentShaderPixelInterlock], ...]}`.

use std::collections::BTreeSet;

use crate::air::abi::{MemberKind, OutputRole, ParamRole, StructRef};
use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::select::SelectCx;
use crate::lower::stage_io::IoObject;
use crate::pipeline::sym::Stage;
use crate::spirv::ir::{ExecMode, IoDir, Sampling};

/// The raster order group a parameter role names, if any.
fn role_group(role: &ParamRole) -> Option<u32> {
    match role {
        ParamRole::Buffer(a) | ParamRole::ArgumentBuffer(a) | ParamRole::Threadgroup(a) => {
            a.raster_order_group
        }
        ParamRole::Texture(t) => t.raster_order_group,
        ParamRole::ColorInput {
            raster_order_group, ..
        } => *raster_order_group,
        _ => None,
    }
}

/// The layouts parameter role `role` names directly: a buffer's pointee, an imageblock's cell
/// and master.
fn role_layouts(role: &ParamRole) -> [Option<StructRef>; 2] {
    match role {
        ParamRole::Buffer(a) | ParamRole::ArgumentBuffer(a) | ParamRole::Threadgroup(a) => {
            [a.layout, None]
        }
        ParamRole::Imageblock(ib) => [ib.layout, ib.master],
        _ => [None, None],
    }
}

/// Whether the function names a raster order group that needs the interlock: on a parameter its
/// body uses, a color output whose attachment the function reads (a used color input of the same
/// attachment), or an argument-buffer or imageblock member of a layout its used parameters and its
/// outputs reach. `e.structs` holds every layout of the module, so a layout only another function
/// names does not count. A group's color output the function does not read needs none, and neither
/// does a group on a parameter the body never uses ([module docs](self)).
pub(super) fn declared(cx: &SelectCx<'_>) -> bool {
    let e = cx.entry;
    let used = crate::lower::users::params_used(cx.body);
    let params = || e.ep.params.iter().filter(|b| used.contains(&b.param));
    let read = |attachment| {
        params()
            .any(|b| matches!(b.role, ParamRole::ColorInput { index, .. } if index == attachment))
    };
    let named = params().any(|b| role_group(&b.role).is_some());
    let outputs = e.ep.outputs.iter().any(|o| match o.role {
        OutputRole::RenderTarget {
            attachment,
            raster_order_group: Some(_),
            ..
        } => read(attachment),
        _ => false,
    });
    if named || outputs {
        return true;
    }
    let mut stack: Vec<StructRef> = params()
        .flat_map(|b| role_layouts(&b.role))
        .chain(e.ep.outputs.iter().flat_map(|o| match o.role {
            OutputRole::ImageblockData(ib) => [ib.layout, ib.master],
            _ => [None, None],
        }))
        .flatten()
        .collect();
    let mut seen = BTreeSet::new();
    while let Some(r) = stack.pop() {
        if !seen.insert(r) {
            continue;
        }
        let Some(s) = usize::try_from(r.0).ok().and_then(|i| e.structs.get(i)) else {
            continue;
        };
        for m in &s.members {
            if m.raster_order_group.is_some() {
                return true;
            }
            stack.extend(m.layout);
            if let MemberKind::Indirect(a) = &m.kind {
                if role_group(&a.role).is_some() {
                    return true;
                }
                stack.extend(role_layouts(&a.role).into_iter().flatten());
            }
        }
    }
    false
}

/// Whether the planned stage interface makes the module run at sample rate: it reads `SampleId`
/// or declares a sample-interpolated input.
fn sample_rate(objects: &[IoObject]) -> bool {
    objects.iter().any(|o| match o {
        IoObject::Builtin { builtin, dir, .. } => {
            *builtin == crate::spv::BuiltIn::SampleId && *dir == IoDir::Input
        }
        IoObject::User { dir, deco, .. } => {
            *dir == IoDir::Input && deco.sampling == Sampling::Sample
        }
    })
}

/// The interlock mode of a function in a raster order group, or its refusal. Called once the
/// stage interface is planned (`fragment::finish`).
pub(super) fn mode(cx: &SelectCx<'_>) -> Result<ExecMode, Error> {
    if cx.stage != Stage::Fragment {
        // Metal orders raster order groups between fragments only.
        return Err(Unsupported::not_implemented(Feature::RasterOrderGroups).into());
    }
    let f = cx.profile.fields().fragment;
    let per_sample = sample_rate(&cx.families.stage_io.objects);
    if f.pixel_interlock {
        return Ok(ExecMode::PixelInterlockOrdered);
    }
    if f.sample_interlock && per_sample {
        return Ok(ExecMode::SampleInterlockOrdered);
    }
    let pixel = vec![Capability::Feature(VkFeature::FragmentShaderPixelInterlock)];
    let alternatives = if per_sample {
        vec![
            pixel,
            vec![Capability::Feature(
                VkFeature::FragmentShaderSampleInterlock,
            )],
        ]
    } else {
        vec![pixel]
    };
    Err(Unsupported::needs_any_of(Feature::RasterOrderGroups, alternatives).into())
}
