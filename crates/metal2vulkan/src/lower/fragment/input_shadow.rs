//! The `half` shadow of an extended-range attachment on a device that orders framebuffer fetch
//! through rasterization-order input attachments (lavapipe).
//!
//! Metal's tile memory holds a `BGR10_XR`, `BGRA10_XR` or `RGB10A8_2P_XR10` color (and the `_sRGB`
//! twins) as `half` through a render pass and rounds it to the codes only at the store, so a blend
//! or a fetch onto what an earlier draw of the pass wrote sees that draw's `half` (`fetch`'s module
//! docs, "The `half` shadow"). Where the draw's render pass instance holds the attachment's shadow
//! (`LoweringInputs::extended_shadows`) on such a device, the attachment stays the subpass's: its
//! codes are read through its input attachment and written through the output variable, ordered
//! by rasterization-order attachment access, as without the shadow (`input_blend`, `color`). The
//! shadow is a second storage image of the attachment's extent in `Rgba16f`
//! (`ResourceKind::ExtendedShadow`, `fetch::declare_shadow`), one sample, or a multisampled image of
//! the pipeline's samples read and written at the invocation's `SampleId`. The function's body runs
//! inside the ordered pixel interlock (`fragment::wrap_entry`), which orders each fragment's read and
//! write of the shadow after every earlier fragment's of the pixel in primitive order, across the
//! draws of the instance, as the storage form's are. A `[[color(n)]]` read and a blend's destination
//! are the shadow's texel as Metal reads the tile (`extended_range::tile_read`), and every write
//! stores the value it computes there, unclamped, the blend's result or the unblended output's
//! `half` (the channels a write mask keeps as the shadow held them), beside the codes' encode,
//! unless the invocation was discarded. Every value the shadow holds is a `half`, so the device's
//! conversions both ways are exact, and the codes always hold the encode of the last `half` written
//! to each written channel.
//!
//! The device's depth and stencil tests run after the function, so a store of the shadow before
//! them would keep a fragment they drop and its codes do not: beside a depth or stencil attachment
//! the module declares early fragment tests, which drop the fragment before its codes and its shadow
//! alike, where that keeps Metal's answers (Metal's function declares them, or it discards nothing,
//! writes no other memory and returns no depth or stencil, on a driver that keeps an early-tests
//! function's image writes at the pipeline's samples). Every other shadowed function is refused
//! (`NotImplemented(FramebufferFetch)`), and so is one that returns a sample mask, one in a pipeline
//! with alpha to coverage (both drop samples of the codes after the function), one that loops over
//! its covered samples, and in a multisampled pipeline one that does not read the attachment (Metal
//! runs it once per pixel, where the shadow holds each sample): the runtime keeps such a draw's
//! colors on the codes (`render/extended_shadow.rs`), which it announces.

use std::collections::BTreeMap;

use crate::air::abi::{AbiInt, OutputRole, Stage as AbiStage};
use crate::device::profile::DriverWorkaround;
use crate::device::requirements::{FormatUsage, VkFeature};
use crate::device::DeviceProfile;
use crate::error::{Attachment, Capability, Error, Feature, RuntimeInput, Unsupported};
use crate::lower::arith::extended_range;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io;
use crate::lower::sync::memory_model;
use crate::metal::format::ExtendedRange;
use crate::metal::ColorWriteMask;
use crate::pipeline::sym::Stage;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{MemAccess, UnOp};
use crate::vk;

use super::{fetch, InputBlend, StorageSamples};

/// An extended-range attachment whose `half` shadow the draw's render pass instance holds, on a
/// device that fetches through input attachments (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputShadow {
    /// The attachment's form.
    pub form: ExtendedRange,
    /// How the shadow holds the pipeline's samples: one, or a multisampled image of them.
    pub samples: StorageSamples,
    /// The format's channels (red, green, blue, alpha) the function's write keeps as the shadow
    /// held them, those its write mask leaves out; `None` where the function does not write the
    /// attachment, and all four where its write stores none of them.
    pub kept: Option<[bool; 4]>,
}

impl InputShadow {
    /// Whether the function stores into the shadow: it writes some of the format's channels.
    pub fn stores(&self) -> bool {
        self.kept.is_some_and(|kept| !kept.iter().all(|&k| k))
    }
}

/// The shadow's format.
const SHADOW_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// What the device lacks for the shadow on `p` (see the module docs): the ordered pixel interlock,
/// fragment stores, the shadow's format as a storage image, and, `multisampled`, multisampled
/// storage images.
fn device_form(p: &DeviceProfile, multisampled: bool) -> Result<(), Unsupported> {
    let f = p.fields();
    let mut missing = Vec::new();
    if !f.fragment.pixel_interlock {
        missing.push(Capability::Feature(VkFeature::FragmentShaderPixelInterlock));
    }
    if !f.leaf.fragment_stores_and_atomics {
        missing.push(Capability::Feature(VkFeature::FragmentStoresAndAtomics));
    }
    if multisampled && !f.images.storage_multisample {
        missing.push(Capability::Feature(
            VkFeature::ShaderStorageImageMultisample,
        ));
    }
    if !p.supports_format(SHADOW_FORMAT, FormatUsage::StorageImage) {
        missing.push(Capability::Format {
            format: SHADOW_FORMAT,
            usage: FormatUsage::StorageImage,
        });
    }
    match missing.is_empty() {
        true => Ok(()),
        false => Err(Unsupported::needs_any_of(
            Feature::FramebufferFetch,
            vec![missing],
        )),
    }
}

/// Plan the shadows of the extended-range attachments the draw's instance holds
/// (`LoweringInputs::extended_shadows`) on a device that fetches through input attachments, before
/// the stage interface and after the blends over input attachments (`input_blend::plan`): each one
/// the function writes through its first output or reads, with its samples and the channels its
/// write keeps, the early tests beside a depth or stencil attachment, the pixel's `FragCoord` and
/// the invocation's sample (see the module docs).
pub(super) fn plan(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.stage != Stage::Fragment
        || cx.inputs.extended_shadows.is_empty()
        || !fetch::inputs_order(cx.profile)
    {
        return Ok(());
    }
    let multisampled = cx.inputs.multisampled;
    device_form(cx.profile, multisampled)?;
    let Some(inputs) = cx.inputs.framebuffer_fetch.clone() else {
        return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
    };
    let body = fetch::body_facts(cx);
    let read = fetch::colors_read(cx, &body.used)?;
    let outputs = &cx.entry.ep.outputs;
    let returns = |role: fn(&OutputRole) -> bool| outputs.iter().any(|o| role(&o.role));
    let exports = returns(|r| matches!(r, OutputRole::Depth(_) | OutputRole::Stencil));
    let sample_mask = returns(|r| matches!(r, OutputRole::SampleMask));
    let declared_early = matches!(
        cx.entry.ep.stage,
        AbiStage::Fragment {
            early_fragment_tests: true
        }
    );
    let refused = |color: u32| -> Error {
        let refused = Unsupported::not_implemented(Feature::FramebufferFetch);
        refused.at_attachment(Attachment::Color(color)).into()
    };
    let mut planned = BTreeMap::new();
    for &color in &cx.inputs.extended_shadows {
        let Some(form) = (cx.inputs.extended_range_colors.get(&color))
            .map(|e| e.form)
            .filter(|f| !f.padded() && !f.bfloat())
        else {
            return Err(refused(color));
        };
        let writes = outputs.iter().any(|o| {
            matches!(
                o.role,
                OutputRole::RenderTarget {
                    attachment: AbiInt::Const(a),
                    index: 0,
                    ..
                } if a == color
            )
        });
        let reads = read.contains(&color)
            || (cx.families.fragment.input_blends)
                .get(&color)
                .is_some_and(InputBlend::reads);
        if !(writes || reads) || sample_mask || inputs.alpha_to_coverage || (multisampled && !reads)
        {
            return Err(refused(color));
        }
        let kept = match writes {
            true => {
                let Some(a) = inputs.attachments.get(&color).copied().flatten() else {
                    return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
                };
                let kept = fetch::kept_channels(a.pixel_format, a.write_mask, true);
                Some(
                    [
                        ColorWriteMask::RED,
                        ColorWriteMask::GREEN,
                        ColorWriteMask::BLUE,
                        ColorWriteMask::ALPHA,
                    ]
                    .map(|c| kept.contains(c)),
                )
            }
            false => None,
        };
        let samples = match multisampled {
            true => StorageSamples::Multisampled,
            false => StorageSamples::One,
        };
        cx.families.arith.converts(form);
        planned.insert(
            color,
            InputShadow {
                form,
                samples,
                kept,
            },
        );
    }
    // Beside a depth or stencil attachment, the shadow's stores follow the tests only under early
    // tests (see the module docs).
    let stores = planned.values().any(InputShadow::stores);
    if inputs.depth_or_stencil && stores && !declared_early {
        let dropped = multisampled
            && cx
                .profile
                .fields()
                .workarounds
                .contains(&DriverWorkaround::EarlyTestsDropMultisampledImageWrites);
        if exports || body.discards || body.side_effects || dropped {
            let color = planned.keys().next().copied().unwrap_or_default();
            return Err(refused(color));
        }
        cx.families.fragment.fetch_early_tests = true;
    }
    fetch::plan_sample(cx)?;
    let coord = stage_io::frag_coord(cx);
    cx.families.fragment.fetch_coord = Some(coord);
    cx.descriptors.storage_images = (cx.descriptors.storage_images)
        .saturating_add(u32::try_from(planned.len()).unwrap_or(u32::MAX));
    cx.families.fragment.input_shadows = planned;
    Ok(())
}

/// Whether the function holds attachment `color`'s shadow on a device that fetches through input
/// attachments.
pub(crate) fn shadows(cx: &EmitCx<'_>, color: u32) -> Option<InputShadow> {
    cx.plan.families.fragment.input_shadows.get(&color).copied()
}

/// The tile value attachment `color`'s shadow holds at the invocation's pixel (and sample), read
/// inside the interlock: Metal's read of what the tile holds (`extended_range::tile_read`).
pub(crate) fn read(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
) -> BResult<SValue> {
    let Some(shadow) = shadows(cx, color) else {
        return Err(cx.fail(format!("attachment {color} holds no shadow")));
    };
    let (image, at, ops) = fetch::shadow_and_texel(cx, fb, color, shadow.samples)?;
    let mem = memory_model::coherence(cx.plan).texel(fb)?;
    let texel = fb.read(image, at, &ops, mem)?;
    extended_range::tile_read(fb, shadow.form, texel)
}

/// Store `v`, the `half`s a write of attachment `color` leaves in Metal's tile (a `vec4`), into
/// its shadow at the invocation's pixel (and sample), the channels the write keeps as the shadow
/// held them, unless the invocation was discarded. Nothing where the write stores none of the
/// format's channels.
pub(crate) fn store(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    v: SValue,
) -> BResult<()> {
    let Some(shadow) = shadows(cx, color) else {
        return Err(cx.fail(format!("attachment {color} holds no shadow")));
    };
    let Some(kept) = shadow.kept.filter(|_| shadow.stores()) else {
        return Ok(());
    };
    let v = match kept.iter().any(|&k| k) {
        true => {
            let held = read(cx, fb, color)?;
            let lanes: Vec<u32> = (kept.iter().zip(0..4u32))
                .map(|(&kept, lane)| match kept {
                    true => 4 + lane,
                    false => lane,
                })
                .collect();
            fb.vector_shuffle(v, held, &lanes)?
        }
        false => v,
    };
    let (image, at, ops) = fetch::shadow_and_texel(cx, fb, color, shadow.samples)?;
    let mem = memory_model::coherence(cx.plan).texel(fb)?;
    let Some(g) = cx.decl.families.fragment.discarded else {
        return fb.write(image, at, v, &ops, mem);
    };
    let p = fb.module().global_value(g)?;
    let discarded = fb.load(p, MemAccess::default())?;
    let kept = fb.unary(UnOp::LogicalNot, discarded)?;
    let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
        fb.write(image, at, v, &ops, mem)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), kept, arm, None)?;
    Ok(())
}

/// An upper estimate of the words the shadows add at `rets` returns: the object and its type, and
/// at each return the coordinate, a read where the write keeps channels, the guard, and the
/// write; and a read per `uses` of a `[[color(n)]]` input.
pub(super) fn words(cx: &SelectCx<'_>, uses: u64, rets: u64) -> u64 {
    const OBJECT: u64 = 40;
    const READ: u64 = 64;
    const WRITE: u64 = 96;
    let shadows = &cx.families.fragment.input_shadows;
    if shadows.is_empty() {
        return 0;
    }
    let reads = uses.saturating_mul(READ);
    shadows
        .values()
        .map(|s| {
            let write = match s.stores() {
                true => (READ + WRITE).saturating_mul(rets.max(1)),
                false => 0,
            };
            OBJECT.saturating_add(write)
        })
        .fold(reads, u64::saturating_add)
}
