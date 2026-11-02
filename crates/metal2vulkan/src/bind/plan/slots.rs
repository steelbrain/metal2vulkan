//! Buffer-slot planning shared by compute and render plans: the Metal table bounds, one
//! [`PlannedBuffer`] per statically used buffer slot, the `BufferRangeLimit` variant a shape's
//! length classes require, and the `uses` of bound buffers and used resources.

use std::collections::BTreeMap;

use super::{AccessClass, BoundKind, ByteRange, PlannedBuffer, ResourceUse, UseTarget};
use crate::bind::state::{BindShape, BufferShape, TextureShape, UsedClass, UsedShape};
use crate::device::profile::Limits;
use crate::device::requirements::VkFeature;
use crate::device::{mechanism, DeviceProfile};
use crate::error::{BindError, Capability, Feature, Unsupported};
use crate::metal;
use crate::pipeline::artifact::RuntimeDependency;
use crate::pipeline::facts::{Mechanism, RuntimeFact, UniformViewChoice};
use crate::pipeline::interface::SlotInterface;
use crate::pipeline::interface::{AccessSet, BufferForm, BufferSlot, Window};
use crate::pipeline::layout::{BindingSource, PipelineLayout};
use crate::pipeline::sym::{
    PushField, SampledType, SlotClass, SlotRef, SnapshotForm, SpecKey, Stage,
};
use crate::vk;

use super::compute::internal;
use super::resources::ResourcePlanner;

/// Metal's argument-table sizes: `buffers[31]` (and `threadgroup_lengths[31]`), `textures[128]`,
/// `samplers[16]`.
pub(crate) const MAX_BUFFERS: u32 = 31;
pub(crate) const MAX_TEXTURES: u32 = 128;
pub(crate) const MAX_SAMPLERS: u32 = 16;

/// Every slot index of `shape` inside Metal's argument tables.
pub(crate) fn check_slots(shape: &BindShape) -> Result<(), BindError> {
    for (&stage, s) in &shape.stages {
        let tables = [
            (SlotClass::Buffer, MAX_BUFFERS, s.buffers.keys().max()),
            (SlotClass::Texture, MAX_TEXTURES, s.textures.keys().max()),
            (SlotClass::Sampler, MAX_SAMPLERS, s.samplers.keys().max()),
            // A threadgroup-memory index names no argument slot; it is reported with the buffer
            // table's class, whose size it shares.
            (
                SlotClass::Buffer,
                MAX_BUFFERS,
                s.threadgroup_lengths.keys().max(),
            ),
        ];
        for (class, max, top) in tables {
            if let Some(&index) = top.filter(|&&i| i >= max) {
                return Err(BindError::SlotOutOfRange {
                    slot: SlotRef {
                        stage,
                        class,
                        index,
                    },
                });
            }
        }
    }
    Ok(())
}

/// Where each layout binding sits, by source.
pub(crate) fn bindings(layout: &PipelineLayout) -> BTreeMap<BindingSource, (u32, u32)> {
    let mut out = BTreeMap::new();
    for set in &layout.sets {
        for b in &set.bindings {
            out.insert(b.source, (set.set, b.binding));
        }
    }
    out
}

impl BoundKind {
    /// What `shape` binds at a buffer slot.
    pub(crate) fn of(shape: Option<&BufferShape>) -> BoundKind {
        match shape {
            None => BoundKind::Unbound,
            Some(&BufferShape::Buffer {
                device_address,
                length_class,
                ..
            }) => BoundKind::Buffer {
                device_address,
                length_class,
            },
            Some(BufferShape::Bytes { .. }) => BoundKind::Bytes,
        }
    }
}

/// The plan of buffer slot `slot` with interface `b` and what the shape binds there: its descriptor
/// binding (descriptor forms), the view's descriptor base alignment and range limit, and a
/// `MissingDeviceAddress` refusal for a device-address slot bound without an address. The base
/// alignment is `max(min*BufferOffsetAlignment, 16)` (`views::base_alignment`): a descriptor byte
/// is then congruent to its Metal address modulo 16, which a `uvec4` view of the slot relies on.
pub(crate) fn plan_buffer(
    slot: SlotRef,
    b: &BufferSlot,
    shape: Option<&BufferShape>,
    limits: &Limits,
    bindings: &BTreeMap<BindingSource, (u32, u32)>,
) -> Result<PlannedBuffer, BindError> {
    let bound = BoundKind::of(shape);
    let (min_offset, max_range) = if b.form.uniform_view() {
        (
            limits.min_uniform_buffer_offset_alignment,
            u64::from(limits.max_uniform_buffer_range),
        )
    } else {
        (
            limits.min_storage_buffer_offset_alignment,
            u64::from(limits.max_storage_buffer_range),
        )
    };
    let alignment = crate::memory::views::base_alignment(min_offset);
    let binding = match b.form {
        BufferForm::Descriptor(_) => Some(
            bindings
                .get(&BindingSource::Buffer(slot))
                .copied()
                .ok_or_else(|| internal(format!("descriptor slot {slot:?} has no binding")))?,
        ),
        BufferForm::Psb => {
            if matches!(
                bound,
                BoundKind::Bytes
                    | BoundKind::Buffer {
                        device_address: false,
                        ..
                    }
            ) {
                return Err(BindError::MissingDeviceAddress { slot });
            }
            None
        }
    };
    Ok(PlannedBuffer {
        slot,
        window: b.window,
        form: b.form,
        access: b.access,
        binding,
        alignment,
        max_range,
        bound,
        disjoint: false,
    })
}

/// The `SpecKey::RangeAtBindOffset` of buffer slot `b`, bound as `shape` states: whether the
/// descriptor range starts at its bind offset (`instantiate::starts_at_bind`, under the view's
/// descriptor base alignment, as [`plan_buffer`] takes it): inline bytes, at offset 0 of the
/// consumer's aligned upload; a buffer, at an offset whose alignment class
/// (`BufferShape::Buffer::offset_alignment`: `BindState::view_base_alignment` where the runtime
/// tells it apart at the slot and it divides the offset, else a power of two up to 4) the base
/// alignment divides. A slot bound to nothing takes false, the form that reads at the residual:
/// its residual is 0 (`instantiate::slot_field`), so either form reads the Metal offset alone of
/// the same descriptor, and false keeps the key of a draw that leaves the slot unbound the key
/// that reads every such slot at its residual, which a consumer can make before any draw binds
/// it (the runtime's creation, `pipeline/predicted.rs`).
pub(crate) fn range_at_bind_offset(
    b: &BufferSlot,
    shape: Option<&BufferShape>,
    limits: &Limits,
) -> bool {
    let min = match b.form.uniform_view() {
        true => limits.min_uniform_buffer_offset_alignment,
        false => limits.min_storage_buffer_offset_alignment,
    };
    let offset = match shape {
        Some(&BufferShape::Buffer {
            offset_alignment, ..
        }) => u64::from(offset_alignment),
        Some(BufferShape::Bytes { .. }) => 0,
        None => return false,
    };
    let alignment = crate::memory::views::base_alignment(min);
    crate::bind::instantiate::starts_at_bind(offset, b.window, alignment)
}

/// The value of `key` for a draw or dispatch of `shape`, whose resources `resources` plans, where
/// the bound resources decide it: a slot's presence, a bound sampler's words, a bound texture's
/// class, swizzle, sparseness and view axes. It is what a specialization constant of `key` is set
/// to, the one place a plan computes it. `None` for a key the bound resources do not decide (a
/// threadgroup's, a tile's, a pass's).
pub(crate) fn bound_value(
    key: SpecKey,
    shape: &BindShape,
    resources: &ResourcePlanner<'_>,
) -> Result<Option<u64>, BindError> {
    Ok(Some(match key {
        SpecKey::Presence(slot) => u64::from(present(shape, slot)),
        SpecKey::SamplerState { slot, fields } => resources.sampler_state(slot, fields),
        SpecKey::SamplerMinLevel(slot) => resources.sampler_min_level(slot),
        SpecKey::SamplerAnisotropy(slot) => resources.sampler_anisotropy(slot),
        SpecKey::NearestPixelSampler(slot) => resources.nearest_pixel_sampler(slot),
        // An unbound slot, or one bound to a placeholder format, reads its neutral: its
        // out-of-bounds texel is all zero.
        SpecKey::OutOfBoundsAlpha(slot) => bound_texture(shape, slot)
            .and_then(TextureShape::out_of_bounds_alpha)
            .unwrap_or(0),
        SpecKey::BicubicTexels(slot) => resources.bicubic_texels(slot)?,
        SpecKey::TexelFormat(slot) => resources.texel_format(slot),
        SpecKey::UnmeasuredTexels(slot) => resources.unmeasured_texels(slot),
        SpecKey::PlainHalfTexels(slot) => resources.plain_half_texels(slot),
        SpecKey::TextureSwizzle(slot) => resources.texture_swizzle(slot),
        SpecKey::SparseTexture(slot) => sparse_texture(shape, slot),
        SpecKey::ConstantLanes(slot) => constant_lanes(shape, slot),
        SpecKey::OffsetBaseAxes(slot) => offset_base_axes(shape, slot),
        SpecKey::RangeAtBindOffset(slot) => match resources.slots.get(&slot) {
            Some(SlotInterface::Buffer(b)) => u64::from(range_at_bind_offset(
                b,
                bound_buffer(shape, slot),
                &resources.profile.fields().limits,
            )),
            _ => 0,
        },
        _ => return Ok(None),
    }))
}

/// Record in `resources`' template the value of each key `push` places as a push field
/// (`PushField::Specialized`), the value its specialization constant would take
/// ([`bound_value`]), which instantiation writes there (`instantiate::slot_field`). A key the
/// bound resources do not decide is not one a function pushes: an internal error.
pub(crate) fn record_specialized(
    push: &super::PushTemplate,
    shape: &BindShape,
    resources: &mut ResourcePlanner<'_>,
) -> Result<(), BindError> {
    for placed in &push.fields {
        let PushField::Specialized(key) = placed.field else {
            continue;
        };
        let value = bound_value(key, shape, resources)?.ok_or_else(|| {
            internal(format!("a pushed {key:?}, which no bound resource decides"))
        })?;
        let value = u32::try_from(value)
            .map_err(|_| internal(format!("a pushed {key:?} of {value}, past a push word")))?;
        let recorded = &mut resources.template.specialized;
        if !recorded.iter().any(|(k, _)| *k == key) {
            recorded.push((key, value));
        }
    }
    Ok(())
}

/// Whether `shape` binds anything at `slot`.
fn present(shape: &BindShape, slot: SlotRef) -> bool {
    shape
        .stages
        .get(&slot.stage)
        .is_some_and(|s| match slot.class {
            SlotClass::Buffer => s.buffers.contains_key(&slot.index),
            SlotClass::Texture => s.textures.contains_key(&slot.index),
            SlotClass::Sampler => s.samplers.contains_key(&slot.index),
        })
}

/// The `SpecKey::SparseTexture` of `slot` for `shape`: 1 where the slot binds a sparse texture
/// (`TextureFacts::sparse`), 0 where it binds one that is not sparse or nothing.
fn sparse_texture(shape: &BindShape, slot: SlotRef) -> u64 {
    u64::from(bound_texture(shape, slot).is_some_and(TextureShape::sparse))
}

/// The `SpecKey::ConstantLanes` of `slot` for `shape`: the lanes the bound sparse texture's format
/// and swizzle make constants (`TextureShape::constant_lanes`), 0 where the slot binds a texture
/// that is not sparse or nothing.
fn constant_lanes(shape: &BindShape, slot: SlotRef) -> u64 {
    bound_texture(shape, slot).map_or(0, TextureShape::constant_lanes)
}

/// The `SpecKey::OffsetBaseAxes` of `slot` for `shape`: the plane axes on which the bound view's
/// parent's base-level size, which Metal's texel-offset rule reads, is not a power of two where
/// the view's own is (`TextureShape::offset_base_axes`); 0 where the slot binds any other texture
/// or nothing.
fn offset_base_axes(shape: &BindShape, slot: SlotRef) -> u64 {
    bound_texture(shape, slot).map_or(0, TextureShape::offset_base_axes)
}

/// The texture slots whose function decodes an 8-bit sRGB texel (`RuntimeFact::SrgbTexture`) at
/// which `shape` binds a texture of such a format that `profile`'s device holds as its UNORM twin
/// (`metal::texel_format::srgb_twin_on`: a format it cannot sample), whose views read the codes:
/// the plan's `VariantKey::srgb_textures`. So is each slot of `cross` that the variant reads in
/// the float class (an integer slot), whose accesses there take `float` results, which the codes
/// serve as a float slot's: the variant's compile states `SrgbTexture` there. A slot bound in an
/// sRGB format the device samples reads the device's decode and joins nothing. Derived from the
/// device's format features alone, as the runtime's image plan holds the twin; the variant's
/// compile states the same facts, so its plan for the same shape states the same key.
pub(crate) fn srgb_variant(
    profile: &DeviceProfile,
    dependencies: &[RuntimeDependency],
    shape: &BindShape,
    cross: &BTreeMap<SlotRef, SampledType>,
) -> std::collections::BTreeSet<SlotRef> {
    let read_float = cross
        .iter()
        .filter(|(_, ty)| **ty == SampledType::Float)
        .map(|(slot, _)| *slot);
    dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::SrgbTexture { slot } => Some(slot),
            _ => None,
        })
        .chain(read_float)
        .filter(|slot| {
            matches!(bound_texture(shape, *slot), Some(TextureShape::Texture { pixel_format, .. })
                if metal::texel_format::srgb_twin_on(*pixel_format, profile))
        })
        .collect()
}

/// The form each of the snapshot slots `slots` keeps its texels in (`VariantKey::snapshot_textures`,
/// `SnapshotForm::of`): by the bound texture's sampled view, which the snapshot's copy fetches
/// through, and the slot's declared sampled type. [`SnapshotForm::Bits`] at a slot the variant
/// reads in another class (`cross`), or one whose sampled view the device has no format for (an
/// 8-bit sRGB format held as its UNORM twin among them). The variant's compile reads the same
/// key, so its plan for the same shape states the same forms.
pub(crate) fn snapshot_forms(
    profile: &DeviceProfile,
    slots: &std::collections::BTreeSet<SlotRef>,
    interface: &BTreeMap<SlotRef, SlotInterface>,
    shape: &BindShape,
    cross: &BTreeMap<SlotRef, SampledType>,
) -> BTreeMap<SlotRef, SnapshotForm> {
    let form = |slot: SlotRef| {
        let Some(SlotInterface::Texture(t)) = interface.get(&slot) else {
            return SnapshotForm::Bits;
        };
        let Some(TextureShape::Texture { pixel_format, .. }) = bound_texture(shape, slot) else {
            return SnapshotForm::Bits;
        };
        if cross.contains_key(&slot) {
            return SnapshotForm::Bits;
        }
        let usages: std::collections::BTreeSet<_> =
            [crate::device::requirements::FormatUsage::SampledImage].into();
        let Ok(m) = metal::format::vk_format(*pixel_format, &usages, profile) else {
            return SnapshotForm::Bits;
        };
        SnapshotForm::of(m.format, t.sampled_type)
    };
    slots.iter().map(|&slot| (slot, form(slot))).collect()
}

/// The sampled texture slots whose function reads their texels in any numeric class
/// (`RuntimeFact::CrossClassReads`) at which `shape` binds a texture whose format's class is not
/// the slot's declared one and which the variant reads in that class
/// (`resources::cross_class`), each with that class: the plan's `VariantKey::cross_class`. The
/// variant's compile states the same facts, and its interface the same declared types, so its
/// plan for the same shape states the same key.
pub(crate) fn cross_variant(
    dependencies: &[RuntimeDependency],
    slots: &BTreeMap<SlotRef, SlotInterface>,
    shape: &BindShape,
) -> BTreeMap<SlotRef, SampledType> {
    dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::CrossClassReads { slot } => Some(slot),
            _ => None,
        })
        .filter_map(|slot| {
            let Some(SlotInterface::Texture(t)) = slots.get(&slot) else {
                return None;
            };
            let bound = bound_texture(shape, slot)?;
            // A storage slot whose integer texels take the bound format's signedness reads an
            // integer texture of either signedness in its own class: only a float one crosses, as
            // the binding sees it (`resources::plain_texture`).
            let numeric = dependencies
                .iter()
                .any(|d| d.fact == RuntimeFact::StorageTextureNumericType { slot });
            let t = match numeric {
                true => super::resources::bound_integer_type(*t, Some(bound)),
                false => *t,
            };
            let class = super::resources::cross_class(&t, bound)?;
            Some((slot, class))
        })
        .collect()
}

/// The texture slots whose function reads a two-plane YCbCr texture only as its in-place form
/// reads it (`RuntimeFact::YCbCrInPlace`) at which `shape` binds a texture whose planes are read
/// in place (`TextureShape::Texture::ycbcr_in_place`), and each runtime sampler the function
/// samples the slot through states a filter the form reads ([`in_place_word`]): the plan's
/// `VariantKey::ycbcr_in_place`. Any other slot binds the texture's image. The variant's compile
/// states the same facts, so its plan for the same shape states the same key.
pub(crate) fn ycbcr_in_place_variant(
    dependencies: &[RuntimeDependency],
    shape: &BindShape,
) -> std::collections::BTreeSet<SlotRef> {
    dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::YCbCrInPlace { slot, samplers, .. } => Some((slot, samplers)),
            _ => None,
        })
        .filter(|&(slot, samplers)| {
            let in_place = matches!(
                bound_texture(shape, slot),
                Some(TextureShape::Texture {
                    ycbcr_in_place: true,
                    ..
                })
            );
            let stage = shape.stages.get(&slot.stage);
            let words = (0..32).filter(|n| samplers >> n & 1 != 0).map(|n| {
                stage
                    .and_then(|s| s.samplers.get(&n))
                    .map_or(0, |s| s.word.0)
            });
            in_place && words.into_iter().all(in_place_word)
        })
        .map(|(slot, _)| slot)
        .collect()
}

/// The slots of `in_place` (the plan's `VariantKey::ycbcr_in_place`) whose function's every sample
/// the device's own sampler Y′CbCr conversion takes (`RuntimeFact::YCbCrInPlace`'s `device`), at
/// which `shape` binds a texture whose planes are also held as an image it samples
/// (`TextureShape::Texture::ycbcr_planes_sampler`), and each runtime sampler the function samples
/// the slot through addresses no border on either plane axis ([`planes_word`]), each with that
/// conversion: the plan's `VariantKey::ycbcr_planes_sampled`. The variant's compile states the
/// same facts, so its plan for the same shape states the same key.
pub(crate) fn ycbcr_planes_sampled_variant(
    dependencies: &[RuntimeDependency],
    shape: &BindShape,
    in_place: &std::collections::BTreeSet<SlotRef>,
) -> BTreeMap<SlotRef, crate::bind::YCbCrPlanesSampler> {
    dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::YCbCrInPlace {
                slot,
                samplers,
                device: true,
            } if in_place.contains(&slot) => Some((slot, samplers)),
            _ => None,
        })
        .filter_map(|(slot, samplers)| {
            let sampler = match bound_texture(shape, slot) {
                Some(&TextureShape::Texture {
                    ycbcr_planes_sampler: Some(sampler),
                    ..
                }) => sampler,
                _ => return None,
            };
            let stage = shape.stages.get(&slot.stage);
            let mut words = (0..32).filter(|n| samplers >> n & 1 != 0).map(|n| {
                stage
                    .and_then(|s| s.samplers.get(&n))
                    .map_or(0, |s| s.word.0)
            });
            words.all(planes_word).then_some((slot, sampler))
        })
        .collect()
}

/// Whether a runtime sampler of word `word` samples a two-plane YCbCr texture in a way the device's
/// own sampler Y′CbCr conversion of its planes' image takes: no clamp-to-zero or border color on
/// the `s` or `t` axis, which a 4:2:0 texture reads as the border where that sampler clamps to the
/// edge (every other mode a 4:2:0 texture samples as clamp-to-edge). Its filter and coordinate
/// space are the function's to apply.
pub(crate) fn planes_word(word: u32) -> bool {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::SamplerAddressMode as M;
    let border = [M::ClampToZero.as_raw(), M::ClampToBorderColor.as_raw()];
    (0..2).all(|axis| {
        let mode = (word & SamplerWord::ADDRESS[axis]) >> SamplerWord::ADDRESS_SHIFT[axis];
        !border.contains(&mode)
    })
}

/// Whether a runtime sampler of word `word` samples a two-plane YCbCr texture in a way its
/// in-place form reads: one filter for minification and magnification (nearest or linear), no
/// anisotropy, no cubic filter, no comparison and the weighted-average reduction. Its mip filter
/// and level clamps read the texture's one level, as Metal's sample of it reads it whatever its
/// mip filter (`lower::sampler::in_place`); every address mode, border color and coordinate
/// space is the form's to address. An unbound slot's word is 0, a nearest sampler's.
pub(crate) fn in_place_word(word: u32) -> bool {
    use crate::metal::sampler::SamplerWord;
    let filter = word & (SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR);
    let one_filter = filter == 0 || filter == SamplerWord::LINEAR;
    let other = SamplerWord::CUBIC
        | SamplerWord::ANISOTROPIC
        | SamplerWord::COMPARE
        | SamplerWord::REDUCTION;
    one_filter && word & other == 0
}

/// The texture slots a constexpr sampler samples (`RuntimeFact::ConstexprSampled`) at which
/// `shape` binds a texture whose address-mode map changes a mode the slot's constexpr samplers
/// state, each with that map restricted to those modes (every other mode to itself): the plan's
/// `VariantKey::address_maps`. A texture whose map keeps them, every ordinary texture's among
/// them, adds nothing, so its draws keep their key; textures whose maps agree on them share a
/// variant. The variant's compile states the same facts, so its plan for the same shape states
/// the same key.
pub(crate) fn address_map_variant(
    dependencies: &[RuntimeDependency],
    shape: &BindShape,
) -> BTreeMap<SlotRef, metal::sampler::AddressModeMap> {
    let mut out = BTreeMap::new();
    for d in dependencies {
        let RuntimeFact::ConstexprSampled { slot, modes } = d.fact else {
            continue;
        };
        let Some(&TextureShape::Texture { address_modes, .. }) = bound_texture(shape, slot) else {
            continue;
        };
        let stated = out
            .entry(slot)
            .or_insert(metal::sampler::AddressModeMap::IDENTITY);
        for mode in modes {
            *stated = stated.with(mode, address_modes.mode(mode));
        }
    }
    out.retain(|_, map| !map.is_identity());
    out
}

/// The runtime sampler slots whose function samples through them (`RuntimeFact::CubicSampler`)
/// at which `shape` binds a sampler whose min or mag filter is `Cubic`, each with the bound word
/// masked by `SamplerWord::CUBIC_STATE`: the plan's `VariantKey::cubic_samplers`. Metal samples
/// through such a sampler with its bicubic filter of that state. The variant's
/// compile states the same facts, so its plan for the same shape states the same key. A bound
/// word whose state the bicubic filter is not measured under, whatever the function samples
/// (`metal::bicubic::runtime_cubic_measured`: a mip filter, anisotropy, another reduction), is
/// `NotImplemented(SamplerBicubicFilter)` here, as the variant's compile would refuse it: the
/// draw is refused before any variant is compiled for it.
pub(crate) fn cubic_variant(
    dependencies: &[RuntimeDependency],
    shape: &BindShape,
) -> Result<std::collections::BTreeMap<SlotRef, u32>, BindError> {
    use crate::metal::sampler::SamplerWord;
    dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::CubicSampler { slot } => Some(slot),
            _ => None,
        })
        .filter_map(|slot| {
            let bound = shape
                .stages
                .get(&slot.stage)
                .and_then(|s| s.samplers.get(&slot.index))?;
            let word = bound.word.0 & SamplerWord::CUBIC_STATE;
            bound.shape.cubic.then(
                || match crate::metal::bicubic::runtime_cubic_measured(word) {
                    true => Ok((slot, word)),
                    false => Err(BindError::Unsupported(Unsupported::not_implemented(
                        Feature::SamplerBicubicFilter,
                    ))),
                },
            )
        })
        .collect()
}

/// The runtime sampler slots whose samples Metal's anisotropic filter, emulated, reaches
/// (`RuntimeFact::AnisotropicSampler`) at which `shape` binds a sampler whose word states
/// anisotropy (`SamplerWord::ANISOTROPIC`, `maxAnisotropy` above 1): the plan's
/// `VariantKey::anisotropic_samplers`. A function compiled without a slot named there takes the
/// plain sample at it, which is Metal's only where the bound word states no anisotropy, so the
/// variant names every such slot the word of whose bound sampler states it. The variant's compile
/// states the same facts, so its plan for the same shape states the same key.
pub(crate) fn anisotropic_variant(
    dependencies: &[RuntimeDependency],
    shape: &BindShape,
) -> std::collections::BTreeSet<SlotRef> {
    use crate::metal::sampler::SamplerWord;
    dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::AnisotropicSampler { slot } => Some(slot),
            _ => None,
        })
        .filter(|slot| {
            shape
                .stages
                .get(&slot.stage)
                .and_then(|s| s.samplers.get(&slot.index))
                .is_some_and(|bound| bound.word.0 & SamplerWord::ANISOTROPIC != 0)
        })
        .collect()
}

/// The texture `shape` binds at `slot`.
/// What `shape` binds at buffer `slot`.
fn bound_buffer(shape: &BindShape, slot: SlotRef) -> Option<&BufferShape> {
    shape
        .stages
        .get(&slot.stage)
        .and_then(|s| s.buffers.get(&slot.index))
}

fn bound_texture(shape: &BindShape, slot: SlotRef) -> Option<&TextureShape> {
    shape
        .stages
        .get(&slot.stage)
        .and_then(|s| s.textures.get(&slot.index))
}

/// Check every slot whose function observes the residency of its accesses
/// (`RuntimeFact::SparseTexture`). A texture that is not sparse is resident everywhere, so its word
/// is answered on every device. A sparse one needs the module's tile-map arm (`tile_map`, on every
/// profile where the slot has one) or its sparse arm (`sparse_arm`, compiled only where the profile
/// enables `shaderResourceResidency` and the slot has no tile map). A slot the function writes
/// drops its writes to a sparse texture under the same specialization, as Metal does
/// (`RuntimeFact::SparseTextureWrites`). A sparse texture is refused, `Unsupported{SparseTexture {
/// slot }}`:
/// - `NotRepresentable` at a slot the function updates by a texture atomic, whose answer on Metal
///   is not deterministic;
/// - at a slot of observed residency without either arm, naming `shaderResourceResidency` where
///   the device lacks it (an access the tile map does not answer, which the device's residency
///   would), and `NotImplemented` where the device has it but the slot's accesses have no sparse
///   form (a bicubic sample).
///
/// This is the shader's side only: the sparse arm reports the residency of the Vulkan image
/// bound and the tile-map arm that of the tile map bound, and that they mirror Metal's tile
/// mappings is the consumer's contract (`docs/API.md`; the profile does not describe sparse
/// binding).
pub(crate) fn check_sparse_textures(
    dependencies: &[RuntimeDependency],
    profile: &DeviceProfile,
    shape: &BindShape,
) -> Result<(), BindError> {
    for d in dependencies {
        if let RuntimeFact::SparseTextureWrites { slot, atomic: true } = d.fact {
            if sparse_texture(shape, slot) != 0 {
                let feature = Feature::SparseTexture { slot };
                return Err(BindError::Unsupported(Unsupported::not_representable(
                    feature,
                )));
            }
        }
    }
    for d in dependencies {
        let RuntimeFact::SparseTexture {
            slot,
            sparse_arm,
            tile_map,
        } = d.fact
        else {
            continue;
        };
        if sparse_arm || tile_map || sparse_texture(shape, slot) == 0 {
            continue;
        }
        let residency = profile
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderResourceResidency);
        let feature = Feature::SparseTexture { slot };
        return Err(BindError::Unsupported(if residency {
            Unsupported::not_implemented(feature)
        } else {
            Unsupported::needs(
                feature,
                Capability::Feature(VkFeature::ShaderResourceResidency),
            )
        }));
    }
    Ok(())
}

/// Mark each planned buffer whose uniform view no written slot may alias
/// (`RuntimeFact::UniformViewAliasing`): instantiation compares their Metal windows.
pub(crate) fn mark_disjoint(buffers: &mut [PlannedBuffer], dependencies: &[RuntimeDependency]) {
    for b in buffers {
        b.disjoint = dependencies
            .iter()
            .any(|d| d.fact == RuntimeFact::UniformViewAliasing { slot: b.slot });
    }
}

/// The read-only slots Profile F's per-stage storage-buffer fallback must keep as storage views
/// for this shape (`VariantKey::storage_view_slots`): the slots the pipeline was compiled keeping
/// (`Kept`), and, when a slot the pipeline views as a uniform buffer is large, every large
/// candidate ([`large_view_candidates`]), so the fallback views none of them. The variant's
/// compile keeps and restates every slot of the key, and views only candidates the shape proves
/// small, so its plan for the same shape states the same key (a fixed point, in one compile).
/// Otherwise the key is the pipeline's own.
pub(crate) fn storage_view_variant(
    dependencies: &[RuntimeDependency],
    buffers: &[PlannedBuffer],
    profile: &DeviceProfile,
) -> std::collections::BTreeSet<SlotRef> {
    let mut kept = std::collections::BTreeSet::new();
    let large = large_view_candidates(dependencies, buffers, profile, None);
    let mut viewed_large = false;
    for d in dependencies {
        let RuntimeFact::UniformViewRange { slot, choice } = d.fact else {
            continue;
        };
        match choice {
            UniformViewChoice::Kept => {
                kept.insert(slot);
            }
            UniformViewChoice::Viewed => viewed_large |= large.contains(&slot),
            _ => {}
        }
    }
    if viewed_large {
        kept.extend(large);
    }
    kept
}

/// The candidates of Profile F's per-stage storage-buffer fallback (`UniformViewRange` facts of
/// any choice, of `stage` when given) at which the shape binds a buffer whose length class does
/// not prove that a uniform view's range fits `maxUniformBufferRange`. A plan whose variant adds a
/// storage buffer to a stage (a newly pulled vertex layout) names them all, since that variant's
/// fallback may view any of them.
pub(crate) fn large_view_candidates(
    dependencies: &[RuntimeDependency],
    buffers: &[PlannedBuffer],
    profile: &DeviceProfile,
    stage: Option<Stage>,
) -> std::collections::BTreeSet<SlotRef> {
    let max = u64::from(profile.fields().limits.max_uniform_buffer_range);
    dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::UniformViewRange { slot, .. } => Some(slot),
            _ => None,
        })
        .filter(|slot| stage.is_none_or(|s| slot.stage == s))
        .filter(|&slot| {
            buffers.iter().any(|b| {
                b.slot == slot
                    && matches!(b.bound, BoundKind::Buffer { length_class, .. }
                        if !class_fits(length_class, max))
            })
        })
        .collect()
}

/// Whether every range of an allocation of length class `class` fits `max_range`: its padded
/// length is below `2^class` (`BindShape` length class).
pub(crate) fn class_fits(class: u8, max_range: u64) -> bool {
    (1u128 << class.min(127)) - 1 <= u128::from(max_range)
}

/// The slots a `BufferRangeLimit` variant must move to device addresses for this shape: every slot
/// whose fact the profile honors by `Variant` and whose bound buffer's length class does not prove
/// that its range fits the view's limit. A slot already in device-address form (the variant's own
/// pipeline) is listed as well, so planning the variant's pipeline with the same shape gives the
/// same key (a fixed point). Profile F honors the fact at instantiation (`DescriptorRangeExceeded`,
/// exact) and needs no variant.
pub(crate) fn psb_variant(
    dependencies: &[RuntimeDependency],
    buffers: &[PlannedBuffer],
    profile: &DeviceProfile,
) -> std::collections::BTreeSet<SlotRef> {
    let mut out = std::collections::BTreeSet::new();
    for d in dependencies {
        let RuntimeFact::BufferRangeLimit { slot } = d.fact else {
            continue;
        };
        let large = buffers
            .iter()
            .any(|b| b.slot == slot && range_variant(slot, b.bound, profile));
        if large {
            out.insert(slot);
        }
    }
    out
}

/// Whether a slot with the `BufferRangeLimit` fact bound as `bound` joins the variant's
/// device-address slots: the profile honors the fact by `Variant` and the bound buffer's length
/// class does not prove that its range fits `maxStorageBufferRange`.
pub(crate) fn range_variant(slot: SlotRef, bound: BoundKind, profile: &DeviceProfile) -> bool {
    let storage = u64::from(profile.fields().limits.max_storage_buffer_range);
    mechanism(&RuntimeFact::BufferRangeLimit { slot }, profile) == Mechanism::Variant
        && matches!(bound, BoundKind::Buffer { length_class, .. } if !class_fits(length_class, storage))
}

/// The `uses` entry of a bound buffer slot: its proven access, `Uniform` for a uniform view and
/// `Storage` otherwise, `stages`, and the window as a range when it starts at or after the bind
/// offset.
pub(crate) fn buffer_use(b: &PlannedBuffer, stages: vk::PipelineStageFlags2) -> ResourceUse {
    ResourceUse {
        resource: UseTarget::Slot(b.slot),
        access: b.access,
        class: if b.form.uniform_view() {
            AccessClass::Uniform
        } else {
            AccessClass::Storage
        },
        stages,
        range: match b.window {
            Window::Bounded { start, len } if start >= 0 => Some(ByteRange {
                offset: start as u64,
                len,
            }),
            _ => None,
        },
    }
}

/// The `uses` of the used resources a pipeline can reach (the handback): buffers and heaps when a
/// function reads the resident root table (`pointer`), textures and heaps when one resolves handles
/// (`handle`). The access is the guest's declared `MTLResourceUsage` (`Read` and `Sample` read,
/// `Write` writes; no usage is `READ_WRITE`), since nothing proves a narrower one, but in the
/// stages whose function writes no memory: there a resource the usage lets the function write is
/// read, in a use of its own, as such a function stores to no buffer or texture and updates none
/// atomically (`StageInterface::writes_memory`). `writers` holds the pipeline stages whose function
/// may write memory. `stages` maps a used resource's render stages to the pipeline stages that
/// reach it (a compute plan maps every resource to `COMPUTE_SHADER`), and a resource reaching none
/// is not listed.
pub(crate) fn used_uses(
    used: &BTreeMap<u32, UsedShape>,
    pointer: bool,
    handle: bool,
    stages: impl Fn(metal::RenderStages) -> vk::PipelineStageFlags2,
    writers: vk::PipelineStageFlags2,
) -> Vec<ResourceUse> {
    let mut out = Vec::new();
    for (&token, u) in used {
        let reachable = match u.class {
            UsedClass::Buffer => pointer,
            UsedClass::Texture(_) => handle,
            UsedClass::Heap => pointer || handle,
        };
        let stages = stages(u.stages);
        if !reachable || stages.is_empty() {
            continue;
        }
        let read = u
            .usage
            .intersects(metal::ResourceUsage::READ | metal::ResourceUsage::SAMPLE);
        let write = u.usage.contains(metal::ResourceUsage::WRITE);
        let access = match (read, write) {
            (false, false) => AccessSet::READ_WRITE,
            (read, write) => AccessSet { read, write },
        };
        let class = match u.class {
            UsedClass::Texture(_) if write => AccessClass::StorageImage,
            UsedClass::Texture(_) => AccessClass::Sampled,
            UsedClass::Buffer | UsedClass::Heap => AccessClass::Storage,
        };
        let (writing, reading) = match access.write {
            true => (
                stages.intersection(writers),
                vk::PipelineStageFlags2::from_raw(stages.as_raw() & !writers.as_raw()),
            ),
            false => (stages, vk::PipelineStageFlags2::empty()),
        };
        for (stages, access) in [(writing, access), (reading, AccessSet::READ)] {
            if !stages.is_empty() {
                out.push(ResourceUse {
                    resource: UseTarget::Used(token),
                    access,
                    class,
                    stages,
                    range: None,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
