//! Texture, sampler, combined image-sampler, texel-buffer, and input-attachment writes, shared by
//! compute and render plans.
//!
//! Every such binding of the layout gets one write:
//!
//! - **Declared type.** A binding's image type is derived from the slot's interface (the Metal
//!   texture type and sampled type), the binding's descriptor type (sampled or storage, image or
//!   texel buffer), and the pipeline's facts: a cube with `ImageViewShape` is the 2D-array view
//!   the function declared, a combined binding's depth comparison is its symbol's, and a storage
//!   binding's format is the variant's storage-image choice, `Unknown` without one, except a
//!   `ulong` slot with `UlongAtomicView`, which is `R64_UINT`.
//! - **Bound and compatible:** a texture view of the bound format through `metal::format`
//!   (sampled: `vk_format` for `SampledImage`, with the emulation's component mapping; storage:
//!   the `StorageImageChoice`'s view format; `R64_UINT` for a `ulong` slot with
//!   `UlongAtomicView`, an alias of the `MUTABLE_FORMAT` image's `RG32_UINT` texels), a
//!   texel-buffer view format, or the runtime sampler slot, whose `VkSampler` instantiation makes
//!   from the bound descriptor for the binding's use.
//! - **Unbound, or bound but incompatible** with the declared type (another texture type, sample
//!   count, or numeric type, a texture buffer for an image, a color texture under a depth
//!   comparison): the neutral of exactly the declared type; an incompatible binding is also a
//!   `PlanDiagnostic::IncompatibleBinding` (guest undefined behavior).
//! - **Variant**: a read or written storage slot's choice joins the plan's `VariantKey` unless the
//!   declaration a pipeline compiled without one has serves it (an `Identity` choice without an
//!   alias, where the profile accesses the slot's uses without a format), so the common path stays
//!   empty. A slot whose texels take the bound format's numeric type
//!   (`RuntimeFact::StorageTextureNumericType`) always joins it: a formatless choice keeps the
//!   type.
//! - **Shape checks:** a runtime sampler's shape passes `metal::sampler::check_sampler` for the
//!   binding's use (a non-average reduction needs `samplerFilterMinmax` and no comparison;
//!   mirror-clamp, border colors, and LOD clamps; anisotropy needs nothing, its sampler isotropic
//!   on a device without `samplerAnisotropy` and where the binding samples a cube with explicit
//!   gradients), so `bind::sampler_desc` refuses per bind only what depends on the bound image
//!   format; `SamplerState` specializes from the bound sampler's word, masked by the fields the
//!   function reads.
//! - **Bicubic texels**: a texture slot sampled through Metal's bicubic filter specializes
//!   `BicubicTexels` from the bound format's family (`metal::bicubic::texels`), and refuses a
//!   format of no measured family.
//! - **Half texels**: a texture or texture buffer slot read or sampled with a `half` result, or
//!   sampled through Metal's emulated linear filter, specializes `TexelFormat` from the bound
//!   format's class (`metal::texel_format::texel_format`).
//! - **Separate bindings**: a separate texture's write is the image its combined binding's would
//!   be, with the same checks, and a separate sampler's is a constexpr sampler's layout immutable
//!   sampler (no write) or the runtime sampler for its use. A runtime sampler meets its textures in
//!   the shader, so each texture's write checks the bound sampler's filtering against its view at
//!   plan time, which `bind::sampler_desc` does per bind for a combined write.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    AccessClass, ImageSource, ImageViewDesc, PlanDiagnostic, PlannedTexel, ResourceTemplate,
    ResourceUse, SamplerSource, UseTarget, WriteSource,
};
use crate::bind::neutral::NeutralResource;
use crate::bind::state::{SamplerShape, StageShape, TextureShape};
use crate::cache::VariantKey;
use crate::device::requirements::{FormatUsage, VkLimit};
use crate::device::DeviceProfile;
use crate::error::{BindError, Capability, Error, Feature, LowerFamily, Unsupported};
use crate::metal::bicubic::{texels as bicubic_texels, BicubicTexels};
use crate::metal::format::{
    accessed_without_format, aspects, direct_format, is_depth_stencil, sampled_type, srgb_raw_twin,
    storage_choice, texel_buffer_alpha_in_red, vk_format, writes_dropped, ycbcr_writes_stored,
    ExtendedRange, FormatMapping, StorageUse,
};
use crate::metal::sampler::{
    check_border_view, check_sampler, opaque_black_reads_transparent, vk_filtering,
    OpaqueBlackBorder, SamplerUse, SamplerWord,
};
use crate::metal::texel_format::{reads_unswizzled, srgb_twin_on, texel_format, TexelFormat};
use crate::metal::{PixelFormat, SamplerMipFilter, TextureSwizzleChannels, TextureType};
use crate::pipeline::artifact::RuntimeDependency;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{
    AccessSet, SamplerSlot, SlotInterface, TextureAccess, TextureSlot,
};
use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
use crate::pipeline::sym::{
    CombinedSampler, ImageAccess, ImageDim, ImageHandleClass, SampledType, SamplerRef, SlotClass,
    SlotRef, Stage, StorageFormatClass,
};
use crate::vk;

use super::compute::internal;

/// A `metal`/`device` error as a bind error: capability refusals stay typed, a bound resource's
/// invalid descriptor is the guest's, anything else is an invariant failure.
pub(crate) fn bind_error(e: Error) -> BindError {
    match e {
        Error::Unsupported(u) => BindError::Unsupported(u),
        Error::InvalidDescriptor(d) => BindError::InvalidDescriptor(d),
        Error::Internal(i) => BindError::Internal(i),
        other => internal(format!("{other:?}")),
    }
}

/// The `VkSampler` of the sampler bound at a plan's `WriteSource::Sampler`,
/// `WriteSource::ArgumentSampler` or `SamplerSource::Slot`: `metal::sampler::vk_sampler` for the
/// write's use (its `compare` and `integer`) on `device`, its opaque-black border
/// made as the write says (`opaque_black`, transparent black, or opaque black with the view's
/// mapping in `border_components`), and, when the sampler reads an image of `image_format` (a
/// combined write's view format), the format's linear-filtering and min/max features the
/// descriptor needs (checked against the bound sampler, not at texture creation). The plan already refused what the sampler's shape decides
/// (`metal::sampler::check_sampler`), so for a state of the plan's shape only the format features
/// can refuse here.
pub fn sampler_desc(
    device: &crate::device::Device,
    desc: &crate::bind::state::SamplerDescriptor,
    use_: SamplerUse,
    opaque_black: OpaqueBlackBorder,
    image_format: Option<vk::Format>,
) -> Result<vk::SamplerDesc, BindError> {
    let profile = device.profile();
    let mut out = crate::metal::sampler::vk_sampler(desc, use_, profile).map_err(bind_error)?;
    let black = [
        vk::BorderColor::FLOAT_OPAQUE_BLACK,
        vk::BorderColor::INT_OPAQUE_BLACK,
    ];
    if black.contains(&out.border_color) {
        match opaque_black {
            OpaqueBlackBorder::Stated => {}
            OpaqueBlackBorder::Transparent => {
                out.border_color = if out.border_color == black[0] {
                    vk::BorderColor::FLOAT_TRANSPARENT_BLACK
                } else {
                    vk::BorderColor::INT_TRANSPARENT_BLACK
                };
            }
            OpaqueBlackBorder::Mapped(components) => out.border_components = Some(components),
        }
    }
    if let Some(format) = image_format {
        check_filter_features(profile, &out, format)?;
    }
    Ok(out)
}

/// The format features Vulkan sampler `s` needs of the image view format `format` it reads. A
/// linear filter or mip mode needs `SAMPLED_IMAGE_FILTER_LINEAR` only under weighted-average
/// reduction without depth comparison (VUID-vkCmdDispatch-magFilter-04553,
/// VUID-vkCmdDispatch-mipmapMode-04770: a comparison sampler filters the comparison results, which
/// needs nothing of the format), and `SAMPLED_IMAGE_FILTER_MINMAX` only under a min or max
/// reduction (VUID-vkCmdDispatch-magFilter-09598, VUID-vkCmdDispatch-mipmapMode-09599). A nearest
/// sampler needs neither. A runtime sampler's ([`sampler_desc`]) and a layout's immutable one's
/// ([`ResourcePlanner::combined`]) are checked alike.
fn check_filter_features(
    profile: &DeviceProfile,
    s: &vk::SamplerDesc,
    format: vk::Format,
) -> Result<(), BindError> {
    require_filter_features(profile, desc_filtering(s), s.compare.is_some(), format)
}

/// The filtering of Vulkan sampler `s`: whether some filter or its mip mode is linear, and its
/// reduction.
pub(crate) fn desc_filtering(s: &vk::SamplerDesc) -> (bool, vk::SamplerReductionMode) {
    let linear = s.mag_filter == vk::Filter::LINEAR
        || s.min_filter == vk::Filter::LINEAR
        || s.mipmap_mode == vk::SamplerMipmapMode::LINEAR;
    (linear, s.reduction)
}

/// [`check_filter_features`] at plan time for the runtime sampler of `shape` bound for `use_`: the
/// filtering `metal::sampler::vk_sampler` makes of it for the use (`metal::sampler::vk_filtering`),
/// against the view format `format`. A separate sampler's write carries no image, so
/// `bind::sampler_desc` cannot check its textures' views per bind.
fn check_shape_filter_features(
    profile: &DeviceProfile,
    shape: &SamplerShape,
    use_: SamplerUse,
    format: vk::Format,
) -> Result<(), BindError> {
    require_filter_features(profile, vk_filtering(shape, use_), use_.compare, format)
}

/// The format features a sampler that filters linearly or not and reduces with `reduction`
/// (`filtering`), comparing or not, needs of the view format `format` it reads.
pub(crate) fn require_filter_features(
    profile: &DeviceProfile,
    (linear, reduction): (bool, vk::SamplerReductionMode),
    compare: bool,
    format: vk::Format,
) -> Result<(), BindError> {
    use vk::SamplerReductionMode as R;
    let filtered = linear && reduction == R::WEIGHTED_AVERAGE && !compare;
    let minmax = linear && matches!(reduction, R::MIN | R::MAX);
    for (needed, usage) in [
        (filtered, FormatUsage::SampledImageFilterLinear),
        (minmax, FormatUsage::SampledImageFilterMinmax),
    ] {
        if needed && !profile.supports_format(format, usage) {
            return Err(BindError::Unsupported(Unsupported::needs(
                Feature::Format,
                Capability::Format { format, usage },
            )));
        }
    }
    Ok(())
}

/// The immutable sampler of each separate sampler binding of `layout`, by its stage and use: `None`
/// for a runtime one. A separate texture's write reads its sampler's.
pub(crate) fn separate_samplers(
    layout: &crate::pipeline::layout::PipelineLayout,
) -> BTreeMap<(Stage, CombinedSampler), Option<&vk::SamplerDesc>> {
    layout
        .sets
        .iter()
        .flat_map(|set| &set.bindings)
        .filter_map(|b| match b.source {
            crate::pipeline::layout::BindingSource::SeparateSampler { stage, sampler } => {
                Some(((stage, sampler), b.immutable_sampler.as_ref()))
            }
            _ => None,
        })
        .collect()
}

/// Whether the Vulkan sampler `s` replaces texels with an opaque-black built-in border: some
/// axis clamps to the border and the color is `FLOAT_OPAQUE_BLACK` or `INT_OPAQUE_BLACK`.
pub(crate) fn opaque_black_border(s: &vk::SamplerDesc) -> bool {
    [s.address_mode_u, s.address_mode_v, s.address_mode_w]
        .contains(&vk::SamplerAddressMode::CLAMP_TO_BORDER)
        && matches!(
            s.border_color,
            vk::BorderColor::FLOAT_OPAQUE_BLACK | vk::BorderColor::INT_OPAQUE_BLACK
        )
}

/// What one plan's resource writes need from the pipeline, the device, and the shape.
pub(crate) struct ResourcePlanner<'a> {
    pub profile: &'a DeviceProfile,
    pub shape: &'a BTreeMap<Stage, StageShape>,
    pub slots: &'a BTreeMap<SlotRef, SlotInterface>,
    pub dependencies: &'a [RuntimeDependency],
    pub stage_flags: fn(Stage) -> vk::PipelineStageFlags2,
    /// Outputs.
    pub variant: VariantKey,
    pub diagnostics: Vec<PlanDiagnostic>,
    pub uses: Vec<ResourceUse>,
    pub template: ResourceTemplate,
    /// The texture-buffer slots whose texel view reads an `A8Unorm` texture buffer's bytes as
    /// `R8_UNORM` with no storage class to convert them (`metal::format::texel_buffer_alpha_in_red`):
    /// their reads take Metal's alpha from red through the slot's swizzle
    /// ([`ResourcePlanner::texture_swizzle`]).
    pub alpha_red_texels: BTreeSet<SlotRef>,
}

/// The type a texture binding declares (the neutral's input).
fn declared(
    t: &TextureSlot,
    ty: vk::DescriptorType,
    cube_as_array: bool,
    depth_compare: bool,
    format: StorageFormatClass,
) -> Result<ImageHandleClass, BindError> {
    use TextureType as T;
    let (dim, arrayed, multisampled) = match t.texture_type {
        T::Type1D => (ImageDim::D1, false, false),
        T::Type1DArray => (ImageDim::D1, true, false),
        T::Type2D => (ImageDim::D2, false, false),
        T::Type2DArray => (ImageDim::D2, true, false),
        T::Type2DMultisample => (ImageDim::D2, false, true),
        T::Type2DMultisampleArray => (ImageDim::D2, true, true),
        T::Cube if cube_as_array => (ImageDim::D2, true, false),
        T::CubeArray if cube_as_array => (ImageDim::D2, true, false),
        T::Cube => (ImageDim::Cube, false, false),
        T::CubeArray => (ImageDim::Cube, true, false),
        T::Type3D => (ImageDim::D3, false, false),
        T::TextureBuffer => (ImageDim::Buffer, false, false),
    };
    let buffer = matches!(
        ty,
        vk::DescriptorType::UNIFORM_TEXEL_BUFFER | vk::DescriptorType::STORAGE_TEXEL_BUFFER
    );
    if buffer != (dim == ImageDim::Buffer) {
        return Err(internal(format!(
            "a {:?} texture slot bound as {ty:?}",
            t.texture_type
        )));
    }
    let storage = matches!(
        ty,
        vk::DescriptorType::STORAGE_IMAGE | vk::DescriptorType::STORAGE_TEXEL_BUFFER
    );
    Ok(ImageHandleClass {
        dim,
        arrayed,
        multisampled,
        sampled_type: t.sampled_type,
        depth_compare,
        access: if storage {
            ImageAccess::Storage
        } else {
            ImageAccess::Sampled
        },
        format: if storage {
            format
        } else {
            StorageFormatClass::NotStorage
        },
    })
}

/// The neutral of `class`.
pub(crate) fn neutral(class: &ImageHandleClass) -> Result<NeutralResource, BindError> {
    NeutralResource::for_image(class)
        .ok_or_else(|| internal(format!("no neutral for the declared type {class:?}")))
}

/// Whether a bound texture of `shape` is what the declared `t` (with `class`) may read: the same
/// Metal texture type, the declared numeric type, and a depth format under comparison.
fn compatible(shape: &TextureShape, t: &TextureSlot, class: &ImageHandleClass) -> bool {
    match *shape {
        TextureShape::Texture {
            texture_type,
            pixel_format,
            sample_count,
            ..
        } => {
            texture_type == t.texture_type
                && (sample_count > 1) == class.multisampled
                && texel_type_binds(pixel_format, t.sampled_type)
                && (!class.depth_compare || is_depth_stencil(pixel_format))
        }
        TextureShape::Buffer { pixel_format, .. } => {
            class.dim == ImageDim::Buffer && texel_type_binds(pixel_format, t.sampled_type)
        }
    }
}

/// Whether a bound texture of `shape` is what the declared `t` (with `class`) may read but for
/// its numeric type: the same Metal texture type and sample count, a depth format under
/// comparison, and another numeric type. The binder writes the declared type's neutral there
/// unless the slot's variant reads the texture in its own class ([`cross_class`]), and says so
/// (`PlanDiagnostic::CrossClassNeutral`).
fn only_numeric(shape: &TextureShape, t: &TextureSlot, class: &ImageHandleClass) -> bool {
    let Some(ty) = bound_sampled_type(shape.pixel_format()) else {
        return false;
    };
    let typed = TextureSlot {
        sampled_type: ty,
        ..*t
    };
    !compatible(shape, t, class) && compatible(shape, &typed, class)
}

/// The numeric class Metal reads a texture of `shape` in at a slot declared `t`, where it is not
/// `t`'s and the slot's variant reads the texture in it (`VariantKey::cross_class`): a texture of
/// `t`'s Metal texture type (any sample count it has, sparse or not), or a texture buffer at a
/// texture-buffer slot, of a color format whose class (`metal::format::sampled_type`) is `Float`,
/// `Sint` or `Uint` and not `t`'s, which is one of those three too. Neither a depth or stencil
/// format, nor one another variant reads raw (an extended-range or bfloat one; a two-plane YCbCr
/// one). Plain Metal reads such a texture in its own class and hands back the words as `t`'s type,
/// a `half` result each word's low 16 bits (measured on Apple M3 Max and M4 for 28 formats); and
/// (on an Apple M3 Max) a `short` or `ushort` result an integer word's
/// low 16 bits or a float texel's `half` bits, a storage, multisampled, sparse or texture-buffer
/// read alike, and a write stores the written lanes' bits as the format's own class
/// (`lower::texture::cross_write`). Two writes are left out: a storage slot bound to a swizzled
/// texture (the swizzle's order against the cross-class conversion is not measured), and an
/// integer slot's write to an `RGB9E5Float` texture (its rounding is not measured). The slot's
/// compile must state `RuntimeFact::CrossClassReads`.
pub(crate) fn cross_class(t: &TextureSlot, shape: &TextureShape) -> Option<SampledType> {
    let (pixel_format, same_type) = match *shape {
        TextureShape::Texture {
            texture_type,
            pixel_format,
            sample_count,
            swizzle,
            ..
        } => {
            let ms = matches!(
                texture_type,
                TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray
            );
            let unswizzled_store = !t.access.write || swizzle.is_default();
            (
                pixel_format,
                texture_type == t.texture_type && (sample_count > 1) == ms && unswizzled_store,
            )
        }
        TextureShape::Buffer { pixel_format, .. } => {
            (pixel_format, t.texture_type == TextureType::TextureBuffer)
        }
    };
    let classes = [SampledType::Float, SampledType::Sint, SampledType::Uint];
    let bound = sampled_type(pixel_format)?;
    let integer_write = t.access.write && t.sampled_type != SampledType::Float;
    let taken = same_type
        && !t.depth_compare
        && classes.contains(&t.sampled_type)
        && bound != t.sampled_type
        && crate::metal::texel_format::crosses_classes(pixel_format)
        && !(integer_write && pixel_format == PixelFormat::RGB9E5Float);
    taken.then_some(bound)
}

/// `t`, a slot whose integer texels take their signedness from the bound format
/// (`RuntimeFact::StorageTextureNumericType`), with the bound integer format's numeric type: its
/// own is that of the format its compile was given, which may be another integer format's (a
/// runtime's stand-in). The bound format's choice then keys the variant that compiles the slot with
/// its type. A slot or format of another type keeps `t`'s.
pub(super) fn bound_integer_type(t: TextureSlot, shape: Option<&TextureShape>) -> TextureSlot {
    let integer = |ty: SampledType| matches!(ty, SampledType::Sint | SampledType::Uint);
    let bound = match shape {
        Some(
            TextureShape::Texture { pixel_format, .. } | TextureShape::Buffer { pixel_format, .. },
        ) => sampled_type(*pixel_format),
        None => None,
    };
    match bound {
        Some(ty) if integer(ty) && integer(t.sampled_type) => TextureSlot {
            sampled_type: ty,
            ..t
        },
        _ => t,
    }
}

/// Whether Metal binds a texture of `pf` to a parameter of texel type `ty`: the format's numeric
/// type, and for `ulong` only `RG32Uint`, which Metal's API validation requires ("incompatible
/// with the data type (`MTLDataTypeULong`)" for every other format, measured on Apple M3 Max).
fn texel_type_binds(pf: PixelFormat, ty: SampledType) -> bool {
    match ty {
        SampledType::Uint64 => pf == PixelFormat::RG32Uint,
        _ => bound_sampled_type(pf) == Some(ty),
    }
}

/// Metal's numeric type of a texture of `pf` bound at a slot: the format table's
/// (`sampled_type`), and `Float` for a color format the table holds no representation of (the
/// 4:2:2 formats, whose texels the runtime holds raw for copies, and ASTC's 4x2 and 8x4
/// footprints).
/// Metal reads each as floats, so a slot of that type holds such a texture, and its binding is
/// the format's own refusal (it has no sampled view), not a mismatch a neutral stands in for.
/// A slot has an interface only where the function uses it, so a texture bound at an index the
/// function never reads is refused nothing. A heap's classes keep the table's type
/// (`heap::fit`): a resident texture the heap cannot view is the neutral at its element.
fn bound_sampled_type(pf: PixelFormat) -> Option<SampledType> {
    sampled_type(pf).or_else(|| {
        let placeholder = matches!(pf, PixelFormat::Invalid | PixelFormat::Unspecialized);
        (!placeholder && aspects(pf).is_none()).then_some(SampledType::Float)
    })
}

/// The aspect a sampled view of `pf` selects: depth for a depth or combined format, stencil for a
/// stencil-only one, color otherwise.
fn view_aspect(pf: PixelFormat) -> vk::ImageAspectFlags {
    let a = aspects(pf).unwrap_or(vk::ImageAspectFlags::COLOR);
    if a.contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageAspectFlags::DEPTH
    } else if a.contains(vk::ImageAspectFlags::STENCIL) {
        vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

/// The component mapping of a view that reads a depth format's depth without comparison: the
/// depth in red, green and blue and alpha one, `(d, d, d, 1)`, as Metal reads a depth texel
/// through a color texture type (`texture2d<float>`, `texture2d<half>`), where Vulkan converts it
/// to `(d, 0, 0, 1)`. In the canonical form `TextureSwizzleChannels::compose` makes (red from
/// itself is `IDENTITY`). Metal's border texels of such a read are the same replication of the
/// border's red: `(0, 0, 0, 1)` for either black and `(1, 1, 1, 1)` for opaque white, which the
/// device reads through this mapping too (Vulkan swizzles after the border's replacement). A
/// comparison's view keeps the identity: its value is one scalar.
pub(crate) const DEPTH_AS_COLOR: vk::ComponentMapping = vk::ComponentMapping {
    r: vk::ComponentSwizzle::IDENTITY,
    g: vk::ComponentSwizzle::R,
    b: vk::ComponentSwizzle::R,
    a: vk::ComponentSwizzle::ONE,
};

/// The component mapping of a sampled view of `pf`'s aspect `aspect` whose format maps with
/// `components`, read with comparison or not (`compare`): [`DEPTH_AS_COLOR`] for a depth aspect
/// without comparison (the format table's depth candidates map with the identity), else
/// `components`.
fn sampled_components(
    aspect: vk::ImageAspectFlags,
    components: vk::ComponentMapping,
    compare: bool,
) -> vk::ComponentMapping {
    match aspect == vk::ImageAspectFlags::DEPTH && !compare {
        true => DEPTH_AS_COLOR,
        false => components,
    }
}

/// The view of a sampled image of `pf` declared as `class` on `profile`: `vk_format` for
/// `SampledImage` with the emulation's component mapping, the depth (or stencil-only) aspect, in
/// `SHADER_READ_ONLY_OPTIMAL`. A depth that no comparison reads reads as Metal's `(d, d, d, 1)`
/// ([`DEPTH_AS_COLOR`]).
pub(crate) fn sampled_view(
    profile: &DeviceProfile,
    pf: PixelFormat,
    class: &ImageHandleClass,
) -> Result<ImageViewDesc, BindError> {
    let usages = [FormatUsage::SampledImage].into();
    let m = vk_format(pf, &usages, profile).map_err(bind_error)?;
    let aspect = view_aspect(pf);
    Ok(ImageViewDesc {
        view_type: view_type(class)?,
        format: m.format,
        aspect,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        components: sampled_components(aspect, m.components, class.depth_compare),
        level: None,
        shader_swizzle: false,
    })
}

/// The view type of a declared image class (the binder's cube-as-2D-array view included).
pub(crate) fn view_type(class: &ImageHandleClass) -> Result<vk::ImageViewType, BindError> {
    use vk::ImageViewType as V;
    Ok(match (class.dim, class.arrayed) {
        (ImageDim::D1, false) => V::TYPE_1D,
        (ImageDim::D1, true) => V::TYPE_1D_ARRAY,
        (ImageDim::D2, false) => V::TYPE_2D,
        (ImageDim::D2, true) => V::TYPE_2D_ARRAY,
        (ImageDim::D3, false) => V::TYPE_3D,
        (ImageDim::Cube, false) => V::CUBE,
        (ImageDim::Cube, true) => V::CUBE_ARRAY,
        _ => return Err(internal(format!("no view type for {class:?}"))),
    })
}

/// Whether a storage slot of `use_` needs its storage choice `c` of a texture or texture buffer
/// viewed as `view` as a variant: a pipeline compiled without a choice declares the slot `Unknown`
/// with class `Identity` wherever the device accesses its uses without a format, which serves a
/// choice that converts nothing and aliases nothing exactly when the view's format serves those
/// uses without one ([`accessed_without_format`]: Vulkan grants it per format and per image or
/// buffer, not by the device's features alone). Any other choice is a compile input.
fn needs_variant(
    c: &StorageImageChoice,
    view: vk::Format,
    use_: StorageUse,
    profile: &DeviceProfile,
) -> bool {
    !(c.class == StorageImageClass::Identity
        && c.alias.is_none()
        && !c.x_channel
        && accessed_without_format(view, use_, profile))
}

/// The declared storage format class of a choice.
fn storage_class(c: Option<&StorageImageChoice>) -> StorageFormatClass {
    match c {
        Some(c) if c.vk_format != vk::Format::UNDEFINED => {
            // A view format without a SPIR-V image format is declared formatless.
            if crate::metal::format::spirv_image_format(c.vk_format).is_some() {
                StorageFormatClass::Format(c.vk_format)
            } else {
                StorageFormatClass::Unknown
            }
        }
        _ => StorageFormatClass::Unknown,
    }
}

fn texture_access(a: TextureAccess) -> AccessSet {
    AccessSet {
        read: a.read || a.sample,
        write: a.write,
    }
}

impl ResourcePlanner<'_> {
    fn has_fact(&self, fact: RuntimeFact) -> bool {
        self.dependencies.iter().any(|d| d.fact == fact)
    }

    fn stage_shape(&self, stage: Stage) -> Option<&StageShape> {
        self.shape.get(&stage)
    }

    fn texture_slot(&self, slot: SlotRef) -> Result<&TextureSlot, BindError> {
        match self.slots.get(&slot) {
            Some(SlotInterface::Texture(t)) => Ok(t),
            _ => Err(internal(format!(
                "texture binding {slot:?} without its interface"
            ))),
        }
    }

    fn texture_shape(&self, slot: SlotRef) -> Option<&TextureShape> {
        self.stage_shape(slot.stage)
            .and_then(|s| s.textures.get(&slot.index))
    }

    fn sampler_shape(&self, slot: SlotRef) -> Option<&SamplerShape> {
        self.stage_shape(slot.stage)
            .and_then(|s| s.samplers.get(&slot.index))
            .map(|s| &s.shape)
    }

    /// The neutral `n` for `slot`, declared `t` (with `class`), reported as a substitution where
    /// `bound`, the shape there, binds something: as `PlanDiagnostic::CrossClassNeutral` where
    /// only its numeric type differs ([`only_numeric`]), else as `IncompatibleBinding`.
    fn substitute(
        &mut self,
        slot: SlotRef,
        n: NeutralResource,
        bound: Option<&TextureShape>,
        (t, class): (&TextureSlot, &ImageHandleClass),
    ) -> NeutralResource {
        if let Some(shape) = bound {
            let d = match only_numeric(shape, t, class) {
                true => PlanDiagnostic::CrossClassNeutral {
                    slot,
                    substituted: n,
                },
                false => PlanDiagnostic::IncompatibleBinding {
                    slot,
                    substituted: n,
                },
            };
            self.diagnostics.push(d);
        }
        n
    }

    /// The class the variant reads the texture bound at sampled slot `slot`, declared `t`, in
    /// ([`cross_class`]), where the slot's compile states `RuntimeFact::CrossClassReads`.
    fn cross_of(&self, slot: SlotRef, t: &TextureSlot) -> Option<SampledType> {
        if !self.has_fact(RuntimeFact::CrossClassReads { slot }) {
            return None;
        }
        cross_class(t, self.texture_shape(slot)?)
    }

    /// The write of texture slot `slot`'s tile map: the sparse texture's map, read by the slot's
    /// stage, or the zero buffer where the bound texture is not sparse (the module reads its map
    /// only for a sparse texture, under `SpecKey::SparseTexture`).
    pub(crate) fn tile_map(&mut self, slot: SlotRef) -> WriteSource {
        let sparse = self
            .shape
            .get(&slot.stage)
            .and_then(|s| s.textures.get(&slot.index))
            .is_some_and(TextureShape::sparse);
        if !sparse {
            return WriteSource::Neutral(NeutralResource::ZeroBuffer {
                min_bytes: crate::bind::neutral::zero_buffer_bytes(self.profile),
            });
        }
        self.uses.push(ResourceUse {
            resource: UseTarget::TileMap(slot),
            access: AccessSet::READ,
            class: AccessClass::Storage,
            stages: (self.stage_flags)(slot.stage),
            range: None,
        });
        WriteSource::TileMap { slot }
    }

    /// The write of `slot`'s binding under its in-place variant: the planes' texel buffer of the
    /// two-plane YCbCr texture the shape binds there (`TextureFacts::ycbcr_in_place`), a read of
    /// their bytes; the neutral texel buffer where it binds none, which the plan's choice of the
    /// variant does not leave (`slots::ycbcr_in_place_variant`).
    fn in_place_write(&mut self, slot: SlotRef) -> (WriteSource, NeutralResource) {
        let neutral = NeutralResource::TexelBuffer {
            format: vk::Format::R8G8_UINT,
        };
        match self.texture_shape(slot) {
            Some(TextureShape::Texture {
                ycbcr_in_place: true,
                ..
            }) => {
                self.uses.push(ResourceUse {
                    resource: UseTarget::Slot(slot),
                    access: AccessSet::READ,
                    class: AccessClass::YCbCrPlanes,
                    stages: (self.stage_flags)(slot.stage),
                    range: None,
                });
                (WriteSource::YCbCrInPlace { slot }, neutral)
            }
            _ => (WriteSource::Neutral(neutral), neutral),
        }
    }

    /// The write of `slot`'s binding under its planes-image variant (`BindingSource::YCbCrPlanes`,
    /// `VariantKey::ycbcr_planes_sampled`): the image of the planes of the two-plane YCbCr texture
    /// the shape binds there, viewed with the device's conversion of `TextureShape::Texture::
    /// ycbcr_planes_sampler`, a read of the planes' bytes. The plan takes the variant only where
    /// the shape binds such a texture there (`slots::ycbcr_planes_sampled_variant`), so any other
    /// is the binder's defect.
    pub(crate) fn planes_image_write(&mut self, slot: SlotRef) -> Result<WriteSource, BindError> {
        match self.texture_shape(slot) {
            Some(TextureShape::Texture {
                ycbcr_planes_sampler: Some(_),
                ..
            }) => {
                self.uses.push(ResourceUse {
                    resource: UseTarget::Slot(slot),
                    access: AccessSet::READ,
                    class: AccessClass::YCbCrPlanes,
                    stages: (self.stage_flags)(slot.stage),
                    range: None,
                });
                Ok(WriteSource::YCbCrPlanesImage { slot })
            }
            _ => Err(internal(format!(
                "a planes image's binding of {slot:?}, which binds no texture whose planes are \
                 held as an image"
            ))),
        }
    }

    fn texture_use(&mut self, slot: SlotRef, t: &TextureSlot, storage: bool) {
        self.uses.push(ResourceUse {
            resource: UseTarget::Slot(slot),
            access: texture_access(t.access),
            class: if storage {
                AccessClass::StorageImage
            } else {
                AccessClass::Sampled
            },
            stages: (self.stage_flags)(slot.stage),
            range: None,
        });
    }

    /// The write of a plain texture binding `(set, binding)` of type `ty` fed by `slot`.
    pub fn texture(
        &mut self,
        set: u32,
        binding: u32,
        ty: vk::DescriptorType,
        slot: SlotRef,
    ) -> Result<WriteSource, BindError> {
        Ok(self.texture_write(set, binding, ty, slot)?.0)
    }

    /// The writes of storage slot `slot`'s `count` level views, binding `(set, binding)` of type
    /// `ty` (`RuntimeFact::StorageTextureLevelViews`): element `i` views level `i` of the bound
    /// texture alone, and every element past its last level is the neutral of the declared type, as
    /// is every element where the slot binds the neutral. `count` is the most levels a texture of
    /// the slot's type has on the device, so a texture with more is larger than the device's
    /// largest image of its type: refused, naming that limit.
    pub fn level_views(
        &mut self,
        set: u32,
        binding: u32,
        ty: vk::DescriptorType,
        slot: SlotRef,
        count: u32,
    ) -> Result<Vec<WriteSource>, BindError> {
        if ty != vk::DescriptorType::STORAGE_IMAGE
            || !self.has_fact(RuntimeFact::StorageTextureLevelViews { slot })
        {
            return Err(internal(format!(
                "a {ty:?} binding of {count} descriptors at {slot:?} without level views"
            )));
        }
        let (write, neutral) = self.texture_write(set, binding, ty, slot)?;
        let (view, levels) = match write {
            WriteSource::Texture { view, .. } => (view, self.bound_levels(slot)),
            WriteSource::Neutral(_) => return Ok(vec![write; count as usize]),
            other => return Err(internal(format!("level views of {other:?}"))),
        };
        if levels > count {
            // Level `levels - 1` exists only on a texture at least `2^(levels - 1)` texels wide.
            let t = self.texture_slot(slot)?.texture_type;
            let l = &self.profile.fields().limits;
            let (limit, available) = match t {
                TextureType::Type1D | TextureType::Type1DArray => {
                    (VkLimit::MaxImageDimension1D, l.max_image_dimension_1d)
                }
                TextureType::Cube | TextureType::CubeArray => {
                    (VkLimit::MaxImageDimensionCube, l.max_image_dimension_cube)
                }
                TextureType::Type3D => (VkLimit::MaxImageDimension3D, l.max_image_dimension_3d),
                _ => (VkLimit::MaxImageDimension2D, l.max_image_dimension_2d),
            };
            return Err(BindError::Unsupported(Unsupported::needs(
                Feature::Lowering(LowerFamily::Texture),
                Capability::Limit {
                    limit,
                    required: 1u64 << (levels - 1).min(63),
                    available: u64::from(available),
                },
            )));
        }
        Ok((0..count)
            .map(|i| {
                if i < levels {
                    WriteSource::Texture {
                        slot,
                        view: ImageViewDesc {
                            level: Some(i),
                            ..view
                        },
                    }
                } else {
                    WriteSource::Neutral(neutral)
                }
            })
            .collect())
    }

    /// The level count of the texture the shape binds at `slot`.
    fn bound_levels(&self, slot: SlotRef) -> u32 {
        match self.texture_shape(slot) {
            Some(TextureShape::Texture {
                mipmap_level_count, ..
            }) => *mipmap_level_count,
            _ => 0,
        }
    }

    /// The write of a plain texture binding, and the neutral of its declared type. A slot whose
    /// level count the function reads records the count the binder pushes
    /// (`PushField::TextureLevels`): the bound texture's, or 0 where the neutral stands in (an
    /// unbound slot, or a texture of another type). Such a slot binds a texture whatever its usage,
    /// as plain Metal writes one made without `ShaderWrite` at every level (the consumer binds its
    /// usage shadow where its image lacks the storage usage).
    fn texture_write(
        &mut self,
        set: u32,
        binding: u32,
        ty: vk::DescriptorType,
        slot: SlotRef,
    ) -> Result<(WriteSource, NeutralResource), BindError> {
        // A texture slot that is no texture buffer is declared a uniform texel buffer only under
        // its in-place variant (`VariantKey::ycbcr_in_place`).
        if ty == vk::DescriptorType::UNIFORM_TEXEL_BUFFER
            && self.texture_slot(slot)?.texture_type != TextureType::TextureBuffer
        {
            return Ok(self.in_place_write(slot));
        }
        let levels_read = self.has_fact(RuntimeFact::StorageTextureLevels { slot });
        let write = self.plain_texture(set, binding, ty, slot)?;
        self.write_rounding(slot, &write.0)?;
        if levels_read {
            let levels = match write.0 {
                WriteSource::Texture { .. } => self.bound_levels(slot),
                _ => 0,
            };
            self.template.texture_levels.push((slot, levels));
        }
        if self.has_fact(RuntimeFact::StorageTextureIntegerChannels { slot }) {
            let word = match (&write.0, self.texture_shape(slot)) {
                (
                    WriteSource::Texture { .. } | WriteSource::TexelBuffer { .. },
                    Some(
                        TextureShape::Texture { pixel_format, .. }
                        | TextureShape::Buffer { pixel_format, .. },
                    ),
                ) => crate::metal::format::integer_channels_word(*pixel_format),
                _ => 0,
            };
            self.template.texture_integer_channels.push((slot, word));
        }
        Ok(write)
    }

    /// The refusal of `slot` bound (`write`, not a neutral) to a texture whose mode-2 write the
    /// function states it has not built (`RuntimeFact::StorageTextureWriteRounding`): an
    /// `RG11B10Float` or `RGB9E5Float` texture or texture buffer, and a half-float one where the
    /// function writes `float` texels there (`metal::texel_format::mode_two_rounds`),
    /// `NotImplemented(PrivateRenderState(TextureWriteRounding))`. Any other binding draws as
    /// Metal's.
    fn write_rounding(&self, slot: SlotRef, write: &WriteSource) -> Result<(), BindError> {
        let fact = |halves| RuntimeFact::StorageTextureWriteRounding { slot, halves };
        let halves = self.has_fact(fact(true));
        if !(halves || self.has_fact(fact(false))) || matches!(write, WriteSource::Neutral(_)) {
            return Ok(());
        }
        let pf = match self.texture_shape(slot) {
            Some(
                TextureShape::Texture { pixel_format, .. }
                | TextureShape::Buffer { pixel_format, .. },
            ) => *pixel_format,
            _ => return Ok(()),
        };
        match crate::metal::texel_format::mode_two_rounds(pf, halves) {
            true => Err(BindError::Unsupported(Unsupported::not_implemented(
                Feature::PrivateRenderState(crate::error::PrivateRenderField::TextureWriteRounding),
            ))),
            false => Ok(()),
        }
    }

    /// The write of slot `slot`'s texture binding `(set, binding)` of type `ty`, through the view
    /// [`ResourcePlanner::texel_view`] gives a sampled one, and the neutral of its declared type.
    fn plain_texture(
        &mut self,
        set: u32,
        binding: u32,
        ty: vk::DescriptorType,
        slot: SlotRef,
    ) -> Result<(WriteSource, NeutralResource), BindError> {
        let shape = self.texture_shape(slot).copied();
        let numeric = self.has_fact(RuntimeFact::StorageTextureNumericType { slot });
        let t = *self.texture_slot(slot)?;
        let t = if numeric {
            bound_integer_type(t, shape.as_ref())
        } else {
            t
        };
        // A texture of another numeric class the slot's variant reads in its own class
        // (`VariantKey::cross_class`, which `slots::cross_variant` states): its image is declared
        // of that class, so the binding is the bound texture's view.
        let crossed = self.cross_of(slot, &t).is_some();
        let t = match self.cross_of(slot, &t) {
            Some(ty) => TextureSlot {
                sampled_type: ty,
                ..t
            },
            None => t,
        };
        let cube_as_array = self.has_fact(RuntimeFact::ImageViewShape { slot });
        let storage = matches!(
            ty,
            vk::DescriptorType::STORAGE_IMAGE | vk::DescriptorType::STORAGE_TEXEL_BUFFER
        );
        // A slot a `ulong` atomic updates is declared `R64ui`: its texture is viewed as
        // `R64_UINT`, whatever choice its format would otherwise make.
        let ulong_atomic = storage && self.has_fact(RuntimeFact::UlongAtomicView { slot });
        let base = declared(&t, ty, cube_as_array, false, StorageFormatClass::Unknown)?;
        let compat = shape.filter(|s| compatible(s, &t, &base));
        // A texture whose writes Metal drops (a 4:2:2 texture, and a YCbCr one;
        // `metal::format::writes_dropped`) at a storage slot its function only writes: plain Metal
        // completes the writes and stores nothing (a filled texture keeps its bytes; its validation
        // stops the binding), so the slot takes its neutral, which the writes leave as Metal leaves
        // the texture. A slot that reads it keeps the format's refusal (`metal::format`).
        let only_written = storage && t.access.write && !t.access.read && !t.access.sample;
        let compat = compat.filter(|s| {
            !(only_written
                && matches!(s, TextureShape::Texture { .. })
                && writes_dropped(s.pixel_format()))
        });
        // A 4:4:4 two-plane YCbCr texture whose writes Metal stores into its planes
        // (`metal::format::ycbcr_writes_stored`) at a storage slot its function only writes: the
        // function writes each texel's bits and a mark through an `R32G32B32A32_UINT` view of the
        // texture's write shadow, which the consumer encodes into the planes as Metal stores
        // them (`StorageImageClass::YCbCrWrite`). A slot that reads it keeps the format's
        // refusal.
        let ycbcr_write = only_written
            && matches!(compat, Some(TextureShape::Texture { pixel_format, .. })
                if ycbcr_writes_stored(pixel_format));
        // A texture of one of Apple's extended-range formats joins the variant, under which the
        // function reads and writes its raw texels through a view of their unsigned integer
        // format: a texture's, not yet a texture buffer's or a sparse texture's.
        let extended = match compat {
            Some(s) => match ExtendedRange::of(s.pixel_format()) {
                Some(_) if matches!(s, TextureShape::Buffer { .. }) || s.sparse() => {
                    return Err(BindError::Unsupported(Unsupported::not_implemented(
                        Feature::Format,
                    )));
                }
                form => form,
            },
            None => None,
        };
        let base = match extended {
            Some(_) => ImageHandleClass {
                sampled_type: SampledType::Uint,
                ..base
            },
            None if ycbcr_write => ImageHandleClass {
                sampled_type: SampledType::Uint,
                ..base
            },
            None => base,
        };
        // The storage choice of a compatible bound format for the slot's accesses, when its texels
        // are read or written; it joins the variant unless the default declaration serves it and
        // the function does not depend on the format's numeric type.
        let use_ = if t.texture_type == TextureType::TextureBuffer {
            StorageUse::texel_buffer(t.access.read, t.access.write)
        } else {
            StorageUse::texture(t.access.read, t.access.write)
        };
        let accessed = storage && (t.access.read || t.access.write) && !ulong_atomic;
        let choice = match compat {
            Some(TextureShape::Texture { .. }) if ycbcr_write => Some((
                StorageImageChoice::new(
                    vk::Format::R32G32B32A32_UINT,
                    StorageImageClass::YCbCrWrite,
                ),
                vk::Format::R32G32B32A32_UINT,
            )),
            Some(
                TextureShape::Texture { pixel_format, .. }
                | TextureShape::Buffer { pixel_format, .. },
            ) if accessed => {
                let c = storage_choice(pixel_format, use_, self.profile).map_err(bind_error)?;
                let view = direct_format(pixel_format)
                    .ok_or_else(|| internal("a storage texture without a format"))?;
                Some((c, view))
            }
            _ => None,
        };
        // A texture of another numeric class that the slot's variant writes: its choice declares
        // the bound format, whose channels the write saturates or rounds to as Metal stores them
        // (`lower::texture::cross_write`).
        let choice = match choice {
            Some((mut c, view)) if crossed && t.access.write => {
                if c.vk_format == vk::Format::UNDEFINED {
                    c.vk_format = view;
                    c.numeric = None;
                }
                Some((c, view))
            }
            other => other,
        };
        // An `RGB9E5Float` texture through `R32_UINT` (`StorageImageClass::SharedExponentBits`):
        // the slot is declared of the alias's numeric type, as its neutral is.
        let base = match choice {
            Some((c, _)) if c.class == StorageImageClass::SharedExponentBits => ImageHandleClass {
                sampled_type: SampledType::Uint,
                ..base
            },
            _ => base,
        };
        let mut variant = choice
            .filter(|(c, view)| {
                (crossed && t.access.write)
                    || numeric
                    || needs_variant(c, *view, use_, self.profile)
            })
            .map(|(c, _)| c);
        let choice = choice.map(|(c, _)| c);
        if compat.is_none() && accessed {
            // The neutral of the default declaration is `R32` of the slot's numeric type (`RG32`
            // for `ulong`). Where it lacks the without-format usage its uses need (a texel buffer's
            // buffer features carry no implied bit), the slot is declared with that format, as for
            // a bound texture of it, and the neutral of the declared format stands in.
            let neutral = crate::bind::neutral::storage_neutral(base.sampled_type);
            if !accessed_without_format(neutral, use_, self.profile) {
                variant = Some(StorageImageChoice::new(
                    neutral,
                    StorageImageClass::Identity,
                ));
            }
        }
        let class = ImageHandleClass {
            format: if ulong_atomic {
                StorageFormatClass::Format(vk::Format::R64_UINT)
            } else if storage {
                storage_class(variant.as_ref())
            } else {
                StorageFormatClass::NotStorage
            },
            ..base
        };
        let n = neutral(&class)?;
        if let Some(c) = variant {
            self.variant.storage_images.insert(slot, c);
        }
        if let Some(form) = extended {
            self.variant.extended_range.insert(slot, form);
        }
        let Some(shape) = compat else {
            let write = WriteSource::Neutral(self.substitute(slot, n, shape.as_ref(), (&t, &base)));
            return Ok((write, n));
        };
        self.texture_use(slot, &t, storage);
        match shape {
            TextureShape::Buffer {
                pixel_format,
                swizzle,
            } => {
                // Metal applies a texture buffer's swizzle to a function's reads, and a texel
                // buffer view has none: the function applies it where its stage states so
                // (`SpecKey::TextureSwizzle`). Every read of a texture-buffer slot does but a
                // `ulong` one, whose swizzled texel Metal is not measured to give; a slot only
                // queried or written reads no texel to swizzle.
                if !swizzle.is_default()
                    && t.access.read
                    && !self.has_fact(RuntimeFact::TextureSwizzle { slot })
                {
                    return Err(BindError::Unsupported(Unsupported::not_implemented(
                        Feature::TextureSwizzle,
                    )));
                }
                let usage = if storage {
                    FormatUsage::StorageTexelBuffer
                } else {
                    FormatUsage::UniformTexelBuffer
                };
                let view = match choice {
                    // The `R64_UINT` view of a `ulong` slot's bytes.
                    _ if ulong_atomic => Some(vk::Format::R64_UINT),
                    // The storage texel view reads the choice's view format (an alias of the
                    // bound format's bytes, or an `A8Unorm` texture buffer's bytes as `R8_UNORM`,
                    // Metal's alpha in red), which must itself support the usage.
                    Some(c) => Some(
                        match (c.alias, c.class) {
                            (Some(_), _) | (None, StorageImageClass::AlphaInRed) => {
                                Some(c.vk_format)
                            }
                            (None, _) => direct_format(pixel_format),
                        }
                        .ok_or_else(|| internal("a storage choice without a format"))?,
                    ),
                    // An `A8Unorm` texture buffer's bytes as `R8_UNORM` where the device views
                    // no `A8_UNORM` texels for the usage: the slot's reads take Metal's alpha
                    // from red through its swizzle, which every texture-buffer read applies but a
                    // `ulong` one, whose texel no `A8Unorm` texture buffer binds.
                    None => match texel_buffer_alpha_in_red(pixel_format, usage, self.profile) {
                        Some(held) => {
                            if t.access.read && !self.has_fact(RuntimeFact::TextureSwizzle { slot })
                            {
                                return Err(BindError::Unsupported(Unsupported::not_implemented(
                                    Feature::Format,
                                )));
                            }
                            self.alpha_red_texels.insert(slot);
                            Some(held)
                        }
                        None => None,
                    },
                };
                let format = match view {
                    Some(view) => {
                        if !self.profile.supports_format(view, usage) {
                            return Err(BindError::Unsupported(Unsupported::needs(
                                Feature::TexelBuffer,
                                Capability::Format {
                                    format: view,
                                    usage,
                                },
                            )));
                        }
                        view
                    }
                    None => {
                        let usages = [usage].into();
                        let m =
                            vk_format(pixel_format, &usages, self.profile).map_err(bind_error)?;
                        m.format
                    }
                };
                let l = &self.profile.fields().limits;
                let block = format
                    .block()
                    .ok_or_else(|| internal(format!("a texel view of {format:?}")))?;
                self.template.texel_buffers.push(PlannedTexel {
                    set,
                    binding,
                    slot,
                    alignment: l.min_texel_buffer_offset_alignment.max(1),
                    max_elements: u64::from(l.max_texel_buffer_elements),
                    block_size: u64::from(block.size),
                    texels_per_block: u64::from(block.texels),
                });
                let write = WriteSource::TexelBuffer {
                    slot,
                    format,
                    window: crate::pipeline::interface::Window::Whole,
                };
                Ok((write, n))
            }
            TextureShape::Texture {
                pixel_format,
                swizzle,
                ..
            } => {
                // A storage view takes no swizzle (`VUID-VkWriteDescriptorSet-descriptorType-00336`),
                // and plain Metal reads a swizzled texture's texels swizzled through a storage
                // binding and writes them with the swizzle undone: the function does both under
                // the variant, which takes the swizzle as the slot's `SpecKey::TextureSwizzle`,
                // and the view takes none. A `ulong` atomic's slot reads and writes no texel, and
                // keeps the consumer's refusal of the swizzled view. An extended-range texture's
                // sampled view holds its raw codes, which a swizzle in the view's component mapping
                // would move before the function decodes them (a `One` reading the code 1, a border
                // texel unswizzled), where plain Metal swizzles the decoded texel, border texels
                // included: the function applies the swizzle there too, under the
                // same variant, to the texels it reads, samples and gathers, and the view takes
                // none.
                let reads_extended =
                    !storage && extended.is_some() && (t.access.read || t.access.sample);
                // A packed normalized format's texel converts by each channel's own code scale
                // (`TexelFormat::PACKED`), which a swizzle in the view's component mapping would
                // move onto another channel's lane before the function converts it, so its
                // class would drop the field and a read, gather or one-texel sample keep the
                // device's widening of a 4-, 5- or 6-bit code (NVIDIA's and ANV's replicate its
                // bits, where Metal answers `c / (2^b - 1)`): on a device
                // that does not sample through Metal's own sampler, the function converts the
                // unswizzled texel by the format's whole class and swizzles it, under the same
                // variant, and the view takes none.
                let reads_packed = !storage
                    && extended.is_none()
                    && (t.access.read || t.access.sample)
                    && swizzled_packed(self.profile, pixel_format, swizzle);
                let decodes_srgb = !storage
                    && (t.access.read || t.access.sample)
                    && self.swizzles_decoded_srgb(slot, pixel_format, swizzle);
                let shader_swizzle = (accessed || reads_extended || reads_packed || decodes_srgb)
                    && !swizzle.is_default();
                if shader_swizzle {
                    self.variant.swizzled_storage.insert(slot);
                }
                let view = if storage {
                    let format = match choice {
                        _ if ulong_atomic => vk::Format::R64_UINT,
                        Some(c) if c.alias.is_some() => c.vk_format,
                        // The write shadow's own format, and the `R8_UNORM` image an `A8Unorm`
                        // texture holds its alpha in.
                        Some(c)
                            if matches!(
                                c.class,
                                StorageImageClass::YCbCrWrite | StorageImageClass::AlphaInRed
                            ) =>
                        {
                            c.vk_format
                        }
                        _ => direct_format(pixel_format)
                            .ok_or_else(|| internal("a storage texture without a format"))?,
                    };
                    ImageViewDesc {
                        view_type: view_type(&class)?,
                        format,
                        aspect: vk::ImageAspectFlags::COLOR,
                        layout: vk::ImageLayout::GENERAL,
                        components: vk::ComponentMapping::IDENTITY,
                        level: None,
                        shader_swizzle,
                    }
                } else {
                    match self.texel_view(pixel_format, &class)? {
                        Some((view, _)) => ImageViewDesc {
                            shader_swizzle,
                            ..view
                        },
                        None => return Ok((WriteSource::Neutral(n), n)),
                    }
                };
                Ok((WriteSource::Texture { slot, view }, n))
            }
        }
    }

    /// The view of a sampled image of `pf` declared as `class`.
    fn sampled_view(
        &self,
        pf: PixelFormat,
        class: &ImageHandleClass,
    ) -> Result<ImageViewDesc, BindError> {
        sampled_view(self.profile, pf, class)
    }

    /// The view a binding of a texture slot reads a texture of `pf` through (declared `class`):
    /// its sampled view, an sRGB format's decoding its codes. Where the device cannot sample an
    /// 8-bit sRGB format, whose sampled texture it holds as the format's UNORM twin
    /// (`metal::format::srgb_raw_twin`), the twin's sampled view, which reads the codes, under the
    /// slot's `VariantKey::srgb_textures`, whose function decodes them. With the view, the format
    /// whose sampled view it is (the twin's, where the twin serves).
    fn texel_view(
        &self,
        pf: PixelFormat,
        class: &ImageHandleClass,
    ) -> Result<Option<(ImageViewDesc, PixelFormat)>, BindError> {
        let missing = match self.sampled_view(pf, class) {
            Ok(own) => return Ok(Some((own, pf))),
            Err(missing) => missing,
        };
        match srgb_raw_twin(pf) {
            Some(linear) => self
                .sampled_view(linear, class)
                .map(|view| Some((view, linear)))
                .map_err(|_| missing),
            None => Err(missing),
        }
    }

    /// Whether slot `slot`'s function decodes the codes of a texture of `pf` bound there with
    /// `swizzle` by the device's transfer (`RuntimeFact::SrgbTexture`, an 8-bit sRGB format the
    /// device holds as its UNORM twin, `srgb_twin_on`) where the swizzle moves a code between a
    /// color lane and alpha: a view's component mapping would move it before the decode (a color
    /// code onto alpha, read as alpha's `c / 255`; alpha's code onto a color lane, decoded), where plain Metal swizzles the decoded texel. The function then decodes the
    /// unswizzled texel and swizzles it, every access of the slot alike, under
    /// `VariantKey::swizzled_storage`, and the slot's views take no swizzle. A swizzle among the
    /// color lanes, or to a constant (0 and 1 decode to themselves), reads the same either way and
    /// keeps the view's. A format the device samples is decoded by the device's sRGB view, which
    /// decodes before its component mapping swizzles, swizzle and all.
    fn swizzles_decoded_srgb(
        &self,
        slot: SlotRef,
        pf: PixelFormat,
        swizzle: TextureSwizzleChannels,
    ) -> bool {
        use crate::metal::TextureSwizzle as S;
        let color = |s: S| matches!(s, S::Red | S::Green | S::Blue);
        let crosses =
            color(swizzle.alpha) || [swizzle.red, swizzle.green, swizzle.blue].contains(&S::Alpha);
        crosses
            && self.has_fact(RuntimeFact::SrgbTexture { slot })
            && srgb_twin_on(pf, self.profile)
    }

    /// The runtime sampler `slot` for a use, with the shape's value checks
    /// (`metal::sampler::check_sampler`: reduction; address modes, border colors, and LOD
    /// clamps), or the neutral sampler when the slot is unbound. The write keeps the use's
    /// comparison, texel type and explicit cube gradients, which make the `VkSampler`,
    /// and how its opaque-black border is made (`ResourcePlanner::opaque_black`); the shape alone
    /// decides the cube-gradient check, so a descriptor of the plan's shape never fails it per
    /// bind.
    fn runtime_sampler(
        &mut self,
        slot: SlotRef,
        use_: SamplerUse,
        opaque_black: OpaqueBlackBorder,
    ) -> Result<SamplerSource, BindError> {
        let compare = use_.compare;
        let Some(s) = self.sampler_shape(slot).copied() else {
            return Ok(SamplerSource::Neutral(NeutralResource::Sampler { compare }));
        };
        check_sampler(&s, use_, self.profile).map_err(bind_error)?;
        Ok(SamplerSource::Slot {
            slot,
            compare,
            integer: use_.integer_texture,
            opaque_black,
        })
    }

    /// The mapping of `pf`'s sampled view that a sampler without comparison reads
    /// (`metal::format::vk_format`, a depth's [`DEPTH_AS_COLOR`]): the view an opaque-black
    /// border is read through. A comparison's sampler meets depth textures alone, whose
    /// opaque-black texel is their transparent-black one.
    fn sampled_mapping(&self, pf: PixelFormat) -> Result<FormatMapping, BindError> {
        let usages = [FormatUsage::SampledImage].into();
        let m = vk_format(pf, &usages, self.profile).map_err(bind_error)?;
        Ok(FormatMapping {
            components: sampled_components(view_aspect(pf), m.components, false),
            ..m
        })
    }

    /// The border check of an opaque-black border read through the sampled view of `pf` made with
    /// `swizzle` by a sampler made for every view alike (`metal::sampler::check_border_view`).
    fn check_border(
        &self,
        pf: PixelFormat,
        swizzle: TextureSwizzleChannels,
    ) -> Result<(), BindError> {
        let m = self.sampled_mapping(pf)?;
        check_border_view(true, pf, &m, swizzle, self.profile).map_err(bind_error)
    }

    /// How the runtime sampler `slot`, meeting the `textures` (each its format and swizzle), makes
    /// its border where the shape's is opaque black (`metal::sampler::opaque_black_border`):
    /// transparent black where Metal reads that texel through every one of them, which every device
    /// reads exactly; else opaque black as stated where every view reads Metal's texel so, or with
    /// one view mapping every view has (`borderColorSwizzle` without
    /// `borderColorSwizzleFromImage`). Textures whose views need different mappings, or a view the
    /// device cannot read Metal's texel through, refuse (`Unsupported{SamplerBorderColor}`): the
    /// first that does.
    fn opaque_black(
        &self,
        slot: SlotRef,
        textures: &[(PixelFormat, TextureSwizzleChannels)],
    ) -> Result<OpaqueBlackBorder, BindError> {
        if !self
            .sampler_shape(slot)
            .is_some_and(|s| s.opaque_black_border)
        {
            return Ok(OpaqueBlackBorder::Stated);
        }
        if textures
            .iter()
            .all(|&(pf, swizzle)| opaque_black_reads_transparent(pf, swizzle))
        {
            return Ok(OpaqueBlackBorder::Transparent);
        }
        // One sampler for every texture: as stated, or with the one mapping each view needs.
        let mut reads = Vec::with_capacity(textures.len());
        for &(pf, swizzle) in textures {
            let m = self.sampled_mapping(pf)?;
            reads.push(
                crate::metal::sampler::opaque_black_border(pf, &m, swizzle, self.profile)
                    .map_err(bind_error)?,
            );
        }
        match reads.split_first() {
            Some((first, rest)) if rest.iter().all(|r| r == first) => Ok(*first),
            Some(_) => {
                // The views need different mappings: only the view's own swizzle of the border
                // (`borderColorSwizzleFromImage`) serves them all.
                for &(pf, swizzle) in textures {
                    self.check_border(pf, swizzle)?;
                }
                Ok(OpaqueBlackBorder::Stated)
            }
            None => Ok(OpaqueBlackBorder::Stated),
        }
    }

    /// The write of a plain sampler binding fed by `slot` (its interface's comparison, float
    /// border colors). A separate sampler meets its textures in the shader, and the interface
    /// does not state which: a bound sampler with an opaque-black border is made for every
    /// texture the shape binds at a slot of its stage that the function samples
    /// (`ResourcePlanner::sampled_textures`, `ResourcePlanner::opaque_black`; a pair that never
    /// meets can refuse, never read a border that is not Metal's).
    pub fn sampler(&mut self, slot: SlotRef) -> Result<WriteSource, BindError> {
        let compare = match self.slots.get(&slot) {
            Some(SlotInterface::Sampler(SamplerSlot { compare, .. })) => *compare,
            _ => {
                return Err(internal(format!(
                    "sampler binding {slot:?} without its interface"
                )))
            }
        };
        let sampled = self.sampled_textures(slot.stage);
        let opaque_black = self.opaque_black(slot, &sampled)?;
        self.sampler_write(slot, SamplerUse::new(compare, false), opaque_black)
    }

    /// The format and swizzle of each texture the shape binds at a slot of `stage` that the
    /// function samples: what a sampler of the stage may meet where the interface does not say.
    fn sampled_textures(&self, stage: Stage) -> Vec<(PixelFormat, TextureSwizzleChannels)> {
        self.slots
            .iter()
            .filter(|(s, i)| {
                s.stage == stage && matches!(i, SlotInterface::Texture(t) if t.access.sample)
            })
            .filter_map(|(s, _)| match self.texture_shape(*s) {
                Some(&TextureShape::Texture {
                    pixel_format,
                    swizzle,
                    ..
                }) => Some((pixel_format, swizzle)),
                _ => None,
            })
            .collect()
    }

    /// The write of the runtime sampler `slot` for `use_` (`ResourcePlanner::runtime_sampler`),
    /// or the neutral sampler where none is bound.
    fn sampler_write(
        &mut self,
        slot: SlotRef,
        use_: SamplerUse,
        opaque_black: OpaqueBlackBorder,
    ) -> Result<WriteSource, BindError> {
        Ok(match self.runtime_sampler(slot, use_, opaque_black)? {
            SamplerSource::Slot {
                slot,
                compare,
                integer,
                opaque_black,
            } => WriteSource::Sampler {
                slot,
                compare,
                integer,
                opaque_black,
            },
            SamplerSource::Neutral(n) => WriteSource::Neutral(n),
            SamplerSource::Immutable => {
                return Err(internal("a runtime sampler binding is immutable"))
            }
        })
    }

    /// The write of a heap sampler binding fed by the runtime sampler `slot`: the sampler bound
    /// there made for the use `c` (its comparison, the texture's numeric type, and whether it takes
    /// explicit cube gradients in the isotropic form, which makes it isotropic), or the neutral
    /// sampler where none is bound. Its opaque-black border and its filtering meet the
    /// heap's members, which the heap planner checks (`HeapPlanner::array`).
    pub fn heap_sampler(
        &mut self,
        slot: SlotRef,
        c: CombinedSampler,
    ) -> Result<WriteSource, BindError> {
        match self.slots.get(&slot) {
            Some(SlotInterface::Sampler(_)) => {}
            _ => {
                return Err(internal(format!(
                    "heap sampler binding {slot:?} without its interface"
                )))
            }
        }
        self.sampler_write(slot, c.sampler_use(), OpaqueBlackBorder::Stated)
    }

    /// The write of a combined image-sampler binding: the texture's own view type (a cube is a cube
    /// view) with the symbol's comparison, and the runtime sampler for the symbol's use or the
    /// layout's immutable sampler (`immutable`). A bound texture read with an
    /// opaque-black border (the runtime sampler's shape, or the immutable sampler's clamp-to-border
    /// modes and color) must read it exactly through its view
    /// (`metal::sampler::check_border_view`). The immutable sampler's filters must be ones the
    /// bound view's format supports ([`check_filter_features`]), as a runtime sampler's are when it
    /// is instantiated.
    pub fn combined(
        &mut self,
        texture: SlotRef,
        sampler: CombinedSampler,
        immutable: Option<&vk::SamplerDesc>,
    ) -> Result<WriteSource, BindError> {
        let (image, sampler) = self.pair(texture, sampler, immutable)?;
        Ok(WriteSource::CombinedImageSampler { image, sampler })
    }

    /// The write of a separate texture binding: the image [`ResourcePlanner::combined`] writes for
    /// the same pair, with the same checks, where `immutable` is the stage's separate sampler's (a
    /// constexpr one's). A runtime sampler's write meets no image, so here its bound shape's
    /// filtering is checked against the view's format (and a neutral's) at plan time
    /// (`check_shape_filter_features`), as `bind::sampler_desc` checks a combined write's per
    /// bind: a sampler shared by textures of several formats passes for each, and the first pair
    /// that does not is the plan's refusal.
    pub fn separate_texture(
        &mut self,
        texture: SlotRef,
        sampler: CombinedSampler,
        immutable: Option<&vk::SamplerDesc>,
    ) -> Result<WriteSource, BindError> {
        let (image, source) = self.pair(texture, sampler, immutable)?;
        if let SamplerSource::Slot { slot, .. } = source {
            if let Some(shape) = self.sampler_shape(slot).copied() {
                let format = match &image {
                    ImageSource::Texture { view, .. } => Some(view.format),
                    ImageSource::Neutral(n) => n.format(),
                };
                if let Some(format) = format {
                    check_shape_filter_features(
                        self.profile,
                        &shape,
                        sampler.sampler_use(),
                        format,
                    )?;
                }
            }
        }
        Ok(match image {
            ImageSource::Texture { slot, view } => WriteSource::Texture { slot, view },
            ImageSource::Neutral(n) => WriteSource::Neutral(n),
        })
    }

    /// The write of separate sampler binding `sampler` of `stage`, whose layout binding carries
    /// `immutable`: none for a constexpr sampler, which is that immutable sampler, and for a
    /// runtime one the sampler bound at its slot made for the use, or the neutral sampler, its
    /// opaque-black border made for every texture the stage samples, as a plain sampler's
    /// ([`ResourcePlanner::sampler`]). Its textures' writes check each pair
    /// ([`ResourcePlanner::separate_texture`]).
    pub fn separate_sampler(
        &mut self,
        stage: Stage,
        sampler: CombinedSampler,
        immutable: Option<&vk::SamplerDesc>,
    ) -> Result<Option<WriteSource>, BindError> {
        match (sampler.sampler, immutable) {
            (SamplerRef::Constexpr(_), Some(_)) => Ok(None),
            (SamplerRef::Slot(index), None) => {
                let slot = SlotRef {
                    stage,
                    class: SlotClass::Sampler,
                    index,
                };
                match self.slots.get(&slot) {
                    Some(SlotInterface::Sampler(_)) => {}
                    _ => {
                        return Err(internal(format!(
                            "separate sampler binding {slot:?} without its interface"
                        )))
                    }
                }
                let sampled = self.sampled_textures(stage);
                let opaque_black = self.opaque_black(slot, &sampled)?;
                self.sampler_write(slot, sampler.sampler_use(), opaque_black)
                    .map(Some)
            }
            (SamplerRef::Constexpr(_), None) => Err(internal(
                "a constexpr separate sampler without its immutable sampler",
            )),
            (SamplerRef::Slot(_), Some(_)) => Err(internal(
                "a runtime separate sampler with an immutable sampler",
            )),
        }
    }

    /// The image of the pair (`texture`, `sampler`) of a combined or separate binding, with the
    /// pair's checks, and the sampler source of a combined write ([`ResourcePlanner::combined`]).
    fn pair(
        &mut self,
        texture: SlotRef,
        sampler: CombinedSampler,
        immutable: Option<&vk::SamplerDesc>,
    ) -> Result<(ImageSource, SamplerSource), BindError> {
        let t = *self.texture_slot(texture)?;
        // A texture of another numeric class the slot's variant reads in its own class
        // (`VariantKey::cross_class`): the variant's layout declares its image of that class and
        // the sampler's use for it. A layout of another use (the pipeline compiled without the
        // variant, whose plan only names it) takes the declared class's neutral image and the
        // neutral sampler, and no `VkSampler` of the bound state.
        let cross = self.cross_of(texture, &t);
        let unlike = cross.is_some_and(|ty| (ty != SampledType::Float) != sampler.integer);
        let t = match cross {
            Some(ty) if !unlike => TextureSlot {
                sampled_type: ty,
                ..t
            },
            _ => t,
        };
        let class = declared(
            &t,
            vk::DescriptorType::SAMPLED_IMAGE,
            false,
            sampler.compare,
            StorageFormatClass::NotStorage,
        )?;
        let n = neutral(&class)?;
        if unlike {
            let source = match sampler.sampler {
                SamplerRef::Slot(_) => SamplerSource::Neutral(NeutralResource::Sampler {
                    compare: sampler.compare,
                }),
                SamplerRef::Constexpr(_) => SamplerSource::Immutable,
            };
            return Ok((ImageSource::Neutral(n), source));
        }
        let shape = self.texture_shape(texture).copied();
        // With the image, the bound texture the sampler reads: the format whose view it reads (an
        // sRGB format's twin where the twin's view serves) and its swizzle. An extended-range
        // texture's variant binds no sampler.
        let mut extended = false;
        let (image, read) = match shape.filter(|s| compatible(s, &t, &class)) {
            // An extended-range texture is sampled by fetches of its raw texels under the slot's
            // variant, which this plan asks for: the variant's function reads the slot's own image
            // and only the word of a runtime sampler, which must state a filter the fetches
            // emulate.
            Some(TextureShape::Texture {
                pixel_format,
                swizzle,
                ..
            }) if ExtendedRange::of(pixel_format).is_some() => {
                if let Some(form) = ExtendedRange::of(pixel_format) {
                    self.variant.extended_range.insert(texture, form);
                }
                // A swizzled one's function swizzles the decoded texels (`plain_texture`, which
                // the variant's slot takes).
                if !swizzle.is_default() {
                    self.variant.swizzled_storage.insert(texture);
                }
                if let (SamplerRef::Slot(index), Some(form)) =
                    (sampler.sampler, ExtendedRange::of(pixel_format))
                {
                    let slot = SlotRef {
                        stage: texture.stage,
                        class: SlotClass::Sampler,
                        index,
                    };
                    let volume = t.texture_type == TextureType::Type3D;
                    self.extended_sampler(slot, form, volume)?;
                }
                extended = true;
                (ImageSource::Neutral(n), Vec::new())
            }
            Some(TextureShape::Texture {
                pixel_format,
                swizzle,
                ..
            }) => match self.texel_view(pixel_format, &class)? {
                Some((view, viewed)) => {
                    self.texture_use(texture, &t, false);
                    // A swizzled packed normalized texture's sampled view takes no swizzle where
                    // the function converts its texels by the class and swizzles them, under the
                    // variant (`texture_write`'s `reads_packed`): its sampler reads the
                    // unswizzled texture's border texel, which the function swizzles.
                    let shader_swizzle = !swizzle.is_default()
                        && (swizzled_packed(self.profile, pixel_format, swizzle)
                            || self.swizzles_decoded_srgb(texture, pixel_format, swizzle));
                    if shader_swizzle {
                        self.variant.swizzled_storage.insert(texture);
                    }
                    let image = ImageSource::Texture {
                        slot: texture,
                        view: ImageViewDesc {
                            shader_swizzle,
                            ..view
                        },
                    };
                    let seen = match shader_swizzle {
                        true => TextureSwizzleChannels::DEFAULT,
                        false => swizzle,
                    };
                    (image, vec![(viewed, seen)])
                }
                None => (ImageSource::Neutral(n), Vec::new()),
            },
            _ => (
                ImageSource::Neutral(self.substitute(texture, n, shape.as_ref(), (&t, &class))),
                Vec::new(),
            ),
        };
        let sampler = match sampler.sampler {
            // The variant's fetches read the word alone (`extended_sampler`), so this plan, which
            // only names the variant, makes no `VkSampler` of the state: a mode the device's
            // sampler lacks (`MirrorClampToEdge`) is the fetches' to address.
            SamplerRef::Slot(_) if extended => SamplerSource::Neutral(NeutralResource::Sampler {
                compare: sampler.compare,
            }),
            SamplerRef::Slot(index) => {
                let slot = SlotRef {
                    stage: texture.stage,
                    class: SlotClass::Sampler,
                    index,
                };
                let opaque_black = self.opaque_black(slot, &read)?;
                self.runtime_sampler(slot, sampler.sampler_use(), opaque_black)?
            }
            // The layout's sampler is made once for every texture: its border must read Metal's
            // texel through this one's view as it is.
            SamplerRef::Constexpr(_) => {
                if immutable.is_some_and(opaque_black_border) {
                    for &(pf, swizzle) in &read {
                        self.check_border(pf, swizzle)?;
                    }
                }
                SamplerSource::Immutable
            }
        };
        if let (ImageSource::Texture { view, .. }, SamplerSource::Immutable, Some(s)) =
            (&image, &sampler, immutable)
        {
            check_filter_features(self.profile, s, view.format)?;
        }
        Ok((image, sampler))
    }

    /// Check the runtime sampler bound at `slot`, which samples an extended-range texture of
    /// `form` (a 3D one where `volume`) by fetches: its word must state one filter for
    /// minification and magnification (nearest or linear), no anisotropy, and no mip filter, a
    /// nearest one, or a linear one where the form's levels' blend is built
    /// (`ExtendedRange::blends_levels`; not a 3D texture's) and the clamp's maximum is past every
    /// texture's last level (its word's `SamplerWord::LEVEL_MAX` 0, a nearest mip
    /// filter's level at it 15: the function clamps a linear one's level by the clamp's minimum
    /// alone, `lower::sampler::extended`): else `NotImplemented(Format)`. Every address mode is the
    /// fetches' (a `MirrorClampToEdge` axis's taps mirrored once and clamped; a padded form's under
    /// every mode, measured). An unbound slot's word is 0, a nearest sampler's. A word that states
    /// a `Cubic` filter is the plan's cubic-sampler variant's (`VariantKey::cubic_samplers`), whose
    /// function samples through Metal's bicubic filter of its state: the plan refuses a word whose
    /// state that filter is not measured under before it names the variant
    /// (`slots::cubic_variant`), and the variant's function refuses what else the filter's checks
    /// refuse for its sample (`lower::sampler::extended`).
    fn extended_sampler(
        &self,
        slot: SlotRef,
        form: ExtendedRange,
        volume: bool,
    ) -> Result<(), BindError> {
        let word = self
            .stage_shape(slot.stage)
            .and_then(|s| s.samplers.get(&slot.index))
            .map_or(0, |s| s.word.0);
        let filter = word & (SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR);
        let one_filter = filter == 0 || filter == SamplerWord::LINEAR;
        let mip = (word & SamplerWord::MIP_FILTER) >> SamplerWord::MIP_FILTER_SHIFT;
        let levels = match mip {
            0 => true,
            m if m == SamplerMipFilter::Nearest.as_raw() => true,
            m if m == SamplerMipFilter::Linear.as_raw() => {
                form.blends_levels() && !volume && word & SamplerWord::LEVEL_MAX == 0
            }
            _ => false,
        };
        let isotropic = word & SamplerWord::ANISOTROPIC == 0;
        if word & SamplerWord::CUBIC != 0 || (one_filter && levels && isotropic) {
            return Ok(());
        }
        Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::Format,
        )))
    }

    /// The `SamplerState { slot, fields }` specialization: the bound sampler's word masked by
    /// `fields`, the bits the function reads; 0 where no sampler is bound.
    pub fn sampler_state(&self, slot: SlotRef, fields: u32) -> u64 {
        let word = self
            .stage_shape(slot.stage)
            .and_then(|s| s.samplers.get(&slot.index))
            .map_or(0, |s| s.word.0);
        u64::from(word & fields)
    }

    /// The `SamplerMinLevel(slot)` specialization: the level, in sixty-fourths, that Metal's
    /// `calculate_clamped_lod` answers outside a fragment function through the sampler bound to
    /// `slot` before the texture's last level bounds it; 0 for a slot bound to no sampler.
    pub fn sampler_min_level(&self, slot: SlotRef) -> u64 {
        self.stage_shape(slot.stage)
            .and_then(|s| s.samplers.get(&slot.index))
            .map_or(0, |s| u64::from(s.min_level))
    }

    /// The `SamplerAnisotropy(slot)` specialization: `log2` of the degree the sampler bound to
    /// `slot` filters with; 0 for a slot bound to an isotropic sampler or to none.
    pub fn sampler_anisotropy(&self, slot: SlotRef) -> u64 {
        self.stage_shape(slot.stage)
            .and_then(|s| s.samplers.get(&slot.index))
            .map_or(0, |s| u64::from(s.degree))
    }

    /// The `BicubicTexels(slot)` specialization: how Metal's bicubic filter treats the texels of
    /// the format bound at texture `slot` (`metal::bicubic::texels`), with the border texels of the
    /// bound texture's swizzle. A slot whose combined binding reads the neutral (unbound, bound
    /// incompatibly, or bound in an extended-range format, whose texels the slot's variant this
    /// plan asks for samples itself, `VariantKey::extended_range`) takes `BicubicTexels::NEUTRAL`;
    /// another compatible format of no measured family is refused
    /// `NotImplemented(SamplerBicubicFilter)`.
    pub fn bicubic_texels(&self, slot: SlotRef) -> Result<u64, BindError> {
        let t = *self.texture_slot(slot)?;
        let class = declared(
            &t,
            vk::DescriptorType::SAMPLED_IMAGE,
            false,
            false,
            StorageFormatClass::NotStorage,
        )?;
        let texels = match self.texture_shape(slot) {
            _ if self.variant.extended_range.contains_key(&slot) => BicubicTexels::NEUTRAL,
            Some(s @ TextureShape::Texture { pixel_format, .. }) if compatible(s, &t, &class) => {
                bicubic_texels(*pixel_format)
                    .ok_or(BindError::Unsupported(Unsupported::not_implemented(
                        Feature::SamplerBicubicFilter,
                    )))?
                    .with_border(*pixel_format, self.border_swizzle(slot, s))
            }
            _ => BicubicTexels::NEUTRAL,
        };
        Ok(u64::from(texels.0))
    }

    /// The swizzle the border texels of texture `slot`'s class are read through: the bound
    /// texture's, but none for a packed normalized texture whose function swizzles its texels
    /// (`VariantKey::swizzled_storage`, [`swizzled_packed`]), whose border texel the function
    /// swizzles with them.
    fn border_swizzle(&self, slot: SlotRef, shape: &TextureShape) -> TextureSwizzleChannels {
        match self.variant.swizzled_storage.contains(&slot)
            && texel_format(shape.pixel_format()).packed_exact()
        {
            true => TextureSwizzleChannels::default(),
            false => shape.swizzle(),
        }
    }

    /// The `TexelFormat(slot)` specialization: the class of the texture or texture buffer bound at
    /// `slot` (`metal::texel_format::texel_format`), by which the function converts its texels to a
    /// `half` result and emulates its linear samples, with the border texels of the bound texture's
    /// swizzle, where the class states its channels (a cube gather's corner alone states none). A
    /// packed format's field ([`TexelFormat::PACKED`]) is kept only for a texture without a
    /// swizzle, or with one that reads every texel as none does
    /// (`metal::texel_format::reads_unswizzled`: a texture whose alpha reads 1): a swizzle permutes
    /// its channels before the function converts them by their code scales or sums them on their
    /// grids, so a swizzled texture's class is [`TexelFormat::swizzled`], but at a slot whose
    /// function swizzles the converted texel (`VariantKey::swizzled_storage`: a storage slot's; an
    /// extended-range sampled one's; a packed normalized sampled one's on a device without Metal's
    /// own sampler, whose border texels are then the unswizzled texture's). A slot bound to nothing
    /// takes [`TexelFormat::OTHER`]: a neutral reads zero, which every class converts alike.
    pub fn texel_format(&self, slot: SlotRef) -> u64 {
        let class = match self.texture_shape(slot) {
            Some(shape) => {
                let class = match texel_format(shape.pixel_format()) {
                    class if !class.has_channels() => class.0,
                    class => {
                        BicubicTexels(class.0)
                            .with_border(shape.pixel_format(), self.border_swizzle(slot, shape))
                            .0
                    }
                };
                // A slot whose function applies the swizzle (a storage slot's; an extended-range
                // one's; a packed normalized one's) converts the texel before it swizzles it, by
                // its own channels' class; a swizzle that reads every texel as none does keeps the
                // class (an alpha the format lacks read as 1).
                match reads_unswizzled(shape.pixel_format(), shape.swizzle())
                    || self.variant.swizzled_storage.contains(&slot)
                {
                    true => class,
                    false => TexelFormat(class).swizzled().0,
                }
            }
            None => TexelFormat::OTHER.0,
        };
        u64::from(class)
    }

    /// The `UnmeasuredTexels(slot)` specialization: 1 where a texture or texture buffer is bound
    /// at `slot` and its class ([`texel_format`](Self::texel_format)) is unmeasured on a device
    /// that samples through Metal's own texture unit
    /// (`TexelFormat::unmeasured_on_metal_sampler`), else 0, a slot bound to nothing's among
    /// them, and one bound to a placeholder format, which reads its neutral (its zero texel
    /// converts alike under every class).
    pub fn unmeasured_texels(&self, slot: SlotRef) -> u64 {
        match self.texture_shape(slot) {
            Some(shape) if shape.pixel_format() != crate::metal::PixelFormat::Invalid => {
                u64::from(TexelFormat(self.texel_format(slot) as u32).unmeasured_on_metal_sampler())
            }
            _ => 0,
        }
    }

    /// The `PlainHalfTexels(slot)` specialization: 1 where the class of the texture or texture
    /// buffer bound at `slot` ([`texel_format`](Self::texel_format)) converts every texel to
    /// `half` as its `float` value rounded to nearest even on a device that samples through
    /// Metal's own texture unit (`TexelFormat::plain_on_metal_sampler`), else 0; 1 for a slot
    /// bound to nothing and one bound to a placeholder format (its zero texel converts alike
    /// under every class).
    pub fn plain_half_texels(&self, slot: SlotRef) -> u64 {
        match self.texture_shape(slot) {
            Some(shape) if shape.pixel_format() != crate::metal::PixelFormat::Invalid => {
                u64::from(TexelFormat(self.texel_format(slot) as u32).plain_on_metal_sampler())
            }
            _ => 1,
        }
    }

    /// The `TextureSwizzle(slot)` specialization: the swizzle of the texture buffer bound at
    /// `slot`, as its word, which the function applies to its reads, or of the texture bound at a
    /// storage slot of the plan's `VariantKey::swizzled_storage`, which it applies to its reads and
    /// undoes on its writes, or at an extended-range sampled slot of it, which it applies to the
    /// decoded texels it reads, samples and gathers. A slot bound to nothing, or to another
    /// texture, takes the identity's.
    pub fn texture_swizzle(&self, slot: SlotRef) -> u64 {
        let swizzle = match self.texture_shape(slot) {
            Some(&TextureShape::Buffer { swizzle, .. })
                if self.alpha_red_texels.contains(&slot) =>
            {
                swizzle.alpha_in_red()
            }
            Some(&TextureShape::Buffer { swizzle, .. }) => swizzle,
            Some(&TextureShape::Texture { swizzle, .. })
                if self.variant.swizzled_storage.contains(&slot) =>
            {
                swizzle
            }
            _ => TextureSwizzleChannels::DEFAULT,
        };
        u64::from(swizzle.word())
    }
}

/// Whether a sampled slot bound to a texture of `pf` made with `swizzle` swizzles in its function
/// (`VariantKey::swizzled_storage`) on `profile`'s device: a packed normalized format
/// (`TexelFormat::packed_exact`), whose texel the function converts by each channel's own code
/// scale, under a swizzle that reads some texel as no unswizzled texture does
/// (`reads_unswizzled`), on a device that does not sample through Metal's own sampler (where the
/// device's sample is Metal's, swizzle and all, `VkProperty::MetalSampler`).
fn swizzled_packed(
    profile: &DeviceProfile,
    pf: PixelFormat,
    swizzle: TextureSwizzleChannels,
) -> bool {
    !profile
        .fields()
        .enabled
        .properties
        .contains(&crate::device::requirements::VkProperty::MetalSampler)
        && texel_format(pf).packed_exact()
        && !reads_unswizzled(pf, swizzle)
}

#[cfg(test)]
mod tests;
