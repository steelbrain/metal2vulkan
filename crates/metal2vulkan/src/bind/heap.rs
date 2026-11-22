//! The set-1 heap's classes, as the lowering reads them.
//!
//! The heap arrays on a profile are of the image classes the core can declare and the four
//! sampler classes. The texture lowering declares the handle table (`Synthetic(HandleTable)`, one
//! per stage that loads a heap texture) and one heap array per class a heap texture reaches, and
//! link numbers them (set 1 for the arrays). The consumer writes the arrays and the tables; this
//! module holds what the lowering decides from a class:
//!
//! - [`fit`]: how a texture fits a class's array: the Metal texture type's dimension,
//!   arrayedness, and multisampling, a sample count above 1 exactly when multisampled, the numeric
//!   type, a depth format for a depth-comparison class, `ShaderRead` usage for a sampled class and
//!   `ShaderWrite` or Apple's private `BlockWritesOnly` for a storage one (Metal's validation takes
//!   a write of either), which never holds a depth or stencil format. A storage class holds a
//!   texture whose writes Metal drops (`metal::format::writes_dropped`) as its neutral
//!   ([`Fit::Dropped`]), and a sampled float class holds a form a heap reads by its raw texels as
//!   its neutral ([`Fit::Raw`]).
//! - [`tier_read_gaps`]: Metal's contract defines a `read_write` read of its read-write tier
//!   formats, so a function that reads a storage class through a handle is refused where one of
//!   them would be held for its writes alone: on a device that reads storage without a format by
//!   the formats' own bits alone (`DeviceProfile::storage_reads_by_format_bits`), a storage view
//!   whose format lacks the read bit.
//! - [`storage_conversions`]: the aliases a storage access through a handle converts by (an 8-bit
//!   sRGB texture's UNORM twin, a blue-first texture's RGBA-order twin), where the handle carries
//!   the texture's class and red-blue order.
//! - The bits of `SpecKey::HeapRawTextures` ([`HELD`], [`FILTERED`]) and of
//!   `SpecKey::HeapNanFamilies` ([`NAN_SIXTEEN`], [`NAN_WIDE`], [`NAN_SMALL`], [`NAN_OTHER`]).

use std::collections::{BTreeMap, BTreeSet};

use crate::bind::state::{TextureFacts, TextureShape};
use crate::device::requirements::FormatUsage;
use crate::device::DeviceProfile;
use crate::error::{BindError, Error, Feature, InternalError, InternalKind, Unsupported};
use crate::metal::format::{
    is_depth_stencil, red_blue_exchanged, sampled_type, storage_choice, storage_view_formats,
    vk_format, without_format_gaps, writes_dropped, StorageUse,
};
use crate::metal::{self, TextureType};
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{ImageAccess, ImageDim, ImageHandleClass, SampledType};
use crate::vk;

/// An invariant failure of the view derivation.
fn internal(detail: impl Into<String>) -> BindError {
    BindError::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// A `metal`/`device` error as a [`BindError`]: capability refusals stay typed, a descriptor
/// error stays the descriptor's, anything else is an invariant failure.
fn bind_error(e: Error) -> BindError {
    match e {
        Error::Unsupported(u) => BindError::Unsupported(u),
        Error::InvalidDescriptor(d) => BindError::InvalidDescriptor(d),
        Error::Internal(i) => BindError::Internal(i),
        other => internal(format!("{other:?}")),
    }
}

/// How a member fits a heap class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fit {
    /// The member is an element of the class's array.
    Holds,
    /// An unswizzled texture of a form a heap reads (`ExtendedRange::HEAP_READ`, not sparse) in a
    /// sampled float class: the class's array holds its neutral, and the unsigned integer class
    /// of its shape holds its raw texels, which a function's read decodes by the form its
    /// handle's class index names. A stage that samples heap textures does not read it so.
    Raw,
    /// It fits in everything but its numeric type.
    NumericType,
    /// A storage class's member whose writes Metal drops (`metal::format::writes_dropped`): the
    /// class's array holds its neutral, which the writes leave as Metal leaves the texture.
    Dropped,
    /// It is another class's.
    No,
}

/// The dimension, arrayedness, and multisampling of a Metal texture type.
fn dims(t: TextureType) -> (ImageDim, bool, bool) {
    use TextureType as T;
    match t {
        T::Type1D => (ImageDim::D1, false, false),
        T::Type1DArray => (ImageDim::D1, true, false),
        T::Type2D => (ImageDim::D2, false, false),
        T::Type2DArray => (ImageDim::D2, true, false),
        T::Type2DMultisample => (ImageDim::D2, false, true),
        T::Type2DMultisampleArray => (ImageDim::D2, true, true),
        T::Cube => (ImageDim::Cube, false, false),
        T::CubeArray => (ImageDim::Cube, true, false),
        T::Type3D => (ImageDim::D3, false, false),
        T::TextureBuffer => (ImageDim::Buffer, false, false),
    }
}

/// How a texture of view facts `shape` fits the heap array of `class` (the module docs).
pub(crate) fn fit(shape: &TextureShape, class: &ImageHandleClass) -> Fit {
    let (texture_type, pixel_format, sample_count, usage) = match *shape {
        TextureShape::Texture {
            texture_type: TextureType::TextureBuffer,
            pixel_format,
            usage,
            ..
        } => return buffer_fit(pixel_format, usage, class),
        TextureShape::Texture {
            texture_type,
            pixel_format,
            sample_count,
            usage,
            ..
        } => (texture_type, pixel_format, sample_count, usage),
    };
    let storage = class.access == ImageAccess::Storage;
    // Metal's validation takes a write to a texture of `BlockWritesOnly` as one of `ShaderWrite`.
    let needs = if storage {
        metal::TextureUsage::SHADER_WRITE | metal::TextureUsage::BLOCK_WRITES_ONLY
    } else {
        metal::TextureUsage::SHADER_READ
    };
    let shaped = dims(texture_type) == (class.dim, class.arrayed, class.multisampled)
        && class.dim != ImageDim::Buffer
        && (sample_count > 1) == class.multisampled
        && (!class.depth_compare || is_depth_stencil(pixel_format))
        && usage.intersects(needs)
        && !(storage && is_depth_stencil(pixel_format));
    // A resident texture of a format the table holds no representation of (ASTC's 4x2 and 8x4
    // footprints) fits no class: its element is the neutral, and nothing is refused, since the heap
    // does not know which of its textures the function reads.
    match sampled_type(pixel_format) {
        _ if !shaped => Fit::No,
        _ if storage && writes_dropped(pixel_format) => Fit::Dropped,
        // A form a heap reads: its raw texels in the unsigned integer class, the neutral in the
        // float one.
        _ if raw_held(shape) && class.access == ImageAccess::Sampled => match class.sampled_type {
            SampledType::Uint => Fit::Holds,
            SampledType::Float => Fit::Raw,
            _ => Fit::NumericType,
        },
        Some(t) if t == class.sampled_type => Fit::Holds,
        Some(_) => Fit::NumericType,
        None => Fit::No,
    }
}

/// How a texture buffer of `pixel_format` and `usage` fits the heap array of `class`: a
/// texel-buffer class of its numeric type holds it where its usage has the class's access, as an
/// image class's member's must, and one of another numeric type fits it in everything but that.
fn buffer_fit(
    pixel_format: metal::PixelFormat,
    usage: metal::TextureUsage,
    class: &ImageHandleClass,
) -> Fit {
    let needs = match class.access {
        ImageAccess::Storage => {
            metal::TextureUsage::SHADER_WRITE | metal::TextureUsage::BLOCK_WRITES_ONLY
        }
        _ => metal::TextureUsage::SHADER_READ,
    };
    if class.dim != ImageDim::Buffer
        || class.arrayed
        || class.multisampled
        || class.depth_compare
        || !usage.intersects(needs)
    {
        return Fit::No;
    }
    match sampled_type(pixel_format) {
        _ if class.access == ImageAccess::Storage && writes_dropped(pixel_format) => Fit::Dropped,
        Some(t) if t == class.sampled_type => Fit::Holds,
        Some(_) => Fit::NumericType,
        None => Fit::No,
    }
}

/// The format of the texel buffer view through which the texel-buffer class `class` holds a
/// texture buffer of `pixel_format` and `swizzle` that fits it ([`buffer_fit`]): its direct
/// format, which a uniform class's view takes as a slot's does and a storage class's formatless
/// declaration serves for every access the device lets a heap make (`heap_storage_use`). A texel
/// buffer view has no component mapping, so `Unsupported` for a swizzled texture buffer (a slot's
/// read applies the swizzle in its function, a heap read does not), for a format the device views
/// only through a converting or swizzled stand-in, and for one it cannot view for the class's
/// usage or access without a format.
fn member_texel_view(
    profile: &DeviceProfile,
    pixel_format: metal::PixelFormat,
    swizzle: metal::TextureSwizzleChannels,
    class: &ImageHandleClass,
) -> Result<vk::Format, BindError> {
    if !swizzle.is_default() {
        return Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::TextureSwizzle,
        )));
    }
    let storage = class.access == ImageAccess::Storage;
    let usage = match storage {
        true => FormatUsage::StorageTexelBuffer,
        false => FormatUsage::UniformTexelBuffer,
    };
    let m = vk_format(pixel_format, &[usage].into(), profile).map_err(bind_error)?;
    if m.emulation != crate::metal::format::FormatEmulation::Direct {
        return Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::ArgumentBufferTextures,
        )));
    }
    if storage {
        let images = &profile.fields().images;
        let use_ = StorageUse::texel_buffer(
            images.storage_read_without_format,
            images.storage_write_without_format,
        );
        let gaps = without_format_gaps(m.format, use_, profile);
        if !gaps.is_empty() {
            return Err(BindError::Unsupported(Unsupported::needs_any_of(
                Feature::ArgumentBufferTextures,
                vec![gaps],
            )));
        }
    }
    Ok(m.format)
}

/// Whether a member of facts `shape` is a texture a heap reads by its raw texels: an unswizzled,
/// not sparse texture of one sample, of a form of `ExtendedRange::HEAP_READ`.
pub(crate) fn raw_held(shape: &TextureShape) -> bool {
    let TextureShape::Texture {
        pixel_format,
        swizzle,
        sparse,
        sample_count,
        ..
    } = *shape;
    !sparse
        && sample_count <= 1
        && swizzle.is_default()
        && crate::metal::format::ExtendedRange::of(pixel_format)
            .and_then(crate::metal::format::ExtendedRange::heap_read)
            .is_some()
}

/// The bit of `SpecKey::HeapRawTextures`' value that a texture the stage's handles can name is held
/// by its raw texels alone, whose decode the function's `float` read, sample or
/// gather takes (`lower::texture::heap_raw`).
pub(crate) const HELD: u32 = 1;

/// The bit of `SpecKey::HeapRawTextures`' value that such a texture is of a form whose sample the
/// function filters (`lower::sampler::heap_raw`): every form of `ExtendedRange::HEAP_READ` but a
/// bfloat one, whose sample reads the nearest texel under either filter.
pub(crate) const FILTERED: u32 = 2;

/// The bit of `SpecKey::HeapNanFamilies` of a 16-bit float format: its written NaN, a `half`'s or
/// a `float`'s, is the device's conversion's.
pub(crate) const NAN_SIXTEEN: u32 = 1;

/// The bit of `SpecKey::HeapNanFamilies` of a 32-bit float format: its written `half` NaN is the
/// device's widening's.
pub(crate) const NAN_WIDE: u32 = 2;

/// The bit of `SpecKey::HeapNanFamilies` of a packed small float format (`RG11B10Float`,
/// `RGB9E5Float`): its written NaN is the device's conversion's.
pub(crate) const NAN_SMALL: u32 = 4;

/// The bit of `SpecKey::HeapNanFamilies` of a float format of none of those families whose texels
/// may hold a NaN: its written NaN is the device's.
pub(crate) const NAN_OTHER: u32 = 8;

/// The format of the view through which the storage heap array of `class` holds a member of
/// facts `shape` on `profile`: the direct format (an `A8Unorm` texture held in red its
/// `R8_UNORM` image), which the class's formatless declaration must serve. A member whose choice
/// is an alias a heap access converts by ([`storage_conversions`]) takes that alias.
/// `Unsupported` where the array cannot view it: an extended-range format (its raw texels need
/// its function's decode), a member whose storage choice or format the formatless declaration
/// cannot serve (another alias, or a converting one whose swizzle loses its class), or an
/// `A8Unorm` one held in red whose swizzle moves its alpha (its handle's class is not
/// `A8Unorm`'s).
fn member_storage_format(
    profile: &DeviceProfile,
    shape: &TextureShape,
) -> Result<vk::Format, BindError> {
    let TextureShape::Texture { pixel_format, .. } = *shape;
    if crate::metal::format::ExtendedRange::of(pixel_format).is_some() {
        return Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::ArgumentBufferTextures,
        )));
    }
    // The class does not say which accesses the function makes, so the member serves
    // every access the device lets a heap make.
    let use_ = heap_storage_use(profile);
    let c = storage_choice(pixel_format, use_, profile).map_err(bind_error)?;
    // The class's one formatless declaration cannot convert per element: a member is viewed
    // through its own format.
    let per_element = c.class == StorageImageClass::Identity;
    let aliased = c.alias.is_some();
    // An `A8Unorm` texture held in red (`StorageImageClass::AlphaInRed`): its `R8_UNORM`
    // image, whose texels a heap access converts where the handle carries `A8Unorm`'s class
    // (`metal::texel_format::heap_alpha_in_red_index`), which a swizzle that moves its alpha
    // does not keep.
    let alpha_in_red = c.class == StorageImageClass::AlphaInRed;
    // A texture viewed through its storage choice's alias, whose channels a heap access
    // exchanges or whose sRGB codes it decodes and encodes where the handle carries its
    // class and red-blue order ([`storage_conversions`]): the alias, which its image lists
    // among its view formats.
    let converted = aliased
        && !c.x_channel
        && storage_conversions(profile).contains(&StorageConversion {
            index: crate::metal::texel_format::heap_class_index(pixel_format, shape.swizzle()),
            red_blue: red_blue_exchanged(pixel_format),
            class: c.class,
        })
        && storage_view_formats(pixel_format, profile).contains(&c.vk_format);
    if alpha_in_red
        && crate::metal::texel_format::heap_class_index(pixel_format, shape.swizzle())
            != crate::metal::texel_format::heap_alpha_in_red_index()
    {
        return Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::ArgumentBufferTextures,
        )));
    }
    if !(per_element || alpha_in_red || converted) || (aliased && !converted) || c.x_channel {
        return Err(BindError::Unsupported(Unsupported::not_implemented(
            Feature::ArgumentBufferTextures,
        )));
    }
    let format = match alpha_in_red || converted {
        true => c.vk_format,
        false => crate::metal::format::direct_format(pixel_format)
            .ok_or_else(|| internal("a storage heap member without a format"))?,
    };
    // The declaration names no format, so the member's format must serve those accesses
    // without one. Where the formats' own bits alone grant a read, a member whose view's
    // format lacks the read bit is held for its writes and flagged
    // (`memory::handles::FLAG_STORAGE_UNREADABLE`).
    let gaps = without_format_gaps(format, use_, profile);
    if !gaps.is_empty() {
        return Err(BindError::Unsupported(Unsupported::needs_any_of(
            Feature::ArgumentBufferTextures,
            vec![gaps],
        )));
    }
    Ok(format)
}

/// What a storage view of `format` (`texel_buffer`: a texel buffer view) lacks for a read without a
/// format on `profile` where the profile reads storage so by the formats' own bits alone
/// (`DeviceProfile::storage_reads_by_format_bits`); empty on every other profile, whose storage
/// arrays hold only views every access the device's features grant serves ([`heap_storage_use`]).
fn storage_read_gaps(
    profile: &DeviceProfile,
    format: vk::Format,
    texel_buffer: bool,
) -> Vec<crate::error::Capability> {
    if !profile.storage_reads_by_format_bits() {
        return Vec::new();
    }
    let use_ = StorageUse {
        texel_buffer,
        read: true,
        write: false,
    };
    without_format_gaps(format, use_, profile)
}

/// What `profile` lacks to read, without a format, the views through which a heap storage array of
/// `class` holds textures of Metal's read-write tier formats on the profile
/// (`device::metal::read_write_tier_formats`), whose `read_write` reads Metal's contract defines:
/// each such view's [`storage_read_gaps`], so empty but where the profile reads storage without a
/// format by the formats' own bits alone. A function that reads the array through a handle needs
/// them (`lower::texture::slot`): without them it would read such a member as
/// the null element (`memory::handles::FLAG_STORAGE_UNREADABLE`).
pub(crate) fn tier_read_gaps(
    profile: &DeviceProfile,
    class: &ImageHandleClass,
) -> Vec<crate::error::Capability> {
    use TextureType as T;
    let mut gaps = Vec::new();
    if !profile.storage_reads_by_format_bits() || class.access != ImageAccess::Storage {
        return gaps;
    }
    let types = [
        T::Type1D,
        T::Type1DArray,
        T::Type2D,
        T::Type2DArray,
        T::Type2DMultisample,
        T::Type2DMultisampleArray,
        T::Cube,
        T::CubeArray,
        T::Type3D,
        T::TextureBuffer,
    ];
    let shape = (class.dim, class.arrayed, class.multisampled);
    let Some(texture_type) = types.into_iter().find(|&t| dims(t) == shape) else {
        return gaps;
    };
    let tiers = crate::device::metal::read_write_tier_formats(profile);
    for &pixel_format in tiers.iter().flat_map(|tier| tier.iter()) {
        let mut facts = TextureFacts::new(texture_type, pixel_format);
        facts.usage = metal::TextureUsage::SHADER_READ | metal::TextureUsage::SHADER_WRITE;
        let member = facts.shape();
        if fit(&member, class) != Fit::Holds {
            continue;
        }
        let view = match class.dim {
            ImageDim::Buffer => {
                member_texel_view(profile, pixel_format, member.swizzle(), class).map(|f| (f, true))
            }
            _ => member_storage_format(profile, &member).map(|f| (f, false)),
        };
        // A tier format the array cannot hold at all is not held for its writes either.
        let Ok((format, texel_buffer)) = view else {
            continue;
        };
        for gap in storage_read_gaps(profile, format, texel_buffer) {
            if !gaps.contains(&gap) {
                gaps.push(gap);
            }
        }
    }
    gaps
}

/// A conversion a storage access through a heap handle makes of the texels of a member its
/// storage array views through the member's storage choice's alias ([`storage_conversions`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct StorageConversion {
    /// The member's texel class index (`metal::texel_format::heap_class_index`), which its
    /// handle carries.
    pub index: u32,
    /// Its pixel format is blue-first (`metal::format::red_blue_exchanged`), which its handle
    /// carries (`memory::handles::FLAG_RED_BLUE`).
    pub red_blue: bool,
    /// The conversion: `SwizzleBgra` exchanges red and blue, `SrgbEncode` decodes and encodes the
    /// sRGB codes of the color lanes by the device's transfer, `SrgbEncodeBgra` both.
    pub class: StorageImageClass,
}

/// The conversions of the storage elements of a heap on `profile`, one for each texel class and
/// red-blue order whose formats' storage choices for every access the device lets a heap make
/// ([`heap_storage_use`]) agree on one alias class that a function converts by (`SwizzleBgra`,
/// `SrgbEncode` or `SrgbEncodeBgra`, as a slot of that choice converts, an RGBX format's left
/// out): an 8-bit sRGB texture where the device stores no sRGB format, and a blue-first one where
/// it stores its RGBA-order twin only. A class and order whose formats choose otherwise (another
/// alias, or `Identity` beside a converting one) has none: its converting member is refused, as
/// before. A member of such a choice is viewed through its alias ([`member_storage_format`]) and the
/// texture lowering converts where the handle carries the class and order
/// (`lower::texture::heap_storage_conversions`); both come from this list, so the two agree.
pub(crate) fn storage_conversions(profile: &DeviceProfile) -> Vec<StorageConversion> {
    let use_ = heap_storage_use(profile);
    let mut chosen: BTreeMap<(u32, bool), BTreeSet<Option<StorageImageClass>>> = BTreeMap::new();
    for &pf in metal::PixelFormat::ALL {
        let Ok(c) = storage_choice(pf, use_, profile) else {
            continue;
        };
        let converts = matches!(
            c.class,
            StorageImageClass::SwizzleBgra
                | StorageImageClass::SrgbEncode
                | StorageImageClass::SrgbEncodeBgra
        ) && c.alias.is_some()
            && !c.x_channel;
        let index = crate::metal::texel_format::heap_class_index(
            pf,
            crate::metal::TextureSwizzleChannels::DEFAULT,
        );
        chosen
            .entry((index, red_blue_exchanged(pf)))
            .or_default()
            .insert(converts.then_some(c.class));
    }
    chosen
        .into_iter()
        .filter_map(|((index, red_blue), classes)| {
            match classes.into_iter().collect::<Vec<_>>().as_slice() {
                [Some(class)] => Some(StorageConversion {
                    index,
                    red_blue,
                    class: *class,
                }),
                _ => None,
            }
        })
        .collect()
}

/// The accesses a storage heap element may take: a read or a write through the class's
/// formatless declaration wherever the device's `shaderStorageImage{Read,Write}WithoutFormat`
/// allows one (the lowering refuses a heap access without its feature, `declare_heap`).
fn heap_storage_use(profile: &DeviceProfile) -> StorageUse {
    let images = &profile.fields().images;
    StorageUse::texture(
        images.storage_read_without_format,
        images.storage_write_without_format,
    )
}

#[cfg(test)]
mod tests;
