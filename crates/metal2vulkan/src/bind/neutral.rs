//! Neutral resources: stand-ins for statically used bindings with nothing (compatible) bound.
//!
//! Metal leaves reading an unbound argument undefined; Vulkan requires a valid descriptor of the
//! declared type for every layout binding. A plan therefore writes a neutral of the exact
//! descriptor, view, and numeric type the shader declares ([`NeutralResource::for_image`]), and
//! [`Device::neutral_kinds`] is the finite set a device can ask for, so the consumer creates each
//! kind once.
//!
//! Neutral image and texel-buffer formats are chosen from Vulkan's mandatory format support
//! (`R8G8B8A8_{UNORM,SINT,UINT}` sampled, `D16_UNORM` depth, `R32_{SFLOAT,SINT,UINT}` storage and
//! texel buffers), except typed storage images, which must use exactly their declared format and
//! are listed only where the profile's format table supports it.


use crate::device::profile::DeviceProfile;
use crate::device::requirements::FormatUsage;
use crate::device::Device;
use crate::pipeline::sym::{
    ImageAccess, ImageDim, ImageHandleClass, SampledType, StorageFormatClass,
};
use crate::vk;

/// A resource the consumer creates once per kind and binds wherever a plan asks for it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NeutralResource {
    /// A zero-filled buffer of at least `min_bytes`, for uniform and storage buffer bindings.
    ZeroBuffer { min_bytes: u32 },
    /// A write-only buffer of at least `min_bytes` that contained stores are redirected to. No
    /// shader ever observes its contents.
    Sink { min_bytes: u32 },
    /// A 1×1(×1) image with one level and one layer per view layer, viewed as `view_type`.
    Image {
        view_type: vk::ImageViewType,
        /// The numeric type the shader reads; the image format has this numeric type.
        sampled_type: SampledType,
        /// A depth image for depth-comparison sampling (depth format).
        depth_compare: bool,
        samples: u32,
        /// `Some(format)` for a storage image of exactly this format; `None` for a sampled image.
        storage_format: Option<vk::Format>,
    },
    /// A texel buffer view of this format.
    TexelBuffer { format: vk::Format },
    /// A sampler; `compare` enables depth comparison.
    Sampler { compare: bool },
}

/// Sample count of every multisampled neutral image: Vulkan requires 4 in
/// `sampledImage{Color,Integer,Depth}SampleCounts` and, with `shaderStorageImageMultisample`,
/// `storageImageSampleCounts`.
const NEUTRAL_SAMPLES: u32 = 4;

impl NeutralResource {
    /// Bytes of the sink buffer.
    pub const SINK_BYTES: u32 = 64;

    /// The neutral for an image or texel-buffer binding whose declared type is `class`, or `None`
    /// when `class` is not a type the core declares (a 3D array, a multisampled image that is not
    /// 2D, a depth-comparison image that is not float, is a storage image, or is not 2D or cube, or
    /// a storage format of another numeric type). Depth comparison is limited to 2D and cube
    /// (arrayed or not, 2D also multisampled) because Metal has no other depth texture type, and
    /// Vulkan does not guarantee a depth format for 1D or 3D images, so a neutral of that shape
    /// could fail image creation.
    pub fn for_image(class: &ImageHandleClass) -> Option<NeutralResource> {
        let storage = class.access == ImageAccess::Storage;
        let format = match (storage, class.format) {
            (false, StorageFormatClass::NotStorage) => None,
            (true, StorageFormatClass::Unknown) => Some(storage_neutral(class.sampled_type)),
            (true, StorageFormatClass::Format(f)) if serves(f, class.sampled_type) => Some(f),
            _ => return None,
        };
        if class.depth_compare
            && (storage
                || class.sampled_type != SampledType::Float
                || !matches!(class.dim, ImageDim::D2 | ImageDim::Cube))
        {
            return None;
        }
        if class.dim == ImageDim::Buffer {
            if class.arrayed || class.multisampled || class.depth_compare {
                return None;
            }
            return Some(NeutralResource::TexelBuffer {
                format: format.unwrap_or_else(|| storage_neutral(class.sampled_type)),
            });
        }
        if class.multisampled && class.dim != ImageDim::D2 {
            return None;
        }
        let view_type = match (class.dim, class.arrayed) {
            (ImageDim::D1, false) => vk::ImageViewType::TYPE_1D,
            (ImageDim::D1, true) => vk::ImageViewType::TYPE_1D_ARRAY,
            (ImageDim::D2, false) => vk::ImageViewType::TYPE_2D,
            (ImageDim::D2, true) => vk::ImageViewType::TYPE_2D_ARRAY,
            (ImageDim::D3, false) => vk::ImageViewType::TYPE_3D,
            (ImageDim::Cube, false) => vk::ImageViewType::CUBE,
            (ImageDim::Cube, true) => vk::ImageViewType::CUBE_ARRAY,
            _ => return None,
        };
        Some(NeutralResource::Image {
            view_type,
            sampled_type: class.sampled_type,
            depth_compare: class.depth_compare,
            samples: if class.multisampled {
                NEUTRAL_SAMPLES
            } else {
                1
            },
            storage_format: format,
        })
    }

    /// The format of the image or texel buffer the consumer creates, or `None` for buffers and
    /// samplers.
    pub fn format(&self) -> Option<vk::Format> {
        match *self {
            NeutralResource::Image {
                sampled_type,
                depth_compare,
                storage_format,
                ..
            } => Some(storage_format.unwrap_or(if depth_compare {
                vk::Format::D16_UNORM
            } else {
                match sampled_type {
                    SampledType::Sint => vk::Format::R8G8B8A8_SINT,
                    SampledType::Uint => vk::Format::R8G8B8A8_UINT,
                    // A `ulong` texture is `RG32Uint`.
                    SampledType::Uint64 => vk::Format::R32G32_UINT,
                    _ => vk::Format::R8G8B8A8_UNORM,
                }
            })),
            NeutralResource::TexelBuffer { format } => Some(format),
            _ => None,
        }
    }
}

/// The format of a formatless storage declaration's neutral of a numeric type: its 32-bit
/// single-channel format (mandatory for storage images and both texel-buffer usages), or, for
/// `ulong`, `RG32_UINT`, the view of the only format Metal binds to it, which its slot declares.
pub(crate) fn storage_neutral(ty: SampledType) -> vk::Format {
    match ty {
        SampledType::Sint => vk::Format::R32_SINT,
        SampledType::Uint => vk::Format::R32_UINT,
        SampledType::Uint64 => vk::Format::R32G32_UINT,
        _ => vk::Format::R32_SFLOAT,
    }
}

/// Whether a storage image of `format` serves a declaration of numeric type `ty`: the format's
/// own numeric type, or, for `ulong`, `RG32_UINT`, and `R64_UINT` where a `ulong` atomic updates
/// the slot.
fn serves(format: vk::Format, ty: SampledType) -> bool {
    match ty {
        SampledType::Uint64 => matches!(format, vk::Format::R32G32_UINT | vk::Format::R64_UINT),
        _ => numeric_type(format) == Some(ty),
    }
}

/// The sampled type a shader declares to read or write `format`: the registry's numeric type of
/// the format's color components, or of its only depth or stencil aspect
/// ([`vk::Format::numeric_type`], generated from `vk.xml`'s `numericFormat`). `None` for a
/// combined depth/stencil format, a scaled-integer or boolean format, `UNDEFINED`, and values
/// the pinned registry does not name.
pub(crate) fn numeric_type(format: vk::Format) -> Option<SampledType> {
    match format.numeric_type()? {
        vk::NumericType::Float => Some(SampledType::Float),
        vk::NumericType::Uint => Some(SampledType::Uint),
        vk::NumericType::Sint => Some(SampledType::Sint),
        vk::NumericType::ScaledSint | vk::NumericType::Bool => None,
    }
}

/// Every image type the new core can declare on `profile` (heap classes and per-slot images): each
/// dimension, arrayed-ness, and sample count SPIR-V allows, each numeric type, depth comparison for
/// float 2D and cube images, sampled or storage with each storage format class the profile
/// supports. Cube arrays need `imageCubeArray`, multisampled storage images
/// `shaderStorageImageMultisample`.
pub fn image_classes(profile: &DeviceProfile) -> Vec<ImageHandleClass> {
    let f = profile.fields();
    let types = [SampledType::Float, SampledType::Sint, SampledType::Uint];
    let shapes = [
        (ImageDim::D1, false, false),
        (ImageDim::D1, true, false),
        (ImageDim::D2, false, false),
        (ImageDim::D2, true, false),
        (ImageDim::D2, false, true),
        (ImageDim::D2, true, true),
        (ImageDim::D3, false, false),
        (ImageDim::Cube, false, false),
        (ImageDim::Cube, true, false),
        (ImageDim::Buffer, false, false),
    ];
    let mut out = Vec::new();
    for (dim, arrayed, multisampled) in shapes {
        if dim == ImageDim::Cube && arrayed && !f.images.cube_array {
            continue;
        }
        let class = |sampled_type, depth_compare, access, format| ImageHandleClass {
            dim,
            arrayed,
            multisampled,
            sampled_type,
            depth_compare,
            access,
            format,
        };
        for ty in types {
            out.push(class(
                ty,
                false,
                ImageAccess::Sampled,
                StorageFormatClass::NotStorage,
            ));
            // Depth textures exist only as 2D (arrayed, multisampled) and cube (arrayed) in
            // Metal, and only those shapes are guaranteed a depth format in Vulkan.
            if ty == SampledType::Float && matches!(dim, ImageDim::D2 | ImageDim::Cube) {
                out.push(class(
                    ty,
                    true,
                    ImageAccess::Sampled,
                    StorageFormatClass::NotStorage,
                ));
            }
            if multisampled && !f.images.storage_multisample {
                continue;
            }
            out.push(class(
                ty,
                false,
                ImageAccess::Storage,
                StorageFormatClass::Unknown,
            ));
            let usage = if dim == ImageDim::Buffer {
                FormatUsage::StorageTexelBuffer
            } else {
                FormatUsage::StorageImage
            };
            for (&format, usages) in &f.storage_formats.formats {
                if usages.contains(&usage) && numeric_type(format) == Some(ty) {
                    let format = StorageFormatClass::Format(format);
                    out.push(class(ty, false, ImageAccess::Storage, format));
                }
            }
        }
    }
    out
}

impl Device {}

/// Bytes of the `ZeroBuffer` neutral on `profile`: large enough for any uniform-buffer view
/// (`maxUniformBufferRange`, capped at 64 KiB), a multiple of 16.
pub fn zero_buffer_bytes(profile: &DeviceProfile) -> u32 {
    let range = profile
        .fields()
        .limits
        .max_uniform_buffer_range
        .min(1 << 16);
    (range & !15).max(16)
}

#[cfg(test)]
mod tests;
