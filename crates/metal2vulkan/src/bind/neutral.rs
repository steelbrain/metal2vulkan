//! The numeric type a declaration of a format takes, and the format of a formatless storage
//! declaration's neutral: the stand-in a consumer binds where nothing compatible is bound, which
//! Vulkan requires for every statically used binding and Metal leaves undefined to read. Its
//! formats are of Vulkan's mandatory format support (`R32_{SFLOAT,SINT,UINT}` for storage images
//! and texel buffers).

use crate::pipeline::sym::SampledType;
use crate::vk;

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

#[cfg(test)]
mod tests;
