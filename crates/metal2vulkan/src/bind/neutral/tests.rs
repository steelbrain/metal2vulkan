//! The neutral resources: the zero buffer covers any uniform view, an undeclarable type has none,
//! and their formats follow the declared type and the registry's numeric formats.

use super::*;

#[test]
fn numeric_types_come_from_the_registry_numeric_formats() {
    for (format, ty) in [
        (vk::Format::R8G8B8A8_UNORM, SampledType::Float),
        (vk::Format::R8G8B8A8_SRGB, SampledType::Float),
        (vk::Format::B10G11R11_UFLOAT_PACK32, SampledType::Float),
        (vk::Format::A2B10G10R10_UINT_PACK32, SampledType::Uint),
        (vk::Format::R16G16_SINT, SampledType::Sint),
        (vk::Format::R8G8_USCALED, SampledType::Float),
        (vk::Format::D32_SFLOAT, SampledType::Float),
        (vk::Format::S8_UINT, SampledType::Uint),
        (vk::Format::BC7_UNORM_BLOCK, SampledType::Float),
        (vk::Format::G8B8G8R8_422_UNORM, SampledType::Float),
    ] {
        assert_eq!(numeric_type(format), Some(ty), "{format:?}");
    }
    // A combined depth/stencil format reads per aspect; scaled-integer and boolean formats have
    // no sampled type the core declares.
    for format in [
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::R16G16_SFIXED5_NV,
        vk::Format::R8_BOOL_ARM,
        vk::Format::UNDEFINED,
        vk::Format::from_raw(0x7fff_0000),
    ] {
        assert_eq!(numeric_type(format), None, "{format:?}");
    }
}
