//! The neutral resources: the zero buffer covers any uniform view, an undeclarable type has none,
//! and their formats follow the declared type and the registry's numeric formats.

use std::path::PathBuf;

use super::*;
use crate::device::{Device, DeviceDescription, DevicePolicy};

/// A device for every committed profile (`validation/profiles/`, read-only), sorted by stem.
fn committed_devices() -> Vec<(String, Device)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, Device)> = std::fs::read_dir(&dir)
        .expect("validation/profiles is readable")
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).expect("profile is readable");
            let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
            let device =
                Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device derives");
            (stem, device)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(out.len() >= 13, "committed profiles: {}", out.len());
    out
}

#[test]
fn zero_buffer_covers_any_uniform_view_and_is_padded() {
    for (stem, device) in committed_devices() {
        let bytes = zero_buffer_bytes(device.profile());
        let range = device.profile().fields().limits.max_uniform_buffer_range;
        assert_eq!(bytes % 16, 0, "{stem}");
        assert!(bytes >= range.min(1 << 16) & !15, "{stem}");
        assert!(
            bytes >= 16384,
            "{stem}: Vulkan's minimum maxUniformBufferRange"
        );
    }
}

fn class(dim: ImageDim, arrayed: bool, multisampled: bool) -> ImageHandleClass {
    ImageHandleClass {
        dim,
        arrayed,
        multisampled,
        sampled_type: SampledType::Float,
        depth_compare: false,
        access: ImageAccess::Sampled,
        format: StorageFormatClass::NotStorage,
    }
}

#[test]
fn undeclarable_types_have_no_neutral() {
    let none = [
        class(ImageDim::D3, true, false),
        class(ImageDim::D1, false, true),
        class(ImageDim::Cube, false, true),
        class(ImageDim::Buffer, true, false),
        ImageHandleClass {
            sampled_type: SampledType::Uint,
            depth_compare: true,
            ..class(ImageDim::D2, false, false)
        },
        ImageHandleClass {
            access: ImageAccess::Storage,
            depth_compare: true,
            format: StorageFormatClass::Unknown,
            ..class(ImageDim::D2, false, false)
        },
        // A storage format of another numeric type than the declared one.
        ImageHandleClass {
            access: ImageAccess::Storage,
            format: StorageFormatClass::Format(vk::Format::R32_UINT),
            ..class(ImageDim::D2, false, false)
        },
        // Sampled images carry no storage format, storage images always one.
        ImageHandleClass {
            format: StorageFormatClass::Unknown,
            ..class(ImageDim::D2, false, false)
        },
        ImageHandleClass {
            access: ImageAccess::Storage,
            ..class(ImageDim::D2, false, false)
        },
    ];
    // Depth comparison exists only for 2D and cube images.
    let depth = [
        (ImageDim::D1, false),
        (ImageDim::D1, true),
        (ImageDim::D3, false),
        (ImageDim::Buffer, false),
    ]
    .map(|(dim, arrayed)| ImageHandleClass {
        depth_compare: true,
        ..class(dim, arrayed, false)
    });
    for c in none.into_iter().chain(depth) {
        assert_eq!(NeutralResource::for_image(&c), None, "{c:?}");
    }
}

#[test]
fn neutral_formats_follow_the_declared_type() {
    let mut c = class(ImageDim::D2, true, false);
    assert_eq!(
        NeutralResource::for_image(&c).and_then(|n| n.format()),
        Some(vk::Format::R8G8B8A8_UNORM)
    );
    c.sampled_type = SampledType::Sint;
    assert_eq!(
        NeutralResource::for_image(&c).and_then(|n| n.format()),
        Some(vk::Format::R8G8B8A8_SINT)
    );
    c.access = ImageAccess::Storage;
    c.format = StorageFormatClass::Unknown;
    assert_eq!(
        NeutralResource::for_image(&c).and_then(|n| n.format()),
        Some(vk::Format::R32_SINT)
    );
    let texel = ImageHandleClass {
        sampled_type: SampledType::Uint,
        ..class(ImageDim::Buffer, false, false)
    };
    assert_eq!(
        NeutralResource::for_image(&texel),
        Some(NeutralResource::TexelBuffer {
            format: vk::Format::R32_UINT
        })
    );
    assert_eq!(NeutralResource::Sampler { compare: true }.format(), None);
}

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
