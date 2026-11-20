//! The required test: on every committed profile, every image type the core can declare has a
//! neutral in `Device::neutral_kinds()` that satisfies Vulkan's view/format compatibility rules.

use std::collections::BTreeSet;
use std::path::PathBuf;

use super::*;
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::testing::is_moltenvk_capture;

/// Formats whose use here is guaranteed by Vulkan's mandatory format support.
const MANDATORY: &[(vk::Format, FormatUsage)] = &[
    (vk::Format::R8G8B8A8_UNORM, FormatUsage::SampledImage),
    (vk::Format::R8G8B8A8_SINT, FormatUsage::SampledImage),
    (vk::Format::R8G8B8A8_UINT, FormatUsage::SampledImage),
    (vk::Format::D16_UNORM, FormatUsage::SampledImage),
    (vk::Format::R32_SFLOAT, FormatUsage::StorageImage),
    (vk::Format::R32_SINT, FormatUsage::StorageImage),
    (vk::Format::R32_UINT, FormatUsage::StorageImage),
    (vk::Format::R32_SFLOAT, FormatUsage::UniformTexelBuffer),
    (vk::Format::R32_SINT, FormatUsage::UniformTexelBuffer),
    (vk::Format::R32_UINT, FormatUsage::UniformTexelBuffer),
    (vk::Format::R32_SFLOAT, FormatUsage::StorageTexelBuffer),
    (vk::Format::R32_SINT, FormatUsage::StorageTexelBuffer),
    (vk::Format::R32_UINT, FormatUsage::StorageTexelBuffer),
];

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
                Device::from_queries(desc, DevicePolicy::default()).expect("device derives");
            (stem, device)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(out.len() >= 13, "committed profiles: {}", out.len());
    out
}

/// Whether `profile` supports `usage` on `format`: its format table decides when it lists the
/// format; otherwise only Vulkan's mandatory support counts.
fn supported(profile: &DeviceProfile, format: vk::Format, usage: FormatUsage) -> bool {
    let f = profile.fields();
    let table = match usage {
        FormatUsage::StorageImage | FormatUsage::StorageTexelBuffer => &f.storage_formats,
        _ => &f.image_formats,
    };
    match table.formats.get(&format) {
        Some(usages) => usages.contains(&usage),
        None => MANDATORY.contains(&(format, usage)),
    }
}

/// The view type an image of `dim`/`arrayed` must be viewed as (VUID-vkCmdDraw-viewType-07752).
fn view_type_for(dim: ImageDim, arrayed: bool) -> vk::ImageViewType {
    match (dim, arrayed) {
        (ImageDim::D1, false) => vk::ImageViewType::TYPE_1D,
        (ImageDim::D1, true) => vk::ImageViewType::TYPE_1D_ARRAY,
        (ImageDim::D2, false) => vk::ImageViewType::TYPE_2D,
        (ImageDim::D2, true) => vk::ImageViewType::TYPE_2D_ARRAY,
        (ImageDim::D3, false) => vk::ImageViewType::TYPE_3D,
        (ImageDim::Cube, false) => vk::ImageViewType::CUBE,
        (ImageDim::Cube, true) => vk::ImageViewType::CUBE_ARRAY,
        other => panic!("no view type for {other:?}"),
    }
}

/// Checks the Vulkan compatibility rules between a declared image type and its neutral.
fn assert_compatible(stem: &str, class: &ImageHandleClass, n: &NeutralResource, p: &DeviceProfile) {
    let ctx = format!("{stem}: {class:?} -> {n:?}");
    let format = n.format().unwrap_or_else(|| panic!("{ctx}: no format"));
    // The view format's numeric type matches the declared sampled type
    // (VUID-vkCmdDraw-format-07753); a `ulong` slot's image is `RG32` unsigned.
    assert!(serves(format, class.sampled_type), "{ctx}");
    assert_eq!(
        numeric_type(format),
        Some(match class.sampled_type {
            SampledType::Uint64 => SampledType::Uint,
            other => other,
        }),
        "{ctx}"
    );
    let storage = class.access == ImageAccess::Storage;
    // A typed storage image is viewed with exactly its declared format
    // (VUID-vkCmdDraw-OpTypeImage-07027/07028); formatless ones need a storage-capable format.
    if let StorageFormatClass::Format(declared) = class.format {
        assert_eq!(format, declared, "{ctx}");
    }
    match *n {
        NeutralResource::TexelBuffer { .. } => {
            assert_eq!(class.dim, ImageDim::Buffer, "{ctx}");
            let usage = if storage {
                FormatUsage::StorageTexelBuffer
            } else {
                FormatUsage::UniformTexelBuffer
            };
            assert!(supported(p, format, usage), "{ctx}: {usage:?}");
        }
        NeutralResource::Image {
            view_type,
            depth_compare,
            samples,
            storage_format,
            ..
        } => {
            assert_ne!(class.dim, ImageDim::Buffer, "{ctx}");
            assert_eq!(view_type, view_type_for(class.dim, class.arrayed), "{ctx}");
            // MS=1 needs a multisampled view and MS=0 a single-sampled one
            // (VUID-vkCmdDraw-MS-07288 and friends).
            assert_eq!(samples > 1, class.multisampled, "{ctx}");
            assert_eq!(storage_format.is_some(), storage, "{ctx}");
            assert_eq!(depth_compare, class.depth_compare, "{ctx}");
            if depth_compare {
                assert_eq!(format, vk::Format::D16_UNORM, "{ctx}: depth format");
                // Depth formats are guaranteed (and Metal depth textures exist) only for 2D and
                // cube images, so a depth neutral of another shape could fail image creation.
                assert!(
                    matches!(
                        view_type,
                        vk::ImageViewType::TYPE_2D
                            | vk::ImageViewType::TYPE_2D_ARRAY
                            | vk::ImageViewType::CUBE
                            | vk::ImageViewType::CUBE_ARRAY
                    ),
                    "{ctx}: depth view type"
                );
            }
            if view_type == vk::ImageViewType::CUBE_ARRAY {
                assert!(p.fields().images.cube_array, "{ctx}: imageCubeArray");
            }
            if storage && samples > 1 {
                assert!(p.fields().images.storage_multisample, "{ctx}");
            }
            let usage = if storage {
                FormatUsage::StorageImage
            } else {
                FormatUsage::SampledImage
            };
            assert!(supported(p, format, usage), "{ctx}: {usage:?}");
        }
        _ => panic!("{ctx}: not an image or texel buffer"),
    }
}

#[test]
fn every_declarable_image_has_a_compatible_neutral_on_every_profile() {
    for (stem, device) in committed_devices() {
        let profile = device.profile();
        let kinds: BTreeSet<NeutralResource> = device.neutral_kinds().into_iter().collect();
        let mut classes = image_classes(profile);
        assert!(!classes.is_empty(), "{stem}");
        classes.extend(ulong_classes(profile));
        let depth_dims: BTreeSet<(ImageDim, bool, bool)> = classes
            .iter()
            .filter(|c| c.depth_compare)
            .map(|c| (c.dim, c.arrayed, c.multisampled))
            .collect();
        for shape in [
            (ImageDim::D2, false, false),
            (ImageDim::D2, true, false),
            (ImageDim::D2, false, true),
            (ImageDim::D2, true, true),
            (ImageDim::Cube, false, false),
        ] {
            assert!(depth_dims.contains(&shape), "{stem}: depth {shape:?}");
        }
        assert!(
            depth_dims
                .iter()
                .all(|&(dim, _, _)| matches!(dim, ImageDim::D2 | ImageDim::Cube)),
            "{stem}: depth only for 2D and cube: {depth_dims:?}"
        );
        for class in &classes {
            let n = NeutralResource::for_image(class)
                .unwrap_or_else(|| panic!("{stem}: no neutral for {class:?}"));
            assert!(
                kinds.contains(&n),
                "{stem}: {n:?} missing from neutral_kinds"
            );
            assert_compatible(&stem, class, &n, profile);
        }
        for n in [
            NeutralResource::ZeroBuffer {
                min_bytes: zero_buffer_bytes(profile),
            },
            NeutralResource::Sink { min_bytes: 64 },
            NeutralResource::Sampler { compare: false },
            NeutralResource::Sampler { compare: true },
        ] {
            assert!(kinds.contains(&n), "{stem}: {n:?} missing");
        }
    }
}

#[test]
fn neutral_kinds_are_exactly_the_derived_set() {
    for (stem, device) in committed_devices() {
        let kinds = device.neutral_kinds();
        let sorted: Vec<NeutralResource> = kinds
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(kinds, sorted, "{stem}: fixed order, no duplicates");
        let images: BTreeSet<NeutralResource> = image_classes(device.profile())
            .iter()
            .chain(&ulong_classes(device.profile()))
            .filter_map(NeutralResource::for_image)
            .collect();
        assert_eq!(kinds.len(), images.len() + 4, "{stem}");
    }
}

/// A `ulong` slot that a `ulong` atomic updates is declared `R64ui`: its `R64_UINT` neutral is
/// listed exactly where the profile has what that slot needs, image and texel buffer apart, and
/// never on MoltenVK, which has no 64-bit image atomics.
#[test]
fn a_ulong_atomic_neutral_is_listed_exactly_where_its_slot_compiles() {
    let r64 = |c: &ImageHandleClass| {
        c.sampled_type == SampledType::Uint64
            && c.format == StorageFormatClass::Format(vk::Format::R64_UINT)
    };
    let mut listed = 0;
    for (stem, device) in committed_devices() {
        let profile = device.profile();
        for buffer in [false, true] {
            let compiles = profile.ulong_atomic_gaps(buffer).is_empty();
            let n = ulong_classes(profile)
                .iter()
                .filter(|c| r64(c) && (c.dim == ImageDim::Buffer) == buffer)
                .count();
            // Five image shapes (1D, 1D array, 2D, 2D array, 3D), or the texture buffer.
            let shapes = if buffer { 1 } else { 5 };
            assert_eq!(
                n,
                if compiles { shapes } else { 0 },
                "{stem}: buffer {buffer}"
            );
            listed += n;
        }
        if is_moltenvk_capture(&stem) {
            assert!(!ulong_classes(profile).iter().any(r64), "{stem}");
        }
    }
    assert!(listed > 0, "no committed profile lists an R64_UINT neutral");
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
