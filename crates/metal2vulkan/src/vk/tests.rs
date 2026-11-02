use std::collections::{BTreeSet, HashSet};

use super::*;

/// Round trips and uniqueness for one generated enum type.
macro_rules! check_enum {
    ($($ty:ident),* $(,)?) => {$({
        let known = $ty::KNOWN;
        assert!(!known.is_empty(), "{} has no values", $ty::VK_NAME);
        let raws: BTreeSet<_> = known.iter().map(|v| v.as_raw()).collect();
        assert_eq!(raws.len(), known.len(), "{} repeats a raw value", $ty::VK_NAME);
        let mut names = HashSet::new();
        for &v in known {
            assert_eq!($ty::from_raw(v.as_raw()), v);
            let name = v.name().expect("known values are named");
            assert!(names.insert(name), "{name} is repeated");
            assert_eq!($ty::from_name(name), Some(v), "{name}");
            assert_eq!(format!("{v:?}"), name);
        }
    })*};
}

/// Round trips and single-bit consistency for one generated flags type.
macro_rules! check_flags {
    ($($ty:ident),* $(,)?) => {$({
        let known = $ty::KNOWN;
        assert!(!known.is_empty(), "{} has no values", $ty::VK_NAME);
        let mut names = HashSet::new();
        let mut singles = $ty::empty();
        for &(v, name) in known {
            assert!(names.insert(name), "{name} is repeated");
            assert_eq!($ty::from_name(name), Some(v), "{name}");
            assert_eq!($ty::from_raw(v.as_raw()), v);
            if v.as_raw().count_ones() == 1 {
                assert!(!singles.intersects(v), "{name} reuses a bit");
                singles |= v;
            }
        }
        assert_eq!(singles, $ty::known_bits());
    })*};
}

#[test]
fn generated_enums_round_trip() {
    check_enum!(
        BlendFactor,
        BlendOp,
        BorderColor,
        CompareOp,
        ComponentSwizzle,
        DescriptorType,
        DriverId,
        DynamicState,
        Filter,
        Format,
        FrontFace,
        ImageLayout,
        ImageTiling,
        ImageType,
        ImageViewType,
        IndexType,
        LogicOp,
        PhysicalDeviceType,
        PolygonMode,
        PrimitiveTopology,
        SamplerAddressMode,
        SamplerMipmapMode,
        SamplerReductionMode,
        ShaderFloatControlsIndependence,
        StencilOp,
        VertexInputRate,
    );
}

#[test]
fn generated_flags_round_trip() {
    check_flags!(
        AccessFlags,
        AccessFlags2,
        BufferUsageFlags,
        ColorComponentFlags,
        CullModeFlags,
        DescriptorBindingFlags,
        DescriptorSetLayoutCreateFlags,
        FormatFeatureFlags,
        FormatFeatureFlags2,
        ImageAspectFlags,
        ImageCreateFlags,
        ImageUsageFlags,
        PipelineShaderStageCreateFlags,
        PipelineStageFlags,
        PipelineStageFlags2,
        SampleCountFlags,
        ShaderStageFlags,
        SubgroupFeatureFlags,
    );
}

/// Values fixed by the Vulkan specification, core and extension-numbered, checked independently
/// of the generator.
#[test]
fn registry_values_match_the_specification() {
    assert_eq!(Format::UNDEFINED.as_raw(), 0);
    assert_eq!(Format::R8G8B8A8_UNORM.as_raw(), 37);
    assert_eq!(Format::B8G8R8A8_UNORM.as_raw(), 44);
    assert_eq!(Format::R32G32B32A32_SFLOAT.as_raw(), 109);
    assert_eq!(Format::D32_SFLOAT.as_raw(), 126);
    assert_eq!(Format::ASTC_12X12_SRGB_BLOCK.as_raw(), 184);
    assert_eq!(Format::G8B8G8R8_422_UNORM.as_raw(), 1_000_156_000);
    assert_eq!(Format::A4R4G4B4_UNORM_PACK16.as_raw(), 1_000_340_000);
    assert_eq!(DescriptorType::STORAGE_BUFFER.as_raw(), 7);
    assert_eq!(DescriptorType::INPUT_ATTACHMENT.as_raw(), 10);
    assert_eq!(DriverId::MOLTENVK.as_raw(), 14);
    assert_eq!(DriverId::MESA_LLVMPIPE.as_raw(), 13);
    assert_eq!(DynamicState::STENCIL_REFERENCE.as_raw(), 8);
    assert_eq!(DynamicState::CULL_MODE.as_raw(), 1_000_267_000);
    assert_eq!(SamplerAddressMode::MIRROR_CLAMP_TO_EDGE.as_raw(), 4);
    assert_eq!(ImageViewType::CUBE_ARRAY.as_raw(), 6);
    assert_eq!(IndexType::UINT8.as_raw(), 1_000_265_000);
    assert_eq!(ShaderStageFlags::COMPUTE.as_raw(), 0x20);
    assert_eq!(ShaderStageFlags::ALL_GRAPHICS.as_raw(), 0x1f);
    assert_eq!(PipelineStageFlags2::COMPUTE_SHADER.as_raw(), 0x800);
    assert_eq!(AccessFlags2::SHADER_STORAGE_WRITE.as_raw(), 1 << 34);
    assert_eq!(FormatFeatureFlags::STORAGE_IMAGE.as_raw(), 0x2);
    assert_eq!(
        FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT.as_raw(),
        1 << 31
    );
    assert_eq!(SubgroupFeatureFlags::QUAD.as_raw(), 0x80);
    assert_eq!(SampleCountFlags::COUNT_4.as_raw(), 0x4);
    assert_eq!(ImageCreateFlags::MUTABLE_FORMAT.as_raw(), 0x8);
}

#[test]
fn aliases_and_unknown_values() {
    assert_eq!(
        Format::from_name("VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT"),
        Some(Format::A4R4G4B4_UNORM_PACK16)
    );
    assert_eq!(Format::from_name("VK_FORMAT_NOT_A_FORMAT"), None);
    let unknown = Format::from_raw(123_456);
    assert!(!unknown.is_known());
    assert_eq!(unknown.name(), None);
    assert_eq!(format!("{unknown:?}"), "VkFormat(123456)");
    assert!(!Format::KNOWN.contains(&unknown));
}

#[test]
fn flags_operations_and_debug() {
    let vf = ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT;
    assert!(vf.contains(ShaderStageFlags::VERTEX));
    assert!(!vf.contains(ShaderStageFlags::COMPUTE));
    assert!(vf.intersects(ShaderStageFlags::ALL_GRAPHICS));
    assert_eq!(
        vf.difference(ShaderStageFlags::VERTEX),
        ShaderStageFlags::FRAGMENT
    );
    assert_eq!(
        format!("{vf:?}"),
        "VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT"
    );
    assert_eq!(
        format!("{:?}", ShaderStageFlags::ALL_GRAPHICS),
        "VK_SHADER_STAGE_ALL_GRAPHICS"
    );
    assert_eq!(format!("{:?}", ShaderStageFlags::empty()), "0x0");
    let odd = ShaderStageFlags::COMPUTE | ShaderStageFlags::from_raw(1 << 30);
    assert_eq!(
        format!("{odd:?}"),
        "VK_SHADER_STAGE_COMPUTE_BIT | 0x40000000"
    );
    // Composite values are not single bits, so they do not widen the known set.
    assert!(!ShaderStageFlags::known_bits().contains(ShaderStageFlags::ALL));
    let mut acc = AccessFlags2::empty();
    acc |= AccessFlags2::SHADER_STORAGE_READ;
    acc &= AccessFlags2::SHADER_STORAGE_READ | AccessFlags2::SHADER_STORAGE_WRITE;
    assert_eq!(acc, AccessFlags2::SHADER_STORAGE_READ);
    assert!(AccessFlags2::default().is_empty());
}

#[test]
fn api_version_packing() {
    // VK_API_VERSION_1_2 and VK_API_VERSION_1_3.
    assert_eq!(
        ApiVersion::from_packed((1 << 22) | (2 << 12)),
        ApiVersion::V1_2
    );
    assert_eq!(ApiVersion::V1_3.packed(), (1 << 22) | (3 << 12));
    // Patch and variant fields are dropped.
    assert_eq!(
        ApiVersion::from_packed((1 << 22) | (3 << 12) | 250),
        ApiVersion::V1_3
    );
}

#[test]
fn sampler_desc_equality_is_total() {
    let desc = SamplerDesc {
        mag_filter: Filter::LINEAR,
        min_filter: Filter::NEAREST,
        mipmap_mode: SamplerMipmapMode::LINEAR,
        address_mode_u: SamplerAddressMode::REPEAT,
        address_mode_v: SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_w: SamplerAddressMode::MIRRORED_REPEAT,
        mip_lod_bias: f32::NAN,
        max_anisotropy: Some(16.0),
        compare: None,
        min_lod: 0.0,
        max_lod: 1000.0,
        border_color: BorderColor::FLOAT_TRANSPARENT_BLACK,
        reduction: SamplerReductionMode::WEIGHTED_AVERAGE,
        border_components: None,
        ycbcr: None,
        unnormalized: false,
    };
    assert_eq!(desc, desc);
    let set: HashSet<SamplerDesc> = [desc, desc].into_iter().collect();
    assert_eq!(set.len(), 1);
    let other = SamplerDesc {
        max_anisotropy: None,
        ..desc
    };
    assert_ne!(desc, other);
    // A sampler made for one view mapping is another sampler.
    let mapped = SamplerDesc {
        border_components: Some(ComponentMapping::IDENTITY),
        ycbcr: None,
        ..desc
    };
    assert_ne!(desc, mapped);
    // So is one that reads texel coordinates.
    let pixel = SamplerDesc {
        unnormalized: true,
        ..desc
    };
    assert_ne!(desc, pixel);
    assert_eq!(ComponentMapping::default(), ComponentMapping::IDENTITY);
}

#[test]
fn values_are_send_sync() {
    fn send_sync<T: Send + Sync + Copy>() {}
    send_sync::<Format>();
    send_sync::<ShaderStageFlags>();
    send_sync::<SamplerDesc>();
    send_sync::<SpecializationMapEntry>();
}

#[test]
fn shader_stage_is_exactly_one_named_bit() {
    let mut seen = ShaderStageFlags::empty();
    for &stage in ShaderStage::KNOWN {
        let bits = stage.as_raw();
        assert_eq!(bits.count_ones(), 1, "{stage:?}");
        assert!(!seen.intersects(stage.flags()), "{stage:?} repeats a bit");
        seen |= stage.flags();
        assert_eq!(ShaderStage::from_raw(bits), Some(stage));
        assert_eq!(ShaderStage::try_from(bits), Ok(stage));
        assert_eq!(ShaderStage::from_flags(stage.into()), Some(stage));
        assert_eq!(ShaderStage::from_name(stage.name()), Some(stage));
        assert_eq!(
            ShaderStageFlags::from_name(stage.name()),
            Some(stage.flags())
        );
        assert_eq!(format!("{stage:?}"), stage.name());
    }
    // Every single bit of the flag type is a stage, and nothing else is.
    assert_eq!(seen, ShaderStageFlags::known_bits());
    assert_eq!(ShaderStage::COMPUTE.as_raw(), 0x20);
    assert_eq!(ShaderStage::FRAGMENT.flags(), ShaderStageFlags::FRAGMENT);
    for rejected in [
        0,
        ShaderStageFlags::ALL_GRAPHICS.as_raw(),
        ShaderStageFlags::ALL.as_raw(),
        (ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT).as_raw(),
        1 << 30,
    ] {
        assert_eq!(ShaderStage::from_raw(rejected), None, "{rejected:#x}");
        assert_eq!(ShaderStage::try_from(rejected), Err(rejected));
    }
    assert_eq!(ShaderStage::from_name("VK_SHADER_STAGE_ALL_GRAPHICS"), None);
    assert_eq!(
        ShaderStage::from_name("VK_SHADER_STAGE_MESH_BIT_NV"),
        Some(ShaderStage::MESH_EXT)
    );
}

#[cfg(feature = "serde")]
#[test]
fn shader_stage_serde_rejects_non_single_bits() {
    let json = serde_json::to_string(&ShaderStage::VERTEX).expect("serialize");
    assert_eq!(json, "1");
    let back: ShaderStage = serde_json::from_str("32").expect("deserialize");
    assert_eq!(back, ShaderStage::COMPUTE);
    assert!(serde_json::from_str::<ShaderStage>("31").is_err());
    assert!(serde_json::from_str::<ShaderStage>("0").is_err());
}

#[cfg(feature = "serde")]
#[test]
fn serde_uses_raw_values() {
    let json = serde_json::to_string(&Format::B8G8R8A8_UNORM).expect("serialize");
    assert_eq!(json, "44");
    let back: Format = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, Format::B8G8R8A8_UNORM);
    let flags = ShaderStageFlags::VERTEX | ShaderStageFlags::COMPUTE;
    let json = serde_json::to_string(&flags).expect("serialize");
    assert_eq!(json, "33");
    let back: ShaderStageFlags = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, flags);
}

/// `Format::block` is the registry's `<formats>` (generated): block bytes and texels, the packed
/// width, and the shared component width, which give the vertex-input alignment.
#[test]
fn format_blocks_come_from_the_registry() {
    let block = |f: Format| f.block().unwrap_or_else(|| panic!("{f:?} has no block"));
    let r8 = block(Format::R8_UNORM);
    assert_eq!(
        (r8.size, r8.texels, r8.packed_bits, r8.component_bits),
        (1, 1, None, Some(8))
    );
    assert_eq!(block(Format::R32G32B32A32_SFLOAT).size, 16);
    assert_eq!(block(Format::R16G16B16_SFLOAT).size, 6);
    assert_eq!(block(Format::R16G16B16_SFLOAT).vertex_alignment(), Some(2));
    assert_eq!(block(Format::R32G32B32_SFLOAT).vertex_alignment(), Some(4));
    assert_eq!(block(Format::R64G64_SFLOAT).vertex_alignment(), Some(8));
    let packed = block(Format::A2B10G10R10_UNORM_PACK32);
    assert_eq!((packed.size, packed.packed_bits), (4, Some(32)));
    assert_eq!(packed.vertex_alignment(), Some(4), "the whole packed word");
    let bc1 = block(Format::BC1_RGB_UNORM_BLOCK);
    assert_eq!((bc1.size, bc1.texels, bc1.component_bits), (8, 16, None));
    assert_eq!(bc1.vertex_alignment(), None);
    let ds = block(Format::D24_UNORM_S8_UINT);
    assert_eq!(
        (ds.size, ds.component_bits),
        (4, None),
        "mixed component widths"
    );
    assert_eq!(Format::UNDEFINED.block(), None);
    assert_eq!(Format::from_raw(-7).block(), None);
    // Every named format but `UNDEFINED` has a block of at least one byte and one texel.
    for &f in Format::KNOWN {
        match f.block() {
            Some(b) => assert!(b.size > 0 && b.texels > 0, "{f:?}"),
            None => assert_eq!(f, Format::UNDEFINED),
        }
    }
}

#[test]
fn numeric_types_come_from_the_registry() {
    let block = |f: Format| f.block().unwrap_or_else(|| panic!("{f:?} has no block"));
    let srgb = block(Format::R8G8B8A8_SRGB);
    assert_eq!(
        (srgb.color, srgb.depth, srgb.stencil),
        (Some(NumericFormat::Srgb), None, None),
        "an sRGB format's linear alpha is part of the SRGB numeric format"
    );
    let ds = block(Format::D24_UNORM_S8_UINT);
    assert_eq!(
        (ds.color, ds.depth, ds.stencil),
        (None, Some(NumericFormat::Unorm), Some(NumericFormat::Uint))
    );
    for (format, ty) in [
        (Format::R8G8B8A8_UNORM, Some(NumericType::Float)),
        (Format::A2B10G10R10_SNORM_PACK32, Some(NumericType::Float)),
        (Format::R16G16B16A16_USCALED, Some(NumericType::Float)),
        (Format::E5B9G9R9_UFLOAT_PACK32, Some(NumericType::Float)),
        (Format::BC6H_SFLOAT_BLOCK, Some(NumericType::Float)),
        (Format::R32G32_UINT, Some(NumericType::Uint)),
        (Format::R8_SINT, Some(NumericType::Sint)),
        (Format::D16_UNORM, Some(NumericType::Float)),
        (Format::S8_UINT, Some(NumericType::Uint)),
        (Format::R16G16_SFIXED5_NV, Some(NumericType::ScaledSint)),
        (Format::R8_BOOL_ARM, Some(NumericType::Bool)),
        (Format::D32_SFLOAT_S8_UINT, None),
        (Format::UNDEFINED, None),
        (Format::from_raw(-7), None),
    ] {
        assert_eq!(format.numeric_type(), ty, "{format:?}");
    }
    // Every named format has exactly the aspects its components have: color, or depth and/or
    // stencil, never both.
    for &f in Format::KNOWN {
        let Some(b) = f.block() else { continue };
        assert!(
            b.color.is_some() != (b.depth.is_some() || b.stencil.is_some()),
            "{f:?}: {b:?}"
        );
    }
}

#[test]
fn format_exposure_comes_from_the_registry() {
    let v = |major, minor| ApiVersion { major, minor };
    let exposure = |f: Format| f.exposure().unwrap_or_else(|| panic!("{f:?} is not named"));
    assert_eq!(exposure(Format::R8G8B8A8_UNORM).core, Some(v(1, 0)));
    assert!(exposure(Format::R8G8B8A8_UNORM).extensions.is_empty());
    let a8 = exposure(Format::A8_UNORM);
    assert_eq!(
        (a8.core, a8.extensions),
        (Some(v(1, 4)), &["VK_KHR_maintenance5"][..])
    );
    let hdr = exposure(Format::ASTC_4X4_SFLOAT_BLOCK);
    assert_eq!(
        (hdr.core, hdr.extensions),
        (Some(v(1, 3)), &["VK_EXT_texture_compression_astc_hdr"][..])
    );
    let pvrtc = exposure(Format::PVRTC1_4BPP_SRGB_BLOCK_IMG);
    assert_eq!(
        (pvrtc.core, pvrtc.extensions),
        (None, &["VK_IMG_format_pvrtc"][..])
    );
    let ycbcr = exposure(Format::G8B8G8R8_422_UNORM);
    assert_eq!(ycbcr.core, Some(v(1, 1)));
    assert!(!a8.exposed(v(1, 3), |_| false));
    assert!(a8.exposed(v(1, 4), |_| false));
    assert!(a8.exposed(v(1, 2), |e| e == "VK_KHR_maintenance5"));
    assert!(!pvrtc.exposed(v(1, 4), |_| false));
    assert_eq!(Format::UNDEFINED.exposure(), None);
    assert_eq!(Format::from_raw(-7).exposure(), None);
    // Every value below the extension range is core 1.0; every named format has an origin.
    for &f in Format::KNOWN {
        let Some(x) = f.exposure() else {
            assert_eq!(f, Format::UNDEFINED);
            continue;
        };
        assert!(x.core.is_some() || !x.extensions.is_empty(), "{f:?}");
        if (1..1_000_000_000).contains(&f.as_raw()) {
            assert_eq!(x.core, Some(v(1, 0)), "{f:?}");
        }
    }
}
