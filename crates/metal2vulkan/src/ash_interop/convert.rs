//! `From` conversions between the plain-data [`crate::vk`] values and `ash::vk`.
//!
//! Both sides are transparent newtypes over the same registry numbers, so every conversion is a
//! raw-value copy in either direction. A value the pinned `vk.xml` does not name converts
//! unchanged (newer drivers report values older registries lack).

use crate::vk;

/// `From` both ways for registry enums and bitmasks of the same raw type.
macro_rules! raw_both_ways {
    ($($ty:ident),* $(,)?) => {
        $(
            impl From<vk::$ty> for ash::vk::$ty {
                fn from(v: vk::$ty) -> ash::vk::$ty {
                    ash::vk::$ty::from_raw(v.as_raw())
                }
            }

            impl From<ash::vk::$ty> for vk::$ty {
                fn from(v: ash::vk::$ty) -> vk::$ty {
                    vk::$ty::from_raw(v.as_raw())
                }
            }
        )*
    };
}

raw_both_ways!(
    // Enums.
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
    // Bitmasks.
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

impl From<vk::ShaderStage> for ash::vk::ShaderStageFlags {
    fn from(stage: vk::ShaderStage) -> ash::vk::ShaderStageFlags {
        ash::vk::ShaderStageFlags::from_raw(stage.as_raw())
    }
}

impl TryFrom<ash::vk::ShaderStageFlags> for vk::ShaderStage {
    /// The flags, when they are not exactly one named stage bit.
    type Error = ash::vk::ShaderStageFlags;

    fn try_from(flags: ash::vk::ShaderStageFlags) -> Result<vk::ShaderStage, Self::Error> {
        vk::ShaderStage::from_raw(flags.as_raw()).ok_or(flags)
    }
}

impl From<vk::ComponentMapping> for ash::vk::ComponentMapping {
    fn from(m: vk::ComponentMapping) -> ash::vk::ComponentMapping {
        ash::vk::ComponentMapping {
            r: m.r.into(),
            g: m.g.into(),
            b: m.b.into(),
            a: m.a.into(),
        }
    }
}

impl From<ash::vk::ComponentMapping> for vk::ComponentMapping {
    fn from(m: ash::vk::ComponentMapping) -> vk::ComponentMapping {
        vk::ComponentMapping {
            r: m.r.into(),
            g: m.g.into(),
            b: m.b.into(),
            a: m.a.into(),
        }
    }
}

impl From<vk::SpecializationMapEntry> for ash::vk::SpecializationMapEntry {
    fn from(e: vk::SpecializationMapEntry) -> ash::vk::SpecializationMapEntry {
        ash::vk::SpecializationMapEntry {
            constant_id: e.constant_id,
            offset: e.offset,
            // Entries are scalars of at most 8 bytes; `u32` always fits `usize`.
            size: e.size as usize,
        }
    }
}

/// The `VkSamplerCreateInfo` of a sampler description, with the reduction mode chained through
/// `reduction` when it is not the default weighted average, and the border's component mapping
/// through `border` when the description has one
/// (`SamplerDesc::border_components`). A description of a Y′CbCr conversion
/// (`SamplerDesc::ycbcr`) also needs the caller's `VkSamplerYcbcrConversionInfo` of its conversion
/// chained, which this does not make.
///
/// `reduction` and `border` are caller storage: they must outlive the returned structure.
pub fn sampler_create_info<'a>(
    desc: &vk::SamplerDesc,
    reduction: &'a mut ash::vk::SamplerReductionModeCreateInfo<'static>,
    border: &'a mut ash::vk::SamplerBorderColorComponentMappingCreateInfoEXT<'static>,
) -> ash::vk::SamplerCreateInfo<'a> {
    let mut info = ash::vk::SamplerCreateInfo::default()
        .mag_filter(desc.mag_filter.into())
        .min_filter(desc.min_filter.into())
        .mipmap_mode(desc.mipmap_mode.into())
        .address_mode_u(desc.address_mode_u.into())
        .address_mode_v(desc.address_mode_v.into())
        .address_mode_w(desc.address_mode_w.into())
        .mip_lod_bias(desc.mip_lod_bias)
        .anisotropy_enable(desc.max_anisotropy.is_some())
        .max_anisotropy(desc.max_anisotropy.unwrap_or(1.0))
        .compare_enable(desc.compare.is_some())
        .compare_op(desc.compare.unwrap_or(vk::CompareOp::NEVER).into())
        .min_lod(desc.min_lod)
        .max_lod(desc.max_lod)
        .border_color(desc.border_color.into())
        .unnormalized_coordinates(desc.unnormalized);
    if desc.reduction != vk::SamplerReductionMode::WEIGHTED_AVERAGE {
        *reduction = ash::vk::SamplerReductionModeCreateInfo::default()
            .reduction_mode(desc.reduction.into());
        info = info.push_next(reduction);
    }
    if let Some(components) = desc.border_components {
        *border = ash::vk::SamplerBorderColorComponentMappingCreateInfoEXT::default()
            .components(components.into())
            .srgb(false);
        info = info.push_next(border);
    }
    info
}
