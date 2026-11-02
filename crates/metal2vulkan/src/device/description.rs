//! `DeviceDescription`: plain data about one Vulkan device as the consumer creates it.
//!
//! A description records what the consumer **enables**, not what the hardware has: output may
//! only declare what was enabled at `vkCreateDevice`. It is the only input of
//! `Device::new`; Khronos profile JSON is parsed into a description
//! ([`DeviceDescription::from_khronos_json`]), never directly into a `DeviceProfile`.
//!
//! Absence means "not supported": a feature, extension, or format usage the description does not
//! list is unavailable, a missing `max*` limit is 0, and a missing alignment is the worst case.

use std::collections::{BTreeMap, BTreeSet};

use crate::device::profile::{FloatControls, Limits};
use crate::device::requirements::{ApiVersion, FormatUsage, VkExtension, VkFeature};
use crate::vk;

/// One Vulkan physical device as the consumer will create it.
///
/// With feature `serde` it (and [`DevicePolicy`](crate::device::DevicePolicy)) serializes, so a
/// consumer that compiles in a child process can hand the child the pair from which it rebuilds
/// the same `Device` (`docs/HOWTO.md` section 5).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceDescription {
    /// The device's Vulkan version (`VkPhysicalDeviceProperties::apiVersion`, capped by the
    /// instance version the consumer uses). Below 1.2 is refused by `Device::new`.
    pub api_version: ApiVersion,
    /// `VkPhysicalDeviceDriverProperties::driverID`, when known.
    pub driver_id: Option<vk::DriverId>,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    /// `VkPhysicalDeviceProperties::deviceName`. No translation reads it but through the tables
    /// keyed on a measured GPU, where a `deviceID` names more than one GPU: MoltenVK's on Apple's
    /// GPUs is the macOS release and the GPU family, so a release measured on one GPU of such an
    /// identity names that GPU by its name (`driver_facts`). Empty where unknown.
    #[cfg_attr(feature = "serde", serde(default))]
    pub device_name: String,
    pub device_type: Option<vk::PhysicalDeviceType>,
    /// Enabled device extensions the core consumes.
    pub extensions: BTreeSet<VkExtension>,
    /// Enabled device features the core consumes. A feature exposed through an extension is only
    /// listed when that extension (or the core version that promoted it) is available.
    pub features: BTreeSet<VkFeature>,
    pub limits: Limits,
    pub subgroup: SubgroupProperties,
    pub float_controls: FloatControls,
    /// `filterMinmaxSingleComponentFormats`.
    pub filter_minmax_single_component_formats: bool,
    pub formats: FormatSupport,
    /// `rgba4OpaqueBlackSwizzled` (`VkPhysicalDeviceMaintenance10PropertiesKHR`): false where the
    /// device does not report it.
    pub rgba4_opaque_black_swizzled: bool,
    /// `robustBufferAccessUpdateAfterBind` (`VkPhysicalDeviceVulkan12Properties`): whether
    /// `robustBufferAccess` may be enabled together with the update-after-bind features of the
    /// buffer descriptor types (VUID-VkDeviceCreateInfo-robustBufferAccess-10247; the enable-set
    /// rules of `device::enables` read it). True only where the device or document reports it
    /// true: a document that does not state it describes devices that may report false, so it is
    /// held to the rule. No translation reads it.
    pub robust_buffer_access_update_after_bind: bool,
    /// The device lists `VK_KHR_format_feature_flags2`, which, like Vulkan 1.3, reports
    /// `VkFormatProperties3` ([`DeviceDescription::has_format_feature_flags2`]): a fact of the
    /// query, which an enable set that leaves the extension out keeps. Enabled, the extension is
    /// [`VkExtension::KhrFormatFeatureFlags2`] among the description's extensions.
    pub khr_format_feature_flags2: bool,
    /// `VK_EXT_device_generated_commands`: `Some` with the device's
    /// `VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT` where the device supports the
    /// extension and its `deviceGeneratedCommands` feature, `None` otherwise.
    /// [`DeviceDescription::query`] and `from_khronos_json` fill it whenever the device (or the
    /// profile document) supports it; clear it where you do not enable the extension, since the
    /// device replay's generated-commands path (`bind::replay`), the only reader, then records
    /// commands that need it. No translation reads it, so, like
    /// `robust_buffer_access_update_after_bind`, it is not a profile field: two devices that
    /// differ only in it share every artifact.
    pub generated_commands: Option<GeneratedCommandsProperties>,
    /// The device's allocation sizes ([`AllocationLimits`]): what one allocation and one buffer may
    /// hold. They follow the machine's installed memory on integrated GPUs (ANV, MoltenVK), so two
    /// machines with one GPU and one driver differ in them. No translation reads them, so, like
    /// `robust_buffer_access_update_after_bind`, they are not profile fields: two devices that
    /// differ only in them share every artifact and every profile fingerprint. The Metal-facing
    /// `maxBufferLength` and the runtime's heap sizes read them from the `Device`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub allocation: AllocationLimits,
}

/// A device's allocation sizes, [`DeviceDescription::allocation`]: device facts that size
/// allocations and buffers, never translation inputs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AllocationLimits {
    /// `maxMemoryAllocationSize` (Vulkan 1.1): the largest single memory allocation.
    pub max_memory_allocation_size: u64,
    /// `maxBufferSize` (Vulkan 1.3 / `VK_KHR_maintenance4`); 0 when the device does not report it.
    pub max_buffer_size: u64,
}

impl AllocationLimits {
    /// Limits of `max_memory_allocation_size` per allocation and `max_buffer_size` per buffer (0:
    /// not reported).
    pub fn new(max_memory_allocation_size: u64, max_buffer_size: u64) -> AllocationLimits {
        AllocationLimits {
            max_memory_allocation_size,
            max_buffer_size,
        }
    }
}

/// What a device states of `VK_EXT_device_generated_commands`
/// (`VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT`): the limits and stages the device
/// replay's generated-commands path checks before it sets anything up.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GeneratedCommandsProperties {
    /// `maxIndirectPipelineCount`: the pipelines an indirect execution set may hold.
    pub max_indirect_pipeline_count: u32,
    /// `maxIndirectSequenceCount`: the sequences one execution may run.
    pub max_indirect_sequence_count: u32,
    /// `maxIndirectCommandsTokenCount`.
    pub max_indirect_commands_token_count: u32,
    /// `maxIndirectCommandsTokenOffset`: the largest offset of a token in a sequence.
    pub max_indirect_commands_token_offset: u32,
    /// `maxIndirectCommandsIndirectStride`: the largest sequence stride.
    pub max_indirect_commands_indirect_stride: u32,
    /// `supportedIndirectCommandsShaderStages`.
    pub supported_stages: vk::ShaderStageFlags,
    /// `supportedIndirectCommandsShaderStagesPipelineBinding`: the stages whose pipelines an
    /// execution set token may bind.
    pub pipeline_binding_stages: vk::ShaderStageFlags,
}

impl GeneratedCommandsProperties {
    /// Properties of `max_indirect_pipeline_count` pipelines, `max_indirect_sequence_count`
    /// sequences, `max_indirect_commands_token_count` tokens at offsets to
    /// `max_indirect_commands_token_offset`, strides to `max_indirect_commands_indirect_stride`,
    /// and the stages `supported_stages` and `pipeline_binding_stages`.
    pub fn new(
        [
            max_indirect_pipeline_count,
            max_indirect_sequence_count,
            max_indirect_commands_token_count,
            max_indirect_commands_token_offset,
            max_indirect_commands_indirect_stride,
        ]: [u32; 5],
        supported_stages: vk::ShaderStageFlags,
        pipeline_binding_stages: vk::ShaderStageFlags,
    ) -> GeneratedCommandsProperties {
        GeneratedCommandsProperties {
            max_indirect_pipeline_count,
            max_indirect_sequence_count,
            max_indirect_commands_token_count,
            max_indirect_commands_token_offset,
            max_indirect_commands_indirect_stride,
            supported_stages,
            pipeline_binding_stages,
        }
    }
}

/// `VkPhysicalDeviceProperties::vendorID` of Apple's GPUs.
pub(crate) const APPLE_VENDOR_ID: u32 = 0x106B;

impl DeviceDescription {
    /// A description of a device of `api_version` that enables nothing and reports zero limits.
    /// Callers fill the public fields; `Device::new` validates the result.
    pub fn new(api_version: ApiVersion) -> DeviceDescription {
        DeviceDescription {
            api_version,
            driver_id: None,
            driver_version: 0,
            vendor_id: 0,
            device_id: 0,
            device_name: String::new(),
            device_type: None,
            extensions: BTreeSet::new(),
            features: BTreeSet::new(),
            limits: Limits::default(),
            subgroup: SubgroupProperties::default(),
            float_controls: FloatControls::default(),
            filter_minmax_single_component_formats: false,
            formats: FormatSupport::default(),
            rgba4_opaque_black_swizzled: false,
            robust_buffer_access_update_after_bind: false,
            khr_format_feature_flags2: false,
            generated_commands: None,
            allocation: AllocationLimits::default(),
        }
    }

    /// Whether the device reports `VkFormatProperties3` (Vulkan 1.3, or
    /// [`DeviceDescription::khr_format_feature_flags2`]): then [`DeviceDescription::formats`]
    /// holds its `VkFormatFeatureFlags2`, and a storage image or storage texel buffer is accessed
    /// through an image declared without a format exactly where its format's
    /// `STORAGE_READ_WITHOUT_FORMAT` / `STORAGE_WRITE_WITHOUT_FORMAT` say, in the optimal-tiling
    /// features for an image and the buffer features for a texel buffer
    /// (`VUID-vkCmdDispatch-OpTypeImage-07027` to `-07030`), together with the image bits the
    /// specification implies for the formats of its "Formats Without Shader Storage Format" list.
    /// Otherwise the device reports `VkFormatProperties` only, and
    /// `shaderStorageImage{Read,Write}WithoutFormat` decide for every storage format.
    pub fn has_format_feature_flags2(&self) -> bool {
        self.api_version >= ApiVersion::V1_3 || self.khr_format_feature_flags2
    }

    /// Whether the feature is enabled.
    pub fn has(&self, feature: VkFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Whether `GLSL.std.450 InterpolateAtOffset` reads at the pixel center plus its offset, as
    /// GLSL.std.450 states ("An offset of (0, 0) identifies the center of the pixel"): every
    /// driver but MoltenVK, whose SPIRV-Cross converts the instruction to Metal's
    /// `interpolate_at_offset(offset + 0.4375)`, which Metal measures from the pixel's upper-left
    /// corner, so offset 0 reads a sixteenth of a pixel from the center. That conversion is read in
    /// SPIRV-Cross's source and Metal's reading measured on the Apple M3 Max; no release of
    /// MoltenVK is measured to read at the center, so none takes it: a device without it has
    /// `VkProperty::InterpolateAtOffsetFromSevenSixteenths`, where translation moves a Metal
    /// offset by 7/16 and reads nothing else through the instruction.
    pub(crate) fn interpolates_at_offset_centered(&self) -> bool {
        self.driver_id != Some(vk::DriverId::MOLTENVK)
    }
}

/// Subgroup properties (`VkPhysicalDeviceVulkan11Properties` subgroup members and
/// `VkPhysicalDeviceSubgroupSizeControlProperties`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubgroupProperties {
    /// `subgroupSize`: the value of `SubgroupSize` in SPIR-V 1.5 modules without a pinned size.
    pub size: u32,
    pub supported_stages: vk::ShaderStageFlags,
    pub supported_operations: vk::SubgroupFeatureFlags,
    pub quad_operations_in_all_stages: bool,
    /// `minSubgroupSize` (0 when the device reports no subgroup size control properties).
    pub min_size: u32,
    /// `maxSubgroupSize` (0 when unreported).
    pub max_size: u32,
    pub max_compute_workgroup_subgroups: u32,
    /// `requiredSubgroupSizeStages`.
    pub required_size_stages: vk::ShaderStageFlags,
}

/// Format feature support, keyed by format (`vkGetPhysicalDeviceFormatProperties` answers).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FormatSupport {
    pub formats: BTreeMap<vk::Format, FormatProperties>,
}

impl FormatSupport {
    /// Every format the translator can need answers for: every Vulkan format a Metal pixel or
    /// vertex format can map to (`metal::format`, `metal::vertex`). Sorted, without duplicates.
    /// Some are defined by an extension or a core version above 1.2 (`A8_UNORM`, the ASTC HDR and
    /// PVRTC blocks; `vk::Format::exposure`), and a physical device may be asked about those only
    /// where it supports them: query [`queries_for`](FormatSupport::queries_for).
    pub fn queries() -> &'static [vk::Format] {
        static QUERIES: std::sync::OnceLock<Vec<vk::Format>> = std::sync::OnceLock::new();
        QUERIES.get_or_init(crate::metal::format::queried_formats)
    }

    /// The formats of [`queries`](FormatSupport::queries) a device of effective version `api`
    /// with `extensions` available exposes: the ones the consumer runs
    /// `vkGetPhysicalDeviceFormatProperties` on (with feature `ash`, `DeviceDescription::query`
    /// does it). A format the device does not expose is never usable on it, so its answer is not
    /// needed.
    pub fn queries_for(api: ApiVersion, extensions: &BTreeSet<VkExtension>) -> Vec<vk::Format> {
        FormatSupport::queries()
            .iter()
            .copied()
            .filter(|&f| crate::device::enables::format_exposed(f, api, extensions))
            .collect()
    }

    /// Whether `format` supports `usage` (optimal tiling for images, buffer features for vertex
    /// and texel buffers). A format the description does not list supports nothing.
    pub fn supports(&self, format: vk::Format, usage: FormatUsage) -> bool {
        self.formats.get(&format).is_some_and(|p| p.supports(usage))
    }

    /// Every usage `format` supports.
    pub fn usages(&self, format: vk::Format) -> BTreeSet<FormatUsage> {
        ALL_FORMAT_USAGES
            .iter()
            .copied()
            .filter(|&u| self.supports(format, u))
            .collect()
    }
}

/// `VkFormatProperties3` (or `VkFormatProperties`, whose bits are its low 32 bits, on a device
/// without it: [`DeviceDescription::has_format_feature_flags2`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FormatProperties {
    pub linear_tiling: vk::FormatFeatureFlags2,
    pub optimal_tiling: vk::FormatFeatureFlags2,
    pub buffer: vk::FormatFeatureFlags2,
    /// The sample counts of a color attachment of the format: `VkImageFormatProperties::
    /// sampleCounts` of a 2D, optimal-tiling image with `COLOR_ATTACHMENT` usage and no flags,
    /// for a format whose optimal tiling has `COLOR_ATTACHMENT`. Vulkan requires them to include
    /// the device's `framebufferColorSampleCounts` (or `framebufferIntegerColorSampleCounts`
    /// for an integer format), and a device may support more for one format. Empty where the
    /// description does not state them: a profile document without its venue capture's
    /// counts, or a format that is no color attachment.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_attachment_sample_counts: vk::SampleCountFlags,
}

/// `VkFormatFeatureFlags` as the low 32 bits of `VkFormatFeatureFlags2`, which they are by
/// definition (a `VkFormatProperties` answer).
pub(crate) fn widen(flags: vk::FormatFeatureFlags) -> vk::FormatFeatureFlags2 {
    vk::FormatFeatureFlags2::from_raw(u64::from(flags.as_raw()))
}

/// Every [`FormatUsage`], in declaration order.
pub const ALL_FORMAT_USAGES: &[FormatUsage] = &[
    FormatUsage::VertexBuffer,
    FormatUsage::UniformTexelBuffer,
    FormatUsage::StorageTexelBuffer,
    FormatUsage::StorageTexelBufferAtomic,
    FormatUsage::SampledImage,
    FormatUsage::SampledImageFilterLinear,
    FormatUsage::SampledImageFilterMinmax,
    FormatUsage::StorageImage,
    FormatUsage::StorageImageAtomic,
    FormatUsage::ColorAttachment,
    FormatUsage::ColorAttachmentBlend,
    FormatUsage::DepthStencilAttachment,
    FormatUsage::StorageImageReadWithoutFormat,
    FormatUsage::StorageImageWriteWithoutFormat,
    FormatUsage::StorageTexelBufferReadWithoutFormat,
    FormatUsage::StorageTexelBufferWriteWithoutFormat,
];

impl FormatProperties {
    /// The properties a `VkFormatProperties` answer states: its bits are the low 32 bits of
    /// `VkFormatFeatureFlags2`.
    pub fn from_legacy(
        linear_tiling: vk::FormatFeatureFlags,
        optimal_tiling: vk::FormatFeatureFlags,
        buffer: vk::FormatFeatureFlags,
    ) -> FormatProperties {
        FormatProperties {
            linear_tiling: widen(linear_tiling),
            optimal_tiling: widen(optimal_tiling),
            buffer: widen(buffer),
            color_attachment_sample_counts: vk::SampleCountFlags::empty(),
        }
    }

    /// Whether these properties support `usage`. Image usages read optimal tiling (the tiling the
    /// binder creates images with); vertex and texel-buffer usages read buffer features. The
    /// without-format usages read the bits as reported; the profile adds what the specification
    /// implies for them.
    pub fn supports(&self, usage: FormatUsage) -> bool {
        use vk::FormatFeatureFlags2 as F;
        let (flags, bit) = match usage {
            FormatUsage::VertexBuffer => (self.buffer, F::VERTEX_BUFFER),
            FormatUsage::UniformTexelBuffer => (self.buffer, F::UNIFORM_TEXEL_BUFFER),
            FormatUsage::StorageTexelBuffer => (self.buffer, F::STORAGE_TEXEL_BUFFER),
            FormatUsage::StorageTexelBufferAtomic => (self.buffer, F::STORAGE_TEXEL_BUFFER_ATOMIC),
            FormatUsage::SampledImage => (self.optimal_tiling, F::SAMPLED_IMAGE),
            FormatUsage::SampledImageFilterLinear => {
                (self.optimal_tiling, F::SAMPLED_IMAGE_FILTER_LINEAR)
            }
            FormatUsage::SampledImageFilterMinmax => {
                (self.optimal_tiling, F::SAMPLED_IMAGE_FILTER_MINMAX)
            }
            FormatUsage::StorageImage => (self.optimal_tiling, F::STORAGE_IMAGE),
            FormatUsage::StorageImageAtomic => (self.optimal_tiling, F::STORAGE_IMAGE_ATOMIC),
            FormatUsage::ColorAttachment => (self.optimal_tiling, F::COLOR_ATTACHMENT),
            FormatUsage::ColorAttachmentBlend => (self.optimal_tiling, F::COLOR_ATTACHMENT_BLEND),
            FormatUsage::DepthStencilAttachment => {
                (self.optimal_tiling, F::DEPTH_STENCIL_ATTACHMENT)
            }
            FormatUsage::StorageImageReadWithoutFormat => {
                (self.optimal_tiling, F::STORAGE_READ_WITHOUT_FORMAT)
            }
            FormatUsage::StorageImageWriteWithoutFormat => {
                (self.optimal_tiling, F::STORAGE_WRITE_WITHOUT_FORMAT)
            }
            FormatUsage::StorageTexelBufferReadWithoutFormat => {
                (self.buffer, F::STORAGE_READ_WITHOUT_FORMAT)
            }
            FormatUsage::StorageTexelBufferWriteWithoutFormat => {
                (self.buffer, F::STORAGE_WRITE_WITHOUT_FORMAT)
            }
        };
        flags.contains(bit)
    }
}

/// The features the core consumes that a Vulkan version makes **unconditionally** required: a
/// conformant device of that version always supports them (vk.xml `<feature>` blocks,
/// `<require>` without `depends`, Vulkan-Headers `vulkan-sdk-1.4.341.0`).
///
/// Required-to-support is not enabled: the consumer must still enable them at `vkCreateDevice`.
/// Khronos profile documents omit members their base version already requires, so the JSON
/// reader adds these for members the document does not mention (never over an explicit `false`).
///
/// Notably `uniformBufferStandardLayout` is required by Vulkan 1.2 (the `u32[]` UBO view therefore
/// needs no optional feature on any 1.2 device that enables it).
pub fn required_features(version: ApiVersion) -> impl Iterator<Item = VkFeature> {
    const V1_0: &[VkFeature] = &[VkFeature::RobustBufferAccess];
    const V1_1: &[VkFeature] = &[VkFeature::Multiview];
    const V1_2: &[VkFeature] = &[
        VkFeature::UniformBufferStandardLayout,
        VkFeature::ShaderSubgroupExtendedTypes,
    ];
    const V1_3: &[VkFeature] = &[
        VkFeature::ShaderTerminateInvocation,
        VkFeature::ShaderDemoteToHelperInvocation,
        VkFeature::Synchronization2,
        VkFeature::ShaderZeroInitializeWorkgroupMemory,
        VkFeature::RobustImageAccess,
        VkFeature::SubgroupSizeControl,
        VkFeature::ComputeFullSubgroups,
        VkFeature::Maintenance4,
        VkFeature::VulkanMemoryModel,
        VkFeature::VulkanMemoryModelDeviceScope,
        VkFeature::BufferDeviceAddress,
    ];
    const V1_4: &[VkFeature] = &[
        VkFeature::ImageCubeArray,
        VkFeature::IndependentBlend,
        VkFeature::SampleRateShading,
        VkFeature::DrawIndirectFirstInstance,
        VkFeature::DepthClamp,
        VkFeature::DepthBiasClamp,
        VkFeature::SamplerAnisotropy,
        VkFeature::FragmentStoresAndAtomics,
        VkFeature::ShaderStorageImageExtendedFormats,
        VkFeature::ShaderImageGatherExtended,
        VkFeature::ShaderInt16,
        VkFeature::StorageBuffer16BitAccess,
        VkFeature::SamplerMirrorClampToEdge,
        VkFeature::ScalarBlockLayout,
        VkFeature::ShaderInt8,
        VkFeature::StorageBuffer8BitAccess,
        VkFeature::VertexAttributeInstanceRateDivisor,
        VkFeature::DynamicRenderingLocalRead,
        VkFeature::Maintenance5,
        VkFeature::FullDrawIndexUint32,
    ];
    let tiers: [(u32, &'static [VkFeature]); 5] =
        [(0, V1_0), (1, V1_1), (2, V1_2), (3, V1_3), (4, V1_4)];
    tiers
        .into_iter()
        .filter(move |&(minor, _)| version >= ApiVersion { major: 1, minor })
        .flat_map(|(_, features)| features.iter().copied())
}
