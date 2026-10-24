//! What a device the runtime creates enables at `vkCreateDevice` (validation's enable-chain
//! machinery, its engine half, moved here).
//!
//! The runtime enables exactly the [`EnableSet`] it computed (the description's
//! `recommended_enables` for the policy, closed by `with_enabled`), plus the extensions the
//! caller's options name, `timelineSemaphore` and `hostQueryReset`, which the runtime needs itself
//! (each required in Vulkan 1.2). Each feature is one `VkBool32` member of one feature structure
//! ([`MEMBERS`]); a structure is chained when one of its members is set, and an extension structure
//! only through its enabled extension, or from the core version that promoted it
//! (`Structure::promotion`, the registry's promotions, which the tests hold to the description's),
//! at the device's version (its own, capped by the instance's). From Vulkan 1.3 a member 1.3
//! promoted is set in `VkPhysicalDeviceVulkan13Features` ([`CORE_1_3`]), never in its extension
//! structure; from 1.4 a member 1.4 promoted is set in its own structure, which 1.4 makes core
//! (`ash 0.38` has no `VkPhysicalDeviceVulkan14Features`), as the description reads it
//! (`ash_interop::query`); a feature core without a member from its version (extended dynamic state
//! from 1.3) sets nothing. Each extension of the vocabulary is enabled under the first of its names
//! the device lists ([`EXTENSIONS`]).

use super::error::{Error, ExtensionFeature, RuntimeGap};
use ash::vk;
use metal2vulkan::device::{ApiVersion, EnableSet, VkExtension, VkFeature};
use std::collections::BTreeSet;
use std::ffi::{c_char, CStr, CString};

/// `VkPhysicalDeviceMaintenance8FeaturesKHR`, which `ash 0.38` predates, laid out as the registry
/// defines it (extension number 575).
#[repr(C)]
#[derive(Clone, Copy)]
struct Maintenance8Features {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    maintenance8: vk::Bool32,
}

impl Default for Maintenance8Features {
    fn default() -> Self {
        Maintenance8Features {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            maintenance8: vk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceMaintenance8FeaturesKHR`, which begins
// with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for Maintenance8Features {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_574_000);
}
// SAFETY: the registry lists `VkDeviceCreateInfo` in the structure's `structextends`.
unsafe impl vk::ExtendsDeviceCreateInfo for Maintenance8Features {}

/// `VkPhysicalDeviceShaderFmaFeaturesKHR`, which `ash 0.38` predates, laid out as the registry
/// defines it (extension number 580).
#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderFmaFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    shader_fma_float16: vk::Bool32,
    shader_fma_float32: vk::Bool32,
    shader_fma_float64: vk::Bool32,
}

impl Default for ShaderFmaFeatures {
    fn default() -> Self {
        ShaderFmaFeatures {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            shader_fma_float16: vk::FALSE,
            shader_fma_float32: vk::FALSE,
            shader_fma_float64: vk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceShaderFmaFeaturesKHR`, which begins with
// `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for ShaderFmaFeatures {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_579_000);
}
// SAFETY: the registry lists `VkDeviceCreateInfo` in the structure's `structextends`.
unsafe impl vk::ExtendsDeviceCreateInfo for ShaderFmaFeatures {}

/// `VkPhysicalDeviceDepthClampControlFeaturesEXT`, which `ash 0.38` predates, laid out as the
/// registry defines it (extension number 583).
#[repr(C)]
#[derive(Clone, Copy)]
struct DepthClampControlFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    depth_clamp_control: vk::Bool32,
}

impl Default for DepthClampControlFeatures {
    fn default() -> Self {
        DepthClampControlFeatures {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            depth_clamp_control: vk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceDepthClampControlFeaturesEXT`, which
// begins with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for DepthClampControlFeatures {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_582_000);
}
// SAFETY: the registry lists `VkDeviceCreateInfo` in the structure's `structextends`.
unsafe impl vk::ExtendsDeviceCreateInfo for DepthClampControlFeatures {}

/// `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`, which `ash 0.38` predates, laid out as
/// the registry defines it (extension number 573).
#[repr(C)]
#[derive(Clone, Copy)]
struct GeneratedCommandsFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    device_generated_commands: vk::Bool32,
    dynamic_generated_pipeline_layout: vk::Bool32,
}

impl Default for GeneratedCommandsFeatures {
    fn default() -> Self {
        GeneratedCommandsFeatures {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            device_generated_commands: vk::FALSE,
            dynamic_generated_pipeline_layout: vk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`,
// which begins with `sType` and `pNext` as every chained structure must, and this is its
// structure type.
unsafe impl vk::TaggedStructure for GeneratedCommandsFeatures {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_572_000);
}
// SAFETY: the registry lists `VkDeviceCreateInfo` in the structure's `structextends`.
unsafe impl vk::ExtendsDeviceCreateInfo for GeneratedCommandsFeatures {}

/// `VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR` (and its `_EXT` alias), which `ash
/// 0.38` predates, laid out as the registry defines it (extension number 362).
#[repr(C)]
#[derive(Clone, Copy)]
struct FifoLatestReadyFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    present_mode_fifo_latest_ready: vk::Bool32,
}

impl Default for FifoLatestReadyFeatures {
    fn default() -> Self {
        FifoLatestReadyFeatures {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            present_mode_fifo_latest_ready: vk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR`,
// which begins with `sType` and `pNext` as every chained structure must, and this is its
// structure type (the `_EXT` one is an alias).
unsafe impl vk::TaggedStructure for FifoLatestReadyFeatures {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_361_000);
}
// SAFETY: the registry lists `VkDeviceCreateInfo` in the structure's `structextends`.
unsafe impl vk::ExtendsDeviceCreateInfo for FifoLatestReadyFeatures {}
// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` in the structure's `structextends`.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for FifoLatestReadyFeatures {}

/// The name of `VK_KHR_pipeline_binary` (extension number 484), which `ash 0.38` predates and the
/// vocabulary does not hold.
pub(super) const PIPELINE_BINARY: &CStr = c"VK_KHR_pipeline_binary";

/// `VK_KHR_pipeline_binary`'s structure type `n`: the registry numbers extension 484's from
/// `1000483000`.
const fn pipeline_binary_structure(n: i32) -> vk::StructureType {
    vk::StructureType::from_raw(1_000_483_000 + n)
}

/// `VkPhysicalDevicePipelineBinaryFeaturesKHR`, laid out as the registry defines it.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct PipelineBinaryFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    pub(super) pipeline_binaries: vk::Bool32,
}

impl Default for PipelineBinaryFeatures {
    fn default() -> Self {
        PipelineBinaryFeatures {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            pipeline_binaries: vk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDevicePipelineBinaryFeaturesKHR`, which begins
// with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for PipelineBinaryFeatures {
    const STRUCTURE_TYPE: vk::StructureType = pipeline_binary_structure(0);
}
// SAFETY: the registry lists `VkDeviceCreateInfo` in the structure's `structextends`.
unsafe impl vk::ExtendsDeviceCreateInfo for PipelineBinaryFeatures {}
// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` in the structure's `structextends`.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for PipelineBinaryFeatures {}

/// `VkPhysicalDevicePipelineBinaryPropertiesKHR`, laid out as the registry defines it: whether
/// the driver keeps an internal cache of pipelines, and whether a device may be made with it off.
#[repr(C)]
pub(super) struct PipelineBinaryProperties {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    internal_cache: vk::Bool32,
    pub(super) internal_cache_control: vk::Bool32,
    prefers_internal_cache: vk::Bool32,
    precompiled_internal_cache: vk::Bool32,
    compressed_data: vk::Bool32,
}

impl Default for PipelineBinaryProperties {
    fn default() -> Self {
        PipelineBinaryProperties {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            internal_cache: vk::FALSE,
            internal_cache_control: vk::FALSE,
            prefers_internal_cache: vk::FALSE,
            precompiled_internal_cache: vk::FALSE,
            compressed_data: vk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDevicePipelineBinaryPropertiesKHR`, which begins
// with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for PipelineBinaryProperties {
    const STRUCTURE_TYPE: vk::StructureType = pipeline_binary_structure(4);
}
// SAFETY: the registry lists `VkPhysicalDeviceProperties2` in the structure's `structextends`.
unsafe impl vk::ExtendsPhysicalDeviceProperties2 for PipelineBinaryProperties {}

/// `VkDevicePipelineBinaryInternalCacheControlKHR`, laid out as the registry defines it.
#[repr(C)]
#[derive(Clone, Copy)]
struct InternalCacheControl {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    disable_internal_cache: vk::Bool32,
}

impl Default for InternalCacheControl {
    fn default() -> Self {
        InternalCacheControl {
            s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null(),
            disable_internal_cache: vk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkDevicePipelineBinaryInternalCacheControlKHR`, which
// begins with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for InternalCacheControl {
    const STRUCTURE_TYPE: vk::StructureType = pipeline_binary_structure(8);
}
// SAFETY: the registry lists `VkDeviceCreateInfo` in the structure's `structextends`.
unsafe impl vk::ExtendsDeviceCreateInfo for InternalCacheControl {}

/// The name of `VK_EXT_device_generated_commands`, which the vocabulary does not hold.
const GENERATED_COMMANDS: &CStr = c"VK_EXT_device_generated_commands";

/// The names of `VK_EXT_host_image_copy` and of the extensions it requires below Vulkan 1.3,
/// where they are core (the registry's `depends`), which the vocabulary does not hold.
const HOST_IMAGE_COPY: [&CStr; 3] = [
    c"VK_EXT_host_image_copy",
    c"VK_KHR_copy_commands2",
    c"VK_KHR_format_feature_flags2",
];

/// The extensions a device of version `api` enables with `hostImageCopy`: none from Vulkan 1.4,
/// which made it core; `VK_EXT_host_image_copy` at 1.3; and below, the two extensions it
/// requires too.
pub(super) fn host_image_copy_names(api: ApiVersion) -> &'static [&'static CStr] {
    if api >= V1_4 {
        &[]
    } else if api >= ApiVersion::V1_3 {
        &HOST_IMAGE_COPY[..1]
    } else {
        &HOST_IMAGE_COPY
    }
}

/// The names of `VK_NV_linear_color_attachment` and of the extension it requires below Vulkan
/// 1.3, where it is core (the registry's `depends`: `VK_KHR_format_feature_flags2` or Vulkan
/// 1.3).
const LINEAR_COLOR_ATTACHMENT: [&CStr; 2] = [
    ash::nv::linear_color_attachment::NAME,
    c"VK_KHR_format_feature_flags2",
];

/// The names of `VK_EXT_graphics_pipeline_library` and of `VK_KHR_pipeline_library`, which it
/// requires (the registry's `depends`).
pub(super) const GRAPHICS_PIPELINE_LIBRARY: [&CStr; 2] = [
    ash::ext::graphics_pipeline_library::NAME,
    ash::khr::pipeline_library::NAME,
];

/// The extensions a device of version `api` enables with `linearColorAttachment`: the extension,
/// and below Vulkan 1.3 the one it requires. The runtime's instance is at Vulkan 1.2 unless the
/// consumer asks another (`DeviceOptions::instance_api_version`), which caps every device's.
pub(super) fn linear_color_attachment_names(api: ApiVersion) -> &'static [&'static CStr] {
    match api >= ApiVersion::V1_3 {
        true => &LINEAR_COLOR_ATTACHMENT[..1],
        false => &LINEAR_COLOR_ATTACHMENT,
    }
}

/// One feature structure the runtime can chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Structure {
    /// `VkPhysicalDeviceFeatures` (`pEnabledFeatures`).
    Core,
    V11,
    V12,
    /// `VkPhysicalDeviceVulkan13Features`, chained from Vulkan 1.3 when a member 1.3 promoted is
    /// set ([`CORE_1_3`]).
    V13,
    SubgroupSizeControl,
    Demote,
    Terminate,
    ZeroInitialize,
    Synchronization2,
    Maintenance4,
    SubgroupRotate,
    LocalRead,
    Divisor,
    AtomicFloat,
    AtomicFloat2,
    ImageAtomicInt64,
    ExplicitLayout,
    MaximalReconvergence,
    /// `VkPhysicalDeviceShaderQuadControlFeaturesKHR`.
    QuadControl,
    UniformControlFlow,
    Derivatives,
    Barycentric,
    Interlock,
    RasterOrder,
    Maintenance8,
    ShaderFma,
    /// `VkPhysicalDeviceMaintenance5FeaturesKHR`.
    Maintenance5,
    /// `VkPhysicalDeviceHostImageCopyFeaturesEXT`, outside the vocabulary: linked only by
    /// [`Enables::enable_host_image_copy`].
    HostImageCopy,
    BorderColorSwizzle,
    /// `VkPhysicalDeviceDepthBiasControlFeaturesEXT`.
    DepthBiasControl,
    /// `VkPhysicalDeviceDepthClipEnableFeaturesEXT`.
    DepthClipEnable,
    /// `VkPhysicalDeviceDepthClampControlFeaturesEXT`.
    DepthClampControl,
    /// `VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT`.
    ListRestart,
    /// `VkPhysicalDeviceProvokingVertexFeaturesEXT`.
    ProvokingVertex,
    RayQuery,
    Acceleration,
    /// `VkPhysicalDeviceMeshShaderFeaturesEXT`.
    MeshShader,
    ExtendedDynamicState,
    /// `VkPhysicalDeviceExtendedDynamicState2FeaturesEXT`.
    ExtendedDynamicState2,
    /// `VkPhysicalDeviceExtendedDynamicState3FeaturesEXT`.
    ExtendedDynamicState3,
    /// `VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT`.
    FeedbackLoop,
    Robustness2,
    ImageRobustness,
    PortabilitySubset,
    /// `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`, outside the vocabulary: linked only by
    /// [`Enables::enable_generated_commands`].
    GeneratedCommands,
    /// `VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR`, outside the vocabulary: linked only by
    /// [`Enables::enable_listed`], where the caller lists its extension.
    SwapchainMaintenance1,
    /// `VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR`, outside the vocabulary: linked
    /// only by [`Enables::enable_listed`], where the caller lists its extension.
    FifoLatestReady,
    /// `VkPhysicalDeviceLinearColorAttachmentFeaturesNV`, outside the vocabulary: linked only by
    /// [`Enables::enable_linear_color_attachment`].
    LinearColorAttachment,
    /// `VkPhysicalDevicePipelineBinaryFeaturesKHR`, outside the vocabulary: linked only by
    /// [`Enables::turn_driver_pipeline_cache_off`].
    PipelineBinary,
    /// `VkDevicePipelineBinaryInternalCacheControlKHR`, outside the vocabulary and no feature
    /// structure: linked only by [`Enables::turn_driver_pipeline_cache_off`].
    InternalCacheControl,
    /// `VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT`, outside the vocabulary: linked only
    /// by [`Enables::enable_graphics_pipeline_library`].
    GraphicsPipelineLibrary,
}

/// Vulkan 1.4.
const V1_4: ApiVersion = ApiVersion { major: 1, minor: 4 };

/// How a core version took over an extension structure (the registry's promotions, pinned
/// `vk.xml`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Promotion {
    /// From this version the structure's members are core whether or not the device lists its
    /// extension: a member 1.3 promoted is set in `VkPhysicalDeviceVulkan13Features`
    /// (`CORE_1_3`), a member 1.4 promoted in this structure, which 1.4 makes core.
    Members(ApiVersion),
    /// From this version the structure's functionality is core without a member: there is
    /// nothing to set.
    Functionality(ApiVersion),
}

impl Structure {
    /// The core version that took the structure over, where one did.
    pub(super) fn promotion(self) -> Option<Promotion> {
        use Promotion::{Functionality, Members};
        match self {
            Structure::SubgroupSizeControl
            | Structure::Demote
            | Structure::Terminate
            | Structure::ZeroInitialize
            | Structure::Synchronization2
            | Structure::Maintenance4
            | Structure::ImageRobustness => Some(Members(ApiVersion::V1_3)),
            Structure::SubgroupRotate
            | Structure::LocalRead
            | Structure::Divisor
            | Structure::Maintenance5
            | Structure::HostImageCopy => Some(Members(V1_4)),
            Structure::ExtendedDynamicState | Structure::ExtendedDynamicState2 => {
                Some(Functionality(ApiVersion::V1_3))
            }
            Structure::Core
            | Structure::V11
            | Structure::V12
            | Structure::V13
            | Structure::AtomicFloat
            | Structure::AtomicFloat2
            | Structure::ImageAtomicInt64
            | Structure::ExplicitLayout
            | Structure::MaximalReconvergence
            | Structure::QuadControl
            | Structure::UniformControlFlow
            | Structure::Derivatives
            | Structure::Barycentric
            | Structure::Interlock
            | Structure::RasterOrder
            | Structure::Maintenance8
            | Structure::ShaderFma
            | Structure::BorderColorSwizzle
            | Structure::DepthBiasControl
            | Structure::DepthClipEnable
            | Structure::DepthClampControl
            | Structure::ListRestart
            | Structure::ProvokingVertex
            | Structure::RayQuery
            | Structure::Acceleration
            | Structure::MeshShader
            | Structure::ExtendedDynamicState3
            | Structure::FeedbackLoop
            | Structure::Robustness2
            | Structure::PortabilitySubset
            | Structure::GeneratedCommands
            | Structure::SwapchainMaintenance1
            | Structure::FifoLatestReady
            | Structure::LinearColorAttachment
            | Structure::PipelineBinary
            | Structure::InternalCacheControl
            | Structure::GraphicsPipelineLibrary => None,
        }
    }

    /// The vocabulary extensions any one of which exposes the structure at Vulkan 1.2 (empty for
    /// the core structures).
    pub(super) fn extensions(self) -> &'static [VkExtension] {
        use VkExtension as X;
        match self {
            Structure::Core | Structure::V11 | Structure::V12 | Structure::V13 => &[],
            Structure::SubgroupSizeControl => &[X::ExtSubgroupSizeControl],
            Structure::Demote => &[X::ExtShaderDemoteToHelperInvocation],
            Structure::Terminate => &[X::KhrShaderTerminateInvocation],
            Structure::ZeroInitialize => &[X::KhrZeroInitializeWorkgroupMemory],
            Structure::Synchronization2 => &[X::KhrSynchronization2],
            Structure::Maintenance4 => &[X::KhrMaintenance4],
            Structure::SubgroupRotate => &[X::KhrShaderSubgroupRotate],
            Structure::LocalRead => &[X::KhrDynamicRenderingLocalRead],
            Structure::Divisor => &[X::KhrVertexAttributeDivisor, X::ExtVertexAttributeDivisor],
            Structure::AtomicFloat => &[X::ExtShaderAtomicFloat],
            Structure::AtomicFloat2 => &[X::ExtShaderAtomicFloat2],
            Structure::ImageAtomicInt64 => &[X::ExtShaderImageAtomicInt64],
            Structure::ExplicitLayout => &[X::KhrWorkgroupMemoryExplicitLayout],
            Structure::MaximalReconvergence => &[X::KhrShaderMaximalReconvergence],
            Structure::QuadControl => &[X::KhrShaderQuadControl],
            Structure::UniformControlFlow => &[X::KhrShaderSubgroupUniformControlFlow],
            Structure::Derivatives => &[X::KhrComputeShaderDerivatives],
            Structure::Barycentric => &[X::KhrFragmentShaderBarycentric],
            Structure::Interlock => &[X::ExtFragmentShaderInterlock],
            Structure::RasterOrder => &[X::ExtRasterizationOrderAttachmentAccess],
            Structure::Maintenance8 => &[X::KhrMaintenance8],
            Structure::ShaderFma => &[X::KhrShaderFma],
            Structure::Maintenance5 => &[X::KhrMaintenance5],
            Structure::HostImageCopy => &[],
            Structure::BorderColorSwizzle => &[X::ExtBorderColorSwizzle],
            Structure::DepthBiasControl => &[X::ExtDepthBiasControl],
            Structure::DepthClipEnable => &[X::ExtDepthClipEnable],
            Structure::DepthClampControl => &[X::ExtDepthClampControl],
            Structure::ListRestart => &[X::ExtPrimitiveTopologyListRestart],
            Structure::ProvokingVertex => &[X::ExtProvokingVertex],
            Structure::RayQuery => &[X::KhrRayQuery],
            Structure::Acceleration => &[X::KhrAccelerationStructure],
            Structure::MeshShader => &[X::ExtMeshShader],
            Structure::ExtendedDynamicState => &[X::ExtExtendedDynamicState],
            Structure::ExtendedDynamicState2 => &[X::ExtExtendedDynamicState2],
            Structure::ExtendedDynamicState3 => &[X::ExtExtendedDynamicState3],
            Structure::FeedbackLoop => &[X::ExtAttachmentFeedbackLoopLayout],
            Structure::Robustness2 => &[X::ExtRobustness2],
            Structure::ImageRobustness => &[X::ExtImageRobustness],
            Structure::PortabilitySubset => &[X::KhrPortabilitySubset],
            Structure::GeneratedCommands
            | Structure::SwapchainMaintenance1
            | Structure::FifoLatestReady
            | Structure::LinearColorAttachment
            | Structure::PipelineBinary
            | Structure::InternalCacheControl
            | Structure::GraphicsPipelineLibrary => &[],
        }
    }
}

/// Every feature structure the runtime can chain. A structure that is not chained stays zeroed;
/// its `pNext` link is null except during the one call that chains it.
#[derive(Clone, Copy, Default)]
struct FeatureChain {
    core: vk::PhysicalDeviceFeatures,
    v11: vk::PhysicalDeviceVulkan11Features<'static>,
    v12: vk::PhysicalDeviceVulkan12Features<'static>,
    v13: vk::PhysicalDeviceVulkan13Features<'static>,
    subgroup_size_control: vk::PhysicalDeviceSubgroupSizeControlFeatures<'static>,
    demote: vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures<'static>,
    terminate: vk::PhysicalDeviceShaderTerminateInvocationFeatures<'static>,
    zero_initialize: vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures<'static>,
    synchronization2: vk::PhysicalDeviceSynchronization2Features<'static>,
    maintenance4: vk::PhysicalDeviceMaintenance4Features<'static>,
    subgroup_rotate: vk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR<'static>,
    local_read: vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR<'static>,
    // One structure type serves `VK_KHR_` and `VK_EXT_vertex_attribute_divisor`.
    divisor: vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR<'static>,
    atomic_float: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT<'static>,
    atomic_float2: vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT<'static>,
    image_atomic_int64: vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT<'static>,
    explicit_layout: vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR<'static>,
    maximal_reconvergence: vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR<'static>,
    quad_control: vk::PhysicalDeviceShaderQuadControlFeaturesKHR<'static>,
    uniform_control_flow: vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR<'static>,
    // The `KHR` structure type is the `NV` one (a registry alias).
    derivatives: vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV<'static>,
    barycentric: vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR<'static>,
    interlock: vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT<'static>,
    raster_order: vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT<'static>,
    maintenance8: Maintenance8Features,
    shader_fma: ShaderFmaFeatures,
    maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR<'static>,
    host_image_copy: vk::PhysicalDeviceHostImageCopyFeaturesEXT<'static>,
    border_color_swizzle: vk::PhysicalDeviceBorderColorSwizzleFeaturesEXT<'static>,
    depth_bias_control: vk::PhysicalDeviceDepthBiasControlFeaturesEXT<'static>,
    depth_clip_enable: vk::PhysicalDeviceDepthClipEnableFeaturesEXT<'static>,
    depth_clamp_control: DepthClampControlFeatures,
    list_restart: vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT<'static>,
    provoking_vertex: vk::PhysicalDeviceProvokingVertexFeaturesEXT<'static>,
    ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
    acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    extended_dynamic_state: vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT<'static>,
    extended_dynamic_state2: vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT<'static>,
    extended_dynamic_state3: vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static>,
    feedback_loop: vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT<'static>,
    robustness2: vk::PhysicalDeviceRobustness2FeaturesEXT<'static>,
    image_robustness: vk::PhysicalDeviceImageRobustnessFeatures<'static>,
    portability_subset: vk::PhysicalDevicePortabilitySubsetFeaturesKHR<'static>,
    generated_commands: GeneratedCommandsFeatures,
    swapchain_maintenance1: vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'static>,
    fifo_latest_ready: FifoLatestReadyFeatures,
    linear_color_attachment: vk::PhysicalDeviceLinearColorAttachmentFeaturesNV<'static>,
    pipeline_binary: PipelineBinaryFeatures,
    internal_cache_control: InternalCacheControl,
    graphics_pipeline_library: vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT<'static>,
}

/// One `VkBool32` member of a [`FeatureChain`] structure, named by the vocabulary.
pub(super) struct Member {
    pub(super) structure: Structure,
    pub(super) feature: VkFeature,
    write: fn(&mut FeatureChain) -> &mut vk::Bool32,
}

/// Defines a table of [`Member`]s, [`MEMBERS`] or [`CORE_1_3`], from
/// `Structure.field.member => feature` rows.
macro_rules! members {
    ($(#[$doc:meta])* $table:ident: $($structure:ident . $field:ident . $member:ident => $feature:ident;)*) => {
        $(#[$doc])*
        pub(super) const $table: &[Member] = &[$(Member {
            structure: Structure::$structure,
            feature: VkFeature::$feature,
            write: {
                fn write(chain: &mut FeatureChain) -> &mut vk::Bool32 {
                    &mut chain.$field.$member
                }
                write
            },
        },)*];
    };
}

members! {
    /// Every vocabulary feature's member, each once: a member 1.3 promoted in its extension
    /// structure, which the device exposes below 1.3.
    MEMBERS:
    Core.core.robust_buffer_access => RobustBufferAccess;
    Core.core.shader_int16 => ShaderInt16;
    Core.core.shader_int64 => ShaderInt64;
    Core.core.shader_float64 => ShaderFloat64;
    Core.core.vertex_pipeline_stores_and_atomics => VertexPipelineStoresAndAtomics;
    Core.core.fragment_stores_and_atomics => FragmentStoresAndAtomics;
    Core.core.geometry_shader => GeometryShader;
    Core.core.tessellation_shader => TessellationShader;
    Core.core.sample_rate_shading => SampleRateShading;
    Core.core.dual_src_blend => DualSrcBlend;
    Core.core.logic_op => LogicOp;
    Core.core.alpha_to_one => AlphaToOne;
    Core.core.independent_blend => IndependentBlend;
    Core.core.shader_clip_distance => ShaderClipDistance;
    Core.core.shader_cull_distance => ShaderCullDistance;
    Core.core.draw_indirect_first_instance => DrawIndirectFirstInstance;
    Core.core.multi_draw_indirect => MultiDrawIndirect;
    Core.core.depth_clamp => DepthClamp;
    Core.core.depth_bias_clamp => DepthBiasClamp;
    Core.core.fill_mode_non_solid => FillModeNonSolid;
    Core.core.image_cube_array => ImageCubeArray;
    Core.core.shader_image_gather_extended => ShaderImageGatherExtended;
    Core.core.shader_resource_min_lod => ShaderResourceMinLod;
    Core.core.shader_resource_residency => ShaderResourceResidency;
    Core.core.sampler_anisotropy => SamplerAnisotropy;
    Core.core.shader_storage_image_multisample => ShaderStorageImageMultisample;
    Core.core.shader_storage_image_extended_formats => ShaderStorageImageExtendedFormats;
    Core.core.shader_storage_image_read_without_format => ShaderStorageImageReadWithoutFormat;
    Core.core.shader_storage_image_write_without_format => ShaderStorageImageWriteWithoutFormat;
    Core.core.shader_uniform_buffer_array_dynamic_indexing => ShaderUniformBufferArrayDynamicIndexing;
    Core.core.shader_sampled_image_array_dynamic_indexing => ShaderSampledImageArrayDynamicIndexing;
    Core.core.shader_storage_buffer_array_dynamic_indexing => ShaderStorageBufferArrayDynamicIndexing;
    Core.core.shader_storage_image_array_dynamic_indexing => ShaderStorageImageArrayDynamicIndexing;
    Core.core.multi_viewport => MultiViewport;
    Core.core.occlusion_query_precise => OcclusionQueryPrecise;
    Core.core.full_draw_index_uint32 => FullDrawIndexUint32;
    Core.core.shader_tessellation_and_geometry_point_size => ShaderTessellationAndGeometryPointSize;
    V11.v11.storage_buffer16_bit_access => StorageBuffer16BitAccess;
    V11.v11.uniform_and_storage_buffer16_bit_access => UniformAndStorageBuffer16BitAccess;
    V11.v11.storage_push_constant16 => StoragePushConstant16;
    V11.v11.storage_input_output16 => StorageInputOutput16;
    V11.v11.multiview => Multiview;
    V11.v11.multiview_tessellation_shader => MultiviewTessellationShader;
    V11.v11.multiview_geometry_shader => MultiviewGeometryShader;
    V11.v11.shader_draw_parameters => ShaderDrawParameters;
    V12.v12.sampler_mirror_clamp_to_edge => SamplerMirrorClampToEdge;
    V12.v12.storage_buffer8_bit_access => StorageBuffer8BitAccess;
    V12.v12.uniform_and_storage_buffer8_bit_access => UniformAndStorageBuffer8BitAccess;
    V12.v12.storage_push_constant8 => StoragePushConstant8;
    V12.v12.shader_buffer_int64_atomics => ShaderBufferInt64Atomics;
    V12.v12.shader_shared_int64_atomics => ShaderSharedInt64Atomics;
    V12.v12.shader_float16 => ShaderFloat16;
    V12.v12.shader_int8 => ShaderInt8;
    V12.v12.shader_input_attachment_array_dynamic_indexing => ShaderInputAttachmentArrayDynamicIndexing;
    V12.v12.shader_uniform_texel_buffer_array_dynamic_indexing => ShaderUniformTexelBufferArrayDynamicIndexing;
    V12.v12.shader_storage_texel_buffer_array_dynamic_indexing => ShaderStorageTexelBufferArrayDynamicIndexing;
    V12.v12.shader_sampled_image_array_non_uniform_indexing => ShaderSampledImageArrayNonUniformIndexing;
    V12.v12.shader_storage_buffer_array_non_uniform_indexing => ShaderStorageBufferArrayNonUniformIndexing;
    V12.v12.shader_storage_image_array_non_uniform_indexing => ShaderStorageImageArrayNonUniformIndexing;
    V12.v12.shader_uniform_texel_buffer_array_non_uniform_indexing => ShaderUniformTexelBufferArrayNonUniformIndexing;
    V12.v12.shader_storage_texel_buffer_array_non_uniform_indexing => ShaderStorageTexelBufferArrayNonUniformIndexing;
    V12.v12.descriptor_binding_sampled_image_update_after_bind => DescriptorBindingSampledImageUpdateAfterBind;
    V12.v12.descriptor_binding_storage_image_update_after_bind => DescriptorBindingStorageImageUpdateAfterBind;
    V12.v12.descriptor_binding_storage_buffer_update_after_bind => DescriptorBindingStorageBufferUpdateAfterBind;
    V12.v12.descriptor_binding_uniform_texel_buffer_update_after_bind => DescriptorBindingUniformTexelBufferUpdateAfterBind;
    V12.v12.descriptor_binding_storage_texel_buffer_update_after_bind => DescriptorBindingStorageTexelBufferUpdateAfterBind;
    V12.v12.descriptor_binding_update_unused_while_pending => DescriptorBindingUpdateUnusedWhilePending;
    V12.v12.descriptor_binding_partially_bound => DescriptorBindingPartiallyBound;
    V12.v12.descriptor_binding_variable_descriptor_count => DescriptorBindingVariableDescriptorCount;
    V12.v12.runtime_descriptor_array => RuntimeDescriptorArray;
    V12.v12.sampler_filter_minmax => SamplerFilterMinmax;
    V12.v12.draw_indirect_count => DrawIndirectCount;
    V12.v12.scalar_block_layout => ScalarBlockLayout;
    V12.v12.uniform_buffer_standard_layout => UniformBufferStandardLayout;
    V12.v12.shader_subgroup_extended_types => ShaderSubgroupExtendedTypes;
    V12.v12.buffer_device_address => BufferDeviceAddress;
    V12.v12.vulkan_memory_model => VulkanMemoryModel;
    V12.v12.vulkan_memory_model_device_scope => VulkanMemoryModelDeviceScope;
    V12.v12.shader_output_viewport_index => ShaderOutputViewportIndex;
    V12.v12.shader_output_layer => ShaderOutputLayer;
    SubgroupSizeControl.subgroup_size_control.subgroup_size_control => SubgroupSizeControl;
    SubgroupSizeControl.subgroup_size_control.compute_full_subgroups => ComputeFullSubgroups;
    Demote.demote.shader_demote_to_helper_invocation => ShaderDemoteToHelperInvocation;
    Terminate.terminate.shader_terminate_invocation => ShaderTerminateInvocation;
    ZeroInitialize.zero_initialize.shader_zero_initialize_workgroup_memory => ShaderZeroInitializeWorkgroupMemory;
    Synchronization2.synchronization2.synchronization2 => Synchronization2;
    Maintenance4.maintenance4.maintenance4 => Maintenance4;
    SubgroupRotate.subgroup_rotate.shader_subgroup_rotate => ShaderSubgroupRotate;
    SubgroupRotate.subgroup_rotate.shader_subgroup_rotate_clustered => ShaderSubgroupRotateClustered;
    LocalRead.local_read.dynamic_rendering_local_read => DynamicRenderingLocalRead;
    Divisor.divisor.vertex_attribute_instance_rate_divisor => VertexAttributeInstanceRateDivisor;
    Divisor.divisor.vertex_attribute_instance_rate_zero_divisor => VertexAttributeInstanceRateZeroDivisor;
    AtomicFloat.atomic_float.shader_buffer_float32_atomics => ShaderBufferFloat32Atomics;
    AtomicFloat.atomic_float.shader_buffer_float32_atomic_add => ShaderBufferFloat32AtomicAdd;
    AtomicFloat.atomic_float.shader_buffer_float64_atomics => ShaderBufferFloat64Atomics;
    AtomicFloat.atomic_float.shader_buffer_float64_atomic_add => ShaderBufferFloat64AtomicAdd;
    AtomicFloat.atomic_float.shader_shared_float32_atomics => ShaderSharedFloat32Atomics;
    AtomicFloat.atomic_float.shader_shared_float32_atomic_add => ShaderSharedFloat32AtomicAdd;
    AtomicFloat.atomic_float.shader_shared_float64_atomics => ShaderSharedFloat64Atomics;
    AtomicFloat.atomic_float.shader_shared_float64_atomic_add => ShaderSharedFloat64AtomicAdd;
    AtomicFloat.atomic_float.shader_image_float32_atomics => ShaderImageFloat32Atomics;
    AtomicFloat.atomic_float.shader_image_float32_atomic_add => ShaderImageFloat32AtomicAdd;
    AtomicFloat2.atomic_float2.shader_buffer_float16_atomics => ShaderBufferFloat16Atomics;
    AtomicFloat2.atomic_float2.shader_buffer_float16_atomic_add => ShaderBufferFloat16AtomicAdd;
    AtomicFloat2.atomic_float2.shader_buffer_float16_atomic_min_max => ShaderBufferFloat16AtomicMinMax;
    AtomicFloat2.atomic_float2.shader_buffer_float32_atomic_min_max => ShaderBufferFloat32AtomicMinMax;
    AtomicFloat2.atomic_float2.shader_buffer_float64_atomic_min_max => ShaderBufferFloat64AtomicMinMax;
    AtomicFloat2.atomic_float2.shader_shared_float16_atomics => ShaderSharedFloat16Atomics;
    AtomicFloat2.atomic_float2.shader_shared_float16_atomic_add => ShaderSharedFloat16AtomicAdd;
    AtomicFloat2.atomic_float2.shader_shared_float16_atomic_min_max => ShaderSharedFloat16AtomicMinMax;
    AtomicFloat2.atomic_float2.shader_shared_float32_atomic_min_max => ShaderSharedFloat32AtomicMinMax;
    AtomicFloat2.atomic_float2.shader_shared_float64_atomic_min_max => ShaderSharedFloat64AtomicMinMax;
    AtomicFloat2.atomic_float2.shader_image_float32_atomic_min_max => ShaderImageFloat32AtomicMinMax;
    ImageAtomicInt64.image_atomic_int64.shader_image_int64_atomics => ShaderImageInt64Atomics;
    ExplicitLayout.explicit_layout.workgroup_memory_explicit_layout => WorkgroupMemoryExplicitLayout;
    ExplicitLayout.explicit_layout.workgroup_memory_explicit_layout_scalar_block_layout => WorkgroupMemoryExplicitLayoutScalarBlockLayout;
    ExplicitLayout.explicit_layout.workgroup_memory_explicit_layout8_bit_access => WorkgroupMemoryExplicitLayout8BitAccess;
    ExplicitLayout.explicit_layout.workgroup_memory_explicit_layout16_bit_access => WorkgroupMemoryExplicitLayout16BitAccess;
    MaximalReconvergence.maximal_reconvergence.shader_maximal_reconvergence => ShaderMaximalReconvergence;
    QuadControl.quad_control.shader_quad_control => ShaderQuadControl;
    UniformControlFlow.uniform_control_flow.shader_subgroup_uniform_control_flow => ShaderSubgroupUniformControlFlow;
    Derivatives.derivatives.compute_derivative_group_quads => ComputeDerivativeGroupQuads;
    Derivatives.derivatives.compute_derivative_group_linear => ComputeDerivativeGroupLinear;
    Barycentric.barycentric.fragment_shader_barycentric => FragmentShaderBarycentric;
    Interlock.interlock.fragment_shader_pixel_interlock => FragmentShaderPixelInterlock;
    Interlock.interlock.fragment_shader_sample_interlock => FragmentShaderSampleInterlock;
    RasterOrder.raster_order.rasterization_order_color_attachment_access => RasterizationOrderColorAttachmentAccess;
    RasterOrder.raster_order.rasterization_order_depth_attachment_access => RasterizationOrderDepthAttachmentAccess;
    RasterOrder.raster_order.rasterization_order_stencil_attachment_access => RasterizationOrderStencilAttachmentAccess;
    Maintenance8.maintenance8.maintenance8 => Maintenance8;
    ShaderFma.shader_fma.shader_fma_float16 => ShaderFmaFloat16;
    ShaderFma.shader_fma.shader_fma_float32 => ShaderFmaFloat32;
    Maintenance5.maintenance5.maintenance5 => Maintenance5;
    BorderColorSwizzle.border_color_swizzle.border_color_swizzle => BorderColorSwizzle;
    BorderColorSwizzle.border_color_swizzle.border_color_swizzle_from_image => BorderColorSwizzleFromImage;
    DepthBiasControl.depth_bias_control.depth_bias_control => DepthBiasControl;
    DepthBiasControl.depth_bias_control.float_representation => FloatRepresentation;
    DepthBiasControl.depth_bias_control.depth_bias_exact => DepthBiasExact;
    DepthClipEnable.depth_clip_enable.depth_clip_enable => DepthClipEnable;
    DepthClampControl.depth_clamp_control.depth_clamp_control => DepthClampControl;
    ListRestart.list_restart.primitive_topology_list_restart => PrimitiveTopologyListRestart;
    ListRestart.list_restart.primitive_topology_patch_list_restart => PrimitiveTopologyPatchListRestart;
    ProvokingVertex.provoking_vertex.provoking_vertex_last => ProvokingVertexLast;
    RayQuery.ray_query.ray_query => RayQuery;
    Acceleration.acceleration_structure.acceleration_structure => AccelerationStructure;
    MeshShader.mesh_shader.task_shader => TaskShader;
    MeshShader.mesh_shader.mesh_shader => MeshShader;
    ExtendedDynamicState.extended_dynamic_state.extended_dynamic_state => ExtendedDynamicState;
    ExtendedDynamicState2.extended_dynamic_state2.extended_dynamic_state2 => ExtendedDynamicState2;
    ExtendedDynamicState3.extended_dynamic_state3.extended_dynamic_state3_color_write_mask => ExtendedDynamicState3ColorWriteMask;
    FeedbackLoop.feedback_loop.attachment_feedback_loop_layout => AttachmentFeedbackLoopLayout;
    Robustness2.robustness2.robust_buffer_access2 => RobustBufferAccess2;
    Robustness2.robustness2.robust_image_access2 => RobustImageAccess2;
    Robustness2.robustness2.null_descriptor => NullDescriptor;
    ImageRobustness.image_robustness.robust_image_access => RobustImageAccess;
    PortabilitySubset.portability_subset.constant_alpha_color_blend_factors => ConstantAlphaColorBlendFactors;
    PortabilitySubset.portability_subset.vertex_attribute_access_beyond_stride => VertexAttributeAccessBeyondStride;
    PortabilitySubset.portability_subset.sampler_mip_lod_bias => SamplerMipLodBias;
    PortabilitySubset.portability_subset.mutable_comparison_samplers => MutableComparisonSamplers;
    PortabilitySubset.portability_subset.shader_sample_rate_interpolation_functions => ShaderSampleRateInterpolationFunctions;
    PortabilitySubset.portability_subset.multisample_array_image => MultisampleArrayImage;
    PortabilitySubset.portability_subset.image_view2_d_on3_d_image => ImageView2DOn3DImage;
    PortabilitySubset.portability_subset.separate_stencil_mask_ref => SeparateStencilMaskRef;
    PortabilitySubset.portability_subset.image_view_format_swizzle => ImageViewFormatSwizzle;
}

members! {
    /// The member of each vocabulary feature Vulkan 1.3 promoted, in
    /// `VkPhysicalDeviceVulkan13Features`: where the device's version is 1.3 or later, its only
    /// member (Vulkan forbids chaining it beside the promoted extension structures).
    CORE_1_3:
    V13.v13.robust_image_access => RobustImageAccess;
    V13.v13.shader_demote_to_helper_invocation => ShaderDemoteToHelperInvocation;
    V13.v13.shader_terminate_invocation => ShaderTerminateInvocation;
    V13.v13.subgroup_size_control => SubgroupSizeControl;
    V13.v13.compute_full_subgroups => ComputeFullSubgroups;
    V13.v13.synchronization2 => Synchronization2;
    V13.v13.shader_zero_initialize_workgroup_memory => ShaderZeroInitializeWorkgroupMemory;
    V13.v13.maintenance4 => Maintenance4;
}

/// Every device extension of the vocabulary by its names. Where one vocabulary extension has two
/// names (a `KHR` promotion of an `EXT` or `NV`, an `ARM` or `VALVE` spelling), the first the
/// device lists is enabled.
const EXTENSIONS: &[(VkExtension, &CStr)] = {
    use VkExtension as X;
    &[
        (X::ExtShaderStencilExport, c"VK_EXT_shader_stencil_export"),
        (X::ExtSubgroupSizeControl, c"VK_EXT_subgroup_size_control"),
        (
            X::ExtShaderDemoteToHelperInvocation,
            c"VK_EXT_shader_demote_to_helper_invocation",
        ),
        (
            X::KhrShaderTerminateInvocation,
            c"VK_KHR_shader_terminate_invocation",
        ),
        (
            X::KhrZeroInitializeWorkgroupMemory,
            c"VK_KHR_zero_initialize_workgroup_memory",
        ),
        (X::KhrSynchronization2, c"VK_KHR_synchronization2"),
        (X::KhrMaintenance4, c"VK_KHR_maintenance4"),
        (X::KhrShaderSubgroupRotate, c"VK_KHR_shader_subgroup_rotate"),
        (
            X::KhrDynamicRenderingLocalRead,
            c"VK_KHR_dynamic_rendering_local_read",
        ),
        (
            X::KhrVertexAttributeDivisor,
            c"VK_KHR_vertex_attribute_divisor",
        ),
        (
            X::ExtVertexAttributeDivisor,
            c"VK_EXT_vertex_attribute_divisor",
        ),
        (X::ExtShaderAtomicFloat, c"VK_EXT_shader_atomic_float"),
        (X::ExtShaderAtomicFloat2, c"VK_EXT_shader_atomic_float2"),
        (
            X::ExtShaderImageAtomicInt64,
            c"VK_EXT_shader_image_atomic_int64",
        ),
        (
            X::KhrWorkgroupMemoryExplicitLayout,
            c"VK_KHR_workgroup_memory_explicit_layout",
        ),
        (
            X::KhrShaderMaximalReconvergence,
            c"VK_KHR_shader_maximal_reconvergence",
        ),
        (X::KhrShaderQuadControl, c"VK_KHR_shader_quad_control"),
        (
            X::KhrShaderSubgroupUniformControlFlow,
            c"VK_KHR_shader_subgroup_uniform_control_flow",
        ),
        (
            X::KhrComputeShaderDerivatives,
            c"VK_KHR_compute_shader_derivatives",
        ),
        (
            X::KhrComputeShaderDerivatives,
            c"VK_NV_compute_shader_derivatives",
        ),
        (
            X::KhrFragmentShaderBarycentric,
            c"VK_KHR_fragment_shader_barycentric",
        ),
        (
            X::ExtFragmentShaderInterlock,
            c"VK_EXT_fragment_shader_interlock",
        ),
        (
            X::ExtRasterizationOrderAttachmentAccess,
            c"VK_EXT_rasterization_order_attachment_access",
        ),
        (
            X::ExtRasterizationOrderAttachmentAccess,
            c"VK_ARM_rasterization_order_attachment_access",
        ),
        (X::KhrMaintenance8, c"VK_KHR_maintenance8"),
        (X::KhrShaderFma, c"VK_KHR_shader_fma"),
        (
            X::ExtDepthRangeUnrestricted,
            c"VK_EXT_depth_range_unrestricted",
        ),
        (X::ExtDepthBiasControl, c"VK_EXT_depth_bias_control"),
        (X::ExtDepthClipEnable, c"VK_EXT_depth_clip_enable"),
        (X::ExtDepthClampControl, c"VK_EXT_depth_clamp_control"),
        (X::ExtSampleLocations, c"VK_EXT_sample_locations"),
        (
            X::ExtPrimitiveTopologyListRestart,
            c"VK_EXT_primitive_topology_list_restart",
        ),
        (X::ExtProvokingVertex, c"VK_EXT_provoking_vertex"),
        (X::KhrRayQuery, c"VK_KHR_ray_query"),
        (
            X::KhrAccelerationStructure,
            c"VK_KHR_acceleration_structure",
        ),
        (
            X::KhrDeferredHostOperations,
            c"VK_KHR_deferred_host_operations",
        ),
        (X::ExtRobustness2, c"VK_EXT_robustness2"),
        (X::ExtRobustness2, c"VK_KHR_robustness2"),
        (X::ExtImageRobustness, c"VK_EXT_image_robustness"),
        (X::ExtExtendedDynamicState, c"VK_EXT_extended_dynamic_state"),
        (
            X::ExtExtendedDynamicState2,
            c"VK_EXT_extended_dynamic_state2",
        ),
        (
            X::ExtExtendedDynamicState3,
            c"VK_EXT_extended_dynamic_state3",
        ),
        (
            X::ExtAttachmentFeedbackLoopLayout,
            c"VK_EXT_attachment_feedback_loop_layout",
        ),
        (X::KhrPushDescriptor, c"VK_KHR_push_descriptor"),
        (
            X::ExtTextureCompressionAstcHdr,
            c"VK_EXT_texture_compression_astc_hdr",
        ),
        (X::ImgFormatPvrtc, c"VK_IMG_format_pvrtc"),
        (X::ExtBorderColorSwizzle, c"VK_EXT_border_color_swizzle"),
        (X::ExtCustomBorderColor, c"VK_EXT_custom_border_color"),
        (X::KhrMaintenance5, c"VK_KHR_maintenance5"),
        (X::KhrDynamicRendering, c"VK_KHR_dynamic_rendering"),
        (X::KhrPortabilitySubset, c"VK_KHR_portability_subset"),
        (
            X::ExtShaderViewportIndexLayer,
            c"VK_EXT_shader_viewport_index_layer",
        ),
        (X::ExtExternalMemoryHost, c"VK_EXT_external_memory_host"),
        (
            X::ExtMutableDescriptorType,
            c"VK_EXT_mutable_descriptor_type",
        ),
        (
            X::ExtMutableDescriptorType,
            c"VK_VALVE_mutable_descriptor_type",
        ),
        (X::ExtDescriptorBuffer, c"VK_EXT_descriptor_buffer"),
        (X::ExtMeshShader, c"VK_EXT_mesh_shader"),
        (
            X::KhrShaderUntypedPointers,
            c"VK_KHR_shader_untyped_pointers",
        ),
        (X::KhrCooperativeMatrix, c"VK_KHR_cooperative_matrix"),
        (X::KhrShaderBfloat16, c"VK_KHR_shader_bfloat16"),
        (X::QcomTileShading, c"VK_QCOM_tile_shading"),
        (X::KhrFormatFeatureFlags2, c"VK_KHR_format_feature_flags2"),
        (X::ExtPostDepthCoverage, c"VK_EXT_post_depth_coverage"),
    ]
};

/// One feature of an extension outside the vocabulary that a caller may list
/// (`DeviceOptions::device_extensions`): its structure, its name, and its member.
struct ListedFeature {
    structure: Structure,
    name: &'static str,
    member: fn(&mut FeatureChain) -> &mut vk::Bool32,
}

/// Each name of an extension outside the vocabulary whose features the runtime enables where
/// the caller lists it, with its feature structure (the `KHR` and `EXT` names of one extension
/// share one).
const LISTED: [(&CStr, Structure); 4] = [
    (
        c"VK_KHR_swapchain_maintenance1",
        Structure::SwapchainMaintenance1,
    ),
    (
        c"VK_EXT_swapchain_maintenance1",
        Structure::SwapchainMaintenance1,
    ),
    (
        c"VK_KHR_present_mode_fifo_latest_ready",
        Structure::FifoLatestReady,
    ),
    (
        c"VK_EXT_present_mode_fifo_latest_ready",
        Structure::FifoLatestReady,
    ),
];

/// Every feature of [`LISTED`]'s structures.
const LISTED_FEATURES: [ListedFeature; 2] = [
    ListedFeature {
        structure: Structure::SwapchainMaintenance1,
        name: "swapchainMaintenance1",
        member: |chain| &mut chain.swapchain_maintenance1.swapchain_maintenance1,
    },
    ListedFeature {
        structure: Structure::FifoLatestReady,
        name: "presentModeFifoLatestReady",
        member: |chain| &mut chain.fifo_latest_ready.present_mode_fifo_latest_ready,
    },
];

/// The feature structures of the extensions `names` lists that the runtime knows ([`LISTED`]),
/// each once, with the first of `names` that names it.
pub(super) fn listed_structures(names: &[CString]) -> Vec<(&'static CStr, Structure)> {
    let mut listed: Vec<(&'static CStr, Structure)> = Vec::new();
    for name in names {
        let known = LISTED.iter().find(|(known, _)| *known == name.as_c_str());
        if let Some(&(name, structure)) = known {
            if listed.iter().all(|&(_, taken)| taken != structure) {
                listed.push((name, structure));
            }
        }
    }
    listed
}

/// Which features of `listed`'s structures the device `physical` of `instance` has
/// (`vkGetPhysicalDeviceFeatures2` with those structures chained), by name.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, which uses it at Vulkan 1.1 or later, and lists
/// each extension `listed` names.
pub(super) unsafe fn listed_reported(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    listed: &[(&'static CStr, Structure)],
) -> impl Fn(&'static str) -> bool {
    let mut chain = FeatureChain::default();
    {
        let c = &mut chain;
        let has = |structure| listed.iter().any(|&(_, taken)| taken == structure);
        let mut features = vk::PhysicalDeviceFeatures2::default();
        if has(Structure::SwapchainMaintenance1) {
            features = features.push_next(&mut c.swapchain_maintenance1);
        }
        if has(Structure::FifoLatestReady) {
            features = features.push_next(&mut c.fifo_latest_ready);
        }
        // SAFETY: the caller's contract; the chain lives for the call, of structures the listed
        // extensions define.
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
    }
    move |name| {
        let mut chain = chain;
        LISTED_FEATURES
            .iter()
            .find(|feature| feature.name == name)
            .is_some_and(|feature| *(feature.member)(&mut chain) == vk::TRUE)
    }
}

/// Chains every structure of `$chain` whose [`Structure`] `$linked` holds onto `$info`.
macro_rules! link_extensions {
    ($info:ident, $chain:ident, $linked:ident) => {
        link_extensions!(@ $info, $chain, $linked;
            V13 => v13,
            SubgroupSizeControl => subgroup_size_control,
            Demote => demote,
            Terminate => terminate,
            ZeroInitialize => zero_initialize,
            Synchronization2 => synchronization2,
            Maintenance4 => maintenance4,
            SubgroupRotate => subgroup_rotate,
            LocalRead => local_read,
            Divisor => divisor,
            AtomicFloat => atomic_float,
            AtomicFloat2 => atomic_float2,
            ImageAtomicInt64 => image_atomic_int64,
            ExplicitLayout => explicit_layout,
            MaximalReconvergence => maximal_reconvergence,
            QuadControl => quad_control,
            UniformControlFlow => uniform_control_flow,
            Derivatives => derivatives,
            Barycentric => barycentric,
            Interlock => interlock,
            RasterOrder => raster_order,
            Maintenance8 => maintenance8,
            ShaderFma => shader_fma,
            Maintenance5 => maintenance5,
            HostImageCopy => host_image_copy,
            BorderColorSwizzle => border_color_swizzle,
            DepthBiasControl => depth_bias_control,
            DepthClipEnable => depth_clip_enable,
            DepthClampControl => depth_clamp_control,
            ListRestart => list_restart,
            ProvokingVertex => provoking_vertex,
            RayQuery => ray_query,
            Acceleration => acceleration_structure,
            MeshShader => mesh_shader,
            ExtendedDynamicState => extended_dynamic_state,
            ExtendedDynamicState2 => extended_dynamic_state2,
            ExtendedDynamicState3 => extended_dynamic_state3,
            FeedbackLoop => feedback_loop,
            Robustness2 => robustness2,
            ImageRobustness => image_robustness,
            PortabilitySubset => portability_subset,
            GeneratedCommands => generated_commands,
            SwapchainMaintenance1 => swapchain_maintenance1,
            FifoLatestReady => fifo_latest_ready,
            LinearColorAttachment => linear_color_attachment,
            PipelineBinary => pipeline_binary,
            InternalCacheControl => internal_cache_control,
            GraphicsPipelineLibrary => graphics_pipeline_library
        )
    };
    (@ $info:ident, $chain:ident, $linked:ident; $($variant:ident => $field:ident),*) => {
        $(if $linked.contains(&Structure::$variant) {
            $info = $info.push_next(&mut $chain.$field);
        })*
    };
}

/// What one `vkCreateDevice` enables: the feature chain, the extension structures it links, and
/// the extension names.
pub(super) struct Enables {
    chain: FeatureChain,
    linked: BTreeSet<Structure>,
    names: Vec<CString>,
}

impl Enables {
    /// The enables of `set` on a device of version `api` (its own, capped by the instance's)
    /// listing `listed`, plus the extensions `extra` names (the caller's options). A feature with
    /// no member, a vocabulary extension the device lists under none of its names, or a feature
    /// whose structure neither an enabled extension nor `api` exposes is an error: the device is
    /// never created on a list that would silently drop one.
    pub(super) fn of(
        set: &EnableSet,
        api: ApiVersion,
        listed: &BTreeSet<CString>,
        extra: &[CString],
    ) -> Result<Enables, Error> {
        let mut enables = Enables {
            chain: FeatureChain::default(),
            linked: BTreeSet::new(),
            names: Vec::new(),
        };
        // The runtime's own needs, outside the vocabulary, each core and required in Vulkan 1.2:
        // events and the completion of command buffers are timeline semaphores, and timestamp
        // slots are reset on the host (`gpu_time.rs`).
        enables.chain.v12.timeline_semaphore = vk::TRUE;
        enables.chain.v12.host_query_reset = vk::TRUE;
        for &extension in &set.extensions {
            let name = names_of(extension)
                .find(|name| listed.contains(*name))
                .ok_or(Error::NotImplemented(RuntimeGap::ExtensionEnable(
                    extension,
                )))?;
            enables.push_name(name.to_owned());
        }
        for name in extra {
            enables.push_name(name.clone());
        }
        for &feature in &set.features {
            let unmatched = || Error::NotImplemented(RuntimeGap::FeatureEnable(feature));
            let in_table = |table: &'static [Member]| {
                table
                    .iter()
                    .find(|member| member.feature == feature)
                    .ok_or_else(unmatched)
            };
            let member = in_table(MEMBERS)?;
            // The core version that took the feature's structure over, where `api` reaches it.
            let core = match member.structure.promotion() {
                // Core functionality without a member at `api`: nothing to set.
                Some(Promotion::Functionality(version)) if api >= version => continue,
                Some(Promotion::Members(version)) if api >= version => Some(version),
                _ => None,
            };
            let member = if core == Some(ApiVersion::V1_3) {
                in_table(CORE_1_3)?
            } else {
                member
            };
            let structure = member.structure;
            let exposed = core.is_some()
                || structure.extensions().is_empty()
                || structure
                    .extensions()
                    .iter()
                    .any(|&extension| set.extensions.contains(&extension));
            if !exposed {
                return Err(unmatched());
            }
            if !matches!(structure, Structure::Core | Structure::V11 | Structure::V12) {
                enables.linked.insert(structure);
            }
            *(member.write)(&mut enables.chain) = vk::TRUE;
        }
        Ok(enables)
    }

    /// Enables the portability subset's `imageViewFormatReinterpretation`, the runtime's own need
    /// outside the vocabulary (Metal's texture views of another component layout,
    /// `view_plan.rs`), on a device whose subset has it and whose set enables the subset.
    pub(super) fn enable_view_reinterpretation(&mut self) {
        self.chain
            .portability_subset
            .image_view_format_reinterpretation = vk::TRUE;
        self.linked.insert(Structure::PortabilitySubset);
    }

    /// Enables `VK_EXT_device_generated_commands` and its `deviceGeneratedCommands`, the
    /// runtime's own need outside the vocabulary (an execution of commands a kernel writes on
    /// the GPU alone), on a device that has them and whose set enables what the extension requires
    /// (`bufferDeviceAddress` and `VK_KHR_maintenance5`).
    pub(super) fn enable_generated_commands(&mut self) {
        self.chain.generated_commands.device_generated_commands = vk::TRUE;
        self.linked.insert(Structure::GeneratedCommands);
        self.push_name(GENERATED_COMMANDS.to_owned());
    }

    /// Enables `hostImageCopy` and the extensions `names` ([`host_image_copy_names`] of the
    /// device's version), the runtime's own need outside the vocabulary: a
    /// heap's colour texture takes host-transfer usage on a driver whose compressed images hide
    /// writes through an alias and whose host-transfer images keep their bytes (`texture/heap.rs`),
    /// on a device that has the feature and lists the names.
    pub(super) fn enable_host_image_copy(&mut self, names: &[&CStr]) {
        self.chain.host_image_copy.host_image_copy = vk::TRUE;
        self.linked.insert(Structure::HostImageCopy);
        for &name in names {
            self.push_name(name.to_owned());
        }
    }

    /// Enables `features`' sparse members (`sparse::device::wanted`), the runtime's own need
    /// outside the vocabulary: sparse heaps and their textures.
    pub(super) fn enable_sparse(&mut self, features: &vk::PhysicalDeviceFeatures) {
        let core = &mut self.chain.core;
        core.sparse_binding |= features.sparse_binding;
        core.sparse_residency_image2_d |= features.sparse_residency_image2_d;
        core.sparse_residency_image3_d |= features.sparse_residency_image3_d;
        core.sparse_residency2_samples |= features.sparse_residency2_samples;
        core.sparse_residency4_samples |= features.sparse_residency4_samples;
        core.sparse_residency8_samples |= features.sparse_residency8_samples;
        core.sparse_residency16_samples |= features.sparse_residency16_samples;
    }

    /// Enables `VK_NV_linear_color_attachment` and its `linearColorAttachment`, the runtime's own
    /// need outside the vocabulary: a texture over a buffer's memory, a no-copy buffer's host
    /// pages included, renders into its linear image in place where every color attachment of
    /// the render pass instance is linear (`linear_attachment.rs`). `names` are the extension's
    /// and, below Vulkan 1.3, `VK_KHR_format_feature_flags2`'s, which it requires
    /// ([`linear_color_attachment_names`]; below 1.3 too).
    pub(super) fn enable_linear_color_attachment(&mut self, names: &[&CStr]) {
        self.chain.linear_color_attachment.linear_color_attachment = vk::TRUE;
        self.linked.insert(Structure::LinearColorAttachment);
        for &name in names {
            self.push_name(name.to_owned());
        }
    }

    /// Enables `VK_KHR_pipeline_binary`, with its `pipelineBinaries` where `binaries` (the device
    /// has it), and makes the device with its driver's internal pipeline cache off
    /// (`VkDevicePipelineBinaryInternalCacheControlKHR::disableInternalCache`), the runtime's own
    /// need outside the vocabulary: the shader cache's records are the device's only pipeline
    /// cache. For a device whose driver allows it (`pipelineBinaryInternalCacheControl`) and whose
    /// set enables `VK_KHR_maintenance5`, which the extension requires (`create.rs`).
    pub(super) fn turn_driver_pipeline_cache_off(&mut self, binaries: bool) {
        if binaries {
            self.chain.pipeline_binary.pipeline_binaries = vk::TRUE;
            self.linked.insert(Structure::PipelineBinary);
        }
        self.chain.internal_cache_control.disable_internal_cache = vk::TRUE;
        self.linked.insert(Structure::InternalCacheControl);
        self.push_name(PIPELINE_BINARY.to_owned());
    }

    /// Enables `VK_EXT_graphics_pipeline_library` with its `graphicsPipelineLibrary`, and
    /// `VK_KHR_pipeline_library`, which it requires: the runtime's own need outside the
    /// vocabulary, where the device links libraries fast (`create.rs`): a render variant's
    /// `VkPipeline`s are linked from libraries of their parts (`pipeline/libraries.rs`).
    pub(super) fn enable_graphics_pipeline_library(&mut self) {
        self.chain
            .graphics_pipeline_library
            .graphics_pipeline_library = vk::TRUE;
        self.linked.insert(Structure::GraphicsPipelineLibrary);
        for name in GRAPHICS_PIPELINE_LIBRARY {
            self.push_name(name.to_owned());
        }
    }

    /// Enables every feature of `listed`'s structures ([`listed_structures`] of the caller's
    /// extensions) that `reported` ([`listed_reported`]) says the device has, chaining each
    /// structure. A feature the device lacks refuses the device, naming the extension the caller
    /// listed and the feature ([`Error::ExtensionFeature`]): the extension is never enabled
    /// without its features.
    pub(super) fn enable_listed(
        &mut self,
        listed: &[(&'static CStr, Structure)],
        reported: impl Fn(&'static str) -> bool,
    ) -> Result<(), Error> {
        for &(extension, structure) in listed {
            for feature in LISTED_FEATURES
                .iter()
                .filter(|feature| feature.structure == structure)
            {
                if !reported(feature.name) {
                    return Err(Error::ExtensionFeature(ExtensionFeature {
                        extension: extension.to_str().unwrap_or_default(),
                        feature: feature.name,
                    }));
                }
                *(feature.member)(&mut self.chain) = vk::TRUE;
            }
            self.linked.insert(structure);
        }
        Ok(())
    }

    /// Enables each of the caller's optional extensions `names` (`DeviceOptions::
    /// optional_device_extensions`, each listed by the device), where `reported`
    /// ([`listed_reported`] of their [`listed_structures`]) says the device has every feature of
    /// its structure that the runtime knows ([`LISTED`]), chaining the structure with those
    /// features set; an extension of no known structure is enabled as it is. One the device lacks
    /// a feature of is left out.
    pub(super) fn enable_where_featured(
        &mut self,
        names: &[CString],
        reported: impl Fn(&'static str) -> bool,
    ) {
        for name in names {
            let structure = LISTED
                .iter()
                .find(|(known, _)| *known == name.as_c_str())
                .map(|&(_, structure)| structure);
            if let Some(structure) = structure {
                let features = || {
                    LISTED_FEATURES
                        .iter()
                        .filter(move |feature| feature.structure == structure)
                };
                if !features().all(|feature| reported(feature.name)) {
                    continue;
                }
                for feature in features() {
                    *(feature.member)(&mut self.chain) = vk::TRUE;
                }
                self.linked.insert(structure);
            }
            self.push_name(name.clone());
        }
    }

    /// The names of the extensions it enables.
    pub(super) fn names(&self) -> impl Iterator<Item = &CStr> {
        self.names.iter().map(CString::as_c_str)
    }

    fn push_name(&mut self, name: CString) {
        if !self.names.contains(&name) {
            self.names.push(name);
        }
    }

    /// Hands `create` the `VkDeviceCreateInfo` with `queues` and these enables. Every chained
    /// structure lives until `create` returns.
    pub(super) fn with_create_info<R>(
        &mut self,
        queues: &[vk::DeviceQueueCreateInfo<'_>],
        create: impl FnOnce(&vk::DeviceCreateInfo<'_>) -> R,
    ) -> R {
        let names: Vec<*const c_char> = self.names.iter().map(|name| name.as_ptr()).collect();
        let mut chain = self.chain;
        let linked = &self.linked;
        let c = &mut chain;
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(queues)
            .enabled_extension_names(&names)
            .enabled_features(&c.core)
            .push_next(&mut c.v11)
            .push_next(&mut c.v12);
        link_extensions!(info, c, linked);
        create(&info)
    }
}

/// The names of vocabulary extension `extension`, in preference order.
pub(super) fn names_of(extension: VkExtension) -> impl Iterator<Item = &'static CStr> {
    EXTENSIONS
        .iter()
        .filter(move |(named, _)| *named == extension)
        .map(|(_, name)| *name)
}
