//! `DeviceDescription::query`: read one physical device through `ash`.
//!
//! The result is what the physical device **supports**. A description must record what the
//! consumer **enables** (`device::description`), so the consumer removes every feature and
//! extension it does not enable at `vkCreateDevice` before calling `Device::new`.
//!
//! Which structures are chained follows the same exposure rule as the Khronos JSON reader
//! (`device::khronos`), at the **effective** API version `min(instance, device)` (the Vulkan
//! spec limits a physical device's core functionality to the instance's version): a core
//! `VkPhysicalDeviceVulkanXYFeatures` structure from effective version `X.Y`, and an extension structure when its extension is available or its promotion version is
//! reached, never both for one member (a structure the device does not expose is never chained,
//! which would be invalid usage). Every [`VkFeature`] has exactly one source, checked by an
//! exhaustive match.

use std::collections::BTreeSet;

use ash::vk as avk;

use crate::device::description::{
    AllocationLimits, DeviceDescription, FormatProperties,
    GeneratedCommandsProperties as DgcProperties, SubgroupProperties,
};
use crate::device::khronos::extension_from_name;
use crate::device::profile::{FloatControls, FloatWidthControls, Independence, Limits, MeshLimits};
use crate::device::requirements::{ApiVersion, VkFeature};
use crate::device::FormatSupport;
use crate::error::{Capability, DescriptorError, Error, Feature, Unsupported};
use crate::vk;

/// Whether a `VkBool32` is true.
fn on(b: avk::Bool32) -> bool {
    b != avk::FALSE
}

/// The device's available extensions and effective version, deciding which structures may be
/// chained.
struct Exposure {
    /// `min(instance apiVersion, device apiVersion)`.
    api: ApiVersion,
    extensions: BTreeSet<String>,
}

impl Exposure {
    fn core(&self, minor: u32) -> bool {
        self.api >= ApiVersion { major: 1, minor }
    }

    fn ext(&self, names: &[&str]) -> bool {
        names.iter().any(|n| self.extensions.contains(*n))
    }

    /// An extension structure promoted to core `1.minor` (`None`: never promoted).
    fn promoted(&self, names: &[&str], minor: Option<u32>) -> bool {
        self.ext(names) || minor.is_some_and(|m| self.core(m))
    }
}

/// `VkPhysicalDeviceMaintenance8FeaturesKHR`, which `ash 0.38` (Vulkan 1.3.281) predates, laid
/// out as the registry defines it (`vk.xml`: `sType`, `pNext`, `VkBool32 maintenance8`; extension
/// number 575, so its structure type is `1000000000 + 574 * 1000`). It extends
/// `VkPhysicalDeviceFeatures2`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Maintenance8Features {
    s_type: avk::StructureType,
    p_next: *mut std::ffi::c_void,
    maintenance8: avk::Bool32,
}

impl Default for Maintenance8Features {
    fn default() -> Self {
        Maintenance8Features {
            s_type: <Self as avk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            maintenance8: avk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceMaintenance8FeaturesKHR`, which begins
// with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl avk::TaggedStructure for Maintenance8Features {
    const STRUCTURE_TYPE: avk::StructureType = avk::StructureType::from_raw(1_000_574_000);
}

// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` in the structure's `structextends`.
unsafe impl avk::ExtendsPhysicalDeviceFeatures2 for Maintenance8Features {}

/// `VkPhysicalDeviceShaderFmaFeaturesKHR`, which `ash 0.38` predates, laid out as the registry
/// defines it (`vk.xml`: `sType`, `pNext`, `VkBool32 shaderFmaFloat16`, `shaderFmaFloat32`,
/// `shaderFmaFloat64`; extension number 580, so its structure type is
/// `1000000000 + 579 * 1000`). It extends `VkPhysicalDeviceFeatures2`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ShaderFmaFeatures {
    s_type: avk::StructureType,
    p_next: *mut std::ffi::c_void,
    shader_fma_float16: avk::Bool32,
    shader_fma_float32: avk::Bool32,
    shader_fma_float64: avk::Bool32,
}

impl Default for ShaderFmaFeatures {
    fn default() -> Self {
        ShaderFmaFeatures {
            s_type: <Self as avk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            shader_fma_float16: avk::FALSE,
            shader_fma_float32: avk::FALSE,
            shader_fma_float64: avk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceShaderFmaFeaturesKHR`, which begins with
// `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl avk::TaggedStructure for ShaderFmaFeatures {
    const STRUCTURE_TYPE: avk::StructureType = avk::StructureType::from_raw(1_000_579_000);
}

// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` in the structure's `structextends`.
unsafe impl avk::ExtendsPhysicalDeviceFeatures2 for ShaderFmaFeatures {}

/// `VkPhysicalDeviceMaintenance10PropertiesKHR`, which `ash 0.38` predates, laid out as the
/// registry defines it (`vk.xml`: `sType`, `pNext`, `VkBool32 rgba4OpaqueBlackSwizzled`,
/// `resolveSrgbFormatAppliesTransferFunction`, `resolveSrgbFormatSupportsTransferFunctionControl`;
/// extension number 631, so its structure type is `1000000000 + 630 * 1000 + 1`). It extends
/// `VkPhysicalDeviceProperties2`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Maintenance10Properties {
    s_type: avk::StructureType,
    p_next: *mut std::ffi::c_void,
    pub(crate) rgba4_opaque_black_swizzled: avk::Bool32,
    resolve_srgb_format_applies_transfer_function: avk::Bool32,
    resolve_srgb_format_supports_transfer_function_control: avk::Bool32,
}

impl Default for Maintenance10Properties {
    fn default() -> Self {
        Maintenance10Properties {
            s_type: <Self as avk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            rgba4_opaque_black_swizzled: avk::FALSE,
            resolve_srgb_format_applies_transfer_function: avk::FALSE,
            resolve_srgb_format_supports_transfer_function_control: avk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceMaintenance10PropertiesKHR`, which begins
// with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl avk::TaggedStructure for Maintenance10Properties {
    const STRUCTURE_TYPE: avk::StructureType = avk::StructureType::from_raw(1_000_630_001);
}

// SAFETY: the registry lists `VkPhysicalDeviceProperties2` in the structure's `structextends`.
unsafe impl avk::ExtendsPhysicalDeviceProperties2 for Maintenance10Properties {}

/// `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`, which `ash 0.38` predates, laid out as
/// the registry defines it (`vk.xml`: `sType`, `pNext`, `VkBool32 deviceGeneratedCommands`,
/// `dynamicGeneratedPipelineLayout`; extension number 573, so its structure type is
/// `1000000000 + 572 * 1000`). It extends `VkPhysicalDeviceFeatures2`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GeneratedCommandsFeatures {
    s_type: avk::StructureType,
    p_next: *mut std::ffi::c_void,
    device_generated_commands: avk::Bool32,
    dynamic_generated_pipeline_layout: avk::Bool32,
}

impl Default for GeneratedCommandsFeatures {
    fn default() -> Self {
        GeneratedCommandsFeatures {
            s_type: <Self as avk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            device_generated_commands: avk::FALSE,
            dynamic_generated_pipeline_layout: avk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`, which
// begins with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl avk::TaggedStructure for GeneratedCommandsFeatures {
    const STRUCTURE_TYPE: avk::StructureType = avk::StructureType::from_raw(1_000_572_000);
}

// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` in the structure's `structextends`.
unsafe impl avk::ExtendsPhysicalDeviceFeatures2 for GeneratedCommandsFeatures {}

/// `VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT`, which `ash 0.38` predates, laid out as
/// the registry defines it (`vk.xml`: six `uint32_t` limits, the input-mode flags, three
/// `VkShaderStageFlags`, two `VkBool32`; structure type `1000000000 + 572 * 1000 + 1`). It extends
/// `VkPhysicalDeviceProperties2`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GeneratedCommandsProperties {
    s_type: avk::StructureType,
    p_next: *mut std::ffi::c_void,
    max_indirect_pipeline_count: u32,
    max_indirect_shader_object_count: u32,
    max_indirect_sequence_count: u32,
    max_indirect_commands_token_count: u32,
    max_indirect_commands_token_offset: u32,
    max_indirect_commands_indirect_stride: u32,
    supported_input_modes: avk::Flags,
    supported_stages: avk::ShaderStageFlags,
    supported_stages_pipeline_binding: avk::ShaderStageFlags,
    supported_stages_shader_binding: avk::ShaderStageFlags,
    transform_feedback: avk::Bool32,
    multi_draw_indirect_count: avk::Bool32,
}

impl Default for GeneratedCommandsProperties {
    fn default() -> Self {
        GeneratedCommandsProperties {
            s_type: <Self as avk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            max_indirect_pipeline_count: 0,
            max_indirect_shader_object_count: 0,
            max_indirect_sequence_count: 0,
            max_indirect_commands_token_count: 0,
            max_indirect_commands_token_offset: 0,
            max_indirect_commands_indirect_stride: 0,
            supported_input_modes: 0,
            supported_stages: avk::ShaderStageFlags::empty(),
            supported_stages_pipeline_binding: avk::ShaderStageFlags::empty(),
            supported_stages_shader_binding: avk::ShaderStageFlags::empty(),
            transform_feedback: avk::FALSE,
            multi_draw_indirect_count: avk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT`,
// which begins with `sType` and `pNext` as every chained structure must, and this is its
// structure type.
unsafe impl avk::TaggedStructure for GeneratedCommandsProperties {
    const STRUCTURE_TYPE: avk::StructureType = avk::StructureType::from_raw(1_000_572_001);
}

// SAFETY: the registry lists `VkPhysicalDeviceProperties2` in the structure's `structextends`.
unsafe impl avk::ExtendsPhysicalDeviceProperties2 for GeneratedCommandsProperties {}

/// `VkPhysicalDeviceDepthClampControlFeaturesEXT`, which `ash 0.38` predates, laid out as the
/// registry defines it (`vk.xml`: `sType`, `pNext`, `VkBool32 depthClampControl`; extension
/// number 583, so its structure type is `1000000000 + 582 * 1000`). It extends
/// `VkPhysicalDeviceFeatures2`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DepthClampControlFeatures {
    s_type: avk::StructureType,
    p_next: *mut std::ffi::c_void,
    depth_clamp_control: avk::Bool32,
}

impl Default for DepthClampControlFeatures {
    fn default() -> Self {
        DepthClampControlFeatures {
            s_type: <Self as avk::TaggedStructure>::STRUCTURE_TYPE,
            p_next: std::ptr::null_mut(),
            depth_clamp_control: avk::FALSE,
        }
    }
}

// SAFETY: the layout is the registry's `VkPhysicalDeviceDepthClampControlFeaturesEXT`, which
// begins with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl avk::TaggedStructure for DepthClampControlFeatures {
    const STRUCTURE_TYPE: avk::StructureType = avk::StructureType::from_raw(1_000_582_000);
}

// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` in the structure's `structextends`.
unsafe impl avk::ExtendsPhysicalDeviceFeatures2 for DepthClampControlFeatures {}

/// Every feature structure the core reads. Structures that were not chained stay zeroed.
#[derive(Default)]
struct FeatureChain {
    core: avk::PhysicalDeviceFeatures,
    v11: avk::PhysicalDeviceVulkan11Features<'static>,
    v12: avk::PhysicalDeviceVulkan12Features<'static>,
    v13: avk::PhysicalDeviceVulkan13Features<'static>,
    // Promoted to 1.3: read only below 1.3.
    image_robustness: avk::PhysicalDeviceImageRobustnessFeatures<'static>,
    subgroup_size_control: avk::PhysicalDeviceSubgroupSizeControlFeatures<'static>,
    demote: avk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures<'static>,
    terminate: avk::PhysicalDeviceShaderTerminateInvocationFeatures<'static>,
    zero_init: avk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures<'static>,
    sync2: avk::PhysicalDeviceSynchronization2Features<'static>,
    maintenance4: avk::PhysicalDeviceMaintenance4Features<'static>,
    // Promoted to 1.4 (no `Vulkan14Features` in `ash 0.38`): read through their own structures.
    rotate: avk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR<'static>,
    local_read: avk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR<'static>,
    divisor: avk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR<'static>,
    maintenance5: avk::PhysicalDeviceMaintenance5FeaturesKHR<'static>,
    // Extension-only.
    robustness2: avk::PhysicalDeviceRobustness2FeaturesEXT<'static>,
    atomic_float: avk::PhysicalDeviceShaderAtomicFloatFeaturesEXT<'static>,
    atomic_float2: avk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT<'static>,
    image_atomic_int64: avk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT<'static>,
    explicit_layout: avk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR<'static>,
    maximal_reconvergence: avk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR<'static>,
    quad_control: avk::PhysicalDeviceShaderQuadControlFeaturesKHR<'static>,
    uniform_control_flow: avk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR<'static>,
    derivatives: avk::PhysicalDeviceComputeShaderDerivativesFeaturesNV<'static>,
    barycentric: avk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR<'static>,
    interlock: avk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT<'static>,
    raster_order: avk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT<'static>,
    dynamic_state: avk::PhysicalDeviceExtendedDynamicStateFeaturesEXT<'static>,
    dynamic_state2: avk::PhysicalDeviceExtendedDynamicState2FeaturesEXT<'static>,
    dynamic_state3: avk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static>,
    feedback_loop: avk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT<'static>,
    maintenance8: Maintenance8Features,
    shader_fma: ShaderFmaFeatures,
    border_color_swizzle: avk::PhysicalDeviceBorderColorSwizzleFeaturesEXT<'static>,
    depth_bias_control: avk::PhysicalDeviceDepthBiasControlFeaturesEXT<'static>,
    depth_clip_enable: avk::PhysicalDeviceDepthClipEnableFeaturesEXT<'static>,
    depth_clamp_control: DepthClampControlFeatures,
    list_restart: avk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT<'static>,
    provoking_vertex: avk::PhysicalDeviceProvokingVertexFeaturesEXT<'static>,
    ray_query: avk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
    acceleration_structure: avk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    portability_subset: avk::PhysicalDevicePortabilitySubsetFeaturesKHR<'static>,
    generated_commands: GeneratedCommandsFeatures,
    mesh_shader: avk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
}

const EXT_IMAGE_ROBUSTNESS: &[&str] = &["VK_EXT_image_robustness"];
const EXT_SUBGROUP_SIZE_CONTROL: &[&str] = &["VK_EXT_subgroup_size_control"];
const EXT_DEMOTE: &[&str] = &["VK_EXT_shader_demote_to_helper_invocation"];
const EXT_TERMINATE: &[&str] = &["VK_KHR_shader_terminate_invocation"];
const EXT_ZERO_INIT: &[&str] = &["VK_KHR_zero_initialize_workgroup_memory"];
const EXT_SYNC2: &[&str] = &["VK_KHR_synchronization2"];
const EXT_MAINTENANCE4: &[&str] = &["VK_KHR_maintenance4"];
const EXT_ROTATE: &[&str] = &["VK_KHR_shader_subgroup_rotate"];
const EXT_LOCAL_READ: &[&str] = &["VK_KHR_dynamic_rendering_local_read"];
const EXT_MAINTENANCE5: &[&str] = &["VK_KHR_maintenance5"];
const EXT_DIVISOR: &[&str] = &[
    "VK_KHR_vertex_attribute_divisor",
    "VK_EXT_vertex_attribute_divisor",
];
const EXT_ROBUSTNESS2: &[&str] = &["VK_EXT_robustness2", "VK_KHR_robustness2"];
const EXT_ATOMIC_FLOAT: &[&str] = &["VK_EXT_shader_atomic_float"];
const EXT_ATOMIC_FLOAT2: &[&str] = &["VK_EXT_shader_atomic_float2"];
const EXT_IMAGE_ATOMIC_INT64: &[&str] = &["VK_EXT_shader_image_atomic_int64"];
const EXT_EXPLICIT_LAYOUT: &[&str] = &["VK_KHR_workgroup_memory_explicit_layout"];
const EXT_MAXIMAL_RECONVERGENCE: &[&str] = &["VK_KHR_shader_maximal_reconvergence"];
const EXT_QUAD_CONTROL: &[&str] = &["VK_KHR_shader_quad_control"];
const EXT_UNIFORM_CONTROL_FLOW: &[&str] = &["VK_KHR_shader_subgroup_uniform_control_flow"];
const EXT_DERIVATIVES: &[&str] = &[
    "VK_KHR_compute_shader_derivatives",
    "VK_NV_compute_shader_derivatives",
];
const EXT_BARYCENTRIC: &[&str] = &["VK_KHR_fragment_shader_barycentric"];
const EXT_INTERLOCK: &[&str] = &["VK_EXT_fragment_shader_interlock"];
const EXT_RASTER_ORDER: &[&str] = &[
    "VK_EXT_rasterization_order_attachment_access",
    "VK_ARM_rasterization_order_attachment_access",
];
const EXT_DYNAMIC_STATE: &[&str] = &["VK_EXT_extended_dynamic_state"];
const EXT_DYNAMIC_STATE2: &[&str] = &["VK_EXT_extended_dynamic_state2"];
const EXT_DYNAMIC_STATE3: &[&str] = &["VK_EXT_extended_dynamic_state3"];
const EXT_FEEDBACK_LOOP: &[&str] = &["VK_EXT_attachment_feedback_loop_layout"];
const EXT_MAINTENANCE8: &[&str] = &["VK_KHR_maintenance8"];
const EXT_SHADER_FMA: &[&str] = &["VK_KHR_shader_fma"];
const EXT_BORDER_COLOR_SWIZZLE: &[&str] = &["VK_EXT_border_color_swizzle"];
const EXT_DEPTH_BIAS_CONTROL: &[&str] = &["VK_EXT_depth_bias_control"];
const EXT_DEPTH_CLIP_ENABLE: &[&str] = &["VK_EXT_depth_clip_enable"];
const EXT_DEPTH_CLAMP_CONTROL: &[&str] = &["VK_EXT_depth_clamp_control"];
const EXT_LIST_RESTART: &[&str] = &["VK_EXT_primitive_topology_list_restart"];
const EXT_PROVOKING_VERTEX: &[&str] = &["VK_EXT_provoking_vertex"];
const EXT_SAMPLE_LOCATIONS: &[&str] = &["VK_EXT_sample_locations"];
const EXT_RAY_QUERY: &[&str] = &["VK_KHR_ray_query"];
const EXT_ACCELERATION_STRUCTURE: &[&str] = &["VK_KHR_acceleration_structure"];
const EXT_PORTABILITY_SUBSET: &[&str] = &["VK_KHR_portability_subset"];
const EXT_GENERATED_COMMANDS: &[&str] = &["VK_EXT_device_generated_commands"];
const EXT_MESH_SHADER: &[&str] = &["VK_EXT_mesh_shader"];

/// Which feature structures are chained on a device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Chained {
    v13: bool,
    image_robustness: bool,
    subgroup_size_control: bool,
    demote: bool,
    terminate: bool,
    zero_init: bool,
    sync2: bool,
    maintenance4: bool,
    rotate: bool,
    local_read: bool,
    divisor: bool,
    maintenance5: bool,
    robustness2: bool,
    atomic_float: bool,
    atomic_float2: bool,
    image_atomic_int64: bool,
    explicit_layout: bool,
    maximal_reconvergence: bool,
    quad_control: bool,
    uniform_control_flow: bool,
    derivatives: bool,
    barycentric: bool,
    interlock: bool,
    raster_order: bool,
    dynamic_state: bool,
    dynamic_state2: bool,
    dynamic_state3: bool,
    feedback_loop: bool,
    maintenance8: bool,
    shader_fma: bool,
    border_color_swizzle: bool,
    depth_bias_control: bool,
    depth_clip_enable: bool,
    depth_clamp_control: bool,
    list_restart: bool,
    provoking_vertex: bool,
    ray_query: bool,
    acceleration_structure: bool,
    portability_subset: bool,
    generated_commands: bool,
    mesh_shader: bool,
}

impl Chained {
    fn of(x: &Exposure) -> Chained {
        let v13 = x.core(3);
        let pre13 = |names: &[&str]| !v13 && x.ext(names);
        Chained {
            v13,
            image_robustness: pre13(EXT_IMAGE_ROBUSTNESS),
            subgroup_size_control: pre13(EXT_SUBGROUP_SIZE_CONTROL),
            demote: pre13(EXT_DEMOTE),
            terminate: pre13(EXT_TERMINATE),
            zero_init: pre13(EXT_ZERO_INIT),
            sync2: pre13(EXT_SYNC2),
            maintenance4: pre13(EXT_MAINTENANCE4),
            rotate: x.promoted(EXT_ROTATE, Some(4)),
            local_read: x.promoted(EXT_LOCAL_READ, Some(4)),
            divisor: x.promoted(EXT_DIVISOR, Some(4)),
            maintenance5: x.promoted(EXT_MAINTENANCE5, Some(4)),
            robustness2: x.ext(EXT_ROBUSTNESS2),
            atomic_float: x.ext(EXT_ATOMIC_FLOAT),
            atomic_float2: x.ext(EXT_ATOMIC_FLOAT2),
            image_atomic_int64: x.ext(EXT_IMAGE_ATOMIC_INT64),
            explicit_layout: x.ext(EXT_EXPLICIT_LAYOUT),
            maximal_reconvergence: x.ext(EXT_MAXIMAL_RECONVERGENCE),
            quad_control: x.ext(EXT_QUAD_CONTROL),
            uniform_control_flow: x.ext(EXT_UNIFORM_CONTROL_FLOW),
            derivatives: x.ext(EXT_DERIVATIVES),
            barycentric: x.ext(EXT_BARYCENTRIC),
            interlock: x.ext(EXT_INTERLOCK),
            raster_order: x.ext(EXT_RASTER_ORDER),
            // Core in 1.3 without a feature bit.
            dynamic_state: !v13 && x.ext(EXT_DYNAMIC_STATE),
            dynamic_state2: !v13 && x.ext(EXT_DYNAMIC_STATE2),
            dynamic_state3: x.ext(EXT_DYNAMIC_STATE3),
            feedback_loop: x.ext(EXT_FEEDBACK_LOOP),
            maintenance8: x.ext(EXT_MAINTENANCE8),
            shader_fma: x.ext(EXT_SHADER_FMA),
            border_color_swizzle: x.ext(EXT_BORDER_COLOR_SWIZZLE),
            depth_bias_control: x.ext(EXT_DEPTH_BIAS_CONTROL),
            depth_clip_enable: x.ext(EXT_DEPTH_CLIP_ENABLE),
            depth_clamp_control: x.ext(EXT_DEPTH_CLAMP_CONTROL),
            list_restart: x.ext(EXT_LIST_RESTART),
            provoking_vertex: x.ext(EXT_PROVOKING_VERTEX),
            ray_query: x.ext(EXT_RAY_QUERY),
            acceleration_structure: x.ext(EXT_ACCELERATION_STRUCTURE),
            portability_subset: x.ext(EXT_PORTABILITY_SUBSET),
            generated_commands: x.ext(EXT_GENERATED_COMMANDS),
            mesh_shader: x.ext(EXT_MESH_SHADER),
        }
    }
}

impl FeatureChain {
    /// Runs `vkGetPhysicalDeviceFeatures2` with the structures `chained` selects.
    ///
    /// # Safety
    /// `pd` is a physical device of `instance`, and `chained` was built from an effective version
    /// (`min(instance, device)`) of at least 1.2.
    unsafe fn query(
        instance: &ash::Instance,
        pd: avk::PhysicalDevice,
        chained: Chained,
    ) -> FeatureChain {
        let mut c = FeatureChain::default();
        let core = {
            let mut f2 = avk::PhysicalDeviceFeatures2::default()
                .push_next(&mut c.v11)
                .push_next(&mut c.v12);
            macro_rules! link {
                ($($field:ident),*) => {
                    $(if chained.$field {
                        f2 = f2.push_next(&mut c.$field);
                    })*
                };
            }
            link!(
                v13,
                image_robustness,
                subgroup_size_control,
                demote,
                terminate,
                zero_init,
                sync2,
                maintenance4,
                rotate,
                local_read,
                divisor,
                maintenance5,
                robustness2,
                atomic_float,
                atomic_float2,
                image_atomic_int64,
                explicit_layout,
                maximal_reconvergence,
                quad_control,
                uniform_control_flow,
                derivatives,
                barycentric,
                interlock,
                raster_order,
                dynamic_state,
                dynamic_state2,
                dynamic_state3,
                feedback_loop,
                maintenance8,
                shader_fma,
                border_color_swizzle,
                depth_bias_control,
                depth_clip_enable,
                depth_clamp_control,
                list_restart,
                provoking_vertex,
                ray_query,
                acceleration_structure,
                portability_subset,
                generated_commands,
                mesh_shader
            );
            // SAFETY: the caller guarantees `pd` belongs to `instance` at version ≥ 1.2; every
            // chained structure is exposed (`Chained::of`) and lives until the call returns.
            unsafe { instance.get_physical_device_features2(pd, &mut f2) };
            f2.features
        };
        c.core = core;
        c
    }
}

/// The feature table: one source per [`VkFeature`]. The generated `match` has no wildcard, so
/// adding a `VkFeature` variant without a source here is a compile error, and [`ALL_FEATURES`]
/// is generated from the same arms.
macro_rules! feature_sources {
    ($c:ident, $x:ident; $($feature:ident => $source:expr;)*) => {
        /// Every `VkFeature`, in table order.
        pub(crate) const ALL_FEATURES: &[VkFeature] = &[$(VkFeature::$feature),*];

        /// Whether the device supports `feature`, from the queried structures.
        fn supports(feature: VkFeature, $c: &FeatureChain, $x: &Exposure) -> bool {
            match feature {
                $(VkFeature::$feature => $source,)*
            }
        }
    };
}

/// A 1.3 core member, or its pre-1.3 extension structure's member.
macro_rules! v13_or {
    ($c:ident . $member:ident, $ext:ident . $ext_member:ident) => {
        on($c.v13.$member) || on($c.$ext.$ext_member)
    };
}

feature_sources! { c, x;
    RobustBufferAccess => on(c.core.robust_buffer_access);
    RobustBufferAccess2 => on(c.robustness2.robust_buffer_access2);
    RobustImageAccess => v13_or!(c.robust_image_access, image_robustness.robust_image_access);
    RobustImageAccess2 => on(c.robustness2.robust_image_access2);
    NullDescriptor => on(c.robustness2.null_descriptor);
    ShaderInt8 => on(c.v12.shader_int8);
    ShaderInt16 => on(c.core.shader_int16);
    ShaderInt64 => on(c.core.shader_int64);
    ShaderFloat16 => on(c.v12.shader_float16);
    ShaderFloat64 => on(c.core.shader_float64);
    StorageBuffer8BitAccess => on(c.v12.storage_buffer8_bit_access);
    UniformAndStorageBuffer8BitAccess => on(c.v12.uniform_and_storage_buffer8_bit_access);
    StoragePushConstant8 => on(c.v12.storage_push_constant8);
    StorageBuffer16BitAccess => on(c.v11.storage_buffer16_bit_access);
    UniformAndStorageBuffer16BitAccess => on(c.v11.uniform_and_storage_buffer16_bit_access);
    StoragePushConstant16 => on(c.v11.storage_push_constant16);
    StorageInputOutput16 => on(c.v11.storage_input_output16);
    ShaderBufferInt64Atomics => on(c.v12.shader_buffer_int64_atomics);
    ShaderSharedInt64Atomics => on(c.v12.shader_shared_int64_atomics);
    ShaderImageInt64Atomics => on(c.image_atomic_int64.shader_image_int64_atomics);
    ShaderBufferFloat32Atomics => on(c.atomic_float.shader_buffer_float32_atomics);
    ShaderBufferFloat32AtomicAdd => on(c.atomic_float.shader_buffer_float32_atomic_add);
    ShaderBufferFloat32AtomicMinMax => on(c.atomic_float2.shader_buffer_float32_atomic_min_max);
    ShaderSharedFloat32Atomics => on(c.atomic_float.shader_shared_float32_atomics);
    ShaderSharedFloat32AtomicAdd => on(c.atomic_float.shader_shared_float32_atomic_add);
    ShaderSharedFloat32AtomicMinMax => on(c.atomic_float2.shader_shared_float32_atomic_min_max);
    ShaderImageFloat32Atomics => on(c.atomic_float.shader_image_float32_atomics);
    ShaderImageFloat32AtomicAdd => on(c.atomic_float.shader_image_float32_atomic_add);
    ShaderImageFloat32AtomicMinMax => on(c.atomic_float2.shader_image_float32_atomic_min_max);
    BufferDeviceAddress => on(c.v12.buffer_device_address);
    ScalarBlockLayout => on(c.v12.scalar_block_layout);
    UniformBufferStandardLayout => on(c.v12.uniform_buffer_standard_layout);
    VulkanMemoryModel => on(c.v12.vulkan_memory_model);
    VulkanMemoryModelDeviceScope => on(c.v12.vulkan_memory_model_device_scope);
    WorkgroupMemoryExplicitLayout => on(c.explicit_layout.workgroup_memory_explicit_layout);
    WorkgroupMemoryExplicitLayout8BitAccess =>
        on(c.explicit_layout.workgroup_memory_explicit_layout8_bit_access);
    WorkgroupMemoryExplicitLayout16BitAccess =>
        on(c.explicit_layout.workgroup_memory_explicit_layout16_bit_access);
    ShaderZeroInitializeWorkgroupMemory => v13_or!(
        c.shader_zero_initialize_workgroup_memory,
        zero_init.shader_zero_initialize_workgroup_memory
    );
    VertexPipelineStoresAndAtomics => on(c.core.vertex_pipeline_stores_and_atomics);
    FragmentStoresAndAtomics => on(c.core.fragment_stores_and_atomics);
    ShaderDrawParameters => on(c.v11.shader_draw_parameters);
    GeometryShader => on(c.core.geometry_shader);
    ShaderTessellationAndGeometryPointSize =>
        on(c.core.shader_tessellation_and_geometry_point_size);
    TessellationShader => on(c.core.tessellation_shader);
    SampleRateShading => on(c.core.sample_rate_shading);
    DualSrcBlend => on(c.core.dual_src_blend);
    LogicOp => on(c.core.logic_op);
    AlphaToOne => on(c.core.alpha_to_one);
    IndependentBlend => on(c.core.independent_blend);
    ShaderClipDistance => on(c.core.shader_clip_distance);
    ShaderCullDistance => on(c.core.shader_cull_distance);
    ShaderOutputLayer => on(c.v12.shader_output_layer);
    ShaderOutputViewportIndex => on(c.v12.shader_output_viewport_index);
    Multiview => on(c.v11.multiview);
    MultiviewTessellationShader => on(c.v11.multiview_tessellation_shader);
    MultiviewGeometryShader => on(c.v11.multiview_geometry_shader);
    FragmentShaderBarycentric => on(c.barycentric.fragment_shader_barycentric);
    FragmentShaderPixelInterlock => on(c.interlock.fragment_shader_pixel_interlock);
    FragmentShaderSampleInterlock => on(c.interlock.fragment_shader_sample_interlock);
    ShaderDemoteToHelperInvocation => v13_or!(
        c.shader_demote_to_helper_invocation,
        demote.shader_demote_to_helper_invocation
    );
    ShaderTerminateInvocation => v13_or!(
        c.shader_terminate_invocation,
        terminate.shader_terminate_invocation
    );
    DynamicRenderingLocalRead => on(c.local_read.dynamic_rendering_local_read);
    RasterizationOrderColorAttachmentAccess =>
        on(c.raster_order.rasterization_order_color_attachment_access);
    RasterizationOrderDepthAttachmentAccess =>
        on(c.raster_order.rasterization_order_depth_attachment_access);
    RasterizationOrderStencilAttachmentAccess =>
        on(c.raster_order.rasterization_order_stencil_attachment_access);
    VertexAttributeInstanceRateDivisor => on(c.divisor.vertex_attribute_instance_rate_divisor);
    VertexAttributeInstanceRateZeroDivisor =>
        on(c.divisor.vertex_attribute_instance_rate_zero_divisor);
    DrawIndirectFirstInstance => on(c.core.draw_indirect_first_instance);
    MultiDrawIndirect => on(c.core.multi_draw_indirect);
    DepthClamp => on(c.core.depth_clamp);
    DepthBiasClamp => on(c.core.depth_bias_clamp);
    FillModeNonSolid => on(c.core.fill_mode_non_solid);
    ImageCubeArray => on(c.core.image_cube_array);
    ShaderImageGatherExtended => on(c.core.shader_image_gather_extended);
    ShaderResourceMinLod => on(c.core.shader_resource_min_lod);
    ShaderResourceResidency => on(c.core.shader_resource_residency);
    SamplerAnisotropy => on(c.core.sampler_anisotropy);
    SamplerFilterMinmax => on(c.v12.sampler_filter_minmax);
    DrawIndirectCount => on(c.v12.draw_indirect_count);
    SamplerMirrorClampToEdge => on(c.v12.sampler_mirror_clamp_to_edge);
    ShaderStorageImageMultisample => on(c.core.shader_storage_image_multisample);
    ShaderStorageImageExtendedFormats => on(c.core.shader_storage_image_extended_formats);
    ShaderStorageImageReadWithoutFormat => on(c.core.shader_storage_image_read_without_format);
    ShaderStorageImageWriteWithoutFormat => on(c.core.shader_storage_image_write_without_format);
    OcclusionQueryPrecise => on(c.core.occlusion_query_precise);
    FullDrawIndexUint32 => on(c.core.full_draw_index_uint32);
    RuntimeDescriptorArray => on(c.v12.runtime_descriptor_array);
    DescriptorBindingPartiallyBound => on(c.v12.descriptor_binding_partially_bound);
    DescriptorBindingVariableDescriptorCount =>
        on(c.v12.descriptor_binding_variable_descriptor_count);
    DescriptorBindingSampledImageUpdateAfterBind =>
        on(c.v12.descriptor_binding_sampled_image_update_after_bind);
    DescriptorBindingStorageImageUpdateAfterBind =>
        on(c.v12.descriptor_binding_storage_image_update_after_bind);
    DescriptorBindingStorageBufferUpdateAfterBind =>
        on(c.v12.descriptor_binding_storage_buffer_update_after_bind);
    DescriptorBindingUniformTexelBufferUpdateAfterBind =>
        on(c.v12.descriptor_binding_uniform_texel_buffer_update_after_bind);
    DescriptorBindingStorageTexelBufferUpdateAfterBind =>
        on(c.v12.descriptor_binding_storage_texel_buffer_update_after_bind);
    DescriptorBindingUpdateUnusedWhilePending =>
        on(c.v12.descriptor_binding_update_unused_while_pending);
    ShaderSampledImageArrayNonUniformIndexing =>
        on(c.v12.shader_sampled_image_array_non_uniform_indexing);
    ShaderStorageImageArrayNonUniformIndexing =>
        on(c.v12.shader_storage_image_array_non_uniform_indexing);
    ShaderStorageBufferArrayNonUniformIndexing =>
        on(c.v12.shader_storage_buffer_array_non_uniform_indexing);
    ShaderUniformTexelBufferArrayNonUniformIndexing =>
        on(c.v12.shader_uniform_texel_buffer_array_non_uniform_indexing);
    ShaderStorageTexelBufferArrayNonUniformIndexing =>
        on(c.v12.shader_storage_texel_buffer_array_non_uniform_indexing);
    SubgroupSizeControl =>
        v13_or!(c.subgroup_size_control, subgroup_size_control.subgroup_size_control);
    ComputeFullSubgroups =>
        v13_or!(c.compute_full_subgroups, subgroup_size_control.compute_full_subgroups);
    ShaderSubgroupExtendedTypes => on(c.v12.shader_subgroup_extended_types);
    ShaderMaximalReconvergence => on(c.maximal_reconvergence.shader_maximal_reconvergence);
    ShaderQuadControl => on(c.quad_control.shader_quad_control);
    ShaderSubgroupUniformControlFlow =>
        on(c.uniform_control_flow.shader_subgroup_uniform_control_flow);
    ComputeDerivativeGroupQuads => on(c.derivatives.compute_derivative_group_quads);
    ComputeDerivativeGroupLinear => on(c.derivatives.compute_derivative_group_linear);
    Maintenance4 => v13_or!(c.maintenance4, maintenance4.maintenance4);
    Synchronization2 => v13_or!(c.synchronization2, sync2.synchronization2);
    // Core in 1.3 without a feature bit; the extension's own bit below 1.3.
    ExtendedDynamicState => x.core(3) || on(c.dynamic_state.extended_dynamic_state);
    ExtendedDynamicState2 => x.core(3) || on(c.dynamic_state2.extended_dynamic_state2);
    ExtendedDynamicState3ColorWriteMask =>
        on(c.dynamic_state3.extended_dynamic_state3_color_write_mask);
    AttachmentFeedbackLoopLayout => on(c.feedback_loop.attachment_feedback_loop_layout);
    ShaderUniformBufferArrayDynamicIndexing =>
        on(c.core.shader_uniform_buffer_array_dynamic_indexing);
    ShaderSampledImageArrayDynamicIndexing =>
        on(c.core.shader_sampled_image_array_dynamic_indexing);
    ShaderStorageBufferArrayDynamicIndexing =>
        on(c.core.shader_storage_buffer_array_dynamic_indexing);
    ShaderStorageImageArrayDynamicIndexing =>
        on(c.core.shader_storage_image_array_dynamic_indexing);
    ShaderUniformTexelBufferArrayDynamicIndexing =>
        on(c.v12.shader_uniform_texel_buffer_array_dynamic_indexing);
    ShaderStorageTexelBufferArrayDynamicIndexing =>
        on(c.v12.shader_storage_texel_buffer_array_dynamic_indexing);
    ShaderInputAttachmentArrayDynamicIndexing =>
        on(c.v12.shader_input_attachment_array_dynamic_indexing);
    ShaderBufferFloat16Atomics => on(c.atomic_float2.shader_buffer_float16_atomics);
    ShaderBufferFloat16AtomicAdd => on(c.atomic_float2.shader_buffer_float16_atomic_add);
    ShaderBufferFloat16AtomicMinMax => on(c.atomic_float2.shader_buffer_float16_atomic_min_max);
    ShaderSharedFloat16Atomics => on(c.atomic_float2.shader_shared_float16_atomics);
    ShaderSharedFloat16AtomicAdd => on(c.atomic_float2.shader_shared_float16_atomic_add);
    ShaderSharedFloat16AtomicMinMax => on(c.atomic_float2.shader_shared_float16_atomic_min_max);
    ShaderBufferFloat64Atomics => on(c.atomic_float.shader_buffer_float64_atomics);
    ShaderBufferFloat64AtomicAdd => on(c.atomic_float.shader_buffer_float64_atomic_add);
    ShaderBufferFloat64AtomicMinMax => on(c.atomic_float2.shader_buffer_float64_atomic_min_max);
    ShaderSharedFloat64Atomics => on(c.atomic_float.shader_shared_float64_atomics);
    ShaderSharedFloat64AtomicAdd => on(c.atomic_float.shader_shared_float64_atomic_add);
    ShaderSharedFloat64AtomicMinMax => on(c.atomic_float2.shader_shared_float64_atomic_min_max);
    WorkgroupMemoryExplicitLayoutScalarBlockLayout =>
        on(c.explicit_layout.workgroup_memory_explicit_layout_scalar_block_layout);
    ShaderSubgroupRotate => on(c.rotate.shader_subgroup_rotate);
    ShaderSubgroupRotateClustered => on(c.rotate.shader_subgroup_rotate_clustered);
    MultiViewport => on(c.core.multi_viewport);
    Maintenance8 => on(c.maintenance8.maintenance8);
    BorderColorSwizzle => on(c.border_color_swizzle.border_color_swizzle);
    BorderColorSwizzleFromImage => on(c.border_color_swizzle.border_color_swizzle_from_image);
    DepthBiasControl => on(c.depth_bias_control.depth_bias_control);
    FloatRepresentation => on(c.depth_bias_control.float_representation);
    DepthBiasExact => on(c.depth_bias_control.depth_bias_exact);
    DepthClipEnable => on(c.depth_clip_enable.depth_clip_enable);
    DepthClampControl => on(c.depth_clamp_control.depth_clamp_control);
    PrimitiveTopologyListRestart => on(c.list_restart.primitive_topology_list_restart);
    PrimitiveTopologyPatchListRestart =>
        on(c.list_restart.primitive_topology_patch_list_restart);
    ProvokingVertexLast => on(c.provoking_vertex.provoking_vertex_last);
    RayQuery => on(c.ray_query.ray_query);
    AccelerationStructure => on(c.acceleration_structure.acceleration_structure);
    ConstantAlphaColorBlendFactors =>
        on(c.portability_subset.constant_alpha_color_blend_factors);
    VertexAttributeAccessBeyondStride =>
        on(c.portability_subset.vertex_attribute_access_beyond_stride);
    SamplerMipLodBias => on(c.portability_subset.sampler_mip_lod_bias);
    MutableComparisonSamplers => on(c.portability_subset.mutable_comparison_samplers);
    ShaderSampleRateInterpolationFunctions =>
        on(c.portability_subset.shader_sample_rate_interpolation_functions);
    MultisampleArrayImage => on(c.portability_subset.multisample_array_image);
    ImageView2DOn3DImage => on(c.portability_subset.image_view2_d_on3_d_image);
    SeparateStencilMaskRef => on(c.portability_subset.separate_stencil_mask_ref);
    ImageViewFormatSwizzle => on(c.portability_subset.image_view_format_swizzle);
    ShaderFmaFloat16 => on(c.shader_fma.shader_fma_float16);
    ShaderFmaFloat32 => on(c.shader_fma.shader_fma_float32);
    Maintenance5 => on(c.maintenance5.maintenance5);
    TaskShader => on(c.mesh_shader.task_shader);
    MeshShader => on(c.mesh_shader.mesh_shader);
}

/// The property structures read.
#[derive(Default)]
struct PropertyChain {
    core: avk::PhysicalDeviceProperties,
    v11: avk::PhysicalDeviceVulkan11Properties<'static>,
    v12: avk::PhysicalDeviceVulkan12Properties<'static>,
    v13: avk::PhysicalDeviceVulkan13Properties<'static>,
    subgroup_size_control: avk::PhysicalDeviceSubgroupSizeControlProperties<'static>,
    maintenance4: avk::PhysicalDeviceMaintenance4Properties<'static>,
    push_descriptor: avk::PhysicalDevicePushDescriptorPropertiesKHR<'static>,
    divisor_khr: avk::PhysicalDeviceVertexAttributeDivisorPropertiesKHR<'static>,
    divisor_ext: avk::PhysicalDeviceVertexAttributeDivisorPropertiesEXT<'static>,
    portability: avk::PhysicalDevicePortabilitySubsetPropertiesKHR<'static>,
    maintenance10: Maintenance10Properties,
    acceleration_structure: avk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    generated_commands: GeneratedCommandsProperties,
    sample_locations: avk::PhysicalDeviceSampleLocationsPropertiesEXT<'static>,
    mesh_shader: avk::PhysicalDeviceMeshShaderPropertiesEXT<'static>,
    provoking_vertex: avk::PhysicalDeviceProvokingVertexPropertiesEXT<'static>,
}

/// Which property structures are chained.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PropertiesChained {
    v13: bool,
    subgroup_size_control: bool,
    maintenance4: bool,
    push_descriptor: bool,
    divisor_khr: bool,
    divisor_ext: bool,
    portability: bool,
    maintenance10: bool,
    acceleration_structure: bool,
    generated_commands: bool,
    sample_locations: bool,
    mesh_shader: bool,
    provoking_vertex: bool,
}

impl PropertiesChained {
    fn of(x: &Exposure) -> PropertiesChained {
        let v13 = x.core(3);
        let divisor_khr = x.promoted(&["VK_KHR_vertex_attribute_divisor"], Some(4));
        PropertiesChained {
            v13,
            subgroup_size_control: !v13 && x.ext(EXT_SUBGROUP_SIZE_CONTROL),
            maintenance4: !v13 && x.ext(EXT_MAINTENANCE4),
            push_descriptor: x.promoted(&["VK_KHR_push_descriptor"], Some(4)),
            divisor_khr,
            divisor_ext: !divisor_khr && x.ext(&["VK_EXT_vertex_attribute_divisor"]),
            portability: x.ext(&["VK_KHR_portability_subset"]),
            maintenance10: x.ext(&["VK_KHR_maintenance10"]),
            acceleration_structure: x.ext(EXT_ACCELERATION_STRUCTURE),
            generated_commands: x.ext(EXT_GENERATED_COMMANDS),
            sample_locations: x.ext(EXT_SAMPLE_LOCATIONS),
            mesh_shader: x.ext(EXT_MESH_SHADER),
            provoking_vertex: x.ext(EXT_PROVOKING_VERTEX),
        }
    }
}

impl PropertyChain {
    /// Runs `vkGetPhysicalDeviceProperties2`.
    ///
    /// # Safety
    /// As [`FeatureChain::query`].
    unsafe fn query(
        instance: &ash::Instance,
        pd: avk::PhysicalDevice,
        chained: PropertiesChained,
    ) -> PropertyChain {
        let mut c = PropertyChain::default();
        let core = {
            let mut p2 = avk::PhysicalDeviceProperties2::default()
                .push_next(&mut c.v11)
                .push_next(&mut c.v12);
            macro_rules! link {
                ($($field:ident),*) => {
                    $(if chained.$field {
                        p2 = p2.push_next(&mut c.$field);
                    })*
                };
            }
            link!(
                v13,
                subgroup_size_control,
                maintenance4,
                push_descriptor,
                divisor_khr,
                divisor_ext,
                portability,
                maintenance10,
                acceleration_structure,
                generated_commands,
                sample_locations,
                mesh_shader,
                provoking_vertex
            );
            // SAFETY: as in `FeatureChain::query`.
            unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
            p2.properties
        };
        c.core = core;
        c
    }
}

impl DeviceDescription {
    /// Reads one physical device: its version, identity, available extensions and features, the
    /// limits and properties the core consumes, and the format properties of exactly
    /// [`FormatSupport::queries_for`] its version and available extensions.
    ///
    /// The result describes what the device **supports**; remove what the consumer does not
    /// enable before `Device::new`. `instance_api_version` is the `VkApplicationInfo::apiVersion`
    /// the instance was created with (e.g. `vk::ApiVersion::from_packed(app_info.api_version)`):
    /// the Vulkan spec limits the device's core functionality to `min(instance, device)` version,
    /// so the description's `api_version`, the structures chained, and every feature sourced from
    /// a core version follow that minimum, never the device's own version alone. Only the
    /// physical-device query entry points are called; nothing is created and no device is opened.
    ///
    /// Errors: an effective version below Vulkan 1.2 is `Unsupported{VulkanVersion}` (nothing
    /// beyond `vkGetPhysicalDeviceProperties` is called then); a failed extension enumeration is
    /// `InvalidDescriptor(DeviceDescription)`.
    ///
    /// # Safety
    /// `physical_device` is a physical device enumerated from `instance`, and
    /// `instance_api_version` is at most the `apiVersion` `instance` was created with (a lower
    /// value is valid and only narrows the description; a higher one makes the chained core
    /// structures invalid usage).
    pub unsafe fn query(
        instance: &ash::Instance,
        physical_device: avk::PhysicalDevice,
        instance_api_version: ApiVersion,
    ) -> Result<DeviceDescription, Error> {
        // SAFETY: the caller's contract (a Vulkan 1.0 entry point).
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let api = vk::ApiVersion::from_packed(props.api_version).min(instance_api_version);
        if api < ApiVersion::V1_2 {
            return Err(Unsupported::needs(
                Feature::VulkanVersion,
                Capability::ApiVersion(ApiVersion::V1_2),
            )
            .into());
        }
        // SAFETY: the caller's contract.
        let listed = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|e| {
                Error::InvalidDescriptor(DescriptorError::DeviceDescription {
                    detail: format!("vkEnumerateDeviceExtensionProperties failed: {e}"),
                })
            })?;
        let extensions: BTreeSet<String> = listed
            .iter()
            .filter_map(|e| e.extension_name_as_c_str().ok())
            .filter_map(|c| c.to_str().ok())
            .map(str::to_owned)
            .collect();
        let exposure = Exposure { api, extensions };

        // SAFETY: the caller's contract; the chains follow `Chained::of`.
        let features =
            unsafe { FeatureChain::query(instance, physical_device, Chained::of(&exposure)) };
        // SAFETY: as above.
        let properties = unsafe {
            PropertyChain::query(instance, physical_device, PropertiesChained::of(&exposure))
        };

        let mut desc = DeviceDescription::new(api);
        desc.driver_version = props.driver_version;
        desc.vendor_id = props.vendor_id;
        desc.device_id = props.device_id;
        desc.device_name = props
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        desc.device_type = Some(props.device_type.into());
        desc.driver_id = Some(properties.v12.driver_id.into());
        desc.extensions = exposure
            .extensions
            .iter()
            .filter_map(|n| extension_from_name(n))
            .collect();
        desc.features = ALL_FEATURES
            .iter()
            .copied()
            .filter(|&f| supports(f, &features, &exposure))
            .collect();
        desc.limits = limits(
            &props.limits,
            &properties,
            &PropertiesChained::of(&exposure),
        );
        desc.allocation = allocation(&properties, &PropertiesChained::of(&exposure));
        desc.subgroup = subgroup(&properties, &PropertiesChained::of(&exposure));
        desc.float_controls = float_controls(&properties.v12);
        desc.filter_minmax_single_component_formats =
            on(properties.v12.filter_minmax_single_component_formats);
        desc.rgba4_opaque_black_swizzled = on(properties.maintenance10.rgba4_opaque_black_swizzled);
        // `VK_EXT_device_generated_commands` where the device supports its feature, with its
        // limits and stages.
        desc.generated_commands =
            on(features.generated_commands.device_generated_commands).then(|| {
                let g = &properties.generated_commands;
                DgcProperties::new(
                    [
                        g.max_indirect_pipeline_count,
                        g.max_indirect_sequence_count,
                        g.max_indirect_commands_token_count,
                        g.max_indirect_commands_token_offset,
                        g.max_indirect_commands_indirect_stride,
                    ],
                    vk::ShaderStageFlags::from_raw(g.supported_stages.as_raw()),
                    vk::ShaderStageFlags::from_raw(g.supported_stages_pipeline_binding.as_raw()),
                )
            });
        desc.robust_buffer_access_update_after_bind =
            on(properties.v12.robust_buffer_access_update_after_bind);
        // No query reports where the driver departs from the specifications; the driver and
        // its release decide it, measured (`device::driver_facts`). `VkFormatProperties3` where the
        // effective version or an available extension has it, whose storage-without-format bits
        // decide per format.
        desc.khr_format_feature_flags2 =
            exposure.extensions.contains("VK_KHR_format_feature_flags2");
        // A driver that reports some linear-tiling features from memory it has not initialized
        // (`DriverFact::LinearTilingFeaturesUninitialized`) is read without them.
        let unreliable = desc.unreliable_linear_tiling();
        for format in FormatSupport::queries_for(api, &desc.extensions) {
            let mut p = if desc.has_format_feature_flags2() {
                let mut p3 = avk::FormatProperties3::default();
                let mut p2 = avk::FormatProperties2::default().push_next(&mut p3);
                // SAFETY: the caller's contract; the device's version or one of its available
                // extensions defines every queried format, and `VkFormatProperties3`
                // (`vkGetPhysicalDeviceFormatProperties2` is Vulkan 1.1, below the floor).
                unsafe {
                    instance.get_physical_device_format_properties2(
                        physical_device,
                        format.into(),
                        &mut p2,
                    )
                };
                FormatProperties {
                    linear_tiling: vk::FormatFeatureFlags2::from_raw(
                        p3.linear_tiling_features.as_raw(),
                    ),
                    optimal_tiling: vk::FormatFeatureFlags2::from_raw(
                        p3.optimal_tiling_features.as_raw(),
                    ),
                    buffer: vk::FormatFeatureFlags2::from_raw(p3.buffer_features.as_raw()),
                    color_attachment_sample_counts: vk::SampleCountFlags::empty(),
                }
            } else {
                // SAFETY: as above.
                let p = unsafe {
                    instance.get_physical_device_format_properties(physical_device, format.into())
                };
                FormatProperties::from_legacy(
                    vk::FormatFeatureFlags::from_raw(p.linear_tiling_features.as_raw()),
                    vk::FormatFeatureFlags::from_raw(p.optimal_tiling_features.as_raw()),
                    vk::FormatFeatureFlags::from_raw(p.buffer_features.as_raw()),
                )
            };
            p.linear_tiling = p.linear_tiling.difference(unreliable);
            // SAFETY: as above.
            p.color_attachment_sample_counts =
                unsafe { color_attachment_sample_counts(instance, physical_device, format, &p) }?;
            if p != FormatProperties::default() {
                desc.formats.formats.insert(format, p);
            }
        }
        Ok(desc)
    }
}

/// The device's allocation sizes: `maxMemoryAllocationSize`, and `maxBufferSize` where Vulkan 1.3
/// or `VK_KHR_maintenance4` reports it (0 otherwise).
fn allocation(p: &PropertyChain, chained: &PropertiesChained) -> AllocationLimits {
    AllocationLimits::new(
        p.v11.max_memory_allocation_size,
        if chained.v13 {
            p.v13.max_buffer_size
        } else if chained.maintenance4 {
            p.maintenance4.max_buffer_size
        } else {
            0
        },
    )
}

/// The sample counts of a color attachment of `format` with properties `p`
/// ([`FormatProperties::color_attachment_sample_counts`]): those of a 2D, optimal-tiling image with
/// `COLOR_ATTACHMENT` usage and no flags where `p`'s optimal tiling has `COLOR_ATTACHMENT`, and
/// none otherwise or where the device takes no such image.
///
/// # Safety
/// `physical_device` belongs to `instance`, whose version or available extensions define
/// `format`.
unsafe fn color_attachment_sample_counts(
    instance: &ash::Instance,
    physical_device: avk::PhysicalDevice,
    format: vk::Format,
    p: &FormatProperties,
) -> Result<vk::SampleCountFlags, Error> {
    if !p
        .optimal_tiling
        .contains(vk::FormatFeatureFlags2::COLOR_ATTACHMENT)
    {
        return Ok(vk::SampleCountFlags::empty());
    }
    let info = avk::PhysicalDeviceImageFormatInfo2::default()
        .format(format.into())
        .ty(avk::ImageType::TYPE_2D)
        .tiling(avk::ImageTiling::OPTIMAL)
        .usage(avk::ImageUsageFlags::COLOR_ATTACHMENT);
    let mut properties = avk::ImageFormatProperties2::default();
    // SAFETY: the caller's contract (`vkGetPhysicalDeviceImageFormatProperties2` is Vulkan 1.1,
    // below the floor).
    match unsafe {
        instance.get_physical_device_image_format_properties2(
            physical_device,
            &info,
            &mut properties,
        )
    } {
        Ok(()) => Ok(vk::SampleCountFlags::from_raw(
            properties.image_format_properties.sample_counts.as_raw(),
        )),
        Err(avk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(vk::SampleCountFlags::empty()),
        Err(e) => Err(Error::InvalidDescriptor(
            DescriptorError::DeviceDescription {
                detail: format!(
                    "vkGetPhysicalDeviceImageFormatProperties2 of a {format:?} color \
                     attachment failed: {e}"
                ),
            },
        )),
    }
}

fn limits(l: &avk::PhysicalDeviceLimits, p: &PropertyChain, chained: &PropertiesChained) -> Limits {
    let v12 = &p.v12;
    Limits {
        max_compute_work_group_size: l.max_compute_work_group_size,
        max_compute_work_group_count: l.max_compute_work_group_count,
        max_compute_work_group_invocations: l.max_compute_work_group_invocations,
        max_compute_shared_memory_size: l.max_compute_shared_memory_size,
        max_push_constants_size: l.max_push_constants_size,
        max_push_descriptors: if chained.push_descriptor {
            p.push_descriptor.max_push_descriptors
        } else {
            0
        },
        max_bound_descriptor_sets: l.max_bound_descriptor_sets,
        max_per_stage_descriptor_samplers: l.max_per_stage_descriptor_samplers,
        max_per_stage_descriptor_uniform_buffers: l.max_per_stage_descriptor_uniform_buffers,
        max_per_stage_descriptor_storage_buffers: l.max_per_stage_descriptor_storage_buffers,
        max_per_stage_descriptor_sampled_images: l.max_per_stage_descriptor_sampled_images,
        max_per_stage_descriptor_storage_images: l.max_per_stage_descriptor_storage_images,
        max_per_stage_descriptor_input_attachments: l.max_per_stage_descriptor_input_attachments,
        max_per_stage_resources: l.max_per_stage_resources,
        max_descriptor_set_samplers: l.max_descriptor_set_samplers,
        max_descriptor_set_uniform_buffers: l.max_descriptor_set_uniform_buffers,
        max_descriptor_set_storage_buffers: l.max_descriptor_set_storage_buffers,
        max_descriptor_set_sampled_images: l.max_descriptor_set_sampled_images,
        max_descriptor_set_storage_images: l.max_descriptor_set_storage_images,
        max_descriptor_set_input_attachments: l.max_descriptor_set_input_attachments,
        max_descriptor_set_update_after_bind_samplers: v12
            .max_descriptor_set_update_after_bind_samplers,
        max_descriptor_set_update_after_bind_sampled_images: v12
            .max_descriptor_set_update_after_bind_sampled_images,
        max_descriptor_set_update_after_bind_storage_images: v12
            .max_descriptor_set_update_after_bind_storage_images,
        max_descriptor_set_update_after_bind_storage_buffers: v12
            .max_descriptor_set_update_after_bind_storage_buffers,
        max_per_stage_descriptor_update_after_bind_samplers: v12
            .max_per_stage_descriptor_update_after_bind_samplers,
        max_per_stage_descriptor_update_after_bind_sampled_images: v12
            .max_per_stage_descriptor_update_after_bind_sampled_images,
        max_per_stage_descriptor_update_after_bind_storage_images: v12
            .max_per_stage_descriptor_update_after_bind_storage_images,
        max_per_stage_descriptor_update_after_bind_storage_buffers: v12
            .max_per_stage_descriptor_update_after_bind_storage_buffers,
        max_per_stage_update_after_bind_resources: v12.max_per_stage_update_after_bind_resources,
        max_storage_buffer_range: l.max_storage_buffer_range,
        max_uniform_buffer_range: l.max_uniform_buffer_range,
        min_storage_buffer_offset_alignment: l.min_storage_buffer_offset_alignment,
        min_uniform_buffer_offset_alignment: l.min_uniform_buffer_offset_alignment,
        min_texel_buffer_offset_alignment: l.min_texel_buffer_offset_alignment,
        max_texel_buffer_elements: l.max_texel_buffer_elements,
        max_sampler_lod_bias_bits: l.max_sampler_lod_bias.to_bits(),
        max_sampler_anisotropy_bits: l.max_sampler_anisotropy.to_bits(),
        min_texel_offset: l.min_texel_offset,
        max_texel_offset: l.max_texel_offset,
        min_texel_gather_offset: l.min_texel_gather_offset,
        max_texel_gather_offset: l.max_texel_gather_offset,
        min_interpolation_offset_bits: l.min_interpolation_offset.to_bits(),
        max_interpolation_offset_bits: l.max_interpolation_offset.to_bits(),
        max_image_dimension_1d: l.max_image_dimension1_d,
        max_image_dimension_2d: l.max_image_dimension2_d,
        max_image_dimension_3d: l.max_image_dimension3_d,
        max_image_dimension_cube: l.max_image_dimension_cube,
        max_image_array_layers: l.max_image_array_layers,
        max_vertex_input_attributes: l.max_vertex_input_attributes,
        max_vertex_input_bindings: l.max_vertex_input_bindings,
        max_vertex_input_attribute_offset: l.max_vertex_input_attribute_offset,
        max_vertex_input_binding_stride: l.max_vertex_input_binding_stride,
        max_vertex_output_components: l.max_vertex_output_components,
        max_tessellation_generation_level: l.max_tessellation_generation_level,
        max_tessellation_evaluation_output_components: l
            .max_tessellation_evaluation_output_components,
        max_fragment_input_components: l.max_fragment_input_components,
        max_fragment_output_attachments: l.max_fragment_output_attachments,
        max_fragment_dual_src_attachments: l.max_fragment_dual_src_attachments,
        max_fragment_combined_output_resources: l.max_fragment_combined_output_resources,
        max_color_attachments: l.max_color_attachments,
        max_clip_distances: l.max_clip_distances,
        max_cull_distances: l.max_cull_distances,
        max_combined_clip_and_cull_distances: l.max_combined_clip_and_cull_distances,
        max_vertex_attrib_divisor: if chained.divisor_khr {
            p.divisor_khr.max_vertex_attrib_divisor
        } else if chained.divisor_ext {
            p.divisor_ext.max_vertex_attrib_divisor
        } else {
            0
        },
        max_per_stage_descriptor_acceleration_structures: if chained.acceleration_structure {
            p.acceleration_structure
                .max_per_stage_descriptor_acceleration_structures
        } else {
            0
        },
        max_descriptor_set_acceleration_structures: if chained.acceleration_structure {
            p.acceleration_structure
                .max_descriptor_set_acceleration_structures
        } else {
            0
        },
        // The KHR (1.4) property; an EXT-only divisor cannot report it and reads `false`.
        supports_non_zero_first_instance: chained.divisor_khr
            && on(p.divisor_khr.supports_non_zero_first_instance),
        // The portability subset's stride rule (exposed means enabled).
        min_vertex_input_binding_stride_alignment: chained
            .portability
            .then_some(p.portability.min_vertex_input_binding_stride_alignment),
        framebuffer_color_sample_counts: vk::SampleCountFlags::from_raw(
            l.framebuffer_color_sample_counts.as_raw(),
        ),
        framebuffer_depth_sample_counts: vk::SampleCountFlags::from_raw(
            l.framebuffer_depth_sample_counts.as_raw(),
        ),
        max_multiview_view_count: p.v11.max_multiview_view_count,
        max_multiview_instance_index: p.v11.max_multiview_instance_index,
        framebuffer_integer_color_sample_counts: vk::SampleCountFlags::from_raw(
            p.v12.framebuffer_integer_color_sample_counts.as_raw(),
        ),
        framebuffer_stencil_sample_counts: vk::SampleCountFlags::from_raw(
            l.framebuffer_stencil_sample_counts.as_raw(),
        ),
        framebuffer_no_attachments_sample_counts: vk::SampleCountFlags::from_raw(
            l.framebuffer_no_attachments_sample_counts.as_raw(),
        ),
        storage_image_sample_counts: vk::SampleCountFlags::from_raw(
            l.storage_image_sample_counts.as_raw(),
        ),
        max_framebuffer_width: l.max_framebuffer_width,
        max_framebuffer_height: l.max_framebuffer_height,
        max_framebuffer_layers: l.max_framebuffer_layers,
        max_draw_indexed_index_value: l.max_draw_indexed_index_value,
        max_tessellation_patch_size: l.max_tessellation_patch_size,
        max_geometry_input_components: l.max_geometry_input_components,
        max_geometry_output_components: l.max_geometry_output_components,
        max_geometry_total_output_components: l.max_geometry_total_output_components,
        supported_depth_resolve_modes: vk::ResolveModeFlags::from_raw(
            v12.supported_depth_resolve_modes.as_raw(),
        ),
        supported_stencil_resolve_modes: vk::ResolveModeFlags::from_raw(
            v12.supported_stencil_resolve_modes.as_raw(),
        ),
        independent_resolve_none: on(v12.independent_resolve_none),
        independent_resolve: on(v12.independent_resolve),
        // Zero where `VK_EXT_sample_locations` is not listed (the structure is not chained).
        sample_location_sample_counts: vk::SampleCountFlags::from_raw(
            p.sample_locations.sample_location_sample_counts.as_raw(),
        ),
        variable_sample_locations: on(p.sample_locations.variable_sample_locations),
        // `false` where `VK_EXT_provoking_vertex` is not listed (the structure is not chained).
        provoking_vertex_mode_per_pipeline: on(p
            .provoking_vertex
            .provoking_vertex_mode_per_pipeline),
        // Zero where `VK_EXT_mesh_shader` is not listed (the structure is not chained).
        mesh: mesh_limits(&p.mesh_shader),
        max_point_size: crate::device::whole_point_size(l.point_size_range[1]),
    }
}

/// `VkPhysicalDeviceMeshShaderPropertiesEXT`'s limits.
fn mesh_limits(m: &avk::PhysicalDeviceMeshShaderPropertiesEXT<'_>) -> MeshLimits {
    MeshLimits {
        max_task_work_group_total_count: m.max_task_work_group_total_count,
        max_task_work_group_count: m.max_task_work_group_count,
        max_task_work_group_invocations: m.max_task_work_group_invocations,
        max_task_work_group_size: m.max_task_work_group_size,
        max_task_payload_size: m.max_task_payload_size,
        max_task_shared_memory_size: m.max_task_shared_memory_size,
        max_task_payload_and_shared_memory_size: m.max_task_payload_and_shared_memory_size,
        max_mesh_work_group_total_count: m.max_mesh_work_group_total_count,
        max_mesh_work_group_count: m.max_mesh_work_group_count,
        max_mesh_work_group_invocations: m.max_mesh_work_group_invocations,
        max_mesh_work_group_size: m.max_mesh_work_group_size,
        max_mesh_shared_memory_size: m.max_mesh_shared_memory_size,
        max_mesh_payload_and_shared_memory_size: m.max_mesh_payload_and_shared_memory_size,
        max_mesh_output_memory_size: m.max_mesh_output_memory_size,
        max_mesh_payload_and_output_memory_size: m.max_mesh_payload_and_output_memory_size,
        max_mesh_output_components: m.max_mesh_output_components,
        max_mesh_output_vertices: m.max_mesh_output_vertices,
        max_mesh_output_primitives: m.max_mesh_output_primitives,
        max_mesh_output_layers: m.max_mesh_output_layers,
        max_mesh_multiview_view_count: m.max_mesh_multiview_view_count,
    }
}

fn subgroup(p: &PropertyChain, chained: &PropertiesChained) -> SubgroupProperties {
    let v11 = &p.v11;
    // Subgroup size control properties: 1.3 core, or the extension structure below 1.3.
    let (min_size, max_size, max_wg_subgroups, required) = if chained.v13 {
        let v = &p.v13;
        (
            v.min_subgroup_size,
            v.max_subgroup_size,
            v.max_compute_workgroup_subgroups,
            v.required_subgroup_size_stages,
        )
    } else if chained.subgroup_size_control {
        let v = &p.subgroup_size_control;
        (
            v.min_subgroup_size,
            v.max_subgroup_size,
            v.max_compute_workgroup_subgroups,
            v.required_subgroup_size_stages,
        )
    } else {
        (0, 0, 0, avk::ShaderStageFlags::empty())
    };
    SubgroupProperties {
        size: v11.subgroup_size,
        supported_stages: v11.subgroup_supported_stages.into(),
        supported_operations: v11.subgroup_supported_operations.into(),
        quad_operations_in_all_stages: on(v11.subgroup_quad_operations_in_all_stages),
        min_size,
        max_size,
        max_compute_workgroup_subgroups: max_wg_subgroups,
        required_size_stages: required.into(),
    }
}

fn float_controls(v12: &avk::PhysicalDeviceVulkan12Properties<'_>) -> FloatControls {
    let independence = |i: avk::ShaderFloatControlsIndependence| match i {
        avk::ShaderFloatControlsIndependence::ALL => Independence::All,
        avk::ShaderFloatControlsIndependence::TYPE_32_ONLY => Independence::Only32Bit,
        // `NONE` or a value this version does not know: the most constrained contract.
        _ => Independence::None,
    };
    FloatControls {
        f16: FloatWidthControls {
            denorm_preserve: on(v12.shader_denorm_preserve_float16),
            denorm_flush_to_zero: on(v12.shader_denorm_flush_to_zero_float16),
            rounding_rte: on(v12.shader_rounding_mode_rte_float16),
            rounding_rtz: on(v12.shader_rounding_mode_rtz_float16),
            signed_zero_inf_nan_preserve: on(v12.shader_signed_zero_inf_nan_preserve_float16),
        },
        f32: FloatWidthControls {
            denorm_preserve: on(v12.shader_denorm_preserve_float32),
            denorm_flush_to_zero: on(v12.shader_denorm_flush_to_zero_float32),
            rounding_rte: on(v12.shader_rounding_mode_rte_float32),
            rounding_rtz: on(v12.shader_rounding_mode_rtz_float32),
            signed_zero_inf_nan_preserve: on(v12.shader_signed_zero_inf_nan_preserve_float32),
        },
        f64: FloatWidthControls {
            denorm_preserve: on(v12.shader_denorm_preserve_float64),
            denorm_flush_to_zero: on(v12.shader_denorm_flush_to_zero_float64),
            rounding_rte: on(v12.shader_rounding_mode_rte_float64),
            rounding_rtz: on(v12.shader_rounding_mode_rtz_float64),
            signed_zero_inf_nan_preserve: on(v12.shader_signed_zero_inf_nan_preserve_float64),
        },
        denorm_independence: independence(v12.denorm_behavior_independence),
        rounding_independence: independence(v12.rounding_mode_independence),
    }
}
