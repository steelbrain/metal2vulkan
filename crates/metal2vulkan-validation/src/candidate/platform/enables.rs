//! What a Vulkan context enables at `vkCreateDevice`.
//!
//! A context is a Vulkan 1.2 consumer (charter: 1.2 is the floor): its instance is created at API
//! 1.2, and everything above the floor is reached through its device extension. A context
//! restricted to a profile of a later version runs at the profile's version, where the structures
//! that version promoted are core: chained with no extension, which the profile need not list, and
//! those 1.3 promoted carried by `VkPhysicalDeviceVulkan13Features`, as the product's query reads
//! them. One [`FeatureChain`] holds every feature structure a context may chain; the supported
//! answer and the enabled request are two instances of it, and a structure is chained (to query or
//! to create) exactly when the context's version has it as core, or its extension is available or
//! enabled ([`Available::exposing`]), since chaining a structure of an absent extension below its
//! core version is invalid usage.
//!
//! Two lists decide what is enabled:
//!
//! - [`old_engine_list`]: the old executor's list, unchanged, the base every engine context
//!   enables (so its opened profile, and its rows, do not move; the pinned chains of
//!   `tests::every_committed_profiles_context_enables_its_pinned_chain` hold it).
//! - [`engine_list`] (engine contexts, on top of the old list): the product's
//!   `DeviceDescription::recommended_enables` for the device's queried description, less what no
//!   encoding reads ([`NOT_ENABLED`]), with the robustness features its leg decides
//!   (`coverage::ROBUSTNESS_FEATURES`): every one the description states in the stated leg, so the
//!   opened profile compiles the robust containment encodings the device states, and none in the
//!   robustness-off leg, so it compiles the in-shader ones. Extended dynamic state is enabled where
//!   the device lists it: a render pipeline states it as a requirement, and the render executor
//!   records its states. The [`ExecutorFeature`]s (sparse binding and residency) are enabled where
//!   the device supports them, for the executor's own sparse case textures: no pipeline requires
//!   them, so the opened profile does not see them.
//!
//! What a context enabled is read back from the enabled chain, member by member, through the
//! one table ([`MEMBERS`]) that also writes it: [`enabled_set`] in the engine's vocabulary
//! (the `EnableSet` that restricts the queried description, `DeviceDescription::with_enabled`).

use crate::coverage::{Leg, ROBUSTNESS_FEATURES};
use ash::{vk, Instance};
use metal2vulkan::device::{ApiVersion, VkExtension, VkFeature};
use metal2vulkan::device::{DeviceDescription, DevicePolicy, EnableSet};
use std::collections::BTreeSet;
use std::ffi::{c_char, CStr, CString};

/// `VkPhysicalDeviceMaintenance8FeaturesKHR`, which `ash 0.38` (Vulkan 1.3.281) predates, laid
/// out as the registry defines it (`sType`, `pNext`, `VkBool32 maintenance8`; extension number
/// 575, so its structure type is `1000000000 + 574 * 1000`). The product's query reads the same
/// structure (`ash_interop::query`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Maintenance8Features {
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

// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` and `VkDeviceCreateInfo` in the
// structure's `structextends`.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for Maintenance8Features {}
// SAFETY: as above.
unsafe impl vk::ExtendsDeviceCreateInfo for Maintenance8Features {}

/// `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`, which `ash 0.38` predates, laid out as
/// the registry defines it (`sType`, `pNext`, `VkBool32 deviceGeneratedCommands`,
/// `dynamicGeneratedPipelineLayout`; extension number 573, so its structure type is
/// `1000000000 + 572 * 1000`). The product's query reads the same structure
/// (`ash_interop::query`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GeneratedCommandsFeatures {
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

// SAFETY: the layout is the registry's `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`, which
// begins with `sType` and `pNext` as every chained structure must, and this is its structure type.
unsafe impl vk::TaggedStructure for GeneratedCommandsFeatures {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1_000_572_000);
}

// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` and `VkDeviceCreateInfo` in the
// structure's `structextends`.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for GeneratedCommandsFeatures {}
// SAFETY: as above.
unsafe impl vk::ExtendsDeviceCreateInfo for GeneratedCommandsFeatures {}

/// `VkPhysicalDeviceShaderFmaFeaturesKHR`, which `ash 0.38` predates, laid out as the registry
/// defines it (`sType`, `pNext`, `VkBool32 shaderFmaFloat16`, `shaderFmaFloat32`,
/// `shaderFmaFloat64`; extension number 580, so its structure type is `1000000000 + 579 * 1000`).
/// The product's query reads the same structure (`ash_interop::query`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ShaderFmaFeatures {
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

// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` and `VkDeviceCreateInfo` in the
// structure's `structextends`.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for ShaderFmaFeatures {}
// SAFETY: as above.
unsafe impl vk::ExtendsDeviceCreateInfo for ShaderFmaFeatures {}

/// `VkPhysicalDeviceDepthClampControlFeaturesEXT`, which `ash 0.38` predates, laid out as the
/// registry defines it (`sType`, `pNext`, `VkBool32 depthClampControl`; extension number 583, so
/// its structure type is `1000000000 + 582 * 1000`). The product's query reads the same structure
/// (`ash_interop::query`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct DepthClampControlFeatures {
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

// SAFETY: the registry lists `VkPhysicalDeviceFeatures2` and `VkDeviceCreateInfo` in the
// structure's `structextends`.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for DepthClampControlFeatures {}
// SAFETY: as above.
unsafe impl vk::ExtendsDeviceCreateInfo for DepthClampControlFeatures {}

/// One feature structure a context can chain at API 1.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Structure {
    /// `VkPhysicalDeviceFeatures` (`pEnabledFeatures`).
    Core,
    V11,
    V12,
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
    /// `VkPhysicalDeviceShaderFmaFeaturesKHR`.
    ShaderFma,
    FloatControls2,
    Maintenance5,
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
    /// `VkPhysicalDeviceMeshShaderFeaturesEXT`.
    MeshShader,
    AstcHdr,
    RayQuery,
    /// `VkPhysicalDeviceAccelerationStructureFeaturesKHR`.
    Acceleration,
    /// `VkPhysicalDeviceExtendedDynamicStateFeaturesEXT`.
    ExtendedDynamicState,
    /// `VkPhysicalDeviceExtendedDynamicState2FeaturesEXT`.
    ExtendedDynamicState2,
    /// `VkPhysicalDeviceExtendedDynamicState3FeaturesEXT`.
    ExtendedDynamicState3,
    /// `VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT`.
    FeedbackLoop,
    /// `VkPhysicalDeviceRobustness2FeaturesEXT` (`robustBufferAccess2`, `robustImageAccess2`).
    Robustness2,
    /// `VkPhysicalDeviceImageRobustnessFeatures` (`robustImageAccess`, core only from 1.3).
    ImageRobustness,
    PortabilitySubset,
    /// `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`: the device replay's
    /// generated-commands path, enabled only by [`enable_generated_commands`].
    GeneratedCommands,
}

impl Structure {
    /// Every extension structure, in chain order.
    const EXTENSIONS: [Structure; 42] = [
        Structure::SubgroupSizeControl,
        Structure::Demote,
        Structure::Terminate,
        Structure::ZeroInitialize,
        Structure::Synchronization2,
        Structure::Maintenance4,
        Structure::SubgroupRotate,
        Structure::LocalRead,
        Structure::Divisor,
        Structure::AtomicFloat,
        Structure::AtomicFloat2,
        Structure::ImageAtomicInt64,
        Structure::ExplicitLayout,
        Structure::MaximalReconvergence,
        Structure::QuadControl,
        Structure::UniformControlFlow,
        Structure::Derivatives,
        Structure::Barycentric,
        Structure::Interlock,
        Structure::RasterOrder,
        Structure::Maintenance8,
        Structure::ShaderFma,
        Structure::FloatControls2,
        Structure::Maintenance5,
        Structure::BorderColorSwizzle,
        Structure::DepthBiasControl,
        Structure::DepthClipEnable,
        Structure::DepthClampControl,
        Structure::ListRestart,
        Structure::ProvokingVertex,
        Structure::MeshShader,
        Structure::AstcHdr,
        Structure::RayQuery,
        Structure::Acceleration,
        Structure::ExtendedDynamicState,
        Structure::ExtendedDynamicState2,
        Structure::ExtendedDynamicState3,
        Structure::FeedbackLoop,
        Structure::Robustness2,
        Structure::ImageRobustness,
        Structure::PortabilitySubset,
        Structure::GeneratedCommands,
    ];

    /// The device extensions any one of which exposes the structure at API 1.2 (empty for the
    /// core structures). Where two names share one structure type (a promoted `KHR` of an `EXT`
    /// or `NV`, the `ARM` spelling of an `EXT`), the first available is enabled.
    fn extensions(self) -> &'static [&'static CStr] {
        match self {
            Structure::Core | Structure::V11 | Structure::V12 => &[],
            Structure::SubgroupSizeControl => &[c"VK_EXT_subgroup_size_control"],
            Structure::Demote => &[c"VK_EXT_shader_demote_to_helper_invocation"],
            Structure::Terminate => &[c"VK_KHR_shader_terminate_invocation"],
            Structure::ZeroInitialize => &[c"VK_KHR_zero_initialize_workgroup_memory"],
            Structure::Synchronization2 => &[c"VK_KHR_synchronization2"],
            Structure::Maintenance4 => &[c"VK_KHR_maintenance4"],
            Structure::SubgroupRotate => &[c"VK_KHR_shader_subgroup_rotate"],
            Structure::LocalRead => &[c"VK_KHR_dynamic_rendering_local_read"],
            Structure::Divisor => &[
                c"VK_KHR_vertex_attribute_divisor",
                c"VK_EXT_vertex_attribute_divisor",
            ],
            Structure::AtomicFloat => &[c"VK_EXT_shader_atomic_float"],
            Structure::AtomicFloat2 => &[c"VK_EXT_shader_atomic_float2"],
            Structure::ImageAtomicInt64 => &[c"VK_EXT_shader_image_atomic_int64"],
            Structure::ExplicitLayout => &[c"VK_KHR_workgroup_memory_explicit_layout"],
            Structure::MaximalReconvergence => &[c"VK_KHR_shader_maximal_reconvergence"],
            Structure::QuadControl => &[c"VK_KHR_shader_quad_control"],
            Structure::UniformControlFlow => &[c"VK_KHR_shader_subgroup_uniform_control_flow"],
            Structure::Derivatives => &[
                c"VK_KHR_compute_shader_derivatives",
                c"VK_NV_compute_shader_derivatives",
            ],
            Structure::Barycentric => &[c"VK_KHR_fragment_shader_barycentric"],
            Structure::Interlock => &[c"VK_EXT_fragment_shader_interlock"],
            Structure::RasterOrder => &[
                c"VK_EXT_rasterization_order_attachment_access",
                c"VK_ARM_rasterization_order_attachment_access",
            ],
            Structure::Maintenance8 => &[c"VK_KHR_maintenance8"],
            Structure::ShaderFma => &[c"VK_KHR_shader_fma"],
            Structure::FloatControls2 => &[c"VK_KHR_shader_float_controls2"],
            Structure::Maintenance5 => &[c"VK_KHR_maintenance5"],
            Structure::BorderColorSwizzle => &[c"VK_EXT_border_color_swizzle"],
            Structure::DepthBiasControl => &[c"VK_EXT_depth_bias_control"],
            Structure::DepthClipEnable => &[c"VK_EXT_depth_clip_enable"],
            Structure::DepthClampControl => &[c"VK_EXT_depth_clamp_control"],
            Structure::ListRestart => &[c"VK_EXT_primitive_topology_list_restart"],
            Structure::ProvokingVertex => &[c"VK_EXT_provoking_vertex"],
            // `VK_KHR_spirv_1_4`, which it requires, is core at the 1.2 floor.
            Structure::MeshShader => &[c"VK_EXT_mesh_shader"],
            Structure::AstcHdr => &[c"VK_EXT_texture_compression_astc_hdr"],
            Structure::RayQuery => &[c"VK_KHR_ray_query"],
            Structure::Acceleration => &[c"VK_KHR_acceleration_structure"],
            Structure::ExtendedDynamicState => &[c"VK_EXT_extended_dynamic_state"],
            Structure::ExtendedDynamicState2 => &[c"VK_EXT_extended_dynamic_state2"],
            Structure::ExtendedDynamicState3 => &[c"VK_EXT_extended_dynamic_state3"],
            Structure::FeedbackLoop => &[c"VK_EXT_attachment_feedback_loop_layout"],
            // One structure type serves both (vk.xml aliases the `EXT` type to the `KHR` one).
            Structure::Robustness2 => &[c"VK_EXT_robustness2", c"VK_KHR_robustness2"],
            Structure::ImageRobustness => &[c"VK_EXT_image_robustness"],
            Structure::PortabilitySubset => &[c"VK_KHR_portability_subset"],
            Structure::GeneratedCommands => &[c"VK_EXT_device_generated_commands"],
        }
    }

    /// The device extensions its extension requires on an API-1.2 device (vk.xml `depends`,
    /// beyond what 1.2 provides): `VK_KHR_dynamic_rendering` for the two that build on it,
    /// `VK_EXT_shader_atomic_float` for its successor, `VK_EXT_custom_border_color` for the
    /// border swizzle, and `VK_KHR_deferred_host_operations` (with, for ray queries,
    /// `VK_KHR_acceleration_structure`) for the ray tracing structures.
    fn requires(self) -> &'static [&'static CStr] {
        match self {
            Structure::LocalRead | Structure::Maintenance5 => &[c"VK_KHR_dynamic_rendering"],
            // Below 1.3, `VK_KHR_maintenance5` (or `VK_KHR_extended_flags`, which no device here
            // lists), which itself requires `VK_KHR_dynamic_rendering`.
            Structure::GeneratedCommands => &[c"VK_KHR_maintenance5", c"VK_KHR_dynamic_rendering"],
            Structure::AtomicFloat2 => &[c"VK_EXT_shader_atomic_float"],
            Structure::BorderColorSwizzle => &[c"VK_EXT_custom_border_color"],
            Structure::RayQuery => &[
                c"VK_KHR_acceleration_structure",
                c"VK_KHR_deferred_host_operations",
            ],
            Structure::Acceleration => &[c"VK_KHR_deferred_host_operations"],
            Structure::QuadControl => &[c"VK_KHR_shader_maximal_reconvergence"],
            _ => &[],
        }
    }

    /// The core version that promoted the structure (vk.xml): a context at that version or later
    /// chains it with no extension enabled. `None` for the base structures (always chained) and
    /// the ones no version up to 1.4 promoted.
    pub(super) fn core_since(self) -> Option<ApiVersion> {
        match self {
            Structure::SubgroupSizeControl
            | Structure::Demote
            | Structure::Terminate
            | Structure::ZeroInitialize
            | Structure::Synchronization2
            | Structure::Maintenance4
            | Structure::ImageRobustness
            | Structure::AstcHdr => Some(ApiVersion::V1_3),
            Structure::SubgroupRotate
            | Structure::LocalRead
            | Structure::Divisor
            | Structure::Maintenance5
            | Structure::FloatControls2 => Some(ApiVersion { major: 1, minor: 4 }),
            _ => None,
        }
    }

    /// Whether Vulkan 1.3 promoted the structure: a context at 1.3 or later queries and enables
    /// its members through `VkPhysicalDeviceVulkan13Features` ([`FeatureChain::v13`]), as the
    /// product's query reads them, never through the structure itself.
    fn in_vulkan13(self) -> bool {
        self.core_since() == Some(ApiVersion::V1_3)
    }

    /// Of `set`, the structures a context at `version` chains themselves, and whether it chains
    /// `VkPhysicalDeviceVulkan13Features` for the others ([`Structure::in_vulkan13`]).
    fn chained_at(version: ApiVersion, set: &BTreeSet<Structure>) -> (BTreeSet<Structure>, bool) {
        let carried =
            |structure: &Structure| version >= ApiVersion::V1_3 && structure.in_vulkan13();
        (
            set.iter().filter(|s| !carried(s)).copied().collect(),
            set.iter().any(carried),
        )
    }

    /// `VkPhysicalDeviceFeatures` and the Vulkan 1.1 and 1.2 structures, which every context
    /// chains.
    fn is_base(self) -> bool {
        matches!(self, Structure::Core | Structure::V11 | Structure::V12)
    }

    /// Whether a context at `version` has the structure as core: a base one always, a promoted
    /// one from its version.
    fn core_at(self, version: ApiVersion) -> bool {
        self.is_base() || self.core_since().is_some_and(|core| core <= version)
    }

    /// [`Structure::requires`] less what a context at `version` has as core.
    fn requires_at(self, version: ApiVersion) -> impl Iterator<Item = &'static CStr> {
        self.requires()
            .iter()
            .copied()
            .filter(move |name| promoted(name).is_none_or(|core| core > version))
    }
}

/// The core version that promoted a device extension a structure requires (vk.xml), if one up to
/// 1.4 did: `VK_KHR_dynamic_rendering` in 1.3, `VK_KHR_maintenance5` in 1.4.
fn promoted(name: &CStr) -> Option<ApiVersion> {
    if name == c"VK_KHR_dynamic_rendering" {
        Some(ApiVersion::V1_3)
    } else if name == c"VK_KHR_maintenance5" {
        Some(ApiVersion { major: 1, minor: 4 })
    } else {
        None
    }
}

/// Where a context takes the commands that record the extended dynamic states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DynamicStateCommands {
    /// Vulkan 1.3's core entry points: the functionality is core from 1.3, without a feature.
    Core,
    /// `VK_EXT_extended_dynamic_state`'s, which the context enabled.
    Extension,
}

/// Where a context at `version` takes the extended dynamic states' commands, if it has them: the
/// core ones from Vulkan 1.3 (a context restricted to a 1.3 profile, which need not list the
/// extension), else the extension's where the context `enabled` it.
pub(super) fn dynamic_state_commands(
    version: ApiVersion,
    enabled: bool,
) -> Option<DynamicStateCommands> {
    if version >= ApiVersion::V1_3 {
        Some(DynamicStateCommands::Core)
    } else {
        enabled.then_some(DynamicStateCommands::Extension)
    }
}

/// The device extensions a physical device lists, as a context at `version` sees them.
#[derive(Clone, Debug)]
pub(super) struct Available {
    names: BTreeSet<CString>,
    /// The context's API version (its instance's, at most the device's).
    version: ApiVersion,
}

impl Available {
    /// What a context at `version` sees (a context restricted to a later profile runs at its
    /// version).
    pub(super) fn at(version: ApiVersion, names: impl IntoIterator<Item = CString>) -> Available {
        Available {
            names: names.into_iter().collect(),
            version,
        }
    }

    pub(super) fn has(&self, name: &CStr) -> bool {
        self.names.contains(name)
    }

    /// How the device exposes `structure` to the context: `Some(None)` for a structure its
    /// version has as core ([`Structure::core_since`]), `Some(Some(name))` for the first available
    /// extension that exposes it and whose requirements (those the version does not have as core)
    /// are available too, `None` when it is not exposed. At Vulkan 1.2 only the base structures
    /// are core.
    pub(super) fn exposing(&self, structure: Structure) -> Option<Option<&'static CStr>> {
        if structure.core_at(self.version) {
            return Some(None);
        }
        if !structure
            .requires_at(self.version)
            .all(|name| self.has(name))
        {
            return None;
        }
        structure
            .extensions()
            .iter()
            .copied()
            .find(|name| self.has(name))
            .map(Some)
    }

    /// Every extension structure the device exposes to the context: what [`FeatureChain::query`]
    /// may chain.
    pub(super) fn exposed(&self) -> BTreeSet<Structure> {
        Structure::EXTENSIONS
            .into_iter()
            .filter(|&structure| self.exposing(structure).is_some())
            .collect()
    }
}

/// Every feature structure a context can chain at API 1.2. A structure that is not chained
/// stays zeroed. Its `pNext` links are null except during the one call that chains it.
#[derive(Clone, Copy, Default)]
pub(super) struct FeatureChain {
    pub(super) core: vk::PhysicalDeviceFeatures,
    pub(super) v11: vk::PhysicalDeviceVulkan11Features<'static>,
    pub(super) v12: vk::PhysicalDeviceVulkan12Features<'static>,
    /// In a context at Vulkan 1.3 or later, what carries the members of the structures 1.3
    /// promoted ([`Structure::in_vulkan13`]) to and from the device: the query reads them here
    /// and spreads them into those structures' fields, and creation gathers the enabled ones here.
    /// The profiles layer reports a promoted structure only where the profile states it, and a
    /// requirements document states its version's features in this one.
    pub(super) v13: vk::PhysicalDeviceVulkan13Features<'static>,
    pub(super) subgroup_size_control: vk::PhysicalDeviceSubgroupSizeControlFeatures<'static>,
    pub(super) demote: vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures<'static>,
    pub(super) terminate: vk::PhysicalDeviceShaderTerminateInvocationFeatures<'static>,
    pub(super) zero_initialize: vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures<'static>,
    pub(super) synchronization2: vk::PhysicalDeviceSynchronization2Features<'static>,
    pub(super) maintenance4: vk::PhysicalDeviceMaintenance4Features<'static>,
    pub(super) subgroup_rotate: vk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR<'static>,
    pub(super) local_read: vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR<'static>,
    // One structure type serves `VK_KHR_` and `VK_EXT_vertex_attribute_divisor` (vk.xml aliases
    // the `EXT` type to the `KHR` one).
    pub(super) divisor: vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR<'static>,
    pub(super) atomic_float: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT<'static>,
    pub(super) atomic_float2: vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT<'static>,
    pub(super) image_atomic_int64: vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT<'static>,
    pub(super) explicit_layout: vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR<'static>,
    pub(super) maximal_reconvergence:
        vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR<'static>,
    pub(super) quad_control: vk::PhysicalDeviceShaderQuadControlFeaturesKHR<'static>,
    pub(super) uniform_control_flow:
        vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR<'static>,
    // The `KHR` structure type is the `NV` one (a vk.xml alias).
    pub(super) derivatives: vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV<'static>,
    pub(super) barycentric: vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR<'static>,
    pub(super) interlock: vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT<'static>,
    pub(super) raster_order:
        vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT<'static>,
    pub(super) maintenance8: Maintenance8Features,
    pub(super) shader_fma: ShaderFmaFeatures,
    pub(super) float_controls2: vk::PhysicalDeviceShaderFloatControls2FeaturesKHR<'static>,
    pub(super) maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR<'static>,
    pub(super) border_color_swizzle: vk::PhysicalDeviceBorderColorSwizzleFeaturesEXT<'static>,
    pub(super) depth_bias_control: vk::PhysicalDeviceDepthBiasControlFeaturesEXT<'static>,
    pub(super) depth_clip_enable: vk::PhysicalDeviceDepthClipEnableFeaturesEXT<'static>,
    pub(super) depth_clamp_control: DepthClampControlFeatures,
    pub(super) list_restart: vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT<'static>,
    pub(super) provoking_vertex: vk::PhysicalDeviceProvokingVertexFeaturesEXT<'static>,
    pub(super) mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    pub(super) astc_hdr: vk::PhysicalDeviceTextureCompressionASTCHDRFeatures<'static>,
    pub(super) ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
    pub(super) acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    pub(super) extended_dynamic_state: vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT<'static>,
    pub(super) extended_dynamic_state2: vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT<'static>,
    pub(super) extended_dynamic_state3: vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static>,
    pub(super) feedback_loop: vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT<'static>,
    pub(super) robustness2: vk::PhysicalDeviceRobustness2FeaturesEXT<'static>,
    pub(super) image_robustness: vk::PhysicalDeviceImageRobustnessFeatures<'static>,
    /// A portable implementation reports here everything it CANNOT do, and the structure
    /// defaults to all-false: enabling `VK_KHR_portability_subset` without chaining it asks for
    /// none of it.
    pub(super) portability_subset: vk::PhysicalDevicePortabilitySubsetFeaturesKHR<'static>,
    pub(super) generated_commands: GeneratedCommandsFeatures,
}

/// Chains every structure of `$chain` whose [`Structure`] `$linked` holds onto `$info`, in
/// [`Structure::EXTENSIONS`] order.
macro_rules! link_extensions {
    ($info:ident, $chain:ident, $linked:ident) => {
        link_extensions!(@ $info, $chain, $linked;
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
            FloatControls2 => float_controls2,
            Maintenance5 => maintenance5,
            BorderColorSwizzle => border_color_swizzle,
            DepthBiasControl => depth_bias_control,
            DepthClipEnable => depth_clip_enable,
            DepthClampControl => depth_clamp_control,
            ListRestart => list_restart,
            ProvokingVertex => provoking_vertex,
            MeshShader => mesh_shader,
            AstcHdr => astc_hdr,
            RayQuery => ray_query,
            Acceleration => acceleration_structure,
            ExtendedDynamicState => extended_dynamic_state,
            ExtendedDynamicState2 => extended_dynamic_state2,
            ExtendedDynamicState3 => extended_dynamic_state3,
            FeedbackLoop => feedback_loop,
            Robustness2 => robustness2,
            ImageRobustness => image_robustness,
            PortabilitySubset => portability_subset,
            GeneratedCommands => generated_commands
        )
    };
    (@ $info:ident, $chain:ident, $linked:ident; $($variant:ident => $field:ident),*) => {
        $(if $linked.contains(&Structure::$variant) {
            $info = $info.push_next(&mut $chain.$field);
        })*
    };
}

impl FeatureChain {
    /// Null every `pNext` link. `push_next` walks the chain it is handed, so a structure must
    /// not carry a link from an earlier call into the next one.
    fn unlink(&mut self) {
        macro_rules! null {
            ($($field:ident),*) => {
                $(self.$field.p_next = std::ptr::null_mut();)*
            };
        }
        null!(
            v11,
            v12,
            v13,
            subgroup_size_control,
            demote,
            terminate,
            zero_initialize,
            synchronization2,
            maintenance4,
            subgroup_rotate,
            local_read,
            divisor,
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
            maintenance8,
            shader_fma,
            float_controls2,
            maintenance5,
            border_color_swizzle,
            depth_bias_control,
            depth_clip_enable,
            depth_clamp_control,
            list_restart,
            provoking_vertex,
            mesh_shader,
            astc_hdr,
            ray_query,
            acceleration_structure,
            extended_dynamic_state,
            extended_dynamic_state2,
            extended_dynamic_state3,
            feedback_loop,
            robustness2,
            image_robustness,
            portability_subset,
            generated_commands
        );
    }

    /// What `physical` supports, reading exactly the extension structures `available` exposes
    /// ([`Available::exposed`], the base structures always), those Vulkan 1.3 promoted through
    /// [`FeatureChain::v13`] in a context at 1.3 or later.
    ///
    /// # Safety
    /// `physical` was enumerated from `instance`, created at API 1.2 or later, and `available` is
    /// what the device lists, at most at the version the instance and the device share.
    pub(super) unsafe fn query(
        instance: &Instance,
        physical: vk::PhysicalDevice,
        available: &Available,
    ) -> FeatureChain {
        let (exposed, v13) = Structure::chained_at(available.version, &available.exposed());
        let mut chain = FeatureChain::default();
        let c = &mut chain;
        let core = {
            let mut info = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut c.v11)
                .push_next(&mut c.v12);
            if v13 {
                info = info.push_next(&mut c.v13);
            }
            link_extensions!(info, c, exposed);
            // SAFETY: the caller's contract; every chained structure lives until the call
            // returns and is core at the version or of an extension the device lists.
            unsafe { instance.get_physical_device_features2(physical, &mut info) };
            info.features
        };
        chain.core = core;
        if v13 {
            chain.spread_v13();
        }
        chain.unlink();
        chain
    }

    /// Copy the members [`FeatureChain::v13`] carries into the fields of the structures Vulkan
    /// 1.3 promoted, where the rest of the model reads them.
    fn spread_v13(&mut self) {
        let v = self.v13;
        self.subgroup_size_control.subgroup_size_control = v.subgroup_size_control;
        self.subgroup_size_control.compute_full_subgroups = v.compute_full_subgroups;
        self.demote.shader_demote_to_helper_invocation = v.shader_demote_to_helper_invocation;
        self.terminate.shader_terminate_invocation = v.shader_terminate_invocation;
        self.zero_initialize.shader_zero_initialize_workgroup_memory =
            v.shader_zero_initialize_workgroup_memory;
        self.synchronization2.synchronization2 = v.synchronization2;
        self.maintenance4.maintenance4 = v.maintenance4;
        self.image_robustness.robust_image_access = v.robust_image_access;
        self.astc_hdr.texture_compression_astc_hdr = v.texture_compression_astc_hdr;
    }

    /// [`FeatureChain::v13`] for creation: the members of each structure Vulkan 1.3 promoted that
    /// `linked` holds, and nothing else.
    fn gather_v13(
        &self,
        linked: &BTreeSet<Structure>,
    ) -> vk::PhysicalDeviceVulkan13Features<'static> {
        let mut v = vk::PhysicalDeviceVulkan13Features::default();
        let has = |structure| linked.contains(&structure);
        if has(Structure::SubgroupSizeControl) {
            v.subgroup_size_control = self.subgroup_size_control.subgroup_size_control;
            v.compute_full_subgroups = self.subgroup_size_control.compute_full_subgroups;
        }
        if has(Structure::Demote) {
            v.shader_demote_to_helper_invocation = self.demote.shader_demote_to_helper_invocation;
        }
        if has(Structure::Terminate) {
            v.shader_terminate_invocation = self.terminate.shader_terminate_invocation;
        }
        if has(Structure::ZeroInitialize) {
            v.shader_zero_initialize_workgroup_memory =
                self.zero_initialize.shader_zero_initialize_workgroup_memory;
        }
        if has(Structure::Synchronization2) {
            v.synchronization2 = self.synchronization2.synchronization2;
        }
        if has(Structure::Maintenance4) {
            v.maintenance4 = self.maintenance4.maintenance4;
        }
        if has(Structure::ImageRobustness) {
            v.robust_image_access = self.image_robustness.robust_image_access;
        }
        if has(Structure::AstcHdr) {
            v.texture_compression_astc_hdr = self.astc_hdr.texture_compression_astc_hdr;
        }
        v
    }

    /// Hands `create` the `VkDeviceCreateInfo` with this chain as the enabled features: the core
    /// structures always, and exactly the extension structures `linked` names, in a context at
    /// `version` 1.3 or later those Vulkan 1.3 promoted gathered into [`FeatureChain::v13`]. The
    /// caller makes the `vkCreateDevice` call, in `device.rs` behind the hang tripwire's check
    /// (the one raw device open, `gpu::tests::device_opens_exist_only_behind_the_tripwire_check`).
    /// Every chained structure lives until `create` returns, and the chain is unlinked after.
    pub(super) fn with_create_info<R>(
        &mut self,
        queues: &[vk::DeviceQueueCreateInfo<'_>],
        extensions: &[*const c_char],
        linked: &BTreeSet<Structure>,
        version: ApiVersion,
        create: impl FnOnce(&vk::DeviceCreateInfo<'_>) -> R,
    ) -> R {
        let (chained, v13) = Structure::chained_at(version, linked);
        self.unlink();
        if v13 {
            self.v13 = self.gather_v13(linked);
        }
        let created = {
            let c = &mut *self;
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(queues)
                .enabled_extension_names(extensions)
                .enabled_features(&c.core)
                .push_next(&mut c.v11)
                .push_next(&mut c.v12);
            if v13 {
                info = info.push_next(&mut c.v13);
            }
            link_extensions!(info, c, chained);
            create(&info)
        };
        self.unlink();
        created
    }
}

/// A feature an engine context enables for this executor's own objects rather than for any
/// pipeline: no module requires it, and neither engine's vocabulary names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum ExecutorFeature {
    /// `sparseBinding`: a sparse case texture's image binds its memory on the queue.
    SparseBinding,
    /// `sparseResidencyImage2D`: a 2D (or 2D array) sparse case texture is a sparse resident
    /// image.
    SparseResidencyImage2D,
    /// `sparseResidencyImage3D`: a 3D sparse case texture is a sparse resident image.
    SparseResidencyImage3D,
}

impl ExecutorFeature {
    /// The member's Vulkan name.
    pub(super) const fn name(self) -> &'static str {
        match self {
            ExecutorFeature::SparseBinding => "sparseBinding",
            ExecutorFeature::SparseResidencyImage2D => "sparseResidencyImage2D",
            ExecutorFeature::SparseResidencyImage3D => "sparseResidencyImage3D",
        }
    }
}

/// One `VkBool32` member of a [`FeatureChain`] structure.
pub(super) struct Member {
    pub(super) structure: Structure,
    /// The member's field in its structure (ash's name for the `VkBool32`).
    pub(super) name: &'static str,
    /// The engine's name for the member, when its vocabulary has one.
    pub(super) feature: Option<VkFeature>,
    /// The executor's own name for it, when an engine context enables it for the executor's
    /// objects ([`ExecutorFeature`]).
    pub(super) executor: Option<ExecutorFeature>,
    read: fn(&FeatureChain) -> vk::Bool32,
    write: fn(&mut FeatureChain) -> &mut vk::Bool32,
}

impl Member {
    pub(super) fn is_set(&self, chain: &FeatureChain) -> bool {
        (self.read)(chain) == vk::TRUE
    }

    pub(super) fn set(&self, chain: &mut FeatureChain) {
        *(self.write)(chain) = vk::TRUE;
    }
}

/// Defines [`MEMBERS`] from `structure.field.member => feature` rows, each with an optional
/// `, executor` name.
macro_rules! members {
    ($($structure:ident . $field:ident . $member:ident => $feature:expr
        $(, $executor:expr)?;)*) => {
        /// Every member a context reads or writes, each once: the one table that enables a
        /// feature and names what was enabled.
        pub(super) const MEMBERS: &[Member] = &[$(Member {
            structure: Structure::$structure,
            name: stringify!($member),
            feature: $feature,
            executor: members!(@executor $($executor)?),
            read: {
                fn read(chain: &FeatureChain) -> vk::Bool32 {
                    chain.$field.$member
                }
                read
            },
            write: {
                fn write(chain: &mut FeatureChain) -> &mut vk::Bool32 {
                    &mut chain.$field.$member
                }
                write
            },
        },)*];
    };
    (@executor) => {
        None
    };
    (@executor $executor:expr) => {
        Some($executor)
    };
}

use ExecutorFeature as X;
use VkFeature as F;

members! {
    Core.core.robust_buffer_access => Some(F::RobustBufferAccess);
    Core.core.shader_int16 => Some(F::ShaderInt16);
    Core.core.shader_int64 => Some(F::ShaderInt64);
    Core.core.shader_float64 => Some(F::ShaderFloat64);
    Core.core.vertex_pipeline_stores_and_atomics => Some(F::VertexPipelineStoresAndAtomics);
    Core.core.fragment_stores_and_atomics => Some(F::FragmentStoresAndAtomics);
    Core.core.geometry_shader => Some(F::GeometryShader);
    Core.core.tessellation_shader => Some(F::TessellationShader);
    Core.core.sample_rate_shading => Some(F::SampleRateShading);
    Core.core.alpha_to_one => Some(F::AlphaToOne);
    Core.core.dual_src_blend => Some(F::DualSrcBlend);
    Core.core.logic_op => Some(F::LogicOp);
    Core.core.independent_blend => Some(F::IndependentBlend);
    Core.core.shader_clip_distance => Some(F::ShaderClipDistance);
    Core.core.shader_cull_distance => Some(F::ShaderCullDistance);
    Core.core.draw_indirect_first_instance => Some(F::DrawIndirectFirstInstance);
    Core.core.multi_draw_indirect => Some(F::MultiDrawIndirect);
    Core.core.depth_clamp => Some(F::DepthClamp);
    Core.core.depth_bias_clamp => Some(F::DepthBiasClamp);
    Core.core.fill_mode_non_solid => Some(F::FillModeNonSolid);
    Core.core.image_cube_array => Some(F::ImageCubeArray);
    Core.core.shader_image_gather_extended => Some(F::ShaderImageGatherExtended);
    Core.core.shader_resource_min_lod => Some(F::ShaderResourceMinLod);
    Core.core.shader_resource_residency => Some(F::ShaderResourceResidency);
    Core.core.sampler_anisotropy => Some(F::SamplerAnisotropy);
    Core.core.shader_storage_image_multisample => Some(F::ShaderStorageImageMultisample);
    Core.core.shader_storage_image_extended_formats => Some(F::ShaderStorageImageExtendedFormats);
    Core.core.shader_storage_image_read_without_format =>
        Some(F::ShaderStorageImageReadWithoutFormat);
    Core.core.shader_storage_image_write_without_format =>
        Some(F::ShaderStorageImageWriteWithoutFormat);
    Core.core.shader_uniform_buffer_array_dynamic_indexing =>
        Some(F::ShaderUniformBufferArrayDynamicIndexing);
    Core.core.shader_sampled_image_array_dynamic_indexing =>
        Some(F::ShaderSampledImageArrayDynamicIndexing);
    Core.core.shader_storage_buffer_array_dynamic_indexing =>
        Some(F::ShaderStorageBufferArrayDynamicIndexing);
    Core.core.shader_storage_image_array_dynamic_indexing =>
        Some(F::ShaderStorageImageArrayDynamicIndexing);
    Core.core.multi_viewport => Some(F::MultiViewport);
    Core.core.occlusion_query_precise => Some(F::OcclusionQueryPrecise);
    Core.core.full_draw_index_uint32 => Some(F::FullDrawIndexUint32);
    Core.core.shader_tessellation_and_geometry_point_size =>
        Some(F::ShaderTessellationAndGeometryPointSize);
    Core.core.sparse_binding => None, X::SparseBinding;
    Core.core.sparse_residency_image2_d => None, X::SparseResidencyImage2D;
    Core.core.sparse_residency_image3_d => None, X::SparseResidencyImage3D;
    V11.v11.storage_buffer16_bit_access => Some(F::StorageBuffer16BitAccess);
    V11.v11.uniform_and_storage_buffer16_bit_access => Some(F::UniformAndStorageBuffer16BitAccess);
    V11.v11.storage_push_constant16 => Some(F::StoragePushConstant16);
    V11.v11.storage_input_output16 => Some(F::StorageInputOutput16);
    V11.v11.multiview => Some(F::Multiview);
    V11.v11.multiview_tessellation_shader => Some(F::MultiviewTessellationShader);
    V11.v11.multiview_geometry_shader => Some(F::MultiviewGeometryShader);
    V11.v11.shader_draw_parameters => Some(F::ShaderDrawParameters);
    V11.v11.variable_pointers => None;
    V11.v11.variable_pointers_storage_buffer => None;
    V12.v12.sampler_mirror_clamp_to_edge => Some(F::SamplerMirrorClampToEdge);
    V12.v12.storage_buffer8_bit_access => Some(F::StorageBuffer8BitAccess);
    V12.v12.uniform_and_storage_buffer8_bit_access => Some(F::UniformAndStorageBuffer8BitAccess);
    V12.v12.storage_push_constant8 => Some(F::StoragePushConstant8);
    V12.v12.shader_buffer_int64_atomics => Some(F::ShaderBufferInt64Atomics);
    V12.v12.shader_shared_int64_atomics => Some(F::ShaderSharedInt64Atomics);
    V12.v12.shader_float16 => Some(F::ShaderFloat16);
    V12.v12.shader_int8 => Some(F::ShaderInt8);
    V12.v12.shader_input_attachment_array_dynamic_indexing =>
        Some(F::ShaderInputAttachmentArrayDynamicIndexing);
    V12.v12.shader_uniform_texel_buffer_array_dynamic_indexing =>
        Some(F::ShaderUniformTexelBufferArrayDynamicIndexing);
    V12.v12.shader_storage_texel_buffer_array_dynamic_indexing =>
        Some(F::ShaderStorageTexelBufferArrayDynamicIndexing);
    V12.v12.shader_sampled_image_array_non_uniform_indexing =>
        Some(F::ShaderSampledImageArrayNonUniformIndexing);
    V12.v12.shader_storage_buffer_array_non_uniform_indexing =>
        Some(F::ShaderStorageBufferArrayNonUniformIndexing);
    V12.v12.shader_storage_image_array_non_uniform_indexing =>
        Some(F::ShaderStorageImageArrayNonUniformIndexing);
    V12.v12.shader_uniform_texel_buffer_array_non_uniform_indexing =>
        Some(F::ShaderUniformTexelBufferArrayNonUniformIndexing);
    V12.v12.shader_storage_texel_buffer_array_non_uniform_indexing =>
        Some(F::ShaderStorageTexelBufferArrayNonUniformIndexing);
    V12.v12.descriptor_binding_sampled_image_update_after_bind =>
        Some(F::DescriptorBindingSampledImageUpdateAfterBind);
    V12.v12.descriptor_binding_storage_image_update_after_bind =>
        Some(F::DescriptorBindingStorageImageUpdateAfterBind);
    V12.v12.descriptor_binding_storage_buffer_update_after_bind =>
        Some(F::DescriptorBindingStorageBufferUpdateAfterBind);
    V12.v12.descriptor_binding_uniform_texel_buffer_update_after_bind =>
        Some(F::DescriptorBindingUniformTexelBufferUpdateAfterBind);
    V12.v12.descriptor_binding_storage_texel_buffer_update_after_bind =>
        Some(F::DescriptorBindingStorageTexelBufferUpdateAfterBind);
    V12.v12.descriptor_binding_update_unused_while_pending =>
        Some(F::DescriptorBindingUpdateUnusedWhilePending);
    V12.v12.descriptor_binding_partially_bound => Some(F::DescriptorBindingPartiallyBound);
    V12.v12.descriptor_binding_variable_descriptor_count =>
        Some(F::DescriptorBindingVariableDescriptorCount);
    V12.v12.runtime_descriptor_array => Some(F::RuntimeDescriptorArray);
    V12.v12.sampler_filter_minmax => Some(F::SamplerFilterMinmax);
    V12.v12.draw_indirect_count => Some(F::DrawIndirectCount);
    V12.v12.scalar_block_layout => Some(F::ScalarBlockLayout);
    V12.v12.uniform_buffer_standard_layout => Some(F::UniformBufferStandardLayout);
    V12.v12.shader_subgroup_extended_types => Some(F::ShaderSubgroupExtendedTypes);
    V12.v12.buffer_device_address => Some(F::BufferDeviceAddress);
    V12.v12.vulkan_memory_model => Some(F::VulkanMemoryModel);
    V12.v12.vulkan_memory_model_device_scope => Some(F::VulkanMemoryModelDeviceScope);
    V12.v12.shader_output_viewport_index => Some(F::ShaderOutputViewportIndex);
    V12.v12.shader_output_layer => Some(F::ShaderOutputLayer);
    SubgroupSizeControl.subgroup_size_control.subgroup_size_control => Some(F::SubgroupSizeControl);
    SubgroupSizeControl.subgroup_size_control.compute_full_subgroups =>
        Some(F::ComputeFullSubgroups);
    Demote.demote.shader_demote_to_helper_invocation => Some(F::ShaderDemoteToHelperInvocation);
    Terminate.terminate.shader_terminate_invocation => Some(F::ShaderTerminateInvocation);
    ZeroInitialize.zero_initialize.shader_zero_initialize_workgroup_memory =>
        Some(F::ShaderZeroInitializeWorkgroupMemory);
    Synchronization2.synchronization2.synchronization2 => Some(F::Synchronization2);
    Maintenance4.maintenance4.maintenance4 => Some(F::Maintenance4);
    SubgroupRotate.subgroup_rotate.shader_subgroup_rotate => Some(F::ShaderSubgroupRotate);
    SubgroupRotate.subgroup_rotate.shader_subgroup_rotate_clustered =>
        Some(F::ShaderSubgroupRotateClustered);
    LocalRead.local_read.dynamic_rendering_local_read => Some(F::DynamicRenderingLocalRead);
    Divisor.divisor.vertex_attribute_instance_rate_divisor =>
        Some(F::VertexAttributeInstanceRateDivisor);
    Divisor.divisor.vertex_attribute_instance_rate_zero_divisor =>
        Some(F::VertexAttributeInstanceRateZeroDivisor);
    AtomicFloat.atomic_float.shader_buffer_float32_atomics => Some(F::ShaderBufferFloat32Atomics);
    AtomicFloat.atomic_float.shader_buffer_float32_atomic_add =>
        Some(F::ShaderBufferFloat32AtomicAdd);
    AtomicFloat.atomic_float.shader_buffer_float64_atomics => Some(F::ShaderBufferFloat64Atomics);
    AtomicFloat.atomic_float.shader_buffer_float64_atomic_add =>
        Some(F::ShaderBufferFloat64AtomicAdd);
    AtomicFloat.atomic_float.shader_shared_float32_atomics => Some(F::ShaderSharedFloat32Atomics);
    AtomicFloat.atomic_float.shader_shared_float32_atomic_add =>
        Some(F::ShaderSharedFloat32AtomicAdd);
    AtomicFloat.atomic_float.shader_shared_float64_atomics => Some(F::ShaderSharedFloat64Atomics);
    AtomicFloat.atomic_float.shader_shared_float64_atomic_add =>
        Some(F::ShaderSharedFloat64AtomicAdd);
    AtomicFloat.atomic_float.shader_image_float32_atomics => Some(F::ShaderImageFloat32Atomics);
    AtomicFloat.atomic_float.shader_image_float32_atomic_add =>
        Some(F::ShaderImageFloat32AtomicAdd);
    AtomicFloat2.atomic_float2.shader_buffer_float16_atomics => Some(F::ShaderBufferFloat16Atomics);
    AtomicFloat2.atomic_float2.shader_buffer_float16_atomic_add =>
        Some(F::ShaderBufferFloat16AtomicAdd);
    AtomicFloat2.atomic_float2.shader_buffer_float16_atomic_min_max =>
        Some(F::ShaderBufferFloat16AtomicMinMax);
    AtomicFloat2.atomic_float2.shader_buffer_float32_atomic_min_max =>
        Some(F::ShaderBufferFloat32AtomicMinMax);
    AtomicFloat2.atomic_float2.shader_buffer_float64_atomic_min_max =>
        Some(F::ShaderBufferFloat64AtomicMinMax);
    AtomicFloat2.atomic_float2.shader_shared_float16_atomics => Some(F::ShaderSharedFloat16Atomics);
    AtomicFloat2.atomic_float2.shader_shared_float16_atomic_add =>
        Some(F::ShaderSharedFloat16AtomicAdd);
    AtomicFloat2.atomic_float2.shader_shared_float16_atomic_min_max =>
        Some(F::ShaderSharedFloat16AtomicMinMax);
    AtomicFloat2.atomic_float2.shader_shared_float32_atomic_min_max =>
        Some(F::ShaderSharedFloat32AtomicMinMax);
    AtomicFloat2.atomic_float2.shader_shared_float64_atomic_min_max =>
        Some(F::ShaderSharedFloat64AtomicMinMax);
    AtomicFloat2.atomic_float2.shader_image_float32_atomic_min_max =>
        Some(F::ShaderImageFloat32AtomicMinMax);
    ImageAtomicInt64.image_atomic_int64.shader_image_int64_atomics =>
        Some(F::ShaderImageInt64Atomics);
    ExplicitLayout.explicit_layout.workgroup_memory_explicit_layout =>
        Some(F::WorkgroupMemoryExplicitLayout);
    ExplicitLayout.explicit_layout.workgroup_memory_explicit_layout_scalar_block_layout =>
        Some(F::WorkgroupMemoryExplicitLayoutScalarBlockLayout);
    ExplicitLayout.explicit_layout.workgroup_memory_explicit_layout8_bit_access =>
        Some(F::WorkgroupMemoryExplicitLayout8BitAccess);
    ExplicitLayout.explicit_layout.workgroup_memory_explicit_layout16_bit_access =>
        Some(F::WorkgroupMemoryExplicitLayout16BitAccess);
    MaximalReconvergence.maximal_reconvergence.shader_maximal_reconvergence =>
        Some(F::ShaderMaximalReconvergence);
    QuadControl.quad_control.shader_quad_control => Some(F::ShaderQuadControl);
    UniformControlFlow.uniform_control_flow.shader_subgroup_uniform_control_flow =>
        Some(F::ShaderSubgroupUniformControlFlow);
    Derivatives.derivatives.compute_derivative_group_quads => Some(F::ComputeDerivativeGroupQuads);
    Derivatives.derivatives.compute_derivative_group_linear =>
        Some(F::ComputeDerivativeGroupLinear);
    Barycentric.barycentric.fragment_shader_barycentric => Some(F::FragmentShaderBarycentric);
    Interlock.interlock.fragment_shader_pixel_interlock => Some(F::FragmentShaderPixelInterlock);
    Interlock.interlock.fragment_shader_sample_interlock => Some(F::FragmentShaderSampleInterlock);
    RasterOrder.raster_order.rasterization_order_color_attachment_access =>
        Some(F::RasterizationOrderColorAttachmentAccess);
    RasterOrder.raster_order.rasterization_order_depth_attachment_access =>
        Some(F::RasterizationOrderDepthAttachmentAccess);
    RasterOrder.raster_order.rasterization_order_stencil_attachment_access =>
        Some(F::RasterizationOrderStencilAttachmentAccess);
    Maintenance8.maintenance8.maintenance8 => Some(F::Maintenance8);
    ShaderFma.shader_fma.shader_fma_float16 => Some(F::ShaderFmaFloat16);
    ShaderFma.shader_fma.shader_fma_float32 => Some(F::ShaderFmaFloat32);
    FloatControls2.float_controls2.shader_float_controls2 => None;
    Maintenance5.maintenance5.maintenance5 => Some(F::Maintenance5);
    BorderColorSwizzle.border_color_swizzle.border_color_swizzle => Some(F::BorderColorSwizzle);
    BorderColorSwizzle.border_color_swizzle.border_color_swizzle_from_image =>
        Some(F::BorderColorSwizzleFromImage);
    DepthBiasControl.depth_bias_control.depth_bias_control => Some(F::DepthBiasControl);
    DepthBiasControl.depth_bias_control.float_representation => Some(F::FloatRepresentation);
    DepthBiasControl.depth_bias_control.depth_bias_exact => Some(F::DepthBiasExact);
    DepthClipEnable.depth_clip_enable.depth_clip_enable => Some(F::DepthClipEnable);
    DepthClampControl.depth_clamp_control.depth_clamp_control => Some(F::DepthClampControl);
    ListRestart.list_restart.primitive_topology_list_restart =>
        Some(F::PrimitiveTopologyListRestart);
    ListRestart.list_restart.primitive_topology_patch_list_restart =>
        Some(F::PrimitiveTopologyPatchListRestart);
    ProvokingVertex.provoking_vertex.provoking_vertex_last => Some(F::ProvokingVertexLast);
    ProvokingVertex.provoking_vertex.transform_feedback_preserves_provoking_vertex => None;
    MeshShader.mesh_shader.task_shader => Some(F::TaskShader);
    MeshShader.mesh_shader.mesh_shader => Some(F::MeshShader);
    AstcHdr.astc_hdr.texture_compression_astc_hdr => None;
    RayQuery.ray_query.ray_query => Some(F::RayQuery);
    Acceleration.acceleration_structure.acceleration_structure => Some(F::AccelerationStructure);
    ExtendedDynamicState.extended_dynamic_state.extended_dynamic_state =>
        Some(F::ExtendedDynamicState);
    ExtendedDynamicState2.extended_dynamic_state2.extended_dynamic_state2 =>
        Some(F::ExtendedDynamicState2);
    ExtendedDynamicState3.extended_dynamic_state3.extended_dynamic_state3_color_write_mask =>
        Some(F::ExtendedDynamicState3ColorWriteMask);
    FeedbackLoop.feedback_loop.attachment_feedback_loop_layout =>
        Some(F::AttachmentFeedbackLoopLayout);
    Robustness2.robustness2.robust_buffer_access2 => Some(F::RobustBufferAccess2);
    Robustness2.robustness2.robust_image_access2 => Some(F::RobustImageAccess2);
    Robustness2.robustness2.null_descriptor => Some(F::NullDescriptor);
    ImageRobustness.image_robustness.robust_image_access => Some(F::RobustImageAccess);
    PortabilitySubset.portability_subset.constant_alpha_color_blend_factors =>
        Some(F::ConstantAlphaColorBlendFactors);
    PortabilitySubset.portability_subset.vertex_attribute_access_beyond_stride =>
        Some(F::VertexAttributeAccessBeyondStride);
    PortabilitySubset.portability_subset.sampler_mip_lod_bias => Some(F::SamplerMipLodBias);
    PortabilitySubset.portability_subset.mutable_comparison_samplers =>
        Some(F::MutableComparisonSamplers);
    PortabilitySubset.portability_subset.shader_sample_rate_interpolation_functions =>
        Some(F::ShaderSampleRateInterpolationFunctions);
    PortabilitySubset.portability_subset.multisample_array_image =>
        Some(F::MultisampleArrayImage);
    PortabilitySubset.portability_subset.image_view2_d_on3_d_image =>
        Some(F::ImageView2DOn3DImage);
    PortabilitySubset.portability_subset.separate_stencil_mask_ref =>
        Some(F::SeparateStencilMaskRef);
    PortabilitySubset.portability_subset.image_view_format_swizzle =>
        Some(F::ImageViewFormatSwizzle);
    GeneratedCommands.generated_commands.device_generated_commands => None;
}

/// The engine's vocabulary features [`engine_list`] never enables, each with why. None of
/// them is a pipeline requirement, and the product reads none of them. Their extensions stay
/// enabled where the product recommends them: an extension enabled without its feature members
/// changes nothing the product reads. `extendedDynamicState` is not here: a render pipeline
/// requires it where it makes a state dynamic, and the render executor records every such state but
/// a per-bind stride. The robustness features are not here either: the product reads them to choose
/// the containment encoding, and the context's leg decides them.
pub(super) const NOT_ENABLED: &[(VkFeature, &str)] = &[(
    F::NullDescriptor,
    "no encoding reads it; it shares `VK_EXT_robustness2`'s structure with the robust-access \
         members, which the leg decides",
)];

/// One device extension a context can enable, with its name in the engine's vocabulary.
struct ExtensionName {
    name: &'static CStr,
    vocabulary: Option<VkExtension>,
}

/// Every device extension a context can enable, but the runtime probe's own
/// ([`enable_runtime_probe`]). Where two names share one vocabulary extension (a `KHR` promotion
/// of an `EXT` or `NV`, an `ARM` spelling of an `EXT`), the first the device lists is enabled.
const EXTENSIONS: &[ExtensionName] = {
    use VkExtension as X;
    const fn e(name: &'static CStr, vocabulary: Option<VkExtension>) -> ExtensionName {
        ExtensionName { name, vocabulary }
    }
    &[
        e(
            c"VK_EXT_shader_stencil_export",
            Some(X::ExtShaderStencilExport),
        ),
        e(
            c"VK_EXT_subgroup_size_control",
            Some(X::ExtSubgroupSizeControl),
        ),
        e(
            c"VK_EXT_shader_demote_to_helper_invocation",
            Some(X::ExtShaderDemoteToHelperInvocation),
        ),
        e(
            c"VK_KHR_shader_terminate_invocation",
            Some(X::KhrShaderTerminateInvocation),
        ),
        e(
            c"VK_KHR_zero_initialize_workgroup_memory",
            Some(X::KhrZeroInitializeWorkgroupMemory),
        ),
        e(c"VK_KHR_synchronization2", Some(X::KhrSynchronization2)),
        e(c"VK_KHR_maintenance4", Some(X::KhrMaintenance4)),
        e(
            c"VK_KHR_shader_subgroup_rotate",
            Some(X::KhrShaderSubgroupRotate),
        ),
        e(
            c"VK_KHR_dynamic_rendering_local_read",
            Some(X::KhrDynamicRenderingLocalRead),
        ),
        e(
            c"VK_KHR_vertex_attribute_divisor",
            Some(X::KhrVertexAttributeDivisor),
        ),
        e(
            c"VK_EXT_vertex_attribute_divisor",
            Some(X::ExtVertexAttributeDivisor),
        ),
        e(c"VK_EXT_shader_atomic_float", Some(X::ExtShaderAtomicFloat)),
        e(
            c"VK_EXT_shader_atomic_float2",
            Some(X::ExtShaderAtomicFloat2),
        ),
        e(
            c"VK_EXT_shader_image_atomic_int64",
            Some(X::ExtShaderImageAtomicInt64),
        ),
        e(
            c"VK_KHR_workgroup_memory_explicit_layout",
            Some(X::KhrWorkgroupMemoryExplicitLayout),
        ),
        e(
            c"VK_KHR_shader_maximal_reconvergence",
            Some(X::KhrShaderMaximalReconvergence),
        ),
        e(c"VK_KHR_shader_quad_control", Some(X::KhrShaderQuadControl)),
        e(
            c"VK_KHR_shader_subgroup_uniform_control_flow",
            Some(X::KhrShaderSubgroupUniformControlFlow),
        ),
        e(
            c"VK_KHR_compute_shader_derivatives",
            Some(X::KhrComputeShaderDerivatives),
        ),
        e(
            c"VK_NV_compute_shader_derivatives",
            Some(X::KhrComputeShaderDerivatives),
        ),
        e(
            c"VK_KHR_fragment_shader_barycentric",
            Some(X::KhrFragmentShaderBarycentric),
        ),
        e(
            c"VK_EXT_fragment_shader_interlock",
            Some(X::ExtFragmentShaderInterlock),
        ),
        e(
            c"VK_EXT_rasterization_order_attachment_access",
            Some(X::ExtRasterizationOrderAttachmentAccess),
        ),
        e(
            c"VK_ARM_rasterization_order_attachment_access",
            Some(X::ExtRasterizationOrderAttachmentAccess),
        ),
        e(c"VK_KHR_maintenance8", Some(X::KhrMaintenance8)),
        e(c"VK_KHR_shader_fma", Some(X::KhrShaderFma)),
        // The product recommends it where the device lists it: the executor's copies into a depth
        // aspect then keep Metal's texels outside `[0, 1]`.
        e(
            c"VK_EXT_depth_range_unrestricted",
            Some(X::ExtDepthRangeUnrestricted),
        ),
        e(c"VK_KHR_ray_query", Some(X::KhrRayQuery)),
        e(
            c"VK_KHR_acceleration_structure",
            Some(X::KhrAccelerationStructure),
        ),
        e(
            c"VK_KHR_deferred_host_operations",
            Some(X::KhrDeferredHostOperations),
        ),
        e(c"VK_EXT_robustness2", Some(X::ExtRobustness2)),
        e(c"VK_KHR_robustness2", Some(X::ExtRobustness2)),
        e(c"VK_EXT_image_robustness", Some(X::ExtImageRobustness)),
        e(
            c"VK_EXT_extended_dynamic_state",
            Some(X::ExtExtendedDynamicState),
        ),
        e(
            c"VK_EXT_extended_dynamic_state2",
            Some(X::ExtExtendedDynamicState2),
        ),
        e(
            c"VK_EXT_extended_dynamic_state3",
            Some(X::ExtExtendedDynamicState3),
        ),
        e(
            c"VK_EXT_attachment_feedback_loop_layout",
            Some(X::ExtAttachmentFeedbackLoopLayout),
        ),
        e(c"VK_KHR_push_descriptor", Some(X::KhrPushDescriptor)),
        e(
            c"VK_EXT_texture_compression_astc_hdr",
            Some(X::ExtTextureCompressionAstcHdr),
        ),
        e(c"VK_IMG_format_pvrtc", Some(X::ImgFormatPvrtc)),
        e(
            c"VK_EXT_border_color_swizzle",
            Some(X::ExtBorderColorSwizzle),
        ),
        e(c"VK_EXT_custom_border_color", Some(X::ExtCustomBorderColor)),
        e(c"VK_EXT_depth_bias_control", Some(X::ExtDepthBiasControl)),
        e(c"VK_EXT_depth_clip_enable", Some(X::ExtDepthClipEnable)),
        e(c"VK_EXT_depth_clamp_control", Some(X::ExtDepthClampControl)),
        e(c"VK_EXT_sample_locations", Some(X::ExtSampleLocations)),
        e(
            c"VK_EXT_primitive_topology_list_restart",
            Some(X::ExtPrimitiveTopologyListRestart),
        ),
        e(c"VK_EXT_provoking_vertex", Some(X::ExtProvokingVertex)),
        e(c"VK_EXT_post_depth_coverage", Some(X::ExtPostDepthCoverage)),
        e(c"VK_EXT_mesh_shader", Some(X::ExtMeshShader)),
        e(c"VK_KHR_shader_float_controls2", None),
        e(c"VK_KHR_maintenance5", Some(X::KhrMaintenance5)),
        e(c"VK_KHR_dynamic_rendering", Some(X::KhrDynamicRendering)),
        e(c"VK_KHR_portability_subset", Some(X::KhrPortabilitySubset)),
        e(c"VK_EXT_device_generated_commands", None),
        e(
            c"VK_KHR_format_feature_flags2",
            Some(X::KhrFormatFeatureFlags2),
        ),
    ]
};

/// The enabled chain and extensions of one context.
pub(super) struct Enables {
    pub(super) chain: FeatureChain,
    /// The extension structures chained at `vkCreateDevice`.
    pub(super) linked: BTreeSet<Structure>,
    /// The device extensions enabled, each with what it requires.
    pub(super) extensions: BTreeSet<&'static CStr>,
    /// The robustness features the device's description states that the stated leg could not
    /// enable (its chain does not report one, or no enabled extension exposes its structure):
    /// no robustness feature is then enabled, and every run on the context is a typed refusal
    /// naming these ([`engine_list`]).
    pub(super) unmet_robustness: BTreeSet<VkFeature>,
    /// The context's API version ([`Available`]'s).
    pub(super) version: ApiVersion,
}

impl Enables {
    fn new(version: ApiVersion) -> Enables {
        Enables {
            chain: FeatureChain::default(),
            linked: BTreeSet::new(),
            extensions: BTreeSet::new(),
            unmet_robustness: BTreeSet::new(),
            version,
        }
    }

    /// Chain `structure` and enable the extension `available` exposes it by (with what that
    /// extension requires), or none where the context's version has it as core. `false` when
    /// the device does not expose it.
    fn expose(&mut self, structure: Structure, available: &Available) -> bool {
        let Some(extension) = available.exposing(structure) else {
            return false;
        };
        if !structure.is_base() {
            self.linked.insert(structure);
        }
        if let Some(name) = extension {
            self.extensions.insert(name);
            self.extensions.extend(structure.requires_at(self.version));
        }
        true
    }

    /// Chain `structure` when the context's version has it as core or an extension already
    /// enabled exposes it (with what that extension requires), enabling nothing more. `false`
    /// when neither does.
    fn link(&mut self, structure: Structure) -> bool {
        if structure.core_at(self.version) {
            if !structure.is_base() {
                self.linked.insert(structure);
            }
            return true;
        }
        let exposed = structure
            .extensions()
            .iter()
            .any(|name| self.extensions.contains(name))
            && structure
                .requires_at(self.version)
                .all(|name| self.extensions.contains(name));
        if exposed {
            self.linked.insert(structure);
        }
        exposed
    }

    /// Enable `member` when `supported` has it and the device exposes its structure.
    fn copy(&mut self, member: &Member, supported: &FeatureChain, available: &Available) {
        if member.is_set(supported) && self.expose(member.structure, available) {
            member.set(&mut self.chain);
        }
    }

    /// The `ppEnabledExtensionNames` pointers, valid for the `'static` names.
    pub(super) fn extension_names(&self) -> Vec<*const c_char> {
        self.extensions.iter().map(|name| name.as_ptr()).collect()
    }
}

/// The member of `MEMBERS` a feature names.
fn member_of(feature: VkFeature) -> Option<&'static Member> {
    MEMBERS
        .iter()
        .find(|member| member.feature == Some(feature))
}

/// The member of `MEMBERS` that is field `name` of `structure`.
fn named_member(structure: Structure, name: &str) -> Option<&'static Member> {
    MEMBERS
        .iter()
        .find(|member| member.structure == structure && member.name == name)
}

/// The old executor's enable list, as it stood before the engine had its own: the members
/// every module under test declared (censused over the corpus), plus the portability subset
/// (everything a portable implementation says it has), `VK_EXT_shader_stencil_export`, and
/// `maintenance5` (which defines an unwritten `PointSize` for the old executor's point draws;
/// its extension requires `VK_KHR_dynamic_rendering` on 1.2). Unchanged since: it is the base of
/// every engine context's list ([`engine_list`]), so a change here moves every opened
/// profile.
pub(super) fn old_engine_list(supported: &FeatureChain, available: &Available) -> Enables {
    let mut enables = Enables::new(available.version);
    for feature in [
        F::SamplerAnisotropy,
        F::SampleRateShading,
        F::TessellationShader,
        F::ShaderInt16,
        F::ShaderInt64,
        F::ShaderClipDistance,
        F::ShaderCullDistance,
        F::ShaderStorageImageExtendedFormats,
        F::ImageCubeArray,
        F::GeometryShader,
        F::FragmentStoresAndAtomics,
        F::VertexPipelineStoresAndAtomics,
        F::StorageInputOutput16,
        F::StorageBuffer16BitAccess,
        F::UniformAndStorageBuffer16BitAccess,
        F::StoragePushConstant16,
        F::StorageBuffer8BitAccess,
        F::UniformAndStorageBuffer8BitAccess,
        F::StoragePushConstant8,
        F::ShaderFloat16,
        F::ShaderInt8,
        F::ShaderSubgroupExtendedTypes,
        F::ShaderOutputViewportIndex,
        F::ShaderOutputLayer,
        F::BufferDeviceAddress,
        F::FragmentShaderPixelInterlock,
        F::ShaderDemoteToHelperInvocation,
        F::ShaderBufferFloat32AtomicAdd,
        F::ShaderSharedFloat32AtomicAdd,
        F::FragmentShaderBarycentric,
    ] {
        let member = member_of(feature).expect("every listed feature has a member");
        enables.copy(member, supported, available);
    }
    // The members the engine's vocabulary does not name.
    for (structure, name) in [
        (Structure::V11, "variable_pointers"),
        (Structure::V11, "variable_pointers_storage_buffer"),
        (Structure::FloatControls2, "shader_float_controls2"),
    ] {
        let member = named_member(structure, name).expect("every named member is listed");
        enables.copy(member, supported, available);
    }
    let maintenance5 = MEMBERS
        .iter()
        .find(|member| member.structure == Structure::Maintenance5)
        .expect("maintenance5 is listed");
    enables.copy(maintenance5, supported, available);
    if enables.expose(Structure::PortabilitySubset, available) {
        // Ask a portable implementation for exactly what it says it has (only the members:
        // `unlink` already cleared the query's link).
        enables.chain.portability_subset = supported.portability_subset;
    }
    let stencil_export = c"VK_EXT_shader_stencil_export";
    if available.has(stencil_export) {
        enables.extensions.insert(stencil_export);
    }
    enables
}

/// Enable `VK_EXT_device_generated_commands` and its `deviceGeneratedCommands` on top of a new
/// engine's `enables`, where the device supports the feature and the list already enables
/// `VK_KHR_maintenance5` (the extension's requirement at API 1.2, and the source of the
/// `VkPipelineCreateFlags2CreateInfo` its pipelines are created with), or, in a context at a
/// version that has it as core (1.4), its `maintenance5`: the device replay then runs the
/// generated-commands path. Whether it did.
pub(super) fn enable_generated_commands(
    enables: &mut Enables,
    supported: &FeatureChain,
    available: &Available,
) -> bool {
    let member = MEMBERS
        .iter()
        .find(|member| member.structure == Structure::GeneratedCommands)
        .expect("deviceGeneratedCommands is listed");
    let maintenance5 = enables.extensions.contains(c"VK_KHR_maintenance5")
        || (Structure::Maintenance5.core_at(enables.version)
            && MEMBERS
                .iter()
                .filter(|member| member.structure == Structure::Maintenance5)
                .all(|member| enabled(enables, member)));
    if !member.is_set(supported) || !maintenance5 {
        return false;
    }
    if !enables.expose(Structure::GeneratedCommands, available) {
        return false;
    }
    member.set(&mut enables.chain);
    true
}

/// What the runtime probe's context enabled beyond the engine's list
/// ([`enable_runtime_probe`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RuntimeProbeExtensions {
    /// `VK_EXT_external_memory_host`: importing host memory without a copy.
    pub(super) external_memory_host: bool,
    /// `VK_EXT_image_drm_format_modifier`: images with an explicit memory layout.
    pub(super) drm_format_modifier: bool,
}

/// Enable what the runtime probe measures, on top of an engine's `enables`, where the device
/// lists it: `VK_EXT_external_memory_host` (which needs `VK_KHR_external_memory`, core from Vulkan
/// 1.1) and `VK_EXT_image_drm_format_modifier` (which needs `VK_KHR_image_format_list` below Vulkan
/// 1.2; its other requirements are core from 1.1). Neither has a feature to enable. Only the
/// probe's own context calls it: no case runs where it enabled anything.
pub(super) fn enable_runtime_probe(
    enables: &mut Enables,
    available: &Available,
) -> RuntimeProbeExtensions {
    let host = ash::ext::external_memory_host::NAME;
    let external_memory_host = available.has(host);
    if external_memory_host {
        enables.extensions.insert(host);
    }
    let modifier = ash::ext::image_drm_format_modifier::NAME;
    let format_list = ash::khr::image_format_list::NAME;
    let list_core = enables.version >= ApiVersion::V1_2;
    let drm_format_modifier = available.has(modifier) && (list_core || available.has(format_list));
    if drm_format_modifier {
        enables.extensions.insert(modifier);
        if !list_core {
            enables.extensions.insert(format_list);
        }
    }
    RuntimeProbeExtensions {
        external_memory_host,
        drm_format_modifier,
    }
}

/// The engine's enable list: on top of `enables`, the product's
/// `DeviceDescription::recommended_enables` for `description` (the device's own queried
/// description, at the context's API 1.2, or the profile a restriction reports) under the default
/// policy the opened device is derived with, less [`NOT_ENABLED`], with the robustness features
/// ([`ROBUSTNESS_FEATURES`]) `leg` decides: every one the description states in
/// [`Leg::Stated`], none in [`Leg::RobustnessOff`]. Each recommended extension is enabled by the
/// first name the device lists for it, each recommended feature by its member (chaining its
/// structure through an enabled extension), and each member no vocabulary names
/// (`textureCompressionASTC_HDR`, `maintenance5`) where an enabled extension exposes its structure
/// and the device supports it; and each [`ExecutorFeature`] where the device supports it.
///
/// A stated robustness feature the feature chain does not report, or whose structure no enabled
/// extension exposes, is not an error: no robustness feature is enabled then, and the stated ones
/// are [`Enables::unmet_robustness`], so the context opens and every run on it is a typed refusal
/// rather than a run of the in-shader encoding in the stated leg. Any other disagreement between
/// the product's description and this device's own lists (a recommended extension with no listed
/// name, a recommended feature with no member, one the feature chain does not report, or one
/// whose structure no enabled extension exposes) is an error: the context is not opened on a list
/// that would silently drop it.
pub(super) fn engine_list(
    mut enables: Enables,
    description: &DeviceDescription,
    supported: &FeatureChain,
    available: &Available,
    leg: Leg,
) -> Result<Enables, String> {
    let recommended = description.recommended_enables(&DevicePolicy::default());
    for &extension in &recommended.extensions {
        let name = EXTENSIONS
            .iter()
            .filter(|named| named.vocabulary == Some(extension))
            .map(|named| named.name)
            .find(|name| available.has(name))
            .ok_or_else(|| format!("the device lists no name of recommended {extension:?}"))?;
        enables.extensions.insert(name);
    }
    let mut robustness = Vec::new();
    for &feature in &recommended.features {
        if NOT_ENABLED.iter().any(|(excluded, _)| *excluded == feature) {
            continue;
        }
        // Core functionality without a member at the context's version (extended dynamic state
        // from 1.3): nothing to enable.
        if !feature.exposure().has_member(enables.version) {
            continue;
        }
        if ROBUSTNESS_FEATURES.contains(&feature) {
            if leg == Leg::Stated {
                robustness.push(feature);
            }
            continue;
        }
        let member = member_of(feature)
            .ok_or_else(|| format!("no member enables recommended {feature:?}"))?;
        if !member.is_set(supported) {
            return Err(format!(
                "the device's description lists {feature:?}, its feature chain does not"
            ));
        }
        if !enables.link(member.structure) {
            return Err(format!(
                "recommended {feature:?} without an enabled extension exposing {:?}",
                member.structure
            ));
        }
        member.set(&mut enables.chain);
    }
    // The stated robustness features, all or none: `robustBufferAccess2` without
    // `robustBufferAccess` is invalid usage, and a context missing any runs nothing anyway.
    let members = robustness
        .iter()
        .map(|&feature| {
            member_of(feature)
                .filter(|member| member.is_set(supported) && enables.link(member.structure))
                .ok_or(feature)
        })
        .collect::<Vec<_>>();
    enables.unmet_robustness = members.iter().filter_map(|member| member.err()).collect();
    if enables.unmet_robustness.is_empty() {
        for member in members.into_iter().flatten() {
            member.set(&mut enables.chain);
        }
    }
    for member in MEMBERS.iter().filter(|member| member.feature.is_none()) {
        let extension_structure = !member.structure.extensions().is_empty();
        if extension_structure && member.is_set(supported) && enables.link(member.structure) {
            member.set(&mut enables.chain);
        }
    }
    for member in MEMBERS.iter().filter(|member| member.executor.is_some()) {
        enables.copy(member, supported, available);
    }
    Ok(enables)
}

/// The executor features `enables` enabled ([`ExecutorFeature`]).
pub(super) fn executor_features(enables: &Enables) -> BTreeSet<ExecutorFeature> {
    MEMBERS
        .iter()
        .filter(|member| enabled(enables, member))
        .filter_map(|member| member.executor)
        .collect()
}

/// Whether the enabled chain of `enables` sets `member`'s bit through a structure it chains.
fn enabled(enables: &Enables, member: &Member) -> bool {
    let chained = match member.structure {
        Structure::Core | Structure::V11 | Structure::V12 => true,
        structure => enables.linked.contains(&structure),
    };
    chained && member.is_set(&enables.chain)
}

/// What `enables` enabled, in the engine's vocabulary: every member with an engine name
/// its chain sets, and every enabled extension the vocabulary names. The queried description is
/// restricted to it (`DeviceDescription::with_enabled`, which refuses a set that is not closed);
/// what a pipeline requires beyond it is reported as not enabled.
pub(super) fn enabled_set(enables: &Enables) -> EnableSet {
    let mut set = EnableSet::new();
    set.features = MEMBERS
        .iter()
        .filter(|member| enabled(enables, member))
        .filter_map(|member| member.feature)
        .collect();
    set.extensions = EXTENSIONS
        .iter()
        .filter(|e| enables.extensions.contains(e.name))
        .filter_map(|e| e.vocabulary)
        .collect();
    set
}

#[cfg(test)]
mod tests;
