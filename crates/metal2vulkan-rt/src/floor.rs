//! The floor: the Vulkan version, extensions, features, limits, queue family and memory types the
//! runtime requires of every device, and the optional features it uses only where a device reports
//! them, each choosing one of two forms.
//!
//! [`PhysicalDevice::check`](crate::PhysicalDevice::check) queries a physical device and decides;
//! the decision is a pure function of what it read, so a test can hand it any set of answers.
//! Which memory types a buffer can use Vulkan answers only for a created device, so
//! [`Device::new`](crate::Device::new) makes that last choice on the device.

use ash::vk;
use metal2vulkan::device::{EnableSet, VkExtension, VkFeature};
use std::ffi::{CStr, CString};
use std::fmt;

/// The Vulkan version the runtime requires of the instance and of every device.
pub const VERSION: u32 = vk::API_VERSION_1_3;

/// The instance extensions the runtime requires: presenting goes through a swapchain on the
/// runtime's own device.
pub const INSTANCE_EXTENSIONS: &[&CStr] = &[ash::khr::surface::NAME];

/// The instance extensions instance creation enables where the loader offers them: the listing of
/// portability devices, which the floor then checks as it checks any device.
pub const OPTIONAL_INSTANCE_EXTENSIONS: &[&CStr] = &[ash::khr::portability_enumeration::NAME];

/// The device extensions the runtime requires and device creation enables: presenting, null
/// descriptors and image reads out of range answered by the device, and the GPU's clock read with
/// the host's, for GPU times on the host clock ([`host_time`](crate::host_time)).
pub const DEVICE_EXTENSIONS: &[&CStr] = &[
    ash::khr::swapchain::NAME,
    ash::ext::robustness2::NAME,
    ash::ext::calibrated_timestamps::NAME,
];

/// The time domains the floor requires the device to calibrate together: its own clock, and the
/// host clock of [`host_time`](crate::host_time) ([`HOST_DOMAIN`](crate::HOST_DOMAIN)).
pub const TIME_DOMAINS: [vk::TimeDomainKHR; 2] =
    [vk::TimeDomainKHR::DEVICE, crate::times::HOST_DOMAIN];

/// The device extensions device creation enables where the device lists them: the SIMD groups'
/// reconvergence modes, each for the optional features of its own, the portability subset, which
/// Vulkan requires enabled where a device lists it, with the portability features the floor asks
/// of such a device, and the vertex attribute divisor's two extensions, for the step-rate feature
/// of their own: the KHR, and the EXT it was promoted from, whose feature and pipeline structures
/// are the same and which a device may list alone.
pub const OPTIONAL_EXTENSIONS: &[&CStr] = &[
    ash::khr::shader_maximal_reconvergence::NAME,
    ash::khr::shader_subgroup_uniform_control_flow::NAME,
    ash::khr::portability_subset::NAME,
    ash::khr::vertex_attribute_divisor::NAME,
    ash::ext::vertex_attribute_divisor::NAME,
];

/// The device extensions of the attachment feedback loop (`VK_EXT_attachment_feedback_loop_layout`
/// and its dynamic state), through which a draw reads, through a texture, an attachment its own
/// pass renders (Metal's texture read of a render target): device creation enables both, with
/// their features, where the device reads attachments (`Capabilities::attachment_reads`), lists
/// both and reports both features, and neither otherwise. A device that reads attachments without
/// them reads through a copy of the image (`Capabilities::feedback_loop`).
pub const ATTACHMENT_READ_EXTENSIONS: &[&CStr] = &[
    ash::ext::attachment_feedback_loop_layout::NAME,
    ash::ext::attachment_feedback_loop_dynamic_state::NAME,
];

/// The translator's name for an extension device creation enables that its shaders may use, or
/// that its description of the device must enable: the portability subset, where the device lists
/// it.
fn shader_extension(name: &CStr) -> Option<VkExtension> {
    if name == ash::khr::shader_maximal_reconvergence::NAME {
        Some(VkExtension::KhrShaderMaximalReconvergence)
    } else if name == ash::khr::shader_subgroup_uniform_control_flow::NAME {
        Some(VkExtension::KhrShaderSubgroupUniformControlFlow)
    } else if name == ash::ext::robustness2::NAME {
        Some(VkExtension::ExtRobustness2)
    } else if name == ash::khr::portability_subset::NAME {
        Some(VkExtension::KhrPortabilitySubset)
    } else if name == ash::khr::vertex_attribute_divisor::NAME {
        Some(VkExtension::KhrVertexAttributeDivisor)
    } else if name == ash::ext::vertex_attribute_divisor::NAME {
        Some(VkExtension::ExtVertexAttributeDivisor)
    } else {
        None
    }
}

/// Slots in each texture array, the sampled and the storage: the most textures alive at once,
/// each at its index in both. Vulkan's least update-after-bind sampled and storage images, per
/// stage and per set, is 500,000 on a device that supports `descriptorIndexing`.
pub(crate) const TEXTURES: u32 = 65_536;

/// Slots in each sampler array. 4000 is Vulkan's minimum `maxSamplerAllocationCount`, so every
/// device can hold a sampler in every slot.
pub(crate) const SAMPLERS: u32 = 4000;

/// Sampler arrays: one per use of a sampler (plain, depth comparison, integer texture).
pub(crate) const SAMPLER_ARRAYS: u32 = 3;

/// Bytes of push constants: a dispatch region and 28 buffer addresses (the translator's push
/// window on `Device::from_queries`' profile), which every captured device reports, so no query
/// decides it.
pub(crate) const PUSH_CONSTANT_BYTES: u32 = 256;

/// Threads in one compute workgroup: Metal's most threads per threadgroup.
pub(crate) const WORKGROUP_INVOCATIONS: u32 = 1024;

/// A compute workgroup's largest extent in each dimension. Metal allows 1024 threads along any
/// axis; 64 along the third is the least a lab GPU offers.
pub(crate) const WORKGROUP_SIZE: [u32; 3] = [1024, 1024, 64];

/// The least `timestampValidBits` Vulkan allows a queue family that has timestamps.
pub(crate) const TIMESTAMP_BITS: u32 = 36;

/// The usage every buffer the runtime makes is created with: each buffer usage Vulkan 1.3 defines,
/// and device addresses. A memory type that holds a buffer of these usages holds a buffer of any
/// subset of them.
pub const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

/// The flags of the memory type Private storage lives in: memory the GPU reads at its own speed.
/// Vulkan guarantees a buffer one such type.
pub(crate) const PRIVATE_MEMORY: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

/// The flags of the memory type Shared storage lives in: memory the host maps and reads at the
/// speed of its own memory.
pub(crate) const SHARED_MEMORY: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw()
        | vk::MemoryPropertyFlags::HOST_CACHED.as_raw(),
);

/// The flags of the memory type the parameter ring lives in: memory the GPU reads at its own speed
/// and the host writes without a flush.
pub(crate) const RING_MEMORY: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw()
        | vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);

/// The Vulkan names of the memory property flags the floor asks for.
pub(crate) const MEMORY_FLAG_NAMES: [(vk::MemoryPropertyFlags, &str); 4] = [
    (
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT",
    ),
    (
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",
    ),
    (
        vk::MemoryPropertyFlags::HOST_COHERENT,
        "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT",
    ),
    (
        vk::MemoryPropertyFlags::HOST_CACHED,
        "VK_MEMORY_PROPERTY_HOST_CACHED_BIT",
    ),
];

/// The feature structures of Vulkan 1.3 that the floor reads and device creation enables, each
/// without its `p_next` link.
#[derive(Clone, Copy, Default)]
pub struct Features {
    /// `VkPhysicalDeviceFeatures`.
    pub core: vk::PhysicalDeviceFeatures,
    /// `VkPhysicalDeviceVulkan11Features`.
    pub vulkan11: vk::PhysicalDeviceVulkan11Features<'static>,
    /// `VkPhysicalDeviceVulkan12Features`.
    pub vulkan12: vk::PhysicalDeviceVulkan12Features<'static>,
    /// `VkPhysicalDeviceVulkan13Features`.
    pub vulkan13: vk::PhysicalDeviceVulkan13Features<'static>,
    /// `VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR`, of
    /// `VK_KHR_shader_maximal_reconvergence`: read and enabled only where the device lists it.
    pub maximal_reconvergence: vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR<'static>,
    /// `VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR`, of
    /// `VK_KHR_shader_subgroup_uniform_control_flow`: read and enabled only where the device lists
    /// it.
    pub subgroup_uniform_control_flow:
        vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR<'static>,
    /// `VkPhysicalDeviceRobustness2FeaturesEXT`, of `VK_EXT_robustness2`: `nullDescriptor` and
    /// `robustImageAccess2` are enabled, never the robust buffer access beside them.
    pub robustness2: vk::PhysicalDeviceRobustness2FeaturesEXT<'static>,
    /// `VkPhysicalDevicePortabilitySubsetFeaturesKHR`, of `VK_KHR_portability_subset`: read and
    /// enabled only where the device lists it, and then only the portability features the floor asks
    /// of such a device.
    pub portability: vk::PhysicalDevicePortabilitySubsetFeaturesKHR<'static>,
    /// `VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT`, of
    /// `VK_EXT_attachment_feedback_loop_layout`: read where the device lists it, and enabled with
    /// the other attachment-read features alone.
    pub feedback_loop: vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT<'static>,
    /// `VkPhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT`, of
    /// `VK_EXT_attachment_feedback_loop_dynamic_state`: likewise.
    pub feedback_loop_dynamic:
        vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT<'static>,
    /// `VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR`, of `VK_KHR_vertex_attribute_divisor`
    /// and `VK_EXT_vertex_attribute_divisor` (the same structure): read and enabled only where the
    /// device lists one of them, and then only the step-rate feature.
    pub divisor: vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR<'static>,
}

/// One feature: the Vulkan structure that reports it, its member as ash spells it, the member, and
/// the translator's name for it where the translator's shaders may use it.
pub(crate) struct Feature {
    pub(crate) structure: &'static str,
    pub(crate) member: &'static str,
    pub(crate) field: fn(&mut Features) -> &mut vk::Bool32,
    pub(crate) translator: Option<VkFeature>,
}

macro_rules! feature {
    (@structure core) => {
        "VkPhysicalDeviceFeatures"
    };
    (@structure vulkan11) => {
        "VkPhysicalDeviceVulkan11Features"
    };
    (@structure vulkan12) => {
        "VkPhysicalDeviceVulkan12Features"
    };
    (@structure vulkan13) => {
        "VkPhysicalDeviceVulkan13Features"
    };
    (@structure maximal_reconvergence) => {
        "VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR"
    };
    (@structure subgroup_uniform_control_flow) => {
        "VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR"
    };
    (@structure robustness2) => {
        "VkPhysicalDeviceRobustness2FeaturesEXT"
    };
    (@structure portability) => {
        "VkPhysicalDevicePortabilitySubsetFeaturesKHR"
    };
    (@structure feedback_loop) => {
        "VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT"
    };
    (@structure feedback_loop_dynamic) => {
        "VkPhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT"
    };
    (@structure divisor) => {
        "VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR"
    };
    (@translator) => {
        None
    };
    (@translator $name:ident) => {
        Some(VkFeature::$name)
    };
    ($part:ident . $member:ident $(=> $name:ident)?) => {
        Feature {
            structure: feature!(@structure $part),
            member: stringify!($member),
            field: |features| &mut features.$part.$member,
            translator: feature!(@translator $($name)?),
        }
    };
}

/// The features the runtime requires.
pub(crate) const REQUIRED: &[Feature] = &[
    // A buffer's address is its device address, used directly by shaders.
    feature!(vulkan12.buffer_device_address => BufferDeviceAddress),
    feature!(core.shader_int64 => ShaderInt64),
    // One descriptor array per image class, indexed by a resource ID read at run time.
    feature!(vulkan12.runtime_descriptor_array => RuntimeDescriptorArray),
    feature!(vulkan12.descriptor_binding_partially_bound => DescriptorBindingPartiallyBound),
    feature!(vulkan12.descriptor_binding_update_unused_while_pending => DescriptorBindingUpdateUnusedWhilePending),
    feature!(vulkan12.descriptor_binding_sampled_image_update_after_bind => DescriptorBindingSampledImageUpdateAfterBind),
    feature!(vulkan12.descriptor_binding_storage_image_update_after_bind => DescriptorBindingStorageImageUpdateAfterBind),
    feature!(core.shader_sampled_image_array_dynamic_indexing => ShaderSampledImageArrayDynamicIndexing),
    feature!(core.shader_storage_image_array_dynamic_indexing => ShaderStorageImageArrayDynamicIndexing),
    feature!(vulkan12.shader_sampled_image_array_non_uniform_indexing => ShaderSampledImageArrayNonUniformIndexing),
    feature!(vulkan12.shader_storage_image_array_non_uniform_indexing => ShaderStorageImageArrayNonUniformIndexing),
    feature!(core.shader_storage_image_write_without_format => ShaderStorageImageWriteWithoutFormat),
    // A sampler state's anisotropy is the device's own filter.
    feature!(core.sampler_anisotropy => SamplerAnisotropy),
    // Every array element no texture owns is a null descriptor: a nil, freed or unowned texture
    // index reads zeros, answers 0 to its size queries and drops its writes, as Metal's nil
    // texture does, with no code in the shader.
    feature!(robustness2.null_descriptor => NullDescriptor),
    // The translator's shaders: 8-, 16- and 64-bit integers, 8- and 16-bit values in buffers,
    // scalar and std430 layout of buffer, uniform and push-constant blocks, and the Vulkan memory
    // model with device scope for atomics and barriers. Threadgroup memory is implicitly laid
    // out (plain arrays and scalars), which needs no feature.
    feature!(core.shader_int16 => ShaderInt16),
    feature!(vulkan12.shader_int8 => ShaderInt8),
    feature!(vulkan11.storage_buffer16_bit_access => StorageBuffer16BitAccess),
    feature!(vulkan12.storage_buffer8_bit_access => StorageBuffer8BitAccess),
    feature!(vulkan12.scalar_block_layout => ScalarBlockLayout),
    feature!(vulkan12.uniform_buffer_standard_layout => UniformBufferStandardLayout),
    feature!(vulkan12.vulkan_memory_model => VulkanMemoryModel),
    feature!(vulkan12.vulkan_memory_model_device_scope => VulkanMemoryModelDeviceScope),
    // SIMD groups at Metal's width of 32: a kernel's subgroup size pinned where the device lists
    // compute in `requiredSubgroupSizeStages`, and its subgroups required full. Both are Vulkan
    // 1.3's.
    feature!(vulkan13.subgroup_size_control => SubgroupSizeControl),
    feature!(vulkan13.compute_full_subgroups => ComputeFullSubgroups),
    // An image access out of range: a write or an atomic is dropped and a read answers zero, as
    // Metal's, with no guard in the shader. Vulkan 1.3's.
    feature!(vulkan13.robust_image_access => RobustImageAccess),
    // An image read out of range answers Metal's texel exactly, alpha 1 for a format without
    // alpha, a fetch past the view's levels included, so a read takes no check in the shader.
    feature!(robustness2.robust_image_access2 => RobustImageAccess2),
    // A render pipeline's color attachments blend each by its own state, as Metal's do, and a
    // fragment function writes and atomically updates device memory, as Metal's may. Vulkan 1.4
    // requires both of every graphics device.
    feature!(core.independent_blend => IndependentBlend),
    feature!(core.fragment_stores_and_atomics => FragmentStoresAndAtomics),
    // What Vulkan 1.1 to 1.3 require of every device and the translator's shaders use where they
    // are enabled: multiview, subgroup operations on 8-, 16- and 64-bit values, a fragment's
    // demotion to a helper and its termination, and workgroup memory zeroed at its declaration.
    feature!(vulkan11.multiview => Multiview),
    feature!(vulkan12.shader_subgroup_extended_types => ShaderSubgroupExtendedTypes),
    feature!(vulkan13.shader_demote_to_helper_invocation => ShaderDemoteToHelperInvocation),
    feature!(vulkan13.shader_terminate_invocation => ShaderTerminateInvocation),
    feature!(vulkan13.shader_zero_initialize_workgroup_memory => ShaderZeroInitializeWorkgroupMemory),
    // Submission, barriers and render passes.
    feature!(vulkan12.timeline_semaphore),
    // GPU times: a command buffer's two timestamp queries reset on the host as it retires, so no
    // command buffer the GPU runs resets a query. Vulkan 1.2 requires it.
    feature!(vulkan12.host_query_reset),
    feature!(vulkan13.synchronization2 => Synchronization2),
    feature!(vulkan13.dynamic_rendering),
];

/// The features of the portability subset the runtime requires of a device that lists
/// `VK_KHR_portability_subset`. A device that does not list it offers each of them by Vulkan's
/// definition, so it is not asked.
pub(crate) const PORTABILITY: &[Feature] = &[
    // A texture's view maps its format's components and Metal's swizzle.
    feature!(portability.image_view_format_swizzle => ImageViewFormatSwizzle),
    // The comparison sampler array takes each sampler state's sampler as it is made.
    feature!(portability.mutable_comparison_samplers => MutableComparisonSamplers),
    // `MTLTextureType2DMultisampleArray`.
    feature!(portability.multisample_array_image => MultisampleArrayImage),
];

/// The features the runtime uses only where the device reports them. Each answer chooses one of
/// two forms; no other feature does.
pub(crate) const OPTIONAL: &[Feature] = &[
    feature!(vulkan12.shader_float16 => ShaderFloat16),
    feature!(core.shader_storage_image_read_without_format => ShaderStorageImageReadWithoutFormat),
    feature!(vulkan12.shader_buffer_int64_atomics => ShaderBufferInt64Atomics),
    feature!(vulkan12.shader_shared_int64_atomics => ShaderSharedInt64Atomics),
    feature!(core.shader_float64 => ShaderFloat64),
    // The SIMD groups' reconvergence: maximal where the device offers it, else subgroup-uniform
    // control flow where it offers that.
    feature!(maximal_reconvergence.shader_maximal_reconvergence => ShaderMaximalReconvergence),
    feature!(subgroup_uniform_control_flow.shader_subgroup_uniform_control_flow => ShaderSubgroupUniformControlFlow),
];

/// The features of the attachment feedback loop ([`ATTACHMENT_READ_EXTENSIONS`]): enabled together
/// where the device reads attachments and reports both, and neither otherwise; a device that reads
/// attachments without them reads through a copy of the image.
pub(crate) const ATTACHMENT_READS: &[Feature] = &[
    feature!(feedback_loop.attachment_feedback_loop_layout),
    feature!(feedback_loop_dynamic.attachment_feedback_loop_dynamic_state),
];

/// The features device creation enables where the device reports them, for Metal's per-instance
/// step at a rate above 1, which the translator builds in one form, a binding's divisor, and
/// refuses on a device without them (`Feature::VertexStepRate`). No answer chooses a second form.
pub(crate) const STEP_RATES: &[Feature] = &[feature!(
    divisor.vertex_attribute_instance_rate_divisor => VertexAttributeInstanceRateDivisor
)];

/// What the translator's shaders may use of what device creation enables: each enabled feature of
/// `capabilities` the translator names, and each enabled extension it names.
pub(crate) fn shader_enables(capabilities: &Capabilities) -> EnableSet {
    let mut features = capabilities.features;
    let mut enables = EnableSet::new();
    for feature in (REQUIRED.iter().chain(OPTIONAL))
        .chain(PORTABILITY)
        .chain(STEP_RATES)
    {
        if let Some(name) = feature.translator {
            if *(feature.field)(&mut features) == vk::TRUE {
                enables.features.insert(name);
            }
        }
    }
    enables.extensions.extend(
        capabilities
            .extensions
            .iter()
            .filter_map(|&name| shader_extension(name)),
    );
    enables
}

/// The floor's description of a feature the consumer names: its structure, member and field.
pub(crate) fn consumer_feature(feature: crate::ConsumerFeature) -> Feature {
    match feature {
        crate::ConsumerFeature::SamplerYcbcrConversion => {
            feature!(vulkan11.sampler_ycbcr_conversion)
        }
    }
}

/// What [`check`] reads from a physical device.
pub(crate) struct Queried {
    /// `VkPhysicalDeviceProperties::apiVersion`.
    pub(crate) version: u32,
    pub(crate) limits: vk::PhysicalDeviceLimits,
    pub(crate) vulkan12: vk::PhysicalDeviceVulkan12Properties<'static>,
    pub(crate) features: Features,
    pub(crate) extensions: Vec<CString>,
    pub(crate) families: Vec<vk::QueueFamilyProperties>,
    pub(crate) memory: vk::PhysicalDeviceMemoryProperties,
    /// The time domains the device calibrates together, read where it lists
    /// `VK_EXT_calibrated_timestamps`.
    pub(crate) time_domains: Vec<vk::TimeDomainKHR>,
}

/// One item of the floor that a device lacks.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Missing {
    /// The device's Vulkan version (`apiVersion`) is below 1.3. The other items are read from
    /// structures Vulkan 1.3 defines, so such a device is named by its version alone.
    Version(u32),
    /// A device extension.
    Extension(&'static CStr),
    /// A feature: the Vulkan structure that reports it, and its member's Vulkan name.
    Feature {
        /// The structure, such as `VkPhysicalDeviceVulkan12Features`.
        structure: &'static str,
        /// The member, such as `bufferDeviceAddress`.
        member: String,
    },
    /// A limit below what the runtime uses. A Boolean limit reads as 0 or 1, and the queue
    /// family's `timestampValidBits` is one of these.
    Limit {
        /// The limit's Vulkan name.
        name: &'static str,
        /// The device's value.
        device: u32,
        /// The least value the runtime uses.
        needed: u32,
    },
    /// A queue family with both `VK_QUEUE_GRAPHICS_BIT` and `VK_QUEUE_COMPUTE_BIT`.
    QueueFamily,
    /// A memory type with these property flags, as Vulkan names them, for buffers of every usage
    /// the runtime gives them. [`BUFFER_USAGE`] is that usage.
    MemoryType {
        /// The flags, joined by ` | `.
        flags: String,
    },
    /// A time domain of [`TIME_DOMAINS`] the device does not calibrate
    /// (`vkGetPhysicalDeviceCalibrateableTimeDomainsEXT`), as Vulkan names it.
    TimeDomain(&'static str),
    /// The layout of one of the runtime's descriptor sets, `set` 0 (empty) or 1 (the texture and
    /// sampler arrays), which `vkGetDescriptorSetLayoutSupport` reports unsupported. Vulkan
    /// answers it only for a created device, so `Device::new` asks it, and
    /// `PhysicalDevice::check` cannot.
    SetLayout {
        /// The set.
        set: u32,
    },
}

impl fmt::Display for Missing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Missing::Version(version) => write!(
                f,
                "Vulkan {}.{}.{}, below 1.3",
                vk::api_version_major(*version),
                vk::api_version_minor(*version),
                vk::api_version_patch(*version)
            ),
            Missing::Extension(name) => write!(f, "{}", name.to_string_lossy()),
            Missing::Feature { structure, member } => write!(f, "{structure}.{member}"),
            Missing::Limit {
                name,
                device,
                needed,
            } => write!(f, "{name} {device}, below {needed}"),
            Missing::QueueFamily => f.write_str("a queue family with graphics and compute"),
            Missing::MemoryType { flags } => write!(f, "a memory type with {flags} for buffers"),
            Missing::TimeDomain(name) => write!(f, "the calibrateable time domain {name}"),
            Missing::SetLayout { set } => write!(
                f,
                "support for descriptor set {set}'s layout (vkGetDescriptorSetLayoutSupport)"
            ),
        }
    }
}

/// A device below the runtime's floor, with every item it lacks.
///
/// Metal has no counterpart: every Metal device runs every Metal program its family supports. The
/// runtime refuses a Vulkan device below its floor at creation, instead of refusing work later.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BelowFloor {
    /// Each item the device lacks, in the floor's order: the version, extensions, features,
    /// limits, the queue family, memory types, then time domains; or, from `Device::new`, a set
    /// layout.
    pub missing: Vec<Missing>,
}

impl fmt::Display for BelowFloor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the device is below the runtime's floor: it lacks ")?;
        for (index, item) in self.missing.iter().enumerate() {
            if index > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

impl std::error::Error for BelowFloor {}

/// What a device above the floor offers: the answers that choose between two forms, and what
/// device creation enables.
///
/// Metal's counterparts are per-family capabilities (`supportsFamily`), and for storage reads
/// `readWriteTextureSupport`; here each is one Vulkan feature. The SIMD groups' other answers
/// (subgroup sizes, stages and operations) are properties the translator reads itself.
pub struct Capabilities {
    /// `shaderFloat16`: `half` arithmetic in 16 bits.
    pub float16: bool,
    /// `shaderStorageImageReadWithoutFormat`: storage-image reads without a declared format, on
    /// every format. Where it is false, a format's `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT`
    /// (`VkFormatProperties3`) answers for that format.
    pub storage_read_without_format: bool,
    /// `shaderBufferInt64Atomics`: 64-bit integer atomics on buffers.
    pub buffer_int64_atomics: bool,
    /// `shaderSharedInt64Atomics`: 64-bit integer atomics on threadgroup memory.
    pub shared_int64_atomics: bool,
    /// `shaderFloat64`: 64-bit floats.
    pub float64: bool,
    /// `shaderMaximalReconvergence`: SIMD groups reconverge maximally, as Metal's do.
    pub maximal_reconvergence: bool,
    /// `shaderSubgroupUniformControlFlow`: SIMD groups reconverge where control flow is uniform,
    /// the mode where maximal reconvergence is not offered.
    pub subgroup_uniform_control_flow: bool,
    /// The device's descriptor limits hold the sampled-texture array's second elements beside the
    /// floor's arrays: a draw may read, through a texture, an attachment its own pass renders, as
    /// Metal reads a render target. It reads through the attachment feedback loop where
    /// [`feedback_loop`](Self::feedback_loop) holds, and through a copy of the image otherwise.
    pub attachment_reads: bool,
    /// The attachment reads hold, and so does every attachment-read feature, with
    /// [`ATTACHMENT_READ_EXTENSIONS`]: a draw reads an attachment through the attachment feedback
    /// loop.
    pub feedback_loop: bool,
    /// The features device creation enables: every one the floor requires, and each of the
    /// optional and step-rate ones the device reports.
    pub features: Features,
    /// The device extensions device creation enables: [`DEVICE_EXTENSIONS`], then each of
    /// [`OPTIONAL_EXTENSIONS`] the device lists, then [`ATTACHMENT_READ_EXTENSIONS`] where
    /// [`feedback_loop`](Self::feedback_loop) holds.
    pub extensions: Vec<&'static CStr>,
    /// The first queue family with graphics and compute, whose timestamps the floor checked.
    pub queue_family: u32,
    /// The device's memory types and heaps, for device creation to choose from.
    pub memory: vk::PhysicalDeviceMemoryProperties,
}

/// The memory types the runtime allocates its buffers from, each the first memory type with its
/// flags that a buffer of [`BUFFER_USAGE`] can use.
///
/// Metal's counterparts are storage modes: `MTLStorageModePrivate` and `MTLStorageModeShared`
/// memory, and the memory `setBytes:length:atIndex:` copies into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryTypes {
    /// Device-local: Private storage.
    pub private: u32,
    /// Host-visible, host-coherent and host-cached: Shared storage.
    pub shared: u32,
    /// Device-local, host-visible and host-coherent: the parameter ring.
    pub ring: u32,
}

/// Checks a physical device against the floor.
///
/// Returns what the device offers above the floor, or every item of the floor it lacks. For memory
/// it checks only that a type with each set of flags exists; [`memory_types`] finishes the check
/// on the created device.
///
/// Metal has no counterpart; see [`BelowFloor`].
///
/// # Safety
///
/// `physical` is a physical device `instance` enumerated, and `instance` was created with an
/// `apiVersion` of [`VERSION`] or later.
pub(crate) unsafe fn check(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Result<Capabilities, BelowFloor> {
    // SAFETY: the caller's contract.
    decide(&unsafe { query(entry, instance, physical) })
}

/// Reads what [`decide`] needs. A device below Vulkan 1.3 gets its version read alone.
///
/// # Safety
///
/// As [`check`].
unsafe fn query(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Queried {
    // SAFETY: the caller's contract; each structure below is one Vulkan 1.3 defines, read only
    // from a device of Vulkan 1.3 or later, or one of an extension the device lists.
    unsafe {
        let version = instance
            .get_physical_device_properties(physical)
            .api_version;
        let mut queried = Queried {
            version,
            limits: vk::PhysicalDeviceLimits::default(),
            vulkan12: vk::PhysicalDeviceVulkan12Properties::default(),
            features: Features::default(),
            extensions: Vec::new(),
            families: Vec::new(),
            memory: vk::PhysicalDeviceMemoryProperties::default(),
            time_domains: Vec::new(),
        };
        if version < VERSION {
            return queried;
        }
        let mut properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut queried.vulkan12);
        instance.get_physical_device_properties2(physical, &mut properties);
        queried.limits = properties.properties.limits;
        queried.vulkan12.p_next = std::ptr::null_mut();

        // A device whose extensions cannot be listed offers none the floor can rely on.
        queried.extensions = instance
            .enumerate_device_extension_properties(physical)
            .unwrap_or_default()
            .iter()
            .filter_map(|extension| extension.extension_name_as_c_str().ok())
            .map(CStr::to_owned)
            .collect();
        // An extension's feature structure may be read only from a device that lists the
        // extension; on another, its features stay false.
        let listed = |extension: &CStr| {
            queried
                .extensions
                .iter()
                .any(|name| name.as_c_str() == extension)
        };
        let maximal_reconvergence_listed = listed(ash::khr::shader_maximal_reconvergence::NAME);
        let uniform_control_flow_listed =
            listed(ash::khr::shader_subgroup_uniform_control_flow::NAME);
        let robustness2_listed = listed(ash::ext::robustness2::NAME);
        let portability_listed = listed(ash::khr::portability_subset::NAME);
        let feedback_loop_listed = listed(ash::ext::attachment_feedback_loop_layout::NAME);
        let feedback_loop_dynamic_listed =
            listed(ash::ext::attachment_feedback_loop_dynamic_state::NAME);
        let divisor_listed = listed(ash::khr::vertex_attribute_divisor::NAME)
            || listed(ash::ext::vertex_attribute_divisor::NAME);

        let Features {
            core,
            vulkan11,
            vulkan12,
            vulkan13,
            maximal_reconvergence,
            subgroup_uniform_control_flow,
            robustness2,
            portability,
            feedback_loop,
            feedback_loop_dynamic,
            divisor,
        } = &mut queried.features;
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut *vulkan11)
            .push_next(&mut *vulkan12)
            .push_next(&mut *vulkan13);
        if maximal_reconvergence_listed {
            features = features.push_next(&mut *maximal_reconvergence);
        }
        if uniform_control_flow_listed {
            features = features.push_next(&mut *subgroup_uniform_control_flow);
        }
        if robustness2_listed {
            features = features.push_next(&mut *robustness2);
        }
        if portability_listed {
            features = features.push_next(&mut *portability);
        }
        if feedback_loop_listed {
            features = features.push_next(&mut *feedback_loop);
        }
        if feedback_loop_dynamic_listed {
            features = features.push_next(&mut *feedback_loop_dynamic);
        }
        if divisor_listed {
            features = features.push_next(&mut *divisor);
        }
        instance.get_physical_device_features2(physical, &mut features);
        *core = features.features;
        vulkan11.p_next = std::ptr::null_mut();
        vulkan12.p_next = std::ptr::null_mut();
        vulkan13.p_next = std::ptr::null_mut();
        maximal_reconvergence.p_next = std::ptr::null_mut();
        subgroup_uniform_control_flow.p_next = std::ptr::null_mut();
        robustness2.p_next = std::ptr::null_mut();
        portability.p_next = std::ptr::null_mut();
        feedback_loop.p_next = std::ptr::null_mut();
        feedback_loop_dynamic.p_next = std::ptr::null_mut();
        divisor.p_next = std::ptr::null_mut();

        queried.families = instance.get_physical_device_queue_family_properties(physical);
        queried.memory = instance.get_physical_device_memory_properties(physical);
        // A device whose time domains cannot be listed calibrates none.
        if listed(ash::ext::calibrated_timestamps::NAME) {
            queried.time_domains = ash::ext::calibrated_timestamps::Instance::new(entry, instance)
                .get_physical_device_calibrateable_time_domains(physical)
                .unwrap_or_default();
        }
        queried
    }
}

/// Decides from what [`query`] read.
pub(crate) fn decide(device: &Queried) -> Result<Capabilities, BelowFloor> {
    if device.version < VERSION {
        return Err(BelowFloor {
            missing: vec![Missing::Version(device.version)],
        });
    }
    let mut missing = Vec::new();
    for &extension in DEVICE_EXTENSIONS {
        if !device
            .extensions
            .iter()
            .any(|name| name.as_c_str() == extension)
        {
            missing.push(Missing::Extension(extension));
        }
    }
    let mut reported = device.features;
    let mut features = Features::default();
    let portability = device
        .extensions
        .iter()
        .any(|name| name.as_c_str() == ash::khr::portability_subset::NAME);
    let portability: &[Feature] = if portability { PORTABILITY } else { &[] };
    for feature in REQUIRED.iter().chain(portability) {
        if *(feature.field)(&mut reported) == vk::TRUE {
            *(feature.field)(&mut features) = vk::TRUE;
        } else {
            missing.push(Missing::Feature {
                structure: feature.structure,
                member: vulkan_name(feature.member),
            });
        }
    }
    for (name, value, needed) in limits(device) {
        if value < needed {
            missing.push(Missing::Limit {
                name,
                device: value,
                needed,
            });
        }
    }
    let family = (0u32..).zip(&device.families).find(|(_, family)| {
        family
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
    });
    match family {
        None => missing.push(Missing::QueueFamily),
        Some((_, family)) if family.timestamp_valid_bits < TIMESTAMP_BITS => {
            missing.push(Missing::Limit {
                name: "timestampValidBits",
                device: family.timestamp_valid_bits,
                needed: TIMESTAMP_BITS,
            })
        }
        Some(_) => {}
    }
    // Every memory type counts as one a buffer can use until the device answers.
    if let Err(lacking) = choose(&device.memory, u32::MAX) {
        missing.extend(lacking);
    }
    for domain in TIME_DOMAINS {
        if !device.time_domains.contains(&domain) {
            missing.push(Missing::TimeDomain(time_domain_name(domain)));
        }
    }
    let Some((queue_family, _)) = family.filter(|_| missing.is_empty()) else {
        return Err(BelowFloor { missing });
    };
    for feature in OPTIONAL.iter().chain(STEP_RATES) {
        *(feature.field)(&mut features) = *(feature.field)(&mut reported);
    }
    let listed =
        |extension: &CStr| (device.extensions.iter()).any(|name| name.as_c_str() == extension);
    let attachment_reads = second_elements_fit(device);
    let feedback_loop = attachment_reads
        && ATTACHMENT_READ_EXTENSIONS.iter().all(|&name| listed(name))
        && (ATTACHMENT_READS.iter()).all(|feature| *(feature.field)(&mut reported) == vk::TRUE);
    if feedback_loop {
        for feature in ATTACHMENT_READS {
            *(feature.field)(&mut features) = vk::TRUE;
        }
    }
    Ok(Capabilities {
        float16: features.vulkan12.shader_float16 == vk::TRUE,
        storage_read_without_format: features.core.shader_storage_image_read_without_format
            == vk::TRUE,
        buffer_int64_atomics: features.vulkan12.shader_buffer_int64_atomics == vk::TRUE,
        shared_int64_atomics: features.vulkan12.shader_shared_int64_atomics == vk::TRUE,
        float64: features.core.shader_float64 == vk::TRUE,
        maximal_reconvergence: features.maximal_reconvergence.shader_maximal_reconvergence
            == vk::TRUE,
        subgroup_uniform_control_flow: features
            .subgroup_uniform_control_flow
            .shader_subgroup_uniform_control_flow
            == vk::TRUE,
        attachment_reads,
        feedback_loop,
        features,
        extensions: DEVICE_EXTENSIONS
            .iter()
            .chain(
                OPTIONAL_EXTENSIONS
                    .iter()
                    .filter(|&&extension| listed(extension)),
            )
            .chain(ATTACHMENT_READ_EXTENSIONS.iter().filter(|_| feedback_loop))
            .copied()
            .collect(),
        queue_family,
        memory: device.memory,
    })
}

/// Chooses the memory types on a device created from `capabilities`.
///
/// Vulkan answers which memory types a buffer can use only for a created device, so this is the
/// floor's last step: it asks once for a buffer of [`BUFFER_USAGE`] and takes, for each of
/// [`MemoryTypes`], the first memory type with its flags that such a buffer can use. A device whose
/// buffers can use none of the types [`check`] found is below the floor.
///
/// Metal has no counterpart; see [`BelowFloor`].
///
/// # Safety
///
/// `device` was created from the physical device `capabilities` came from, with
/// `capabilities.features` enabled.
pub(crate) unsafe fn memory_types(
    device: &ash::Device,
    capabilities: &Capabilities,
) -> Result<MemoryTypes, BelowFloor> {
    let buffer = vk::BufferCreateInfo::default()
        .size(65_536)
        .usage(BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let question = vk::DeviceBufferMemoryRequirements::default().create_info(&buffer);
    let mut answer = vk::MemoryRequirements2::default();
    // SAFETY: the caller's contract; `BUFFER_USAGE`'s device addresses need
    // `bufferDeviceAddress`, which the floor requires and `capabilities.features` enables.
    unsafe { device.get_device_buffer_memory_requirements(&question, &mut answer) };
    choose(
        &capabilities.memory,
        answer.memory_requirements.memory_type_bits,
    )
    .map_err(|missing| BelowFloor { missing })
}

/// For each of [`MemoryTypes`], the first memory type with its flags whose bit is set in `usable`,
/// or a [`Missing::MemoryType`] for each that has none.
pub(crate) fn choose(
    memory: &vk::PhysicalDeviceMemoryProperties,
    usable: u32,
) -> Result<MemoryTypes, Vec<Missing>> {
    let first = |flags: vk::MemoryPropertyFlags| {
        (0u32..)
            .zip(memory.memory_types_as_slice())
            .find(|(index, kind)| usable & (1 << index) != 0 && kind.property_flags.contains(flags))
            .map(|(index, _)| index)
    };
    match (
        first(PRIVATE_MEMORY),
        first(SHARED_MEMORY),
        first(RING_MEMORY),
    ) {
        (Some(private), Some(shared), Some(ring)) => Ok(MemoryTypes {
            private,
            shared,
            ring,
        }),
        // Vulkan guarantees the device-local type, so only a device that breaks that rule is
        // refused for it.
        (private, shared, ring) => Err([
            (private, PRIVATE_MEMORY),
            (shared, SHARED_MEMORY),
            (ring, RING_MEMORY),
        ]
        .into_iter()
        .filter(|(index, _)| index.is_none())
        .map(|(_, flags)| Missing::MemoryType {
            flags: flag_names(flags),
        })
        .collect()),
    }
}

/// The Vulkan names of `flags`, joined by ` | `.
pub(crate) fn flag_names(flags: vk::MemoryPropertyFlags) -> String {
    MEMORY_FLAG_NAMES
        .iter()
        .filter(|(flag, _)| flags.contains(*flag))
        .map(|(_, name)| *name)
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Whether the device's descriptor limits hold the sampled-texture array's second elements
/// (`arrays::sampled_images`) beside the floor's arrays: what the attachment reads take beyond
/// [`limits`].
fn second_elements_fit(device: &Queried) -> bool {
    let limits = &device.limits;
    let descriptors = &device.vulkan12;
    let sampled = 2 * TEXTURES;
    descriptors.max_per_stage_descriptor_update_after_bind_sampled_images >= sampled
        && descriptors.max_descriptor_set_update_after_bind_sampled_images >= sampled
        && descriptors.max_per_stage_update_after_bind_resources
            >= sampled + TEXTURES + limits.max_color_attachments
        && descriptors.max_update_after_bind_descriptors_in_all_pools
            >= sampled + TEXTURES + SAMPLER_ARRAYS * SAMPLERS
}

/// Each limit the floor sets: its Vulkan name, the device's value, and the least value the runtime
/// uses.
pub(crate) fn limits(device: &Queried) -> [(&'static str, u32, u32); 16] {
    let limits = &device.limits;
    let descriptors = &device.vulkan12;
    [
        (
            "maxPerStageDescriptorUpdateAfterBindSampledImages",
            descriptors.max_per_stage_descriptor_update_after_bind_sampled_images,
            TEXTURES,
        ),
        (
            "maxDescriptorSetUpdateAfterBindSampledImages",
            descriptors.max_descriptor_set_update_after_bind_sampled_images,
            TEXTURES,
        ),
        (
            "maxPerStageDescriptorUpdateAfterBindStorageImages",
            descriptors.max_per_stage_descriptor_update_after_bind_storage_images,
            TEXTURES,
        ),
        (
            "maxDescriptorSetUpdateAfterBindStorageImages",
            descriptors.max_descriptor_set_update_after_bind_storage_images,
            TEXTURES,
        ),
        (
            "maxPerStageDescriptorUpdateAfterBindSamplers",
            descriptors.max_per_stage_descriptor_update_after_bind_samplers,
            SAMPLER_ARRAYS * SAMPLERS,
        ),
        (
            "maxDescriptorSetUpdateAfterBindSamplers",
            descriptors.max_descriptor_set_update_after_bind_samplers,
            SAMPLER_ARRAYS * SAMPLERS,
        ),
        (
            "maxSamplerAllocationCount",
            limits.max_sampler_allocation_count,
            SAMPLERS,
        ),
        // Samplers do not count against it; a fragment stage's colour attachments do.
        (
            "maxPerStageUpdateAfterBindResources",
            descriptors.max_per_stage_update_after_bind_resources,
            2 * TEXTURES + limits.max_color_attachments,
        ),
        (
            "maxUpdateAfterBindDescriptorsInAllPools",
            descriptors.max_update_after_bind_descriptors_in_all_pools,
            2 * TEXTURES + SAMPLER_ARRAYS * SAMPLERS,
        ),
        (
            "maxPushConstantsSize",
            limits.max_push_constants_size,
            PUSH_CONSTANT_BYTES,
        ),
        (
            "maxComputeWorkGroupInvocations",
            limits.max_compute_work_group_invocations,
            WORKGROUP_INVOCATIONS,
        ),
        (
            "maxComputeWorkGroupSize[0]",
            limits.max_compute_work_group_size[0],
            WORKGROUP_SIZE[0],
        ),
        (
            "maxComputeWorkGroupSize[1]",
            limits.max_compute_work_group_size[1],
            WORKGROUP_SIZE[1],
        ),
        (
            "maxComputeWorkGroupSize[2]",
            limits.max_compute_work_group_size[2],
            WORKGROUP_SIZE[2],
        ),
        (
            "timestampComputeAndGraphics",
            limits.timestamp_compute_and_graphics,
            vk::TRUE,
        ),
        // Every slot of a render pass descriptor (`MTLRenderPassDescriptor.colorAttachments`),
        // so a pass of any slots draws.
        (
            "maxColorAttachments",
            limits.max_color_attachments,
            crate::pass::MAX_COLOR_ATTACHMENTS as u32,
        ),
    ]
}

/// A member's Vulkan name from ash's spelling: `buffer_device_address` is `bufferDeviceAddress`.
pub(crate) fn vulkan_name(member: &str) -> String {
    let mut name = String::with_capacity(member.len());
    let mut upper = false;
    for c in member.chars() {
        if c == '_' {
            upper = true;
        } else if upper {
            name.push(c.to_ascii_uppercase());
            upper = false;
        } else {
            name.push(c);
        }
    }
    name
}

/// The Vulkan name of a time domain of [`TIME_DOMAINS`].
pub(crate) fn time_domain_name(domain: vk::TimeDomainKHR) -> &'static str {
    match domain {
        vk::TimeDomainKHR::DEVICE => "VK_TIME_DOMAIN_DEVICE_EXT",
        vk::TimeDomainKHR::CLOCK_MONOTONIC => "VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT",
        vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW => "VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT",
        vk::TimeDomainKHR::QUERY_PERFORMANCE_COUNTER => {
            "VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT"
        }
        _ => "an unnamed time domain",
    }
}
