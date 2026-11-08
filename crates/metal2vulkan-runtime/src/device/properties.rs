//! What a device reports about itself: the Vulkan facts `query` reads of the physical device, from
//! which `Device::assemble` builds the [`DeviceProperties`](crate::DeviceProperties) Metal's device
//! queries answer from (`capabilities.rs`).
//!
//! The same read answers [`PhysicalDeviceIdentity`] of a physical device no device was made of
//! ([`Device::identities_with`](crate::Device::identities_with)), so the identity a consumer reads
//! before making a device is the one the device then answers.

use crate::capabilities::DeviceIdentity;
use ash::vk;
use metal2vulkan::vk::ApiVersion;

/// The kind of physical device, from `VkPhysicalDeviceType`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// A discrete GPU.
    Discrete,
    /// A GPU integrated with the host.
    Integrated,
    /// A GPU of a virtual machine.
    Virtual,
    /// A CPU implementation (lavapipe, SwiftShader).
    Cpu,
    /// Any other device.
    Other,
}

impl DeviceKind {
    pub(super) fn of(ty: vk::PhysicalDeviceType) -> DeviceKind {
        match ty {
            vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::Integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::Virtual,
            vk::PhysicalDeviceType::CPU => DeviceKind::Cpu,
            _ => DeviceKind::Other,
        }
    }

    /// Whether a device of the kind shares the host's memory (`hasUnifiedMemory`): an integrated
    /// GPU and a CPU implementation do; a discrete GPU does not, and a virtual GPU or another kind
    /// is not known to.
    pub(crate) fn has_unified_memory(self) -> bool {
        matches!(self, DeviceKind::Integrated | DeviceKind::Cpu)
    }

    /// The order `Device::system_default` prefers kinds in: discrete, integrated, virtual, CPU,
    /// other.
    pub(super) fn preference(self) -> u8 {
        match self {
            DeviceKind::Discrete => 0,
            DeviceKind::Integrated => 1,
            DeviceKind::Virtual => 2,
            DeviceKind::Cpu => 3,
            DeviceKind::Other => 4,
        }
    }
}

/// `registryID` of a device with `uuid`: its two halves, little-endian, exclusive-ored, and 1
/// when that is 0 (Metal never answers 0).
pub(crate) fn registry_id(uuid: &[u8; 16]) -> u64 {
    let half = |at: usize| {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&uuid[at..at + 8]);
        u64::from_le_bytes(bytes)
    };
    match half(0) ^ half(8) {
        0 => 1,
        id => id,
    }
}

/// A physical device's identity, read without making a device
/// ([`Device::identities_with`](crate::Device::identities_with),
/// [`Device::system_default_identity_with`](crate::Device::system_default_identity_with)).
///
/// No Metal counterpart: Metal's program holds an `MTLDevice` and asks it, so every one of these
/// is a property of a device that exists. The runtime answers them of the physical device alone,
/// so a consumer can tell which GPU it is about to get before paying for it. Every field but
/// [`recommended_max_working_set_size`](Self::recommended_max_working_set_size), which no device
/// property answers, is the one the device made of that physical device answers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalDeviceIdentity {
    /// `name`: `VkPhysicalDeviceProperties::deviceName`
    /// ([`DeviceProperties::name`](crate::DeviceProperties::name)).
    pub name: String,
    /// The kind of physical device (`VkPhysicalDeviceProperties::deviceType`), which the order
    /// [`Device::system_default_with`](crate::Device::system_default_with) prefers physical
    /// devices in is taken from. No Metal counterpart
    /// ([`DeviceProperties::kind`](crate::DeviceProperties::kind)).
    pub kind: DeviceKind,
    /// `VkPhysicalDeviceProperties::vendorID`
    /// ([`DeviceProperties::vendor_id`](crate::DeviceProperties::vendor_id)).
    pub vendor_id: u32,
    /// `VkPhysicalDeviceProperties::deviceID`
    /// ([`DeviceProperties::device_id`](crate::DeviceProperties::device_id)).
    pub device_id: u32,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`
    /// ([`DeviceProperties::device_uuid`](crate::DeviceProperties::device_uuid)).
    pub device_uuid: [u8; 16],
    /// `registryID`: the ID [`Device::registry_id`](crate::Device::registry_id) of the device of
    /// this physical device answers, and [`Device::with_registry_id`](crate::Device::with_registry_id)
    /// opens it by.
    pub registry_id: u64,
    /// The Vulkan version the device of this physical device runs at: the lower of the instance's
    /// and the physical device's. No Metal counterpart
    /// ([`DeviceProperties::api_version`](crate::DeviceProperties::api_version)).
    pub api_version: ApiVersion,
    /// The physical device's PCI bus address
    /// ([`DeviceProperties::pci_bus`](crate::DeviceProperties::pci_bus)).
    pub pci_bus: Option<(u32, u32, u32, u32)>,
    /// The physical device's DRM render node
    /// ([`DeviceProperties::drm_render`](crate::DeviceProperties::drm_render)).
    pub drm_render: Option<(i64, i64)>,
    /// The size in bytes of the physical device's largest `DEVICE_LOCAL` memory heap
    /// (`VkPhysicalDeviceMemoryProperties::memoryHeaps`), as its Vulkan driver reports it; the
    /// device's [`DeviceProperties::max_buffer_length`](crate::DeviceProperties::max_buffer_length)
    /// is at most three quarters of it, to 16 KiB. Named after Metal's nearest counterpart,
    /// `recommendedMaxWorkingSetSize`, which it is not: Metal's is its estimate of the memory the
    /// GPU can use without hurting its performance, and this is the heap's whole size, fixed for
    /// the physical device, with nothing taken off for the system or for other processes' use
    /// (what `VK_EXT_memory_budget` would report) and no device made to read it. On an Apple M3
    /// Max of 64 GiB (macOS 27.0.1, MoltenVK 1.4.2) it is 68,719,476,736, the host's whole
    /// memory, where Metal's `recommendedMaxWorkingSetSize` is 55,662,788,608. No device
    /// property answers it. `None` only where the driver reports no `DEVICE_LOCAL` heap, which
    /// Vulkan requires one of.
    pub recommended_max_working_set_size: Option<u64>,
}

impl PhysicalDeviceIdentity {
    /// The identity in `identity`, which [`query`] read of a physical device: the fields the
    /// device made of it answers, from the one read.
    pub(crate) fn of(identity: &DeviceIdentity) -> PhysicalDeviceIdentity {
        PhysicalDeviceIdentity {
            name: identity.name.clone(),
            kind: identity.kind,
            vendor_id: identity.vendor_id,
            device_id: identity.device_id,
            device_uuid: identity.device_uuid,
            registry_id: registry_id(&identity.device_uuid),
            api_version: identity.api_version,
            pci_bus: identity.pci_bus,
            drm_render: identity.drm_render,
            recommended_max_working_set_size: identity.device_local_heap,
        }
    }
}

/// The identity of `physical`, and its largest memory allocation
/// (`VkPhysicalDeviceVulkan11Properties::maxMemoryAllocationSize`), which the sub-allocator
/// reads.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, created at `instance_api_version`, which is
/// Vulkan 1.2 or later.
pub(super) unsafe fn query(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    instance_api_version: metal2vulkan::vk::ApiVersion,
) -> (DeviceIdentity, u64) {
    let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut v11);
    // SAFETY: the caller's contract; `VkPhysicalDeviceVulkan11Properties` is core in 1.2.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let core = properties.properties;
    // SAFETY: the caller's contract.
    let (pci_bus, drm_render) = unsafe { bus_addresses(instance, physical) };
    // SAFETY: the caller's contract.
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    let heaps =
        &memory.memory_heaps[..(memory.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS)];
    let device_local_heap = heaps
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max();
    let identity = DeviceIdentity {
        name: core
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        kind: DeviceKind::of(core.device_type),
        vendor_id: core.vendor_id,
        device_id: core.device_id,
        device_uuid: v11.device_uuid,
        pci_bus,
        drm_render,
        api_version: metal2vulkan::vk::ApiVersion::from_packed(core.api_version)
            .min(instance_api_version),
        max_viewports: core.limits.max_viewports,
        max_sampler_allocation_count: core.limits.max_sampler_allocation_count,
        // SAFETY: the caller's contract.
        gpu_core_count: unsafe { gpu_core_count(instance, physical) },
        sampled_sample_counts: metal2vulkan::vk::SampleCountFlags::from_raw(
            (core.limits.sampled_image_color_sample_counts
                & core.limits.sampled_image_depth_sample_counts)
                .as_raw(),
        ),
        device_local_heap,
    };
    (identity, v11.max_memory_allocation_size)
}

/// The PCI bus address and the DRM render node of `physical`
/// ([`DeviceProperties::pci_bus`](crate::DeviceProperties::pci_bus),
/// [`DeviceProperties::drm_render`](crate::DeviceProperties::drm_render)): each read through its
/// own extension, and only where the driver lists it (a query of properties enables nothing);
/// `None` where it lists none, and `None` for both where the extensions cannot be listed at all.
///
/// # Safety
///
/// As [`query`]'s.
unsafe fn bus_addresses(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> (Option<(u32, u32, u32, u32)>, Option<(i64, i64)>) {
    // SAFETY: the caller's contract.
    let listed = unsafe { instance.enumerate_device_extension_properties(physical) };
    let Ok(listed) = listed else {
        return (None, None);
    };
    let has = |name: &std::ffi::CStr| {
        listed
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    };
    let pci = has(ash::ext::pci_bus_info::NAME);
    let drm = has(ash::ext::physical_device_drm::NAME);
    if !(pci || drm) {
        return (None, None);
    }
    let mut bus = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut nodes = vk::PhysicalDeviceDrmPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default();
    if pci {
        properties = properties.push_next(&mut bus);
    }
    if drm {
        properties = properties.push_next(&mut nodes);
    }
    // SAFETY: the caller's contract; each structure chained is of an extension the device lists.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    bus_addresses_of(pci.then_some(&bus), drm.then_some(&nodes))
}

/// The PCI bus address in `bus` and the DRM render node in `nodes`, the structures the driver
/// filled for the extensions it lists ([`bus_addresses`]): `None` for one the driver does not
/// list, and `None` for the render node of a driver that reports the device has none
/// (`hasRender` false), whose `renderMajor` and `renderMinor` say nothing.
pub(crate) fn bus_addresses_of(
    bus: Option<&vk::PhysicalDevicePCIBusInfoPropertiesEXT>,
    nodes: Option<&vk::PhysicalDeviceDrmPropertiesEXT>,
) -> (Option<(u32, u32, u32, u32)>, Option<(i64, i64)>) {
    let pci = bus.map(|bus| {
        (
            bus.pci_domain,
            bus.pci_bus,
            bus.pci_device,
            bus.pci_function,
        )
    });
    let render = nodes
        .filter(|nodes| nodes.has_render == vk::TRUE)
        .map(|nodes| (nodes.render_major, nodes.render_minor));
    (pci, render)
}

/// `gpuCoreCount` of `physical` as its driver reports it
/// ([`DeviceProperties::gpu_core_count`](crate::DeviceProperties::gpu_core_count)): through the
/// first of NVIDIA's, AMD's and Arm's extensions that the driver lists, each read only where
/// listed (a query of properties enables nothing); `None` where it lists none, or reports 0.
///
/// # Safety
///
/// As [`query`]'s.
unsafe fn gpu_core_count(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Option<u32> {
    // SAFETY: the caller's contract.
    let listed = unsafe { instance.enumerate_device_extension_properties(physical) }.ok()?;
    let has = |name: &std::ffi::CStr| {
        listed
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    };
    let nvidia = has(ash::nv::shader_sm_builtins::NAME);
    let amd = has(ash::amd::shader_core_properties::NAME);
    let amd_active = has(ash::amd::shader_core_properties2::NAME);
    let arm = has(ash::arm::shader_core_builtins::NAME);
    let mut sms = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
    let mut cores = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
    let mut active = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
    let mut arm_cores = vk::PhysicalDeviceShaderCoreBuiltinsPropertiesARM::default();
    let mut properties = vk::PhysicalDeviceProperties2::default();
    if nvidia {
        properties = properties.push_next(&mut sms);
    }
    if amd {
        properties = properties.push_next(&mut cores);
    }
    if amd_active {
        properties = properties.push_next(&mut active);
    }
    if arm {
        properties = properties.push_next(&mut arm_cores);
    }
    if !(nvidia || amd || amd_active || arm) {
        return None;
    }
    // SAFETY: the caller's contract; each structure chained is of an extension the device lists.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let amd_total = cores
        .shader_engine_count
        .saturating_mul(cores.shader_arrays_per_engine_count)
        .saturating_mul(cores.compute_units_per_shader_array);
    [
        (nvidia, sms.shader_sm_count),
        (amd_active, active.active_compute_unit_count),
        (amd, amd_total),
        (arm, arm_cores.shader_core_count),
    ]
    .into_iter()
    .find_map(|(listed, count)| (listed && count > 0).then_some(count))
}

/// The `driverUUID` and `deviceUUID` of `physical`.
///
/// # Safety
///
/// As [`query`]'s.
pub(super) unsafe fn uuids(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> [[u8; 16]; 2] {
    let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut v11);
    // SAFETY: the caller's contract; `VkPhysicalDeviceVulkan11Properties` is core in 1.2.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    [v11.driver_uuid, v11.device_uuid]
}
