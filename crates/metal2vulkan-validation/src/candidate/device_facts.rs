//! What an engine row records of the Vulkan device that ran it: the device's identity and driver
//! (`VkPhysicalDeviceProperties`, `VkPhysicalDeviceDriverProperties`), the tools active on it, and
//! the facts the row's environment has always carried beside them. The executor's own context reads
//! its tools here.

use crate::coverage::ExecutionClass;
use ash::vk;
use std::ffi::{c_char, CStr};

/// What a row's environment records as its device's `layers`: every engine run, on the
/// executor's context or through the runtime, runs under `VK_LAYER_KHRONOS_validation` with core
/// and synchronization validation and a message sink that fails the row. Rows record the name the
/// context had beside the old engine's unvalidated one, deleted with that engine's executor.
pub(crate) const RECORDED_LAYERS: &str = "Validation";

/// One tool `vkGetPhysicalDeviceToolPropertiesEXT` reports active on a device.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ActiveTool {
    pub(crate) name: String,
    pub(crate) version: String,
    /// `VkToolPurposeFlags`.
    pub(crate) purposes: u32,
    /// The layer that implements the tool, if one does.
    pub(crate) layer: String,
}

/// The text of a NUL-terminated Vulkan string array.
fn text(raw: &[c_char]) -> String {
    // SAFETY: Vulkan's fixed-size strings are NUL-terminated within their arrays.
    unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// The tools active on `physical` (`vkGetPhysicalDeviceToolPropertiesEXT`, `VK_EXT_tooling_info`,
/// which the loader, the validation layer, and the profiles layer each provide). An instance
/// where nothing provides the entry point has no tool that reports itself, so the list is empty:
/// the profiles layer always provides it.
///
/// # Safety
/// `physical` was enumerated from `instance`, which `entry` created.
pub(crate) unsafe fn active_tools(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Result<Vec<ActiveTool>, String> {
    // SAFETY: a valid instance and a NUL-terminated name.
    let address = unsafe {
        entry.get_instance_proc_addr(
            instance.handle(),
            c"vkGetPhysicalDeviceToolPropertiesEXT".as_ptr(),
        )
    };
    if address.is_none() {
        return Ok(Vec::new());
    }
    let loader = ash::ext::tooling_info::Instance::new(entry, instance);
    // SAFETY: the entry point exists (above); `physical` belongs to `instance`.
    let tools = unsafe { loader.get_physical_device_tool_properties(physical) }
        .map_err(|error| format!("vkGetPhysicalDeviceToolPropertiesEXT: {error}"))?;
    Ok(tools
        .iter()
        .map(|tool| ActiveTool {
            name: text(&tool.name),
            version: text(&tool.version),
            purposes: tool.purposes.as_raw(),
            layer: text(&tool.layer),
        })
        .collect())
}

/// What a row records of the device that ran it, read from the physical device.
#[derive(Clone, Debug)]
pub(crate) struct DeviceFacts {
    /// `deviceName`.
    pub(crate) device: String,
    /// The device is a CPU implementation (`VK_PHYSICAL_DEVICE_TYPE_CPU`: lavapipe, SwiftShader).
    pub(crate) cpu: bool,
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    /// `driverVersion`, as the driver packs it.
    pub(crate) driver_version: u32,
    /// `apiVersion`, packed.
    pub(crate) api_version: u32,
    /// `VkDriverId` as its raw value (`VK_DRIVER_ID_MOLTENVK` is 14, `MESA_LLVMPIPE` 13).
    pub(crate) driver_id: i32,
    pub(crate) driver_name: String,
    pub(crate) driver_info: String,
    pub(crate) tools: Vec<ActiveTool>,
    /// `shaderSubgroupExtendedTypes`, as the device supports it.
    pub(crate) shader_subgroup_extended_types: bool,
    /// The portability subset's `multisampleArrayImage` (false where the device lists no
    /// subset).
    pub(crate) multisample_array_image: bool,
    pub(crate) max_sampler_anisotropy: f32,
    /// `minStorageBufferOffsetAlignment`.
    pub(crate) storage_buffer_alignment: u64,
    pub(crate) max_storage_buffer_range: u64,
}

impl DeviceFacts {
    /// The facts of `physical`.
    ///
    /// # Safety
    /// `physical` was enumerated from `instance`, which `entry` created at Vulkan 1.2 or later.
    pub(crate) unsafe fn query(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
    ) -> Result<DeviceFacts, String> {
        // SAFETY: the caller's contract.
        let tools = unsafe { active_tools(entry, instance, physical) }?;
        // SAFETY: the caller's contract.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(|error| format!("enumerate Vulkan device extensions: {error}"))?;
        let portability = extensions.iter().any(|extension| {
            text(&extension.extension_name).as_bytes() == b"VK_KHR_portability_subset"
        });
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: the caller's contract; the driver properties are core in 1.2.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let core = properties.properties;
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut subset = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut v12);
        if portability {
            features = features.push_next(&mut subset);
        }
        // SAFETY: the caller's contract; the subset's structure is chained only where the device
        // lists its extension.
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
        Ok(DeviceFacts {
            device: text(&core.device_name),
            cpu: core.device_type == vk::PhysicalDeviceType::CPU,
            vendor_id: core.vendor_id,
            device_id: core.device_id,
            driver_version: core.driver_version,
            api_version: core.api_version,
            driver_id: driver.driver_id.as_raw(),
            driver_name: text(&driver.driver_name),
            driver_info: text(&driver.driver_info),
            tools,
            shader_subgroup_extended_types: v12.shader_subgroup_extended_types == vk::TRUE,
            multisample_array_image: portability && subset.multisample_array_image == vk::TRUE,
            max_sampler_anisotropy: core.limits.max_sampler_anisotropy,
            storage_buffer_alignment: core.limits.min_storage_buffer_offset_alignment,
            max_storage_buffer_range: u64::from(core.limits.max_storage_buffer_range),
        })
    }

    /// The class of the device ([`ExecutionClass`]): a CPU implementation, or its driver.
    pub(crate) fn execution_class(&self) -> ExecutionClass {
        if self.cpu {
            ExecutionClass::CpuIcd
        } else {
            ExecutionClass::Host(crate::coverage::HostClass(
                metal2vulkan::vk::DriverId::from_raw(self.driver_id),
            ))
        }
    }

    /// Whether the driver is MoltenVK.
    pub(crate) fn is_moltenvk(&self) -> bool {
        self.driver_id == vk::DriverId::MOLTENVK.as_raw()
    }

    /// The environment members of these facts, as every engine row records them: the
    /// device's identity (`c3::venues::DeviceIdentity::recorded` reads it), its driver, the
    /// active tools, the host, and the facts beside them. `sampler_anisotropy` is whether the
    /// device enabled `samplerAnisotropy`.
    pub(crate) fn environment(&self, sampler_anisotropy: bool) -> serde_json::Value {
        serde_json::json!({
            "device": self.device,
            "device_id": self.device_id,
            "vendor_id": self.vendor_id,
            "driver_version": self.driver_version,
            "api_version": self.api_version,
            "driver_id": self.driver_id,
            "driver_name": self.driver_name,
            "driver_info": self.driver_info,
            "shader_subgroup_extended_types": self.shader_subgroup_extended_types,
            "multisample_array_image": self.multisample_array_image,
            "sampler_anisotropy": sampler_anisotropy,
            "max_sampler_anisotropy": self.max_sampler_anisotropy,
            "storage_buffer_alignment": self.storage_buffer_alignment,
            "max_storage_buffer_range": self.max_storage_buffer_range,
            "architecture": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "tools": self.tools,
        })
    }
}
