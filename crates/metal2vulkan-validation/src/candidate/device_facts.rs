//! What an engine row records of the Vulkan device that ran it beside its identity: the tools
//! active on it and the layers it ran under.

use ash::vk;
use std::ffi::{c_char, CStr};

/// What a row's environment records as its device's `layers`: every engine run, on the
/// executor's context, runs under `VK_LAYER_KHRONOS_validation` with core and synchronization
/// validation and a message sink that fails the row. Rows record the name the
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
