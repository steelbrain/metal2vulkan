mod command;
mod device;
mod floor;
mod memory;
mod one_path;

/// Each physical device of the device cfg's kind the loader lists, with its name: CPU devices
/// under `m2v_cpu_icd`, the others under `m2v_gpu`. The legs pick one with `VK_DRIVER_FILES` and
/// `MESA_VK_DEVICE_SELECT`.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) fn devices_of_the_cfgs_kind(
    instance: &crate::Instance,
) -> Vec<(String, crate::PhysicalDevice)> {
    let (_, vk_instance) = instance.raw();
    let physicals = instance.physical_devices().expect("the physical devices");
    physicals
        .into_iter()
        .filter_map(|physical| {
            // SAFETY: `physical` is one of `instance`'s.
            let properties = unsafe { vk_instance.get_physical_device_properties(physical.raw()) };
            let cpu = properties.device_type == ash::vk::PhysicalDeviceType::CPU;
            let name = properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            (cpu == cfg!(m2v_cpu_icd)).then_some((name, physical))
        })
        .collect()
}
