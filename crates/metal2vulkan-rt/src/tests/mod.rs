mod command;
mod compute;
mod device;
mod floor;
mod hazard;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod kernels;
mod memory;
mod null_descriptor;
mod one_path;
mod readback;

/// Every product source file under `src/`, the tests excepted, with its path under `src/`, in
/// path order.
pub(crate) fn product_sources() -> Vec<(String, String)> {
    use std::path::{Path, PathBuf};
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()))
            .map(|entry| entry.expect("a directory entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "tests") {
                    continue;
                }
                walk(&path, out);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                out.push(path);
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    assert!(!files.is_empty(), "no source under {}", root.display());
    files
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
            let label = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .display()
                .to_string();
            (label, text)
        })
        .collect()
}

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
