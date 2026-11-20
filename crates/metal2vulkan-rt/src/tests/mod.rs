mod ahead;
mod aliased_encoders;
mod alpha_test;
mod argument_tier;
mod arrays;
mod bicubic;
mod blit;
mod border;
mod cache;
mod choice;
mod command;
mod compute;
mod copy_out;
mod copy_over;
mod cube_views;
mod d3d11;
mod device;
mod dma_bufs;
mod draw;
mod export;
mod fence;
mod fetch;
mod first_use_sites;
mod floor;
mod hazard;
mod heap;
mod image_reads;
mod import;
mod import_overs;
mod import_time;
mod kept;
mod kernel_map;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod kernels;
mod large;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod layouts;
mod linear_planes;
#[cfg(m2v_gpu)]
mod locks;
mod lowered;
mod maker_yield;
mod managed;
mod memory;
mod mipmaps;
mod narrow;
mod nil_slots;
mod null_descriptor;
mod one_path;
mod over;
mod over_aliases;
mod over_causes;
mod pass;
mod patch;
mod pixel_format_info;
mod places;
mod present;
mod push_window;
mod read_write_tier;
mod readback;
mod render;
mod resolve;
mod ring;
mod sampler;
mod sampling;
mod shadow;
mod sites;
mod spill;
mod staging;
#[cfg(m2v_gpu)]
mod stamps;
mod state_answers;
mod storage_element;
mod submit_infos;
mod surface;
mod swizzle;
mod take_split;
mod texture;
mod texture_accesses;
mod translate;
mod upload;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod validation;
mod view;

/// This crate's directory: the test process's `CARGO_MANIFEST_DIR` where it holds the crate's
/// sources (cargo and nextest set it for each test, and an archive run elsewhere remaps it to the
/// workspace it runs in), else the directory the tests were built in.
pub(crate) fn crate_dir() -> std::path::PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .filter(|dir| dir.join("src").join("lib.rs").is_file())
        .unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").into())
}

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
    let root = crate_dir().join("src");
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

/// The sampler indices of a test's states, whose kernels number no constexpr sampler: a read's
/// or a write's sampler takes nothing at its index. A kernel that numbers a constexpr sampler is
/// refused as one that reads what the test does not supply.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) fn no_constexpr_samplers(
    samplers: &[crate::NumberedSampler],
) -> Result<Vec<u32>, crate::Error> {
    match samplers
        .iter()
        .any(|sampler| matches!(sampler, crate::NumberedSampler::Constexpr(_)))
    {
        true => Err(crate::Error::Unsupported(crate::Gap::KernelInput)),
        false => Ok(vec![0; samplers.len()]),
    }
}

/// Whether `device`'s pipeline cache keeps a blob, once the reaper has seeded it: none where its
/// driver's data of an empty cache begins with no header (`cache::usable_header`), as lavapipe's
/// does, which is printed.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) fn keeps_a_blob(name: &str, device: &crate::Device) -> bool {
    let caches = &device.core().caches;
    crate::tests::command::on_device::within("the seed", || caches.is_seeded());
    let kept = caches.blob().is_some();
    if !kept {
        println!("{name}: its pipeline cache data begins with no header, so it keeps no blob");
    }
    kept
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
