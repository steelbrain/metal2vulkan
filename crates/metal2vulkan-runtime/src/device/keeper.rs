//! The drivers' keeper: once the runtime has made a device through a Vulkan loader library, one
//! bare instance of that library lives for the rest of the process, so the loader keeps every
//! driver it loaded for it.
//!
//! The loader unloads a driver when the last instance using it is destroyed and loads it again
//! for the next. Not every driver survives that: NVIDIA's Linux driver takes a new piece of
//! glibc's static TLS at each load and never gives it back, so after enough device lifetimes in
//! one process the loader cannot load it and ignores it, and the runtime finds no device. Metal
//! never fails to make a device, and most applications keep their instance for the process's life
//! anyway.
//!
//! The keeper is bare: no layers, no messenger, and no extension but
//! `VK_KHR_portability_enumeration` where the device's instance enabled it, which the runtime does
//! where the loader lists it (without it, a loader that sees only a portability driver refuses the
//! instance): the keeper asks the loader nothing of its own. Nothing is made from it and it is
//! never destroyed. It is made once a device was made (`vkCreateDevice`), while that device's
//! instance is alive, and before the device is primed and registers the exit hook
//! (`exit_guard.rs`); adopted devices never make one. Its instance can still load drivers
//! (measured on a host with Intel, lavapipe and NVIDIA drivers and no Intel GPU: both Intel
//! drivers, which the loader unloaded as it made the device's instance, loaded again for the
//! keeper's). Made before the registration, what it loads comes before the load count the
//! registration keeps, so the next device made on what stays loaded is not primed again
//! (`tests::primed_devices`).

use super::create::new_instance;
use super::{lock, InstanceShared};
use ash::vk;
use std::sync::Mutex;

/// One value per key, made at most once, under one lock.
pub(crate) struct Kept<K, T> {
    entries: Mutex<Vec<(K, T)>>,
}

impl<K: PartialEq, T> Kept<K, T> {
    /// No value kept.
    pub(crate) const fn new() -> Kept<K, T> {
        Kept {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Keep the value `make` makes for `key`, unless one is kept already. `make` runs under the
    /// lock, so concurrent first calls make one value; when it makes none, nothing is kept and
    /// the next call tries again. Whether a value is kept for `key` afterwards.
    pub(crate) fn keep(&self, key: K, make: impl FnOnce() -> Option<T>) -> bool {
        let mut entries = lock(&self.entries);
        if entries.iter().any(|(kept, _)| *kept == key) {
            return true;
        }
        match make() {
            Some(value) => {
                entries.push((key, value));
                true
            }
            None => false,
        }
    }

    /// How many values are kept.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        lock(&self.entries).len()
    }
}

/// A loader library and its bare instance, kept for the life of the process.
struct Keeper {
    /// Keeps the loader library loaded.
    _entry: ash::Entry,
    /// Never destroyed: while it lives, the loader keeps its drivers.
    _instance: vk::Instance,
}

/// One keeper per loader library, keyed by the library itself: the address of its
/// `vkGetInstanceProcAddr`. The OS loads a library once however it is named (a symlink, a
/// searched name, an explicit path), and the keeper holds it loaded, so the address names one
/// library for the life of the process.
static KEEPERS: Kept<usize, Keeper> = Kept::new();

/// Keep the drivers of `instance`'s loader library loaded for the life of the process. Called
/// once a device was made from `instance`, which the runtime created, while it is alive, before
/// that device is primed (the module docs). A keeper that cannot be made leaves the device as it
/// is, and the next device tries again.
pub(super) fn keep_drivers_loaded(instance: &InstanceShared) {
    let entry = &instance.entry;
    let library = entry.static_fn().get_instance_proc_addr as usize;
    let portability = instance
        .extensions
        .iter()
        .any(|name| name.as_c_str() == ash::khr::portability_enumeration::NAME);
    KEEPERS.keep(library, || {
        #[cfg(test)]
        if LOAD_AS_KEPT.load(std::sync::atomic::Ordering::SeqCst) {
            crate::loaded::force();
        }
        let instance = bare_instance(entry, portability)?;
        Some(Keeper {
            _entry: entry.clone(),
            _instance: instance,
        })
    });
}

/// Tests: count a load (`loaded::force`) as the keeper is made, as a driver the device's
/// instance let go loads again for the keeper's on a host with several drivers.
#[cfg(test)]
pub(crate) static LOAD_AS_KEPT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// A bare instance of `entry`: no layers, no messenger, portability enumeration where
/// `portability` says the device's instance enabled it.
fn bare_instance(entry: &ash::Entry, portability: bool) -> Option<vk::Instance> {
    let extensions = [ash::khr::portability_enumeration::NAME.as_ptr()];
    let application = vk::ApplicationInfo::default()
        .engine_name(c"metal2vulkan")
        .api_version(metal2vulkan::vk::ApiVersion::V1_2.packed());
    let info = vk::InstanceCreateInfo::default().application_info(&application);
    let info = if portability {
        info.enabled_extension_names(&extensions)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
    } else {
        info
    };
    // SAFETY: `info` and everything it points at live for the call.
    let instance = unsafe { new_instance(entry, &info) }.ok()?;
    Some(instance.handle())
}
