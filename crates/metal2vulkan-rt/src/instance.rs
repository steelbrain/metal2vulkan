//! The Vulkan instance the runtime creates, and the physical devices it lists.

use crate::error::Error;
use crate::floor;
use ash::vk;
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;

/// No Metal counterpart: Metal has no instance. The Vulkan instance the runtime creates for its
/// devices, at the floor's version ([`floor::VERSION`]) with the floor's instance extensions
/// ([`floor::INSTANCE_EXTENSIONS`]), and those of [`floor::OPTIONAL_INSTANCE_EXTENSIONS`] and the
/// consumer's that the loader offers. Where the loader offers `VK_KHR_portability_enumeration`, it
/// lists portability devices too (`VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`), each
/// checked against the floor as any other.
///
/// A cheap handle: clones share one instance, which lives until the last clone, and the last
/// [`PhysicalDevice`] and [`Device`](crate::Device) made from it, are dropped. Whatever the
/// consumer makes with [`Instance::raw`] (a window surface) it destroys before that.
#[derive(Clone)]
pub struct Instance {
    pub(crate) shared: Arc<Shared>,
}

pub(crate) struct Shared {
    entry: ash::Entry,
    instance: ash::Instance,
    extensions: Vec<CString>,
}

impl Instance {
    /// Loads the Vulkan loader and creates the instance with the floor's instance extensions, the
    /// portability listing where the loader offers it, and each of `extensions` the loader offers: the window system's (such as `VK_KHR_xlib_surface`
    /// or `VK_KHR_win32_surface`) and any other the consumer uses. A name the floor already lists,
    /// or listed twice, is enabled once. [`Instance::extensions`] says which were enabled; the
    /// consumer names an extension's dependencies too. The loader loads the drivers the system
    /// installed; [`Instance::with_drivers`] gives it the consumer's own instead.
    ///
    /// Err: [`Error::Loader`] where no loader loads; [`Error::Vulkan`] where the instance is not
    /// made, the floor's extensions missing among the causes (`VK_ERROR_EXTENSION_NOT_PRESENT`).
    pub fn new(extensions: &[&CStr]) -> Result<Instance, Error> {
        // SAFETY: no driver is given.
        unsafe { Instance::with_drivers(extensions, &[]) }
    }

    /// No Metal counterpart (Metal is part of the system): [`Instance::new`], with `drivers` the
    /// only Vulkan drivers the loader loads, where any is given. Each is a driver the consumer
    /// loaded itself, from its own distribution, given by its `vkGetInstanceProcAddr`: for a
    /// driver built for the Vulkan loader, the `vk_icdGetInstanceProcAddr` it exports. The
    /// instance then enables `VK_LUNARG_direct_driver_loading` and gives the loader the list in
    /// its exclusive mode (`VK_DIRECT_DRIVER_LOADING_MODE_EXCLUSIVE_LUNARG`): no driver the system
    /// installed, or that the loader's variables name, is loaded. An instance extension is offered
    /// where the loader or one of the drivers lists it, as the loader does not ask such drivers
    /// for theirs. With no driver, it is [`Instance::new`].
    ///
    /// Err: as [`Instance::new`]; [`Error::NoDirectDriverLoading`] where a driver is given and the
    /// loader does not offer `VK_LUNARG_direct_driver_loading`.
    ///
    /// # Safety
    ///
    /// Each of `drivers` is a Vulkan driver's `vkGetInstanceProcAddr`, whose library stays loaded
    /// while the instance, or anything made from it, lives.
    pub unsafe fn with_drivers(
        extensions: &[&CStr],
        drivers: &[vk::PFN_vkGetInstanceProcAddr],
    ) -> Result<Instance, Error> {
        // SAFETY: loading the loader runs its initialisation and nothing else.
        let entry = unsafe { ash::Entry::load() }.map_err(Error::Loader)?;
        // SAFETY: the loader is loaded.
        let loader = names(unsafe { entry.enumerate_instance_extension_properties(None) })?;
        let mut offered = Vec::new();
        for &driver in drivers {
            // SAFETY: by the caller, a driver's `vkGetInstanceProcAddr`, which answers the global
            // commands for no instance.
            let own = unsafe {
                ash::Entry::from_static_fn(ash::StaticFn {
                    get_instance_proc_addr: driver,
                })
            };
            // SAFETY: the driver is loaded, by the caller.
            offered.extend(names(unsafe {
                own.enumerate_instance_extension_properties(None)
            })?);
        }
        let names = enabled(&loader, &offered, extensions, !drivers.is_empty())?;
        let pointers: Vec<*const c_char> = names.iter().map(|name| name.as_ptr()).collect();
        let application = vk::ApplicationInfo::default().api_version(floor::VERSION);
        // Where the loader offers the listing of portability devices, they are listed too.
        let portability = ash::khr::portability_enumeration::NAME;
        let flags = match names.iter().any(|name| name.as_c_str() == portability) {
            true => vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
            false => vk::InstanceCreateFlags::empty(),
        };
        let infos: Vec<vk::DirectDriverLoadingInfoLUNARG> = (drivers.iter())
            .map(|&driver| {
                vk::DirectDriverLoadingInfoLUNARG::default()
                    .pfn_get_instance_proc_addr(Some(driver))
            })
            .collect();
        let mut list = vk::DirectDriverLoadingListLUNARG::default()
            .mode(vk::DirectDriverLoadingModeLUNARG::EXCLUSIVE)
            .drivers(&infos);
        let mut info = vk::InstanceCreateInfo::default()
            .flags(flags)
            .application_info(&application)
            .enabled_extension_names(&pointers);
        if !drivers.is_empty() {
            info = info.push_next(&mut list);
        }
        // SAFETY: `info` and everything it points to live through the call; each driver is
        // loaded, by the caller.
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(Error::vulkan("vkCreateInstance"))?;
        Ok(Instance {
            shared: Arc::new(Shared {
                entry,
                instance,
                extensions: names,
            }),
        })
    }

    /// No Metal counterpart: the instance extensions enabled, the floor's first.
    pub fn extensions(&self) -> &[CString] {
        &self.shared.extensions
    }

    /// `MTLCopyAllDevices`: the physical devices in the loader's order. The floor is checked when
    /// one is opened ([`Device::new`](crate::Device::new)), not here.
    pub fn physical_devices(&self) -> Result<Vec<PhysicalDevice>, Error> {
        // SAFETY: the instance is live.
        let handles = unsafe { self.shared.instance.enumerate_physical_devices() }
            .map_err(Error::vulkan("vkEnumeratePhysicalDevices"))?;
        Ok(handles
            .into_iter()
            .map(|raw| PhysicalDevice {
                instance: self.clone(),
                raw,
            })
            .collect())
    }

    /// No Metal counterpart: the loader and the instance, for the consumer's own calls (making a
    /// window surface). The runtime keeps owning both; the consumer destroys neither.
    pub fn raw(&self) -> (&ash::Entry, &ash::Instance) {
        (&self.shared.entry, &self.shared.instance)
    }

    /// The instance's function table.
    pub(crate) fn vk(&self) -> &ash::Instance {
        &self.shared.instance
    }
}

/// The names of the instance extensions `listed` holds.
fn names(listed: Result<Vec<vk::ExtensionProperties>, vk::Result>) -> Result<Vec<CString>, Error> {
    let listed = listed.map_err(Error::vulkan("vkEnumerateInstanceExtensionProperties"))?;
    Ok((listed.iter())
        .filter_map(|properties| properties.extension_name_as_c_str().ok())
        .map(CStr::to_owned)
        .collect())
}

/// The instance extensions instance creation enables: the floor's, then each of the optional
/// ones' and of `extensions` that the `loader` or the consumer's drivers (`drivers`) list,
/// once each, and where the consumer gives drivers (`direct`), direct driver loading, which only
/// the loader offers. Err: [`Error::NoDirectDriverLoading`] where `direct` and the loader does
/// not list it.
pub(crate) fn enabled(
    loader: &[CString],
    drivers: &[CString],
    extensions: &[&CStr],
    direct: bool,
) -> Result<Vec<CString>, Error> {
    let mut names: Vec<CString> = (floor::INSTANCE_EXTENSIONS.iter())
        .map(|&name| name.to_owned())
        .collect();
    let listed = |name: &CStr| {
        loader
            .iter()
            .chain(drivers)
            .any(|offered| offered.as_c_str() == name)
    };
    for &name in floor::OPTIONAL_INSTANCE_EXTENSIONS.iter().chain(extensions) {
        if listed(name) && !names.iter().any(|enabled| enabled.as_c_str() == name) {
            names.push(name.to_owned());
        }
    }
    if direct {
        let direct = ash::lunarg::direct_driver_loading::NAME;
        if !loader.iter().any(|offered| offered.as_c_str() == direct) {
            return Err(Error::NoDirectDriverLoading);
        }
        if !names.iter().any(|enabled| enabled.as_c_str() == direct) {
            names.push(direct.to_owned());
        }
    }
    Ok(names)
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: every physical device and device made from the instance holds a clone of it, so
        // none is alive; the consumer's own objects are gone by its contract.
        unsafe { self.instance.destroy_instance(None) };
    }
}

/// No Metal counterpart: one entry of `MTLCopyAllDevices` before it is opened, a physical device
/// of one [`Instance`]. [`Device::new`](crate::Device::new) opens it; holding it keeps the instance
/// alive.
#[derive(Clone)]
pub struct PhysicalDevice {
    instance: Instance,
    raw: vk::PhysicalDevice,
}

impl PhysicalDevice {
    /// No Metal counterpart: the Vulkan handle, for the consumer's own queries (its name, its
    /// surface support).
    pub fn raw(&self) -> vk::PhysicalDevice {
        self.raw
    }

    /// No Metal counterpart (every Metal device runs every program its family supports): checks
    /// the device against the runtime's floor. Returns what it offers above the floor, or every
    /// item of the floor it lacks; for memory it checks only that a type with each set of flags
    /// exists, and [`Device::new`](crate::Device::new) finishes the check on the created device.
    pub fn check(&self) -> Result<floor::Capabilities, floor::BelowFloor> {
        let (entry, instance) = self.instance.raw();
        // SAFETY: the instance listed the device, and was created at the floor's version.
        unsafe { floor::check(entry, instance, self.raw) }
    }

    /// The instance that listed it.
    pub(crate) fn instance(&self) -> &Instance {
        &self.instance
    }
}
