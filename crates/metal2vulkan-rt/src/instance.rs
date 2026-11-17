//! The Vulkan instance the runtime creates, and the physical devices it lists.

use crate::error::Error;
use crate::floor;
use ash::vk;
use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;

/// No Metal counterpart: Metal has no instance. The Vulkan instance the runtime creates for its
/// devices, at the floor's version ([`floor::VERSION`]) with the floor's instance extensions
/// ([`floor::INSTANCE_EXTENSIONS`]) and those the consumer names that the loader offers.
///
/// A cheap handle: clones share one instance, which lives until the last clone, and the last
/// [`PhysicalDevice`] and [`Device`](crate::Device) made from it, are dropped. Whatever the
/// consumer makes with [`Instance::raw`] (a window surface) it destroys before that.
#[derive(Clone)]
pub struct Instance {
    shared: Arc<Shared>,
}

struct Shared {
    entry: ash::Entry,
    instance: ash::Instance,
    extensions: Vec<CString>,
}

impl Instance {
    /// Loads the Vulkan loader and creates the instance with the floor's instance extensions, and
    /// each of `extensions` the loader offers: the window system's (such as `VK_KHR_xlib_surface`
    /// or `VK_KHR_win32_surface`) and any other the consumer uses. A name the floor already lists,
    /// or listed twice, is enabled once. [`Instance::extensions`] says which were enabled; the
    /// consumer names an extension's dependencies too.
    ///
    /// Err: [`Error::Loader`] where no loader loads; [`Error::Vulkan`] where the instance is not
    /// made, the floor's extensions missing among the causes (`VK_ERROR_EXTENSION_NOT_PRESENT`).
    pub fn new(extensions: &[&CStr]) -> Result<Instance, Error> {
        // SAFETY: loading the loader runs its initialisation and nothing else.
        let entry = unsafe { ash::Entry::load() }.map_err(Error::Loader)?;
        // SAFETY: the loader is loaded.
        let offered = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(Error::vulkan("vkEnumerateInstanceExtensionProperties"))?;
        let mut names: Vec<CString> = floor::INSTANCE_EXTENSIONS
            .iter()
            .map(|&name| name.to_owned())
            .collect();
        for &name in extensions {
            let listed = offered
                .iter()
                .any(|properties| properties.extension_name_as_c_str() == Ok(name));
            if listed && !names.iter().any(|enabled| enabled.as_c_str() == name) {
                names.push(name.to_owned());
            }
        }
        let pointers: Vec<*const c_char> = names.iter().map(|name| name.as_ptr()).collect();
        let application = vk::ApplicationInfo::default().api_version(floor::VERSION);
        let info = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_extension_names(&pointers);
        // SAFETY: `info` and everything it points to live through the call.
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

    /// The instance that listed it.
    pub(crate) fn instance(&self) -> &Instance {
        &self.instance
    }
}
