//! Device extensions the consumer names beyond the floor's, each with the features of it the
//! consumer needs, enabled where the device offers both.

use ash::vk;
use std::ffi::{c_void, CStr, CString};
use std::sync::Arc;

/// No Metal counterpart: a device extension the consumer uses beyond the floor's, such as
/// `VK_EXT_swapchain_maintenance1` for its presents, and the features of it the consumer needs.
///
/// [`Device::new`](crate::Device::new) enables it where the device lists the extension and
/// offers those features, and passes it over otherwise; [`Device::extensions`](crate::Device::extensions)
/// says which were enabled. The consumer names an extension's dependencies too.
#[derive(Clone)]
pub struct DeviceExtension {
    name: CString,
    features: Option<Arc<dyn Features>>,
}

impl DeviceExtension {
    /// An extension with no features to enable.
    pub fn new(name: &CStr) -> DeviceExtension {
        DeviceExtension {
            name: name.to_owned(),
            features: None,
        }
    }

    /// An extension with its feature structure. `wanted` has the members the consumer needs set,
    /// and `needs` says whether the device's answer, the same structure as
    /// `vkGetPhysicalDeviceFeatures2` fills it, offers them. Where it does, device creation chains
    /// `wanted` as it is.
    ///
    /// `T` is a structure the floor's features do not already chain: not
    /// `VkPhysicalDeviceVulkan11Features`, `12`, `13`,
    /// `VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR`,
    /// `VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR`,
    /// `VkPhysicalDeviceShaderQuadControlFeaturesKHR`,
    /// `VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT`,
    /// `VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR`,
    /// `VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT` or
    /// `VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR`.
    pub fn with_features<T>(name: &CStr, wanted: T, needs: fn(&T) -> bool) -> DeviceExtension
    where
        T: vk::ExtendsPhysicalDeviceFeatures2
            + vk::ExtendsDeviceCreateInfo
            + Copy
            + Default
            + Send
            + Sync
            + 'static,
    {
        DeviceExtension {
            name: name.to_owned(),
            features: Some(Arc::new(Wanted { wanted, needs })),
        }
    }

    /// The extension's name.
    pub fn name(&self) -> &CStr {
        &self.name
    }
}

/// No Metal counterpart: a Vulkan feature the consumer needs for its own raw Vulkan work on the
/// runtime's device, such as a two-plane image it imports and samples itself.
///
/// [`Device::new`](crate::Device::new) enables each of
/// [`DeviceOptions::features`](crate::DeviceOptions::features) the device reports, and
/// [`Device::features`](crate::Device::features) says which were enabled, so the consumer can
/// refuse its own path with its own reason. The runtime's own work never reads them: the
/// translator sees only the floor's features.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConsumerFeature {
    /// `samplerYcbcrConversion` (`VkPhysicalDeviceVulkan11Features`): samplers and image views
    /// with a YCbCr conversion.
    SamplerYcbcrConversion,
}

/// A feature structure as the consumer asks for it, whatever its type.
trait Features: Send + Sync {
    /// Whether the device offers what the consumer needs.
    ///
    /// # Safety
    ///
    /// `physical` is one of `instance`'s physical devices, of Vulkan 1.1 or later.
    unsafe fn offered(&self, instance: &ash::Instance, physical: vk::PhysicalDevice) -> bool;

    /// A copy of the structure to enable, for the device-creation chain.
    fn enabling(&self) -> Box<dyn Link>;
}

struct Wanted<T> {
    wanted: T,
    needs: fn(&T) -> bool,
}

impl<T> Features for Wanted<T>
where
    T: vk::ExtendsPhysicalDeviceFeatures2
        + vk::ExtendsDeviceCreateInfo
        + Copy
        + Default
        + Send
        + Sync
        + 'static,
{
    unsafe fn offered(&self, instance: &ash::Instance, physical: vk::PhysicalDevice) -> bool {
        let mut answer = T::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut answer);
        // SAFETY: the caller's contract; `answer` lives through the call.
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
        (self.needs)(&answer)
    }

    fn enabling(&self) -> Box<dyn Link> {
        Box::new(self.wanted)
    }
}

/// A structure that extends `VkDeviceCreateInfo`, as the head of a Vulkan structure.
pub(crate) trait Link {
    /// The structure's `sType` and `pNext`.
    fn base(&mut self) -> *mut vk::BaseOutStructure<'static>;
}

impl<T: vk::ExtendsDeviceCreateInfo> Link for T {
    fn base(&mut self) -> *mut vk::BaseOutStructure<'static> {
        // Every Vulkan structure that extends another begins with `sType` and `pNext`.
        (self as *mut T).cast()
    }
}

/// The consumer's extensions a device enables: those it lists, past the floor's and without
/// repeats, whose features it offers; with the structures to chain for them.
///
/// # Safety
///
/// `physical` is one of `instance`'s physical devices, of Vulkan 1.3.
pub(crate) unsafe fn choose(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    listed: &[vk::ExtensionProperties],
    floor: &[&CStr],
    asked: &[DeviceExtension],
) -> (Vec<CString>, Vec<Box<dyn Link>>) {
    let mut names: Vec<CString> = floor.iter().map(|&name| name.to_owned()).collect();
    let mut links = Vec::new();
    for extension in asked {
        let name = extension.name.as_c_str();
        let offered = listed
            .iter()
            .any(|properties| properties.extension_name_as_c_str() == Ok(name));
        if !offered || names.iter().any(|enabled| enabled.as_c_str() == name) {
            continue;
        }
        if let Some(features) = &extension.features {
            // SAFETY: the caller's contract.
            if !unsafe { features.offered(instance, physical) } {
                continue;
            }
            links.push(features.enabling());
        }
        names.push(name.to_owned());
    }
    (names, links)
}

/// Puts `links` in front of the chain `head` begins.
///
/// # Safety
///
/// `head` is null or the first structure of a valid chain; `links` outlive every use of the
/// chain.
pub(crate) unsafe fn chain(head: *const c_void, links: &mut [Box<dyn Link>]) -> *const c_void {
    let mut next = head.cast_mut().cast::<vk::BaseOutStructure<'static>>();
    for link in links.iter_mut() {
        let base = link.base();
        // SAFETY: `base` heads a live structure the box owns.
        unsafe { (*base).p_next = next };
        next = base;
    }
    next.cast_const().cast()
}
