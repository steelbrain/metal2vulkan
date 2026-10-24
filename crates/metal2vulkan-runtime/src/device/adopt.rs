//! Adopting the consumer's Vulkan device: `unsafe Device::from_raw`.
//!
//! The consumer made the instance and the device and keeps them: the runtime loads their
//! function pointers through the consumer's `vkGetInstanceProcAddr`, reads the physical device
//! into a description, restricts it to the [`EnableSet`] the consumer enabled, and builds its
//! compile device from that. Dropping the runtime device destroys only what the runtime made
//! from the device (its pools, memory and objects); the instance and the `VkDevice` stay the
//! consumer's.

use super::{device_error, Device, InstanceShared, Parts};
use crate::error::Error;
use crate::gpu_time;
use crate::image_barrier::MemoryReads;
use crate::loader_word::LoaderWord;
use crate::metal_level::{choose, Host};
use crate::shader_cache::{CacheSetup, ShaderCacheOptions};
use crate::sparse::device::SparseFeatures;
use ash::vk;
use metal2vulkan::device::{DeviceDescription, DevicePolicy, EnableSet, VkExtension};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use std::sync::Arc;

/// The consumer's Vulkan objects a runtime device adopts ([`Device::from_raw`]). No Metal
/// counterpart: an interop hook.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RawDevice {
    /// The `vkGetInstanceProcAddr` the instance was made through.
    pub get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    /// The consumer's instance.
    pub instance: vk::Instance,
    /// The `VkApplicationInfo::apiVersion` the instance was created with.
    pub instance_api_version: metal2vulkan::vk::ApiVersion,
    /// The physical device the device was made on.
    pub physical_device: vk::PhysicalDevice,
    /// The consumer's device, made with what [`enabled`](Self::enabled) states.
    pub device: vk::Device,
    /// The queue family the runtime submits to: it must support graphics and compute work.
    pub queue_family: u32,
    /// The queue of that family the runtime submits to (0 by default). The runtime takes it
    /// under its own lock; the consumer must not submit to it concurrently.
    pub queue_index: u32,
    /// Exactly what the consumer enabled at `vkCreateDevice`, in the compile layer's vocabulary.
    pub enabled: EnableSet,
    /// The extensions outside that vocabulary the consumer enabled that the runtime uses (none
    /// by default): `VK_KHR_calibrated_timestamps` or `VK_EXT_calibrated_timestamps` places
    /// command buffers' GPU times on the host's clock
    /// ([`CommandBuffer::gpu_start_time`](crate::CommandBuffer::gpu_start_time));
    /// `VK_NV_linear_color_attachment`, with its `linearColorAttachment` feature enabled beside
    /// it (and, below Vulkan 1.3, `VK_KHR_format_feature_flags2`, which it requires there), makes
    /// a texture over a buffer's memory render into its linear image in place;
    /// `VK_KHR_external_memory_fd` with `VK_EXT_external_memory_dma_buf` serve
    /// `HostMemory::with_dma_buf`, and with them `VK_EXT_image_drm_format_modifier` with
    /// `VK_EXT_queue_family_foreign` make textures over its pages images rather than shadows.
    pub device_extensions: Vec<CString>,
    /// Where the device keeps translated shaders: by default, the platform's per-user cache
    /// directory.
    pub shader_cache: ShaderCacheOptions,
    /// The sparse members of the core features the consumer enabled (`sparseBinding`,
    /// `sparseResidencyImage2D`, `sparseResidencyImage3D`, `sparseResidency2Samples` to
    /// `16Samples`; every other member is ignored), outside the compile layer's vocabulary: the
    /// runtime makes sparse heaps on what they enable, where the queue family binds sparse
    /// memory. None by default.
    pub sparse_features: SparseFeatures,
    /// Whether the consumer enabled the portability subset's `imageViewFormatReinterpretation`
    /// (`VkPhysicalDevicePortabilitySubsetFeaturesKHR`), outside the compile layer's vocabulary:
    /// false by default, the answer that is never wrong. On a device whose `enabled` lists
    /// `VK_KHR_portability_subset` (MoltenVK), the runtime makes a texture view of another
    /// component layout than its texture's (`RGBA8Unorm` as `R32Uint`, `RGB10A2Unorm` as
    /// `RGB10A2Uint`) only where this is true, and refuses it where it is false (`Unsupported`,
    /// `Feature::ViewFormatReinterpretation`): Vulkan makes no such view on a device the feature
    /// was not enabled on. A device without the subset makes such views whatever this says.
    pub view_format_reinterpretation: bool,
    /// Whether the consumer enabled `hostImageCopy` (`VkPhysicalDeviceHostImageCopyFeatures`,
    /// with `VK_EXT_host_image_copy` and the extensions it requires below Vulkan 1.4), outside
    /// the compile layer's vocabulary: false by default, the answer that is never wrong. Where
    /// it is true, a `Shared` or `Managed` texture's `replace_region` and `get_bytes` copy on the
    /// host, with no submission and no wait for the GPU, where the format and the memory take
    /// host-transfer usage; where it is false they copy on a submission. On a driver whose
    /// compressed images hide writes through an alias and that keeps a host-transfer image's bytes
    /// (ANV), a heap's colour texture takes host-transfer usage only where this is true, and where
    /// it is false a linear image where one serves, else the optimal image, a recorded
    /// near-equivalent (`heap-texture-hides-aliased-writes`).
    pub host_image_copy: bool,
    /// Answer command buffers' GPU times from the host's clock instead of GPU timestamps, as
    /// [`DeviceOptions::gpu_times_from_host`](crate::DeviceOptions::gpu_times_from_host) says:
    /// false by default, Metal's answer, except on a driver whose timestamps wait for every
    /// queue's, which answers from the host's clock whatever this says. Metal has no such
    /// setting.
    pub gpu_times_from_host: bool,
    /// The degradation kinds the device announces at every occurrence, not once, as
    /// [`DeviceOptions::announce_each`](crate::DeviceOptions::announce_each) says: none by
    /// default.
    pub announce_each: BTreeSet<crate::Degradation>,
    /// The degradation kinds that refuse a render encoder's call met while lowering it after it
    /// returned, as
    /// [`DeviceOptions::lowering_refuses`](crate::DeviceOptions::lowering_refuses) says: none by
    /// default.
    pub lowering_refuses: BTreeSet<crate::Degradation>,
    /// The macOS release whose Metal the device answers as, as
    /// [`DeviceOptions::metal_level`](crate::DeviceOptions::metal_level) says: `None` by default,
    /// the host's.
    pub metal_level: Option<crate::MetalLevel>,
}

impl RawDevice {
    /// The objects of one device, its queue family's first queue.
    pub fn new(
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        instance: vk::Instance,
        instance_api_version: metal2vulkan::vk::ApiVersion,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        queue_family: u32,
        enabled: EnableSet,
    ) -> RawDevice {
        RawDevice {
            get_instance_proc_addr,
            instance,
            instance_api_version,
            physical_device,
            device,
            queue_family,
            queue_index: 0,
            enabled,
            device_extensions: Vec::new(),
            shader_cache: ShaderCacheOptions::default(),
            sparse_features: SparseFeatures::default(),
            view_format_reinterpretation: false,
            host_image_copy: false,
            gpu_times_from_host: false,
            announce_each: BTreeSet::new(),
            lowering_refuses: BTreeSet::new(),
            metal_level: None,
        }
    }
}

impl Device {
    /// Adopt the consumer's device: a runtime device that works on `raw`, with its compile
    /// device made with `policy` from the description restricted to `raw.enabled`.
    ///
    /// Errors: a shader cache directory that cannot be used (`ShaderCache`), a device the
    /// compile layer refuses (below Vulkan 1.2, or `raw.enabled` not a valid set for the device:
    /// `Unsupported` or `Compile`), one whose timeline semaphores step less than Vulkan
    /// requires (`Unsupported`, `Feature::TimelineSemaphores`), and a MoltenVK device without
    /// Metal argument buffers (`DriverSetting`): the consumer's instance must leave MoltenVK's
    /// `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` on. And, with no `raw.metal_level`, a macOS host
    /// below every measured release, or whose release cannot be read (`Unproven`,
    /// `Problem::HostReleaseUnmeasured`).
    ///
    /// # Safety
    ///
    /// - `raw.instance` was created through `raw.get_instance_proc_addr` with
    ///   `raw.instance_api_version`, `raw.physical_device` was enumerated from it, and
    ///   `raw.device` was created on that physical device with exactly `raw.enabled` enabled,
    ///   the extensions `raw.device_extensions` names among the others it enabled, and queue
    ///   `raw.queue_index` of family `raw.queue_family`.
    /// - The device also enabled `timelineSemaphore` (core in Vulkan 1.2, outside the compile
    ///   layer's vocabulary): events and the completion of command buffers are timeline
    ///   semaphores.
    /// - `raw.view_format_reinterpretation` is true only if the device enabled the portability
    ///   subset's `imageViewFormatReinterpretation`.
    /// - `raw.host_image_copy` is true only if the device enabled `hostImageCopy`, with
    ///   `VK_EXT_host_image_copy` where the device's Vulkan version (the instance's or the
    ///   physical device's, the lower) is below 1.4, and `VK_KHR_copy_commands2` and
    ///   `VK_KHR_format_feature_flags2` where it is below 1.3.
    /// - All of them outlive the runtime device and every object made from it.
    /// - The device's first word is still the one the loader wrote (no driver has overwritten
    ///   it).
    /// - The consumer does not use the queue, or any object the runtime makes, concurrently
    ///   with the runtime.
    pub unsafe fn from_raw(raw: RawDevice, policy: DevicePolicy) -> Result<Device, Error> {
        let cache = CacheSetup::of(&raw.shader_cache, "from_raw")?;
        let metal_level = choose(raw.metal_level, Host::this(), "from_raw")?;
        let static_fn = ash::StaticFn {
            get_instance_proc_addr: raw.get_instance_proc_addr,
        };
        // SAFETY: the caller's contract: the loader's (or a driver's) `vkGetInstanceProcAddr`.
        let entry = unsafe { ash::Entry::from_static_fn(static_fn) };
        // SAFETY: the caller's contract: the instance was made through this entry point.
        let instance = unsafe { ash::Instance::load(entry.static_fn(), raw.instance) };
        // SAFETY: the caller's contract.
        let description = unsafe {
            DeviceDescription::query(&instance, raw.physical_device, raw.instance_api_version)
        }
        .map_err(device_error)?;
        // SAFETY: the caller's contract.
        unsafe { super::create::timeline_conformant(&instance, raw.physical_device) }?;
        super::create::argument_buffers_kept(&description)?;
        let mut enabled = description
            .with_enabled(&raw.enabled)
            .map_err(device_error)?;
        // The vocabulary does not say whether `VK_EXT_device_generated_commands` was enabled:
        // the runtime makes none of its objects on such a device.
        enabled.generated_commands = None;
        let compile =
            metal2vulkan::device::Device::new(enabled.clone(), policy).map_err(device_error)?;
        // SAFETY: the caller's contract: the word is still the loader's.
        let loader_word = unsafe { LoaderWord::of_device(raw.device) };
        // SAFETY: the device is live; the loader's `vkGetDeviceProcAddr` reads the word, which
        // is the loader's.
        let device = unsafe {
            loader_word.restore();
            ash::Device::load(instance.fp_v1_0(), raw.device)
        };
        // The vocabulary does not say whether a portability subset's
        // `imageViewFormatReinterpretation` was enabled: the consumer says so beside it, and a
        // device with the subset whose consumer does not reinterprets nothing.
        let view_reinterpretation = raw.view_format_reinterpretation
            || !raw
                .enabled
                .extensions
                .contains(&VkExtension::KhrPortabilitySubset);
        // Nor whether `hostImageCopy` was: the consumer says so beside it.
        let host_image_copy = raw.host_image_copy;
        // Nor whether `VK_NV_linear_color_attachment` was: the consumer lists it with its feature
        // (`RawDevice::device_extensions`), at any version, as a made device takes it: a device
        // made with it enabled has what it requires (Vulkan 1.3, or `VK_KHR_format_feature_flags2`,
        // whose formats' properties the description then reads).
        let linear_color_attachment = raw
            .device_extensions
            .iter()
            .any(|name| name.as_c_str() == ash::nv::linear_color_attachment::NAME);
        // Nor whether `VK_EXT_graphics_pipeline_library` was: the consumer lists it and
        // `VK_KHR_pipeline_library` with its feature, and the device's properties say whether it
        // links fast, as a made device takes it.
        let listed: BTreeSet<CString> = raw.device_extensions.iter().cloned().collect();
        // Nor whether the dma-buf route's were (`dma_buf.rs`): the consumer lists them.
        let dma_buf = super::create::dma_buf_names(&listed).is_some();
        let dma_buf_images = super::create::dma_buf_image_names(&listed).is_some();
        let graphics_pipeline_library = super::create::graphics_pipeline_library(&listed, || {
            // SAFETY: the consumer's physical device of its instance, which lists the
            // extension (the consumer enabled it), at Vulkan 1.2 or later.
            unsafe {
                super::create::graphics_pipeline_library_supported(&instance, raw.physical_device)
            }
        });
        // The reads its global memory barriers name: those of the family the runtime submits to,
        // which supports graphics and compute work, with every extension the consumer enabled
        // that the description or the consumer names.
        // SAFETY: the caller's contract: the physical device was enumerated from the instance.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(raw.physical_device) };
        let family = (families.get(raw.queue_family as usize)).map_or(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            |family| family.queue_flags,
        );
        let mut named: Vec<&CStr> = (raw.enabled.extensions.iter())
            .flat_map(|&extension| crate::enables::names_of(extension))
            .collect();
        named.extend(raw.device_extensions.iter().map(CString::as_c_str));
        let memory_reads = MemoryReads::of(family, named);
        let extensions = raw.device_extensions.clone();
        let parts = Parts {
            instance: Arc::new(InstanceShared {
                entry,
                raw: instance,
                api_version: raw.instance_api_version,
                extensions: Vec::new(),
                owned: false,
                messenger: None,
                relay: None,
                settings: [0; 32],
                dropped_layers: Vec::new(),
            }),
            physical: raw.physical_device,
            raw: device,
            loader_word,
            queue_family: raw.queue_family,
            queues: vec![(raw.queue_family, raw.queue_index)],
            present_queues: 0,
            staging: None,
            compile,
            description: enabled,
            enabled: raw.enabled,
            extensions,
            creation_feedback: raw
                .device_extensions
                .iter()
                .any(|name| name.as_c_str() == ash::ext::pipeline_creation_feedback::NAME),
            dma_buf,
            dma_buf_images,
            calibration: gpu_time::Extension::among(&raw.device_extensions.into_iter().collect()),
            gpu_times_from_host: raw.gpu_times_from_host,
            // An adopted device keeps no execution record: validation makes its own devices.
            record_executions: false,
            announce_each: raw.announce_each,
            lowering_refuses: raw.lowering_refuses,
            view_reinterpretation,
            generated_commands: false,
            host_image_copy,
            linear_color_attachment,
            graphics_pipeline_library,
            sparse_features: raw.sparse_features.features(),
            memory_reads,
            owned: false,
            cache,
            metal_level,
        };
        // SAFETY: the caller's contract.
        unsafe { Device::assemble(parts) }
    }
}
