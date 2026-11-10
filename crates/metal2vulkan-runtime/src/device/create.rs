//! Devices the runtime creates: `Device::system_default` and `Device::all`
//! (`MTLCreateSystemDefaultDevice`, `MTLCopyAllDevices`) and their `_with` forms, which take
//! [`DeviceOptions`].
//!
//! The runtime makes a Vulkan 1.2 instance, with `VK_KHR_portability_enumeration` where the loader
//! lists it, then one `VkDevice` per qualifying physical device with the description's
//! [`recommended_enables`](metal2vulkan::device::DeviceDescription::recommended_enables) for the
//! options' policy plus the extensions the options name, and adopts each (create is built on
//! adopt). The compile device is made before `vkCreateDevice`, so a device the compile layer
//! refuses is never opened.
//!
//! Once a device has been made through a loader library, the runtime keeps that library and every
//! Vulkan driver it loaded for the life of the process (`keeper.rs`), so later devices never
//! reload a driver.
//!
//! A physical device has one runtime device per options while any handle to it lives, as Metal
//! has one `MTLDevice` per GPU (an M3 Max and an M4: `MTLCreateSystemDefaultDevice()` twice is one
//! object, `MTLCopyAllDevices()[0]` that object, by pointer and `registryID`).
//! `system_default_with` and `all_with` with equal options ([`DeviceOptions`]' fields, the
//! message sink by identity) hand out the devices they made that still live (`MADE`, held
//! weakly), made from one instance; a device still tears down with its last handle, and a later
//! call makes a fresh one. Devices of other options are other devices (an event handle does not
//! cross to them, `SharedEventAcrossDevices`), as is every adopted device.
//!
//! Everything a caller can change here is an explicit option: layers and their settings,
//! extensions, recommended features and extensions to withhold, where debug-utils messages go,
//! which loader opens, which kind of physical device qualifies, and the shader cache directory.
//! Nothing is read from the environment, except where the platform keeps the default shader cache
//! directory (`HOME`, and `XDG_CACHE_HOME` off macOS) and, on macOS, where the Vulkan SDK is
//! (`VULKAN_SDK`), one of the places the default loader is looked for (`default_loaders`).

use super::instance_requirements::instance_requirements_met;
use super::keeper::keep_drivers_loaded;
use super::line::{Line, Recipe};
use super::properties::{DeviceKind, PhysicalDeviceIdentity};
use super::{device_error, holds, lock, Device, InstanceShared, Parts};
use crate::degradation::{Degradation, DegradeNotice};
use crate::enables::{host_image_copy_names, listed_reported, listed_structures, Enables};
use crate::error::{vulkan, Error, InstanceApiVersion, LoaderNotFound};
use crate::exit_guard;
use crate::gpu_time;
use crate::image_barrier::MemoryReads;
use crate::loader_word::DeviceWord;
use crate::release;
use crate::shader_cache::{CacheSetup, ShaderCacheOptions};
use crate::sparse;
use ash::vk;
use metal2vulkan::device::{
    ApiVersion, DeviceDescription, DevicePolicy, EnableSet, VkExtension, VkFeature,
};
use metal2vulkan::error::{Feature, Unsupported};
use sha2::Digest;
use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

/// How the runtime creates its instance and devices. No Metal counterpart: these choose the
/// Vulkan side (`MTLCreateSystemDefaultDevice` takes nothing).
#[non_exhaustive]
#[derive(Clone, Default)]
pub struct DeviceOptions {
    /// The policy each device's compile device is made with, and whose
    /// `recommended_enables` each device enables.
    pub policy: DevicePolicy,
    /// Instance layers to enable, in order (the first is nearest the application). A layer that
    /// is not installed fails creation.
    pub layers: Vec<CString>,
    /// Settings of those layers, chained at instance creation through `VK_EXT_layer_settings`,
    /// which the instance then enables (a layer that takes settings provides it; without one,
    /// creation fails).
    pub layer_settings: Vec<LayerSetting>,
    /// Instance extensions beyond the ones the runtime enables itself.
    pub instance_extensions: Vec<CString>,
    /// Device extensions beyond the recommended enables (a swapchain, for example). Where one
    /// is an extension whose feature structure the runtime knows (`VK_KHR_` or
    /// `VK_EXT_swapchain_maintenance1`: `swapchainMaintenance1`; `VK_KHR_` or
    /// `VK_EXT_present_mode_fifo_latest_ready`: `presentModeFifoLatestReady`;
    /// `VK_KHR_video_maintenance1`: `videoMaintenance1`; `VK_KHR_video_maintenance2`:
    /// `videoMaintenance2`; `VK_KHR_unified_image_layouts`: `unifiedImageLayouts` and
    /// `unifiedImageLayoutsVideo`) and the device lists it, the device enables the structure's
    /// features too; a device that lacks one Vulkan requires of every device listing the
    /// extension (all of them but `unifiedImageLayoutsVideo`) is refused
    /// ([`Error::ExtensionFeature`]) rather than enable the
    /// extension without it. `unifiedImageLayoutsVideo` is enabled where the device has it
    /// (`vkGetPhysicalDeviceFeatures2` on [`DeviceHandles`](crate::DeviceHandles) answers
    /// whether it does). One whose registry dependency names an instance extension the
    /// instance would not enable (`VK_KHR_swapchain` without `VK_KHR_surface` in
    /// `instance_extensions`, for example) refuses the options before the instance is made
    /// ([`Error::InstanceExtensionRequired`]).
    pub device_extensions: Vec<CString>,
    /// Device extensions each device enables only where it can, never refused for one: where the
    /// device lists it, the instance enables every instance extension its registry dependency
    /// names (as for `device_extensions`), and, for an extension whose feature structure the
    /// runtime knows (as for `device_extensions`), the device has every required feature of it,
    /// which it then enables too, with each optional one it has (`unifiedImageLayoutsVideo`).
    /// Elsewhere the device leaves it out, and is made all the same. Which a
    /// device enabled, [`Device::enables_extension`](crate::Device::enables_extension) answers
    /// (window-system extensions, for a consumer whose windows present where the device can).
    /// One also in `device_extensions` is that list's. Metal has no such setting.
    pub optional_device_extensions: Vec<CString>,
    /// Features of the recommended enables a created device leaves disabled (the robustness
    /// features, for a run of the in-shader containment encoding, for example). The compile
    /// device is made from what is enabled, so it compiles for that. Withholding a feature another
    /// enabled feature requires refuses the device (`DeviceDescription::with_enabled`'s check).
    pub withheld_features: BTreeSet<VkFeature>,
    /// Extensions a created device leaves disabled, of the recommended enables and
    /// `VK_EXT_external_memory_host` (a run without `VK_EXT_shader_stencil_export`, the
    /// shader-side stencil's, for example). The compile device is made from what is enabled, so it
    /// compiles for that. Withholding an extension an enabled feature or another enabled extension
    /// requires refuses the device (`DeviceDescription::with_enabled`'s check).
    pub withheld_extensions: BTreeSet<VkExtension>,
    /// Where the instance's `VK_EXT_debug_utils` messages of warning and error severity go. When
    /// set, the instance enables `VK_EXT_debug_utils` and reports through it from its creation
    /// to its destruction. Its devices announce their degradations here too, once per device and
    /// kind ([`DebugMessage::degradation`]); without a sink, on stderr.
    pub messages: Option<MessageSink>,
    /// The Vulkan loader library to open instead of the default ones: by default the platform's
    /// own by name, then on macOS `libvulkan.1.dylib` in the Vulkan SDK (`VULKAN_SDK`),
    /// Homebrew's `/opt/homebrew/lib` and `/usr/local/lib`. A loader that opens from none of
    /// them is [`Error::LoaderNotFound`], naming each.
    pub loader: Option<PathBuf>,
    /// Only physical devices of these kinds qualify; empty admits every kind.
    pub kinds: Vec<DeviceKind>,
    /// Where the devices keep translated shaders: by default, the platform's per-user cache
    /// directory.
    pub shader_cache: ShaderCacheOptions,
    /// Leave `VK_EXT_device_generated_commands` disabled on a device that has it. The runtime
    /// enables it otherwise (with `VK_KHR_maintenance5`, which it requires), and runs an
    /// execution of commands a kernel writes on the GPU alone where every record it may hold is
    /// covered; without it every such execution takes the host step. The compile device states the
    /// extension only where it is enabled.
    pub withhold_generated_commands: bool,
    /// Make no Vulkan view of an image in another component layout, on any device, as a device
    /// whose `VK_KHR_portability_subset` withholds `imageViewFormatReinterpretation` makes none:
    /// a texture view of another layout (`RGBA8Unorm` as `R32Uint`) then holds its texels apart,
    /// in a shadow the encoders copy (`Degradation::ShadowedTextureView`), and the subset's
    /// feature stays disabled. For runs of the path such a device takes.
    pub withhold_view_reinterpretation: bool,
    /// The Vulkan API version the instance requests (`VkApplicationInfo::apiVersion`), at which
    /// the runtime reads each physical device's description; Vulkan 1.2 when `None`. Metal has
    /// no such setting: it is a Vulkan-side option, for a consumer whose layers read later
    /// structures only under a later instance version (the profiles layer reads a profile's
    /// Vulkan 1.3 and later structures only then). A request below 1.2, the runtime's floor, or
    /// above the version the loader reports (`vkEnumerateInstanceVersion`) refuses creation
    /// (`Error::InstanceApiVersion`); the runtime never clamps it.
    pub instance_api_version: Option<ApiVersion>,
    /// Answer command buffers' `GPUStartTime` and `GPUEndTime` from the host's clock instead of
    /// GPU timestamps, which no command buffer then writes. Off by default: Metal's answer,
    /// except on a driver whose timestamps wait for every queue's
    /// (`DriverFact::TimestampsWaitForEveryQueue`: MoltenVK 1.4.1 on Apple's GPUs), which answers
    /// from the host's clock whatever this says. There, a command buffer with work answers as its
    /// GPU start its `kernelEndTime` (or its queue's latest GPU end before it, where later) and as
    /// its GPU end the host time the runtime saw it complete: the window its work ran in, which
    /// keeps every rule of Metal's own times (`gpu_time.rs`), but not the GPU's own times. The
    /// device announces `Degradation::GpuTimesFromHost` once, at the first such answer. It spares
    /// each command buffer its two timestamps, which cost about 150 us of latency on MoltenVK; a
    /// device whose queue family writes no timestamps then answers GPU times instead of refusing
    /// them. Metal has no such setting.
    pub gpu_times_from_host: bool,
    /// The class of Mac each device presents ([`MacClass`](crate::MacClass)), whose Metal's
    /// answers it gives where a Mac's GPU decides them: what a new `Private` texture reads before
    /// anything writes it, the encoding a GPU blit through a compressed texture writes, and a
    /// `half` NaN clear's sign ([`DeviceProperties::lossless_compression`](crate::DeviceProperties::lossless_compression)
    /// states it). `None` by default: the class of the device's GPU
    /// ([`MacClass::of_vendor`](crate::MacClass::of_vendor): an Apple GPU presents an Apple
    /// silicon Mac, every other GPU a Mac with an Intel or AMD GPU). A consumer that presents
    /// another Mac than its GPU's (an x86-64 macOS guest, an Intel or AMD Mac, on an Apple GPU)
    /// declares it. No Metal counterpart: a Mac's GPU is what it is.
    pub mac_class: Option<crate::MacClass>,
    /// The degradation kinds each device announces at every occurrence, not once (empty by
    /// default: each kind once per device). Each occurrence's notice is the same
    /// [`DegradeNotice`] through the same sink (or stderr), on the thread whose call announces
    /// it, so a program that runs tasks on threads over one device sees each task's: where each
    /// kind's notice surfaces is in its [`Degradation`] docs. A kind a
    /// pipeline variant's making announces (a stage's precision notes, specialized or not, and
    /// the other kinds its stages or requirements state) occurs at each dispatch or draw that
    /// binds the variant, which announces it on the encoding thread (on the committing or
    /// scheduler thread for an indirect command's), not at the making: the device makes a
    /// variant once. No Metal counterpart.
    pub announce_each: BTreeSet<crate::Degradation>,
    /// The kinds of degradation that, met while the runtime lowers a render encoder's call after
    /// it returned (`RenderLoweringPolicy::Worker`), refuse the call rather than
    /// being announced: at every occurrence there, the call fails its command buffer
    /// (`CommandBufferError::CallRefused`), whose refusal is the notice
    /// (`Error::Degraded`), and nothing of that encoder or of any after it runs. Empty by default,
    /// where every degradation is announced and the work runs; a consumer that ends its program
    /// at a kind's notice in the call names that kind here, so the work it ends at never runs.
    /// No Metal counterpart.
    pub lowering_refuses: BTreeSet<crate::Degradation>,
    /// Keep each command buffer's execution record
    /// ([`CommandBuffer::executions`](crate::CommandBuffer::executions)): the pipeline each
    /// dispatch and draw ran, the bind shape it was planned for, and its `VkPipeline`s' creation
    /// times. Off by default: the device's encoders build no record and keep nothing for one, so
    /// a draw pays nothing for a record no one reads. Validation's executor and lanes turn it on.
    /// No Metal counterpart.
    #[doc(hidden)]
    pub record_executions: bool,
    /// Make each device's check of the first move of an image over host pages fail whatever the
    /// moves did (`in_place_check.rs`), and keep no record of it: every texture over host pages
    /// is then a shadow, its notice naming the failed check. Off by default. A test hook, for the
    /// lanes that hold the shadow to a failed check. No Metal counterpart.
    #[doc(hidden)]
    pub fail_in_place_check: bool,
    /// Take each device's submission check of merged submissions (`submit_check.rs`,
    /// `Check::Merge`) as holding without its answer, and keep no record of its checks: under
    /// `CompletionPolicy::Grouped` its batches then wait in groups whatever the timing it
    /// measured, where a driver that runs one `vkQueueSubmit` of several `VkSubmitInfo`s as
    /// several kernel submissions runs them as correctly, saving none. Its import tax and follows
    /// checks answer as ever. Off by default. A test hook, for the lanes that hold a device to the
    /// grouped path on every run. No Metal counterpart.
    #[doc(hidden)]
    pub assume_submissions_merge: bool,
    /// Turn each device's pending audit on: a diagnostic that checks, at each begin of a
    /// `VkCommandBuffer`, reset or give-back of its pool and submission of it again, that the
    /// GPU completed every submission that carries it, and logs each one that did not (id name
    /// `metal2vulkan-pending-audit`, `pending_audit.rs`); it changes no work. Off by default;
    /// `METAL2VULKAN_PENDING_AUDIT`, set, turns it on for every device too. For the tests that
    /// hold it to a finding. No Metal counterpart.
    #[doc(hidden)]
    pub pending_audit: bool,
    /// Color formats each device takes as no color attachment, as a device without the
    /// attachment feature of them does: their tilings' `COLOR_ATTACHMENT` and blend features and
    /// their attachment sample counts are left out of the description the device plans and
    /// compiles by (`withhold_attachments`). Empty by default. A test hook, for the lanes that run
    /// the path a device that renders a format only through the view of its word takes
    /// (`metal::format::WordAttachment`, its multisampled textures held in the word's format) on a
    /// device that renders the format. No Metal counterpart.
    #[doc(hidden)]
    pub withheld_attachment_formats: BTreeSet<metal2vulkan::vk::Format>,
    /// Where each device hands its pipeline records: one [`PipelineRecord`](crate::PipelineRecord)
    /// per pipeline state creation that made its state (compute, render, mesh render and tile
    /// render, and the additional-binary-functions forms; and the one-sample or factor-scaling
    /// twin the runtime makes of a render state where a draw first needs it), on the creating
    /// thread before the call returns, as the program's own code (a panic in it is caught). A
    /// creation that fails, a binary archive's add and a warm pass's make none. A consumer keeps
    /// them, in the order they came, and hands a later process's device them
    /// ([`Device::warm`](crate::Device::warm)) to make the same translations and `VkPipeline`s
    /// before its program asks. `None` by default: no record is made. Devices of one hook are
    /// devices of one option (it is compared by identity, as [`messages`](Self::messages) is).
    /// No Metal counterpart: Metal keeps its own compiled pipelines in the OS's caches.
    pub pipeline_records: Option<Arc<dyn Fn(crate::PipelineRecord) + Send + Sync>>,
    /// Queues each device makes for the caller beside its own, one per entry, of a family whose
    /// flags contain the entry's (`VIDEO_DECODE_KHR` for a consumer's video decode, for
    /// example), which [`DeviceHandles::extra_queues`](crate::DeviceHandles::extra_queues)
    /// returns. Each is a queue no runtime work uses: of the first family whose flags contain
    /// the entry's and that has a queue the device's own plan leaves free, at its lowest free
    /// index; where no such family has one left, the last queue of such a family the device would
    /// take for itself, which it then leaves to the caller (a queue reserved for presenting
    /// first, then the round robin's last, then the staging queue's), never its first queue,
    /// where internal submissions go. A device with no queue for an entry (no family whose flags
    /// contain it, empty flags, or only the device's first queue) is made all the same, without
    /// it, and says so once in an informational message
    /// (`metal2vulkan-extra-queues-absent`, through [`messages`](Self::messages) or on stderr).
    /// The runtime never submits to these queues: the caller is their only submitter, and
    /// orders its work against the runtime's through semaphores
    /// ([`Event::reserve_vk_signal`](crate::Event::reserve_vk_signal),
    /// [`Event::vk_wait`](crate::Event::vk_wait), and a shared event's same hooks). The runtime's
    /// own buffers and images stay shared among its own queue families
    /// ([`DeviceHandles::queue_families`](crate::DeviceHandles::queue_families)), which need not
    /// include an extra queue's: a submission to an extra queue of another family uses only
    /// objects made to be shared with that family, the caller's own (which Metal-side code
    /// reaches through [`Device::new_buffer_with_vk_buffer`](crate::Device::new_buffer_with_vk_buffer)
    /// or [`Device::new_texture_with_vk_image_plane`](crate::Device::new_texture_with_vk_image_plane)).
    /// Empty by default. No Metal counterpart: a Vulkan-side option.
    pub extra_queues: Vec<vk::QueueFlags>,
    /// How many queues of its own queue families each device reserves for the caller, which
    /// [`DeviceHandles::reserved_queues`](crate::DeviceHandles::reserved_queues) returns: queues
    /// it would otherwise lend the round robin its command queues share
    /// ([`CommandQueue`](crate::CommandQueue)), so that round robin shrinks by each one reserved,
    /// and the runtime never submits to one (no command queue, presenting or not, takes one; no
    /// internal or staging copy goes there). They are the round robin's last queues: it keeps its
    /// first four, and the two queues reserved for presenting queues keep theirs, so a device
    /// reserves only queues past those six (on an RTX 5080, whose graphics family has 16, one
    /// reserved queue leaves the round robin 13 and presenting 2). A family with queues past the
    /// round robin's ceiling of 16 and presenting's two gives those first, and its round robin
    /// does not shrink. A device with fewer to spare reserves those it has (none on MoltenVK's
    /// four families of one queue each, which its round robin takes, or lavapipe's one queue), is
    /// made all the same, and says so once in an informational message
    /// (`metal2vulkan-reserved-queues-absent`, through [`messages`](Self::messages) or on
    /// stderr). They are reserved before the [`extra_queues`](Self::extra_queues) are placed, so
    /// no extra queue takes one.
    ///
    /// A reserved queue's family is one of the device's own
    /// ([`DeviceHandles::queue_families`](crate::DeviceHandles::queue_families)), which its
    /// buffers and images are shared among (`EXCLUSIVE` on a device of one family, `CONCURRENT`
    /// among them otherwise): a submission there may use the runtime's buffers and images
    /// ([`Buffer::vk_buffer`](crate::Buffer::vk_buffer),
    /// [`Texture::vk_image`](crate::Texture::vk_image)) as well as the caller's, with no
    /// ownership transfer, under those handles' contracts: while the object lives, ordered
    /// against the runtime's work on it through semaphores
    /// ([`Event::vk_wait`](crate::Event::vk_wait),
    /// [`Event::reserve_vk_signal`](crate::Event::reserve_vk_signal), a shared event's same
    /// hooks, [`CommandBuffer::encode_wait_for_vk_semaphore`](crate::CommandBuffer::encode_wait_for_vk_semaphore),
    /// [`CommandBuffer::encode_signal_vk_semaphore`](crate::CommandBuffer::encode_signal_vk_semaphore))
    /// or completion, and an image the runtime made left in `VK_IMAGE_LAYOUT_GENERAL`. The caller
    /// is the queue's only submitter: its submissions there sit in no command queue's order, and
    /// hold none back. 0 by default. No Metal counterpart: a Vulkan-side option.
    pub reserved_queues: u32,
    /// Fewer `VkQueue`s for each device than its queue plan takes: at most
    /// [`round_robin`](QueueLimits::round_robin) for the round robin its command queues share,
    /// and at most [`presenting`](QueueLimits::presenting) reserved for presenting queues. `None`
    /// (the default) takes the plan: the round robin up to 16 (at least four where the families
    /// have them, NVIDIA's graphics family of 16 giving it 14) and two for presenting.
    ///
    /// What it changes: how many `VkQueue`s the device makes, and so what making and dropping it
    /// cost. Each queue is made with the device and destroyed with it: on an RTX 5080 Laptop GPU
    /// (NVIDIA 615.71.09), one device made and dropped at a time, the plan's 17 queues (14 for the
    /// round robin, 2 for presenting and the staging queue) took 445 ms in `vkCreateDevice` and
    /// 110 ms in `vkDestroyDevice`, and 2 queues (limits of one for the round robin and none for
    /// presenting) 145 ms and 44 ms, about 20 ms a queue to make; and the driver makes devices
    /// one at a time across processes. On a device whose families have no queue to spare (RADV's
    /// and lavapipe's one graphics queue) the limits leave the plan as it is.
    ///
    /// What it costs a program: its command queues share `VkQueue`s past the limit, so one
    /// queue's command buffers wait behind another's whole command buffer where Metal's would not
    /// ([`CommandQueue`](crate::CommandQueue)); a queue made by
    /// [`Device::new_command_queue_unshared`](crate::Device::new_command_queue_unshared) may share
    /// too, and one made for presenting past the limit's shares with the round robin
    /// (`shared-present-queue`) or with an earlier presenting queue. The reserved and extra
    /// queues take the queues the limits leave. Wherever the limits change a placement from what
    /// the plan gives, the device says so in an informational message
    /// (`metal2vulkan-queue-limits-reached`, through [`messages`](Self::messages) or on stderr):
    /// a command queue, unshared or not, that takes a `VkQueue` another took where the plan gives
    /// it one of its own; a presenting queue past the limit's where the plan reserves it one;
    /// reserved or extra queues placed otherwise than the plan places them (as the device is
    /// made); and each read of the device's queue sharing (`Device::settle_queue_switch_check`,
    /// `Device::queue_switch_check_report`) on a device whose queues the limits changed. No Metal
    /// counterpart: a Vulkan-side option, for tests and tools that make many devices.
    pub queue_limits: Option<QueueLimits>,
}

/// The most `VkQueue`s a device takes for its command queues ([`DeviceOptions::queue_limits`]).
/// No Metal counterpart: a Vulkan-side option.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueueLimits {
    /// The most the round robin its command queues share takes: at least one (0 is taken as 1),
    /// and the plan's ceiling of 16 at most.
    pub round_robin: u32,
    /// The most reserved for presenting queues: the plan's two at most.
    pub presenting: u32,
}

impl std::fmt::Debug for DeviceOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Every field is named, so a field added later is written here or this fails to compile.
        let DeviceOptions {
            policy,
            layers,
            layer_settings,
            instance_extensions,
            device_extensions,
            optional_device_extensions,
            withheld_features,
            withheld_extensions,
            messages,
            loader,
            kinds,
            shader_cache,
            withhold_generated_commands,
            withhold_view_reinterpretation,
            instance_api_version,
            gpu_times_from_host,
            mac_class,
            announce_each,
            lowering_refuses,
            record_executions,
            fail_in_place_check,
            assume_submissions_merge,
            pending_audit,
            withheld_attachment_formats,
            pipeline_records,
            extra_queues,
            reserved_queues,
            queue_limits,
        } = self;
        f.debug_struct("DeviceOptions")
            .field("policy", policy)
            .field("layers", layers)
            .field("layer_settings", layer_settings)
            .field("instance_extensions", instance_extensions)
            .field("device_extensions", device_extensions)
            .field("optional_device_extensions", optional_device_extensions)
            .field("withheld_features", withheld_features)
            .field("withheld_extensions", withheld_extensions)
            .field("messages", messages)
            .field("loader", loader)
            .field("kinds", kinds)
            .field("shader_cache", shader_cache)
            .field("withhold_generated_commands", withhold_generated_commands)
            .field(
                "withhold_view_reinterpretation",
                withhold_view_reinterpretation,
            )
            .field("instance_api_version", instance_api_version)
            .field("gpu_times_from_host", gpu_times_from_host)
            .field("mac_class", mac_class)
            .field("announce_each", announce_each)
            .field("lowering_refuses", lowering_refuses)
            .field("record_executions", record_executions)
            .field("fail_in_place_check", fail_in_place_check)
            .field("assume_submissions_merge", assume_submissions_merge)
            .field("pending_audit", pending_audit)
            .field("withheld_attachment_formats", withheld_attachment_formats)
            .field(
                "pipeline_records",
                &pipeline_records.as_ref().map(|_| "hook"),
            )
            .field(
                "extra_queues",
                &(extra_queues.iter())
                    .map(|&flags| super::handles::QueueFlagsText(flags))
                    .collect::<Vec<_>>(),
            )
            .field("reserved_queues", reserved_queues)
            .field("queue_limits", queue_limits)
            .finish()
    }
}

/// One setting of an instance layer (`VkLayerSettingEXT`).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerSetting {
    /// The layer, as [`DeviceOptions::layers`] names it.
    pub layer: CString,
    /// The setting's name, as the layer documents it.
    pub name: CString,
    /// The setting's value.
    pub value: LayerSettingValue,
}

impl LayerSetting {
    /// `layer`'s setting `name` of `value`.
    pub fn new(layer: CString, name: CString, value: LayerSettingValue) -> LayerSetting {
        LayerSetting { layer, name, value }
    }
}

/// A layer setting's value.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerSettingValue {
    /// One `VkBool32`.
    Bool(bool),
    /// Strings.
    Strings(Vec<CString>),
}

/// The `VkLayerSettingEXT`s of `settings`, pointing into them and into `bools`, which holds one
/// `VkBool32` per setting (its value for a boolean one), and `strings`, which holds each setting's
/// string pointers (empty for a boolean one): both made by [`setting_values`].
pub(crate) fn layer_settings<'a>(
    settings: &'a [LayerSetting],
    bools: &'a [vk::Bool32],
    strings: &'a [Vec<*const c_char>],
) -> Vec<vk::LayerSettingEXT<'a>> {
    settings
        .iter()
        .zip(bools.iter().zip(strings))
        .map(|(setting, (value, pointers))| {
            let mut raw = vk::LayerSettingEXT::default()
                .layer_name(&setting.layer)
                .setting_name(&setting.name);
            // `LayerSettingEXT::values` counts bytes, not values: the count is set directly.
            match &setting.value {
                LayerSettingValue::Bool(_) => {
                    raw = raw.ty(vk::LayerSettingTypeEXT::BOOL32);
                    raw.value_count = 1;
                    raw.p_values = std::ptr::from_ref(value).cast();
                }
                LayerSettingValue::Strings(_) => {
                    raw = raw.ty(vk::LayerSettingTypeEXT::STRING);
                    raw.value_count = pointers.len() as u32;
                    raw.p_values = pointers.as_ptr().cast();
                }
            }
            raw
        })
        .collect()
}

/// The values [`layer_settings`] points at: per setting, its `VkBool32` and its string pointers.
pub(crate) fn setting_values(
    settings: &[LayerSetting],
) -> (Vec<vk::Bool32>, Vec<Vec<*const c_char>>) {
    settings
        .iter()
        .map(|setting| match &setting.value {
            LayerSettingValue::Bool(value) => (vk::Bool32::from(*value), Vec::new()),
            LayerSettingValue::Strings(strings) => (
                vk::FALSE,
                strings.iter().map(|string| string.as_ptr()).collect(),
            ),
        })
        .unzip()
}

/// No Metal counterpart: a receiver of the instance's debug-utils messages, and of its devices'
/// degrade notices. It is called on whichever thread the driver, the layer, or the runtime
/// reports from, so it must be `Send + Sync`; a panic inside it is caught.
#[derive(Clone)]
pub struct MessageSink {
    receive: Arc<dyn Fn(&DebugMessage) + Send + Sync>,
    /// The least severity of the messages it hears, but degrade notices
    /// ([`with_least_severity`](Self::with_least_severity)).
    least: MessageSeverity,
}

impl MessageSink {
    /// A sink calling `receive` with each message, of every severity.
    pub fn new(receive: impl Fn(&DebugMessage) + Send + Sync + 'static) -> MessageSink {
        MessageSink {
            receive: Arc::new(receive),
            least: MessageSeverity::Verbose,
        }
    }

    /// No Metal counterpart: the sink, hearing no message of a severity below `least` but the
    /// devices' degrade notices ([`DebugMessage::degradation`]), which it hears whatever `least`
    /// says. The runtime makes none of the messages of its own the sink skips: it formats no text
    /// for them (a host import freed on a device that keeps MoltenVK's residency set, a `Verbose`
    /// line, `metal2vulkan-host-import-freed`, the only one that may come at every call).
    pub fn with_least_severity(self, least: MessageSeverity) -> MessageSink {
        MessageSink { least, ..self }
    }

    /// Whether `other` is this sink (a clone of it, hearing the same severities), not merely one
    /// that does the same.
    pub(crate) fn same(&self, other: &MessageSink) -> bool {
        Arc::ptr_eq(&self.receive, &other.receive) && self.least == other.least
    }

    /// Whether the sink hears messages of `severity` that are no degrade notice.
    pub(crate) fn hears(&self, severity: MessageSeverity) -> bool {
        severity >= self.least
    }

    /// Hand `message` to the sink, where it hears it; a panic inside it is caught. On a thread
    /// inside a passing hold of the device's resources, once its outermost ended
    /// (`resource_ids.rs`): the program's code runs in none.
    pub(crate) fn deliver(&self, message: &DebugMessage) {
        if message.degradation.is_none() && !self.hears(message.severity) {
            return;
        }
        if crate::resource_ids::held_past_passing_hold(self, message) {
            return;
        }
        crate::exit_guard::program(|| (self.receive)(message));
    }
}

impl std::fmt::Debug for MessageSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MessageSink")
    }
}

/// One debug-utils message, or a device's degrade notice. No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugMessage {
    /// The message's severity; a degrade notice's is `Warning`.
    pub severity: MessageSeverity,
    /// What the message is about.
    pub kind: MessageKind,
    /// `pMessageIdName` (a validation message's VUID), empty when absent; a degrade notice's
    /// code.
    pub id_name: String,
    /// `messageIdNumber`; 0 for a degrade notice.
    pub id: i32,
    /// `pMessage`; a degrade notice's text.
    pub text: String,
    /// The runtime's own notice of a degradation it takes, once per device and kind: never a
    /// driver's or a layer's message. Match its [`Degradation`], not the text.
    pub degradation: Option<DegradeNotice>,
}

/// A message's severity.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MessageSeverity {
    /// `VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT`.
    Verbose,
    /// `VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT`.
    Info,
    /// `VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT`, and the runtime's degrade notices.
    Warning,
    /// `VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT`.
    Error,
}

/// What a message is about (`VkDebugUtilsMessageTypeFlagsEXT`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MessageKind {
    /// A general event.
    pub general: bool,
    /// A violation of the specification (a validation message).
    pub validation: bool,
    /// A use that is valid but not optimal.
    pub performance: bool,
}

/// The instance's persistent messenger and the sink it reports to.
pub(super) struct Messenger {
    loader: ash::ext::debug_utils::Instance,
    handle: vk::DebugUtilsMessengerEXT,
    /// The callback's user data; it outlives the instance (dropped after `vkDestroyInstance`,
    /// which reports through the messenger chained at creation).
    sink: Box<MessageSink>,
}

impl Messenger {
    /// The sink the messenger reports to.
    pub(super) fn sink(&self) -> &MessageSink {
        &self.sink
    }

    /// Destroy the persistent messenger; the instance is destroyed next.
    pub(super) fn destroy(&self) {
        // SAFETY: the messenger belongs to the live instance and is destroyed once.
        unsafe { release::messenger(&self.loader, self.handle) };
    }
}

/// The debug-utils callback: hands the message to the [`MessageSink`] at `user`.
pub(super) unsafe extern "system" fn deliver(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    let text = |pointer: *const c_char| {
        if pointer.is_null() {
            String::new()
        } else {
            // SAFETY: the layer passes NUL-terminated strings that live for the call.
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        }
    };
    if data.is_null() || user.is_null() {
        return vk::FALSE;
    }
    // SAFETY: `data` is the layer's callback data for this call.
    let data = unsafe { &*data };
    // SAFETY: `user` is the boxed sink the messenger was created with, alive with the instance.
    let sink = unsafe { &*user.cast::<MessageSink>() };
    use vk::DebugUtilsMessageSeverityFlagsEXT as S;
    use vk::DebugUtilsMessageTypeFlagsEXT as T;
    let message = DebugMessage {
        severity: if severity.contains(S::ERROR) {
            MessageSeverity::Error
        } else if severity.contains(S::WARNING) {
            MessageSeverity::Warning
        } else if severity.contains(S::INFO) {
            MessageSeverity::Info
        } else {
            MessageSeverity::Verbose
        },
        kind: MessageKind {
            general: types.contains(T::GENERAL),
            validation: types.contains(T::VALIDATION),
            performance: types.contains(T::PERFORMANCE),
        },
        id_name: text(data.p_message_id_name),
        id: data.message_id_number,
        text: text(data.p_message),
        degradation: None,
    };
    // A driver diagnostic the device's driver facts measure this thread's current call to
    // report is withheld, and counted (`driver_facts.rs`).
    if crate::driver_facts::withholds(&message) {
        return vk::FALSE;
    }
    // On the runtime's thread making a call for an asking thread, held: the asking thread
    // delivers it once the call returns (`compiles.rs`).
    if super::compiles::held(sink, &message) {
        return vk::FALSE;
    }
    // A panic must not unwind into the driver.
    sink.deliver(&message);
    vk::FALSE
}

/// The instance extensions a loader lists, read with one `vkEnumerateInstanceExtensionProperties`:
/// the loader scans its driver and layer manifests again at each such call, and loads and asks
/// each driver it finds, so the making of an instance reads the list once and answers every
/// question of it from this.
pub(crate) struct ListedExtensions(Vec<vk::ExtensionProperties>);

/// The room [`ListedExtensions::of`] gives its first call: more extensions than a loader lists
/// (20 on lavapipe's alone, 26 with RADV's beside it).
pub(crate) const LISTED_ROOM: usize = 64;

impl ListedExtensions {
    /// The instance extensions `entry`'s loader lists, read in one call into room for
    /// [`LISTED_ROOM`] of them, and again into twice the room while the loader answers
    /// `VK_INCOMPLETE`. Not the usual two-call read (the count, then the list), which is two
    /// scans of the manifests.
    pub(crate) fn of(entry: &ash::Entry) -> Result<ListedExtensions, Error> {
        let enumerate = entry.fp_v1_0().enumerate_instance_extension_properties;
        let mut room = LISTED_ROOM;
        loop {
            let mut listed = vec![vk::ExtensionProperties::default(); room];
            let mut count = u32::try_from(room).unwrap_or(u32::MAX);
            // SAFETY: a global command of a loaded entry, for no layer, into `count` properties.
            let answer = unsafe { enumerate(std::ptr::null(), &mut count, listed.as_mut_ptr()) };
            match answer {
                vk::Result::SUCCESS => {
                    listed.truncate(count as usize);
                    return Ok(ListedExtensions(listed));
                }
                vk::Result::INCOMPLETE => room *= 2,
                error => return Err(vulkan(error)),
            }
        }
    }

    /// Whether the loader listed the instance extension `name`.
    pub(crate) fn lists(&self, name: &CStr) -> bool {
        self.0
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    }
}

/// The instance API version `requested` asks for (`DeviceOptions::instance_api_version`):
/// Vulkan 1.2 when `None`. Refused below 1.2, and above `loader`'s answer, the version the
/// loader reports, asked only for a request.
pub(crate) fn instance_version(
    requested: Option<ApiVersion>,
    loader: impl FnOnce() -> Result<ApiVersion, Error>,
) -> Result<ApiVersion, Error> {
    let Some(requested) = requested else {
        return Ok(ApiVersion::V1_2);
    };
    if requested < ApiVersion::V1_2 {
        return Err(Error::InstanceApiVersion(InstanceApiVersion::BelowFloor {
            requested,
        }));
    }
    let loader = loader()?;
    if requested > loader {
        return Err(Error::InstanceApiVersion(InstanceApiVersion::AboveLoader {
            requested,
            loader,
        }));
    }
    Ok(requested)
}

/// The Vulkan loader libraries the runtime tries, in order, when the options name none: the
/// platform's own by name, which the dynamic linker searches for, then on macOS
/// `libvulkan.1.dylib` in the Vulkan SDK's `lib` directory (`vulkan_sdk`, the `VULKAN_SDK`
/// variable, where set), in Homebrew's `/opt/homebrew/lib` and in `/usr/local/lib`, which no
/// default search path holds there.
pub(crate) fn default_loaders(vulkan_sdk: Option<&Path>) -> Vec<PathBuf> {
    if cfg!(target_os = "macos") {
        let directories = vulkan_sdk
            .map(|sdk| sdk.join("lib"))
            .into_iter()
            .chain(["/opt/homebrew/lib", "/usr/local/lib"].map(PathBuf::from));
        ["libvulkan.dylib", "libvulkan.1.dylib"]
            .map(PathBuf::from)
            .into_iter()
            .chain(directories.map(|directory| directory.join("libvulkan.1.dylib")))
            .collect()
    } else if cfg!(windows) {
        vec![PathBuf::from("vulkan-1.dll")]
    } else if cfg!(target_os = "android") {
        vec![PathBuf::from("libvulkan.so")]
    } else {
        vec![PathBuf::from("libvulkan.so.1")]
    }
}

/// The Vulkan loader: `loader` where the options name one, else the first of
/// [`default_loaders`] that opens. [`Error::LoaderNotFound`] names each library tried.
pub(crate) fn open_loader(loader: Option<&Path>) -> Result<ash::Entry, Error> {
    let tried = match loader {
        Some(path) => vec![path.to_path_buf()],
        None => default_loaders(std::env::var_os("VULKAN_SDK").as_deref().map(Path::new)),
    };
    for path in &tried {
        // SAFETY: opening the loader runs its initialization; the runtime assumes nothing else.
        if let Ok(entry) = unsafe { ash::Entry::load_from(path) } {
            return Ok(entry);
        }
    }
    Err(Error::LoaderNotFound(LoaderNotFound { tried }))
}

/// `vkCreateInstance` of `info` from `entry`: every instance the runtime creates, its devices' and
/// the drivers' keeper's (`keeper.rs`), is made here, and the exit guard's hook registers again
/// after it, whatever it answered ([`exit_guard::instance_made`]): a layer registers an exit
/// handler at each such call.
///
/// # Safety
///
/// `info` and everything it points at live for the call.
pub(super) unsafe fn new_instance(
    entry: &ash::Entry,
    info: &vk::InstanceCreateInfo<'_>,
) -> ash::prelude::VkResult<ash::Instance> {
    // SAFETY: the caller's.
    let made = unsafe { entry.create_instance(info, None) };
    exit_guard::instance_made();
    made
}

/// The instance of `options`.
fn create_instance(options: &DeviceOptions) -> Result<Arc<InstanceShared>, Error> {
    instance_of(open_loader(options.loader.as_deref())?, options)
}

/// The instance of `options` from the loader `entry`, which lists its instance extensions once
/// ([`ListedExtensions`]).
pub(crate) fn instance_of(
    entry: ash::Entry,
    options: &DeviceOptions,
) -> Result<Arc<InstanceShared>, Error> {
    let listed = ListedExtensions::of(&entry)?;
    let portability = listed.lists(ash::khr::portability_enumeration::NAME);
    let mut extensions: Vec<&CStr> = Vec::new();
    if portability {
        extensions.push(ash::khr::portability_enumeration::NAME);
    }
    // The relay of a driver's command-buffer failures needs a messenger of its own
    // (`relay.rs`): debug utils wherever the instance lists them.
    let relays = listed.lists(ash::ext::debug_utils::NAME);
    if options.messages.is_some() || relays {
        extensions.push(ash::ext::debug_utils::NAME);
    }
    // MoltenVK's Metal argument buffers, asked for where the instance can take layer settings,
    // ahead of the consumer's own, which win (`argument_buffers.rs`); and, where the relay hears
    // its failures, that it keep a device after a command buffer fails (`relay.rs`).
    let asks = listed.lists(ash::ext::layer_settings::NAME);
    let chained_settings: Vec<LayerSetting> = asks
        .then(super::argument_buffers::setting)
        .into_iter()
        .chain((asks && relays).then(super::relay::setting))
        .chain(options.layer_settings.iter().cloned())
        .collect();
    if !chained_settings.is_empty() {
        extensions.push(ash::ext::layer_settings::NAME);
    }
    for extra in &options.instance_extensions {
        if !extensions.contains(&extra.as_c_str()) {
            extensions.push(extra);
        }
    }
    // The instance extensions the caller's device extensions require: one the instance would
    // not enable refuses the options before the instance is made, where a driver need not.
    instance_requirements_met(&options.device_extensions, &extensions)?;
    let extension_names: Vec<*const c_char> = extensions.iter().map(|n| n.as_ptr()).collect();
    let api_version = instance_version(options.instance_api_version, || {
        // SAFETY: a global command of a loaded entry.
        let version = unsafe { entry.try_enumerate_instance_version() }.map_err(vulkan)?;
        // A loader without `vkEnumerateInstanceVersion` is a Vulkan 1.0 loader.
        Ok(version.map_or(ApiVersion { major: 1, minor: 0 }, ApiVersion::from_packed))
    })?;
    let application = vk::ApplicationInfo::default()
        .engine_name(c"metal2vulkan")
        .api_version(api_version.packed());
    let sink = options.messages.clone().map(Box::new);
    let mut messenger_info = sink.as_ref().map(|sink| {
        use vk::DebugUtilsMessageSeverityFlagsEXT as S;
        use vk::DebugUtilsMessageTypeFlagsEXT as T;
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(S::WARNING | S::ERROR)
            .message_type(T::GENERAL | T::VALIDATION | T::PERFORMANCE)
            .pfn_user_callback(Some(deliver))
            .user_data(std::ptr::from_ref::<MessageSink>(sink).cast_mut().cast())
    });
    let mut chained = messenger_info;
    let (bools, strings) = setting_values(&chained_settings);
    let settings = layer_settings(&chained_settings, &bools, &strings);
    let mut settings_info = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
    let layer_names: Vec<*const c_char> = options.layers.iter().map(|n| n.as_ptr()).collect();
    let mut info = vk::InstanceCreateInfo::default()
        .application_info(&application)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_names)
        .flags(if portability {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        });
    // The chain is built once, before the attempts below: `push_next` writes the pushed
    // structure's own `pNext`, so pushing these onto a second create info would leave the one
    // pushed last pointing at the one pushed first, whose `pNext` the second push then sets to
    // itself -- a cycle the loader walks without end, allocating as it goes.
    if let Some(chained) = chained.as_mut() {
        info = info.push_next(chained);
    }
    if !settings.is_empty() {
        info = info.push_next(&mut settings_info);
    }
    // A layer the loader cannot provide is not something the device lacks: the layers are
    // diagnostics, so a caller that asked for a device gets one. The instance of the layers asked
    // for and, where the loader answers `VK_ERROR_LAYER_NOT_PRESENT` ([`drops_layers`]), of no
    // layer at all, whose answer is the caller's: two `vkCreateInstance` calls at most, the same
    // create info but for its layer list, which the second has none of, so the arm cannot run
    // twice. What was left out is announced as each device is made ([`layers_dropped`]).
    let mut layers: &[CString] = &options.layers;
    let mut dropped: Vec<CString> = Vec::new();
    let raw = loop {
        // SAFETY: `info` and everything it points at live for the call.
        match unsafe { new_instance(&entry, &info) } {
            Ok(raw) => break raw,
            Err(vk::Result::ERROR_INCOMPATIBLE_DRIVER) => return Err(Error::NoDevice),
            Err(error) if drops_layers(error, layers) => {
                dropped = options.layers.clone();
                layers = &[];
                info = info.enabled_layer_names(&[]);
            }
            Err(error) => return Err(vulkan(error)),
        }
    };
    let messenger = match (sink, messenger_info.as_mut()) {
        (Some(sink), Some(info)) => {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &raw);
            // SAFETY: the instance enabled `VK_EXT_debug_utils`; the sink lives in the messenger.
            match unsafe { loader.create_debug_utils_messenger(info, None) } {
                Ok(handle) => Some(Messenger {
                    loader,
                    handle,
                    sink,
                }),
                Err(error) => {
                    // SAFETY: nothing was made from the instance.
                    unsafe { release::instance(&raw) };
                    return Err(vulkan(error));
                }
            }
        }
        _ => None,
    };
    let relay = if relays {
        // SAFETY: the instance is live, from `entry`, and enabled `VK_EXT_debug_utils`.
        match unsafe { super::relay::Messenger::new(&entry, &raw) } {
            Ok(relay) => Some(relay),
            Err(error) => {
                if let Some(messenger) = &messenger {
                    messenger.destroy();
                }
                // SAFETY: nothing else was made from the instance.
                unsafe { release::instance(&raw) };
                return Err(vulkan(error));
            }
        }
    } else {
        None
    };
    let settings = sha2::Sha256::digest(format!("{chained_settings:?}").as_bytes()).into();
    Ok(Arc::new(InstanceShared {
        entry,
        raw,
        api_version,
        extensions: extensions.iter().map(|&name| name.to_owned()).collect(),
        owned: true,
        messenger,
        relay,
        settings,
        dropped_layers: dropped,
    }))
}

/// Whether `error`, which `vkCreateInstance` of `layers` answered, is answered by creating the
/// instance again with no layer at all: a missing layer's answer alone, and only where a layer was
/// asked for (`VK_ERROR_LAYER_NOT_PRESENT` with none asked for is the loader's own answer, with
/// nothing of ours to drop). Every other failure is the caller's.
pub(crate) fn drops_layers(error: vk::Result, layers: &[CString]) -> bool {
    error == vk::Result::ERROR_LAYER_NOT_PRESENT && !layers.is_empty()
}

/// The informational message a device of an instance created without the layers `dropped`, which
/// its caller asked for and the loader could not provide, sends as it is made: no degrade notice,
/// as nothing of Metal's answers differs (the layers are diagnostics). `None` where none was
/// dropped.
pub(crate) fn layers_dropped(dropped: &[CString], device: &str) -> Option<String> {
    if dropped.is_empty() {
        return None;
    }
    let names: Vec<String> = dropped
        .iter()
        .map(|name| name.to_string_lossy().into_owned())
        .collect();
    Some(format!(
        "metal2vulkan: the device on {device} was made without the instance layer(s) {}, which          the loader answered VK_ERROR_LAYER_NOT_PRESENT for: a layer it does not list, or one it          lists and cannot load (a layer manifest naming its library by a bare filename, which the          loader looks for on the system library search path). The device is the same in every          other way and its answers are Metal's; what those layers would have reported is missing          [{LAYERS_DROPPED}]"
    , names.join(", ")))
}

/// The id name of the informational message a device made without the layers its caller asked for
/// sends as it is made ([`layers_dropped`]).
pub(crate) const LAYERS_DROPPED: &str = "metal2vulkan-instance-layers-dropped";

/// The physical devices of `instance` that `options` admit, in the order
/// `Device::system_default` prefers them (by [`DeviceKind`], then enumeration order).
fn ranked(
    instance: &InstanceShared,
    options: &DeviceOptions,
) -> Result<Vec<vk::PhysicalDevice>, Error> {
    // SAFETY: the instance is live.
    let physical = unsafe { instance.raw.enumerate_physical_devices() }.map_err(vulkan)?;
    let mut ranked: Vec<(u8, vk::PhysicalDevice)> = physical
        .into_iter()
        .filter_map(|physical| {
            // SAFETY: `physical` was enumerated from the instance.
            let ty = unsafe { instance.raw.get_physical_device_properties(physical) }.device_type;
            let kind = DeviceKind::of(ty);
            (options.kinds.is_empty() || options.kinds.contains(&kind))
                .then_some((kind.preference(), physical))
        })
        .collect();
    ranked.sort_by_key(|&(preference, _)| preference);
    Ok(ranked.into_iter().map(|(_, physical)| physical).collect())
}

/// The registry ID (`Device::registry_id`) of each of `physical`, physical devices of `instance`,
/// in their order: from its `deviceUUID` (`VkPhysicalDeviceIDProperties`, core in Vulkan 1.1);
/// one below Vulkan 1.2, of which the runtime makes no device, is left out.
fn identified(
    instance: &InstanceShared,
    physical: Vec<vk::PhysicalDevice>,
) -> Vec<(u64, vk::PhysicalDevice)> {
    physical
        .into_iter()
        .filter_map(|physical| {
            // SAFETY: `physical` was enumerated from the instance.
            let core = unsafe { instance.raw.get_physical_device_properties(physical) };
            let api = metal2vulkan::vk::ApiVersion::from_packed(core.api_version);
            if api < metal2vulkan::vk::ApiVersion::V1_2 {
                return None;
            }
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            // SAFETY: as above; the instance is at Vulkan 1.2 or later, and the device too, so
            // the structure (core in 1.1) may be chained.
            unsafe { (instance.raw).get_physical_device_properties2(physical, &mut properties) };
            Some((super::properties::registry_id(&id.device_uuid), physical))
        })
        .collect()
}

/// The fewest `VkQueue`s a created device takes for the round robin logical queues share, where
/// the families taken have that many.
pub(crate) const LEAST_QUEUES: u32 = 4;

/// The most it takes for that round robin, where the families taken have queues to spare past
/// [`PRESENT_QUEUES`].
///
/// Two logical queues that share a `VkQueue` submit to one Vulkan queue, which runs its
/// submissions in order, so one's command buffers wait for the other's whole command buffer,
/// where Metal's queues wait for each other only inside one command (`queue.rs`, measured). A
/// device with queues to spare therefore gives each logical queue one of its own rather than
/// leaving them idle.
///
/// The ceiling is NVIDIA's graphics family of 16, the largest measured. What a further queue
/// costs is its `VkQueue`'s lock, timeline semaphore and four counters, and, on a driver that
/// takes cross-queue timeline waits, one wait in a batch's `vkQueueSubmit` the first time it sees
/// that queue's value move (about 0.5 us on an RTX 5080, 1.1 us on ms01's Iris Xe, 1.6 us on a
/// Radeon 780M). The one driver measured with queues to spare, NVIDIA's 615.71.09, stalls its
/// submissions on semaphore operations (`DriverFact::SemaphoreOperationsStallSubmits`) and so
/// takes no such wait at all: its batches begin with a barrier instead, whose cost does not grow
/// with the queues.
pub(crate) const ROUND_ROBIN_CEILING: u32 = 16;

/// The most `VkQueue`s a created device reserves for presenting queues
/// (`Device::new_command_queue_for_presenting`), outside the round robin: the taken families'
/// queues past the round robin's, where they have them, before any reserved for the caller.
const PRESENT_QUEUES: u32 = 2;

/// The queues a created device takes ([`queue_plan`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QueuePlan {
    /// The first family that does graphics and compute work (and so transfers): the device's
    /// primary family, its first queue's.
    pub(crate) primary: u32,
    /// The queues, each its family and index there, the primary family's first queue first.
    pub(crate) queues: Vec<(u32, u32)>,
    /// The queues reserved for presenting queues, after `queues` in the same order: at most
    /// [`PRESENT_QUEUES`], none where the families taken have no more.
    pub(crate) present: Vec<(u32, u32)>,
    /// The queues reserved for the caller (`DeviceOptions::reserved_queues`), after `present` in
    /// the same order: at most as many as asked, none where the families taken have no more.
    pub(crate) reserved: Vec<(u32, u32)>,
    /// The queue reserved for staging copies (`device/staging.rs`): the first queue of the first
    /// family of transfers alone whose copies take any texel (a granularity of one texel), which
    /// is none of the families above; `None` where there is none.
    pub(crate) staging: Option<(u32, u32)>,
}

/// The queues a device of `families` takes: where every family is equivalent to the first that
/// does graphics and compute work (the same flags, queue count, timestamp bits and transfer
/// granularity: MoltenVK's four families of one queue each, one Metal command queue per queue),
/// one queue of each family first, then the families' next queues in turn; otherwise that
/// family's queues alone. Queues of another family are
/// another Metal command queue on MoltenVK, whose command buffers the GPU runs beside the
/// first's; queues of one family there share one.
///
/// The round robin logical queues share takes those queues less [`PRESENT_QUEUES`] and less the
/// `reserved` the caller asks for (`DeviceOptions::reserved_queues`), at least [`LEAST_QUEUES`]
/// and at most [`ROUND_ROBIN_CEILING`], so a family with queues to spare lends them to the round
/// robin rather than leaving them idle (NVIDIA's 16 give 14) and no device takes fewer than it
/// took before. The next queues in that order, at most [`PRESENT_QUEUES`], are reserved for
/// presenting queues, and the next, at most `reserved`, for the caller: the round robin keeps its
/// first [`LEAST_QUEUES`] and presenting its queues before the caller has any (NVIDIA's 16 with
/// one reserved give the round robin 13). A queue of a family of transfers alone
/// is reserved for staging copies: a copy engine, which runs beside the graphics family's work
/// where another of its queues waits behind it; an asynchronous compute queue is none (its work
/// waited up to 1.05 s on a Radeon 780M and 1.19 s on an RTX 3090 behind a kernel that filled the
/// GPU). `None` where no family does graphics and compute work.
pub(crate) fn queue_plan(
    families: &[vk::QueueFamilyProperties],
    reserved: u32,
) -> Option<QueuePlan> {
    queue_plan_within(families, reserved, None)
}

/// [`queue_plan`] within `limits` (`DeviceOptions::queue_limits`): the round robin at most
/// `limits.round_robin` (at least one, its floor of [`LEAST_QUEUES`] no more than that) and at
/// most `limits.presenting` queues reserved for presenting, limits past the plan's taking the
/// plan's. `None` is the plan.
pub(crate) fn queue_plan_within(
    families: &[vk::QueueFamilyProperties],
    reserved: u32,
    limits: Option<QueueLimits>,
) -> Option<QueuePlan> {
    let (ceiling, presenting) = match limits {
        None => (ROUND_ROBIN_CEILING, PRESENT_QUEUES),
        Some(limits) => (
            limits.round_robin.clamp(1, ROUND_ROBIN_CEILING),
            limits.presenting.min(PRESENT_QUEUES),
        ),
    };
    let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
    let (primary, first) = families
        .iter()
        .enumerate()
        .find(|(_, family)| family.queue_count > 0 && family.queue_flags.contains(wanted))?;
    let equivalent = |family: &vk::QueueFamilyProperties| {
        family.queue_flags == first.queue_flags
            && family.queue_count == first.queue_count
            && family.timestamp_valid_bits == first.timestamp_valid_bits
            && family.min_image_transfer_granularity == first.min_image_transfer_granularity
    };
    let taken: Vec<u32> = if families.iter().all(equivalent) {
        (0..families.len() as u32).collect()
    } else {
        vec![primary as u32]
    };
    let mut order = (0..first.queue_count)
        .flat_map(|index| taken.iter().map(move |&family| (family, index)))
        .collect::<Vec<_>>();
    let past = (presenting as usize).saturating_add(reserved as usize);
    let round_robin = (order.len().saturating_sub(past))
        .clamp(LEAST_QUEUES.min(ceiling) as usize, ceiling as usize)
        .min(order.len());
    let mut rest = order.split_off(round_robin).into_iter();
    let present = rest.by_ref().take(presenting as usize).collect();
    let reserved = rest.take(reserved as usize).collect();
    // Transfers alone (sparse binding and protected memory aside: no graphics, compute, video or
    // optical flow work), and copies of any texel.
    let unit = vk::Extent3D {
        width: 1,
        height: 1,
        depth: 1,
    };
    let besides =
        vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING | vk::QueueFlags::PROTECTED;
    let staging = (families.iter().enumerate())
        .find(|&(index, family)| {
            !taken.contains(&(index as u32))
                && family.queue_count > 0
                && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && (family.queue_flags & !besides).is_empty()
                && family.min_image_transfer_granularity == unit
        })
        .map(|(index, _)| (index as u32, 0));
    // Where every family is equivalent, the primary is family 0, so either way the primary
    // family's first queue leads (internal submissions go there).
    Some(QueuePlan {
        primary: primary as u32,
        queues: order,
        present,
        reserved,
        staging,
    })
}

/// What the queue plan gives a device of `families` whose queue limits
/// (`DeviceOptions::queue_limits`) gave it `plan` and `extra_places`, for `reserved` reserved and
/// `extra` extra queues asked for: where the limits changed any of it, the plan's; `None` where
/// they changed nothing.
pub(crate) fn planned_queues(
    families: &[vk::QueueFamilyProperties],
    reserved: u32,
    extra: &[vk::QueueFlags],
    plan: &QueuePlan,
    extra_places: &[Option<(u32, u32)>],
) -> Option<PlannedQueues> {
    let mut planned = queue_plan(families, reserved)?;
    let planned_extra = extra_queue_places(families, &mut planned, extra);
    (planned != *plan || planned_extra != extra_places).then(|| PlannedQueues {
        round_robin: planned.queues.len(),
        presenting: planned.present.len(),
        reserved: planned.reserved,
        extra: planned_extra,
    })
}

/// What the queue plan gives a device whose queue limits changed its queues ([`planned_queues`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlannedQueues {
    /// The round robin's `VkQueue`s.
    pub(crate) round_robin: usize,
    /// The `VkQueue`s reserved for presenting queues.
    pub(crate) presenting: usize,
    /// The queues reserved for the caller, each its family and index there.
    pub(crate) reserved: Vec<(u32, u32)>,
    /// Each extra queue's place, `None` for one the device has none of.
    pub(crate) extra: Vec<Option<(u32, u32)>>,
}

/// Whether the `made`th queue counted from 0 of the round robin, or of the presenting queues,
/// that the limits give `limited` `VkQueue`s takes one another took where the plan's `planned`
/// give it one of its own (`DeviceOptions::queue_limits`).
pub(crate) fn shares_past_limit(made: u64, limited: usize, planned: usize) -> bool {
    made >= limited as u64 && made < planned as u64
}

/// The informational message a device whose queue limits changed what the queue plan gives says
/// where they did, `what` saying what (`DeviceOptions::queue_limits`).
pub(crate) fn queue_limits_reached(device: &str, what: &str) -> String {
    format!(
        "metal2vulkan: on {device}, {what}: the device's queue limits \
         (DeviceOptions::queue_limits) changed that from what its queue plan gives \
         [{QUEUE_LIMITS_REACHED}]"
    )
}

/// The id name of the informational message of [`queue_limits_reached`].
pub(crate) const QUEUE_LIMITS_REACHED: &str = "metal2vulkan-queue-limits-reached";

/// The queue of each of `asked` (`DeviceOptions::extra_queues`) on a device of `families` whose
/// own queues are `plan`'s, in `asked`'s order, `None` for one the device has none of: a queue the
/// plan leaves free, of the first family whose flags contain the asked flags and that has one,
/// at its lowest free index; else the last queue the plan reserved for presenting, then the last
/// of the round robin's but its first (internal submissions go there), then the staging queue, of
/// such a family, which `plan` then gives up; never a queue reserved for the caller. Flags that
/// are empty, or that no family's contain, take none.
pub(crate) fn extra_queue_places(
    families: &[vk::QueueFamilyProperties],
    plan: &mut QueuePlan,
    asked: &[vk::QueueFlags],
) -> Vec<Option<(u32, u32)>> {
    let mut extra: Vec<(u32, u32)> = Vec::new();
    let mut places = Vec::with_capacity(asked.len());
    for &flags in asked {
        let serves = |family: u32| {
            !flags.is_empty()
                && (families.get(family as usize)).is_some_and(|f| f.queue_flags.contains(flags))
        };
        let taken = |place: (u32, u32)| {
            plan.queues.contains(&place)
                || plan.present.contains(&place)
                || plan.reserved.contains(&place)
                || plan.staging == Some(place)
                || extra.contains(&place)
        };
        let free = (0..families.len() as u32)
            .filter(|&family| serves(family))
            .find_map(|family| {
                (0..families[family as usize].queue_count)
                    .map(|index| (family, index))
                    .find(|&place| !taken(place))
            });
        let place = free.or_else(|| {
            if let Some(at) = plan.present.iter().rposition(|&(f, _)| serves(f)) {
                return Some(plan.present.remove(at));
            }
            let round_robin = plan.queues.iter().skip(1).rposition(|&(f, _)| serves(f));
            if let Some(at) = round_robin {
                return Some(plan.queues.remove(at + 1));
            }
            plan.staging.filter(|&(f, _)| serves(f)).inspect(|_| {
                plan.staging = None;
            })
        });
        if let Some(place) = place {
            extra.push(place);
        }
        places.push(place);
    }
    places
}

/// The informational message a device made with extra queues it has none of for `absent`
/// (`DeviceOptions::extra_queues`) sends as it is made; `None` where it made every one.
pub(crate) fn extra_queues_absent(absent: &[vk::QueueFlags], device: &str) -> Option<String> {
    if absent.is_empty() {
        return None;
    }
    let flags: Vec<String> = (absent.iter())
        .map(|&flags| format!("{:?}", super::handles::QueueFlagsText(flags)))
        .collect();
    Some(format!(
        "metal2vulkan: the device on {device} has no queue for the extra queue(s) of {} the \
         options asked for (no queue family whose flags contain them, or none left past the \
         device's first queue); the device was made without them, and \
         DeviceHandles::extra_queues leaves them out [{EXTRA_QUEUES_ABSENT}]",
        flags.join("; ")
    ))
}

/// The id name of the informational message a device made without some of the extra queues its
/// options asked for sends as it is made ([`extra_queues_absent`]).
pub(crate) const EXTRA_QUEUES_ABSENT: &str = "metal2vulkan-extra-queues-absent";

/// The informational message a device that reserved only `reserved` of the `asked` queues
/// (`DeviceOptions::reserved_queues`) sends as it is made; `None` where it reserved every one.
pub(crate) fn reserved_queues_absent(asked: u32, reserved: usize, device: &str) -> Option<String> {
    if reserved >= asked as usize {
        return None;
    }
    Some(format!(
        "metal2vulkan: the device on {device} reserved {reserved} of the {asked} queue(s) of its \
         own queue families the options asked for: its round robin keeps its first \
         {LEAST_QUEUES} queues and presenting its {PRESENT_QUEUES} before the caller has any, \
         and its families have no more; the device was made with those it reserved, which \
         DeviceHandles::reserved_queues holds [{RESERVED_QUEUES_ABSENT}]"
    ))
}

/// The id name of the informational message a device made with fewer reserved queues than its
/// options asked for sends as it is made ([`reserved_queues_absent`]).
pub(crate) const RESERVED_QUEUES_ABSENT: &str = "metal2vulkan-reserved-queues-absent";

/// `description` with each of `withheld` no color attachment
/// (`DeviceOptions::withheld_attachment_formats`): its tilings' `COLOR_ATTACHMENT` and blend
/// features and its attachment sample counts left out.
pub(crate) fn withhold_attachments(
    description: &mut DeviceDescription,
    withheld: &BTreeSet<metal2vulkan::vk::Format>,
) {
    use metal2vulkan::vk::{FormatFeatureFlags2, SampleCountFlags};
    let attachment =
        FormatFeatureFlags2::COLOR_ATTACHMENT | FormatFeatureFlags2::COLOR_ATTACHMENT_BLEND;
    for format in withheld {
        if let Some(properties) = description.formats.formats.get_mut(format) {
            properties.optimal_tiling = properties.optimal_tiling.difference(attachment);
            properties.linear_tiling = properties.linear_tiling.difference(attachment);
            properties.color_attachment_sample_counts = SampleCountFlags::empty();
        }
    }
}

/// What a device of `description` the runtime creates under `options` enables: the
/// description's recommended enables for the options' policy and `VK_EXT_external_memory_host`
/// where listed (the runtime's own need: buffers over host memory without a copy), less
/// the withheld features and extensions. `DeviceDescription::with_enabled` checks it is a valid
/// set.
pub(crate) fn enable_set(description: &DeviceDescription, options: &DeviceOptions) -> EnableSet {
    let mut set = description.recommended_enables(&options.policy);
    set.features
        .retain(|feature| !options.withheld_features.contains(feature));
    let host_memory = VkExtension::ExtExternalMemoryHost;
    if description.extensions.contains(&host_memory) {
        set.extensions.insert(host_memory);
    }
    set.extensions
        .retain(|extension| !options.withheld_extensions.contains(extension));
    set
}

/// The extensions of the dma-buf route (`dma_buf.rs`) of a device listing `listed`:
/// `VK_KHR_external_memory_fd` and the `VK_EXT_external_memory_dma_buf` that requires it, where the
/// device lists both. The runtime's own need, outside the vocabulary, so outside the enable set,
/// the compile profile and its fingerprint (the shader cache's key).
pub(crate) fn dma_buf_names(listed: &BTreeSet<CString>) -> Option<[&'static CStr; 2]> {
    let names = [
        ash::khr::external_memory_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
    ];
    names
        .iter()
        .all(|name| listed.contains(*name))
        .then_some(names)
}

/// The extension of buffers whose memory another API imports (`exported_memory.rs`) of a device
/// listing `listed`: `VK_KHR_external_memory_fd`, where the device lists it. The runtime's own
/// need, outside the vocabulary as the dma-buf route's are ([`dma_buf_names`], which enable it
/// too).
pub(crate) fn memory_fd_name(listed: &BTreeSet<CString>) -> Option<&'static CStr> {
    let name = ash::khr::external_memory_fd::NAME;
    listed.contains(name).then_some(name)
}

/// The extension of the acquire of a dma-buf-exportable buffer's memory from the foreign agent
/// that wrote it (`exported_memory.rs`, `Buffer::external_write_done`) of a device listing
/// `listed`: `VK_EXT_queue_family_foreign`, where the device lists it and the dma-buf route's
/// ([`dma_buf_names`]). The runtime's own need, outside the vocabulary as the route's are; the
/// route's image extensions name it too ([`dma_buf_image_names`]), which enable it then.
pub(crate) fn foreign_family_name(listed: &BTreeSet<CString>) -> Option<&'static CStr> {
    let name = ash::ext::queue_family_foreign::NAME;
    (dma_buf_names(listed).is_some() && listed.contains(name)).then_some(name)
}

/// The extension of shared events whose timeline semaphore another API imports
/// (`exported_semaphore.rs`) of a device listing `listed`: `VK_KHR_external_semaphore_fd`, where
/// the device lists it. The runtime's own need, outside the vocabulary.
pub(crate) fn semaphore_fd_name(listed: &BTreeSet<CString>) -> Option<&'static CStr> {
    let name = ash::khr::external_semaphore_fd::NAME;
    listed.contains(name).then_some(name)
}

/// The extensions of the images of textures over the dma-buf route's pages (`dma_buf.rs`) of a
/// device listing `listed`: `VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign`,
/// where the device lists both and the route's two ([`dma_buf_names`]). Outside the vocabulary as
/// the route's are; what they require is core at the runtime's Vulkan 1.2 floor.
pub(crate) fn dma_buf_image_names(listed: &BTreeSet<CString>) -> Option<[&'static CStr; 2]> {
    let names = [
        ash::ext::image_drm_format_modifier::NAME,
        ash::ext::queue_family_foreign::NAME,
    ];
    (dma_buf_names(listed).is_some() && names.iter().all(|name| listed.contains(*name)))
        .then_some(names)
}

/// Whether a device of `description` the runtime creates under `options` enables
/// `VK_EXT_device_generated_commands`: where the description states it (the extension and its
/// `deviceGeneratedCommands`), `options` do not withhold it, and `set` enables
/// `bufferDeviceAddress`, which it requires. `set` then gains `VK_KHR_maintenance5` and its
/// feature, which it requires too, unless they are withheld or the device lacks them, when it is
/// not enabled.
pub(crate) fn generated_commands(
    description: &DeviceDescription,
    options: &DeviceOptions,
    set: &mut EnableSet,
) -> bool {
    let maintenance5 = (VkExtension::KhrMaintenance5, VkFeature::Maintenance5);
    let enabled = description.generated_commands.is_some()
        && !options.withhold_generated_commands
        && set.features.contains(&VkFeature::BufferDeviceAddress)
        && description.extensions.contains(&maintenance5.0)
        && description.features.contains(&maintenance5.1)
        && !options.withheld_extensions.contains(&maintenance5.0)
        && !options.withheld_features.contains(&maintenance5.1);
    if enabled {
        set.extensions.insert(maintenance5.0);
        set.features.insert(maintenance5.1);
    }
    enabled
}

/// Whether a device the runtime creates enables `hostImageCopy`, with the extensions `names`
/// (`enables::host_image_copy_names` of its version), the runtime's own need outside the
/// vocabulary: where the device lists `names`, and `supported`, asked only then, answers that it
/// has the feature. A `Shared` or `Managed` texture's CPU copies run on the host there
/// (`texture/host_copy.rs`), and a heap's colour texture takes host-transfer usage where its driver
/// serves it (`texture/heap.rs`).
pub(crate) fn host_image_copy(
    names: &[&CStr],
    listed: &BTreeSet<CString>,
    supported: impl FnOnce() -> bool,
) -> bool {
    names.iter().all(|&name| listed.contains(name)) && supported()
}

/// Whether `physical` has `hostImageCopy`.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, and the instance uses it at Vulkan 1.4 or later or
/// it lists `VK_EXT_host_image_copy`.
unsafe fn host_image_copy_supported(
    instance: &InstanceShared,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut host = vk::PhysicalDeviceHostImageCopyFeaturesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut host);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe {
        instance
            .raw
            .get_physical_device_features2(physical, &mut features)
    };
    host.host_image_copy == vk::TRUE
}

/// Whether `physical` has `zeroInitializeDeviceMemory`.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, made at Vulkan 1.1 or later, and lists
/// `VK_EXT_zero_initialize_device_memory`.
unsafe fn zero_initialize_device_memory_supported(
    instance: &InstanceShared,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut zero = crate::enables::ZeroInitializeDeviceMemoryFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut zero);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe {
        instance
            .raw
            .get_physical_device_features2(physical, &mut features)
    };
    zero.zero_initialize_device_memory == vk::TRUE
}

/// Whether a device that lists `listed` links its render pipelines from graphics pipeline
/// libraries: it lists `VK_EXT_graphics_pipeline_library` and `VK_KHR_pipeline_library`, which it
/// requires, and `supported` (asked only then) says it has the feature, links fast and takes each
/// library's interpolation decorations independently.
pub(crate) fn graphics_pipeline_library(
    listed: &BTreeSet<CString>,
    supported: impl FnOnce() -> bool,
) -> bool {
    crate::enables::GRAPHICS_PIPELINE_LIBRARY
        .iter()
        .all(|name| listed.contains(*name))
        && supported()
}

/// Whether `physical` has `graphicsPipelineLibrary`, and links libraries fast
/// (`graphicsPipelineLibraryFastLinking`) with each library's interpolation decorations its own
/// (`graphicsPipelineLibraryIndependentInterpolationDecoration`): a library of a fragment
/// function is then linked with any pre-rasterization library, whatever decorations its outputs
/// carry.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, made at Vulkan 1.1 or later, and lists
/// `VK_EXT_graphics_pipeline_library`.
pub(super) unsafe fn graphics_pipeline_library_supported(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut library = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut library);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    let mut linking = vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut linking);
    // SAFETY: as above.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    library.graphics_pipeline_library == vk::TRUE
        && linking.graphics_pipeline_library_fast_linking == vk::TRUE
        && linking.graphics_pipeline_library_independent_interpolation_decoration == vk::TRUE
}

/// Whether `physical` has `samplerYcbcrConversion` (Vulkan 1.1's).
///
/// # Safety
///
/// `physical` was enumerated from `instance`, made at Vulkan 1.2 or later.
unsafe fn ycbcr_conversion_supported(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut v11);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    v11.sampler_ycbcr_conversion == vk::TRUE
}

/// Whether `physical` has `VK_EXT_ycbcr_2plane_444_formats`' `ycbcr2plane444Formats`.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, made at Vulkan 1.2 or later, and lists the
/// extension.
unsafe fn ycbcr_444_supported(instance: &ash::Instance, physical: vk::PhysicalDevice) -> bool {
    let mut formats = vk::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut formats);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    formats.ycbcr2plane444_formats == vk::TRUE
}

/// Whether a device that lists `listed` enables `VK_EXT_pageable_device_local_memory` and its
/// `pageableDeviceLocalMemory`: it lists the extension and `VK_EXT_memory_priority`, which it
/// requires, and has the feature (`supported`, asked only then).
pub(crate) fn pageable_device_local_memory(
    listed: &BTreeSet<CString>,
    supported: impl FnOnce() -> bool,
) -> bool {
    (crate::enables::PAGEABLE_DEVICE_LOCAL_MEMORY.iter()).all(|&name| listed.contains(name))
        && supported()
}

/// Whether `physical` has `pageableDeviceLocalMemory`.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, made at Vulkan 1.1 or later, and lists
/// `VK_EXT_pageable_device_local_memory`.
unsafe fn pageable_device_local_memory_supported(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut pageable = vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut pageable);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    pageable.pageable_device_local_memory == vk::TRUE
}

/// Whether a device of version `api` (its own, capped by the instance's) that lists `listed`
/// enables `VK_NV_linear_color_attachment` and its `linearColorAttachment`: it lists the
/// extension and has what it requires, `VK_KHR_format_feature_flags2` or Vulkan 1.3, where that
/// is core, and has the feature (`supported`, asked only then). Below Vulkan 1.3 too, where it
/// lists `VK_KHR_format_feature_flags2`, which it enables with it
/// (`enables::linear_color_attachment_names`); the runtime's instance is at Vulkan 1.2 unless the
/// consumer asks another, so NVIDIA's devices never enabled it (the consumer's probe).
pub(crate) fn linear_color_attachment(
    api: ApiVersion,
    listed: &BTreeSet<CString>,
    supported: impl FnOnce() -> bool,
) -> bool {
    (api >= ApiVersion::V1_3 || listed.contains(c"VK_KHR_format_feature_flags2"))
        && listed.contains(ash::nv::linear_color_attachment::NAME)
        && supported()
}

/// Whether `physical` has `linearColorAttachment`.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, made at Vulkan 1.1 or later, and lists
/// `VK_NV_linear_color_attachment`.
unsafe fn linear_color_attachment_supported(
    instance: &InstanceShared,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut linear = vk::PhysicalDeviceLinearColorAttachmentFeaturesNV::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut linear);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe {
        instance
            .raw
            .get_physical_device_features2(physical, &mut features)
    };
    linear.linear_color_attachment == vk::TRUE
}

/// Why a device whose set is `set` and that lists `listed` keeps its driver's internal pipeline
/// cache, `None` where the runtime makes it with that cache off
/// ([`Enables::turn_driver_pipeline_cache_off`]): it lists `VK_KHR_pipeline_binary`, `set` enables
/// `VK_KHR_maintenance5`, which the extension requires below Vulkan 1.4 (the runtime's instance is
/// at Vulkan 1.2 unless the consumer asks another), and the driver allows it (`control`, asked
/// only then: `pipelineBinaryInternalCacheControl`).
pub(crate) fn driver_cache_kept(
    api: ApiVersion,
    set: &EnableSet,
    listed: &BTreeSet<CString>,
    control: impl FnOnce() -> bool,
) -> Option<&'static str> {
    let maintenance5 = api >= ApiVersion { major: 1, minor: 4 }
        || set.extensions.contains(&VkExtension::KhrMaintenance5);
    if !listed.contains(crate::enables::PIPELINE_BINARY) {
        Some("its driver lists no `VK_KHR_pipeline_binary`, whose device structure turns it off")
    } else if !maintenance5 {
        Some(
            "the device enables no `VK_KHR_maintenance5`, which `VK_KHR_pipeline_binary` \
             requires",
        )
    } else if !control() {
        Some("its driver's `pipelineBinaryInternalCacheControl` is false: it keeps the cache on")
    } else {
        None
    }
}

/// Whether `physical`'s driver lets a device be made with its internal pipeline cache off
/// (`pipelineBinaryInternalCacheControl`).
///
/// # Safety
///
/// `physical` was enumerated from `instance`, made at Vulkan 1.1 or later, and lists
/// `VK_KHR_pipeline_binary`.
unsafe fn internal_cache_control(instance: &InstanceShared, physical: vk::PhysicalDevice) -> bool {
    let mut binary = crate::enables::PipelineBinaryProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut binary);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe {
        instance
            .raw
            .get_physical_device_properties2(physical, &mut properties)
    };
    binary.internal_cache_control == vk::TRUE
}

/// Whether `physical` has `pipelineBinaries`.
///
/// # Safety
///
/// As [`internal_cache_control`].
unsafe fn pipeline_binaries_supported(
    instance: &InstanceShared,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut binary = crate::enables::PipelineBinaryFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut binary);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe {
        instance
            .raw
            .get_physical_device_features2(physical, &mut features)
    };
    binary.pipeline_binaries == vk::TRUE
}

/// Whether `physical`'s portability subset has `imageViewFormatReinterpretation`.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, made at Vulkan 1.1 or later, and lists
/// `VK_KHR_portability_subset`.
unsafe fn reinterpretation_supported(
    instance: &InstanceShared,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut subset = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut subset);
    // SAFETY: the caller's contract; the chain lives for the call.
    unsafe {
        instance
            .raw
            .get_physical_device_features2(physical, &mut features)
    };
    subset.image_view_format_reinterpretation == vk::TRUE
}

/// The `maxTimelineSemaphoreValueDifference` Vulkan requires of every device (its required
/// limits): `2^31 - 1`.
const MIN_TIMELINE_VALUE_DIFFERENCE: u64 = (1 << 31) - 1;

/// Refuses a device whose timeline semaphores cannot step as Vulkan requires: one whose
/// `maxTimelineSemaphoreValueDifference` (`available`) is below Vulkan's minimum is not conformant
/// (`Unsupported`, `Feature::TimelineSemaphores` naming the value). The runtime's command buffers
/// and events are timeline semaphores, and its first signal, priming the device (`prime.rs`), is
/// already a step.
pub(crate) fn timeline_difference(available: u64) -> Result<(), Error> {
    if available >= MIN_TIMELINE_VALUE_DIFFERENCE {
        return Ok(());
    }
    Err(Error::Unsupported(Unsupported::not_representable(
        Feature::TimelineSemaphores {
            max_value_difference: available,
        },
    )))
}

/// [`timeline_difference`] of `physical`'s `maxTimelineSemaphoreValueDifference`.
///
/// # Safety
/// `physical` was enumerated from `instance`, made at Vulkan 1.2 or later.
pub(super) unsafe fn timeline_conformant(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Result<(), Error> {
    let mut timeline = vk::PhysicalDeviceTimelineSemaphoreProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut timeline);
    // SAFETY: the caller's; the structure is core in Vulkan 1.2.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    timeline_difference(timeline.max_timeline_semaphore_value_difference)
}

/// Refuse a MoltenVK device of `description` running without Metal argument buffers
/// (`argument_buffers.rs`).
pub(super) fn argument_buffers_kept(description: &DeviceDescription) -> Result<(), Error> {
    use super::argument_buffers::{check, Facts};
    check(Facts::of(description).argument_buffers())
}

/// A device on `physical`, made from `instance`.
/// A device of `physical` made with `options`, a generation handle of it: the first generation of
/// a new line, or with `recovering`, the next generation of that line (`line.rs`), made the same
/// way.
pub(super) fn create_on(
    instance: &Arc<InstanceShared>,
    physical: vk::PhysicalDevice,
    options: &DeviceOptions,
    cache: &CacheSetup,
    recovering: Option<&Arc<Line>>,
) -> Result<Device, Error> {
    // SAFETY: `physical` was enumerated from the instance, created at `api_version`.
    let mut description =
        unsafe { DeviceDescription::query(&instance.raw, physical, instance.api_version) }
            .map_err(device_error)?;
    withhold_attachments(&mut description, &options.withheld_attachment_formats);
    // SAFETY: as above.
    unsafe { timeline_conformant(&instance.raw, physical) }?;
    argument_buffers_kept(&description)?;
    // SAFETY: `physical` was enumerated from the instance.
    let families = unsafe {
        instance
            .raw
            .get_physical_device_queue_family_properties(physical)
    };
    let mut plan = queue_plan_within(&families, options.reserved_queues, options.queue_limits)
        .ok_or(Error::NoDevice)?;
    let extra_places = extra_queue_places(&families, &mut plan, &options.extra_queues);
    let planned = options.queue_limits.and_then(|_| {
        planned_queues(
            &families,
            options.reserved_queues,
            &options.extra_queues,
            &plan,
            &extra_places,
        )
    });
    let reserved_places = plan.reserved.clone();
    let queue_family = plan.primary;
    let api = description.api_version;
    let mut set = enable_set(&description, options);
    let generated_commands = generated_commands(&description, options, &mut set);
    let mut enabled = description.with_enabled(&set).map_err(device_error)?;
    if !generated_commands {
        enabled.generated_commands = None;
    }
    let compile =
        metal2vulkan::device::Device::new(enabled.clone(), options.policy).map_err(device_error)?;
    // SAFETY: as above.
    let listed: BTreeSet<CString> =
        unsafe { instance.raw.enumerate_device_extension_properties(physical) }
            .map_err(vulkan)?
            .iter()
            .filter_map(|extension| extension.extension_name_as_c_str().ok())
            .map(CStr::to_owned)
            .collect();
    // The runtime's own need, outside the vocabulary, where listed: GPU times on the host's
    // clock (`gpu_time.rs`).
    let calibration = gpu_time::Extension::among(&listed);
    // The runtime's own need, outside the vocabulary, wherever listed (Not only on a driver that
    // compiles every library of saved cache data at load): which stages a variant's cache held
    // (`shader_cache/variant.rs`), and whether the driver's cache held a pipeline
    // (`shader_cache/mod.rs`). Listed, not core 1.3: the runtime's instance is Vulkan 1.2 by
    // default.
    let feedback_name = ash::ext::pipeline_creation_feedback::NAME;
    let creation_feedback = listed.contains(feedback_name);
    let feedback_asked = options
        .device_extensions
        .iter()
        .any(|name| name.as_c_str() == feedback_name);
    // The runtime's own needs, outside the vocabulary, where listed (`dma_buf.rs`).
    let dma_buf = dma_buf_names(&listed);
    let dma_buf_images = dma_buf_image_names(&listed);
    // Wherever listed (`exported_memory.rs`); the dma-buf route's names enable it already.
    let memory_fd = memory_fd_name(&listed);
    // Beside the route, wherever listed (`exported_memory.rs`); the route's image names enable it
    // already where they are enabled.
    let foreign_family = foreign_family_name(&listed);
    // Wherever listed (`exported_semaphore.rs`).
    let semaphore_fd = semaphore_fd_name(&listed);
    let extra: Vec<CString> = options
        .device_extensions
        .iter()
        .cloned()
        .chain(calibration.map(|extension| extension.name().to_owned()))
        .chain((creation_feedback && !feedback_asked).then(|| feedback_name.to_owned()))
        .chain(dma_buf.into_iter().flatten().map(CStr::to_owned))
        .chain((memory_fd.filter(|_| dma_buf.is_none())).map(CStr::to_owned))
        .chain(semaphore_fd.map(CStr::to_owned))
        .chain(dma_buf_images.into_iter().flatten().map(CStr::to_owned))
        .chain((foreign_family.filter(|_| dma_buf_images.is_none())).map(CStr::to_owned))
        .collect();
    let mut enables = Enables::of(&set, api, &listed, &extra)?;
    // The features of the caller's extensions whose structures the runtime knows, as the device
    // has them: one it lacks refuses the device.
    // An extension the device does not list is not asked about: the create call refuses it.
    let caller: Vec<_> = listed_structures(&options.device_extensions)
        .into_iter()
        .filter(|(name, _)| listed.contains(*name))
        .collect();
    // The caller's optional extensions the device may enable: those it lists whose instance
    // requirements the instance enables, enabled where it has their features.
    let instance_names: Vec<&CStr> = instance.extensions.iter().map(CString::as_c_str).collect();
    let optional: Vec<CString> = (options.optional_device_extensions.iter())
        .filter(|name| listed.contains(*name) && !options.device_extensions.contains(name))
        .filter(|name| {
            instance_requirements_met(std::slice::from_ref(name), &instance_names).is_ok()
        })
        .cloned()
        .collect();
    let asked: Vec<_> = caller
        .iter()
        .copied()
        .chain(listed_structures(&optional))
        .collect();
    // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2, and lists each
    // extension `asked` names.
    let reported =
        (!asked.is_empty()).then(|| unsafe { listed_reported(&instance.raw, physical, &asked) });
    let has = |feature: &'static str| reported.as_ref().is_some_and(|reported| reported(feature));
    enables.enable_listed(&caller, has)?;
    enables.enable_where_featured(&optional, has);
    // The runtime's own need, outside the vocabulary: Metal's views of another component layout
    // (`view_plan.rs`), which a portability subset allows only with
    // `imageViewFormatReinterpretation`, enabled where it has it, unless the options withhold
    // such views (then shadowed, `texture/view_shadow.rs`).
    let portability = set.extensions.contains(&VkExtension::KhrPortabilitySubset);
    // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2, and lists the
    // subset's extension.
    let view_reinterpretation = !options.withhold_view_reinterpretation
        && (!portability || unsafe { reinterpretation_supported(instance, physical) });
    if portability && view_reinterpretation {
        enables.enable_view_reinterpretation();
    }
    if generated_commands {
        enables.enable_generated_commands();
    }
    let host_names = host_image_copy_names(api);
    let host_image_copy = host_image_copy(host_names, &listed, || {
        // SAFETY: `physical` was enumerated from the instance, which uses it at `api`; asked
        // only where `api` makes the feature core or the device lists its extension.
        unsafe { host_image_copy_supported(instance, physical) }
    });
    if host_image_copy {
        enables.enable_host_image_copy(host_names);
    }
    // The runtime's own need, outside the vocabulary, where listed with its feature: a `Shared`
    // or `Managed` texture's image that copies on the host starts in memory the driver zeroed
    // (`texture/host_copy.rs`).
    let zero_initialize_device_memory = host_image_copy
        && listed.contains(crate::enables::ZERO_INITIALIZE_DEVICE_MEMORY)
        // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2, and lists the
        // extension.
        && unsafe { zero_initialize_device_memory_supported(instance, physical) };
    if zero_initialize_device_memory {
        enables.enable_zero_initialize_device_memory();
    }
    // The runtime's own need, outside the vocabulary, where listed with its feature and what it
    // requires: textures over buffers' memory render into their linear images in place
    // (`linear_attachment.rs`).
    let linear_color_attachment = linear_color_attachment(api, &listed, || {
        // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2, and lists the
        // extension.
        unsafe { linear_color_attachment_supported(instance, physical) }
    });
    if linear_color_attachment {
        enables.enable_linear_color_attachment(crate::enables::linear_color_attachment_names(api));
    }
    // The runtime's own need, outside the vocabulary, where the device lists the extensions and
    // links libraries fast: its render pipelines are linked from libraries of their parts
    // (`pipeline/libraries.rs`).
    let graphics_pipeline_library = graphics_pipeline_library(&listed, || {
        // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2, and lists the
        // extensions.
        unsafe { graphics_pipeline_library_supported(&instance.raw, physical) }
    });
    if graphics_pipeline_library {
        enables.enable_graphics_pipeline_library();
    }
    // The runtime's own need, outside the vocabulary, where the device has it: two-plane YCbCr
    // textures convert through the device's own conversion (`ycbcr_device.rs`).
    // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2.
    let ycbcr_conversion = unsafe { ycbcr_conversion_supported(&instance.raw, physical) };
    if ycbcr_conversion {
        enables.enable_ycbcr_conversion();
    }
    // And its two-plane 4:4:4 formats: core from Vulkan 1.3, else where the device lists
    // `VK_EXT_ycbcr_2plane_444_formats` with its feature.
    let ycbcr_444 = ycbcr_conversion
        && (api >= ApiVersion::V1_3
            || listed.contains(ash::ext::ycbcr_2plane_444_formats::NAME)
                // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2, and
                // lists the extension.
                && unsafe { ycbcr_444_supported(&instance.raw, physical) });
    if ycbcr_444 && api < ApiVersion::V1_3 {
        enables.enable_ycbcr_2plane_444();
    }
    // The runtime's own need, outside the vocabulary, where listed with its feature: the driver
    // may move device-local memory to host-local memory to make room rather than fail an
    // allocation, as Metal pages a discrete GPU's resources (`memory.rs`'s
    // `overflow_candidates`, which places them past a full device-local heap where it fails). On
    // an RTX 5080 Laptop GPU (Linux) with it, 72 `Private` buffers of 256 MiB, past its 17.09 GB
    // device-local heap, were made in device-local memory; without it the 60th overflowed.
    let pageable = pageable_device_local_memory(&listed, || {
        // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2, and lists the
        // extension.
        unsafe { pageable_device_local_memory_supported(&instance.raw, physical) }
    });
    if pageable {
        enables.enable_pageable_device_local_memory();
    }
    // The runtime's own need, outside the vocabulary: the device's pipelines are cached in the
    // shader cache's records alone, the driver's internal cache off where it allows; else the
    // device says once that the driver keeps it.
    let cache_kept = driver_cache_kept(api, &set, &listed, || {
        // SAFETY: `physical` was enumerated from the instance, made at Vulkan 1.2, and lists the
        // extension.
        unsafe { internal_cache_control(instance, physical) }
    });
    if cache_kept.is_none() {
        // SAFETY: as above.
        let binaries = unsafe { pipeline_binaries_supported(instance, physical) };
        enables.turn_driver_pipeline_cache_off(binaries);
    }
    // The runtime's own need, outside the vocabulary: sparse heaps (`sparse/device.rs`), where
    // the queue family binds sparse memory (every family taken is equivalent to the primary).
    // SAFETY: `physical` was enumerated from the instance.
    let available = unsafe { instance.raw.get_physical_device_features(physical) };
    let sparse_features = families
        .get(queue_family as usize)
        .filter(|family| family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING))
        .map_or_else(vk::PhysicalDeviceFeatures::default, |_| {
            sparse::device::wanted(&available)
        });
    enables.enable_sparse(&sparse_features);
    // One create info per family taken, of as many queues as the plan takes of it.
    let mut taken: Vec<(u32, usize)> = Vec::new();
    let extra: Vec<(u32, u32)> = extra_places.iter().flatten().copied().collect();
    let all = (plan.queues.iter().chain(&plan.present))
        .chain(&plan.reserved)
        .chain(&plan.staging);
    for &(family, _) in all.chain(&extra) {
        match taken.iter_mut().find(|(at, _)| *at == family) {
            Some((_, count)) => *count += 1,
            None => taken.push((family, 1)),
        }
    }
    // One priority per queue of the family the plan and the extra queues take most of, so the
    // slice each create info names is always within it: the round robin alone takes up to
    // `ROUND_ROBIN_CEILING` of one family, presenting `PRESENT_QUEUES` more, and the reserved and
    // extra queues at most the family's own count.
    let priorities = vec![1.0f32; taken.iter().map(|&(_, count)| count).max().unwrap_or(0)];
    let queues: Vec<vk::DeviceQueueCreateInfo<'_>> = taken
        .iter()
        .map(|&(family, count)| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities[..count])
        })
        .collect();
    // At the user-initiated class: MoltenVK compiles its command shaders as it makes the device.
    let raw = enables
        .with_create_info(&queues, |info| {
            let asked = super::compiles::Sent::new(info);
            let instance = &instance.raw;
            super::compiles::at_user_initiated(move || {
                // SAFETY: the create info and its chain live for the call, and the asking thread
                // waits for it; `physical` is the instance's.
                unsafe { instance.create_device(physical, asked.into_inner(), None) }
            })
        })
        .map_err(vulkan)?;
    // SAFETY: the device was just created: its word is the loader's.
    let loader_word = unsafe { DeviceWord::of_device(raw.handle()) };
    let primary = families.get(queue_family as usize);
    let memory_reads = MemoryReads::of(
        primary.map_or(vk::QueueFlags::empty(), |family| family.queue_flags),
        enables.names(),
    );
    let reserved = plan.reserved.len();
    let parts = Parts {
        instance: Arc::clone(instance),
        physical,
        raw,
        loader_word,
        queue_family,
        present_queues: plan.present.len(),
        planned: (planned.as_ref()).map(|planned| (planned.round_robin, planned.presenting)),
        queues: plan.queues.into_iter().chain(plan.present).collect(),
        staging: plan.staging,
        reserved_queues: plan.reserved,
        extra_queues: (options.extra_queues.iter().zip(&extra_places))
            .filter_map(|(&flags, place)| place.map(|(family, index)| (flags, family, index)))
            .collect(),
        compile,
        description: enabled,
        enabled: set,
        extensions: enables.names().map(CStr::to_owned).collect(),
        calibration,
        gpu_times_from_host: options.gpu_times_from_host,
        mac_class: options.mac_class,
        announce_each: options.announce_each.clone(),
        lowering_refuses: options.lowering_refuses.clone(),
        record_executions: options.record_executions,
        pipeline_records: options.pipeline_records.clone(),
        fail_in_place_check: options.fail_in_place_check,
        assume_submissions_merge: options.assume_submissions_merge,
        pending_audit: options.pending_audit,
        view_reinterpretation,
        generated_commands,
        host_image_copy,
        zero_initialize_device_memory,
        linear_color_attachment,
        graphics_pipeline_library,
        ycbcr_conversion,
        ycbcr_444,
        creation_feedback: creation_feedback || feedback_asked,
        dma_buf: dma_buf.is_some(),
        dma_buf_images: dma_buf_images.is_some(),
        memory_fd: memory_fd.is_some(),
        foreign_family: foreign_family.is_some(),
        semaphore_fd: semaphore_fd.is_some(),
        sparse_features,
        memory_reads,
        owned: true,
        cache: cache.clone(),
        line: recovering.cloned(),
        recipe: match recovering {
            Some(_) => None,
            None => Some(Recipe {
                instance: Arc::clone(instance),
                physical,
                options: options.clone(),
                cache: cache.clone(),
            }),
        },
    };
    // SAFETY: the device was made from these parts just now.
    let device = unsafe { Device::assemble(parts) }?;
    if let Some(text) = layers_dropped(&instance.dropped_layers, device.name()) {
        device.inform(LAYERS_DROPPED, text);
    }
    let absent: Vec<vk::QueueFlags> = (options.extra_queues.iter().zip(&extra_places))
        .filter(|(_, place)| place.is_none())
        .map(|(&flags, _)| flags)
        .collect();
    if let Some(text) = extra_queues_absent(&absent, device.name()) {
        device.inform(EXTRA_QUEUES_ABSENT, text);
    }
    if let Some(text) = reserved_queues_absent(options.reserved_queues, reserved, device.name()) {
        device.inform(RESERVED_QUEUES_ABSENT, text);
    }
    if let Some(planned) = &planned {
        if planned.reserved != reserved_places {
            let what = format!(
                "the queues reserved for the caller are {reserved_places:?} (family, index), \
                 where the plan reserves {:?}",
                planned.reserved
            );
            device.inform(
                QUEUE_LIMITS_REACHED,
                queue_limits_reached(device.name(), &what),
            );
        }
        if planned.extra != extra_places {
            let what = format!(
                "the extra queues are at {extra_places:?} (family, index), where the plan places \
                 them at {:?}",
                planned.extra
            );
            device.inform(
                QUEUE_LIMITS_REACHED,
                queue_limits_reached(device.name(), &what),
            );
        }
    }
    if let Some(why) = cache_kept {
        let object = format!("the device, on {}", device.name());
        if kept_cache_is_metals(device.properties().description.driver_id) {
            device.inform(
                METAL_CACHE_KEPT,
                format!(
                    "metal2vulkan: {object}: {why}; its internal pipeline cache is Metal's own \
                     shader cache, which every Metal program on the host keeps: nothing differs \
                     from Metal [{METAL_CACHE_KEPT}]"
                ),
            );
        } else {
            device.degrade(Degradation::DriverPipelineCacheKept, || {
                (object, why.to_owned())
            });
        }
    }
    Ok(device)
}

/// The id name of the informational message a device whose driver keeps its internal pipeline
/// cache, that cache being Metal's own ([`kept_cache_is_metals`]), sends as it is made.
pub(crate) const METAL_CACHE_KEPT: &str = "metal2vulkan-driver-pipeline-cache-kept";

/// Whether the internal pipeline cache a driver of identity `driver` keeps is Metal's own shader
/// cache: MoltenVK's, whose pipelines are Metal pipeline states that Metal caches on the host
/// for every Metal program alike, so keeping it is Metal's behaviour exactly. Every other
/// driver's is a second cache beside the shader cache's records, which costs a warm making
/// and announces `driver-pipeline-cache-kept`.
pub(crate) fn kept_cache_is_metals(driver: Option<metal2vulkan::vk::DriverId>) -> bool {
    driver == Some(metal2vulkan::vk::DriverId::MOLTENVK)
}

/// The devices `system_default_with` and `all_with` made, by the options they were made with
/// (the module docs): each group's instance and its devices by physical device, held weakly. A
/// group whose instance is gone has no device left (each device holds its instance), and is
/// dropped at the next call.
static MADE: Mutex<Vec<Made>> = Mutex::new(Vec::new());

/// One group of `MADE`: the devices made with equal options.
struct Made {
    /// The options, less the message sink, which the instance holds (`InstanceShared::sink`),
    /// and the pipeline records' hook, which `records` names: the group keeps neither alive past
    /// its devices.
    options: DeviceOptions,
    /// The options' pipeline records' hook, which each of the group's devices holds.
    records: Option<std::sync::Weak<dyn Fn(crate::PipelineRecord) + Send + Sync>>,
    instance: Weak<InstanceShared>,
    devices: Vec<(vk::PhysicalDevice, Weak<Line>)>,
}

impl Made {
    /// The group's instance, while a device of it lives, where it was made with `options`.
    fn instance_for(&self, options: &DeviceOptions) -> Option<Arc<InstanceShared>> {
        let instance = self.instance.upgrade()?;
        let sinks = match (&options.messages, instance.sink()) {
            (None, None) => true,
            (Some(asked), Some(made)) => asked.same(made),
            _ => false,
        };
        let records = match (&options.pipeline_records, &self.records) {
            (None, None) => true,
            (Some(asked), Some(made)) => std::sync::Weak::ptr_eq(&Arc::downgrade(asked), made),
            _ => false,
        };
        (sinks && records && same_options(&self.options, options)).then_some(instance)
    }

    /// The live device of `physical`, if the group has one: a root handle of its line.
    fn device(&self, physical: vk::PhysicalDevice) -> Option<Device> {
        self.devices
            .iter()
            .find(|(of, _)| *of == physical)
            .and_then(|(_, line)| line.upgrade())
            .and_then(holds::revive_root)
    }
}

/// The live devices of the group of `options` (equal options and the same message sink), for the
/// tests: none where no such group lives.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn live_devices_with(options: &DeviceOptions) -> usize {
    let made = lock(&MADE);
    let of = made
        .iter()
        .find(|group| group.instance_for(options).is_some());
    of.map_or(0, |group| {
        let devices = group.devices.iter();
        devices.filter(|(_, line)| line.strong_count() > 0).count()
    })
}

/// Whether `a` and `b` are equal options, but for their message sinks and pipeline records'
/// hooks (compared by `Made`).
fn same_options(a: &DeviceOptions, b: &DeviceOptions) -> bool {
    // Every field is named, so a field added later is compared or this fails to compile.
    let DeviceOptions {
        policy,
        layers,
        layer_settings,
        instance_extensions,
        device_extensions,
        optional_device_extensions,
        withheld_features,
        withheld_extensions,
        messages: _,
        loader,
        kinds,
        shader_cache,
        withhold_generated_commands,
        withhold_view_reinterpretation,
        instance_api_version,
        gpu_times_from_host,
        mac_class,
        announce_each,
        lowering_refuses,
        record_executions,
        fail_in_place_check,
        assume_submissions_merge,
        pending_audit,
        withheld_attachment_formats,
        pipeline_records: _,
        extra_queues,
        reserved_queues,
        queue_limits,
    } = a;
    *policy == b.policy
        && *layers == b.layers
        && *layer_settings == b.layer_settings
        && *instance_extensions == b.instance_extensions
        && *device_extensions == b.device_extensions
        && *optional_device_extensions == b.optional_device_extensions
        && *withheld_features == b.withheld_features
        && *withheld_extensions == b.withheld_extensions
        && *loader == b.loader
        && *kinds == b.kinds
        && *shader_cache == b.shader_cache
        && *withhold_generated_commands == b.withhold_generated_commands
        && *withhold_view_reinterpretation == b.withhold_view_reinterpretation
        && *instance_api_version == b.instance_api_version
        && *gpu_times_from_host == b.gpu_times_from_host
        && *mac_class == b.mac_class
        && *announce_each == b.announce_each
        && *lowering_refuses == b.lowering_refuses
        && *record_executions == b.record_executions
        && *fail_in_place_check == b.fail_in_place_check
        && *assume_submissions_merge == b.assume_submissions_merge
        && *pending_audit == b.pending_audit
        && *withheld_attachment_formats == b.withheld_attachment_formats
        && *extra_queues == b.extra_queues
        && *reserved_queues == b.reserved_queues
        && *queue_limits == b.queue_limits
}

/// The group of `made` for `options`: the live one, or a new one on a new instance. Its index.
fn group(
    made: &mut Vec<Made>,
    options: &DeviceOptions,
) -> Result<(usize, Arc<InstanceShared>), Error> {
    made.retain(|group| group.instance.strong_count() > 0);
    if let Some(found) = made
        .iter()
        .enumerate()
        .find_map(|(at, group)| group.instance_for(options).map(|instance| (at, instance)))
    {
        return Ok(found);
    }
    let instance = create_instance(options)?;
    made.push(Made {
        options: DeviceOptions {
            messages: None,
            pipeline_records: None,
            ..options.clone()
        },
        records: options.pipeline_records.as_ref().map(Arc::downgrade),
        instance: Arc::downgrade(&instance),
        devices: Vec::new(),
    });
    Ok((made.len() - 1, instance))
}

/// The device of `physical` in `group`: its live one, or one made now and kept there.
fn device_of(
    group: &mut Made,
    instance: &Arc<InstanceShared>,
    physical: vk::PhysicalDevice,
    options: &DeviceOptions,
    cache: &CacheSetup,
) -> Result<Device, Error> {
    if let Some(device) = group.device(physical) {
        return Ok(device);
    }
    let device = create_on(instance, physical, options, cache, None)?.root();
    group.devices.retain(|(_, line)| line.strong_count() > 0);
    group
        .devices
        .push((physical, Arc::downgrade(device.line())));
    Ok(device)
}

impl Device {
    /// `MTLCreateSystemDefaultDevice`: [`Device::system_default_with`] the default options.
    pub fn system_default() -> Result<Device, Error> {
        Device::system_default_with(&DeviceOptions::default())
    }

    /// The device of the preferred qualifying physical device (discrete, then integrated,
    /// virtual, CPU): the first one, in that order, the runtime can create. The error of the
    /// first refusal when none can be, and [`Error::NoDevice`] when there is none.
    ///
    /// A physical device has one device per options, as Metal's GPU has one `MTLDevice`: while
    /// a handle to the device this or [`Device::all_with`] made with equal options lives (the
    /// same message sink, a clone of it, and every other field equal), the call returns that
    /// device. Once its last handle drops, the device tears down, and the next call makes a
    /// fresh one.
    ///
    /// Once it has made a device, the runtime keeps the loader library and every Vulkan driver
    /// it loaded for the life of the process: one bare instance, with no layers and no
    /// messenger, is never destroyed. The loader would otherwise unload the drivers with the
    /// last device and load them again for the next, which some drivers cannot do indefinitely.
    pub fn system_default_with(options: &DeviceOptions) -> Result<Device, Error> {
        let cache = CacheSetup::of(&options.shader_cache, "system_default_with")?;
        let mut made = lock(&MADE);
        let (at, instance) = group(&mut made, options)?;
        let mut refusal = None;
        for physical in ranked(&instance, options)? {
            match device_of(&mut made[at], &instance, physical, options, &cache) {
                Ok(device) => {
                    keep_drivers_loaded(&instance);
                    return Ok(device);
                }
                Err(error) => {
                    refusal.get_or_insert(error);
                }
            }
        }
        Err(refusal.unwrap_or(Error::NoDevice))
    }

    /// The identity of the physical device [`Device::system_default_with`] makes its device of
    /// with `options`, read without making any device: the first of
    /// [`Device::identities_with`]'s, whose order (the preferred qualifying kind, then
    /// enumeration order) and physical devices are the ones that call tries, so the two cannot
    /// name different physical devices. A runtime extension, no Metal call (Metal's program holds
    /// an `MTLDevice` to ask it anything).
    ///
    /// The physical device named is the one [`Device::system_default_with`] tries first. Where
    /// that call then refuses it as it makes its device (unsupported, or without a graphics and
    /// compute queue) it falls to the next, and makes a device of another physical device than
    /// the one named here, as [`Device::registry_ids_with`] lists such a physical device all the
    /// same.
    ///
    /// [`Error::NoDevice`] where `options` admit no physical device at Vulkan 1.2 or later, of
    /// which the runtime makes no device: that call answers `NoDevice` too where they admit none
    /// at all, and the refusal of the physical device it tried where they admit one the runtime
    /// cannot make a device of.
    pub fn system_default_identity_with(
        options: &DeviceOptions,
    ) -> Result<PhysicalDeviceIdentity, Error> {
        let first = Device::identities_with(options)?.into_iter().next();
        first.ok_or(Error::NoDevice)
    }

    /// The registry IDs ([`Device::registry_id`], `registryID`) of the physical devices
    /// [`Device::all_with`] makes its devices of, in its order, read without making any device: a
    /// runtime extension, no Metal call (Metal lists its devices, not their IDs). They are the
    /// [`PhysicalDeviceIdentity::registry_id`]s of [`Device::identities_with`], one per physical
    /// device it lists: one the runtime refuses as it makes its device (unsupported, or without a
    /// graphics and compute queue), which [`Device::all_with`] leaves out, is listed all the
    /// same, and [`Device::with_registry_id`] answers its refusal.
    pub fn registry_ids_with(options: &DeviceOptions) -> Result<Vec<u64>, Error> {
        let identities = Device::identities_with(options)?.into_iter();
        Ok(identities.map(|identity| identity.registry_id).collect())
    }

    /// The identity of each physical device `options` admit, in [`Device::system_default_with`]'s
    /// order (the preferred qualifying kind, then enumeration order), read without making any
    /// device. No Metal call: a runtime extension,
    /// whose nearest Metal counterpart is `MTLCopyAllDevices`, which answers devices that exist
    /// where this answers what a device of each physical device would answer, before any is
    /// made, so a consumer can choose a GPU and make only its device
    /// ([`Device::with_registry_id`]).
    ///
    /// The physical devices of `options.kinds` at Vulkan 1.2 or later are listed. Each identity
    /// is read by the property reads the device made of it takes its own from
    /// (`properties::query`), with no `vkCreateDevice`, no compile device and no queue. One the
    /// runtime refuses as it makes its device (unsupported, or without a graphics and compute
    /// queue), which [`Device::all_with`] leaves out, is listed all the same, and
    /// [`Device::with_registry_id`] answers its refusal. The instance is the one of the devices of
    /// equal options while one lives (as [`Device::system_default_with`] takes it), else one made
    /// for the call, which goes with it. [`Device::registry_ids_with`] and
    /// [`Device::system_default_identity_with`] answer from this list.
    pub fn identities_with(options: &DeviceOptions) -> Result<Vec<PhysicalDeviceIdentity>, Error> {
        let mut made = lock(&MADE);
        let (_, instance) = group(&mut made, options)?;
        let ranked = ranked(&instance, options)?;
        let identities = identified(&instance, ranked)
            .into_iter()
            .map(|(_, physical)| {
                // SAFETY: `physical` was enumerated from the instance, created at Vulkan 1.2 or
                // later, and `identified` kept only physical devices at Vulkan 1.2 or later.
                let (identity, _) = unsafe {
                    super::properties::query(&instance.raw, physical, instance.api_version)
                };
                PhysicalDeviceIdentity::of(&identity)
            });
        let identities = identities.collect();
        drop(made);
        // The instance made for the call goes here, outside the lock.
        drop(instance);
        Ok(identities)
    }

    /// The device of the physical device whose registry ID ([`Device::registry_id`]) is
    /// `registry_id`, among those [`Device::registry_ids_with`] lists for `options` (the first,
    /// in its order, where two share one), making no device of any other: a runtime extension
    /// (Metal's program finds the device of an ID among `MTLCopyAllDevices`'s, all of which exist
    /// already). As [`Device::system_default_with`] makes its device: while a handle to the
    /// device of equal options lives, it is that device, and the drivers stay loaded once a
    /// device was made. [`Error::NoDevice`] where no listed physical device has the ID; the
    /// refusal of the device where the runtime cannot make it.
    pub fn with_registry_id(registry_id: u64, options: &DeviceOptions) -> Result<Device, Error> {
        let cache = CacheSetup::of(&options.shader_cache, "with_registry_id")?;
        let mut made = lock(&MADE);
        let (at, instance) = group(&mut made, options)?;
        let ranked = ranked(&instance, options)?;
        let Some((_, physical)) =
            (identified(&instance, ranked).into_iter()).find(|&(id, _)| id == registry_id)
        else {
            return Err(Error::NoDevice);
        };
        let device = device_of(&mut made[at], &instance, physical, options, &cache)?;
        keep_drivers_loaded(&instance);
        Ok(device)
    }

    /// `MTLCopyAllDevices`: [`Device::all_with`] the default options.
    pub fn all() -> Result<Vec<Device>, Error> {
        Device::all_with(&DeviceOptions::default())
    }

    /// One device per qualifying physical device, in the order of
    /// [`Device::system_default_with`], sharing one instance. A physical device the runtime or
    /// the compile layer cannot use (unsupported, or without a graphics and compute queue) is
    /// left out; any other failure fails the call. Once it has made a device, the runtime keeps
    /// the drivers loaded for the life of the process, as [`Device::system_default_with`] does.
    /// A device of equal options that lives is returned again, as there (Metal's
    /// `MTLCopyAllDevices()[0]` is `MTLCreateSystemDefaultDevice()`).
    pub fn all_with(options: &DeviceOptions) -> Result<Vec<Device>, Error> {
        let cache = CacheSetup::of(&options.shader_cache, "all_with")?;
        let mut made = lock(&MADE);
        let (at, instance) = group(&mut made, options)?;
        let mut devices = Vec::new();
        for physical in ranked(&instance, options)? {
            match device_of(&mut made[at], &instance, physical, options, &cache) {
                Ok(device) => devices.push(device),
                Err(Error::Unsupported(_) | Error::NoDevice) => {}
                Err(error) => return Err(error),
            }
        }
        if !devices.is_empty() {
            keep_drivers_loaded(&instance);
        }
        Ok(devices)
    }
}

#[cfg(test)]
mod tests {
    use super::{deliver, MessageSink};
    use ash::vk;
    use metal2vulkan::device::{DriverDiagnostic, DriverFact};
    use metal2vulkan::vk::{DynamicState as D, PrimitiveTopology as T};
    use std::ffi::CString;
    use std::sync::{Arc, Mutex};

    /// The callback withholds a diagnostic the thread's current call is measured to report
    /// (MoltenVK 1.4.1's warning of a pipeline whose restart is dynamic), and counts it, inside
    /// that call alone: the same message outside it, from another thread while it runs, at
    /// another severity or kind, or with another text, reaches the sink, as every message does
    /// in a call that expects none.
    #[test]
    fn a_measured_diagnostic_is_withheld_inside_its_call_alone() {
        let warning = DriverFact::DynamicRestartWarns
            .pipeline_creation_diagnostics(
                &[D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE],
                T::TRIANGLE_LIST,
                &[],
            )
            .pop()
            .expect("the warning of a dynamic restart");
        let received: Arc<Mutex<Vec<String>>> = Arc::default();
        let sink = {
            let received = Arc::clone(&received);
            MessageSink::new(move |message| {
                received
                    .lock()
                    .unwrap()
                    .push(format!("{} {}", message.id_name, message.text))
            })
        };
        let user = std::ptr::from_ref(&sink) as usize;
        use vk::DebugUtilsMessageSeverityFlagsEXT as S;
        use vk::DebugUtilsMessageTypeFlagsEXT as K;
        let report = |severity: S, kind: K, text: &str| sent(user, &warning, severity, kind, text);
        let delivered = || received.lock().unwrap().len();
        report(S::WARNING, K::VALIDATION, &warning.text);
        assert_eq!(delivered(), 1, "outside the call");
        let ((), withheld) = crate::driver_facts::expecting(vec![warning.clone()], || {
            report(S::WARNING, K::VALIDATION, &warning.text);
            assert_eq!(delivered(), 1, "inside the call");
            // Another thread, started as the runtime starts its own and joined before the sink
            // goes.
            let other = warning.clone();
            crate::exit_guard::spawn(std::thread::Builder::new(), move || {
                sent(user, &other, S::WARNING, K::VALIDATION, &other.text)
            })
            .unwrap()
            .join()
            .unwrap();
            assert_eq!(delivered(), 2, "from another thread");
            report(S::ERROR, K::VALIDATION, &warning.text);
            report(S::WARNING, K::GENERAL, &warning.text);
            report(
                S::WARNING,
                K::VALIDATION,
                "VK_ERROR_FEATURE_NOT_PRESENT: Metal does not support something else.",
            );
            assert_eq!(delivered(), 5, "another severity, kind or text");
            report(S::WARNING, K::VALIDATION, &warning.text);
        });
        assert_eq!(withheld, 2, "withheld inside the call");
        report(S::WARNING, K::VALIDATION, &warning.text);
        assert_eq!(delivered(), 6, "after the call");
        let ((), withheld) = crate::driver_facts::expecting(Vec::new(), || {
            report(S::WARNING, K::VALIDATION, &warning.text);
        });
        assert_eq!((withheld, delivered()), (0, 7), "a call that expects none");
    }

    /// A kept internal pipeline cache is Metal's own on MoltenVK alone, which says so in an
    /// informational message; every other driver's is a second cache beside the shader cache's
    /// records, announced as `driver-pipeline-cache-kept`.
    #[test]
    fn a_kept_cache_is_metals_on_moltenvk_alone() {
        assert!(super::kept_cache_is_metals(Some(
            metal2vulkan::vk::DriverId::MOLTENVK
        )));
        let others = [
            Some(metal2vulkan::vk::DriverId::NVIDIA_PROPRIETARY),
            Some(metal2vulkan::vk::DriverId::MESA_RADV),
            Some(metal2vulkan::vk::DriverId::INTEL_OPEN_SOURCE_MESA),
            Some(metal2vulkan::vk::DriverId::MESA_LLVMPIPE),
            None,
        ];
        for driver in others {
            assert!(!super::kept_cache_is_metals(driver), "{driver:?}");
        }
    }

    /// A sink with a least severity hears no message below it but degrade notices, which it hears
    /// whatever their severity; a sink of `MessageSink::new` hears every severity; and a sink with
    /// another least severity is not the same sink, though it calls the same receiver.
    #[test]
    fn a_sink_hears_no_message_below_its_least_severity_but_notices() {
        use super::{DebugMessage, MessageKind, MessageSeverity as S};
        use crate::{Degradation, DegradeNotice};
        let received: Arc<Mutex<Vec<(S, bool)>>> = Arc::default();
        let every = {
            let received = Arc::clone(&received);
            MessageSink::new(move |message| {
                let notice = message.degradation.is_some();
                received.lock().unwrap().push((message.severity, notice))
            })
        };
        let message = |severity, notice: bool| {
            let mut message = DebugMessage {
                severity,
                kind: MessageKind::default(),
                id_name: String::new(),
                id: 0,
                text: String::new(),
                degradation: None,
            };
            if notice {
                let kind = Degradation::UnsizedPoints;
                let notice = DegradeNotice::new(kind, String::new(), String::new());
                message = DebugMessage {
                    severity,
                    ..notice.message()
                };
            }
            message
        };
        let all = [S::Verbose, S::Info, S::Warning, S::Error];
        for least in all {
            let sink = every.clone().with_least_severity(least);
            assert_eq!(sink.same(&every), least == S::Verbose, "{least:?}");
            for severity in all {
                assert_eq!(sink.hears(severity), severity >= least);
                for notice in [false, true] {
                    sink.deliver(&message(severity, notice));
                    let heard = received.lock().unwrap().pop();
                    let expected = (notice || severity >= least).then_some((severity, notice));
                    assert_eq!(heard, expected, "{least:?}: {severity:?}, notice {notice}");
                }
            }
        }
    }

    /// Deliver `warning`'s name and number with `text` at `severity` and `kind` to the sink at
    /// `user`.
    fn sent(
        user: usize,
        warning: &DriverDiagnostic,
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        kind: vk::DebugUtilsMessageTypeFlagsEXT,
        text: &str,
    ) {
        let id_name = CString::new(warning.id_name).unwrap();
        let text = CString::new(text).unwrap();
        let data = vk::DebugUtilsMessengerCallbackDataEXT::default()
            .message_id_name(&id_name)
            .message_id_number(warning.id)
            .message(&text);
        // SAFETY: the callback data and its strings live for the call, and `user` is the sink,
        // alive until every call of the test returned.
        unsafe { deliver(severity, kind, &data, user as *mut std::ffi::c_void) };
    }
}
