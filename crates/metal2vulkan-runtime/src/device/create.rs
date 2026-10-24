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
use super::properties::DeviceKind;
use super::{device_error, lock, Device, DeviceShared, InstanceShared, Parts};
use crate::degradation::{Degradation, DegradeNotice};
use crate::enables::{host_image_copy_names, listed_reported, listed_structures, Enables};
use crate::error::{vulkan, Error, InstanceApiVersion, LoaderNotFound};
use crate::exit_guard;
use crate::gpu_time;
use crate::image_barrier::MemoryReads;
use crate::loader_word::LoaderWord;
use crate::metal_level::{choose, Chosen, Host};
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
#[derive(Clone, Debug, Default)]
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
    /// `VK_EXT_present_mode_fifo_latest_ready`: `presentModeFifoLatestReady`) and the device
    /// lists it, the device enables the structure's features too; a device that lacks one is
    /// refused ([`Error::ExtensionFeature`](crate::Error::ExtensionFeature)) rather than enable
    /// the extension without it. One whose registry dependency names an instance extension the
    /// instance would not enable (`VK_KHR_swapchain` without `VK_KHR_surface` in
    /// `instance_extensions`, for example) refuses the options before the instance is made
    /// ([`Error::InstanceExtensionRequired`](crate::Error::InstanceExtensionRequired)).
    pub device_extensions: Vec<CString>,
    /// Device extensions each device enables only where it can, never refused for one: where the
    /// device lists it, the instance enables every instance extension its registry dependency
    /// names (as for `device_extensions`), and, for an extension whose feature structure the
    /// runtime knows (as for `device_extensions`), the device has every feature of it, which it
    /// then enables too. Elsewhere the device leaves it out, and is made all the same. Which a
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
    /// The degradation kinds each device announces at every occurrence, not once (empty by
    /// default: each kind once per device). Each occurrence's notice is the same
    /// [`DegradeNotice`] through the same sink (or stderr), on the thread whose call announces
    /// it, so a program that runs tasks on threads over one device sees each task's: where each
    /// kind's notice surfaces is in its [`Degradation`](crate::Degradation) docs. A kind a
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
    /// The macOS release whose Metal each device answers as, where releases answer differently
    /// ([`MetalLevel`](crate::MetalLevel): which pixel formats a descriptor takes and Metal
    /// describes). `None` by default: on a macOS host, the host's own release, or the newest
    /// measured release below it, announced once as `metal2vulkan-degrade-metal-level-below-host`
    /// (a host below every measured release, or whose release cannot be read, refuses the device,
    /// `Error::Unproven` with `Problem::HostReleaseUnmeasured`); on any other host, the newest,
    /// [`MetalLevel::NEWEST`](crate::MetalLevel::NEWEST). Named, the devices answer at it on any
    /// host, unannounced: for hosts that are not macOS, and for tests. No Metal counterpart:
    /// Metal answers as the release it runs on.
    pub metal_level: Option<crate::MetalLevel>,
    /// Keep each command buffer's execution record
    /// ([`CommandBuffer::executions`](crate::CommandBuffer::executions)): the pipeline each
    /// dispatch and draw ran, the bind shape it was planned for, and its `VkPipeline`s' creation
    /// times. Off by default: the device's encoders build no record and keep nothing for one, so
    /// a draw pays nothing for a record no one reads. Validation's executor and lanes turn it on.
    /// No Metal counterpart.
    #[doc(hidden)]
    pub record_executions: bool,
    /// Take each device as an identity no measurement names
    /// ([`DeviceDescription::assumed_unmeasured`]), whatever its driver, GPU and release are: as
    /// another GPU of its driver at its release. The device then takes the paths an unmeasured
    /// identity of its driver takes, in translation and in the runtime: no deviation or defect
    /// measured on one GPU alone, every exact path in place of an answer fact, every defect its
    /// driver is measured with assumed, and the informational message
    /// `metal2vulkan-unmeasured-driver` naming them as the device is made. Off by default. A
    /// diagnostic, for running a measured device as an unmeasured one of its driver runs (a
    /// report from a GPU no venue measured). No Metal counterpart.
    pub assume_unmeasured_identity: bool,
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

    /// Hand `message` to the sink, where it hears it; a panic inside it is caught.
    pub(crate) fn deliver(&self, message: &DebugMessage) {
        if message.degradation.is_none() && !self.hears(message.severity) {
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
    /// driver's or a layer's message. Match its [`Degradation`](crate::Degradation), not the text.
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

/// Whether `entry`'s loader lists `VK_KHR_portability_enumeration`.
pub(super) fn lists_portability(entry: &ash::Entry) -> Result<bool, Error> {
    lists_instance_extension(entry, ash::khr::portability_enumeration::NAME)
}

/// Whether the loader lists the instance extension `name`.
fn lists_instance_extension(entry: &ash::Entry, name: &CStr) -> Result<bool, Error> {
    // SAFETY: a global command of a loaded entry.
    let available =
        unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(vulkan)?;
    Ok(available
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(name)))
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
    let entry = open_loader(options.loader.as_deref())?;
    let portability = lists_portability(&entry)?;
    let mut extensions: Vec<&CStr> = Vec::new();
    if portability {
        extensions.push(ash::khr::portability_enumeration::NAME);
    }
    // The relay of a driver's command-buffer failures needs a messenger of its own
    // (`relay.rs`): debug utils wherever the instance lists them.
    let relays = lists_instance_extension(&entry, ash::ext::debug_utils::NAME)?;
    if options.messages.is_some() || relays {
        extensions.push(ash::ext::debug_utils::NAME);
    }
    // MoltenVK's Metal argument buffers, asked for where the instance can take layer settings,
    // ahead of the consumer's own, which win (`argument_buffers.rs`); and, where the relay hears
    // its failures, that it keep a device after a command buffer fails (`relay.rs`).
    let asks = lists_instance_extension(&entry, ash::ext::layer_settings::NAME)?;
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

/// The most `VkQueue`s a created device takes: logical queues share them round robin.
const MAX_QUEUES: u32 = 4;

/// The most `VkQueue`s a created device reserves for presenting queues
/// (`Device::new_command_queue_for_presenting`), outside the round robin: the taken families'
/// queues past [`MAX_QUEUES`], where they have them.
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
    /// The queue reserved for staging copies (`device/staging.rs`): the first queue of the first
    /// family of transfers alone whose copies take any texel (a granularity of one texel), which
    /// is none of the families above; `None` where there is none.
    pub(crate) staging: Option<(u32, u32)>,
}

/// The queues a device of `families` takes: where every family is equivalent to the first that
/// does graphics and compute work (the same flags, queue count, timestamp bits and transfer
/// granularity: MoltenVK's four families of one queue each, one Metal command queue per queue),
/// one queue of each family first, then the families' next queues in turn; otherwise that
/// family's queues alone. At most [`MAX_QUEUES`] either way. Queues of another family are
/// another Metal command queue on MoltenVK, whose command buffers the GPU runs beside the
/// first's; queues of one family there share one. The next queues in that order, at most
/// [`PRESENT_QUEUES`], are reserved for presenting queues. A queue of a family of transfers alone
/// is reserved for staging copies: a copy engine, which runs beside the graphics family's work
/// where another of its queues waits behind it; an asynchronous compute queue is none (its work
/// waited up to 1.05 s on a Radeon 780M and 1.19 s on an RTX 3090 behind a kernel that filled the
/// GPU). `None` where no family does graphics and compute work.
pub(crate) fn queue_plan(families: &[vk::QueueFamilyProperties]) -> Option<QueuePlan> {
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
    let present = order.split_off(order.len().min(MAX_QUEUES as usize));
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
        present: present.into_iter().take(PRESENT_QUEUES as usize).collect(),
        staging,
    })
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

/// `description` as `options` take its identity: an identity no measurement names where they
/// assume one ([`DeviceOptions::assume_unmeasured_identity`]), else as the device reports it.
pub(crate) fn identity_as_options(
    mut description: DeviceDescription,
    options: &DeviceOptions,
) -> DeviceDescription {
    description.assumed_unmeasured = options.assume_unmeasured_identity;
    description
}

/// A device on `physical`, made from `instance`, answering at `level`.
fn create_on(
    instance: &Arc<InstanceShared>,
    physical: vk::PhysicalDevice,
    options: &DeviceOptions,
    cache: &CacheSetup,
    level: Chosen,
) -> Result<Device, Error> {
    // SAFETY: `physical` was enumerated from the instance, created at `api_version`.
    let description =
        unsafe { DeviceDescription::query(&instance.raw, physical, instance.api_version) }
            .map_err(device_error)?;
    let description = identity_as_options(description, options);
    // SAFETY: as above.
    unsafe { timeline_conformant(&instance.raw, physical) }?;
    argument_buffers_kept(&description)?;
    // SAFETY: `physical` was enumerated from the instance.
    let families = unsafe {
        instance
            .raw
            .get_physical_device_queue_family_properties(physical)
    };
    let plan = queue_plan(&families).ok_or(Error::NoDevice)?;
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
    let extra: Vec<CString> = options
        .device_extensions
        .iter()
        .cloned()
        .chain(calibration.map(|extension| extension.name().to_owned()))
        .chain((creation_feedback && !feedback_asked).then(|| feedback_name.to_owned()))
        .chain(dma_buf.into_iter().flatten().map(CStr::to_owned))
        .chain(dma_buf_images.into_iter().flatten().map(CStr::to_owned))
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
    let priorities = [1.0f32; (MAX_QUEUES + PRESENT_QUEUES) as usize];
    let mut taken: Vec<(u32, usize)> = Vec::new();
    for &(family, _) in plan.queues.iter().chain(&plan.present).chain(&plan.staging) {
        match taken.iter_mut().find(|(at, _)| *at == family) {
            Some((_, count)) => *count += 1,
            None => taken.push((family, 1)),
        }
    }
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
    let loader_word = unsafe { LoaderWord::of_device(raw.handle()) };
    let primary = families.get(queue_family as usize);
    let memory_reads = MemoryReads::of(
        primary.map_or(vk::QueueFlags::empty(), |family| family.queue_flags),
        enables.names(),
    );
    let parts = Parts {
        instance: Arc::clone(instance),
        physical,
        raw,
        loader_word,
        queue_family,
        present_queues: plan.present.len(),
        queues: plan.queues.into_iter().chain(plan.present).collect(),
        staging: plan.staging,
        compile,
        description: enabled,
        enabled: set,
        extensions: enables.names().map(CStr::to_owned).collect(),
        calibration,
        gpu_times_from_host: options.gpu_times_from_host,
        announce_each: options.announce_each.clone(),
        lowering_refuses: options.lowering_refuses.clone(),
        record_executions: options.record_executions,
        view_reinterpretation,
        generated_commands,
        host_image_copy,
        linear_color_attachment,
        graphics_pipeline_library,
        creation_feedback: creation_feedback || feedback_asked,
        dma_buf: dma_buf.is_some(),
        dma_buf_images: dma_buf_images.is_some(),
        sparse_features,
        memory_reads,
        owned: true,
        cache: cache.clone(),
        metal_level: level,
    };
    // SAFETY: the device was made from these parts just now.
    let device = unsafe { Device::assemble(parts) }?;
    if let Some(text) = layers_dropped(&instance.dropped_layers, device.name()) {
        device.inform(LAYERS_DROPPED, text);
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
    /// The options, less the message sink, which the instance holds (`InstanceShared::sink`):
    /// the group keeps no sink alive past its devices.
    options: DeviceOptions,
    instance: Weak<InstanceShared>,
    devices: Vec<(vk::PhysicalDevice, Weak<DeviceShared>)>,
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
        (sinks && same_options(&self.options, options)).then_some(instance)
    }

    /// The live device of `physical`, if the group has one.
    fn device(&self, physical: vk::PhysicalDevice) -> Option<Device> {
        self.devices
            .iter()
            .find(|(of, _)| *of == physical)
            .and_then(|(_, device)| device.upgrade())
            .map(Device)
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
        devices
            .filter(|(_, device)| device.strong_count() > 0)
            .count()
    })
}

/// Whether `a` and `b` are equal options, but for their message sinks (compared by `Made`).
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
        announce_each,
        lowering_refuses,
        metal_level,
        record_executions,
        assume_unmeasured_identity,
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
        && *announce_each == b.announce_each
        && *lowering_refuses == b.lowering_refuses
        && *metal_level == b.metal_level
        && *record_executions == b.record_executions
        && *assume_unmeasured_identity == b.assume_unmeasured_identity
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
            ..options.clone()
        },
        instance: Arc::downgrade(&instance),
        devices: Vec::new(),
    });
    Ok((made.len() - 1, instance))
}

/// The device of `physical` in `group`: its live one, or one made now and kept there, answering
/// at `level`.
fn device_of(
    group: &mut Made,
    instance: &Arc<InstanceShared>,
    physical: vk::PhysicalDevice,
    options: &DeviceOptions,
    cache: &CacheSetup,
    level: Chosen,
) -> Result<Device, Error> {
    if let Some(device) = group.device(physical) {
        return Ok(device);
    }
    let device = create_on(instance, physical, options, cache, level)?;
    group.devices.retain(|(_, weak)| weak.strong_count() > 0);
    group.devices.push((physical, Arc::downgrade(&device.0)));
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
    ///
    /// With no `options.metal_level`, a macOS host below every measured release of Metal, or
    /// whose release cannot be read, refuses every device (`Unproven`,
    /// `Problem::HostReleaseUnmeasured`).
    pub fn system_default_with(options: &DeviceOptions) -> Result<Device, Error> {
        let cache = CacheSetup::of(&options.shader_cache, "system_default_with")?;
        let level = choose(options.metal_level, Host::this(), "system_default_with")?;
        let mut made = lock(&MADE);
        let (at, instance) = group(&mut made, options)?;
        let mut refusal = None;
        for physical in ranked(&instance, options)? {
            match device_of(&mut made[at], &instance, physical, options, &cache, level) {
                Ok(device) => {
                    keep_drivers_loaded(&instance.entry);
                    return Ok(device);
                }
                Err(error) => {
                    refusal.get_or_insert(error);
                }
            }
        }
        Err(refusal.unwrap_or(Error::NoDevice))
    }

    /// The registry IDs ([`Device::registry_id`], `registryID`) of the physical devices
    /// [`Device::all_with`] makes its devices of, in its order, read without making any device: a
    /// runtime extension, no Metal call (Metal lists its devices, not their IDs). The physical
    /// devices of `options.kinds` at Vulkan 1.2 or later are listed, each by its `deviceUUID`; one
    /// the runtime refuses as it makes its device (unsupported, or without a graphics and compute
    /// queue), which [`Device::all_with`] leaves out, is listed all the same, and
    /// [`Device::with_registry_id`] answers its refusal. The instance is the one of the devices of
    /// equal options while one lives (as [`Device::system_default_with`] takes it), else one made
    /// for the call, which goes with it.
    pub fn registry_ids_with(options: &DeviceOptions) -> Result<Vec<u64>, Error> {
        let mut made = lock(&MADE);
        let (_, instance) = group(&mut made, options)?;
        let ranked = ranked(&instance, options)?;
        let ids = identified(&instance, ranked).into_iter().map(|(id, _)| id);
        let ids = ids.collect();
        drop(made);
        // The instance made for the call goes here, outside the lock.
        drop(instance);
        Ok(ids)
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
        let level = choose(options.metal_level, Host::this(), "with_registry_id")?;
        let mut made = lock(&MADE);
        let (at, instance) = group(&mut made, options)?;
        let ranked = ranked(&instance, options)?;
        let Some((_, physical)) =
            (identified(&instance, ranked).into_iter()).find(|&(id, _)| id == registry_id)
        else {
            return Err(Error::NoDevice);
        };
        let device = device_of(&mut made[at], &instance, physical, options, &cache, level)?;
        keep_drivers_loaded(&instance.entry);
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
        let level = choose(options.metal_level, Host::this(), "all_with")?;
        let mut made = lock(&MADE);
        let (at, instance) = group(&mut made, options)?;
        let mut devices = Vec::new();
        for physical in ranked(&instance, options)? {
            match device_of(&mut made[at], &instance, physical, options, &cache, level) {
                Ok(device) => devices.push(device),
                Err(Error::Unsupported(_) | Error::NoDevice) => {}
                Err(error) => return Err(error),
            }
        }
        if !devices.is_empty() {
            keep_drivers_loaded(&instance.entry);
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
