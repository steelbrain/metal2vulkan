//! Vulkan device creation, enabled capabilities, and per-thread context lifetime.

use super::enables::{self, Available, FeatureChain};
use super::loader_word::LoaderWord;
use super::profiles::{
    self, ActiveTool, ProfileRestriction, ProfilesLayer, ProfilesSettings, RestrictionError,
    PROFILES_LAYER,
};
use crate::coverage::{ExecutionClass, Leg};
use crate::gpu::wait::{self, SubmissionBackend};
use crate::gpu::DeviceType;
use ash::{vk, Device, Entry, Instance};
use metal2vulkan::device::{ApiVersion, VkFeature};
use metal2vulkan::device::{DeviceDescription, EnableSet};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// What a row's environment records as its context's `layers`
/// (`device_facts::RECORDED_LAYERS`). Every context runs under `VK_LAYER_KHRONOS_validation` with
/// core and synchronization validation, and a message sink ([`VulkanContext::take_messages`])
/// that fails the row: every engine candidate run and every L4 acceptance run. It enables the
/// engine's list on top of the base list ([`enables::engine_list`]).
pub(super) use crate::candidate::device_facts::RECORDED_LAYERS;

/// The validation layer every engine run enables.
pub(super) const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The robustness leg of every engine context this process opens: decided once, by
/// [`run_robustness_off`] or by the first context opened without it.
static LEG: OnceLock<Leg> = OnceLock::new();

/// Run every engine context this process opens in the robustness-off leg (a command's
/// `--robustness-off`): enable no robustness feature, so the product compiles the in-shader
/// containment encodings. Call it once, before any device opens: the leg is decided once per
/// process, and a context already opened in the stated leg refuses it.
pub fn run_robustness_off() -> Result<(), String> {
    LEG.set(Leg::RobustnessOff).map_err(|_| {
        "the robustness leg is already decided: pass --robustness-off once, before any device \
         opens"
            .to_string()
    })
}

/// The leg of every engine context this process opens ([`run_robustness_off`]); reading it
/// decides that it is the stated leg if none was set.
pub fn requested_leg() -> Leg {
    *LEG.get_or_init(|| Leg::Stated)
}

/// Run `body` against this thread's Vulkan context for `device_type`, under the validation layer
/// ([`RECORDED_LAYERS`]), creating the device on first use. One device of each type serves every
/// call on the thread: device creation dominates the cost of a single case, and a corpus sweep
/// makes thousands of calls. The layer must be installed: without it this fails, and nothing
/// runs unvalidated.
///
/// A context is only ever created and used by its own thread. Each GPU call re-checks the hang
/// tripwire before touching the device.
pub(super) fn with_validated_context<R>(
    device_type: DeviceType,
    body: impl FnOnce(&VulkanContext) -> Result<R, String>,
) -> Result<R, String> {
    with_slot(device_type, Purpose::Executor, body)
}

/// The check of the device `device_type` opens on (the engine's validated context, which the
/// run then uses) against the C3 venue profile `venue` and the identities the venue allows,
/// `allowed` ([`profiles::venue_check`]): its identity (device name, vendor and device IDs, driver
/// ID, version, and info) one of `allowed`, and its description as the context queried it, before a
/// restriction or the enables, read at API 1.2. A restricted context runs a profile, not a venue's
/// device, and is refused.
pub fn venue_check(
    device_type: DeviceType,
    venue: &crate::c3::VenueProfile,
    allowed: &[crate::c3::DeviceIdentity],
) -> Result<crate::c3::VenueCheck, String> {
    with_validated_context(device_type, |context| {
        if let Some(restriction) = context.restriction {
            return Err(format!(
                "a context restricted to {} runs no venue's device",
                restriction.stem
            ));
        }
        // SAFETY: the context's physical device belongs to its live instance.
        let properties = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical)
        };
        let live = crate::c3::DeviceIdentity {
            device: extension_name(&properties.device_name)
                .to_string_lossy()
                .into_owned(),
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            driver_id: context.driver.driver_id,
            driver_version: properties.driver_version,
            driver_info: context.driver.driver_info.clone(),
        };
        let queried = context.queried_description.as_ref().map_err(Clone::clone)?;
        profiles::venue_check(
            venue,
            allowed,
            &live,
            &crate::acceptance::read_at(queried, ApiVersion::V1_2),
        )
    })
}

/// [`with_validated_context`] for the runtime probe (`corpus-vulkan --probe-runtime`): a validated
/// context of its own, in its own slot, that also enables what the probe measures where the device
/// lists it ([`enables::enable_runtime_probe`]). No case runs on it, and no executor context
/// enables those extensions.
pub(super) fn with_runtime_probe_context<R>(
    device_type: DeviceType,
    body: impl FnOnce(&VulkanContext) -> Result<R, String>,
) -> Result<R, String> {
    with_slot(device_type, Purpose::RuntimeProbe, body)
}

/// What a context is opened for, which picks its slot on the thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Purpose {
    /// An executor's context ([`with_validated_context`]).
    Executor,
    /// The runtime probe's validated context ([`with_runtime_probe_context`]).
    RuntimeProbe,
}

fn with_slot<R>(
    device_type: DeviceType,
    purpose: Purpose,
    body: impl FnOnce(&VulkanContext) -> Result<R, String>,
) -> Result<R, String> {
    #[derive(Default)]
    struct Contexts {
        cpu: Option<VulkanContext>,
        gpu: Option<VulkanContext>,
        cpu_probe: Option<VulkanContext>,
        gpu_probe: Option<VulkanContext>,
    }
    thread_local! {
        static CONTEXTS: RefCell<Contexts> = RefCell::new(Contexts::default());
    }
    if device_type == DeviceType::Gpu {
        crate::gpu::tripwire::refuse_if_tripped()?;
    }
    CONTEXTS.with(|contexts| {
        let mut contexts = contexts.borrow_mut();
        let slot = match (device_type, purpose) {
            (DeviceType::Cpu, Purpose::Executor) => &mut contexts.cpu,
            (DeviceType::Gpu, Purpose::Executor) => &mut contexts.gpu,
            (DeviceType::Cpu, Purpose::RuntimeProbe) => &mut contexts.cpu_probe,
            (DeviceType::Gpu, Purpose::RuntimeProbe) => &mut contexts.gpu_probe,
        };
        if slot.is_none() {
            *slot = Some(VulkanContext::new(device_type, purpose)?);
        }
        body(slot.as_ref().expect("initialized Vulkan context"))
    })
}

/// What device selection needs to know about one enumerated physical device.
#[derive(Clone, Copy)]
struct PhysicalCandidate {
    device_type: vk::PhysicalDeviceType,
    api_version: u32,
    compute_family: Option<u32>,
}

/// Enumerate `instance`'s physical devices and pick the one [`select_physical_device`] selects
/// for `wanted`: `(device, compute queue family)`.
fn select_enumerated(
    instance: &Instance,
    wanted: DeviceType,
) -> Result<(vk::PhysicalDevice, u32), String> {
    // SAFETY: `instance` is live.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|error| format!("enumerate Vulkan devices: {error}"))?;
    let candidates = physical_devices.iter().map(|&physical| {
        // SAFETY: `physical` was enumerated from `instance`.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        // SAFETY: as above.
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        PhysicalCandidate {
            device_type: properties.device_type,
            api_version: properties.api_version,
            compute_family: families
                .iter()
                .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|family| family as u32),
        }
    });
    select_physical_device(wanted, candidates)
        .map(|(index, family)| (physical_devices[index], family))
}

/// The error a CPU selection reports when no CPU implementation is present.
const NO_CPU_DEVICE: &str = "no CPU device: no Vulkan 1.2 compute-capable CPU implementation \
                             (lavapipe, SwiftShader) is available, and a GPU is never substituted";

/// Pick the device `wanted` names from `candidates`, in enumeration order. There is no ranking
/// across kinds: a CPU request matches only `CPU` devices and never falls back to a GPU, and a
/// GPU request matches only discrete, integrated, or virtual GPUs and never falls back to a CPU
/// implementation. Among several matches the loader's first is taken.
fn select_physical_device(
    wanted: DeviceType,
    candidates: impl IntoIterator<Item = PhysicalCandidate>,
) -> Result<(usize, u32), String> {
    candidates
        .into_iter()
        .enumerate()
        .find_map(|(index, candidate)| {
            let kind_matches = match wanted {
                DeviceType::Cpu => candidate.device_type == vk::PhysicalDeviceType::CPU,
                DeviceType::Gpu => matches!(
                    candidate.device_type,
                    vk::PhysicalDeviceType::DISCRETE_GPU
                        | vk::PhysicalDeviceType::INTEGRATED_GPU
                        | vk::PhysicalDeviceType::VIRTUAL_GPU
                ),
            };
            (kind_matches && candidate.api_version >= vk::API_VERSION_1_2)
                .then_some(candidate.compute_family)
                .flatten()
                .map(|family| (index, family))
        })
        .ok_or_else(|| match wanted {
            DeviceType::Cpu => NO_CPU_DEVICE.to_string(),
            DeviceType::Gpu => "no Vulkan 1.2 compute-capable GPU is available".to_string(),
        })
}

#[test]
fn device_selection_never_crosses_device_kinds() {
    let device = |device_type, compute_family| PhysicalCandidate {
        device_type,
        api_version: vk::API_VERSION_1_2,
        compute_family,
    };
    let gpus = [
        device(vk::PhysicalDeviceType::INTEGRATED_GPU, Some(0)),
        device(vk::PhysicalDeviceType::DISCRETE_GPU, Some(1)),
        device(vk::PhysicalDeviceType::OTHER, Some(0)),
    ];
    // This host's shape: MoltenVK exposes only GPUs, so a CPU request fails outright.
    let error = select_physical_device(DeviceType::Cpu, gpus).unwrap_err();
    assert_eq!(
        error,
        "no CPU device: no Vulkan 1.2 compute-capable CPU implementation (lavapipe, SwiftShader) \
         is available, and a GPU is never substituted"
    );
    // No "discrete first" ranking: the loader's first GPU wins.
    assert_eq!(
        select_physical_device(DeviceType::Gpu, gpus).unwrap(),
        (0, 0)
    );

    let mixed = [
        device(vk::PhysicalDeviceType::DISCRETE_GPU, Some(0)),
        device(vk::PhysicalDeviceType::CPU, None),
        device(vk::PhysicalDeviceType::CPU, Some(2)),
        device(vk::PhysicalDeviceType::VIRTUAL_GPU, Some(0)),
    ];
    assert_eq!(
        select_physical_device(DeviceType::Cpu, mixed).unwrap(),
        (2, 2)
    );
    assert_eq!(
        select_physical_device(DeviceType::Gpu, mixed).unwrap(),
        (0, 0)
    );

    let cpu_only = [device(vk::PhysicalDeviceType::CPU, Some(0))];
    assert_eq!(
        select_physical_device(DeviceType::Cpu, cpu_only).unwrap(),
        (0, 0)
    );
    assert!(select_physical_device(DeviceType::Gpu, cpu_only).is_err());

    let old_cpu = [PhysicalCandidate {
        api_version: vk::API_VERSION_1_1,
        ..cpu_only[0]
    }];
    assert!(select_physical_device(DeviceType::Cpu, old_cpu).is_err());
}

/// The commands that record the extended dynamic states: Vulkan 1.3's core entry points in a
/// context at 1.3 or later, where the functionality is core without a feature (a profile of that
/// version need not list the extension), else `VK_EXT_extended_dynamic_state`'s, where the context
/// enabled it, with `VK_EXT_extended_dynamic_state2`'s primitive restart where it enabled that too.
/// vk.xml aliases each command of the extensions to the core one, so the tables hold the same
/// signatures.
#[derive(Clone, Copy)]
pub(super) struct ExtendedDynamicState {
    pub(super) cull_mode: vk::PFN_vkCmdSetCullMode,
    pub(super) front_face: vk::PFN_vkCmdSetFrontFace,
    pub(super) primitive_topology: vk::PFN_vkCmdSetPrimitiveTopology,
    pub(super) depth_test_enable: vk::PFN_vkCmdSetDepthTestEnable,
    pub(super) depth_write_enable: vk::PFN_vkCmdSetDepthWriteEnable,
    pub(super) depth_compare_op: vk::PFN_vkCmdSetDepthCompareOp,
    pub(super) stencil_test_enable: vk::PFN_vkCmdSetStencilTestEnable,
    pub(super) stencil_op: vk::PFN_vkCmdSetStencilOp,
    /// Binds vertex buffers with their strides, for a pipeline whose strides are dynamic state
    /// (`VERTEX_INPUT_BINDING_STRIDE`).
    pub(super) bind_vertex_buffers2: vk::PFN_vkCmdBindVertexBuffers2,
    /// `vkCmdSetPrimitiveRestartEnable`, Vulkan 1.3's core command or
    /// `VK_EXT_extended_dynamic_state2`'s where the context enabled it, for a pipeline that makes
    /// the topology and restart dynamic.
    pub(super) primitive_restart_enable: Option<vk::PFN_vkCmdSetPrimitiveRestartEnable>,
}

impl ExtendedDynamicState {
    /// Vulkan 1.3's commands, which `device` loaded when it was created.
    fn core(device: &Device) -> ExtendedDynamicState {
        let f = device.fp_v1_3();
        ExtendedDynamicState {
            cull_mode: f.cmd_set_cull_mode,
            front_face: f.cmd_set_front_face,
            primitive_topology: f.cmd_set_primitive_topology,
            depth_test_enable: f.cmd_set_depth_test_enable,
            depth_write_enable: f.cmd_set_depth_write_enable,
            depth_compare_op: f.cmd_set_depth_compare_op,
            stencil_test_enable: f.cmd_set_stencil_test_enable,
            stencil_op: f.cmd_set_stencil_op,
            bind_vertex_buffers2: f.cmd_bind_vertex_buffers2,
            primitive_restart_enable: Some(f.cmd_set_primitive_restart_enable),
        }
    }

    /// `VK_EXT_extended_dynamic_state`'s commands, from `extension`'s table, with
    /// `VK_EXT_extended_dynamic_state2`'s restart from `second`'s where the context enabled it.
    fn extension(
        extension: &ash::ext::extended_dynamic_state::Device,
        second: Option<&ash::ext::extended_dynamic_state2::Device>,
    ) -> ExtendedDynamicState {
        let f = extension.fp();
        ExtendedDynamicState {
            cull_mode: f.cmd_set_cull_mode_ext,
            front_face: f.cmd_set_front_face_ext,
            primitive_topology: f.cmd_set_primitive_topology_ext,
            depth_test_enable: f.cmd_set_depth_test_enable_ext,
            depth_write_enable: f.cmd_set_depth_write_enable_ext,
            depth_compare_op: f.cmd_set_depth_compare_op_ext,
            stencil_test_enable: f.cmd_set_stencil_test_enable_ext,
            stencil_op: f.cmd_set_stencil_op_ext,
            bind_vertex_buffers2: f.cmd_bind_vertex_buffers2_ext,
            primitive_restart_enable: second.map(|s| s.fp().cmd_set_primitive_restart_enable_ext),
        }
    }
}

pub(super) struct VulkanContext {
    /// The backend every wait on this device reports in a hang record.
    pub(super) backend: SubmissionBackend,
    _entry: Entry,
    pub(super) instance: Instance,
    pub(super) device: Device,
    /// The commands of the extended dynamic states, where the context has them
    /// ([`ExtendedDynamicState`]): the render executor records a pipeline's extended dynamic
    /// states through them.
    pub(super) extended_dynamic_state: Option<ExtendedDynamicState>,
    pub(super) physical: vk::PhysicalDevice,
    pub(super) queue_family: u32,
    pub(super) queue: vk::Queue,
    pub(super) sampler_anisotropy: bool,
    shader_subgroup_extended_types: bool,
    multisample_array_image: bool,
    pub(super) max_sampler_anisotropy: f32,
    pub(super) storage_buffer_alignment: u64,
    pub(super) max_storage_buffer_range: u64,
    /// What this device enabled, in the engine's vocabulary.
    pub(super) enabled: EnableSet,
    /// The profile of this device as opened: its queried properties, limits, formats, and float
    /// controls, restricted to [`VulkanContext::enabled`] (L4 acceptance checks every requirement
    /// beyond enablement against it; `Err` makes such a requirement unchecked).
    pub(super) opened: Result<metal2vulkan::device::DeviceProfile, String>,
    /// The description [`VulkanContext::opened`] derives from: `DeviceDescription::query` on
    /// the opened physical device, restricted to [`VulkanContext::enabled`]
    /// (`DeviceDescription::with_enabled`). L4 acceptance checks enablement against it, and the
    /// engine executor derives its `Device` from it under its own policy.
    pub(super) opened_description: Result<metal2vulkan::device::DeviceDescription, String>,
    /// `DeviceDescription::query` on the opened physical device at API 1.2, before a restriction
    /// replaces it or the enables restrict it: the device as it is, which a venue's run holds to
    /// the venue's profile ([`venue_check`]).
    queried_description: Result<metal2vulkan::device::DeviceDescription, String>,
    /// The ICD's `VkPhysicalDeviceDriverProperties`, the backend identity of every row.
    pub(super) driver: DriverIdentity,
    /// The class of the device that executes ([`ExecutionClass::of_description`] of its queried
    /// identity): the device every census names, also under a restriction, whose description
    /// ([`VulkanContext::opened_description`]) is P's and may name no device.
    pub(super) execution_class: Option<ExecutionClass>,
    /// The tools active on the device (`vkGetPhysicalDeviceToolPropertiesEXT`), recorded with
    /// every row.
    pub(super) tools: Vec<ActiveTool>,
    /// The committed profile `VK_LAYER_KHRONOS_profiles` restricts the device to, checked
    /// ([`profiles::check_restricted`]).
    pub(super) restriction: Option<&'static ProfileRestriction>,
    /// The extensions the restricted context hides ([`ProfileRestriction::hidden`]; empty when
    /// unrestricted).
    hidden: BTreeSet<String>,
    /// The robustness leg the context runs under ([`requested_leg`]).
    pub(super) leg: Leg,
    /// The robustness features the stated leg could not enable
    /// (`enables::Enables::unmet_robustness`): every run on the context is refused naming them.
    pub(super) unmet_robustness: BTreeSet<VkFeature>,
    /// What the context can do with a sparse case texture.
    pub(super) sparse_images: SparseImages,
    /// Whether the context enabled `VK_EXT_device_generated_commands` with its
    /// `deviceGeneratedCommands` ([`enables::enable_generated_commands`]).
    pub(super) generated_commands: bool,
    /// What the runtime probe's context enabled beyond the engine's list
    /// ([`with_runtime_probe_context`]); `None` on every other context.
    pub(super) runtime_probe: Option<enables::RuntimeProbeExtensions>,
    /// The validation layer's messenger.
    messenger: (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT),
    /// The sink the messenger writes.
    messages: &'static MessageSink,
    /// The loader's word at the start of [`VulkanContext::device`]'s handle, as `vkCreateDevice`
    /// left it, put back before every call the loader serves itself ([`LoaderWord`]).
    loader_word: LoaderWord,
}

/// What a context can do with a sparse case texture: the executor features it enabled
/// (`sparseBinding`, `sparseResidencyImage2D`, `sparseResidencyImage3D`, each where the device
/// supports it, on an engine context only), and whether its queue binds sparse memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct SparseImages {
    pub(super) features: BTreeSet<enables::ExecutorFeature>,
    /// The context's queue family has `VK_QUEUE_SPARSE_BINDING_BIT`.
    pub(super) queue: bool,
}

/// Which Vulkan implementation a context opened: rows from different ICDs are different backends,
/// never relabeled as one another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DriverIdentity {
    /// `VkDriverId` as its raw value (`VK_DRIVER_ID_MOLTENVK` is 14, `MESA_LLVMPIPE` 13).
    pub(super) driver_id: i32,
    pub(super) driver_name: String,
    pub(super) driver_info: String,
}

/// Whether the messenger also writes each failing message to stderr as the layer reports it
/// ([`echo_validation_messages`]).
static ECHO_MESSAGES: AtomicBool = AtomicBool::new(false);

/// Write every message that fails a row to stderr as well, the moment the layer reports it. The
/// engine device worker turns this on for its whole process: a worker that a driver fault kills
/// never drains its sink, and its parent reports the crash with the worker's stderr, so a crash
/// after a validation message shows the message.
pub fn echo_validation_messages() {
    ECHO_MESSAGES.store(true, Ordering::Relaxed);
}

/// What the validation layer reported that fails a row: every `ERROR` message, and every
/// `WARNING` of the validation type. Bounded so a flood cannot grow without limit.
///
/// But for one message about the driver: on a context whose opened profile takes
/// `DriverDeviation::MetalVertexSimdgroups` ([`MessageSink::unadvertised_vertex`]), the layer's
/// report of a vertex module's subgroup operations outside `subgroupSupportedStages`
/// (`candidate/unadvertised_stage.rs`) fails nothing: the translation relies on the driver's
/// measured answer there, as the module's requirements state. Nor does a driver diagnostic every
/// pipeline making of the opened device is measured to report ([`MessageSink::making`]), nor, on a
/// device whose driver runs the set layouts its support query refuses
/// ([`MessageSink::unsupported_sets`]), the layer's report of such a layout
/// (`candidate/unsupported_set.rs`).
pub(super) struct MessageSink {
    messages: Mutex<Vec<String>>,
    /// Whether the context's opened profile takes `DriverDeviation::MetalVertexSimdgroups`, set
    /// once it is derived.
    unadvertised_vertex: AtomicBool,
    /// The diagnostics the opened device's driver facts measure every pipeline making to report,
    /// whatever its state: its warning of each color attachment it cannot blend
    /// (`DriverFact::BlendlessColorAttachmentsWarn`), for each format the device lists as a color
    /// attachment without blending (`DeviceDescription::blendless_color_formats`). The driver
    /// reports one only as it makes a render pipeline, naming the format, so a message of exactly
    /// one of them fails no row; set once the opened description is known.
    making: OnceLock<Vec<metal2vulkan::device::DriverDiagnostic>>,
    /// Whether the opened device's driver runs the descriptor set layouts its
    /// `vkGetDescriptorSetLayoutSupport` refuses, as the runtime makes them there
    /// (`metal2vulkan_runtime::makes_layouts_past_support`: a heap's set 1 on MoltenVK), set once
    /// the opened description is known.
    unsupported_sets: AtomicBool,
}

impl MessageSink {
    /// The most messages kept between two drains; one more entry says the rest were dropped.
    const MAX_MESSAGES: usize = 64;
    /// The longest message text kept.
    const MAX_MESSAGE_BYTES: usize = 2048;

    fn record(&self, message: String) {
        let mut messages = self.messages.lock().unwrap_or_else(|p| p.into_inner());
        if messages.len() < Self::MAX_MESSAGES {
            messages.push(message);
        } else if messages.len() == Self::MAX_MESSAGES {
            messages.push("further validation messages were dropped".into());
        }
    }

    fn take(&self) -> Vec<String> {
        let mut messages = self.messages.lock().unwrap_or_else(|p| p.into_inner());
        messages.drain(..).collect()
    }
}

/// Whether the validation layer's message of `severity` and `types` fails the row.
fn fails_row(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
) -> bool {
    severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        || (severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
            && types.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION))
}

/// The messenger callback: records a failing message in the context's [`MessageSink`].
unsafe extern "system" fn record_validation_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !fails_row(severity, types) || user.is_null() || data.is_null() {
        return vk::FALSE;
    }
    // SAFETY: the layer passes a live callback-data struct.
    if withholds(severity, types, unsafe { &*data }) {
        return vk::FALSE;
    }
    // SAFETY: `user` is the `&'static MessageSink` the messenger was created with, and the layer
    // passes a live callback-data struct.
    let (sink, data) = unsafe { (&*user.cast::<MessageSink>(), &*data) };
    let text = |pointer: *const std::ffi::c_char| {
        if pointer.is_null() {
            String::new()
        } else {
            // SAFETY: a non-null string of the callback data is NUL-terminated (Vulkan spec).
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        }
    };
    if sink.unadvertised_vertex.load(Ordering::Relaxed)
        && crate::candidate::unadvertised_stage::shape(
            &text(data.p_message_id_name),
            &text(data.p_message),
        )
    {
        return vk::FALSE;
    }
    if sink.unsupported_sets.load(Ordering::Relaxed)
        && crate::candidate::unsupported_set::shape(
            &text(data.p_message_id_name),
            &text(data.p_message),
        )
    {
        return vk::FALSE;
    }
    if severity == vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        && types.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
        && sink.making.get().is_some_and(|making| {
            let (id_name, text) = (text(data.p_message_id_name), text(data.p_message));
            making
                .iter()
                .any(|d| d.id_name == id_name && d.id == data.message_id_number && d.text == text)
        })
    {
        return vk::FALSE;
    }
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "error"
    } else {
        "warning"
    };
    let mut message = format!(
        "{level} {}: {}",
        text(data.p_message_id_name),
        text(data.p_message)
    );
    if message.len() > MessageSink::MAX_MESSAGE_BYTES {
        let mut end = MessageSink::MAX_MESSAGE_BYTES;
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        message.truncate(end);
    }
    if ECHO_MESSAGES.load(Ordering::Relaxed) {
        echo(&message);
    }
    sink.record(message);
    vk::FALSE
}

thread_local! {
    /// The driver diagnostics the thread's current driver call is measured to report
    /// ([`expecting_diagnostics`]).
    static EXPECTED_DIAGNOSTICS: RefCell<Vec<metal2vulkan::device::DriverDiagnostic>> =
        const { RefCell::new(Vec::new()) };
}

/// Run `call`, a driver call on this thread that the executing driver's facts measure to report
/// `diagnostics` (`DriverFact::pipeline_creation_diagnostics`), withholding exactly those messages
/// from the context's sink while it runs, as the runtime does (`runtime/src/driver_facts.rs`):
/// the driver reports them from the calling thread inside the call, so a message reported on
/// another thread, or after the call returns, still fails the row.
pub(super) fn expecting_diagnostics<T>(
    diagnostics: Vec<metal2vulkan::device::DriverDiagnostic>,
    call: impl FnOnce() -> T,
) -> T {
    if diagnostics.is_empty() {
        return call();
    }
    /// Clears the thread's expectation however `call` ends.
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            let _ = EXPECTED_DIAGNOSTICS.try_with(|expected| expected.borrow_mut().clear());
        }
    }
    EXPECTED_DIAGNOSTICS.with(|expected| *expected.borrow_mut() = diagnostics);
    let _clear = Clear;
    call()
}

/// Whether the message is a diagnostic the thread's current driver call is measured to report
/// ([`expecting_diagnostics`]): a warning of the validation kind whose name, number and text are
/// the diagnostic's exactly.
fn withholds(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: &vk::DebugUtilsMessengerCallbackDataEXT<'_>,
) -> bool {
    if severity != vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        || !types.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
        || data.p_message_id_name.is_null()
        || data.p_message.is_null()
    {
        return false;
    }
    // SAFETY: non-null strings of the callback data are NUL-terminated (Vulkan spec).
    let (id_name, text) = unsafe {
        (
            CStr::from_ptr(data.p_message_id_name),
            CStr::from_ptr(data.p_message),
        )
    };
    EXPECTED_DIAGNOSTICS
        .try_with(|expected| {
            expected.try_borrow().is_ok_and(|expected| {
                expected.iter().any(|d| {
                    id_name.to_bytes() == d.id_name.as_bytes()
                        && data.message_id_number == d.id
                        && text.to_bytes() == d.text.as_bytes()
                })
            })
        })
        .unwrap_or(false)
}

/// How many messages this process has echoed ([`echo_validation_messages`]).
static ECHOED: AtomicUsize = AtomicUsize::new(0);

/// Write `message` to stderr if this process has echoed fewer than a sink keeps
/// ([`MessageSink::MAX_MESSAGES`]: a device worker runs one case), and say once that the rest are
/// not. A stderr that cannot be written loses the line; a panic here would abort in the layer.
fn echo(message: &str) {
    use std::io::Write as _;
    let echoed = ECHOED.fetch_add(1, Ordering::Relaxed);
    let mut stderr = std::io::stderr();
    let _ = if echoed < MessageSink::MAX_MESSAGES {
        writeln!(stderr, "VK_LAYER_KHRONOS_validation: {message}")
    } else if echoed == MessageSink::MAX_MESSAGES {
        writeln!(
            stderr,
            "VK_LAYER_KHRONOS_validation: further validation messages are not echoed"
        )
    } else {
        Ok(())
    };
}

#[test]
fn only_errors_and_validation_warnings_fail_a_row() {
    use vk::DebugUtilsMessageSeverityFlagsEXT as S;
    use vk::DebugUtilsMessageTypeFlagsEXT as T;
    assert!(fails_row(S::ERROR, T::GENERAL));
    assert!(fails_row(S::ERROR, T::PERFORMANCE));
    assert!(fails_row(S::WARNING, T::VALIDATION));
    assert!(!fails_row(S::WARNING, T::GENERAL));
    assert!(!fails_row(S::WARNING, T::PERFORMANCE));
    assert!(!fails_row(S::INFO, T::VALIDATION));
    assert!(!fails_row(S::VERBOSE, T::VALIDATION));
    let sink = MessageSink {
        messages: Mutex::new(Vec::new()),
        unadvertised_vertex: AtomicBool::new(false),
        making: OnceLock::new(),
        unsupported_sets: AtomicBool::new(false),
    };
    for n in 0..100 {
        sink.record(format!("m{n}"));
    }
    let kept = sink.take();
    assert_eq!(kept.len(), MessageSink::MAX_MESSAGES + 1);
    assert_eq!(kept[0], "m0");
    assert!(sink.take().is_empty());
}

/// The executor's callback withholds a diagnostic the executing driver is measured to report
/// (MoltenVK 1.4.1's warning of a pipeline whose restart is dynamic) inside the call that
/// expects it alone: the same message outside it, from another thread while it runs, at another
/// severity, or with another text fails the row, as the runtime's callback relays it.
#[test]
fn a_measured_diagnostic_fails_no_row_inside_its_call_alone() {
    use metal2vulkan::device::DriverFact;
    use metal2vulkan::vk::{DynamicState as D, PrimitiveTopology as P};
    use std::ffi::CString;
    use vk::DebugUtilsMessageSeverityFlagsEXT as S;
    use vk::DebugUtilsMessageTypeFlagsEXT as T;
    let warning = DriverFact::DynamicRestartWarns
        .pipeline_creation_diagnostics(&[D::PRIMITIVE_RESTART_ENABLE], P::TRIANGLE_STRIP, &[])
        .pop()
        .expect("the warning of a dynamic restart");
    let sink = MessageSink {
        messages: Mutex::new(Vec::new()),
        unadvertised_vertex: AtomicBool::new(false),
        making: OnceLock::new(),
        unsupported_sets: AtomicBool::new(false),
    };
    let user = std::ptr::from_ref(&sink) as usize;
    let report = |severity: S, text: &str| {
        let id_name = CString::new(warning.id_name).unwrap();
        let text = CString::new(text).unwrap();
        let data = vk::DebugUtilsMessengerCallbackDataEXT::default()
            .message_id_name(&id_name)
            .message_id_number(warning.id)
            .message(&text);
        // SAFETY: the callback data and its strings live for the call, and `user` is the sink,
        // alive for the test.
        unsafe {
            record_validation_message(
                severity,
                T::VALIDATION,
                &data,
                user as *mut std::ffi::c_void,
            )
        };
    };
    report(S::WARNING, &warning.text);
    assert_eq!(sink.take().len(), 1, "outside the call");
    expecting_diagnostics(vec![warning.clone()], || {
        report(S::WARNING, &warning.text);
        assert_eq!(sink.take().len(), 0, "inside the call");
        std::thread::scope(|scope| {
            scope.spawn(|| report(S::WARNING, &warning.text));
        });
        report(S::ERROR, &warning.text);
        report(
            S::WARNING,
            "VK_ERROR_FEATURE_NOT_PRESENT: Metal does not support something else.",
        );
        assert_eq!(sink.take().len(), 3, "another thread, severity or text");
    });
    report(S::WARNING, &warning.text);
    assert_eq!(sink.take().len(), 1, "after the call");
}

/// Refuse unless `VK_LAYER_KHRONOS_validation` is installed with the three instance extensions a
/// validated context enables through it (`VK_EXT_debug_utils` for the sink,
/// `VK_EXT_validation_features` for synchronization validation, `VK_EXT_layer_settings` for
/// [`VALIDATION_LAYER_SETTINGS`]). Nothing falls back to an unvalidated run, or to one whose
/// repeated messages the layer drops.
fn require_validation_layer(entry: &Entry) -> Result<(), String> {
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .map_err(|error| format!("enumerate Vulkan instance layers: {error}"))?;
    if !layers
        .iter()
        .any(|layer| extension_name(&layer.layer_name) == VALIDATION_LAYER)
    {
        return Err(format!(
            "{} is not installed; an engine candidate or L4 run never proceeds without it",
            VALIDATION_LAYER.to_string_lossy()
        ));
    }
    let provided = unsafe { entry.enumerate_instance_extension_properties(Some(VALIDATION_LAYER)) }
        .map_err(|error| format!("enumerate validation-layer extensions: {error}"))?;
    let global = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(|error| format!("enumerate Vulkan instance extensions: {error}"))?;
    for wanted in [
        ash::ext::debug_utils::NAME,
        ash::ext::validation_features::NAME,
        ash::ext::layer_settings::NAME,
    ] {
        if !provided
            .iter()
            .chain(&global)
            .any(|extension| extension_name(&extension.extension_name) == wanted)
        {
            return Err(format!(
                "{} does not provide {}",
                VALIDATION_LAYER.to_string_lossy(),
                wanted.to_string_lossy()
            ));
        }
    }
    Ok(())
}

/// The validation layer's boolean settings a validated context sets at instance creation
/// (`VkLayerSettingsCreateInfoEXT`): `enable_message_limit` off, so the layer reports every
/// occurrence of a message. By default it reports each message ID at most
/// `duplicate_message_limit` (10) times per instance, and a context serves every row of its
/// thread, so from the eleventh row with one defect on, rows would pass with it unreported. A
/// `static`, so the setting values the chain points at outlive every instance creation.
static VALIDATION_LAYER_SETTINGS: [(&CStr, vk::Bool32); 1] = [(c"enable_message_limit", vk::FALSE)];

/// Refuse unless `VK_LAYER_KHRONOS_profiles` is installed and provides `VK_EXT_layer_settings`,
/// through which a restricted context configures it.
fn require_profiles_layer(entry: &Entry) -> Result<(), String> {
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .map_err(|error| format!("enumerate Vulkan instance layers: {error}"))?;
    if !layers
        .iter()
        .any(|layer| extension_name(&layer.layer_name) == PROFILES_LAYER)
    {
        return Err(RestrictionError::LayerMissing.to_string());
    }
    let provided = unsafe { entry.enumerate_instance_extension_properties(Some(PROFILES_LAYER)) }
        .map_err(|error| format!("enumerate profiles-layer extensions: {error}"))?;
    if !provided.iter().any(|extension| {
        extension_name(&extension.extension_name) == ash::ext::layer_settings::NAME
    }) {
        return Err(format!(
            "{} does not provide {}",
            PROFILES_LAYER.to_string_lossy(),
            ash::ext::layer_settings::NAME.to_string_lossy()
        ));
    }
    Ok(())
}

/// Hands `create` the `VkInstanceCreateInfo` of a context or of one of its probes
/// ([`create_instance`]): API `api_version` ([`context_version`] for both; the loader's for
/// [`capture_formats`]),
/// `VK_KHR_portability_enumeration` where the loader lists it (`portability`); for a validated
/// context (`messenger`), `VK_LAYER_KHRONOS_validation` with the three extensions it provides
/// ([`require_validation_layer`]), synchronization validation, [`VALIDATION_LAYER_SETTINGS`], and
/// `messenger` chained so instance creation and destruction report into the sink too; and with
/// `profiles`, `VK_LAYER_KHRONOS_profiles` below the validation layer (the first enabled layer is
/// the nearest the application, so the validation layer checks what the profiles layer reports)
/// configured by its settings. Every chained structure lives until `create` returns.
fn with_instance_create_info<R>(
    api_version: u32,
    portability: bool,
    messenger: Option<vk::DebugUtilsMessengerCreateInfoEXT<'_>>,
    profiles: Option<&ProfilesSettings>,
    create: impl FnOnce(&vk::InstanceCreateInfo<'_>) -> R,
) -> R {
    let mut extension_names = Vec::new();
    if portability {
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
    }
    let mut layer_names = Vec::new();
    if messenger.is_some() {
        layer_names.push(VALIDATION_LAYER.as_ptr());
        extension_names.extend([
            ash::ext::debug_utils::NAME.as_ptr(),
            ash::ext::validation_features::NAME.as_ptr(),
        ]);
    }
    if profiles.is_some() {
        layer_names.push(PROFILES_LAYER.as_ptr());
    }
    if messenger.is_some() || profiles.is_some() {
        extension_names.push(ash::ext::layer_settings::NAME.as_ptr());
    }
    // Core validation is on by default; synchronization validation is enabled explicitly.
    let enabled_validation = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
    let mut validation_features =
        vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled_validation);
    let mut settings: Vec<vk::LayerSettingEXT<'_>> = Vec::new();
    if messenger.is_some() {
        settings.extend(VALIDATION_LAYER_SETTINGS.iter().map(|(name, value)| {
            // `LayerSettingEXT::values` counts bytes, not values: one `VkBool32` is one value.
            let mut setting = vk::LayerSettingEXT::default()
                .layer_name(VALIDATION_LAYER)
                .setting_name(name)
                .ty(vk::LayerSettingTypeEXT::BOOL32);
            setting.value_count = 1;
            setting.p_values = std::ptr::from_ref(value).cast();
            setting
        }));
    }
    if let Some(profiles) = profiles {
        settings.extend(profiles.settings());
    }
    let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
    let mut instance_messenger = messenger;
    let application = vk::ApplicationInfo::default()
        .application_name(c"metal2vulkan-validation")
        .application_version(1)
        .engine_name(c"metal2vulkan-validation")
        .engine_version(1)
        .api_version(api_version);
    let mut info = vk::InstanceCreateInfo::default()
        .application_info(&application)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_names)
        .flags(if portability {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        });
    if !settings.is_empty() {
        info = info.push_next(&mut layer_settings);
    }
    if let Some(messenger) = instance_messenger.as_mut() {
        info = info
            .push_next(&mut validation_features)
            .push_next(messenger);
    }
    create(&info)
}

/// A validated context's instance reports every occurrence of a message: it enables
/// `VK_EXT_layer_settings` beside the validation layer and chains `enable_message_limit = false`
/// for that layer. An instance without a messenger (the formats capture's, [`capture_formats`])
/// enables no layer and chains nothing. A restricted
/// context stacks the profiles layer below the validation layer and configures it with the
/// committed file; the native probe enables the profiles layer alone with its emulation off.
#[test]
fn a_validated_instance_reports_every_repeated_message() {
    /// `(extension names, layer names, (layer, setting, type, values) of every chained setting)`,
    /// a boolean value spelled as its number.
    type Seen = (
        Vec<String>,
        Vec<String>,
        Vec<(String, String, i32, Vec<String>)>,
    );
    fn seen(info: &vk::InstanceCreateInfo<'_>) -> Seen {
        let text = |pointer: *const std::ffi::c_char| {
            // SAFETY: every name in the create info is a NUL-terminated static string.
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        let list = |names: *const *const std::ffi::c_char, count: u32| -> Vec<String> {
            if count == 0 {
                return Vec::new();
            }
            // SAFETY: the create info's arrays hold `count` live pointers.
            unsafe { std::slice::from_raw_parts(names, count as usize) }
                .iter()
                .map(|&name| text(name))
                .collect()
        };
        let mut settings = Vec::new();
        let mut next = info.p_next.cast::<vk::BaseInStructure<'_>>();
        while !next.is_null() {
            // SAFETY: every chained structure is live while `create` runs.
            let base = unsafe { &*next };
            if base.s_type == vk::StructureType::LAYER_SETTINGS_CREATE_INFO_EXT {
                // SAFETY: the structure type names the structure.
                let chained = unsafe { &*next.cast::<vk::LayerSettingsCreateInfoEXT<'_>>() };
                // SAFETY: `setting_count` settings, each with `value_count` 32-bit booleans.
                let all = unsafe {
                    std::slice::from_raw_parts(chained.p_settings, chained.setting_count as usize)
                };
                for setting in all {
                    let count = setting.value_count as usize;
                    let values = if setting.ty == vk::LayerSettingTypeEXT::STRING {
                        // SAFETY: `value_count` NUL-terminated strings.
                        unsafe {
                            std::slice::from_raw_parts(
                                setting.p_values.cast::<*const std::ffi::c_char>(),
                                count,
                            )
                        }
                        .iter()
                        .map(|&value| text(value))
                        .collect()
                    } else {
                        // SAFETY: `value_count` 32-bit booleans.
                        unsafe {
                            std::slice::from_raw_parts(setting.p_values.cast::<vk::Bool32>(), count)
                        }
                        .iter()
                        .map(u32::to_string)
                        .collect()
                    };
                    settings.push((
                        text(setting.p_layer_name),
                        text(setting.p_setting_name),
                        setting.ty.as_raw(),
                        values,
                    ));
                }
            }
            next = base.p_next;
        }
        (
            list(
                info.pp_enabled_extension_names,
                info.enabled_extension_count,
            ),
            list(info.pp_enabled_layer_names, info.enabled_layer_count),
            settings,
        )
    }
    let messenger = vk::DebugUtilsMessengerCreateInfoEXT::default();
    let (extensions, layers, settings) = with_instance_create_info(
        packed(context_version(None)),
        false,
        Some(messenger),
        None,
        seen,
    );
    assert_eq!(
        extensions,
        [
            "VK_EXT_debug_utils",
            "VK_EXT_validation_features",
            "VK_EXT_layer_settings"
        ]
    );
    assert_eq!(layers, ["VK_LAYER_KHRONOS_validation"]);
    let message_limit = (
        "VK_LAYER_KHRONOS_validation".to_owned(),
        "enable_message_limit".to_owned(),
        vk::LayerSettingTypeEXT::BOOL32.as_raw(),
        vec!["0".to_owned()],
    );
    assert_eq!(settings, std::slice::from_ref(&message_limit));
    let (extensions, layers, settings) =
        with_instance_create_info(packed(context_version(None)), true, None, None, seen);
    assert_eq!(extensions, ["VK_KHR_portability_enumeration"]);
    assert!(layers.is_empty());
    assert!(settings.is_empty());

    // Restricted: the profiles layer below the validation layer, configured with the file.
    let restriction = ProfileRestriction::committed("vk12-floor").unwrap();
    let hidden: BTreeSet<String> = ["VK_EXT_shader_viewport_index_layer".to_owned()].into();
    let restrict = ProfilesSettings::new(ProfilesLayer::Restrict(&restriction, &hidden)).unwrap();
    let (extensions, layers, settings) = with_instance_create_info(
        packed(context_version(Some(&restriction))),
        false,
        Some(messenger),
        Some(&restrict),
        seen,
    );
    assert_eq!(
        extensions,
        [
            "VK_EXT_debug_utils",
            "VK_EXT_validation_features",
            "VK_EXT_layer_settings"
        ]
    );
    assert_eq!(
        layers,
        ["VK_LAYER_KHRONOS_validation", "VK_LAYER_KHRONOS_profiles"]
    );
    let profiles = |name: &str, ty: vk::LayerSettingTypeEXT, values: &[&str]| {
        (
            "VK_LAYER_KHRONOS_profiles".to_owned(),
            name.to_owned(),
            ty.as_raw(),
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>(),
        )
    };
    let (bool32, string) = (
        vk::LayerSettingTypeEXT::BOOL32,
        vk::LayerSettingTypeEXT::STRING,
    );
    assert_eq!(
        settings,
        [
            message_limit,
            profiles("profile_emulation", bool32, &["1"]),
            profiles(
                "profile_file",
                string,
                &[&restriction.file.to_string_lossy()]
            ),
            profiles("profile_name", string, &["VP_M2V_vk12_floor"]),
            profiles(
                "simulate_capabilities",
                string,
                &[
                    "SIMULATE_API_VERSION_BIT",
                    "SIMULATE_FEATURES_BIT",
                    "SIMULATE_PROPERTIES_BIT",
                    "SIMULATE_EXTENSIONS_BIT",
                    "SIMULATE_FORMATS_BIT",
                ]
            ),
            profiles(
                "default_feature_values",
                string,
                &["DEFAULT_FEATURE_VALUES_FALSE"]
            ),
            profiles("debug_fail_on_error", bool32, &["0"]),
            profiles("debug_reports", string, &[]),
            profiles("emulate_portability", bool32, &["0"]),
            profiles(
                "exclude_device_extensions",
                string,
                &["VK_EXT_shader_viewport_index_layer"]
            ),
        ]
    );
    // Hiding nothing passes no exclusion.
    let none = BTreeSet::new();
    let presenting = ProfilesSettings::new(ProfilesLayer::Restrict(&restriction, &none)).unwrap();
    let (_, _, settings) = with_instance_create_info(
        packed(context_version(None)),
        false,
        None,
        Some(&presenting),
        seen,
    );
    assert!(settings
        .iter()
        .all(|(_, name, _, _)| name != "exclude_device_extensions"));

    // The native probe: the profiles layer alone, its emulation off.
    let probe = ProfilesSettings::new(ProfilesLayer::PassThrough).unwrap();
    let (extensions, layers, settings) = with_instance_create_info(
        packed(context_version(None)),
        false,
        None,
        Some(&probe),
        seen,
    );
    assert_eq!(extensions, ["VK_EXT_layer_settings"]);
    assert_eq!(layers, ["VK_LAYER_KHRONOS_profiles"]);
    assert_eq!(
        settings,
        [
            profiles("profile_emulation", bool32, &["0"]),
            profiles("emulate_portability", bool32, &["0"]),
        ]
    );
}

/// The API version of the instance of a context restricted to `restriction`, or of none, and of
/// its probes: Vulkan 1.2 unrestricted, and restricted the profile's own version with 1.2 its floor
/// ([`ProfileRestriction::context_version`]), so the context reads the profile, and the device, at
/// the version the product reads the profile at.
fn context_version(restriction: Option<&ProfileRestriction>) -> ApiVersion {
    restriction.map_or(ApiVersion::V1_2, ProfileRestriction::context_version)
}

/// `version` as a `VkApplicationInfo::apiVersion`.
fn packed(version: ApiVersion) -> u32 {
    vk::make_api_version(0, version.major, version.minor, 0)
}

/// Create a Vulkan instance at `api_version` from [`with_instance_create_info`]'s create info: the
/// only instance creation in this crate. It runs only inside [`VulkanContext::new`] (the context's
/// instance and its two probes, [`native_description`] and [`presented_extensions`]), which runs
/// only after [`with_slot`] has checked the hang tripwire, and inside [`capture_formats`], after
/// its own check.
fn create_instance(
    entry: &Entry,
    api_version: u32,
    portability: bool,
    messenger: Option<vk::DebugUtilsMessengerCreateInfoEXT<'_>>,
    profiles: Option<&ProfilesSettings>,
) -> Result<Instance, vk::Result> {
    with_instance_create_info(api_version, portability, messenger, profiles, |info| {
        // SAFETY: every pointer in `info` outlives the call. A validated context's layer provides
        // the extensions it enables through it (`require_validation_layer`), and so does the
        // profiles layer of a restricted context or a probe (`require_profiles_layer`).
        let instance = unsafe { entry.create_instance(info, None) };
        // A layer registered an exit handler as the call began: the runtime's, where a runtime
        // device lives in this process (a lane's), registers again after it.
        metal2vulkan_runtime::program_instance_made();
        instance
    })
}

/// The description of the device `device_type` selects, as the device is: read through an
/// instance whose profiles layer has its emulation off, so neither this process's restriction
/// nor a profile an environment names (whose settings take precedence over the application's,
/// but which is loaded only with emulation on) is applied, queried at API `version` (the
/// context's, [`context_version`]) and destroyed.
fn native_description(
    entry: &Entry,
    device_type: DeviceType,
    portability: bool,
    version: ApiVersion,
) -> Result<DeviceDescription, String> {
    native_probe(entry, device_type, portability, version).map(|(description, _)| description)
}

/// [`native_description`], and the device's identity as a context's environment records it (its
/// name, IDs, driver version, API version, and driver), read through the same probe instance.
fn native_probe(
    entry: &Entry,
    device_type: DeviceType,
    portability: bool,
    version: ApiVersion,
) -> Result<(DeviceDescription, serde_json::Value), String> {
    let probe = ProfilesSettings::new(ProfilesLayer::PassThrough)?;
    let instance = create_instance(entry, packed(version), portability, None, Some(&probe))
        .map_err(|error| format!("create the native probe's Vulkan instance: {error}"))?;
    let described = select_enumerated(&instance, device_type).and_then(|(physical, _)| {
        // SAFETY: `physical` was enumerated from `instance`, created at API `version`.
        let description = unsafe { DeviceDescription::query(&instance, physical, version) }
            .map_err(|error| format!("query the native device: {error}"))?;
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: `physical` belongs to `instance`, created at API 1.2 or later.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let device = &properties.properties;
        let identity = serde_json::json!({
            "device": extension_name(&device.device_name).to_string_lossy(),
            "device_id": device.device_id,
            "vendor_id": device.vendor_id,
            "driver_version": device.driver_version,
            "api_version": device.api_version,
            "driver_id": driver.driver_id.as_raw(),
            "driver_name": extension_name(&driver.driver_name).to_string_lossy(),
            "driver_info": extension_name(&driver.driver_info).to_string_lossy(),
            "architecture": std::env::consts::ARCH,
            "os": std::env::consts::OS,
        });
        Ok((description, identity))
    });
    // SAFETY: nothing created from the instance outlives it.
    unsafe { instance.destroy_instance(None) };
    described
}

/// A restriction the device cannot take: this process is restricted to a profile that states
/// capabilities the device `device_type` selects, as it is, lacks. What a typed refusal of it
/// records: the profile and the members, the device as the native probe describes it, and its
/// identity as a context's environment records it.
pub(super) struct RefusedRestriction {
    pub(super) restriction: &'static ProfileRestriction,
    pub(super) gaps: Vec<String>,
    pub(super) native: DeviceDescription,
    pub(super) identity: serde_json::Value,
}

/// The restricted context's first check ([`profiles::check_within`]), made before any context
/// opens: `Some` when this process is restricted to a profile the device lacks capabilities of,
/// `None` when it is unrestricted or the device takes the profile. A context makes the same check
/// and fails without a record; an engine executor asks here first, so that the refusal is
/// typed and recorded with the device's facts. Every other failure of the check is an error, as
/// in the context.
pub(super) fn refused_restriction(
    device_type: DeviceType,
) -> Result<Option<RefusedRestriction>, String> {
    let Some(restriction) = profiles::current_restriction() else {
        return Ok(None);
    };
    if device_type == DeviceType::Gpu {
        crate::gpu::tripwire::refuse_if_tripped()?;
    }
    let entry = load_entry()?;
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(|error| format!("enumerate Vulkan instance extensions: {error}"))?;
    let portability = available.iter().any(|extension| {
        extension_name(&extension.extension_name) == ash::khr::portability_enumeration::NAME
    });
    require_profiles_layer(&entry)?;
    let version = restriction.context_version();
    let (native, identity) = native_probe(&entry, device_type, portability, version)?;
    match profiles::check_within(restriction, &native, version) {
        Ok(()) => Ok(None),
        Err(RestrictionError::Exceeds { gaps, .. }) => Ok(Some(RefusedRestriction {
            restriction,
            gaps,
            native,
            identity,
        })),
        Err(error) => Err(error.to_string()),
    }
}

/// This process's restriction as the runtime path opens its device under it: the restricted
/// context's steps before its device, on the device `device_type` selects.
pub(crate) enum Restricting {
    /// The device takes P: what the runtime device's options and its second check need.
    Takes(Box<RuntimeRestriction>),
    /// The device lacks capabilities P states: the typed refusal, recorded as the executor's
    /// context records it ([`restriction_refusal`](super::compute::restriction_refusal)).
    Refused(Box<crate::candidate::engine::RunAttempt>),
}

/// What a runtime device restricted to P is made with: P, the device as it is (read through the
/// layer with its emulation off, which the second check compares the layer's kept members with)
/// and its identity, and the extensions the layer presents for P that P does not list, which it
/// hides.
pub(crate) struct RuntimeRestriction {
    restriction: &'static ProfileRestriction,
    native: DeviceDescription,
    identity: serde_json::Value,
    hidden: BTreeSet<String>,
}

/// The restricted context's steps before its device, for the runtime path's `executor`: `None`
/// when this process runs unrestricted; else the first check ([`profiles::check_within`])
/// decides between a typed refusal and a restriction the device takes, whose hidden extensions
/// are read as a context reads them. Every other failure is an error, as in the context.
pub(crate) fn runtime_restriction(
    device_type: DeviceType,
    executor: &str,
) -> Result<Option<Restricting>, String> {
    let Some(restriction) = profiles::current_restriction() else {
        return Ok(None);
    };
    restricting(restriction, device_type, executor).map(Some)
}

/// What [`lane_restriction`] finds for a lane test's committed profile.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) enum LaneRestriction {
    /// `VK_LAYER_KHRONOS_profiles` is not installed.
    NoLayer,
    /// The device lacks capabilities the profile states (the first check's refusal, which a
    /// restricted context records as a typed refusal): the device's name and the members it
    /// lacks.
    Lacks { device: String, gaps: Vec<String> },
    /// The device takes the profile: what the runtime device's options and its second check need.
    Takes(Box<RuntimeRestriction>),
}

/// The runtime path's restriction to committed profile `stem` for a lane test, whatever this
/// process's own restriction is (a test does not restrict its process), on the device
/// `device_type` selects. Every failure other than the first check's refusal is an error.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn lane_restriction(
    stem: &str,
    device_type: DeviceType,
) -> Result<LaneRestriction, String> {
    if !profiles_layer_installed()? {
        return Ok(LaneRestriction::NoLayer);
    }
    // A restriction lives as long as the process, as this process's own would.
    let restriction: &'static ProfileRestriction =
        Box::leak(Box::new(profiles::presentable(stem)?));
    Ok(match first_check(restriction, device_type)? {
        FirstCheck::Takes(restriction) => LaneRestriction::Takes(restriction),
        FirstCheck::Lacks(refused) => LaneRestriction::Lacks {
            device: refused.identity["device"]
                .as_str()
                .unwrap_or("the device")
                .to_owned(),
            gaps: refused.gaps,
        },
    })
}

/// Whether `VK_LAYER_KHRONOS_profiles` is installed, for a lane test that restricts its device
/// through it.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn profiles_layer_installed() -> Result<bool, String> {
    let entry = load_entry()?;
    // SAFETY: a global command of a loaded entry.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .map_err(|error| format!("enumerate Vulkan instance layers: {error}"))?;
    Ok(layers
        .iter()
        .any(|layer| extension_name(&layer.layer_name) == PROFILES_LAYER))
}

/// [`runtime_restriction`]'s steps for `restriction`: the first check decides between a typed
/// refusal, recorded as `executor`'s context records it, and a restriction the device takes.
fn restricting(
    restriction: &'static ProfileRestriction,
    device_type: DeviceType,
    executor: &str,
) -> Result<Restricting, String> {
    Ok(match first_check(restriction, device_type)? {
        FirstCheck::Takes(restriction) => Restricting::Takes(restriction),
        FirstCheck::Lacks(refused) => Restricting::Refused(Box::new(
            super::compute::restriction_refusal(&refused, executor),
        )),
    })
}

/// The first check's ([`profiles::check_within`]) verdict on a restriction.
enum FirstCheck {
    /// The device takes P.
    Takes(Box<RuntimeRestriction>),
    /// The device lacks capabilities P states: what a typed refusal of it records.
    Lacks(Box<RefusedRestriction>),
}

/// The restricted context's steps before its device for `restriction`, on the device
/// `device_type` selects: the first check ([`profiles::check_within`]) decides between a refusal
/// and a restriction the device takes, whose hidden extensions are read as a context reads them.
/// Every other failure is an error, as in the context.
fn first_check(
    restriction: &'static ProfileRestriction,
    device_type: DeviceType,
) -> Result<FirstCheck, String> {
    if device_type == DeviceType::Gpu {
        crate::gpu::tripwire::refuse_if_tripped()?;
    }
    let entry = load_entry()?;
    // SAFETY: a global command of a loaded entry.
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(|error| format!("enumerate Vulkan instance extensions: {error}"))?;
    let portability = available.iter().any(|extension| {
        extension_name(&extension.extension_name) == ash::khr::portability_enumeration::NAME
    });
    require_profiles_layer(&entry)?;
    let version = restriction.context_version();
    let (native, identity) = native_probe(&entry, device_type, portability, version)?;
    match profiles::check_within(restriction, &native, version) {
        Ok(()) => {}
        Err(RestrictionError::Exceeds { gaps, .. }) => {
            return Ok(FirstCheck::Lacks(Box::new(RefusedRestriction {
                restriction,
                gaps,
                native,
                identity,
            })));
        }
        Err(error) => return Err(error.to_string()),
    }
    let presented = presented_extensions(&entry, device_type, portability, restriction)?;
    let hidden = restriction.hidden(presented.iter().map(String::as_str));
    Ok(FirstCheck::Takes(Box::new(RuntimeRestriction {
        restriction,
        native,
        identity,
        hidden,
    })))
}

impl RuntimeRestriction {
    /// The committed profile's stem.
    pub(crate) fn stem(&self) -> &str {
        &self.restriction.stem
    }

    /// `options` restricted to P as a restricted context's instance is: the profiles layer below
    /// the validation layer (which `options` stacks first), configured with P and the hidden
    /// extensions, and the instance at P's version ([`ProfileRestriction::context_version`]).
    pub(crate) fn restrict(
        &self,
        options: &mut metal2vulkan_runtime::DeviceOptions,
    ) -> Result<(), String> {
        let settings =
            ProfilesSettings::new(ProfilesLayer::Restrict(self.restriction, &self.hidden))?;
        options.layers.push(PROFILES_LAYER.to_owned());
        options.layer_settings.extend(settings.runtime_settings());
        options.instance_api_version = Some(self.restriction.context_version());
        Ok(())
    }

    /// What `executor`'s typed refusal of the restricted device itself, before it opened (a
    /// device the runtime refuses at creation), records of the device as it is, with the
    /// restriction it was made under (`compute::native_refused_on`, as a refusal of the
    /// restriction records it).
    pub(crate) fn refused_on(
        &self,
        executor: &str,
    ) -> Result<crate::candidate::engine::RefusedOn, String> {
        let layer = serde_json::json!({
            "profile": self.restriction.stem,
            "profile_name": self.restriction.profile_name,
            "profile_sha256": self.restriction.sha256,
            "hidden_extensions": self.hidden,
        });
        super::compute::native_refused_on(&self.native, &self.identity, executor, layer)
    }

    /// The restricted context's second check ([`profiles::check_restricted`]) on `device`,
    /// opened with [`RuntimeRestriction::restrict`]'s options, read through its raw handles: the
    /// layer active, its kept members the native device's, and every other member P's.
    pub(crate) fn check(
        &self,
        device: &metal2vulkan_runtime::Device,
    ) -> Result<crate::candidate::RestrictedDevice, String> {
        let version = self.restriction.context_version();
        // SAFETY: only physical-device queries of the device's own instance, created at P's
        // version; nothing is created, destroyed, or submitted, and the handles are not kept.
        let (tools, queried) = unsafe {
            let handles = device.raw();
            let tools =
                profiles::active_tools(handles.entry, handles.instance, handles.physical_device)?;
            let queried =
                DeviceDescription::query(handles.instance, handles.physical_device, version)
                    .map_err(|error| format!("query the restricted device: {error}"))?;
            (tools, queried)
        };
        let translated =
            profiles::check_restricted(self.restriction, &self.native, &tools, &queried, version)
                .map_err(|error| error.to_string())?;
        let record = serde_json::json!({
            "tool": profiles::profiles_tool(&tools),
            "profile": self.restriction.stem,
            "profile_name": self.restriction.profile_name,
            "profile_sha256": self.restriction.sha256,
            "hidden_extensions": self.hidden,
        });
        let kept = profiles::differences(&translated, &queried)
            .into_iter()
            .map(|difference| difference.path)
            .collect();
        Ok(crate::candidate::RestrictedDevice {
            translated,
            kept,
            record,
        })
    }
}

/// The device extensions the device `device_type` selects presents under `restriction`, read
/// through an instance whose profiles layer restricts it and hides nothing, then destroyed: P's
/// extensions and the ones the layer adds for P's version.
fn presented_extensions(
    entry: &Entry,
    device_type: DeviceType,
    portability: bool,
    restriction: &ProfileRestriction,
) -> Result<Vec<String>, String> {
    let hidden = BTreeSet::new();
    let probe = ProfilesSettings::new(ProfilesLayer::Restrict(restriction, &hidden))?;
    let version = packed(restriction.context_version());
    let instance = create_instance(entry, version, portability, None, Some(&probe))
        .map_err(|error| format!("create the presentation probe's Vulkan instance: {error}"))?;
    let presented = select_enumerated(&instance, device_type).and_then(|(physical, _)| {
        // SAFETY: `physical` was enumerated from `instance`.
        unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(|error| format!("enumerate the presented device extensions: {error}"))
    });
    // SAFETY: nothing created from the instance outlives it.
    unsafe { instance.destroy_instance(None) };
    Ok(presented?
        .iter()
        .map(|extension| {
            extension_name(&extension.extension_name)
                .to_string_lossy()
                .into_owned()
        })
        .collect())
}

impl VulkanContext {
    /// The runtime's driver facts of the driver that executes: its queried description's (the
    /// device as it is, under a restriction too); none where the query failed.
    pub(super) fn executing_driver_facts(&self) -> BTreeSet<metal2vulkan::device::DriverFact> {
        self.queried_description
            .as_ref()
            .map(DeviceDescription::driver_facts)
            .unwrap_or_default()
    }

    /// Whether the device replay's generated-commands executions are preprocessed apart, as the
    /// runtime's are on the driver that executes (`generated_commands::preprocess_apart` of its
    /// queried description, the device as it is, under a restriction too); also where the query
    /// failed, since preprocessing apart runs right on every driver.
    pub(super) fn preprocesses_generated_commands_apart(&self) -> bool {
        match &self.queried_description {
            Ok(description) => {
                metal2vulkan_runtime::generated_commands::preprocess_apart(description)
            }
            Err(_) => true,
        }
    }

    /// Open a device of `device_type` for `purpose`: the only Vulkan device creation in this crate,
    /// and with its probes the only caller of [`create_instance`].
    fn new(device_type: DeviceType, purpose: Purpose) -> Result<Self, String> {
        let entry = load_entry()?;
        let available = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|error| format!("enumerate Vulkan instance extensions: {error}"))?;
        let portability = available.iter().any(|extension| {
            extension_name(&extension.extension_name) == ash::khr::portability_enumeration::NAME
        });
        require_validation_layer(&entry)?;
        // A restricted context reads the device as it is first, through the profiles layer with its
        // emulation off, and goes on only if the device has P; then it reads which extensions the
        // layer presents for P, to hide the ones P does not list.
        let restriction = profiles::current_restriction();
        let version = context_version(restriction);
        let (native, hidden) = match restriction {
            Some(restriction) => {
                require_profiles_layer(&entry)?;
                let native = native_description(&entry, device_type, portability, version)?;
                profiles::check_within(restriction, &native, version)
                    .map_err(|error| error.to_string())?;
                let presented =
                    presented_extensions(&entry, device_type, portability, restriction)?;
                let hidden = restriction.hidden(presented.iter().map(String::as_str));
                (Some(native), hidden)
            }
            None => (None, BTreeSet::new()),
        };
        let profiles_settings = restriction
            .map(|restriction| ProfilesSettings::new(ProfilesLayer::Restrict(restriction, &hidden)))
            .transpose()?;
        // Leaked on purpose: one small sink per context, and a context lives for its thread, so
        // the messenger's user pointer can never dangle, whatever path creation takes.
        let messages: &'static MessageSink = Box::leak(Box::new(MessageSink {
            messages: Mutex::new(Vec::new()),
            unadvertised_vertex: AtomicBool::new(false),
            making: OnceLock::new(),
            unsupported_sets: AtomicBool::new(false),
        }));
        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(record_validation_message))
            .user_data(std::ptr::from_ref(messages).cast_mut().cast());
        let instance = create_instance(
            &entry,
            packed(version),
            portability,
            Some(messenger_info),
            profiles_settings.as_ref(),
        )
        .map_err(|error| format!("create Vulkan instance: {error}"))?;
        // The device, and the tools active on it, which every row records. A context a profile
        // restricts without a request still opens and runs; the store refuses its rows
        // (`CandidateObservation::validate_content`), so they are never recorded as native.
        let selected = select_enumerated(&instance, device_type).and_then(|(physical, family)| {
            // SAFETY: `physical` was enumerated from `instance`, which `entry` created.
            let tools = unsafe { profiles::active_tools(&entry, &instance, physical) }?;
            Ok((physical, family, tools))
        });
        let (physical, queue_family, tools) = match selected {
            Ok(selected) => selected,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error);
            }
        };
        // The messenger for everything after instance creation. Created before the device, so
        // device creation reports into the sink too.
        let messenger = {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            // SAFETY: `messenger_info` is live for the call; its user pointer is the leaked sink.
            let handle = unsafe { loader.create_debug_utils_messenger(&messenger_info, None) }
                .map_err(|error| format!("create the validation messenger: {error}"))?;
            (loader, handle)
        };
        let device_extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(|error| format!("enumerate Vulkan device extensions: {error}"))?;
        // The structures core for the device are its version's, at most the instance's (the
        // product's query reads the same minimum).
        // SAFETY: `physical` belongs to `instance`.
        let device_api = unsafe { instance.get_physical_device_properties(physical) }.api_version;
        let device_version = ApiVersion {
            major: vk::api_version_major(device_api),
            minor: vk::api_version_minor(device_api),
        };
        let available = Available::at(
            version.min(device_version),
            device_extensions
                .iter()
                .map(|extension| extension_name(&extension.extension_name).to_owned()),
        );
        // SAFETY: `physical` was enumerated from `instance`, created at API `version`, and
        // `available` is what the device lists, at the version the instance and device share.
        let supported = unsafe { FeatureChain::query(&instance, physical, &available) };
        // What the device supports, in the product's vocabulary: the engine's list is the
        // product's recommendation for it, and the opened description is it restricted to what
        // was enabled.
        // SAFETY: `physical` was enumerated from `instance`, which was created with API
        // `version`, the version passed here; the query only calls physical-device entry points.
        let mut description = unsafe { DeviceDescription::query(&instance, physical, version) }
            .map_err(|error| format!("query the device: {error}"));
        // The device as it is, which a venue's run holds to the venue's profile.
        let queried_description = description.clone();
        // The device that executes, from the queried identity: under a restriction the layer
        // keeps it from the device, and `check_restricted` proves it the native device's.
        let execution_class = description
            .as_ref()
            .ok()
            .and_then(ExecutionClass::of_description);
        // A restricted context runs only where, through the layer, the device reports P: checked
        // before the device is created. It translates for P (`check_restricted`).
        if let (Some(restriction), Some(native)) = (restriction, &native) {
            let checked = description
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|queried| {
                    profiles::check_restricted(restriction, native, &tools, queried, version)
                        .map_err(|error| error.to_string())
                });
            match checked {
                Ok(profile) => description = Ok(profile),
                Err(error) => {
                    // SAFETY: nothing else was created from the instance.
                    unsafe {
                        let (loader, handle) = &messenger;
                        loader.destroy_debug_utils_messenger(*handle, None);
                        instance.destroy_instance(None);
                    }
                    return Err(error);
                }
            }
        }
        let leg = requested_leg();
        let mut enables = enables::engine_list(
            enables::old_engine_list(&supported, &available),
            description.as_ref().map_err(Clone::clone)?,
            &supported,
            &available,
            leg,
        )
        .map_err(|error| format!("the engine's enable list: {error}"))?;
        // The device replay's generated-commands path: an engine context enables
        // `VK_EXT_device_generated_commands` where its device supports it, unless this process runs
        // without it.
        let generated_commands = crate::candidate::requested_generated_commands()
            && enables::enable_generated_commands(&mut enables, &supported, &available);
        // The runtime probe's own context enables what it measures.
        let runtime_probe = (purpose == Purpose::RuntimeProbe)
            .then(|| enables::enable_runtime_probe(&mut enables, &available));
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let extension_names = enables.extension_names();
        let device = enables
            .chain
            .with_create_info(
                &queue_info,
                &extension_names,
                &enables.linked,
                enables.version,
                |info| {
                    // SAFETY: `physical` belongs to `instance`; every name is one the device
                    // lists (with what it requires), every linked structure is core at the
                    // context's version or exposed by one of them (`Enables`), and the chained
                    // structures live for the call.
                    unsafe { instance.create_device(physical, info, None) }
                },
            )
            .map_err(|error| format!("create Vulkan device: {error}"))?;
        // SAFETY: the loader just returned the handle; the context destroys it last.
        let loader_word = unsafe { LoaderWord::of_device(device.handle()) };
        // The loader's `vkGetDeviceQueue` and `vkGetDeviceProcAddr` (which loads an extension's
        // commands) read the device's loader word, which nothing has reported on since it was
        // recorded ([`LoaderWord`]).
        #[expect(
            clippy::disallowed_methods,
            reason = "the loader word was recorded from the handle just returned"
        )]
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let extended_dynamic_state = match enables::dynamic_state_commands(
            enables.version,
            enables
                .extensions
                .contains(ash::ext::extended_dynamic_state::NAME),
        ) {
            Some(enables::DynamicStateCommands::Core) => Some(ExtendedDynamicState::core(&device)),
            Some(enables::DynamicStateCommands::Extension) => {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the loader word was recorded from the handle just returned"
                )]
                let extension = ash::ext::extended_dynamic_state::Device::new(&instance, &device);
                let second = if enables
                    .extensions
                    .contains(ash::ext::extended_dynamic_state2::NAME)
                {
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "the loader word was recorded from the handle just returned"
                    )]
                    let second = ash::ext::extended_dynamic_state2::Device::new(&instance, &device);
                    Some(second)
                } else {
                    None
                };
                Some(ExtendedDynamicState::extension(&extension, second.as_ref()))
            }
            None => None,
        };
        let mut driver_properties = vk::PhysicalDeviceDriverProperties::default();
        let mut properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut driver_properties);
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let limits = properties.properties.limits;
        let max_sampler_anisotropy = limits.max_sampler_anisotropy;
        let driver = DriverIdentity {
            driver_id: driver_properties.driver_id.as_raw(),
            driver_name: extension_name(&driver_properties.driver_name)
                .to_string_lossy()
                .into_owned(),
            driver_info: extension_name(&driver_properties.driver_info)
                .to_string_lossy()
                .into_owned(),
        };
        let enabled = enables::enabled_set(&enables);
        let opened_description = description
            .and_then(|description| crate::acceptance::opened_description(description, &enabled))
            .map(|mut description| {
                // The description records what the context enabled: the generated-commands
                // facts only where it enabled the extension.
                if !generated_commands {
                    description.generated_commands = None;
                }
                description
            });
        if let Ok(description) = &opened_description {
            let blendless = description.blendless_color_formats();
            let making = (description.driver_facts().into_iter())
                .flat_map(|fact| {
                    fact.pipeline_creation_diagnostics(
                        &[],
                        metal2vulkan::vk::PrimitiveTopology::TRIANGLE_LIST,
                        &blendless,
                    )
                })
                .collect();
            let _ = messages.making.set(making);
            messages.unsupported_sets.store(
                metal2vulkan_runtime::makes_layouts_past_support(description),
                Ordering::Relaxed,
            );
        }
        let opened = opened_description
            .clone()
            .and_then(crate::acceptance::derived_profile);
        if opened.as_ref().is_ok_and(|profile| {
            profile
                .takes_driver_deviation(metal2vulkan::error::DriverDeviation::MetalVertexSimdgroups)
        }) {
            messages.unadvertised_vertex.store(true, Ordering::Relaxed);
        }
        let on = |value: vk::Bool32| value == vk::TRUE;
        // The portability subset's `multisampleArrayImage`, recorded in the row environment
        // (false where the device lists no subset).
        let multisample_array_image = available.has(ash::khr::portability_subset::NAME)
            && on(supported.portability_subset.multisample_array_image);
        // SAFETY: `physical` belongs to the live instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let sparse_images = SparseImages {
            features: enables::executor_features(&enables),
            queue: families
                .get(queue_family as usize)
                .is_some_and(|family| family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)),
        };
        Ok(Self {
            backend: SubmissionBackend::vulkan(device_type),
            _entry: entry,
            instance,
            device,
            extended_dynamic_state,
            physical,
            queue_family,
            queue,
            sampler_anisotropy: on(enables.chain.core.sampler_anisotropy),
            shader_subgroup_extended_types: on(supported.v12.shader_subgroup_extended_types),
            multisample_array_image,
            max_sampler_anisotropy,
            storage_buffer_alignment: limits.min_storage_buffer_offset_alignment,
            max_storage_buffer_range: u64::from(limits.max_storage_buffer_range),
            enabled,
            opened,
            opened_description,
            queried_description,
            driver,
            execution_class,
            tools,
            restriction,
            hidden,
            leg,
            unmet_robustness: enables.unmet_robustness,
            sparse_images,
            generated_commands,
            runtime_probe,
            messenger,
            messages,
            loader_word,
        })
    }

    /// Allocate command buffers through the loader's `vkAllocateCommandBuffers`, which reads the
    /// device's loader word ([`LoaderWord`]): the word is put back first.
    #[expect(
        clippy::disallowed_methods,
        reason = "the one call site, after the loader word is put back"
    )]
    pub(super) fn allocate_command_buffers(
        &self,
        info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> ash::prelude::VkResult<Vec<vk::CommandBuffer>> {
        // SAFETY: the device is live, and its context is used by its own thread only.
        unsafe {
            self.loader_word.restore();
            self.device.allocate_command_buffers(info)
        }
    }

    /// `VK_EXT_device_generated_commands`'s commands, loaded through the loader's
    /// `vkGetDeviceProcAddr`, which reads the device's loader word ([`LoaderWord`]): the word is
    /// put back first. `None` where the context did not enable the extension (or a command does not
    /// load).
    #[expect(
        clippy::disallowed_methods,
        reason = "the one call site, after the loader word is put back"
    )]
    pub(super) fn generated_commands(
        &self,
    ) -> Option<metal2vulkan_runtime::generated_commands::Commands> {
        if !self.generated_commands {
            return None;
        }
        let handle = self.device.handle();
        let load = |name: &CStr| {
            // SAFETY: the device is live, its context is used by its own thread only, and the
            // word is put back before each load.
            unsafe {
                self.loader_word.restore();
                self.instance.get_device_proc_addr(handle, name.as_ptr())
            }
        };
        // SAFETY: each name is the extension's command of this device, which enabled it.
        unsafe { metal2vulkan_runtime::generated_commands::Commands::load(handle, load) }
    }

    /// `vkGetMemoryHostPointerPropertiesEXT` (`VK_EXT_external_memory_host`), loaded through the
    /// loader's `vkGetDeviceProcAddr`, which reads the device's loader word ([`LoaderWord`]): the
    /// word is put back first. `None` where the context did not enable the extension (only the
    /// runtime probe's context does) or the command does not load.
    #[expect(
        clippy::disallowed_methods,
        reason = "the one call site, after the loader word is put back"
    )]
    pub(super) fn memory_host_pointer_properties(
        &self,
    ) -> Option<vk::PFN_vkGetMemoryHostPointerPropertiesEXT> {
        if !self
            .runtime_probe
            .is_some_and(|probe| probe.external_memory_host)
        {
            return None;
        }
        // SAFETY: the device is live, its context is used by its own thread only, and the word is
        // put back before the load.
        let command = unsafe {
            self.loader_word.restore();
            self.instance.get_device_proc_addr(
                self.device.handle(),
                c"vkGetMemoryHostPointerPropertiesEXT".as_ptr(),
            )
        }?;
        // SAFETY: the registry gives the command this signature.
        Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(),
                vk::PFN_vkGetMemoryHostPointerPropertiesEXT,
            >(command)
        })
    }

    /// `VK_KHR_acceleration_structure`'s commands, loaded through the loader's
    /// `vkGetDeviceProcAddr`, which reads the device's loader word ([`LoaderWord`]): the word is
    /// put back first. The commands work only where the context enabled the extension; otherwise
    /// each loads as a stub that panics when called.
    #[expect(
        clippy::disallowed_methods,
        reason = "the one call site, after the loader word is put back"
    )]
    pub(super) fn acceleration_structure(&self) -> ash::khr::acceleration_structure::Device {
        // SAFETY: the device is live, and its context is used by its own thread only.
        unsafe { self.loader_word.restore() };
        ash::khr::acceleration_structure::Device::new(&self.instance, &self.device)
    }

    /// Everything the validation layer reported since the last call, and clear the sink.
    pub(super) fn take_messages(&self) -> Vec<String> {
        self.messages.take()
    }

    /// What the environment records of the robustness leg: the leg and the robustness features the
    /// context enabled.
    pub(super) fn robustness(&self) -> serde_json::Value {
        serde_json::json!({
            "leg": self.leg,
            "enabled": crate::coverage::ROBUSTNESS_FEATURES
                .iter()
                .filter(|feature| self.enabled.features.contains(feature))
                .map(|feature| format!("{feature:?}"))
                .collect::<Vec<_>>(),
            "unmet": self
                .unmet_robustness
                .iter()
                .map(|feature| format!("{feature:?}"))
                .collect::<Vec<_>>(),
        })
    }

    /// What the environment records of the profiles layer: `null` unrestricted, else the layer's
    /// tool as the device reports it and the committed profile it restricts to.
    pub(super) fn profiles_layer(&self) -> serde_json::Value {
        match self.restriction {
            None => serde_json::Value::Null,
            Some(restriction) => serde_json::json!({
                "tool": profiles::profiles_tool(&self.tools),
                "profile": restriction.stem,
                "profile_name": restriction.profile_name,
                "profile_sha256": restriction.sha256,
                "hidden_extensions": self.hidden,
            }),
        }
    }

    pub(super) fn environment(&self) -> serde_json::Value {
        let properties = unsafe { self.instance.get_physical_device_properties(self.physical) };
        let device = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        serde_json::json!({
            "device": device,
            "device_id": properties.device_id,
            "vendor_id": properties.vendor_id,
            "driver_version": properties.driver_version,
            "api_version": properties.api_version,
            "shader_subgroup_extended_types": self.shader_subgroup_extended_types,
            "multisample_array_image": self.multisample_array_image,
            "sampler_anisotropy": self.sampler_anisotropy,
            "max_sampler_anisotropy": self.max_sampler_anisotropy,
            "storage_buffer_alignment": self.storage_buffer_alignment,
            "max_storage_buffer_range": self.max_storage_buffer_range,
            "architecture": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "tools": self.tools,
        })
    }

    pub(super) fn memory_type(
        &self,
        bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical)
        };
        (0..properties.memory_type_count)
            .find(|index| {
                bits & (1 << index) != 0
                    && properties.memory_types[*index as usize]
                        .property_flags
                        .contains(required)
            })
            .ok_or_else(|| {
                format!(
                    "no Vulkan memory type satisfies property flags {:#x}",
                    required.as_raw()
                )
            })
    }
}

impl VulkanContext {
    /// Wait, under the hang protocol, for `fence` of a submission `operation` just made.
    pub(super) fn wait_for_fence(&self, fence: vk::Fence, operation: &str) -> Result<(), String> {
        wait::vulkan::wait_for_fence(&self.device, fence, self.backend, operation)
    }

    /// The device's viewport limits, which each region of a rate-mapped pass's plan must respect:
    /// `maxViewportDimensions` and `viewportBoundsRange`.
    pub(super) fn region_limits(&self) -> metal2vulkan::bind::RegionLimits {
        // SAFETY: `physical` was enumerated from `instance`, which is live.
        let limits = unsafe { self.instance.get_physical_device_properties(self.physical) }.limits;
        metal2vulkan::bind::RegionLimits::new(
            limits.max_viewport_dimensions,
            limits.viewport_bounds_range,
        )
    }
}

impl Drop for VulkanContext {
    /// Destroys the device only after a bounded drain of its queue. If the drain fails the device
    /// and instance are leaked: destroying them while work may be in flight could fault the GPU.
    /// A drain that times out follows the hang protocol and does not return.
    #[expect(
        clippy::disallowed_methods,
        reason = "the one `vkDestroyDevice`, after the loader word is put back"
    )]
    fn drop(&mut self) {
        if let Err(error) = wait::vulkan::drain_queue(&self.device, self.queue, self.backend) {
            eprintln!("leaking the Vulkan device: {error}");
            return;
        }
        // SAFETY: the device is live until the call, its context is used by its own thread only,
        // and the loader's `vkDestroyDevice` reads the device's loader word first.
        unsafe {
            self.loader_word.restore();
            self.device.destroy_device(None);
            let (loader, messenger) = &self.messenger;
            loader.destroy_debug_utils_messenger(*messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn load_entry() -> Result<Entry, String> {
    if let Ok(path) = std::env::var("METAL2VULKAN_LIBVULKAN") {
        if !path.trim().is_empty() {
            return unsafe { Entry::load_from(path.trim()) }
                .map_err(|error| format!("load Vulkan library {path}: {error}"));
        }
    }
    if let Ok(entry) = unsafe { Entry::load() } {
        return Ok(entry);
    }
    for path in vulkan_library_candidates() {
        if !path.is_file() {
            continue;
        }
        if let Ok(entry) = unsafe { Entry::load_from(&path) } {
            return Ok(entry);
        }
    }
    Err(
            "no Vulkan loader found; install vulkan-loader (and MoltenVK on macOS) or set METAL2VULKAN_LIBVULKAN"
                .into(),
        )
}

fn vulkan_library_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        for relative in [
            "lib/libvulkan.dylib",
            "lib/libvulkan.1.dylib",
            "macOS/lib/libvulkan.dylib",
            "lib/libvulkan.so.1",
        ] {
            paths.push(Path::new(&sdk).join(relative));
        }
    }
    for path in [
        "/opt/homebrew/lib/libvulkan.dylib",
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "/usr/lib/libvulkan.so.1",
        "/usr/local/lib/libvulkan.so.1",
    ] {
        paths.push(PathBuf::from(path));
    }
    paths
}

fn extension_name(raw: &[std::ffi::c_char]) -> &CStr {
    unsafe { CStr::from_ptr(raw.as_ptr()) }
}

/// The `VkFormatProperties3` supplement of a venue profile's `vulkaninfo --json` capture, which
/// never writes that structure (`corpus-vulkan --capture-formats`): the device `device_type`
/// selects, picked as a venue run picks it ([`select_enumerated`], from the same loader), and for
/// every format the pinned registry names that the device exposes, its three
/// `VkFormatFeatureFlags2` masks and the sample counts of a color attachment of it
/// (`VkImageFormatProperties::sampleCounts` of a 2D, optimal-tiling image with `COLOR_ATTACHMENT`
/// usage and no flags, where its optimal tiling has `COLOR_ATTACHMENT`, and 0 otherwise or where
/// the device takes no such image). Physical-device queries only: no device is created and nothing
/// is submitted; a GPU capture checks the hang tripwire first, as every device open does, and opens
/// its instance through [`create_instance`]. The instance runs at the loader's version, as
/// `vulkaninfo`'s does, so a format the device exposes only at its own core version is queried too.
/// Each mask is its raw 64-bit value; `scripts/regen-device-profiles` names the bits with the
/// pinned registry and checks the low 32 against the capture's `VkFormatProperties`. The identity
/// is the device's, as a row's `environment` records it, so the regeneration can check both files
/// describe one device.
pub fn capture_formats(device_type: DeviceType) -> Result<serde_json::Value, String> {
    if device_type == DeviceType::Gpu {
        crate::gpu::tripwire::refuse_if_tripped()?;
    }
    let entry = load_entry()?;
    // SAFETY: a global command of a loaded entry.
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(|error| format!("enumerate Vulkan instance extensions: {error}"))?;
    let portability = available.iter().any(|extension| {
        extension_name(&extension.extension_name) == ash::khr::portability_enumeration::NAME
    });
    // SAFETY: a global command of a loaded entry.
    let instance_version = unsafe { entry.try_enumerate_instance_version() }
        .map_err(|error| format!("enumerate the Vulkan instance version: {error}"))?
        .unwrap_or(vk::API_VERSION_1_0);
    let instance = create_instance(&entry, instance_version, portability, None, None)
        .map_err(|error| format!("create the capture's Vulkan instance: {error}"))?;
    let captured = select_enumerated(&instance, device_type).and_then(|(physical, _)| {
        // SAFETY: `physical` was enumerated from `instance`, created at `instance_version`.
        unsafe { describe_formats(&instance, physical, instance_version) }
    });
    // SAFETY: nothing created from the instance outlives it.
    unsafe { instance.destroy_instance(None) };
    captured
}

/// [`capture_formats`]'s document for `physical`.
///
/// # Safety
/// `physical` was enumerated from `instance`, created at API `instance_version` (at least 1.1).
unsafe fn describe_formats(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    instance_version: u32,
) -> Result<serde_json::Value, String> {
    use metal2vulkan::vk as mvk;
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    // SAFETY: the caller's contract; `VkPhysicalDeviceDriverProperties` is Vulkan 1.2, the
    // floor of every device a venue runs.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let core = properties.properties;
    let device_api = mvk::ApiVersion::from_packed(core.api_version);
    let effective = device_api.min(mvk::ApiVersion::from_packed(instance_version));
    // SAFETY: the caller's contract.
    let listed = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map_err(|error| format!("enumerate Vulkan device extensions: {error}"))?
        .iter()
        .map(|extension| {
            extension_name(&extension.extension_name)
                .to_string_lossy()
                .into_owned()
        })
        .collect::<BTreeSet<_>>();
    if effective < ApiVersion::V1_3 && !listed.contains("VK_KHR_format_feature_flags2") {
        return Err(format!(
            "the device reports no VkFormatProperties3: its API version at this instance is \
             {}.{} and it lists no VK_KHR_format_feature_flags2",
            effective.major, effective.minor
        ));
    }
    let mut formats = serde_json::Map::new();
    for &format in mvk::Format::KNOWN {
        let (Some(name), Some(exposure)) = (format.name(), format.exposure()) else {
            continue;
        };
        if !exposure.exposed(effective, |extension| listed.contains(extension)) {
            continue;
        }
        let mut p3 = vk::FormatProperties3::default();
        let mut p2 = vk::FormatProperties2::default().push_next(&mut p3);
        // SAFETY: the caller's contract; the device exposes `format` (at its effective version
        // or through an extension it lists), and `VkFormatProperties3` (checked above).
        unsafe {
            instance.get_physical_device_format_properties2(
                physical,
                vk::Format::from_raw(format.as_raw()),
                &mut p2,
            )
        };
        let samples = if p3
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags2::COLOR_ATTACHMENT)
        {
            let info = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(vk::Format::from_raw(format.as_raw()))
                .ty(vk::ImageType::TYPE_2D)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);
            let mut image = vk::ImageFormatProperties2::default();
            // SAFETY: as above; `image` is live for the call.
            match unsafe {
                instance.get_physical_device_image_format_properties2(physical, &info, &mut image)
            } {
                Ok(()) => image.image_format_properties.sample_counts.as_raw(),
                Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => 0,
                Err(error) => {
                    return Err(format!(
                        "vkGetPhysicalDeviceImageFormatProperties2 of a {name} color attachment: \
                         {error}"
                    ))
                }
            }
        } else {
            0
        };
        formats.insert(
            name.to_string(),
            serde_json::json!({
                "linearTilingFeatures": p3.linear_tiling_features.as_raw(),
                "optimalTilingFeatures": p3.optimal_tiling_features.as_raw(),
                "bufferFeatures": p3.buffer_features.as_raw(),
                "colorAttachmentSampleCounts": samples,
            }),
        );
    }
    let text = |raw: &[std::ffi::c_char]| extension_name(raw).to_string_lossy().into_owned();
    Ok(serde_json::json!({
        "capture": "corpus-vulkan --capture-formats",
        "instance_api_version": instance_version,
        "device": {
            "device": text(&core.device_name),
            "vendor_id": core.vendor_id,
            "device_id": core.device_id,
            "device_type": core.device_type.as_raw(),
            "api_version": core.api_version,
            "driver_id": driver.driver_id.as_raw(),
            "driver_version": core.driver_version,
            "driver_info": text(&driver.driver_info),
        },
        "formats": formats,
    }))
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tests {
    use super::{Purpose, VulkanContext};
    use ash::vk::{self, Handle};

    /// `ICD_LOADER_MAGIC` (`vulkan/vk_icd.h`): what MoltenVK 1.4.1 writes over the loader's word.
    const ICD_LOADER_MAGIC: usize = 0x01CD_C0DE;

    /// Write over the loader's word of `context`'s device what MoltenVK writes there when it
    /// reports a message about the device.
    fn overwrite_loader_word(context: &VulkanContext) {
        let entry: *mut usize =
            std::ptr::with_exposed_provenance_mut(context.device.handle().as_raw() as usize);
        // SAFETY: the device is live, and its first entry is the loader's, which the context puts
        // back before the loader reads it.
        unsafe { entry.write(ICD_LOADER_MAGIC) };
    }

    /// The loader's own entry points the context calls after a driver overwrote the loader's word
    /// of the device still run. They are `vkAllocateCommandBuffers`, `vkGetDeviceProcAddr`
    /// (`VK_KHR_acceleration_structure`'s commands, which load whether or not the extension is
    /// enabled: an absent command loads as one that panics), and `vkDestroyDevice` when the context
    /// drops. Without the context putting the word back, the loader reads `ICD_LOADER_MAGIC` as its
    /// dispatch table and faults at `0x1cdc0de`. Our probe measured this at 9caf79a2, before
    /// the `vkGetDeviceProcAddr` step was added: the test passed on lavapipe and on MoltenVK, and
    /// on lavapipe a build without the allocation's and `Drop`'s restores failed it by `SIGSEGV`.
    /// Other drivers are expected to behave the same, since the context has no layer and no driver
    /// reads the word, so the overwrite reaches only the loader.
    #[test]
    fn the_loaders_device_calls_run_after_a_driver_overwrote_its_word() {
        let context = VulkanContext::new(crate::gpu::TEST_DEVICE, Purpose::Executor).unwrap();
        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family);
        // SAFETY: the device is live; the pool is destroyed before the context drops.
        let pool = unsafe { context.device.create_command_pool(&pool_info, None) }.unwrap();
        overwrite_loader_word(&context);
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffers = context.allocate_command_buffers(&allocate).unwrap();
        assert_eq!(buffers.len(), 1);
        // SAFETY: nothing was recorded into or submitted from the pool's buffer.
        unsafe { context.device.destroy_command_pool(pool, None) };
        overwrite_loader_word(&context);
        let commands = context.acceleration_structure();
        assert_eq!(commands.device(), context.device.handle());
        overwrite_loader_word(&context);
        drop(context);
    }

    /// The formats supplement describes the device a venue run selects, and its masks are what the
    /// product's own query reads for every format it queries: each format of the description is in
    /// the capture with the same three `VkFormatFeatureFlags2` masks, and each format the capture
    /// reports with a feature that the description queries is in it.
    #[test]
    fn the_formats_capture_is_the_product_queried_formats_of_the_venue_device() {
        use metal2vulkan::vk as mvk;
        let captured = super::capture_formats(crate::gpu::TEST_DEVICE).unwrap();
        let context = VulkanContext::new(crate::gpu::TEST_DEVICE, Purpose::Executor).unwrap();
        // SAFETY: `physical` belongs to the context's live instance.
        let properties = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical)
        };
        assert_eq!(captured["device"]["vendor_id"], properties.vendor_id);
        assert_eq!(captured["device"]["device_id"], properties.device_id);
        assert_eq!(
            captured["device"]["driver_version"],
            properties.driver_version
        );
        assert_eq!(captured["device"]["driver_id"], context.driver.driver_id);
        assert_eq!(
            captured["device"]["driver_info"],
            context.driver.driver_info.as_str()
        );
        // SAFETY: as above; the instance was created at API 1.2.
        let description = unsafe {
            metal2vulkan::device::DeviceDescription::query(
                &context.instance,
                context.physical,
                super::ApiVersion::V1_2,
            )
        }
        .unwrap();
        assert!(description.has_format_feature_flags2());
        let formats = captured["formats"].as_object().unwrap();
        for (format, props) in &description.formats.formats {
            let name = format.name().unwrap();
            let masks = &formats[name];
            assert_eq!(
                masks["linearTilingFeatures"],
                props.linear_tiling.as_raw(),
                "{name}"
            );
            assert_eq!(
                masks["optimalTilingFeatures"],
                props.optimal_tiling.as_raw(),
                "{name}"
            );
            assert_eq!(masks["bufferFeatures"], props.buffer.as_raw(), "{name}");
            assert_eq!(
                masks["colorAttachmentSampleCounts"],
                props.color_attachment_sample_counts.as_raw(),
                "{name}"
            );
        }
        let queried = mvk::Format::KNOWN.iter().filter(|format| {
            metal2vulkan::device::FormatSupport::queries_for(
                description.api_version,
                &description.extensions,
            )
            .contains(format)
        });
        for format in queried {
            let masks = &formats[format.name().unwrap()];
            let any = [
                "linearTilingFeatures",
                "optimalTilingFeatures",
                "bufferFeatures",
            ]
            .iter()
            .any(|member| masks[*member] != 0);
            assert_eq!(
                any,
                description.formats.formats.contains_key(format),
                "{format:?}"
            );
        }
    }
}
