//! A case's run through the Metal-like runtime, `metal2vulkan_runtime`: the one copy the runtime
//! lane's device tests (`runtime_lane`) and the engine executor's runtime path (`runtime_run`)
//! share.
//!
//! A kernel case runs as a Metal program would run it: a compute pipeline state of the oracle's
//! descriptor ([`compute_descriptor`]: its kernel, the functions it links, its stage-input
//! descriptor), its buffers, textures, sampler states, threadgroup memory, imageblock size and
//! stage-in region set on a compute encoder, its dispatch, and its output read back after the
//! command buffer completed. Each texture is made as the Metal oracle makes it: its type, pixel
//! format, extent and usage (the case's role united with the AIR's declared access,
//! [`texture_usage`]), `Shared`, its levels (one unless the case declares more), its bytes written
//! per level and slice with `replace_region`, or, for a texture buffer, a texture over a `Shared`
//! buffer of its bytes (`Buffer::new_texture`); a multisampled texture `Private`, filled sample by
//! sample by render passes the case's command buffer runs before its dispatch (`runtime_samples`);
//! a texture output is read per slice with `get_bytes`. Each sampler state is the oracle's
//! descriptor (`resources::sampler_descriptor`). An argument buffer's members are made as the
//! oracle makes them and written into their owner at their fields as its argument encoder writes
//! them ([`bound_buffers`]): a buffer, `Shared`, of its bytes, by its `gpu_address`; a texture as a
//! bound one is made (a multisampled one filled as a bound one is), its usage its role united with
//! the declared access of the AIR member at its field ([`argument_texture_usage`]), by its
//! `gpu_resource_id`; a sampler state supporting argument buffers, of the oracle's descriptor
//! (`resources::argument_sampler_descriptor`), by its `gpu_resource_id`. The buffers and textures
//! are made resident for reading and writing (`use_resources`), as the oracle's
//! `useResource:usage:` does; a sampler state is no resource. A buffer or argument-buffer buffer
//! holding a rasterization-rate map's parameter data is made as the oracle makes it
//! ([`rate_map_buffer`]): the runtime makes the map (`new_rasterization_rate_map`) and copies its
//! data to offset 0 of a `Shared` buffer of the size it answers (`copy_parameter_data_to_buffer`).
//! A buffer, argument-buffer buffer or texture buffer that is an out-of-bounds target (or holds the
//! stage input that is one) lies in a placement heap of its own between guards
//! ([`guarded_buffer`]): every byte of the heap outside the target's length padded to 16 is its
//! guard, and what a run changed there is counted once it completed ([`Committed::guard_counts`]).
//!
//! A tile kernel, a kernel case over a tile pass's render targets, dispatches through a tile
//! pipeline state on a render encoder ([`encoded_tile`], the `tile` module's rule). A render case
//! draws as a Metal program would draw it ([`encoded_render`], [`commit_render`], the `render`
//! module's rule), a vertex buffer that is an out-of-bounds target between guards as a kernel's
//! buffer target is.
//!
//! Every runtime call's refusal stays typed ([`CaseError::Runtime`]), so the executor can tell a
//! gap the runtime names from the product's refusal.

use crate::air_library::AirLibrary;
use crate::candidate::engine::argument_buffers::owner_bytes;
use crate::candidate::engine::guards::{
    buffer_target, embedded_guard_reach, guard_reach, texel_guard_reach, GuardCount, GuardReach,
    MIN_GUARD,
};
use crate::candidate::engine::resources::{
    argument_sampler_descriptor, pixel_format, sampler_descriptor, texture_type,
};
use crate::candidate::engine::tables::{FunctionTable, TableBuffers};
use crate::candidate::engine::{ComputeCase, ComputeOutput, LinkedModule};
use crate::candidate::runtime_samples::{multisampled_texture_with, SampleFill};
use crate::candidate::{unadvertised_stage, unsupported_set};
use crate::case::out_of_bounds::OutOfBoundsResource;
use crate::case::{ArgumentBufferSamplerResource, AuthoredCase, RateMapLiteral, TextureType};
use crate::check::interface::{
    declared_texture_atomic, declared_texture_writable, declared_writable, texture_shader_usage,
    AirInterface,
};
use crate::executor_contract::out_of_bounds::{GUARD_FILL, PAD_FILL};
use crate::executor_contract::stage_inputs;
use crate::gpu::DeviceType;
use crate::literal::{subresource_span, LiteralResources, LiteralTexture, SubresourceSpan};
use metal2vulkan::bind::{Dispatch, FunctionTableWords};
use metal2vulkan::metal::{ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, LinkedFunctions, Stage as ShaderStage, StageSource,
};
use metal2vulkan_air::abi::ParamRole;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipelineState, CpuCacheMode, DebugMessage,
    Degradation, Device, DeviceKind, DeviceOptions, Error, Execution, Function, FunctionHandle,
    HazardTrackingMode, Heap, HeapDescriptor, HeapType, LayerSetting, LayerSettingValue,
    MessageSeverity, MessageSink, Region, ResourceOptions, SamplerState, SamplerStateOptions,
    ShaderCacheOptions, Size, StorageMode, Texture, TextureDescriptor, VisibleFunctionTable,
    VisibleFunctionTableDescriptor,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod commands;
mod host_wait_gap;
pub(crate) mod imageblock;
mod layer_claim;
mod raster_order_gap;
mod render;
mod sparse;
mod tile;
pub use commands::{encoded_execution, pipeline_states, CaseCommands, TemplateState};
/// The lane's own check of the creation messages a device heard reads it too
/// (`runtime_lane::creation_heard_checked`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use layer_claim::loader_layer_message;
pub use layer_claim::{layer_reported_nothing, EnabledLayers, LayerClaim, LAYERS_DROPPED};
pub use render::{
    commit_render, encoded_mesh, encoded_render, mesh_pipeline_state, render_pipeline_descriptor,
    render_pipeline_state, CommittedRender, EncodedRender, MeshInputs, RenderInputs,
};
pub use tile::{encoded_tile, tile_descriptor, tile_tables};

/// Why a case did not run through the runtime.
#[derive(Debug)]
pub enum CaseError {
    /// The runtime call `call` returned `error`.
    Runtime { call: String, error: Box<Error> },
    /// The runtime call `call`, making or drawing the pipeline of a helper of the runner's own (a
    /// multisampled texture's sample fill, a sample readback, `runtime_samples`; a fragment
    /// imageblock's initializer and resolver, `imageblock.rs`: the work the Metal oracle does
    /// around the case's draw), returned `error`: not the case's own pipeline.
    Helper { call: String, error: Box<Error> },
    /// The case cannot run so: an input of a later runtime step, or a literal that does not fit.
    Case(String),
}

impl CaseError {
    /// The runtime's refusal of `call`.
    pub fn runtime(call: impl Into<String>) -> impl FnOnce(Error) -> CaseError {
        let call = call.into();
        move |error| CaseError::Runtime {
            call,
            error: Box::new(error),
        }
    }

    /// The runtime's refusal of `call`, which makes a helper's pipeline ([`CaseError::Helper`]).
    pub fn helper(call: impl Into<String>) -> impl FnOnce(Error) -> CaseError {
        let call = call.into();
        move |error| CaseError::Helper {
            call,
            error: Box::new(error),
        }
    }
}

impl fmt::Display for CaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaseError::Runtime { call, error } | CaseError::Helper { call, error } => {
                write!(f, "{call}: {error}")
            }
            CaseError::Case(why) => f.write_str(why),
        }
    }
}

impl From<String> for CaseError {
    fn from(why: String) -> CaseError {
        CaseError::Case(why)
    }
}

impl From<&str> for CaseError {
    fn from(why: &str) -> CaseError {
        CaseError::Case(why.into())
    }
}

impl From<CaseError> for String {
    fn from(error: CaseError) -> String {
        error.to_string()
    }
}

/// Whether every [`Messages`] sink also writes each failing message to stderr as the layer
/// reports it ([`echo_messages`]).
static ECHO: AtomicBool = AtomicBool::new(false);

/// Write every failing message a [`Messages`] sink keeps to stderr as well, the moment the layer
/// reports it: the engine device worker's setting for its whole process, so a worker a driver
/// fault kills still shows the messages before the fault.
pub(crate) fn echo_messages() {
    ECHO.store(true, Ordering::Relaxed);
}

/// Whether every [`Messages`] sink also writes each failing message to stderr as one
/// `LANE-VALIDATION` line naming its test ([`report_each_message`]).
static REPORT: AtomicBool = AtomicBool::new(false);

/// Write every failing message a [`Messages`] sink receives to stderr as one line, `LANE-VALIDATION
/// <test>: <id name> (<id>): <text>`, past the list's bound and whether or not its test reads the
/// list: the runtime lane's setting when its operational switch enables more of the layer's checks
/// (`runtime_lane::LaneChecks`), so a run's every report can be counted and triaged. A message the
/// sink lets pass (the layer's host-wait gap) prints its own `LAYER-GAP` line instead.
pub(crate) fn report_each_message() {
    REPORT.store(true, Ordering::Relaxed);
}

/// What the validation layer reported that fails a run: every error, and every warning of the
/// validation type (the rule the executor's own context applies); and, apart, the runtime's
/// degrade notices (`DebugMessage::degradation`), which fail nothing. Bounded so a flood cannot
/// grow without limit.
///
/// Two layer gaps pass, on the layer releases whose source was read with their cause in place:
/// the layer's synchronization validation forgetting a semaphore chain once the host waited on a
/// timeline (`runtime_case/host_wait_gap.rs`); and the layer's synchronization validation not
/// reading rasterization-order attachment access, so that it reports an input attachment read of
/// the runtime's pipelines made with it (named so) after an earlier draw's color write
/// (`runtime_case/raster_order_gap.rs`). Each message passed prints a `LAYER-GAP` line with the
/// layer's source, and a test's count of them prints once its messages go. On any other release
/// each fails, marked "re-read WaitForSemaphore in layer X.Y.Z" or "re-read the input attachment
/// ordering in layer X.Y.Z".
///
/// One message about the driver passes too, once the runtime announced it: the layer's report
/// of a vertex module's subgroup operations outside `subgroupSupportedStages`
/// (`candidate/unadvertised_stage.rs`), after the device's
/// `Degradation::UnadvertisedVertexSimdgroups` notice, which it gives only where its profile
/// takes `DriverDeviation::MetalVertexSimdgroups`. Each such message prints an `UNADVERTISED-STAGE`
/// line, and a test's count of them prints once its messages go; before the notice it fails.
/// So does the layer's report of a descriptor set layout the driver's support query refuses
/// (`candidate/unsupported_set.rs`), after the device's `Degradation::SetPastLayoutSupport` notice,
/// which it gives only where its driver runs what the query refuses
/// (`RuntimeDriverDefect::SetLayoutSupportUnderstated`), each printing an `UNSUPPORTED-SET` line.
#[derive(Clone, Default)]
pub struct Messages(Arc<Mutex<Reported>>);

/// The messages [`Messages`] keeps.
struct Reported {
    failing: Vec<DebugMessage>,
    notices: Vec<DebugMessage>,
    /// How many of the layer gap's messages passed.
    layer_gaps: usize,
    /// Whether a device announced `Degradation::UnadvertisedVertexSimdgroups`.
    unadvertised_vertex: bool,
    /// How many of the layer's messages about such a vertex module passed.
    unadvertised_messages: usize,
    /// Whether a device announced `Degradation::SetPastLayoutSupport`.
    unsupported_sets: bool,
    /// How many of the layer's messages about such a set layout passed.
    unsupported_set_messages: usize,
    /// The thread that made the list: the test's.
    test: String,
    /// What the devices reporting here enabled of the instance layers they asked for, and the
    /// claims stated against it (`layer_claim`).
    layers: layer_claim::LayerFacts,
    /// How many notices of each kind were taken so far (`Messages::take_notices`), less those
    /// put back (`Messages::restore_notices`): a device announces a kind once, so a later making
    /// or use of it announces nothing again.
    taken: BTreeMap<Degradation, usize>,
}

impl Default for Reported {
    fn default() -> Reported {
        Reported {
            failing: Vec::new(),
            notices: Vec::new(),
            layer_gaps: 0,
            unadvertised_vertex: false,
            unadvertised_messages: 0,
            unsupported_sets: false,
            unsupported_set_messages: 0,
            taken: BTreeMap::new(),
            layers: layer_claim::LayerFacts::default(),
            test: std::thread::current()
                .name()
                .unwrap_or("an unnamed thread")
                .to_owned(),
        }
    }
}

impl Drop for Reported {
    fn drop(&mut self) {
        self.layers.report(&self.test);
        if self.layer_gaps != 0 {
            eprintln!(
                "LAYER-GAP count {}: {} layer gap message(s) passed",
                self.test, self.layer_gaps
            );
        }
        if self.unadvertised_messages != 0 {
            eprintln!(
                "UNADVERTISED-STAGE count {}: {} message(s) about the vertex stage's subgroups \
                 passed",
                self.test, self.unadvertised_messages
            );
        }
        if self.unsupported_set_messages != 0 {
            eprintln!(
                "UNSUPPORTED-SET count {}: {} message(s) about set layouts the driver's support \
                 query refuses passed",
                self.test, self.unsupported_set_messages
            );
        }
    }
}

/// The `specVersion` of the validation layer the runtime's loader offers, read once.
static LAYER_VERSION: OnceLock<Option<u32>> = OnceLock::new();

impl Messages {
    /// The most messages kept of each list.
    const MAX: usize = 64;

    /// A device reporting here is about to be made asking for the instance layers `layers`: what
    /// it enables of them is the instance's answer, which it announces as it is made
    /// (`layer_claim`). Called once per device, before it opens; a later device's list replaces
    /// an earlier one's.
    pub fn device_asked_layers(&self, layers: &[std::ffi::CString]) {
        let mut reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        reported.layers.asked(layers);
    }

    /// Take the informational message `id_name`, text `text`, where it is one this list keeps
    /// apart (the message a device sends as it is made without the layers it asked for); whether
    /// it was taken. The sink's own path for such a message, and the way a test states one
    /// without a device.
    pub fn heard_informational(&self, id_name: &str, text: &str) -> bool {
        let mut reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let test = reported.test.clone();
        reported.layers.heard(&test, id_name, text)
    }

    /// What the instance of the devices reporting here enabled of the instance layers they asked
    /// for: the layers asked for, less those it announced it was made without.
    pub fn enabled_layers(&self) -> EnabledLayers {
        let reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        reported.layers.enabled()
    }

    /// The text of the message a device reporting here sent as it was made without the layers it
    /// asked for, which the loader could not provide; `None` where none was.
    pub fn layers_dropped_message(&self) -> Option<String> {
        let reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        reported.layers.dropped_message().map(str::to_owned)
    }

    /// State the layer-dependent claim `claim` against what the instance enabled
    /// ([`layer_reported_nothing`], which reads this list's messages first).
    pub(crate) fn state_layer_claim(&self, claim: &str) -> LayerClaim {
        let mut reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let test = reported.test.clone();
        reported.layers.claim(&test, claim)
    }

    /// The failing messages reported so far, removed, with the loader's own messages about a
    /// layer it could not provide that no device of this list explained (`layer_claim`).
    pub fn take(&self) -> Vec<DebugMessage> {
        let mut reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let mut unexplained = reported.layers.release();
        let mut failing = reported.failing.split_off(0);
        unexplained.append(&mut failing);
        unexplained
    }

    /// The degrade notices delivered so far, removed.
    pub fn take_notices(&self) -> Vec<DebugMessage> {
        let mut reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let notices = reported.notices.split_off(0);
        for notice in notices.iter().filter_map(|m| m.degradation.as_ref()) {
            *reported.taken.entry(notice.degradation).or_insert(0) += 1;
        }
        notices
    }

    /// The kinds of the notices taken so far ([`Messages::take_notices`]): those a device
    /// reporting here announced before, which it does not announce again.
    pub fn taken_kinds(&self) -> BTreeSet<Degradation> {
        let reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        reported.taken.keys().copied().collect()
    }

    /// Start [`Messages::taken_kinds`] over for a new device reporting here, whose making
    /// announced `made` (taken apart from the test's notices): a device announces each kind
    /// once, its own, so the kinds an earlier device announced are the new one's to announce
    /// again.
    pub fn device_made(&self, made: &[DebugMessage]) {
        let mut reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        reported.taken.clear();
        for notice in made.iter().filter_map(|m| m.degradation.as_ref()) {
            *reported.taken.entry(notice.degradation).or_insert(0) += 1;
        }
    }

    /// Whether every device reporting here was torn down within `wait`: the sink a device's
    /// instance holds goes after `vkDestroyInstance`, its teardown's last step, wherever that
    /// teardown runs (the device's reaper, after a making still running), so only this list's
    /// own shares remain.
    pub fn unshared_within(&self, wait: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + wait;
        while Arc::strong_count(&self.0) > 1 {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        true
    }

    /// Put `notices`, taken earlier, back before those delivered since: the runtime lane's device
    /// constructor takes a new device's creation notices apart from those a test has not read.
    pub fn restore_notices(&self, mut notices: Vec<DebugMessage>) {
        let mut reported = self.0.lock().unwrap_or_else(|p| p.into_inner());
        for notice in notices.iter().filter_map(|m| m.degradation.as_ref()) {
            let kind = notice.degradation;
            if let Some(count) = reported.taken.get_mut(&kind) {
                *count -= 1;
                if *count == 0 {
                    reported.taken.remove(&kind);
                }
            }
        }
        notices.append(&mut reported.notices);
        reported.notices = notices;
    }

    /// How many of the layer gaps' messages passed so far (`runtime_case/host_wait_gap.rs`,
    /// `runtime_case/raster_order_gap.rs`).
    pub fn layer_gaps(&self) -> usize {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).layer_gaps
    }

    /// How many hold the message list: this handle and its clones, and the instance of each
    /// device made to report to it, which holds it (through its debug messenger's sink) until the
    /// instance is destroyed, the last step of the device's teardown. So with one handle and no
    /// device left, 1.
    pub fn holders(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// The sink a device reports to.
    pub fn sink(&self) -> MessageSink {
        let messages = Arc::clone(&self.0);
        let layer =
            *LAYER_VERSION.get_or_init(|| host_wait_gap::layer_version(loader().as_deref()));
        MessageSink::new(move |message: &DebugMessage| {
            let fails = message.severity == MessageSeverity::Error
                || (message.severity == MessageSeverity::Warning && message.kind.validation);
            let mut reported = messages.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(notice) = &message.degradation {
                if notice.degradation == Degradation::UnadvertisedVertexSimdgroups {
                    reported.unadvertised_vertex = true;
                }
                if notice.degradation == Degradation::SetPastLayoutSupport {
                    reported.unsupported_sets = true;
                }
                if reported.notices.len() < Self::MAX {
                    reported.notices.push(message.clone());
                }
                return;
            }
            // The runtime's pipeline creation log, on where `METAL2VULKAN_LOG_PIPELINES` is set:
            // printed as it comes, never a failure.
            if message.id_name == "metal2vulkan-pipeline-creation" {
                eprintln!("{}", message.text);
                return;
            }
            // What the instance enabled of the layers the device asked for: informational, so it
            // fails nothing, and kept rather than dropped with the rest, since a claim of what the
            // validation layer reported holds nothing without it (`layer_claim`).
            let test = reported.test.clone();
            if reported
                .layers
                .heard(&test, &message.id_name, &message.text)
            {
                return;
            }
            // The loader's own message about an instance layer it cannot provide, reported through
            // the messenger chained at creation before the call answers
            // `VK_ERROR_LAYER_NOT_PRESENT`: held until the device says whether it was made
            // without that layer, which passes it (`layer_claim`), and failing with the rest
            // where nothing says so.
            if layer_claim::loader_layer_message(&message.id_name, &message.text)
                && reported.layers.hold(message)
            {
                return;
            }
            if !fails {
                return;
            }
            if reported.unadvertised_vertex
                && unadvertised_stage::shape(&message.id_name, &message.text)
            {
                reported.unadvertised_messages += 1;
                eprintln!(
                    "UNADVERTISED-STAGE {}: passed, announced as {}: {} ({}): {}",
                    reported.test,
                    Degradation::UnadvertisedVertexSimdgroups.code(),
                    message.id_name,
                    message.id,
                    message.text.replace('\n', " ")
                );
                return;
            }
            if reported.unsupported_sets && unsupported_set::shape(&message.id_name, &message.text)
            {
                reported.unsupported_set_messages += 1;
                eprintln!(
                    "UNSUPPORTED-SET {}: passed, announced as {}: {} ({}): {}",
                    reported.test,
                    Degradation::SetPastLayoutSupport.code(),
                    message.id_name,
                    message.id,
                    message.text.replace('\n', " ")
                );
                return;
            }
            let gap = host_wait_gap::shape(message.id, &message.id_name, &message.text);
            if gap && layer.is_some_and(host_wait_gap::read) {
                reported.layer_gaps += 1;
                eprintln!(
                    "LAYER-GAP {}: passed, the layer's host-wait gap ({}): {} ({}): {}",
                    reported.test,
                    host_wait_gap::SOURCE,
                    message.id_name,
                    message.id,
                    message.text.replace('\n', " ")
                );
                return;
            }
            let raster_gap = raster_order_gap::shape(message.id, &message.id_name, &message.text);
            if raster_gap && layer.is_some_and(raster_order_gap::read) {
                reported.layer_gaps += 1;
                eprintln!(
                    "LAYER-GAP {}: passed, the layer's rasterization-order gap ({}): {} ({}): {}",
                    reported.test,
                    raster_order_gap::SOURCE,
                    message.id_name,
                    message.id,
                    message.text.replace('\n', " ")
                );
                return;
            }
            let mut message = message.clone();
            if raster_gap {
                let read: Vec<String> = raster_order_gap::READ
                    .iter()
                    .map(|&version| host_wait_gap::release(version))
                    .collect();
                message.text = format!(
                    "re-read the input attachment ordering in layer {}: the layer's \
                     rasterization-order gap passes only on the releases read ({}). {}",
                    layer.map_or_else(|| "of an unknown release".into(), host_wait_gap::release),
                    read.join(", "),
                    message.text
                );
                eprintln!(
                    "LAYER-GAP {}: failed, {}",
                    reported.test,
                    message.text.replace('\n', " ")
                );
            }
            if gap {
                let read: Vec<String> = host_wait_gap::READ
                    .iter()
                    .map(|&version| host_wait_gap::release(version))
                    .collect();
                message.text = format!(
                    "re-read WaitForSemaphore in layer {}: the layer's host-wait gap passes only \
                     on the releases read ({}). {}",
                    layer.map_or_else(|| "of an unknown release".into(), host_wait_gap::release),
                    read.join(", "),
                    message.text
                );
                eprintln!(
                    "LAYER-GAP {}: failed, {}",
                    reported.test,
                    message.text.replace('\n', " ")
                );
            }
            if ECHO.load(Ordering::Relaxed) {
                use std::io::Write as _;
                let _ = writeln!(
                    std::io::stderr(),
                    "VK_LAYER_KHRONOS_validation: {message:?}"
                );
            }
            if REPORT.load(Ordering::Relaxed) {
                use std::io::Write as _;
                let _ = writeln!(
                    std::io::stderr(),
                    "LANE-VALIDATION {}: {} ({}): {}",
                    reported.test,
                    message.id_name,
                    message.id,
                    message.text.replace('\n', " ")
                );
            }
            if reported.failing.len() < Self::MAX {
                reported.failing.push(message);
            }
        })
    }
}

/// The Vulkan loader library the runtime opens: its default name when that opens (the system
/// library path), else the first installed of the places the executor's own context also looks
/// (`candidate::platform::device`): Homebrew's and `/usr/local`'s on macOS.
pub fn loader() -> Option<PathBuf> {
    // SAFETY: opening the loader library runs only its initializers; the entry is dropped.
    if unsafe { ash::Entry::load() }.is_ok() {
        return None;
    }
    [
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/opt/homebrew/lib/libvulkan.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.dylib",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}

/// The validation layer every device validation makes through the runtime enables.
pub const VALIDATION_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";

/// The validation layer's settings the executor's devices are made with, as its own context sets
/// them: every occurrence of a message reported (`enable_message_limit` off: by default the layer
/// reports each message ID at most ten times per instance), and synchronization validation on.
pub fn executor_layer_settings() -> Vec<LayerSetting> {
    [(c"enable_message_limit", false), (c"validate_sync", true)]
        .into_iter()
        .map(|(name, value)| {
            LayerSetting::new(
                VALIDATION_LAYER.to_owned(),
                name.to_owned(),
                LayerSettingValue::Bool(value),
            )
        })
        .collect()
}

/// The options of a device validation makes through the runtime: the validation layer reporting
/// to `messages`, its shader cache in `cache`, and the loader [`loader`] finds. Its policy emulates
/// Metal's sampler exactly (`SamplingPolicy::Exact`) and takes Metal's answer wherever the policy
/// offers the device's instead (`DevicePolicy::exact_answers`): it converts texels as Metal does,
/// computes Metal's maths exactly, rounds `half` results as Metal does, and answers Metal's NaN,
/// minimum and maximum and rounding order. Validation compares Metal's answers, which native device
/// sampling and the device's forms answer only within their recorded bounds
/// (`PrecisionNote::DeviceSampling`, `DeviceTexelConversion`; `NativeTranscendentals`,
/// `NativeDivision`, `NativeFma`; `NativeHalfConversion`, `NativeHalfQuantize`;
/// `NativeNanEncoding`, `NativeMinMax`; `NativeDot`, `NativeMix`). The lane of the device's forms
/// sets them back one at a time (`runtime_lane::native_maths`). Its devices keep execution records
/// (`DeviceOptions::record_executions`), which the executor and the lanes read.
pub fn validated_options(messages: &Messages, cache: &Path) -> DeviceOptions {
    let mut options = DeviceOptions::default();
    options.policy = options.policy.exact_answers();
    options.policy.sampling = metal2vulkan::device::SamplingPolicy::Exact;
    options.shader_cache = ShaderCacheOptions::in_directory(cache);
    options.loader = loader();
    options.layers.push(VALIDATION_LAYER.to_owned());
    options.messages = Some(messages.sink());
    // What a claim of what the validation layer reported stands on is what the instance enabled of
    // this list, so the list is told what the device asks for (`layer_claim`).
    messages.device_asked_layers(&options.layers);
    // The executor and the lanes read what each dispatch and draw ran.
    options.record_executions = true;
    options
}

/// The runtime's default device among the physical devices of `device_type`, made with
/// `options`, whose kinds this sets. A GPU is opened only while the hang tripwire is absent: every
/// device validation makes through the runtime opens here (`gpu.rs`'s test).
pub fn open_device(device_type: DeviceType, options: DeviceOptions) -> Result<Device, CaseError> {
    let options = checked_options(device_type, options)?;
    Device::system_default_with(&options).map_err(CaseError::runtime("system_default_with"))
}

/// Every device of `device_type` the runtime lists, made with `options`, whose kinds this sets, as
/// [`open_device`] makes the default one: opened only while the hang tripwire is absent.
pub fn open_devices(
    device_type: DeviceType,
    options: DeviceOptions,
) -> Result<Vec<Device>, CaseError> {
    let options = checked_options(device_type, options)?;
    Device::all_with(&options).map_err(CaseError::runtime("all_with"))
}

/// `options` with the kinds of `device_type`, once the hang tripwire is found absent for a GPU.
fn checked_options(
    device_type: DeviceType,
    mut options: DeviceOptions,
) -> Result<DeviceOptions, CaseError> {
    if device_type == DeviceType::Gpu {
        crate::gpu::tripwire::refuse_if_tripped()?;
    }
    options.kinds = match device_type {
        DeviceType::Cpu => vec![DeviceKind::Cpu],
        DeviceType::Gpu => vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
    };
    Ok(options)
}

/// A shader cache directory one run or test owns: a new, empty directory under the OS temp
/// directory, removed when dropped (after the devices that use it), so a warm cache never stands
/// in for a translation a run measures.
pub struct ScratchCache(PathBuf);

impl ScratchCache {
    /// A new directory, not yet made (the device makes it).
    pub fn new() -> std::io::Result<ScratchCache> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let dir = std::env::temp_dir().join(format!(
            "m2v-lane-shader-cache-{}-{}-{nanos}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::remove_dir_all(&dir) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
            _ => Ok(ScratchCache(dir)),
        }
    }

    /// The directory.
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchCache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The usage the Metal oracle gives each texture of `case`, by Metal texture slot: its role
/// united with the AIR's declared access at its slot, with `ShaderAtomic` where a texture atomic
/// reaches it; an element of a texture array takes its array's declared access, and a later bind
/// of a slot wins (`metal::platform::textures`).
pub fn texture_usage(
    case: &AuthoredCase,
    interface: &AirInterface,
) -> Result<BTreeMap<u32, TextureUsage>, String> {
    let resources = LiteralResources::prepare(case)?;
    let mut usage = BTreeMap::new();
    for texture in &resources.textures {
        let mut texture_usage = texture_shader_usage(
            texture.role,
            declared_texture_writable(interface, texture.binding),
        );
        if declared_texture_atomic(interface, texture.binding) {
            texture_usage |= TextureUsage::SHADER_ATOMIC;
        }
        usage.insert(texture.binding, texture_usage);
    }
    for array in &resources.texture_arrays {
        let writable = declared_texture_writable(interface, array.binding);
        for (element, texture) in (0u32..).zip(&array.elements) {
            let slot = array
                .binding
                .checked_add(element)
                .ok_or("a texture array's slot overflows")?;
            usage.insert(slot, texture_shader_usage(texture.role, writable));
        }
    }
    Ok(usage)
}

/// The usage the Metal oracle gives each of a function's argument-buffer `textures`, by its owner
/// and field offset: its role united with the declared access of the AIR member at its field
/// (`metal::platform::argument_buffers`), without the atomic usage a bound texture may take.
pub fn argument_texture_usage(
    textures: &BTreeMap<(u32, u32), LiteralTexture>,
    interface: &AirInterface,
) -> Result<BTreeMap<(u32, u32), TextureUsage>, String> {
    textures
        .iter()
        .map(|(&(owner, field), texture)| {
            let what = format!("argument-buffer texture {owner}+{field}");
            let (_, member) = interface
                .member_at(owner, field)
                .ok_or_else(|| format!("{what} is no AIR argument-buffer member"))?;
            let ParamRole::Texture(declared) = &member.role else {
                return Err(format!("{what} disagrees with its AIR member role"));
            };
            let writable = declared_writable(declared.access);
            Ok((
                (owner, field),
                texture_shader_usage(texture.role, Some(writable)),
            ))
        })
        .collect()
}

/// A Shared buffer of `bytes` (one zero byte when empty: Metal has no empty buffer).
pub fn buffer_with(device: &Device, bytes: &[u8]) -> Result<Buffer, CaseError> {
    buffer_with_options(device, bytes, ResourceOptions::shared())
}

/// A buffer of `options` (a storage mode the host reaches) holding `bytes`.
pub fn buffer_with_options(
    device: &Device,
    bytes: &[u8],
    options: ResourceOptions,
) -> Result<Buffer, CaseError> {
    let buffer = device
        .new_buffer(bytes.len().max(1) as u64, options)
        .map_err(CaseError::runtime("new_buffer"))?;
    write_contents(&buffer, bytes)?;
    Ok(buffer)
}

/// Write `bytes` at the start of `buffer`, which the host reaches and no command uses yet.
fn write_contents(buffer: &Buffer, bytes: &[u8]) -> Result<(), CaseError> {
    if bytes.len() as u64 > buffer.length() {
        return Err("bytes past the buffer's length".into());
    }
    let contents = buffer.contents().ok_or("the buffer has no contents")?;
    // SAFETY: the buffer holds at least `bytes.len()` bytes at `contents`, and no command uses
    // it yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    Ok(())
}

/// A `Shared` buffer of `bytes` (one zero byte when empty) placed in a tracked placement heap of
/// its own between guards of `reach`: an out-of-bounds target's. The buffer lies at the first
/// multiple of the heap's placement alignment at least `reach.before` (and [`MIN_GUARD`]) bytes
/// into the heap, and the heap holds at least `reach.after` (and [`MIN_GUARD`]) bytes past the
/// buffer's length padded to 16. Those padding bytes (the contract: a descriptor's range may
/// include them) are the memory bound to the target past its end, holding `PAD_FILL`; every other
/// byte of the heap is its guard, holding `GUARD_FILL` ([`guard_count`] counts what a run changed).
/// Tracked, as a buffer of its own is.
pub fn guarded_buffer(
    device: &Device,
    bytes: &[u8],
    reach: GuardReach,
) -> Result<Buffer, CaseError> {
    let options = ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    );
    let length = bytes.len().max(1) as u64;
    let align = device.heap_buffer_size_and_align(length, options).align;
    let front = u64::from(reach.before)
        .max(MIN_GUARD)
        .next_multiple_of(align);
    let padded = length.next_multiple_of(16);
    let size = front + padded + u64::from(reach.after).max(MIN_GUARD);
    let heap = device
        .new_heap(&HeapDescriptor::new(HeapType::Placement, size, options))
        .map_err(CaseError::runtime("new_heap"))?;
    let mut fill = vec![GUARD_FILL; heap.size() as usize];
    fill[(front + length) as usize..(front + padded) as usize].fill(PAD_FILL);
    write_contents(&placed(&heap, heap.size(), 0)?, &fill)?;
    let buffer = placed(&heap, length, front)?;
    write_contents(&buffer, bytes)?;
    Ok(buffer)
}

/// What a run changed around `buffer`, placed by [`guarded_buffer`], once no pending command
/// writes its heap: the guard bytes (every byte of the heap outside the buffer's length padded to
/// 16) that no longer hold `GUARD_FILL`, and the padding bytes past its length that no longer
/// hold `PAD_FILL`.
pub fn guard_count(buffer: &Buffer) -> Result<GuardCount, CaseError> {
    let heap = buffer
        .heap()
        .filter(|heap| heap.heap_type() == HeapType::Placement)
        .ok_or("a guarded target outside a placement heap")?;
    let bytes = read_buffer(&placed(heap, heap.size(), 0)?, 0, heap.size())?;
    let front = buffer.heap_offset() as usize;
    let end = front + buffer.length() as usize;
    let padded = front + buffer.length().next_multiple_of(16) as usize;
    let changed = |bytes: &[u8], fill: u8| bytes.iter().filter(|&&byte| byte != fill).count();
    Ok(GuardCount {
        escaped: (changed(&bytes[..front], GUARD_FILL) + changed(&bytes[padded..], GUARD_FILL))
            as u64,
        past_end: changed(&bytes[end..padded], PAD_FILL) as u64,
    })
}

/// A `Shared` buffer of `length` bytes `offset` bytes into `heap`, of the heap's modes.
fn placed(heap: &Heap, length: u64, offset: u64) -> Result<Buffer, CaseError> {
    heap.new_buffer_at_offset(length, ResourceOptions::shared(), offset)
        .map_err(CaseError::runtime("new_buffer_at_offset"))
}

/// A `Shared` buffer of the runtime's parameter data of `map`, made as the Metal oracle makes
/// Metal's (`metal::platform::buffers::rate_map_buffer`): the device makes the map
/// (`new_rasterization_rate_map`) and copies its data (`copy_parameter_data_to_buffer`) to offset 0
/// of a buffer of the size it answers (`parameter_buffer_size_and_align`).
pub fn rate_map_buffer(device: &Device, map: &RateMapLiteral) -> Result<Buffer, CaseError> {
    let map = device
        .new_rasterization_rate_map(&map.descriptor())
        .map_err(CaseError::runtime("new_rasterization_rate_map"))?;
    let length = map.parameter_buffer_size_and_align().size.max(1);
    let buffer = device
        .new_buffer(length, ResourceOptions::shared())
        .map_err(CaseError::runtime("new_buffer"))?;
    map.copy_parameter_data_to_buffer(&buffer, 0)
        .map_err(CaseError::runtime("copy_parameter_data_to_buffer"))?;
    Ok(buffer)
}

/// `length` bytes of `buffer` from `offset`. No command that writes them is pending.
pub fn read_buffer(buffer: &Buffer, offset: u64, length: u64) -> Result<Vec<u8>, CaseError> {
    let end = offset
        .checked_add(length)
        .filter(|&end| end <= buffer.length())
        .ok_or("a read past the buffer")?;
    let contents = buffer
        .contents()
        .ok_or("a Private buffer has no contents")?;
    // SAFETY: `[offset, end)` is inside the buffer's bytes, and no pending command writes them
    // (the caller's contract).
    let bytes = unsafe {
        std::slice::from_raw_parts(
            contents.as_ptr().add(offset as usize),
            (end - offset) as usize,
        )
    };
    Ok(bytes.to_vec())
}

/// The Metal texture of `literal` on `device`, made as the Metal oracle makes it (the module's
/// rule) with `usage`, holding the literal's bytes. A texture buffer is the oracle's too: a
/// texture over a `Shared` buffer of the literal's bytes, from its first byte, its row the
/// bytes' length. A multisampled texture is filled by a render pass the case's command buffer
/// runs ([`multisampled_texture_with`]: [`CaseResources::of`]'s textures, bound and in argument
/// buffers, and the render runner's), so it is refused here. A sparse texture is made from a
/// sparse heap and mapped before the work (`sparse.rs`).
pub fn texture_with(
    device: &Device,
    literal: &LiteralTexture,
    usage: TextureUsage,
) -> Result<Texture, CaseError> {
    let label = format!("texture {}", literal.binding);
    if literal.sample_count != 1 {
        return Err(format!("{label}: a multisampled texture its command buffer fills").into());
    }
    if let Some(mappings) = &literal.sparse {
        return sparse::texture_with(device, literal, mappings, usage);
    }
    if literal.texture_type == TextureType::Buffer {
        return texture_buffer_over(&buffer_with(device, &literal.bytes)?, literal, usage);
    }
    let descriptor = texture_descriptor(literal, usage)?;
    let texture = device
        .new_texture(&descriptor)
        .map_err(CaseError::runtime(format!("{label}: new_texture")))?;
    // Each level's bytes, level 0 first, written at its level as the oracle writes them; a
    // one-level texture's level 0 is every byte.
    for level in 0..literal.mip_levels {
        let (dimensions, bytes) = literal.level(level)?;
        let span = subresource_span(literal.texture_type, [0; 3], dimensions, 1)?;
        let (row, image, slice) = span_strides(&span, literal);
        let needed = slice * span.slice_count as usize;
        if bytes.len() < needed || needed == 0 {
            return Err(format!(
                "{label} supplies {} bytes but its level {level} needs {needed}",
                bytes.len()
            )
            .into());
        }
        for index in 0..span.slice_count as usize {
            texture
                .replace_region(
                    span_region(&span),
                    level.into(),
                    u64::from(span.base_slice) + index as u64,
                    &bytes[index * slice..(index + 1) * slice],
                    row as u64,
                    image as u64,
                )
                .map_err(CaseError::runtime(format!("{label}: replace_region")))?;
        }
    }
    Ok(texture)
}

/// The texture buffer of `literal`, of `usage`, over `buffer`, which holds the literal's bytes
/// from its first: the oracle's texture over a buffer, its row the bytes' length.
fn texture_buffer_over(
    buffer: &Buffer,
    literal: &LiteralTexture,
    usage: TextureUsage,
) -> Result<Texture, CaseError> {
    let label = format!("texture {}", literal.binding);
    if literal.texture_type != TextureType::Buffer {
        return Err(format!("{label}: a texture over a buffer that is no texture buffer").into());
    }
    buffer
        .new_texture(
            &texture_descriptor(literal, usage)?,
            0,
            literal.bytes.len() as u64,
        )
        .map_err(CaseError::runtime(format!(
            "{label}: new_texture over a buffer"
        )))
}

/// The descriptor the Metal oracle makes `literal`'s texture with, of `usage`: its type, pixel
/// format and extent, `Shared`, its levels (one but where the case declares more).
pub fn texture_descriptor(
    literal: &LiteralTexture,
    usage: TextureUsage,
) -> Result<TextureDescriptor, String> {
    let layout = literal.layout()?;
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = texture_type(literal.texture_type);
    descriptor.pixel_format = pixel_format(literal.format);
    descriptor.width = layout.width.into();
    descriptor.height = layout.height.into();
    descriptor.depth = layout.depth.into();
    // Metal's `arrayLength` counts cubes for a cube array.
    descriptor.array_length = match literal.texture_type {
        TextureType::Cube => 1,
        TextureType::CubeArray => (layout.array_layers / 6).into(),
        _ => layout.array_layers.into(),
    };
    descriptor.mipmap_level_count = literal.mip_levels.into();
    descriptor.usage = usage;
    descriptor.resource_options = ResourceOptions::shared();
    Ok(descriptor)
}

/// The bytes of one row, one image and one slice of `span` of `literal`, tightly packed.
fn span_strides(span: &SubresourceSpan, literal: &LiteralTexture) -> (usize, usize, usize) {
    let row = span.size[0] as usize * literal.format.bytes_per_pixel();
    let image = row * span.size[1] as usize;
    (row, image, image * span.size[2] as usize)
}

/// The region of one slice `span` names.
fn span_region(span: &SubresourceSpan) -> Region {
    Region::new_3d(
        span.origin[0].into(),
        span.origin[1].into(),
        span.origin[2].into(),
        span.size[0].into(),
        span.size[1].into(),
        span.size[2].into(),
    )
}

/// The `(origin, dimensions)` selection, in the case's spelling, of `texture` (made from
/// `literal`), read per slice with `get_bytes` as the Metal oracle reads it. No command that
/// writes it is pending.
pub fn read_texture(
    texture: &Texture,
    literal: &LiteralTexture,
    origin: [u32; 3],
    dimensions: [u32; 3],
) -> Result<Vec<u8>, CaseError> {
    let span = subresource_span(literal.texture_type, origin, dimensions, 1)?;
    let (row, image, slice) = span_strides(&span, literal);
    let mut output = vec![0u8; slice * span.slice_count as usize];
    for (index, bytes) in output.chunks_exact_mut(slice.max(1)).enumerate() {
        texture
            .get_bytes(
                bytes,
                row as u64,
                image as u64,
                span_region(&span),
                0,
                u64::from(span.base_slice) + index as u64,
            )
            .map_err(CaseError::runtime(format!(
                "texture {}: get_bytes",
                literal.binding
            )))?;
    }
    Ok(output)
}

/// What a case binds besides its buffers: textures and sampler states, by Metal slot; the
/// buffers, textures and sampler states its argument buffers hold, by owner slot and field
/// offset; its visible-function tables, by Metal buffer index, which its pipeline state makes
/// ([`visible_function_tables`]; [`encoded_case`] takes them); the fills of its multisampled
/// textures, which the command buffer that reads them runs first ([`SampleFill::encode`]); and
/// the buffer under each texture buffer an out-of-bounds target guards ([`guarded_buffer`]), by
/// Metal slot, whose heap its guard count reads ([`Committed::guard_counts`]).
#[derive(Default)]
pub struct CaseResources {
    pub textures: BTreeMap<u32, Texture>,
    pub samplers: BTreeMap<u32, SamplerState>,
    pub argument_buffers: BTreeMap<(u32, u32), Buffer>,
    pub argument_textures: BTreeMap<(u32, u32), Texture>,
    pub argument_samplers: BTreeMap<(u32, u32), SamplerState>,
    pub tables: BTreeMap<u32, VisibleFunctionTable>,
    pub fills: Vec<SampleFill>,
    pub guarded_texture_buffers: BTreeMap<u32, Buffer>,
    pub commands: CaseCommands,
}

impl CaseResources {
    /// `compute`'s textures (holding their bytes, each of its `usage`, or for an argument-buffer
    /// texture its `argument_usage`) and sampler states on `device`, bound and in argument
    /// buffers, and its argument-buffer buffers (a rate map's, [`rate_map_buffer`]); no table (a
    /// pipeline state makes those). An argument-buffer buffer or a texture buffer that is an
    /// out-of-bounds target lies in a heap of its own between guards ([`guarded_buffer`]).
    pub fn of(
        device: &Device,
        compute: &ComputeCase,
        usage: &BTreeMap<u32, TextureUsage>,
        argument_usage: &BTreeMap<(u32, u32), TextureUsage>,
    ) -> Result<CaseResources, CaseError> {
        let mut textures = BTreeMap::new();
        let mut fills = Vec::new();
        let mut guarded_texture_buffers = BTreeMap::new();
        for (&slot, literal) in &compute.textures {
            let usage = usage
                .get(&slot)
                .copied()
                .ok_or_else(|| format!("texture {slot} has no usage"))?;
            let texture = if let Some(reach) = texel_guard_reach(compute, slot) {
                let buffer = guarded_buffer(device, &literal.bytes, reach)?;
                let texture = texture_buffer_over(&buffer, literal, usage)?;
                guarded_texture_buffers.insert(slot, buffer);
                texture
            } else {
                filled_texture(device, literal, usage, &mut fills)?
            };
            textures.insert(slot, texture);
        }
        let samplers = compute
            .samplers
            .iter()
            .map(|(&slot, sampler)| {
                let state = device
                    .new_sampler_state(&sampler_descriptor(sampler))
                    .map_err(CaseError::runtime(format!(
                        "sampler {slot}: new_sampler_state"
                    )))?;
                Ok((slot, state))
            })
            .collect::<Result<_, CaseError>>()?;
        // A command buffer's field holds the command buffer, not the record storage the mapped
        // case states for the executor (`commands`).
        let argument_buffers = compute
            .argument_buffer_buffers
            .iter()
            .filter(|(key, _)| !compute.command_buffers.contains_key(key))
            .map(|(&key, bytes)| {
                let buffer = match (
                    compute.argument_buffer_rate_maps.get(&key),
                    embedded_guard_reach(compute, key),
                ) {
                    (Some(map), _) => rate_map_buffer(device, map)?,
                    (None, Some(reach)) => guarded_buffer(device, bytes, reach)?,
                    (None, None) => buffer_with(device, bytes)?,
                };
                Ok((key, buffer))
            })
            .collect::<Result<_, CaseError>>()?;
        let argument_textures = argument_textures(
            device,
            &compute.argument_buffer_textures,
            argument_usage,
            &mut fills,
        )?;
        let argument_samplers = argument_samplers(device, &compute.argument_buffer_samplers)?;
        Ok(CaseResources {
            textures,
            samplers,
            argument_buffers,
            argument_textures,
            argument_samplers,
            tables: BTreeMap::new(),
            fills,
            guarded_texture_buffers,
            commands: CaseCommands::of(device, &compute.command_buffers)?,
        })
    }
}

/// A function's argument-buffer `textures` on `device`, by `(owner, field)`, each holding its
/// contents with its `usage` ([`argument_texture_usage`]), a multisampled one's fill joining
/// `fills` ([`filled_texture`]).
fn argument_textures(
    device: &Device,
    textures: &BTreeMap<(u32, u32), LiteralTexture>,
    usage: &BTreeMap<(u32, u32), TextureUsage>,
    fills: &mut Vec<SampleFill>,
) -> Result<BTreeMap<(u32, u32), Texture>, CaseError> {
    textures
        .iter()
        .map(|(&key, literal)| {
            let usage = usage
                .get(&key)
                .copied()
                .ok_or_else(|| format!("argument-buffer texture {key:?} has no usage"))?;
            Ok((key, filled_texture(device, literal, usage, fills)?))
        })
        .collect()
}

/// A function's argument-buffer `samplers` on `device`, by `(owner, field)`: each a sampler state
/// made for argument buffers (`support_argument_buffers`), as the Metal oracle makes it.
fn argument_samplers(
    device: &Device,
    samplers: &BTreeMap<(u32, u32), ArgumentBufferSamplerResource>,
) -> Result<BTreeMap<(u32, u32), SamplerState>, CaseError> {
    let mut options = SamplerStateOptions::default();
    options.support_argument_buffers = true;
    samplers
        .iter()
        .map(|(&key, sampler)| {
            let state = device
                .new_sampler_state_with_options(&argument_sampler_descriptor(sampler), options)
                .map_err(CaseError::runtime(format!(
                    "argument-buffer sampler {key:?}: new_sampler_state"
                )))?;
            Ok((key, state))
        })
        .collect()
}

/// The word each member's owner holds at its field, by `(owner, field)`, as Metal's argument
/// encoder writes it: `members` of each kind, a field two members name refused.
fn member_words(
    members: impl IntoIterator<Item = ((u32, u32), u64)>,
) -> Result<BTreeMap<(u32, u32), u64>, CaseError> {
    let mut fields = BTreeMap::new();
    for (key, word) in members {
        if fields.insert(key, word).is_some() {
            return Err(format!("argument-buffer field {key:?} holds two members").into());
        }
    }
    Ok(fields)
}

/// The texture of `literal` on `device` with `usage`, as [`texture_with`] makes it, or a
/// multisampled one ([`multisampled_texture_with`]) whose fill, a render pass the case's command
/// buffer runs before its work, joins `fills`, as the Metal oracle fills one wherever the case
/// holds it.
fn filled_texture(
    device: &Device,
    literal: &LiteralTexture,
    usage: TextureUsage,
    fills: &mut Vec<SampleFill>,
) -> Result<Texture, CaseError> {
    if literal.sample_count == 1 {
        return texture_with(device, literal, usage);
    }
    let (texture, fill) = multisampled_texture_with(device, literal, usage)?;
    fills.push(fill);
    Ok(texture)
}

/// `case`'s bound buffers, each holding its bytes, or a rate map's the runtime's parameter data
/// ([`rate_map_buffer`]): an argument buffer's owner also holds, at each field, what Metal's
/// argument encoder writes there for the member of `resources` there: an argument-buffer buffer's
/// `gpu_address`, or a texture's or sampler state's `gpu_resource_id`. A field two members name
/// is refused. A buffer that is an out-of-bounds target, or holds the stage input that is one,
/// lies in a heap of its own between guards ([`guarded_buffer`]).
pub fn bound_buffers(
    device: &Device,
    case: &ComputeCase,
    resources: &CaseResources,
) -> Result<BTreeMap<u32, Buffer>, CaseError> {
    let buffers = resources
        .argument_buffers
        .iter()
        .map(|(&key, buffer)| (key, buffer.gpu_address().0));
    let textures = resources
        .argument_textures
        .iter()
        .map(|(&key, texture)| (key, texture.gpu_resource_id().0));
    let samplers = resources
        .argument_samplers
        .iter()
        .map(|(&key, sampler)| (key, sampler.gpu_resource_id().0));
    let fields = member_words(
        buffers
            .chain(textures)
            .chain(samplers)
            .chain(resources.commands.words()),
    )?;
    case.buffers
        .keys()
        .map(|&index| {
            if let Some(map) = case.rate_maps.get(&index) {
                return Ok((index, rate_map_buffer(device, map)?));
            }
            let bytes = owner_bytes(&case.buffers, index, &fields)?;
            let buffer = match guard_reach(case, index) {
                Some(reach) => guarded_buffer(device, &bytes, reach)?,
                None => buffer_with(device, &bytes)?,
            };
            Ok((index, buffer))
        })
        .collect()
}

/// The descriptor of `case`'s compute pipeline state, as the Metal oracle states it: its kernel
/// `function` (made with the case's constant values) linking `linked`, and the case's stage-input
/// descriptor (`stageInputDescriptor`), Metal's defaults otherwise.
pub fn compute_descriptor(
    function: &Function,
    linked: LinkedFunctions,
    case: &ComputeCase,
) -> ComputePipelineDescriptor {
    let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(
        function.function_descriptor().with_linked(linked),
    ));
    descriptor.stage_input = case.stage_input.clone();
    descriptor
}

/// The functions a case links, made on a device ([`linked_functions`]): what its pipeline
/// descriptor links (`functions`), and each function made, by its module's SHA-256 and its Metal
/// name, the pair a visible-function table's entry names ([`visible_function_tables`]).
pub struct LinkedSet {
    pub functions: LinkedFunctions,
    made: BTreeMap<(String, String), Function>,
}

impl LinkedSet {
    /// The function `name` of the module whose SHA-256 is `module_sha256`, if the set links it.
    fn function(&self, module_sha256: &str, name: &str) -> Option<&Function> {
        self.made.get(&(module_sha256.to_owned(), name.to_owned()))
    }
}

/// The functions `modules` supply, each library's in turn, every one made with `constants`, as
/// the Metal oracle links them (`load_linked_functions`): a value for a constant a function does
/// not declare is ignored.
pub fn linked_functions(
    device: &Device,
    modules: &[LinkedModule],
    constants: &metal2vulkan::library::FunctionConstantValues,
) -> Result<LinkedSet, CaseError> {
    let mut functions = LinkedFunctions::new();
    let mut made = BTreeMap::new();
    for module in modules {
        let library = device
            .new_library_with_air(&module.bitcode)
            .map_err(CaseError::runtime("new_library_with_air"))?;
        for name in &module.library.functions {
            let function = library
                .new_function_with_name_constant_values(name, constants)
                .map_err(CaseError::runtime("new_function_with_name_constant_values"))?;
            functions = functions.with_function(function.linked_function());
            made.insert((module.module_sha256.clone(), name.clone()), function);
        }
    }
    Ok(LinkedSet { functions, made })
}

/// A kernel case's visible-function tables made for its pipeline state
/// ([`visible_function_tables`]): each table by Metal buffer index, and the bytes the runtime
/// wrote to each (`FunctionTableWords` of its size, each entry the word of the handle the state
/// gave), keyed by the kernel stage and the index as the executor's artifact records a table's
/// buffer (`tables::TableBuffers`).
pub struct CaseTables {
    pub tables: BTreeMap<u32, VisibleFunctionTable>,
    pub words: TableBuffers,
}

/// The visible-function tables of `case` for `pipeline`, by Metal buffer index, as the Metal
/// oracle makes them (`newVisibleFunctionTableWithDescriptor:` and `setFunction:atIndex:` of
/// `functionHandleWithFunction:`): a table of the authored size, each authored entry the handle
/// `pipeline` gives the function it holds among `linked`, every other entry null. An entry whose
/// function `linked` does not hold, or that the pipeline gives no handle, cannot run (the executor
/// links every entry's module into `functions`, as the oracle does). Each table's bytes are the
/// words the runtime writes: an entry's is the word of the handle `linked`'s functions give its
/// function (`LinkedFunctions::function_handle`), which is the handle the state answers, the state
/// linking those functions.
pub fn visible_function_tables(
    pipeline: &ComputePipelineState,
    case: &ComputeCase,
    linked: &LinkedSet,
) -> Result<CaseTables, CaseError> {
    tables_of(
        (ShaderStage::Kernel, &case.function_tables),
        linked,
        |descriptor| pipeline.new_visible_function_table(descriptor),
        |function| pipeline.function_handle(function),
    )
}

/// The visible-function tables `authored` of a function of `stage`, each made by `make` and each
/// entry the handle `handle` gives the function it holds among `linked`, as
/// [`visible_function_tables`] states it for a kernel; a render state's stage makes its own
/// (`render.rs`), its words keyed by that stage.
fn tables_of(
    (stage, authored): (ShaderStage, &BTreeMap<u32, FunctionTable>),
    linked: &LinkedSet,
    make: impl Fn(&VisibleFunctionTableDescriptor) -> Result<VisibleFunctionTable, Error>,
    handle: impl Fn(&Function) -> Result<Option<FunctionHandle>, Error>,
) -> Result<CaseTables, CaseError> {
    let mut made = CaseTables {
        tables: BTreeMap::new(),
        words: TableBuffers::new(),
    };
    for (&binding, authored) in authored {
        let descriptor = VisibleFunctionTableDescriptor::new(authored.size.into());
        let table = make(&descriptor).map_err(CaseError::runtime(format!(
            "table {binding}: new_visible_function_table"
        )))?;
        let mut words = FunctionTableWords::new(authored.size);
        for (&index, entry) in &authored.entries {
            let function = linked
                .function(&entry.module_sha256, &entry.function)
                .ok_or_else(|| {
                    format!(
                        "table {binding} entry {index}: {} of module {} is not linked",
                        entry.function, entry.module_sha256
                    )
                })?;
            let no_handle = || {
                format!(
                    "table {binding} entry {index}: the pipeline gives linked function {} no \
                     handle",
                    entry.function
                )
            };
            let handle = handle(function)
                .map_err(CaseError::runtime(format!(
                    "table {binding} entry {index}: function_handle"
                )))?
                .ok_or_else(no_handle)?;
            table
                .set_function(Some(&handle), index)
                .map_err(CaseError::runtime(format!(
                    "table {binding}: set_function {index}"
                )))?;
            let word = linked
                .functions
                .function_handle(&function.linked_function())
                .ok_or_else(no_handle)?;
            if !words.set(index, Some(word)) {
                return Err(format!("table {binding} has no entry {index}").into());
            }
        }
        made.tables.insert(binding, table);
        made.words.insert((stage, binding), words.to_bytes());
    }
    Ok(made)
}

/// Encode `case`'s dispatch on `encoder` with `pipeline`, over `buffers` and `resources` (by Metal
/// slot, its tables by Metal buffer index, its argument-buffer buffers and textures resident for
/// reading and writing), the case's threadgroup memory lengths, its imageblock size
/// (`setImageblockWidth:height:`), and, for a case with stage inputs dispatching a grid of threads,
/// the case's stage-in region or that grid's from the origin, the region the oracle sets
/// (`stage_inputs::region`). A stage input of a dynamic layout's data is bound with the stride form
/// at its case's stride ([`ComputeCase::attribute_strides`]).
pub fn encode_case(
    encoder: &mut ComputeCommandEncoder,
    pipeline: &ComputePipelineState,
    case: &ComputeCase,
    buffers: &BTreeMap<u32, Buffer>,
    resources: &CaseResources,
) -> Result<(), CaseError> {
    encoder
        .set_compute_pipeline_state(pipeline)
        .map_err(CaseError::runtime("set_compute_pipeline_state"))?;
    for (&index, buffer) in buffers {
        match case.attribute_strides.get(&index) {
            Some(&stride) => encoder
                .set_buffer_with_stride(Some(buffer), 0, stride.into(), index)
                .map_err(CaseError::runtime(format!(
                    "set_buffer_with_stride {index}"
                )))?,
            None => encoder
                .set_buffer(Some(buffer), 0, index)
                .map_err(CaseError::runtime(format!("set_buffer {index}")))?,
        }
    }
    for (&index, table) in &resources.tables {
        encoder
            .set_visible_function_table(Some(table), index)
            .map_err(CaseError::runtime(format!(
                "set_visible_function_table {index}"
            )))?;
    }
    for (&index, texture) in &resources.textures {
        encoder
            .set_texture(Some(texture), index)
            .map_err(CaseError::runtime(format!("set_texture {index}")))?;
    }
    for (&index, sampler) in &resources.samplers {
        encoder
            .set_sampler_state(Some(sampler), index)
            .map_err(CaseError::runtime(format!("set_sampler_state {index}")))?;
    }
    let members: Vec<_> = resources
        .argument_buffers
        .values()
        .map(Buffer::as_resource)
        .chain(
            resources
                .argument_textures
                .values()
                .map(Texture::as_resource),
        )
        .chain(resources.commands.resources())
        .collect();
    if !members.is_empty() {
        encoder
            .use_resources(&members, ResourceUsage::READ | ResourceUsage::WRITE)
            .map_err(CaseError::runtime("use_resources"))?;
    }
    for (&index, &length) in &case.threadgroup_lengths {
        encoder
            .set_threadgroup_memory_length(length, index)
            .map_err(CaseError::runtime(format!(
                "set_threadgroup_memory_length {index}"
            )))?;
    }
    if let Some([width, height]) = case.imageblock {
        encoder
            .set_imageblock_size(width.into(), height.into())
            .map_err(CaseError::runtime("set_imageblock_size"))?;
    }
    if let (Some(_), Dispatch::Threads { threads, .. }) = (&case.stage_input, case.dispatch) {
        let region = stage_inputs::region(case.stage_in_region, threads);
        let [x, y, z] = region.origin.map(u64::from);
        let [width, height, depth] = region.size.map(u64::from);
        encoder
            .set_stage_in_region(Region::new_3d(x, y, z, width, height, depth))
            .map_err(CaseError::runtime("set_stage_in_region"))?;
    }
    let size = |axes: [u32; 3]| Size::new(axes[0].into(), axes[1].into(), axes[2].into());
    match case.dispatch {
        Dispatch::Threads {
            threads,
            threads_per_group,
        } => encoder
            .dispatch_threads(size(threads), size(threads_per_group))
            .map_err(CaseError::runtime("dispatch")),
        Dispatch::Threadgroups {
            groups,
            threads_per_group,
        } => encoder
            .dispatch_threadgroups(size(groups), size(threads_per_group))
            .map_err(CaseError::runtime("dispatch")),
        other => Err(format!("the dispatch {other:?} is a later runtime step").into()),
    }
}

/// A kernel case's work encoded on its device and not yet committed ([`encoded_case`], or a tile
/// kernel's [`encoded_tile`]): its command buffer, what it binds, a tile kernel's pass's
/// attachments by color index, and what its dispatch executed (the command buffer's executions
/// after its fills'). The executions are known before anything runs, so a caller decides from them
/// whether to run the work ([`Encoded::commit`]); dropped uncommitted, it never runs.
pub struct Encoded {
    pub command_buffer: CommandBuffer,
    pub buffers: BTreeMap<u32, Buffer>,
    pub resources: CaseResources,
    pub attachments: BTreeMap<u32, Texture>,
    pub executions: Vec<Execution>,
}

/// A kernel case's work committed on its device ([`Encoded::commit`]).
pub struct Committed {
    pub command_buffer: CommandBuffer,
    pub buffers: BTreeMap<u32, Buffer>,
    pub resources: CaseResources,
    pub attachments: BTreeMap<u32, Texture>,
    pub executions: Vec<Execution>,
}

/// Encode `case`'s dispatch on `device` with `pipeline` (the pipeline state of its entry), its
/// visible-function `tables` ([`visible_function_tables`]), the pipeline state of each token its
/// commands and argument buffers name (`templates`, each made with
/// `support_indirect_command_buffers`), and the textures' `usage` and `argument_usage`
/// ([`CaseResources::of`]): its resources holding their bytes and its command buffers the host's
/// commands (`commands`), one command buffer with the fills of its multisampled textures, then one
/// compute encoder, ended, then the copy of each `Private` command buffer into its twin. The
/// encoded case holds the tables with its other resources.
pub fn encoded_case(
    device: &Device,
    case: &ComputeCase,
    usage: &BTreeMap<u32, TextureUsage>,
    argument_usage: &BTreeMap<(u32, u32), TextureUsage>,
    pipeline: &ComputePipelineState,
    (tables, templates): (
        BTreeMap<u32, VisibleFunctionTable>,
        &BTreeMap<u64, TemplateState>,
    ),
) -> Result<Encoded, CaseError> {
    let mut resources = CaseResources::of(device, case, usage, argument_usage)?;
    resources.tables = tables;
    resources.commands.states = pipeline_states(&case.objects, templates)?;
    let buffers = bound_buffers(device, case, &resources)?;
    commands::encode_host_commands(
        &case.command_buffers,
        &resources.commands,
        &buffers,
        templates,
    )?;
    let queue = device
        .new_command_queue()
        .map_err(CaseError::runtime("new_command_queue"))?;
    let command_buffer = queue
        .command_buffer()
        .map_err(CaseError::runtime("command_buffer"))?;
    for fill in &resources.fills {
        fill.encode(&command_buffer)?;
    }
    let filled = command_buffer.executions().len();
    let mut encoder = command_buffer
        .compute_command_encoder()
        .map_err(CaseError::runtime("compute_command_encoder"))?;
    encode_case(&mut encoder, pipeline, case, &buffers, &resources)?;
    encoder
        .end_encoding()
        .map_err(CaseError::runtime("end_encoding"))?;
    commands::encode_twin_copies(&command_buffer, &resources.commands)?;
    // Each ended encoder's executions join the command buffer's in encoding order (a blit adds
    // none).
    let executions = command_buffer.executions().split_off(filled);
    Ok(Encoded {
        command_buffer,
        buffers,
        resources,
        attachments: BTreeMap::new(),
        executions,
    })
}

impl Encoded {
    /// Commit the encoded work.
    pub fn commit(self) -> Result<Committed, CaseError> {
        let Encoded {
            command_buffer,
            buffers,
            resources,
            attachments,
            executions,
        } = self;
        command_buffer
            .commit()
            .map_err(CaseError::runtime("commit"))?;
        Ok(Committed {
            command_buffer,
            buffers,
            resources,
            attachments,
            executions,
        })
    }
}

/// [`encoded_case`] of a case whose commands and argument buffers name no pipeline token,
/// committed. Nothing is committed when a call is refused.
pub fn commit_case(
    device: &Device,
    case: &ComputeCase,
    usage: &BTreeMap<u32, TextureUsage>,
    argument_usage: &BTreeMap<(u32, u32), TextureUsage>,
    pipeline: &ComputePipelineState,
    tables: BTreeMap<u32, VisibleFunctionTable>,
) -> Result<Committed, CaseError> {
    encoded_case(
        device,
        case,
        usage,
        argument_usage,
        pipeline,
        (tables, &BTreeMap::new()),
    )?
    .commit()
}

impl Committed {
    /// What the run changed around each of `case`'s targets it placed between guards
    /// ([`guard_count`]), by target, once the command buffer completed: a bound buffer's (or the
    /// stage input's whose data it binds), an argument-buffer buffer's, and a texture buffer's.
    pub fn guard_counts(
        &self,
        case: &ComputeCase,
    ) -> Result<BTreeMap<OutOfBoundsResource, GuardCount>, CaseError> {
        let mut counts = BTreeMap::new();
        for (&index, buffer) in &self.buffers {
            if guard_reach(case, index).is_some() {
                counts.insert(buffer_target(case, index), guard_count(buffer)?);
            }
        }
        for (&key, buffer) in &self.resources.argument_buffers {
            if embedded_guard_reach(case, key).is_some() {
                let (buffer_binding, field_offset) = key;
                let target = OutOfBoundsResource::ArgumentBufferBuffer {
                    buffer_binding,
                    field_offset,
                };
                counts.insert(target, guard_count(buffer)?);
            }
        }
        for (&binding, buffer) in &self.resources.guarded_texture_buffers {
            counts.insert(
                OutOfBoundsResource::Texture { binding },
                guard_count(buffer)?,
            );
        }
        Ok(counts)
    }

    /// The bytes `case`'s output selects, once the command buffer completed.
    pub fn output(&self, case: &ComputeCase) -> Result<Vec<u8>, CaseError> {
        match case.output {
            ComputeOutput::Buffer {
                binding,
                offset,
                length,
            } => read_buffer(
                self.buffers
                    .get(&binding)
                    .ok_or_else(|| format!("the output buffer {binding} is not bound"))?,
                offset,
                length,
            ),
            ComputeOutput::Texture {
                binding,
                origin,
                dimensions,
            } => read_texture(
                self.resources
                    .textures
                    .get(&binding)
                    .ok_or_else(|| format!("the output texture {binding} is not bound"))?,
                case.textures
                    .get(&binding)
                    .ok_or_else(|| format!("the output texture {binding} is not declared"))?,
                origin,
                dimensions,
            ),
            ComputeOutput::ArgumentBufferBuffer {
                owner,
                field_offset,
                offset,
                length,
            } => read_buffer(
                self.resources
                    .argument_buffers
                    .get(&(owner, field_offset))
                    .ok_or_else(|| {
                        format!("no argument-buffer buffer at {owner}+{field_offset}")
                    })?,
                offset,
                length,
            ),
            ComputeOutput::ArgumentBufferTexture {
                owner,
                field_offset,
                origin,
                dimensions,
            } => read_texture(
                self.resources
                    .argument_textures
                    .get(&(owner, field_offset))
                    .ok_or_else(|| {
                        format!("no argument-buffer texture at {owner}+{field_offset}")
                    })?,
                case.argument_buffer_textures
                    .get(&(owner, field_offset))
                    .ok_or_else(|| {
                        format!("argument-buffer texture {owner}+{field_offset} undeclared")
                    })?,
                origin,
                dimensions,
            ),
            ComputeOutput::RenderTarget {
                index,
                origin,
                dimensions,
            } => read_texture(
                self.attachments
                    .get(&index)
                    .ok_or_else(|| format!("the output attachment {index} is not attached"))?,
                case.tile
                    .as_ref()
                    .and_then(|pass| pass.attachments.get(&index))
                    .ok_or_else(|| format!("the output attachment {index} is not declared"))?,
                [origin[0], origin[1], 0],
                [dimensions[0], dimensions[1], 1],
            ),
            ComputeOutput::None => Ok(Vec::new()),
        }
    }
}
