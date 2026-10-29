use crate::gpu::DeviceType;
use crate::observation::Backend;
use metal2vulkan::device::DevicePolicy;
use std::sync::OnceLock;

/// The exit statuses of the candidate executors' processes (`corpus-vulkan`, `corpus-moltenvk`):
/// one per [`ExecutorError`](crate::check::admission::ExecutorError) kind that is not a failure, so a caller that runs them as processes
/// (`corpus-vulkan --batch`) reads the kind from the status. A failure exits 1, and so does a
/// mismatch, whose row the store then holds; a timed-out device wait exits
/// [`crate::gpu::wait::HANG_EXIT_CODE`].
pub mod exit_status {
    /// A run refused by policy (`execution_refused{...}`), distinct from a failure (1).
    pub const REFUSED: i32 = 3;
    /// A case the engine executor cannot run (`not_supported{...}`): the product refused it,
    /// the device did not enable what it needs, or the executor does not implement it.
    pub const NOT_SUPPORTED: i32 = 4;
    /// An engine run whose device worker died by a signal (`device_crash{...}`): a driver or
    /// executor fault, never evidence.
    pub const DEVICE_CRASH: i32 = 5;
    /// A case with no current Metal observation to compare against
    /// (`needs_metal_qualification{...}`): qualify it on Metal first. Nothing ran.
    pub const NEEDS_METAL: i32 = 6;
    /// A run recorded under a C3 venue whose device is unlike the venue's profile
    /// (`venue_mismatch{...}`). Nothing was recorded, and a batch stops.
    pub const VENUE_MISMATCH: i32 = 7;
}

/// Create an engine pipeline on `device` from the binder layout its artifact carries, time
/// the creation, and destroy it. Nothing is bound or dispatched.
pub fn create_acceptance_pipeline(
    device: DeviceType,
    pipeline: &crate::acceptance::RowPipeline,
) -> Result<crate::acceptance::Acceptance, crate::acceptance::AcceptanceError> {
    platform::create_acceptance_pipeline(device, pipeline)
}

/// Restrict every Vulkan context this process opens to committed device profile `stem` through
/// `VK_LAYER_KHRONOS_profiles` (a command's `--profiles-layer P`). Call it once, before any device
/// opens. Each restricted context checks that the device has the profile and reports exactly it
/// before its device is created, and its rows record the layer and the profile and are stored in
/// the profile's own slot.
pub fn restrict_devices_to(stem: &str) -> Result<(), String> {
    platform::restrict_devices_to(stem)
}

/// Refuses committed profile `stem` as a restriction when the profiles layer cannot present it as
/// the product reads it (a requirements document): what `restrict_devices_to` checks first, and a
/// batch restricted to it checks before its first run, so no run is tried.
pub fn check_presentable(stem: &str) -> Result<(), String> {
    platform::check_presentable(stem)
}

/// The `VkFormatProperties3` supplement of a venue profile's `vulkaninfo --json` capture
/// (`corpus-vulkan --capture-formats`): the identity of the device `device_type` selects, as a
/// venue run selects it, and the three `VkFormatFeatureFlags2` masks of every format the pinned
/// registry names that it exposes. Physical-device queries only.
pub fn capture_formats(device_type: DeviceType) -> Result<serde_json::Value, String> {
    platform::capture_formats(device_type)
}

/// The runtime's device measurements (`corpus-vulkan --probe-runtime`): on the device `device_type`
/// selects, as a venue run selects it, in a validated context of its own, host memory imported
/// without a copy (`VK_EXT_external_memory_host`) and used by a dispatch, linear images over a
/// buffer's memory (pitches, aliasing, and explicit layouts through
/// `VK_EXT_image_drm_format_modifier`), and the fragment's depth (whether `FragCoord.z`, clamped to
/// the viewport's range, is the depth the device's own test stores). The report is JSON. A GPU
/// probe checks the hang tripwire first, as every device open does, and every wait is bounded.
pub fn probe_runtime(device_type: DeviceType) -> Result<serde_json::Value, String> {
    platform::probe_runtime(device_type)
}

/// The committed profile every Vulkan context this process opens is restricted to
/// ([`restrict_devices_to`]), which an engine device worker applies before it opens a device.
/// Reading it decides that there is none if none was set.
pub(crate) fn requested_restriction() -> Option<String> {
    platform::requested_restriction()
}

/// Run every engine context this process opens in the robustness-off leg (a command's
/// `--robustness-off`): it enables no robustness feature, so the product compiles the in-shader
/// containment encodings, and its rows are stored in the leg's own slot. Without it an engine
/// context enables every robustness feature its device's description states (the stated leg). Call
/// it once, before any device opens.
pub fn run_robustness_off() -> Result<(), String> {
    platform::run_robustness_off()
}

/// The robustness leg every engine context this process opens runs under
/// ([`run_robustness_off`]), which an engine device worker applies before it opens a device.
/// Reading it decides that it is the stated leg if none was set.
pub fn requested_leg() -> crate::coverage::Leg {
    platform::requested_leg()
}

/// Whether every engine context this process opens enables `VK_EXT_device_generated_commands`
/// where its device supports it, so the device replay takes the generated-commands path: decided
/// once, by [`run_without_generated_commands`] (or a device worker's
/// [`decide_generated_commands`]) or by the first read without it (enabled).
static GENERATED_COMMANDS: OnceLock<bool> = OnceLock::new();

/// Open every engine context this process opens without `VK_EXT_device_generated_commands`
/// (a command's `--without-generated-commands`), so the device replay takes K14c's path on a device
/// that has the extension. A measurement of that path, never the default's evidence: the engine
/// executor compares and returns its observation but does not store it. Call it once, before any
/// device opens.
pub fn run_without_generated_commands() -> Result<(), String> {
    decide_generated_commands(false)
}

/// Decide this process's generated-commands setting ([`run_without_generated_commands`]); a
/// device worker applies its parent's before it opens a device.
pub(crate) fn decide_generated_commands(enabled: bool) -> Result<(), String> {
    if *GENERATED_COMMANDS.get_or_init(|| enabled) == enabled {
        Ok(())
    } else {
        Err("the generated-commands setting is already decided: pass \
             --without-generated-commands once, before any device opens"
            .into())
    }
}

/// Whether this process's engine contexts enable `VK_EXT_device_generated_commands` where
/// their device supports it ([`run_without_generated_commands`]); reading it decides that they do
/// if nothing was set.
pub fn requested_generated_commands() -> bool {
    *GENERATED_COMMANDS.get_or_init(|| true)
}

/// The policy an engine `Device` of this process is created under: the default policy with
/// Metal's answer wherever the policy offers the device's instead (`DevicePolicy::exact_answers`:
/// the runtime's heap compression, completion, batch order, render lowering, host-page textures
/// and pipeline waits, and each `half` operation rounded on a device without `shaderFloat16`), as
/// Metal's answers are what the candidate is compared with. Where the device has a native form it
/// changes no translation: the device's arithmetic, sampling and conversions are the only forms
/// there, recorded as precision notes where they can differ from Metal's.
pub fn requested_policy() -> DevicePolicy {
    DevicePolicy::default().exact_answers()
}

/// Write every validation-layer message that fails a row to stderr as the layer reports it, for
/// this whole process (the engine device worker's setting).
pub(crate) fn echo_validation_messages() {
    platform::echo_validation_messages();
    runtime_case::echo_messages();
}

fn validate_backend_host(backend: Backend) -> Result<(), String> {
    if backend == Backend::Moltenvk && !cfg!(target_os = "macos") {
        return Err("MoltenVK candidate execution requires macOS".into());
    }
    if backend == Backend::Vulkan && cfg!(target_os = "macos") {
        return Err(
            "native Vulkan candidate execution is not available on macOS; use corpus-moltenvk"
                .into(),
        );
    }
    Ok(())
}

pub(crate) mod device_facts;
pub mod engine;
pub mod runtime_case;
pub mod runtime_samples;
pub(crate) mod unadvertised_stage;
pub(crate) mod unsupported_set;

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod platform;
/// The runtime path's restriction to a committed profile for a lane test
/// (`runtime_lane/absent_reads.rs`).
#[cfg(all(
    test,
    any(m2v_gpu, m2v_cpu_icd),
    any(target_os = "linux", target_os = "macos")
))]
pub(crate) use platform::{lane_restriction, profiles_layer_installed, LaneRestriction};
pub(crate) use platform::{runtime_restriction, Restricting, RuntimeRestriction};

/// What the runtime path's second check of a profile restriction found on its device
/// (`RuntimeRestriction::check`): the description the executor's context translates for under the
/// profile, the members where the description the runtime read through the profiles layer differs
/// from it (the kept ones, which the layer reads from the device), and the environment's
/// `profiles_layer` record.
pub(crate) struct RestrictedDevice {
    pub(crate) translated: metal2vulkan::device::DeviceDescription,
    pub(crate) kept: Vec<String>,
    pub(crate) record: serde_json::Value,
}
/// The loader-word guard, for the runtime lane's tests that make their own devices
/// (`runtime_lane/presenting.rs`).
#[cfg(all(
    test,
    any(m2v_gpu, m2v_cpu_icd),
    any(target_os = "linux", target_os = "macos")
))]
pub(crate) use platform::LoaderWord;

#[cfg(test)]
pub(crate) mod tests;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use super::*;
    use crate::check::admission::ExecutorError;

    pub fn restrict_devices_to(_stem: &str) -> Result<(), String> {
        Err("candidate execution is supported only on Linux and macOS".into())
    }

    pub fn capture_formats(_device_type: DeviceType) -> Result<serde_json::Value, String> {
        Err("device captures are supported only on Linux and macOS".into())
    }

    pub fn probe_runtime(_device_type: DeviceType) -> Result<serde_json::Value, String> {
        Err("device probes are supported only on Linux and macOS".into())
    }

    pub fn venue_check(
        _device_type: DeviceType,
        _venue: &crate::c3::VenueProfile,
        _allowed: &[crate::c3::DeviceIdentity],
    ) -> Result<crate::c3::VenueCheck, String> {
        Err("candidate execution is supported only on Linux and macOS".into())
    }

    pub fn create_acceptance_pipeline(
        _device: DeviceType,
        _pipeline: &crate::acceptance::RowPipeline,
    ) -> Result<crate::acceptance::Acceptance, crate::acceptance::AcceptanceError> {
        Err(crate::acceptance::AcceptanceError::DeviceUnavailable(
            "pipeline creation is supported only on Linux and macOS".into(),
        ))
    }

    pub fn execute_engine(
        _device: DeviceType,
        _case_id: &str,
        _case: &super::engine::ComputeCase,
        _compile: impl FnOnce(
            &crate::worker::compile::WorkerDevice,
            &metal2vulkan::bind::BindShape,
        ) -> Result<
            super::engine::CompiledCase<metal2vulkan::pipeline::ComputePipeline>,
            ExecutorError,
        >,
        _children: impl FnMut(
            &crate::worker::compile::WorkerDevice,
            u64,
            &metal2vulkan::bind::BindShape,
            &super::engine::ChildCommand,
        ) -> Result<super::engine::CompiledChild, ExecutorError>,
        _registered: &[metal2vulkan::bind::ReplayVariant],
        _backend: Backend,
    ) -> super::engine::RunAttempt {
        super::engine::RunAttempt::of(Err(
            "candidate execution is supported only on Linux and macOS".into(),
        ))
    }

    pub fn execute_engine_render(
        _device: DeviceType,
        _case_id: &str,
        _case: &super::engine::graphics::RenderCase,
        _compile: impl FnOnce(
            &crate::worker::compile::WorkerDevice,
            &metal2vulkan::bind::BindShape,
        ) -> Result<
            super::engine::CompiledCase<metal2vulkan::pipeline::RenderPipeline>,
            ExecutorError,
        >,
        _children: impl FnMut(
            &crate::worker::compile::WorkerDevice,
            u64,
            &metal2vulkan::bind::BindShape,
            &super::engine::ChildCommand,
        ) -> Result<super::engine::CompiledChild, ExecutorError>,
        _backend: Backend,
    ) -> super::engine::RunAttempt {
        super::engine::RunAttempt::of(Err(
            "candidate execution is supported only on Linux and macOS".into(),
        ))
    }

    pub fn echo_validation_messages() {}

    pub fn requested_restriction() -> Option<String> {
        None
    }

    pub fn check_presentable(_stem: &str) -> Result<(), String> {
        Err("profile restriction is supported only on Linux and macOS".into())
    }

    pub fn run_robustness_off() -> Result<(), String> {
        Err("candidate execution is supported only on Linux and macOS".into())
    }

    pub fn requested_leg() -> crate::coverage::Leg {
        crate::coverage::Leg::Stated
    }

    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub fn opened_worker_device(
        _device: DeviceType,
    ) -> Result<crate::worker::compile::WorkerDevice, ExecutorError> {
        Err("candidate execution is supported only on Linux and macOS".into())
    }

    /// No restriction runs here ([`restrict_devices_to`] refuses one).
    pub(crate) enum Restricting {
        Takes(Box<RuntimeRestriction>),
        Refused(Box<crate::candidate::engine::RunAttempt>),
    }

    /// No value: no restriction runs here.
    pub(crate) enum RuntimeRestriction {}

    impl RuntimeRestriction {
        pub(crate) fn stem(&self) -> &str {
            match *self {}
        }

        pub(crate) fn restrict(
            &self,
            _options: &mut metal2vulkan_runtime::DeviceOptions,
        ) -> Result<(), String> {
            match *self {}
        }

        pub(crate) fn check(
            &self,
            _device: &metal2vulkan_runtime::Device,
        ) -> Result<super::RestrictedDevice, String> {
            match *self {}
        }

        pub(crate) fn refused_on(
            &self,
            _executor: &str,
        ) -> Result<crate::worker::device::OpenedDevice, String> {
            match *self {}
        }
    }

    pub(crate) fn runtime_restriction(
        _device_type: DeviceType,
        _executor: &str,
    ) -> Result<Option<Restricting>, String> {
        Ok(None)
    }
}
