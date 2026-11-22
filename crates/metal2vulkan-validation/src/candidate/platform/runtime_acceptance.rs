//! Driver acceptance (L4) of kernel rows through the runtime, as a guest's kernels are made: on
//! the sweep's one runtime device ([`RuntimeKernels`], opened once for the sweep's device kind,
//! restricted to the process's profile where it is, `runtime_restriction`), each row's AIR as a
//! library, its function specialized with the row's function-constant values, one
//! `ComputePipelineState`, and one dispatch of one representative threadgroup, encoded into a
//! command buffer that is never committed, so nothing executes.
//!
//! The representative threadgroup is the state's thread execution width of threads (32), or its
//! `maxTotalThreadsPerThreadgroup` where that is fewer. The runtime decides which of the state's
//! modules that threadgroup runs, as it does for a guest's dispatch, and the dispatch makes the
//! `VkPipeline` the driver is asked for: its creation is the runtime's count of it
//! (`Metrics::pipeline_creation_nanos`). A kernel that writes no memory records nothing and is made
//! no pipeline (`Metrics::dispatches_without_effects`): [`Acceptance::NoEffect`].
//!
//! The make is held to the per-attempt ceilings (`runtime::within_ceilings`), either past it a
//! tooling error. The device's instance is made under `VK_LAYER_KHRONOS_validation`, whose
//! variables are set only while it is made (`runtime::under_validation_variables`); the layer logs
//! to the process's one log for the device's life. A row's messages are what the log gained while
//! the row's state was made and dispatched and its objects dropped: any makes the row
//! [`Acceptance::InvalidUsage`]. A message logged while the device opens stops the sweep before
//! any row, since no row on that device would be evidence; one logged at its teardown fails the
//! sweep ([`RuntimeKernels::finish`]). The runtime's typed refusal of a row on the device (the
//! translator's, a known gap, a limit) is [`Acceptance::RuntimeRefused`]; a Vulkan error of the
//! make or the dispatch, the driver's, [`Acceptance::Refused`].

use super::runtime::{self, sampler_grant, VALIDATION_LAYER};
use super::runtime_restriction;
use crate::acceptance::{Acceptance, AcceptanceError, RuntimeRow};
use crate::engine::bounded_detail;
use crate::gpu::DeviceType;
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use metal2vulkan_runtime as rt;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Duration;

/// The executor a row's environment names.
const EXECUTOR: &str = "runtime-acceptance";

/// The sweep's one runtime device of its kind, every kernel row's state made and dispatched on
/// it, as the module documentation says.
pub struct RuntimeKernels {
    // Dropped before its instance.
    device: rt::Device,
    instance: rt::Instance,
    device_type: DeviceType,
    log: Log,
}

impl RuntimeKernels {
    /// The runtime's device of `device_type`, under the validation layer and the process's
    /// restriction, its whole sampler range granted once (a grant lasts as long as the device),
    /// which every row's constexpr samplers are numbered in. Only a device that cannot be used (none of the kind, the hang tripwire, a
    /// restriction it cannot take, a message the layer logs while it opens) is
    /// [`AcceptanceError::DeviceUnavailable`].
    pub fn open(device_type: DeviceType) -> Result<RuntimeKernels, AcceptanceError> {
        let unavailable = AcceptanceError::DeviceUnavailable;
        if device_type == DeviceType::Gpu {
            crate::gpu::tripwire::refuse_if_tripped().map_err(unavailable)?;
        }
        let layered =
            match runtime_restriction::layered(device_type, EXECUTOR).map_err(unavailable)? {
                Ok(layered) => layered,
                Err(refused) => return Err(unavailable(refused.to_string())),
            };
        let log = Log::new();
        let (instance, device) = runtime::under_validation_variables(&log.path, || {
            layered.instance(|instance| {
                let (device, _, _) =
                    runtime::made_device(instance, device_type, EXECUTOR, &layered)?;
                Ok((instance.clone(), device))
            })
        })
        .map_err(unavailable)?;
        // A grant lasts as long as the device: every row's constexpr samplers are numbered in it.
        sampler_grant(&device).map_err(|error| unavailable(error.to_string()))?;
        let mut kernels = RuntimeKernels {
            device,
            instance,
            device_type,
            log,
        };
        if let Some(messages) = kernels.log.unread().map_err(unavailable)? {
            return Err(unavailable(format!(
                "{VALIDATION_LAYER} reported messages while the runtime's device opened, so no \
                 row on it is evidence: {}",
                bounded_detail(&messages)
            )));
        }
        Ok(kernels)
    }

    /// Make `row`'s pipeline state on the sweep's device and encode one dispatch of it, and report
    /// what the runtime, the driver and the layer answered. The hang tripwire set is
    /// [`AcceptanceError::DeviceUnavailable`].
    pub fn accept(&mut self, row: &RuntimeRow<'_>) -> Result<Acceptance, AcceptanceError> {
        if self.device_type == DeviceType::Gpu {
            crate::gpu::tripwire::refuse_if_tripped()
                .map_err(AcceptanceError::DeviceUnavailable)?;
        }
        let result = made(&self.device, row);
        match (result, self.log.unread().map_err(AcceptanceError::Tooling)?) {
            (Ok(_), Some(messages)) => {
                Ok(Acceptance::InvalidUsage(vec![bounded_detail(&messages)]))
            }
            (result, _) => result,
        }
    }

    /// Drop the device and wait for its teardown: `Err` past the wait, or with what the layer
    /// logged after the last row.
    pub fn finish(self) -> Result<(), String> {
        let RuntimeKernels {
            device,
            instance,
            mut log,
            ..
        } = self;
        drop(device);
        runtime::torn_down(&instance)?;
        match log.unread()? {
            None => Ok(()),
            Some(messages) => Err(format!(
                "{VALIDATION_LAYER} reported messages at the runtime's device's teardown: {}",
                bounded_detail(&messages)
            )),
        }
    }
}

/// The validation layer's log for the device's life, removed with it, and how much of it the
/// sweep has read.
struct Log {
    path: PathBuf,
    read: u64,
}

impl Log {
    fn new() -> Log {
        let path = runtime::validation_log();
        let _ = std::fs::remove_file(&path);
        Log { path, read: 0 }
    }

    /// What the layer logged since the last read, if anything.
    fn unread(&mut self) -> Result<Option<String>, String> {
        let failed = |error: std::io::Error| format!("read {}: {error}", self.path.display());
        let mut file = match std::fs::File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(failed(error)),
        };
        file.seek(SeekFrom::Start(self.read)).map_err(failed)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(failed)?;
        self.read += bytes.len() as u64;
        let text = String::from_utf8_lossy(&bytes);
        Ok((!text.trim().is_empty()).then(|| text.into_owned()))
    }
}

impl Drop for Log {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// `row`'s state made on `device`, and its dispatch encoded.
fn made(device: &rt::Device, row: &RuntimeRow<'_>) -> Result<Acceptance, AcceptanceError> {
    let function =
        match Library::from_air(row.bitcode).and_then(|library| library.function(row.entry)) {
            Ok(function) => function,
            Err(error) => {
                return Ok(Acceptance::RuntimeRefused(format!(
                    "the translator: {error}"
                )))
            }
        };
    let mut function = FunctionDescriptor::new(function).with_specialized(true);
    function.constants = row.constants.clone();
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(function));
    let before = device.counters().metrics();
    let made = runtime::within_ceilings(|numbered| {
        device.new_compute_pipeline_state(&descriptor, |samplers| numbered(samplers))
    })
    .map_err(AcceptanceError::Tooling)?;
    let state = match made {
        Ok(state) => state,
        Err(error) => return answer(error),
    };
    let width = state
        .thread_execution_width()
        .min(state.max_total_threads_per_threadgroup())
        .max(1);
    if let Err(error) = dispatched(device, &state, [width, 1, 1]) {
        return answer(error);
    }
    let after = device.counters().metrics();
    if after.dispatches_without_effects > before.dispatches_without_effects {
        return Ok(Acceptance::NoEffect);
    }
    let nanos = after
        .pipeline_creation_nanos
        .saturating_sub(before.pipeline_creation_nanos);
    Ok(Acceptance::Created {
        create: Duration::from_nanos(nanos),
    })
}

/// One threadgroup of `threads_per_threadgroup` of `state` encoded into a command buffer of
/// `device` that is dropped uncommitted: the runtime gives back what it recorded, and nothing runs.
fn dispatched(
    device: &rt::Device,
    state: &rt::ComputePipelineState,
    threads_per_threadgroup: [u32; 3],
) -> Result<(), rt::Error> {
    let queue = device.new_command_queue(1)?;
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder(rt::DispatchType::Serial)?;
    encoder.set_compute_pipeline_state(state)?;
    encoder.dispatch_threadgroups([1, 1, 1], threads_per_threadgroup)?;
    encoder.end_encoding();
    Ok(())
}

/// The runtime's `error` for a row: a Vulkan error, the driver's refusal; the translator's
/// refusal, a known gap or a limit, the runtime's; any other a tooling error.
fn answer(error: rt::Error) -> Result<Acceptance, AcceptanceError> {
    match error {
        rt::Error::Vulkan { .. } => Ok(Acceptance::Refused(error.to_string())),
        rt::Error::Translation { .. } | rt::Error::Unsupported(_) | rt::Error::Limit(_) => {
            Ok(Acceptance::RuntimeRefused(error.to_string()))
        }
        error => Err(AcceptanceError::Tooling(error.to_string())),
    }
}
