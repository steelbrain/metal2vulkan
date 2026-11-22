//! Driver acceptance (L4) of a kernel row through the runtime, as a guest's kernel is made: the
//! row's AIR as a library, its function specialized with the row's function-constant values, one
//! `ComputePipelineState` on the sweep's device (restricted to the process's profile where it is,
//! `runtime_restriction`), and one dispatch of one representative threadgroup, encoded into a
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
//! tooling error. Each attempt opens its own device under `VK_LAYER_KHRONOS_validation`
//! (`runtime::validated`); any message the layer logs makes the row [`Acceptance::InvalidUsage`].
//! The runtime's typed refusal of the row on the device (the translator's, a known gap, a limit)
//! is [`Acceptance::RuntimeRefused`]; a Vulkan error of the make or the dispatch, the driver's,
//! [`Acceptance::Refused`].

use super::runtime::{self, sampler_grant};
use super::runtime_restriction;
use crate::acceptance::{Acceptance, AcceptanceError, RuntimeRow};
use crate::gpu::DeviceType;
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use metal2vulkan_runtime as rt;
use std::time::Duration;

/// The executor a row's environment names.
const EXECUTOR: &str = "runtime-acceptance";

/// Make `row`'s pipeline state on the runtime's device of `device_type` and encode one dispatch of
/// it, as the module documentation says, and report what the runtime and the driver did. Only
/// opening the device (none of the kind, the hang tripwire, a restriction it cannot take) is
/// [`AcceptanceError::DeviceUnavailable`].
pub fn create_runtime_kernel(
    device_type: DeviceType,
    row: &RuntimeRow<'_>,
) -> Result<Acceptance, AcceptanceError> {
    let (created, messages) = runtime::validated(|| attempt(device_type, row));
    match (created, messages) {
        (Ok(_), Some(messages)) => Ok(Acceptance::InvalidUsage(vec![
            crate::engine::bounded_detail(&messages),
        ])),
        (created, _) => created,
    }
}

/// [`create_runtime_kernel`]'s attempt, under the validation layer.
fn attempt(device_type: DeviceType, row: &RuntimeRow<'_>) -> Result<Acceptance, AcceptanceError> {
    let unavailable = AcceptanceError::DeviceUnavailable;
    if device_type == DeviceType::Gpu {
        crate::gpu::tripwire::refuse_if_tripped().map_err(unavailable)?;
    }
    let layered = match runtime_restriction::layered(device_type, EXECUTOR).map_err(unavailable)? {
        Ok(layered) => layered,
        Err(refused) => return Err(AcceptanceError::DeviceUnavailable(refused.to_string())),
    };
    let opened = layered.instance(|instance| {
        runtime::opened(instance, device_type, EXECUTOR, &layered, |device, _, _| {
            made(device, row)
        })
    });
    opened.map_err(unavailable)?
}

/// `row`'s state made on `device`, and its dispatch encoded.
fn made(device: &rt::Device, row: &RuntimeRow<'_>) -> Result<Acceptance, AcceptanceError> {
    let _grant =
        sampler_grant(device).map_err(|error| AcceptanceError::Tooling(error.to_string()))?;
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
