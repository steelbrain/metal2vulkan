//! The engine compute executor through the runtime (`metal2vulkan-runtime`).
//!
//! A kernel case the engine routes here (`engine::runs_through_runtime`) runs as a consumer runs
//! it, on the runtime's public API alone: its instance, the loader's first physical device of the
//! requested kind (as the executor's own context selects one), a device with the default
//! options, a library and a function from the row's AIR with the case's constants and linked
//! functions, one compute pipeline state, one Shared buffer per bound buffer and per
//! argument-buffer buffer, each bound and argument-buffer texture and sampler as the Metal oracle
//! makes them (`runtime_textures`), and one command buffer of one compute encoder and one
//! dispatch, waited for under the hang protocol, then read back. An argument buffer holds each
//! member's `gpuAddress` or `gpuResourceID` at its field, written into the owner's bytes before
//! the owner is made, as a program writes them (`engine::argument_buffers::owner_bytes`), and the
//! encoder uses each member buffer and texture for reading and writing (`useResource:usage:`), as
//! the oracle does.
//!
//! The process runs it under `VK_LAYER_KHRONOS_validation`, which the loader's and the layer's
//! own variables enable before the instance is made: core and synchronization validation, every
//! message reported, each error and warning written to the process's log, removed once read. A
//! device that lists no active validation tool fails the run; so does any message in the log once
//! the device is gone, as a message fails a row of the executor's own context.
//!
//! The pipeline's make (`new_compute_pipeline_state`) is held to the per-attempt ceilings: its
//! peak live allocations to 500 MiB (`worker::alloc::bounded`: an allocation past it ends the
//! process) and its wall time to 20 s. The driver's creation of its pipelines is held to the
//! case's bound (`budgets::creation_within`). Every pipeline the dispatch runs is made before the
//! submission (`ComputePipelineState::prepare_dispatch`).
//!
//! The row records what the runtime answers: the state's precision notes and the runtime's
//! fingerprint (`engine::runtime_fingerprint`). It records no SPIR-V, status, subgroup shape,
//! containment or artifact: the runtime keeps its modules to itself. Its census's strategies are
//! the float reductions the case's AIR calls (`check::interface::float_reductions`), which bound
//! `PrecisionNote::ReductionOrder` per operation, as the translator's census of the same
//! operations does.

use crate::candidate::device_facts::{active_tools, ActiveTool, RECORDED_LAYERS};
use crate::candidate::engine::argument_buffers::{self, FieldKey};
use crate::candidate::engine::{
    product_refusal, runtime_fingerprint, select_output, ComputeCase, ComputeOutput, ComputeRun,
    RefusedOn, RunAttempt, RuntimeKernel,
};
use crate::case::Stage;
use crate::check::admission::{ExecutorError, NotSupported};
use crate::coverage::{ExecutedCensus, ExecutionClass, HostClass, Leg, ModuleCensus};
use crate::gpu::wait::new_runtime::{wait_until_retired, Retired};
use crate::gpu::wait::{with_submission_label, SubmissionBackend, SubmissionLabel};
use crate::gpu::DeviceType;
use crate::hash::sha256_bytes;
use crate::observation::TRANSLATOR_FINGERPRINT;
use ash::vk;
use metal2vulkan::bind::Dispatch;
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, NumberedSampler, StageSource,
};
use metal2vulkan_runtime as rt;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::path::Path;
use std::time::{Duration, Instant};

/// The executor a row's environment names.
const EXECUTOR: &str = "runtime-compute";

/// The validation layer.
const VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";

/// How long the run waits for the runtime's device to tear down once its last handle is gone,
/// before the validation log is read.
const TEARDOWN: Duration = Duration::from_secs(60);

/// Run `case`, whose kernel and buffers are `kernel`'s, through the runtime on the device of
/// `device_type`, as the module documentation says. Every refusal the runtime types is typed: the
/// translator's refusal of the kernel as the product's ([`product_refusal`]), named with the
/// device it opened. A known gap of the runtime fails the run: the case's structure classified it
/// as one the runtime runs. Every other error, and any validation message, fails it.
pub fn execute_runtime(
    device_type: DeviceType,
    case_id: &str,
    case: &ComputeCase,
    kernel: RuntimeKernel<'_>,
) -> RunAttempt {
    // One log per process, at one path, so the layer writes this run's messages there whether it
    // reads its variables at each instance or once; a log of an earlier run is gone first.
    let log = std::env::temp_dir().join(format!(
        "metal2vulkan-validation-runtime-{}.log",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&log);
    validate_process(&log);
    let attempt = match on_device(device_type, case_id, case, &kernel) {
        Ok(attempt) => attempt,
        Err(error) => RunAttempt::of(Err(ExecutorError::Failed(error))),
    };
    let messages = std::fs::read_to_string(&log);
    let _ = std::fs::remove_file(&log);
    match messages {
        Ok(messages) if !messages.trim().is_empty() => {
            let ran = match &attempt.result {
                Ok(_) => String::new(),
                Err(error) => format!(" (the run also failed: {error})"),
            };
            RunAttempt::of(Err(ExecutorError::Failed(format!(
                "{VALIDATION_LAYER} reported messages; the run is not evidence{ran}: {}",
                crate::engine::bounded_detail(&messages)
            ))))
        }
        _ => attempt,
    }
}

/// Enable the validation layer for this process's next instance, through the loader's and the
/// layer's variables (`docs/VALIDATION.md`), its messages logged to `log`: the loader and the
/// layer read them when an instance is made. A device worker makes one instance at a time, on
/// this thread, and no other thread reads the environment.
fn validate_process(log: &Path) {
    for (name, value) in [
        ("VK_INSTANCE_LAYERS", VALIDATION_LAYER),
        ("VK_KHRONOS_VALIDATION_REPORT_FLAGS", "error,warn"),
        (
            "VK_KHRONOS_VALIDATION_DEBUG_ACTION",
            "VK_DBG_LAYER_ACTION_LOG_MSG",
        ),
        ("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "true"),
        ("VK_KHRONOS_VALIDATION_ENABLE_MESSAGE_LIMIT", "false"),
    ] {
        std::env::set_var(name, value);
    }
    std::env::set_var("VK_KHRONOS_VALIDATION_LOG_FILENAME", log);
}

/// The run on the device of `device_type`, every object of it gone before this returns, and the
/// runtime's device torn down. `Err` where no device of the kind opens under the layer.
fn on_device(
    device_type: DeviceType,
    case_id: &str,
    case: &ComputeCase,
    kernel: &RuntimeKernel<'_>,
) -> Result<RunAttempt, String> {
    opened(device_type, |device, identity, environment| {
        let result = run(
            device,
            device_type,
            case_id,
            case,
            kernel,
            identity,
            environment,
        );
        let refused_on = matches!(result, Err(ExecutorError::NotSupported(_))).then(|| RefusedOn {
            environment: environment.clone(),
            translator_fingerprint: runtime_fingerprint(),
            artifact: None,
        });
        RunAttempt { result, refused_on }
    })
}

/// `body` on the runtime's device of `device_type`, with the identity and environment a row
/// records of it, every object of it gone before this returns, and the device torn down. `Err`
/// where no device of the kind opens under the layer.
fn opened<T>(
    device_type: DeviceType,
    body: impl FnOnce(&rt::Device, &Identity, &serde_json::Value) -> T,
) -> Result<T, String> {
    if device_type == DeviceType::Gpu {
        crate::gpu::tripwire::refuse_if_tripped()?;
    }
    let instance =
        rt::Instance::new(&[]).map_err(|error| format!("the runtime's instance: {error}"))?;
    let value = {
        let (physical, identity) = select(&instance, device_type)?;
        let (entry, vk_instance) = instance.raw();
        // SAFETY: `physical` is one of the instance's physical devices, and the instance is live.
        let tools = unsafe { active_tools(entry, vk_instance, physical.raw()) }?;
        if !tools.iter().any(|tool| tool.layer == VALIDATION_LAYER) {
            return Err(format!(
                "{} lists no active validation tool: the loader found no {VALIDATION_LAYER}, and \
                 a run never proceeds without it",
                identity.device
            ));
        }
        let environment = environment(&identity, &tools);
        let device = metal2vulkan_runtime::Device::new(&physical, rt::DeviceOptions::default())
            .map_err(|error| format!("the runtime's device on {}: {error}", identity.device))?;
        body(&device, &identity, &environment)
    };
    if !instance.wait_until_devices_torn_down(TEARDOWN) {
        return Err(format!(
            "the runtime's device did not tear down within {} s",
            TEARDOWN.as_secs()
        ));
    }
    Ok(value)
}

/// `body` on the runtime's device of `device_type` under the validation layer, as a run opens it
/// ([`execute_runtime`]): what a device test needs of the runtime's answer to one call, beside a
/// run. `Err` where no device of the kind opens, or where the layer reported a message.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
pub fn on_runtime_device<T>(
    device_type: DeviceType,
    body: impl FnOnce(&rt::Device) -> T,
) -> Result<T, String> {
    let log = std::env::temp_dir().join(format!(
        "metal2vulkan-validation-runtime-{}.log",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&log);
    validate_process(&log);
    let value = opened(device_type, |device, _, _| body(device));
    let messages = std::fs::read_to_string(&log);
    let _ = std::fs::remove_file(&log);
    match messages {
        Ok(messages) if !messages.trim().is_empty() => Err(format!(
            "{VALIDATION_LAYER} reported messages: {}",
            crate::engine::bounded_detail(&messages)
        )),
        _ => value,
    }
}

/// The make of `entry`'s compute pipeline state from `bitcode` on `device`, with no constant, no
/// linked function and no sampler it numbers: the runtime's own answer, a device test's to judge.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
pub fn make_kernel(
    device: &rt::Device,
    bitcode: &[u8],
    entry: &str,
) -> Result<rt::ComputePipelineState, rt::Error> {
    let function = Library::from_air(bitcode)
        .and_then(|library| library.function(entry))
        .map_err(rt::Error::Translation)?;
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(
        FunctionDescriptor::new(function).with_specialized(true),
    ));
    device.new_compute_pipeline_state(&descriptor, |samplers: &[NumberedSampler]| {
        Ok(vec![0; samplers.len()])
    })
}

/// What a row records of the device that ran it.
struct Identity {
    device: String,
    device_id: u32,
    vendor_id: u32,
    driver_version: u32,
    api_version: u32,
    device_type: vk::PhysicalDeviceType,
    driver_id: i32,
    driver_name: String,
    driver_info: String,
}

/// The loader's first physical device of `device_type`'s kind, as the executor's own context
/// selects one: a CPU implementation for a CPU run, a discrete, integrated or virtual GPU for a
/// GPU run, never one for the other.
fn select(
    instance: &rt::Instance,
    device_type: DeviceType,
) -> Result<(rt::PhysicalDevice, Identity), String> {
    let (_, vk_instance) = instance.raw();
    let physical_devices = instance
        .physical_devices()
        .map_err(|error| format!("the runtime's physical devices: {error}"))?;
    for physical in physical_devices {
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: `physical` is one of the instance's physical devices, the instance is live and
        // at the floor's version, which has `vkGetPhysicalDeviceProperties2`.
        unsafe { vk_instance.get_physical_device_properties2(physical.raw(), &mut properties) };
        let device = properties.properties;
        let text = |name: Result<&CStr, std::ffi::FromBytesUntilNulError>| {
            name.map_or_else(
                |_| String::new(),
                |name| name.to_string_lossy().into_owned(),
            )
        };
        let kind = match device_type {
            DeviceType::Cpu => device.device_type == vk::PhysicalDeviceType::CPU,
            DeviceType::Gpu => matches!(
                device.device_type,
                vk::PhysicalDeviceType::DISCRETE_GPU
                    | vk::PhysicalDeviceType::INTEGRATED_GPU
                    | vk::PhysicalDeviceType::VIRTUAL_GPU
            ),
        };
        if !kind {
            continue;
        }
        let identity = Identity {
            device: text(device.device_name_as_c_str()),
            device_id: device.device_id,
            vendor_id: device.vendor_id,
            driver_version: device.driver_version,
            api_version: device.api_version,
            device_type: device.device_type,
            driver_id: driver.driver_id.as_raw(),
            driver_name: text(driver.driver_name_as_c_str()),
            driver_info: text(driver.driver_info_as_c_str()),
        };
        return Ok((physical, identity));
    }
    Err(format!(
        "no {device_type:?} device: the runtime's instance lists none of that kind, and another \
         kind is never substituted"
    ))
}

/// The environment of a row on the device `identity` names, with the tools active on it: its
/// identity as the executor's own context records it, the executor, the validation it ran under,
/// no profiles-layer restriction, and the stated leg.
fn environment(identity: &Identity, tools: &[ActiveTool]) -> serde_json::Value {
    serde_json::json!({
        "device": identity.device,
        "device_id": identity.device_id,
        "vendor_id": identity.vendor_id,
        "driver_version": identity.driver_version,
        "api_version": identity.api_version,
        "driver_id": identity.driver_id,
        "driver_name": identity.driver_name,
        "driver_info": identity.driver_info,
        "architecture": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "tools": tools,
        "executor": EXECUTOR,
        "layers": RECORDED_LAYERS,
        "synchronization_validation": true,
        "profiles_layer": null,
        "robustness": {"leg": Leg::Stated},
    })
}

/// The runtime's `error` as the executor's: the translator's refusal as the product's, a pixel
/// format the runtime makes no image of on this device as the device's (`NotEnabled`, as the
/// executor's own context records a format its query lacks), every other error a failure
/// ([`execute_runtime`]).
fn runtime_error(what: &str, error: rt::Error) -> ExecutorError {
    match error {
        rt::Error::Translation(error) => product_refusal(&error),
        rt::Error::Unsupported(gap @ rt::Gap::TextureFormat { .. }) => {
            ExecutorError::NotSupported(NotSupported::NotEnabled(vec![format!("{what}: {gap}")]))
        }
        rt::Error::Unsupported(gap) => ExecutorError::Failed(format!(
            "{what}: the runtime refused a case its structure classifies as one it runs \
             ({gap:?})"
        )),
        error => ExecutorError::Failed(format!("{what}: {error}")),
    }
}

/// Make `case`'s pipeline state on `device`, bind its buffers, dispatch it once, wait, and read
/// its output back.
fn run(
    device: &rt::Device,
    device_type: DeviceType,
    case_id: &str,
    case: &ComputeCase,
    kernel: &RuntimeKernel<'_>,
    identity: &Identity,
    environment: &serde_json::Value,
) -> Result<ComputeRun, ExecutorError> {
    // Each constexpr sampler the kernel numbers takes an index of the device's sampler range from
    // its bottom, one per state words, as Metal gives equal samplers one ID; a read's or a
    // write's takes nothing at its index. Each bound sampler takes one from its top
    // (`runtime_textures`).
    let range = rt::IndexClass::Sampler.range();
    let _grant = device
        .grant_indices(
            rt::IndexClass::Sampler,
            range.start,
            range.end - range.start,
        )
        .map_err(|error| runtime_error("grant the sampler indices", error))?;
    let descriptor = descriptor(case, kernel)?;
    let (threads_per_group, dispatch) = match case.dispatch {
        Dispatch::Threads {
            threads,
            threads_per_group,
        } => (threads_per_group, rt::Dispatch::Threads(threads)),
        Dispatch::Threadgroups {
            groups,
            threads_per_group,
        } => (threads_per_group, rt::Dispatch::Threadgroups(groups)),
        other => {
            return Err(ExecutorError::Failed(format!(
                "a run through the runtime dispatches by threads or threadgroups, not {other:?}"
            )))
        }
    };
    let lengths: Vec<(u32, u32)> = case
        .threadgroup_lengths
        .iter()
        .map(|(&index, &length)| (index, length))
        .collect();
    let before = device.counters().metrics().pipeline_creation_nanos;
    let state = make(device, &descriptor)?;
    // Every pipeline the dispatch runs, made before anything is submitted.
    state.prepare_dispatch(dispatch, threads_per_group, &lengths);
    let created = device
        .counters()
        .metrics()
        .pipeline_creation_nanos
        .saturating_sub(before);
    crate::budgets::creation_within(
        "the case's pipelines",
        Duration::from_nanos(created),
        case.creation_bound,
    )
    .map_err(ExecutorError::Failed)?;
    let members = case
        .argument_buffer_buffers
        .iter()
        .map(|(&(owner, field), bytes)| {
            let what = format!("argument-buffer buffer {owner}+{field}");
            Ok(((owner, field), Placed::new(device, &what, bytes)?))
        })
        .collect::<Result<BTreeMap<FieldKey, Placed>, ExecutorError>>()?;
    let bound = super::runtime_textures::make(device, case, kernel, runtime_error)?;
    let mut words = bound.words();
    words.extend(
        members
            .iter()
            .map(|(&key, placed)| (key, placed.buffer.gpu_address())),
    );
    let buffers = kernel
        .buffers
        .keys()
        .map(|&index| {
            let bytes = argument_buffers::owner_bytes(&kernel.buffers, index, &words)
                .map_err(ExecutorError::Failed)?;
            Ok((
                index,
                Placed::new(device, &format!("buffer {index}"), &bytes)?,
            ))
        })
        .collect::<Result<BTreeMap<u32, Placed>, ExecutorError>>()?;
    let label = SubmissionLabel {
        case_id: case_id.into(),
        module_sha256: sha256_bytes(kernel.bitcode),
    };
    with_submission_label(label, || {
        dispatched(
            device,
            device_type,
            &state,
            (&buffers, &members, &bound),
            case,
            (dispatch, threads_per_group),
        )
    })?;
    // A texture output is read back as the oracle reads one; a run through the runtime binds no
    // attachment.
    let output = match case.output {
        ComputeOutput::Texture {
            binding,
            origin,
            dimensions,
        } => {
            let (Some(made), Some(texture)) =
                (bound.textures.get(&binding), case.textures.get(&binding))
            else {
                return Err(ExecutorError::Failed(format!(
                    "output texture {binding} is not bound"
                )));
            };
            super::runtime_textures::read_back(made, texture, origin, dimensions, runtime_error)?
        }
        ComputeOutput::ArgumentBufferTexture {
            owner,
            field_offset,
            origin,
            dimensions,
        } => {
            let key = (owner, field_offset);
            let (Some(made), Some(texture)) = (
                bound.members.get(&key),
                case.argument_buffer_textures.get(&key),
            ) else {
                return Err(ExecutorError::Failed(format!(
                    "output argument-buffer texture {owner}+{field_offset} is not made"
                )));
            };
            super::runtime_textures::read_back(made, texture, origin, dimensions, runtime_error)?
        }
        output => {
            let finals = buffers
                .iter()
                .map(|(&index, placed)| (index, placed.bytes()))
                .collect::<BTreeMap<u32, Vec<u8>>>();
            let embedded = members
                .iter()
                .map(|(&key, placed)| (key, placed.bytes()))
                .collect::<BTreeMap<FieldKey, Vec<u8>>>();
            select_output(
                output,
                &finals,
                &embedded,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .map_err(ExecutorError::Failed)?
        }
    };
    let census = ExecutedCensus {
        out_of_bounds: None,
        rate_map_pass: None,
        device: Some(match identity.device_type {
            vk::PhysicalDeviceType::CPU => ExecutionClass::CpuIcd,
            _ => ExecutionClass::Host(HostClass(metal2vulkan::vk::DriverId::from_raw(
                identity.driver_id,
            ))),
        }),
        product: TRANSLATOR_FINGERPRINT.into(),
        restricted_to: None,
        leg: Some(Leg::Stated),
        modules: vec![ModuleCensus {
            stage: Stage::Kernel,
            subgroup: None,
            strategies: kernel
                .reductions
                .iter()
                .map(|&name| name.to_owned())
                .collect(),
            precision: state.precision_notes().into_iter().collect(),
            containment: None,
        }],
        artifact: None,
    };
    Ok(ComputeRun {
        output,
        spv_sha256: None,
        translator_fingerprint: runtime_fingerprint(),
        environment: environment.clone(),
        status: None,
        comparison: census.comparison_class(),
        census,
    })
}

/// The case's `MTLComputePipelineDescriptor` as the Metal oracle makes it
/// (`metal::platform::pipelines`): its kernel from the row's library, specialized with the case's
/// constants (`newFunctionWithName:constantValues:`), linking each function of the modules it
/// links, each made with the same constants (`acceptance::linked_functions`).
fn descriptor(
    case: &ComputeCase,
    kernel: &RuntimeKernel<'_>,
) -> Result<ComputePipelineDescriptor, ExecutorError> {
    let function = Library::from_air(kernel.bitcode)
        .and_then(|library| library.function(&case.entry))
        .map_err(|error| product_refusal(&error))?;
    let libraries: Vec<&[u8]> = kernel
        .linked
        .iter()
        .map(|module| module.bitcode.as_slice())
        .collect();
    let linked: Vec<crate::acceptance::LinkedLibrary> = kernel
        .linked
        .iter()
        .map(|module| module.library.clone())
        .collect();
    let (linked, _) = crate::acceptance::linked_functions(&libraries, &linked, &case.constants)
        .map_err(|error| product_refusal(&error))?;
    let mut function = FunctionDescriptor::new(function)
        .with_specialized(true)
        .with_linked(linked);
    function.constants = case.constants.clone();
    Ok(ComputePipelineDescriptor::new(StageSource::Function(
        function,
    )))
}

/// `descriptor`'s state on `device`, made within the per-attempt ceilings: its peak live
/// allocations within 500 MiB (an allocation past it ends the process), its wall time within
/// 20 s.
fn make(
    device: &rt::Device,
    descriptor: &ComputePipelineDescriptor,
) -> Result<rt::ComputePipelineState, ExecutorError> {
    if !crate::worker::alloc::installed() {
        return Err(ExecutorError::Failed(
            "the counting allocator is not installed: the pipeline's make cannot be held to its \
             memory ceiling"
                .into(),
        ));
    }
    let mut constexprs: BTreeMap<[u64; 2], u32> = BTreeMap::new();
    let first = rt::IndexClass::Sampler.range().start;
    // A numbered sampler of a kind this run does not know fails it, once the make returns.
    let mut unknown = None;
    let numbered = |samplers: &[NumberedSampler]| {
        let ids = samplers.iter().map(|sampler| match sampler {
            NumberedSampler::Constexpr(words) => {
                let next = first + constexprs.len() as u32;
                *constexprs.entry(*words).or_insert(next)
            }
            NumberedSampler::Read | NumberedSampler::Write => 0,
            other => {
                unknown = Some(format!("{other:?}"));
                0
            }
        });
        Ok::<Vec<u32>, rt::Error>(ids.collect())
    };
    let started = Instant::now();
    let (made, peak) = crate::worker::alloc::bounded(crate::worker::MEMORY_LIMIT_BYTES, || {
        device.new_compute_pipeline_state(descriptor, numbered)
    });
    let took = started.elapsed();
    if took > crate::worker::DEADLINE {
        return Err(ExecutorError::Failed(format!(
            "the pipeline's make took {} ms, past the {} s ceiling (peak live allocations {} \
             bytes)",
            took.as_millis(),
            crate::worker::DEADLINE.as_secs(),
            peak.unwrap_or(0)
        )));
    }
    let made = made.map_err(|error| runtime_error("the pipeline's make", error))?;
    match unknown {
        Some(sampler) => Err(ExecutorError::Failed(format!(
            "the pipeline numbers a sampler of a kind a run through the runtime does not know: \
             {sampler}"
        ))),
        None => Ok(made),
    }
}

/// A buffer of the run: a Shared block of its own, and the buffer over it holding the bytes.
struct Placed {
    buffer: rt::Buffer,
    _block: rt::Block,
}

impl Placed {
    /// A Shared buffer holding `bytes`, `what` the case names it.
    fn new(device: &rt::Device, what: &str, bytes: &[u8]) -> Result<Placed, ExecutorError> {
        let length = bytes.len() as u64;
        let block = device
            .take_block(rt::MemoryClass::Shared, length)
            .map_err(|error| runtime_error(what, error))?;
        // Every byte is written below.
        let buffer = device
            .new_buffer(block.id, 0, length, None)
            .map_err(|error| runtime_error(what, error))?;
        let contents = buffer
            .contents()
            .ok_or_else(|| ExecutorError::Failed(format!("{what} has no contents")))?;
        // SAFETY: a Shared buffer's contents are its `length` bytes, mapped while it lives, and
        // nothing else writes them: no work names it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len());
        }
        Ok(Placed {
            buffer,
            _block: block,
        })
    }

    /// Its bytes now.
    fn bytes(&self) -> Vec<u8> {
        let length = self.buffer.length() as usize;
        match self.buffer.contents() {
            // SAFETY: as in `new`; the work that named it has retired.
            Some(contents) => {
                unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec()
            }
            None => Vec::new(),
        }
    }
}

/// Encode `case`'s dispatch of `state` over `buffers` and `bound`'s textures and samplers in one
/// command buffer, commit it and wait until it has retired, under the hang protocol; a command
/// buffer that did not complete fails the run.
fn dispatched(
    device: &rt::Device,
    device_type: DeviceType,
    state: &rt::ComputePipelineState,
    (buffers, members, bound): (
        &BTreeMap<u32, Placed>,
        &BTreeMap<FieldKey, Placed>,
        &super::runtime_textures::Bound,
    ),
    case: &ComputeCase,
    (dispatch, threads_per_group): (rt::Dispatch, [u32; 3]),
) -> Result<(), ExecutorError> {
    let fail = |what: &'static str| move |error: rt::Error| runtime_error(what, error);
    let queue = device.new_command_queue(1).map_err(fail("the queue"))?;
    let command_buffer = queue.command_buffer().map_err(fail("the command buffer"))?;
    let retired = Retired::watch(&command_buffer).map_err(fail("the completed handler"))?;
    let mut encoder = command_buffer
        .compute_command_encoder(rt::DispatchType::Serial)
        .map_err(fail("the compute encoder"))?;
    encoder
        .set_compute_pipeline_state(state)
        .map_err(fail("the pipeline state"))?;
    for (&index, placed) in buffers {
        encoder
            .set_buffer(Some(&placed.buffer), 0, index)
            .map_err(fail("a buffer"))?;
    }
    for (&index, texture) in &bound.textures {
        encoder
            .set_texture(Some(texture), index)
            .map_err(fail("a texture"))?;
    }
    for (&index, sampler) in &bound.samplers {
        encoder
            .set_sampler_state(Some(sampler), index)
            .map_err(fail("a sampler"))?;
    }
    let used = members
        .values()
        .map(|placed| rt::Resource::Buffer(&placed.buffer))
        .chain(bound.members.values().map(rt::Resource::Texture));
    for resource in used {
        encoder
            .use_resource(resource, rt::ResourceUsage::READ | rt::ResourceUsage::WRITE)
            .map_err(fail("an argument-buffer member's use"))?;
    }
    for (&index, &length) in &case.threadgroup_lengths {
        encoder
            .set_threadgroup_memory_length(length, index)
            .map_err(fail("a threadgroup memory length"))?;
    }
    match dispatch {
        rt::Dispatch::Threads(threads) => encoder.dispatch_threads(threads, threads_per_group),
        rt::Dispatch::Threadgroups(groups) => {
            encoder.dispatch_threadgroups(groups, threads_per_group)
        }
        rt::Dispatch::Indirect => {
            return Err(ExecutorError::Failed(
                "a run through the runtime dispatches by threads or threadgroups".into(),
            ))
        }
    }
    .map_err(fail("the dispatch"))?;
    encoder.end_encoding();
    command_buffer.commit().map_err(fail("the commit"))?;
    wait_until_retired(
        &queue,
        &command_buffer,
        &retired,
        SubmissionBackend::vulkan(device_type),
        "a dispatch through the runtime",
    );
    match command_buffer.status() {
        rt::CommandBufferStatus::Completed => Ok(()),
        status => Err(ExecutorError::Failed(format!(
            "the command buffer ended {status:?}: {:?}",
            command_buffer.error()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{rt, runtime_error};
    use crate::check::admission::{ExecutorError, NotSupported};

    /// A pixel format the runtime makes no image of on the device is the device's refusal, as the
    /// executor's own context records a format its query lacks; another gap of the runtime fails
    /// the row, as the case's structure classifies it as one the runtime runs.
    #[test]
    fn a_format_the_runtime_makes_no_image_of_is_the_devices_refusal() {
        let format = rt::Error::Unsupported(rt::Gap::TextureFormat {
            pixel_format: rt::PixelFormat::R8UnormSRGB,
        });
        assert!(matches!(
            runtime_error("texture 1", format),
            ExecutorError::NotSupported(NotSupported::NotEnabled(gaps))
                if gaps.len() == 1 && gaps[0].starts_with("texture 1: ")
        ));
        let depth = rt::Error::Unsupported(rt::Gap::FloatDepthUpload {
            pixel_format: rt::PixelFormat::Depth32Float,
        });
        assert!(matches!(
            runtime_error("texture 0 level 0", depth),
            ExecutorError::Failed(_)
        ));
    }
}
