//! The engine compute executor's Vulkan half.
//!
//! It is a consumer of the binder and nothing else: the `Device` is derived from the opened
//! device's own queried description, the pipeline layout is the artifact's
//! ([`LayoutObjects`], shared with L4 acceptance), and every descriptor write, range, push byte,
//! push hole, root-table byte, specialization, and group count comes from
//! `ComputePipeline::plan` and `BindingPlan::instantiate_dispatch`. The context runs under `VK_LAYER_KHRONOS_validation`; a message fails the row. The
//! pipeline comes from the caller's compile (a killable compile worker in every tool), never from a
//! compile in this process that holds the device.

use super::acceptance::{unmet_requirements, LayoutError, LayoutObjects};
use super::buffers::{create_guarded_host_buffer, create_host_buffer, HostBuffer};
use super::device::{
    refused_restriction, with_validated_context, RefusedRestriction, VulkanContext,
};
use super::plan_resources::{padded, Guest, PlanResources, TextureKey};
use crate::candidate::engine::argument_buffers::{
    field_words, owner_bytes, records_storage, validate_records,
};
use crate::candidate::engine::guards::{
    buffer_target, embedded_guard_reach, guard_reach, texel_guard_reach, with_escaped, GuardCount,
    GuardReach, Placement,
};
use crate::candidate::engine::tables::TableBuffers;
use crate::candidate::engine::{
    bind_refusal, bind_state, compute_shape, engine_fingerprint, out_of_bounds_run,
    product_refusal, records_ids, resources, run_allocations, sampler_ids, select_output,
    status_verdict, synthetic_used_addresses, texture_tokens, BufferAddress, CaseStructure,
    CompiledCase, ComputeCase, ComputeOutput, ComputeRun, RunAllocation, RunAttempt,
};
use crate::case::out_of_bounds::OutOfBoundsResource;
use crate::check::admission::{ExecutorError, NotSupported};
use crate::coverage::{ExecutedCensus, ModuleCensus};
use crate::executor_contract::out_of_bounds::{self, OutOfBoundsRun};
use crate::gpu::wait::{with_submission_label, SubmissionLabel};
use crate::gpu::DeviceType;
use crate::hash::sha256_bytes;
use crate::literal::LiteralTexture;
use crate::observation::{Backend, TRANSLATOR_FINGERPRINT};
use crate::worker::compile::WorkerDevice;
use ash::vk;
use metal2vulkan::bind::BindShape;
use metal2vulkan::bind::BindingPlan;
use metal2vulkan::bind::{
    DispatchCommands, DispatchRegion, ExecutionStatus, NeutralResource, PushHole, ReplayVariant,
    RootTable, SamplerDescriptor,
};
use metal2vulkan::device::VkFeature;
use metal2vulkan::device::{Device, DeviceDescription, DeviceProfile};
use metal2vulkan::pipeline::{ComputePipeline, StageModule};
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::sync::Arc;
use std::time::Duration;

mod device_replay;
mod draw_replay;
mod replay;
use crate::candidate::engine::{ChildCommand, CompiledChild};
pub(super) use device_replay::{replay_commands, OnDevice, ReplayPath};
pub(in crate::candidate) use replay::CompileChild;
pub(super) use replay::{Live, Replay};

/// One region's push-constant writes: `(stages, offset, bytes)`, holes filled.
type PushWrites = Vec<(vk::ShaderStageFlags, u32, Vec<u8>)>;

/// Compile `case` for the device `device_type` opens with `compile` (given the opened device's
/// description and policy, and the shape the dispatch binds, which decides the lowering inputs; it
/// returns the pipeline and its visible-function tables' bytes), plan and execute its dispatch, and
/// read the status before the output. The commands of each executed command buffer run after it,
/// compiled with `children`: on the device over the `registered` replay variants where the product
/// sets the device replay up, by the host replay elsewhere. Every refusal is typed; only a device,
/// driver, or tool failure is `Failed`. A typed refusal after the device opened names that device
/// ([`RunAttempt::refused_on`]).
pub fn execute_engine(
    device_type: DeviceType,
    case_id: &str,
    case: &ComputeCase,
    compile: impl FnOnce(
        &WorkerDevice,
        &BindShape,
    ) -> Result<CompiledCase<ComputePipeline>, ExecutorError>,
    children: impl FnMut(
        &WorkerDevice,
        u64,
        &BindShape,
        &ChildCommand,
    ) -> Result<CompiledChild, ExecutorError>,
    registered: &[ReplayVariant],
    backend: Backend,
) -> RunAttempt {
    if let Some(refused) = refused_attempt(device_type, "engine-compute") {
        return refused;
    }
    with_validated_context(device_type, |context| {
        let result = run(
            context,
            case_id,
            case,
            (compile, children),
            registered,
            backend,
        );
        Ok(attempt(context, result, "engine-compute"))
    })
    .unwrap_or_else(|error| RunAttempt::of(Err(ExecutorError::Failed(error))))
}

/// The attempt of an `executor` run on `device_type` when this process's restriction is one the
/// device cannot take ([`refused_restriction`]): a typed refusal naming the profile and the
/// members, recorded with the device as it is ([`restriction_refusal`]). `None` when the run goes
/// on to open its context; a failure of the check itself is the attempt's failure.
pub(super) fn refused_attempt(device_type: DeviceType, executor: &str) -> Option<RunAttempt> {
    match refused_restriction(device_type) {
        Ok(None) => None,
        Ok(Some(refused)) => Some(restriction_refusal(&refused, executor)),
        Err(error) => Some(RunAttempt::of(Err(ExecutorError::Failed(error)))),
    }
}

/// The typed refusal of `refused` by `executor`, with the facts of the device as it is that its
/// record keeps ([`native_refused_on`]).
pub(super) fn restriction_refusal(refused: &RefusedRestriction, executor: &str) -> RunAttempt {
    let layer = serde_json::json!({
        "profile": refused.restriction.stem,
        "profile_name": refused.restriction.profile_name,
        "profile_sha256": refused.restriction.sha256,
        "refused": refused.gaps,
    });
    match native_refused_on(&refused.native, &refused.identity, executor, layer) {
        Ok(opened) => RunAttempt {
            result: Err(NotSupported::Restriction {
                profile: refused.restriction.stem.clone(),
                gaps: refused.gaps.clone(),
            }
            .into()),
            refused_on: Some(opened),
        },
        Err(error) => RunAttempt::of(Err(ExecutorError::Failed(format!(
            "the device lacks capabilities of profile {}, and {error}",
            refused.restriction.stem
        )))),
    }
}

/// What a typed refusal before any device opened records of the device as it is, read through
/// the profiles layer with its emulation off (`native`, `identity`): the device's identity,
/// `executor`, the validation layer, the profiles layer's restriction record (`profiles_layer`),
/// the requested leg, and the profile the device's own description derives (its fingerprint is in
/// the translator fingerprint, as a row's is). No device opened, so the environment records no
/// enabled features or active tools. A description the product cannot derive a profile from is the
/// error.
pub(crate) fn native_refused_on(
    native: &DeviceDescription,
    identity: &serde_json::Value,
    executor: &str,
    profiles_layer: serde_json::Value,
) -> Result<crate::candidate::engine::RefusedOn, String> {
    let device = WorkerDevice::new(native.clone(), crate::candidate::requested_policy())
        .map_err(|error| format!("its own description derives no profile: {error}"))?;
    let profile = device.device().profile();
    let fingerprint: String = profile
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let mut environment = identity.clone();
    if let Some(object) = environment.as_object_mut() {
        for (key, value) in [
            ("executor", serde_json::json!(executor)),
            ("layers", serde_json::json!(super::device::RECORDED_LAYERS)),
            ("synchronization_validation", serde_json::json!(true)),
            ("profiles_layer", profiles_layer),
            (
                "robustness",
                serde_json::json!({"leg": crate::candidate::requested_leg()}),
            ),
            ("profile_fingerprint", serde_json::json!(fingerprint)),
            ("profile", serde_json::json!(format!("{profile:?}"))),
        ] {
            object.insert(key.into(), value);
        }
    }
    Ok(crate::candidate::engine::RefusedOn {
        environment,
        translator_fingerprint: engine_fingerprint(profile),
        artifact: None,
    })
}

/// The attempt `result` was on `context`: a typed refusal names the device the context opened,
/// with the environment `executor`'s row on it would record. A refusal before a device was
/// described (the product refused the description itself) names none.
pub(super) fn attempt(
    context: &VulkanContext,
    result: Result<ComputeRun, ExecutorError>,
    executor: &str,
) -> RunAttempt {
    let refused_on = match &result {
        Err(ExecutorError::NotSupported(_)) => {
            opened_device(context)
                .ok()
                .map(|target| crate::candidate::engine::RefusedOn {
                    environment: environment(context, target.device(), executor),
                    translator_fingerprint: engine_fingerprint(target.device().profile()),
                    artifact: None,
                })
        }
        _ => None,
    };
    RunAttempt { result, refused_on }
}

/// The `Device` the context `device_type` opens: its queried description restricted to what it
/// enabled, under the default policy ([`opened_device`]). What a device test compiles for, so its
/// profile is the opened device's, never a committed one.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub fn opened_worker_device(device_type: DeviceType) -> Result<WorkerDevice, ExecutorError> {
    with_validated_context(device_type, |context| Ok(opened_device(context)))
        .map_err(ExecutorError::Failed)?
}

/// The `Device` of the opened context: its queried description restricted to what the context
/// enabled (`DeviceDescription::with_enabled`), under the default policy, or with the unordered
/// framebuffer fetch when this process opted in (`candidate::requested_policy`).
pub(super) fn opened_device(context: &VulkanContext) -> Result<WorkerDevice, ExecutorError> {
    let description = context
        .opened_description
        .clone()
        .map_err(ExecutorError::Failed)?;
    WorkerDevice::new(description, crate::candidate::requested_policy())
        .map_err(|e| product_refusal(&e))
}

/// Rows from different ICDs are different backends: the MoltenVK backend runs only on MoltenVK, and
/// native Vulkan never on it.
pub(super) fn check_backend(
    context: &VulkanContext,
    backend: Backend,
) -> Result<(), ExecutorError> {
    let moltenvk = context.driver.driver_id == vk::DriverId::MOLTENVK.as_raw();
    match backend {
        Backend::Moltenvk if !moltenvk => Err(format!(
            "the moltenvk backend opened {} (driver id {}), not MoltenVK",
            context.driver.driver_name, context.driver.driver_id
        )
        .into()),
        Backend::Vulkan if moltenvk => {
            Err("the native vulkan backend opened MoltenVK; use the moltenvk backend".into())
        }
        Backend::Moltenvk | Backend::Vulkan => Ok(()),
    }
}

fn run(
    context: &VulkanContext,
    case_id: &str,
    case: &ComputeCase,
    (compile, mut children): (
        impl FnOnce(&WorkerDevice, &BindShape) -> Result<CompiledCase<ComputePipeline>, ExecutorError>,
        impl FnMut(
            &WorkerDevice,
            u64,
            &BindShape,
            &ChildCommand,
        ) -> Result<CompiledChild, ExecutorError>,
    ),
    registered: &[ReplayVariant],
    backend: Backend,
) -> Result<ComputeRun, ExecutorError> {
    check_backend(context, backend)?;
    let target = opened_device(context)?;
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    let shape = compute_shape(case, addressed).map_err(ExecutorError::Failed)?;
    // The census and the precision notes come from the artifact the compile worker returned and
    // this process decoded (`pipeline.stage`), never from a compile of its own.
    let CompiledCase { pipeline, tables } = compile(&target, &shape)?;
    let device = target.device();
    let missing = unmet_requirements(context, &pipeline.requirements);
    if !missing.is_empty() {
        return Err(NotSupported::NotEnabled(missing).into());
    }
    // How the compiled module contains each out-of-bounds target on this device, decided before
    // anything runs; a form the device leaves uncontained is refused, never run. A kernel fetches
    // each stage input as the descriptor it was compiled under says.
    let attributes = out_of_bounds::Fetch::of_stage_input(case.stage_input.as_ref());
    let out_of_bounds = out_of_bounds_run(
        context.execution_class,
        device.profile(),
        &out_of_bounds::ModuleFacts {
            stage: Stage::Kernel,
            slots: &pipeline.interface.slots,
            attributes: &attributes,
            resident_roots: pipeline.interface.needs_pointer_translation,
            handle_table: pipeline.interface.needs_handle_translation,
            strategies: &pipeline.stage.strategies,
        },
        &case.out_of_bounds,
    )?;
    let spv_sha256 = sha256_bytes(&pipeline.stage.spirv_bytes());
    let label = SubmissionLabel {
        case_id: case_id.into(),
        module_sha256: spv_sha256.clone(),
    };
    let mut compile_child = |token: u64, shape: &BindShape, command: &ChildCommand| {
        children(&target, token, shape, command)
    };
    let executed = with_submission_label(label, || {
        execute(
            context,
            device,
            (&pipeline, &tables, &shape),
            case,
            (&mut compile_child, registered),
        )
    });
    // Every object of the run is destroyed by now, so destruction reports here too.
    let messages = context.take_messages();
    if !messages.is_empty() {
        let run = match &executed {
            Ok(_) => String::new(),
            Err(error) => format!(" (the run also failed: {error})"),
        };
        return Err(ExecutorError::Failed(format!(
            "VK_LAYER_KHRONOS_validation reported {} message(s); the run is not evidence{run}: {}",
            messages.len(),
            messages.join(" | ")
        )));
    }
    let Executed {
        output,
        status,
        children,
        escaped,
        replayed,
    } = executed?;
    // The census covers every module the run executed: the producer's, under the specialization
    // its plan for the shape gives, and each replayed command's.
    let planned = pipeline.plan(device, &shape).ok();
    let mut modules = module_censuses(
        std::slice::from_ref(&pipeline.stage),
        device.profile(),
        planned.as_ref(),
    )?;
    modules.extend(children);
    let out_of_bounds = out_of_bounds.map(|run| with_escaped(run, &escaped, Placement::Allocation));
    let census = executed_census(context, modules, out_of_bounds);
    let mut environment = environment(context, device, "engine-compute");
    command_replay(&mut environment, &replayed);
    Ok(ComputeRun {
        output,
        spv_sha256,
        translator_fingerprint: engine_fingerprint(device.profile()),
        environment,
        status,
        comparison: census.comparison_class(),
        census,
    })
}

/// The census of the stage modules a run executed, in pipeline order, on the opened device. The
/// modules are those of the pipeline the compile worker returned and this process decoded,
/// never of a compile of its own; the row's comparison class is derived from this census, so a
/// stored row re-derives it. `out_of_bounds` is how the run contained the case's out-of-bounds
/// targets.
pub(super) fn executed_census(
    context: &VulkanContext,
    modules: Vec<ModuleCensus>,
    out_of_bounds: Option<OutOfBoundsRun>,
) -> ExecutedCensus {
    ExecutedCensus {
        out_of_bounds,
        rate_map_pass: None,
        // The class of the device that executed, also under a restriction.
        device: context.execution_class,
        product: TRANSLATOR_FINGERPRINT.into(),
        restricted_to: context
            .restriction
            .map(|restriction| restriction.stem.clone()),
        leg: Some(context.leg),
        modules,
        artifact: None,
    }
}

/// Record in `environment` which replay ran each executed command buffer, in key order
/// (`command_replay`, [`ReplayPath::name`]), where any ran.
pub(super) fn command_replay(environment: &mut serde_json::Value, replayed: &[ReplayPath]) {
    if let (false, Some(object)) = (replayed.is_empty(), environment.as_object_mut()) {
        let paths: Vec<&str> = replayed.iter().map(|path| path.name()).collect();
        object.insert("command_replay".into(), serde_json::json!(paths));
    }
}

/// The census of each module of a pipeline the run executed
/// (`ModuleCensus::of_pipeline_modules_under`): its stages, compiled for the opened device's
/// `profile`, with the containment encodings that profile gives them, and the precision notes that
/// hold under the `plan` it ran with, where it is known (its specialization and the values it
/// pushes).
pub(super) fn module_censuses(
    stages: &[StageModule],
    profile: &DeviceProfile,
    plan: Option<&BindingPlan>,
) -> Result<Vec<ModuleCensus>, ExecutorError> {
    ModuleCensus::of_pipeline_modules_under(stages, profile, plan).map_err(ExecutorError::Failed)
}

/// The candidate environment of `executor`: the device facts, the ICD identity, the layers and
/// active tools, the profiles layer's restriction, and the profile the case compiled for (its
/// fingerprint is in the translator fingerprint).
pub(super) fn environment(
    context: &VulkanContext,
    device: &Device,
    executor: &str,
) -> serde_json::Value {
    let mut environment = context.environment();
    let fingerprint: String = device
        .profile()
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if let Some(object) = environment.as_object_mut() {
        for (key, value) in [
            ("executor", serde_json::json!(executor)),
            ("driver_id", serde_json::json!(context.driver.driver_id)),
            ("driver_name", serde_json::json!(context.driver.driver_name)),
            ("driver_info", serde_json::json!(context.driver.driver_info)),
            ("layers", serde_json::json!(super::device::RECORDED_LAYERS)),
            ("synchronization_validation", serde_json::json!(true)),
            ("profiles_layer", context.profiles_layer()),
            ("robustness", context.robustness()),
            ("profile_fingerprint", serde_json::json!(fingerprint)),
            (
                "profile",
                serde_json::json!(format!("{:?}", device.profile())),
            ),
        ] {
            object.insert(key.into(), value);
        }
    }
    environment
}

/// Every object of one run beyond its buffers and layout, destroyed on every exit path (before
/// the layout, a field declared after them).
struct Objects<'a> {
    device: &'a ash::Device,
    module: vk::ShaderModule,
    pipelines: Vec<vk::Pipeline>,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    layout: LayoutObjects<'a>,
}

impl Drop for Objects<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this run and is destroyed once; the
        // run waited for its submission (under the hang protocol) before returning normally.
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            for pipeline in &self.pipelines {
                self.device.destroy_pipeline(*pipeline, None);
            }
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

/// What [`execute`] read after the run: the output, the status, the census of each module a
/// replayed command executed, the guard and padding bytes that changed around each guarded buffer
/// and texture buffer, by out-of-bounds target, and which replay ran each executed command buffer,
/// in key order.
struct Executed {
    output: Vec<u8>,
    status: metal2vulkan::bind::StatusReport,
    children: Vec<ModuleCensus>,
    escaped: BTreeMap<OutOfBoundsResource, GuardCount>,
    replayed: Vec<ReplayPath>,
}

/// A buffer the run allocated, with its device address when the device has them.
pub(super) struct Allocation {
    pub(super) buffer: HostBuffer,
    pub(super) address: Option<u64>,
}

/// A host-visible buffer holding `bytes` (padded), usable as a storage, uniform, and transfer
/// buffer plus `extra`, with its device address when `addressed`.
pub(super) fn allocate(
    context: &VulkanContext,
    label: &str,
    bytes: &[u8],
    addressed: bool,
    extra: vk::BufferUsageFlags,
) -> Result<Allocation, String> {
    allocate_in(context, label, bytes, addressed, extra, None)
}

/// [`allocate`], inside a guard of `guard` when one is given ([`create_guarded_host_buffer`]).
pub(super) fn allocate_in(
    context: &VulkanContext,
    label: &str,
    bytes: &[u8],
    addressed: bool,
    extra: vk::BufferUsageFlags,
    guard: Option<GuardReach>,
) -> Result<Allocation, String> {
    let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::UNIFORM_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | extra;
    if addressed {
        usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    }
    let buffer = match guard {
        Some(reach) => create_guarded_host_buffer(context, label, &padded(bytes), usage, reach)?,
        None => create_host_buffer(context, label, &padded(bytes), usage)?,
    };
    let address = addressed.then(|| {
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer);
        // SAFETY: the buffer was created with SHADER_DEVICE_ADDRESS on a device that enabled
        // `bufferDeviceAddress`, and its memory is bound.
        unsafe { context.device.get_buffer_device_address(&info) }
    });
    Ok(Allocation { buffer, address })
}

fn not_implemented(what: impl Into<String>) -> ExecutorError {
    NotSupported::NotImplemented(format!("engine candidate: {}", what.into())).into()
}

/// Allocate the case's buffers (argument-buffer buffers first, so each owner is created with their
/// addresses, its textures' and samplers' resource ids, and its command buffers', [`field_words`],
/// [`owner_bytes`]) and its visible-function tables' buffers (holding `tables`' bytes), plan and
/// instantiate its dispatch, create what every plan write names (`plan_resources`: textures,
/// argument-buffer textures, heap arrays, handle tables, samplers, texel buffers, acceleration
/// structures, and neutrals included), record and submit it, wait, and read the status and then
/// the output ([`Executed`]). `compile` compiles the children the case's executed command
/// buffers dispatch, each in a compile worker: on the device, the `registered` replay variants.
fn execute(
    context: &VulkanContext,
    device: &Device,
    (pipeline, tables, compiled_for): (&ComputePipeline, &TableBuffers, &BindShape),
    case: &ComputeCase,
    (compile, registered): (&mut CompileChild<'_>, &[ReplayVariant]),
) -> Result<Executed, ExecutorError> {
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    // With a device address an allocation's guest address lies a fixed offset above it
    // (`BufferAddress::addressed`), else at its synthetic base.
    let own_guest = |allocation: &Allocation, synthetic: BufferAddress| match allocation.address {
        Some(host) => BufferAddress::addressed(host).map_err(ExecutorError::Failed),
        None => Ok(synthetic),
    };
    let synthetic_used = synthetic_used_addresses(case).map_err(ExecutorError::Failed)?;
    let records = records_ids(case).map_err(ExecutorError::Failed)?;
    let mut embedded = BTreeMap::new();
    let mut embedded_addresses = BTreeMap::new();
    for (&key, bytes) in &case.argument_buffer_buffers {
        // An out-of-bounds target is allocated inside its guard.
        let allocation = allocate_in(
            context,
            &format!("argument-buffer buffer {}+{}", key.0, key.1),
            bytes,
            addressed,
            vk::BufferUsageFlags::empty(),
            embedded_guard_reach(case, key),
        )?;
        // A command buffer's record storage lies at its ID's records address.
        let address = match records.get(&key) {
            Some(&id) => records_storage(id, allocation.address),
            None => own_guest(&allocation, synthetic_used[&key])?,
        };
        embedded_addresses.insert(key, address);
        embedded.insert(key, allocation);
    }
    let tokens = texture_tokens(case).map_err(ExecutorError::Failed)?;
    let fields = field_words(
        &embedded_addresses,
        &records,
        &tokens,
        &sampler_ids(case).map_err(ExecutorError::Failed)?,
        case.objects.iter().map(|(&key, &word)| (key, word)),
    )
    .map_err(ExecutorError::Failed)?;
    let mut buffers = BTreeMap::new();
    for &index in case.buffers.keys() {
        let bytes = owner_bytes(&case.buffers, index, &fields).map_err(ExecutorError::Failed)?;
        // An out-of-bounds target is allocated inside its guard.
        let allocation = allocate_in(
            context,
            &format!("buffer {index}"),
            &bytes,
            addressed,
            vk::BufferUsageFlags::empty(),
            guard_reach(case, index),
        )?;
        buffers.insert(index, allocation);
    }
    let mut table_buffers = BTreeMap::new();
    for (&index, table) in &case.function_tables {
        let bytes = table_bytes(tables, Stage::Kernel, index, table.byte_len())?;
        let allocation = allocate(
            context,
            &format!("visible function table {index}"),
            bytes,
            addressed,
            vk::BufferUsageFlags::empty(),
        )?;
        table_buffers.insert(index, allocation);
    }
    let addresses: BTreeMap<u32, BufferAddress> = buffers
        .iter()
        .chain(&table_buffers)
        .map(|(&index, allocation)| {
            Ok((
                index,
                own_guest(allocation, BufferAddress::synthetic(index))?,
            ))
        })
        .collect::<Result<_, ExecutorError>>()?;
    // The commands the host encodes, written into their record storage before the producer runs,
    // each buffer at its case buffer's guest address.
    for (key, command) in &case.command_buffers {
        if command.commands.is_empty() {
            continue;
        }
        let storage = embedded
            .get(key)
            .ok_or_else(|| format!("command buffer {key:?} has no record storage"))?;
        // SAFETY: the storage was created for this run and nothing has been submitted yet.
        let mapped = unsafe { storage.buffer.map()? };
        command
            .encode_host_records(mapped.bytes, |buffer| {
                addresses.get(&buffer).map(|address| address.guest)
            })
            .map_err(ExecutorError::Failed)?;
    }
    let state = bind_state(case, &addresses, &embedded_addresses).map_err(ExecutorError::Failed)?;
    // The compile settled the plan's variant for this shape (`variants`): planning is a function
    // of the pipeline, the device's profile, and the shape, so the same shape plans the variant
    // the pipeline was compiled with.
    let shape = state.shape();
    if shape != *compiled_for {
        return Err(ExecutorError::Failed(format!(
            "the dispatch binds shape {shape:?}, not the {compiled_for:?} it was compiled for"
        )));
    }
    let plan = pipeline
        .plan(device, &shape)
        .map_err(|e| bind_refusal(&e))?;
    if let Some(diagnostic) = plan.diagnostics.first() {
        return Err(incompatible_binding(diagnostic));
    }
    let commands = plan
        .instantiate_dispatch(&state, &case.dispatch)
        .map_err(|e| bind_refusal(&e))?;
    if commands.indirect_clamp.is_some() || commands.regions.iter().any(|r| r.indirect.is_some()) {
        return Err(not_implemented("indirect dispatches"));
    }
    let status = allocate(
        context,
        "execution status",
        &ExecutionStatus::initial(),
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    let slot = |class, index| SlotRef {
        stage: Stage::Kernel,
        class,
        index,
    };
    let guest_buffers: BTreeMap<SlotRef, vk::Buffer> = buffers
        .iter()
        .chain(&table_buffers)
        .map(|(&index, allocation)| (slot(SlotClass::Buffer, index), allocation.buffer.buffer))
        .collect();
    let textures: BTreeMap<SlotRef, &LiteralTexture> = case
        .textures
        .iter()
        .map(|(&index, texture)| (slot(SlotClass::Texture, index), texture))
        .collect();
    let texel_guards: BTreeMap<SlotRef, GuardReach> = case
        .textures
        .keys()
        .filter_map(|&binding| {
            let reach = texel_guard_reach(case, binding)?;
            Some((slot(SlotClass::Texture, binding), reach))
        })
        .collect();
    let used_textures: BTreeMap<u32, &LiteralTexture> = case
        .argument_buffer_textures
        .iter()
        .map(|(key, texture)| (tokens[key], texture))
        .collect();
    let samplers: BTreeMap<SlotRef, SamplerDescriptor> = case
        .samplers
        .iter()
        .map(|(&index, sampler)| {
            (
                slot(SlotClass::Sampler, index),
                resources::sampler_descriptor(sampler),
            )
        })
        .collect();
    let structures: BTreeMap<SlotRef, CaseStructure> = case
        .acceleration_structures
        .iter()
        .map(|(&index, structure)| (slot(SlotClass::Buffer, index), structure.clone()))
        .collect();
    let constant_blobs: BTreeMap<Stage, Arc<[u8]>> = pipeline
        .stage
        .constant_blob
        .iter()
        .map(|blob| (Stage::Kernel, blob.clone()))
        .collect();
    let observed = match case.output {
        ComputeOutput::Texture { binding, .. } => {
            Some(TextureKey::Slot(slot(SlotClass::Texture, binding)))
        }
        ComputeOutput::ArgumentBufferTexture {
            owner,
            field_offset,
            ..
        } => tokens
            .get(&(owner, field_offset))
            .copied()
            .map(TextureKey::Used),
        ComputeOutput::RenderTarget { index, .. } => Some(TextureKey::TileAttachment(index)),
        ComputeOutput::None
        | ComputeOutput::Buffer { .. }
        | ComputeOutput::ArgumentBufferBuffer { .. } => None,
    };
    let no_attachments = BTreeMap::new();
    let tile_attachments = case
        .tile
        .as_ref()
        .map_or(&no_attachments, |pass| &pass.tiled);
    let mut resources = PlanResources::create(
        context,
        device,
        &plan.descriptors,
        &Guest {
            buffers: &guest_buffers,
            textures: &textures,
            texel_guards: &texel_guards,
            used_textures: &used_textures,
            samplers: &samplers,
            argument_samplers: &state.argument_samplers,
            constant_blobs: &constant_blobs,
            descriptor_ranges: &commands.descriptor_ranges,
            descriptor_bytes: &commands.descriptor_bytes,
            structures: &structures,
            // A dispatch draws no patches and has no attachments or fragment imageblock.
            tessellation_factors: None,
            attachments: &BTreeMap::new(),
            imageblock: None,
            tile_attachments,
            shadow_stencil: None,
            status: Some(status.buffer.buffer),
        },
        observed,
    )?;
    // Consumer-owned objects behind the push holes.
    let root = commands
        .root_table
        .as_ref()
        .map(|table| upload_root_table(context, table, addressed))
        .transpose()?;
    let regions = push_bytes(&commands, root.as_ref().map(|(table, _)| table), &status)?;

    let layout = match LayoutObjects::create(&context.device, &pipeline.layout) {
        Ok(layout) => layout,
        Err(LayoutError::Shape(why) | LayoutError::Refused(why)) => {
            return Err(ExecutorError::Failed(why))
        }
    };
    let mut objects = Objects {
        device: &context.device,
        module: vk::ShaderModule::null(),
        pipelines: Vec::new(),
        command_pool: vk::CommandPool::null(),
        fence: vk::Fence::null(),
        layout,
    };
    let module_info = vk::ShaderModuleCreateInfo::default().code(&pipeline.stage.spirv);
    // SAFETY: `module_info` points at live SPIR-V words for the call.
    objects.module = unsafe { context.device.create_shader_module(&module_info, None) }
        .map_err(|error| format!("create shader module: {error}"))?;
    let region_pipelines = create_pipelines(
        context,
        &mut objects,
        pipeline,
        &commands.regions,
        case.creation_bound,
    )?;
    let sets = resources.write_sets(context, &objects.layout)?;
    record_and_submit(
        context,
        &mut objects,
        &resources,
        &sets,
        &commands.regions,
        &regions,
        &region_pipelines,
    )?;

    // The status first: nothing is read from a run whose indirect count was clamped or that
    // missed the resident set.
    // SAFETY: the submission completed (the fence signaled), and nothing else maps it.
    let status_bytes: [u8; ExecutionStatus::BYTES] = unsafe { status.buffer.map()? }.bytes
        [..ExecutionStatus::BYTES]
        .try_into()
        .expect("the status buffer holds 32 bytes");
    let report = status_verdict(&status_bytes)?;
    // Each command buffer's records as the host contract decodes them: an encoding fault fails the
    // run. Then the commands the case executes run, over the run's allocations.
    let mut records = BTreeMap::new();
    for &key in case.command_buffers.keys() {
        let allocation = embedded
            .get(&key)
            .ok_or_else(|| format!("command buffer {key:?} has no record storage"))?;
        let len = case.argument_buffer_buffers[&key].len();
        // SAFETY: as for the status buffer.
        let bytes = unsafe { allocation.buffer.map()? }.bytes[..len].to_vec();
        records.insert(key, bytes);
    }
    validate_records(
        case.command_buffers
            .iter()
            .map(|(&key, command)| (key, command)),
        &records,
    )
    .map_err(ExecutorError::Failed)?;
    // Each allocation at its guest address and the length the guest sees (not the padding), in
    // the run's order (`run_allocations`), which the device replay's resident set follows.
    let live = run_allocations(case)
        .into_iter()
        .map(|(held, length)| {
            let (address, allocation) = match held {
                RunAllocation::Buffer(index) => (addresses[&index], &buffers[&index]),
                RunAllocation::FunctionTable(index) => (addresses[&index], &table_buffers[&index]),
                RunAllocation::ArgumentBuffer(key) => (embedded_addresses[&key], &embedded[&key]),
            };
            Live {
                buffer: allocation.buffer.buffer,
                address,
                length,
            }
        })
        .collect::<Vec<_>>();
    let (executed, replayed) = replay_commands(
        context,
        device,
        case.command_buffers
            .iter()
            .map(|(&key, command)| (key, command)),
        Replay {
            live: &live,
            records: &records,
            addresses: &|_, buffer| addresses.get(&buffer).map(|address| address.guest),
            compile,
        },
        OnDevice {
            registered,
            storage: &|key| embedded.get(&key).and_then(|allocation| allocation.address),
            writes: pipeline.interface.command_writes,
        },
    )?;
    let mut finals = BTreeMap::new();
    let mut escaped = BTreeMap::new();
    for (&index, allocation) in &buffers {
        let len = case.buffers[&index].len();
        // SAFETY: as for the status buffer; every replayed command completed too.
        let bytes = unsafe { allocation.buffer.map()? }.bytes[..len].to_vec();
        finals.insert(index, bytes);
        // SAFETY: as for the status buffer.
        if let Some(count) = unsafe { allocation.buffer.escaped()? } {
            escaped.insert(buffer_target(case, index), count);
        }
    }
    let mut embedded_finals = BTreeMap::new();
    for (&key, allocation) in &embedded {
        let len = case.argument_buffer_buffers[&key].len();
        // SAFETY: as for the status buffer.
        let bytes = unsafe { allocation.buffer.map()? }.bytes[..len].to_vec();
        embedded_finals.insert(key, bytes);
        // SAFETY: as for the status buffer.
        if let Some(count) = unsafe { allocation.buffer.escaped()? } {
            escaped.insert(
                OutOfBoundsResource::ArgumentBufferBuffer {
                    buffer_binding: key.0,
                    field_offset: key.1,
                },
                count,
            );
        }
    }
    // SAFETY: as for the status buffer.
    for (slot, count) in unsafe { resources.texel_guard_counts()? } {
        escaped.insert(
            OutOfBoundsResource::Texture {
                binding: slot.index,
            },
            count,
        );
    }
    let mut final_textures = BTreeMap::new();
    let mut embedded_final_textures = BTreeMap::new();
    let mut final_attachments = BTreeMap::new();
    match (observed, case.output) {
        // An attachment no plan write reaches keeps its contents, as the pass loads and stores
        // it (`final_texture`).
        (Some(key @ TextureKey::TileAttachment(index)), _) => {
            if let Some(attachment) = tile_attachments.get(&index) {
                // SAFETY: as for the status buffer.
                let attachment = unsafe { resources.final_texture(key, attachment)? };
                final_attachments.insert(index, attachment);
            }
        }
        (Some(key @ TextureKey::Slot(observed)), _) => {
            if let Some(texture) = textures.get(&observed) {
                // SAFETY: as for the status buffer.
                let texture = unsafe { resources.final_texture(key, texture)? };
                final_textures.insert(observed.index, texture);
            }
        }
        (
            Some(key @ TextureKey::Used(_)),
            ComputeOutput::ArgumentBufferTexture {
                owner,
                field_offset,
                ..
            },
        ) => {
            if let Some(texture) = case.argument_buffer_textures.get(&(owner, field_offset)) {
                // SAFETY: as for the status buffer.
                let texture = unsafe { resources.final_texture(key, texture)? };
                embedded_final_textures.insert((owner, field_offset), texture);
            }
        }
        _ => {}
    }
    let output = select_output(
        case.output,
        &finals,
        &embedded_finals,
        &final_textures,
        &embedded_final_textures,
        &final_attachments,
    )
    .map_err(ExecutorError::Failed)?;
    drop(objects);
    drop(resources);
    Ok(Executed {
        output,
        status: report,
        children: executed,
        escaped,
        replayed,
    })
}

/// The bytes of the visible-function table at `index` of `stage` (the compile's), which must be the
/// `length` its bind state binds.
pub(super) fn table_bytes(
    tables: &TableBuffers,
    stage: Stage,
    index: u32,
    length: u64,
) -> Result<&[u8], ExecutorError> {
    match tables.get(&(stage, index)) {
        Some(bytes) if bytes.len() as u64 == length => Ok(bytes),
        Some(bytes) => Err(format!(
            "the {stage:?} visible function table {index} has {} bytes, not {length}",
            bytes.len()
        )
        .into()),
        None => Err(format!("the {stage:?} visible function table {index} has no words").into()),
    }
}

/// Upload a resident root table with its sink entry filled: `(table, sink)`; the table's device
/// address fills its `RootTableAddress` holes.
pub(super) fn upload_root_table(
    context: &VulkanContext,
    table: &RootTable,
    addressed: bool,
) -> Result<(Allocation, Allocation), ExecutorError> {
    let sink = allocate(
        context,
        "sink",
        &[0; NeutralResource::SINK_BYTES as usize],
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    let sink_address = sink
        .address
        .ok_or("the root table needs device addresses")?;
    let mut bytes = table.bytes.clone();
    let at = RootTable::SINK_HOST_OFFSET as usize;
    bytes
        .get_mut(at..at + 8)
        .ok_or("the root table has no sink entry")?
        .copy_from_slice(&sink_address.to_le_bytes());
    let table = allocate(
        context,
        "root table",
        &bytes,
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    match table.address {
        Some(address) if address % 16 == 0 => Ok((table, sink)),
        other => Err(format!("the root table's address {other:?} is not 16-aligned").into()),
    }
}

/// A plan that substituted a neutral for a bound resource of another type (guest undefined
/// behavior): not evidence either way.
pub(super) fn incompatible_binding(
    diagnostic: &metal2vulkan::bind::PlanDiagnostic,
) -> ExecutorError {
    NotSupported::Bind(format!(
        "{diagnostic:?}: the case binds a resource the function does not declare (Metal leaves \
         this undefined)"
    ))
    .into()
}

/// Each region's push-constant writes with the consumer's holes filled: the root table's and the
/// status buffer's device addresses.
fn push_bytes(
    commands: &DispatchCommands,
    root: Option<&Allocation>,
    status: &Allocation,
) -> Result<Vec<PushWrites>, ExecutorError> {
    push_writes(commands, |hole| match *hole {
        PushHole::RootTableAddress { offset, .. } => Ok((
            offset,
            root.and_then(|table| table.address)
                .ok_or("a root-table hole without a root table")?,
        )),
        PushHole::ExecutionStatusAddress { offset, .. } => Ok((
            offset,
            status
                .address
                .ok_or("a status-address hole without device addresses")?,
        )),
        other => Err(not_implemented(format!("the push hole {other:?}"))),
    })
}

/// Each region's push-constant writes with every hole of `commands` filled: `hole` gives each
/// hole's offset and the device address it takes.
fn push_writes(
    commands: &DispatchCommands,
    hole: impl Fn(&PushHole) -> Result<(u32, u64), ExecutorError>,
) -> Result<Vec<PushWrites>, ExecutorError> {
    let holes = commands
        .push_holes
        .iter()
        .map(hole)
        .collect::<Result<Vec<_>, _>>()?;
    let mut regions = Vec::new();
    for region in &commands.regions {
        let mut writes: PushWrites = region
            .push_constants
            .iter()
            .map(|w| (w.stages.into(), w.offset, w.bytes.clone()))
            .collect();
        for &(offset, address) in &holes {
            let write = writes
                .iter_mut()
                .find(|(_, start, bytes)| {
                    *start <= offset && offset as usize + 8 <= *start as usize + bytes.len()
                })
                .ok_or_else(|| format!("no push write covers the hole at {offset}"))?;
            let at = (offset - write.1) as usize;
            write.2[at..at + 8].copy_from_slice(&address.to_le_bytes());
        }
        regions.push(writes);
    }
    Ok(regions)
}

/// One compute pipeline per distinct region specialization (with its stage flags and required
/// subgroup size), each creation held to `bound` (the case's, `ComputeCase::creation_bound`), and
/// each region's index into `objects.pipelines`.
fn create_pipelines(
    context: &VulkanContext,
    objects: &mut Objects<'_>,
    pipeline: &ComputePipeline,
    regions: &[DispatchRegion],
    bound: Duration,
) -> Result<Vec<usize>, ExecutorError> {
    let entry = CString::new(pipeline.stage.entry_point.as_str())
        .map_err(|_| "entry point name contains NUL")?;
    let mut keys: Vec<&DispatchRegion> = Vec::new();
    let mut indices = Vec::with_capacity(regions.len());
    for region in regions {
        let same = |known: &&DispatchRegion| {
            known.specialization == region.specialization
                && known.stage_flags == region.stage_flags
                && known.required_subgroup_size == region.required_subgroup_size
        };
        if let Some(index) = keys.iter().position(same) {
            indices.push(index);
            continue;
        }
        let map: Vec<vk::SpecializationMapEntry> = region
            .specialization
            .entries
            .iter()
            .map(|&entry| entry.into())
            .collect();
        let spec = vk::SpecializationInfo::default()
            .map_entries(&map)
            .data(&region.specialization.data);
        let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(region.required_subgroup_size.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(objects.module)
            .name(&entry)
            .flags(region.stage_flags.into());
        if !map.is_empty() {
            stage = stage.specialization_info(&spec);
        }
        if region.required_subgroup_size.is_some() {
            stage = stage.push_next(&mut subgroup);
        }
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(objects.layout.pipeline_layout);
        let clock = crate::budgets::CreationClock::start();
        // SAFETY: every pointer in `info` outlives the call; the result is destroyed with
        // `objects`.
        let created = unsafe {
            context
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, error)| format!("the driver refused the compute pipeline: {error}"))?;
        objects.pipelines.extend(created.first().copied());
        crate::budgets::creation_within("the compute pipeline", clock.read(), bound)?;
        indices.push(keys.len());
        keys.push(region);
    }
    Ok(indices)
}

/// Record every region in order (a compute-to-compute barrier between regions, which only adds
/// an ordering Metal's single dispatch allows, and a shader-to-host barrier at the end), submit,
/// and wait under the hang protocol.
fn record_and_submit(
    context: &VulkanContext,
    objects: &mut Objects<'_>,
    resources: &PlanResources<'_>,
    sets: &[vk::DescriptorSet],
    regions: &[DispatchRegion],
    push: &[PushWrites],
    pipelines: &[usize],
) -> Result<(), ExecutorError> {
    let device = &context.device;
    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family);
    // SAFETY: the queue family is the context's.
    objects.command_pool = unsafe { device.create_command_pool(&pool_info, None) }
        .map_err(|error| format!("create command pool: {error}"))?;
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(objects.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // The pool is live; the buffer is freed with it.
    let command = context
        .allocate_command_buffers(&allocate_info)
        .map_err(|error| format!("allocate command buffer: {error}"))?[0];
    let shader_to_shader = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    let shader_to_host = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    // SAFETY: every handle recorded is live until the fence signals; the buffer is recorded once.
    unsafe {
        device
            .begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|error| format!("begin command buffer: {error}"))?;
        resources.record_uploads(command);
        if !sets.is_empty() {
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                objects.layout.pipeline_layout,
                0,
                sets,
                &[],
            );
        }
        for (index, region) in regions.iter().enumerate() {
            if index > 0 {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[shader_to_shader],
                    &[],
                    &[],
                );
            }
            device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                objects.pipelines[pipelines[index]],
            );
            for (stages, offset, bytes) in &push[index] {
                device.cmd_push_constants(
                    command,
                    objects.layout.pipeline_layout,
                    *stages,
                    *offset,
                    bytes,
                );
            }
            let [x, y, z] = region.group_count;
            device.cmd_dispatch(command, x, y, z);
        }
        resources.record_readback(command);
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[shader_to_host],
            &[],
            &[],
        );
        device
            .end_command_buffer(command)
            .map_err(|error| format!("end command buffer: {error}"))?;
        objects.fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|error| format!("create completion fence: {error}"))?;
        let commands = [command];
        let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
        device
            .queue_submit(context.queue, &submit, objects.fence)
            .map_err(|error| format!("submit command buffer: {error}"))?;
    }
    context.wait_for_fence(objects.fence, "engine candidate execution")?;
    Ok(())
}
