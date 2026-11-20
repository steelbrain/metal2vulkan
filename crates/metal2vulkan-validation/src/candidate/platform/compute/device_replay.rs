//! Device replay of executed compute commands in the engine compute executor.
//!
//! After the producer completed and every command buffer's records passed the host contract's
//! decode (as for the host replay, `validate_records`), each executed compute command buffer runs
//! its execution range on the device where the product sets the device replay up
//! (`replay_preprocess_module`: a device with device addresses) and the case registers a
//! variant: the records are never decoded to drive what runs. The consumer side is the
//! product's contract (`docs/API.md`, "GPU-encoded compute commands on the device"):
//!
//! - every registered variant ([`ReplayVariant`]: a compute template's pipeline token, threadgroup
//!   size, and threadgroup lengths, `engine::replay_variants`) is compiled in a compile worker
//!   with `LoweringInputs::replay_bindings` (`ChildCommand::ReplayVariant`), planned for a bind
//!   state that sets its threadgroup lengths, binds no kernel buffer, and makes every allocation
//!   of the run resident (as the oracle's `useResource` does), and instantiated with
//!   `BindingPlan::instantiate_replay`;
//! - one command buffer records the product's preprocess over the range (its scratch holding the
//!   parameters, the resident root table with the sink filled, the variants, and the parent
//!   encoder's state, as `replay_preprocess` states them, with the product's most regions per
//!   command, `MAX_REPLAY_REGIONS`), a barrier, and then per command in range order, per region
//!   (a ragged grid's edges and a group count's chunks past the device's limit), and per variant
//!   `vkCmdDispatchIndirect` from the arguments of the command's region with the command's slot
//!   table as the root table and the region's record as the region, a compute barrier between
//!   commands;
//! - after the wait, the status: a replay fault (a command dispatched nothing) refuses the row
//!   for the product's typed limits and fails it for every other reason
//!   (`engine::replay_fault_verdict`); the census holds each variant once per command whose
//!   arguments selected it in any region.
//!
//! Where the context enabled `VK_EXT_device_generated_commands` (`VulkanContext::
//! generated_commands`), the variants are compiled first and the product's
//! `replay_preprocess_generated` sets up the generated-commands path: the same preprocess writes
//! each command's sequences and each possible segment's count in place of the arguments, and the
//! command buffer records one `vkCmdExecuteGeneratedCommandsEXT` per command of the range
//! ([`generated`]) in place of the dispatches. Where the product refuses that path
//! (`ReplayRefusal::GeneratedCommands`: a limit the device states, or variants that bind other
//! descriptors), K14c's dispatches run instead; the environment names which ran.
//!
//! A render command buffer runs on the device where the runtime's GPU form draws it
//! (`draw_replay`); one the shared rules send to the host step, a compute one on a device without
//! device addresses, and one of a case that registers no variant run the host replay
//! (`replay::execute_commands`).

use super::draw_replay::{draw_commands, Drawing};
use super::replay::{bound_table, execute_commands, require_enabled};
use super::*;
use crate::candidate::engine::{
    replay_fault_verdict, replay_regions, replay_resident, replay_variant_state,
};
use crate::case::indirect_commands::CommandBufferResource;
use metal2vulkan::bind::indirect_commands::{
    BufferStage, CommandRecord, Inheritance, RegisteredBuffer, StateSource,
};
use metal2vulkan::bind::{
    replay_preprocess, replay_preprocess_generated, replay_preprocess_module, BindState,
    BindingPlan, GeneratedLayout, GeneratedVariant, IndirectArgs, ReplayExecution, ReplayLayout,
    ReplayPreprocessModule, ReplayRefusal,
};
use metal2vulkan::error::BindError;

mod generated;

/// Which replay ran an executed command buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::candidate::platform) enum ReplayPath {
    /// The product's preprocess and indirect dispatches of the registered variants.
    Device,
    /// The product's preprocess and the generated-commands executions of its token stream.
    Generated,
    /// The host replay of the decoded records.
    Host,
    /// The product's draw replay preprocess and one indirect draw per command of the render
    /// template's draw replay variant (`draw_replay`).
    Draws,
    /// The product's draw replay preprocess, its merge passes, and the merged draw of the render
    /// template's merged variant (`draw_replay`).
    MergedDraws,
}

impl ReplayPath {
    /// The name the candidate's environment records.
    pub(in crate::candidate::platform) fn name(self) -> &'static str {
        match self {
            ReplayPath::Device => "device",
            ReplayPath::Generated => "generated",
            ReplayPath::Host => "host",
            ReplayPath::Draws => "device-draws",
            ReplayPath::MergedDraws => "merged-draws",
        }
    }
}

/// What the device replay needs beyond the host replay's [`Replay`]: the variants the case
/// registers, the device address of each command buffer's record storage (its header), by key,
/// and what the functions of the case's own dispatch or draw may write into the records of every
/// command buffer of their kinds (`PipelineInterface::command_writes`: its encoder makes each
/// resident for writing, as the case runner's does).
pub(in crate::candidate::platform) struct OnDevice<'a, K> {
    pub(in crate::candidate::platform) registered: &'a [ReplayVariant],
    pub(in crate::candidate::platform) storage: &'a dyn Fn(K) -> Option<u64>,
    pub(in crate::candidate::platform) writes: metal2vulkan::pipeline::CommandWrites,
}

/// Execute each of `command_buffers` that the case executes, in key order: a compute command
/// buffer on the device ([`device_commands`]) where the product sets the device replay up and
/// `on_device` registers a variant, a render one on the device where the runtime's GPU form draws
/// it ([`draw_commands`]), every other one by the host replay ([`execute_commands`]). The census
/// of every module a command executed, and which replay ran each executed command buffer.
pub(in crate::candidate::platform) fn replay_commands<'c, K: Ord + Copy + std::fmt::Debug>(
    context: &VulkanContext,
    device: &Device,
    command_buffers: impl IntoIterator<Item = (K, &'c CommandBufferResource)>,
    replay: Replay<'_, '_, K>,
    on_device: OnDevice<'_, K>,
) -> Result<(replay::Executed, Vec<ReplayPath>), ExecutorError> {
    let Replay {
        live,
        records,
        addresses,
        compile,
    } = replay;
    let mut executed = Vec::new();
    let mut paths = Vec::new();
    // The preprocess module, built once: `None` where the product does not set the device replay
    // up on this device (no device addresses).
    let mut preprocess: Option<Option<ReplayPreprocessModule>> = None;
    for (key, command) in command_buffers {
        if command.execute.is_none() {
            continue;
        }
        if command.is_render() {
            let drawing = Drawing {
                live,
                addresses,
                storage: (on_device.storage)(key),
                writes: on_device.writes,
            };
            if let Some((census, path)) =
                draw_commands(context, device, (key, command), drawing, compile)?
            {
                executed.extend(census);
                paths.push(path);
                continue;
            }
        }
        let module = if command.is_render() || on_device.registered.is_empty() {
            None
        } else {
            if preprocess.is_none() {
                preprocess = Some(preprocess_module(device)?);
            }
            preprocess.as_ref().and_then(Option::as_ref)
        };
        match module {
            Some(module) => {
                let storage = (on_device.storage)(key).ok_or_else(|| {
                    format!("command buffer {key:?} has no record storage address")
                })?;
                let (census, path) = device_commands(
                    context,
                    device,
                    (key, command),
                    module,
                    Commands {
                        live,
                        addresses,
                        storage,
                        registered: on_device.registered,
                    },
                    compile,
                )?;
                executed.extend(census);
                paths.push(path);
            }
            None => {
                executed.extend(execute_commands(
                    context,
                    device,
                    [(key, command)],
                    Replay {
                        live,
                        records,
                        addresses,
                        compile: &mut *compile,
                    },
                )?);
                paths.push(ReplayPath::Host);
            }
        }
    }
    Ok((executed, paths))
}

/// The product's preprocess module for `device`, or `None` where the product states the device
/// replay unsupported (no device addresses): the host replay runs there.
fn preprocess_module(device: &Device) -> Result<Option<ReplayPreprocessModule>, ExecutorError> {
    match replay_preprocess_module(device) {
        Ok(module) => Ok(Some(module)),
        Err(metal2vulkan::error::Error::Unsupported(_)) => Ok(None),
        Err(error) => Err(product_refusal(&error)),
    }
}

/// What one command buffer's device replay runs over: the run's allocations `live`, the guest
/// address of each case buffer of the command buffer's owner (`addresses`, for the parent's
/// bindings), the record storage's device address `storage`, and the registered variants.
struct Commands<'a, K> {
    live: &'a [Live],
    addresses: &'a dyn Fn(K, u32) -> Option<u64>,
    storage: u64,
    registered: &'a [ReplayVariant],
}

/// Replay the execution range of the command buffer `key` on the device: compile each registered
/// variant ([`prepare`]), set up the generated-commands path where the context enabled it and the
/// product sets it up ([`generated::Generated::create`]) or else each variant's dispatches
/// ([`Variant::create`]), upload the preprocess's scratch, record the preprocess and every
/// command's executions or indirect dispatches in one command buffer, submit, wait, and read the
/// status. The census of each variant once per command whose arguments (or sequences) selected
/// it, and which path ran.
fn device_commands<K: Ord + Copy + std::fmt::Debug>(
    context: &VulkanContext,
    device: &Device,
    (key, command): (K, &CommandBufferResource),
    module: &ReplayPreprocessModule,
    commands: Commands<'_, K>,
    compile: &mut CompileChild<'_>,
) -> Result<(replay::Executed, ReplayPath), ExecutorError> {
    let Commands {
        live,
        addresses,
        storage,
        registered,
    } = commands;
    let fail = |why: String| ExecutorError::Failed(format!("command buffer {key:?}: {why}"));
    let Some(parent) = command
        .parent_bindings(|buffer| addresses(key, buffer))
        .map_err(ExecutorError::Failed)?
    else {
        return Ok((Vec::new(), ReplayPath::Device));
    };
    let [start, end] = command.execution_range;
    let count = end
        .checked_sub(start)
        .ok_or_else(|| fail("a reversed execution range".into()))?;
    if count == 0 {
        return Ok((Vec::new(), ReplayPath::Device));
    }
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    if !addressed {
        return Err(fail(
            "the device replay needs device addresses the context did not enable".into(),
        ));
    }
    // Every allocation of the run is resident, as the oracle's `useResource` makes it.
    let resident = replay_resident(
        live.iter()
            .map(|allocation| (allocation.address, allocation.length)),
    )
    .map_err(ExecutorError::Failed)?;
    let mut execution = ReplayExecution::new(command.capacity, start, count);
    execution.inheritance = command.inheritance();
    execution.parent_pipeline = parent.pipeline;
    // Commands that inherit the encoder's buffers bind its kernel buffers, each at its
    // allocation's device address (`ReplayExecution::parent_buffers`).
    if execution.inheritance.buffers == StateSource::Parent {
        let buffers = live
            .iter()
            .map(|allocation| {
                RegisteredBuffer::new(allocation, allocation.address.guest, allocation.length)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| fail(error.to_string()))?;
        let encoder = Inheritance {
            pipeline: StateSource::Parent,
            buffers: StateSource::Parent,
        };
        let empty = CommandRecord::default();
        let resolved = empty
            .resolve_bindings(encoder, &parent)
            .map_err(|error| fail(error.to_string()))?;
        let (table, _) =
            bound_table(&resolved, Stage::Kernel, BufferStage::Kernel, &buffers).map_err(fail)?;
        execution.parent_buffers = table.buffers;
    }
    // As many regions per command as keep the range's dispatches within the product's bound.
    execution.regions = replay_regions(count, registered.len());
    let k14c = || {
        replay_preprocess(device, &resident, &execution, registered).map_err(|e| bind_refusal(&e))
    };
    // The generated-commands path needs the variants' plans and commands before the preprocess;
    // K14c's path keeps its order, the preprocess's refusals first.
    let dgc = context.generated_commands();
    let (pre, prepared) = match dgc {
        Some(_) => {
            let prepared = prepare_all(context, device, registered, &resident, compile)?;
            let set_up = {
                let generated: Vec<GeneratedVariant<'_>> =
                    prepared.iter().map(Prepared::generated).collect();
                replay_preprocess_generated(device, &resident, &execution, registered, &generated)
            };
            match set_up {
                Ok(pre) => (pre, prepared),
                Err(BindError::InvalidReplay {
                    reason: ReplayRefusal::GeneratedCommands { .. },
                }) => (k14c()?, prepared),
                Err(error) => return Err(bind_refusal(&error)),
            }
        }
        None => {
            let pre = k14c()?;
            (
                pre,
                prepare_all(context, device, registered, &resident, compile)?,
            )
        }
    };
    let layout = pre.layout;
    require_preprocess(context, module)?;
    let stages: Vec<StageModule> = prepared.iter().map(|p| p.stage.clone()).collect();
    let mut replayed = match (dgc, &pre.generated, &layout.generated) {
        (Some(dgc), Some(replay), Some(generated)) => Replayed::Generated(
            Box::new(generated::Generated::create(
                context, device, dgc, &prepared, replay, generated,
            )?),
            *generated,
        ),
        _ => {
            let mut variants = Vec::with_capacity(prepared.len());
            for p in prepared {
                variants.push(Variant::create(context, device, p)?);
            }
            Replayed::Dispatches(variants)
        }
    };

    // The consumer's objects: the sink the resident root table names, the scratch, the status.
    let sink = allocate(
        context,
        "device replay sink",
        &[0; NeutralResource::SINK_BYTES as usize],
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    let sink_address = sink.address.ok_or("the sink has no device address")?;
    let scratch_bytes = scratch_bytes(&pre, sink_address)?;
    // The generated path's preprocess memory, sized for the executions the layout states.
    if let Replayed::Generated(generated, layout) = &mut replayed {
        generated.reserve_preprocess(context, layout)?;
    }
    let scratch = allocate(
        context,
        "device replay scratch",
        &scratch_bytes,
        addressed,
        vk::BufferUsageFlags::INDIRECT_BUFFER,
    )?;
    let scratch_address = match scratch.address {
        // Every slot table and the resident root table are root tables: 16-aligned.
        Some(address) if address % 16 == 0 => address,
        other => {
            return Err(fail(format!(
                "the scratch's address {other:?} is not 16-aligned"
            )))
        }
    };
    let status = allocate(
        context,
        "device replay status",
        &ExecutionStatus::initial(),
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    let status_address = status.address.ok_or("the status has no device address")?;
    let mut push = pre.push_constants.clone();
    for hole in &pre.push_holes {
        let (offset, address) = match *hole {
            PushHole::RootTableAddress { offset, .. } => {
                (offset, scratch_address + layout.resident_offset)
            }
            PushHole::ExecutionStatusAddress { offset, .. } => (offset, status_address),
            PushHole::CommandRecordsAddress { offset, .. } => (offset, storage),
            other => return Err(not_implemented(format!("the push hole {other:?}"))),
        };
        push.get_mut(offset as usize..offset as usize + 8)
            .ok_or_else(|| fail(format!("the preprocess push bytes end before {offset}")))?
            .copy_from_slice(&address.to_le_bytes());
    }
    let mut objects = preprocess_objects(context, (&module.spirv, &module.layout), "preprocess")?;
    record_and_submit(
        context,
        &mut objects,
        &replayed,
        Recorded {
            layout: &layout,
            group_count: pre.group_count,
            push: &push,
            scratch: scratch.buffer.buffer,
            scratch_address,
            status_address,
        },
    )?;
    // SAFETY: the submission completed (the fence signaled), and nothing else maps it.
    let status_bytes: [u8; ExecutionStatus::BYTES] = unsafe { status.buffer.map()? }.bytes
        [..ExecutionStatus::BYTES]
        .try_into()
        .expect("the status buffer holds ExecutionStatus::BYTES bytes");
    let report = ExecutionStatus::decode(&status_bytes);
    if report.replay_fault {
        return Err(replay_fault_verdict(
            &format!("command buffer {key:?}"),
            report.replay_faults,
        ));
    }
    status_verdict(&status_bytes).map_err(|e| match e {
        ExecutorError::Failed(why) => fail(why),
        refused => refused,
    })?;
    // Each variant a command's arguments (or sequences) selected in any region (every group
    // count nonzero), once per command.
    // SAFETY: as for the status buffer.
    let arguments = unsafe { scratch.buffer.map()? };
    let selected = |slot: u32, index: usize| -> Result<bool, ExecutorError> {
        let mut selected = false;
        for region in 0..layout.regions {
            let (variant, at) = match &replayed {
                Replayed::Dispatches(_) => (
                    index,
                    layout.argument_offset(slot, region, index as u32) as usize,
                ),
                Replayed::Generated(_, generated) => {
                    let at = generated.sequence_offset(slot, region);
                    let word = arguments
                        .bytes
                        .get(at as usize..at as usize + 4)
                        .ok_or_else(|| {
                            fail(format!("the sequences of command {slot} end early"))
                        })?;
                    let variant = u32::from_le_bytes(word.try_into().expect("four bytes"));
                    let at = at + u64::from(generated.dispatch_offset());
                    (variant as usize, at as usize)
                }
            };
            if variant >= stages.len() {
                return Err(fail(format!(
                    "a sequence of command {slot} selects variant {variant} of {}",
                    stages.len()
                )));
            }
            selected |= variant == index
                && arguments
                    .bytes
                    .get(at..at + 12)
                    .ok_or_else(|| fail(format!("the arguments of command {slot} end early")))?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|word| *word != [0; 4]);
        }
        Ok(selected)
    };
    let mut executed = Vec::new();
    for slot in 0..layout.slots {
        for (index, stage) in stages.iter().enumerate() {
            if selected(slot, index)? {
                executed.extend(module_censuses(
                    std::slice::from_ref(stage),
                    device.profile(),
                    None,
                )?);
            }
        }
    }
    drop(arguments);
    drop(objects);
    let path = match replayed {
        Replayed::Dispatches(_) => ReplayPath::Device,
        Replayed::Generated(..) => ReplayPath::Generated,
    };
    Ok((executed, path))
}

/// What the replay records after the preprocess and its barrier: each variant's indirect
/// dispatches per command and region (K14c), or the generated-commands executions of the
/// preprocess's token stream, with its layout.
enum Replayed<'c> {
    Dispatches(Vec<Variant<'c>>),
    Generated(Box<generated::Generated<'c>>, GeneratedLayout),
}

/// Refuse the preprocess where the context did not enable what it requires.
fn require_preprocess(
    context: &VulkanContext,
    module: &ReplayPreprocessModule,
) -> Result<(), ExecutorError> {
    let missing = unmet_requirements(context, &module.requirements);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(NotSupported::NotEnabled(missing).into())
    }
}

/// The scratch's initial bytes: the parameters, the resident root table with its sink entry's
/// `host` field at `sink`, the resident roots' extents, the variants, and the parent's state,
/// each at its layout offset; the slot tables and arguments are the preprocess's to write.
fn scratch_bytes(pre: &metal2vulkan::bind::ReplayPreprocess, sink: u64) -> Result<Vec<u8>, String> {
    let layout = pre.layout;
    let len = usize::try_from(layout.bytes).map_err(|_| "the scratch exceeds the address space")?;
    let mut bytes = vec![0u8; len];
    let mut resident = pre.resident.bytes.clone();
    let at = RootTable::SINK_HOST_OFFSET as usize;
    resident
        .get_mut(at..at + 8)
        .ok_or("the resident root table has no sink entry")?
        .copy_from_slice(&sink.to_le_bytes());
    // The generated path's variant templates, which its preprocess reads.
    let templates = match (&pre.generated, &layout.generated) {
        (Some(replay), Some(generated)) => Some((generated.templates_offset, &replay.templates)),
        (None, None) => None,
        _ => return Err("the preprocess states half of the generated-commands path".into()),
    };
    for (offset, part) in [
        (layout.params_offset(), &pre.params),
        (layout.resident_offset, &resident),
        (layout.extents_offset, &pre.extents),
        (layout.variants_offset, &pre.variants),
        (layout.parent_offset, &pre.parent),
    ]
    .into_iter()
    .chain(templates)
    {
        let offset = offset as usize;
        bytes
            .get_mut(offset..offset + part.len())
            .ok_or("a scratch section runs past the scratch")?
            .copy_from_slice(part);
    }
    Ok(bytes)
}

/// One registered variant, compiled, planned, and instantiated (no Vulkan object yet): its stage
/// module (for the census), its pipeline, its plan, and its replay commands.
struct Prepared {
    stage: StageModule,
    pipeline: ComputePipeline,
    plan: BindingPlan,
    commands: DispatchCommands,
    /// What a failure of this variant names.
    name: String,
}

impl Prepared {
    /// The variant as the generated-commands path binds it.
    fn generated(&self) -> GeneratedVariant<'_> {
        GeneratedVariant::new(&self.pipeline, &self.plan, &self.commands)
    }

    fn fail(&self, why: String) -> ExecutorError {
        ExecutorError::Failed(format!("{}: {why}", self.name))
    }

    /// The constant blob a write of the plan may read, by stage.
    fn constant_blobs(&self) -> BTreeMap<Stage, Arc<[u8]>> {
        self.pipeline
            .stage
            .constant_blob
            .iter()
            .map(|blob| (Stage::Kernel, blob.clone()))
            .collect()
    }

    /// What the plan's writes name: a variant binds no kernel buffer (each command's slot table
    /// names its buffers), and its descriptor contents are its commands'.
    fn resources<'c>(
        &self,
        context: &'c VulkanContext,
        device: &Device,
    ) -> Result<PlanResources<'c>, ExecutorError> {
        PlanResources::create(
            context,
            device,
            &self.plan.descriptors,
            &Guest {
                buffers: &BTreeMap::new(),
                textures: &BTreeMap::new(),
                texel_guards: &BTreeMap::new(),
                tile_attachments: &BTreeMap::new(),
                shadow_stencil: None,
                status: None,
                used_textures: &BTreeMap::new(),
                samplers: &BTreeMap::new(),
                // A generated-commands variant's state lists no argument sampler
                // (`replay_variant_state`).
                argument_samplers: &[],
                constant_blobs: &self.constant_blobs(),
                descriptor_ranges: &self.commands.descriptor_ranges,
                descriptor_bytes: &self.commands.descriptor_bytes,
                structures: &BTreeMap::new(),
                tessellation_factors: None,
                attachments: &BTreeMap::new(),
                imageblock: None,
            },
            None,
        )
    }
}

/// Every registered variant, [`prepare`]d, in their order.
fn prepare_all(
    context: &VulkanContext,
    device: &Device,
    registered: &[ReplayVariant],
    resident: &BindState,
    compile: &mut CompileChild<'_>,
) -> Result<Vec<Prepared>, ExecutorError> {
    let mut prepared = Vec::with_capacity(registered.len());
    for variant in registered {
        prepared.push(prepare(context, device, variant, resident, compile)?);
    }
    Ok(prepared)
}

/// Compile `variant` in a compile worker for the bind state of `resident` with its threadgroup
/// lengths and no kernel buffer, plan it (no diagnostic), and instantiate it with
/// `instantiate_replay` (one region with the replay's indirect arguments and no root table).
fn prepare(
    context: &VulkanContext,
    device: &Device,
    variant: &ReplayVariant,
    resident: &BindState,
    compile: &mut CompileChild<'_>,
) -> Result<Prepared, ExecutorError> {
    let token = variant.pipeline.get();
    let name = format!(
        "the replay variant of pipeline token {token} for {:?}",
        variant.threads_per_threadgroup
    );
    let fail = |why: String| ExecutorError::Failed(format!("{name}: {why}"));
    let state = replay_variant_state(resident, variant);
    let shape = state.shape();
    let child = ChildCommand::ReplayVariant {
        threads_per_group: variant.threads_per_threadgroup,
    };
    let pipeline = match compile(token, &shape, &child)? {
        CompiledChild::Compute(compiled) => compiled.pipeline,
        CompiledChild::Render(..) => return Err(fail("the token names a render template".into())),
    };
    require_enabled(context, &pipeline.requirements)?;
    let plan = pipeline
        .plan(device, &shape)
        .map_err(|e| bind_refusal(&e))?;
    if let Some(diagnostic) = plan.diagnostics.first() {
        return Err(incompatible_binding(diagnostic));
    }
    let commands = plan
        .instantiate_replay(&state, variant.threads_per_threadgroup)
        .map_err(|e| bind_refusal(&e))?;
    let replayed = matches!(
        commands.regions.as_slice(),
        [region] if region.indirect == Some(IndirectArgs::Replay)
    );
    if !replayed || commands.root_table.is_some() {
        return Err(fail(format!(
            "the binder instantiated {} region(s), not one replayed region without a root \
             table",
            commands.regions.len()
        )));
    }
    Ok(Prepared {
        stage: pipeline.stage.clone(),
        pipeline,
        plan,
        commands,
        name,
    })
}

/// One registered variant set up for K14c's dispatches: its instantiated commands, its pipeline
/// (one region), its descriptor sets, and the objects behind them (the pipeline objects destroyed
/// before the plan's resources, a field declared after).
struct Variant<'c> {
    commands: DispatchCommands,
    sets: Vec<vk::DescriptorSet>,
    objects: Objects<'c>,
    resources: PlanResources<'c>,
}

impl<'c> Variant<'c> {
    /// Create the pipeline of `prepared` (with its own layout) and what its plan writes name.
    fn create(
        context: &'c VulkanContext,
        device: &Device,
        prepared: Prepared,
    ) -> Result<Variant<'c>, ExecutorError> {
        let mut resources = prepared.resources(context, device)?;
        let Prepared {
            pipeline, commands, ..
        } = &prepared;
        let layout = match LayoutObjects::create(&context.device, &pipeline.layout) {
            Ok(layout) => layout,
            Err(LayoutError::Shape(why) | LayoutError::Refused(why)) => {
                return Err(prepared.fail(why))
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
        // A replayed command's template: the shared ceiling (`ComputeCase::creation_bound`).
        create_pipelines(
            context,
            &mut objects,
            pipeline,
            &commands.regions,
            crate::budgets::PIPELINE_CREATION_CEILING,
        )?;
        let sets = resources.write_sets(context, &objects.layout)?;
        Ok(Variant {
            commands: prepared.commands,
            sets,
            objects,
            resources,
        })
    }
}

/// An owned kernel's (`what`: a replay's preprocess, a merge pass) module of `spirv`, pipeline, and
/// layout (its own, `layout`: [`ReplayPreprocessModule::layout`] and the like), with the command
/// pool and fence of the replay's submission still to create.
pub(super) fn preprocess_objects<'c>(
    context: &'c VulkanContext,
    (spirv, layout): (&[u32], &metal2vulkan::pipeline::PipelineLayout),
    what: &str,
) -> Result<Objects<'c>, ExecutorError> {
    let layout = match LayoutObjects::create(&context.device, layout) {
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
    let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
    // SAFETY: `module_info` points at live SPIR-V words for the call.
    objects.module = unsafe { context.device.create_shader_module(&module_info, None) }
        .map_err(|error| format!("create the {what}'s shader module: {error}"))?;
    let entry = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(objects.module)
        .name(entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(objects.layout.pipeline_layout);
    let clock = crate::budgets::CreationClock::start();
    // SAFETY: every pointer in `info` outlives the call; the result is destroyed with `objects`.
    let created = unsafe {
        context
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, error)| format!("the driver refused the {what} pipeline: {error}"))?;
    objects.pipelines.extend(created.first().copied());
    crate::budgets::creation_within_ceiling(&format!("the {what} pipeline"), clock.read())?;
    Ok(objects)
}

/// What the replay's command buffer records beyond the pipelines: the scratch's layout, buffer,
/// and address, the preprocess's group count and push bytes, and the status's address.
struct Recorded<'a> {
    layout: &'a ReplayLayout,
    group_count: [u32; 3],
    push: &'a [u8],
    scratch: vk::Buffer,
    scratch_address: u64,
    status_address: u64,
}

/// Record the variants' uploads, the preprocess, its barrier, and per command in range order
/// either per region and per variant the variant's indirect dispatch from the arguments of the
/// command's region with the command's slot table and the region's record, or the command's
/// generated-commands execution ([`generated::record_executions`]); a compute barrier between
/// commands, a shader-to-host barrier at the end. Submit and wait under the hang protocol.
fn record_and_submit(
    context: &VulkanContext,
    objects: &mut Objects<'_>,
    replayed: &Replayed<'_>,
    recorded: Recorded<'_>,
) -> Result<(), ExecutorError> {
    let Recorded {
        layout,
        group_count,
        push,
        scratch,
        scratch_address,
        status_address,
    } = recorded;
    let variants: &[Variant<'_>] = match replayed {
        Replayed::Dispatches(variants) => variants,
        Replayed::Generated(..) => &[],
    };
    // A variant's push writes for a command's region: its slot table, its region record, and the
    // status. Each is written as its dispatch is recorded, so what the recording holds stays one
    // variant's writes whatever the range.
    let writes = |variant: &Variant<'_>, slot: u32, region: u32| {
        push_writes(&variant.commands, |hole| match *hole {
            PushHole::RootTableAddress { offset, .. } => {
                Ok((offset, scratch_address + layout.table_offset(slot)))
            }
            PushHole::ReplayRegionAddress { offset, .. } => {
                Ok((offset, scratch_address + layout.region_offset(slot, region)))
            }
            PushHole::ExecutionStatusAddress { offset, .. } => Ok((offset, status_address)),
            other => Err(not_implemented(format!("the push hole {other:?}"))),
        })
        .map(|writes| writes.into_iter().next().unwrap_or_default())
    };
    // Every hole is one the recording fills, before anything is recorded.
    for variant in variants {
        writes(variant, 0, 0)?;
    }
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
    // The generated path's executions read the token stream in their preprocessing stage too.
    let (preprocess_stage, preprocess_read) = match replayed {
        Replayed::Dispatches(_) => (vk::PipelineStageFlags::empty(), vk::AccessFlags::empty()),
        Replayed::Generated(..) => (generated::PREPROCESS_STAGE, generated::PREPROCESS_READ),
    };
    let preprocessed = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ | preprocess_read,
        );
    let between = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
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
        for variant in variants {
            variant.resources.record_uploads(command);
        }
        if let Replayed::Generated(generated, _) = replayed {
            generated.resources.record_uploads(command);
        }
        device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            objects.pipelines[0],
        );
        device.cmd_push_constants(
            command,
            objects.layout.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push,
        );
        let [x, y, z] = group_count;
        device.cmd_dispatch(command, x, y, z);
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::DRAW_INDIRECT
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | preprocess_stage,
            vk::DependencyFlags::empty(),
            &[preprocessed],
            &[],
            &[],
        );
        if let Replayed::Generated(generated, generated_layout) = replayed {
            generated::record_executions(
                device,
                command,
                generated,
                generated_layout,
                scratch_address,
                &between,
            )?;
        }
        if let Replayed::Dispatches(variants) = replayed {
            for slot in 0..layout.slots {
                if slot > 0 {
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[between],
                        &[],
                        &[],
                    );
                }
                for (index, variant) in variants.iter().enumerate() {
                    let pipeline_layout = variant.objects.layout.pipeline_layout;
                    if !variant.sets.is_empty() {
                        device.cmd_bind_descriptor_sets(
                            command,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline_layout,
                            0,
                            &variant.sets,
                            &[],
                        );
                    }
                    device.cmd_bind_pipeline(
                        command,
                        vk::PipelineBindPoint::COMPUTE,
                        variant.objects.pipelines[0],
                    );
                    // A command's regions are one Metal dispatch: no barrier between them.
                    for region in 0..layout.regions {
                        for (stages, offset, bytes) in writes(variant, slot, region)? {
                            device.cmd_push_constants(
                                command,
                                pipeline_layout,
                                stages,
                                offset,
                                &bytes,
                            );
                        }
                        device.cmd_dispatch_indirect(
                            command,
                            scratch,
                            layout.argument_offset(slot, region, index as u32),
                        );
                    }
                }
            }
        }
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
    context.wait_for_fence(objects.fence, "engine candidate device replay")?;
    Ok(())
}
