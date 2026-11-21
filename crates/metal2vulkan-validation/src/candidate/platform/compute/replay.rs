//! Host replay of executed indirect commands in the engine executors.
//!
//! After the producer's dispatch or draw completed and its records decoded under the host
//! contract, each command buffer the case executes runs its execution range in order, one command
//! at a time: the command's state (or the parent encoder's, where the descriptor inherits it)
//! selects a compiled template by its pipeline token, and binds each buffer address to the
//! allocation of the run that holds it at its interior offset (`BufferBinding::resolve_buffer`).
//! A dispatch binds the command's kernel buffers and threadgroup lengths, and its template is
//! compiled for that bind shape and the command's dispatch; a draw binds the command's vertex
//! buffers, and its template (a rasterization-disabled vertex function, which the checker
//! requires of every render template an executed command may select) is compiled for that bind
//! shape, drawn with the command's primitive, counts, and base instance in a pass of its own
//! (`render::submit_draw_child`). Each child is compiled in a compile worker, planned, and
//! submitted over the run's own allocations, each waited for before the next (an ordering
//! Metal's barriers, and its render encoder's draw order, allow). A record whose pipeline has no
//! template, whose address no allocation holds, whose primitive type Metal does not define or is
//! outside its template's topology class, whose dynamic attribute stride, stage-in region, or
//! imageblock size no template reads, or whose bind the plan refuses or asks another variant for
//! fails the run: nothing is guessed. A draw's fragment buffers pass the record decode's index
//! policy (`CommandBuffer::decode`) and are neither resolved nor bound: no render template has a
//! fragment stage.

use super::*;
use crate::candidate::engine::argument_buffers::used_buffer;
use crate::candidate::engine::{ChildCommand, CompiledChild};
use crate::case::indirect_commands::{authored_primitive, CommandBufferResource};
use metal2vulkan::bind::indirect_commands::{
    Action, BufferBinding, BufferStage, RegisteredBuffer, ResolvedBindings,
};
use metal2vulkan::bind::{BindState, BoundBuffer, Dispatch, DrawCall, StageTable};
use metal2vulkan::metal::RenderStages;

/// One allocation of the run a replayed command may bind or reach through a pointer.
pub(in crate::candidate::platform) struct Live {
    pub(in crate::candidate::platform) buffer: vk::Buffer,
    pub(in crate::candidate::platform) address: BufferAddress,
    pub(in crate::candidate::platform) length: u64,
}

/// Compile the template of a pipeline token for a bind shape and a command, in a compile worker.
pub(in crate::candidate) type CompileChild<'a> =
    dyn FnMut(u64, &BindShape, &ChildCommand) -> Result<CompiledChild, ExecutorError> + 'a;

/// What a replay executed: each child's module census, for the run's census.
pub(in crate::candidate::platform) type Executed = Vec<ModuleCensus>;

/// What a replay runs over: the run's allocations `live` (every case buffer and argument-buffer
/// buffer), the final records `records` (by command-buffer key), the guest
/// address of each case buffer of a command buffer's owner (`addresses`, by its key and the
/// buffer's Metal index, for the parent's bindings), and the compile of each child.
pub(in crate::candidate::platform) struct Replay<'a, 'b, K> {
    pub(in crate::candidate::platform) live: &'a [Live],
    pub(in crate::candidate::platform) records: &'a BTreeMap<K, Vec<u8>>,
    pub(in crate::candidate::platform) addresses: &'a dyn Fn(K, u32) -> Option<u64>,
    pub(in crate::candidate::platform) compile: &'a mut CompileChild<'b>,
}

/// Execute each of `command_buffers` (by key, in key order: the order the case's identity sorts
/// them in, and the oracle's, `case::indirect_commands::executed_in_key_order`) that the case
/// executes, as `replay` states, on the context's device.
pub(in crate::candidate::platform) fn execute_commands<'c, K: Ord + Copy + std::fmt::Debug>(
    context: &VulkanContext,
    device: &Device,
    command_buffers: impl IntoIterator<Item = (K, &'c CommandBufferResource)>,
    replay: Replay<'_, '_, K>,
) -> Result<Executed, ExecutorError> {
    let Replay {
        live,
        records,
        addresses,
        compile,
    } = replay;
    let mut executed = Vec::new();
    let executing = command_buffers
        .into_iter()
        .filter(|(_, command)| command.execute.is_some())
        .collect::<Vec<_>>();
    if executing.is_empty() {
        return Ok(executed);
    }
    let registered = live
        .iter()
        .map(|allocation| {
            RegisteredBuffer::new(allocation, allocation.address.guest, allocation.length)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ExecutorError::Failed(error.to_string()))?;
    for (key, command) in executing {
        let parent = match command
            .parent_bindings(|buffer| addresses(key, buffer))
            .map_err(ExecutorError::Failed)?
        {
            Some(parent) => parent,
            None => continue,
        };
        let bytes = records
            .get(&key)
            .ok_or_else(|| format!("command buffer {key:?} has no final records"))?;
        let decoded = command
            .validated_records(bytes)
            .map_err(ExecutorError::Failed)?;
        let start = command.execution_range[0];
        for (offset, record) in decoded.enumerate() {
            let index = start + offset as u32;
            let what = format!("command {index} of command buffer {key:?}");
            let fail = |why: String| ExecutorError::Failed(format!("{what}: {why}"));
            let resolved = record
                .resolve_bindings(command.inheritance(), &parent)
                .map_err(|error| fail(error.to_string()))?;
            if record.stage_in_region != [0; 6] || record.imageblock_size != [0; 2] {
                return Err(fail(
                    "a stage-in region or an imageblock size, which no replayed template reads"
                        .into(),
                ));
            }
            let child = match record.action {
                Action::Empty => continue,
                Action::DispatchThreadgroups { grid, local_size } => {
                    ChildCommand::Dispatch(Dispatch::Threadgroups {
                        groups: grid,
                        threads_per_group: local_size,
                    })
                }
                Action::DispatchThreads { grid, local_size } => {
                    ChildCommand::Dispatch(Dispatch::Threads {
                        threads: grid,
                        threads_per_group: local_size,
                    })
                }
                Action::DrawPrimitives {
                    primitive_type,
                    vertex_start,
                    vertex_count,
                    instance_count,
                    base_instance,
                } => ChildCommand::Draw {
                    draw: DrawCall::Primitives {
                        vertex_start,
                        vertex_count,
                        instance_count,
                        base_instance,
                    },
                    primitive: authored_primitive(primitive_type).ok_or_else(|| {
                        fail(format!(
                            "primitive type {primitive_type} is no Metal primitive"
                        ))
                    })?,
                },
                // The executor's context draws no index or patch buffer a record names, and no
                // mesh draw.
                Action::DrawIndexedPrimitives { .. }
                | Action::DrawPatches { .. }
                | Action::DrawIndexedPatches { .. }
                | Action::DrawMeshThreadgroups { .. }
                | Action::DrawMeshThreads { .. } => return Err(fail(
                    "an indexed, patch or mesh draw, which the executor's context does not replay"
                        .into(),
                )),
            };
            let token = resolved
                .pipeline
                .ok_or_else(|| fail("no pipeline".into()))?
                .get();
            // A dispatch binds the command's kernel buffers and threadgroup lengths, a draw its
            // vertex buffers (its template has no fragment stage).
            let (stage, buffers, stages) = match child {
                ChildCommand::Dispatch(_) => {
                    (Stage::Kernel, BufferStage::Kernel, RenderStages::empty())
                }
                ChildCommand::Draw { .. } => (
                    Stage::Vertex,
                    BufferStage::Vertex,
                    RenderStages::VERTEX | RenderStages::FRAGMENT,
                ),
                // A record's action is a dispatch or a draw, never a device replay's variant.
                ChildCommand::ReplayVariant { .. } | ChildCommand::DrawReplay { .. } => {
                    return Err(fail(
                        "a host-replayed command runs no replay variant".into(),
                    ))
                }
            };
            let (table, guest) =
                bound_table(&resolved, stage, buffers, &registered).map_err(fail)?;
            let mut state = BindState::default();
            state.stages.insert(stage, table);
            // Every allocation of the run is resident, as the oracle's `useResource` makes it.
            for (token, allocation) in live.iter().enumerate() {
                state.used_resources.push(
                    used_buffer(
                        token,
                        allocation.length as usize,
                        allocation.address,
                        stages,
                    )
                    .map_err(ExecutorError::Failed)?,
                );
            }
            let shape = state.shape();
            let stages = match (compile(token, &shape, &child)?, child) {
                (CompiledChild::Compute(pipeline), ChildCommand::Dispatch(dispatch)) => {
                    require_enabled(context, &pipeline.requirements)?;
                    submit_child(context, device, &pipeline, &state, &dispatch, &guest)
                        .map_err(|error| refine(error, &fail))?;
                    vec![pipeline.stage.clone()]
                }
                (CompiledChild::Render(case, pipeline), ChildCommand::Draw { primitive, .. }) => {
                    require_enabled(context, &pipeline.requirements)?;
                    super::super::render::submit_draw_child(
                        context, device, &case, &pipeline, &state, primitive, &guest,
                    )
                    .map_err(|error| refine(error, &fail))?;
                    pipeline.stages.clone()
                }
                _ => {
                    return Err(fail(format!(
                        "pipeline token {token} names a template of the other command family"
                    )))
                }
            };
            executed.extend(super::module_censuses(&stages, device.profile(), None)?);
        }
    }
    Ok(executed)
}

/// A failed child's error, named for its command.
fn refine(error: ExecutorError, fail: &impl Fn(String) -> ExecutorError) -> ExecutorError {
    match error {
        ExecutorError::Failed(why) => fail(why),
        other => other,
    }
}

/// Refuse a child whose requirements the context did not enable.
pub(super) fn require_enabled(
    context: &VulkanContext,
    requirements: &metal2vulkan::device::Requirements,
) -> Result<(), ExecutorError> {
    let missing = unmet_requirements(context, requirements);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(NotSupported::NotEnabled(missing).into())
    }
}

/// The bind table of `stage` from the command's resolved `buffers` bindings, each at the
/// allocation of `registered` that holds its address (at least one byte of it), and the
/// allocation each slot binds; for a dispatch, the command's nonzero threadgroup lengths too. A
/// dynamic attribute stride, which no replayed template reads (no template has stage inputs or
/// vertex inputs), or an address no allocation holds, is the command's failure.
pub(super) fn bound_table(
    resolved: &ResolvedBindings<'_>,
    stage: Stage,
    buffers: BufferStage,
    registered: &[RegisteredBuffer<'_, Live>],
) -> Result<(StageTable, BTreeMap<SlotRef, vk::Buffer>), String> {
    let mut table = StageTable::default();
    let mut guest = BTreeMap::new();
    for (slot, binding) in resolved.buffers(buffers).iter().enumerate() {
        // A slot no setter set binds nothing; a nil buffer's (its offset below 2^32) binds
        // nothing and reads no stride.
        let Some(binding) = binding.filter(BufferBinding::names_buffer) else {
            continue;
        };
        if binding.attribute_stride.is_some() {
            return Err(format!(
                "buffer {slot} has a dynamic attribute stride, which no replayed template reads"
            ));
        }
        let Some(found) = binding
            .resolve_buffer(registered, 1)
            .map_err(|error| format!("buffer {slot}: {error}"))?
        else {
            continue;
        };
        let slot = slot as u32;
        table.buffers.insert(
            slot,
            BoundBuffer::Buffer {
                length: found.resource.length,
                offset: found.offset,
                gpu_address: found.resource.address.guest,
                device_address: found.resource.address.host,
                attribute_stride: None,
            },
        );
        guest.insert(
            SlotRef {
                stage,
                class: SlotClass::Buffer,
                index: slot,
            },
            found.resource.buffer,
        );
    }
    if stage == Stage::Kernel {
        for (slot, &length) in resolved.threadgroup_memory_lengths.iter().enumerate() {
            if length != 0 {
                table.threadgroup_lengths.insert(slot as u32, length);
            }
        }
    }
    Ok((table, guest))
}

/// Plan `pipeline` for `state` (the plan must name no other variant and no neutral), create what
/// the plan names over the run's allocations `guest` (by slot), record and submit `dispatch`,
/// wait, and require a clean status.
fn submit_child(
    context: &VulkanContext,
    device: &Device,
    pipeline: &ComputePipeline,
    state: &BindState,
    dispatch: &Dispatch,
    guest: &BTreeMap<SlotRef, vk::Buffer>,
) -> Result<(), ExecutorError> {
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    let shape = state.shape();
    let plan = pipeline
        .plan(device, &shape)
        .map_err(|e| bind_refusal(&e))?;
    if let Some(diagnostic) = plan.diagnostics.first() {
        return Err(incompatible_binding(diagnostic));
    }
    let commands = plan
        .instantiate_dispatch(state, dispatch)
        .map_err(|e| bind_refusal(&e))?;
    if commands.regions.iter().any(|r| r.indirect.is_some()) {
        return Err(not_implemented("indirect dispatches"));
    }
    let status = allocate(
        context,
        "replayed command status",
        &ExecutionStatus::initial(),
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    let constant_blobs: BTreeMap<Stage, Arc<[u8]>> = pipeline
        .stage
        .constant_blob
        .iter()
        .map(|blob| (Stage::Kernel, blob.clone()))
        .collect();
    let mut resources = PlanResources::create(
        context,
        device,
        &plan.descriptors,
        &Guest {
            buffers: guest,
            textures: &BTreeMap::new(),
            status: Some(status.buffer.buffer),
            used_textures: &BTreeMap::new(),
            samplers: &BTreeMap::new(),
            argument_samplers: &state.argument_samplers,
            constant_blobs: &constant_blobs,
            descriptor_ranges: &commands.descriptor_ranges,
            descriptor_bytes: &commands.descriptor_bytes,
            tessellation_factors: None,
            attachments: &BTreeMap::new(),
        },
        None,
    )?;
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
    // An executed command's template: the shared ceiling (`ComputeCase::creation_bound`).
    let region_pipelines = create_pipelines(
        context,
        &mut objects,
        pipeline,
        &commands.regions,
        crate::budgets::PIPELINE_CREATION_CEILING,
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
    // SAFETY: the submission completed (the fence signaled), and nothing else maps it.
    let status_bytes: [u8; ExecutionStatus::BYTES] = unsafe { status.buffer.map()? }.bytes
        [..ExecutionStatus::BYTES]
        .try_into()
        .expect("the status buffer holds ExecutionStatus::BYTES bytes");
    status_verdict(&status_bytes)?;
    drop(objects);
    drop(resources);
    Ok(())
}
