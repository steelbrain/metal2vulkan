//! Device replay of executed render commands in the engine executors: an execution of render
//! commands runs on the device where the runtime's GPU form draws it, by the rules both share
//! (`bind::replay`'s draw replay rules), so the executor compiles and draws what the runtime does,
//! and the records never return to the host to decide what runs.
//!
//! - **Which executions.** The rules decide from the case's facts alone, never from the records:
//!   a range of commands of a buffer of draws of one primitive type alone under the encoder's
//!   pipeline state, with their own buffers or the parent's where they inherit them
//!   (`draw_replay_primitive_type`, `DrawReplay::inheritance`), from what the records
//!   may hold: the host's setters' (`CommandBufferResource::host_writes`) and what the functions of
//!   the case's own dispatch or draw may write where they write render commands
//!   (`PipelineInterface::command_writes`, its `kinds`: a route reaches the buffers of its kind
//!   alone, as the runtime's does; its encoder makes every command buffer resident for writing, as
//!   the case runner's does); the inherited
//!   state's functions (`draw_replay_state`) and attributes (`draw_replay_vertex_layouts`: the
//!   variant pulls every fixed-function layout, `variants::compile_render_for_shape`), and its
//!   draw instantiated in the variant's form (`draw_replay_draw`); and a device with device
//!   addresses (`draw_replay_module`, `draw_replay_preprocess`). The runtime's rules of its own hold of every case the runner
//!   makes: it passes no check, counts no visibility result, and its pass (the template's
//!   scaffolding) reaches no tile memory and draws nothing before the execution. An execution a
//!   rule keeps from the device, or whose variant, plan, draw or preprocess the product refuses,
//!   runs the host replay (`replay::execute_commands`), as the runtime records it at its host step.
//! - **What runs.** The parent's render template's draw replay variant (`ChildCommand::
//!   DrawReplay`), compiled in a compile worker for the bind state of the execution's encoder (the
//!   parent's vertex buffers bound, every allocation of the run resident in the vertex and fragment
//!   stages, as the oracle's `useResource` makes them): merged into one draw where the commands
//!   draw a list type or points and the merged form's rules hold (`draw_replay_merges`), else one
//!   draw per command. One command buffer records the product's preprocess over the range (its
//!   scratch holding the parameters, the resident root table of every allocation with the sink
//!   filled, the extents, the parent's buffers and a merged replay's header), a merged replay's two
//!   merge passes (`draw_merge_module`), then, in the template's pass, the variant's draws: per
//!   command one `vkCmdDrawIndirect` of one draw at its arguments with its slot tables at the
//!   variant's root-table holes, or the merged draw's two (`render::submit_replayed_draws`).
//! - **After the wait**, the status: a replay fault refuses the row for the product's typed limits
//!   and fails it otherwise (`engine::replay_fault_verdict`), then the status verdict. The census
//!   holds the variant once per execution: the one draw of it the runtime's execution record holds.
//!   The artifact holds it as a pipeline the authored compile census does not compile, as the
//!   runtime path's holds every pipeline its executed commands ran.

use super::device_replay::{preprocess_objects, ReplayPath};
use super::replay::{bound_table, require_enabled, Executed};
use super::*;
use crate::candidate::engine::argument_buffers::used_buffer;
use crate::candidate::engine::graphics::RenderCase;
use crate::candidate::engine::{replay_fault_verdict, replay_resident};
use crate::candidate::platform::render::{
    submit_replayed_draws, OwnedPass, ReplayedDraw, ReplayedDraws,
};
use crate::case::indirect_commands::{authored_primitive, CommandBufferResource};
use metal2vulkan::bind::indirect_commands::{
    BufferStage, CommandRecord, Inheritance, RegisteredBuffer, StateSource,
};
use metal2vulkan::bind::{
    draw_merge_module, draw_replay_draw, draw_replay_merges, draw_replay_module,
    draw_replay_preprocess, draw_replay_primitive_type, draw_replay_slots, draw_replay_state,
    draw_replay_vertex_layouts, BindState, DrawCall, DrawCommands, DrawMergeModule, DrawReplay,
    DrawReplayPreprocess, IndirectRef, MergePhase,
};
use metal2vulkan::error::Error;
use metal2vulkan::metal::RenderStages;
use metal2vulkan::pipeline::{CommandWrites, RenderPipeline};

/// What one render command buffer's device replay runs over: the run's allocations `live`, the
/// guest address of each case buffer of the command buffer's owner (`addresses`, for the parent's
/// bindings), the record storage's device address (`storage`, none without device addresses), and
/// what the functions of the case's own dispatch or draw may write into the records of a buffer of
/// their kinds (`writes`).
pub(super) struct Drawing<'a, K> {
    pub(super) live: &'a [Live],
    pub(super) addresses: &'a dyn Fn(K, u32) -> Option<u64>,
    pub(super) storage: Option<u64>,
    pub(super) writes: CommandWrites,
}

/// Replay the execution of render command buffer `key` on the device where the shared rules take
/// it (the module docs), compiling its variant with `compile`; `None` where they send it to the
/// host replay. The census of the variant that drew, and which form ran.
pub(super) fn draw_commands<K: Ord + Copy + std::fmt::Debug>(
    context: &VulkanContext,
    device: &Device,
    (key, command): (K, &CommandBufferResource),
    drawing: Drawing<'_, K>,
    compile: &mut CompileChild<'_>,
) -> Result<Option<(Executed, ReplayPath)>, ExecutorError> {
    let Drawing {
        live,
        addresses,
        storage,
        writes,
    } = drawing;
    let fail = |why: String| ExecutorError::Failed(format!("command buffer {key:?}: {why}"));
    // The call's rules and the buffer's, from what the records may hold.
    let [start, end] = command.execution_range;
    let count = end
        .checked_sub(start)
        .ok_or_else(|| fail("a reversed execution range".into()))?;
    let Some(execution) = command.execute.as_ref().filter(|_| count > 0) else {
        return Ok(None);
    };
    // The case's functions reach these records where they write render commands: a kernel that
    // encodes dispatches writes no draw.
    let writes = match writes.writes_kinds(CommandWrites::RENDER_COMMANDS) {
        true => writes,
        false => CommandWrites::NONE,
    };
    let written = command.host_writes().map_err(fail)?.union(writes);
    let types = command.descriptor.command_types();
    let inherits = command.inherit_pipeline_state;
    let Ok(primitive_type) = draw_replay_primitive_type(types, inherits, written) else {
        return Ok(None);
    };
    let primitive = authored_primitive(primitive_type.as_raw()).ok_or_else(|| {
        fail(format!(
            "primitive type {primitive_type:?} is no Metal primitive"
        ))
    })?;
    let token = (execution.pipeline.as_ref())
        .ok_or_else(|| fail("the commands inherit a pipeline the parent does not set".into()))?
        .token;
    // The device sets the draw replay up where it has device addresses.
    let module = match draw_replay_module(device) {
        Ok(module) => module,
        Err(Error::Unsupported(_)) => return Ok(None),
        Err(error) => return Err(product_refusal(&error)),
    };
    let addressed = context
        .enabled
        .features
        .contains(&VkFeature::BufferDeviceAddress);
    let storage = match (addressed, storage) {
        (true, Some(storage)) => storage,
        _ => {
            return Err(fail(
                "the draw replay needs device addresses the context did not enable".into(),
            ))
        }
    };
    let parent = command
        .parent_bindings(|buffer| addresses(key, buffer))
        .map_err(ExecutorError::Failed)?
        .ok_or_else(|| fail("an execution without its parent's state".into()))?;

    // The execution's encoder's bind state: its vertex buffers (the parent's), and every
    // allocation of the run resident in the vertex and fragment stages, as the oracle's
    // `useResource` makes it. No render template has a fragment stage.
    let registered = live
        .iter()
        .map(|allocation| {
            RegisteredBuffer::new(allocation, allocation.address.guest, allocation.length)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ExecutorError::Failed(error.to_string()))?;
    let encoder = Inheritance {
        pipeline: StateSource::Parent,
        buffers: StateSource::Parent,
    };
    let empty = CommandRecord::default();
    let resolved = empty
        .resolve_bindings(encoder, &parent)
        .map_err(|error| fail(error.to_string()))?;
    let (table, guest) =
        bound_table(&resolved, Stage::Vertex, BufferStage::Vertex, &registered).map_err(fail)?;
    let mut state = BindState::default();
    state.stages.insert(Stage::Vertex, table.clone());
    for (token, allocation) in live.iter().enumerate() {
        state.used_resources.push(
            used_buffer(
                token,
                allocation.length as usize,
                allocation.address,
                RenderStages::VERTEX | RenderStages::FRAGMENT,
            )
            .map_err(ExecutorError::Failed)?,
        );
    }
    // The preprocess's resident set: every allocation of the run, as the host step resolves a
    // command's buffers in it; and the encoder's vertex buffers, which commands that inherit them
    // bind (`draw_replay_preprocess`'s state).
    let mut resident = replay_resident(
        live.iter()
            .map(|allocation| (allocation.address, allocation.length)),
    )
    .map_err(ExecutorError::Failed)?;
    resident.stages.insert(Stage::Vertex, table);
    let mut replay = DrawReplay::new(
        command.capacity,
        start,
        count,
        vec![primitive_type.as_raw()],
    );
    replay.inheritance = command.inheritance();

    // One draw per command's variant first: its functions' facts are the state's, which decide
    // whether the merged form may run at all, as the runtime's state decides it before it compiles
    // the merged variant. Merged where the commands draw a list type or points and its rules hold.
    let at = (context, device, &state, &resident);
    let Some(per_command) = compiled(token, &state, false, compile)? else {
        return Ok(None);
    };
    let merges = draw_replay_merges(primitive_type)
        && draw_replay_state(&per_command.1.graphics, true).is_ok();
    let merged = match merges {
        true => match compiled(token, &state, true, compile)? {
            Some(merged) => drawn(at, merged, &replay, true)?,
            None => None,
        },
        false => None,
    };
    let variant = match merged {
        Some(merged) => merged,
        None => match drawn(at, per_command, &replay, false)? {
            Some(variant) => variant,
            None => return Ok(None),
        },
    };

    // The replay's objects: the sink the resident root table names, the scratch, the status, and
    // the owned kernels' pipelines.
    let Variant {
        merged,
        case,
        pipeline,
        plan,
        commands,
        pre,
    } = variant;
    let layout = pre.layout;
    require_enabled(context, &module.requirements)?;
    let sink = allocate(
        context,
        "draw replay sink",
        &[0; NeutralResource::SINK_BYTES as usize],
        addressed,
        vk::BufferUsageFlags::empty(),
    )?;
    let sink_address = sink.address.ok_or("the sink has no device address")?;
    let scratch = allocate(
        context,
        "draw replay scratch",
        &scratch_bytes(&pre, sink_address)?,
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
        "draw replay status",
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
            other => return Err(fail(format!("the preprocess's push hole {other:?}"))),
        };
        push.get_mut(offset as usize..offset as usize + 8)
            .ok_or_else(|| fail(format!("the preprocess push bytes end before {offset}")))?
            .copy_from_slice(&address.to_le_bytes());
    }
    let mut kernels = vec![preprocess_objects(
        context,
        (&module.spirv, &module.layout),
        "draw replay preprocess",
    )?];
    let mut pushes = vec![(push, pre.group_count)];
    if let Some(sections) = layout.merged {
        for phase in [MergePhase::Blocks, MergePhase::Starts] {
            let module = draw_merge_module(device, phase).map_err(|e| product_refusal(&e))?;
            require_enabled(context, &module.requirements)?;
            kernels.push(preprocess_objects(
                context,
                (&module.spirv, &module.layout),
                "draw merge pass",
            )?);
            let header = scratch_address + sections.header_offset;
            pushes.push((
                DrawMergeModule::push_constants(header, status_address),
                sections.group_count(),
            ));
        }
    }
    let passes: Vec<OwnedPass> = kernels
        .iter()
        .zip(pushes)
        .map(|(kernel, (push, group_count))| OwnedPass {
            pipeline: kernel.pipelines[0],
            layout: kernel.layout.pipeline_layout,
            push,
            group_count,
        })
        .collect();

    // The draws: per command its arguments, after the push writes of its slot tables at the
    // variant's root-table holes (the first command's the draw's own push writes carry); or the
    // merged draw's two, its header at the variant's merged-draw hole.
    let table = |hole: &PushHole, command: u32| match *hole {
        PushHole::RootTableAddress { stages, .. } => {
            let stages = vk::ShaderStageFlags::from(stages);
            let offset = if stages == vk::ShaderStageFlags::VERTEX {
                layout.vertex_table_offset(command)
            } else if stages == vk::ShaderStageFlags::FRAGMENT {
                layout.fragment_table_offset(command)
            } else {
                return None;
            };
            Some(scratch_address + offset)
        }
        PushHole::MergedDrawAddress { .. } => {
            (layout.merged).map(|sections| scratch_address + sections.header_offset)
        }
        _ => None,
    };
    let draws: Vec<ReplayedDraw> = match layout.merged {
        Some(sections) => (0..2)
            .map(|index| ReplayedDraw {
                push: Vec::new(),
                arguments: sections.draw_offset(index),
            })
            .collect(),
        None => (0..layout.commands)
            .map(|command| ReplayedDraw {
                push: match command {
                    0 => Vec::new(),
                    _ => (commands.push_holes.iter())
                        .filter_map(|hole| match *hole {
                            PushHole::RootTableAddress { stages, offset } => {
                                let address = table(hole, command)?;
                                Some((stages.into(), offset, address.to_le_bytes().to_vec()))
                            }
                            _ => None,
                        })
                        .collect(),
                },
                arguments: layout.argument_offset(command, 0),
            })
            .collect(),
    };
    submit_replayed_draws(
        context,
        device,
        (&case, primitive),
        (&pipeline, &plan, &commands),
        (&state, &guest),
        (&status, &|hole: &PushHole| table(hole, 0)),
        &ReplayedDraws {
            passes: &passes,
            scratch: scratch.buffer.buffer,
            draws: &draws,
        },
    )
    .map_err(|error| match error {
        ExecutorError::Failed(why) => fail(why),
        refused => refused,
    })?;
    // SAFETY: the submission completed (the fence signaled), and nothing else maps it.
    let status_bytes: [u8; ExecutionStatus::BYTES] = unsafe { status.buffer.map()? }.bytes
        [..ExecutionStatus::BYTES]
        .try_into()
        .map_err(|_| "the status buffer is shorter than ExecutionStatus")?;
    let report = ExecutionStatus::decode(&status_bytes);
    if report.replay_fault {
        return Err(replay_fault_verdict(
            &format!("command buffer {key:?}"),
            report.replay_faults,
        ));
    }
    status_verdict(&status_bytes).map_err(|error| match error {
        ExecutorError::Failed(why) => fail(why),
        refused => refused,
    })?;
    drop(kernels);
    let census = module_censuses(&pipeline.stages, device.profile(), None)?;
    let path = match merged {
        true => ReplayPath::MergedDraws,
        false => ReplayPath::Draws,
    };
    Ok(Some((census, path)))
}

/// The draw replay variant of the template of `token`, `merged` into one draw or not, compiled in
/// a compile worker for the shape of `state`: its template's case and pipeline; `None` where the
/// product refuses it, as the runtime's GPU form takes its host step where its state does not
/// settle the variant.
fn compiled(
    token: u64,
    state: &BindState,
    merged: bool,
    compile: &mut CompileChild<'_>,
) -> Result<Option<(RenderCase, RenderPipeline)>, ExecutorError> {
    match compile(token, &state.shape(), &ChildCommand::DrawReplay { merged }) {
        Ok(CompiledChild::Render(case, compiled)) => Ok(Some((*case, compiled.pipeline))),
        Ok(CompiledChild::Compute(_)) => Err(ExecutorError::Failed(format!(
            "an executed command's pipeline token {token} names a compute template for render \
             commands"
        ))),
        Err(ExecutorError::NotSupported(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

/// One form's variant ready to draw: its template's case and pipeline, the plan and the one
/// indirect draw instantiated for the encoder's bind state, and the replay's preprocess.
struct Variant {
    merged: bool,
    case: RenderCase,
    pipeline: RenderPipeline,
    plan: BindingPlan,
    commands: DrawCommands,
    pre: DrawReplayPreprocess,
}

/// The `compiled` variant in its form (`merged` or one draw per command) where the shared rules
/// hold of it: the inherited state's (`draw_replay_state`), its plan for `state` (no diagnostic)
/// and its one indirect draw instantiated for it (`draw_replay_draw`), and the product's
/// preprocess of `replay` over `resident` with the slots its functions read; `None` where a rule
/// breaks or the product refuses the plan, the draw or the preprocess, as the runtime's GPU form
/// takes its host step.
fn drawn(
    (context, device, state, resident): (&VulkanContext, &Device, &BindState, &BindState),
    (case, pipeline): (RenderCase, RenderPipeline),
    replay: &DrawReplay,
    merged: bool,
) -> Result<Option<Variant>, ExecutorError> {
    // Its attributes: none whose layout's stride is set per bind (the variant pulls the rest).
    if draw_replay_state(&pipeline.graphics, merged).is_err()
        || draw_replay_vertex_layouts(&pipeline.vertex_input).is_err()
    {
        return Ok(None);
    }
    require_enabled(context, &pipeline.requirements)?;
    let Ok(plan) = pipeline.plan(device, &state.shape()) else {
        return Ok(None);
    };
    if let Some(diagnostic) = plan.diagnostics.first() {
        return Err(incompatible_binding(diagnostic));
    }
    let draw = DrawCall::Indirect {
        indirect: IndirectRef::new(0, 0),
        indexed: None,
    };
    let Ok(commands) = plan.instantiate_draw(state, &draw) else {
        return Ok(None);
    };
    if draw_replay_draw(&commands, (&plan, &plan.variant), merged).is_err() {
        return Ok(None);
    }
    let mut replay = replay.clone();
    (replay.vertex_slots, replay.fragment_slots) = draw_replay_slots(&pipeline.interface);
    replay.merged = merged;
    let Ok(pre) = draw_replay_preprocess(device, resident, &replay) else {
        return Ok(None);
    };
    Ok(Some(Variant {
        merged,
        case,
        pipeline,
        plan,
        commands,
        pre,
    }))
}

/// The scratch's initial bytes: the parameters, the resident root table with its sink entry's
/// `host` field at `sink`, the resident roots' extents, the parent's buffers, and a merged
/// replay's header, each at its layout offset; the slot tables and arguments (and a merged
/// replay's sections past its header) are the preprocess's and the passes' to write.
fn scratch_bytes(pre: &DrawReplayPreprocess, sink: u64) -> Result<Vec<u8>, String> {
    let layout = pre.layout;
    let len = usize::try_from(layout.bytes).map_err(|_| "the scratch exceeds the address space")?;
    let mut bytes = vec![0u8; len];
    let mut resident = pre.resident.bytes.clone();
    let at = RootTable::SINK_HOST_OFFSET as usize;
    resident
        .get_mut(at..at + 8)
        .ok_or("the resident root table has no sink entry")?
        .copy_from_slice(&sink.to_le_bytes());
    let header = (layout.merged).map(|sections| (sections.header_offset, &pre.header));
    for (offset, part) in [
        (layout.params_offset(), &pre.params),
        (layout.resident_offset, &resident),
        (layout.extents_offset, &pre.extents),
        (layout.parent_offset, &pre.parent),
    ]
    .into_iter()
    .chain(header)
    {
        let offset = offset as usize;
        bytes
            .get_mut(offset..offset + part.len())
            .ok_or("a scratch section runs past the scratch")?
            .copy_from_slice(part);
    }
    Ok(bytes)
}
