//! The GPU form of an execution of render commands the GPU may write: its records never return
//! to the host. `commit` records the pass for good where every such execution of it takes this
//! form ([`DeferredPass::record_on_gpu`](super::DeferredPass::record_on_gpu)); otherwise a host
//! step records the pass from the records as the GPU and the CPU left them (the module docs of
//! `render/executions.rs`), which the device announces (`Degradation::HostReplayedIndirectCommands`),
//! naming the first rule below that the pass broke ([`HostStepCause`]).
//!
//! - **What takes it** ([`RenderExecution::gpu_form`]), decided at commit from the call and from
//!   what the buffer's records may hold (`written_commands.rs`), never from the records: an
//!   execution of a range given at the call, not empty, with no visibility result mode set, whose
//!   caller's check, if it has one (`execute_commands_in_buffer_checked`), has nothing to see
//!   there (`IndirectCommandCheckNeeds`: no draw of points, a primitive type the writes do not
//!   state counting as one, where it looks for them; the encoder's state encodes no indirect
//!   commands, where it looks for such states), so it is never called; of a buffer of draws of
//!   primitives alone (`IndirectCommandType::DRAW`) that inherits the encoder's pipeline state
//!   and not its buffers, and whose records the CPU and every route of a command buffer that made
//!   it resident for writing (`write_routes.rs`) may fill with draws of one primitive type,
//!   buffers, resets and cleared barriers alone; in a pass whose fragment stages wrote no bytes before the call (no copy of them is
//!   read, `render/pass_start_reads.rs`), with no tile threadgroup memory argument set, and where
//!   the pass has a shadowed texture, whose command buffer made resident past the call nothing
//!   that meets a shadowed texture's truth (`shadows::meets_truths`), as the host step's rule.
//!   The state is one the pass takes (`check_pipeline`), neither a tessellation nor a mesh one,
//!   whose fragment function writes no memory, nor its vertex function where it rasterizes (no
//!   command's draw waits on another's, `render_target::after_geometry_writes`), and whose vertex
//!   function fetches no attribute; its replay variant (`VariantKey::replay_bindings`) compiles
//!   and draws in the execution's instance as one indirect draw whose arguments the
//!   device reads (`DrawCommands::indirect_needs_arguments` clear): no amplification drawn apart,
//!   view, rate map, fragment imageblock, fetched or storage attachment, shader-side stencil or
//!   depth test, or color in another format. And the device has device addresses
//!   (`bind::draw_replay_preprocess`).
//! - **What runs.** Before the pass, the replay's preprocess (`bind::draw_replay_module`, an
//!   owned kernel) over the range: it reads each record where the GPU and the CPU left it, checks
//!   it as the host's decode does, and writes the command's vertex and fragment slot tables and
//!   its `VkDrawIndirectCommand` into the scratch, a region of the command buffer's upload arena;
//!   then a barrier from everything before to everything after. In the execution's instance, one
//!   draw per command, in order, of the inherited state's replay variant: the first command's
//!   draws through the encoder's draw (`RenderCommandEncoder::draw`, `BufferForms::Replay`), which
//!   binds the pipeline, the sets, the dynamic state and the push constants with its slot tables,
//!   and each later command's pushes its own slot tables and draws (`vkCmdDrawIndirect` of one
//!   draw at its arguments). Its `vertex_id`, `instance_id` and `base_instance` are the command's
//!   own, as Metal's.
//! - **Shadowed textures.** In a pass with one, the pass's copies go around its instances as for
//!   the host step and any draw (`render/shadows.rs`): a shadow is copied in before the instance
//!   that reads it and copied back, or merged, after the pass where it writes it; the commands'
//!   draws use the shadows their call gathered, the first command's draw binding them as the
//!   encoder's draw does, and the later ones drawing with its tables.
//! - **Merged** (`bind::DrawReplay::merged`), where the commands draw points, lines or triangles
//!   (a list type), the vertex function writes no memory, and the state's merged variant
//!   (`VariantKey::merged_draws`) compiles with a fragment function that reads no buffer and no
//!   `[[primitive_id]]`: the preprocess, then the two merge passes (`bind::draw_merge_module`,
//!   owned kernels), each after a barrier from everything before, write each command's first
//!   vertex in one merged draw and its two `VkDrawIndirectCommand`s; the pass draws them, the
//!   first through the encoder's draw (`BufferForms::Merged`), and the vertex function finds its
//!   command, vertex and instance (`bind::replay`'s merged docs), in the commands' order. One
//!   draw call for the whole range, whatever its count. Otherwise one draw per command, as
//!   above.
//! - **What the commands reach.** Their buffers' addresses resolve in the replay's resident set:
//!   the command buffer's resident resources, then the device's live buffers Metal reaches
//!   unnamed (`live_buffers.rs`), as the host step resolves them. One in neither is unbound
//!   (reading zero) and counted in the execution report (its resident misses), and the command
//!   draws, as the host step's.
//! - **Decided again as the batch is ready.** What the records may hold can grow after the
//!   commit: a CPU setter, or a route encoded since. Where it grew by the time the pass's batch is
//!   otherwise ready to submit (`DeferredPass::outgrown`, `Segment::needs_step`), the host step
//!   records the pass instead, as for any pass whose records the GPU may write, and calls the
//!   caller's check where it then has something to see. Every writer the runtime orders before the pass was committed by then: a
//!   queue's batches go in commit order, and a batch that waits for an event is ready only once a
//!   submitted signal reaches it. A CPU write after that races the GPU, as on Metal.
//! - **Faults.** A record the host's decode refuses draws nothing, and the execution report
//!   states why (`bind::ReplayFaults`), as the host step's; the others draw.

use std::ops::Range;
use std::sync::Arc;

use super::super::bound::{Bound, Recorded};
use super::super::line_width::LineRoute;
use super::super::pass_start_reads::PassStartTables;
use super::super::provoking::ProvokingForms;
use super::super::{Assembly, DrawBuffers, PrivateForm, Recording, RenderLowering};
use super::{HostStepCause, PassStorage, RenderExecution, Rendered, CALL};
use crate::binding::PushWrite;
use crate::blit::{reads_by, writes_by};
use crate::command_buffer::{CommandBuffer, ReplayTable};
use crate::device::lock;
use crate::draw_source::DrawSource;
use crate::encoder::EncoderCore;
use crate::error::{internal, Error};
use crate::executed_commands::RecordsRead;
use crate::indirect_execute::ExecutionRange;
use crate::owned_kernel::OwnedModule;
use crate::pipeline::{BufferForms, GraphicsState, RenderStateKind, Settled};
use crate::residency::Residency;
use crate::sampler::ArgumentSamplers;
use crate::tracker::{Access, Barrier};
use crate::visibility::Visibility;
use crate::Device;
use crate::RenderPipelineState;
use ash::vk;
use metal2vulkan::bind::{
    draw_replay_draw, draw_replay_merges, draw_replay_preprocess, draw_replay_primitive_type,
    draw_replay_slots, draw_replay_state, BindState, DrawCall, DrawMergeModule, DrawReplay,
    DrawReplayLayout, DrawReplayPreprocess, IndirectRef, MergePhase, NeutralResource, PushHole,
    DRAW_ARGUMENT_BYTES,
};
use metal2vulkan::metal::PrimitiveType;
use metal2vulkan::pipeline::CommandWrites;

/// An execution's GPU form, decided at commit ([`RenderExecution::gpu_form`]).
pub(super) struct GpuForm {
    /// What the buffer's records may hold, as the decision found it: a route encoded since that
    /// adds to it sends the pass to the host step (`DeferredPass::outgrown`).
    pub(super) decided: CommandWrites,
    /// The primitive type every command that draws draws.
    primitive_type: PrimitiveType,
    /// The replay's preprocess over the command buffer's resident set and the device's live
    /// buffers as they lay then, which the command buffer's first submission points at the set as
    /// it lies then.
    pre: DrawReplayPreprocess,
}

/// The slot table one push hole of a replayed draw names: the vertex function's or the fragment
/// function's.
#[derive(Clone, Copy)]
enum Table {
    Vertex,
    Fragment,
}

/// A push hole of a replayed draw that takes a command's slot table: its stages and offset.
pub(in crate::render) struct TableHole {
    stages: vk::ShaderStageFlags,
    offset: u32,
    table: Table,
}

/// A draw replay's draws, which the encoder that records an execution's GPU form records in
/// place of its draw's own (`RenderCommandEncoder::draw`): every command's of the range, the
/// first through the draw's recording.
pub(in crate::render) struct ReplayedDraws {
    /// The scratch's device address, and its `VkBuffer` and offset there.
    address: u64,
    buffer: vk::Buffer,
    offset: u64,
    layout: DrawReplayLayout,
}

impl ReplayedDraws {
    /// The address of a merged replay's header, which its vertex function's
    /// `PushHole::MergedDrawAddress` takes (`binding::Tables::merged_header`); `None` for draws
    /// per command.
    pub(in crate::render) fn merged_header(&self) -> Option<u64> {
        (self.layout.merged).map(|m| self.address + m.header_offset)
    }

    /// The holes of `holes` (the draw's push holes) that take a slot table, each filled in
    /// `pushes` (the draw's push writes) with the first command's: the binder's root tables
    /// are none of the commands'. Refused (internal) for a hole of other stages than one
    /// function's, one no push write covers, or any in a merged replay, whose functions read no
    /// slot table at a hole.
    pub(in crate::render) fn first_tables(
        &self,
        holes: &[PushHole],
        pushes: &mut [PushWrite],
    ) -> Result<Vec<TableHole>, Error> {
        let mut tables = Vec::new();
        for hole in holes {
            let PushHole::RootTableAddress { stages, offset } = *hole else {
                continue;
            };
            if self.layout.merged.is_some() {
                return Err(internal("a merged draw's slot table at a hole"));
            }
            let stages: vk::ShaderStageFlags = stages.into();
            let table = match stages {
                vk::ShaderStageFlags::VERTEX => Table::Vertex,
                vk::ShaderStageFlags::FRAGMENT => Table::Fragment,
                _ => return Err(internal("a replayed draw's slot table of several stages")),
            };
            let hole = TableHole {
                stages,
                offset,
                table,
            };
            let write = pushes
                .iter_mut()
                .find(|(of, start, bytes)| {
                    *of == stages
                        && *start <= offset
                        && offset as usize + 8 <= *start as usize + bytes.len()
                })
                .ok_or_else(|| internal("no push write covers a replayed draw's slot table"))?;
            let at = (offset - write.1) as usize;
            write.2[at..at + 8].copy_from_slice(&self.table(&hole, 0).to_le_bytes());
            tables.push(hole);
        }
        Ok(tables)
    }

    /// The device address of `hole`'s slot table of the `command`th command of the range.
    fn table(&self, hole: &TableHole, command: u32) -> u64 {
        self.address
            + match hole.table {
                Table::Vertex => self.layout.vertex_table_offset(command),
                Table::Fragment => self.layout.fragment_table_offset(command),
            }
    }

    /// Record the commands' draws: `first`, the first command's recording, then for each later
    /// command the push writes of its slot tables at `holes` and one `vkCmdDrawIndirect` of one
    /// draw at its arguments, in order; the pipeline, sets and dynamic state `first` bound hold
    /// for them all. A merged replay's: `first`, the merged draw's first, then its second. The
    /// state commands they recorded.
    ///
    /// # Safety
    ///
    /// As [`Recording::record`]'s; the scratch is live until the command buffer completes.
    pub(in crate::render) unsafe fn record(
        &self,
        first: &Recording<'_>,
        bound: &mut Bound,
        holes: &[TableHole],
    ) -> Recorded {
        // SAFETY: the caller's contract.
        let mut recorded = unsafe { first.record(bound) };
        let (raw, command_buffer, layout) = (first.raw, first.command_buffer, first.layout);
        if let Some(merged) = self.layout.merged {
            // SAFETY: the caller's contract; the draw lies in the scratch, which holds
            // `INDIRECT_BUFFER` usage as every upload chunk does.
            unsafe {
                raw.cmd_draw_indirect(
                    command_buffer,
                    self.buffer,
                    self.offset + merged.draw_offset(1),
                    1,
                    DRAW_ARGUMENT_BYTES as u32,
                )
            };
            return recorded;
        }
        let mut writes: Vec<PushWrite> = holes
            .iter()
            .map(|hole| (hole.stages, hole.offset, vec![0; 8]))
            .collect();
        for command in 1..self.layout.commands {
            for (write, hole) in writes.iter_mut().zip(holes) {
                write
                    .2
                    .copy_from_slice(&self.table(hole, command).to_le_bytes());
            }
            // SAFETY: the caller's contract; each write lies in a push range of its stages under
            // `layout`, as the first command's write it lies in did, and the arguments lie in the
            // scratch, which holds `INDIRECT_BUFFER` usage as every upload chunk does.
            unsafe {
                let pushed = match writes.is_empty() {
                    true => None,
                    false => Some(bound.pushes(layout, &writes)).filter(|p| p.any()),
                };
                if let Some(pushed) = pushed {
                    for (at, (stages, offset, bytes)) in writes.iter().enumerate() {
                        if pushed.has(at) {
                            raw.cmd_push_constants(command_buffer, layout, *stages, *offset, bytes);
                        }
                    }
                    recorded.states += 1;
                }
                raw.cmd_draw_indirect(
                    command_buffer,
                    self.buffer,
                    self.offset + self.layout.argument_offset(command, 0),
                    1,
                    DRAW_ARGUMENT_BYTES as u32,
                );
            }
        }
        recorded
    }
}

impl RenderExecution {
    /// The execution's GPU form, settled through an encoder of `core` that records nothing, over
    /// the pass's `storage`, with `residency`, the command buffer's (the module docs); else the
    /// first rule that sends it to the host step. Returns the core. Announces nothing: it draws
    /// nothing.
    pub(super) fn gpu_form(
        &self,
        core: EncoderCore,
        (storage, residency): (PassStorage, &Residency),
    ) -> (EncoderCore, Result<GpuForm, HostStepCause>) {
        let (decided, primitive_type, range) = match self.gpu_commands() {
            Ok(commands) => commands,
            Err(cause) => return (core, Err(cause)),
        };
        let Some(pipeline) = &self.pipeline else {
            return (
                core,
                Err(HostStepCause::Rule("the encoder has no pipeline state")),
            );
        };
        let RenderStateKind::Graphics(state) = pipeline.kind() else {
            let rule = "the inherited state is a mesh or a tile state";
            return (core, Err(HostStepCause::Rule(rule)));
        };
        // Merged where the commands draw a list type or points, else one draw per command.
        let merges = draw_replay_merges(primitive_type);
        let commands = (decided, primitive_type, range);
        let inputs = (storage, residency);
        let mut core = core;
        if merges {
            let (back, form) = self.form_of(
                core,
                inputs.clone(),
                (pipeline, state),
                commands.clone(),
                true,
            );
            core = back;
            if let Ok(form) = form {
                return (core, Ok(form));
            }
        }
        self.form_of(core, inputs, (pipeline, state), commands, false)
    }

    /// The execution's GPU form for `commands`, `merged` into one draw or one draw per command,
    /// its `pipeline`'s graphics `state` settled through an encoder of `core` that records nothing
    /// in that form's buffer forms (the merged variant or the replay variant), over the pass's
    /// `storage` with `residency`; else the rule that keeps it from that form. Returns the core.
    fn form_of(
        &self,
        core: EncoderCore,
        (storage, residency): (PassStorage, &Residency),
        (pipeline, state): (&RenderPipelineState, &GraphicsState),
        (decided, primitive_type, range): (CommandWrites, PrimitiveType, Range<u32>),
        merged: bool,
    ) -> (EncoderCore, Result<GpuForm, HostStepCause>) {
        let snapshot = &self.snapshot;
        let forms = match merged {
            true => BufferForms::Merged,
            false => BufferForms::Replay,
        };
        let visibility = Visibility::settling(snapshot.layout.visibility.clone(), None);
        let mut encoder = self.replay_encoder(
            core,
            (vk::CommandBuffer::null(), visibility),
            (storage, snapshot.layout.clone(), residency.clone()),
            forms,
        );
        let slots = encoder
            .shape_inputs()
            .check_pipeline(CALL, pipeline, &encoder.layout)
            .and_then(|()| {
                encoder.pipeline = Some(pipeline.clone());
                encoder.replay_draw_of(state, primitive_type)
            });
        let RenderLowering { core, .. } = encoder;
        let form = match slots {
            Ok(Ok(slots)) => {
                let commands = (decided, primitive_type, range);
                self.preprocess(core.device(), residency, commands, slots, merged)
            }
            Ok(Err(cause)) => Err(cause),
            Err(error) => Err(HostStepCause::Refused(error.to_string())),
        };
        (EncoderCore::unboxed(core), form)
    }

    /// What the buffer's records may hold, the primitive type and the range of an execution that
    /// may take the GPU form, from the call and that alone (the module docs); else the first rule
    /// that sends it to the host step.
    fn gpu_commands(&self) -> Result<(CommandWrites, PrimitiveType, Range<u32>), HostStepCause> {
        let rule = |holds: bool, rule: &'static str| match holds {
            true => Ok(()),
            false => Err(HostStepCause::Rule(rule)),
        };
        let ExecutionRange::Fixed(range) = &self.range else {
            return Err(HostStepCause::Rule(
                "the execution reads its range from a buffer",
            ));
        };
        let descriptor = self.icb.descriptor();
        let written = self.icb.written().writes();
        rule(!range.is_empty(), "the execution's range is empty")?;
        rule(self.lent.is_none(), "the encoder counts visibility results")?;
        let inherits = (
            descriptor.inherit_pipeline_state,
            descriptor.inherit_buffers,
        );
        let primitive_type =
            draw_replay_primitive_type(descriptor.command_types, inherits, written)
                .map_err(HostStepCause::Replay)?;
        self.check_has_nothing_to_see(written)?;
        rule(!self.reaches_tile_memory(), "the pass reaches tile memory")?;
        rule(
            !self.snapshot.pass_start.fragment_wrote(),
            "the pass's fragment functions wrote memory before the execution",
        )?;
        Ok((written, primitive_type, range.clone()))
    }

    /// Whether the caller's check, if any, has nothing to see among commands whose records may
    /// hold `written`, under the encoder's state, which the buffer inherits: neither draws of
    /// points (a primitive type the writes do not state counts as one) nor a state that may encode
    /// indirect commands where it looks for them (`RenderExecution::check_sees`); else which of
    /// those it may see.
    fn check_has_nothing_to_see(&self, written: CommandWrites) -> Result<(), HostStepCause> {
        match self.check_sees(written) {
            (false, false) => Ok(()),
            (points, encodes) => Err(HostStepCause::Check { points, encodes }),
        }
    }

    /// The replay's preprocess of commands `range` drawing `primitive_type`, each stage's slot
    /// table holding `slots` (the vertex function's, then the fragment function's), over
    /// `residency`'s resident resources and the device's live buffers, decided on the records'
    /// `decided` writes, `merged` into one draw or not; else why it cannot run (the binder's
    /// refusal, a device without device addresses among others).
    fn preprocess(
        &self,
        device: &Device,
        residency: &Residency,
        (decided, primitive_type, range): (CommandWrites, PrimitiveType, Range<u32>),
        (vertex_slots, fragment_slots): (u32, u32),
        merged: bool,
    ) -> Result<GpuForm, HostStepCause> {
        let capacity = u32::try_from(self.icb.size())
            .map_err(|_| HostStepCause::Rule("the buffer holds 2^32 commands or more"))?;
        let live = (device.shared().live_buffers.table(device))
            .map_err(|error| HostStepCause::Refused(error.to_string()))?;
        let mut state = BindState::default();
        state.used_resources = residency.used_resources();
        state.used_in_token_order = true;
        state.resident_ranges = live.ranges().collect();
        let count = range.end - range.start;
        let mut replay =
            DrawReplay::new(capacity, range.start, count, vec![primitive_type.as_raw()]);
        replay.inheritance = self.icb.inheritance();
        replay.vertex_slots = vertex_slots;
        replay.fragment_slots = fragment_slots;
        replay.merged = merged;
        let pre = draw_replay_preprocess(device.compile_device(), &state, &replay)
            .map_err(|error| HostStepCause::Refused(error.to_string()))?;
        Ok(GpuForm {
            decided,
            primitive_type,
            pre,
        })
    }

    /// Record the execution's GPU form `form` (the module docs) through `core`, a segment of
    /// `command_buffer` with `residency` (the command buffer's): the preprocess and the barrier
    /// after it, then the commands' draws into a secondary begun against the execution's
    /// instance, over the pass's `storage`, in the call's layout as `rendered` backs it. Returns
    /// the encoder that drew them, its runs to end ([`finish`](Self::finish)).
    pub(super) fn record_gpu(
        &self,
        mut core: EncoderCore,
        (command_buffer, residency): (&CommandBuffer, &Residency),
        form: &GpuForm,
        (storage, rendered): (PassStorage, &mut Rendered),
    ) -> Result<RenderLowering, Error> {
        let snapshot = &self.snapshot;
        let device = command_buffer.device().clone();
        let pipeline = (self.pipeline.as_ref())
            .ok_or_else(|| internal("an execution's GPU form without its pipeline state"))?;
        // The GPU reads the records: the handlers learn the state their commands run.
        command_buffer.report_executed(&self.icb, RecordsRead::Gpu, residency, || {
            (Vec::new(), pipeline.can_encode_indirect_commands())
        });
        let pre = &form.pre;
        let kernel = device.owned_kernel(OwnedModule::DrawReplayPreprocess)?;
        let no_address = || internal("a draw replay on a device without device addresses");
        let records = self.icb.storage();
        let records_address = records.device_address().ok_or_else(no_address)?;
        let (scratch, status) = {
            let mut cb = command_buffer.state();
            let sink = cb
                .uploads
                .upload(&device, &[], u64::from(NeutralResource::SINK_BYTES))?
                .device_address()
                .ok_or_else(no_address)?;
            let head = scratch_head(pre, sink)?;
            let scratch = cb.uploads.upload(&device, &head, pre.layout.bytes)?;
            let status = cb.status_address(&device)?.ok_or_else(no_address)?;
            (scratch, status)
        };
        let address = scratch.device_address().ok_or_else(no_address)?;
        let mut push = pre.push_constants.clone();
        for hole in &pre.push_holes {
            let (offset, value) = match *hole {
                PushHole::RootTableAddress { offset, .. } => {
                    (offset, address + pre.layout.resident_offset)
                }
                PushHole::ExecutionStatusAddress { offset, .. } => (offset, status),
                PushHole::CommandRecordsAddress { offset, .. } => (offset, records_address),
                _ => {
                    return Err(internal(
                        "a draw replay preprocess hole the runtime does not fill",
                    ))
                }
            };
            push.get_mut(offset as usize..offset as usize + 8)
                .ok_or_else(|| internal("a draw replay preprocess hole past its push bytes"))?
                .copy_from_slice(&value.to_le_bytes());
        }
        // The preprocess reads the records and writes the scratch past what the host uploaded.
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let (chunk, at) = scratch.buffer();
        let uses = [
            reads_by(
                records,
                0..records.length(),
                Access::new(compute, vk::AccessFlags2::SHADER_STORAGE_READ),
            ),
            writes_by(
                chunk,
                at + pre.layout.uploaded..at + pre.layout.bytes,
                Access::new(compute, vk::AccessFlags2::SHADER_STORAGE_WRITE),
            ),
        ];
        let group_count = pre.group_count;
        core.command(compute, &uses, |raw, command_buffer| {
            // SAFETY: the segment's command buffer is recording outside a render pass; the kernel
            // is the device's, and its push bytes name the scratch, the status words and the
            // records, which live until the command buffer completes.
            unsafe {
                raw.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline,
                );
                raw.cmd_push_constants(
                    command_buffer,
                    kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                let [x, y, z] = group_count;
                raw.cmd_dispatch(command_buffer, x, y, z);
            }
        });
        // Its writes before the draws' indirect reads and their functions' reads of the slot
        // tables (and before a merged replay's passes), the pass begun after it.
        core.barrier(Barrier::everything());
        if let Some(merged) = pre.layout.merged {
            let push = DrawMergeModule::push_constants(address + merged.header_offset, status);
            let uses = [writes_by(
                chunk,
                at + merged.header_offset..at + pre.layout.bytes,
                Access::new(
                    compute,
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            )];
            let [x, y, z] = merged.group_count();
            for phase in [MergePhase::Blocks, MergePhase::Starts] {
                let kernel = device.owned_kernel(OwnedModule::DrawMerge(phase))?;
                core.command(compute, &uses, |raw, command_buffer| {
                    // SAFETY: as the preprocess's: its push bytes name the scratch's header and
                    // the status words, which live until the command buffer completes.
                    unsafe {
                        raw.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.pipeline,
                        );
                        raw.cmd_push_constants(
                            command_buffer,
                            kernel.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            &push,
                        );
                        raw.cmd_dispatch(command_buffer, x, y, z);
                    }
                });
                // Each pass's writes before what reads them next: the second pass, the draws.
                core.barrier(Barrier::everything());
            }
        }
        // Its resident set as the live buffers and what its encoders made resident lie now: the
        // command buffer's first submission points its table at the set as it lies then.
        command_buffer.state().replay_table(ReplayTable {
            upload: scratch.clone(),
            at: pre.layout.resident_offset,
            address: address + pre.layout.resident_offset,
        });
        let draws = command_buffer.begin_draws(&snapshot.compatible)?;
        let raw = device.shared().vk();
        let visibility =
            Visibility::borrowing(snapshot.layout.visibility.clone(), None, raw, draws);
        let layout = rendered.backing(&snapshot.layout)?;
        let (forms, first_at) = match pre.layout.merged {
            Some(merged) => (BufferForms::Merged, merged.draw_offset(0)),
            None => (BufferForms::Replay, pre.layout.argument_offset(0, 0)),
        };
        let mut encoder = self.replay_encoder(
            core,
            (draws, visibility),
            (storage, layout, residency.clone()),
            forms,
        );
        // The first draw reads its arguments where the preprocess (or the last merge pass)
        // wrote them.
        let first = DrawSource::new(chunk, at + first_at, DRAW_ARGUMENT_BYTES, 4, false, false)?;
        let draw = DrawCall::Indirect {
            indirect: IndirectRef::new(0, first.at),
            indexed: None,
        };
        encoder.replayed = Some(ReplayedDraws {
            address,
            buffer: scratch.raw(),
            offset: scratch.offset(),
            layout: pre.layout,
        });
        (encoder.shape_inputs()).check_pipeline(CALL, pipeline, &encoder.layout)?;
        encoder.pipeline = Some(pipeline.clone());
        let buffers = DrawBuffers {
            indirect: Some(first),
            ..DrawBuffers::default()
        };
        let assembly = Assembly::Primitives(form.primitive_type);
        encoder.draw(CALL, assembly, draw, buffers, false)?;
        encoder.replayed = None;
        let counts = &device.shared().counts;
        counts.gpu_form_commands(u64::from(pre.layout.commands));
        if pre.layout.merged.is_some() {
            counts.merged_draw_commands(u64::from(pre.layout.commands));
        }
        Ok(encoder)
    }
}

/// The bytes of a draw replay's scratch the host uploads: the parameters, the resident root
/// table with its sink entry's `host` field at `sink`, the resident roots' extents, the
/// encoder's buffers and a merged replay's header, each at its layout offset; the slot tables
/// and arguments past them are the preprocess's to write.
fn scratch_head(pre: &DrawReplayPreprocess, sink: u64) -> Result<Vec<u8>, Error> {
    let layout = pre.layout;
    let len =
        usize::try_from(layout.uploaded).map_err(|_| internal("a replay scratch too large"))?;
    let mut bytes = vec![0u8; len];
    let mut resident = pre.resident.bytes.clone();
    let at = metal2vulkan::bind::RootTable::SINK_HOST_OFFSET as usize;
    resident
        .get_mut(at..at + 8)
        .ok_or_else(|| internal("a resident root table without a sink entry"))?
        .copy_from_slice(&sink.to_le_bytes());
    let header = layout.merged.map(|m| (m.header_offset, &pre.header));
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
            .ok_or_else(|| internal("a replay scratch section past what the host uploads"))?
            .copy_from_slice(part);
    }
    Ok(bytes)
}

impl RenderLowering {
    /// The buffer slots the vertex and fragment functions of a command's draw of `state` as
    /// `primitive_type` read, through this encoder (an execution's at its call, in the replay
    /// form or the merged one, its `forms`), where that draw records as the GPU form's draws do
    /// (the module docs); else the rule it breaks, as the host-step notice words it. Announces
    /// nothing: it draws nothing.
    fn replay_draw_of(
        &self,
        state: &crate::pipeline::GraphicsState,
        primitive_type: PrimitiveType,
    ) -> Result<Result<(u32, u32), HostStepCause>, Error> {
        let graphics = &state.base().graphics;
        let merged = self.forms == BufferForms::Merged;
        let rules = [
            (
                crate::tessellation::patches(graphics).is_some(),
                "the inherited state tessellates",
            ),
            (
                state.descriptor().vertex().is_none(),
                "the inherited state has no vertex function",
            ),
            (
                state.reads_tile_memory(),
                "the inherited state reads tile memory",
            ),
            (
                self.rate_map.is_some(),
                "the pass has a rasterization rate map",
            ),
        ];
        if let Some((_, rule)) = rules.iter().find(|(breaks, _)| *breaks) {
            return Ok(Err(HostStepCause::Rule(rule)));
        }
        if let Err(rule) = draw_replay_state(graphics, merged) {
            return Ok(Err(HostStepCause::Replay(rule)));
        }
        if !self.storage_attachments(graphics)?.is_empty() {
            let rule = "the inherited state writes an attachment as storage";
            return Ok(Err(HostStepCause::Rule(rule)));
        }
        let settings =
            self.draw_settings(CALL, state, Assembly::Primitives(primitive_type), false)?;
        // Lines at a width other than 1 are the host replay's to draw, which announces their width
        // (`render/line_width.rs`).
        if settings.line != LineRoute::Plain {
            return Ok(Err(HostStepCause::Rule(
                "the encoder's line width is not 1",
            )));
        }
        if *self.current.shape.locations != *settings.routing.locations {
            let rule = "the inherited state's color outputs go to other locations than the pass's";
            return Ok(Err(HostStepCause::Rule(rule)));
        }
        if !self.reformats(state, &settings.routing)?.is_empty() {
            let rule = "the inherited state writes a color in another format than its attachment's";
            return Ok(Err(HostStepCause::Rule(rule)));
        }
        let mut argument_samplers = ArgumentSamplers::taken(self.core.device());
        let function_tables = state.function_tables();
        let live = Arc::clone(&self.live);
        let pass_start = PassStartTables::default();
        let sampler_maps = self.sampler_maps(state, &pass_start)?;
        let mut live = lock(&live);
        let (Settled { variant, plan, .. }, _) = self.settle_draw(
            &mut live,
            (
                state,
                &settings,
                None,
                PrivateForm::default(),
                ProvokingForms::default(),
            ),
            (&pass_start, &sampler_maps),
            (&mut argument_samplers, &mut None),
            (&function_tables, &mut None),
        )?;
        let draw = DrawCall::Indirect {
            indirect: IndirectRef::new(0, 0),
            indexed: None,
        };
        let commands = plan
            .instantiate_draw(&live.state, &draw)
            .map_err(Error::Bind)?;
        if let Err(rule) = draw_replay_draw(&commands, (&plan, variant.key()), merged) {
            return Ok(Err(HostStepCause::Replay(rule)));
        }
        Ok(Ok(draw_replay_slots(&variant.pipeline().interface)))
    }
}
