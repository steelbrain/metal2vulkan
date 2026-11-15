//! The GPU route of a host step that fans a rate-mapped pass's triangles out (`render/fan_out.rs`):
//! an execution whose records the GPU may write, and which takes its merged GPU form but for the
//! pass's rasterization-rate map (`gpu.rs`; the decision passes over that one rule, `cause.rs`),
//! has its records decoded on the GPU. The host never reads them.
//!
//! - **What runs first.** The step records the form's prework for each such execution of the pass
//!   (`RenderExecution::prework`: the head carried into a scratch of its own, the preprocess, and
//!   the two merge passes). Then a copy of each one's arguments (each command's
//!   `VkDrawIndirectCommand`, zero for a command that draws nothing) and index entries (the device
//!   address of each command's first index and its index size, `bind::DRAW_INDEX_ENTRY_BYTES`)
//!   into the command buffer's upload arena, behind a barrier to the host. The step submits that
//!   with a fence and waits for it, as for its probe (`DeferredPass::submit_probe`). The host
//!   reads those words alone, no record (`WorkCounts::indirect_records_read`).
//! - **Its commands' draws.** Each command that draws is drawn as the host step draws a command
//!   it read (`RenderExecution::replay`): a draw of primitives, or an indexed draw of the index
//!   buffer the host resolves the index entry's address to (the resident set's range that holds
//!   it gives its Metal address, which resolves as a record's does), with its index count,
//!   instances, base vertex and base instance. It runs in the state's replay variant
//!   (`BufferForms::Replay`), whose vertex and fragment functions read the command's buffers in
//!   the slot tables its preprocess wrote, at the addresses its draws push
//!   (`PushHole::RootTableAddress`, [`CommandTables`]). Its vertex reuse,
//!   `vertex_id`, `instance_id` and primitive order are an indexed draw's. The probe and the
//!   routes take those draws as they take the draws of the commands the host read
//!   (`render/fan_out.rs`): probed, then fanned out, boxed, in parts or region by region, and an
//!   indexed one cut into ranges of its indices as any is (`rate_mapped_pass.rs`, **Pieces**).
//!   `commit`'s settle of the pass finds the same executions (`DeferredPass::route_forms`) and
//!   begins the replay variants' `VkPipeline`s of their draws, the probe's and the fan-out's, on
//!   the device's pool (`DeferredPass::settle`), so the step makes none of them itself
//!   (`WorkCounts::host_step_pipelines_made`).
//! - **Where the host reads the records instead**, as without the route: an index size the
//!   preprocess does not write, an index entry's address in no range of the resident set, and a
//!   step whose prework did not complete.
//! - **What the pass holds.** The scratch, the head and the live buffers' table its resident set's
//!   ranges lie in, until the command buffer completes. The prework's resident set is the command
//!   buffer's as the step resolves it (`residency.rs`), as the host's read resolves the records'
//!   addresses, so no first submission points it again (`command_buffer/first_submission.rs`).
//! - Counted: `WorkCounts::rate_map_gpu_routes` and `rate_map_gpu_route_wait_nanos`.

use std::sync::Arc;
use std::time::Instant;

use super::super::{Assembly, IndexedDraw};
use super::cause::Passed;
use super::gpu::{CommandTables, GpuForm, Prework, RATE_MAP};
use super::{DeferredPass, Draw, HostStepCause, StepCommands, StepDraws};
use crate::blit::{reads, writes};
use crate::command_buffer::CommandBuffer;
use crate::encoder::{EncoderCore, RecordedSegment};
use crate::error::{internal, Error};
use crate::render_values::DrawPrimitiveType;
use crate::residency::Residency;
use crate::tracker::Barrier;
use crate::upload::Upload;
use ash::vk;
use metal2vulkan::bind::{
    DrawCall, ReplayFaults, ResidentRange, DRAW_ARGUMENT_BYTES, DRAW_INDEX_ENTRY_BYTES,
};
use metal2vulkan::metal::{IndexType, PrimitiveType};

/// An execution's GPU route at its host step, its prework run (the module docs): where its
/// commands' slot tables lie, the type its commands draw, and each command that draws, its draw,
/// in order.
pub(super) struct GpuRoute {
    tables: CommandTables,
    primitive_type: PrimitiveType,
    draws: Vec<RoutedDraw>,
}

impl GpuRoute {
    /// Its commands as a host step's probe and routes take them: each that draws, its draw; from
    /// no command the host read, with no fault the host found.
    pub(super) fn step_commands(self) -> StepCommands {
        let commands = StepDraws::Routed(self.draws, self.primitive_type, self.tables);
        (commands, 0, ReplayFaults::default())
    }
}

/// A command's draw as the GPU route's preprocess decoded it (the module docs): its place in the
/// preprocess's range, whose slot tables its draws push; its vertex or index count, instances,
/// vertex start or base vertex, and base instance; and, for an indexed draw, its index type and
/// the Metal address of its first index.
#[derive(Clone, Copy)]
pub(in crate::render) struct RoutedDraw {
    pub(super) command: u32,
    count: u32,
    instances: u32,
    first: u32,
    base_instance: u32,
    indices: Option<(IndexType, u64)>,
}

impl RoutedDraw {
    /// Its draw of `primitive_type`, its index buffer at its Metal address, as a record's
    /// (`Asked::of`).
    pub(super) fn draw(&self, primitive_type: PrimitiveType) -> Draw<u64> {
        match self.indices {
            None => {
                let draw = DrawCall::Primitives {
                    vertex_start: self.first,
                    vertex_count: self.count,
                    instance_count: self.instances,
                    base_instance: self.base_instance,
                };
                Draw::Primitives(Assembly::Primitives(primitive_type), draw)
            }
            Some((index_type, address)) => {
                let draw = IndexedDraw {
                    primitive_type: DrawPrimitiveType::Public(primitive_type),
                    index_count: self.count,
                    index_type,
                    instance_count: self.instances,
                    // The preprocess's first vertex of an indexed draw: Metal's base vertex.
                    base_vertex: i32::from_le_bytes(self.first.to_le_bytes()),
                    base_instance: self.base_instance,
                };
                Draw::Indexed(draw, address)
            }
        }
    }

    /// Whether it draws indices, and its count and instances (the encoder trail's).
    pub(super) fn traced(&self) -> (bool, u32, u32) {
        (self.indices.is_some(), self.count, self.instances)
    }
}

/// An execution's prework as the step recorded it: its form, where its scratch lies, and the upload
/// its words for the host are copied into.
type Recorded = (GpuForm, Prework, Upload);

impl DeferredPass {
    /// The merged GPU forms the GPU route takes for the pass's executions, by index (the module
    /// docs): the decision again, passing over the rate map's rule alone, for each execution of a
    /// rate-mapped pass without shadowed textures whose records the GPU may write; `None` for the
    /// others. Through encoders of `core`, which record nothing, with `residency`, the command
    /// buffer's: at the host step, which takes them, and at `commit`, whose settle begins the
    /// pipelines of the step's draws of them (`DeferredPass::settle`). Returns the core.
    pub(super) fn route_forms(
        &self,
        core: EncoderCore,
        residency: &Residency,
    ) -> (EncoderCore, Vec<Option<GpuForm>>) {
        let mut passed = Passed::default();
        passed.pass(&HostStepCause::Rule(RATE_MAP));
        let mut core = core;
        let mut forms = Vec::with_capacity(self.executions.len());
        for execution in &self.executions {
            let takes = execution.snapshot.rate_map.is_some()
                && execution.shadows.is_empty()
                && execution.icb.gpu_written();
            if !takes {
                forms.push(None);
                continue;
            }
            let storage = self.storage.clone();
            let (back, form) = execution.gpu_form(core, (storage, residency), (&passed, true));
            core = back;
            forms.push(form.ok().filter(GpuForm::merged));
        }
        (core, forms)
    }

    /// The GPU routes of the pass's executions at its host step, by index (the module docs):
    /// `None` for an execution that takes none, whose records the step reads; and the recording
    /// of their prework, which ran, for the pass's segment to join. Nothing where none takes one.
    pub(super) fn gpu_routes(
        &self,
        command_buffer: &CommandBuffer,
        residency: &Residency,
    ) -> Result<(Vec<Option<GpuRoute>>, Option<RecordedSegment>), Error> {
        let none = || -> Vec<Option<GpuRoute>> { self.executions.iter().map(|_| None).collect() };
        let core = EncoderCore::segment(command_buffer, false)?;
        let (mut core, forms) = self.route_forms(core, residency);
        if forms.iter().all(Option::is_none) {
            return Ok((none(), None));
        }
        let mut recorded: Vec<Option<Recorded>> = Vec::with_capacity(forms.len());
        for (execution, form) in self.executions.iter().zip(forms) {
            let Some(form) = form else {
                recorded.push(None);
                continue;
            };
            let prework = execution.prework(&mut core, command_buffer, (&form, true))?;
            let words = read_back(&mut core, command_buffer, &form, &prework)?;
            core.retain(Arc::clone(form.live()));
            recorded.push(Some((form, prework, words)));
        }
        core.barrier(Barrier::to_host());
        let mut segment = core.end_segment()?;
        let what = "a rate-mapped pass's GPU route";
        // SAFETY: the segment's command buffers, of the command buffer's pool, ended and recorded
        // for this submission alone; what they use the segment holds, which the pass's joins, and
        // the host waits for the submission before the step ends.
        let fenced = unsafe { command_buffer.submit_fenced(&segment.raws, what) }?;
        segment.raws.clear();
        segment.piece_ends.clear();
        segment.executions.clear();
        let started = Instant::now();
        let completed = match &fenced {
            Some(fenced) => command_buffer.wait_fenced(std::slice::from_ref(fenced), what)?,
            None => false,
        };
        let counts = &command_buffer.device().shared().counts;
        counts.rate_map_gpu_route_waited(started.elapsed());
        if !completed {
            return Ok((none(), Some(segment)));
        }
        let routes: Vec<Option<GpuRoute>> = (recorded.into_iter())
            .map(|recorded| {
                let (form, prework, words) = recorded?;
                route(&form, (&prework, &words), residency)
            })
            .collect();
        counts.rate_map_gpu_routes(routes.iter().flatten().count() as u64);
        Ok((routes, Some(segment)))
    }
}

/// Record into `core`, a segment of `command_buffer`, a copy of `form`'s arguments and index
/// entries from `prework`'s scratch into an upload of the command buffer's arena, after the
/// barrier that follows the preprocess: the upload, the arguments first.
fn read_back(
    core: &mut EncoderCore,
    command_buffer: &CommandBuffer,
    form: &GpuForm,
    prework: &Prework,
) -> Result<Upload, Error> {
    let layout = form.layout();
    let merged = (layout.merged).ok_or_else(|| internal("a GPU route that does not merge"))?;
    let commands = u64::from(layout.commands);
    let arguments = commands * u64::from(layout.types) * DRAW_ARGUMENT_BYTES;
    let entries = commands * DRAW_INDEX_ENTRY_BYTES;
    let device = command_buffer.device();
    let words = (command_buffer.state().uploads).upload(device, &[], arguments + entries)?;
    let (scratch, lead) = (&prework.scratch, prework.lead);
    let (chunk, at) = words.buffer();
    let (from, src) = scratch.at(lead);
    let (to, dst) = chunk.at(at);
    let regions = [
        vk::BufferCopy {
            src_offset: src + layout.arguments_offset,
            dst_offset: dst,
            size: arguments,
        },
        vk::BufferCopy {
            src_offset: src + merged.indices_offset,
            dst_offset: dst + arguments,
            size: entries,
        },
    ];
    let uses = [
        reads(
            scratch,
            lead + layout.arguments_offset..lead + layout.arguments_offset + arguments,
        ),
        reads(
            scratch,
            lead + merged.indices_offset..lead + merged.indices_offset + entries,
        ),
        writes(chunk, at..at + arguments + entries),
    ];
    core.command(
        vk::PipelineStageFlags2::TRANSFER,
        &uses,
        |raw, command_buffer| {
            // SAFETY: the segment's command buffer is recording outside a render pass; the
            // scratch lives until it completes (the segment holds it) and so does the upload (the
            // arena); both regions lie in both: the arguments and the entries in the scratch's
            // layout, and the upload holds them both.
            unsafe { raw.cmd_copy_buffer(command_buffer, from, to, &regions) };
        },
    );
    Ok(words)
}

/// The route of `form`'s commands from the words its prework wrote into `words`
/// ([`read_back`]), the indices' addresses resolved in `residency`, the command buffer's, and the
/// form's live buffers: `None` where the host reads the records instead (the module docs).
fn route(
    form: &GpuForm,
    (prework, words): (&Prework, &Upload),
    residency: &Residency,
) -> Option<GpuRoute> {
    let layout = form.layout();
    if layout.types != 1 {
        return None;
    }
    let commands = usize::try_from(layout.commands).ok()?;
    let arguments = commands * DRAW_ARGUMENT_BYTES as usize;
    let len = arguments + commands * DRAW_INDEX_ENTRY_BYTES as usize;
    let mut bytes = vec![0u8; len];
    // SAFETY: the submission that wrote the words completed, its writes made visible to the host
    // by its last barrier; the bytes are writable and apart from the upload.
    unsafe { words.copy_out(0, bytes.as_mut_ptr(), len as u64) };
    let (arguments, entries) = bytes.split_at(arguments);
    // The ranges the preprocess resolved the indices' addresses in, by device address
    // (`RenderExecution::preprocess`): the command buffer's resident buffers and heaps, then the
    // live buffers.
    let mut ranges: Vec<ResidentRange> = residency.resident_ranges();
    ranges.extend(form.live().ranges());
    ranges.sort_by_key(|range| range.host);
    // The Metal address of device address `address`: of the nearest range below that holds it,
    // an address at a range's end included, as a record's resolves.
    let guest = |address: u64| {
        let below = ranges.partition_point(|range| range.host <= address);
        (ranges[..below].iter().rev())
            .find(|range| address - range.host <= range.length)
            .map(|range| range.guest + (address - range.host))
    };
    let word = |bytes: &[u8], at: usize| {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    let mut draws = Vec::new();
    let arguments = arguments.as_chunks::<{ DRAW_ARGUMENT_BYTES as usize }>().0;
    let entries = entries.as_chunks::<{ DRAW_INDEX_ENTRY_BYTES as usize }>().0;
    for (command, (argument, entry)) in arguments.iter().zip(entries).enumerate() {
        // `VkDrawIndirectCommand`: `vertexCount`, `instanceCount`, `firstVertex`,
        // `firstInstance`; an indexed draw's index count and base vertex.
        let [count, instances, first, base_instance] = [0, 4, 8, 12].map(|at| word(argument, at));
        if count == 0 || instances == 0 {
            continue;
        }
        // The entry: its first index's device address, the indices from there to the buffer's
        // end, and the index size.
        let mut address = [0u8; 8];
        address.copy_from_slice(&entry[..8]);
        let address = u64::from_le_bytes(address);
        let indices = match word(entry, 12) {
            0 => None,
            2 => Some((IndexType::UInt16, guest(address)?)),
            4 => Some((IndexType::UInt32, guest(address)?)),
            _ => return None,
        };
        draws.push(RoutedDraw {
            command: u32::try_from(command).ok()?,
            count,
            instances,
            first,
            base_instance,
            indices,
        });
    }
    Some(GpuRoute {
        tables: CommandTables::new(prework.address, *layout),
        primitive_type: form.primitive_type(),
        draws,
    })
}
