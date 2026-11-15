//! The GPU route of a host step that fans a rate-mapped pass's triangles out (`render/fan_out.rs`):
//! an execution whose records the GPU may write, and which takes its merged GPU form but for the
//! pass's rasterization-rate map (`gpu.rs`; the decision passes over that one rule, `cause.rs`),
//! has its records decoded on the GPU. The host never reads them.
//!
//! - **What runs first.** The step records the form's prework for each such execution of the pass
//!   (`RenderExecution::prework`: the head carried into a scratch of its own, the preprocess, and
//!   the two merge passes). Then a copy of each one's prefix (each command's first corner in the
//!   merged draw, a `u64`) and of its merged draw's two `VkDrawIndirectCommand`s into the command
//!   buffer's upload arena, behind a barrier to the host. The step submits that with a fence and
//!   waits for it, as for its probe (`DeferredPass::submit_probe`). The host reads those words
//!   alone, no record (`WorkCounts::indirect_records_read`).
//! - **Its commands' draws.** Each command that draws is one direct draw of its corners in the
//!   state's merged variant (`BufferForms::Merged`), with the merged replay's header pushed
//!   (`PushHole::MergedDrawAddress`): corners `[first, next)`, the prefix's, the last command's
//!   next the total. Its vertex function finds the command, its vertex and its instance as the
//!   merged draw's does (`bind::replay`'s merged docs): its `VertexIndex` is the corner, as in a
//!   merged draw's first draw's first instance (a total below `MERGED_CHUNK`). The probe and the
//!   routes take those draws as they take the draws of the commands the host read
//!   (`render/fan_out.rs`): probed, then fanned out, boxed, in parts or region by region. Vulkan
//!   rasterizes one draw's primitives in instance order, then in order within an instance
//!   ("Primitive Order"), and the draws in order, so a pixel takes the commands' fragments in the
//!   commands' order and each command's in its own, as the merged draw's.
//! - **Where the host reads the records instead**, as without the route: a total of
//!   `MERGED_CHUNK` corners or more, a prefix that does not rise to the total, and a step whose
//!   prework did not complete.
//! - **What the pass holds.** The scratch, the head and the live buffers' table its resident set's
//!   ranges lie in, until the command buffer completes. The prework's resident set is the command
//!   buffer's as the step resolves it (`residency.rs`), as the host's read resolves the records'
//!   addresses, so no first submission points it again (`command_buffer/first_submission.rs`).
//! - Counted: `WorkCounts::rate_map_gpu_routes` and `rate_map_gpu_route_wait_nanos`.

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use super::cause::Passed;
use super::gpu::{GpuForm, Prework, RATE_MAP};
use super::{DeferredPass, HostStepCause, StepCommands, StepDraws};
use crate::blit::{reads, writes};
use crate::command_buffer::CommandBuffer;
use crate::encoder::{EncoderCore, RecordedSegment};
use crate::error::{internal, Error};
use crate::residency::Residency;
use crate::tracker::Barrier;
use crate::upload::Upload;
use ash::vk;
use metal2vulkan::bind::{ReplayFaults, DRAW_ARGUMENT_BYTES};
use metal2vulkan::metal::PrimitiveType;

/// An execution's GPU route at its host step, its prework run (the module docs): the merged
/// replay's header's device address, the type its commands draw, and each command that draws, its
/// corners in the merged draw, in order.
pub(super) struct GpuRoute {
    header: u64,
    primitive_type: PrimitiveType,
    corners: Vec<Range<u32>>,
}

impl GpuRoute {
    /// Its commands as a host step's probe and routes take them: each that draws, its corners;
    /// from no command the host read, with no fault the host found, and its header.
    pub(super) fn step_commands(self) -> StepCommands {
        let commands = StepDraws::Corners(self.corners, self.primitive_type);
        (commands, 0, ReplayFaults::default(), Some(self.header))
    }
}

/// An execution's prework as the step recorded it: its form, where its scratch lies, and the upload
/// its words for the host are copied into.
type Recorded = (GpuForm, Prework, Upload);

impl DeferredPass {
    /// The GPU routes of the pass's executions at its host step, by index (the module docs):
    /// `None` for an execution that takes none, whose records the step reads; and the recording
    /// of their prework, which ran, for the pass's segment to join. Nothing where none takes one.
    pub(super) fn gpu_routes(
        &self,
        command_buffer: &CommandBuffer,
        residency: &Residency,
    ) -> Result<(Vec<Option<GpuRoute>>, Option<RecordedSegment>), Error> {
        let none = || -> Vec<Option<GpuRoute>> { self.executions.iter().map(|_| None).collect() };
        // The decision again, passing over the rate map's rule alone; merged forms alone.
        let mut passed = Passed::default();
        passed.pass(&HostStepCause::Rule(RATE_MAP));
        let mut core = EncoderCore::segment(command_buffer, false)?;
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
            let (back, form) = execution.gpu_form(core, (storage, residency), &passed);
            core = back;
            forms.push(form.ok().filter(GpuForm::merged));
        }
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
                route(&form, &prework, &words)
            })
            .collect();
        counts.rate_map_gpu_routes(routes.iter().flatten().count() as u64);
        Ok((routes, Some(segment)))
    }
}

/// Record into `core`, a segment of `command_buffer`, a copy of `form`'s prefix and its merged
/// draw's two `VkDrawIndirectCommand`s from `prework`'s scratch into an upload of the command
/// buffer's arena, after the barrier that follows the last merge pass: the upload, the prefix
/// first.
fn read_back(
    core: &mut EncoderCore,
    command_buffer: &CommandBuffer,
    form: &GpuForm,
    prework: &Prework,
) -> Result<Upload, Error> {
    let layout = form.layout();
    let merged = (layout.merged).ok_or_else(|| internal("a GPU route that does not merge"))?;
    let prefix = 8 * u64::from(layout.commands);
    let draws = 2 * DRAW_ARGUMENT_BYTES;
    let device = command_buffer.device();
    let words = (command_buffer.state().uploads).upload(device, &[], prefix + draws)?;
    let (scratch, lead) = (&prework.scratch, prework.lead);
    let (chunk, at) = words.buffer();
    let (from, src) = scratch.at(lead);
    let (to, dst) = chunk.at(at);
    let regions = [
        vk::BufferCopy {
            src_offset: src + merged.prefix_offset,
            dst_offset: dst,
            size: prefix,
        },
        vk::BufferCopy {
            src_offset: src + merged.draws_offset,
            dst_offset: dst + prefix,
            size: draws,
        },
    ];
    let uses = [
        reads(
            scratch,
            lead + merged.prefix_offset..lead + merged.prefix_offset + prefix,
        ),
        reads(
            scratch,
            lead + merged.draws_offset..lead + merged.draws_offset + draws,
        ),
        writes(chunk, at..at + prefix + draws),
    ];
    core.command(
        vk::PipelineStageFlags2::TRANSFER,
        &uses,
        |raw, command_buffer| {
            // SAFETY: the segment's command buffer is recording outside a render pass; the
            // scratch lives until it completes (the segment holds it) and so does the upload (the
            // arena); both regions lie in both: the prefix and the draws in the scratch's layout,
            // and the upload holds them both.
            unsafe { raw.cmd_copy_buffer(command_buffer, from, to, &regions) };
        },
    );
    Ok(words)
}

/// The route of `form`'s commands from the words its prework wrote into `words`
/// ([`read_back`]): `None` where the host reads the records instead (the module docs).
fn route(form: &GpuForm, prework: &Prework, words: &Upload) -> Option<GpuRoute> {
    let layout = form.layout();
    let merged = layout.merged?;
    let commands = usize::try_from(layout.commands).ok()?;
    let len = 8 * commands + 2 * DRAW_ARGUMENT_BYTES as usize;
    let mut bytes = vec![0u8; len];
    // SAFETY: the submission that wrote the words completed, its writes made visible to the host
    // by its last barrier; the bytes are writable and apart from the upload.
    unsafe { words.copy_out(0, bytes.as_mut_ptr(), len as u64) };
    let (prefix, draws) = bytes.split_at(8 * commands);
    let word =
        |at: usize| u32::from_le_bytes([draws[at], draws[at + 1], draws[at + 2], draws[at + 3]]);
    // `VkDrawIndirectCommand`: `vertexCount`, `instanceCount`, `firstVertex`, `firstInstance`.
    // The first draw's whole chunks, and the second's vertices.
    let (chunks, rest) = (word(4), word(16));
    if chunks > 0 {
        return None;
    }
    let total = u64::from(rest);
    let firsts: Vec<u64> = (prefix.as_chunks::<8>().0.iter())
        .map(|first| u64::from_le_bytes(*first))
        .collect();
    let mut corners = Vec::new();
    for (at, &first) in firsts.iter().enumerate() {
        let next = firsts.get(at + 1).copied().unwrap_or(total);
        if first > next || next > total {
            return None;
        }
        if next > first {
            corners.push(u32::try_from(first).ok()?..u32::try_from(next).ok()?);
        }
    }
    Some(GpuRoute {
        header: prework.address + merged.header_offset,
        primitive_type: form.primitive_type(),
        corners,
    })
}
