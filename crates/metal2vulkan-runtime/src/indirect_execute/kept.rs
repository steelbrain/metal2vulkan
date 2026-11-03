//! An execution of compute commands the CPU encoded, recorded chunk by chunk so a later one runs
//! the chunks no write reached again (`kept_commands.rs`).
//!
//! - **Which executions.** One `commit` records from the CPU's records (not the host step), of a
//!   range in the call, outside a block of command predication, with no check of the caller's
//!   that may see something among its commands (one whose needs rule out what the buffer's
//!   records may hold calls nothing at `commit` either, `Execute::check_sees`), no inherited
//!   table that binds anything, in a command buffer with no executed-commands handler that is
//!   not encoded again after its commit; no chunk of its range last recorded otherwise
//!   than plainly; and a chunk of its range run before, no write reaching it since
//!   (`kept_commands.rs`, **From the second execution**): a range's first execution records as
//!   one that keeps nothing.
//! - **The key.** A chunk is recorded from its records and from the inputs [`ComputeKey`] holds:
//!   the inherited pipeline state, whether the encoder runs its dispatches in encoded order, and
//!   the resources made resident (`Residency::kept_key`, one list an execution, compared whole
//!   once a list its kept chunks hold: `kept_commands::ResidentMet`). A kept chunk runs again only
//!   under an equal key.
//! - **Each chunk** runs its kept recording again, or records apart into a primary of its own
//!   (from a context of its own where an execution ran it before, no write reaching it since,
//!   else from its command buffer's own, kept by none), as a segment of its own
//!   (`EncoderCore::segment`): after a barrier from everything before,
//!   and with one to everything after, so it orders against nothing else of the execution's, as a
//!   command's barrier at its start would. Its recording is plain where it noted no shadowed
//!   texture's work, named no detached buffer, named no released state, recorded into one command
//!   buffer, held only shared objects, and reached nothing its command buffer alone holds
//!   (`CommandBuffer::own_reached`); a plain chunk is kept, any other noted, so later executions
//!   of it record as before until a write reaches it. The execution's segment is its chunks', in
//!   order.

use super::{Execute, ExecutionRange};
use crate::command_buffer::CommandBuffer;
use crate::encoder::{EncoderCore, RecordedSegment};
use crate::error::Error;
use crate::indirect_command_buffer::Reader;
use crate::kept_commands::{
    self, Kept, KeptChunk, KeptContext, KeptEffects, ResidentKey, ResidentMet,
};
use crate::pipeline::{BufferForms, ComputePipelineState};
use crate::residency::Residency;
use metal2vulkan::bind::indirect_commands::BufferStage;
use metal2vulkan::bind::ReplayFaults;
use std::ops::Range;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// What a chunk of an execution's dispatches is recorded from beside its records (the module
/// docs).
#[derive(Clone, PartialEq)]
pub(super) struct ComputeKey {
    pipeline: Option<ComputePipelineState>,
    ordered: bool,
    resident: ResidentKey,
}

/// An execution that keeps its commands (the module docs): its key, what is kept of its range's
/// chunks in order, and the CPU's writes of the buffer as they were looked up.
pub(super) struct Keeping {
    key: ComputeKey,
    kept: Vec<Kept>,
    read_at: u64,
}

impl Execute {
    /// Whether the execution keeps its commands (the module docs), recorded by `command_buffer`
    /// against `residency`: its key and its chunks.
    pub(super) fn keeping(
        &self,
        command_buffer: &CommandBuffer,
        residency: &Residency,
    ) -> Option<Keeping> {
        let ExecutionRange::Fixed(range) = &self.range else {
            return None;
        };
        let inherited = self.icb.descriptor().inherit_buffers;
        let keeps = !range.is_empty()
            && !self.icb.gpu_written()
            && !self.check_sees()
            && self.gate.is_none()
            && self.preprocess_gate.is_none()
            && (!inherited || self.table.binds_nothing())
            && !crate::replay::is_replaying()
            && !command_buffer.reports_executed();
        if !keeps {
            return None;
        }
        let (kept, read_at) = self.icb.kept_chunks(range.clone())?;
        let key = ComputeKey {
            pipeline: self.pipeline.clone(),
            ordered: self.ordered,
            resident: ResidentKey::new(residency.kept_key()),
        };
        Some(Keeping { key, kept, read_at })
    }

    /// Record the commands chunk by chunk into a segment of `command_buffer` (the module docs),
    /// resolving against `residency`, the command buffer's: `record`'s, where the execution keeps
    /// its commands (`keeping`).
    pub(super) fn record_kept(
        &self,
        command_buffer: &CommandBuffer,
        residency: &Residency,
        keeping: Keeping,
    ) -> Result<RecordedSegment, Error> {
        let (range, faults) = self.range()?;
        let Keeping {
            mut key,
            kept,
            read_at,
        } = keeping;
        // Counted before the records are read: a write meanwhile records it again.
        self.read_at.store(read_at, Ordering::Release);
        let counts = &command_buffer.device().shared().counts;
        let mut met = ResidentMet::default();
        let mut parts = Vec::new();
        for ((chunk, covered), kept) in kept_commands::chunks(range).zip(kept) {
            let (kept, keeps): (Option<Arc<KeptChunk>>, bool) = match kept {
                Kept::Recording(kept) => (Some(kept), true),
                Kept::Again => (None, true),
                Kept::First => (None, false),
            };
            let again = kept.and_then(|kept| {
                let resident = &kept.key::<ComputeKey>()?.resident;
                if !met.same(&mut key.resident, resident, counts) {
                    return None;
                }
                let held = kept.runs_again(command_buffer, (&key, &covered), residency)?;
                Some((kept, held))
            });
            if let Some((kept, held)) = again {
                let mut core = EncoderCore::capture(command_buffer);
                core.run_kept(&kept.effects, held)?;
                let raw = kept.raw;
                let misses = kept.misses;
                core.retain_shared(kept);
                let mut part = core.finish_capture()?.into_segment(vec![raw])?;
                part.resident_misses = misses;
                counts.indirect_chunk_kept(u64::from(covered.end - covered.start));
                parts.push(part);
                continue;
            }
            // A chunk's first execution records from its command buffer's own.
            let context = match keeps {
                true => Some(KeptContext::new(command_buffer)?),
                false => None,
            };
            let reached = command_buffer.own_reached();
            if let Some(context) = &context {
                command_buffer.lend_kept(Some(context));
            }
            let apart = self.record_apart(command_buffer, residency, covered.clone());
            if context.is_some() {
                command_buffer.lend_kept(None);
            }
            let (mut part, resolved, (moved, plain)) = apart?;
            let held = part.retained.all_shared();
            let keeps = plain && command_buffer.own_reached() == reached && part.raws.len() == 1;
            // A first execution's chunk that would not be kept is noted as one that is not.
            let Some(context) = context else {
                if !keeps || held.is_none() {
                    self.icb.keep_chunk(chunk, None, read_at);
                }
                parts.push(part);
                continue;
            };
            match (keeps, held) {
                (true, Some(held)) => {
                    let effects = KeptEffects::new(
                        part.summary.clone(),
                        (part.keys.clone(), part.taken.named().clone()),
                        &held,
                        part.executions.clone(),
                    );
                    let kept = KeptChunk::new(
                        (Arc::clone(&context), part.raws[0]),
                        covered,
                        Box::new(key.clone()),
                        (resolved, moved),
                        effects,
                        (part.resident_misses, ReplayFaults::default()),
                    );
                    self.icb.keep_chunk(chunk, Some(Arc::new(kept)), read_at);
                }
                (true, None) | (false, _) => self.icb.keep_chunk(chunk, None, read_at),
            }
            part.retained.shared(context);
            parts.push(part);
        }
        let mut parts = parts.into_iter();
        let mut segment = parts.next().unwrap_or_else(EncoderCore::unrecorded_segment);
        for part in parts {
            segment.join(part);
        }
        segment.faults = ReplayFaults(segment.faults.0 | faults.0);
        Ok(segment)
    }

    /// Record commands `covered` apart (the module docs) into a segment of `command_buffer`,
    /// resolving against `residency`: the segment, the addresses they named and what they
    /// resolved to, the no-copy buffers' moves before it read its records, and whether it is
    /// plain but for what its command buffer alone holds.
    fn record_apart(
        &self,
        command_buffer: &CommandBuffer,
        residency: &Residency,
        covered: Range<u32>,
    ) -> Result<(RecordedSegment, Vec<kept_commands::Resolution>, (u64, bool)), Error> {
        let moved = crate::buffer::host_cells_moved();
        let (commands, released) = self.icb.records(covered, Reader::Commit)?;
        self.icb.degrade_released(&released);
        let descriptor = *self.icb.descriptor();
        let addresses = commands
            .iter()
            .filter(|_| !descriptor.inherit_buffers)
            .flat_map(|(record, _)| {
                let bindings = record.bindings.buffers(BufferStage::Kernel);
                let count = descriptor.max_kernel_buffer_bind_count;
                (0..count).zip(bindings).filter_map(|(_, binding)| *binding)
            })
            .filter(|binding| binding.names_buffer())
            .map(|binding| binding.address);
        let resolved = kept_commands::resolutions(addresses, residency);
        let mut core = EncoderCore::segment(command_buffer, self.ordered)?;
        let mut misses = 0u32;
        let forms = (residency, BufferForms::Bound);
        self.dispatch(&mut core, commands.iter(), forms, &mut misses)?;
        let shadows = core.used_shadows();
        let mut segment = core.end_segment()?;
        segment.resident_misses = misses;
        let plain = !shadows && !segment.taken.dead() && released.is_empty();
        Ok((segment, resolved, (moved, plain)))
    }
}
