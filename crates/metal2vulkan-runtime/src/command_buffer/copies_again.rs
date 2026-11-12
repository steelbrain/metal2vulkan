//! Copied host buffers copied in again after an event wait (`copied.rs`, **Coherence**).
//!
//! A commit copies into each copy its commands reach the bytes the CPU changed in its pages,
//! before any of its work is submitted. A batch that waits for an event runs its commands once
//! the event reaches the value, and Metal's reads there see the pages as they stand then: a CPU
//! write made after the commit and before the signal (the host's, or a command buffer's committed
//! after it) is read. So a batch whose wait the commit found not reached, as it stood before the
//! commit's copy in, copies in again as it is submitted, which the scheduler does only once the
//! wait is met: the copies the command buffer's commands reach, as at the commit, take the bytes
//! the CPU changed since. Counted: `WorkCounts::copied_copy_ins_after_waits`.
//!
//! A batch with no wait, and one whose every wait the commit found reached, copies in nothing
//! more: every CPU write before its commands may run was made before the commit's copy in. A wait
//! a GPU signal meets that a copy's merge back must come before is held on the host until that
//! merge (`event_timeline.rs`), so the copy in again after it takes the merged bytes too.

use super::batches::Batch;
use super::{lock, recording_error, CommandBufferShared};
use crate::copied::CopiesReached;
use crate::error::{CommandBufferError, Error, MemoryKind};
use crate::event::EventShared;
use crate::event_timeline::WaitFor;
use ash::vk;
use std::sync::Arc;

/// The batches of a committed command buffer that copy in again, and the copies its commands
/// reach.
pub(super) struct CopiesAgain {
    /// The batches, by index, not yet submitted.
    batches: Vec<usize>,
    /// The copies its commands may read, and those they may write, as the commit found them
    /// (`CopiedBuffers::copy_in`).
    reached: Arc<(CopiesReached, CopiesReached)>,
}

impl CopiesAgain {
    /// The indices of `batches` that wait for an event value not reached now.
    pub(super) fn unreached(batches: &[Batch<Arc<EventShared>, vk::Semaphore>]) -> Vec<usize> {
        (batches.iter().enumerate())
            .filter(|(_, batch)| {
                (batch.waits.iter())
                    .any(|(event, value)| event.wait(*value) != Some(WaitFor::Reached))
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// The copies in again of `batches`, reaching `reached`; `None` where there are none.
    pub(super) fn new(
        batches: Vec<usize>,
        reached: (CopiesReached, CopiesReached),
    ) -> Option<CopiesAgain> {
        (!batches.is_empty()).then(|| CopiesAgain {
            batches,
            reached: Arc::new(reached),
        })
    }
}

impl CommandBufferShared {
    /// Copy in again for batch `index`, where its commit left one for it, once: as it is
    /// submitted, its waits met. Copied with no lock of the command buffer's held, as a copy over
    /// logged pages waits for the log's publication; where the host has no memory for a copy's
    /// base, the command buffer fails as out of memory, as its commit would have.
    pub(super) fn copy_in_again(&self, index: usize) -> Result<(), CommandBufferError> {
        let reached = {
            let mut state = lock(&self.state);
            let Some(again) = &mut state.copies_again else {
                return Ok(());
            };
            let Some(at) = again.batches.iter().position(|&batch| batch == index) else {
                return Ok(());
            };
            again.batches.swap_remove(at);
            let reached = Arc::clone(&again.reached);
            if again.batches.is_empty() {
                state.copies_again = None;
            }
            reached
        };
        let device = &self.device;
        let shared = device.shared();
        let (used, written) = &*reached;
        let copied_in = shared.copied.copy_in(
            &shared.counts,
            shared.dirty.tally(),
            (used, written),
            || shared.dirty.in_flight(),
            &mut |length| device.copied_base(length),
        );
        if copied_in.is_err() {
            return Err(recording_error(
                device,
                Error::OutOfMemory(MemoryKind::Host),
            ));
        }
        shared.counts.copied_copy_ins_after_waits();
        Ok(())
    }
}
