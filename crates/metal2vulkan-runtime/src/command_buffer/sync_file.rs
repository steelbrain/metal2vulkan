//! A command buffer's completion as a sync file (a runtime extension: Metal has none): a binary
//! semaphore its last batch signals, exported as `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`,
//! which a `VkDevice` of another driver imports and waits for on its own GPU
//! ([`CommandBuffer::add_completion_sync_file`], [`CommandBuffer::export_completion_sync_file`]).
//!
//! - **The signal.** Asked for before the commit, the semaphore (one of the device's idle ones, or
//!   a new one: `exported_semaphore.rs`) is signaled by the command buffer's last batch: the
//!   commit adds it to the batches as a signal of one of the caller's binary semaphores
//!   (`Batches::binary_signal`), after its last encoder and past every point where the commit may
//!   still be refused, so it goes into the last batch's end with the queue's timeline value, after
//!   the barrier that makes every write available to the host (`submit.rs`). A batch that
//!   signals a binary semaphore never waits in a group for one kernel submission (`group.rs`),
//!   and a command buffer with one needs a submission even with no GPU work: a submission of the
//!   signals alone, after every earlier submission of its `VkQueue`, so it completes in its
//!   queue's order. Every host step runs before the batch it records for, so none follows the
//!   last batch's submission.
//! - **The export** waits for that submission only, never for the GPU, then exports the payload
//!   (`vkGetSemaphoreFdKHR`), which Vulkan allows once the signal operation, and every signal
//!   operation it depends on, was submitted (`VUID-VkSemaphoreGetFdInfoKHR-handleType-03254`):
//!   every wait of the runtime's batches is for a signal submitted before it, but for a signal the
//!   caller reserved (`Event::reserve_vk_signal`), whose submission that call's contract orders
//!   before such an export. The export moves the payload out, as a wait would (copy transference),
//!   so a command buffer's sync file is exported once.
//! - **Ends the GPU does not see.** A command buffer that finished (`Completed` or `Error`) before
//!   its signal was submitted (a failed submission or host step, a lost device, a revoked queue,
//!   its GPU hang holding its last batch) or before the export looked answers a sync file already
//!   signaled (`None`): it ended. One whose completion writes the caller's pages on the host after
//!   its GPU work (a copied host buffer's merge back, `copied.rs`; a two-copy buffer's copy out
//!   into pages no import maps, `managed_copy.rs`) cannot be ordered on the GPU: its export waits
//!   for its completion on the host and answers `None`, announced
//!   (`sync-file-after-host-copies`).
//! - **The semaphore's end.** Its command buffer's completion takes it, every batch that names it
//!   completed: back among the device's idle ones where an export moved its payload out and the
//!   command buffer completed, so it is unsignaled with nothing pending; else released. A command
//!   buffer never committed gives it back as it goes.

use super::own::{running, scheduled_wait, Answer};
use super::{CommandBuffer, CommandBufferShared, State};
use crate::contexts;
use crate::copied::CopiesReached;
use crate::degradation::Degradation;
use crate::device::lock;
use crate::error::{internal, invalid, DeadlockCause, Error, Problem, RuntimeGap, SyncFileProblem};
use crate::exit_guard;
use crate::managed_copy::AtCompletion;
use crate::release;
use crate::values::CommandBufferStatus;
use ash::vk;
use std::os::fd::OwnedFd;
use std::sync::atomic::Ordering;

/// A command buffer's completion sync file ([`State::sync_file`]).
pub(super) struct SyncFile {
    /// Its binary semaphore, made exportable as a sync file, until its command buffer's
    /// completion or release takes it.
    semaphore: Option<vk::Semaphore>,
    /// The commit added its signal to the last batch.
    placed: bool,
    /// The batch that signals it was submitted.
    submitted: bool,
    /// An export answered.
    taken: bool,
    /// An export moved its payload out (`vkGetSemaphoreFdKHR` returned).
    exported: bool,
}

/// Whether the command buffer finished: its transition to `Completed` or `Error` ran.
fn finished(state: &State) -> bool {
    matches!(
        state.status,
        CommandBufferStatus::Completed | CommandBufferStatus::Error
    )
}

/// Whether an export may answer now: the sync file's signal was submitted, or the command buffer
/// finished.
fn answerable(state: &State) -> bool {
    finished(state) || state.sync_file.as_ref().is_some_and(|sync| sync.submitted)
}

impl State {
    /// The commit: the sync file's signal goes into the last batch, where one was asked for.
    pub(super) fn place_sync_file(&mut self) {
        let Some(sync) = &mut self.sync_file else {
            return;
        };
        if let (Some(semaphore), false) = (sync.semaphore, sync.placed) {
            self.batches.binary_signal(semaphore);
            sync.placed = true;
        }
    }

    /// The command buffer's work ended: its sync file's semaphore, to give back
    /// ([`CommandBufferShared::release_sync_file`]), and whether the device may keep it idle.
    pub(super) fn take_sync_file(&mut self) -> Option<(vk::Semaphore, bool)> {
        let completed = self.status == CommandBufferStatus::Completed;
        let sync = self.sync_file.as_mut()?;
        let idle = !sync.placed || (completed && sync.exported);
        sync.semaphore.take().map(|semaphore| (semaphore, idle))
    }
}

impl CommandBufferShared {
    /// Batch submitted with `signals`: where they hold the sync file's, it was submitted, and an
    /// export waiting for that is woken.
    pub(super) fn sync_file_submitted(&self, state: &mut State, signals: &[vk::Semaphore]) {
        let Some(sync) = &mut state.sync_file else {
            return;
        };
        if sync
            .semaphore
            .is_some_and(|semaphore| signals.contains(&semaphore))
        {
            sync.submitted = true;
            let parked = state.parked > 0;
            self.wake_parked(parked);
        }
    }

    /// Give back the sync file's semaphore `taken` ([`State::take_sync_file`]), nothing pending
    /// using it: kept idle by the device where the flag says it may be, else released. Once the
    /// process is exiting it is left to its end.
    pub(super) fn release_sync_file(&self, taken: Option<(vk::Semaphore, bool)>) {
        let Some((semaphore, idle)) = taken else {
            return;
        };
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        let Some(sync_fd) = &shared.sync_fd_semaphores else {
            return;
        };
        // SAFETY: the device's semaphore, made by `take`; its command buffer's work ended (or was
        // never submitted), so nothing pending uses it; an idle one is unsignaled: never
        // signaled, or its payload moved out by an export after its signal completed.
        unsafe {
            match idle {
                true => sync_fd.give(shared.vk(), semaphore),
                false => {
                    release::object(shared.vk(), semaphore);
                }
            }
        }
    }

    /// Whether its completion writes the caller's pages on the host after its GPU work: a copy
    /// out of a two-copy buffer into pages no import maps, or the merge back of a copied host
    /// buffer its commands may have written (`completion.rs`'s `finish`). `writes_unnamed` is
    /// its state's. Read once its last batch was submitted, its writes all noted.
    fn host_copies_after(&self, writes_unnamed: bool) -> bool {
        if lock(&self.at_completion)
            .iter()
            .any(AtCompletion::writes_pages)
        {
            return true;
        }
        let copied = &self.device.shared().copied;
        if !copied.any() {
            return false;
        }
        let mut written = CopiesReached::default();
        match self.unnoted.load(Ordering::Relaxed) || writes_unnamed {
            true => written.reached_all(),
            false => lock(&self.dirty).copies_written(&mut written),
        }
        copied.merges_back(&written)
    }
}

impl CommandBuffer {
    /// No Metal counterpart (a runtime extension): ask for the command buffer's completion as a
    /// sync file, which [`export_completion_sync_file`](Self::export_completion_sync_file) hands
    /// out once the command buffer is committed. Its last batch then also signals a binary
    /// semaphore of the device's, exportable as `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`
    /// (`VK_KHR_external_semaphore_fd`), which a `VkDevice` of any driver imports: that device's
    /// submission waits on its own GPU for this command buffer's work to end, where a wait for
    /// [`wait_until_completed`](Self::wait_until_completed) blocks a thread. A second call
    /// changes nothing.
    ///
    /// The signal comes after every command of the command buffer, behind the barrier that makes
    /// their writes available to the host, and after every earlier submission of its queue's
    /// `VkQueue`. It costs the command buffer a submission where it needs no GPU work, and its last
    /// batch never joins a grouped submission (`CompletionPolicy::Grouped`).
    ///
    /// Refused after the commit (`Invalid`, [`SyncFileProblem::Committed`]), on a lost device
    /// (`DeviceLost`), and `NotImplemented` ([`RuntimeGap::SyncFileExport`]) on a device without
    /// `VK_KHR_external_semaphore_fd` or that exports no binary semaphore as a sync file: Metal
    /// on Apple's GPUs (MoltenVK) among them. The driver's failure to make the semaphore refuses
    /// it too.
    pub fn add_completion_sync_file(&self) -> Result<(), Error> {
        const CALL: &str = "add_completion_sync_file";
        let device = &self.0.device;
        let shared = device.shared();
        let Some(sync_fd) = &shared.sync_fd_semaphores else {
            return Err(Error::NotImplemented(RuntimeGap::SyncFileExport));
        };
        if device.is_lost() {
            return Err(device.lost_refusal());
        }
        let mut state = self.state();
        if state.enqueued() {
            return Err(invalid(CALL, Problem::SyncFile(SyncFileProblem::Committed)));
        }
        if state.sync_file.is_some() {
            return Ok(());
        }
        // SAFETY: the device is live: this command buffer holds it.
        let semaphore = unsafe { sync_fd.take(shared.vk()) }?;
        state.sync_file = Some(SyncFile {
            semaphore: Some(semaphore),
            placed: false,
            submitted: false,
            taken: false,
            exported: false,
        });
        Ok(())
    }

    /// No Metal counterpart (a runtime extension): the command buffer's completion as a sync
    /// file, which [`add_completion_sync_file`](Self::add_completion_sync_file) asked for before
    /// the commit: a new descriptor (`vkGetSemaphoreFdKHR`,
    /// `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`), the caller's to close, which a `VkDevice`
    /// of any driver imports (`VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`) and waits for. `None` is a sync
    /// file already signaled, which Vulkan imports as the descriptor -1: there is nothing to wait
    /// for.
    ///
    /// It blocks only until the command buffer's last batch, which signals the semaphore, was
    /// submitted (as [`wait_until_scheduled`](Self::wait_until_scheduled) blocks, without waiting
    /// for the scheduled handlers), never for the GPU's work: after
    /// [`commit_and_submit`](Self::commit_and_submit) it answers at once. Its answers:
    /// - a descriptor (or the driver's -1, `None`) once the signal was submitted;
    /// - `None` where the command buffer finished first, `Completed` or `Error`: a failed
    ///   submission or host step, a lost device or a revoked queue, which never submit the
    ///   signal, or a completion before the export looked;
    /// - `None` once the command buffer completed, where its completion writes the caller's pages
    ///   on the host after its GPU work, which no GPU wait orders: the merge back of a copied host
    ///   buffer its commands may have written, or a two-copy buffer's copy out into pages no
    ///   import maps. That export blocks until the completion, announced
    ///   ([`Degradation::SyncFileAfterHostCopies`]).
    ///
    /// The importer orders after its wait only what its own memory access makes visible: the
    /// command buffer's writes are available to the host as its work ends. On a device loss the
    /// command buffer fails; what an importer's wait for a descriptor handed out before does
    /// then is its driver's.
    ///
    /// Refused for a command buffer no `add_completion_sync_file` asked for (`Invalid`,
    /// `Problem::NotExportable`); before its commit ([`SyncFileProblem::NotCommitted`]); after an
    /// export answered ([`SyncFileProblem::Exported`]: one export per command buffer, as the
    /// export moves the payload out; duplicate the descriptor instead); where its wait would
    /// deadlock, as `wait_until_scheduled`'s (`WaitWouldDeadlock`), or the host copies' wait for
    /// the completion, as `wait_until_completed`'s; and for the driver's failure.
    pub fn export_completion_sync_file(&self) -> Result<Option<OwnedFd>, Error> {
        const CALL: &str = "export_completion_sync_file";
        let ready = {
            let state = self.state();
            match &state.sync_file {
                None => return Err(invalid(CALL, Problem::NotExportable)),
                Some(sync) if sync.taken => {
                    return Err(invalid(CALL, Problem::SyncFile(SyncFileProblem::Exported)))
                }
                Some(_) => {}
            }
            if !state.enqueued() {
                return Err(invalid(
                    CALL,
                    Problem::SyncFile(SyncFileProblem::NotCommitted),
                ));
            }
            answerable(&state)
        };
        if !ready {
            self.wait_for_sync_file()?;
        }
        let (done, writes_unnamed) = {
            let state = self.state();
            (finished(&state), state.writes_unnamed)
        };
        let device = &self.0.device;
        let shared = device.shared();
        if !done && self.0.host_copies_after(writes_unnamed) {
            device.degrade(Degradation::SyncFileAfterHostCopies, || {
                (
                    "a command buffer's completion sync file".to_owned(),
                    "its completion writes the caller's pages on the host after its GPU work \
                     (a copied host buffer's merge back, or a two-copy buffer's copy out into \
                     pages no import maps), which no GPU wait orders"
                        .to_owned(),
                )
            });
            self.wait_until_completed()?;
            let mut state = self.state();
            let Some(sync) = state.sync_file.as_mut().filter(|sync| !sync.taken) else {
                return Err(invalid(CALL, Problem::SyncFile(SyncFileProblem::Exported)));
            };
            sync.taken = true;
            shared.counts.sync_file_after_host_copies();
            return Ok(None);
        }
        let mut state = self.state();
        let done = finished(&state);
        let Some(sync) = state.sync_file.as_mut().filter(|sync| !sync.taken) else {
            return Err(invalid(CALL, Problem::SyncFile(SyncFileProblem::Exported)));
        };
        let semaphore = match (done, sync.semaphore) {
            (false, Some(semaphore)) => semaphore,
            _ => {
                sync.taken = true;
                return Ok(None);
            }
        };
        // A sync file was asked for on this device, which so exports one.
        let sync_fd = (shared.sync_fd_semaphores.as_ref())
            .ok_or_else(|| internal("a sync file asked for on a device that exports none"))?;
        // SAFETY: the semaphore lives until the command buffer's completion takes it, under the
        // state's lock this export holds, and the command buffer has not finished; its signal was
        // submitted, and so was every signal its batch waits for (the module docs); nothing waits
        // for it.
        let file = unsafe { sync_fd.export(semaphore) }?;
        sync.taken = true;
        sync.exported = true;
        shared.counts.sync_file_exported();
        Ok(file)
    }

    /// Block until the sync file's signal was submitted or the command buffer finished, as
    /// [`wait_until_scheduled`](Self::wait_until_scheduled) blocks.
    fn wait_for_sync_file(&self) -> Result<(), Error> {
        let _blocked = match scheduled_wait(running(), self.0.target(), false) {
            Answer::Refuse(cause) => return Err(cause.into()),
            Answer::Return | Answer::Wait => None,
            Answer::Block(from) => Some(contexts::block(from, self.0.context())?),
        };
        self.settle_commit();
        self.0.waits_for_scheduling();
        let mut state = self.state();
        while !answerable(&state) {
            if exit_guard::closed() {
                return Err(DeadlockCause::Exiting.into());
            }
            state.parked += 1;
            state = self
                .0
                .changed
                .wait_timeout(state, exit_guard::WAKE)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
            state.parked -= 1;
        }
        Ok(())
    }
}
