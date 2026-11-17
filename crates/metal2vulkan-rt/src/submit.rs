//! The submit thread: each queue's FIFO of command buffers, the holds at event waits, the upload
//! that goes ahead of later commits, and the batches it passes to `vkQueueSubmit2`.

use crate::command::{CbInner, CommandBufferError, Pending, Segment};
use crate::complete;
use crate::counters::Metric;
use crate::device::{lock, Core};
use crate::error::Error;
use crate::event::EventInner;
use crate::fill;
use crate::memory::Block;
use crate::queue::QueueInner;
use crate::record::{self, Reach, Recorder};
use crate::times;
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// The submit side: the FIFOs and the upload, and the submit thread's wake-up.
pub(crate) struct Work {
    state: Mutex<WorkState>,
    changed: Condvar,
}

pub(crate) struct WorkState {
    /// The queues with command buffers enqueued or committed and not yet fully submitted.
    fifos: Vec<Fifo>,
    /// The next command buffer's place in the commit order.
    next_seq: u64,
    /// The progress value of the last batch.
    progress: u64,
    /// Zero fills recorded and not yet submitted.
    upload: Option<Upload>,
    /// Something changed that may let a held segment go.
    dirty: bool,
    /// The submit thread is to end: it submits what is ready, then ends what is held.
    stop: bool,
    /// Quiesced or torn down: commits and uploads are refused.
    closed: bool,
    lost: bool,
}

struct Fifo {
    queue: Arc<QueueInner>,
    entries: VecDeque<Entry>,
}

struct Entry {
    cb: Arc<CbInner>,
    /// None while enqueued and not committed.
    pending: Option<Pending>,
}

/// Zero fills recorded into one command buffer, submitted ahead of everything in the next batch,
/// which holds every block they fill until it retires.
pub(crate) struct Upload {
    pub(crate) recorder: Recorder,
    command: vk::CommandBuffer,
    pub(crate) blocks: Vec<Arc<Block>>,
}

/// What one `vkQueueSubmit2` submits.
struct Batch {
    value: u64,
    upload: Option<Upload>,
    segments: Vec<Staged>,
    /// Command buffers whose last segment is in the batch.
    finished: Vec<(Arc<CbInner>, Pending)>,
    /// Events the batch signals that the completion thread is not yet watching.
    watch: Vec<Arc<EventInner>>,
}

/// One segment's submit info.
#[derive(Default)]
struct Staged {
    waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    command: Option<vk::CommandBufferSubmitInfo<'static>>,
    signals: Vec<vk::SemaphoreSubmitInfo<'static>>,
}

/// An ended command buffer, with how it ended.
pub(crate) type Ended = (Arc<CbInner>, Pending, CommandBufferError);

impl Work {
    pub(crate) fn new() -> Work {
        Work {
            state: Mutex::new(WorkState {
                fifos: Vec::new(),
                next_seq: 0,
                progress: 0,
                upload: None,
                dirty: false,
                stop: false,
                closed: false,
                lost: false,
            }),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, WorkState> {
        lock(&self.state)
    }

    /// Wakes the submit thread after a change [`WorkState`] recorded.
    pub(crate) fn wake(&self) {
        self.changed.notify_one();
    }

    /// A host value rose: held segments may go.
    pub(crate) fn poke(&self) {
        self.lock().dirty = true;
        self.wake();
    }

    /// Records the zero fill of `offset..end` of `block` into the upload, which goes ahead of
    /// every later commit.
    ///
    /// Err: [`Error::Quiesced`]; [`Error::Vulkan`] where a command pool or buffer cannot be made.
    pub(crate) fn zero(
        &self,
        core: &Core,
        block: &Arc<Block>,
        offset: u64,
        end: u64,
    ) -> Result<(), Error> {
        let mut state = self.lock();
        if state.closed {
            return Err(Error::Quiesced);
        }
        let device = &core.device;
        let upload = match state.upload.take() {
            Some(upload) => upload,
            None => {
                let mut recorder = core.recorders.take(device)?;
                match recorder.next(device) {
                    Ok(command) => Upload {
                        recorder,
                        command,
                        blocks: Vec::new(),
                    },
                    Err(error) => {
                        core.recorders.give_back(core, recorder);
                        return Err(error);
                    }
                }
            }
        };
        let upload = state.upload.insert(upload);
        // SAFETY: the upload's command buffer is recording, under the lock; the block's buffer has
        // every usage, and `offset..end` lies in it.
        unsafe {
            fill::record(
                device,
                upload.command,
                core.filler.zeros(),
                block.memory.buffer,
                offset,
                end,
            )
        };
        if !upload.blocks.iter().any(|held| Arc::ptr_eq(held, block)) {
            upload.blocks.push(Arc::clone(block));
        }
        Ok(())
    }

    /// Ends `queue`'s held and unsubmitted command buffers with `NotRun`.
    pub(crate) fn abandon(&self, core: &Core, queue: &QueueInner) {
        let ended = {
            let mut state = self.lock();
            match state.fifos.iter().position(|fifo| fifo.queue.is(queue)) {
                Some(index) => {
                    let fifo = state.fifos.swap_remove(index);
                    end(fifo.entries, |_| CommandBufferError::NotRun)
                }
                None => Vec::new(),
            }
        };
        core.done.end_all(core, ended);
    }

    /// Closes the device to commits and uploads, and tells the submit thread to end.
    pub(crate) fn stop(&self) {
        let mut state = self.lock();
        state.closed = true;
        state.stop = true;
        state.dirty = true;
        self.wake();
    }

    /// At a loss: the submit thread ends everything it holds.
    pub(crate) fn lose(&self) {
        let mut state = self.lock();
        state.lost = true;
        state.dirty = true;
        self.wake();
    }

    /// The progress value of the last batch.
    pub(crate) fn progress(&self) -> u64 {
        self.lock().progress
    }
}

impl WorkState {
    pub(crate) fn closed(&self) -> bool {
        self.closed
    }

    pub(crate) fn lost(&self) -> bool {
        self.lost
    }

    /// The next place in the commit order.
    pub(crate) fn seq(&mut self) -> u64 {
        self.next_seq += 1;
        self.next_seq
    }

    fn fifo(&mut self, queue: &Arc<QueueInner>) -> &mut Fifo {
        let index = match self.fifos.iter().position(|fifo| fifo.queue.is(queue)) {
            Some(index) => index,
            None => {
                self.fifos.push(Fifo {
                    queue: Arc::clone(queue),
                    entries: VecDeque::new(),
                });
                self.fifos.len() - 1
            }
        };
        &mut self.fifos[index]
    }

    /// Takes `cb`'s place in its queue's FIFO.
    pub(crate) fn enqueue(&mut self, cb: &Arc<CbInner>) {
        self.fifo(&cb.queue).entries.push_back(Entry {
            cb: Arc::clone(cb),
            pending: None,
        });
    }

    /// Hands `cb` to the submit thread: at its place where it was enqueued, else at its queue's
    /// end.
    pub(crate) fn commit(&mut self, cb: &Arc<CbInner>, pending: Pending) {
        self.dirty = true;
        let fifo = self.fifo(&cb.queue);
        match fifo
            .entries
            .iter_mut()
            .find(|entry| Arc::ptr_eq(&entry.cb, cb))
        {
            Some(entry) => entry.pending = Some(pending),
            None => fifo.entries.push_back(Entry {
                cb: Arc::clone(cb),
                pending: Some(pending),
            }),
        }
    }

    /// The FIFO whose head goes next: of the committed heads whose next segment is ready, the
    /// earliest committed.
    fn next_ready(&self) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;
        for (index, fifo) in self.fifos.iter().enumerate() {
            let Some(Entry {
                pending: Some(pending),
                ..
            }) = fifo.entries.front()
            else {
                continue;
            };
            if best.is_some_and(|(_, seq)| seq < pending.seq) {
                continue;
            }
            if ready(&pending.gpu.segments[pending.next]) {
                best = Some((index, pending.seq));
            }
        }
        best.map(|(index, _)| index)
    }

    /// Takes every ready segment, in commit order, into one batch; None where none is ready.
    fn stage(&mut self) -> Option<Batch> {
        let value = self.progress + 1;
        let mut segments = Vec::new();
        let mut finished = Vec::new();
        let mut watch = Vec::new();
        while let Some(index) = self.next_ready() {
            let fifo = &mut self.fifos[index];
            let Some(Entry {
                pending: Some(pending),
                ..
            }) = fifo.entries.front_mut()
            else {
                break;
            };
            segments.push(stage(&pending.gpu.segments[pending.next], &mut watch));
            pending.next += 1;
            pending.value = value;
            if pending.next == pending.gpu.segments.len() {
                if let Some(Entry {
                    cb,
                    pending: Some(pending),
                }) = fifo.entries.pop_front()
                {
                    finished.push((cb, pending));
                }
                if fifo.entries.is_empty() {
                    self.fifos.swap_remove(index);
                }
            }
        }
        if segments.is_empty() {
            return None;
        }
        self.progress = value;
        Some(Batch {
            value,
            upload: self.upload.take(),
            segments,
            finished,
            watch,
        })
    }
}

/// Whether every wait of `segment` is met by the host's value or by a submitted signal.
fn ready(segment: &Segment) -> bool {
    segment.waits.iter().all(|(event, value)| {
        let state = lock(&event.state);
        state.floor >= *value || state.submitted >= *value
    })
}

/// A ready segment's submit info. A wait the host's value meets submits no semaphore wait; a
/// signal not above the highest value submitted or set is left out, so the semaphore only rises.
fn stage(segment: &Segment, watch: &mut Vec<Arc<EventInner>>) -> Staged {
    let mut staged = Staged {
        command: segment
            .command
            .map(|command| vk::CommandBufferSubmitInfo::default().command_buffer(command)),
        ..Staged::default()
    };
    for (event, value) in &segment.waits {
        if lock(&event.state).floor >= *value {
            continue;
        }
        staged.waits.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(event.semaphore)
                .value(*value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
    }
    for (event, value) in &segment.signals {
        let mut state = lock(&event.state);
        if *value <= state.submitted.max(state.floor) {
            continue;
        }
        state.submitted = *value;
        state.pending.push_back(*value);
        if !state.watched {
            state.watched = true;
            watch.push(Arc::clone(event));
        }
        staged.signals.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(event.semaphore)
                .value(*value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
    }
    staged
}

/// Ends FIFO entries: a committed one now, one not yet committed at its commit; `error` answers
/// for each, with its pending part where it is committed.
fn end(
    entries: VecDeque<Entry>,
    error: impl Fn(Option<&Pending>) -> CommandBufferError,
) -> Vec<Ended> {
    let mut ended = Vec::new();
    for Entry { cb, pending } in entries {
        match pending {
            Some(pending) => {
                let error = error(Some(&pending));
                ended.push((cb, pending, error));
            }
            None => cb.end_uncommitted(error(None)),
        }
    }
    ended
}

/// The submit thread, until it is stopped.
pub(crate) fn run(core: Arc<Core>) {
    let work = &core.work;
    let mut state = work.lock();
    loop {
        if state.lost {
            let ended = drain(&mut state, |pending| CommandBufferError::DeviceLost {
                in_flight: pending.is_some_and(|pending| pending.value > 0),
            });
            let upload = state.upload.take();
            drop(state);
            if let Some(upload) = upload {
                upload.retire(&core);
            }
            core.done.end_all(&core, ended);
            state = work.lock();
        } else if let Some(batch) = state.stage() {
            drop(state);
            submit(&core, batch);
            state = work.lock();
            continue;
        }
        if state.stop {
            let ended = drain(&mut state, |_| CommandBufferError::NotRun);
            let upload = state.upload.take();
            drop(state);
            if let Some(upload) = upload {
                upload.retire(&core);
            }
            core.done.end_all(&core, ended);
            return;
        }
        while !state.dirty {
            state = work
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.dirty = false;
    }
}

/// Ends every FIFO's entries.
fn drain(
    state: &mut WorkState,
    error: impl Fn(Option<&Pending>) -> CommandBufferError,
) -> Vec<Ended> {
    state
        .fifos
        .drain(..)
        .flat_map(|fifo| end(fifo.entries, &error))
        .collect()
}

/// Submits a batch: the upload first, then each segment in order, the last signalling the
/// progress value. Whatever was passed to `vkQueueSubmit2` goes to the completion thread as in
/// flight; a failed submission is taken as a loss.
fn submit(core: &Core, batch: Batch) {
    let Batch {
        value,
        upload,
        mut segments,
        finished,
        watch,
    } = batch;
    let device = &core.device;
    // The upload's fills are visible to every command after it.
    let mut ok = upload.as_ref().is_none_or(|upload| {
        record::barrier(device, upload.command, Reach::Commands);
        core.counters.count(Metric::Barriers);
        // SAFETY: the upload's command buffer was recording under the work lock, which `stage`
        // took it out under.
        unsafe { device.end_command_buffer(upload.command) }.is_ok()
    });
    if let Some(upload) = &upload {
        segments.insert(
            0,
            Staged {
                command: Some(
                    vk::CommandBufferSubmitInfo::default().command_buffer(upload.command),
                ),
                ..Staged::default()
            },
        );
    }
    segments.retain(|staged| {
        staged.command.is_some() || !staged.waits.is_empty() || !staged.signals.is_empty()
    });
    let progress = vk::SemaphoreSubmitInfo::default()
        .semaphore(core.done.progress)
        .value(value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    match segments.last_mut() {
        Some(last) => last.signals.push(progress),
        None => segments.push(Staged {
            signals: vec![progress],
            ..Staged::default()
        }),
    }
    let infos: Vec<vk::SubmitInfo2<'_>> = segments
        .iter()
        .map(|staged| {
            vk::SubmitInfo2::default()
                .wait_semaphore_infos(&staged.waits)
                .command_buffer_infos(staged.command.as_slice())
                .signal_semaphore_infos(&staged.signals)
        })
        .collect();
    if ok {
        let queue = lock(&core.queues.runtime);
        // SAFETY: every command buffer is executable and in no other pending submission; every
        // wait is on a signal submitted earlier, and every signal raises its semaphore; the queue
        // is held.
        ok = unsafe { device.queue_submit2(*queue, &infos, vk::Fence::null()) }.is_ok();
        drop(queue);
        core.counters.count(Metric::Submissions);
    }
    if !ok {
        complete::lose(core);
    }
    let now = times::host_time();
    for (cb, _) in &finished {
        cb.scheduled(now);
    }
    core.done.submitted(core, value, upload, finished, watch);
}

impl Upload {
    /// Recycles the upload's recorder and gives up its blocks: once its submission completed, or
    /// where it never went.
    pub(crate) fn retire(self, core: &Core) {
        core.recorders.give_back(core, self.recorder);
    }
}
