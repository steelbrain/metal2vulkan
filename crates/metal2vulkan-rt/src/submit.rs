//! The submit thread: each queue's FIFO of command buffers, the holds at event waits, the upload
//! that goes ahead of later commits, and the batches it passes to `vkQueueSubmit2`.
//!
//! A batch ends at most [`GROUP`] command buffers. A full one goes at once; one the thread runs out
//! of ready segments for goes at once too, unless [`IN_FLIGHT`] submissions are not yet seen
//! complete: then it is kept, and takes what is committed meanwhile, until one of them completes,
//! it fills, or something waits for it (a host wait on one of its command buffers, an upload to
//! flush, the device stopping).

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
use crate::texture::{Image, Staging};
use crate::times;
use ash::vk;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// The command buffers one batch ends at most.
const GROUP: usize = 4;

/// The submissions not yet seen complete at which a batch not full is kept.
const IN_FLIGHT: usize = 2;

/// The submit side: the FIFOs and the upload, and the submit thread's wake-up.
pub(crate) struct Work {
    state: Mutex<WorkState>,
    changed: Condvar,
    /// The thread keeps a batch: the completion thread wakes it as submissions complete.
    keeping: AtomicBool,
}

pub(crate) struct WorkState {
    /// The queues with command buffers enqueued or committed and not yet fully submitted.
    fifos: Vec<Fifo>,
    /// The next command buffer's place in the commit order.
    next_seq: u64,
    /// The progress value of the last batch submitted.
    progress: u64,
    /// The progress values of the submissions not yet seen complete, in order.
    in_flight: VecDeque<u64>,
    /// The batch staged and kept while [`IN_FLIGHT`] submissions are not seen complete.
    kept: Option<Batch>,
    /// Zero fills recorded and not yet submitted.
    upload: Option<Upload>,
    /// Something changed that may let a held segment go.
    dirty: bool,
    /// The upload is to go at once, alone where no segment is ready ([`Work::flush`]).
    flush: bool,
    /// The host waits for a command buffer committed: what is ready goes at once ([`Work::send`]).
    send: bool,
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

/// Commands recorded into one command buffer apart from any command buffer of the guest's (zero
/// fills, and later transfers), submitted ahead of everything in the next batch, which keeps what
/// they use until it retires.
pub(crate) struct Upload {
    pub(crate) recorder: Recorder,
    command: vk::CommandBuffer,
    held: Vec<Held>,
}

/// What an upload keeps until it retires.
pub(crate) enum Held {
    Block(Arc<Block>),
    /// A texture's image the upload moves to its home layout or copies bytes to or from.
    Texture(Arc<Image>),
    /// The staging buffer of such a copy.
    Staging(Arc<Staging>),
}

impl Held {
    /// Whether `self` and `other` are the same object.
    fn is(&self, other: &Held) -> bool {
        match (self, other) {
            (Held::Block(a), Held::Block(b)) => Arc::ptr_eq(a, b),
            (Held::Texture(a), Held::Texture(b)) => Arc::ptr_eq(a, b),
            (Held::Staging(a), Held::Staging(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
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
                in_flight: VecDeque::new(),
                kept: None,
                upload: None,
                dirty: false,
                flush: false,
                send: false,
                stop: false,
                closed: false,
                lost: false,
            }),
            changed: Condvar::new(),
            keeping: AtomicBool::new(false),
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

    /// Submissions completed: where the thread keeps a batch, it may go.
    pub(crate) fn completed(&self) {
        if self.keeping.load(Ordering::SeqCst) {
            self.poke();
        }
    }

    /// The host waits for a command buffer committed and not yet submitted: what is ready goes
    /// at once, a batch not full included.
    pub(crate) fn send(&self) {
        let mut state = self.lock();
        state.send = true;
        state.dirty = true;
        drop(state);
        self.wake();
    }

    /// Records into the pending upload with `record`, which is given the device and the upload's
    /// command buffer, recording, under the work lock: commands that go ahead of every later
    /// commit, on every queue. The upload keeps each of `held` until it retires.
    ///
    /// Err: [`Error::Quiesced`]; [`Error::Vulkan`] where a command pool or buffer cannot be made.
    pub(crate) fn upload(
        &self,
        core: &Core,
        held: impl IntoIterator<Item = Held>,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
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
                        held: Vec::new(),
                    },
                    Err(error) => {
                        core.recorders.give_back(core, recorder);
                        return Err(error);
                    }
                }
            }
        };
        let upload = state.upload.insert(upload);
        record(device, upload.command);
        for item in held {
            if !upload.held.iter().any(|kept| kept.is(&item)) {
                upload.held.push(item);
            }
        }
        Ok(())
    }

    /// Records the zero fill of `offset..end` of `block` into the upload.
    ///
    /// Err: as [`upload`](Self::upload).
    pub(crate) fn zero(
        &self,
        core: &Core,
        block: &Arc<Block>,
        offset: u64,
        end: u64,
    ) -> Result<(), Error> {
        self.upload(core, [Held::Block(Arc::clone(block))], |device, command| {
            // SAFETY: the upload's command buffer is recording, under the lock; the block's
            // buffer has every usage, and `offset..end` lies in it.
            unsafe {
                fill::record(
                    device,
                    command,
                    core.filler.zeros(),
                    block.memory.buffer,
                    offset,
                    end,
                )
            }
        })
    }

    /// Has the upload submitted at once, alone where no commit is ready: the progress value at
    /// which everything recorded into uploads so far has completed and is visible to the host,
    /// for [`Done::wait`](crate::complete::Done::wait).
    pub(crate) fn flush(&self) -> u64 {
        let mut state = self.lock();
        if state.upload.is_none() {
            // Every upload so far went in a batch already submitted.
            return state.progress;
        }
        state.flush = true;
        state.dirty = true;
        // The next batch submitted, the one kept included, takes the upload.
        let value = state.progress + 1;
        drop(state);
        self.wake();
        value
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

    /// The progress value of the last batch submitted.
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

    /// Takes the ready segments, in commit order, into the batch kept or a new one, until it ends
    /// [`GROUP`] command buffers; None where it holds nothing and no upload is to be flushed.
    fn stage(&mut self) -> Option<Batch> {
        let mut batch = self.kept.take().unwrap_or_else(|| Batch {
            value: self.progress + 1,
            upload: None,
            segments: Vec::new(),
            finished: Vec::new(),
            watch: Vec::new(),
        });
        while batch.finished.len() < GROUP {
            let Some(index) = self.next_ready() else {
                break;
            };
            let fifo = &mut self.fifos[index];
            let Some(Entry {
                pending: Some(pending),
                ..
            }) = fifo.entries.front_mut()
            else {
                break;
            };
            (batch.segments).push(stage(&pending.gpu.segments[pending.next], &mut batch.watch));
            pending.next += 1;
            pending.value = batch.value;
            if pending.next == pending.gpu.segments.len() {
                if let Some(Entry {
                    cb,
                    pending: Some(pending),
                }) = fifo.entries.pop_front()
                {
                    batch.finished.push((cb, pending));
                }
                if fifo.entries.is_empty() {
                    self.fifos.swap_remove(index);
                }
            }
        }
        if batch.segments.is_empty() && !(self.flush && self.upload.is_some()) {
            return None;
        }
        Some(batch)
    }

    /// Whether `batch` goes now: it is full, or something waits for it, or fewer than
    /// [`IN_FLIGHT`] submissions are not seen complete by `completed`.
    fn goes(&mut self, batch: &Batch, completed: u64) -> bool {
        while (self.in_flight.front()).is_some_and(|&value| value <= completed) {
            self.in_flight.pop_front();
        }
        batch.finished.len() >= GROUP
            || self.flush
            || self.send
            || self.stop
            || self.in_flight.len() < IN_FLIGHT
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
            // A kept batch never reached the GPU: its command buffers end as not in flight.
            let mut ended: Vec<Ended> = (state.kept.take().into_iter())
                .flat_map(|batch| batch.finished)
                .map(|(cb, mut pending)| {
                    pending.value = 0;
                    let error = CommandBufferError::DeviceLost { in_flight: false };
                    (cb, pending, error)
                })
                .collect();
            let submitted = state.progress;
            ended.extend(drain(&mut state, |pending| {
                CommandBufferError::DeviceLost {
                    in_flight: pending
                        .is_some_and(|pending| (1..=submitted).contains(&pending.value)),
                }
            }));
            let upload = state.upload.take();
            drop(state);
            if let Some(upload) = upload {
                upload.retire(&core);
            }
            core.done.end_all(&core, ended);
            state = work.lock();
        } else if let Some(mut batch) = state.stage() {
            // Marked before the completed value is read: a completion the read misses sees the
            // mark and wakes the thread.
            work.keeping.store(true, Ordering::SeqCst);
            if state.goes(&batch, core.done.completed()) {
                work.keeping.store(false, Ordering::SeqCst);
                if batch.finished.len() < GROUP {
                    // Everything ready goes in it: no wait is left to answer.
                    state.send = false;
                }
                state.flush = false;
                batch.upload = state.upload.take();
                state.progress = batch.value;
                state.in_flight.push_back(batch.value);
                drop(state);
                submit(&core, batch);
                state = work.lock();
                continue;
            }
            state.kept = Some(batch);
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
    // What the upload wrote is visible to every command after it, and to the host once the
    // batch completes.
    let mut ok = upload.as_ref().is_none_or(|upload| {
        record::barrier(device, upload.command, Reach::Host);
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
    /// Recycles the upload's recorder and gives up what it holds: once its submission completed,
    /// or where it never went.
    pub(crate) fn retire(self, core: &Core) {
        core.recorders.give_back(core, self.recorder);
    }
}
