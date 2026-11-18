//! The submit thread: each queue's FIFO of command buffers, the holds at event waits, the groups
//! held while a queue's earlier submissions run, the upload that goes ahead of later commits, and
//! the submissions it passes to `vkQueueSubmit2`.
//!
//! Each queue groups on its own. While fewer than [`IN_FLIGHT`] of the queue's submissions are
//! not seen complete, each of its command buffers goes at once, alone. While that many run, the
//! ones committed meanwhile are held in one group of at most [`GROUP`], which goes when it fills,
//! when a completion leaves fewer running (the completion thread wakes the thread as it retires),
//! when the host is to wait for one of them ([`Work::send`]), or at teardown. A segment that
//! signals an event ends its group, which goes at once; one that waits on a semaphore leads a new
//! group, the one held going first. A group is one submission with one signal of the progress
//! semaphore, which the completion thread follows in commit order.
//!
//! The upload is bounded: a call that finds it holding [`UPLOAD_ENTRIES`] calls, or
//! [`UPLOAD_BYTES`] of memory it alone keeps, has it submitted and waits for it to complete before
//! it records. A new texture's move to its home layout holds nothing until the next submission,
//! whose upload records it where the texture still lives. Copies stage their bytes in the
//! device's staging region ([`crate::staging`]), which the work lock guards with the upload; the
//! upload goes without a wait once its copies' spans take half the region.

use crate::buffer::DeviceCopy;
use crate::command::{CbInner, CommandBufferError, Handler, Pending, Segment};
use crate::complete;
use crate::counters::{Counters, Metric};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::event::EventInner;
use crate::fill;
use crate::memory::Block;
use crate::over::{self, Leans};
use crate::queue::QueueInner;
use crate::record::{self, Reach, Recorder};
use crate::staging::{At, Oldest, Region, Stage, Staging, REGION_LEN};
use crate::texture::{Image, OverCopy};
use crate::times;
use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};

/// The command buffers one submission ends at most.
const GROUP: usize = 4;

/// The submissions of a queue not yet seen complete at which its command buffers are held in a
/// group.
const IN_FLIGHT: usize = 2;

/// The calls the pending upload records at most: a call past them waits for it to complete.
pub(crate) const UPLOAD_ENTRIES: usize = 4096;

/// The memory the pending upload alone keeps, at most: its staging buffers, and the textures and
/// blocks released while it held them. A call past it waits for the upload to complete; one call
/// larger than it goes alone.
pub(crate) const UPLOAD_BYTES: u64 = 64 << 20;

/// The new textures awaiting their home layout at which the dead among them are first dropped.
pub(crate) const HOMES_PRUNE: usize = 64;

/// The submit side: the FIFOs and the upload, and the submit thread's wake-up.
pub(crate) struct Work {
    state: Mutex<WorkState>,
    changed: Condvar,
    /// A queue holds a group: the completion thread wakes the thread as submissions complete.
    keeping: AtomicBool,
    /// Bytes of textures and blocks released while an upload held them, since the pending upload
    /// began: they count against [`UPLOAD_BYTES`]. A texture's release adds without the work lock,
    /// which its drop may run under.
    released: AtomicU64,
}

pub(crate) struct WorkState {
    /// The queues with command buffers enqueued, committed, held or in flight.
    fifos: Vec<Fifo>,
    /// The next command buffer's place in the commit order.
    next_seq: u64,
    /// The next FIFO entry's place in its queue's order.
    next_place: u64,
    /// The progress value of the last submission.
    progress: u64,
    /// Zero fills, copies and moves to a home layout recorded and not yet submitted.
    upload: Option<Upload>,
    /// Textures made since the last submission, not held: the next submission's upload moves
    /// each that still lives to its home layout. While there are any, there is an upload.
    homes: Vec<Weak<Image>>,
    /// The length of `homes` at which the dead are dropped from it.
    prune_at: usize,
    /// The staging region, from the first copy on.
    region: Option<Region>,
    /// Something changed that may let a held segment or group go.
    dirty: bool,
    /// The upload is to go at once, alone where no group goes ([`Work::flush`]).
    flush: bool,
    /// The submit thread is to end: what is held in groups goes, then what is held at waits ends.
    stop: bool,
    /// Quiesced or torn down: commits and uploads are refused.
    closed: bool,
    lost: bool,
    /// The device's counters: the groups and submissions one queue would not have made.
    counters: Counters,
}

struct Fifo {
    queue: Arc<QueueInner>,
    entries: VecDeque<Entry>,
    /// The group staged and not yet submitted.
    group: Option<Batch>,
    /// The progress values of the queue's submissions not yet seen complete, in order.
    in_flight: VecDeque<u64>,
    /// A host wait is to come for the command buffer at this place: a group from it or before
    /// goes at once.
    send_through: u64,
    /// The progress value at which the last command buffer it ended with nothing held retires:
    /// the next retires no earlier.
    behind: u64,
}

struct Entry {
    cb: Arc<CbInner>,
    /// Its place in its queue's order.
    place: u64,
    /// None while enqueued and not committed.
    pending: Option<Pending>,
    /// Ended before it was submitted: it retires with this error behind its queue's earlier
    /// work.
    error: Option<CommandBufferError>,
}

/// Commands recorded into one command buffer apart from any command buffer of the guest's (zero
/// fills, texture copies, and new textures' moves to their home layouts), submitted ahead of
/// everything in the next submission, which keeps what they use until it retires.
pub(crate) struct Upload {
    pub(crate) recorder: Recorder,
    command: vk::CommandBuffer,
    held: Vec<Held>,
    /// The calls recorded, and one for the home layouts recorded as it is submitted.
    entries: usize,
    /// The bytes of the staging buffers of its own it holds.
    staging: u64,
    /// The staging region's spans its copies use, and their bytes.
    spans: Vec<u64>,
    span_bytes: u64,
    /// The Managed buffers whose consumer's bytes changed, in the order of their first change,
    /// and each one's place in it by its device copy's address.
    modified: Vec<Modified>,
    modified_at: HashMap<usize, usize>,
}

/// A Managed buffer's ranges the consumer changed since the last submission, which the upload
/// copies from the consumer's bytes into the device copy as it is submitted.
struct Modified {
    copy: Arc<DeviceCopy>,
    /// The block of the consumer's bytes, and the buffer's offset in it.
    cpu: Arc<Block>,
    offset: u64,
    /// `start..end` in the buffer, by start, no two overlapping or touching.
    ranges: Vec<(u64, u64)>,
}

/// What an upload keeps until it retires.
pub(crate) enum Held {
    Block(Arc<Block>),
    /// A texture's image the upload moves to its home layout or copies bytes to or from.
    Texture(Arc<Image>),
    /// The staging buffer of such a copy.
    Staging(Arc<Staging>),
    /// A Managed buffer's device copy the upload copies into.
    Copy(Arc<DeviceCopy>),
}

impl Held {
    /// Whether `self` and `other` are the same object.
    fn is(&self, other: &Held) -> bool {
        match (self, other) {
            (Held::Block(a), Held::Block(b)) => Arc::ptr_eq(a, b),
            (Held::Texture(a), Held::Texture(b)) => Arc::ptr_eq(a, b),
            (Held::Staging(a), Held::Staging(b)) => Arc::ptr_eq(a, b),
            (Held::Copy(a), Held::Copy(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Upload {
    /// A new upload, its command buffer recording.
    ///
    /// Err: [`Error::Vulkan`] where a command pool or buffer cannot be made.
    fn begin(core: &Core) -> Result<Upload, Error> {
        let device = &core.device;
        let mut recorder = core.recorders.take(device)?;
        match recorder.next(device) {
            Ok(command) => Ok(Upload {
                recorder,
                command,
                held: Vec::new(),
                entries: 0,
                staging: 0,
                spans: Vec::new(),
                span_bytes: 0,
                modified: Vec::new(),
                modified_at: HashMap::new(),
            }),
            Err(error) => {
                core.recorders.give_back(core, recorder);
                Err(error)
            }
        }
    }

    /// Whether a call of `staging` bytes waits for it: it holds a call, and it is at
    /// [`UPLOAD_ENTRIES`], or the call would take what it alone keeps (with `released`) past
    /// [`UPLOAD_BYTES`].
    fn full(&self, staging: u64, released: u64) -> bool {
        self.entries > 0
            && (self.entries >= UPLOAD_ENTRIES
                || self
                    .staging
                    .saturating_add(released)
                    .saturating_add(staging)
                    > UPLOAD_BYTES)
    }

    /// Records `moves`, images' moves from undefined to their home layouts
    /// ([`Image::homing`]), in one barrier; the upload holds each image.
    fn home(&self, device: &ash::Device, moves: &[vk::ImageMemoryBarrier2<'_>]) {
        if moves.is_empty() {
            return;
        }
        let dependency = vk::DependencyInfo::default().image_memory_barriers(moves);
        // SAFETY: the upload's command buffer is recording, under the work lock; each image is
        // the device's and held by the upload.
        unsafe { device.cmd_pipeline_barrier2(self.command, &dependency) };
    }
}

/// One queue's group: what one `vkQueueSubmit2` submits of it.
struct Batch {
    /// The place of the first command buffer with a segment in it.
    first: u64,
    segments: Vec<Staged>,
    /// Command buffers whose last segment is in it, and those that ended behind them, in the
    /// queue's order.
    finished: Vec<Finished>,
    /// The command buffers whose last segment is in it.
    count: usize,
    /// It holds segments of its queue's head, whose last is not yet staged.
    partial: bool,
    /// Events it signals that the completion thread is not yet watching.
    watch: Vec<Arc<EventInner>>,
    /// What its segments wrote of buffers under textures over a buffer in an image of their own
    /// (`crate::over`).
    leans: Leans,
    /// The copies into those textures' images its upload makes first: one for each texture a
    /// segment leans on the leading copy for.
    overs: Vec<OverCopy>,
}

/// One segment's submit info.
#[derive(Default)]
struct Staged {
    waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    command: Option<vk::CommandBufferSubmitInfo<'static>>,
    signals: Vec<vk::SemaphoreSubmitInfo<'static>>,
}

/// A command buffer of a submission: with an error where it ended before it was submitted.
pub(crate) type Finished = (Arc<CbInner>, Pending, Option<CommandBufferError>);

/// A command buffer that retires with no submission of its own: with how it ended, or with no
/// error where it had nothing to submit.
pub(crate) type Ended = Finished;

/// A command buffer that joined a group, with its scheduled handlers.
pub(crate) type Joined = (Arc<CbInner>, Vec<Handler>);

/// What staging the next ready segment did.
enum Step {
    /// Nothing is ready.
    Idle,
    /// Something was staged or ended; no group goes yet.
    Staged,
    /// The group of the FIFO at this index goes now.
    Goes(usize),
}

impl Work {
    pub(crate) fn new(counters: Counters) -> Work {
        Work {
            state: Mutex::new(WorkState {
                fifos: Vec::new(),
                next_seq: 0,
                next_place: 0,
                progress: 0,
                upload: None,
                homes: Vec::new(),
                prune_at: HOMES_PRUNE,
                region: None,
                dirty: false,
                flush: false,
                stop: false,
                closed: false,
                lost: false,
                counters,
            }),
            changed: Condvar::new(),
            keeping: AtomicBool::new(false),
            released: AtomicU64::new(0),
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

    /// Submissions completed: where a queue holds a group, it may go.
    pub(crate) fn completed(&self) {
        if self.keeping.load(Ordering::SeqCst) {
            self.poke();
        }
    }

    /// The host is to wait for `cb`: where it is committed and not yet submitted, the group it is
    /// in goes at once, or, where it is not yet staged, the group held ahead of it and the one it
    /// joins. Nothing for a command buffer submitted, ended or not committed. It takes the lock
    /// briefly and waits for no GPU work.
    pub(crate) fn send(&self, cb: &Arc<CbInner>) {
        let mut state = self.lock();
        let Some(fifo) = (state.fifos.iter_mut()).find(|fifo| fifo.queue.is(&cb.queue)) else {
            return;
        };
        let held = (fifo.group.as_ref()).and_then(|group| {
            (group.finished.iter())
                .any(|(joined, _, error)| error.is_none() && Arc::ptr_eq(joined, cb))
                .then_some(group.first)
        });
        let place = held.or_else(|| {
            (fifo.entries.iter())
                .find(|entry| {
                    Arc::ptr_eq(&entry.cb, cb) && entry.pending.is_some() && entry.error.is_none()
                })
                .map(|entry| entry.place)
        });
        let Some(place) = place.filter(|&place| place > fifo.send_through) else {
            return;
        };
        fifo.send_through = place;
        state.dirty = true;
        drop(state);
        self.wake();
    }

    /// Records into the pending upload with `record`, which is given the device and the upload's
    /// command buffer, recording, under the work lock: commands that go ahead of every later
    /// commit, on every queue. The upload keeps each of `held` until it retires, and `stage`, the
    /// staging of a copy [`stage`](Self::stage) gave: its span of the region until the
    /// submission completes, or its own staging buffer. A texture of `held` not yet in its home
    /// layout is moved there first.
    ///
    /// Where the upload is full ([`UPLOAD_ENTRIES`], [`UPLOAD_BYTES`]), it is submitted and the
    /// call waits for it to complete first, without the lock. Where its spans of the staging
    /// region reach half the region, it is submitted, and the call does not wait. Where the call
    /// fails, `stage`'s span is given up.
    ///
    /// Err: [`Error::Quiesced`]; [`Error::Vulkan`] where a command pool or buffer cannot be made,
    /// and with `VK_ERROR_DEVICE_LOST` where the device is lost while the call waits.
    pub(crate) fn upload<const N: usize>(
        &self,
        core: &Core,
        stage: Option<&Stage>,
        held: [Held; N],
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), Error> {
        let recorded = self.record_upload(core, stage, held, record);
        if recorded.is_err() {
            if let Some(Stage::Region { number, .. }) = stage {
                self.give_up(*number);
            }
        }
        recorded
    }

    /// The work lock, once the pending upload has room for a call of `staging` bytes of staging of
    /// its own: where it is full ([`UPLOAD_ENTRIES`], [`UPLOAD_BYTES`]), it is submitted and the
    /// call waits for it to complete, without the lock.
    ///
    /// Err: [`Error::Quiesced`]; [`Error::Vulkan`] with `VK_ERROR_DEVICE_LOST` where the device is
    /// lost while the call waits.
    fn room(&self, core: &Core, staging: u64) -> Result<MutexGuard<'_, WorkState>, Error> {
        let mut state = self.lock();
        loop {
            if state.closed {
                return Err(Error::Quiesced);
            }
            let released = self.released.load(Ordering::Relaxed);
            if !(state.upload.as_ref()).is_some_and(|upload| upload.full(staging, released)) {
                return Ok(state);
            }
            let value = state.flush();
            drop(state);
            self.wake();
            core.counters.count(Metric::UploadWaits);
            core.done.wait(core, value)?;
            state = self.lock();
        }
    }

    /// `didModifyRange:` of a Managed buffer with a device copy: the consumer's bytes
    /// `start..end` of the buffer, at `offset` in `cpu`, changed. The next submission's upload
    /// copies them into `copy`, after everything recorded into it before, in one copy command per
    /// buffer, whose ranges are merged where they overlap or touch; the copy reads the bytes when
    /// the GPU reaches it. Each call is one of the upload's calls, bounded as
    /// [`upload`](Self::upload)'s are.
    ///
    /// Err: as [`upload`](Self::upload).
    pub(crate) fn modify(
        &self,
        core: &Core,
        copy: &Arc<DeviceCopy>,
        cpu: &Arc<Block>,
        offset: u64,
        start: u64,
        end: u64,
    ) -> Result<(), Error> {
        let mut state = self.room(core, 0)?;
        let upload = match state.upload.take() {
            Some(upload) => upload,
            None => Upload::begin(core)?,
        };
        let upload = state.upload.insert(upload);
        upload.entries += 1;
        let key = Arc::as_ptr(copy) as usize;
        match upload.modified_at.get(&key) {
            Some(&index) => merge(&mut upload.modified[index].ranges, start, end),
            None => {
                upload.modified_at.insert(key, upload.modified.len());
                upload.modified.push(Modified {
                    copy: Arc::clone(copy),
                    cpu: Arc::clone(cpu),
                    offset,
                    ranges: vec![(start, end)],
                });
            }
        }
        Ok(())
    }

    /// [`upload`](Self::upload) up to the span's give-up on a failure.
    fn record_upload<const N: usize>(
        &self,
        core: &Core,
        stage: Option<&Stage>,
        held: [Held; N],
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), Error> {
        let staging = match stage {
            Some(Stage::Own(staging)) => staging.len,
            _ => 0,
        };
        let mut state = self.room(core, staging)?;
        let device = &core.device;
        let upload = match state.upload.take() {
            Some(upload) => upload,
            None => Upload::begin(core)?,
        };
        let upload = state.upload.insert(upload);
        let moves: Vec<vk::ImageMemoryBarrier2<'static>> = (held.iter())
            .filter_map(|item| match item {
                Held::Texture(image) => image.homing(),
                _ => None,
            })
            .collect();
        upload.home(device, &moves);
        record(device, upload.command);
        upload.entries += 1;
        upload.staging += staging;
        for item in held {
            if !upload.held.iter().any(|kept| kept.is(&item)) {
                upload.held.push(item);
            }
        }
        let mut half = false;
        match stage {
            Some(Stage::Region { number, len, .. }) => {
                upload.spans.push(*number);
                upload.span_bytes += len;
                half = upload.span_bytes >= REGION_LEN / 2;
                if let Some(region) = &mut state.region {
                    region.ring.set(*number, At::Recorded);
                }
            }
            Some(Stage::Own(staging)) => upload.held.push(Held::Staging(Arc::clone(staging))),
            None => {}
        }
        if half {
            state.flush();
            drop(state);
            self.wake();
        }
        Ok(())
    }

    /// Staging for a copy of `len` bytes: a span of the device's staging region, made at the first
    /// copy; a staging buffer of its own for a copy larger than the region. Where the region has
    /// no room, the call waits, without the lock, for what frees its oldest span: the submission that carries it, after submitting the pending upload
    /// where it is in that. `read` keeps the span until [`unread`](Self::unread), for a copy the
    /// host reads once it completed. The caller passes the stage to [`upload`](Self::upload).
    ///
    /// Err: [`Error::Quiesced`]; what making the region or the staging buffer can return;
    /// [`Error::Vulkan`] with `VK_ERROR_DEVICE_LOST` where the device is lost while the call
    /// waits.
    pub(crate) fn stage(&self, core: &Arc<Core>, len: u64, read: bool) -> Result<Stage, Error> {
        if len > REGION_LEN {
            if self.lock().closed {
                return Err(Error::Quiesced);
            }
            return Ok(Stage::Own(Arc::new(Staging::new(core, len)?)));
        }
        let mut state = self.lock();
        if state.region.is_none() {
            drop(state);
            let made = Region::new(core)?;
            state = self.lock();
            if state.region.is_none() {
                state.region = Some(made);
            } else {
                // SAFETY: another call made the region first; nothing used this one.
                unsafe { made.free(core) };
            }
        }
        // The progress value known complete: what the completion thread has seen, or what this
        // call waited for.
        let mut completed = core.done.completed();
        loop {
            if state.closed {
                return Err(Error::Quiesced);
            }
            // Made above, and taken only at teardown.
            let Some(region) = &mut state.region else {
                return Err(Error::Quiesced);
            };
            region.ring.reclaim(completed.max(core.done.completed()));
            if let Some((number, offset)) = region.ring.take(len, read) {
                return Ok(Stage::Region {
                    number,
                    len,
                    buffer: region.buffer,
                    offset,
                    host: region.host(offset),
                });
            }
            let value = match region.ring.oldest() {
                Some(Oldest::Submitted(value)) => value,
                Some(Oldest::Pending) => state.flush(),
                // Another call finishes writing or reading its span.
                Some(Oldest::Busy) | None => {
                    drop(state);
                    std::thread::yield_now();
                    state = self.lock();
                    continue;
                }
            };
            drop(state);
            self.wake();
            core.counters.count(Metric::UploadWaits);
            core.done.wait(core, value)?;
            completed = completed.max(value);
            state = self.lock();
        }
    }

    /// The host no longer reads span `number` of the staging region.
    pub(crate) fn unread(&self, number: u64) {
        if let Some(region) = &mut self.lock().region {
            region.ring.unread(number);
        }
    }

    /// Gives up span `number` of the staging region, with no copy recorded.
    fn give_up(&self, number: u64) {
        if let Some(region) = &mut self.lock().region {
            region.ring.set(number, At::Done);
            region.ring.unread(number);
        }
    }

    /// At teardown: the staging region, for the core to free.
    pub(crate) fn take_region(&mut self) -> Option<Region> {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        state.region.take()
    }

    /// A new texture's `image`: the next submission's upload moves it to its home layout where it
    /// still lives, holding it from then; until then nothing holds it. An upload is begun now if
    /// none is pending, so that the submission has one to record into.
    ///
    /// Err: [`Error::Quiesced`]; [`Error::Vulkan`] where a command pool or buffer cannot be made.
    pub(crate) fn home(&self, core: &Core, image: &Arc<Image>) -> Result<(), Error> {
        let mut state = self.lock();
        if state.closed {
            return Err(Error::Quiesced);
        }
        if state.upload.is_none() {
            state.upload = Some(Upload::begin(core)?);
        }
        if state.homes.len() >= state.prune_at {
            state.homes.retain(|home| home.strong_count() > 0);
            state.prune_at = (state.homes.len() * 2).max(HOMES_PRUNE);
        }
        state.homes.push(Arc::downgrade(image));
        Ok(())
    }

    /// A texture whose image an upload holds is released: its `bytes` count against the pending
    /// upload's [`UPLOAD_BYTES`]. An upload already submitted frees it soon; counting it anyway
    /// can only submit the pending one sooner.
    pub(crate) fn released(&self, bytes: u64) {
        self.released.fetch_add(bytes, Ordering::Relaxed);
    }

    /// The consumer released `block`: where the pending upload holds it, its length counts
    /// against the upload's [`UPLOAD_BYTES`].
    pub(crate) fn release_block(&self, block: &Arc<Block>) {
        let holds = (self.lock().upload.as_ref()).is_some_and(|upload| {
            (upload.held.iter())
                .any(|kept| matches!(kept, Held::Block(kept) if Arc::ptr_eq(kept, block)))
        });
        if holds {
            self.released(block.memory.len);
        }
    }

    /// For the device tests: the pending upload's calls, the bytes it alone keeps, and the new
    /// textures awaiting their home layout.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn pending(&self) -> (usize, u64, usize) {
        let state = self.lock();
        let released = self.released.load(Ordering::Relaxed);
        match &state.upload {
            Some(upload) => (upload.entries, upload.staging + released, state.homes.len()),
            None => (0, released, state.homes.len()),
        }
    }

    /// For the device tests: the staging region's live spans, oldest first; none before the
    /// region is made.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn staging_spans(&self) -> Vec<(u64, u64, crate::staging::At)> {
        let state = self.lock();
        (state.region.as_ref()).map_or_else(Vec::new, |region| region.ring.live())
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
        self.upload(
            core,
            None,
            [Held::Block(Arc::clone(block))],
            |device, command| {
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
            },
        )
    }

    /// Has the upload submitted at once, alone where no group goes: the progress value at which
    /// everything recorded into uploads so far has completed and is visible to the host, for
    /// [`Done::wait`](crate::complete::Done::wait).
    pub(crate) fn flush(&self) -> u64 {
        let value = self.lock().flush();
        self.wake();
        value
    }

    /// Ends `queue`'s command buffers not yet in a submission or a group with `NotRun`. A group
    /// it holds still goes: its command buffers are scheduled.
    pub(crate) fn abandon(&self, core: &Core, queue: &QueueInner) {
        let ended = {
            let mut state = self.lock();
            match state.fifos.iter_mut().find(|fifo| fifo.queue.is(queue)) {
                Some(fifo) => {
                    let entries: Vec<Entry> = fifo.entries.drain(..).collect();
                    let mut ended = Vec::new();
                    for entry in entries {
                        fifo.end(entry, CommandBufferError::NotRun, &mut ended);
                    }
                    ended
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

    /// The progress value of the last submission.
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

    /// Has the upload submitted at once, alone where no group goes: the progress value at which
    /// everything recorded into uploads so far has completed. The caller wakes the thread.
    fn flush(&mut self) -> u64 {
        if self.upload.is_none() {
            // Every upload so far went in a submission already made.
            return self.progress;
        }
        self.flush = true;
        self.dirty = true;
        // The next submission takes the upload.
        self.progress + 1
    }

    /// Takes the pending upload for the submission of progress `value`, with the moves to their
    /// home layouts of the textures made since the last that still live, which it holds from
    /// then; its spans of the staging region are free once `value` completes. `released` starts
    /// again for the next upload.
    fn take_upload(
        &mut self,
        device: &ash::Device,
        released: &AtomicU64,
        value: u64,
    ) -> Option<Upload> {
        let mut upload = self.upload.take()?;
        released.store(0, Ordering::Relaxed);
        if let Some(region) = &mut self.region {
            for &number in &upload.spans {
                region.ring.set(number, At::Submitted(value));
            }
        }
        self.prune_at = HOMES_PRUNE;
        let mut moves = Vec::new();
        for image in self.homes.drain(..).filter_map(|home| home.upgrade()) {
            // One the upload copied to or from was moved before its copy, and is held.
            if let Some(barrier) = image.homing() {
                moves.push(barrier);
                upload.held.push(Held::Texture(image));
            }
        }
        if !moves.is_empty() {
            upload.home(device, &moves);
            upload.entries += 1;
        }
        Some(upload)
    }

    /// Takes the pending upload where it will never be submitted: its spans of the staging region
    /// are given up, and the textures awaiting their home layout need none.
    fn drop_upload(&mut self) -> Option<Upload> {
        self.homes.clear();
        let upload = self.upload.take()?;
        if let Some(region) = &mut self.region {
            for &number in &upload.spans {
                region.ring.set(number, At::Done);
            }
        }
        Some(upload)
    }

    fn fifo(&mut self, queue: &Arc<QueueInner>) -> &mut Fifo {
        let index = match self.fifos.iter().position(|fifo| fifo.queue.is(queue)) {
            Some(index) => index,
            None => {
                self.fifos.push(Fifo {
                    queue: Arc::clone(queue),
                    entries: VecDeque::new(),
                    group: None,
                    in_flight: VecDeque::new(),
                    send_through: 0,
                    behind: 0,
                });
                self.fifos.len() - 1
            }
        };
        &mut self.fifos[index]
    }

    /// Takes `cb`'s place at its queue's end.
    fn push(&mut self, cb: &Arc<CbInner>) -> &mut Entry {
        self.next_place += 1;
        let place = self.next_place;
        let entries = &mut self.fifo(&cb.queue).entries;
        entries.push_back(Entry {
            cb: Arc::clone(cb),
            place,
            pending: None,
            error: None,
        });
        entries.back_mut().expect("an entry just pushed")
    }

    /// Takes `cb`'s place in its queue's FIFO.
    pub(crate) fn enqueue(&mut self, cb: &Arc<CbInner>) {
        self.push(cb);
    }

    /// Hands `cb` to the submit thread: at its place where it was enqueued, else at its queue's
    /// end. With `error` (it ended while enqueued, and its place went), at its queue's end, to
    /// retire with that error behind the work held or in flight before it.
    pub(crate) fn commit(
        &mut self,
        cb: &Arc<CbInner>,
        pending: Pending,
        error: Option<CommandBufferError>,
    ) {
        self.dirty = true;
        let enqueued = (self.fifo(&cb.queue).entries.iter_mut())
            .find(|entry| Arc::ptr_eq(&entry.cb, cb) && entry.pending.is_none());
        let entry = match enqueued {
            Some(entry) => entry,
            None => self.push(cb),
        };
        entry.pending = Some(pending);
        entry.error = error;
    }

    /// The FIFO whose head goes next: of the committed heads that are ended or whose next segment
    /// is ready, the earliest committed.
    fn next_ready(&self) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;
        for (index, fifo) in self.fifos.iter().enumerate() {
            let Some(Entry {
                pending: Some(pending),
                error,
                ..
            }) = fifo.entries.front()
            else {
                continue;
            };
            if best.is_some_and(|(_, seq)| seq < pending.seq) {
                continue;
            }
            if error.is_some() || ready(&pending.gpu.segments[pending.next]) {
                best = Some((index, pending.seq));
            }
        }
        best.map(|(index, _)| index)
    }

    /// Forgets the submissions seen complete by `completed`, and the FIFOs left with nothing.
    fn settle(&mut self, completed: u64) {
        self.fifos.retain_mut(|fifo| {
            while (fifo.in_flight.front()).is_some_and(|&value| value <= completed) {
                fifo.in_flight.pop_front();
            }
            !(fifo.entries.is_empty() && fifo.group.is_none() && fifo.in_flight.is_empty())
        });
    }

    /// Stages the next ready segment, in commit order, into its queue's group, or ends the next
    /// ended command buffer behind its queue's work. A command buffer whose last segment joins a
    /// group is scheduled at `now`, its scheduled handlers going to `joined`; one ended with
    /// nothing held goes to `ended`.
    fn step(&mut self, now: u64, joined: &mut Vec<Joined>, ended: &mut Vec<Ended>) -> Step {
        let Some(index) = self.next_ready() else {
            return Step::Idle;
        };
        let stop = self.stop;
        let others_hold = (self.fifos.iter().enumerate())
            .any(|(other, fifo)| other != index && fifo.group.is_some());
        let fifo = &mut self.fifos[index];
        let Some(entry) = fifo.entries.front_mut() else {
            return Step::Idle;
        };
        if let Some(error) = entry.error {
            if let Some(entry) = fifo.entries.pop_front() {
                fifo.end(entry, error, ended);
            }
            return Step::Staged;
        }
        let Some(pending) = entry.pending.as_mut() else {
            return Step::Idle;
        };
        // A command buffer with nothing for the GPU takes no submission: it is scheduled now and
        // completes with its queue's work before it.
        if pending.gpu.is_empty() {
            if let Some(Entry {
                cb,
                pending: Some(mut pending),
                ..
            }) = fifo.entries.pop_front()
            {
                cb.scheduled(now);
                let scheduled = std::mem::take(&mut pending.handlers.scheduled);
                if !scheduled.is_empty() {
                    joined.push((Arc::clone(&cb), scheduled));
                }
                fifo.behind_held(cb, pending, None, ended);
            }
            return Step::Staged;
        }
        let segment = &pending.gpu.segments[pending.next];
        // A segment that waits on a semaphore leads its group: a driver that merges a submission's
        // entries gathers their waits ahead of the first one's commands.
        let waits = (segment.waits.iter()).any(|(event, value)| lock(&event.state).floor < *value);
        if waits && fifo.group.is_some() {
            return Step::Goes(index);
        }
        // A segment that leans on the leading copy for a buffer an earlier segment of the group
        // wrote leads its group, so its copy follows the write (`crate::over`).
        if (fifo.group.as_ref()).is_some_and(|group| group.leans.conflicts(&segment.overs)) {
            return Step::Goes(index);
        }
        let place = entry.place;
        if fifo.group.is_none() && others_hold {
            self.counters.count(Metric::GroupsBesideAnotherQueue);
        }
        let group = fifo.group.get_or_insert_with(|| Batch {
            first: place,
            segments: Vec::new(),
            finished: Vec::new(),
            count: 0,
            partial: false,
            watch: Vec::new(),
            leans: Leans::default(),
            overs: Vec::new(),
        });
        group.leans.join(&segment.overs);
        for texture in segment.overs.leaning() {
            if let Some((_, copy)) = texture.copied_over() {
                if !group.overs.iter().any(|kept| kept.image == copy.image) {
                    group.overs.push(copy);
                }
            }
        }
        let staged = stage(segment, &mut group.watch);
        // An event's signal goes at once: nothing waits on it behind a held group.
        let signals = !staged.signals.is_empty();
        group.segments.push(staged);
        pending.next += 1;
        if pending.next < pending.gpu.segments.len() {
            group.partial = true;
            return if signals {
                Step::Goes(index)
            } else {
                Step::Staged
            };
        }
        let Some(Entry {
            cb,
            pending: Some(mut pending),
            ..
        }) = fifo.entries.pop_front()
        else {
            return Step::Staged;
        };
        cb.scheduled(now);
        let scheduled = std::mem::take(&mut pending.handlers.scheduled);
        if !scheduled.is_empty() {
            joined.push((Arc::clone(&cb), scheduled));
        }
        group.finished.push((cb, pending, None));
        group.partial = false;
        group.count += 1;
        let goes = signals
            || stop
            || group.count >= GROUP
            || fifo.in_flight.len() < IN_FLIGHT
            || fifo.send_through >= group.first;
        match goes {
            true => Step::Goes(index),
            false => Step::Staged,
        }
    }

    /// The FIFO at `index`'s group, given the next progress value and the upload
    /// ([`take_upload`](Self::take_upload)): the submission it makes.
    fn close(
        &mut self,
        index: usize,
        device: &ash::Device,
        released: &AtomicU64,
    ) -> Option<(u64, Option<Upload>, Batch)> {
        let across: usize = (self.fifos.iter()).map(|fifo| fifo.in_flight.len()).sum();
        let fifo = &mut self.fifos[index];
        let mut batch = fifo.group.take()?;
        let own = fifo.in_flight.len();
        if own < IN_FLIGHT && across >= IN_FLIGHT {
            self.counters.count(Metric::LoneSubmissionsAcrossQueues);
        }
        // Each reason `step` and `unheld` send a group for, all of its own queue's.
        let leads = (fifo.entries.front())
            .and_then(|entry| entry.pending.as_ref())
            .and_then(|pending| pending.gpu.segments.get(pending.next))
            .is_some_and(|segment| {
                (segment.waits.iter()).any(|(event, value)| lock(&event.state).floor < *value)
            });
        let own_reason = self.stop
            || own < IN_FLIGHT
            || fifo.send_through >= batch.first
            || batch.count >= GROUP
            || (batch.segments.iter()).any(|segment| !segment.signals.is_empty())
            || leads;
        if !own_reason {
            self.counters.count(Metric::ForcedFlushesAcrossQueues);
        }
        self.progress += 1;
        let value = self.progress;
        fifo.in_flight.push_back(value);
        if batch.partial {
            if let Some(Entry {
                pending: Some(pending),
                ..
            }) = fifo.entries.front_mut()
            {
                pending.value = value;
            }
        }
        for (_, pending, _) in &mut batch.finished {
            pending.value = value;
        }
        self.flush = false;
        Some((value, self.take_upload(device, released, value), batch))
    }

    /// The first FIFO whose group goes at the end of a pass: unless its queue has [`IN_FLIGHT`]
    /// submissions not seen complete and no host wait is to come for it.
    fn unheld(&self) -> Option<usize> {
        self.fifos.iter().position(|fifo| {
            fifo.group.as_ref().is_some_and(|group| {
                self.stop || fifo.in_flight.len() < IN_FLIGHT || fifo.send_through >= group.first
            })
        })
    }
}

impl Fifo {
    /// Ends `entry` with `error` (its own, where it has one): behind the group held, which it
    /// joins to retire with; else once the queue's submissions in flight and the command buffers
    /// it ended before complete, at once where none is. One not committed ends at its commit.
    fn end(&mut self, entry: Entry, error: CommandBufferError, ended: &mut Vec<Ended>) {
        let Entry {
            cb,
            pending,
            error: own,
            ..
        } = entry;
        let error = own.unwrap_or(error);
        let Some(pending) = pending else {
            cb.end_uncommitted(error);
            return;
        };
        if let Some(group) = &mut self.group {
            // The head whose segments the group holds is ended now, if it was the head.
            group.partial = false;
        }
        self.behind_held(cb, pending, Some(error), ended);
    }

    /// Retires a committed command buffer with no submission of its own: behind the group held,
    /// which it joins to retire with; else once the queue's submissions in flight and the command
    /// buffers it ended before complete, at once where none is.
    fn behind_held(
        &mut self,
        cb: Arc<CbInner>,
        mut pending: Pending,
        error: Option<CommandBufferError>,
        ended: &mut Vec<Ended>,
    ) {
        match &mut self.group {
            Some(group) => group.finished.push((cb, pending, error)),
            None => {
                let in_flight = self.in_flight.back().copied().unwrap_or(0);
                pending.value = pending.value.max(in_flight).max(self.behind);
                self.behind = pending.value;
                ended.push((cb, pending, error));
            }
        }
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
/// for each without its own, with its pending part where it is committed.
fn end(
    entries: VecDeque<Entry>,
    error: impl Fn(Option<&Pending>) -> CommandBufferError,
) -> Vec<Ended> {
    let mut ended = Vec::new();
    for Entry {
        cb,
        pending,
        error: own,
        ..
    } in entries
    {
        let error = own.unwrap_or_else(|| error(pending.as_ref()));
        match pending {
            Some(pending) => ended.push((cb, pending, Some(error))),
            None => cb.end_uncommitted(error),
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
            let submitted = state.progress;
            let error = |pending: Option<&Pending>| CommandBufferError::DeviceLost {
                in_flight: pending.is_some_and(|pending| (1..=submitted).contains(&pending.value)),
            };
            let mut ended = Vec::new();
            for fifo in state.fifos.drain(..) {
                // A group held never reached the GPU: its command buffers end as in flight only
                // where an earlier segment of theirs did.
                for (cb, pending, own) in fifo.group.into_iter().flat_map(|group| group.finished) {
                    let error = own.unwrap_or_else(|| error(Some(&pending)));
                    ended.push((cb, pending, Some(error)));
                }
                ended.extend(end(fifo.entries, error));
            }
            // Nothing more runs: the textures awaiting their home layout need none.
            let upload = state.drop_upload();
            drop(state);
            if let Some(upload) = upload {
                upload.retire(&core);
            }
            core.done.end_all(&core, ended);
            state = work.lock();
        } else {
            state = pass(&core, state);
        }
        if state.stop {
            let ended: Vec<Ended> = (state.fifos.drain(..))
                .flat_map(|fifo| end(fifo.entries, |_| CommandBufferError::NotRun))
                .collect();
            let upload = state.drop_upload();
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

/// One pass of the submit thread: stages what is ready in commit order, submits each group that
/// goes as it closes, then each group its queue no longer holds, then an upload to flush alone.
fn pass<'a>(core: &'a Core, mut state: MutexGuard<'a, WorkState>) -> MutexGuard<'a, WorkState> {
    let work = &core.work;
    // Marked before the completed value is read: a completion the read misses sees the mark and
    // wakes the thread.
    work.keeping.store(true, Ordering::SeqCst);
    state.settle(core.done.completed());
    let mut joined: Vec<Joined> = Vec::new();
    let mut ended: Vec<Ended> = Vec::new();
    let mut sent = false;
    loop {
        let index = match state.step(times::host_time(), &mut joined, &mut ended) {
            Step::Idle => match state.unheld() {
                Some(index) => index,
                None => break,
            },
            Step::Staged => continue,
            Step::Goes(index) => index,
        };
        let Some((value, upload, batch)) = state.close(index, &core.device, &work.released) else {
            continue;
        };
        drop(state);
        // Scheduled handlers reach the completion thread before the submission they are in.
        core.done.scheduled(core, std::mem::take(&mut joined));
        core.done.end_all(core, std::mem::take(&mut ended));
        submit(core, value, upload, batch);
        sent = true;
        state = work.lock();
    }
    if state.flush && !sent {
        let value = state.progress + 1;
        if let Some(upload) = state.take_upload(&core.device, &work.released, value) {
            state.flush = false;
            state.progress = value;
            drop(state);
            submit(core, value, Some(upload), Batch::empty());
            state = work.lock();
        }
    }
    if state.fifos.iter().all(|fifo| fifo.group.is_none()) {
        work.keeping.store(false, Ordering::SeqCst);
    }
    if !joined.is_empty() || !ended.is_empty() {
        drop(state);
        core.done.scheduled(core, joined);
        core.done.end_all(core, ended);
        state = work.lock();
    }
    state
}

impl Batch {
    /// A submission of no command buffer's: an upload alone.
    fn empty() -> Batch {
        Batch {
            first: 0,
            segments: Vec::new(),
            finished: Vec::new(),
            count: 0,
            partial: false,
            watch: Vec::new(),
            leans: Leans::default(),
            overs: Vec::new(),
        }
    }
}

/// Submits a group: the upload first, then each segment in order, the last signalling the
/// progress `value`. Whatever was passed to `vkQueueSubmit2` goes to the completion thread as in
/// flight; a failed submission is taken as a loss.
fn submit(core: &Core, value: u64, upload: Option<Upload>, batch: Batch) {
    let Batch {
        mut segments,
        finished,
        watch,
        overs,
        ..
    } = batch;
    let device = &core.device;
    // An upload begun for textures gone before their move records nothing; one is begun for the
    // copies into textures over a buffer in an image of their own.
    let (mut upload, mut ok) = match upload {
        Some(upload) if upload.entries == 0 && overs.is_empty() => {
            upload.retire(core);
            (None, true)
        }
        None if !overs.is_empty() => match Upload::begin(core) {
            Ok(upload) => (Some(upload), true),
            Err(_) => (None, false),
        },
        upload => (upload, true),
    };
    // What the upload wrote is visible to every command after it, and to the host once the
    // submission completes.
    ok &= upload.as_mut().is_none_or(|upload| {
        upload.copy_modified(core);
        // Each texture over a buffer in an image of its own that a segment leans on takes the
        // buffer's bytes (`crate::over`).
        // SAFETY: the upload's command buffer is recording, and only this thread has it; the
        // command buffers of the submission hold each texture, which holds its buffer, until it
        // retires, and each image is in its home layout, any move there recorded before.
        unsafe { over::record_side(device, upload.command, &overs, true, &core.counters) };
        record::barrier(device, upload.command, Reach::Host);
        core.counters.count(Metric::Barriers);
        // SAFETY: the upload's command buffer was recording under the work lock, which `close`
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
    core.done.submitted(core, value, upload, finished, watch);
}

impl Upload {
    /// Records the copies of the Managed buffers' changed ranges, last, after a barrier that puts
    /// everything recorded before them (a device copy's zero fill among it) first; the upload
    /// holds both blocks of each until it retires. The upload is out of the work lock, on its way
    /// to its submission.
    fn copy_modified(&mut self, core: &Core) {
        if self.modified.is_empty() {
            return;
        }
        let device = &core.device;
        record::barrier(device, self.command, Reach::Commands);
        core.counters.count(Metric::Barriers);
        for modified in self.modified.drain(..) {
            let regions: Vec<vk::BufferCopy> = (modified.ranges.iter())
                .map(|&(start, end)| vk::BufferCopy {
                    src_offset: modified.offset + start,
                    dst_offset: modified.copy.offset() + start,
                    size: end - start,
                })
                .collect();
            // SAFETY: the upload's command buffer is recording, and only this thread has it; both
            // blocks' buffers have every usage, each range lies in the buffer, and the upload holds
            // both blocks until it retires.
            unsafe {
                device.cmd_copy_buffer(
                    self.command,
                    modified.cpu.memory.buffer,
                    modified.copy.buffer(),
                    &regions,
                )
            };
            self.held.push(Held::Block(modified.cpu));
            self.held.push(Held::Copy(modified.copy));
        }
        self.modified_at.clear();
    }

    /// Recycles the upload's recorder and gives up what it holds: once its submission completed,
    /// or where it never went.
    pub(crate) fn retire(self, core: &Core) {
        core.recorders.give_back(core, self.recorder);
    }
}

/// Adds `start..end` to `ranges` (by start, no two overlapping or touching), joining the ranges it
/// overlaps or touches.
pub(crate) fn merge(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    // The first range that ends at or after `start`, and the first that starts after `end`.
    let first = ranges.partition_point(|&(_, stop)| stop < start);
    let last = ranges.partition_point(|&(begin, _)| begin <= end);
    if first == last {
        ranges.insert(first, (start, end));
        return;
    }
    let joined = (start.min(ranges[first].0), end.max(ranges[last - 1].1));
    ranges.drain(first + 1..last);
    ranges[first] = joined;
}
