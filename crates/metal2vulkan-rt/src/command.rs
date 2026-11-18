//! Command buffers: recorded at once into segments that event waits and signals split, and their
//! states from creation to retirement.

use crate::buffer::{Buffer, DeviceCopy};
use crate::counters::Metric;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Refusal};
use crate::event::{Event, EventInner};
use crate::hazard::{Access, Hazards};
use crate::memory::Block;
use crate::over::{self, Overs};
use crate::pipeline::ComputePipelineState;
use crate::queue::QueueInner;
use crate::readback;
use crate::record::{self, Reach, Recorder};
use crate::ring::Writes;
use crate::sampler::SamplerState;
use crate::texture::Texture;
use crate::times::{self, Stamps, Times};
use ash::vk;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// `MTLCommandBuffer`.
///
/// Each call records at once; there is no replay log. Event waits and signals split it into
/// segments, one `VkCommandBuffer` each where it records commands, submitted with the semaphore
/// operations between them. The submit thread holds a segment whose waits are not yet met, and its
/// queue's later command buffers behind it, while the segments before it run: an event the held
/// part waits for can be signalled by the part before it. Each segment's commands end with one
/// barrier that makes their writes visible to every later command and to the host.
///
/// Its GPU times come from two timestamps, written where its first commands begin and where its
/// last segment ends, and read once it completes. A command buffer with no commands, or whose
/// last segment has none, records a `VkCommandBuffer` for them.
///
/// A cheap handle: clones share one command buffer. It holds the device until it is committed and
/// nothing of it after, so a command buffer held forever on an event does not keep the device
/// alive: teardown ends it with [`CommandBufferError::NotRun`].
#[derive(Clone)]
pub struct CommandBuffer {
    inner: Arc<CbInner>,
}

/// `MTLCommandBufferStatus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandBufferStatus {
    /// `MTLCommandBufferStatusNotEnqueued`.
    NotEnqueued,
    /// `MTLCommandBufferStatusEnqueued`.
    Enqueued,
    /// `MTLCommandBufferStatusCommitted`: also while the submit thread holds a later segment.
    Committed,
    /// `MTLCommandBufferStatusScheduled`: the last segment is in a submission: passed to
    /// `vkQueueSubmit2`, or in a group its queue holds while two of its submissions run, which
    /// goes with no other call.
    Scheduled,
    /// `MTLCommandBufferStatusCompleted`.
    Completed,
    /// `MTLCommandBufferStatusError`, with [`CommandBuffer::error`].
    Error,
}

/// `MTLCommandBufferError`, the part this runtime reports.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandBufferError {
    /// The `VkDevice` was lost, or a submission failed, which the runtime takes as a loss. With
    /// one device, a fault anywhere ends every queue's work. `in_flight`: some of the command
    /// buffer had passed `vkQueueSubmit2` (Metal's hang or victim); false for one held or not yet
    /// submitted, and for every commit after the loss.
    DeviceLost {
        /// Whether part of the command buffer had been submitted.
        in_flight: bool,
    },
    /// Ended without its held part running: by [`CommandQueue::abandon`](crate::CommandQueue::abandon),
    /// by [`Device::quiesce`], or at teardown. Segments before the held one may have run.
    NotRun,
}

/// A scheduled or completed handler.
pub(crate) type Handler = Box<dyn FnOnce(&CommandBuffer) + Send>;

/// The command buffer itself: what its handles, its queue's FIFO and the completion thread hold.
pub(crate) struct CbInner {
    pub(crate) queue: Arc<QueueInner>,
    /// Its slot in the queue, and so its pair of timestamp queries.
    slot: u32,
    state: Mutex<CbState>,
    /// Signalled at each status change and once the handlers have run.
    changed: Condvar,
}

struct CbState {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    /// The handlers have run: waits for completion return.
    finished: bool,
    /// Until commit: what is recorded, and the device.
    building: Option<Building>,
    /// Ended while enqueued and not yet committed (an abandon, a loss or a quiesce): the commit
    /// ends it with this error, behind its queue's work held or in flight.
    ended: Option<CommandBufferError>,
    times: Times,
    /// An encoder holds `building`.
    encoding: bool,
}

impl CbState {
    /// What is being recorded. Err: [`Refusal::Committed`]; [`Refusal::EncoderOpen`].
    fn open(&mut self) -> Result<&mut Building, Error> {
        if self.encoding {
            return Err(Error::Refused(Refusal::EncoderOpen));
        }
        self.building
            .as_mut()
            .ok_or(Error::Refused(Refusal::Committed))
    }
}

/// What a command buffer records until its commit; an open encoder holds it.
pub(crate) struct Building {
    pub(crate) device: Device,
    gpu: Gpu,
    handlers: Handlers,
    /// The queue's timestamp pool and the first of the command buffer's two queries.
    stamps: (vk::QueryPool, u32),
    /// Whether the start timestamp is recorded.
    started: bool,
    /// A barrier owed before the next command: an encoder that wrote ended, a memory barrier was
    /// asked for, or a dispatch wrote through an address outside its resources. The segment's
    /// own end barrier pays it too.
    owed: bool,
    /// A dispatch wrote since the last barrier that reached indirect argument reads: a barrier
    /// between dispatches leaves them out unless the dispatch after it is indirect, so an
    /// indirect dispatch after such a write needs one that reaches them.
    args_unordered: bool,
    /// The pipeline bound in the current segment's command buffer.
    bound: Option<vk::Pipeline>,
    /// The current segment wrote memory the host may read: its end barrier reaches the host.
    host_written: bool,
}

/// What the GPU uses: the recorder, the segments, and the blocks kept until retirement.
pub(crate) struct Gpu {
    pub(crate) recorder: Option<Recorder>,
    pub(crate) segments: Vec<Segment>,
    /// Each block the commands use, once: a block released meanwhile is reused only after the
    /// command buffer retires.
    pub(crate) blocks: Vec<Arc<Block>>,
    /// Each Managed buffer's device copy the commands use, once: its place is freed only after
    /// the command buffer retires.
    pub(crate) copies: Vec<Arc<DeviceCopy>>,
    /// The pipeline states the commands use, once each, kept until the command buffer retires.
    pub(crate) pipelines: Vec<ComputePipelineState>,
    /// The textures and sampler states the encoders bound or declared, once each, kept until the
    /// command buffer retires: their elements of set 1 stay theirs while work may read them.
    pub(crate) textures: Vec<Texture>,
    pub(crate) samplers: Vec<SamplerState>,
    /// What the commands wrote into the parameter ring.
    pub(crate) ring: Writes,
}

/// One submission's part of a command buffer: its waits, its commands and its signals.
#[derive(Default)]
pub(crate) struct Segment {
    pub(crate) waits: Vec<(Arc<EventInner>, u64)>,
    pub(crate) command: Option<vk::CommandBuffer>,
    /// At most one per event, the highest value.
    pub(crate) signals: Vec<(Arc<EventInner>, u64)>,
    /// What its commands did to textures over a buffer in an image of their own; once it ends,
    /// what the submit thread needs of that (`crate::over`).
    pub(crate) overs: Overs,
}

pub(crate) struct Handlers {
    pub(crate) scheduled: Vec<Handler>,
    pub(crate) completed: Vec<Handler>,
}

/// A committed command buffer on its way through the submit thread.
pub(crate) struct Pending {
    /// Its place in the device's commit order.
    pub(crate) seq: u64,
    /// The first segment not yet submitted.
    pub(crate) next: usize,
    /// The progress value of the last submission with a segment of it; 0 before the first.
    pub(crate) value: u64,
    pub(crate) gpu: Gpu,
    pub(crate) handlers: Handlers,
}

impl Pending {
    /// How many of its slot's two queries the segments it has passed to the GPU write, the start
    /// first: the start in the first segment with commands, the end in the last segment, which
    /// its commit gave commands.
    pub(crate) fn stamps_written(&self) -> u32 {
        let start =
            (self.gpu.segments[..self.next].iter()).any(|segment| segment.command.is_some());
        let end = self.next == self.gpu.segments.len();
        u32::from(start) + u32::from(end)
    }
}

impl CommandBuffer {
    /// A new command buffer of `queue`, which gave it `slot`, whose two queries are reset.
    pub(crate) fn new(queue: Arc<QueueInner>, device: Device, slot: u32) -> CommandBuffer {
        let stamps = (queue.stamps.pool, Stamps::first(slot));
        CommandBuffer {
            inner: Arc::new(CbInner {
                queue,
                slot,
                state: Mutex::new(CbState {
                    status: CommandBufferStatus::NotEnqueued,
                    error: None,
                    finished: false,
                    building: Some(Building {
                        device,
                        gpu: Gpu {
                            recorder: None,
                            segments: vec![Segment::default()],
                            blocks: Vec::new(),
                            copies: Vec::new(),
                            pipelines: Vec::new(),
                            textures: Vec::new(),
                            samplers: Vec::new(),
                            ring: Writes::default(),
                        },
                        handlers: Handlers {
                            scheduled: Vec::new(),
                            completed: Vec::new(),
                        },
                        stamps,
                        started: false,
                        owed: false,
                        args_unordered: false,
                        bound: None,
                        host_written: false,
                    }),
                    ended: None,
                    times: Times::default(),
                    encoding: false,
                }),
                changed: Condvar::new(),
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, CbState> {
        lock(&self.inner.state)
    }

    /// Runs `f` on what is being recorded. Err: [`Refusal::Committed`];
    /// [`Refusal::EncoderOpen`].
    fn building<R>(&self, f: impl FnOnce(&mut Building) -> Result<R, Error>) -> Result<R, Error> {
        let mut state = self.state();
        f(state.open()?)
    }

    /// Takes what is being recorded for an encoder, until [`close_encoder`](Self::close_encoder).
    /// Err: [`Refusal::Committed`]; [`Refusal::EncoderOpen`].
    pub(crate) fn open_encoder(&self) -> Result<Building, Error> {
        let mut state = self.state();
        state.open()?;
        state.encoding = true;
        Ok(state
            .building
            .take()
            .expect("an open command buffer is recording"))
    }

    /// Gives back what an encoder recorded into.
    pub(crate) fn close_encoder(&self, building: Building) {
        let mut state = self.state();
        state.building = Some(building);
        state.encoding = false;
    }

    /// The device, while the command buffer is not committed. Err: [`Refusal::Committed`].
    fn device(&self) -> Result<Device, Error> {
        self.building(|building| Ok(building.device.clone()))
    }

    /// `encodeSignalEvent:value:`: the event takes `value` once the commands before it complete,
    /// unless a value at least as high is already submitted or set.
    ///
    /// Err: [`Refusal::Committed`]; [`Refusal::OtherDevice`] for another device's event.
    pub fn encode_signal_event(&self, event: &Event, value: u64) -> Result<(), Error> {
        self.building(|building| {
            building.check(event)?;
            building.signal(event.inner(), value);
            Ok(())
        })
    }

    /// `encodeWaitForEvent:value:`: the commands after it run once the event reaches `value`.
    ///
    /// The submit thread holds this segment, and its queue behind it, until the host's value
    /// reaches `value` (then no semaphore wait is submitted) or a signal of at least `value` is
    /// submitted (then the segment waits on the semaphore). Other queues go on.
    ///
    /// Err: [`Refusal::Committed`]; [`Refusal::OtherDevice`] for another device's event;
    /// [`Error::Vulkan`] where ending the segment's commands fails.
    pub fn encode_wait_for_event(&self, event: &Event, value: u64) -> Result<(), Error> {
        self.building(|building| {
            building.check(event)?;
            building.wait(event.inner(), value)
        })
    }

    /// `addScheduledHandler:`: runs on the completion thread once the last segment has passed
    /// `vkQueueSubmit2`, or at once there for a command buffer that ends without running.
    ///
    /// Err: [`Refusal::Committed`].
    pub fn add_scheduled_handler(
        &self,
        handler: impl FnOnce(&CommandBuffer) + Send + 'static,
    ) -> Result<(), Error> {
        self.building(|building| {
            building.handlers.scheduled.push(Box::new(handler));
            Ok(())
        })
    }

    /// `addCompletedHandler:`: runs on the completion thread once the command buffer has
    /// completed or ended, after its scheduled handlers. One thread runs every device's handlers
    /// in turn, so a handler waits for no command buffer.
    ///
    /// Err: [`Refusal::Committed`].
    pub fn add_completed_handler(
        &self,
        handler: impl FnOnce(&CommandBuffer) + Send + 'static,
    ) -> Result<(), Error> {
        self.building(|building| {
            building.handlers.completed.push(Box::new(handler));
            Ok(())
        })
    }

    /// `enqueue`: takes the command buffer's place in its queue now; the queue's later command
    /// buffers wait behind it until it is committed. A second call changes nothing.
    ///
    /// Err: [`Refusal::Committed`]; [`Error::Quiesced`].
    pub fn enqueue(&self) -> Result<(), Error> {
        let device = self.device()?;
        let core = device.core();
        let mut work = core.work.lock();
        if work.closed() {
            return Err(Error::Quiesced);
        }
        let mut state = self.state();
        state.open()?;
        if state.status == CommandBufferStatus::NotEnqueued {
            work.enqueue(&self.inner);
            state.status = CommandBufferStatus::Enqueued;
            self.inner.changed.notify_all();
        }
        Ok(())
    }

    /// `commit`: hands the command buffer to the submit thread and returns. Where nothing is held,
    /// submission follows the device's commit order, and whatever is ready goes in one
    /// `vkQueueSubmit2`. A commit after a loss ends at once with
    /// [`CommandBufferError::DeviceLost`].
    ///
    /// Err: [`Refusal::Committed`]; [`Error::Quiesced`]; [`Error::Vulkan`] where ending the last
    /// segment's commands fails.
    pub fn commit(&self) -> Result<(), Error> {
        // The core, not a handle: the command buffer gives its handle up before another thread
        // can complete it, so a handler that drops the consumer's last handle ends the device on
        // the reaper, never in this call.
        let core = self.building(|building| Ok(Arc::clone(building.device.core())))?;
        let began = crate::host_time();
        let mut work = core.work.lock();
        core.counters
            .commit_lock_wait(crate::host_time().saturating_sub(began));
        if work.closed() {
            return Err(Error::Quiesced);
        }
        let mut state = self.state();
        state.open()?;
        let mut building = state
            .building
            .take()
            .expect("an open command buffer is recording");
        if let Err(error) = building.finish() {
            state.building = Some(building);
            return Err(error);
        }
        let Building {
            device,
            gpu,
            handlers,
            ..
        } = building;
        state.times.kernel_start = times::host_time();
        let pending = Pending {
            seq: work.seq(),
            next: 0,
            value: 0,
            gpu,
            handlers,
        };
        state.status = CommandBufferStatus::Committed;
        self.inner.changed.notify_all();
        let ended = state.ended;
        drop(state);
        // Where it is the last handle, the device ends below, once no lock is held.
        let last = device.release();
        if work.lost() {
            drop(work);
            let error = ended.unwrap_or(CommandBufferError::DeviceLost { in_flight: false });
            core.done
                .end_all(&core, vec![(Arc::clone(&self.inner), pending, error)]);
        } else {
            // One ended while enqueued retires behind its queue's work held or in flight.
            work.commit(&self.inner, pending, ended);
            drop(work);
            core.work.wake();
        }
        drop(core);
        drop(last);
        Ok(())
    }

    /// `waitUntilScheduled`: returns once the command buffer is scheduled, or has completed or
    /// ended. It sends nothing: a command buffer held in a group is scheduled already.
    pub fn wait_until_scheduled(&self) {
        let mut state = self.state();
        while !matches!(
            state.status,
            CommandBufferStatus::Scheduled
                | CommandBufferStatus::Completed
                | CommandBufferStatus::Error
        ) {
            state = self.wait(state);
        }
    }

    /// `waitUntilCompleted`: returns once the command buffer has completed or ended and its
    /// handlers have run. A command buffer held in a group goes at once, as
    /// [`CommandQueue::send_now`](crate::CommandQueue::send_now) sends it.
    pub fn wait_until_completed(&self) {
        self.send();
        let mut state = self.state();
        while !state.finished {
            state = self.wait(state);
        }
    }

    /// Before a host wait for it: where the command buffer is committed and not yet submitted,
    /// its group goes at once (`Work::send`).
    pub(crate) fn send(&self) {
        if !matches!(
            self.status(),
            CommandBufferStatus::Committed | CommandBufferStatus::Scheduled
        ) {
            return;
        }
        if let Some(core) = self.inner.queue.core() {
            core.work.send(&self.inner);
        }
    }

    /// Its queue.
    pub(crate) fn queue(&self) -> &Arc<QueueInner> {
        &self.inner.queue
    }

    /// Whether the command buffer is `queue`'s.
    pub(crate) fn is_of(&self, queue: &QueueInner) -> bool {
        self.inner.queue.is(queue)
    }

    fn wait<'a>(&self, state: MutexGuard<'a, CbState>) -> MutexGuard<'a, CbState> {
        self.inner
            .changed
            .wait(state)
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// `status`.
    pub fn status(&self) -> CommandBufferStatus {
        self.state().status
    }

    /// `error`: set when [`status`](Self::status) is [`CommandBufferStatus::Error`].
    pub fn error(&self) -> Option<CommandBufferError> {
        self.state().error
    }

    /// `kernelStartTime`, in nanoseconds of [`host_time`](crate::host_time) where Metal gives
    /// seconds: the host time at the commit; 0 before it.
    pub fn kernel_start_time(&self) -> u64 {
        self.state().times.kernel_start
    }

    /// `kernelEndTime`, in nanoseconds of [`host_time`](crate::host_time) where Metal gives
    /// seconds: the host time at which it was scheduled, its last segment joining a submission; 0
    /// before then, and for one that ended without being scheduled.
    pub fn kernel_end_time(&self) -> u64 {
        self.state().times.kernel_end
    }

    /// `GPUStartTime`, in nanoseconds of [`host_time`](crate::host_time) where Metal gives
    /// seconds: the timestamp written where its first commands began, placed on the host clock by
    /// the device's last calibration of its clock against the host's
    /// (`vkGetCalibratedTimestampsEXT`, read as the device starts and again at a completion once
    /// 100 ms have passed since the last). 0 until it completes, and for one that ended with an
    /// error.
    pub fn gpu_start_time(&self) -> u64 {
        self.state().times.gpu_start
    }

    /// `GPUEndTime`, in nanoseconds of [`host_time`](crate::host_time) where Metal gives seconds:
    /// the timestamp written where its last segment ended, placed on the host clock as
    /// [`gpu_start_time`](Self::gpu_start_time) is. 0 until it completes, and for one that ended
    /// with an error.
    pub fn gpu_end_time(&self) -> u64 {
        self.state().times.gpu_end
    }

    /// For the device tests: records commands into the current segment with `f`, and keeps each
    /// buffer's block until the command buffer retires. The command buffer is to dispatch nothing
    /// after them: a barrier between dispatches orders only the dispatches before it.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn record_for_test(
        &self,
        buffers: &[&crate::Buffer],
        f: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), Error> {
        self.building(|building| {
            let command = building.command(Reach::Commands)?;
            // What a test records may write anything the host reads.
            building.wrote_host();
            f(&building.device.core().device, command);
            for buffer in buffers {
                building.keep(buffer);
            }
            Ok(())
        })
    }
}

impl Building {
    /// Err: [`Refusal::OtherDevice`] for another device's event.
    fn check(&self, event: &Event) -> Result<(), Error> {
        if event.inner().is_of(self.device.core()) {
            Ok(())
        } else {
            Err(Error::Refused(Refusal::OtherDevice))
        }
    }

    fn last(&mut self) -> &mut Segment {
        let segments = &mut self.gpu.segments;
        let last = segments.len() - 1;
        &mut segments[last]
    }

    /// At commit: records the end timestamp into the last segment, beginning a command buffer for
    /// it where the segment has none, and ends it.
    fn finish(&mut self) -> Result<(), Error> {
        let command = match self.last().command {
            Some(command) => command,
            None => self.begin()?,
        };
        let (pool, first) = self.stamps;
        times::write_end(&self.device.core().device, command, pool, first);
        self.close()
    }

    /// Begins the last segment's command buffer. The command buffer's first begins with the
    /// start timestamp.
    fn begin(&mut self) -> Result<vk::CommandBuffer, Error> {
        let core = Arc::clone(self.device.core());
        let recorder = match self.gpu.recorder.take() {
            Some(recorder) => recorder,
            None => core.recorders.take(&core.device)?,
        };
        let recorder = self.gpu.recorder.insert(recorder);
        let command = recorder.next(&core.device)?;
        if !self.started {
            let (pool, first) = self.stamps;
            times::write_start(&core.device, command, pool, first);
            self.started = true;
        }
        self.last().command = Some(command);
        self.bound = None;
        Ok(command)
    }

    /// Ends the last segment's commands, if it has any.
    fn close(&mut self) -> Result<(), Error> {
        let core = Arc::clone(self.device.core());
        match self.last().command {
            Some(command) => {
                // The images commands wrote go back to their buffers first (`crate::over`).
                let back = self.last().overs.end();
                if !back.is_empty() {
                    // SAFETY: the command buffer records outside any render pass and keeps each
                    // texture, which holds its buffer, until it retires; each image is in its
                    // home layout between commands.
                    unsafe {
                        over::record_side(&core.device, command, &back, false, &core.counters)
                    };
                    self.host_written |= back.iter().any(|copy| copy.shared);
                }
                // Its end barrier pays what is owed and starts the hazard sets again.
                self.owed = false;
                self.args_unordered = false;
                if let Some(recorder) = &mut self.gpu.recorder {
                    recorder.hazards.clear();
                }
                // The host reads what the segment wrote to Shared memory once the command buffer
                // completes, and may read after an event the segment signals.
                let host = self.host_written || !self.last().signals.is_empty();
                self.host_written = false;
                record::end(&core.device, command, host, &core.counters)
            }
            None => Ok(()),
        }
    }

    /// Starts a new segment after the last one.
    fn split(&mut self) -> Result<(), Error> {
        self.close()?;
        self.gpu.segments.push(Segment::default());
        Ok(())
    }

    fn wait(&mut self, event: &Arc<EventInner>, value: u64) -> Result<(), Error> {
        let last = self.last();
        if last.command.is_some() || !last.signals.is_empty() {
            self.split()?;
        }
        self.last().waits.push((Arc::clone(event), value));
        Ok(())
    }

    fn signal(&mut self, event: &Arc<EventInner>, value: u64) {
        let signals = &mut self.last().signals;
        match signals
            .iter_mut()
            .find(|(signalled, _)| Arc::ptr_eq(signalled, event))
        {
            Some((_, highest)) => *highest = (*highest).max(value),
            None => signals.push((Arc::clone(event), value)),
        }
    }

    /// The current segment's command buffer, with the barrier owed recorded first, reaching what
    /// the caller records next (`reach`). Commands after a signal go in a new segment.
    pub(crate) fn command(&mut self, reach: Reach) -> Result<vk::CommandBuffer, Error> {
        let command = self.current()?;
        if self.owed {
            self.barrier(command, reach);
        }
        Ok(command)
    }

    /// The current segment's command buffer, for commands that record a barrier of their own
    /// first that orders every earlier command before every later one and makes their writes
    /// visible to them (a blit's): it pays what is owed, and the hazard sets start again.
    pub(crate) fn ordered_command(&mut self) -> Result<vk::CommandBuffer, Error> {
        let command = self.current()?;
        self.owed = false;
        self.args_unordered = false;
        self.hazards().clear();
        Ok(command)
    }

    /// The current segment's command buffer, begun where it has none. Commands after a signal go
    /// in a new segment.
    fn current(&mut self) -> Result<vk::CommandBuffer, Error> {
        if !self.last().signals.is_empty() {
            self.split()?;
        }
        match self.last().command {
            Some(command) => Ok(command),
            None => self.begin(),
        }
    }

    /// Records a barrier into `command`, the current segment's, reaching what the caller records
    /// next (`reach`), and pays what is owed. The caller starts the hazard sets again where it
    /// has not.
    pub(crate) fn barrier(&mut self, command: vk::CommandBuffer, reach: Reach) {
        let core = self.device.core();
        record::barrier(&core.device, command, reach);
        core.counters.count(Metric::Barriers);
        self.owed = false;
        if reach.reaches_indirect() {
            self.args_unordered = false;
        }
    }

    /// A dispatch wrote memory the host may read: the segment's end barrier reaches the host.
    pub(crate) fn wrote_host(&mut self) {
        self.host_written = true;
    }

    /// A dispatch wrote: an indirect dispatch after it needs a barrier that reaches its
    /// argument read.
    pub(crate) fn wrote(&mut self) {
        self.args_unordered = true;
    }

    /// Whether a dispatch wrote since the last barrier that reached indirect argument reads.
    pub(crate) fn args_unordered(&self) -> bool {
        self.args_unordered
    }

    /// The hazard sets, once the command buffer records: they live with its recorder.
    pub(crate) fn hazards(&mut self) -> &mut Hazards {
        &mut self
            .gpu
            .recorder
            .as_mut()
            .expect("a command buffer that records has a recorder")
            .hazards
    }

    /// Owes a barrier before the next command, and starts the hazard sets again: it orders
    /// everything before it.
    pub(crate) fn owe(&mut self) {
        self.owed = true;
        if let Some(recorder) = &mut self.gpu.recorder {
            recorder.hazards.clear();
        }
    }

    /// Owes a barrier where anything was read or written since the last one.
    pub(crate) fn owe_if_touched(&mut self) {
        if self
            .gpu
            .recorder
            .as_ref()
            .is_some_and(|recorder| recorder.hazards.touched())
        {
            self.owe();
        }
    }

    /// Whether the current command buffer has no pipeline bound yet.
    pub(crate) fn unbound(&self) -> bool {
        self.bound.is_none()
    }

    /// Whether `pipeline` needs binding in the current command buffer; it is bound from now.
    pub(crate) fn bind(&mut self, pipeline: vk::Pipeline) -> bool {
        let needed = self.bound != Some(pipeline);
        self.bound = Some(pipeline);
        needed
    }

    /// Copies `bytes` into the parameter ring, returning their address.
    pub(crate) fn write_ring(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let core = Arc::clone(self.device.core());
        core.counters.count(Metric::RingWrites);
        self.gpu.ring.write(&core, bytes)
    }

    /// Keeps `buffer`'s block, and its device copy where it has one, until the command buffer
    /// retires.
    pub(crate) fn keep(&mut self, buffer: &Buffer) {
        let (block, copy) = buffer.held();
        let blocks = &mut self.gpu.blocks;
        if !blocks.iter().rev().any(|kept| Arc::ptr_eq(kept, block)) {
            blocks.push(Arc::clone(block));
        }
        if let Some(copy) = copy {
            let copies = &mut self.gpu.copies;
            if !copies.iter().rev().any(|kept| Arc::ptr_eq(kept, copy)) {
                copies.push(Arc::clone(copy));
            }
        }
    }

    /// Keeps `state` until the command buffer retires.
    pub(crate) fn keep_pipeline(&mut self, state: &ComputePipelineState) {
        let pipelines = &mut self.gpu.pipelines;
        if !pipelines.iter().rev().any(|kept| kept.is(state)) {
            pipelines.push(state.clone());
        }
    }

    /// Keeps `texture` until the command buffer retires.
    pub(crate) fn keep_texture(&mut self, texture: &Texture) {
        let textures = &mut self.gpu.textures;
        if !textures.iter().rev().any(|kept| kept.id() == texture.id()) {
            textures.push(texture.clone());
        }
    }

    /// Whether the commands track what they write for textures over a buffer in an image of
    /// their own (`crate::over`): while any lives.
    pub(crate) fn copies_over(&self) -> bool {
        self.device.core().copied_overs.load(Ordering::Relaxed) > 0
    }

    /// Before a command that makes `accesses` (by hazard id), `unbound` where it reaches memory
    /// through an address outside its resources: records the copies its textures over a buffer
    /// in an image of their own need first (`crate::over`), in the segment the command goes in,
    /// ordered as a blit's copy is.
    ///
    /// Err: [`Error::Vulkan`] where the command buffer cannot record.
    pub(crate) fn overs_before(
        &mut self,
        accesses: &[(u32, Access)],
        unbound: bool,
    ) -> Result<(), Error> {
        if !self.copies_over() {
            return Ok(());
        }
        if !self.last().signals.is_empty() {
            self.split()?;
        }
        let segments = &mut self.gpu.segments;
        let last = segments.len() - 1;
        let copies = segments[last]
            .overs
            .before(accesses, &self.gpu.textures, unbound);
        if copies.is_empty() {
            return Ok(());
        }
        let command = self.ordered_command()?;
        let core = Arc::clone(self.device.core());
        // SAFETY: the command buffer records outside any render pass and keeps each texture,
        // which holds its buffer, until it retires; each image is in its home layout between
        // commands.
        let host = unsafe { over::record(&core.device, command, &copies, &core.counters) };
        self.host_written |= host;
        Ok(())
    }

    /// After a command that made `accesses` (by hazard id), `unbound` where it wrote through an
    /// address outside its resources (`crate::over`); a command recorded while no texture over a
    /// buffer in an image of its own lived is untracked.
    pub(crate) fn overs_after(&mut self, accesses: &[(u32, Access)], unbound: bool) {
        match self.copies_over() {
            true => self.last().overs.after(accesses, unbound),
            false => self.overs_untracked(),
        }
    }

    /// After a command whose writes were not tracked: a texture over a buffer in an image of its
    /// own first used later takes its bytes in again first, as after a write through an address
    /// outside its resources (`crate::over`).
    pub(crate) fn overs_untracked(&mut self) {
        self.last().overs.after(&[], true);
    }

    /// Keeps `state` until the command buffer retires.
    pub(crate) fn keep_sampler(&mut self, state: &SamplerState) {
        let samplers = &mut self.gpu.samplers;
        if !samplers.iter().rev().any(|kept| kept.is(state)) {
            samplers.push(state.clone());
        }
    }
}

impl CbInner {
    /// The last segment joined a submission at `now`: passed to `vkQueueSubmit2` at once, or held
    /// in its queue's group.
    pub(crate) fn scheduled(&self, now: u64) {
        let mut state = lock(&self.state);
        state.status = CommandBufferStatus::Scheduled;
        state.times.kernel_end = now;
        self.changed.notify_all();
    }

    /// Ended while enqueued and not committed: the commit ends it with `error`.
    pub(crate) fn end_uncommitted(&self, error: CommandBufferError) {
        lock(&self.state).ended = Some(error);
    }

    /// Its start and end timestamps, once it completed.
    pub(crate) fn stamps(&self, core: &Core) -> Option<(u64, u64)> {
        // SAFETY: the queue's pool holds the command buffer's two queries.
        unsafe {
            readback::timestamps(
                &core.device,
                self.queue.stamps.pool,
                Stamps::first(self.slot),
            )
        }
    }
}

/// On the completion thread: runs the scheduled handlers of a command buffer just submitted.
pub(crate) fn run_scheduled(inner: Arc<CbInner>, handlers: Vec<Handler>) {
    let handle = CommandBuffer { inner };
    for handler in handlers {
        handler(&handle);
    }
}

/// On the completion thread: retires a command buffer that completed (`error` None), with its GPU
/// times where they were read, or ended, recycling its recorder and giving up its blocks and its
/// place in the queue with the `written` of its two queries reset, then runs its handlers.
pub(crate) fn retire(
    core: &Core,
    inner: Arc<CbInner>,
    gpu: Gpu,
    handlers: Handlers,
    error: Option<CommandBufferError>,
    gpu_times: Option<(u64, u64)>,
    written: u32,
) {
    {
        let mut state = lock(&inner.state);
        if let Some((start, end)) = gpu_times {
            state.times.gpu_start = start;
            state.times.gpu_end = end;
        }
        state.status = match error {
            Some(_) => CommandBufferStatus::Error,
            None => CommandBufferStatus::Completed,
        };
        state.error = error;
        inner.changed.notify_all();
    }
    let Gpu {
        recorder,
        segments,
        blocks,
        copies,
        pipelines,
        textures,
        samplers,
        ring,
    } = gpu;
    if let Some(recorder) = recorder {
        core.recorders.give_back(core, recorder);
    }
    ring.give_back(core);
    drop((segments, blocks, copies, pipelines, textures, samplers));
    times::reset(
        &core.device,
        inner.queue.stamps.pool,
        Stamps::first(inner.slot),
        written,
    );
    inner.queue.release(inner.slot);
    let handle = CommandBuffer { inner };
    for handler in handlers.scheduled.into_iter().chain(handlers.completed) {
        handler(&handle);
    }
    let mut state = lock(&handle.inner.state);
    state.finished = true;
    handle.inner.changed.notify_all();
}

impl Drop for CbInner {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        // A command buffer never committed gives back its recorder and its place in the queue,
        // whose queries no submission wrote, so they stay reset; a committed one did so when it
        // retired.
        if let Some(Building { device, gpu, .. }) = state.building.take() {
            let core = device.core();
            if let Some(recorder) = gpu.recorder {
                core.recorders.give_back(core, recorder);
            }
            gpu.ring.give_back(core);
            self.queue.release(self.slot);
        }
    }
}
