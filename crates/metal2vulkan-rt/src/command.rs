//! Command buffers: recorded at once into segments that event waits and signals split, and their
//! states from creation to retirement.

use crate::buffer::Buffer;
use crate::capture::CopyIn;
use crate::counters::{Metric, Tally};
use crate::device::{lock, Core, Device};
use crate::error::{Error, Refusal};
use crate::event::{Event, EventInner};
use crate::hazard::{Access, Hazards};
use crate::kept::{Lent, Objects};
use crate::memory::Block;
use crate::over::{self, Overs};
use crate::pipeline::ComputePipelineState;
use crate::queue::QueueInner;
use crate::readback;
use crate::record::{self, Reach, Recorder};
use crate::render::RenderPipelineState;
use crate::ring::{Placed, Writes};
use crate::sampler::SamplerState;
use crate::texture::{Image, OverCopy, Texture};
use crate::times::{self, Pair, Times};
use ash::vk;
use std::sync::atomic::Ordering;
use std::sync::atomic::Ordering::Relaxed;
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
/// last segment ends, and read once it completes. Their pair of queries comes from the device's
/// timestamp pools at its first record and goes back as they are read, or as it drops. A command
/// buffer with no commands, or whose last segment has none, records a `VkCommandBuffer` for them.
/// Where every pair is taken and the device makes no new pool, it runs untimed, its GPU times 0
/// ([`Metrics::untimed_command_buffers`](crate::Metrics::untimed_command_buffers)).
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
    /// `vkQueueSubmit2`, or in a group its queue holds while three of its submissions run, which
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
    /// From its commit until it retires: its pair of timestamp queries, where it took one.
    stamps: Option<Pair>,
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
    /// Whether the first command buffer began: the pair was taken, and where one was, the start
    /// timestamp recorded.
    started: bool,
    /// Its pair of timestamp queries, from its first record; None before it and where it runs
    /// untimed.
    stamps: Option<Pair>,
    /// A barrier owed before the next command: an encoder that wrote ended, a memory barrier was
    /// asked for, a dispatch wrote through an address outside its resources, or a render pass
    /// ended. The segment's own end barrier pays it too.
    owed: bool,
    /// The barrier owed orders a render pass before what follows it ([`Building::owe_pass`]): it
    /// reaches every later command ([`Reach::Commands`]), whatever the command that pays it names.
    owed_pass: bool,
    /// A dispatch wrote since the last barrier that reached indirect argument reads: a barrier
    /// between dispatches leaves them out unless the dispatch after it is indirect, so an
    /// indirect dispatch after such a write needs one that reaches them.
    args_unordered: bool,
    /// The compute pipeline bound in the current segment's command buffer.
    bound: Option<vk::Pipeline>,
    /// The graphics pipeline bound in the current segment's command buffer.
    drawn: Option<vk::Pipeline>,
    /// The push constant bytes the current segment's command buffer holds.
    pushed: Pushed,
    /// The current segment wrote memory the host may read: its end barrier reaches the host.
    host_written: bool,
    /// The vertex buffers bound in the current segment's command buffer: binding, buffer and
    /// offset. Vulkan keeps a binding across renderings and pipeline binds of one command buffer.
    vertex: Vec<(u32, vk::Buffer, u64)>,
    /// What the encoders counted since the last one ended, added to the device's counters as each
    /// ends ([`CommandBuffer::close_encoder`]).
    tally: Tally,
}

/// The push constant bytes a command buffer holds, from offset 0, as its pushes left them.
///
/// Every pipeline, compute and graphics, has the device's one layout and its one push range of
/// [`PUSH_STAGES`](crate::pipeline::PUSH_STAGES), so a push stays across pipeline binds and both
/// bind points until the next one: a dispatch or a draw whose push bytes are already there
/// records none, as a Metal argument set once stays bound for the commands after it.
#[derive(Default)]
pub(crate) struct Pushed {
    bytes: Vec<u8>,
}

impl Pushed {
    /// Whether `push`, written from offset 0, changes the bytes held; they are `push` from now,
    /// with the held bytes past its end kept.
    pub(crate) fn needed(&mut self, push: &[u8]) -> bool {
        if self.bytes.get(..push.len()) == Some(push) {
            return false;
        }
        if self.bytes.len() < push.len() {
            self.bytes.resize(push.len(), 0);
        }
        self.bytes[..push.len()].copy_from_slice(push);
        true
    }
}

/// What the GPU uses: the recorder, the segments, and the blocks kept until retirement.
pub(crate) struct Gpu {
    pub(crate) recorder: Option<Recorder>,
    pub(crate) segments: Vec<Segment>,
    /// Each heap's block the commands use (`useHeap:`), once: a block released meanwhile is
    /// reused only after the command buffer retires.
    pub(crate) blocks: Vec<Arc<Block>>,
    /// The buffers, textures, sampler states and pipeline states the encoders bound, declared or
    /// copied, once each, kept until the command buffer retires: a buffer's block and device copy
    /// with it, and a texture's and a sampler's elements of set 1 stay theirs while work may read
    /// them (`crate::kept`).
    pub(crate) kept: Box<Objects>,
    /// What the commands wrote into the parameter ring.
    pub(crate) ring: Writes,
    /// A texture over a Managed buffer's device copy in an image of its own is kept: the submit
    /// thread looks for the copies in it owes (`crate::over::ManagedOvers`).
    pub(crate) managed: bool,
    /// The images of released textures whose copies back the commands make, kept until the
    /// command buffer retires.
    pub(crate) images: Vec<Arc<Image>>,
}

impl Gpu {
    /// Nothing for the GPU: no segment holds commands, a wait or a signal. Such a command buffer
    /// records no timestamps, takes no submission and captures nothing.
    pub(crate) fn is_empty(&self) -> bool {
        (self.segments.iter()).all(|segment| {
            segment.command.is_none() && segment.waits.is_empty() && segment.signals.is_empty()
        })
    }
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
    /// The Managed buffers with textures over their device copies that its commands wrote: each
    /// such texture owes its copy in from the next submission (`crate::over::ManagedOvers`).
    pub(crate) managed_written: Vec<Buffer>,
    /// The textures over a Managed buffer's device copy in an image of their own that its
    /// commands wrote: its submission settles their copies back (`crate::over::ManagedOvers`).
    pub(crate) managed_wrote: Vec<Texture>,
    /// Of the first segment: what the command buffer's commit captured of Managed buffers'
    /// changed ranges (`crate::capture`). Each copy-in goes ahead of the segment's commands, after
    /// its waits, and is kept until the command buffer retires.
    pub(crate) copy_ins: Vec<CopyIn>,
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
    /// It took a pair of timestamp queries.
    pub(crate) timed: bool,
}

impl Pending {
    /// How many of its pair's two queries the segments it has passed to the GPU write, the start
    /// first, none where it runs untimed: the start in the first segment with commands, the end
    /// in the last segment, which its commit gave commands unless the command buffer has nothing
    /// for the GPU. A segment is passed once the submit thread stages it into its queue's group,
    /// which is always submitted: it goes before any later one, and at a loss or a failed
    /// submission the command buffer ends `DeviceLost`, whose timestamps are never read. So the
    /// completion thread's read, which waits and which it makes only for a command buffer that
    /// completed with both counted, waits only on queries a submission it saw complete wrote.
    pub(crate) fn stamps_written(&self) -> u32 {
        if !self.timed {
            return 0;
        }
        let segments = &self.gpu.segments;
        let start = (segments[..self.next].iter()).any(|segment| segment.command.is_some());
        let end = self.next == segments.len()
            && segments
                .last()
                .is_some_and(|segment| segment.command.is_some());
        u32::from(start) + u32::from(end)
    }
}

impl CommandBuffer {
    /// A new command buffer of `queue`, which counted it live, keeping its objects in `kept`,
    /// empty, the lists of a retired one where the queue had them.
    pub(crate) fn new(queue: Arc<QueueInner>, device: Device, kept: Box<Objects>) -> CommandBuffer {
        CommandBuffer {
            inner: Arc::new(CbInner {
                queue,
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
                            kept,
                            ring: Writes::default(),
                            managed: false,
                            images: Vec::new(),
                        },
                        handlers: Handlers {
                            scheduled: Vec::new(),
                            completed: Vec::new(),
                        },
                        started: false,
                        stamps: None,
                        owed: false,
                        owed_pass: false,
                        args_unordered: false,
                        bound: None,
                        drawn: None,
                        pushed: Pushed::default(),
                        host_written: false,
                        vertex: Vec::new(),
                        tally: Tally::default(),
                    }),
                    ended: None,
                    times: Times::default(),
                    encoding: false,
                    stamps: None,
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

    /// Gives back what an encoder recorded into, adding what it counted to the device's counters:
    /// once an encoder, so that no command it recorded wrote a counter the device's threads share.
    pub(crate) fn close_encoder(&self, mut building: Building) {
        (building.device.core().counters).add(&mut building.tally);
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
        let turn = core.work.turn();
        let work = core.work.lock();
        core.counters
            .commit_lock_wait(crate::host_time().saturating_sub(began));
        if work.closed() {
            return Err(Error::Quiesced);
        }
        // With commands, it takes the Managed buffers' ranges named since the last capture
        // (`crate::capture`); one with none leaves them for the next.
        let empty = self.state().open()?.gpu.is_empty();
        let (mut work, copy_ins) = match empty || work.lost() {
            true => (work, Vec::new()),
            false => core.work.capture(&core, work)?,
        };
        let mut state = self.state();
        if let Err(error) = state.open() {
            work.carry(copy_ins);
            return Err(error);
        }
        let mut building = state
            .building
            .take()
            .expect("an open command buffer is recording");
        if let Err(error) = building.finish() {
            state.building = Some(building);
            work.carry(copy_ins);
            return Err(error);
        }
        (building.device.core().counters).add(&mut building.tally);
        let Building {
            device,
            mut gpu,
            handlers,
            stamps,
            ..
        } = building;
        gpu.segments[0].copy_ins = copy_ins;
        state.times.kernel_start = times::host_time();
        state.stamps = stamps;
        let pending = Pending {
            seq: work.seq(),
            next: 0,
            value: 0,
            gpu,
            handlers,
            timed: stamps.is_some(),
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
                .end_all(&core, vec![(Arc::clone(&self.inner), pending, Some(error))]);
        } else {
            // One ended while enqueued retires behind its queue's work held or in flight.
            work.commit(&self.inner, pending, ended);
            drop(work);
            core.work.wake();
        }
        drop(turn);
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
    /// error. One whose segments hold no commands, no event wait and no event signal writes no
    /// timestamp: its start and end are one time, the latest GPU end of its queue's command
    /// buffers before it, or its commit where that is later, as Metal reports for a command buffer
    /// that does no work.
    pub fn gpu_start_time(&self) -> u64 {
        self.state().times.gpu_start
    }

    /// `GPUEndTime`, in nanoseconds of [`host_time`](crate::host_time) where Metal gives seconds:
    /// the timestamp written where its last segment ended, placed on the host clock as
    /// [`gpu_start_time`](Self::gpu_start_time) is. 0 until it completes, and for one that ended
    /// with an error; its start time for one that writes no timestamp.
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
    /// it where the segment has none, and ends it; nothing where the command buffer has nothing
    /// for the GPU, and no timestamp where it runs untimed.
    fn finish(&mut self) -> Result<(), Error> {
        if self.gpu.is_empty() {
            return Ok(());
        }
        let last = self.last().command;
        let command = match last {
            Some(command) => command,
            None if self.started && self.stamps.is_none() => return self.close(),
            None => self.begin()?,
        };
        if let Some(pair) = self.stamps {
            times::write_end(&self.device.core().device, command, pair.pool, pair.first);
        }
        self.close()
    }

    /// Begins the last segment's command buffer. The command buffer's first takes its pair of
    /// timestamp queries from the device's pools and begins with the start timestamp, or, where
    /// it gets none, counts it untimed.
    fn begin(&mut self) -> Result<vk::CommandBuffer, Error> {
        let core = Arc::clone(self.device.core());
        let recorder = match self.gpu.recorder.take() {
            Some(recorder) => recorder,
            None => core.recorders.take(&core.device)?,
        };
        let recorder = self.gpu.recorder.insert(recorder);
        let command = recorder.next(&core.device)?;
        if !self.started {
            self.started = true;
            self.stamps = core.stamps.take(&core.device, &core.counters);
            match self.stamps {
                Some(pair) => times::write_start(&core.device, command, pair.pool, pair.first),
                None => self.tally.count(Metric::UntimedCommandBuffers),
            }
        }
        self.last().command = Some(command);
        self.bound = None;
        self.drawn = None;
        self.pushed = Pushed::default();
        self.vertex.clear();
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
                self.owed_pass = false;
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
        self.owed_pass = false;
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
        // One owed after a render pass reaches every later command, whatever the caller names.
        let reach = match self.owed && self.owed_pass {
            true => Reach::Commands,
            false => reach,
        };
        record::barrier(&self.device.core().device, command, reach);
        self.tally.count(Metric::Barriers);
        self.owed = false;
        self.owed_pass = false;
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

    /// A render pass ended with no attachment to move home: the barrier that orders it before
    /// later commands is owed, recorded first by the segment's next command, reaching every later
    /// command, or paid by the segment's end barrier where none follows. A command buffer that
    /// ends with a pass records one barrier after it, its end's.
    pub(crate) fn owe_pass(&mut self) {
        self.owe();
        self.owed_pass = true;
    }

    /// Whether the next command opens a command buffer of the segments: the last segment has
    /// none begun, or a signal ends it first. Every command before it is then in a command buffer
    /// that ended on the queue, whose end barrier orders it before every later command
    /// (`record::end`), as it does for a copy-in (`crate::capture`).
    pub(crate) fn opens_command_buffer(&mut self) -> bool {
        let last = self.last();
        last.command.is_none() || !last.signals.is_empty()
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

    /// Whether the current command buffer has no compute pipeline bound yet.
    pub(crate) fn unbound(&self) -> bool {
        self.bound.is_none()
    }

    /// Whether the compute pipeline `pipeline` needs binding in the current command buffer; it is
    /// bound from now.
    pub(crate) fn bind(&mut self, pipeline: vk::Pipeline) -> bool {
        let needed = self.bound != Some(pipeline);
        self.bound = Some(pipeline);
        needed
    }

    /// Whether the graphics pipeline `pipeline` needs binding in the current command buffer, and
    /// whether it is the command buffer's first, before which set 1 is bound at the graphics bind
    /// point; it is bound from now.
    pub(crate) fn bind_graphics(&mut self, pipeline: vk::Pipeline) -> (bool, bool) {
        let first = self.drawn.is_none();
        let needed = self.drawn != Some(pipeline);
        self.drawn = Some(pipeline);
        (needed, first)
    }

    /// Whether vertex buffer `binding` needs binding to `buffer` at `offset` in the current
    /// command buffer: it holds another binding there, or none. It is bound so from now.
    pub(crate) fn vertex_needed(&mut self, binding: u32, buffer: vk::Buffer, offset: u64) -> bool {
        match self.vertex.iter_mut().find(|(held, ..)| *held == binding) {
            Some((_, held, at)) if (*held, *at) == (buffer, offset) => false,
            Some((_, held, at)) => {
                (*held, *at) = (buffer, offset);
                true
            }
            None => {
                self.vertex.push((binding, buffer, offset));
                true
            }
        }
    }

    /// Whether `push`, the push bytes of a dispatch or a draw from offset 0, needs recording in
    /// the current command buffer: its push bytes do not already begin with it. They do from now.
    pub(crate) fn push_needed(&mut self, push: &[u8]) -> bool {
        let needed = self.pushed.needed(push);
        self.tally.count(match needed {
            true => Metric::Pushes,
            false => Metric::PushesHeld,
        });
        needed
    }

    /// Counts one more of `metric`, added to the device's counters once the encoder ends.
    pub(crate) fn count(&mut self, metric: Metric) {
        self.tally.count(metric);
    }

    /// Copies `bytes` into the parameter ring, returning their address.
    pub(crate) fn write_ring(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        Ok(self.write_ring_placed(bytes)?.address)
    }

    /// Copies `bytes` into the parameter ring, returning where they lie.
    pub(crate) fn write_ring_placed(&mut self, bytes: &[u8]) -> Result<Placed, Error> {
        self.tally.count(Metric::RingWrites);
        self.gpu.ring.write(self.device.core(), bytes)
    }

    /// Keeps `buffer`, with its block and its device copy where it has one, until the command
    /// buffer retires.
    pub(crate) fn keep(&mut self, buffer: &Buffer) {
        self.gpu.kept.buffers.hold(buffer);
    }

    /// Keeps `buffer` as [`keep`](Self::keep) does, and lends it for a binding slot: no count
    /// changes but at the recording's first use of it.
    ///
    /// # Safety
    ///
    /// The borrow is used only until the command buffer commits: in an encoder's tables.
    pub(crate) unsafe fn lend_buffer(&mut self, buffer: &Buffer) -> Lent<Buffer> {
        // SAFETY: the caller uses the borrow only before the commit, and the lists are cleared
        // only once the command buffer retired.
        unsafe { self.gpu.kept.buffers.keep(buffer) }
    }

    /// Keeps `block` until the command buffer retires.
    pub(crate) fn keep_block(&mut self, block: &Arc<Block>) {
        let blocks = &mut self.gpu.blocks;
        if !blocks.iter().rev().any(|kept| Arc::ptr_eq(kept, block)) {
            blocks.push(Arc::clone(block));
        }
    }

    /// Keeps `state` until the command buffer retires, and lends it to the encoder that set it.
    ///
    /// # Safety
    ///
    /// As [`lend_buffer`](Self::lend_buffer).
    pub(crate) unsafe fn lend_pipeline(
        &mut self,
        state: &ComputePipelineState,
    ) -> Lent<ComputePipelineState> {
        // SAFETY: as `lend_buffer`.
        unsafe { self.gpu.kept.pipelines.keep(state) }
    }

    /// Keeps `state` until the command buffer retires, and lends it to the encoder that set it.
    ///
    /// # Safety
    ///
    /// As [`lend_buffer`](Self::lend_buffer).
    pub(crate) unsafe fn lend_render_pipeline(
        &mut self,
        state: &RenderPipelineState,
    ) -> Lent<RenderPipelineState> {
        // SAFETY: as `lend_buffer`.
        unsafe { self.gpu.kept.render_pipelines.keep(state) }
    }

    /// Keeps `texture` until the command buffer retires, as Metal's command buffer retains every
    /// object its encoders bind: each object once, a view apart from its parent.
    pub(crate) fn keep_texture(&mut self, texture: &Texture) {
        if self.gpu.kept.textures.hold(texture) {
            self.gpu.managed |= texture.managed_over().is_some();
        }
    }

    /// Keeps `texture` as [`keep_texture`](Self::keep_texture) does, and lends it for a binding
    /// slot.
    ///
    /// # Safety
    ///
    /// As [`lend_buffer`](Self::lend_buffer).
    pub(crate) unsafe fn lend_texture(&mut self, texture: &Texture) -> Lent<Texture> {
        self.keep_texture(texture);
        // SAFETY: as `lend_buffer`; the texture is kept now.
        unsafe { self.gpu.kept.textures.keep(texture) }
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
        let copies = self.overs_owed(accesses, unbound);
        self.record_overs(copies)
    }

    /// The copies a command that makes `accesses` (by hazard id), `unbound` where it reaches
    /// memory through an address outside its resources, needs first (`crate::over`), in the
    /// current segment, which the caller records ([`record_overs`](Self::record_overs)): a draw's,
    /// which ends its pass's rendering to record them. None while no texture over a buffer in an
    /// image of its own lives.
    pub(crate) fn overs_owed(&mut self, accesses: &[(u32, Access)], unbound: bool) -> over::Copies {
        if !self.copies_over() {
            return over::Copies::default();
        }
        let segments = &mut self.gpu.segments;
        let last = segments.len() - 1;
        segments[last]
            .overs
            .before(accesses, self.gpu.kept.textures.as_slice(), unbound)
    }

    /// Records `copies`, ordered as a blit's copy is; nothing where there are none.
    ///
    /// Err: [`Error::Vulkan`] where the command buffer cannot record.
    pub(crate) fn record_overs(&mut self, copies: over::Copies) -> Result<(), Error> {
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

    /// Before a command that reads or writes a Managed buffer's device copy: records the copies
    /// back `backs` owed by textures over it (`crate::over::ManagedOvers`), ordered as a blit's
    /// copy is, and keeps their images until the command buffer retires.
    ///
    /// Err: [`Error::Vulkan`] where the command buffer cannot record.
    pub(crate) fn managed_backs(&mut self, backs: Vec<over::Back>) -> Result<(), Error> {
        if backs.is_empty() {
            return Ok(());
        }
        let command = self.ordered_command()?;
        let core = Arc::clone(self.device.core());
        let copies: Vec<OverCopy> = backs.iter().map(|(_, copy, _)| *copy).collect();
        // SAFETY: the command buffer records outside any render pass and keeps each image, and
        // each buffer through its texture or below, until it retires; each image is in its home
        // layout between commands.
        unsafe { over::record_side(&core.device, command, &copies, false, &core.counters) };
        self.gpu
            .images
            .extend(backs.into_iter().map(|(_, _, image)| image));
        Ok(())
    }

    /// A command writes `texture`: where it is over a Managed buffer's device copy in an image of
    /// its own, it owes its copy back to the readers recorded after the command, and from the
    /// segment's submission to every later reader (`crate::over::ManagedOvers`).
    pub(crate) fn wrote_texture_managed(&mut self, texture: &Texture) {
        let Some(buffer) = texture.managed_over() else {
            return;
        };
        let wrote = &mut self.last().managed_wrote;
        let first = !wrote.iter().any(|kept| kept.id() == texture.id());
        if first {
            wrote.push(texture.clone());
        }
        buffer.over_entries().pend(texture.id(), first);
    }

    /// A command writes `buffer`: where it is a Managed buffer with textures over its device copy,
    /// each owes its copy in from the next submission (`crate::over::ManagedOvers`).
    pub(crate) fn wrote_managed(&mut self, buffer: &Buffer) {
        if buffer.managed_overs().is_none() {
            return;
        }
        let written = &mut self.last().managed_written;
        if !written.iter().any(|kept| kept.id() == buffer.id()) {
            written.push(buffer.clone());
        }
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

    /// Keeps `state` until the command buffer retires, and lends it for a binding slot.
    ///
    /// # Safety
    ///
    /// As [`lend_buffer`](Self::lend_buffer).
    pub(crate) unsafe fn lend_sampler(&mut self, state: &SamplerState) -> Lent<SamplerState> {
        // SAFETY: as `lend_buffer`.
        unsafe { self.gpu.kept.samplers.keep(state) }
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

    /// Its start and end timestamps, once it completed having written both.
    pub(crate) fn stamps(&self, core: &Core) -> Option<(u64, u64)> {
        let pair = lock(&self.state).stamps?;
        // SAFETY: the pair's pool holds its two queries, which a submission that completed wrote.
        unsafe { readback::timestamps(&core.device, pair.pool, pair.first) }
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
/// times where they were read, or ended, recycling its recorder and giving up its blocks, its pair
/// of timestamp queries with the `written` of them reset, and its place in the queue, then runs
/// its handlers.
pub(crate) fn retire(
    core: &Core,
    inner: Arc<CbInner>,
    gpu: Gpu,
    handlers: Handlers,
    error: Option<CommandBufferError>,
    gpu_times: Option<(u64, u64)>,
    written: u32,
) {
    let stamps = {
        let mut state = lock(&inner.state);
        // One with nothing for the GPU, as Metal reports it: its start and end equal, at its
        // queue's previous command buffer's end or its own commit, whichever is later.
        let gpu_times = gpu_times.or_else(|| {
            (error.is_none() && gpu.is_empty()).then(|| {
                let at = (inner.queue.last_end.load(Relaxed)).max(state.times.kernel_start);
                (at, at)
            })
        });
        if let Some((start, end)) = gpu_times {
            state.times.gpu_start = start;
            state.times.gpu_end = end;
            inner.queue.last_end.fetch_max(end, Relaxed);
        }
        state.status = match error {
            Some(_) => CommandBufferStatus::Error,
            None => CommandBufferStatus::Completed,
        };
        state.error = error;
        inner.changed.notify_all();
        state.stamps.take()
    };
    let Gpu {
        recorder,
        segments,
        blocks,
        mut kept,
        ring,
        managed: _,
        images,
    } = gpu;
    if let Some(recorder) = recorder {
        core.recorders.give_back(core, recorder);
    }
    for copy_in in segments.into_iter().flat_map(|segment| segment.copy_ins) {
        copy_in.retire(core);
    }
    ring.give_back(core);
    drop((blocks, images));
    kept.clear();
    if let Some(pair) = stamps {
        times::reset(&core.device, pair.pool, pair.first, written);
        core.stamps.give_back(pair);
    }
    inner.queue.release(kept);
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
        // A command buffer never committed gives back its recorder, its pair of timestamp
        // queries, which no submission wrote, so they stay reset, and its place in the queue; a
        // committed one did so when it retired.
        if let Some(Building {
            device,
            gpu,
            stamps,
            ..
        }) = state.building.take()
        {
            let core = device.core();
            let Gpu {
                recorder,
                mut kept,
                ring,
                ..
            } = gpu;
            if let Some(recorder) = recorder {
                core.recorders.give_back(core, recorder);
            }
            ring.give_back(core);
            if let Some(pair) = stamps {
                core.stamps.give_back(pair);
            }
            kept.clear();
            self.queue.release(kept);
        }
    }
}
