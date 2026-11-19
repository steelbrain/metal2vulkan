//! Command queues: Metal's bound on a queue's uncompleted command buffers, and abandoning a queue.

use crate::command::CommandBuffer;
use crate::counters::{Counters, Gauge};
use crate::device::{lock, Core, Device};
use crate::error::{Error, Refusal};
use crate::kept::Objects;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};

/// `MTLCommandQueue`: one FIFO on the submit thread. Every queue submits to the device's one
/// `VkQueue`.
///
/// A cheap handle: clones share one queue. It holds the device.
#[derive(Clone)]
pub struct CommandQueue {
    inner: Arc<QueueInner>,
    device: Device,
}

/// The queue itself: what its command buffers and the submit thread hold.
pub(crate) struct QueueInner {
    max: u32,
    /// The command buffers made and not yet retired, and the lists retired ones left.
    live: Mutex<Live>,
    /// Signalled when one retires.
    room: Condvar,
    /// The latest GPU end time of its command buffers retired, in host nanoseconds: where a
    /// command buffer with nothing for the GPU places its times.
    pub(crate) last_end: AtomicU64,
    /// The latest kernel end time of its command buffers scheduled, in host nanoseconds: where
    /// the next one's kernel start is placed at the latest, as Metal schedules a queue's command
    /// buffers in order.
    pub(crate) last_scheduled: AtomicU64,
    /// The GPU run, in nanoseconds, its next command buffer is expected to take
    /// ([`expected_run`](Self::expected_run)); `u64::MAX` before one of its command buffers
    /// retired with its GPU times.
    run: AtomicU64,
    counters: Counters,
    /// The device, without keeping it: a command buffer handle that outlives the device's
    /// handles holds its queue, and neither keeps the device from its teardown.
    core: Weak<Core>,
}

/// A queue's live command buffers: those made and not yet retired, and the kept-object lists
/// of retired ones, empty, with their capacity, each in the allocation it came in
/// (`crate::kept::Objects`): together at most the queue's maximum, since each list is a live
/// command buffer's or one retired since.
struct Live {
    count: u32,
    objects: Vec<Objects>,
}

impl Device {
    /// `newCommandQueueWithMaxCommandBufferCount:`; `newCommandQueue` is this call with 64.
    ///
    /// The queue makes no Vulkan object: its command buffers take their pairs of timestamp
    /// queries from the device's pools
    /// ([`Gauges::timestamp_pools`](crate::Gauges::timestamp_pools)).
    ///
    /// Err: [`Refusal::NoCommandBuffers`] for a count of 0.
    pub fn new_command_queue(&self, max_command_buffer_count: u32) -> Result<CommandQueue, Error> {
        if max_command_buffer_count == 0 {
            return Err(Error::Refused(Refusal::NoCommandBuffers));
        }
        let core = self.core();
        Ok(CommandQueue {
            inner: Arc::new(QueueInner {
                max: max_command_buffer_count,
                live: Mutex::new(Live {
                    count: 0,
                    objects: Vec::new(),
                }),
                room: Condvar::new(),
                last_end: AtomicU64::new(0),
                last_scheduled: AtomicU64::new(0),
                run: AtomicU64::new(u64::MAX),
                counters: core.counters.clone(),
                core: Arc::downgrade(core),
            }),
            device: self.clone(),
        })
    }
}

impl CommandQueue {
    /// `commandBuffer`: waits, as Metal does, while the queue's maximum count of its command
    /// buffers are made and not yet completed. `commandBufferWithUnretainedReferences` is the same
    /// call: what a command buffer keeps is one reference per object it uses.
    ///
    /// Err: [`Error::Quiesced`]; [`Refusal::WaitOnOwnThread`] in a handler or a listener of its
    /// device where the queue's maximum count are made and not yet completed.
    pub fn command_buffer(&self) -> Result<CommandBuffer, Error> {
        let core = self.device.core();
        let mut live = lock(&self.inner.live);
        loop {
            if core.closed() {
                return Err(Error::Quiesced);
            }
            if live.count < self.inner.max {
                break;
            }
            if crate::device::on_handler_thread(core) {
                return Err(Error::Refused(Refusal::WaitOnOwnThread));
            }
            live = self
                .inner
                .room
                .wait(live)
                .unwrap_or_else(PoisonError::into_inner);
        }
        live.count += 1;
        let kept = live.objects.pop().unwrap_or_default();
        drop(live);
        core.counters.rise(Gauge::CommandBuffers);
        Ok(CommandBuffer::new(
            Arc::clone(&self.inner),
            self.device.clone(),
            kept,
        ))
    }

    /// No Metal counterpart: ends every command buffer of the queue the submit thread holds at an
    /// event wait or has not yet scheduled with
    /// [`CommandBufferError::NotRun`](crate::CommandBufferError::NotRun), and runs their handlers,
    /// as a Mac drops a dead process's held command buffers. One enqueued and not yet committed
    /// ends so at its commit. Segments already submitted still complete, and so does a group of
    /// scheduled command buffers the queue holds while three of its submissions run. Each ended
    /// command buffer retires in the queue's order, behind its work held or in flight. The queue
    /// stays usable.
    pub fn abandon(&self) {
        let core = self.device.core();
        core.work.abandon(core, &self.inner);
    }

    /// No Metal counterpart: the wait hint. Where `command_buffer`, of this queue, is committed and
    /// not yet submitted, its group goes now, as a host wait for it sends it
    /// ([`CommandBuffer::wait_until_completed`]): held in a group while three of the queue's
    /// submissions run, or committed behind such a group, or at an event wait (then its group goes
    /// as it is staged). For a consumer that learns of a wait the runtime does not see.
    ///
    /// Callable from any thread: it takes the submit side's lock briefly and waits for no GPU
    /// work. A no-op for a command buffer already submitted, completed or ended, not yet
    /// committed, or of another queue: it makes no submission.
    pub fn send_now(&self, command_buffer: &CommandBuffer) {
        if command_buffer.is_of(&self.inner) {
            command_buffer.send();
        }
    }

    /// No Metal counterpart: the device the queue was made on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl QueueInner {
    /// The GPU run, in nanoseconds, its next command buffer is expected to take: its command
    /// buffers' own, by their timestamps, as each retires ([`ran`](Self::ran)); None before one
    /// retired with its GPU times.
    pub(crate) fn expected_run(&self) -> Option<u64> {
        Some(self.run.load(Ordering::Relaxed)).filter(|&run| run != u64::MAX)
    }

    /// One of its command buffers ran `nanos` on the GPU: the expected run rises to a longer one
    /// at once and halves toward a shorter one ([`next_run`]).
    pub(crate) fn ran(&self, nanos: u64) {
        let _ = (self.run).fetch_update(Ordering::Relaxed, Ordering::Relaxed, |run| {
            Some(next_run((run != u64::MAX).then_some(run), nanos))
        });
    }

    /// Whether `queue` is this queue.
    pub(crate) fn is(&self, queue: &QueueInner) -> bool {
        std::ptr::eq(self, queue)
    }

    /// The device, while it lives.
    pub(crate) fn core(&self) -> Option<Arc<Core>> {
        self.core.upgrade()
    }

    /// A command buffer of the queue retired, or was dropped uncommitted: its place goes back,
    /// and `kept`, its kept-object lists emptied, for the next command buffer made. Its gauge
    /// falls first, so a command buffer made in the freed place never reads one more live than
    /// the queue holds.
    pub(crate) fn release(&self, kept: Objects) {
        self.counters.fall(Gauge::CommandBuffers);
        let mut live = lock(&self.live);
        live.count -= 1;
        live.objects.push(kept);
        drop(live);
        self.room.notify_one();
    }
}

/// The expected run after a command buffer that ran `nanos`, from the one before (`None` for
/// none): a longer run raises it at once, so a queue whose work grows stops grouping at its first
/// long command buffer; a shorter one halves the way toward it, so one short command buffer among
/// long ones does not start grouping them again.
pub(crate) fn next_run(expected: Option<u64>, nanos: u64) -> u64 {
    match expected {
        Some(expected) if nanos < expected => expected - (expected - nanos) / 2,
        _ => nanos.min(u64::MAX - 1),
    }
}
