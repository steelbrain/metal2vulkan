//! Command queues: Metal's bound on a queue's uncompleted command buffers, and abandoning a queue.

use crate::command::CommandBuffer;
use crate::counters::{Counters, Gauge};
use crate::device::{lock, Core, Device};
use crate::error::{Error, Refusal};
use crate::kept::Objects;
use crate::reaper::{Free, Job};
use crate::times::Stamps;
use std::sync::atomic::AtomicU64;
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
    /// The slots of the command buffers made and not yet retired.
    slots: Mutex<Slots>,
    /// Signalled when one retires.
    room: Condvar,
    /// Two timestamp queries per slot.
    pub(crate) stamps: Stamps,
    /// The latest GPU end time of its command buffers retired, in host nanoseconds: where a
    /// command buffer with nothing for the GPU places its times.
    pub(crate) last_end: AtomicU64,
    counters: Counters,
    /// The device, without keeping it: a command buffer handle that outlives the device's
    /// handles holds its queue, and neither keeps the device from its teardown.
    core: Weak<Core>,
}

/// A queue's slots: each command buffer made and not yet retired holds one, and with it its pair
/// of timestamp queries.
struct Slots {
    /// Slots below it have been taken at least once.
    next: u32,
    /// Slots given back.
    free: Vec<u32>,
    /// The kept-object lists of retired command buffers, empty, with their capacity
    /// (`crate::kept`): at most one per slot.
    objects: Vec<Objects>,
}

impl Slots {
    fn taken(&self) -> u32 {
        self.next - self.free.len() as u32
    }
}

impl Device {
    /// `newCommandQueueWithMaxCommandBufferCount:`; `newCommandQueue` is this call with 64.
    ///
    /// The queue holds a pool of two timestamp queries per command buffer it may have made at
    /// once, for their GPU times.
    ///
    /// Err: [`Refusal::NoCommandBuffers`] for a count of 0; [`Error::Vulkan`] where the pool
    /// cannot be made.
    pub fn new_command_queue(&self, max_command_buffer_count: u32) -> Result<CommandQueue, Error> {
        if max_command_buffer_count == 0 {
            return Err(Error::Refused(Refusal::NoCommandBuffers));
        }
        let core = self.core();
        Ok(CommandQueue {
            inner: Arc::new(QueueInner {
                max: max_command_buffer_count,
                slots: Mutex::new(Slots {
                    next: 0,
                    free: Vec::new(),
                    objects: Vec::new(),
                }),
                room: Condvar::new(),
                stamps: Stamps::new(core, max_command_buffer_count)?,
                last_end: AtomicU64::new(0),
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
    /// Err: [`Error::Quiesced`].
    pub fn command_buffer(&self) -> Result<CommandBuffer, Error> {
        let core = self.device.core();
        let mut slots = lock(&self.inner.slots);
        loop {
            if core.closed() {
                return Err(Error::Quiesced);
            }
            if slots.taken() < self.inner.max {
                break;
            }
            slots = self
                .inner
                .room
                .wait(slots)
                .unwrap_or_else(PoisonError::into_inner);
        }
        let slot = match slots.free.pop() {
            Some(slot) => slot,
            None => {
                slots.next += 1;
                slots.next - 1
            }
        };
        let kept = slots.objects.pop().unwrap_or_default();
        drop(slots);
        core.counters.rise(Gauge::CommandBuffers);
        Ok(CommandBuffer::new(
            Arc::clone(&self.inner),
            self.device.clone(),
            slot,
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
    /// Whether `queue` is this queue.
    pub(crate) fn is(&self, queue: &QueueInner) -> bool {
        std::ptr::eq(self, queue)
    }

    /// The device, while it lives.
    pub(crate) fn core(&self) -> Option<Arc<Core>> {
        self.core.upgrade()
    }

    /// The command buffer holding `slot` retired, or was dropped uncommitted: the slot goes back,
    /// and `kept`, its kept-object lists emptied, for the next command buffer made. Its gauge falls
    /// first, so a command buffer made in the freed slot never reads one more live than the queue
    /// holds.
    pub(crate) fn release(&self, slot: u32, kept: Objects) {
        self.counters.fall(Gauge::CommandBuffers);
        let mut slots = lock(&self.slots);
        slots.free.push(slot);
        slots.objects.push(kept);
        drop(slots);
        self.room.notify_one();
    }
}

impl Drop for QueueInner {
    fn drop(&mut self) {
        // No command buffer of the queue is left, so no work names the pool. Where the device is
        // already gone, its teardown destroyed the pool.
        if let Some(core) = self.core.upgrade() {
            core.reaper
                .give(Job::Free(Free::QueryPool(self.stamps.pool)));
        }
    }
}
