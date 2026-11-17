//! Command queues: Metal's bound on a queue's uncompleted command buffers, and abandoning a queue.

use crate::command::CommandBuffer;
use crate::counters::Gauge;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Refusal};
use crate::reaper::{Free, Job};
use crate::times::Stamps;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

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
    pub(crate) core: Arc<Core>,
}

/// A queue's slots: each command buffer made and not yet retired holds one, and with it its pair
/// of timestamp queries.
struct Slots {
    /// Slots below it have been taken at least once.
    next: u32,
    /// Slots given back.
    free: Vec<u32>,
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
                }),
                room: Condvar::new(),
                stamps: Stamps::new(core, max_command_buffer_count)?,
                core: Arc::clone(core),
            }),
            device: self.clone(),
        })
    }
}

impl CommandQueue {
    /// `commandBuffer`: waits, as Metal does, while the queue's maximum count of its command
    /// buffers are made and not yet completed. `commandBufferWithUnretainedReferences` is the same
    /// call: what a command buffer keeps is one reference per block it uses.
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
        drop(slots);
        core.counters.rise(Gauge::CommandBuffers);
        Ok(CommandBuffer::new(
            Arc::clone(&self.inner),
            self.device.clone(),
            slot,
        ))
    }

    /// No Metal counterpart: ends every command buffer of the queue the submit thread holds or has
    /// not yet submitted with [`CommandBufferError::NotRun`](crate::CommandBufferError::NotRun),
    /// and runs their handlers, as a Mac drops a dead process's held command buffers. One enqueued
    /// and not yet committed ends so at its commit. Segments already submitted still complete.
    /// The queue stays usable.
    pub fn abandon(&self) {
        let core = self.device.core();
        core.work.abandon(core, &self.inner);
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

    /// The command buffer holding `slot` retired, or was dropped uncommitted: the slot goes back.
    pub(crate) fn release(&self, slot: u32) {
        lock(&self.slots).free.push(slot);
        self.room.notify_one();
        self.core.counters.fall(Gauge::CommandBuffers);
    }
}

impl Drop for QueueInner {
    fn drop(&mut self) {
        // No command buffer of the queue is left, so no work names the pool.
        self.core
            .reaper
            .give(Job::Free(Free::QueryPool(self.stamps.pool)));
    }
}
