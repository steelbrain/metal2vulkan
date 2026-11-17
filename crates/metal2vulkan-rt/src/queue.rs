//! Command queues: Metal's bound on a queue's uncompleted command buffers, and abandoning a queue.

use crate::command::CommandBuffer;
use crate::counters::{Counters, Gauge};
use crate::device::{lock, Device};
use crate::error::{Error, Refusal};
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
    /// Command buffers made and not yet retired.
    count: Mutex<u32>,
    /// Signalled when one retires.
    room: Condvar,
    counters: Counters,
}

impl Device {
    /// `newCommandQueueWithMaxCommandBufferCount:`; `newCommandQueue` is this call with 64.
    ///
    /// Err: [`Refusal::NoCommandBuffers`] for a count of 0.
    pub fn new_command_queue(&self, max_command_buffer_count: u32) -> Result<CommandQueue, Error> {
        if max_command_buffer_count == 0 {
            return Err(Error::Refused(Refusal::NoCommandBuffers));
        }
        Ok(CommandQueue {
            inner: Arc::new(QueueInner {
                max: max_command_buffer_count,
                count: Mutex::new(0),
                room: Condvar::new(),
                counters: self.counters(),
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
        let mut count = lock(&self.inner.count);
        loop {
            if core.closed() {
                return Err(Error::Quiesced);
            }
            if *count < self.inner.max {
                break;
            }
            count = self
                .inner
                .room
                .wait(count)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *count += 1;
        drop(count);
        self.inner.counters.rise(Gauge::CommandBuffers);
        Ok(CommandBuffer::new(
            Arc::clone(&self.inner),
            self.device.clone(),
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

    /// One of the queue's command buffers retired, or was dropped uncommitted.
    pub(crate) fn release(&self) {
        *lock(&self.count) -= 1;
        self.room.notify_one();
        self.counters.fall(Gauge::CommandBuffers);
    }
}
