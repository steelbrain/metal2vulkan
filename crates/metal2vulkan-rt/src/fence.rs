//! Fences: Metal's order between one queue's encoders, which the runtime's commit order and
//! barriers already give, so a fence records no device work.

use crate::device::{Core, Device};
use crate::error::{Error, Gap, Refusal};
use crate::queue::QueueInner;
use std::sync::{Arc, OnceLock, Weak};

/// `MTLFence`: the encoders that wait on it after an update see what the encoders up to the
/// update wrote.
///
/// The runtime records nothing for a fence. A command buffer's encoders run in their order and a
/// queue's command buffers in their commit order, each encoder's writes reaching the commands
/// after it through the barrier at its end or at its segment's end, so a wait after an update
/// already sees them. What differs: work with no fence between it is ordered too, where Metal may
/// run it at once (an empty dispatch committed after a busy one runs beside it on an M3 Max, probe
/// `empty_command_buffer_times.m`). Metal's fence orders encoders of one queue (`MTLEvent` orders
/// queues): the fence keeps the first queue whose encoder uses it, and a use on another queue is
/// [`Gap::FenceOtherQueue`]. A cheap handle: clones share one fence. It does not hold the device.
#[derive(Clone)]
pub struct Fence {
    inner: Arc<FenceInner>,
}

/// The fence itself: its device and the first queue that used it, each held by a weak reference
/// that keeps its address from being reused, so a comparison by address is sound.
struct FenceInner {
    core: Weak<Core>,
    queue: OnceLock<Weak<QueueInner>>,
}

impl Device {
    /// `newFence`. It makes no Vulkan object ([`Fence`]).
    pub fn new_fence(&self) -> Fence {
        Fence {
            inner: Arc::new(FenceInner {
                core: Arc::downgrade(self.core()),
                queue: OnceLock::new(),
            }),
        }
    }
}

impl Fence {
    /// An encoder of `queue`, on the device of `core`, updates or waits on the fence: the fence
    /// keeps the queue if it is the first.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's fence; [`Gap::FenceOtherQueue`] for a
    /// fence another queue used first.
    pub(crate) fn used_on(&self, core: &Arc<Core>, queue: &Arc<QueueInner>) -> Result<(), Error> {
        if !std::ptr::eq(Weak::as_ptr(&self.inner.core), Arc::as_ptr(core)) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        let first = self.inner.queue.get_or_init(|| Arc::downgrade(queue));
        match std::ptr::eq(Weak::as_ptr(first), Arc::as_ptr(queue)) {
            true => Ok(()),
            false => Err(Error::Unsupported(Gap::FenceOtherQueue)),
        }
    }
}
