//! Fences: Metal's order between encoders, of one queue or of two, which the device's one queue
//! order and the barriers already give, so a fence records no device work.

use crate::device::{Core, Device};
use crate::error::{Error, Refusal};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// `MTLFence`: an encoder that waits on it sees what the encoders up to each update submitted
/// before it wrote, on its own queue or another.
///
/// The runtime records nothing for a fence. Every queue's command buffers go to the device's one
/// `VkQueue`, and each encoder's writes reach every command submitted after it through the barrier
/// at its end or at its segment's end, so a wait sees every update submitted before it. A command
/// buffer the submit thread holds while its queue's submissions run, in a group or scheduled and
/// waiting for room, is submitted, as Metal would have submitted it, before a command buffer of
/// another queue that waits on a fence it updates, with its queue's command buffers before it: one
/// held for room so after the waiter was is not waited for. A waiter with nothing for the GPU
/// completes no sooner than the last submission that
/// updates its fence, as Metal's does. An update in a command buffer held at an event wait, or
/// behind one held so or enqueued and not committed, is not waited for: Metal evaluates a wait at
/// its kernel submission against the updates already submitted, and a waiter on another queue runs
/// at once, without the held update's writes (probe `cross_queue_fence_held_update.m` in the
/// conformance crate, an Apple M3 Max, macOS 27.0.1, jobs m3-max-52186, m3-max-52188 and
/// m3-max-52194). What differs: the waiter with nothing for the GPU reports one GPU time, where
/// Metal reports a span over the update's run; and work with no fence between it is ordered too,
/// where Metal may run it at once (an empty dispatch committed after a busy one runs beside it on
/// an M3 Max, probe `empty_command_buffer_times.m`; a dispatch on another queue does too, job
/// m3-max-52188). A cheap handle: clones share one fence. It does not hold the device.
#[derive(Clone)]
pub struct Fence {
    inner: Arc<FenceInner>,
}

/// The fence itself: its device, held by a weak reference that keeps its address from being
/// reused, so a comparison by address is sound.
struct FenceInner {
    core: Weak<Core>,
    /// The progress value of the last submission with a command buffer that updates it, written
    /// and read by the submit thread alone (`crate::submit`).
    submitted: AtomicU64,
}

impl Device {
    /// `newFence`. It makes no Vulkan object ([`Fence`]).
    pub fn new_fence(&self) -> Fence {
        Fence {
            inner: Arc::new(FenceInner {
                core: Arc::downgrade(self.core()),
                submitted: AtomicU64::new(0),
            }),
        }
    }
}

impl Fence {
    /// An encoder on the device of `core` updates or waits on the fence.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's fence.
    pub(crate) fn check(&self, core: &Arc<Core>) -> Result<(), Error> {
        match std::ptr::eq(Weak::as_ptr(&self.inner.core), Arc::as_ptr(core)) {
            true => Ok(()),
            false => Err(Error::Refused(Refusal::OtherDevice)),
        }
    }

    fn is(&self, other: &Fence) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// The fences a command buffer's encoders wait on and update, each once, kept until it retires.
/// Empty, with nothing allocated, for a command buffer that uses none.
#[derive(Default)]
pub(crate) struct Fences {
    waited: Vec<Fence>,
    updated: Vec<Fence>,
}

impl Fences {
    /// An encoder waits on `fence`.
    pub(crate) fn wait(&mut self, fence: &Fence) {
        if !self.waited.iter().any(|kept| kept.is(fence)) {
            self.waited.push(fence.clone());
        }
    }

    /// An encoder updates `fence`.
    pub(crate) fn update(&mut self, fence: &Fence) {
        if !self.updated.iter().any(|kept| kept.is(fence)) {
            self.updated.push(fence.clone());
        }
    }

    /// Whether an encoder waits on a fence.
    pub(crate) fn waits(&self) -> bool {
        !self.waited.is_empty()
    }

    /// Whether an encoder updates a fence an encoder of `waiter` waits on.
    pub(crate) fn updates_for(&self, waiter: &Fences) -> bool {
        (self.updated.iter()).any(|fence| waiter.waited.iter().any(|kept| kept.is(fence)))
    }

    /// The submission of progress `value` carries the command buffer: each fence it updates was
    /// last updated there. Called by the submit thread alone, in the order of the values.
    pub(crate) fn submitted(&self, value: u64) {
        for fence in &self.updated {
            fence.inner.submitted.store(value, Ordering::Relaxed);
        }
    }

    /// The progress value of the last submission that updates a fence an encoder waits on: 0
    /// where none was submitted.
    pub(crate) fn waited_through(&self) -> u64 {
        (self.waited.iter())
            .map(|fence| fence.inner.submitted.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0)
    }
}
