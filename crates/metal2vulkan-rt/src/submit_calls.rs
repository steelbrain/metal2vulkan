//! The runtime's `vkQueueSubmit2` calls inside the driver, as the block-maker thread's memory
//! calls meet them.
//!
//! A driver may hold a submit made while another thread is inside `vkAllocateMemory` or
//! `vkFreeMemory` until that call returns, and hold it through calls made back to back (NVIDIA's
//! Linux driver 620.70.06 does, device-local memory included). So after each of its allocations
//! and frees the block-maker thread waits until every submit call that had entered the driver
//! when its own call returned has come back ([`SubmitCalls::wait_out`]); where none was inside
//! the driver then, it goes on at once. A submit then waits out at most one of its calls, the
//! one it met. A submit never waits for the block-maker thread: its call counts itself in and out
//! under a lock held for an add, and wakes the thread only while it waits. A take that allocates
//! on the consumer's thread makes a single call, and waits for nothing.

use crate::counters::Counters;
use crate::device::lock;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, PoisonError};

/// The device's submit calls inside the driver, and whether the block-maker thread is inside a
/// `vkFreeMemory`.
pub(crate) struct SubmitCalls {
    calls: Mutex<Calls>,
    /// Signalled as a call returns while a thread waits for one ([`Calls::waiting`]).
    returned: Condvar,
    /// Whether the block-maker thread is inside a `vkFreeMemory`, which a driver may hold a
    /// submission made now for ([`Metrics::submissions_during_frees`]).
    ///
    /// [`Metrics::submissions_during_frees`]: crate::Metrics::submissions_during_frees
    freeing: AtomicBool,
    /// The device's counters: the block-maker thread's calls and its waits.
    counters: Counters,
    /// For the device tests: while set, the block-maker thread waits inside each of its frees,
    /// with [`freeing`](Self::freeing) set, before it calls `vkFreeMemory`.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    hold_frees: Mutex<bool>,
}

/// The submit calls counted in and out.
#[derive(Default)]
struct Calls {
    /// The calls that entered the driver.
    entered: u64,
    /// The calls that returned from it.
    returned: u64,
    /// The threads waiting for [`returned`](Self::returned) to reach a count.
    waiting: usize,
}

/// A submit call inside the driver: counted out as it drops, once the call returned.
pub(crate) struct InDriver<'a>(&'a SubmitCalls);

impl Drop for InDriver<'_> {
    fn drop(&mut self) {
        let calls = &self.0;
        let mut counts = lock(&calls.calls);
        counts.returned += 1;
        if counts.waiting > 0 {
            calls.returned.notify_all();
        }
    }
}

impl SubmitCalls {
    /// None inside the driver, for a new device whose counters are `counters`.
    pub(crate) fn new(counters: Counters) -> SubmitCalls {
        SubmitCalls {
            calls: Mutex::new(Calls::default()),
            returned: Condvar::new(),
            freeing: AtomicBool::new(false),
            counters,
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            hold_frees: Mutex::new(false),
        }
    }

    /// Counts in a `vkQueueSubmit2` call made at once after this: it is counted out as the
    /// returned guard drops, after the call returned, whatever it returned.
    pub(crate) fn enter(&self) -> InDriver<'_> {
        lock(&self.calls).entered += 1;
        InDriver(self)
    }

    /// On the block-maker thread, as one of its `vkAllocateMemory` or `vkFreeMemory` calls
    /// returned: waits until every submit call counted in by now has returned, blocking, for no
    /// longer than those calls take; with none inside the driver, it returns at once. The call is
    /// counted in [`Metrics::maker_memory_calls`], and a wait in [`Metrics::maker_yields`] and
    /// [`Metrics::maker_yield_nanos`].
    ///
    /// [`Metrics::maker_memory_calls`]: crate::Metrics::maker_memory_calls
    /// [`Metrics::maker_yields`]: crate::Metrics::maker_yields
    /// [`Metrics::maker_yield_nanos`]: crate::Metrics::maker_yield_nanos
    pub(crate) fn wait_out(&self) {
        let mut counts = lock(&self.calls);
        let entered = counts.entered;
        if counts.returned >= entered {
            drop(counts);
            self.counters.maker_memory_call(None);
            return;
        }
        let since = crate::host_time();
        counts.waiting += 1;
        while counts.returned < entered {
            counts = (self.returned.wait(counts)).unwrap_or_else(PoisonError::into_inner);
        }
        counts.waiting -= 1;
        drop(counts);
        let nanos = crate::host_time().saturating_sub(since);
        self.counters.maker_memory_call(Some(nanos));
    }

    /// On the block-maker thread: `free`, its `vkFreeMemory`, with
    /// [`freeing`](Self::freeing) set for its length, then [`wait_out`](Self::wait_out).
    pub(crate) fn free(&self, free: impl FnOnce()) {
        self.freeing.store(true, Relaxed);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        while *lock(&self.hold_frees) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        free();
        self.freeing.store(false, Relaxed);
        self.wait_out();
    }

    /// Whether the block-maker thread is inside a `vkFreeMemory`, which a driver may hold a
    /// submission made now for ([`Metrics::submissions_during_frees`]).
    ///
    /// [`Metrics::submissions_during_frees`]: crate::Metrics::submissions_during_frees
    pub(crate) fn freeing(&self) -> bool {
        self.freeing.load(Relaxed)
    }

    /// For the tests: the threads waiting for a submit call to return.
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        lock(&self.calls).waiting
    }

    /// For the device tests: holds each of the block-maker thread's frees, with
    /// [`freeing`](Self::freeing) set, while `hold` is true.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn hold_frees(&self, hold: bool) {
        *lock(&self.hold_frees) = hold;
    }
}
