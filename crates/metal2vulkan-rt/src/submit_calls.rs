//! The runtime's `vkQueueSubmit2` calls inside the driver, as the block-maker thread's memory
//! calls meet them.
//!
//! A driver may hold a submit made while another thread is inside `vkAllocateMemory` or
//! `vkFreeMemory` until that call returns, and hold it through calls made back to back (NVIDIA's
//! Linux driver 620.70.06 does, device-local memory included). So as each of its allocations and
//! frees returns, the block-maker thread marks the submit calls inside the driver then, and before
//! its next such call it waits until every one of them has come back ([`SubmitCalls::call`]);
//! where none is left inside, it goes straight on. A submit then waits out at most one of its
//! calls, the one it met, and what the thread made is handed on as soon as its call returned. A
//! submit never waits for the block-maker thread: its call counts itself in and out under a lock
//! held for an add, and wakes the thread only while it waits. A take that allocates on the
//! consumer's thread makes a single call, and waits for nothing.

use crate::counters::Counters;
use crate::device::lock;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, PoisonError};

/// The device's submit calls inside the driver, the ones the block-maker thread's last memory call
/// met, and whether that thread is inside a `vkFreeMemory`.
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
    /// The calls that had entered as the block-maker thread's last memory call returned: its
    /// next one starts once this many have returned.
    met: u64,
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

    /// On the block-maker thread, `call`, one `vkAllocateMemory`: first [`wait_out`] the submit
    /// calls its last memory call met, then the call, then marks the submit calls inside the
    /// driver as it returned.
    ///
    /// [`wait_out`]: Self::wait_out
    pub(crate) fn call<R>(&self, call: impl FnOnce() -> R) -> R {
        self.wait_out();
        let made = call();
        self.mark();
        made
    }

    /// On the block-maker thread, `free`, one `vkFreeMemory`: as [`call`](Self::call), with
    /// [`freeing`](Self::freeing) set for the free's own length alone.
    pub(crate) fn free(&self, free: impl FnOnce()) {
        self.wait_out();
        self.freeing.store(true, Relaxed);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        while *lock(&self.hold_frees) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        free();
        self.freeing.store(false, Relaxed);
        self.mark();
    }

    /// Before one of the block-maker thread's memory calls: waits until every submit call its
    /// last memory call met has returned, blocking, for no longer than those calls take; with none
    /// left inside the driver, it returns at once. The call is counted in
    /// [`Metrics::maker_memory_calls`], and a wait in [`Metrics::maker_yields`] and
    /// [`Metrics::maker_yield_nanos`].
    ///
    /// [`Metrics::maker_memory_calls`]: crate::Metrics::maker_memory_calls
    /// [`Metrics::maker_yields`]: crate::Metrics::maker_yields
    /// [`Metrics::maker_yield_nanos`]: crate::Metrics::maker_yield_nanos
    fn wait_out(&self) {
        let mut counts = lock(&self.calls);
        let met = counts.met;
        if counts.returned >= met {
            drop(counts);
            self.counters.maker_memory_call(None);
            return;
        }
        let since = crate::host_time();
        counts.waiting += 1;
        while counts.returned < met {
            counts = (self.returned.wait(counts)).unwrap_or_else(PoisonError::into_inner);
        }
        counts.waiting -= 1;
        drop(counts);
        let nanos = crate::host_time().saturating_sub(since);
        self.counters.maker_memory_call(Some(nanos));
    }

    /// As one of the block-maker thread's memory calls returned: the submit calls inside the
    /// driver now are the ones its next call waits out.
    fn mark(&self) {
        let mut counts = lock(&self.calls);
        counts.met = counts.entered;
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
