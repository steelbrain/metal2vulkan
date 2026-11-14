//! The setters' writes of an indirect command buffer's records made without its lock (`Batch`),
//! and a device recovery's carry of the buffer, which waits out those of a buffer of pooled
//! memory ([`Writers`]).
//!
//! A write without the lock reads the records' bytes and writes them. A recovery's carry of a
//! buffer whose records lie in the runtime's own pages, as they do on a device that imports host
//! pointers (`fresh_pages.rs`), imports the same pages again: the bytes and their address stay,
//! so such a write lands in the records the carry keeps, before it or after it, and nothing here
//! looks at it. A carry of a buffer of pooled memory, on a device that imports none or where the
//! import failed, moves its records to new memory and resets every command there, and the lost
//! generation's memory goes at that generation's rest, which can come at once (`carry.rs`). So a
//! write to such a buffer that read the bytes, or looked at what the command holds, before the
//! move and wrote after it would write the lost generation's memory, memory let go of, or a
//! command the carry reset. Instead each such write, from its look at what the command holds to
//! its last store, counts itself under way in the buffer by an atomic add; the carry marks
//! itself begun, then waits until none is under way. A write that finds a carry begun and not
//! ended is not made without the lock, and goes under it instead, after the carry. Each write
//! lands before the carry or after it.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// A carry begun, in [`Writers::carries`]: above the carries under way.
const BEGUN: u64 = 1 << 32;

/// One buffer's writes without its lock and its carries (the module docs).
pub(super) struct Writers {
    /// Whether the buffer's records lie in pooled memory, which a carry moves: its writes
    /// without the lock count themselves. Fixed at the buffer's making.
    counted: bool,
    /// The carries of the buffer begun so far, above those under way (in the low 32 bits), while
    /// one of which no write without the lock begins. A write reads the records' bytes again
    /// only where a carry began since it read them last ([`Writing::carries_begun`]).
    carries: AtomicU64,
    /// The writes without the lock under way.
    under_way: AtomicU32,
}

impl Writers {
    /// The writers of a buffer whose records lie in pooled memory (`pooled`), or in the
    /// runtime's own pages.
    pub(super) fn new(pooled: bool) -> Writers {
        Writers {
            counted: pooled,
            carries: AtomicU64::new(0),
            under_way: AtomicU32::new(0),
        }
    }

    /// Whether the buffer's records lie in pooled memory, whose writes without the lock count
    /// themselves ([`enter`](Self::enter)); else they need nothing of this.
    #[inline]
    pub(super) fn counted(&self) -> bool {
        self.counted
    }

    /// A write of a buffer of pooled memory without its lock, from now until the guard goes:
    /// none where a carry of it is under way, the write then to go under the lock.
    pub(super) fn enter(&self) -> Option<Writing<'_>> {
        self.under_way.fetch_add(1, Ordering::SeqCst);
        let carries = self.carries.load(Ordering::SeqCst);
        if !carries.is_multiple_of(BEGUN) {
            self.under_way.fetch_sub(1, Ordering::Release);
            return None;
        }
        Some(Writing {
            under_way: &self.under_way,
            begun: carries / BEGUN,
        })
    }

    /// Stop the writes without the lock of a buffer of pooled memory for a carry, until the guard
    /// goes: none begins, and those under way have ended. None for a buffer in the runtime's own
    /// pages, whose writes need no stop.
    pub(super) fn stop(&self) -> Option<Stopped<'_>> {
        if !self.counted {
            return None;
        }
        self.carries.fetch_add(BEGUN + 1, Ordering::SeqCst);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if self.under_way.load(Ordering::SeqCst) != 0 {
            waits_hook();
        }
        while self.under_way.load(Ordering::SeqCst) != 0 {
            std::thread::yield_now();
        }
        Some(Stopped(self))
    }
}

/// A write without the lock under way ([`Writers::enter`]).
pub(super) struct Writing<'a> {
    /// The count it is under way in.
    under_way: &'a AtomicU32,
    /// The carries of the buffer begun before it, all ended.
    begun: u64,
}

impl Writing<'_> {
    /// The carries of the buffer begun before the write, all ended: the records' bytes read in
    /// an earlier write that found as many are the bytes now.
    pub(super) fn carries_begun(&self) -> u64 {
        self.begun
    }
}

impl Drop for Writing<'_> {
    fn drop(&mut self) {
        self.under_way.fetch_sub(1, Ordering::Release);
    }
}

/// A carry's stop of the writes without the lock ([`Writers::stop`]).
pub(super) struct Stopped<'a>(&'a Writers);

impl Drop for Stopped<'_> {
    fn drop(&mut self) {
        self.0.carries.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
thread_local! {
    /// The hook this thread's next carry runs where it finds a write under way.
    static WAITS_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Have this thread's next carry that finds a write without the lock under way run `hook`
/// before it waits: a device test's proof that the carry waits (`tests::icb_carry_race`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) fn on_carry_waits(hook: impl FnOnce() + 'static) {
    WAITS_HOOK.with(|armed| *armed.borrow_mut() = Some(Box::new(hook)));
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
fn waits_hook() {
    if let Some(hook) = WAITS_HOOK.with(|armed| armed.borrow_mut().take()) {
        hook();
    }
}
