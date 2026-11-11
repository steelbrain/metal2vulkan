//! The calling thread's identity, read at any point of the thread's life.
//!
//! A program's exit handlers run on the thread that called `exit`, after the C library destroyed
//! that thread's thread-locals (macOS's `exit` and the GNU C library's run their destructors
//! first): a thread-local that needs dropping is gone there, read with `with` it panics, and so
//! does `std::thread::current` once the thread's handle went with them. In an exit handler, which
//! cannot unwind, the panic ends the process. Each thread-local of the runtime's that a program's
//! call reaches is read with `try_with` or holds nothing to drop, and the calls ask this module,
//! not `std::thread::current`, which thread they run on.

use std::thread::JoinHandle;

/// This thread's identity among the live threads: the address of a thread-local of nothing to
/// drop, which stays readable to the thread's end. No two live threads share one; a thread that
/// starts after another ended may take its address.
pub(crate) fn id() -> usize {
    thread_local! {
        static HERE: u8 = const { 0 };
    }
    HERE.with(|here| std::ptr::from_ref(here).addr())
}

/// Whether `handle`'s thread is this one: a thread of the runtime's own whose teardown it runs,
/// which it does not join.
pub(crate) fn runs<T>(handle: &JoinHandle<T>) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::thread::JoinHandleExt;
        // SAFETY: reads of two threads' identifiers; `handle`'s thread is neither joined nor
        // detached while the handle lives.
        unsafe { libc::pthread_equal(handle.as_pthread_t(), libc::pthread_self()) != 0 }
    }
    #[cfg(not(unix))]
    {
        handle.thread().id() == std::thread::current().id()
    }
}
