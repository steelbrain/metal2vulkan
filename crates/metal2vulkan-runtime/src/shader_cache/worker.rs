//! The cache's background thread, one per device: the entries the directory is given
//! ([`Store::write_queued`]), the scan at device creation ([`Store::sweep`]) and the scans that
//! bring the directory back under its cap ([`Store::evict`]), off every program's thread; nothing
//! on a pipeline's or a command's path waits for them. (The saves of the device's one driver cache
//! it made, went with that cache; each making now writes its own record, `driver.rs`.)
//!
//! - **What a creation hands over.** An entry's bytes are encoded on the creating thread, which
//!   needs their length, and handed here as a [`Write`]: the `mkdir`, the temporary file, the
//!   write and the `rename` are this thread's. The commit point is still the `rename`, so a reader
//!   of another process reads the whole old entry or the whole new one.
//! - **Bounded.** The queue holds at most [`QUEUED_WRITES`] entries and [`QUEUED_BYTES`] bytes
//!   together, the one being written included. A creation that meets either bound, or whose entry
//!   is larger than the whole bound, writes it on its own thread, exactly as every creation did
//!   before: the bound costs the creation its write, never the entry.
//! - **One write per name.** An entry queued again under a name already queued replaces its bytes
//!   rather than queueing a second write, so two creations racing on one key leave one write and
//!   one `rename`.
//! - **Visible at once.** A queued entry answers this device's lookups from the queue
//!   ([`Asked::pending`]), so a lookup that would have read the file it has not written yet hits
//!   rather than compiling again.
//! - **Drained when the device goes.** [`Worker::stop`] writes what is queued before the thread
//!   ends, so an entry a creation handed over is on disk by the time the device's drop returns.
//!   A process that ends without dropping its devices (an exit, or `exit_guard`'s `end_at_once`
//!   past its bound) loses what is
//!   still queued: at most the bound, each of them an entry the next process compiles again, as it
//!   would have compiled anything this process had not reached yet.

use super::store::Store;
use crate::device::lock;
use crate::error::{Error, MemoryKind};
use crate::exit_guard;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// The most entries the queue holds at once, the one being written included.
pub(crate) const QUEUED_WRITES: usize = 64;

/// The most bytes the queue holds at once, the one being written included: 16 MiB.
pub(crate) const QUEUED_BYTES: u64 = 16 << 20;

/// An entry to write: where it goes, and its bytes.
pub(crate) struct Write {
    pub(crate) path: PathBuf,
    pub(crate) bytes: Vec<u8>,
}

impl Write {
    pub(crate) fn new(path: PathBuf, bytes: Vec<u8>) -> Write {
        Write { path, bytes }
    }
}

/// What the cache thread is asked to do: the entries waiting to be written, and the scans. The
/// store holds it to queue writes and to answer lookups from the queue; the thread holds it to
/// take them.
pub(crate) struct Asked {
    work: Mutex<Work>,
    wake: Condvar,
    /// Set when the device goes: a running scan stops early. The queued writes are still
    /// written.
    stop: AtomicBool,
}

#[derive(Default)]
struct Work {
    evict: bool,
    stopping: bool,
    /// The entries queued, in the order they were handed over, at most [`QUEUED_WRITES`].
    queued: VecDeque<Arc<Write>>,
    /// The one the thread is writing now, which lookups still see.
    writing: Option<Arc<Write>>,
    /// The bytes of `queued` and `writing` together, at most [`QUEUED_BYTES`].
    bytes: u64,
}

impl Asked {
    /// Nothing asked for yet.
    pub(crate) fn new() -> Asked {
        Asked {
            work: Mutex::new(Work::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
        }
    }

    /// Queue `write` for the thread, waking it; `Some(write)` where the queue is at a bound and
    /// the caller writes it itself. An entry under a path already queued replaces its bytes.
    pub(crate) fn queue(&self, write: Write) -> Option<Write> {
        let length = write.bytes.len() as u64;
        let mut work = lock(&self.work);
        if let Some(at) = work
            .queued
            .iter()
            .position(|queued| queued.path == write.path)
        {
            work.bytes = work.bytes - work.queued[at].bytes.len() as u64 + length;
            work.queued[at] = Arc::new(write);
            self.wake.notify_all();
            return None;
        }
        if work.queued.len() + usize::from(work.writing.is_some()) >= QUEUED_WRITES
            || work.bytes + length > QUEUED_BYTES
        {
            return Some(write);
        }
        work.bytes += length;
        work.queued.push_back(Arc::new(write));
        self.wake.notify_all();
        None
    }

    /// The bytes of the entry queued for `path`, which the thread has not written yet: the one it
    /// is writing now included, because its file is not in place until its `rename`.
    pub(crate) fn pending(&self, path: &Path) -> Option<Vec<u8>> {
        let work = lock(&self.work);
        work.queued
            .iter()
            .rev()
            .chain(work.writing.iter())
            .find(|write| write.path == path)
            .map(|write| write.bytes.clone())
    }

    /// Whether an entry for `path` is queued or being written, without taking its bytes
    /// ([`Asked::pending`] clones them, which a question about the queue has no use for).
    pub(crate) fn holds(&self, path: &Path) -> bool {
        let work = lock(&self.work);
        work.queued
            .iter()
            .chain(work.writing.iter())
            .any(|write| write.path == path)
    }

    /// Drop the entry queued for `path`, where one is: its bytes are not the entry after all.
    /// The one being written is past recall, and its file is removed by the caller.
    pub(crate) fn discard(&self, path: &Path) {
        let mut work = lock(&self.work);
        if let Some(at) = work.queued.iter().position(|write| write.path == path) {
            let gone = work.queued.remove(at).expect("the entry just found");
            work.bytes -= gone.bytes.len() as u64;
        }
    }

    /// The next entry to write, which stays visible to lookups until [`Asked::written`].
    fn take(&self) -> Option<Arc<Write>> {
        let mut work = lock(&self.work);
        let write = work.queued.pop_front()?;
        work.writing = Some(Arc::clone(&write));
        Some(write)
    }

    /// The entry the thread was writing is written (or failed): it is no longer the queue's.
    fn written(&self) {
        let mut work = lock(&self.work);
        if let Some(done) = work.writing.take() {
            work.bytes -= done.bytes.len() as u64;
        }
    }

    /// Ask for a scan that brings the directory under its cap.
    fn evict(&self) {
        lock(&self.work).evict = true;
        self.wake.notify_all();
    }

    /// Ask the thread to end: it writes what is queued and returns.
    pub(crate) fn stopping(&self) {
        lock(&self.work).stopping = true;
        self.wake.notify_all();
    }
}

/// The thread and what it is asked to do.
pub(super) struct Worker {
    asked: Arc<Asked>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Worker {
    /// Start the thread over `store`, which was made with `asked`.
    pub(super) fn start(store: Arc<Store>, asked: Arc<Asked>) -> Result<Worker, Error> {
        let thread = {
            let asked = Arc::clone(&asked);
            exit_guard::spawn(
                std::thread::Builder::new().name("metal2vulkan-shader-cache".into()),
                move || run(&asked, &store),
            )
            .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?
        };
        Ok(Worker {
            asked,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Ask for a scan that brings the directory under its cap.
    pub(super) fn evict(&self) {
        self.asked.evict();
    }

    /// Stop the thread, and wait for it: the entries queued are written first.
    pub(super) fn stop(&self) {
        self.asked.stop.store(true, Ordering::Relaxed);
        self.asked.stopping();
        if let Some(handle) = lock(&self.thread).take() {
            let _ = handle.join();
        }
    }
}

/// Write every entry queued; whether the directory is past its cap after them.
pub(crate) fn drain(asked: &Asked, store: &Store) -> bool {
    while let Some(write) = asked.take() {
        store.write_queued(&write.path, &write.bytes);
        asked.written();
    }
    store.over_capacity()
}

/// The thread's loop: the scan at creation, then the entries it is given and the scans it is
/// asked for, until [`Asked::stopping`], whose entries it writes before it returns.
pub(crate) fn run(asked: &Asked, store: &Store) {
    store.sweep(&asked.stop);
    loop {
        let asked_scan;
        {
            let mut work = lock(&asked.work);
            while !work.stopping && !work.evict && work.queued.is_empty() {
                work = asked
                    .wake
                    .wait(work)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if work.stopping {
                drop(work);
                // The device is going: what a creation handed over is written before the drop
                // that stops this thread returns.
                drain(asked, store);
                break;
            }
            asked_scan = work.evict;
            work.evict = false;
        }
        let over = drain(asked, store);
        if asked_scan || over {
            store.evict(&asked.stop);
        }
    }
}
