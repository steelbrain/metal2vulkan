//! The device's free thread: the `vkFreeMemory` of a block of the runtime's own memory that a
//! release let go of runs here, off the releasing thread (`memory.rs`'s `Memory::let_go`), and
//! the `vkDestroySemaphore` of an event's timeline semaphore (`event.rs`). A driver that unmaps a
//! block's pages as it frees it charges that to the free: on an Apple M3 Max an object's release
//! that freed a block took up to 24 ms (a 2048x2048 `Shared` texture's, a 16 MiB buffer's, each
//! the last placement in its block). On an RTX 5080 an event's release took about 400 us, in
//! the kernel driver (the consumer's measurement: median of 5, 1000 events).
//!
//! - **What is posted.** A block of the runtime's own memory, pooled or dedicated, whose last
//!   placement went, with its buffer of addresses, and a block over fresh pages the runtime
//!   mapped for a buffer (`fresh_pages.rs`), its pages unmapped here once its memory is freed.
//!   Never an import of the caller's memory (a no-copy object's, a read-only view of the
//!   caller's pages, a guest RAM chunk): Vulkan has "the application must ensure that the
//!   imported memory range remains valid and accessible for the lifetime of the imported memory
//!   object" (`VkImportMemoryHostPointerInfoEXT`), and the caller may free its pages once the
//!   object's release returns, so such an import is freed before the release returns. A block
//!   that waits for the batches in flight (`freed_memory.rs`) is freed by the handler thread that
//!   sees them complete, and posted here where they had completed as it went.
//! - **And an event's semaphore**, as the event's last holder lets go: every batch that waits for
//!   or signals it holds the event until it completed, a reserved signal of the caller's holds it
//!   until the signal landed, and the caller keeps it while its own submissions that use the
//!   semaphore run (`SharedEvent::vk_semaphore`'s contract), so nothing of the runtime's or the
//!   caller's names the semaphore any more. It is no longer among the device's own timelines
//!   from then (`interop.rs`), and the driver hands its handle out again only once it is
//!   destroyed. Never recycled: its counter only rises, and its handle was the caller's to see.
//! - **Reusable at once.** The block left its pool before it is posted, and the object's
//!   resource ID and Metal address go back with the object: the address space is the runtime's
//!   own (`address.rs`), a pinned address one of its ranges, never a Vulkan device address, and
//!   no address is captured for replay. So reusing an ID or an address needs no memory freed.
//! - **Bounded.** At most [`QUEUE`] frees wait; a block that finds the queue full, the device's
//!   teardown begun, or no thread, is freed where it went.
//! - **Still counted.** A posted block counts against `maxMemoryAllocationCount`
//!   (`allocations.rs`) and against its memory heap's size until its `vkFreeMemory` returns. An
//!   allocation the count's limit leaves no room for waits for the posted frees first, then asks
//!   again; a block whose making would take its heap past the heap's size while frees of that
//!   heap wait, waits for them first, as one whose making failed for room does (`memory.rs`'s
//!   `place`). Each such wait is counted (`WorkCounts::memory_free_waits`).
//! - **Holds nothing of the device.** The thread holds its queue and a share of the device's
//!   functions. The device's teardown waits for every free posted and joins the thread before
//!   `vkDestroyDevice`.

use crate::device::functions::Functions;
use crate::device::lock;
use crate::exit_guard;
use ash::vk;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// The most frees that wait for the thread (the module docs): a burst of releases longer than
/// this frees the rest where they go.
pub(crate) const QUEUE: usize = 64;

/// The device's free thread, once a free started it.
pub(crate) struct FreeThread {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// What the thread and the posters share.
struct Shared {
    functions: Arc<Functions>,
    state: Mutex<State>,
    /// A free was posted, or the thread is asked to stop.
    work: Condvar,
    /// A free completed.
    done: Condvar,
    /// The blocks' frees posted so far.
    posted: AtomicU64,
    /// The semaphores' destroys posted so far.
    semaphores: AtomicU64,
    /// The frees run where they went, the queue full.
    full: AtomicU64,
    /// The waits for posted frees: an allocation at the count's limit or past its heap's size.
    waits: AtomicU64,
    /// The crate's device tests' hold of each free as it begins (`tests/posted_frees.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    held: crate::freed_memory::held::Held,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Free>,
    /// Posted and not yet freed: those queued and the one the thread frees now.
    pending: usize,
    /// Per memory heap, the bytes posted and not yet freed.
    heap_bytes: [u64; vk::MAX_MEMORY_HEAPS],
    stopping: bool,
}

/// What the thread frees.
enum Free {
    Block(Posted),
    Semaphore(vk::Semaphore),
}

/// A block's memory, its buffer of addresses and the pages it was imported from, posted to be
/// freed.
pub(crate) struct Posted {
    pub(crate) memory: vk::DeviceMemory,
    /// Destroyed before the memory is freed; `None` where it went already (`freed_memory.rs`).
    pub(crate) addresses: Option<vk::Buffer>,
    /// What goes once the memory is freed: the runtime's own pages it was imported from.
    pub(crate) after: Option<Box<dyn Send>>,
    /// The memory heap the block lies in.
    pub(crate) heap: usize,
    /// Its bytes.
    pub(crate) size: u64,
}

impl Posted {
    /// Destroy the buffer of addresses and free the memory: what goes after it, for the caller to
    /// let go of with no lock held.
    ///
    /// # Safety
    ///
    /// As [`FreeThread::post`]'s.
    pub(crate) unsafe fn free(self, device: &ash::Device) -> Option<Box<dyn Send>> {
        let Posted {
            memory,
            addresses,
            after,
            ..
        } = self;
        // SAFETY: the caller's.
        unsafe {
            if let Some(addresses) = addresses {
                crate::release::object(device, addresses);
            }
            crate::release::object(device, memory);
        }
        after
    }
}

impl FreeThread {
    /// The free thread of the device whose functions are `functions`, not started.
    pub(crate) fn new(functions: Arc<Functions>) -> FreeThread {
        FreeThread {
            shared: Arc::new(Shared {
                functions,
                state: Mutex::default(),
                work: Condvar::new(),
                done: Condvar::new(),
                posted: AtomicU64::new(0),
                semaphores: AtomicU64::new(0),
                full: AtomicU64::new(0),
                waits: AtomicU64::new(0),
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                held: crate::freed_memory::held::Held::default(),
            }),
            thread: Mutex::new(None),
        }
    }

    /// Post `posted` to be freed on the thread, started unless it runs; given back, to be freed
    /// where it went, where the queue is full, the device's teardown began or the host cannot
    /// start a thread.
    ///
    /// # Safety
    ///
    /// The memory and the buffer are the device's, nothing is placed in the memory or bound to
    /// the buffer, and no pending command reads either.
    pub(crate) unsafe fn post(&self, posted: Posted) -> Result<(), Posted> {
        if !self.started() {
            return Err(posted);
        }
        let mut state = lock(&self.shared.state);
        if state.stopping {
            return Err(posted);
        }
        if state.pending >= QUEUE {
            drop(state);
            self.shared.full.fetch_add(1, Ordering::Relaxed);
            return Err(posted);
        }
        state.pending += 1;
        state.heap_bytes[posted.heap] += posted.size;
        state.queue.push_back(Free::Block(posted));
        drop(state);
        self.shared.posted.fetch_add(1, Ordering::Relaxed);
        self.shared.work.notify_one();
        Ok(())
    }

    /// Post the destroy of an event's timeline semaphore (the module docs), or destroy it on
    /// `device` here where the queue is full, the device's teardown began or the host cannot
    /// start a thread.
    ///
    /// # Safety
    ///
    /// The semaphore is the device's, and nothing waits for or signals it, nor will.
    pub(crate) unsafe fn destroy_semaphore(&self, semaphore: vk::Semaphore, device: &ash::Device) {
        let posted = self.started() && {
            let mut state = lock(&self.shared.state);
            let room = !state.stopping && state.pending < QUEUE;
            if room {
                state.pending += 1;
                state.queue.push_back(Free::Semaphore(semaphore));
            } else if !state.stopping {
                self.shared.full.fetch_add(1, Ordering::Relaxed);
            }
            room
        };
        if posted {
            self.shared.semaphores.fetch_add(1, Ordering::Relaxed);
            self.shared.work.notify_one();
        } else {
            // SAFETY: the caller's.
            unsafe { crate::release::object(device, semaphore) };
        }
    }

    /// [`post`](Self::post) `posted`, or free it on `device` here where it is given back: what
    /// goes after its memory, for the caller to let go of with no lock held, where it was freed
    /// here.
    ///
    /// # Safety
    ///
    /// As [`post`](Self::post)'s.
    pub(crate) unsafe fn post_or_free(
        &self,
        posted: Posted,
        device: &ash::Device,
    ) -> Option<Box<dyn Send>> {
        // SAFETY: the caller's.
        match unsafe { self.post(posted) } {
            Ok(()) => None,
            // SAFETY: the caller's.
            Err(posted) => unsafe { posted.free(device) },
        }
    }

    /// Whether the thread runs, started here where it did not, unless the device's teardown
    /// began.
    fn started(&self) -> bool {
        let mut thread = lock(&self.thread);
        if thread.is_none() {
            if lock(&self.shared.state).stopping {
                return false;
            }
            let shared = Arc::clone(&self.shared);
            let builder = std::thread::Builder::new().name("metal2vulkan-frees".into());
            let Ok(handle) = exit_guard::spawn(builder, move || run(&shared)) else {
                return false;
            };
            *thread = Some(handle);
        }
        true
    }

    /// The bytes of memory heap `heap` posted and not yet freed.
    pub(crate) fn heap_pending(&self, heap: usize) -> u64 {
        lock(&self.shared.state).heap_bytes[heap]
    }

    /// Wait until no free posted is left: whether one was, the wait then counted.
    pub(crate) fn wait(&self) -> bool {
        let mut state = lock(&self.shared.state);
        if state.pending == 0 {
            return false;
        }
        self.shared.waits.fetch_add(1, Ordering::Relaxed);
        while state.pending > 0 {
            state = (self.shared.done.wait(state)).unwrap_or_else(PoisonError::into_inner);
        }
        true
    }

    /// The blocks' frees and the semaphores' destroys posted so far, those run where they went
    /// as the queue was full, and the waits for posted frees.
    pub(crate) fn counts(&self) -> (u64, u64, u64, u64) {
        let shared = &*self.shared;
        (
            shared.posted.load(Ordering::Relaxed),
            shared.semaphores.load(Ordering::Relaxed),
            shared.full.load(Ordering::Relaxed),
            shared.waits.load(Ordering::Relaxed),
        )
    }

    /// The tests' hold of each free as it begins.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn held(&self) -> &crate::freed_memory::held::Held {
        &self.shared.held
    }

    /// The frees posted and not yet run, and whether the thread was started and not yet joined.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn pending(&self) -> (usize, bool) {
        let pending = lock(&self.shared.state).pending;
        (pending, lock(&self.thread).is_some())
    }

    /// The device's teardown, before `vkDestroyDevice`: no free is posted from here on, the
    /// thread frees every one posted, and ends, joined.
    pub(crate) fn stop(&self) {
        lock(&self.shared.state).stopping = true;
        self.shared.work.notify_all();
        if let Some(handle) = lock(&self.thread).take() {
            let _ = handle.join();
        }
    }
}

/// The thread's loop: free each block posted, in turn, until stopped with none left.
fn run(shared: &Shared) {
    loop {
        let posted = {
            let mut state = lock(&shared.state);
            loop {
                if let Some(posted) = state.queue.pop_front() {
                    break posted;
                }
                if state.stopping {
                    return;
                }
                state = (shared.work.wait(state)).unwrap_or_else(PoisonError::into_inner);
            }
        };
        let left = match &posted {
            Free::Block(block) => Some((block.heap, block.size)),
            Free::Semaphore(_) => None,
        };
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        shared.held.pass();
        // The free is a section of the exit guard: while the process exits, the block, and the
        // pages it was imported from, or the semaphore, stay to its end (`exit_guard`).
        match (exit_guard::enter(), posted) {
            (Some(busy), Free::Block(block)) => {
                // SAFETY: `post`'s contract.
                let after = unsafe { block.free(shared.functions.get()) };
                drop(busy);
                drop(after);
            }
            (Some(busy), Free::Semaphore(semaphore)) => {
                // SAFETY: `destroy_semaphore`'s contract.
                unsafe { crate::release::object(shared.functions.get(), semaphore) };
                drop(busy);
            }
            (None, Free::Block(block)) => std::mem::forget(block),
            (None, Free::Semaphore(_)) => {}
        }
        let mut state = lock(&shared.state);
        state.pending -= 1;
        if let Some((heap, size)) = left {
            state.heap_bytes[heap] -= size;
        }
        drop(state);
        shared.done.notify_all();
    }
}
