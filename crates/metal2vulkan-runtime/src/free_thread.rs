//! The device's free thread: the `vkFreeMemory` of a block of the runtime's own memory that a
//! release let go of runs here, off the releasing thread (`memory.rs`'s `Memory::let_go`), and so
//! does a host import's that a completed command buffer let go of last while a deallocator or a
//! keep-alive waits for it (below), off the thread that runs the completed handlers. A
//! driver that unmaps a block's pages as it frees it charges that to the free: on an Apple M3 Max
//! an object's release that freed a block took up to 24 ms (a 2048x2048 `Shared` texture's, a
//! 16 MiB buffer's, each the last placement in its block).
//!
//! - **What is posted.** A block of the runtime's own memory, pooled or dedicated, whose last
//!   placement went, with its buffer of addresses, and a block over fresh pages the runtime
//!   mapped for a buffer (`fresh_pages.rs`), its pages unmapped here once its memory is freed.
//!   An import of the caller's memory (a no-copy object's, a read-only view of the caller's
//!   pages, a guest RAM chunk) only where a completed command buffer's release lets go of it
//!   last ([`AtCompletion`]) and a deallocator or a keep-alive waits for its free
//!   ([`PostedImport`]): Vulkan has "the application must ensure that the imported memory range
//!   remains valid and accessible for the lifetime of the imported memory object"
//!   (`VkImportMemoryHostPointerInfoEXT`), and the caller learns that its pages may go from
//!   that deallocator or keep-alive, which go here after the free. So such a deallocator may run
//!   after the command buffer's waiters returned (`wait_until_completed`), paced behind the
//!   frees posted before it: Metal ties a deallocator to the buffer's deallocation, not to a
//!   wait for a command buffer that used it. Any other import is freed
//!   where it goes: a program's own release frees it before the release returns, as the caller
//!   may free its pages once it returned, and a command buffer's release with nothing waiting
//!   frees it before the command buffer's waiters return. A block
//!   that waits for the batches in flight (`freed_memory.rs`) is freed by the handler thread that
//!   sees them complete, and posted here where they had completed as it went. And a spare command
//!   pool's command buffers that a give-back frees (`spares.rs`'s `PoolFrees`), the pool trimmed,
//!   so that a command buffer's completion, which gives its pool back, frees none: the pool goes
//!   back among the spares once they are freed.
//! - **Not an event's semaphore**, which its release destroys where it goes (`event.rs`): NVIDIA's
//!   driver runs a `vkCreateSemaphore` behind a `vkDestroySemaphore` another thread runs, so a
//!   posted destroy moves its cost onto the program's next event and adds to it
//!   (docs/FAST-PATHS.md, known gaps).
//! - **Reusable at once.** The block left its pool before it is posted, and the object's
//!   resource ID and Metal address go back with the object: the address space is the runtime's
//!   own (`address.rs`), a pinned address one of its ranges, never a Vulkan device address, and
//!   no address is captured for replay. So reusing an ID or an address needs no memory freed.
//! - **Bounded.** At most [`QUEUE`] frees wait; a block, a pool's command buffers or an import
//!   that find the queue full, the device's teardown begun, or no thread, are freed where they
//!   went.
//! - **Still counted.** A posted block counts against `maxMemoryAllocationCount`
//!   (`allocations.rs`) and against its memory heap's size until its `vkFreeMemory` returns. An
//!   allocation the count's limit leaves no room for waits for the posted frees first (a spare
//!   pool's among them), then asks again; a block whose making would take its heap past the heap's size while frees of that
//!   heap wait, waits for them first, as one whose making failed for room does (`memory.rs`'s
//!   `place`). Each such wait is counted (`WorkCounts::memory_free_waits`).
//! - **Paced.** Before each block it frees, the thread waits until as long as its frees before
//!   it took has passed since the last of them ended (`step_aside::pace`): NVIDIA's driver holds
//!   every other thread's `vkQueueSubmit` while it frees memory (docs/FAST-PATHS.md, known gaps),
//!   so a run of posted frees back to back would hold the program's submissions for the whole
//!   run. Never while a caller waits for the posted frees ([`FreeThread::wait`], the device's
//!   teardown). Counted: `WorkCounts::memory_frees_paced`, `memory_frees_paced_nanos`.
//! - **Holds nothing of the device** but a posted import's objects, which hold it as a holder
//!   does (`device/holds.rs`): the program's last drop on a thread of the program's waits for
//!   them, and the thread frees them unpaced once that drop stopped the holds. The thread holds
//!   its queue, a share of the device's functions and of its spare command pools. The device's
//!   teardown waits for the frees posted before it releases the spares ([`FreeThread::drain`]),
//!   and for every free posted and joins the thread before `vkDestroyDevice`; where the program's
//!   last drop ran on a thread of the runtime's, which waits for nothing, a posted import's
//!   objects may hold the device's last share, and the teardown runs here as they go: it frees
//!   the blocks and pools still queued in place of the waits, and ends the thread unjoined.

use crate::device::functions::Functions;
use crate::device::lock;
use crate::exit_guard;
use crate::spares::PoolFrees;
use ash::vk;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
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
    /// The blocks' frees posted so far, and the host imports'.
    posted: AtomicU64,
    imports: AtomicU64,
    /// The frees run where they went, the queue full.
    full: AtomicU64,
    /// The waits for posted frees: an allocation at the count's limit or past its heap's size.
    waits: AtomicU64,
    /// The frees that waited first for those before them (`step_aside::pace`), and the
    /// nanoseconds they waited.
    paced: AtomicU64,
    paced_nanos: AtomicU64,
    /// The crate's device tests' hold of each free as it begins, a block's, a spare pool's or a
    /// host import's (`tests/posted_frees.rs`, `tests/spares.rs`,
    /// `tests/completion_import_frees.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    held: crate::freed_memory::held::Held,
    /// The crate's device tests' stand-in for how long each block's free takes
    /// ([`FreeThread::frees_take`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    takes: Mutex<Option<std::time::Duration>>,
    /// The device's teardown ran on the thread itself, as a posted import's objects went
    /// ([`FreeThread::stop`]): the crate's device tests' sign of it.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    stopped_here: std::sync::atomic::AtomicBool,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Work>,
    /// Posted and not yet freed: those queued and the one the thread frees now.
    pending: usize,
    /// Per memory heap, the bytes posted and not yet freed.
    heap_bytes: [u64; vk::MAX_MEMORY_HEAPS],
    /// Callers waiting for every free posted: the thread's frees do not pace then.
    waiting: usize,
    stopping: bool,
}

/// What is posted to the thread.
enum Work {
    /// A block's memory.
    Block(Posted),
    /// A spare command pool's command buffers.
    Pool(Box<PoolFrees>),
    /// A host import's objects.
    Import(PostedImport),
}

/// A host import's objects, posted to be freed as they drop where a completed command buffer's
/// release let go of them last (the module docs, **What is posted**): their `VkBuffer` and
/// memory, then the deallocators and the keep-alive that waited for the free (`buffer.rs`).
pub(crate) struct PostedImport {
    /// The objects, each handle of the device they hold a holder's
    /// (`Device::held_for_release`).
    pub(crate) objects: Box<dyn Send>,
    /// The device's holds: once its program's last drop stopped them, the thread frees the
    /// objects unpaced, as that drop waits for them.
    pub(crate) holds: Arc<crate::device::holds::Holds>,
}

thread_local! {
    /// A completed command buffer's release of what its commands used runs on this thread now
    /// ([`AtCompletion`]).
    static AT_COMPLETION: Cell<bool> = const { Cell::new(false) };
    /// This thread is a device's free thread.
    static FREE_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// A completed command buffer's release of what its commands used, on this thread, while it lives
/// (`CommandBufferShared::complete`): a host import whose free a deallocator or a keep-alive
/// waits for, let go of last there, is posted to its device's free thread
/// ([`posts_imports`]).
pub(crate) struct AtCompletion {
    was: bool,
}

impl AtCompletion {
    /// The release begins.
    pub(crate) fn enter() -> AtCompletion {
        AtCompletion {
            was: AT_COMPLETION.with(|at| at.replace(true)),
        }
    }
}

impl Drop for AtCompletion {
    fn drop(&mut self) {
        AT_COMPLETION.with(|at| at.set(self.was));
    }
}

/// Whether a host import that goes now, a deallocator or a keep-alive waiting for its free, is
/// posted to the free thread: a completed command buffer's release lets go of it
/// ([`AtCompletion`]), not inside the program's code (a deallocator that release ran).
pub(crate) fn posts_imports() -> bool {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if FREES_IN_PLACE.load(Ordering::Relaxed) {
        return false;
    }
    AT_COMPLETION.with(Cell::get) && !exit_guard::in_program_call()
}

/// The crate's device tests' stand-in for the release before posting: a completed command
/// buffer's release frees every import in place while set (`tests/import_free_hold_bench.rs`'s
/// completion leg, which measures both in one process).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) static FREES_IN_PLACE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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
                imports: AtomicU64::new(0),
                full: AtomicU64::new(0),
                waits: AtomicU64::new(0),
                paced: AtomicU64::new(0),
                paced_nanos: AtomicU64::new(0),
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                held: crate::freed_memory::held::Held::default(),
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                takes: Mutex::new(None),
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                stopped_here: std::sync::atomic::AtomicBool::new(false),
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
        let mut state = match self.room() {
            Ok(state) => state,
            Err(full) => {
                if full {
                    self.shared.full.fetch_add(1, Ordering::Relaxed);
                }
                return Err(posted);
            }
        };
        state.pending += 1;
        state.heap_bytes[posted.heap] += posted.size;
        state.queue.push_back(Work::Block(posted));
        drop(state);
        self.shared.posted.fetch_add(1, Ordering::Relaxed);
        self.shared.work.notify_one();
        Ok(())
    }

    /// Post `frees` to run on the thread, started unless it runs; given back, to run where they
    /// were found, where [`post`](Self::post) would give a block back.
    ///
    /// # Safety
    ///
    /// As [`PoolFrees::free`]'s.
    pub(crate) unsafe fn post_pool(&self, frees: Box<PoolFrees>) -> Result<(), Box<PoolFrees>> {
        let Ok(mut state) = self.room() else {
            return Err(frees);
        };
        state.pending += 1;
        state.queue.push_back(Work::Pool(frees));
        drop(state);
        self.shared.work.notify_one();
        Ok(())
    }

    /// Post `posted`, a host import's objects, to be freed on the thread, started unless it runs;
    /// given back, to be freed where they went, where [`post`](Self::post) would give a block
    /// back. Paced as a block's free.
    pub(crate) fn post_import(&self, posted: PostedImport) -> Result<(), PostedImport> {
        let mut state = match self.room() {
            Ok(state) => state,
            Err(full) => {
                if full {
                    self.shared.full.fetch_add(1, Ordering::Relaxed);
                }
                return Err(posted);
            }
        };
        state.pending += 1;
        state.queue.push_back(Work::Import(posted));
        drop(state);
        self.shared.imports.fetch_add(1, Ordering::Relaxed);
        self.shared.work.notify_one();
        Ok(())
    }

    /// The thread's state with room for one more free, the thread started unless it runs; else
    /// whether the queue was full, rather than the device's teardown begun or no thread started.
    fn room(&self) -> Result<MutexGuard<'_, State>, bool> {
        if !self.started() {
            return Err(false);
        }
        let state = lock(&self.shared.state);
        if state.stopping {
            return Err(false);
        }
        if state.pending >= QUEUE {
            return Err(true);
        }
        Ok(state)
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

    /// Wait until no free posted is left: whether one was, the wait then counted as it begins.
    pub(crate) fn wait(&self) -> bool {
        self.wait_for_posted(true)
    }

    /// Wait until no free posted is left, uncounted: the device's teardown, before it releases its
    /// spare command pools, which those posted keep again (`spares.rs`).
    pub(crate) fn drain(&self) {
        self.wait_for_posted(false);
    }

    /// Wait until no free posted is left: whether one was, the wait counted as it begins where
    /// `counted`. On the thread itself (the device's teardown, where a posted import's objects
    /// held its last share: the module docs), free those still queued here instead, unpaced: the
    /// one it frees now is the teardown's own.
    fn wait_for_posted(&self, counted: bool) -> bool {
        if FREE_THREAD.with(Cell::get) {
            let mut freed = false;
            loop {
                let Some(work) = lock(&self.shared.state).queue.pop_front() else {
                    return freed;
                };
                free(&self.shared, work, false);
                freed = true;
            }
        }
        let mut state = lock(&self.shared.state);
        if state.pending == 0 {
            return false;
        }
        if counted {
            self.shared.waits.fetch_add(1, Ordering::Relaxed);
        }
        state.waiting += 1;
        while state.pending > 0 {
            state = (self.shared.done.wait(state)).unwrap_or_else(PoisonError::into_inner);
        }
        state.waiting -= 1;
        true
    }

    /// The frees posted so far, those run where they went as the queue was full, and the waits
    /// for posted frees.
    pub(crate) fn counts(&self) -> (u64, u64, u64) {
        let shared = &*self.shared;
        (
            shared.posted.load(Ordering::Relaxed),
            shared.full.load(Ordering::Relaxed),
            shared.waits.load(Ordering::Relaxed),
        )
    }

    /// The host imports' frees posted so far ([`post_import`](Self::post_import)).
    pub(crate) fn imports_posted(&self) -> u64 {
        self.shared.imports.load(Ordering::Relaxed)
    }

    /// The frees that waited first for those before them, and the nanoseconds they waited.
    pub(crate) fn paced(&self) -> (u64, u64) {
        let shared = &*self.shared;
        (
            shared.paced.load(Ordering::Relaxed),
            shared.paced_nanos.load(Ordering::Relaxed),
        )
    }

    /// The tests' hold of each free as it begins.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn held(&self) -> &crate::freed_memory::held::Held {
        &self.shared.held
    }

    /// Count each block's free from here on as taking `took`, owed so to the thread's next pace
    /// whatever the free measures (`step_aside::own_calls_take`); `None` measures them again.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn frees_take(&self, took: Option<std::time::Duration>) {
        *lock(&self.shared.takes) = took;
    }

    /// Whether the device's teardown ran on the thread itself ([`stop`](Self::stop)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stopped_here(&self) -> bool {
        self.shared.stopped_here.load(Ordering::SeqCst)
    }

    /// The frees posted and not yet run, and whether the thread was started and not yet joined.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn pending(&self) -> (usize, bool) {
        let pending = lock(&self.shared.state).pending;
        (pending, lock(&self.thread).is_some())
    }

    /// The device's teardown, before `vkDestroyDevice`: no free is posted from here on, the
    /// thread frees every one posted, and ends, joined; on the thread itself (the module docs),
    /// those still queued freed here, and the thread ends unjoined once the teardown returned.
    pub(crate) fn stop(&self) {
        lock(&self.shared.state).stopping = true;
        self.shared.work.notify_all();
        if FREE_THREAD.with(Cell::get) {
            self.wait_for_posted(false);
            drop(lock(&self.thread).take());
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            self.shared.stopped_here.store(true, Ordering::SeqCst);
            return;
        }
        if let Some(handle) = lock(&self.thread).take() {
            let _ = handle.join();
        }
    }
}

/// The thread's loop: run each free posted, in turn, until stopped with none left.
fn run(shared: &Shared) {
    crate::step_aside::mark_a_paced_thread();
    FREE_THREAD.with(|on| on.set(true));
    loop {
        let work = {
            let mut state = lock(&shared.state);
            loop {
                if let Some(work) = state.queue.pop_front() {
                    break work;
                }
                if state.stopping {
                    return;
                }
                state = (shared.work.wait(state)).unwrap_or_else(PoisonError::into_inner);
            }
        };
        free(shared, work, true);
    }
}

/// Run `work`, a free posted, paced first where `paced` (the module docs, **Paced**), and count it
/// freed.
fn free(shared: &Shared, work: Work, paced: bool) {
    let hurry = || {
        let state = lock(&shared.state);
        state.stopping || state.waiting > 0
    };
    let pace = |hurry: &dyn Fn() -> bool| {
        if !paced {
            return;
        }
        if let Some(waited) = crate::step_aside::pace(hurry) {
            shared.paced.fetch_add(1, Ordering::Relaxed);
            let nanos = u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX);
            shared.paced_nanos.fetch_add(nanos, Ordering::Relaxed);
        }
    };
    let freed = match work {
        Work::Block(posted) => {
            let (heap, size) = (posted.heap, posted.size);
            pace(&hurry);
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            shared.held.pass();
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            crate::step_aside::own_calls_take(*lock(&shared.takes));
            // The free is a section of the exit guard: while the process exits, the block,
            // and the pages it was imported from, stay to its end (`exit_guard`).
            match exit_guard::enter() {
                Some(busy) => {
                    // SAFETY: `post`'s contract.
                    let after = unsafe { posted.free(shared.functions.get()) };
                    drop(busy);
                    drop(after);
                }
                None => std::mem::forget(posted),
            }
            Some((heap, size))
        }
        Work::Pool(frees) => {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            shared.held.pass();
            // As a block's: while the process exits, the command buffers stay to its end.
            if let Some(busy) = exit_guard::enter() {
                // SAFETY: `post_pool`'s contract.
                unsafe { (*frees).free(shared.functions.get()) };
                drop(busy);
            }
            None
        }
        Work::Import(PostedImport { objects, holds }) => {
            pace(&|| holds.stopping() || hurry());
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            shared.held.pass();
            drop(holds);
            // Their free, and the deallocators and keep-alive after it, each find the exit
            // themselves (`buffer.rs`).
            drop(objects);
            None
        }
    };
    let mut state = lock(&shared.state);
    state.pending -= 1;
    if let Some((heap, size)) = freed {
        state.heap_bytes[heap] -= size;
    }
    drop(state);
    shared.done.notify_all();
}
