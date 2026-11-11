//! Idle imports of a no-copy buffer's pages, released and imported again at their next use: a
//! runtime-internal residency, which nothing of Metal's observes.
//!
//! RADV lists every `VkDeviceMemory` its device imported in every kernel submission's
//! buffer-object list. Mesa 26.2.4's `radv_alloc_memory` makes every memory but a WSI image's
//! resident in the device's global list (`buffer_make_resident`), its amdgpu winsys keeps out of
//! that list only memory neither imported nor exported (`radv_amdgpu_winsys_bo_make_resident`),
//! and each submission's list is the whole global list (`radv_amdgpu_get_bo_list`). Linux 7.0's
//! amdgpu works per entry at every submission (`amdgpu_cs.c`: lookup and sort, lock, validate,
//! page-table update, fence sync and fence add), about 0.75 us an entry on a Radeon 780M (the
//! consumer's measurement). ANV on i915 lists every `VkDeviceMemory` of the device in every
//! execbuffer (`i915/anv_batch_chain.c`, `setup_execbuf_for_cmd_buffers`), a pointer import (an
//! i915 userptr) costing about 0.16 us of the kernel's time an execbuffer on an Iris Xe of RPL-P,
//! three times an ordinary buffer object. So each import a device holds costs every submission,
//! used or not: a consumer whose guest keeps hundreds of transfer buffers paid much of its
//! submission thread's time for those idle. The runtime's own memory is pooled in few blocks.
//!
//! - **Which imports.** A no-copy buffer's own import (not a heap's, nor a rebind's) where the
//!   device can make it again exactly ([`Again`]):
//!   - of the dma-bufs its pages' source made (`dma_buf.rs`, where the device refused the pages'
//!     pointer), where the device kept a descriptor of each dma-buf within the descriptors the
//!     process's devices keep at most ([`kept_fds_limit`]); an import past them is made as any
//!     other and stays imported;
//!   - of its pages' pointer, on a device whose check found a held pointer import adds to every
//!     submission's cost (`submit_check.rs`, the import tax: RADV and ANV on i915 pay one, NVIDIA's
//!     driver and MoltenVK none measured): the caller's pages stay mapped at the pointer, with the
//!     same backing, while the buffer lives (`HostMemory::new`), so an import again at the same
//!     pointer reads the pages the kept import read, on every driver, and releasing it changes
//!     nothing a command reads. Elsewhere, and in a buffer made before the check answered, a
//!     pointer import stays imported: a release would spare no submission any cost, and the
//!     import again would cost the next blit;
//!   - of a run of a guest RAM alone (`host_ram.rs`), which a buffer over one run inside a chunk
//!     not imported yet was bound in while the device's thread imports the chunk: made again in
//!     the chunk's import where that is made by then (and lays the bytes as the run's did), else
//!     in an import of the run alone again, never the chunk's own import nor a wait for one. The
//!     RAM's pages stay mapped at their addresses while it lives (`HostRam::new`), so either reads
//!     the pages the run's import read.
//! - **Blits alone.** It stays releasable while every read of its `VkBuffer` and memory was a
//!   blit's (`Buffer::blit_at`: a copy or fill naming the buffer, a copy between it and a
//!   texture, `synchronize_resource`), which holds what it read until its command buffer
//!   completed ([`Imported`]), weakly from the commit until the first submission
//!   (`rebind::ImportHold`). Any other read keeps it imported for good from then on, before the
//!   read returns: its Metal address handed out (`gpu_address`, so a binding, an argument
//!   buffer, an argument table, an indirect command buffer or `use_resource`), its `VkBuffer`
//!   handed out (`vk_buffer`), a texture or texel view over its bytes, or a second buffer lying in
//!   the import (`host_aliases.rs`). So no pointer, descriptor or image other than a blit's
//!   command ever holds a released import's objects.
//! - **Released while idle** by the device's residency thread ([`Sweeper`]), every [`TICK`]: an
//!   import no command buffer holds, even weakly (every one that read it completed) and that no
//!   blit took for its window, [`IDLE`] at first and doubled at each import again, up to
//!   [`IDLE_MOST`], so an import used now and then stays imported and an unused one goes. Its
//!   `VkBuffer` is destroyed and its memory freed; the buffer, its Metal address, its host pages,
//!   the dma-bufs and the caller's release stay. The crate's device tests release them at once,
//!   whatever their windows (`tests/host_residency.rs`, `tests/pointer_residency.rs`). The doubling
//!   bounds the churn whatever the program's pattern: an import is made again at most five times
//!   before its window reaches [`IDLE_MOST`], and at most once per such window after.
//! - **The import going** (its last buffer released) lets go of its objects, and the caller's
//!   keep-alive and deallocators go once their memory is freed: as the import goes (in the
//!   release, or on a handler thread where a keep-alive alone waits for the free, `buffer.rs`), or
//!   with the last submitted command buffer that read
//!   them, which holds them, not the buffer, once a rebind or a detach replaced the buffer's pages
//!   (`rebind.rs`, `detach.rs`). One committed and not yet submitted holds them weakly and never
//!   runs the recording that named them, so they go in that rebind or detach, as Metal's
//!   deallocator runs there. So the caller's pages stay mapped while any memory over them lives:
//!   ANV on i915 lists every memory of the device in every execbuffer, and one over
//!   pointer-imported pages since unmapped fails it, the device lost. Where the free waits for the
//!   batches in flight (`freed_memory.rs`) that is announced, as for an import kept.
//! - **Imported again at the next blit,** on the encoding thread, before its command reads the
//!   `VkBuffer`. A dma-buf import from the dma-bufs kept, never from the source again, so the
//!   buffer lies over the pages it was made over even where `host_range_remapped` told the device
//!   they were replaced since, as a buffer made before that call keeps its pages; a dma-buf holds
//!   its pages for as long as it is open, and adds nothing to a submission's list until it is
//!   imported. A pointer import at its pointer, over the pages the kept import read, which
//!   `HostMemory::new`'s contract keeps mapped there while the buffer lives. A guest RAM run's in
//!   its chunk's import, or the run's alone. The new `VkBuffer` lies at the same offsets in the
//!   same memory type; its device address may differ, which no shader saw. Bytes the CPU wrote
//!   through `contents` meanwhile are in those pages, and so read. Counted
//!   (`WorkCounts::host_imports_reimported`); refused where the device cannot import them again (a
//!   dma-buf's: out of memory; a pointer's: as its first import may be, typed as the import
//!   answers), which the blit's call returns.
//! - **Exact.** What the buffer holds is its host pages, which a release never touches; its
//!   commands order by its key, as before; Metal's reach of it unnamed needs its Metal address,
//!   which a releasable buffer never handed out (`live_buffers.rs` holds its range only where
//!   it may be stale, and lists it again at its exposure). A read that keeps it imported finds it
//!   released only where its address was never handed out: it imports it again first.
//! - **Counted.** `WorkCounts::host_imports_released`, `host_imports_reimported`,
//!   `host_imports_resident` (releasable imports holding their memory now),
//!   `host_import_fds_kept`, and `host_imports_kept_for_fds` (imports kept as their descriptors
//!   did not fit); the device's dma-buf memories (`dma_buf_memories_live`) and its
//!   imports of host pages by their pointer (`host_imports_live`) fall as imports are released.

use super::buffer::{FixedPages, HostImport, Imported, Pages, Shape};
use super::device::lock;
use super::dma_buf::{DmaBufRequest, DmaBufSource};
use super::error::{internal, Error};
use super::exit_guard;
use super::host_memory::ImportWindow;
use super::host_ram::RunAgain;
use super::memory::MemoryClass;
use super::Device;
use std::os::fd::OwnedFd;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long an import waits idle before its first release.
pub(super) const IDLE: Duration = Duration::from_secs(2);

/// The longest an import waits idle, its window doubled at each import again.
pub(super) const IDLE_MOST: Duration = Duration::from_secs(64);

/// How often the residency thread looks for idle imports.
pub(super) const TICK: Duration = Duration::from_millis(500);

/// The share of the process's descriptor limit (`RLIMIT_NOFILE`'s soft limit) the device's kept
/// dma-bufs take at most: the rest stays the program's.
const FD_SHARE: u64 = 4;

/// The descriptors of dma-bufs the process's devices keep at most for imports they may release
/// ([`FD_SHARE`] of the process's soft limit): an import past them stays imported.
pub(super) fn kept_fds_limit() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `limit` is a valid `rlimit` to write.
        match unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } {
            0 => limit.rlim_cur / FD_SHARE,
            _ => 0,
        }
    })
}

/// The descriptors the process's devices keep now, within [`kept_fds_limit`].
static KEPT_FDS: AtomicU64 = AtomicU64::new(0);

/// Keep `count` more descriptors within the process's [`kept_fds_limit`]: whether they fit
/// (counted then).
fn keep_fds(count: u64) -> bool {
    let limit = kept_fds_limit();
    let fits = |held: u64| held.checked_add(count).filter(|after| *after <= limit);
    KEPT_FDS
        .try_update(Ordering::AcqRel, Ordering::Acquire, fits)
        .is_ok()
}

/// Milliseconds of the residency's clock: since its first reading in the process.
fn now_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// What imports a releasable import again (the module docs): what it was made of, and the
/// buffer's length, its window in the pages and its memory class.
pub(super) struct Again {
    device: Device,
    pub(super) from: AgainFrom,
    pub(super) length: u64,
    pub(super) window: ImportWindow,
    pub(super) class: MemoryClass,
}

/// What an import is made again of.
pub(super) enum AgainFrom {
    /// A descriptor of each dma-buf its source made, in order, with their lengths: duplicates of
    /// those the device imported. The source stays for the textures over its bytes, which ask it
    /// for dma-bufs of their own (`Device::dma_buf_image`), never for the import's again.
    DmaBufs {
        chunks: Vec<(OwnedFd, u64)>,
        source: DmaBufSource,
        request: DmaBufRequest,
    },
    /// The pointer its pages were imported at, on a device whose held pointer imports cost every
    /// submission.
    Pointer(ImportedAt),
    /// The run of a guest RAM it was an import of alone, made while the run's chunk was not
    /// imported (`host_ram.rs`): bound again in the chunk's import where it is made by then,
    /// else in an import of the run alone again.
    RamRun(RunAgain),
}

/// The pointer a pointer import was made at.
pub(super) struct ImportedAt(pub(super) NonNull<u8>);

// SAFETY: an address, which the residency hands to the driver to import again and never reads
// through; the caller's pages stay mapped there while the buffer lives (`HostMemory::new`).
unsafe impl Send for ImportedAt {}

impl Again {
    /// `chunks` kept, counted against the process's [`kept_fds_limit`] and the device's cap
    /// (only the crate's device tests cap it): `None`, the descriptors closed, where they do not
    /// fit, so that the import stays imported.
    pub(super) fn dma_bufs(
        device: &Device,
        chunks: Vec<(OwnedFd, u64)>,
        source: DmaBufSource,
        request: DmaBufRequest,
        (length, window, class): (u64, ImportWindow, MemoryClass),
    ) -> Option<Again> {
        let shared = device.shared();
        let count = chunks.len() as u64;
        let cap = shared.residency.shared.fds_cap.load(Ordering::Relaxed);
        if !shared.counts.host_import_fds_keep(count, cap) {
            return None;
        }
        if !keep_fds(count) {
            shared.counts.host_import_fds_closed(count);
            return None;
        }
        Some(Again {
            device: device.clone(),
            from: AgainFrom::DmaBufs {
                chunks,
                source,
                request,
            },
            length,
            window,
            class,
        })
    }

    /// The pages imported at `pointer`, made again there: `None` where a held pointer import
    /// costs the device's submissions nothing, so that a release would buy nothing (its check
    /// found so, could not run, or has not answered: `Device::pointer_imports_tax_submissions`),
    /// and the import stays imported.
    pub(super) fn pointer(
        device: &Device,
        pointer: NonNull<u8>,
        (length, window, class): (u64, ImportWindow, MemoryClass),
    ) -> Option<Again> {
        let taxed = device.pointer_imports_tax_submissions();
        taxed.then(|| Again {
            device: device.clone(),
            from: AgainFrom::Pointer(ImportedAt(pointer)),
            length,
            window,
            class,
        })
    }
}

impl Again {
    /// The import of `run` alone, a guest RAM's pages a buffer was bound in while their chunk
    /// was not imported (`host_ram.rs`): made again in the chunk's import, or the run's alone,
    /// from the RAM, which stays mapped while the buffer lives (`HostRam::new`).
    pub(super) fn ram_run(
        device: &Device,
        run: RunAgain,
        (length, window, class): (u64, ImportWindow, MemoryClass),
    ) -> Again {
        Again {
            device: device.clone(),
            from: AgainFrom::RamRun(run),
            length,
            window,
            class,
        }
    }
}

impl Drop for Again {
    fn drop(&mut self) {
        let AgainFrom::DmaBufs { chunks, .. } = &self.from else {
            return;
        };
        let count = chunks.len() as u64;
        self.device.shared().counts.host_import_fds_closed(count);
        KEPT_FDS.fetch_sub(count, Ordering::AcqRel);
    }
}

/// A releasable import's residency (the module docs).
pub(super) struct Residency {
    device: Device,
    state: Mutex<State>,
    /// When a blit last took it, in [`now_ms`]'s milliseconds.
    used: AtomicU64,
    /// No read but a blit's took it yet: cleared for good by the first that did.
    releasable: AtomicBool,
    /// The pages of the device's stand-in (`detach.rs`), which a read takes where the device
    /// cannot import it again (`HostImport::pages`): kept from the start, so that read makes
    /// nothing.
    stand_in: Arc<FixedPages>,
    /// How its first import laid the bytes, which each import again matches.
    shape: Shape,
}

struct State {
    /// Its `VkBuffer` and memory while it is imported.
    held: Option<Arc<Imported>>,
    /// What imports it again, while it may be released; dropped, its descriptors closed, once it
    /// is imported for good.
    again: Option<Again>,
    /// How long it waits idle before it is released, in milliseconds.
    window: u64,
    /// It was released once: its buffer's live-table range may be stale (`live_buffers.rs`).
    released_once: bool,
}

impl Residency {
    /// The residency of an import of `pages`, imported now, which `again` imports again;
    /// `stand_in` the device's stand-in's pages.
    pub(super) fn new(
        device: &Device,
        pages: Pages,
        again: Again,
        stand_in: Arc<FixedPages>,
    ) -> Residency {
        device.shared().counts.host_import_resident(true);
        Residency {
            shape: Shape::of(&pages),
            device: device.clone(),
            state: Mutex::new(State {
                held: Some(Arc::new(Imported::new(device, pages))),
                again: Some(again),
                window: IDLE.as_millis() as u64,
                released_once: false,
            }),
            used: AtomicU64::new(now_ms()),
            releasable: AtomicBool::new(true),
            stand_in,
        }
    }

    /// The device's stand-in's pages.
    pub(super) fn stand_in(&self) -> Arc<FixedPages> {
        Arc::clone(&self.stand_in)
    }

    /// Its `VkBuffer` and memory for a blit's command, imported again where released: what the
    /// command buffer holds until it completed. Refused where the device cannot import them
    /// again.
    pub(super) fn for_blit(&self) -> Result<Arc<Imported>, Error> {
        let mut state = lock(&self.state);
        self.used.store(now_ms(), Ordering::Relaxed);
        if let Some(held) = &state.held {
            return Ok(Arc::clone(held));
        }
        self.import_again(&mut state)
    }

    /// Its `VkBuffer` and memory for any other read, which keeps it imported for good from now
    /// on; imported again where released. `None` where the device cannot import them again.
    pub(super) fn exposed(&self) -> Option<Arc<Imported>> {
        let mut state = lock(&self.state);
        self.releasable.store(false, Ordering::Release);
        let held = match &state.held {
            Some(held) => Arc::clone(held),
            None => self.import_again(&mut state).ok()?,
        };
        // Never released again: its dma-bufs need not stay open.
        state.again = None;
        Some(held)
    }

    /// Keep it imported for good from now on, imported again now where released (where the
    /// device cannot, the next read tries again): whether it was ever released, so a range of it
    /// noted before may be stale. A caller racing another's exposure waits for its import under
    /// the lock, and is told the same.
    pub(super) fn expose(&self) -> bool {
        let mut state = lock(&self.state);
        self.releasable.store(false, Ordering::Release);
        if state.held.is_none() && self.import_again(&mut state).is_err() {
            return state.released_once;
        }
        state.again = None;
        state.released_once
    }

    /// Its `VkBuffer` and memory for a live-table listing (`live_buffers.rs`), which reads it
    /// without exposing it: `None` while a releasable import is released (its range is not
    /// listed then); else as [`Residency::exposed`].
    pub(super) fn listing(&self) -> Option<Arc<Imported>> {
        if !self.releasable.load(Ordering::Acquire) {
            return self.exposed();
        }
        lock(&self.state).held.as_ref().map(Arc::clone)
    }

    /// Release it where it is releasable, imported, held by no command buffer, and idle for its
    /// window (`force`: whatever its window) at `now`: its objects, for the caller to let go of
    /// with no lock held.
    fn release_if_idle(&self, now: u64, force: bool) -> Option<Arc<Imported>> {
        if !self.releasable.load(Ordering::Acquire) {
            return None;
        }
        let mut state = lock(&self.state);
        if !self.releasable.load(Ordering::Acquire) || state.again.is_none() {
            return None;
        }
        let held = state.held.as_ref()?;
        // A command buffer whose blit read it holds it until it completed, weakly from its commit
        // to its first submission (`rebind::ImportHold`); an exposing read holds it while it
        // reads, under this lock.
        if Arc::strong_count(held) > 1 || Arc::weak_count(held) > 0 {
            return None;
        }
        let idle = now.saturating_sub(self.used.load(Ordering::Relaxed));
        if !force && idle < state.window {
            return None;
        }
        state.released_once = true;
        let counts = &self.device.shared().counts;
        counts.host_import_released();
        counts.host_import_resident(false);
        state.held.take()
    }

    /// Import it again from its dma-bufs (`Device::import_dma_bufs_again`) or at its pointer
    /// (`Device::import_pointer_again`), under `state`'s lock, its window doubled: its objects.
    /// Refused where released for good, where the device cannot, and where it lays the bytes
    /// otherwise than the first import (an internal error: the same pages, length and window).
    fn import_again(&self, state: &mut State) -> Result<Arc<Imported>, Error> {
        let again = state
            .again
            .as_ref()
            .ok_or_else(|| internal("an import released without what imports it again"))?;
        let device = &self.device;
        let held = match &again.from {
            AgainFrom::DmaBufs {
                chunks,
                source,
                request,
            } => Imported::new(
                device,
                device.import_dma_bufs_again(chunks, source, *request, again)?,
            ),
            AgainFrom::Pointer(at) => {
                Imported::new(device, device.import_pointer_again(at.0, again)?)
            }
            // The chunk's import where it lays the bytes as the run's did (a bound buffer that
            // does not is freed at once: nothing read it), else the run's alone again.
            AgainFrom::RamRun(run) => {
                let in_chunk = (device.ram_run_in_chunk(run, again))
                    .map(|pages| Imported::new(device, pages))
                    .filter(|held| Shape::of(&held.pages).same(&self.shape));
                match in_chunk {
                    Some(held) => held,
                    None => Imported::new(device, device.ram_run_alone_again(run, again)?),
                }
            }
        };
        let laid = Shape::of(&held.pages);
        let held = Arc::new(held);
        if !laid.same(&self.shape) {
            // Freed at once: nothing read them.
            drop(held);
            return Err(internal("an import again laid its bytes otherwise"));
        }
        state.held = Some(Arc::clone(&held));
        state.window = (2 * state.window).min(IDLE_MOST.as_millis() as u64);
        let counts = &self.device.shared().counts;
        counts.host_import_reimported();
        counts.host_import_resident(true);
        Ok(held)
    }
}

impl Residency {
    /// Its objects, the import going, for the caller to leave what waits for their free with
    /// (`Imported::leave`): freed as the last of their holders lets go, the caller or the last
    /// command buffer that read them, so the caller's pages stay mapped, and its deallocator
    /// waits, while any memory over them lives (the module docs). `None` where it is released,
    /// nothing imported.
    pub(super) fn let_go(&self) -> Option<Arc<Imported>> {
        let held = lock(&self.state).held.take()?;
        self.device.shared().counts.host_import_resident(false);
        Some(held)
    }
}

impl Drop for Residency {
    fn drop(&mut self) {
        if lock(&self.state).held.is_some() {
            self.device.shared().counts.host_import_resident(false);
        }
    }
}

/// The device's releasable imports and the thread that releases them while idle (the module
/// docs): started by the first such import; holds nothing of the device.
#[derive(Default)]
pub(crate) struct Sweeper {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

struct Shared {
    state: Mutex<Swept>,
    /// The thread is asked to stop.
    stop: Condvar,
    /// The descriptors the device keeps at most, within the process's [`kept_fds_limit`]:
    /// unbounded but in the crate's device tests (`tests/host_residency.rs`).
    fds_cap: AtomicU64,
}

impl Default for Shared {
    fn default() -> Shared {
        Shared {
            state: Mutex::default(),
            stop: Condvar::new(),
            fds_cap: AtomicU64::new(u64::MAX),
        }
    }
}

#[derive(Default)]
struct Swept {
    /// The releasable imports, weakly: each goes with its buffer.
    imports: Vec<Weak<HostImport>>,
    stopping: bool,
}

impl Sweeper {
    /// Note `import`, releasable, and start the thread unless it runs: `false` where the host
    /// cannot start one (the import then stays imported).
    pub(super) fn note(&self, import: &Arc<HostImport>) -> bool {
        let mut thread = lock(&self.thread);
        if thread.is_none() {
            let shared = Arc::clone(&self.shared);
            let spawned = exit_guard::spawn(
                std::thread::Builder::new().name("metal2vulkan-residency".into()),
                move || run(&shared),
            );
            match spawned {
                Ok(handle) => *thread = Some(handle),
                Err(_) => return false,
            }
        }
        lock(&self.shared.state)
            .imports
            .push(Arc::downgrade(import));
        true
    }

    /// Release every releasable import idle now (`force`: whatever its window), on this thread:
    /// how many.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
    pub(super) fn sweep(&self, force: bool) -> usize {
        sweep(&self.shared, force)
    }

    /// The device's teardown: the thread ends, joined unless this is it.
    pub(crate) fn stop(&self) {
        lock(&self.shared.state).stopping = true;
        self.shared.stop.notify_all();
        let mut thread = lock(&self.thread);
        let own = thread.as_ref().is_some_and(crate::this_thread::runs);
        if !own {
            if let Some(handle) = thread.take() {
                let _ = handle.join();
            }
        }
    }
}

/// The thread's loop: a sweep every [`TICK`] until stopped.
fn run(shared: &Shared) {
    loop {
        {
            let state = lock(&shared.state);
            let (state, _) = shared
                .stop
                .wait_timeout(state, TICK)
                .unwrap_or_else(PoisonError::into_inner);
            if state.stopping {
                return;
            }
        }
        sweep(shared, false);
    }
}

/// Release the noted imports idle now (`force`: whatever their windows), forgetting those gone:
/// how many. Each import and its objects are let go of with no lock held: either may be its
/// device's last holder.
fn sweep(shared: &Shared, force: bool) -> usize {
    let imports: Vec<Arc<HostImport>> = {
        let mut state = lock(&shared.state);
        state.imports.retain(|import| import.strong_count() > 0);
        state.imports.iter().filter_map(Weak::upgrade).collect()
    };
    let now = now_ms();
    let mut released = 0;
    for import in imports {
        let objects = import
            .residency()
            .and_then(|residency| residency.release_if_idle(now, force));
        released += usize::from(objects.is_some());
        drop(objects);
        drop(import);
    }
    released
}

/// The crate's device tests' hooks (`tests/host_residency.rs`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
impl Device {
    /// Release now every import of a no-copy buffer's dma-bufs that the device releases while
    /// idle, whatever its idle window: those no command buffer holds, whose Metal address was
    /// never handed out and whose bytes only blits read. Each is imported again at its next
    /// blit. How many it released.
    pub(crate) fn release_idle_host_imports(&self) -> usize {
        self.shared().residency.sweep(true)
    }

    /// Keep at most `cap` descriptors of dma-bufs for the device's releasable imports from now
    /// on, within the process's own bound: a no-copy buffer made past it keeps its import
    /// imported, as one past the process's bound does.
    pub(crate) fn cap_kept_host_import_fds(&self, cap: u64) {
        let shared = &self.shared().residency.shared;
        shared.fds_cap.store(cap, Ordering::Relaxed);
    }
}
