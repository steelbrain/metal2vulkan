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
//!     pointer), where the device kept a descriptor of each dma-buf of its own within the
//!     descriptors the process's devices keep at most ([`kept_fds_limit`]), and the `Arc` of each
//!     shared file its other pieces lie in, which opens no descriptor; an import past them is
//!     made as any other and stays imported. A piece of a shared file lies in the device's one
//!     import of the file, which the buffers and heaps bound in it hold, and which leaves the
//!     submissions' list only with the last: such imports go together ("A shared file counts
//!     once", below);
//!   - of its pages' pointer, on a device whose check found a held pointer import adds to every
//!     submission's cost (`submit_check.rs`, the import tax: RADV and ANV on i915 pay one, NVIDIA's
//!     driver and MoltenVK none measured): the caller's pages stay mapped at the pointer, with the
//!     same backing, while the buffer lives (`HostMemory::new`), so an import again at the same
//!     pointer reads the pages the kept import read, on every driver, and releasing it changes
//!     nothing a command reads. Elsewhere, and in a buffer made before the check answered, a
//!     pointer import stays imported: a release would spare no submission any cost, and the
//!     import again would cost the next blit.
//!
//!   A buffer bound in a guest RAM's chunk (`host_ram.rs`) has no import of its own to release.
//! - **Blits alone.** It stays releasable while every read of its `VkBuffer` and memory was a
//!   blit's (`Buffer::blit_at`: a copy or fill naming the buffer, a copy between it and a
//!   texture, `synchronize_resource`), which holds what it read until its command buffer
//!   completed ([`Imported`]), weakly from the commit until the first submission
//!   (`rebind::ImportHold`). Any other read keeps it imported for good from then on, before the
//!   read returns: its Metal address handed out (`gpu_address`, so a binding, an argument
//!   buffer, an argument table, an indirect command buffer or `use_resource`), its `VkBuffer`
//!   handed out (`vk_buffer`), a texture or texel view over its bytes bound to its memory or
//!   `VkBuffer` (a shadow of them, or an image in place in an import of their pointer), a buffer
//!   bound inside its memory with a `VkBuffer` of its own or made to share its buffer's range
//!   (`host_aliases.rs`, `BufferDescriptor::range_of`), or a no-copy heap's own bytes or a
//!   rebound buffer lying in it (a heap's placed buffers' addresses are the heap's). So no
//!   pointer, descriptor or image other than a blit's command ever holds a released import's
//!   objects.
//! - **Several buffers in one import.** Every no-copy buffer made over the bytes the import's
//!   buffers were made over lies in it (`host_aliases.rs`, "One memory, one import"), and reads
//!   it as they do, through the import: its blits take the import's objects as the first
//!   buffer's do, imported again where released, and any other read of it keeps the import
//!   imported for good as the first buffer's would. So the import stays releasable while every
//!   read of each buffer in it was a blit's, or none: a buffer that is never read, made only to
//!   keep the bytes imported (a consumer's holder of a view of its guest's pages), keeps the
//!   import no more than a buffer only blits read does. Counted:
//!   `WorkCounts::host_imports_joined_releasable`. A texture in place over a dma-buf import's
//!   bytes whose image is bound in that import (`texture/in_place_image.rs`, where the import is
//!   dedicated to nothing and the device's check of such images passed; never on RADV, whose such
//!   images require a dedicated allocation) is a read apart as an image in a pointer import is:
//!   the import is kept imported for good from its bind on, and the image adds no import. One
//!   whose `DRM_FORMAT_MOD_LINEAR` image lies in an import of a dma-buf of its own pages, which it
//!   asks the buffer's source for as it is made, is not: it reads the buffer's import as a blit
//!   does (imported again where released, `dma_buf.rs`) and binds none of the buffer's memory, so
//!   the buffer's import stays releasable beside it. Each such image is a listed import of its own
//!   on RADV, and the buffer's import a second one only while blits use it.
//! - **Released while idle** by the device's residency thread ([`Sweeper`]), every [`TICK`]: an
//!   import no command buffer holds, even weakly (every one that read it completed) and that no
//!   blit took for its window, [`IDLE`] at first and doubled at each import again, up to
//!   [`IDLE_MOST`], so an import used now and then stays imported and an unused one goes. Its
//!   `VkBuffer` is destroyed and its memory freed; the buffer, its Metal address, its host pages,
//!   the dma-bufs and the caller's release stay. The crate's device tests release them at once,
//!   whatever their windows (`tests/host_residency.rs`, `tests/pointer_residency.rs`). The doubling
//!   bounds the churn whatever the program's pattern: an import is made again at most five times
//!   before its window reaches [`IDLE_MOST`], and at most once per such window after. A sweep
//!   that releases several paces them (`step_aside::pace`): before each, it waits until as long as
//!   its frees before it took has passed since the last ended, so that the device's other
//!   threads' submissions go between them, as NVIDIA's driver holds every one while it frees an
//!   import (docs/FAST-PATHS.md, known gaps). Not while the device's teardown waits for the
//!   thread.
//! - **A shared file counts once.** Each pass groups the releasable imports by the shared
//!   files their pieces lie in: those that hold one file's import go together, joined through
//!   every file each holds, and each other alone ([`groups`]). A group goes whole or not at all:
//!   all its imports released at once (idle, each past its own window, or past the most below),
//!   and only where every hold of each of its files is one of theirs, so that the release lets go
//!   of each file's import ([`listed`]). A file a heap, a buffer kept imported for good or a
//!   command buffer still holds stays imported, and so do all its pieces: releasing them would
//!   take nothing off the submissions' list and cost each piece an import again at its next use.
//!   Each released import is counted as any other.
//! - **Released past the most,** on a device whose check found a held import adds to every
//!   submission's cost (`submit_check.rs`, the import tax): the releasable imports hold at most
//!   [`RESIDENT_MOST`] imports on the device's list at once, a shared file's once and each
//!   import's own once. After each sweep's idle releases, and as soon as one more may be past the
//!   most (the residency thread is woken: [`Sweeper::resident`]), the thread releases the groups
//!   past it that no command buffer holds, the least recently used blit's first, whatever their
//!   windows, paced as above. A window keeps an import used every few seconds imported for good,
//!   so a program that blits from many buffers in turn kept them all listed: a guest's 3D
//!   benchmark on the RX 7900 XTX (RADV) held 1,000 to 1,470 releasable imports, its submissions
//!   taking 0.8 to 1.0 ms each, against about 0.1 ms at its desktop with about 110 imports.
//!   Counted: `WorkCounts::host_imports_released_over_most`, within `host_imports_released`.
//! - **Parked images.** An image in place over a no-copy buffer's dma-buf pages, kept with its
//!   own import under the import of the pages as its texture goes, for the next texture over the
//!   same pages (`dma_buf/parked.rs`), is one import on the device's list: counted against the
//!   most while it is parked, released past it among the groups in the order of its park, and
//!   released by the thread at its first pass after the park where the device's held imports
//!   cost every submission, else once parked for [`IDLE`] (a parked image is taken or released,
//!   never imported again, so its window never doubles; [`parked_window`]). The device parks
//!   [`PARKED_MOST`] at most.
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
//!   `HostMemory::new`'s contract keeps mapped there while the buffer lives. The new `VkBuffer` lies at the same offsets in the
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
//!   `host_imports_resident` (releasable imports holding their memory now, which the most above
//!   bounds through the imports on the list they hold), `host_imports_held_for_good` (imports a
//!   read other than a blit's kept imported for good, holding their memory now: never released,
//!   whatever the most), `host_import_fds_kept`,
//!   and `host_imports_kept_for_fds` (imports kept as their descriptors did not fit); the
//!   device's dma-buf memories (`dma_buf_memories_live`) and its imports of host pages by their
//!   pointer (`host_imports_live`) fall as imports are released.

use super::buffer::{FixedPages, HostImport, Imported, Pages, Shape};
use super::device::lock;
#[cfg(unix)]
use super::dma_buf::{DmaBufRequest, DmaBufSource, Piece};
use super::error::{internal, Error};
use super::exit_guard;
use super::host_memory::ImportWindow;
use super::memory::MemoryClass;
use super::Device;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long an import waits idle before its first release.
pub(super) const IDLE: Duration = Duration::from_secs(2);

/// The longest an import waits idle, its window doubled at each import again.
pub(super) const IDLE_MOST: Duration = Duration::from_secs(64);

/// How often the residency thread looks for idle imports.
pub(super) const TICK: Duration = Duration::from_millis(500);

/// The imports on its list that the releasable imports of a device whose held imports cost every
/// submission keep at most, a shared file's once (the module docs, "Released past the most"). Each
/// costs every kernel submission about 0.45 us on a Radeon 780M (RADV, the consumer's measurement:
/// an empty command buffer's submission took 0.45 ms more with 1,000 udmabuf dma-bufs imported,
/// whatever their size), so 64 cost about 29 us a submission. A guest's 3D benchmark's blits used
/// about 480 imports a second at 57 frames a second: 64 are about eight frames' worth.
pub(super) const RESIDENT_MOST: usize = 64;

/// The images a device parks at most under the imports of host pages (`dma_buf/parked.rs`): past
/// them an image in place over a no-copy buffer's dma-buf pages goes as it did, its import freed.
/// As many as [`RESIDENT_MOST`], so that on a device whose held imports cost every submission a
/// texture's park never takes that many releasable imports' places; elsewhere the bound alone.
#[cfg(unix)]
pub(super) const PARKED_MOST: u64 = RESIDENT_MOST as u64;

/// The share of the process's descriptor limit (`RLIMIT_NOFILE`'s soft limit) the device's kept
/// dma-bufs take at most: the rest stays the program's.
#[cfg(unix)]
const FD_SHARE: u64 = 4;

/// The descriptors of dma-bufs the process's devices keep at most for imports they may release
/// ([`FD_SHARE`] of the process's soft limit): an import past them stays imported.
#[cfg(unix)]
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
#[cfg(unix)]
static KEPT_FDS: AtomicU64 = AtomicU64::new(0);

/// Keep `count` more descriptors within the process's [`kept_fds_limit`]: whether they fit
/// (counted then).
#[cfg(unix)]
fn keep_fds(count: u64) -> bool {
    let limit = kept_fds_limit();
    let fits = |held: u64| held.checked_add(count).filter(|after| *after <= limit);
    KEPT_FDS
        .try_update(Ordering::AcqRel, Ordering::Acquire, fits)
        .is_ok()
}

/// Milliseconds of the residency's clock: since its first reading in the process.
pub(super) fn now_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// What imports a releasable import again (the module docs): what it was made of, and the
/// buffer's length, its window in the pages and its memory class.
pub(super) struct Again {
    #[cfg(unix)]
    device: Device,
    pub(super) from: AgainFrom,
    pub(super) length: u64,
    pub(super) window: ImportWindow,
    pub(super) class: MemoryClass,
}

/// What an import is made again of.
pub(super) enum AgainFrom {
    /// The pieces its source made, in order: a duplicate of the descriptor of each dma-buf of its
    /// own the device imported, and a clone of each shared file's `Arc` with its piece's place
    /// (`dma_buf.rs`'s `Piece::again`). The source stays for the textures over its bytes, which
    /// ask it for dma-bufs of their own (`Device::dma_buf_image`), never for the import's again.
    #[cfg(unix)]
    DmaBufs {
        chunks: Vec<Piece>,
        source: DmaBufSource,
        request: DmaBufRequest,
    },
    /// The pointer its pages were imported at, on a device whose held pointer imports cost every
    /// submission.
    Pointer(ImportedAt),
}

/// The pointer a pointer import was made at.
pub(super) struct ImportedAt(pub(super) NonNull<u8>);

// SAFETY: an address, which the residency hands to the driver to import again and never reads
// through; the caller's pages stay mapped there while the buffer lives (`HostMemory::new`).
unsafe impl Send for ImportedAt {}

impl Again {
    /// `chunks` kept, their descriptors of their own counted against the process's
    /// [`kept_fds_limit`] and the device's cap (only the crate's device tests cap it): `None`,
    /// the descriptors closed, where they do not fit, so that the import stays imported. A shared
    /// file's `Arc` opens no descriptor and counts none.
    #[cfg(unix)]
    pub(super) fn dma_bufs(
        device: &Device,
        chunks: Vec<Piece>,
        source: DmaBufSource,
        request: DmaBufRequest,
        (length, window, class): (u64, ImportWindow, MemoryClass),
    ) -> Option<Again> {
        let shared = device.shared();
        let count = own_descriptors(&chunks);
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
        if !device.pointer_imports_tax_submissions() {
            return None;
        }
        Some(Again {
            #[cfg(unix)]
            device: device.clone(),
            from: AgainFrom::Pointer(ImportedAt(pointer)),
            length,
            window,
            class,
        })
    }
}

#[cfg(unix)]
impl Drop for Again {
    fn drop(&mut self) {
        let AgainFrom::DmaBufs { chunks, .. } = &self.from else {
            return;
        };
        let count = own_descriptors(chunks);
        self.device.shared().counts.host_import_fds_closed(count);
        KEPT_FDS.fetch_sub(count, Ordering::AcqRel);
    }
}

/// The descriptors of their own `pieces` hold: one per dma-buf of its own.
#[cfg(unix)]
fn own_descriptors(pieces: &[Piece]) -> u64 {
    pieces.iter().filter(|piece| piece.own()).count() as u64
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
    /// The imports on the device's list its import holds ([`Holds::weight`]): what it counts for
    /// against the most while it holds its memory ([`Sweeper::resident`]). Each import again
    /// holds the same, from the same dma-bufs or pointer.
    weight: u64,
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
    /// Whether the import holds its `VkBuffer` and memory now, where its state is free to read
    /// (a failed submission's line, `submit_failure.rs`): it never waits for a sweep.
    pub(super) fn held_now(&self) -> Option<bool> {
        crate::submit_failure::try_lock(&self.state).map(|state| state.held.is_some())
    }

    /// The residency of an import of `pages`, imported now, which `again` imports again;
    /// `stand_in` the device's stand-in's pages.
    pub(super) fn new(
        device: &Device,
        pages: Pages,
        again: Again,
        stand_in: Arc<FixedPages>,
    ) -> Residency {
        let weight = Holds::of(&pages).weight();
        device.shared().counts.host_import_resident(true);
        device.shared().residency.resident(device, true, weight);
        Residency {
            weight,
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

    /// Whether it is releasable still: no read but a blit's took it yet.
    pub(super) fn releasable(&self) -> bool {
        self.releasable.load(Ordering::Acquire)
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
        self.held_for_good(&state);
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
        self.held_for_good(&state);
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

    /// No longer releasable, under `state`'s lock: where it was and holds its memory, one fewer
    /// releasable import does, and one more kept for good.
    fn held_for_good(&self, state: &State) {
        let was = self.releasable.swap(false, Ordering::AcqRel);
        if was && state.held.is_some() {
            let shared = self.device.shared();
            shared.counts.host_import_resident(false);
            shared.residency.resident(&self.device, false, self.weight);
            shared.counts.host_import_held_for_good(true);
        }
    }

    /// It came to hold its memory (`held`) or let go of it, under `state`'s lock: counted as a
    /// releasable import (`WorkCounts::host_imports_resident`, and against the device's most)
    /// while it is one, else as one kept for good (`WorkCounts::host_imports_held_for_good`).
    fn holds_memory(&self, held: bool) {
        let shared = self.device.shared();
        match self.releasable.load(Ordering::Acquire) {
            true => {
                shared.counts.host_import_resident(held);
                shared.residency.resident(&self.device, held, self.weight);
            }
            false => shared.counts.host_import_held_for_good(held),
        }
    }

    /// When a blit last took it and what its import holds, where it is releasable and holds its
    /// memory now: what a pass groups its releases by and orders them by (the module docs).
    fn resident_holds(&self) -> Option<(u64, Holds)> {
        let state = lock(&self.state);
        let held = (state.held.as_ref()).filter(|_| self.releasable.load(Ordering::Acquire))?;
        Some((self.used.load(Ordering::Relaxed), Holds::of(&held.pages)))
    }

    /// Release `group` together, all or none: where each is releasable, imported, held by no
    /// command buffer and idle for its window (`force`: whatever its window) at `now`, and every
    /// hold of each shared file's import their imports hold is one of theirs ([`listed`]). Their
    /// objects, for the caller to let go of with no lock held, and the imports on the device's
    /// list that lets go of. Their locks are taken in one order, by address; no other path holds
    /// two.
    fn release_together(
        group: &[&Residency],
        now: u64,
        force: bool,
    ) -> Option<(Vec<Arc<Imported>>, u64)> {
        let releasable = |residency: &Residency| residency.releasable.load(Ordering::Acquire);
        if !group.iter().all(|residency| releasable(residency)) {
            return None;
        }
        let mut group = group.to_vec();
        group.sort_by_key(|residency| std::ptr::from_ref(*residency).addr());
        let mut states: Vec<MutexGuard<'_, State>> = group
            .iter()
            .map(|residency| lock(&residency.state))
            .collect();
        let mut holds = Vec::with_capacity(group.len());
        for (residency, state) in group.iter().zip(&states) {
            if !releasable(residency) || state.again.is_none() {
                return None;
            }
            let held = state.held.as_ref()?;
            // A command buffer whose blit read it holds it until it completed, weakly from its
            // commit to its first submission (`rebind::ImportHold`); an exposing read holds it
            // while it reads, under its lock.
            if Arc::strong_count(held) > 1 || Arc::weak_count(held) > 0 {
                return None;
            }
            let idle = now.saturating_sub(residency.used.load(Ordering::Relaxed));
            if !force && idle < state.window {
                return None;
            }
            holds.push(Holds::of(&held.pages));
        }
        let listed = listed(&holds)?;
        let mut objects = Vec::with_capacity(group.len());
        for (residency, state) in group.iter().zip(&mut states) {
            state.released_once = true;
            residency.device.shared().counts.host_import_released();
            residency.holds_memory(false);
            objects.extend(state.held.take());
        }
        Some((objects, listed))
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
            #[cfg(unix)]
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
        self.device.shared().counts.host_import_reimported();
        self.holds_memory(true);
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
        let mut state = lock(&self.state);
        let held = state.held.take()?;
        self.holds_memory(false);
        drop(state);
        Some(held)
    }
}

impl Drop for Residency {
    fn drop(&mut self) {
        if lock(&self.state).held.is_some() {
            self.holds_memory(false);
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
    #[cfg(unix)]
    fds_cap: AtomicU64,
    /// The imports on the device's list the releasable imports that hold their memory now hold,
    /// each import's counted ([`Residency::weight`]): a shared file's once for each releasable
    /// import that holds a piece of it.
    resident: AtomicU64,
    /// How many more `resident` counts than the imports on the list it stands for, as the last
    /// release past the most found it (a shared file's holds past its first, and those of files a
    /// release cannot let go of), lowered with `resident`: `resident` less it is what a new one
    /// holds the most to, so that pieces of one file wake no thread.
    gap: AtomicU64,
    /// The most listed imports the device keeps ([`Shared::most`]): `u64::MAX` for the device's
    /// own, as all but the crate's device tests keep.
    most: AtomicU64,
    /// Held by a release of those past the most, so that two never release the same excess.
    over_most: Mutex<()>,
    /// The images parked under the imports of host pages now (`dma_buf/parked.rs`), within
    /// `parked_most`.
    #[cfg(unix)]
    parked: AtomicU64,
    /// The images the device parks at most: [`PARKED_MOST`], but in the crate's device tests.
    #[cfg(unix)]
    parked_most: AtomicU64,
}

impl Default for Shared {
    fn default() -> Shared {
        Shared {
            state: Mutex::default(),
            stop: Condvar::new(),
            #[cfg(unix)]
            fds_cap: AtomicU64::new(u64::MAX),
            resident: AtomicU64::new(0),
            gap: AtomicU64::new(0),
            most: AtomicU64::new(u64::MAX),
            over_most: Mutex::new(()),
            #[cfg(unix)]
            parked: AtomicU64::new(0),
            #[cfg(unix)]
            parked_most: AtomicU64::new(PARKED_MOST),
        }
    }
}

impl Shared {
    /// The imports on its list that `device`'s releasable imports keep at most: [`RESIDENT_MOST`]
    /// where its held imports cost every submission (`Device::pointer_imports_tax_submissions`),
    /// else no bound.
    fn most(&self, device: &Device) -> usize {
        match self.most.load(Ordering::Relaxed) {
            u64::MAX => match device.pointer_imports_tax_submissions() {
                true => RESIDENT_MOST,
                false => usize::MAX,
            },
            most => usize::try_from(most).unwrap_or(usize::MAX),
        }
    }
}

#[derive(Default)]
struct Swept {
    /// The releasable imports, weakly: each goes with its buffer.
    imports: Vec<Weak<HostImport>>,
    /// The imports images were parked under, weakly, each once (`dma_buf/parked.rs`).
    #[cfg(unix)]
    parked: Vec<Weak<HostImport>>,
    /// More releasable imports hold their memory than the most the device keeps: the thread
    /// sweeps at once.
    over: bool,
    stopping: bool,
}

impl Sweeper {
    /// Note `import`, releasable, and start the thread unless it runs: `false` where the host
    /// cannot start one (the import then stays imported).
    pub(super) fn note(&self, import: &Arc<HostImport>) -> bool {
        if !self.started() {
            return false;
        }
        lock(&self.shared.state)
            .imports
            .push(Arc::downgrade(import));
        true
    }

    /// Start the thread unless it runs: `false` where the host cannot start one.
    fn started(&self) -> bool {
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
        true
    }

    /// Take a place for an image of `device` parked under `import` (`dma_buf/parked.rs`): within
    /// [`PARKED_MOST`], the thread started and the import noted, the image counted as one more
    /// import on the device's list ([`Sweeper::resident`]). `false` where the device parks as
    /// many as it parks at most, or the host cannot start the thread: the image is not parked.
    #[cfg(unix)]
    pub(super) fn park(&self, device: &Device, import: &Arc<HostImport>) -> bool {
        let shared = &self.shared;
        let most = shared.parked_most.load(Ordering::Relaxed);
        let fits = |count: u64| (count < most).then_some(count + 1);
        if (shared.parked)
            .try_update(Ordering::AcqRel, Ordering::Acquire, fits)
            .is_err()
        {
            return false;
        }
        if !self.started() {
            shared.parked.fetch_sub(1, Ordering::AcqRel);
            return false;
        }
        {
            let mut state = lock(&shared.state);
            state.parked.retain(|noted| noted.strong_count() > 0);
            let noted = (state.parked.iter()).any(|noted| noted.as_ptr() == Arc::as_ptr(import));
            if !noted {
                state.parked.push(Arc::downgrade(import));
            }
        }
        self.resident(device, true, 1);
        true
    }

    /// `count` images of `device` parked under the imports of its host pages were taken or
    /// released: their places given back.
    #[cfg(unix)]
    pub(super) fn unparked(&self, device: &Device, count: usize) {
        if count == 0 {
            return;
        }
        let fewer = |parked: u64| Some(parked.saturating_sub(count as u64));
        let _ = (self.shared.parked).try_update(Ordering::AcqRel, Ordering::Acquire, fewer);
        self.resident(device, false, count as u64);
    }

    /// Release the parked images over any page of the host bytes `start..end`, rounded out to
    /// `unit` (`Device::host_range_remapped`), on this thread.
    #[cfg(unix)]
    pub(super) fn remapped(&self, start: u64, end: u64, unit: u64) {
        if self.shared.parked.load(Ordering::Acquire) == 0 {
            return;
        }
        for import in parked_noted(&self.shared) {
            let Some(live) = import.upgrade() else {
                continue;
            };
            let images = (live.parked_images()).take_where(|image| image.meets((start, end), unit));
            drop(live);
            crate::dma_buf::parked::release(images);
        }
    }

    /// A releasable import of `device` that holds `weight` imports on its list came to hold its
    /// memory (`up`), or let go of it (a parked image, one: `dma_buf/parked.rs`): one that may
    /// take the listed imports past the most the device keeps (`resident` less [`Shared::gap`])
    /// wakes the thread, which releases those past it. One that lets go lowers the gap too, so
    /// that the count stays a bound: a missed excess can only come from a race with a pass, or a
    /// file a release could not let go of becoming one it can, and the thread's next pass, every
    /// [`TICK`], finds it.
    fn resident(&self, device: &Device, up: bool, weight: u64) {
        let shared = &self.shared;
        if !up {
            let fewer = |count: u64| Some(count.saturating_sub(weight));
            for count in [&shared.resident, &shared.gap] {
                let _ = count.try_update(Ordering::Relaxed, Ordering::Relaxed, fewer);
            }
            return;
        }
        let resident = shared.resident.fetch_add(weight, Ordering::Relaxed) + weight;
        let listed = resident.saturating_sub(shared.gap.load(Ordering::Relaxed));
        if usize::try_from(listed).unwrap_or(usize::MAX) > shared.most(device) {
            lock(&shared.state).over = true;
            shared.stop.notify_all();
        }
    }

    /// Release every releasable import idle now (`force`: whatever its window), and every image
    /// parked for its window (`force`: every one), on this thread: how many imports.
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
    crate::step_aside::mark_a_paced_thread();
    loop {
        {
            let mut state = lock(&shared.state);
            if !state.over && !state.stopping {
                state = (shared.stop.wait_timeout(state, TICK))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            if state.stopping {
                return;
            }
            state.over = false;
        }
        sweep(shared, false);
    }
}

/// What a releasable import holds of its device's imports now (the module docs, "A shared file
/// counts once"): an import of its own, and its holds of shared files' imports.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Holds {
    /// It holds an import of its own: of its pointer, or of dma-bufs of their own.
    pub(super) own: bool,
    /// The shared files' imports it holds, each once.
    pub(super) files: Vec<FileHolds>,
}

/// A releasable import's holds of one shared file's import (`dma_buf.rs`'s `DmaBufFile`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FileHolds {
    /// The file's import, by its address: the device's one import of it.
    pub(super) file: usize,
    /// The holds of it the import's pieces take: one per piece.
    pub(super) pieces: usize,
    /// Every hold of it now, whoever holds it: a release lets go of the import only once every
    /// one is a released import's.
    pub(super) all: usize,
}

impl Holds {
    /// What the import of `pages` holds: one of its pointer, or of dma-bufs of their own, is its
    /// own; each shared file's import it lies in is held once per piece (`DmaBufPages::listed`).
    #[cfg(unix)]
    fn of(pages: &Pages) -> Holds {
        let Some(dma_bufs) = pages.placement().dma_buf_pages() else {
            return Holds {
                own: true,
                files: Vec::new(),
            };
        };
        let (own, files) = dma_bufs.listed();
        let mut holds = Holds {
            own: own || files.is_empty(),
            files: Vec::new(),
        };
        for file in files {
            let at = Arc::as_ptr(file).addr();
            match holds.files.iter_mut().find(|held| held.file == at) {
                Some(held) => held.pieces += 1,
                None => holds.files.push(FileHolds {
                    file: at,
                    pieces: 1,
                    all: Arc::strong_count(file),
                }),
            }
        }
        holds
    }

    /// [`Holds::of`] where no import is of dma-bufs: every import is of its pointer, its own.
    #[cfg(not(unix))]
    fn of(_pages: &Pages) -> Holds {
        Holds {
            own: true,
            files: Vec::new(),
        }
    }

    /// The imports on the device's list it holds: its own, and each shared file's.
    pub(super) fn weight(&self) -> u64 {
        u64::from(self.own) + self.files.len() as u64
    }
}

/// The imports on the device's list that releasing the imports of `holds` together lets go of:
/// each one's own, and each shared file's once, where every hold of each file is one of theirs
/// (each read the same count of its holds); `None` where one is not, the release then leaving the
/// file imported for another holder, which uses it (the module docs).
pub(super) fn listed<'a>(holds: impl IntoIterator<Item = &'a Holds>) -> Option<u64> {
    let mut files: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut own = 0;
    for held in holds {
        own += u64::from(held.own);
        for file in &held.files {
            let (pieces, all) = files.entry(file.file).or_insert((0, file.all));
            if *all != file.all {
                return None;
            }
            *pieces += file.pieces;
        }
    }
    let whole = files.values().all(|(pieces, all)| pieces == all);
    whole.then(|| own + files.len() as u64)
}

/// The imports of `holds` as a pass releases them (the module docs, "A shared file counts once"):
/// those that hold one shared file's import in one group, joined through every file each holds,
/// each other alone; each group's members by their index in `holds`, in order, and what releasing
/// them together lets go of ([`listed`]).
pub(super) fn groups(holds: &[Holds]) -> Vec<(Vec<usize>, Option<u64>)> {
    let mut parent: Vec<usize> = (0..holds.len()).collect();
    let mut first = HashMap::new();
    for (index, held) in holds.iter().enumerate() {
        for file in &held.files {
            let other = *first.entry(file.file).or_insert(index);
            let (one, two) = (root(&mut parent, other), root(&mut parent, index));
            parent[one.max(two)] = one.min(two);
        }
    }
    let mut at = HashMap::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for index in 0..holds.len() {
        let top = root(&mut parent, index);
        let group = *at.entry(top).or_insert_with(|| {
            members.push(Vec::new());
            members.len() - 1
        });
        members[group].push(index);
    }
    (members.into_iter())
        .map(|members| {
            let listed = listed(members.iter().map(|&index| &holds[index]));
            (members, listed)
        })
        .collect()
}

/// The first index of `index`'s group in the forest of `parent` links, its path shortened.
fn root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

/// A group of the noted imports, releasable and holding their memory as a pass found them
/// ([`groups`]): its members weakly, as the pass holds none of them while it waits; the latest
/// time a blit took one of them; and the imports on the device's list releasing them together
/// lets go of, `None` where another holder keeps one of its files' imports ([`listed`]).
struct Group {
    imports: Vec<Weak<HostImport>>,
    used: u64,
    listed: Option<u64>,
}

/// The noted imports releasable and holding their memory now, forgetting those gone, in their
/// groups ([`groups`]), and the most listed imports their device keeps ([`Shared::most`]): `None`
/// where none is.
fn resident_groups(shared: &Shared) -> (Vec<Group>, Option<usize>) {
    let imports: Vec<Weak<HostImport>> = {
        let mut state = lock(&shared.state);
        state.imports.retain(|import| import.strong_count() > 0);
        state.imports.clone()
    };
    let (mut found, mut holds, mut most) = (Vec::new(), Vec::new(), None);
    for import in imports {
        let Some(live) = import.upgrade() else {
            continue;
        };
        let Some(residency) = live.residency() else {
            continue;
        };
        if let Some((used, held)) = residency.resident_holds() {
            most.get_or_insert_with(|| shared.most(&residency.device));
            found.push((import, used));
            holds.push(held);
        }
    }
    let groups = (groups(&holds).into_iter())
        .map(|(members, listed)| Group {
            imports: (members.iter())
                .map(|&index| Weak::clone(&found[index].0))
                .collect(),
            used: (members.iter())
                .map(|&index| found[index].1)
                .max()
                .unwrap_or(0),
            listed,
        })
        .collect();
    (groups, most)
}

/// Release `group` together where it may go now (`Residency::release_together`; `force`: whatever
/// its windows), paced after the frees before it (`step_aside::pace`) while holding none of its
/// imports, so a buffer the program releases meanwhile frees its import in that release; its
/// imports and objects let go of with no lock held, as any may be its device's last holder. The
/// imports released, each counted as released past the most where `past_most`, and the imports on
/// the device's list that let go of; `None` where none was, or one of them is gone.
fn release_group(
    shared: &Shared,
    group: &Group,
    force: bool,
    past_most: bool,
) -> Option<(usize, u64)> {
    let waited = crate::step_aside::pace(|| lock(&shared.state).stopping);
    let imports: Vec<Arc<HostImport>> = (group.imports.iter())
        .map(Weak::upgrade)
        .collect::<Option<_>>()?;
    let residencies: Vec<&Residency> = (imports.iter())
        .map(|import| import.residency())
        .collect::<Option<_>>()?;
    let counts = &residencies.first()?.device.shared().counts;
    if let Some(waited) = waited {
        counts.memory_free_paced(waited);
    }
    let (objects, listed) = Residency::release_together(&residencies, now_ms(), force)?;
    let released = objects.len();
    if past_most {
        (0..released).for_each(|_| counts.host_import_released_over_most());
    }
    drop(objects);
    drop(residencies);
    drop(imports);
    Some((released, listed))
}

/// Release the noted imports idle now (`force`: whatever their windows), a group at a time
/// ([`release_group`]), and the images parked for their window (`force`: every one,
/// [`release_idle_parked`]), then those past the most ([`release_over_most`]): how many imports.
fn sweep(shared: &Shared, force: bool) -> usize {
    let (groups, _) = resident_groups(shared);
    let mut released = 0;
    for group in groups.iter().filter(|group| group.listed.is_some()) {
        if let Some((count, _)) = release_group(shared, group, force, false) {
            released += count;
        }
    }
    #[cfg(unix)]
    release_idle_parked(shared, force);
    released + release_over_most(shared)
}

/// The imports images were parked under, weakly, forgetting those gone.
#[cfg(unix)]
fn parked_noted(shared: &Shared) -> Vec<Weak<HostImport>> {
    let mut state = lock(&shared.state);
    state.parked.retain(|import| import.strong_count() > 0);
    state.parked.clone()
}

/// How long, in the residency's milliseconds, an image of `device` stays parked before the
/// thread releases it: none, so the thread's first pass after the park (within [`TICK`])
/// releases it, where the device's held imports cost every submission
/// (`Device::pointer_imports_tax_submissions`); else [`IDLE`]. There each parked image is one more
/// import on every submission's list, about 0.46 us a submission on a Radeon 780M (RADV), 0.53 ms
/// a second at 1,150 submissions a second, where a take saves the texture's making: on the 780M,
/// 0.13 ms for 7.9 MiB over 2 MiB pages to 2.6 ms for 84 MiB, and 0.7 to 24 ms over 4 KiB pages
/// (`tests::texture_shadow_bench`, medians). Held 2 s, an image never taken cost more than most
/// takes save; held to the first pass, one taken soon after its park is still taken.
#[cfg(unix)]
fn parked_window(device: &Device) -> u64 {
    match device.pointer_imports_tax_submissions() {
        true => 0,
        false => IDLE.as_millis() as u64,
    }
}

/// Release the images parked for their window ([`parked_window`]; `force`: every one), each
/// paced as an import's release is (`step_aside::pace`), holding none of the imports they were
/// parked under while it waits.
#[cfg(unix)]
fn release_idle_parked(shared: &Shared, force: bool) {
    for import in parked_noted(shared) {
        let now = now_ms();
        let Some(live) = import.upgrade() else {
            continue;
        };
        let images = (live.parked_images()).take_where(|image| {
            force || now.saturating_sub(image.since()) >= parked_window(image.device())
        });
        drop(live);
        for image in images {
            release_parked(shared, image);
        }
    }
}

/// Release `image`, a parked image taken out, paced after the frees before it.
#[cfg(unix)]
fn release_parked(shared: &Shared, image: crate::dma_buf::parked::ParkedImage) {
    if let Some(waited) = crate::step_aside::pace(|| lock(&shared.state).stopping) {
        image.device().shared().counts.memory_free_paced(waited);
    }
    crate::dma_buf::parked::release(vec![image]);
}

/// What a release past the most may let go of: a group of the noted imports ([`Group`]), or an
/// image parked under one of them (`dma_buf/parked.rs`), one import on the device's list, by
/// the import it lies under and its `VkImage`.
enum Excess<'a> {
    Group(&'a Group),
    #[cfg(unix)]
    Parked(&'a Weak<HostImport>, ash::vk::Image),
}

impl<'a> Excess<'a> {
    /// The group this is, `None` for a parked image.
    fn group(&self) -> Option<&'a Group> {
        match self {
            Excess::Group(group) => Some(group),
            #[cfg(unix)]
            Excess::Parked(..) => None,
        }
    }
}

/// The images parked under the noted imports, each by when it was parked, the import and its
/// `VkImage`, and the most listed imports their device keeps ([`Shared::most`]): `None` where
/// none is.
#[cfg(unix)]
fn parked_ages(shared: &Shared) -> (Vec<(u64, Weak<HostImport>, ash::vk::Image)>, Option<usize>) {
    let (mut ages, mut most) = (Vec::new(), None);
    for import in parked_noted(shared) {
        let Some(live) = import.upgrade() else {
            continue;
        };
        let parked = live.parked_images().ages();
        if !parked.is_empty() {
            most.get_or_insert_with(|| shared.most(live.device()));
        }
        ages.extend((parked.into_iter()).map(|(since, raw)| (since, Weak::clone(&import), raw)));
    }
    (ages, most)
}

/// Release the noted imports' groups and the images parked under them while their listed
/// imports are past the most their device keeps ([`Shared::most`]), the group a blit took least
/// recently or the image parked first first, whatever their windows, skipping the groups a
/// command buffer holds ([`release_group`]), and note what the count of the imports on the list
/// stands for ([`Shared::gap`]): how many imports it released.
fn release_over_most(shared: &Shared) -> usize {
    let _one = lock(&shared.over_most);
    let (groups, most) = resident_groups(shared);
    #[cfg(unix)]
    let (parked, most) = {
        let (parked, parked_most) = parked_ages(shared);
        (parked, most.or(parked_most))
    };
    let mut excess: Vec<(u64, Excess<'_>)> = (groups.iter())
        .map(|group| (group.used, Excess::Group(group)))
        .collect();
    #[cfg(unix)]
    excess
        .extend((parked.iter()).map(|(since, import, raw)| (*since, Excess::Parked(import, *raw))));
    let listed: u64 = (excess.iter())
        .filter_map(|(_, excess)| match excess {
            Excess::Group(group) => group.listed,
            #[cfg(unix)]
            Excess::Parked(..) => Some(1),
        })
        .sum();
    let most = most.map_or(u64::MAX, |most| u64::try_from(most).unwrap_or(u64::MAX));
    let (mut left, mut released) = (listed, 0);
    excess.sort_by_key(|(used, _)| *used);
    for (_, excess) in &excess {
        if left <= most {
            break;
        }
        #[cfg(unix)]
        if let Excess::Parked(import, raw) = excess {
            // One taken or released meanwhile is on the list no more as a parked image.
            left = left.saturating_sub(1);
            let Some(live) = import.upgrade() else {
                continue;
            };
            let images = (live.parked_images()).take_where(|image| image.raw() == *raw);
            drop(live);
            for image in images {
                release_parked(shared, image);
            }
            continue;
        }
        let Some(group) = excess.group() else {
            continue;
        };
        let Some(listed) = group.listed else {
            continue;
        };
        // One gone meanwhile holds its imports no more.
        if group
            .imports
            .iter()
            .all(|import| import.strong_count() == 0)
        {
            left = left.saturating_sub(listed);
            continue;
        }
        if let Some((count, freed)) = release_group(shared, group, true, true) {
            left = left.saturating_sub(freed);
            released += count;
        }
    }
    let resident = shared.resident.load(Ordering::Relaxed);
    shared
        .gap
        .store(resident.saturating_sub(left), Ordering::Relaxed);
    released
}

/// The crate's device tests' hooks (`tests/host_residency.rs`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
impl Device {
    /// Release now every import of a no-copy buffer's dma-bufs that the device releases while
    /// idle, whatever its idle window: those no command buffer holds, whose Metal address was
    /// never handed out and whose bytes only blits read. Each is imported again at its next
    /// blit. Every parked image goes too (`dma_buf/parked.rs`). How many imports it released.
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

    /// Keep at most `most` imports on the device's list held by its releasable imports from now
    /// on (a shared file's once), whatever the device's import tax: the groups past it are
    /// released, the least recently used first, by the residency thread and by
    /// [`Device::release_host_imports_over_most`].
    pub(crate) fn keep_resident_host_imports(&self, most: u64) {
        let shared = &self.shared().residency.shared;
        shared.most.store(most, Ordering::Relaxed);
    }

    /// Park at most `most` images under the imports of host pages from now on
    /// (`dma_buf/parked.rs`): with 0 none parks, and an image in place goes as it went before
    /// images parked.
    pub(crate) fn park_dma_buf_images_at_most(&self, most: u64) {
        let shared = &self.shared().residency.shared;
        shared.parked_most.store(most, Ordering::Relaxed);
    }

    /// Release now the releasable imports past the most the device keeps resident, on this
    /// thread, whatever their windows: how many.
    pub(crate) fn release_host_imports_over_most(&self) -> usize {
        release_over_most(&self.shared().residency.shared)
    }
}
