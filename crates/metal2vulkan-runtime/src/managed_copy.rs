//! `Managed` no-copy buffers on a device without unified memory: two copies, as Metal keeps a
//! managed resource on a Mac with a discrete GPU.
//!
//! - **What Metal does.** On a Mac with a discrete GPU a `Managed` resource has two copies: the
//!   CPU's, which `contents` maps, and the GPU's own in its memory. `didModifyRange:` tells Metal
//!   the CPU wrote a range, which reaches the GPU's copy before the command buffers committed
//!   after the call; a blit encoder's `synchronizeResource:` copies the GPU's copy to the CPU's
//!   where the command buffer runs it, which the CPU sees once it completed. A CPU write the
//!   program tells nothing of is not seen by the GPU, nor a GPU write it synchronizes nothing of
//!   by the CPU. A `Managed` buffer over the caller's pages
//!   (`newBufferWithBytesNoCopy:length:options:deallocator:`) is no exception: the caller's
//!   pages are its CPU copy. A kernel reading one 256 MiB such buffer and writing another ran at
//!   102.5 GB/s on a Radeon Pro 5300M (macOS 26.7.1), where a `Shared` one ran at 6.1 GB/s, the
//!   bus's speed.
//! - **The runtime's.** Such a buffer is a buffer of device-local memory of the runtime's own
//!   (its `VkBuffer`, which every command binds, its Metal address, its textures', as a
//!   `Managed` buffer the runtime makes on such a device, `managed.rs`), and its host copy is
//!   the caller's pages ([`CallerCopy`]): reached through an import of them (by their pointer, as
//!   dma-bufs, or in a guest RAM's chunk: `host_memory.rs`, `host_ram.rs`), which is a buffer of
//!   the runtime's own over them, outside the Metal address space ([`Side::Import`]), or, where
//!   the device imports none of them (the caller's gate refused it, the device refused the
//!   pointer, the import failed), the pages alone ([`Side::Pages`]): never a copy of them as well
//!   (`copied.rs`), which would be a third. `contents` is the caller's pointer, and the
//!   deallocator and keep-alive go as a no-copy buffer's do.
//! - **Copies in.** The buffer's bytes and the rest of their last page ([`CallerCopy::reach`])
//!   are copied in as it is made, `did_modify_range` copies its range in, and a rebind copies the
//!   new pages in whole: each by the device's next submission, before any of its commands, after
//!   the commands before it on its `VkQueue` (`texture/first_move.rs`), which the call does not
//!   wait for, so every command buffer committed after the call reads them, and none in flight on
//!   that queue sees the copy land under it. A command buffer in flight on another queue that
//!   reads the buffer meanwhile races the copy, as on Metal. Not on the staging queue
//!   (`device/staging.rs`), which waits for no work in flight. From an import, the copy's source is
//!   the import's `VkBuffer`, held (imported again where the device released it) until the
//!   submission completed; from pages no import maps, the call copies the bytes on the CPU into
//!   staging buffers of [`STAGED_CHUNK`] bytes at most, written once and freed once the
//!   submission completed. Past the bound of what is staged (`texture/first_move.rs`'s
//!   `STAGED_BYTES`), the call makes a submission for them and waits.
//! - **Copies out.** `synchronize_resource` copies the buffer's bytes to the import in the blit
//!   encoder's command stream, a blit's write of the import (`blit.rs`); to pages no import maps,
//!   into a staging buffer there, which the command buffer's completion copies to the pages before
//!   anything sees it complete (its events, the scheduler, a waiter, its handlers;
//!   `command_buffer/completion.rs`), freeing the staging buffer.
//! - **Read-only pages** (a rebind's, `replaceBackingWithRanges:readOnly:`): the GPU's writes are
//!   dropped, as Metal drops them. `synchronize_resource` copies nothing out, and the pages are
//!   copied in whole again at the completion of each command buffer that used the buffer, so a
//!   later command buffer reads the pages' bytes and never the GPU's writes. A `Managed` no-copy
//!   buffer made with Metal's private option bit 24 is not two copies: it lies in its pages as a
//!   `Shared` one does (`read_only_view.rs`, `copied.rs`).
//! - **Rebind and detach** (`rebind.rs`, `detach.rs`) are the host copy's: a detach lets go of
//!   it, as a no-copy buffer's lets go of its pages, and keeps the device copy, which the GPU
//!   reads and writes on; a rebind makes a new host copy of the new pages, as at the making (so
//!   never a copy of them either), and copies them in whole. An import's detach is its own, as a
//!   no-copy buffer's.
//! - **Memory.** The device copy holds the buffer's bytes in device-local memory for its life
//!   (`WorkCounts::managed_device_copy_bytes`); staging buffers live only until the submission
//!   or command buffer that uses them completed (`WorkCounts::managed_staging_bytes`). Where no
//!   device-local memory has room for the device copy (an allocation refused, or one that would
//!   overflow into system memory), the buffer lies in its pages, as a `Shared` no-copy buffer
//!   does, announced (`device-memory-overflow`), never refused.
//! - **On unified memory** a `Managed` no-copy buffer lies in its pages, as Apple silicon's does.

use super::buffer::{Buffer, HostCopy, Naming};
use super::device::lock;
use super::error::{internal, Error};
use super::exit_guard;
use super::host_memory::{HostMemory, HostParts, Release};
use super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::Device;
use ash::vk;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

/// The most bytes one staging buffer of a copy in from pages no import maps holds: the bytes
/// past it go in the next.
pub(super) const STAGED_CHUNK: u64 = 32 << 20;

/// The host copy of a `Managed` no-copy buffer on a device without unified memory: the caller's
/// pages, as an import or alone (the module docs).
pub(super) struct CallerCopy {
    /// Where the pages are now: a rebind replaces them.
    side: Mutex<Side>,
    /// The bytes from the buffer's byte 0 its device copy holds and its whole copies in cover:
    /// its length and the rest of its last page, which the caller's pages hold.
    reach: u64,
    /// The bytes of device-local memory the device copy holds, counted while it lives.
    device_bytes: u64,
    device: Device,
}

/// Where a [`CallerCopy`]'s pages are.
#[derive(Clone)]
pub(super) enum Side {
    /// An import of them, which a buffer of the runtime's own over them holds, outside the Metal
    /// address space (made as [`Naming::HostCopy`]): its rebind and detach are the host copy's.
    Import(Buffer),
    /// The pages alone, which no import maps, and whether they are detached.
    Pages(Arc<CallerPages>, bool),
}

/// The caller's pages of a `Managed` no-copy buffer that no import maps ([`Side::Pages`]), given
/// back to the caller's deallocator and keep-alive as they go: once the buffer let go of them and
/// every copy into them or out of them completed.
pub(super) struct CallerPages {
    pointer: NonNull<u8>,
    /// The bytes from the pointer the copies cover ([`CallerCopy::reach`] at most).
    reach: u64,
    /// The process cannot write them, or Metal's private option bit 24 drops the GPU's writes:
    /// nothing is copied out to them.
    drops_writes: bool,
    /// The caller's memory they are, which a recovery makes the host copy again of
    /// (`Buffer::carry`).
    made_of: HostParts,
    /// Their release to the caller's deallocator and keep-alive, set once they are the buffer's.
    release: OnceLock<Release>,
}

// SAFETY: the pointer is the caller's, which `HostMemory::new`'s contract keeps mapped until the
// release; the runtime reads and writes through it only as the module docs say.
unsafe impl Send for CallerPages {}
// SAFETY: as above.
unsafe impl Sync for CallerPages {}

impl Drop for CallerPages {
    fn drop(&mut self) {
        let Some(release) = self.release.take() else {
            return;
        };
        // The exit guard runs no deallocator and drops no keep-alive.
        let Some(busy) = exit_guard::enter() else {
            release.keep();
            return;
        };
        drop(busy);
        // The deallocator runs first, then the keep-alive goes: the deallocator may read the
        // bytes (`host_memory.rs`).
        let keep_alive = release.keep_alive();
        release.run();
        drop(keep_alive);
    }
}

/// Where a blit encoder's `synchronize_resource` copies a `Managed` buffer with a host copy to
/// (`Buffer::host_copy`).
pub(super) enum SyncTarget {
    /// The runtime's own host copy's `VkBuffer`, whose byte 0 is the buffer's.
    Own(vk::Buffer),
    /// The import of the caller's pages: a blit's copy into the buffer over them.
    Import(Buffer),
    /// Pages no import maps: a copy into a staging buffer, copied to the pages at the command
    /// buffer's completion ([`CopyBack`]).
    Pages(Arc<CallerPages>),
}

/// A staging buffer of a two-copy buffer's copies (the module docs), counted while it lives
/// (`WorkCounts::managed_staging_bytes`).
pub(super) struct Staging {
    pub(super) buffer: Buffer,
    bytes: u64,
}

impl Staging {
    /// A staging buffer of `bytes` bytes of host-visible memory, its bytes as placed: write-combined
    /// for a copy in, which the CPU writes and the GPU reads once; host-cached for a copy out,
    /// which the GPU writes and the CPU reads.
    pub(super) fn new(device: &Device, bytes: u64, copy_in: bool) -> Result<Staging, Error> {
        let cache = match copy_in {
            true => CpuCacheMode::WriteCombined,
            false => CpuCacheMode::DefaultCache,
        };
        let options = ResourceOptions::new(StorageMode::Shared, cache, HazardTrackingMode::Tracked);
        let (buffer, _) = device.unzeroed_buffer_with(bytes, options)?;
        device.shared().counts.managed_staging(bytes, true);
        Ok(Staging { buffer, bytes })
    }

    /// Its bytes on the CPU.
    fn contents(&self) -> Result<NonNull<u8>, Error> {
        self.buffer
            .contents()
            .ok_or_else(|| internal("a staging buffer without contents"))
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        let counts = &self.buffer.device().shared().counts;
        counts.managed_staging(self.bytes, false);
    }
}

/// A copy out of a two-copy buffer over pages no import maps, from a staging buffer the command
/// buffer's commands wrote into the pages, made at its completion (`completion.rs`).
pub(super) struct CopyBack {
    staging: Staging,
    pages: Arc<CallerPages>,
    /// Where the bytes go from the pages' pointer.
    at: u64,
}

impl CopyBack {
    /// The copy of `staging`, which a command buffer's commands write, into `pages` from `at`
    /// bytes past their pointer.
    pub(super) fn new(staging: Staging, pages: Arc<CallerPages>, at: u64) -> CopyBack {
        CopyBack { staging, pages, at }
    }

    /// Copy the staging buffer's bytes into the pages: the command buffer completed, its work
    /// visible to the host.
    pub(super) fn run(self) {
        let Ok(from) = self.staging.contents() else {
            return;
        };
        let bytes = self.staging.bytes;
        // SAFETY: the staging buffer holds `bytes` bytes the completed command buffer wrote; the
        // pages hold them at `at`, inside their reach, mapped while `pages` lives
        // (`HostMemory::new`), and nothing of the runtime's writes them meanwhile.
        unsafe {
            std::ptr::copy_nonoverlapping(
                from.as_ptr(),
                self.pages.pointer.as_ptr().add(self.at as usize),
                bytes as usize,
            );
        }
        let counts = &self.staging.buffer.device().shared().counts;
        counts.managed_copied(bytes, false);
    }
}

/// What a command buffer does at its completion for the two-copy buffers its commands used,
/// before anything sees it complete (`completion.rs`).
pub(super) enum AtCompletion {
    /// A copy out into pages no import maps.
    CopyBack(CopyBack),
    /// A copy in whole of the pages of a buffer whose host copy drops the GPU's writes: what the
    /// command buffer wrote reaches no later one.
    Undo(Buffer),
}

impl AtCompletion {
    /// Do it, where the command buffer completed; a copy back of one that failed is dropped (its
    /// commands may not have run), and a copy in is made whatever its result.
    pub(super) fn run(self, completed: bool) {
        match self {
            AtCompletion::CopyBack(back) if completed => back.run(),
            AtCompletion::CopyBack(_) => {}
            AtCompletion::Undo(buffer) => {
                if let Some(caller) = buffer.caller_copy() {
                    buffer.device().shared().counts.managed_write_undone();
                    // Unbounded: the completion makes no submission (the next one takes it).
                    let _ = caller.copy_in(&buffer, 0..caller.reach, false);
                }
            }
        }
    }
}

impl Side {
    /// Give the pages back to `release` once they go: an import's, as a no-copy buffer's pages
    /// go (`buffer.rs`'s `HostPages`); pages no import maps once nothing holds them.
    fn release_to(&self, release: Release) {
        match self {
            Side::Import(import) => {
                if let Some(pages) = import.host_pages() {
                    pages.release_to(release);
                }
            }
            // A refused set leaves the deallocator unrun: never reached, as the release is set
            // once, as the pages become the buffer's.
            Side::Pages(pages, _) => drop(pages.release.set(release)),
        }
    }
}

impl CallerCopy {
    /// The host copy on `to`, the device's next generation, for a recovery that carries the
    /// buffer, `length` bytes of `options`, there (`Buffer::carry`): made again of the caller's
    /// memory its pages were made of, as at the making (an import, or the pages alone), its
    /// deallocator and keep-alive waiting for the buffer's release there and for each import of
    /// the pages to be freed, as a no-copy buffer's (`Release::carry_on`). `None` (the buffer left
    /// the lost generation's) where its pages are detached, or the making fails there.
    pub(super) fn carry(&self, to: &Device, length: u64, options: ResourceOptions) -> Option<Self> {
        const CALL: &str = "new_buffer_with_bytes_no_copy";
        let (parts, held) = match self.side() {
            Side::Import(import) => {
                let pages = import.host_pages()?;
                (pages.made_of()?.clone(), Some(pages))
            }
            Side::Pages(pages, false) => (pages.made_of.clone(), None),
            Side::Pages(_, true) => return None,
        };
        let pointer = parts.pointer;
        // SAFETY: the caller's pages stay mapped while the buffer lives, as the contract of the
        // memory it was made or rebound over says (`HostMemory::new`).
        let side = unsafe { to.caller_side((CALL, length, options), parts, None) }.ok()?;
        // What the caller gave back the pages to waits for the buffer's release there and for
        // each import of them to be freed.
        let carried = match (self.side(), held) {
            (_, Some(pages)) => pages.carry_release(),
            (Side::Pages(pages, _), None) => pages.release.get().and_then(Release::carry_on),
            (Side::Import(_), None) => None,
        };
        if let Some(held) = carried {
            side.release_to(Release::carried(pointer, length, held));
        }
        to.shared()
            .counts
            .managed_device_copy(self.device_bytes, true);
        Some(CallerCopy {
            side: Mutex::new(side),
            reach: self.reach,
            device_bytes: self.device_bytes,
            device: to.clone(),
        })
    }

    /// The bytes of device-local memory its device copy holds.
    pub(super) fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// Where its pages are now.
    fn side(&self) -> Side {
        lock(&self.side).clone()
    }

    /// The bytes from the buffer's byte 0 its whole copies in cover.
    pub(super) fn reach(&self) -> u64 {
        self.reach
    }

    /// The caller's pointer, `contents`; `None` while the pages are detached.
    pub(super) fn contents(&self) -> Option<NonNull<u8>> {
        match self.side() {
            Side::Import(import) => import.contents(),
            Side::Pages(pages, false) => Some(pages.pointer),
            Side::Pages(_, true) => None,
        }
    }

    /// Whether the GPU's writes to the buffer are dropped: its pages are read-only, an import of
    /// a read-only view of them, or memory of Metal's private option bit 24.
    pub(super) fn drops_writes(&self) -> bool {
        match self.side() {
            Side::Import(import) => import.drops_writes(),
            Side::Pages(pages, _) => pages.drops_writes,
        }
    }

    /// Where `synchronize_resource` copies the buffer to: nothing while the pages are detached,
    /// nor where they drop the GPU's writes.
    pub(super) fn sync_target(&self) -> Option<SyncTarget> {
        match self.side() {
            Side::Import(import) if !import.detached() && !import.drops_writes() => {
                Some(SyncTarget::Import(import))
            }
            Side::Pages(pages, false) if !pages.drops_writes => Some(SyncTarget::Pages(pages)),
            Side::Import(_) | Side::Pages(..) => None,
        }
    }

    /// Copy `range` of the pages into `buffer`'s device copy (the module docs, **Copies in**),
    /// clamped to the pages' reach. Where `bounded`, past the bound of what is staged, a
    /// submission waited for makes the copies now; else (a command buffer's completion, which
    /// makes no submission) they wait for the device's next. Nothing while the pages are
    /// detached.
    pub(super) fn copy_in(
        &self,
        buffer: &Buffer,
        range: Range<u64>,
        bounded: bool,
    ) -> Result<(), Error> {
        let range = range.start..range.end.min(self.reach);
        if range.is_empty() {
            return Ok(());
        }
        match self.side() {
            Side::Import(import) => {
                if import.detached() {
                    return Ok(());
                }
                let range = range.start..range.end.min(import.copy_reach());
                if range.is_empty() {
                    return Ok(());
                }
                // A copy of the pages (an import a rebind could not make, `copied.rs`) is not
                // kept up with the CPU's writes but at the commits that use it: the pages
                // themselves are staged.
                if import.copied() {
                    let pointer = import
                        .contents()
                        .ok_or_else(|| internal("a copied host buffer without contents"))?;
                    return stage_in(buffer, pointer, range, bounded);
                }
                let ((source, at), hold) = import.blit_at(range.start)?;
                let held: Box<dyn std::any::Any + Send + Sync> = Box::new((import, hold));
                buffer.copy_in_from((source, at), range, held, bounded)
            }
            Side::Pages(pages, false) => {
                let range = range.start..range.end.min(pages.reach);
                stage_in(buffer, pages.pointer, range, bounded)
            }
            Side::Pages(_, true) => Ok(()),
        }
    }

    /// `detach_backing` of the buffer: the host copy's (the module docs). An import's detach
    /// answers as a no-copy buffer's; pages no import maps detach at once, their deallocator held
    /// until the buffer's next rebind or its release.
    pub(super) fn detach(&self) -> Result<bool, Error> {
        let mut side = lock(&self.side);
        match &*side {
            Side::Import(import) => {
                let import = import.clone();
                drop(side);
                import.detach_backing()
            }
            Side::Pages(pages, _) => {
                *side = Side::Pages(Arc::clone(pages), true);
                Ok(true)
            }
        }
    }

    /// `rebind_bytes_no_copy` of `buffer`: a new host copy of `memory`, made as the buffer's
    /// first was (an import, never a copy of the pages, or the pages alone), in place of the old,
    /// which goes once no copy or command holds it, its deallocator then; then the new pages
    /// copied in whole.
    ///
    /// # Safety
    ///
    /// As [`Buffer::rebind_bytes_no_copy`]'s.
    pub(super) unsafe fn rebind(
        &self,
        buffer: &Buffer,
        memory: HostMemory,
        deallocator: Option<super::host_memory::Deallocator>,
        keep_alive: Option<super::host_memory::KeepAlive>,
    ) -> Result<(), Error> {
        const CALL: &str = "rebind_bytes_no_copy";
        let device = buffer.device();
        let release = super::host_memory::HostRelease::new(deallocator, keep_alive);
        let parts = memory.into_parts();
        let release = release.of(parts.pointer, buffer.length());
        let made = (CALL, buffer.length(), buffer.resource_options());
        // SAFETY: the caller's contract.
        let side = unsafe { device.caller_side(made, parts, release)? };
        // The old host copy goes once nothing holds it, after the lock: its deallocator in the
        // call where nothing does, as Metal runs it at the replace.
        let old = std::mem::replace(&mut *lock(&self.side), side);
        drop(old);
        self.copy_in(buffer, 0..self.reach, true)
    }
}

impl Drop for CallerCopy {
    fn drop(&mut self) {
        (self.device.shared().counts).managed_device_copy(self.device_bytes, false);
    }
}

/// Copy `range` of the bytes at `pointer` (the buffer's byte 0) into `buffer`'s device copy
/// through staging buffers of [`STAGED_CHUNK`] bytes at most, read now.
fn stage_in(
    buffer: &Buffer,
    pointer: NonNull<u8>,
    range: Range<u64>,
    bounded: bool,
) -> Result<(), Error> {
    let device = buffer.device();
    let mut start = range.start;
    while start < range.end {
        let bytes = (range.end - start).min(STAGED_CHUNK);
        let staging = Staging::new(device, bytes, true)?;
        let into = staging.contents()?;
        // SAFETY: the caller's pages hold the bytes from `pointer` to its reach, mapped while the
        // buffer holds them (`HostMemory::new`); the staging buffer holds `bytes` bytes nothing
        // else uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                pointer.as_ptr().add(start as usize),
                into.as_ptr(),
                bytes as usize,
            );
        }
        let source = staging.buffer.at(0);
        let held: Box<dyn std::any::Any + Send + Sync> = Box::new(staging);
        buffer.copy_in_from(source, start..start + bytes, held, bounded)?;
        start += bytes;
    }
    Ok(())
}

/// The refusal a buffer made as [`Naming::HostCopy`] answers where it would be a copy of the
/// caller's pages (`copied.rs`): the host copy is their pages alone instead ([`Side::Pages`]),
/// whose GPU writes are dropped where `drops_writes` (read-only pages, or bit-24 memory).
pub(super) fn uncopied(drops_writes: bool) -> Error {
    internal(match drops_writes {
        true => "a two-copy buffer's read-only host pages are never a copy",
        false => "a two-copy buffer's host pages are never a copy",
    })
}

impl Device {
    /// Whether a no-copy buffer of `storage`, not a heap's own, whose GPU writes are not dropped,
    /// is two copies here (the module docs): `Managed` storage on a device without unified
    /// memory.
    pub(super) fn two_copies(
        &self,
        storage: StorageMode,
        of_heap: bool,
        drops_writes: bool,
    ) -> bool {
        storage == StorageMode::Managed
            && !of_heap
            && !drops_writes
            && !self.properties().has_unified_memory
    }

    /// The host copy of a two-copy buffer of `made`'s call, length and options over the caller's
    /// memory's `parts`, given back to `release`: an import of them, made as a no-copy buffer's
    /// (but never a copy of them), outside the Metal address space, or, where the device imports
    /// none, the pages alone. Refused as a no-copy buffer is, where the pages are.
    ///
    /// # Safety
    ///
    /// The memory of `parts` meets [`HostMemory::new`]'s contract for the length's bytes.
    pub(super) unsafe fn caller_side(
        &self,
        (call, length, options): (&'static str, u64, ResourceOptions),
        parts: HostParts,
        release: Option<Release>,
    ) -> Result<Side, Error> {
        let pointer = parts.pointer;
        let vouched = parts.writable;
        let made_of = parts.clone();
        let outside = super::address::AddressPlacement::Outside.into();
        let made = (call, length, options, outside);
        // SAFETY: the caller's contract.
        let import =
            unsafe { self.one_copy_buffer(made, parts, None, false, false, Naming::HostCopy) };
        let side = match import {
            Ok(import) => Side::Import(import),
            Err(error) if error == uncopied(false) || error == uncopied(true) => {
                let page = (self.shared().host_import.as_ref())
                    .map_or(4096, |importer| importer.page.max(1));
                let at = pointer.as_ptr() as u64;
                let reach = (at + length).next_multiple_of(page) - at;
                // A refusal before the pages' write permission was asked (the caller's gate, the
                // device's refusal of the pointer, a failed import) asks it here: the GPU's
                // writes to pages the process cannot write are dropped, never copied out to them.
                let drops_writes = error == uncopied(true)
                    || (!vouched && self.unwritable(at..at + reach, page)?);
                let pages = CallerPages {
                    pointer,
                    reach,
                    drops_writes,
                    made_of,
                    release: OnceLock::new(),
                };
                Side::Pages(Arc::new(pages), false)
            }
            Err(error) => return Err(error),
        };
        if let Some(release) = release {
            side.release_to(release);
        }
        Ok(side)
    }

    /// Whether the process cannot write the pages of `range` (whole pages of `page` bytes), as
    /// the import of no-copy memory asks it (`host_memory.rs`'s `refuse_unwritable`); refused
    /// where the system cannot say, as that import is.
    fn unwritable(&self, range: Range<u64>, page: u64) -> Result<bool, Error> {
        let start = range.start / page * page;
        let window = super::host_memory::ImportWindow {
            start,
            offset: range.start - start,
            size: range.end - start,
        };
        match super::host_memory::refuse_unwritable(&window, &self.shared().counts) {
            Ok(()) => Ok(false),
            Err(error) if super::host_memory::refused(&error) => Ok(true),
            Err(error) => Err(error),
        }
    }

    /// A two-copy buffer (the module docs) of `made`'s call, length, options and Metal address
    /// placement over the caller's memory's `parts`, given back to `release`, named `naming`: its
    /// host copy ([`Device::caller_side`]) and its device copy, the pages copied in. Where no
    /// device-local memory has room for the device copy, the buffer lies in its pages, as a
    /// `Shared` no-copy buffer does, announced (`device-memory-overflow`).
    ///
    /// # Safety
    ///
    /// The memory of `parts` meets [`HostMemory::new`]'s contract for the length's bytes.
    pub(super) unsafe fn two_copy_buffer(
        &self,
        (call, length, options, placement): (
            &'static str,
            u64,
            ResourceOptions,
            super::address::Place,
        ),
        parts: super::host_memory::HostParts,
        release: Option<Release>,
        naming: Naming,
    ) -> Result<Buffer, Error> {
        let modes = super::buffer::Modes::of(options)?;
        let shared = self.shared();
        let importer = shared
            .host_import
            .as_ref()
            .map_err(|unsupported| Error::Unsupported(unsupported.clone()))?;
        let pointer = parts.pointer.as_ptr() as u64;
        let window = super::host_memory::import_window(pointer, length, importer.page, 1)
            .map_err(|_| super::error::nil(call, super::error::Problem::BufferLength { length }))?;
        let reach = window.size - window.offset;
        // The device copy, first: where it has no room, nothing of the host copy was made.
        let base = super::host_memory::buffer_base(window.offset, 256);
        let size = (base + reach).next_multiple_of(16);
        let device_copy = match self.managed_device_copy(size) {
            Ok(made) => made,
            Err(cause) => {
                // SAFETY: the caller's contract.
                return unsafe {
                    self.in_place(
                        call,
                        (length, options, placement),
                        parts,
                        release,
                        naming,
                        cause,
                    )
                };
            }
        };
        let (raw, memory) = device_copy;
        let discard = |memory: &super::memory::Placement| {
            // SAFETY: the buffer is unused.
            unsafe { crate::release::object(shared.vk(), raw) };
            lock(&shared.memory).free(shared.vk(), memory);
        };
        // SAFETY: the caller's contract.
        let side = match unsafe { self.caller_side((call, length, options), parts, release) } {
            Ok(side) => side,
            Err(error) => {
                discard(&memory);
                return Err(error);
            }
        };
        let (address, reserved) = match self.place_window(placement, &window, length) {
            Ok(held) => held,
            Err(error) => {
                discard(&memory);
                return Err(error);
            }
        };
        let caller = Arc::new(CallerCopy {
            side: Mutex::new(side),
            reach,
            device_bytes: size,
            device: self.clone(),
        });
        shared.counts.managed_device_copy(size, true);
        let buffer = self.buffer_of(super::buffer::BufferParts {
            raw,
            memory,
            offset: base,
            base,
            length,
            reach,
            modes,
            address,
            reserved,
            owner: super::buffer::Owner::Runtime,
            release: None,
            of_heap: false,
            drops_writes: false,
            mirror: Some(HostCopy::Caller(Arc::clone(&caller))),
            #[cfg(unix)]
            again: None,
            naming,
        });
        // The pages' bytes reach every command buffer committed from now on.
        caller.copy_in(&buffer, 0..reach, true)?;
        Ok(buffer)
    }

    /// The no-copy buffer of a two-copy buffer whose device copy found no room for `cause`, in
    /// its pages as a `Shared` one, announced (`device-memory-overflow`) and counted
    /// (`WorkCounts::managed_device_copies_in_place`).
    ///
    /// # Safety
    ///
    /// As [`Device::two_copy_buffer`]'s.
    unsafe fn in_place(
        &self,
        call: &'static str,
        (length, options, placement): (u64, ResourceOptions, super::address::Place),
        parts: super::host_memory::HostParts,
        release: Option<Release>,
        naming: Naming,
        cause: Error,
    ) -> Result<Buffer, Error> {
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        self.shared().counts.managed_device_copy_in_place();
        self.degrade_overflow(|| {
            format!(
                "a `Managed` no-copy buffer of {length} bytes, whose GPU copy found no room \
                 ({cause}), lies in the caller's pages as a `Shared` one"
            )
        });
        let made = (call, length, options, placement);
        // SAFETY: the caller's contract.
        unsafe { self.one_copy_buffer(made, parts, release, false, false, naming) }
    }
}
