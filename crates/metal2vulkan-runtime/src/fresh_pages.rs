//! Large new buffers in fresh pages: a host-visible buffer of the runtime's own longer than half a
//! pooled block lies in anonymous pages the runtime maps for it and imports by their pointer
//! (`VK_EXT_external_memory_host`), not in memory the device allocates, so nothing writes its
//! zeros.
//!
//! - **Why.** A new Metal buffer reads zero (`Device::new_buffer`). Memory the device allocates
//!   holds undefined bytes, so the runtime sets a new buffer's to zero on the host
//!   (`Buffer::clear_first`), and in memory no buffer used before that write is the CPU's first
//!   touch of each page. Measured on an M3 Max under MoltenVK 1.4.1, for 64 MiB, it took 2.1 to
//!   2.3 ms, on top of the allocation's 1.1 to 1.3 ms (MoltenVK's heap and its residency set's
//!   commit). A memset of pages already touched took 0.29 ms. Metal made the buffer in 6 us and
//!   paid for its pages at their first touch. Anonymous pages are zero when mapped. Apple's
//!   mmap(2) says "MAP_ANON Map zero-filled anonymous memory not associated with any specific
//!   file", and Linux's says of `MAP_ANONYMOUS` "its contents are initialized to zero". An import
//!   of them is that memory (`VkImportMemoryHostPointerInfoEXT`). So the buffer reads zero with
//!   nothing written, and the first touch is paid where the program makes it, as on Metal. The
//!   import costs what the allocation it replaces costs (the same commit, 1.0 to 1.2 ms for
//!   64 MiB).
//! - **Which.** A buffer that takes a block of its own (`memory.rs`'s `place`): longer than half a
//!   pooled block ([`FRESH_PAGES_PAST`]), not of a heap, whose bytes the host maps in memory of
//!   its class (`Shared`, write-combined, `Managed` with unified memory). The device must import
//!   host pointers (`host_memory.rs`'s `Importer`), and the class's best memory type for the buffer
//!   must take the pages: else the buffer lies where it would have, made as ever. Measured
//!   (2026-10-24, RADV, Mesa 26.2.4): on the unified-memory Radeon 780M all three classes take the
//!   pages; on the RX 7900 XTX the write-combined class's best type for the buffer is a device-local
//!   host-visible one, which a host-pointer import cannot bind, so such a buffer is placed as ever
//!   and `Buffer::clear_first` sets its bytes to zero. A shorter one
//!   lies in a pooled block. There the bytes a buffer used before are set to zero by a memset of
//!   pages already touched (about 4.5 us per MiB), which is cheaper than an import of fresh ones
//!   (16 us per MiB). An import also costs 17 to 43 us whatever its size, against a pooled
//!   placement's none.
//! - **Recycled.** A parked buffer of the same making (`recycle.rs`) is still taken first, and its
//!   bytes set to zero on the host (0.29 ms for 64 MiB), since its pages hold its last owner's.
//!   Only a buffer made here reads zero unwritten.
//! - **Textures over it** take the shadow (`shadow.rs`). An image bound to imported memory starts
//!   `UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`), and its first move may write any byte
//!   bound to it. Counted: `WorkCounts::shadows_over_fresh_pages`.
//! - **Freed.** The pages are unmapped once the memory imported from them is freed, since the
//!   import requires them mapped for its whole life. That happens with the buffer's block, or
//!   after it where freed memory waits for the batches in flight (`freed_memory.rs`).
//! - **Counted**: `WorkCounts::fresh_page_buffers_made` and `fresh_page_bytes`.

use super::buffer::usage;
use super::device::lock;
use super::error::{vulkan, Error};
use super::memory::{self, MemoryClass, Placement, BLOCK_BYTES};
use crate::release;
use crate::Device;
use ash::vk;
use std::ptr::NonNull;

/// The length past which a host-visible buffer of the runtime's own lies in fresh pages: half a
/// pooled block, past which it takes a block of its own (the module docs).
pub(super) const FRESH_PAGES_PAST: u64 = BLOCK_BYTES / 2;

/// Whether a buffer of `size` bytes (its padded length) in memory of `class` lies in fresh pages
/// where the device takes them (the module docs): `class` is the buffer's after a `Managed`
/// buffer on a device without unified memory took `Private` memory and a host copy.
pub(super) fn takes_fresh_pages(size: u64, class: MemoryClass) -> bool {
    let host_visible = matches!(
        class,
        MemoryClass::Shared | MemoryClass::WriteCombined | MemoryClass::Managed
    );
    host_visible && size > FRESH_PAGES_PAST && cfg!(unix)
}

/// Anonymous private pages the runtime mapped for a buffer of its own, read-write and zero, and
/// unmapped as this goes: after the memory imported from them is freed (`memory.rs`'s `Block`).
#[derive(Debug)]
pub(crate) struct OwnPages {
    pointer: NonNull<u8>,
    length: u64,
}

// SAFETY: the pages are the runtime's alone; who writes them is the resource API's contract
// (Metal's `contents`), and unmapping them is this value's alone.
unsafe impl Send for OwnPages {}
// SAFETY: as above; a shared `OwnPages` answers its pointer and length alone.
unsafe impl Sync for OwnPages {}

impl OwnPages {
    /// `length` bytes of new anonymous private pages, read-write, at a host page; `None` where
    /// the system maps none.
    #[cfg(unix)]
    fn map(length: u64) -> Option<OwnPages> {
        let bytes = usize::try_from(length).ok()?;
        // SAFETY: a new private anonymous mapping, at an address the kernel picks.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return None;
        }
        let pointer = NonNull::new(pointer.cast::<u8>())?;
        Some(OwnPages { pointer, length })
    }

    /// No pages are mapped on this system.
    #[cfg(not(unix))]
    fn map(_length: u64) -> Option<OwnPages> {
        None
    }

    /// Where the pages start.
    pub(super) fn pointer(&self) -> NonNull<u8> {
        self.pointer
    }

    /// Their bytes.
    pub(super) fn length(&self) -> u64 {
        self.length
    }
}

impl Drop for OwnPages {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the mapping is this value's, and nothing reads it any more: the memory
        // imported from it was freed first (`memory.rs`'s `Block`).
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length as usize);
        }
    }
}

impl Device {
    /// A `VkBuffer` of `size` bytes, made for host allocations with every usage of a buffer of
    /// its own and the device-address usage where the device enables it (its address is its own,
    /// as no block's buffer of addresses lies in the pages), bound in fresh pages imported as
    /// memory of `class`: the buffer and its placement, its bytes zero (the module docs). `None`
    /// where the device takes no such pages: it imports no host pointer, the system maps none,
    /// the class's best memory type for the buffer is not among those the pages allow, or the
    /// driver refuses them. Refused where the buffer cannot be made at all.
    pub(super) fn fresh_page_buffer(
        &self,
        size: u64,
        class: MemoryClass,
    ) -> Result<Option<(vk::Buffer, Placement)>, Error> {
        let shared = self.shared();
        let Ok(importer) = &shared.host_import else {
            return Ok(None);
        };
        let raw_device = shared.vk();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage(shared.device_address))
                .push_next(&mut external),
        );
        // SAFETY: the device is live and `info` and its chain valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        shared.counts.buffer_made(info.usage);
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        // Whole units of the host page and the import alignment, as the import requires of the
        // pointer and the length, and room for the buffer's requirements.
        let unit = (importer.page)
            .max(importer.alignment)
            .max(requirements.alignment);
        let placed = (requirements.size.max(size))
            .checked_next_multiple_of(unit)
            .and_then(OwnPages::map)
            .and_then(|pages| {
                // SAFETY: the pages are host memory of this process at a multiple of the unit,
                // which the import alignment divides (both are powers of two).
                let types = unsafe { importer.memory_types(raw_device, pages.pointer()) }.ok()?;
                // The memory type the buffer would take without the pages: the pages take it, or
                // the buffer lies there as ever.
                let best = lock(&shared.memory).best_type(requirements.memory_type_bits, class)?;
                if types & 1 << best == 0 {
                    return None;
                }
                memory::import_own_pages(&shared.memory, raw_device, pages, best).ok()
            });
        let Some(placement) = placed else {
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, raw) };
            return Ok(None);
        };
        // SAFETY: the placement is the pages' memory, at least the buffer's size and of a type it
        // takes, at offset 0, which every alignment divides; the buffer is unbound.
        if let Err(error) = unsafe {
            crate::creation::bind_buffer(raw_device, raw, placement.memory, placement.offset)
        } {
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, raw) };
            lock(&shared.memory).free(raw_device, &placement);
            return Err(vulkan(error));
        }
        shared.counts.fresh_page_buffer_made(placement.size());
        Ok(Some((raw, placement)))
    }
}
