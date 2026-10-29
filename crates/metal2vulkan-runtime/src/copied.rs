//! Copied host buffers: a no-copy buffer whose import of the caller's pages the caller's gate
//! refused ([`HostMemory::with_import_gate`](crate::HostMemory)), or whose import the gate and
//! the device admitted and that then failed (out of memory, another Vulkan error), is made all the
//! same over a copy of its bytes (`Device::buffer_over_host_memory`), announced
//! (`copied-host-pages`). No import maps the pages, so they take no kernel mapping.
//!
//! - **The copy.** A `VkBuffer` of the buffer's length in host-visible, coherent memory of the
//!   runtime's own ([`MemoryClass::Shared`](crate::memory::MemoryClass)), which the GPU uses; the
//!   buffer's `contents` stays the caller's pointer, as Metal's is. Made over pages the process
//!   reads ([`read_caller`]): the first copy reads them through the kernel ([`read_pages`]), and
//!   where the kernel fails the read, with plain loads where the process maps every page with
//!   read permission ([`mapped_readable`]): on Linux the kernel reads no page it cannot pin, such
//!   as a mapping of a dma-buf (`VM_PFNMAP`). Pages not mapped, or mapped without read permission
//!   (memory Metal makes no buffer over), keep the import's own refusal.
//! - **Coherence.** The runtime keeps, for each copy, the bytes the pages and the copy last agreed
//!   on (its snapshot). At each commit, before any of the command buffer's work is submitted,
//!   every byte the CPU changed in the pages since then is written into the copy
//!   ([`CopiedBuffers::copy_in`]). At each batch's completion, before its events are seen
//!   signaled, the scheduler is told, its handlers run or a waiter returns, every byte the GPU
//!   changed in the copy is written back into the pages, where the CPU did not change that byte
//!   since ([`CopiedBuffers::merge_back`]): a byte both changed keeps the CPU's, which the next
//!   commit copies in. Bytes are compared 8 at a time and merged one at a time where they differ,
//!   so a merge writes no byte the GPU did not change. Each batch ends with a barrier that makes
//!   its writes visible to the host (`tracker::Barrier::to_host`).
//! - **Cost.** Every live copy is compared whole at each commit and at each batch's completion,
//!   whichever buffers the work names: O(bytes) per commit and per batch, a fallback's cost
//!   (`docs/FAST-PATHS.md`, known gaps). Counted: `WorkCounts::copied_copy_in_bytes`,
//!   `WorkCounts::copied_merge_back_bytes`.
//! - **Heaps and rebinds.** A no-copy heap's own buffer is copied as a buffer is, where the heap's
//!   textures could lie, with a heap's hazard identity; its buffers lie in the copy at their
//!   offsets, and its textures are shadows of its bytes (`AliasRoute::MemoryShadow` or Metal's
//!   tiles). A buffer's rebind gives it a copy of its new pages laid out as their import would be
//!   (`Device::copied_pages`), so texel views planned at the bytes' offset lie where they lay; a
//!   rebind of a buffer with an image in place over it is refused instead
//!   (`NotImplemented(RebindTextures)`).
//! - **What is not copied.** A heap's rebind and memory whose GPU writes Metal's private option
//!   bit 24 drops import as before. A texture over a copied buffer is a shadow of its bytes
//!   (`shadowed-texture`): no image is bound to the copy.

use crate::device::lock;
use crate::host_memory::mapped_readable;
use crate::work_counts::Counters;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A device's live copied host buffers, compared at each commit and each batch's completion.
#[derive(Default)]
pub(crate) struct CopiedBuffers {
    entries: Mutex<Vec<Entry>>,
    /// How many are live: a commit or completion looks no further where none is.
    live: AtomicUsize,
    /// The next copy's id.
    next: AtomicU64,
}

/// Why a copy was not made ([`CopiedBuffers::add`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NotCopied {
    /// A page is not mapped, or mapped without read permission: memory Metal makes no buffer over.
    Unreadable,
    /// The host has no memory for the copy's snapshot.
    OutOfHostMemory,
}

/// A copy and the caller's pages it mirrors.
struct Entry {
    id: u64,
    /// The caller's bytes (the buffer's `contents`).
    pages: NonNull<u8>,
    /// The copy's bytes, mapped.
    copy: NonNull<u8>,
    /// The bytes the pages and the copy last agreed on.
    snapshot: Box<[u8]>,
}

// SAFETY: the pointers name the caller's pages, which stay mapped while the buffer's import lives
// (`HostMemory::new`), and the copy's mapped memory, which the import holds; both are reached only
// under the registry's lock, which the import's drop takes before either goes ([`Copied`]).
unsafe impl Send for Entry {}

/// A live copy's place in its device's [`CopiedBuffers`]: dropped (unregistered) before the copy's
/// memory is freed, so no comparison reads it after (`HostImport`'s drop).
pub(crate) struct Copied {
    buffers: Arc<CopiedBuffers>,
    id: u64,
}

impl Drop for Copied {
    fn drop(&mut self) {
        let mut entries = lock(&self.buffers.entries);
        if let Some(index) = entries.iter().position(|entry| entry.id == self.id) {
            entries.swap_remove(index);
            self.buffers.live.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl CopiedBuffers {
    /// Register the copy at `copy` of the `length` bytes at `pages`, its bytes read from the pages
    /// now ([`read_caller`]: through the kernel, or where it fails, with plain loads of pages
    /// mapped readable). Refused, registering nothing, where the process cannot read every page,
    /// or the host has no memory for the bytes the copy and the pages last agreed on.
    ///
    /// # Safety
    ///
    /// `copy` is `length` bytes of mapped memory nothing else uses yet, which stays mapped until
    /// the answer is dropped; `pages` the caller's bytes, which stay mapped as long
    /// (`HostMemory::new`), and which the CPU reads without a fault wherever the process maps
    /// them with read permission.
    pub(crate) unsafe fn add(
        self: &Arc<Self>,
        pages: NonNull<u8>,
        copy: NonNull<u8>,
        length: usize,
    ) -> Result<Copied, NotCopied> {
        let read = |pages: NonNull<u8>, into: &mut [u8]| {
            // SAFETY: the caller's bytes, read with plain loads only where the process maps them
            // readable (the caller's contract).
            unsafe { read_caller(pages, into, read_pages, mapped_readable) }
        };
        // SAFETY: the caller's.
        unsafe { self.add_reading(pages, copy, length, read) }
    }

    /// [`CopiedBuffers::add`], the first bytes of the pages read by `read`, which answers whether
    /// it read every one.
    ///
    /// # Safety
    ///
    /// As [`CopiedBuffers::add`]'s.
    pub(crate) unsafe fn add_reading(
        self: &Arc<Self>,
        pages: NonNull<u8>,
        copy: NonNull<u8>,
        length: usize,
        read: impl FnOnce(NonNull<u8>, &mut [u8]) -> bool,
    ) -> Result<Copied, NotCopied> {
        let mut snapshot = Vec::new();
        if snapshot.try_reserve_exact(length).is_err() {
            return Err(NotCopied::OutOfHostMemory);
        }
        snapshot.resize(length, 0u8);
        let mut snapshot = snapshot.into_boxed_slice();
        if !read(pages, &mut snapshot) {
            return Err(NotCopied::Unreadable);
        }
        // SAFETY: the copy is `length` bytes nothing else uses yet (the caller's).
        unsafe { std::ptr::copy_nonoverlapping(snapshot.as_ptr(), copy.as_ptr(), length) };
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.entries).push(Entry {
            id,
            pages,
            copy,
            snapshot,
        });
        self.live.fetch_add(1, Ordering::Relaxed);
        Ok(Copied {
            buffers: Arc::clone(self),
            id,
        })
    }

    /// At a commit: write into each copy the bytes the CPU changed in its pages since they last
    /// agreed, before any of the command buffer's work is submitted.
    pub(crate) fn copy_in(&self, counts: &Counters) {
        if self.live.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut moved = 0u64;
        for entry in lock(&self.entries).iter_mut() {
            // SAFETY: the entry's pointers are live while it is registered (`Entry`).
            moved += unsafe { sync(entry.pages, entry.copy, &mut entry.snapshot, false) };
        }
        counts.copied_copied_in(moved);
    }

    /// At a batch's completion: write back into each copy's pages the bytes the GPU changed in the
    /// copy since they last agreed, where the CPU did not change them since, before anything sees
    /// the batch complete.
    pub(crate) fn merge_back(&self, counts: &Counters) {
        if self.live.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut moved = 0u64;
        for entry in lock(&self.entries).iter_mut() {
            // SAFETY: the entry's pointers are live while it is registered (`Entry`).
            moved += unsafe { sync(entry.copy, entry.pages, &mut entry.snapshot, true) };
        }
        counts.copied_merged_back(moved);
    }
}

/// Write into `to` each byte of `from` that differs from `snapshot`, and take it into `snapshot`;
/// where `keep_to`, a byte of `to` that differs from `snapshot` already (written since they last
/// agreed) is kept, though `snapshot` takes `from`'s, so the next copy the other way carries it.
/// The bytes written.
///
/// # Safety
///
/// `from` and `to` are `snapshot.len()` bytes of mapped memory each.
pub(crate) unsafe fn sync(
    from: NonNull<u8>,
    to: NonNull<u8>,
    snapshot: &mut [u8],
    keep_to: bool,
) -> u64 {
    let (from, to) = (from.as_ptr(), to.as_ptr());
    let mut written = 0u64;
    let mut at = 0;
    while at < snapshot.len() {
        let span = (snapshot.len() - at).min(8);
        let mut was = [0u8; 8];
        was[..span].copy_from_slice(&snapshot[at..at + span]);
        let mut now = [0u8; 8];
        // SAFETY: `span` bytes at `at` lie in `from` (the caller's).
        unsafe { std::ptr::copy_nonoverlapping(from.add(at), now.as_mut_ptr(), span) };
        if now != was {
            let mut there = [0u8; 8];
            if keep_to {
                // SAFETY: as above, in `to`.
                unsafe { std::ptr::copy_nonoverlapping(to.add(at), there.as_mut_ptr(), span) };
            }
            for byte in 0..span {
                if now[byte] == was[byte] {
                    continue;
                }
                if !keep_to || there[byte] == was[byte] {
                    // SAFETY: as above.
                    unsafe { to.add(at + byte).write_volatile(now[byte]) };
                    written += 1;
                }
                snapshot[at + byte] = now[byte];
            }
        }
        at += span;
    }
    written
}

/// Read the caller's `into.len()` bytes at `pages` into `into`: through the kernel (`kernel`,
/// [`read_pages`]), and where it fails, with plain loads where every page of them is mapped with
/// read permission (`readable`, asked for the bytes' addresses, [`mapped_readable`]). Whether
/// they were read: `false` where the kernel failed and a page is not mapped readable.
///
/// The kernel's read is a test that faults on nothing, but on Linux it pins each page, and no
/// page of a `VM_PFNMAP` or `VM_IO` mapping can be pinned: a mapping of a dma-buf (udmabuf's on
/// Linux 7.2, measured) is mapped readable and fails it all the same. The plain loads are the
/// access a copy's comparisons make at each commit ([`sync`]).
///
/// # Safety
///
/// `pages` is `into.len()` bytes the CPU reads without a fault wherever `readable` answers `true`
/// for them.
pub(crate) unsafe fn read_caller(
    pages: NonNull<u8>,
    into: &mut [u8],
    kernel: impl FnOnce(NonNull<u8>, &mut [u8]) -> bool,
    readable: impl FnOnce(u64, u64) -> bool,
) -> bool {
    if kernel(pages, into) {
        return true;
    }
    let start = pages.as_ptr() as u64;
    if !readable(start, start + into.len() as u64) {
        return false;
    }
    // SAFETY: the bytes are mapped readable (the caller's); `into` is the runtime's own, apart.
    unsafe { std::ptr::copy_nonoverlapping(pages.as_ptr(), into.as_mut_ptr(), into.len()) };
    true
}

/// Read the caller's `into.len()` bytes at `pages` into `into` through the kernel, which answers a
/// page the process cannot read (unmapped, or mapped without read permission) with a failure, not a
/// fault: `false` then. On Linux `process_vm_readv` of the process's own pages; on macOS
/// `mach_vm_read_overwrite` of its own task's; elsewhere nothing is read and the answer is `false`.
#[cfg(target_os = "linux")]
pub(crate) fn read_pages(pages: NonNull<u8>, into: &mut [u8]) -> bool {
    let mut done = 0;
    while done < into.len() {
        let local = libc::iovec {
            iov_base: into[done..].as_mut_ptr().cast(),
            iov_len: into.len() - done,
        };
        let remote = libc::iovec {
            iov_base: pages.as_ptr().wrapping_add(done).cast(),
            iov_len: into.len() - done,
        };
        // SAFETY: the local vector names `into`'s bytes, which the call writes; the remote one is
        // read through the kernel, which answers an unreadable page with an error.
        let read = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
        if read <= 0 {
            return false;
        }
        done += read as usize;
    }
    true
}

/// [`read_pages`] on macOS.
#[cfg(target_os = "macos")]
pub(crate) fn read_pages(pages: NonNull<u8>, into: &mut [u8]) -> bool {
    unsafe extern "C" {
        /// `<mach/mach_init.h>`: the port of this process's task.
        static mach_task_self_: libc::mach_port_t;
        fn mach_vm_read_overwrite(
            target_task: libc::vm_map_t,
            address: libc::mach_vm_address_t,
            size: libc::mach_vm_size_t,
            data: libc::mach_vm_address_t,
            outsize: *mut libc::mach_vm_size_t,
        ) -> libc::kern_return_t;
    }
    let mut read = 0;
    // SAFETY: the call writes `into`'s bytes alone, and answers an unreadable page of the source
    // with an error.
    let answer = unsafe {
        mach_vm_read_overwrite(
            mach_task_self_,
            pages.as_ptr() as u64,
            into.len() as u64,
            into.as_mut_ptr() as u64,
            &mut read,
        )
    };
    answer == libc::KERN_SUCCESS && read == into.len() as u64
}

/// [`read_pages`] where the system is not asked: nothing is copied.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn read_pages(_pages: NonNull<u8>, _into: &mut [u8]) -> bool {
    false
}
