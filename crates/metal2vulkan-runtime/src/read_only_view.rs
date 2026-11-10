//! Read-only views of a caller's pages: on macOS, no-copy memory made with Metal's private option
//! bit 24 ([`ResourceOptions::PRIVATE_BIT`](crate::ResourceOptions::PRIVATE_BIT)) is imported
//! through one, so that the GPU's writes to it are dropped, as Metal drops them.
//!
//! - **What Metal does.** Measured on an M3 Max under macOS 27.0.1: a no-copy buffer, or a heap
//!   over the caller's bytes, made with bit 24 takes the GPU's writes nowhere. The pages'
//!   protection is unchanged, and the resource's options leave the bit out. Which writes it takes
//!   nowhere, and which it faults on, splits by the engine that writes
//!   (`metal2vulkan-conformance`'s `probes/readonly_gpu_writes.m`, eleven shapes against a
//!   writable twin of each, eight runs a shape):
//!   - **What a shader or the blit engine writes is dropped and nothing is said.** A blit fill, a
//!     kernel's stores, a vertex-stage store, a fragment-stage store, a kernel or fragment store
//!     through an address loaded out of an argument buffer, and a pass's visibility result all
//!     complete with no error, and neither the CPU nor a later command buffer sees any of it.
//!   - **What faults a render pass is the store-out, not the binding and not the read.** An
//!     attachment over such memory that the pass does not store completes: a colour attachment at
//!     `storeAction = DontCare` over 484 fresh processes, each with a witness buffer proving the
//!     fragment stage ran, and a depth, stencil or combined attachment at `DontCare` over
//!     2,248 of 2,248 (`Depth32Float`, `Depth16Unorm`, `Stencil8`, `Depth32Float_Stencil8` bound
//!     to both points, and 4x multisampled `Depth32Float`, the draw's own values read back
//!     against the clear). `loadAction = Load` completes the same 2,248 of 2,248 for colour,
//!     depth and stencil, so the clean run is not Metal quietly refusing the memory. The same
//!     attachments at `storeAction = Store` end their command buffer with
//!     `MTLCommandBufferErrorDomain` / `kIOGPUCommandBufferCallbackErrorPageFault`, 200 of 200,
//!     and every later command buffer of that process then errors too; the writable control is
//!     clean through `Store`, 306 of 306.
//!   - **A resolve is the same lever: whichever attachment stores out.** Attaching a resolve
//!     destination over such memory and not resolving into it is clean 2,248 of 2,248 on both
//!     read-only routes, and so is reading a read-only multisample primary and resolving out of
//!     it; storing that primary into such memory faults 188 of 188. **Resolving into such memory
//!     is unknown**: 0 page faults over 520 processes, but about 99 per cent of them were
//!     discarded as recovery victims at two spacings, so the cell has no denominator to read a
//!     verdict off. Nothing is built for it. An earlier record called a resolve into such memory
//!     settled and faulting; that shape kept its multisample primary in Metal's own private
//!     memory under `MultisampleResolve`, so it never stored a primary into read-only memory at
//!     all, and its page fault does not reproduce.
//!   - **Later commands of the same command buffer read a dropped write back** from the GPU's
//!     caches, deterministically, and what decides it is the process's state rather than chance. A
//!     blit fill and both compute stores were visible there in every run, the first of a fresh
//!     process included. The visibility result and the three draw-stage stores were visible in
//!     every run but a process's own first: over 61 fresh processes, each running one shape once,
//!     not one saw the write (0 of 61, bit 24 and the `detachBacking` pair alike), and within one
//!     process running a shape 512 times the split was exactly `answer[1/512] same=none` and
//!     `answer[511/512] same=visible` in seven processes of eight (`[2/512]` and `[510/512]` in
//!     the eighth). So the cold first run of a process never reads the dropped write back and every
//!     later run does. It is still not a contract to reproduce: the caches, not Metal, decide it.
//! - **Bit 24 is not the only read-only state, and the other one agrees with it.** A consumer's
//!   production path reaches read-only backing through Apple's SPI instead:
//!   `detachBacking`, then `replaceBackingWithRanges:readOnly:YES` over its pages (the ranges as
//!   one `MTLResourceAddressRangeArray`), offered where the device answers
//!   `supportsResourceDetachBacking`, which this M3 Max does. Measured against bit 24 over the
//!   same programs, eight runs a shape, each with the writable control of its own creation path:
//!   the two forms agree on **all ten shapes measurable under both**, byte for byte, error for
//!   error and first run for first run, so what is written above holds for either state. The SPI pair also
//!   behaves the same on a buffer made with `newBufferWithBytesNoCopy:` as on one made through the
//!   private `MTLBufferDescriptor`, so the creation path does not enter it. A **detached** backing
//!   is a different state and not this one: a command buffer that names a detached resource ends
//!   in error, where one naming a read-only-replaced resource completes.
//! - **Depth and stencil are measured in the bit-24 heap form alone**, and that is a limit of
//!   Metal's rather than of the measurement: `detachBacking` answers NO for a heap of
//!   `newHeapWithDescriptor:` in either storage mode, because Metal offers no heap over the
//!   caller's bytes for it to detach (`MTLHeapDescriptor` answers no `setContents:`, `setNoCopy:`
//!   or `setDeallocator:`, and the device does not respond to `newHeapWithBytesNoCopy:`). The
//!   runtime's own `new_heap_with_bytes_no_copy` is an extension with no Metal counterpart. The
//!   bit is behavioural there rather than declarative: `heap.resourceOptions` reports it absent on
//!   the read-only heap and on its writable twin alike, while `storeAction = Store` faults on the
//!   one and not the other.
//! - **How.** A no-copy buffer made without the bit over a read-only view of the same pages
//!   (`mach_vm_remap` sharing them, then `mprotect` read-only) behaved the same on every count
//!   measured, and the probe above found it agreeing with bit 24 on **every one of its eleven
//!   shapes**, the five that fault included, with the same stability. So the view reproduces
//!   Metal here rather than approximating it. On macOS the device's import of a host pointer is such a buffer (MoltenVK hands
//!   the range to Metal), so the runtime imports the view in place of the caller's pointer
//!   (`Device::buffer_over_host_memory`), while the CPU reads and writes the caller's own
//!   (`Buffer::contents`). The view shares the pages, so the GPU reads what the CPU
//!   writes at the pointer.
//! - **Cost.** Mapping and protecting a view of touched pages took at most 7 us, and unmapping
//!   it at most 17 us, for 16 KiB to 1 GiB (best of 20, the M3 Max under macOS 27.0.1), beside
//!   the import's 11 to 20 ms per GiB.
//! - **Freed.** The view is unmapped once the memory imported from it is freed (`memory.rs`'s
//!   `Block`), as the import requires it mapped for its whole life.
//! - **Elsewhere** the import is of the caller's pages: Linux drivers pin imported pages for
//!   writing and refuse or fault on read-only ones (`host_memory.rs`), so there the GPU's writes
//!   reach the pages, and the difference is announced
//!   ([`Degradation::PrivateResourceOption`](crate::Degradation::PrivateResourceOption)).

use std::ptr::NonNull;

/// Whether this system imports no-copy memory of bit 24 through a read-only view of its pages
/// (the module docs).
pub(super) const VIEWS_DROP_WRITES: bool = cfg!(target_os = "macos");

/// A read-only view of `length` bytes of the caller's pages, mapped by the runtime and unmapped
/// as this goes: after the memory imported from it is freed (`memory.rs`'s `Block`).
#[derive(Debug)]
pub(crate) struct ReadOnlyView {
    pointer: NonNull<u8>,
    #[cfg(unix)]
    length: u64,
}

// SAFETY: the view is the runtime's alone, and nothing writes through it; unmapping it is this
// value's alone.
unsafe impl Send for ReadOnlyView {}
// SAFETY: as above; a shared view answers its pointer alone.
unsafe impl Sync for ReadOnlyView {}

impl ReadOnlyView {
    /// A read-only view of the `length` bytes of pages at `pointer`, sharing them: `None` where
    /// the system maps none.
    ///
    /// # Safety
    ///
    /// `pointer` and `length` are multiples of the host page, and the bytes are mapped memory of
    /// this process.
    #[cfg(target_os = "macos")]
    pub(super) unsafe fn of(pointer: NonNull<u8>, length: u64) -> Option<ReadOnlyView> {
        unsafe extern "C" {
            /// `<mach/mach_init.h>`: the port of this process's task.
            static mach_task_self_: libc::mach_port_t;
            fn mach_vm_remap(
                target_task: libc::vm_map_t,
                target_address: *mut libc::mach_vm_address_t,
                size: libc::mach_vm_size_t,
                mask: libc::mach_vm_address_t,
                flags: libc::c_int,
                src_task: libc::vm_map_t,
                src_address: libc::mach_vm_address_t,
                copy: libc::boolean_t,
                cur_protection: *mut libc::vm_prot_t,
                max_protection: *mut libc::vm_prot_t,
                inheritance: libc::vm_inherit_t,
            ) -> libc::kern_return_t;
        }
        // `<mach/vm_statistics.h>`: the kernel picks the address.
        const VM_FLAGS_ANYWHERE: libc::c_int = 0x0001;
        // `<mach/vm_inherit.h>`: a child process shares the view, as measured (the module docs).
        const VM_INHERIT_SHARE: libc::vm_inherit_t = 0;
        let bytes = usize::try_from(length).ok()?;
        let mut view: libc::mach_vm_address_t = 0;
        let (mut current, mut maximum) = (0, 0);
        // SAFETY: a new mapping, at an address the kernel picks, of the caller's mapped pages
        // (the caller's contract), shared rather than copied, so it reads what they hold.
        let remapped = unsafe {
            let task = mach_task_self_;
            mach_vm_remap(
                task,
                &mut view,
                length,
                0,
                VM_FLAGS_ANYWHERE,
                task,
                pointer.as_ptr() as libc::mach_vm_address_t,
                0,
                &mut current,
                &mut maximum,
                VM_INHERIT_SHARE,
            )
        };
        if remapped != libc::KERN_SUCCESS {
            return None;
        }
        let view = ReadOnlyView {
            pointer: NonNull::new(view as *mut u8)?,
            length,
        };
        // SAFETY: the mapping is the view's own, just made.
        let protected =
            unsafe { libc::mprotect(view.pointer.as_ptr().cast(), bytes, libc::PROT_READ) };
        (protected == 0).then_some(view)
    }

    /// No view is mapped on this system ([`VIEWS_DROP_WRITES`]).
    ///
    /// # Safety
    ///
    /// None: it maps nothing.
    #[cfg(not(target_os = "macos"))]
    pub(super) unsafe fn of(_pointer: NonNull<u8>, _length: u64) -> Option<ReadOnlyView> {
        None
    }

    /// Where the view starts.
    pub(super) fn pointer(&self) -> NonNull<u8> {
        self.pointer
    }
}

impl Drop for ReadOnlyView {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the mapping is this value's, and nothing reads it any more: the memory
        // imported from it was freed first (`memory.rs`'s `Block`).
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length as usize);
        }
    }
}
