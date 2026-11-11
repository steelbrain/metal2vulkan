//! `contents` of a detached no-copy buffer (`detach.rs`): the caller's pointer of the pages it lay
//! in last, as Metal's answers, across a detach and a rebind, for a no-copy buffer in its pages
//! (a `Shared` one), a `Managed` one (two copies on a device without unified memory,
//! `managed_copy.rs`) and a no-copy heap's buffer (its offset from the heap's pointer). While it
//! is detached the pages are mapped with no access: the runtime reads and writes nothing through
//! the pointer, or the process would fault, while the GPU reads the two-copy buffer's device copy
//! and a blit encoder synchronizes it.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, CpuCacheMode, Device, HazardTrackingMode,
    HeapDescriptor, HeapType, HostBacking, HostMemory, ResourceOptions, ResourceRef, StorageMode,
};
use std::ptr::NonNull;

/// The bytes of each mapping: whole pages of any host.
const BYTES: u64 = 64 << 10;

/// Anonymous private pages of the test's own, unmapped when dropped: declared before the device,
/// so dropped after every buffer over them.
struct Pages(NonNull<u8>);

impl Pages {
    fn new(byte: u8) -> Pages {
        // SAFETY: a fresh private anonymous mapping, unmapped in `drop`.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                BYTES as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED);
        let pages = Pages(NonNull::new(at.cast()).unwrap());
        // SAFETY: the mapping's bytes, which nothing uses yet.
        unsafe { std::ptr::write_bytes(pages.0.as_ptr(), byte, BYTES as usize) };
        pages
    }

    /// Let nothing read or write them (`access`, false), or let the process read and write them.
    fn access(&self, access: bool) {
        let protection = match access {
            true => libc::PROT_READ | libc::PROT_WRITE,
            false => libc::PROT_NONE,
        };
        // SAFETY: the test's mapping, whole.
        let done = unsafe { libc::mprotect(self.0.as_ptr().cast(), BYTES as usize, protection) };
        assert_eq!(done, 0);
    }

    /// Them, for a no-copy buffer, heap or rebind.
    ///
    /// # Safety
    ///
    /// They outlive every object over them (declared before the device).
    unsafe fn memory(&self) -> HostMemory {
        // SAFETY: the caller's contract.
        unsafe { HostMemory::new(self.0) }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made, which no object lies in any more.
        unsafe { libc::munmap(self.0.as_ptr().cast(), BYTES as usize) };
    }
}

/// `buffer`'s first 16 bytes as a blit reads them, `None` where its command buffer failed (a
/// detached buffer in its pages: `DetachedBacking`).
fn gpu_read(device: &Device, queue: &CommandQueue, buffer: &Buffer) -> Option<Vec<u8>> {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, &target, 0, 16)
        .unwrap();
    blit.synchronize_resource(ResourceRef::Buffer(buffer))
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    if command_buffer.status() != CommandBufferStatus::Completed {
        return None;
    }
    // SAFETY: the target's bytes, which no command writes now.
    let bytes = unsafe { std::slice::from_raw_parts(target.contents().unwrap().as_ptr(), 16) };
    Some(bytes.to_vec())
}

/// A no-copy buffer of `storage` over `first`: its `contents` stays `first`'s pointer across a
/// detach, during which the runtime touches no page of it, and is `second`'s after a rebind and
/// across a second detach.
fn buffer_leg(storage: StorageMode) {
    let (first, second) = (Pages::new(1), Pages::new(2));
    let scratch = Scratch::new("detached-contents");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    // SAFETY: the pages outlive every buffer over them.
    let buffer =
        unsafe { device.new_buffer_with_bytes_no_copy(first.memory(), BYTES, options, None) }
            .unwrap();
    let two_copies = buffer.host_backing() == Some(HostBacking::DeviceCopy);
    eprintln!(
        "DETACHED-CONTENTS {} {storage:?}: two copies {two_copies}",
        device.name()
    );
    assert_eq!(gpu_read(&device, &queue, &buffer), Some(vec![1; 16]));
    assert_eq!(buffer.detach_backing(), Ok(true));
    assert_eq!(buffer.host_backing(), Some(HostBacking::Detached));
    assert_eq!(buffer.contents(), Some(first.0), "{storage:?}: detached");
    // Nothing of the runtime's reads or writes the pages: they fault now.
    first.access(false);
    let read = gpu_read(&device, &queue, &buffer);
    match two_copies {
        // The GPU's copy stays, and the synchronize copies nothing out.
        true => assert_eq!(read, Some(vec![1; 16])),
        // A command naming the buffer runs nothing.
        false => assert_eq!(read, None),
    }
    assert_eq!(buffer.contents(), Some(first.0));
    first.access(true);
    // SAFETY: as above.
    unsafe { buffer.rebind_bytes_no_copy(second.memory(), None, None) }.unwrap();
    assert_eq!(buffer.contents(), Some(second.0), "{storage:?}: rebound");
    assert_eq!(gpu_read(&device, &queue, &buffer), Some(vec![2; 16]));
    assert_eq!(buffer.detach_backing(), Ok(true));
    assert_eq!(
        buffer.contents(),
        Some(second.0),
        "{storage:?}: detached again"
    );
}

#[test]
fn a_detached_shared_no_copy_buffer_answers_its_last_pointer() {
    buffer_leg(StorageMode::Shared);
}

#[test]
fn a_detached_managed_no_copy_buffer_answers_its_last_pointer() {
    buffer_leg(StorageMode::Managed);
}

/// A no-copy heap's buffer answers its offset from the heap's last pointer while the heap is
/// detached.
#[test]
fn a_detached_heap_s_buffer_answers_its_offset_from_the_heap_s_last_pointer() {
    let pages = Pages::new(3);
    let scratch = Scratch::new("detached-contents-heap");
    let device = device(&scratch.0);
    let descriptor = HeapDescriptor::new(HeapType::Placement, BYTES, ResourceOptions::shared());
    // SAFETY: the pages outlive the heap.
    let heap =
        unsafe { device.new_heap_with_bytes_no_copy(pages.memory(), &descriptor, None, None) }
            .unwrap();
    let offset = 4096;
    let buffer = heap
        .new_buffer_at_offset(256, ResourceOptions::shared(), offset)
        .unwrap();
    let at = NonNull::new(pages.0.as_ptr().wrapping_add(offset as usize)).unwrap();
    assert_eq!(buffer.contents(), Some(at));
    assert_eq!(heap.detach_backing(), Ok(true));
    assert_eq!(buffer.contents(), Some(at), "the heap detached");
}
