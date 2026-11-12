//! One live import of a page (`host_aliases.rs`, "One page, one import"):
//! `VK_EXT_external_memory_host` leaves undefined a host allocation imported twice into one
//! device, so the device never imports again a page a live import of a no-copy buffer or heap
//! holds. A buffer over exactly a heap's bytes lies in the heap's import, and a heap over exactly
//! a live buffer's bytes in the buffer's; a buffer whose pages a live import holds all of is bound
//! inside that import's memory at their offset (`WorkCounts::object_churn
//! .buffers_bound_in_import`); and a buffer or heap whose pages one holds some of (across its
//! edge, or around it) is a copy of them (`WorkCounts::copied_over_live_imports`). Each test
//! counts the device's imports by pointer (`host_pointer_imports_made`), reads what the GPU wrote
//! through each buffer in the host's bytes, and what the host wrote in what the GPU copied out.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. A device that imports no host pointer
//! says so and passes.

use super::blit_holds::{device, Scratch};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, Error, Heap, HeapDescriptor, HeapType,
    HostMemory, ResourceOptions,
};
use std::alloc::Layout;
use std::ptr::NonNull;

/// The bytes of each test's host memory, and of a heap over its first half.
const BYTES: usize = 4 << 20;
const HEAP: u64 = 2 << 20;

/// `BYTES` bytes of zeroed host memory, aligned to 2 MiB, freed when dropped.
struct Pages(NonNull<u8>);

impl Pages {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, 2 << 20).unwrap()
    }

    fn new() -> Pages {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Pages::layout()) };
        Pages(NonNull::new(pointer).unwrap())
    }

    /// The memory `at` bytes in.
    ///
    /// # Safety
    ///
    /// It outlives every buffer and heap made over it.
    unsafe fn at(&self, at: usize) -> HostMemory {
        assert!(at < BYTES);
        // SAFETY: inside the allocation; the caller's contract.
        unsafe { HostMemory::new(NonNull::new_unchecked(self.0.as_ptr().add(at))) }
    }

    /// The bytes `at..at + length`.
    fn bytes(&self, at: usize, length: usize) -> &[u8] {
        assert!(at + length <= BYTES);
        // SAFETY: inside the allocation, which no command writes while the test reads it.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().add(at), length) }
    }

    /// Set the bytes from `at` on to `value`.
    fn set(&self, at: usize, length: usize, value: u8) {
        assert!(at + length <= BYTES);
        // SAFETY: inside the allocation, which no command uses now.
        unsafe { std::ptr::write_bytes(self.0.as_ptr().add(at), value, length) };
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Pages::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Pages::layout()) };
    }
}

/// The device's first imports by pointer (not those of an import released while idle made
/// again, `host_residency.rs`), its buffers bound inside a live import, its copies made over a
/// live import, and the imports made where a live one held some of the pages all the same.
fn imports(device: &Device) -> [u64; 4] {
    let counts = device.work_counts();
    [
        counts.host_pointer_imports_made - counts.host_imports_reimported,
        counts.object_churn.buffers_bound_in_import,
        counts.copied_over_live_imports,
        counts.object_churn.imports_inside_live,
    ]
}

/// `counts` moved on by `by`.
fn plus(counts: [u64; 4], by: [u64; 4]) -> [u64; 4] {
    std::array::from_fn(|at| counts[at] + by[at])
}

/// A no-copy buffer of `length` bytes `at` bytes into `pages`, of `Shared` storage.
fn buffer(device: &Device, pages: &Pages, at: usize, length: u64) -> Buffer {
    // SAFETY: the pages outlive the buffer (dropped after the device), and only the test's
    // commands use them while it lives.
    unsafe {
        device.new_buffer_with_bytes_no_copy(pages.at(at), length, ResourceOptions::shared(), None)
    }
    .unwrap()
}

/// A no-copy automatic heap of `size` bytes `at` bytes into `pages`; `None` where the device
/// imports no host pointer.
fn heap(device: &Device, pages: &Pages, at: usize, size: u64) -> Option<Heap> {
    let descriptor = HeapDescriptor::new(HeapType::Automatic, size, ResourceOptions::shared());
    // SAFETY: as `buffer`'s.
    match unsafe { device.new_heap_with_bytes_no_copy(pages.at(at), &descriptor, None, None) } {
        Ok(heap) => Some(heap),
        Err(Error::Unsupported(gap)) => {
            eprintln!("ONE-IMPORT: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_heap_with_bytes_no_copy: {error:?}"),
    }
}

/// Commit what `encode` records into one blit encoder, and wait for it.
fn blits(queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed().unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The first `length` bytes of `buffer` as the GPU reads them: copied into a `Shared` buffer of
/// the runtime's own.
fn read(device: &Device, queue: &CommandQueue, buffer: &Buffer, length: u64) -> Vec<u8> {
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    blits(queue, |blit| {
        blit.copy_from_buffer_to_buffer(buffer, 0, &out, 0, length)
            .unwrap()
    });
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command writes any more.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// A heap over host pages, then buffers over them: one over exactly the heap's bytes lies in its
/// import; two over part of them (one from a byte past a page's start) are bound inside it; one
/// across its end is a copy. No page is imported a second time, and the GPU's writes through
/// each reach the host's bytes, the host's writes the GPU's reads.
#[test]
fn buffers_over_a_heaps_pages_lie_inside_its_import() {
    let pages = Pages::new();
    let scratch = Scratch::new("one-import-heap-first");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let start = imports(&device);
    let Some(heap) = heap(&device, &pages, 0, HEAP) else {
        return;
    };
    let after_heap = imports(&device);
    if after_heap[0] == start[0] {
        eprintln!("ONE-IMPORT: the heap's pages are a copy here, imported by no pointer");
        return;
    }
    assert_eq!(
        after_heap,
        plus(start, [1, 0, 0, 0]),
        "the heap's own import"
    );
    let exact = buffer(&device, &pages, 0, HEAP);
    assert_eq!(
        imports(&device),
        after_heap,
        "a buffer over the heap's exact bytes"
    );
    let inside = buffer(&device, &pages, 64 << 10, 256 << 10);
    let unaligned = buffer(&device, &pages, (64 << 10) + 0x10, 0x2000);
    assert_eq!(
        imports(&device),
        plus(after_heap, [0, 2, 0, 0]),
        "two buffers over part of the heap's bytes"
    );
    let across = buffer(&device, &pages, HEAP as usize - 4096, 8192);
    let last = plus(after_heap, [0, 2, 1, 0]);
    assert_eq!(imports(&device), last, "a buffer across the heap's end");

    // The GPU's writes through each reach the host's bytes.
    blits(&queue, |blit| {
        blit.fill_buffer(&exact, 0..4096, 0x11).unwrap();
        blit.fill_buffer(&inside, 4096..8192, 0x22).unwrap();
        blit.fill_buffer(&across, 4096..8192, 0x33).unwrap();
    });
    assert!(pages.bytes(0, 4096).iter().all(|&byte| byte == 0x11));
    let inside_at = (64 << 10) + 4096;
    assert!(pages
        .bytes(inside_at, 4096)
        .iter()
        .all(|&byte| byte == 0x22));
    assert!(pages
        .bytes(HEAP as usize, 4096)
        .iter()
        .all(|&byte| byte == 0x33));
    // A placed buffer of the heap's, through the heap's own import.
    let placed = heap.new_buffer(4096, ResourceOptions::shared()).unwrap();
    blits(&queue, |blit| {
        blit.fill_buffer(&placed, 0..4096, 0x44).unwrap()
    });
    assert!(pages.bytes(0, 4096).iter().all(|&byte| byte == 0x44));

    // The host's writes reach the GPU's reads through each.
    pages.set((64 << 10) + 0x10, 0x2000, 0x55);
    assert!(read(&device, &queue, &unaligned, 0x2000)
        .iter()
        .all(|&byte| byte == 0x55));
    pages.set(HEAP as usize - 4096, 8192, 0x66);
    assert!(read(&device, &queue, &across, 8192)
        .iter()
        .all(|&byte| byte == 0x66));
    pages.set(64 << 10, 256 << 10, 0x77);
    assert!(read(&device, &queue, &inside, 256 << 10)
        .iter()
        .all(|&byte| byte == 0x77));
    assert_eq!(imports(&device)[0], last[0], "no import since");
    drop((placed, exact, inside, unaligned, across, heap));
}

/// Buffers over host pages, then heaps: a heap over exactly a live buffer's bytes lies in its
/// import, its placed buffers' writes reaching the host's bytes through it; a heap around a
/// live buffer's pages, which that import holds some of, is a copy; a buffer over part of a live
/// buffer's bytes is bound inside its import. No page is imported a second time.
#[test]
fn heaps_and_buffers_over_a_buffers_pages_lie_inside_its_import() {
    let pages = Pages::new();
    let scratch = Scratch::new("one-import-buffer-first");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let start = imports(&device);
    let first = buffer(&device, &pages, 0, HEAP);
    let after_first = imports(&device);
    if after_first[0] == start[0] {
        eprintln!("ONE-IMPORT: the buffer's pages are a copy here, imported by no pointer");
        return;
    }
    assert_eq!(
        after_first,
        plus(start, [1, 0, 0, 0]),
        "the buffer's own import"
    );
    let Some(over) = heap(&device, &pages, 0, HEAP) else {
        return;
    };
    assert_eq!(
        imports(&device),
        after_first,
        "a heap over the buffer's exact bytes"
    );
    let placed = over.new_buffer(8192, ResourceOptions::shared()).unwrap();
    blits(&queue, |blit| {
        blit.fill_buffer(&placed, 0..8192, 0x5c).unwrap()
    });
    assert!(pages.bytes(0, 8192).iter().all(|&byte| byte == 0x5c));
    pages.set(0, 4096, 0x5d);
    assert!(read(&device, &queue, &placed, 4096)
        .iter()
        .all(|&byte| byte == 0x5d));

    // A buffer over part of the first one's bytes: bound inside its import.
    let part = buffer(&device, &pages, 1 << 20, 64 << 10);
    let bound = plus(after_first, [0, 1, 0, 0]);
    assert_eq!(
        imports(&device),
        bound,
        "a buffer over part of a buffer's bytes"
    );
    blits(&queue, |blit| {
        blit.fill_buffer(&part, 0..4096, 0x6e).unwrap()
    });
    assert!(pages.bytes(1 << 20, 4096).iter().all(|&byte| byte == 0x6e));

    // A small buffer in the second half, then a heap around it: the heap is a copy.
    let small = buffer(&device, &pages, (3 << 20) + 4096, 4096);
    let after_small = plus(bound, [1, 0, 0, 0]);
    assert_eq!(
        imports(&device),
        after_small,
        "a buffer over pages no import holds"
    );
    let copied_heaps = device.work_counts().copied_heaps_made;
    let around = heap(&device, &pages, 2 << 20, HEAP).unwrap();
    assert_eq!(
        imports(&device),
        plus(after_small, [0, 0, 1, 0]),
        "a heap around a live buffer's pages"
    );
    assert_eq!(device.work_counts().copied_heaps_made, copied_heaps + 1);
    let in_copy = around.new_buffer(4096, ResourceOptions::shared()).unwrap();
    blits(&queue, |blit| {
        blit.fill_buffer(&in_copy, 0..4096, 0x7f).unwrap()
    });
    assert!(pages.bytes(2 << 20, 4096).iter().all(|&byte| byte == 0x7f));
    drop((in_copy, around, small, part, placed, over, first));
}
