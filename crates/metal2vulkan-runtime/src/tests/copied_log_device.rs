//! Host pages whose memory came with a log of the writes to them (`dirty_pages.rs`), on a device:
//! the runtime's own CPU writes into them are reported to the log, as the consumer's writer 0, so
//! every copy of the pages takes them (`DirtyPages::wrote`).
//!
//! - **A `replaceRegion` into a copied heap's shadow texture** writes the heap's bytes, the
//!   caller's pages, and reports their range.
//! - **An argument encoder's write** into a no-copy buffer's bytes reports the argument's bytes.
//! - **A GPU write in place**: a blit's fill of a no-copy buffer imported in place reports the
//!   bytes it filled, at the commit and again at the completion, as does one through a second
//!   buffer over the same bytes, which lies in the first's import.
//!
//! And copies in that the log gates (`copied.rs`): each takes what the runtime reported.
//!
//! - **A GPU write in place, then a copy over the same pages**: a fill through a buffer imported
//!   in place reaches a copied buffer over the same pages at its next commit, the copy in having
//!   compared only the pages the log reported (`WorkCounts::copied_copy_in_bytes_skipped`).
//! - **A `replaceRegion` of a copied heap's texture, then a read of the heap's bytes**: the
//!   heap's own copy takes the texels at the next commit.
//!
//! The log is the test's own (`copied_log::TestLog`): it records what it is told.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use super::blit_holds::WAIT;
use super::copied_log::TestLog;
use crate::{
    Buffer, CommandQueue, Device, DeviceKind, DeviceOptions, HostMemory, ImportGate, Region,
    ResourceOptions,
};
use metal2vulkan::bind::ArgumentDescriptor;
use metal2vulkan::metal::{DataType, PixelFormat, TextureUsage};
use std::alloc::Layout;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

/// The bytes of each host memory.
const LENGTH: usize = 64 * 1024;

/// Host memory of whole pages, zeroed, freed when dropped (after the device).
pub(super) struct Pages {
    pub(super) pointer: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    pub(super) fn new() -> Pages {
        let layout = Layout::from_size_align(LENGTH, 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
        Pages { pointer, layout }
    }

    /// Its memory, logged by `log`, under a gate answering `admit` where given.
    pub(super) fn memory(&self, log: &Arc<TestLog>, admit: Option<bool>) -> HostMemory {
        // SAFETY: the memory outlives every object over it and the device (dropped after both),
        // and only their commands and the tests' reads between command buffers use it.
        let memory = unsafe { HostMemory::new(self.pointer) }.with_dirty_pages(Arc::clone(log));
        match admit {
            Some(answer) => memory.with_import_gate(Arc::new(move |_| answer) as ImportGate),
            None => memory,
        }
    }

    /// A no-copy buffer over all of it, logged by `log`, under a gate answering `admit`.
    pub(super) fn buffer(
        &self,
        device: &Device,
        log: &Arc<TestLog>,
        admit: Option<bool>,
    ) -> Buffer {
        let memory = self.memory(log, admit);
        // SAFETY: as in `memory`.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                LENGTH as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// A device of the cfg's kind; `None` where it imports no host memory.
pub(super) fn device() -> Option<Device> {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("COPIED-LOG: the device imports no host memory here; nothing to test");
        return None;
    }
    Some(device)
}

/// The writes `log` was told of by writer 0 (the consumer's, and the runtime's own CPU writes)
/// since it held `before` writes.
fn plain_writes(log: &TestLog, before: usize) -> Vec<Range<u64>> {
    (log.writes.lock().unwrap()[before..].iter())
        .filter(|w| w.2 == 0)
        .map(|w| w.1.clone())
        .collect()
}

/// A `replaceRegion` into a texture of a copied no-copy heap (a shadow of the heap's bytes, the
/// caller's pages) reports the heap's bytes it wrote, inside the texture's span, to the heap's
/// memory's log.
#[test]
fn a_replace_region_into_a_copied_heap_s_shadow_texture_is_reported() {
    let pages = Pages::new();
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(TestLog::default());
    let descriptor = crate::HeapDescriptor::new(
        crate::HeapType::Placement,
        LENGTH as u64,
        ResourceOptions::default(),
    );
    let memory = pages.memory(&log, Some(false));
    // SAFETY: as in `Pages::memory`.
    let heap = unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) };
    let heap = heap.unwrap();
    assert!(
        heap.bytes().is_some_and(Buffer::copied),
        "the heap is a copy"
    );
    let mut texture = crate::TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
    texture.usage = TextureUsage::SHADER_READ;
    texture.resource_options = ResourceOptions::default();
    let span = device.heap_texture_size_and_align(&texture).unwrap().size;
    let texture = heap.new_texture_at_offset(&texture, 0).unwrap();
    let before = log.writes.lock().unwrap().len();
    let region = Region::new_2d(0, 0, 16, 16);
    texture
        .replace_region(region, 0, 0, &[0x5C; 16 * 16 * 4], 16 * 4, 0)
        .unwrap();
    let reported = plain_writes(&log, before);
    assert_eq!(reported.len(), 1, "one report: {reported:?}");
    let range = reported[0].clone();
    assert!(
        range.start < range.end && range.end <= span,
        "inside the span: {range:?}"
    );
    assert!(
        range.end - range.start >= 16 * 16 * 4,
        "every texel: {range:?}"
    );
    drop((texture, heap));
}

/// An argument encoder's write into a no-copy buffer's bytes reports the argument's bytes to the
/// buffer's memory's log; a write into a buffer of the device's own reports nothing.
#[test]
fn an_argument_encoder_s_write_into_logged_pages_is_reported() {
    let pages = Pages::new();
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(TestLog::default());
    let buffer = pages.buffer(&device, &log, None);
    let own = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let mut encoder = device
        .new_argument_encoder_with_arguments(&[ArgumentDescriptor::new(DataType::Pointer, 0)])
        .unwrap();
    encoder.set_argument_buffer(Some(&buffer), 256).unwrap();
    let before = log.writes.lock().unwrap().len();
    encoder.set_buffer(Some(&own), 0, 0).unwrap();
    assert_eq!(
        plain_writes(&log, before),
        vec![256..264],
        "the pointer's bytes"
    );
    encoder.set_argument_buffer(Some(&own), 0).unwrap();
    let before = log.writes.lock().unwrap().len();
    encoder.set_buffer(Some(&own), 0, 0).unwrap();
    assert!(
        plain_writes(&log, before).is_empty(),
        "nothing for a buffer of its own"
    );
    drop((encoder, buffer, own));
}

/// Copy `length` bytes of `source` into a buffer of the device's own through `queue`, to
/// completion: those bytes.
fn read_back(queue: &CommandQueue, source: &Buffer, length: u64) -> Vec<u8> {
    let device = queue.device();
    let into = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(source, 0, &into, 0, length)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    let contents = into.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// Fill `range` of `target` through `queue`, to completion: the writes `log` was told of by
/// writer 0 meanwhile.
fn filled(
    queue: &CommandQueue,
    log: &TestLog,
    target: &Buffer,
    range: Range<u64>,
) -> Vec<Range<u64>> {
    let before = log.writes.lock().unwrap().len();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, range, 0x7E).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    plain_writes(log, before)
}

/// A GPU write through a no-copy buffer imported in place, over logged pages, reports the bytes
/// it wrote to the memory's log, as writer 0, at the commit and again at the completion: a
/// blit's fill of the buffer, and one through a second buffer over the same bytes, which lies in
/// the first's import. A fill of a buffer of the device's own reports nothing.
#[test]
fn a_gpu_write_into_logged_pages_imported_in_place_is_reported() {
    let pages = Pages::new();
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(TestLog::default());
    let first = pages.buffer(&device, &log, None);
    if first.copied() {
        eprintln!("COPIED-LOG: the device copies the pages; no import in place to test");
        return;
    }
    let live = device.work_counts().host_imports_live;
    let second = pages.buffer(&device, &log, None);
    let shares = device.work_counts().host_imports_live == live;
    eprintln!("COPIED-LOG: the second buffer lies in the first's import: {shares}");
    let own = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    for (target, range) in [(&first, 4096..8192), (&second, 12288..16384)] {
        let reported = filled(&queue, &log, target, range.clone());
        assert!(
            reported.len() >= 2 && reported.iter().all(|r| *r == range),
            "the fill's bytes, at the commit and the completion: {reported:?}"
        );
        // SAFETY: the command buffer completed; the bytes are the pages'.
        let byte = unsafe { *pages.pointer.as_ptr().add(range.start as usize) };
        assert_eq!(byte, 0x7E, "filled in place");
    }
    assert!(
        filled(&queue, &log, &own, 0..4096).is_empty(),
        "nothing for a buffer of the device's own"
    );
    drop((first, second, own, queue));
}

/// A fill through a no-copy buffer imported in place over logged pages reaches a copied buffer
/// over the first half of the same pages (its import refused, so a copy whose copy in the log
/// gates): read back after the fill, the copy holds the filled bytes, though its copies in
/// compare only the pages the log reports (`copied_copy_in_bytes_skipped` grows).
#[test]
fn a_gpu_write_in_place_reaches_a_logged_copy_of_the_same_pages() {
    let pages = Pages::new();
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(TestLog::default());
    let in_place = pages.buffer(&device, &log, Some(true));
    let half = (LENGTH / 2) as u64;
    let memory = pages.memory(&log, Some(false));
    // SAFETY: as in `Pages::memory`.
    let copy = unsafe {
        device.new_buffer_with_bytes_no_copy(memory, half, ResourceOptions::default(), None)
    };
    let copy = copy.unwrap();
    if in_place.copied() || !copy.copied() {
        eprintln!("COPIED-LOG: the device copies the admitted pages; nothing in place to test");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    // A first copy in compares every byte, and syncs the copy to a stamp.
    assert_eq!(read_back(&queue, &copy, half), vec![0; half as usize]);
    let skipped = device.work_counts().copied_copy_in_bytes_skipped;
    filled(&queue, &log, &in_place, 4096..8192);
    let bytes = read_back(&queue, &copy, half);
    assert!(
        bytes[4096..8192].iter().all(|b| *b == 0x7E),
        "the copy took the fill"
    );
    assert!(
        bytes[..4096].iter().chain(&bytes[8192..]).all(|b| *b == 0),
        "and nothing else"
    );
    let skipped = device.work_counts().copied_copy_in_bytes_skipped - skipped;
    eprintln!("COPIED-LOG: the copies in skipped {skipped} bytes");
    assert!(skipped > 0, "the copies in were gated by the log");
    drop((in_place, copy, queue));
}

/// A `replaceRegion` of a texture of a copied no-copy heap over logged pages writes the heap's
/// bytes in the pages; a buffer of the heap over the texture's span reads them back after the
/// next commit, the heap's own copy having taken them, though its copies in compare only the
/// pages the log reports.
#[test]
fn a_replace_region_of_a_copied_heap_s_texture_reaches_the_heap_s_copy() {
    let pages = Pages::new();
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(TestLog::default());
    let descriptor = crate::HeapDescriptor::new(
        crate::HeapType::Placement,
        LENGTH as u64,
        ResourceOptions::default(),
    );
    let memory = pages.memory(&log, Some(false));
    // SAFETY: as in `Pages::memory`.
    let heap = unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) };
    let heap = heap.unwrap();
    let mut texture = crate::TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
    texture.usage = TextureUsage::SHADER_READ;
    texture.resource_options = ResourceOptions::default();
    let span = device.heap_texture_size_and_align(&texture).unwrap().size;
    let texture = heap.new_texture_at_offset(&texture, 0).unwrap();
    let bytes = heap
        .new_buffer_at_offset(span, ResourceOptions::default(), 0)
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    assert!(read_back(&queue, &bytes, span).iter().all(|b| *b == 0));
    let skipped = device.work_counts().copied_copy_in_bytes_skipped;
    let region = Region::new_2d(0, 0, 16, 16);
    texture
        .replace_region(region, 0, 0, &[0x5C; 16 * 16 * 4], 16 * 4, 0)
        .unwrap();
    let read = read_back(&queue, &bytes, span);
    let texels = read.iter().filter(|b| **b == 0x5C).count();
    assert_eq!(
        texels,
        16 * 16 * 4,
        "the heap's copy holds every texel's bytes"
    );
    let skipped = device.work_counts().copied_copy_in_bytes_skipped - skipped;
    eprintln!("COPIED-LOG: the copies in skipped {skipped} bytes");
    assert!(skipped > 0, "the copies in were gated by the log");
    drop((bytes, texture, heap, queue));
}
