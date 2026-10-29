//! Copied host buffers on a device (`copied.rs`): a no-copy buffer whose import the caller's gate
//! refuses, or whose import the gate and the device admitted and that then failed, is made over a
//! copy of its pages, whose reads and writes match the caller's bytes at the commit and at the
//! completion.
//!
//! - **The gate refuses.** The gate is asked once, with the buffer's pages, and the buffer is a
//!   copy: no import of the pages is made, its `contents` is the pointer, and the copy is
//!   announced (`copied-host-pages`) and counted as a refusal.
//! - **The admitted import fails.** The gate admits the import, which the driver then fails (the
//!   tests' stand-in, `Device::fail_next_host_imports`, answers it as out of device memory): the
//!   buffer is a copy all the same, announced and counted as degraded; the next buffer imports.
//!   The same without a gate.
//! - **Coherence.** In each, the GPU reads the caller's bytes; a CPU write before a commit reaches
//!   the GPU, those bytes alone copied in; a fill reaches the caller's pages at its completion,
//!   those bytes alone merged back.
//! - **Unreadable pages** (mapped without access, memory Metal makes no buffer over) are never
//!   copied: the buffer is refused, as their import is.
//! - **A dma-buf's mapping** (Linux, where `/dev/udmabuf` opens), whose pages the kernel's read
//!   cannot pin, is copied where the gate refuses its import, with the same coherence.
//! - **Rebinds.** A rebind's new pages are a copy where the gate refuses their import or the
//!   admitted import fails, with the same coherence; the old pages stay as they were, and a later
//!   rebind imports again.
//! - **Heaps.** A no-copy heap's own buffer is a copy in the same two cases; a buffer placed in
//!   it reads the caller's bytes at its offset, and a fill through it reaches those pages alone.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use crate::{
    AddressPlacement, Buffer, BufferContents, BufferDescriptor, CommandBufferStatus, Degradation,
    Device, DeviceKind, DeviceOptions, GpuAddress, HostMemory, ImportGate, ImportRequest,
    MessageSink, ResourceOptions, WorkCounts,
};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);
/// The bytes of each host memory.
const LENGTH: usize = 64 * 1024;

/// Host memory of whole pages, in a pattern of its own, freed when dropped (after the device).
struct Host {
    pointer: NonNull<u8>,
    backing: Backing,
    seed: u8,
}

/// What a [`Host`]'s pages are.
enum Backing {
    /// An allocation of this layout.
    Allocated(Layout),
    /// A mapping of a dma-buf of a `memfd`'s pages, as a consumer maps guest RAM.
    #[cfg(target_os = "linux")]
    DmaBuf {
        _mapping: super::copied::DmaBufMapping,
    },
}

impl Host {
    fn new(seed: u8) -> Host {
        let layout = Layout::from_size_align(LENGTH, 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
        let host = Host {
            pointer,
            backing: Backing::Allocated(layout),
            seed,
        };
        host.write(0, &host.pattern());
        host
    }

    /// Host memory over a dma-buf's mapping (`VM_PFNMAP`), or `None` where `/dev/udmabuf` does
    /// not open, which it says.
    #[cfg(target_os = "linux")]
    fn over_dma_buf(seed: u8) -> Option<Host> {
        let mapping = super::copied::DmaBufMapping::new(&pattern(seed))?;
        Some(Host {
            pointer: mapping.pages,
            backing: Backing::DmaBuf { _mapping: mapping },
            seed,
        })
    }

    /// The pattern it was made with.
    fn pattern(&self) -> Vec<u8> {
        pattern(self.seed)
    }

    /// Its bytes, read while no command buffer runs.
    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the allocation's bytes, which no command buffer writes now.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), LENGTH) }.to_vec()
    }

    /// Write `bytes` at `at`, while no command buffer runs.
    fn write(&self, at: usize, bytes: &[u8]) {
        assert!(at + bytes.len() <= LENGTH);
        // SAFETY: the allocation's bytes, which no command buffer uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.pointer.as_ptr().add(at),
                bytes.len(),
            )
        };
    }

    /// A no-copy buffer over all of it, under a gate where given.
    fn buffer(&self, device: &Device, gate: Option<ImportGate>) -> Buffer {
        // SAFETY: the memory outlives every buffer over it and the device (dropped after both),
        // and only the buffers' commands and the tests' reads between command buffers use it.
        let memory = unsafe { HostMemory::new(self.pointer) };
        let memory = match gate {
            Some(gate) => memory.with_import_gate(gate),
            None => memory,
        };
        // SAFETY: as above.
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

impl Drop for Host {
    fn drop(&mut self) {
        match self.backing {
            // SAFETY: allocated with this layout.
            Backing::Allocated(layout) => unsafe {
                std::alloc::dealloc(self.pointer.as_ptr(), layout)
            },
            // Unmapped by the mapping's own drop.
            #[cfg(target_os = "linux")]
            Backing::DmaBuf { .. } => {}
        }
    }
}

/// The bytes of a [`Host`] of `seed`.
fn pattern(seed: u8) -> Vec<u8> {
    (0..LENGTH)
        .map(|i| ((i % 251) as u8).wrapping_add(seed.wrapping_mul(37)))
        .collect()
}

/// The degrade notices a device announced.
type Heard = Arc<Mutex<Vec<Degradation>>>;

/// A device of the cfg's kind, its notices into the answer's list; `None` where it imports no
/// host memory.
fn device() -> Option<(Device, Heard)> {
    let heard: Heard = Arc::default();
    let hearing = Arc::clone(&heard);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                hearing.lock().unwrap().push(notice.degradation);
            }
        })),
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("COPIED: the device imports no host memory here; nothing to test");
        return None;
    }
    Some((device, heard))
}

/// A gate answering `answer`, each request it is asked into `asked`.
fn gate(asked: &Arc<Mutex<Vec<ImportRequest>>>, answer: bool) -> ImportGate {
    let asked = Arc::clone(asked);
    Arc::new(move |request| {
        asked.lock().unwrap().push(request);
        answer
    })
}

/// Commit one command buffer whose blit encoder `encode` fills, and wait for it.
fn blit(device: &Device, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The buffer's bytes as the GPU reads them: copied into one of the device's own, read there.
fn gpu_bytes(device: &Device, buffer: &Buffer) -> Vec<u8> {
    let own = device
        .new_buffer(LENGTH as u64, ResourceOptions::shared())
        .unwrap();
    blit(device, |encoder| {
        encoder
            .copy_from_buffer_to_buffer(buffer, 0, &own, 0, LENGTH as u64)
            .unwrap();
    });
    let contents = own.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), LENGTH) }.to_vec()
}

/// `buffer` lies in a copy of `host`'s pages: the GPU reads its bytes; a CPU write before a
/// commit reaches the GPU, those bytes alone copied in; a fill reaches its pages alone, those
/// bytes alone merged back.
fn lies_in_a_copy(device: &Device, buffer: &Buffer, host: &Host) {
    assert!(buffer.copied(), "the buffer is a copy");
    assert_eq!(
        buffer.contents(),
        Some(host.pointer),
        "contents are the pointer"
    );
    assert_eq!(
        gpu_bytes(device, buffer),
        host.pattern(),
        "the GPU reads the bytes"
    );
    // A CPU write before the commit reaches the GPU: only the bytes it changed are copied in.
    let pattern = host.pattern();
    let changed_in = pattern[..4096].iter().filter(|&&byte| byte != 0xC3).count() as u64;
    let before = device.work_counts();
    host.write(0, &[0xC3; 4096]);
    let read = gpu_bytes(device, buffer);
    let after_read = device.work_counts();
    assert!(
        read[..4096].iter().all(|&byte| byte == 0xC3),
        "the GPU reads the CPU's write"
    );
    assert_eq!(read[4096..], pattern[4096..], "and the bytes after it");
    assert_eq!(
        after_read.copied_copy_in_bytes - before.copied_copy_in_bytes,
        changed_in,
        "the commit copies in the bytes the CPU changed alone"
    );
    // A fill reaches the pages at its completion: only the bytes it changed are merged back.
    let changed_back = pattern[4096..8192]
        .iter()
        .filter(|&&byte| byte != 0x5A)
        .count() as u64;
    blit(device, |encoder| {
        encoder.fill_buffer(buffer, 4096..8192, 0x5A).unwrap();
    });
    let filled = device.work_counts();
    let bytes = host.bytes();
    assert!(
        bytes[4096..8192].iter().all(|&byte| byte == 0x5A),
        "the fill reached the pages"
    );
    assert!(
        bytes[..4096].iter().all(|&byte| byte == 0xC3),
        "the bytes before it stay"
    );
    assert_eq!(bytes[8192..], pattern[8192..], "the bytes after it stay");
    assert_eq!(
        filled.copied_merge_back_bytes - after_read.copied_merge_back_bytes,
        changed_back,
        "the fill's bytes alone were merged back"
    );
    eprintln!(
        "COPIED: copied in {} bytes, merged back {}",
        after_read.copied_copy_in_bytes - before.copied_copy_in_bytes,
        filled.copied_merge_back_bytes - after_read.copied_merge_back_bytes
    );
}

/// The differences between two counts that a no-copy buffer's making moves: gate refusals,
/// copies made, of them refused and degraded, and pointer imports made.
fn moved(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    [
        after.host_imports_refused_by_host - before.host_imports_refused_by_host,
        after.copied_host_buffers_made - before.copied_host_buffers_made,
        after.copied_host_buffers_refused - before.copied_host_buffers_refused,
        after.copied_host_buffers_degraded - before.copied_host_buffers_degraded,
        after.host_pointer_imports_made - before.host_pointer_imports_made,
    ]
}

/// How many of `heard` are `copied-host-pages`.
fn announced(heard: &Heard) -> usize {
    let notices = heard.lock().unwrap();
    notices
        .iter()
        .filter(|&&kind| kind == Degradation::CopiedHostPages)
        .count()
}

#[test]
fn copied_buffer_is_made_where_the_caller_s_gate_refuses_the_import() {
    let (host, other) = (Host::new(1), Host::new(2));
    let Some((device, heard)) = device() else {
        return;
    };
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    let buffer = host.buffer(&device, Some(gate(&asked, false)));
    let after = device.work_counts();
    // Refused: a copy made, counted as a refusal, no import made.
    assert_eq!(moved(&before, &after), [1, 1, 1, 0, 0]);
    let requests = asked.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "the gate was asked once");
    assert_eq!(requests[0].start, host.pointer, "from the first page");
    assert_eq!(requests[0].length, LENGTH as u64, "through the last");
    assert_eq!(announced(&heard), 1, "announced once");
    lies_in_a_copy(&device, &buffer, &host);
    // A gate that admits it lets the import go on.
    let before = device.work_counts();
    let imported = other.buffer(&device, Some(gate(&asked, true)));
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [0, 0, 0, 0, 1]);
    assert!(!imported.copied(), "imported");
    assert_eq!(gpu_bytes(&device, &imported), other.pattern());
    assert_eq!(asked.lock().unwrap().len(), 2, "asked once more");
    drop((buffer, imported));
}

/// Pages of a dma-buf's mapping (`VM_PFNMAP`, as a consumer maps guest RAM), which the kernel's
/// read cannot pin, whose import the gate refuses, are copied all the same (read with plain loads,
/// mapped readable), with the copy's coherence. Linux only, where `/dev/udmabuf` opens.
#[cfg(target_os = "linux")]
#[test]
fn copied_buffer_is_made_over_a_dma_buf_mapping_the_gate_refuses() {
    let Some(host) = Host::over_dma_buf(9) else {
        return;
    };
    let Some((device, heard)) = device() else {
        return;
    };
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    let buffer = host.buffer(&device, Some(gate(&asked, false)));
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [1, 1, 1, 0, 0]);
    assert_eq!(announced(&heard), 1, "announced once");
    lies_in_a_copy(&device, &buffer, &host);
    drop(buffer);
}

#[test]
fn copied_buffer_is_made_where_an_admitted_import_fails_announced() {
    let (host, other) = (Host::new(3), Host::new(4));
    let Some((device, heard)) = device() else {
        return;
    };
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    device.fail_next_host_imports(1);
    let buffer = host.buffer(&device, Some(gate(&asked, true)));
    let after = device.work_counts();
    // Admitted, its import failed: a copy made, degraded, no refusal.
    assert_eq!(moved(&before, &after), [0, 1, 0, 1, 0]);
    assert_eq!(asked.lock().unwrap().len(), 1, "the gate was asked once");
    assert_eq!(announced(&heard), 1, "announced once");
    lies_in_a_copy(&device, &buffer, &host);
    // With no failure left, the next buffer imports its pages.
    let before = device.work_counts();
    let imported = other.buffer(&device, Some(gate(&asked, true)));
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [0, 0, 0, 0, 1]);
    assert_eq!(gpu_bytes(&device, &imported), other.pattern());
    assert_eq!(announced(&heard), 1, "nothing more announced");
    drop((buffer, imported));
}

#[test]
fn copied_buffer_without_a_gate_whose_import_fails_is_made_announced() {
    let host = Host::new(5);
    let Some((device, heard)) = device() else {
        return;
    };
    let before = device.work_counts();
    device.fail_next_host_imports(1);
    let buffer = host.buffer(&device, None);
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [0, 1, 0, 1, 0]);
    assert_eq!(announced(&heard), 1, "announced once");
    lies_in_a_copy(&device, &buffer, &host);
}

/// Pages mapped without access are memory Metal makes no buffer over: a refusing gate makes no
/// copy of them (reading them would fault), and the buffer is refused.
#[cfg(unix)]
#[test]
fn copied_buffer_is_never_made_over_pages_the_process_cannot_read() {
    let Some((device, heard)) = device() else {
        return;
    };
    // SAFETY: a new private anonymous mapping without access, at an address the kernel chooses.
    let mapped = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            LENGTH,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(mapped, libc::MAP_FAILED);
    let pointer = NonNull::new(mapped.cast::<u8>()).unwrap();
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    // SAFETY: the buffer is refused before anything reads the pages (the test's claim); the
    // mapping outlives the call.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(pointer).with_import_gate(gate(&asked, false)),
            LENGTH as u64,
            ResourceOptions::default(),
            None,
        )
    };
    let after = device.work_counts();
    assert!(made.is_err(), "refused: {made:?}");
    assert_eq!(
        after.copied_host_buffers_made,
        before.copied_host_buffers_made
    );
    assert_eq!(announced(&heard), 0, "nothing announced");
    // SAFETY: the mapping made above, which nothing uses.
    assert_eq!(unsafe { libc::munmap(mapped, LENGTH) }, 0);
}

/// A placed no-copy buffer whose import fails over pages the process cannot read is refused, and
/// the refusal gives its range of the address space back: a buffer placed at the same address
/// next is made, as after Metal's nil.
#[cfg(unix)]
#[test]
fn a_refused_copy_gives_back_its_placed_range() {
    let Some((device, _heard)) = device() else {
        return;
    };
    let at = AddressPlacement::At(GpuAddress(0x100_0001_8000));
    // SAFETY: a new private anonymous mapping without access, at an address the kernel chooses.
    let mapped = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            LENGTH,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(mapped, libc::MAP_FAILED);
    let pointer = NonNull::new(mapped.cast::<u8>()).unwrap();
    // The import fails on every device, so the call tries a copy, which the pages refuse.
    device.fail_next_host_imports(1);
    let contents = BufferContents::NoCopy {
        // SAFETY: the buffer is refused before anything reads the pages (the test's claim); the
        // mapping outlives the call.
        memory: unsafe { HostMemory::new(pointer) },
        deallocator: None,
        keep_alive: None,
    };
    let refused = device.new_buffer_with_descriptor(
        BufferDescriptor::new(LENGTH as u64, ResourceOptions::default(), contents)
            .with_placement(at),
    );
    assert!(refused.is_err(), "refused: {refused:?}");
    let again = device.new_buffer_with_descriptor(
        BufferDescriptor::new(
            LENGTH as u64,
            ResourceOptions::default(),
            BufferContents::Zero,
        )
        .with_placement(at),
    );
    let again = again.expect("the refused buffer's range is free again");
    assert_eq!(again.gpu_address(), GpuAddress(0x100_0001_8000));
    drop(again);
    // SAFETY: the mapping made above, which nothing uses.
    assert_eq!(unsafe { libc::munmap(mapped, LENGTH) }, 0);
}

/// Rebind `buffer` to all of `host`, under a gate where given.
fn rebind(buffer: &Buffer, host: &Host, gate: Option<ImportGate>) -> Result<(), crate::Error> {
    // SAFETY: the memory outlives every buffer over it and the device (dropped after both), and
    // no command that uses the buffer is encoded or pending.
    let memory = unsafe { HostMemory::new(host.pointer) };
    let memory = match gate {
        Some(gate) => memory.with_import_gate(gate),
        None => memory,
    };
    // SAFETY: as above.
    unsafe { buffer.rebind_bytes_no_copy(memory, None, None) }
}

#[test]
fn copied_rebind_pages_are_made_where_the_caller_s_gate_refuses_the_import() {
    let (first, second) = (Host::new(6), Host::new(7));
    let Some((device, heard)) = device() else {
        return;
    };
    let buffer = first.buffer(&device, None);
    assert!(!buffer.copied(), "imported first");
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    rebind(&buffer, &second, Some(gate(&asked, false))).unwrap();
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [1, 1, 1, 0, 0]);
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        1
    );
    let requests = asked.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "the gate was asked once");
    assert_eq!(requests[0].start, second.pointer, "with the new pages");
    assert_eq!(requests[0].length, LENGTH as u64);
    assert_eq!(announced(&heard), 1, "announced once");
    lies_in_a_copy(&device, &buffer, &second);
    assert_eq!(
        first.bytes(),
        first.pattern(),
        "the old pages stay as they were"
    );
    // Rebound again with no gate, it imports.
    let before = device.work_counts();
    rebind(&buffer, &first, None).unwrap();
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [0, 0, 0, 0, 1]);
    assert!(!buffer.copied(), "imported again");
    assert_eq!(gpu_bytes(&device, &buffer), first.pattern());
}

#[test]
fn copied_rebind_pages_are_made_where_an_admitted_import_fails_announced() {
    let (first, second) = (Host::new(8), Host::new(9));
    let Some((device, heard)) = device() else {
        return;
    };
    let buffer = first.buffer(&device, None);
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    device.fail_next_host_imports(1);
    rebind(&buffer, &second, Some(gate(&asked, true))).unwrap();
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [0, 1, 0, 1, 0]);
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        1
    );
    assert_eq!(asked.lock().unwrap().len(), 1, "the gate was asked once");
    assert_eq!(announced(&heard), 1, "announced once");
    lies_in_a_copy(&device, &buffer, &second);
    assert_eq!(
        first.bytes(),
        first.pattern(),
        "the old pages stay as they were"
    );
}

/// A no-copy placement heap over all of `host`, under a gate where given.
fn heap(device: &Device, host: &Host, gate: Option<ImportGate>) -> crate::Heap {
    // SAFETY: the memory outlives every heap over it and the device (dropped after both), and
    // only the heap's resources' commands and the tests' reads between command buffers use it.
    let memory = unsafe { HostMemory::new(host.pointer) };
    let memory = match gate {
        Some(gate) => memory.with_import_gate(gate),
        None => memory,
    };
    let descriptor = crate::HeapDescriptor::new(
        crate::HeapType::Placement,
        LENGTH as u64,
        ResourceOptions::default(),
    );
    // SAFETY: as above.
    unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) }.unwrap()
}

#[test]
fn copied_heap_is_made_where_the_caller_s_gate_refuses_the_import() {
    let (host, other) = (Host::new(10), Host::new(11));
    let Some((device, heard)) = device() else {
        return;
    };
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    let copied = heap(&device, &host, Some(gate(&asked, false)));
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [1, 1, 1, 0, 0]);
    assert_eq!(after.copied_heaps_made - before.copied_heaps_made, 1);
    let requests = asked.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "the gate was asked once");
    assert_eq!(requests[0].start, host.pointer);
    assert_eq!(requests[0].length, LENGTH as u64);
    assert_eq!(announced(&heard), 1, "announced once");
    // Its hazard tracking is an imported heap's.
    let imported = heap(&device, &other, None);
    assert_eq!(
        copied.hazard_tracking_mode(),
        imported.hazard_tracking_mode()
    );
    // A buffer placed at 16 KiB reads the caller's bytes there; a fill through it reaches those
    // pages alone, merged back.
    let at = 16 * 1024;
    let buffer = copied
        .new_buffer_at_offset(LENGTH as u64 - at, ResourceOptions::default(), at)
        .unwrap();
    let own = device
        .new_buffer(LENGTH as u64 - at, ResourceOptions::shared())
        .unwrap();
    blit(&device, |encoder| {
        encoder
            .copy_from_buffer_to_buffer(&buffer, 0, &own, 0, LENGTH as u64 - at)
            .unwrap();
    });
    let pattern = host.pattern();
    let contents = own.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    let read = unsafe { std::slice::from_raw_parts(contents.as_ptr(), LENGTH - at as usize) };
    assert_eq!(
        read,
        &pattern[at as usize..],
        "the GPU reads the heap's bytes"
    );
    let before = device.work_counts();
    blit(&device, |encoder| {
        encoder.fill_buffer(&buffer, 0..4096, 0x5A).unwrap();
    });
    let after = device.work_counts();
    let bytes = host.bytes();
    let (start, end) = (at as usize, at as usize + 4096);
    assert!(bytes[start..end].iter().all(|&byte| byte == 0x5A));
    assert_eq!(bytes[..start], pattern[..start], "the bytes before it stay");
    assert_eq!(bytes[end..], pattern[end..], "the bytes after it stay");
    let changed = pattern[start..end].iter().filter(|&&b| b != 0x5A).count() as u64;
    assert_eq!(
        after.copied_merge_back_bytes - before.copied_merge_back_bytes,
        changed
    );
    drop((buffer, copied, imported));
}

#[test]
fn copied_heap_is_made_where_an_admitted_import_fails_announced() {
    let host = Host::new(12);
    let Some((device, heard)) = device() else {
        return;
    };
    let before = device.work_counts();
    device.fail_next_host_imports(1);
    let copied = heap(&device, &host, None);
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [0, 1, 0, 1, 0]);
    assert_eq!(after.copied_heaps_made - before.copied_heaps_made, 1);
    assert_eq!(announced(&heard), 1, "announced once");
    drop(copied);
}

#[test]
fn copied_heap_s_texture_keeps_its_texels_in_the_caller_s_pages() {
    let host = Host::new(13);
    let Some((device, _heard)) = device() else {
        return;
    };
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let copied = heap(&device, &host, Some(gate(&asked, false)));
    let mut descriptor = crate::TextureDescriptor::texture_2d(
        metal2vulkan::metal::PixelFormat::RGBA8Unorm,
        16,
        16,
        false,
    );
    descriptor.usage = metal2vulkan::metal::TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::default();
    let span = device
        .heap_texture_size_and_align(&descriptor)
        .unwrap()
        .size as usize;
    let texture = copied.new_texture_at_offset(&descriptor, 0).unwrap();
    let bytes = 16 * 16 * 4;
    let size = crate::Size::new(16, 16, 1);
    let origin = crate::Origin { x: 0, y: 0, z: 0 };
    let staging = device
        .new_buffer_with_bytes(&[0xA5; 16 * 16 * 4], ResourceOptions::shared())
        .unwrap();
    let before = device.work_counts();
    blit(&device, |encoder| {
        encoder
            .copy_from_buffer_to_texture(
                &staging,
                0,
                16 * 4,
                bytes,
                size,
                &texture,
                0,
                0,
                origin,
                crate::BlitOption::None,
            )
            .unwrap();
    });
    let after = device.work_counts();
    // The texels reach the caller's pages in the texture's span alone, merged back.
    let pattern = host.pattern();
    assert_eq!(
        host.bytes()[span..],
        pattern[span..],
        "past the texture's span"
    );
    assert!(after.copied_merge_back_bytes > before.copied_merge_back_bytes);
    // A copy out of it reads them back.
    let own = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    blit(&device, |encoder| {
        encoder
            .copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                origin,
                size,
                &own,
                0,
                16 * 4,
                bytes,
                crate::BlitOption::None,
            )
            .unwrap();
    });
    let contents = own.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    let read = unsafe { std::slice::from_raw_parts(contents.as_ptr(), bytes as usize) };
    assert!(
        read.iter().all(|&byte| byte == 0xA5),
        "the texels read back"
    );
    drop((texture, copied));
}
