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
//! - **Rebinds with textures.** A rebind of a buffer with a texture in place over it whose gate
//!   refuses the new pages, and one whose new bytes lie at another offset in their page, give it
//!   a copy, the texture's image made again over it: the texture reads the new texels, its writes
//!   reach the new pages, and the buffer's bytes are the new pages'. A heap's rebind whose gate
//!   refuses, or whose admitted import fails, gives the heap a copy: its placed buffer and its
//!   texture bound to its bytes read and write the new pages. A command buffer committed before
//!   such a rebind and held back reaches the copy, encoded again; a rebind while one not yet
//!   committed used the buffer is refused, the gate unasked.
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

/// Host memory of `length` bytes of whole pages, zeroed, freed when dropped (after the device and
/// every buffer over it).
struct Pages {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    fn new(length: usize) -> Pages {
        let layout = Layout::from_size_align(length, 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
        Pages { pointer, layout }
    }

    /// A no-copy buffer over all of it, its import refused by the gate: a copy.
    fn copied(&self, device: &Device) -> Buffer {
        // SAFETY: the memory outlives every buffer over it and the device (dropped after both).
        let memory = unsafe { HostMemory::new(self.pointer) }.with_import_gate(Arc::new(|_| false));
        // SAFETY: as above.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                self.layout.size() as u64,
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
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) }
    }
}

/// A shared event signaled after a command buffer's write to a copied buffer: a reader that
/// sees the event's value, by reading it as a guest's CPU spins on it or by a host wait, finds
/// the write in the caller's pages, as Metal's writes before a signal are there once its value is
/// seen. The signal is kept behind the merge (`WorkCounts::event_signals_behind_merge`). A large
/// copy beside it makes each merge long, so a value seen before the merge would show.
#[test]
fn a_shared_event_s_value_is_seen_once_the_copy_s_bytes_are_in_the_pages() {
    let host = Host::new(3);
    let large = Pages::new(64 << 20);
    let Some((device, _heard)) = device() else {
        return;
    };
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let buffer = host.buffer(&device, Some(gate(&asked, false)));
    assert!(buffer.copied(), "the buffer is a copy");
    let large = (large.copied(&device), large);
    assert!(large.0.copied(), "the large buffer is a copy");
    let event = device.new_shared_event().unwrap();
    let queue = device.new_command_queue().unwrap();
    for round in 1..=6u64 {
        let fill = 0x10 + round as u8;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(&buffer, 0..4096, fill).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_signal_event(crate::EventRef::from(&event), round)
            .unwrap();
        let before = device.work_counts();
        command_buffer.commit().unwrap();
        // Even rounds spin on the value, odd ones wait on the host.
        if round % 2 == 0 {
            let deadline = std::time::Instant::now() + WAIT;
            while event.signaled_value() < round {
                assert!(std::time::Instant::now() < deadline, "round {round}");
                std::hint::spin_loop();
            }
        } else {
            assert!(
                event.wait_until_signaled_value(round, WAIT),
                "round {round}"
            );
        }
        // SAFETY: the pages' first bytes, which the command buffer wrote before the signal.
        let seen = unsafe { std::slice::from_raw_parts(host.pointer.as_ptr(), 4096) }.to_vec();
        assert!(
            seen.iter().all(|&byte| byte == fill),
            "round {round}: the write is in the pages once the value is seen",
        );
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let after = device.work_counts();
        assert_eq!(
            after.event_signals_behind_merge - before.event_signals_behind_merge,
            1,
            "round {round}: the signal is kept behind the merge",
        );
    }
    drop(large.0);
}

/// The side, bytes per texel and row pitch of the textures over buffers below.
const SIDE: usize = 16;
const TEXEL: usize = 4;
const PITCH: usize = 256;

/// The rows of a texture's texels in `bytes`, its first texel at 0.
fn rows(bytes: &[u8]) -> Vec<u8> {
    (0..SIDE)
        .flat_map(|row| bytes[row * PITCH..row * PITCH + SIDE * TEXEL].to_vec())
        .collect()
}

/// A sampled 16 x 16 RGBA8 texture over `buffer` at 0, its rows `PITCH` apart, and whether it
/// is an image in place over the bytes (`WorkCounts::host_textures_in_place`).
fn texture_over(device: &Device, buffer: &Buffer) -> (crate::Texture, bool) {
    let mut descriptor = crate::TextureDescriptor::texture_2d(
        metal2vulkan::metal::PixelFormat::RGBA8Unorm,
        SIDE as u64,
        SIDE as u64,
        false,
    );
    descriptor.usage = metal2vulkan::metal::TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let before = device.work_counts();
    let texture = buffer.new_texture(&descriptor, 0, PITCH as u64).unwrap();
    let after = device.work_counts();
    let in_place = after.host_textures_in_place > before.host_textures_in_place;
    eprintln!(
        "COPIED: the texture over the buffer is {}",
        if in_place { "in place" } else { "a shadow" }
    );
    (texture, in_place)
}

/// The texels of a 16 x 16 RGBA8 texture, as the GPU copies them out.
fn texels(device: &Device, texture: &crate::Texture) -> Vec<u8> {
    let bytes = (SIDE * SIDE * TEXEL) as u64;
    let own = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    blit(device, |encoder| {
        encoder
            .copy_from_texture_to_buffer(
                texture,
                0,
                0,
                crate::Origin { x: 0, y: 0, z: 0 },
                crate::Size::new(SIDE as u64, SIDE as u64, 1),
                &own,
                0,
                (SIDE * TEXEL) as u64,
                bytes,
                crate::BlitOption::None,
            )
            .unwrap();
    });
    let contents = own.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), bytes as usize) }.to_vec()
}

/// Write `value` into every texel of a 16 x 16 RGBA8 texture, by a GPU copy.
fn write_texels(device: &Device, texture: &crate::Texture, value: u8) {
    let bytes = SIDE * SIDE * TEXEL;
    let staging = device
        .new_buffer_with_bytes(&vec![value; bytes], ResourceOptions::shared())
        .unwrap();
    blit(device, |encoder| {
        encoder
            .copy_from_buffer_to_texture(
                &staging,
                0,
                (SIDE * TEXEL) as u64,
                bytes as u64,
                crate::Size::new(SIDE as u64, SIDE as u64, 1),
                texture,
                0,
                0,
                crate::Origin { x: 0, y: 0, z: 0 },
                crate::BlitOption::None,
            )
            .unwrap();
    });
}

/// The texture over `buffer`, which lies in a copy of the `LENGTH` bytes at `pages` whose
/// pattern is `pattern`, reads those texels; a GPU write into it reaches the texels' bytes in
/// `pages` alone, the bytes between its rows and after it kept.
fn texture_follows_the_copy(
    device: &Device,
    buffer: &Buffer,
    texture: &crate::Texture,
    pages: NonNull<u8>,
    pattern: &[u8],
) {
    assert!(buffer.copied(), "the buffer is a copy");
    assert_eq!(
        buffer.contents(),
        Some(pages),
        "contents are the new pointer"
    );
    assert_eq!(texels(device, texture), rows(pattern), "the new texels");
    assert_eq!(gpu_bytes(device, buffer), pattern, "the buffer's new bytes");
    write_texels(device, texture, 0xA5);
    // SAFETY: the caller's bytes, which no command buffer uses now.
    let now = unsafe { std::slice::from_raw_parts(pages.as_ptr(), LENGTH) }.to_vec();
    for row in 0..SIDE {
        let (start, end) = (row * PITCH, row * PITCH + SIDE * TEXEL);
        assert!(
            now[start..end].iter().all(|&byte| byte == 0xA5),
            "row {row}'s texels reached the pages"
        );
        let next = (row + 1) * PITCH;
        assert_eq!(
            now[end..next],
            pattern[end..next],
            "the bytes after row {row}"
        );
    }
    assert_eq!(
        now[SIDE * PITCH..],
        pattern[SIDE * PITCH..],
        "past the texels"
    );
    assert!(texels(device, texture).iter().all(|&byte| byte == 0xA5));
}

#[test]
fn copied_rebind_makes_a_texture_s_image_in_place_again_over_the_copy() {
    let (first, second) = (Host::new(20), Host::new(21));
    let Some((device, heard)) = device() else {
        return;
    };
    let settled = device.settle_in_place_checks(Duration::from_secs(120));
    let buffer = first.buffer(&device, None);
    let (texture, in_place) = texture_over(&device, &buffer);
    eprintln!("COPIED: in-place verdicts settled: {settled}");
    assert_eq!(texels(&device, &texture), rows(&first.pattern()));
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    rebind(&buffer, &second, Some(gate(&asked, false))).unwrap();
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [1, 1, 1, 0, 0]);
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        1
    );
    assert_eq!(asked.lock().unwrap().len(), 1, "the gate was asked once");
    assert_eq!(announced(&heard), 1, "announced once");
    let images = after.copied_rebind_images_made - before.copied_rebind_images_made;
    assert_eq!(
        images,
        u64::from(in_place),
        "the image in place made over the copy"
    );
    texture_follows_the_copy(
        &device,
        &buffer,
        &texture,
        second.pointer,
        &second.pattern(),
    );
    assert_eq!(first.bytes(), first.pattern(), "the old pages stay");
    // Rebound again with no gate, it imports, the image made again over the import.
    rebind(&buffer, &first, None).unwrap();
    assert!(!buffer.copied(), "imported again");
    assert_eq!(texels(&device, &texture), rows(&first.pattern()));
    eprintln!(
        "COPIED-REBIND-IN-PLACE: texture {} over a gate-refused rebind's copy",
        if in_place { "in place" } else { "a shadow" }
    );
    drop((texture, buffer));
}

/// Host memory of `LENGTH` bytes `at` bytes into a page, in `seed`'s pattern, freed when dropped.
struct OffPage {
    allocation: NonNull<u8>,
    layout: Layout,
    pointer: NonNull<u8>,
    pattern: Vec<u8>,
}

impl OffPage {
    fn new(seed: u8, at: usize) -> OffPage {
        let layout = Layout::from_size_align(LENGTH + (1 << 16), 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let allocation = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
        // SAFETY: `at` is inside the allocation's first 64 KiB.
        let pointer = unsafe { NonNull::new_unchecked(allocation.as_ptr().add(at)) };
        let pattern: Vec<u8> = (0..LENGTH)
            .map(|i| ((i % 241) as u8).wrapping_add(seed.wrapping_mul(29)))
            .collect();
        // SAFETY: the allocation holds `LENGTH` bytes past `pointer`, which nothing uses yet.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), pointer.as_ptr(), LENGTH) };
        OffPage {
            allocation,
            layout,
            pointer,
            pattern,
        }
    }
}

impl Drop for OffPage {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout.
        unsafe { std::alloc::dealloc(self.allocation.as_ptr(), self.layout) };
    }
}

#[test]
fn rebind_to_bytes_moved_in_their_page_under_a_texture_is_a_copy() {
    let first = Host::new(22);
    let moved_bytes = OffPage::new(23, 256);
    let Some((device, heard)) = device() else {
        return;
    };
    device.settle_in_place_checks(Duration::from_secs(120));
    let buffer = first.buffer(&device, None);
    let (texture, in_place) = texture_over(&device, &buffer);
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    // SAFETY: the memory outlives the buffer and the device; no command uses the buffer.
    let memory =
        unsafe { HostMemory::new(moved_bytes.pointer) }.with_import_gate(gate(&asked, true));
    // SAFETY: as above.
    unsafe { buffer.rebind_bytes_no_copy(memory, None, None) }.unwrap();
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [0, 1, 0, 0, 0]);
    assert_eq!(
        after.copied_rebind_pages_moved - before.copied_rebind_pages_moved,
        1
    );
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        1
    );
    assert!(
        asked.lock().unwrap().is_empty(),
        "nothing imported, the gate unasked"
    );
    assert_eq!(announced(&heard), 1, "announced once");
    let images = after.copied_rebind_images_made - before.copied_rebind_images_made;
    assert_eq!(
        images,
        u64::from(in_place),
        "the image in place made over the copy"
    );
    texture_follows_the_copy(
        &device,
        &buffer,
        &texture,
        moved_bytes.pointer,
        &moved_bytes.pattern,
    );
    assert_eq!(first.bytes(), first.pattern(), "the old pages stay");
    // Back at the old offset, it imports.
    rebind(&buffer, &first, None).unwrap();
    assert!(!buffer.copied(), "imported again");
    assert_eq!(texels(&device, &texture), rows(&first.pattern()));
    eprintln!(
        "COPIED-REBIND-MOVED: texture {} over a moved rebind's copy",
        if in_place { "in place" } else { "a shadow" }
    );
    drop((texture, buffer));
}

/// Rebind `heap` to all of `host`, under a gate where given.
fn rebind_heap(
    heap: &crate::Heap,
    host: &Host,
    gate: Option<ImportGate>,
) -> Result<(), crate::Error> {
    // SAFETY: the memory outlives the heap and the device (dropped after both), and no command
    // that uses the heap is encoded or pending.
    let memory = unsafe { HostMemory::new(host.pointer) };
    let memory = match gate {
        Some(gate) => memory.with_import_gate(gate),
        None => memory,
    };
    // SAFETY: as above.
    unsafe { heap.rebind_bytes_no_copy(memory, None, None) }
}

/// `heap`, rebound to a copy of `host`'s pages, holds its placed `buffer` (at `at`) and its
/// `texture` (at 0, `span` bytes) over them: the buffer reads the new bytes, a fill through it
/// reaches those pages alone; the texture's written texels reach the pages in its span alone and
/// read back.
fn heap_follows_the_copy(
    device: &Device,
    (buffer, at): (&Buffer, usize),
    (texture, span): (&crate::Texture, usize),
    host: &Host,
) {
    let pattern = host.pattern();
    assert_eq!(gpu_bytes_of(device, buffer, LENGTH - at), pattern[at..]);
    let before = device.work_counts();
    blit(device, |encoder| {
        encoder.fill_buffer(buffer, 0..4096, 0x5A).unwrap();
    });
    let after = device.work_counts();
    let bytes = host.bytes();
    assert!(bytes[at..at + 4096].iter().all(|&byte| byte == 0x5A));
    assert_eq!(
        bytes[span..at],
        pattern[span..at],
        "between texture and buffer"
    );
    assert_eq!(bytes[at + 4096..], pattern[at + 4096..], "after the fill");
    assert!(after.copied_merge_back_bytes > before.copied_merge_back_bytes);
    write_texels(device, texture, 0xA5);
    let bytes = host.bytes();
    assert_eq!(
        bytes[span..at],
        pattern[span..at],
        "past the texture's span"
    );
    assert!(
        bytes[..span].contains(&0xA5),
        "the texels reached the pages"
    );
    assert!(texels(device, texture).iter().all(|&byte| byte == 0xA5));
}

/// The first `length` bytes of `buffer` as the GPU reads them.
fn gpu_bytes_of(device: &Device, buffer: &Buffer, length: usize) -> Vec<u8> {
    let own = device
        .new_buffer(length as u64, ResourceOptions::shared())
        .unwrap();
    blit(device, |encoder| {
        encoder
            .copy_from_buffer_to_buffer(buffer, 0, &own, 0, length as u64)
            .unwrap();
    });
    let contents = own.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length) }.to_vec()
}

/// A no-copy heap with a texture bound at 0 and a buffer placed at 32 KiB, rebound under a gate
/// answering `gate_answer` where given, the next `fail` imports failing: a copy, counted as
/// `expect` ([`moved`]), the gate asked once where given; rebound again with no gate, it imports.
fn heap_rebind_is_a_copy(gate_answer: Option<bool>, fail: u64, expect: [u64; 5]) {
    let (first, second) = (Host::new(24), Host::new(25));
    let Some((device, heard)) = device() else {
        return;
    };
    let imported = heap(&device, &first, None);
    let mut descriptor = crate::TextureDescriptor::texture_2d(
        metal2vulkan::metal::PixelFormat::RGBA8Unorm,
        SIDE as u64,
        SIDE as u64,
        false,
    );
    descriptor.usage = metal2vulkan::metal::TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::default();
    let span = device
        .heap_texture_size_and_align(&descriptor)
        .unwrap()
        .size as usize;
    let at = 32 * 1024;
    assert!(span <= at);
    let texture = imported.new_texture_at_offset(&descriptor, 0).unwrap();
    let buffer = imported
        .new_buffer_at_offset((LENGTH - at) as u64, ResourceOptions::default(), at as u64)
        .unwrap();
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let before = device.work_counts();
    device.fail_next_host_imports(fail);
    let gate = gate_answer.map(|answer| gate(&asked, answer));
    rebind_heap(&imported, &second, gate).unwrap();
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), expect);
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        1
    );
    let requests = asked.lock().unwrap().clone();
    assert_eq!(
        requests.len(),
        usize::from(gate_answer.is_some()),
        "the gate asked once"
    );
    if let Some(request) = requests.first() {
        assert_eq!(request.start, second.pointer);
        assert_eq!(request.length, LENGTH as u64);
    }
    assert_eq!(announced(&heard), 1, "announced once");
    eprintln!(
        "COPIED-HEAP-REBIND: {} heap image(s) made again over the copy",
        after.copied_rebind_images_made - before.copied_rebind_images_made
    );
    heap_follows_the_copy(&device, (&buffer, at), (&texture, span), &second);
    assert_eq!(first.bytes(), first.pattern(), "the old pages stay");
    // Rebound again with no gate, it imports.
    let before = device.work_counts();
    rebind_heap(&imported, &first, None).unwrap();
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [0, 0, 0, 0, 1]);
    assert_eq!(
        gpu_bytes_of(&device, &buffer, LENGTH - at),
        first.pattern()[at..]
    );
    drop((buffer, texture, imported));
}

#[test]
fn heap_rebind_is_a_copy_where_the_caller_s_gate_refuses_the_import() {
    heap_rebind_is_a_copy(Some(false), 0, [1, 1, 1, 0, 0]);
}

#[test]
fn heap_rebind_is_a_copy_where_an_admitted_import_fails_announced() {
    heap_rebind_is_a_copy(Some(true), 1, [0, 1, 0, 1, 0]);
}

/// A command buffer committed before a gate-refused rebind and held back by an event fills the
/// copy, encoded again, and its fill reaches the new pages at its completion, as one committed
/// after the rebind does; a rebind while a command buffer not yet committed used the buffer is
/// refused, the gate unasked, and the buffer stays.
#[test]
fn copied_rebind_keeps_metal_s_order_around_command_buffers() {
    let (first, second) = (Host::new(26), Host::new(27));
    let Some((device, _heard)) = device() else {
        return;
    };
    device.settle_in_place_checks(Duration::from_secs(120));
    let buffer = first.buffer(&device, None);
    let (texture, _) = texture_over(&device, &buffer);
    let queue = device.new_command_queue().unwrap();
    // A use not yet committed refuses the rebind before the gate is asked.
    let uncommitted = queue.command_buffer().unwrap();
    let mut encoder = uncommitted.blit_command_encoder().unwrap();
    encoder.fill_buffer(&buffer, 0..16, 0x77).unwrap();
    encoder.end_encoding().unwrap();
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let refused = rebind(&buffer, &second, Some(gate(&asked, false)));
    assert!(
        format!("{refused:?}").contains("EncodedUses"),
        "refused as encoded uses: {refused:?}"
    );
    assert!(asked.lock().unwrap().is_empty(), "the gate unasked");
    assert!(!buffer.copied(), "the buffer stays");
    drop(uncommitted);
    // One committed and held back by an event, then the rebind, then one committed after it.
    let event = device.new_shared_event().unwrap();
    let held = queue.command_buffer().unwrap();
    held.encode_wait_for_event(crate::EventRef::from(&event), 1)
        .unwrap();
    let mut encoder = held.blit_command_encoder().unwrap();
    encoder.fill_buffer(&buffer, 0..16, 0x11).unwrap();
    encoder.end_encoding().unwrap();
    held.commit().unwrap();
    let before = device.work_counts();
    rebind(&buffer, &second, Some(gate(&asked, false))).unwrap();
    assert_eq!(asked.lock().unwrap().len(), 1, "the gate asked once");
    assert!(buffer.copied());
    let after_rebind = device.work_counts();
    let next = queue.command_buffer().unwrap();
    let mut encoder = next.blit_command_encoder().unwrap();
    encoder.fill_buffer(&buffer, 16..32, 0x22).unwrap();
    encoder.end_encoding().unwrap();
    next.commit().unwrap();
    assert_eq!(held.status(), CommandBufferStatus::Committed);
    event.set_signaled_value(1);
    for command_buffer in [&held, &next] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
    }
    let pattern = second.pattern();
    let bytes = second.bytes();
    assert_eq!(&bytes[..16], &[0x11; 16], "the held fill, encoded again");
    assert_eq!(&bytes[16..32], &[0x22; 16], "the later fill");
    assert_eq!(bytes[32..], pattern[32..]);
    assert_eq!(first.bytes(), first.pattern(), "the old pages untouched");
    assert_eq!(
        after_rebind.encoders_reencoded - before.encoders_reencoded,
        1,
        "the held encoder encoded again"
    );
    let mut expected = pattern.clone();
    expected[..16].fill(0x11);
    expected[16..32].fill(0x22);
    assert_eq!(texels(&device, &texture), rows(&expected));
    drop((texture, buffer));
}

/// Pages the process maps read-only, as a consumer maps its alias of a read-only replace
/// (`replaceBackingWithRanges:readOnly:`, its maximum protection lowered to read), over shared
/// memory a writable mapping also maps (the guest's own): `LENGTH` bytes in a pattern of `seed`,
/// the read-only view at a multiple of 64 KiB.
#[cfg(any(target_os = "linux", target_os = "macos"))]
struct ReadOnlyHost {
    /// The read-only view's first byte.
    pointer: NonNull<u8>,
    /// The writable mapping of the same memory.
    writable: NonNull<u8>,
    /// The address space the view was placed in (`2 * LENGTH` bytes), unmapped with it.
    reserved: NonNull<u8>,
    /// The shared memory, a `memfd`, which udmabuf takes for the source's dma-bufs.
    #[cfg(target_os = "linux")]
    memory: Arc<std::os::fd::OwnedFd>,
    seed: u8,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl ReadOnlyHost {
    fn new(seed: u8) -> ReadOnlyHost {
        use std::os::fd::AsRawFd;
        let memory = Arc::new(shared_memory());
        let fd = memory.as_raw_fd();
        // SAFETY: a new shared mapping of the memory, at an address the kernel chooses.
        let writable = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                LENGTH,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        assert_ne!(
            writable,
            libc::MAP_FAILED,
            "{}",
            std::io::Error::last_os_error()
        );
        // SAFETY: a new private reservation without access, at an address the kernel chooses.
        let reserved = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                2 * LENGTH,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(
            reserved,
            libc::MAP_FAILED,
            "{}",
            std::io::Error::last_os_error()
        );
        let at = (reserved as usize).next_multiple_of(1 << 16);
        // SAFETY: a read-only shared mapping of the memory over the reservation's own pages.
        let view = unsafe {
            libc::mmap(
                at as *mut libc::c_void,
                LENGTH,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                0,
            )
        };
        assert_eq!(view as usize, at, "{}", std::io::Error::last_os_error());
        let host = ReadOnlyHost {
            pointer: NonNull::new(view.cast()).unwrap(),
            writable: NonNull::new(writable.cast()).unwrap(),
            reserved: NonNull::new(reserved.cast()).unwrap(),
            #[cfg(target_os = "linux")]
            memory,
            seed,
        };
        host.write(0, &pattern(seed));
        host
    }

    /// The pattern it was made with.
    fn pattern(&self) -> Vec<u8> {
        pattern(self.seed)
    }

    /// Its bytes, read through the read-only view while no command buffer runs.
    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the view's bytes, readable, which no command buffer writes now.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), LENGTH) }.to_vec()
    }

    /// Write `bytes` at `at` through the writable mapping, while no command buffer runs.
    fn write(&self, at: usize, bytes: &[u8]) {
        assert!(at + bytes.len() <= LENGTH);
        // SAFETY: the writable mapping's bytes, which no command buffer uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.writable.as_ptr().add(at),
                bytes.len(),
            )
        };
    }

    /// The view as the consumer hands it to the runtime, under `gate`: on Linux with a source of
    /// dma-bufs of its pages (one chunk of the `memfd`'s pages behind each request, through
    /// `/dev/udmabuf`), counting its calls in `calls`.
    ///
    /// # Safety
    ///
    /// The view outlives every buffer, heap and rebind over it and the device.
    unsafe fn memory(
        &self,
        gate: ImportGate,
        calls: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> HostMemory {
        #[cfg(target_os = "linux")]
        let memory = {
            use std::sync::atomic::Ordering;
            let fd = Arc::clone(&self.memory);
            let base = self.pointer.as_ptr() as usize;
            let calls = Arc::clone(calls);
            let source: crate::DmaBufSource = Arc::new(move |request: crate::DmaBufRequest| {
                calls.fetch_add(1, Ordering::AcqRel);
                let offset = (request.start.as_ptr() as usize)
                    .checked_sub(base)
                    .ok_or_else(|| std::io::Error::other("pages before the view"))?;
                Ok(vec![crate::DmaBufChunk {
                    fd: super::host_residency::udmabuf(&fd, offset as u64, request.length)?,
                    length: request.length,
                }])
            });
            // SAFETY: the caller's; the source's chunks hold exactly the view's pages.
            unsafe { HostMemory::with_dma_buf(self.pointer, source) }
        };
        #[cfg(target_os = "macos")]
        let memory = {
            let _ = calls;
            // SAFETY: the caller's.
            unsafe { HostMemory::new(self.pointer) }
        };
        memory.with_import_gate(gate)
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Drop for ReadOnlyHost {
    fn drop(&mut self) {
        // SAFETY: the mappings made by `new`, which nothing uses now: the reservation holds the
        // view.
        unsafe {
            assert_eq!(libc::munmap(self.writable.as_ptr().cast(), LENGTH), 0);
            assert_eq!(libc::munmap(self.reserved.as_ptr().cast(), 2 * LENGTH), 0);
        }
    }
}

/// `LENGTH` bytes of new shared memory, zero.
#[cfg(target_os = "linux")]
fn shared_memory() -> std::os::fd::OwnedFd {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    // SAFETY: a NUL-terminated name.
    let fd = unsafe {
        libc::memfd_create(
            c"m2v-read-only-pages".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    assert!(fd >= 0, "memfd: {}", std::io::Error::last_os_error());
    // SAFETY: a new descriptor, owned at once.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    // SAFETY: sizes and seals the memfd made above, against shrinking, as udmabuf requires.
    unsafe {
        assert_eq!(libc::ftruncate(fd.as_raw_fd(), LENGTH as libc::off_t), 0);
        assert_eq!(
            libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK),
            0
        );
    }
    fd
}

/// `LENGTH` bytes of new shared memory, zero: a POSIX shared memory object, unlinked at once.
#[cfg(target_os = "macos")]
fn shared_memory() -> std::os::fd::OwnedFd {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let name = format!(
        "/m2v-ro-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let name = std::ffi::CString::new(name).unwrap();
    // SAFETY: a NUL-terminated name; a new object, owned at once and unlinked.
    let fd = unsafe {
        let fd = libc::shm_open(
            name.as_ptr(),
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
            0o600 as libc::c_uint,
        );
        assert!(fd >= 0, "shm_open: {}", std::io::Error::last_os_error());
        libc::shm_unlink(name.as_ptr());
        OwnedFd::from_raw_fd(fd)
    };
    // SAFETY: sizes the object made above.
    assert_eq!(
        unsafe { libc::ftruncate(fd.as_raw_fd(), LENGTH as libc::off_t) },
        0
    );
    fd
}

/// `buffer`, whose bytes are those of `host` from `at`, lies in a copy of the read-only pages
/// whose GPU writes are dropped: the GPU reads their bytes; a CPU write through the writable
/// mapping before a commit reaches the GPU; a fill completes, the pages keep their bytes, nothing
/// is merged back, the fill's bytes are dropped (counted), and a later command buffer reads the
/// pages' bytes, not the fill's.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn drops_the_gpu_s_writes(device: &Device, (buffer, at): (&Buffer, usize), host: &ReadOnlyHost) {
    let length = LENGTH - at;
    assert!(buffer.copied(), "the buffer is a copy");
    let pattern = host.pattern();
    assert_eq!(
        gpu_bytes_of(device, buffer, length),
        pattern[at..],
        "the GPU reads the bytes"
    );
    host.write(at, &[0xC3; 4096]);
    let mut expected = pattern.clone();
    expected[at..at + 4096].fill(0xC3);
    assert_eq!(
        gpu_bytes_of(device, buffer, length),
        expected[at..],
        "the GPU reads the CPU's write"
    );
    let dropped = expected[at + 4096..at + 8192]
        .iter()
        .filter(|&&byte| byte != 0x5A)
        .count() as u64;
    let before = device.work_counts();
    blit(device, |encoder| {
        encoder.fill_buffer(buffer, 4096..8192, 0x5A).unwrap();
    });
    let after = device.work_counts();
    assert_eq!(host.bytes(), expected, "the pages keep their bytes");
    assert_eq!(
        after.copied_merge_back_bytes - before.copied_merge_back_bytes,
        0,
        "nothing is merged back"
    );
    assert_eq!(
        after.copied_dropped_bytes - before.copied_dropped_bytes,
        dropped,
        "the fill's bytes are dropped"
    );
    assert_eq!(
        gpu_bytes_of(device, buffer, length),
        expected[at..],
        "a later command buffer reads the pages' bytes"
    );
    eprintln!("COPIED-READ-ONLY: dropped {dropped} bytes the GPU wrote");
}

/// A no-copy buffer over read-only pages (the consumer's no-copy object over its read-only alias)
/// whose import the gate refuses is a copy of them, the source never called, whose GPU writes are
/// dropped.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn copied_buffer_over_read_only_pages_drops_the_gpu_s_writes() {
    let host = ReadOnlyHost::new(28);
    let Some((device, heard)) = device() else {
        return;
    };
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let calls: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
    let before = device.work_counts();
    // SAFETY: the view outlives the buffer and the device (dropped after both).
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            host.memory(gate(&asked, false), &calls),
            LENGTH as u64,
            ResourceOptions::default(),
            None,
        )
    }
    .expect("a buffer over read-only pages the gate refuses is a copy");
    let after = device.work_counts();
    assert_eq!(moved(&before, &after), [1, 1, 1, 0, 0]);
    let requests = asked.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "the gate was asked once");
    assert_eq!(requests[0].start, host.pointer, "with the pages");
    assert_eq!(requests[0].length, LENGTH as u64);
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Acquire),
        0,
        "the source unasked"
    );
    assert_eq!(announced(&heard), 1, "announced once");
    drops_the_gpu_s_writes(&device, (&buffer, 0), &host);
    drop(buffer);
}

/// A buffer's rebind to read-only pages (the consumer's `replaceBackingWithRanges:readOnly:`)
/// whose import the gate refuses gives it a copy of them, the source never called, whose GPU
/// writes are dropped; the old pages stay as they were.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn copied_rebind_over_read_only_pages_drops_the_gpu_s_writes() {
    let (first, second) = (Host::new(29), ReadOnlyHost::new(30));
    let Some((device, heard)) = device() else {
        return;
    };
    let buffer = first.buffer(&device, None);
    assert!(!buffer.copied(), "imported first");
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let calls: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
    let before = device.work_counts();
    // SAFETY: the view outlives the buffer and the device (dropped after both), and no command
    // that uses the buffer is encoded or pending.
    let rebound = unsafe {
        buffer.rebind_bytes_no_copy(second.memory(gate(&asked, false), &calls), None, None)
    };
    let after = device.work_counts();
    rebound.expect("a rebind to read-only pages the gate refuses is a copy");
    assert_eq!(moved(&before, &after), [1, 1, 1, 0, 0]);
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        1
    );
    let requests = asked.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "the gate was asked once");
    assert_eq!(requests[0].start, second.pointer, "with the new pages");
    assert_eq!(requests[0].length, LENGTH as u64);
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Acquire),
        0,
        "the source unasked"
    );
    assert_eq!(announced(&heard), 1, "announced once");
    drops_the_gpu_s_writes(&device, (&buffer, 0), &second);
    assert_eq!(first.bytes(), first.pattern(), "the old pages stay");
}

/// A heap's rebind to read-only pages whose import the gate refuses gives it a copy of them, the
/// source never called, whose GPU writes through a buffer placed in it are dropped.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn copied_heap_rebind_over_read_only_pages_drops_the_gpu_s_writes() {
    let (first, second) = (Host::new(31), ReadOnlyHost::new(32));
    let Some((device, heard)) = device() else {
        return;
    };
    let imported = heap(&device, &first, None);
    let at = 32 * 1024;
    let buffer = imported
        .new_buffer_at_offset((LENGTH - at) as u64, ResourceOptions::default(), at as u64)
        .unwrap();
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let calls: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
    let before = device.work_counts();
    // SAFETY: the view outlives the heap and the device (dropped after both), and no command
    // that uses the heap is encoded or pending.
    let rebound = unsafe {
        imported.rebind_bytes_no_copy(second.memory(gate(&asked, false), &calls), None, None)
    };
    let after = device.work_counts();
    rebound.expect("a heap's rebind to read-only pages the gate refuses is a copy");
    assert_eq!(moved(&before, &after), [1, 1, 1, 0, 0]);
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        1
    );
    let requests = asked.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "the gate was asked once");
    assert_eq!(requests[0].start, second.pointer, "with the new pages");
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Acquire),
        0,
        "the source unasked"
    );
    assert_eq!(announced(&heard), 1, "announced once");
    drops_the_gpu_s_writes(&device, (&buffer, at), &second);
    assert_eq!(first.bytes(), first.pattern(), "the old pages stay");
    drop((buffer, imported));
}

/// A buffer's rebind to read-only pages whose import the gate admits goes to the caller's source
/// (Linux drivers pin a pointer's pages for writing); where the device's import of those dma-bufs
/// then fails (the RTX 5080's, out of device memory), the buffer is a copy of them all the same,
/// counted as degraded, whose GPU writes are dropped. Where the device imports them, or none, the
/// test says what it took and passes.
#[cfg(target_os = "linux")]
#[test]
fn copied_rebind_over_read_only_pages_whose_admitted_import_fails_drops_the_gpu_s_writes() {
    let (first, second) = (Host::new(33), ReadOnlyHost::new(34));
    let Some((device, heard)) = device() else {
        return;
    };
    let buffer = first.buffer(&device, None);
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let calls: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
    let before = device.work_counts();
    // SAFETY: the view outlives the buffer and the device (dropped after both), and no command
    // that uses the buffer is encoded or pending.
    let rebound = unsafe {
        buffer.rebind_bytes_no_copy(second.memory(gate(&asked, true), &calls), None, None)
    };
    let after = device.work_counts();
    let called = calls.load(std::sync::atomic::Ordering::Acquire);
    assert_eq!(asked.lock().unwrap().len(), 1, "the gate was asked once");
    // An import that failed is never the answer: the copy is.
    if let Err(error @ (crate::Error::OutOfMemory(_) | crate::Error::Vulkan(_))) = &rebound {
        panic!("the admitted import failed and no copy was made: {error:?}");
    }
    if !buffer.copied() {
        eprintln!(
            "COPIED-READ-ONLY-ADMITTED: not a copy here ({rebound:?}, the source called {called} \
             times, {} dma-buf imports made): nothing to test",
            after.dma_buf_imports - before.dma_buf_imports
        );
        return;
    }
    rebound.unwrap();
    assert_eq!(moved(&before, &after), [0, 1, 0, 1, 0]);
    assert_eq!(called, 1, "the source was asked once");
    assert_eq!(announced(&heard), 1, "announced once");
    drops_the_gpu_s_writes(&device, (&buffer, 0), &second);
    eprintln!("COPIED-READ-ONLY-ADMITTED: the source's import failed, and the copy was made");
}

/// A batch that only reads a copied buffer leaves it uncompared at its merge back
/// (`copied_merges_skipped`), and one that writes it merges it back.
#[test]
fn a_merge_back_skips_a_copy_its_batch_only_read() {
    let Some((device, _)) = device() else {
        return;
    };
    let host = Host::new(11);
    let asked = Arc::default();
    let buffer = host.buffer(&device, Some(gate(&asked, false)));
    assert!(buffer.copied(), "the buffer is a copy");
    blit(&device, |encoder| {
        encoder.fill_buffer(&buffer, 0..256, 0x3C).unwrap();
    });
    assert_eq!(&host.bytes()[..256], &[0x3C; 256], "a write merged back");
    let before = device.work_counts();
    let read = gpu_bytes(&device, &buffer);
    let after = device.work_counts();
    assert_eq!(read, host.bytes(), "the GPU reads the pages");
    let skipped = after.copied_merges_skipped - before.copied_merges_skipped;
    let compared = after.copied_merges_compared - before.copied_merges_compared;
    eprintln!("COPIED: a reading batch's merge back: {compared} compared, {skipped} skipped");
    assert_eq!(
        (compared, skipped),
        (0, 1),
        "the copy a batch only read is not compared"
    );
    drop(buffer);
}

/// A kernel's stores through a copied buffer's address loaded from memory, with no use of the
/// buffer named (the buffer reached unnamed, as a command buffer's earlier use left it), still
/// merge back into the pages: its command may write memory no use names, so every copy is
/// compared.
#[test]
fn a_write_through_an_address_no_use_names_still_merges_back() {
    let Some((device, _)) = device() else {
        return;
    };
    let host = Host::new(13);
    let asked = Arc::default();
    let buffer = host.buffer(&device, Some(gate(&asked, false)));
    assert!(buffer.copied(), "the buffer is a copy");
    // A command buffer's use of it: the device's live buffers list it, reached unnamed from here.
    let _ = gpu_bytes(&device, &buffer);
    let function = super::encoder_holds::library(&device, "engine/ob2_loaded_pointer.ll")
        .new_function_with_name("ob2_loaded_pointer")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &crate::PipelineOptions::default())
        .unwrap();
    let indices: Vec<u8> = (0..16u32).flat_map(u32::to_le_bytes).collect();
    let out = device.new_buffer(32, ResourceOptions::shared()).unwrap();
    let before = device.work_counts();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_bytes(&indices, 0).unwrap();
    encoder
        .set_bytes(&buffer.gpu_address().0.to_le_bytes(), 1)
        .unwrap();
    encoder.set_buffer(Some(&out), 0, 2).unwrap();
    encoder
        .dispatch_threads(crate::Size::new(8, 1, 1), crate::Size::new(8, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let after = device.work_counts();
    let words: Vec<u32> = (host.bytes()[32..64].as_chunks::<4>().0.iter())
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    let stored: Vec<u32> = (0..8).map(|t| 0xB000_0000 | t).collect();
    eprintln!(
        "COPIED: the unnamed write's merge back: {} compared, {} skipped",
        after.copied_merges_compared - before.copied_merges_compared,
        after.copied_merges_skipped - before.copied_merges_skipped,
    );
    assert_eq!(
        words, stored,
        "the stores through the address are in the pages"
    );
    drop((buffer, out, state, queue));
}
