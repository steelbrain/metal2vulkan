//! The moment a program may unmap the host pages a no-copy buffer lies over (docs/HOWTO.md,
//! "Unmapping host memory"): `VK_EXT_external_memory_host` requires the imported range to stay
//! valid for the imported memory's whole life, so the pages go only once every import of them was
//! freed. A no-copy buffer's keep-alive is dropped, and with none its deallocator runs, only then,
//! whichever of the runtime's threads frees the import:
//! - the releasing thread, or a handler thread where a keep-alive alone waits for the free, for a
//!   buffer no command buffer holds;
//! - the queue's thread that sees a command buffer holding the buffer complete, for one released
//!   while that command buffer waits for an event;
//! - that thread again, for one whose import the device released while idle and its next blit
//!   imported again (`host_residency.rs`), the device held to releasing idle pointer imports
//!   whatever its check found (`tests/pointer_residency.rs`).
//!
//! Each signal reads the device's live host imports (`WorkCounts::host_imports_live`, a census of
//! its imported blocks under the memory's lock, under which each import's `vkFreeMemory` runs) and
//! then unmaps the pages, as a program that unmaps there does, while another thread submits
//! command buffers over the device's own buffers throughout: RADV lists every import of the device
//! in every submission. Every command buffer completes, and the device is neither lost nor
//! recovered.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on Linux, on the kind of device the cfg selects.

use super::host_residency::{device, host_page, release, Scratch};
use super::pointer_residency::release_pointer_imports;
use crate::{
    Buffer, BufferContents, BufferDescriptor, CommandQueue, Device, EventRef, HostMemory,
    ResourceOptions, SharedEvent,
};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::ThreadId;
use std::time::Duration;

/// The rounds, each of every case with every signal.
const ROUNDS: usize = 48;

/// The pages each buffer lies over.
const PAGES: usize = 4;

/// How long the test waits for a signal.
const SIGNAL_WAIT: Duration = Duration::from_secs(30);

/// The bytes the other thread's command buffers copy.
const OTHER_BYTES: u64 = 64 * 1024;

/// How the buffer's import goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    /// The buffer released while no command buffer holds it.
    Released,
    /// The buffer released while a committed command buffer that copies it waits for an event.
    Held,
    /// The buffer's import released while idle, then the buffer copied, its import made again,
    /// and released as in [`Case::Held`].
    Reimported,
}

/// What tells the program the pages may go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    KeepAlive,
    Deallocator,
}

/// What a signal saw as it ran.
#[derive(Debug)]
struct Seen {
    live: u64,
    thread: ThreadId,
}

/// The program's unmap of its pages, run as the signal comes: as its drop.
struct Unmap {
    at: usize,
    len: usize,
    device: Device,
    seen: mpsc::Sender<Seen>,
}

impl Drop for Unmap {
    fn drop(&mut self) {
        let live = self.device.work_counts().host_imports_live;
        // SAFETY: the pages are this test's, mapped for this buffer alone, which let go of them.
        let unmapped = unsafe { libc::munmap(self.at as *mut libc::c_void, self.len) };
        assert_eq!(unmapped, 0, "munmap");
        let thread = std::thread::current().id();
        let _ = self.seen.send(Seen { live, thread });
    }
}

/// New private anonymous pages, `len` bytes, written: their address.
fn pages(len: usize, byte: u8) -> usize {
    // SAFETY: a new private mapping, the test's until its signal unmaps it.
    let at = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(at, libc::MAP_FAILED, "mmap");
    // SAFETY: the new mapping, read-write.
    unsafe { std::ptr::write_bytes(at.cast::<u8>(), byte, len) };
    at as usize
}

/// A no-copy buffer over `len` bytes of new pages, imported by their pointer, which `signal`
/// unmaps.
fn no_copy(device: &Device, len: usize, signal: Signal, seen: &mpsc::Sender<Seen>) -> Buffer {
    let at = pages(len, 0x5a);
    let unmap = Unmap {
        at,
        len,
        device: device.clone(),
        seen: seen.clone(),
    };
    // SAFETY: the pages stay mapped until `unmap` drops: with the keep-alive, or the deallocator.
    let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) };
    let contents = match signal {
        Signal::KeepAlive => BufferContents::NoCopy {
            memory,
            deallocator: None,
            keep_alive: Some(Box::new(unmap)),
        },
        Signal::Deallocator => BufferContents::NoCopy {
            memory,
            deallocator: Some(Box::new(move |_, _| drop(unmap))),
            keep_alive: None,
        },
    };
    let options = ResourceOptions::default();
    let descriptor = BufferDescriptor::new(len as u64, options, contents);
    device.new_buffer_with_descriptor(descriptor).unwrap()
}

/// Commit a copy of `buffer` into `into` behind a wait for `event` to reach `value`.
fn held_copy(
    queue: &CommandQueue,
    buffer: &Buffer,
    into: &Buffer,
    (event, value): (&SharedEvent, u64),
) -> crate::CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(event), value)).unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, into, 0, buffer.length())
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// Copy `buffer` into `into` and wait.
fn copy(queue: &CommandQueue, buffer: &Buffer, into: &Buffer) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, into, 0, buffer.length())
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed().unwrap();
    assert_eq!(command_buffer.error(), None, "a copy of the no-copy buffer");
}

/// Command buffers of copies between two buffers of the device's own, on a queue of their own,
/// submitted one after another until `stop`: how many, and the errors any answered.
fn submitting(device: Device, stop: Arc<AtomicBool>) -> (usize, Vec<String>) {
    let queue = device.new_command_queue().unwrap();
    let from = device
        .new_buffer(OTHER_BYTES, ResourceOptions::shared())
        .unwrap();
    let into = device
        .new_buffer(OTHER_BYTES, ResourceOptions::shared())
        .unwrap();
    let (mut count, mut failed) = (0, Vec::new());
    while !stop.load(Ordering::Acquire) {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        (blit.copy_from_buffer_to_buffer(&from, 0, &into, 0, OTHER_BYTES)).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        if let Err(error) = command_buffer.wait_until_completed() {
            failed.push(format!("wait: {error:?}"));
        }
        if let Some(error) = command_buffer.error() {
            failed.push(format!("{error:?}"));
        }
        count += 1;
    }
    (count, failed)
}

/// Every case with every signal, [`ROUNDS`] times, while another thread submits (the module docs).
#[test]
fn a_no_copy_buffer_s_signal_comes_after_every_free_of_its_pages() {
    const TEST: &str = "unmap order";
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    assert!(
        device.settle_in_place_checks(crate::tests::SETTLE),
        "{TEST}: the in-place checks never settled"
    );
    release_pointer_imports(&device);
    let len = PAGES * host_page();
    let queue = device.new_command_queue().unwrap();
    let into = device
        .new_buffer(len as u64, ResourceOptions::shared())
        .unwrap();
    let event = device.new_shared_event().unwrap();
    let mut value = 0;
    let (seen, signals) = mpsc::channel();
    let here = std::thread::current().id();
    let base = device.work_counts();

    let stop = Arc::new(AtomicBool::new(false));
    let other = std::thread::spawn({
        let (device, stop) = (device.clone(), Arc::clone(&stop));
        move || submitting(device, stop)
    });

    let cases = [Case::Released, Case::Held, Case::Reimported];
    let mut ran = Vec::new();
    for round in 0..ROUNDS {
        for case in cases {
            for signal in [Signal::KeepAlive, Signal::Deallocator] {
                let what = format!("{TEST}: round {round}, {case:?}, {signal:?}");
                let buffer = no_copy(&device, len, signal, &seen);
                let live = device.work_counts().host_imports_live;
                assert_eq!(live, base.host_imports_live + 1, "{what}: one import");
                if case == Case::Reimported {
                    copy(&queue, &buffer, &into);
                    release(&what, &device, 1);
                    let released = device.work_counts().host_imports_live;
                    assert_eq!(released, base.host_imports_live, "{what}: released");
                }
                let held = (case != Case::Released).then(|| {
                    value += 1;
                    held_copy(&queue, &buffer, &into, (&event, value))
                });
                drop(buffer);
                if let Some(command_buffer) = held {
                    // The command buffer holds the buffer: nothing went yet.
                    assert!(signals.try_recv().is_err(), "{what}: signalled while held");
                    event.set_signaled_value(value);
                    command_buffer.wait_until_completed().unwrap();
                    assert_eq!(command_buffer.error(), None, "{what}: the held copy");
                }
                let came = signals.recv_timeout(SIGNAL_WAIT).unwrap();
                assert_eq!(
                    came.live, base.host_imports_live,
                    "{what}: an import of the pages lived as the signal came"
                );
                if case != Case::Released {
                    assert_ne!(came.thread, here, "{what}: freed on the test's thread");
                }
                ran.push((case, signal));
            }
        }
    }

    stop.store(true, Ordering::Release);
    let (submitted, failed) = other.join().unwrap();
    assert!(failed.is_empty(), "{TEST}: the other thread's: {failed:?}");
    let counts = device.work_counts();
    assert!(!device.is_lost(), "{TEST}: the device lost");
    assert_eq!(
        counts.device_recoveries, base.device_recoveries,
        "{TEST}: recovered"
    );
    let reimported = counts.host_imports_reimported - base.host_imports_reimported;
    let released = counts.host_imports_released - base.host_imports_released;
    let expected = (ROUNDS * 2) as u64;
    assert_eq!(released, expected, "{TEST}: imports released while idle");
    assert_eq!(reimported, expected, "{TEST}: imports made again");
    eprintln!(
        "UNMAP-ORDER {TEST}: on {}: {} buffers unmapped at their signals ({} rounds; released and \
         imported again while idle: {released}), {submitted} command buffers of the other thread",
        device.name(),
        ran.len(),
        ROUNDS,
    );
}
