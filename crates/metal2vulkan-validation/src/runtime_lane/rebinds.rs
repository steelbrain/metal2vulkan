//! Rebinding a no-copy buffer, or a heap over the caller's host memory, to other host pages
//! (`rebind_bytes_no_copy`, a runtime extension; the runtime's `rebind.rs`) on the device the cfg
//! selects, under the validation layer:
//!
//! - **What stays.** The buffer or heap and the resources placed in it keep their `gpu_address`,
//!   length and resource IDs; `contents` answers the new pointer at once. The GPU reads and
//!   writes the new pages from the next command on, through bindings (a kernel case of the
//!   consumer's task memory) and blits alike, and the CPU's writes at the new pointer are what it
//!   reads. Nothing of the old pages is copied.
//! - **The ordering rule, by first submission** (before, by commit). A command buffer first
//!   submitted before a rebind keeps the old pages until it completes; one first submitted after it
//!   reaches the new pages, though committed before (`rebind_reach` pins the reaches); a command it
//!   encoded naming the buffer, resolved then, is encoded again by the rebind (It failed
//!   `EncodedUseRebound` before); one encoded after it uses the new pages. Each range's deallocator runs once,
//!   after the last submitted command buffer that used its pages completed and its completed
//!   handlers ran (in the rebind itself where none holds them: one not yet submitted holds
//!   none), and sees that command buffer's writes. A rebind while a command buffer not yet
//!   committed used the buffer, or while an encoder that used it is open, is refused
//!   (`Invalid`, `RebindProblem::EncodedUses`); once it is committed, or dropped, the rebind is
//!   made.
//! - **Refusals.** A buffer of the runtime's memory, a heap's buffer and a heap of the runtime's
//!   memory are refused `Invalid` (`RebindProblem::NotHostMemory`); a buffer with a live texture
//!   over it, given memory at another offset in its page, is given a copy of the new bytes laid
//!   out at the old offset (`copied-host-pages`), not refused; a
//!   heap's memory off the import's unit `Invalid` (`HeapProblem::HostMemoryAlignment`). Each
//!   refused call drops its deallocator unrun and changes nothing.
//! - **Textures.** A heap's textures, bound to its bytes or shadows of them, a texture or a
//!   texture buffer over a no-copy buffer, and a texture over a buffer of a no-copy heap (each
//!   2D texture aliasing the pages or a shadow of them, as the device lays its rows), read and
//!   write the new pages from the next command on, laid out as before; the old pages are left as they were. A command buffer committed
//!   before the rebind and first submitted after it that samples a heap's texture samples the
//!   new pages, the old pages left intact. `REBIND cost` lines print what a heap's rebind takes
//!   with textures bound to its bytes, each of which takes a new image.

use super::host_memory::{
    counting, gpu_fill, gpu_read, host_page, no_copy, not_importable, on_device, recording, run,
    Mapping, Seen, GUEST_PAGE,
};
use super::no_copy_heaps::{blit_into, blit_texels, no_copy_heap, shared, HEAP};
use super::{
    bound_buffers, encode_case, pipeline_state, public_case, read_buffer, run_case,
    texture_descriptor, CaseResources,
};
use crate::candidate::engine::ComputeOutput;
use metal2vulkan::metal::TextureType;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Deallocator, Device, Error, EventRef,
    Heap, HeapDescriptor, HeapProblem, HeapType, HostMemory, Invalid, Origin, Problem,
    RebindProblem, ResourceOptions, RuntimeGap, SharedEvent, Size, Texture, TextureDescriptor,
};
use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

const CALL: &str = "rebind_bytes_no_copy";

fn refused(problem: RebindProblem) -> Error {
    Error::Invalid(Invalid::new(CALL, Problem::Rebind(problem)))
}

/// `len` bytes of fresh host memory holding `counting(len, seed)`.
fn host(len: usize, seed: usize) -> Mapping {
    let mapping = Mapping::anonymous(len);
    mapping.write(0, &counting(len, seed));
    mapping
}

/// The `len` bytes at `pointer`, which the test's memory still holds.
///
/// # Safety
///
/// `len` bytes at `pointer` are mapped and no command writes them now.
unsafe fn at(pointer: NonNull<u8>, range: std::ops::Range<usize>) -> Vec<u8> {
    // SAFETY: the caller's contract.
    unsafe { std::slice::from_raw_parts(pointer.as_ptr().add(range.start), range.len()) }.to_vec()
}

/// The buffers of these tests: two guest pages, as the consumer's task memory.
const LEN: usize = 2 * GUEST_PAGE;

/// A no-copy buffer of `LEN` bytes over `mapping`, whose deallocator records into `seen` and
/// releases the mapping; `None`, printed, where the device imports no host memory.
fn buffer_over(device: &Device, mapping: Mapping, seen: &Arc<Seen>) -> Option<Buffer> {
    let pointer = mapping.pointer();
    // SAFETY: the deallocator releases the mapping, after the buffer's last use of it.
    let made = unsafe {
        no_copy(
            device,
            pointer,
            LEN,
            ResourceOptions::default(),
            Some(recording(seen, mapping)),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(error) if not_importable(&error) => {
            eprintln!("REBIND: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// Rebind `buffer` to `mapping`, whose deallocator records into `seen` and releases it.
pub(super) fn rebind(buffer: &Buffer, mapping: Mapping, seen: &Arc<Seen>) -> Result<(), Error> {
    let pointer = mapping.pointer();
    // SAFETY: the mapping holds the buffer's length until the deallocator releases it, and no
    // other thread encodes or commits meanwhile.
    unsafe {
        buffer.rebind_bytes_no_copy(
            HostMemory::new(pointer),
            Some(recording(seen, mapping)),
            None,
        )
    }
}

/// Rebind `heap` to `mapping`, whose deallocator records into `seen` and releases it.
pub(super) fn rebind_heap(heap: &Heap, mapping: Mapping, seen: &Arc<Seen>) -> Result<(), Error> {
    let pointer = mapping.pointer();
    // SAFETY: as `rebind`'s, for the heap's size.
    unsafe {
        heap.rebind_bytes_no_copy(
            HostMemory::new(pointer),
            Some(recording(seen, mapping)),
            None,
        )
    }
}

/// Where a deallocator ran.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Thread {
    /// On this thread: in the call that let go of the pages last.
    Here,
    /// Either: the pages' last command buffer completed before the call.
    Any,
}

/// Check that `seen`'s deallocator ran once, for `len` bytes at `pointer`, on `thread`, seeing
/// `first` as its first 16 bytes.
#[track_caller]
fn ran(seen: &Seen, pointer: NonNull<u8>, len: usize, first: &[u8], thread: Thread) {
    assert_eq!(
        seen.runs.load(Ordering::SeqCst),
        1,
        "the deallocator's runs"
    );
    assert_eq!(seen.dropped.load(Ordering::SeqCst), 0);
    assert_eq!(
        *seen.args.lock().unwrap(),
        Some((pointer.as_ptr() as usize, len as u64))
    );
    assert_eq!(*seen.bytes.lock().unwrap(), first);
    let here = *seen.thread.lock().unwrap() == Some(std::thread::current().id());
    match thread {
        Thread::Here => assert!(here, "the deallocator ran off the call's thread"),
        Thread::Any => {}
    }
}

/// Check that a refused call dropped `seen`'s deallocator unrun.
fn dropped_unrun(seen: &Seen) {
    assert_eq!(seen.runs.load(Ordering::SeqCst), 0);
    assert_eq!(seen.dropped.load(Ordering::SeqCst), 1);
}

/// A rebound buffer keeps its `gpu_address` and length, answers the new pointer as `contents`,
/// and the GPU reads and writes the new pages; no command buffer held the old ones, so their
/// deallocator ran in the call, seeing the GPU's last write, and the new one runs when the
/// buffer goes.
#[test]
fn a_rebound_buffer_keeps_its_address_and_takes_the_new_pages() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let (old_seen, new_seen) = (Arc::new(Seen::default()), Arc::new(Seen::default()));
        let old = host(LEN, 3);
        let old_pointer = old.pointer();
        let Some(buffer) = buffer_over(device, old, &old_seen) else {
            return;
        };
        let address = buffer.gpu_address();
        assert_eq!(gpu_read(&queue, &buffer, 0..LEN as u64), counting(LEN, 3));
        gpu_fill(&queue, &buffer, 0..16, 0x6b);

        let new = host(LEN, 9);
        let new_pointer = new.pointer();
        rebind(&buffer, new, &new_seen).unwrap();
        ran(&old_seen, old_pointer, LEN, &[0x6b; 16], Thread::Any);
        assert_eq!(buffer.gpu_address(), address);
        assert_eq!(buffer.contents(), Some(new_pointer));
        assert_eq!(buffer.length(), LEN as u64);
        assert_eq!(gpu_read(&queue, &buffer, 0..LEN as u64), counting(LEN, 9));
        gpu_fill(&queue, &buffer, 100..4200, 0xc3);
        // SAFETY: the new mapping lives until the buffer goes; no command writes it now.
        let now = unsafe { at(new_pointer, 0..LEN) };
        assert_eq!(&now[..100], &counting(LEN, 9)[..100]);
        assert!(now[100..4200].iter().all(|&byte| byte == 0xc3));
        assert_eq!(&now[4200..], &counting(LEN, 9)[4200..]);
        assert_eq!(new_seen.runs.load(Ordering::SeqCst), 0);
        drop(buffer);
        ran(&new_seen, new_pointer, LEN, &counting(16, 9), Thread::Here);
        eprintln!("REBIND buffer: {LEN} bytes kept {address:?}");
    });
}

/// Two no-copy buffers over one memory lie in one import: rebinding one moves it alone. It reads
/// and writes the new pages from then on at its address; the other keeps the memory, its pages and
/// its address; the moved buffer's deallocator of the memory runs at the rebind while the other
/// lives on in the import, and the other's when it goes. A third buffer made over the memory then
/// reads what the other wrote.
#[test]
fn rebinding_one_of_two_buffers_over_one_memory_moves_it_alone() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let (moved_seen, kept_seen, new_seen) = (
            Arc::new(Seen::default()),
            Arc::new(Seen::default()),
            Arc::new(Seen::default()),
        );
        let old = host(LEN, 3);
        let old_pointer = old.pointer();
        let over_old = |seen: Option<&Arc<Seen>>| {
            // SAFETY: the test holds the mapping until every buffer over it is gone.
            unsafe {
                no_copy(
                    device,
                    old_pointer,
                    LEN,
                    ResourceOptions::default(),
                    seen.map(|seen| recording(seen, ())),
                )
            }
        };
        let moved = match over_old(Some(&moved_seen)) {
            Err(error) if not_importable(&error) => {
                eprintln!("REBIND: the device imports no host memory: {error:?}");
                return;
            }
            moved => moved.unwrap(),
        };
        let kept = over_old(Some(&kept_seen)).unwrap();
        let (moved_address, kept_address) = (moved.gpu_address(), kept.gpu_address());
        gpu_fill(&queue, &moved, 0..16, 0x6b);
        assert_eq!(gpu_read(&queue, &kept, 0..16), [0x6b; 16]);

        let new = host(LEN, 9);
        let new_pointer = new.pointer();
        rebind(&moved, new, &new_seen).unwrap();
        ran(&moved_seen, old_pointer, LEN, &[0x6b; 16], Thread::Any);
        assert_eq!(kept_seen.runs.load(Ordering::SeqCst), 0);
        assert_eq!(moved.gpu_address(), moved_address);
        assert_eq!(kept.gpu_address(), kept_address);
        assert_eq!(moved.contents(), Some(new_pointer));
        assert_eq!(kept.contents(), Some(old_pointer));
        gpu_fill(&queue, &moved, 0..16, 0x11);
        gpu_fill(&queue, &kept, 16..32, 0x22);
        // SAFETY: the new mapping lives until the moved buffer goes; no command writes it now.
        let now_new = unsafe { at(new_pointer, 0..32) };
        assert_eq!(&now_new[..16], &[0x11; 16]);
        assert_eq!(&now_new[16..], &counting(LEN, 9)[16..32]);
        let now_old = old.read(0..32);
        assert_eq!(&now_old[..16], &[0x6b; 16]);
        assert_eq!(&now_old[16..], &[0x22; 16]);
        assert_eq!(gpu_read(&queue, &moved, 0..32), now_new);
        assert_eq!(gpu_read(&queue, &kept, 0..32), now_old);
        let third = over_old(None).unwrap();
        assert_eq!(gpu_read(&queue, &third, 0..32), now_old);
        drop(third);
        drop(kept);
        ran(&kept_seen, old_pointer, LEN, &[0x6b; 16], Thread::Here);
        drop(moved);
        ran(&new_seen, new_pointer, LEN, &[0x11; 16], Thread::Here);
        drop(old);
        eprintln!("REBIND one of two: {moved_address:?} moved, {kept_address:?} kept");
    });
}

/// A rebind onto the bytes a live no-copy buffer was made over lies in that buffer's import, as a
/// buffer made over them does: it asks the kernel about none of their mappings
/// (`WorkCounts::host_mappings_queried`, which Linux's imports count) and imports nothing, and the
/// GPU reads and writes those bytes through either buffer. Rebound back onto bytes no live buffer
/// was made over, it imports them, asking about their mapping, and reads them.
#[test]
fn a_rebind_onto_a_live_buffer_s_bytes_lies_in_its_import() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let (live_seen, moved_seen, onto_seen, back_seen) = (
            Arc::new(Seen::default()),
            Arc::new(Seen::default()),
            Arc::new(Seen::default()),
            Arc::new(Seen::default()),
        );
        let queried = || device.work_counts().host_mappings_queried;
        let shared = host(LEN, 5);
        let shared_pointer = shared.pointer();
        let Some(live) = buffer_over(device, shared, &live_seen) else {
            return;
        };
        let Some(moved) = buffer_over(device, host(LEN, 6), &moved_seen) else {
            return;
        };
        let before = queried();
        // SAFETY: the shared mapping lives until the live buffer goes, after the moved one leaves
        // it; no other thread encodes or commits meanwhile.
        unsafe {
            moved.rebind_bytes_no_copy(
                HostMemory::new(shared_pointer),
                Some(recording(&onto_seen, ())),
                None,
            )
        }
        .unwrap();
        assert_eq!(queried() - before, 0, "mappings asked for the live import");
        assert_eq!(moved.contents(), Some(shared_pointer));
        assert_eq!(gpu_read(&queue, &moved, 0..64), counting(LEN, 5)[..64]);
        gpu_fill(&queue, &moved, 0..16, 0x5d);
        gpu_fill(&queue, &live, 16..32, 0x2e);
        let both = [[0x5d; 16], [0x2e; 16]].concat();
        // SAFETY: the shared mapping lives; no command writes it now.
        assert_eq!(unsafe { at(shared_pointer, 0..32) }, both);
        assert_eq!(gpu_read(&queue, &live, 0..32), both);
        assert_eq!(gpu_read(&queue, &moved, 0..32), both);
        // Back onto fresh bytes: imported, their mapping asked about where Linux counts it.
        let back = host(LEN, 7);
        let before = queried();
        rebind(&moved, back, &back_seen).unwrap();
        let asked = queried() - before;
        assert_eq!(
            asked > 0,
            cfg!(target_os = "linux"),
            "{asked} mappings asked"
        );
        assert_eq!(gpu_read(&queue, &moved, 0..64), counting(LEN, 7)[..64]);
        ran(&onto_seen, shared_pointer, LEN, &both[..16], Thread::Any);
        drop(moved);
        drop(live);
        eprintln!("REBIND onto a live import: no mapping asked; back: {asked} asked");
    });
}

/// The consumer's task memory, rebound: a public kernel case runs over no-copy buffers, which are
/// then rebound to new pages holding the case's inputs; the case run again reads them and writes
/// its output into the new pages, matching Metal's output, at the same `gpu_address`es.
#[test]
fn a_kernel_runs_over_rebound_task_memory() {
    on_device(|device| {
        let case =
            public_case("saturating-halving-and-wide-product-integers-at-their-edges").unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let mut buffers = BTreeMap::new();
        let mut seen = Vec::new();
        for &index in case.compute.buffers.keys() {
            let first = Arc::new(Seen::default());
            let Some(buffer) = buffer_over(device, host(LEN, index as usize), &first) else {
                return;
            };
            seen.push(first);
            buffers.insert(index, buffer);
        }
        let dispatch = |buffers: &BTreeMap<u32, Buffer>| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encode_case(&mut encoder, &pipeline, &case.compute, buffers, &resources).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        // The kernel's first run, over pages of other bytes.
        dispatch(&buffers);
        let addresses: Vec<_> = buffers.values().map(Buffer::gpu_address).collect();
        let mut pointers = BTreeMap::new();
        for (&index, buffer) in &buffers {
            let mapping = Mapping::anonymous(LEN);
            mapping.write(0, &case.compute.buffers[&index]);
            pointers.insert(index, mapping.pointer());
            let next = Arc::new(Seen::default());
            rebind(buffer, mapping, &next).unwrap();
            seen.push(next);
        }
        dispatch(&buffers);
        let after: Vec<_> = buffers.values().map(Buffer::gpu_address).collect();
        assert_eq!(after, addresses);
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = case.compute.output
        else {
            panic!("an output {:?}", case.compute.output);
        };
        let start = offset as usize;
        // SAFETY: the output's mapping lives until its buffer goes; no command writes it now.
        let output = unsafe { at(pointers[&binding], start..start + length as usize) };
        assert!(
            output == case.metal_output,
            "the output differs from Metal's:\n runtime {output:02x?}\n metal   {:02x?}",
            case.metal_output
        );
        let first_runs: usize = seen[..buffers.len()]
            .iter()
            .map(|seen| seen.runs.load(Ordering::SeqCst))
            .sum();
        assert_eq!(first_runs, buffers.len(), "the first pages' deallocators");
        drop(buffers);
        for seen in &seen {
            assert_eq!(seen.runs.load(Ordering::SeqCst), 1);
            assert_eq!(seen.dropped.load(Ordering::SeqCst), 0);
        }
    });
}

/// A command buffer that fills `range` of `buffer` with `value`, waiting first for `event` to
/// reach 1 where given, committed; with a completed handler that notes `seen`'s runs then.
fn committed_fill(
    queue: &CommandQueue,
    (buffer, range): (&Buffer, std::ops::Range<u64>),
    value: u8,
    wait: Option<&SharedEvent>,
    seen: &Arc<Seen>,
) -> (CommandBuffer, Arc<AtomicUsize>) {
    let command_buffer = queue.command_buffer().unwrap();
    if let Some(event) = wait {
        command_buffer
            .encode_wait_for_event(EventRef::from(event), 1)
            .unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, range, value).unwrap();
    blit.end_encoding().unwrap();
    let in_handler = Arc::new(AtomicUsize::new(usize::MAX));
    {
        let (seen, in_handler) = (Arc::clone(seen), Arc::clone(&in_handler));
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                in_handler.store(seen.runs.load(Ordering::SeqCst), Ordering::SeqCst);
            }))
            .unwrap();
    }
    command_buffer.commit().unwrap();
    (command_buffer, in_handler)
}

/// Rebinds between command buffers committed and not yet submitted: a command buffer first
/// submitted after a rebind of a buffer its blit names writes the pages the buffer lies in then,
/// as Metal's does (the rebind probe's table 2, case 5), the rebind encoding its blit encoder
/// again (`replay.rs`; it once failed `EncodedUseRebound`): the first, committed before the first
/// rebind and held back by an event, encoded again at both rebinds, and the second, encoded between
/// the rebinds and queued behind it, encoded again at the second. The third, encoded after both,
/// writes the third pages beside them. A command buffer not yet submitted holds no pages: each of
/// the first two ranges' deallocators runs in the rebind that let it go, seeing no write, and
/// before either command buffer's completed handler. The buffer's `gpu_address` stays.
#[test]
fn rebinds_before_the_first_submission_reach_the_pages_of_the_last() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let seen: Vec<Arc<Seen>> = (0..3).map(|_| Arc::default()).collect();
        let first = host(LEN, 1);
        let pointers = [first.pointer()];
        let Some(buffer) = buffer_over(device, first, &seen[0]) else {
            return;
        };
        let address = buffer.gpu_address();
        let event = device.new_shared_event().unwrap();
        let before = device.work_counts();
        let (one, one_handler) =
            committed_fill(&queue, (&buffer, 0..16), 0x11, Some(&event), &seen[0]);

        let second = host(LEN, 2);
        let pointers = [pointers[0], second.pointer()];
        rebind(&buffer, second, &seen[1]).unwrap();
        ran(&seen[0], pointers[0], LEN, &counting(16, 1), Thread::Here);
        assert_eq!(buffer.contents(), Some(pointers[1]));
        let (two, two_handler) = committed_fill(&queue, (&buffer, 16..32), 0x22, None, &seen[1]);

        let third = host(LEN, 3);
        let pointers = [pointers[0], pointers[1], third.pointer()];
        rebind(&buffer, third, &seen[2]).unwrap();
        ran(&seen[1], pointers[1], LEN, &counting(16, 2), Thread::Here);
        let (three, _) = committed_fill(&queue, (&buffer, 32..48), 0x33, None, &seen[2]);
        assert_eq!(buffer.gpu_address(), address);

        assert_eq!(one.status(), CommandBufferStatus::Committed);
        event.set_signaled_value(1);
        for command_buffer in [&one, &two, &three] {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        for (name, wrote) in [("one", &one), ("two", &two), ("three", &three)] {
            assert_eq!(wrote.status(), CommandBufferStatus::Completed, "{name}");
            assert_eq!(wrote.error(), None, "{name}");
        }
        assert_eq!(one_handler.load(Ordering::SeqCst), 1);
        assert_eq!(two_handler.load(Ordering::SeqCst), 1);
        assert_eq!(seen[2].runs.load(Ordering::SeqCst), 0);
        // SAFETY: the third mapping lives until the buffer goes; no command writes it now.
        let now = unsafe { at(pointers[2], 0..64) };
        assert_eq!(&now[..16], &[0x11; 16], "one, encoded again twice");
        assert_eq!(&now[16..32], &[0x22; 16], "two, encoded again");
        assert_eq!(&now[32..48], &[0x33; 16], "three");
        assert_eq!(&now[48..], &counting(64, 3)[48..]);
        let after = device.work_counts();
        assert_eq!(after.encoders_reencoded - before.encoders_reencoded, 3);
        assert_eq!(after.reencodes_refused, before.reencodes_refused);
        drop(buffer);
        ran(&seen[2], pointers[2], LEN, &now[..16], Thread::Here);
        eprintln!(
            "CBF-MARK {}: two command buffers rebound before their first submission wrote the \
             last pages, encoded again 3 times",
            device.name()
        );
    });
}

/// A rebind while uncommitted work used the buffer is refused `EncodedUses`, dropping its
/// deallocator unrun and leaving the buffer where it was: an open encoder that used it, then the
/// ended encoder of a command buffer not yet committed. Once that command buffer is committed,
/// the rebind is made; a command buffer that used the buffer and was dropped uncommitted refuses
/// none.
#[test]
fn a_rebind_after_uncommitted_uses_is_refused() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let first_seen = Arc::new(Seen::default());
        let first = host(LEN, 1);
        let first_pointer = first.pointer();
        let Some(buffer) = buffer_over(device, first, &first_seen) else {
            return;
        };
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..16, 0x5e).unwrap();
        let open = Arc::new(Seen::default());
        assert_eq!(
            rebind(&buffer, host(LEN, 2), &open),
            Err(refused(RebindProblem::EncodedUses))
        );
        dropped_unrun(&open);
        blit.end_encoding().unwrap();
        let ended = Arc::new(Seen::default());
        assert_eq!(
            rebind(&buffer, host(LEN, 2), &ended),
            Err(refused(RebindProblem::EncodedUses))
        );
        dropped_unrun(&ended);
        assert_eq!(buffer.contents(), Some(first_pointer));
        assert_eq!(first_seen.runs.load(Ordering::SeqCst), 0);
        run(&command_buffer);
        let second_seen = Arc::new(Seen::default());
        let second = host(LEN, 2);
        let second_pointer = second.pointer();
        rebind(&buffer, second, &second_seen).unwrap();
        ran(&first_seen, first_pointer, LEN, &[0x5e; 16], Thread::Any);

        let dropped = queue.command_buffer().unwrap();
        let mut blit = dropped.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..16, 0x77).unwrap();
        blit.end_encoding().unwrap();
        drop(dropped);
        let third_seen = Arc::new(Seen::default());
        rebind(&buffer, host(LEN, 3), &third_seen).unwrap();
        ran(
            &second_seen,
            second_pointer,
            LEN,
            &counting(16, 2),
            Thread::Here,
        );
        drop(buffer);
        assert_eq!(third_seen.runs.load(Ordering::SeqCst), 1);
    });
}

/// Only memory the caller gave is rebound: a buffer of the runtime's memory, a buffer of a
/// no-copy heap and a heap of the runtime's memory are refused `NotHostMemory`; a no-copy heap
/// given memory off the import's unit `HostMemoryAlignment`. Each refused call drops its
/// deallocator unrun.
#[test]
fn only_the_caller_s_memory_is_rebound() {
    on_device(|device| {
        let ordinary = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let seen = Arc::new(Seen::default());
        assert_eq!(
            rebind(&ordinary, host(LEN, 1), &seen),
            Err(refused(RebindProblem::NotHostMemory))
        );
        dropped_unrun(&seen);
        let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP as u64, shared());
        let heap = device.new_heap(&descriptor).unwrap();
        let seen = Arc::new(Seen::default());
        assert_eq!(
            rebind_heap(&heap, host(HEAP, 1), &seen),
            Err(refused(RebindProblem::NotHostMemory))
        );
        dropped_unrun(&seen);

        let mapping = host(HEAP, 1);
        // SAFETY: the mapping outlives the heap, dropped at the end of the test after it.
        let over = unsafe { no_copy_heap(device, mapping.pointer(), HeapType::Placement, None) };
        let Some(over) = over else {
            return;
        };
        let placed = over.new_buffer_at_offset(4096, shared(), 0).unwrap();
        let seen = Arc::new(Seen::default());
        assert_eq!(
            rebind(&placed, host(LEN, 1), &seen),
            Err(refused(RebindProblem::NotHostMemory))
        );
        dropped_unrun(&seen);
        // Memory 256 bytes past a page.
        let off_unit = Mapping::anonymous(HEAP + 4 * host_page());
        // SAFETY: 256 bytes into the mapping, which holds the heap's bytes past it.
        let off = unsafe { NonNull::new_unchecked(off_unit.pointer().as_ptr().add(256)) };
        let seen = Arc::new(Seen::default());
        let count = Arc::clone(&seen);
        let deallocator: Deallocator = Box::new(move |_, _| {
            count.runs.fetch_add(1, Ordering::SeqCst);
        });
        // SAFETY: the mapping holds the heap's size past `off` and outlives any rebind.
        let made =
            unsafe { over.rebind_bytes_no_copy(HostMemory::new(off), Some(deallocator), None) };
        let Err(Error::Invalid(invalid)) = &made else {
            panic!("an unaligned rebind: {made:?}");
        };
        assert!(
            matches!(
                invalid.problem,
                Problem::Heap(HeapProblem::HostMemoryAlignment { .. })
            ),
            "{made:?}"
        );
        assert_eq!(seen.runs.load(Ordering::SeqCst), 0);
        assert_eq!(placed.contents(), Some(mapping.pointer()));
        drop((placed, over, off_unit));
        drop(mapping);
    });
}

/// The rows of a `side`-square texture of `bytes`-byte texels whose rows lie `pitch` bytes
/// apart in `all`.
fn rows(all: &[u8], side: usize, bytes: usize, pitch: usize) -> Vec<u8> {
    (0..side)
        .flat_map(|row| &all[row * pitch..row * pitch + side * bytes])
        .copied()
        .collect()
}

/// A 2D texture over a no-copy buffer (its linear image aliasing the pages where the device lays
/// its rows as Metal does, made again over the new pages by the rebind; else a shadow of its bytes)
/// reads the new pages after a rebind, and its GPU writes land there; given memory at another
/// offset in its page, the rebind gives the buffer a copy of the new bytes laid out at the old
/// offset (`copied-host-pages`), which the texture reads, that memory's deallocator running at the
/// next rebind. The texture's form is printed.
#[test]
fn a_texture_over_a_rebound_buffer_reads_the_new_pages() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let Some(buffer) = buffer_over(device, host(LEN, 1), &Arc::default()) else {
            return;
        };
        let (side, bytes, pitch) = (16usize, 4usize, 256usize);
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side as u64, side as u64, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let texture = match buffer.new_texture(&descriptor, 0, pitch as u64) {
            Ok(texture) => texture,
            Err(error) => {
                eprintln!("REBIND: no texture over a no-copy buffer: {error:?}");
                return;
            }
        };
        // SAFETY: the handle is only compared, while the texture lives.
        let aliased = unsafe { texture.vk_image() }.is_some();
        eprintln!(
            "REBIND texture over a no-copy buffer: {}",
            if aliased { "aliased" } else { "shadowed" }
        );
        let read = |texture: &Texture| {
            blit_texels(&queue, texture, (side as u64, side as u64), bytes as u64)
        };
        assert_eq!(read(&texture), rows(&counting(LEN, 1), side, bytes, pitch));

        // Memory 256 bytes into a page: a copy of it, laid out at the old offset.
        let off_page = Mapping::anonymous(LEN + host_page());
        off_page.write(256, &counting(LEN, 3));
        // SAFETY: 256 bytes into the mapping, which holds the buffer's length past it.
        let off = unsafe { NonNull::new_unchecked(off_page.pointer().as_ptr().add(256)) };
        let seen = Arc::new(Seen::default());
        let count = DropRecord(Arc::clone(&seen));
        let deallocator: Deallocator = Box::new(move |_, _| drop(count));
        let before = device.work_counts();
        // SAFETY: the mapping holds the length past `off` and outlives the next rebind.
        let made =
            unsafe { buffer.rebind_bytes_no_copy(HostMemory::new(off), Some(deallocator), None) };
        assert_eq!(made, Ok(()));
        let after = device.work_counts();
        assert_eq!(
            after.copied_rebind_pages_moved - before.copied_rebind_pages_moved,
            1,
            "a copy of the moved bytes"
        );
        assert_eq!(buffer.contents(), Some(off));
        assert_eq!(read(&texture), rows(&counting(LEN, 3), side, bytes, pitch));
        assert_eq!(seen.dropped.load(Ordering::SeqCst), 0);

        let next = host(LEN, 2);
        let pointer = next.pointer();
        rebind(&buffer, next, &Arc::default()).unwrap();
        assert_eq!(seen.dropped.load(Ordering::SeqCst), 1);
        drop(off_page);
        assert_eq!(read(&texture), rows(&counting(LEN, 2), side, bytes, pitch));
        let written: Vec<u8> = (0..side * side * bytes).map(|i| (i % 199) as u8).collect();
        blit_into(
            &queue,
            &texture,
            (side as u64, side as u64),
            bytes as u64,
            &written,
        );
        // SAFETY: the new mapping lives until the buffer goes; no command writes it now.
        let now = unsafe { at(pointer, 0..side * pitch) };
        assert_eq!(rows(&now, side, bytes, pitch), written);
    });
}

/// A 2D texture over a buffer of a no-copy heap follows a rebind of the heap as one over a
/// no-copy buffer does: where its linear image aliases the heap's bytes, the rebind makes it
/// again over the new pages at its place; else it is a shadow whose copies read them. It reads
/// the texels the new pages hold, laid out as before, and the GPU's next write lands in the new
/// pages, not in the old ones. The texture's form is printed.
#[test]
fn a_texture_over_a_no_copy_heap_s_buffer_follows_a_rebind() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let first = host(HEAP, 1);
        // SAFETY: the test keeps the mapping until the heap is gone.
        let heap = unsafe { no_copy_heap(device, first.pointer(), HeapType::Placement, None) };
        let Some(heap) = heap else {
            return;
        };
        let offset = 64usize << 10;
        let buffer = heap
            .new_buffer_at_offset(64 << 10, shared(), offset as u64)
            .unwrap();
        let (side, bytes, pitch) = (16usize, 4usize, 256usize);
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side as u64, side as u64, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&descriptor, 0, pitch as u64).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        let aliased = unsafe { texture.vk_image() }.is_some();
        let read = |texture: &Texture| {
            blit_texels(&queue, texture, (side as u64, side as u64), bytes as u64)
        };
        assert_eq!(
            read(&texture),
            rows(&counting(HEAP, 1)[offset..], side, bytes, pitch)
        );
        let next = host(HEAP, 2);
        let next_pointer = next.pointer();
        rebind_heap(&heap, next, &Arc::default()).unwrap();
        first.fill(0);
        assert_eq!(
            read(&texture),
            rows(&counting(HEAP, 2)[offset..], side, bytes, pitch),
            "the texels of the new pages"
        );
        let written: Vec<u8> = (0..side * side * bytes).map(|i| (i % 199) as u8).collect();
        blit_into(
            &queue,
            &texture,
            (side as u64, side as u64),
            bytes as u64,
            &written,
        );
        // SAFETY: the new mapping lives until the heap goes; no command writes it now.
        let now = unsafe { at(next_pointer, offset..offset + side * pitch) };
        assert_eq!(rows(&now, side, bytes, pitch), written);
        assert!(
            first
                .read(offset..offset + side * pitch)
                .iter()
                .all(|&byte| byte == 0),
            "the GPU's write reached the old pages"
        );
        eprintln!(
            "REBIND texture over a no-copy heap's buffer: {}",
            if aliased { "aliased" } else { "shadowed" }
        );
        drop((texture, buffer, heap));
        drop(first);
    });
}

/// Counts its drop into its `Seen`'s `dropped` (a deallocator dropped unrun).
struct DropRecord(Arc<Seen>);

impl Drop for DropRecord {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

/// A case that writes four bytes of a storage texel buffer, its output.
const TEXEL_WRITER: &str = "write-four-bytes-into-a-storage-texel-buffer";

/// A texture buffer over a no-copy buffer views the new pages after a rebind: a public case that
/// writes four bytes of a storage texel buffer, run with the texture buffer bound before and
/// after the rebind, writes the case's output into the pages the buffer lies in each time.
#[test]
fn a_texture_buffer_over_a_rebound_buffer_views_the_new_pages() {
    on_device(|device| {
        let case = public_case(TEXEL_WRITER).unwrap();
        let written = run_case(device, &case).unwrap().output;
        let ComputeOutput::Texture { binding: index, .. } = case.compute.output else {
            panic!("{TEXEL_WRITER}: the case's output is no texture");
        };
        let literal = &case.compute.textures[&index];
        let descriptor = texture_descriptor(literal, case.texture_usage[&index]).unwrap();
        assert_eq!(descriptor.texture_type, TextureType::TextureBuffer);
        let length = literal.bytes.len();
        let first = Mapping::anonymous(LEN);
        first.write(0, &literal.bytes);
        let first_pointer = first.pointer();
        let Some(buffer) = buffer_over(device, first, &Arc::default()) else {
            return;
        };
        let texels = buffer
            .new_texture(&descriptor, 0, length.next_multiple_of(256) as u64)
            .unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let run_over = |texels: &Texture| {
            let mut resources = CaseResources::new(device, &case).unwrap();
            resources.textures.insert(index, texels.clone());
            let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encode_case(&mut encoder, &pipeline, &case.compute, &buffers, &resources).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        run_over(&texels);
        // SAFETY: the first mapping lives until the rebind's deallocator runs, below.
        assert_eq!(unsafe { at(first_pointer, 0..length) }, written);

        let next = Mapping::anonymous(LEN);
        next.write(0, &literal.bytes);
        let pointer = next.pointer();
        rebind(&buffer, next, &Arc::default()).unwrap();
        run_over(&texels);
        // SAFETY: the new mapping lives until the buffer goes; no command writes it now.
        assert_eq!(unsafe { at(pointer, 0..length) }, written);
    });
}

/// The descriptor of a heap texture of `format`, `side` square: a depth texture read and
/// rendered to, a colour one read and written.
fn heap_descriptor(format: PixelFormat, side: u64) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(format, side, side, false);
    descriptor.resource_options = shared();
    descriptor.usage = if metal2vulkan::metal::format::is_depth_stencil(format) {
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET
    } else {
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE
    };
    descriptor
}

/// A texture of [`heap_descriptor`] at `offset` of `heap`: `None`, printed, where the heap
/// refuses it `NotImplemented(HeapTextureMemory)`.
fn heap_texture(heap: &Heap, format: PixelFormat, side: u64, offset: u64) -> Option<Texture> {
    match heap.new_texture_at_offset(&heap_descriptor(format, side), offset) {
        Ok(texture) => Some(texture),
        Err(error @ Error::NotImplemented(RuntimeGap::HeapTextureMemory)) => {
            eprintln!("REBIND heap {format:?}: refused {error:?}");
            None
        }
        Err(error) => panic!("{format:?}: {error:?}"),
    }
}

/// A rebound no-copy heap keeps its buffers and textures: their `gpu_address`es and resource
/// IDs stay, a buffer's `contents` follows its offset into the new pages, and the GPU reads the
/// new bytes through the buffers. A depth texture (a shadow of the heap's bytes on every device
/// measured) and a colour texture (bound to them there), each written by the GPU before the
/// rebind, read after it the texels the new pages hold, given the old pages' bytes over the
/// texture laid out as they were, while the old pages are cleared; the GPU's next write lands in
/// the new pages. A command buffer not yet committed that used a buffer of the heap refuses the
/// rebind `EncodedUses`. Each old range's deallocator runs in the rebind, none pending.
#[test]
fn a_rebound_heap_keeps_its_resources() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let mut mappings = vec![host(HEAP, 1)];
        let mut seen = vec![Arc::new(Seen::default())];
        let pointer = mappings[0].pointer();
        // SAFETY: the test keeps every mapping until the heap is gone.
        let heap = unsafe {
            no_copy_heap(
                device,
                pointer,
                HeapType::Placement,
                Some(recording(&seen[0], ())),
            )
        };
        let Some(heap) = heap else {
            return;
        };
        let a = heap.new_buffer_at_offset(64 << 10, shared(), 0).unwrap();
        let b = heap
            .new_buffer_at_offset(64 << 10, shared(), 128 << 10)
            .unwrap();
        let addresses = [heap.gpu_address(), a.gpu_address(), b.gpu_address()];
        gpu_fill(&queue, &a, 0..16, 0x6b);

        // A command buffer not yet committed that used a buffer of the heap.
        let pending = queue.command_buffer().unwrap();
        let mut blit = pending.blit_command_encoder().unwrap();
        blit.fill_buffer(&b, 0..16, 0x99).unwrap();
        blit.end_encoding().unwrap();
        let refused_seen = Arc::new(Seen::default());
        assert_eq!(
            rebind_heap(&heap, host(HEAP, 9), &refused_seen),
            Err(refused(RebindProblem::EncodedUses))
        );
        dropped_unrun(&refused_seen);
        drop(pending);

        for (format, side, bytes, offset) in [
            (PixelFormat::Depth16Unorm, 32u64, 2u64, 512usize << 10),
            (PixelFormat::RGBA8Unorm, 64, 4, 256 << 10),
        ] {
            let Some(texture) = heap_texture(&heap, format, side, offset as u64) else {
                continue;
            };
            let id = texture.gpu_resource_id();
            let len = (side * side * bytes) as usize;
            let first: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            blit_into(&queue, &texture, (side, side), bytes, &first);
            let size = device
                .heap_texture_size_and_align(&heap_descriptor(format, side))
                .unwrap()
                .size as usize;
            let range = offset..offset + size;
            let old = mappings.len() - 1;
            let laid = mappings[old].read(range.clone());
            // The new pages hold the old pages' bytes over the texture.
            let next = host(HEAP, mappings.len() + 1);
            next.write(offset, &laid);
            let next_seen = Arc::new(Seen::default());
            let next_pointer = next.pointer();
            // SAFETY: the test keeps the mapping until the heap is gone.
            unsafe {
                heap.rebind_bytes_no_copy(
                    HostMemory::new(next_pointer),
                    Some(recording(&next_seen, ())),
                    None,
                )
            }
            .unwrap_or_else(|error| panic!("{format:?}: {error:?}"));
            ran(
                &seen[old],
                mappings[old].pointer(),
                HEAP,
                &mappings[old].read(0..16),
                Thread::Here,
            );
            mappings.push(next);
            seen.push(next_seen);
            mappings[old].fill(0);
            assert_eq!(texture.gpu_resource_id(), id);
            assert_eq!(
                blit_texels(&queue, &texture, (side, side), bytes),
                first,
                "{format:?}: the texels of the new pages"
            );
            let second: Vec<u8> = (0..len).map(|i| (i % 241) as u8 ^ 0x5a).collect();
            blit_into(&queue, &texture, (side, side), bytes, &second);
            let current = mappings.len() - 1;
            assert_ne!(
                mappings[current].read(range.clone()),
                laid,
                "{format:?}: the GPU's write"
            );
            assert!(
                mappings[old].read(range).iter().all(|&byte| byte == 0),
                "{format:?}: the GPU's write reached the old pages"
            );
            eprintln!("REBIND heap {format:?}: rebound, the texels follow the new pages");
        }
        if mappings.len() == 1 {
            let next = host(HEAP, 2);
            let next_seen = Arc::new(Seen::default());
            // SAFETY: the test keeps the mapping until the heap is gone.
            unsafe {
                heap.rebind_bytes_no_copy(
                    HostMemory::new(next.pointer()),
                    Some(recording(&next_seen, ())),
                    None,
                )
            }
            .unwrap();
            mappings.push(next);
            seen.push(next_seen);
        }
        let current = mappings.len() - 1;
        let pointer = mappings[current].pointer();
        assert_eq!(
            [heap.gpu_address(), a.gpu_address(), b.gpu_address()],
            addresses
        );
        assert_eq!(a.contents(), Some(pointer));
        // SAFETY: 128 KiB into the heap's current pages.
        let b_at = unsafe { NonNull::new_unchecked(pointer.as_ptr().add(128 << 10)) };
        assert_eq!(b.contents(), Some(b_at));
        assert_eq!(
            gpu_read(&queue, &b, 0..64 << 10),
            mappings[current].read(128 << 10..192 << 10)
        );
        drop((a, b, heap));
        assert_eq!(seen[current].runs.load(Ordering::SeqCst), 1);
        drop(mappings);
    });
}

/// A case that samples three 2D textures through one sampler into a buffer.
const SAMPLED: &str = "sample-each-of-three-textures-through-one-dynamic-array-index";

/// A colour texture of a no-copy heap, sampled by a command buffer committed before a rebind and
/// held back by an event, samples the texture's image over the new pages, which hold the texels
/// laid out as the old did: the rebind encodes the compute encoder again (`replay.rs`; the command
/// buffer once failed `EncodedUseRebound`), as Metal's buffers reach the new pages (its textures
/// are not measured). The old pages are untouched: a command buffer not yet submitted holds none,
/// so the old range's deallocator runs in the rebind, before the command buffer's completed
/// handler. After the rebind the texture lies in the new pages: the texels copied into it land
/// there laid out as before, and the case reads them.
#[test]
fn a_texture_sampled_before_its_first_submission_reads_its_heap_s_new_pages() {
    on_device(|device| {
        let case = public_case(SAMPLED).unwrap();
        let expected = run_case(device, &case).unwrap().output;
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = case.compute.output
        else {
            panic!("{SAMPLED}: an output {:?}", case.compute.output);
        };
        let queue = device.new_command_queue().unwrap();
        let plain = CaseResources::new(device, &case).unwrap();
        let (&index, original) = plain.textures.iter().next().unwrap();
        let first = Mapping::anonymous(HEAP);
        let seen = [Arc::new(Seen::default()), Arc::new(Seen::default())];
        // SAFETY: the test keeps the mapping until the heap is gone.
        let heap = unsafe {
            no_copy_heap(
                device,
                first.pointer(),
                HeapType::Placement,
                Some(recording(&seen[0], ())),
            )
        };
        let Some(heap) = heap else {
            return;
        };
        let literal = &case.compute.textures[&index];
        let mut descriptor = texture_descriptor(literal, case.texture_usage[&index]).unwrap();
        descriptor.resource_options = shared();
        let texture = match heap.new_texture_at_offset(&descriptor, 0) {
            Ok(texture) => texture,
            Err(error @ Error::NotImplemented(RuntimeGap::HeapTextureMemory)) => {
                eprintln!("REBIND sampled: refused {error:?}");
                return;
            }
            Err(error) => panic!("{SAMPLED}: {error:?}"),
        };
        let copy_in = || {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let size = Size::new(original.width(), original.height(), 1);
            blit.copy_from_texture_to_texture(
                original,
                0,
                0,
                Origin::new(0, 0, 0),
                size,
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
            )
            .unwrap();
            blit.end_encoding().unwrap();
            run(&command_buffer);
        };
        copy_in();
        let size = device
            .heap_texture_size_and_align(&descriptor)
            .unwrap()
            .size as usize;
        let laid = first.read(0..size);
        let pipeline = pipeline_state(device, &case).unwrap();
        let mut resources = CaseResources::new(device, &case).unwrap();
        resources.textures.insert(index, texture.clone());
        let encode = |command_buffer: &CommandBuffer| {
            let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encode_case(&mut encoder, &pipeline, &case.compute, &buffers, &resources).unwrap();
            encoder.end_encoding().unwrap();
            buffers
        };
        let event = device.new_shared_event().unwrap();
        let held = queue.command_buffer().unwrap();
        held.encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let buffers = encode(&held);
        let in_handler = Arc::new(AtomicUsize::new(usize::MAX));
        {
            let (seen, in_handler) = (Arc::clone(&seen[0]), Arc::clone(&in_handler));
            held.add_completed_handler(Box::new(move |_: &CommandBuffer| {
                in_handler.store(seen.runs.load(Ordering::SeqCst), Ordering::SeqCst);
            }))
            .unwrap();
        }
        held.commit().unwrap();

        // The rebind, to pages holding the texels laid out as the old, while the command buffer
        // that samples the texture waits.
        let second = Mapping::anonymous(HEAP);
        second.write(0, &laid);
        let before = device.work_counts();
        // SAFETY: the test keeps the mapping until the heap is gone.
        unsafe {
            heap.rebind_bytes_no_copy(
                HostMemory::new(second.pointer()),
                Some(recording(&seen[1], ())),
                None,
            )
        }
        .unwrap();
        // Its last use, the copy in, completed before the rebind.
        ran(&seen[0], first.pointer(), HEAP, &laid[..16], Thread::Any);
        assert_eq!(held.status(), CommandBufferStatus::Committed);
        event.set_signaled_value(1);
        held.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(held.error(), None);
        assert_eq!(held.status(), CommandBufferStatus::Completed);
        assert_eq!(
            read_buffer(&buffers[&binding], offset, length).unwrap(),
            expected,
            "sampled from the new pages"
        );
        let after = device.work_counts();
        assert!(after.encoders_reencoded > before.encoders_reencoded);
        drop(buffers);
        assert_eq!(first.read(0..size), laid, "the old pages changed");
        assert_eq!(in_handler.load(Ordering::SeqCst), 1);

        // The texture lies in the new pages now.
        copy_in();
        assert_eq!(
            second.read(0..size),
            laid,
            "the texels copied in after the rebind"
        );
        let after = queue.command_buffer().unwrap();
        let buffers = encode(&after);
        run(&after);
        assert_eq!(
            read_buffer(&buffers[&binding], offset, length).unwrap(),
            expected
        );
        drop((resources, texture, heap));
        assert_eq!(seen[1].runs.load(Ordering::SeqCst), 1);
        eprintln!(
            "CBF-MARK {}: a heap texture sampled before its first submission read the new pages",
            device.name()
        );
        drop((first, second));
    });
}

/// What a heap's rebind takes: the import of the new pages, and a new image for each texture
/// bound to the heap's bytes, made and moved to `GENERAL` in one submission. `REBIND cost`
/// lines print the call's wall time with 0, 1, 8 and 32 `RGBA8Unorm` 64x64 textures bound
/// (three rebinds each; the lane's build). Nothing is asserted of the times.
#[test]
fn a_heap_rebind_costs_an_image_per_bound_texture() {
    on_device(|device| {
        let mut mappings = vec![Mapping::anonymous(HEAP)];
        // SAFETY: the test keeps every mapping until the heap is gone.
        let heap =
            unsafe { no_copy_heap(device, mappings[0].pointer(), HeapType::Placement, None) };
        let Some(heap) = heap else {
            return;
        };
        let descriptor = heap_descriptor(PixelFormat::RGBA8Unorm, 64);
        let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
        let step = answer.size.next_multiple_of(answer.align);
        let mut textures = Vec::new();
        for count in [0usize, 1, 8, 32] {
            while textures.len() < count {
                let at = textures.len() as u64 * step;
                let Some(texture) = heap_texture(&heap, PixelFormat::RGBA8Unorm, 64, at) else {
                    return;
                };
                textures.push(texture);
            }
            let mut times = Vec::new();
            for _ in 0..3 {
                let next = Mapping::anonymous(HEAP);
                let start = Instant::now();
                // SAFETY: the test keeps the mapping until the heap is gone.
                unsafe { heap.rebind_bytes_no_copy(HostMemory::new(next.pointer()), None, None) }
                    .unwrap();
                times.push(format!("{:.0}", start.elapsed().as_secs_f64() * 1e6));
                mappings.push(next);
            }
            eprintln!(
                "REBIND cost on {}: {count} bound textures: {} us",
                device.name(),
                times.join(", ")
            );
        }
        drop((textures, heap));
        drop(mappings);
    });
}
