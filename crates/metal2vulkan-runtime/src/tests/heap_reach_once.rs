//! A heap's buffers are listed among the device's live buffers (`live_buffers.rs`) at the first
//! use of the heap since its making or the last replacement of its pages, and each buffer placed
//! in it after that use at its placing (`Heap::reached`, `LiveBuffers::placed`): a later use of the
//! heap walks none of them. Each way a placed buffer comes to be listed, or listed again, is read
//! through an address loaded from memory, which only the live buffers' table resolves:
//!
//! - a buffer placed in a heap a command buffer used;
//! - a buffer placed at the offset a gone one left: the new one is listed, the gone one not, and
//!   the table of the list since shares the set and upload of the one before (the same set);
//! - a listed buffer's range read again (`LiveBuffers::refresh`, as a re-import reads it);
//! - the heap's pages replaced (`LiveBuffers::rebound` with the heap): none of its buffers is
//!   listed until a command buffer uses the heap again, which lists them all;
//! - a buffer placed after a command buffer's commit and before its first submission, which a
//!   shared event only the host signals holds back: the submission's table holds it.
//!
//! Each reading command buffer uses the heap (the indices the kernel reads lie in it), so a reach
//! that walked the heap's buffers again would list a buffer its placing failed to.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `ob2_loaded_pointer` of the
//! owned fixture `validation/fixtures/public/engine/ob2_loaded_pointer.ll` (`llvm-as`): thread `t`
//! of 8 reads word `idx[t]` through the address in its holder into `out[t]`, and writes through it
//! at word `idx[t + 8]`.

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    EventRef, Heap, HeapDescriptor, HeapType, PipelineOptions, ResourceOptions, Size,
};

/// A placed buffer's bytes: the 16 words the kernel reads and writes, and room.
const BYTES: u64 = 256;

/// The words the kernel reads.
const READ: usize = 8;

/// The 16 words a buffer placed for `seed` holds: none of them 0, which a dropped read reads.
fn words(seed: u32) -> Vec<u32> {
    (0..16).map(|at| (seed << 8) | at).collect()
}

/// Write `words` into `buffer`, a `Shared` one.
fn write(buffer: &Buffer, words: &[u32]) {
    let contents = buffer.contents().unwrap();
    // SAFETY: a `Shared` buffer of `BYTES` bytes at least, which no command uses meanwhile.
    unsafe {
        std::ptr::copy_nonoverlapping(words.as_ptr(), contents.as_ptr().cast::<u32>(), words.len());
    }
}

/// A buffer placed in `heap` holding [`words`] of `seed`.
fn placed(heap: &Heap, seed: u32) -> Buffer {
    let buffer = heap.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    write(&buffer, &words(seed));
    buffer
}

/// The kernel and its buffers: the indices in the heap, the holder of the address it reads
/// through, and the words it read.
struct Reader {
    state: ComputePipelineState,
    indices: Buffer,
    holder: Buffer,
    out: Buffer,
}

impl Reader {
    fn new(device: &Device, heap: &Heap) -> Reader {
        let function = library(device, "engine/ob2_loaded_pointer.ll")
            .new_function_with_name("ob2_loaded_pointer")
            .unwrap();
        let state = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let indices = heap.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
        write(&indices, &(0..16).collect::<Vec<u32>>());
        Reader {
            state,
            indices,
            holder: device.new_buffer(8, ResourceOptions::shared()).unwrap(),
            out: device
                .new_buffer(4 * READ as u64, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// Point the holder at `target` and clear what was read.
    fn point(&self, target: &Buffer) {
        let address = target.gpu_address().0;
        write(&self.holder, &[address as u32, (address >> 32) as u32]);
        write(&self.out, &[0; READ]);
    }

    /// Encode the dispatch into `command_buffer`: its use of the indices uses the heap.
    fn encode(&self, command_buffer: &CommandBuffer) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(&self.indices), 0, 0).unwrap();
        encoder.set_buffer(Some(&self.holder), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.out), 0, 2).unwrap();
        let threads = Size::new(READ as u64, 1, 1);
        encoder.dispatch_threads(threads, threads).unwrap();
        encoder.end_encoding().unwrap();
    }

    /// The words the dispatch read, once `command_buffer` completed.
    fn read(&self, command_buffer: &CommandBuffer) -> Vec<u32> {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let contents = self.out.contents().unwrap();
        // SAFETY: the output's own words, which its completed command buffer wrote.
        unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), READ) }.to_vec()
    }

    /// The words a command buffer of its own reads through `target`'s address.
    fn through(&self, queue: &CommandQueue, target: &Buffer) -> Vec<u32> {
        self.point(target);
        let command_buffer = queue.command_buffer().unwrap();
        self.encode(&command_buffer);
        command_buffer.commit().unwrap();
        self.read(&command_buffer)
    }
}

#[test]
fn a_heap_s_buffers_are_listed_at_its_first_use_and_each_later_one_at_its_placing() {
    let scratch = Scratch::new("heap-reach-once");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let descriptor = HeapDescriptor::new(HeapType::Automatic, 64 << 10, ResourceOptions::shared());
    let heap = device.new_heap(&descriptor).unwrap();
    let reader = Reader::new(&device, &heap);
    let live = &device.shared().live_buffers;

    // The heap's first use lists every buffer placed in it.
    let first = placed(&heap, 1);
    assert_eq!(reader.through(&queue, &first), words(1)[..READ]);
    assert!(
        heap.reached().is_empty(),
        "a later use of the heap walks none of its buffers"
    );

    // Placed in a heap a command buffer used: listed at its placing, as no later use lists it.
    let later = placed(&heap, 2);
    assert_eq!(
        reader.through(&queue, &later),
        words(2)[..READ],
        "a buffer placed in a used heap"
    );
    let key = (later.gpu_address().0, later.object_id());
    assert!(live.lists(key.0, key.1), "listed at its placing");

    // Placed at the offset a gone buffer left: the new one is listed, the gone one not. The table
    // made of the list then lists the set the table before the gone one went listed, and shares
    // that set and its upload, writing nothing.
    let before = live.table(&device).unwrap();
    let uploaded = device.work_counts().live_tables_uploaded;
    drop(later);
    let again = placed(&heap, 3);
    assert_eq!(
        again.gpu_address().0,
        key.0,
        "placed where the gone one lay"
    );
    assert_eq!(
        reader.through(&queue, &again),
        words(3)[..READ],
        "a buffer placed where a gone one lay"
    );
    assert!(!live.lists(key.0, key.1), "the gone buffer is not listed");
    assert!(live.lists(key.0, again.object_id()), "the new one is");
    let after = live.table(&device).unwrap();
    assert!(
        after.generation() > before.generation(),
        "a table of the list since"
    );
    assert!(after.shares_set(&before), "of the same set, shared");
    assert_eq!(
        device.work_counts().live_tables_uploaded,
        uploaded,
        "no set uploaded to a new buffer"
    );
    drop((before, after));

    // Its range read again, as a re-import of its pages reads it: still listed.
    live.refresh(&again);
    assert_eq!(
        reader.through(&queue, &again),
        words(3)[..READ],
        "a refreshed buffer"
    );
    assert!(
        live.lists(key.0, again.object_id()),
        "listed after a refresh"
    );

    // The heap's pages replaced: none of its buffers is listed until a command buffer uses the
    // heap again, which lists them all.
    live.rebound(heap.bytes().unwrap(), Some(&heap));
    assert!(
        !live.lists(key.0, again.object_id()),
        "unlisted by the rebind"
    );
    assert_eq!(
        reader.through(&queue, &again),
        words(3)[..READ],
        "a buffer of a heap used again after its pages were replaced"
    );
    assert!(live.lists(key.0, again.object_id()), "listed by the use");

    // Placed after a command buffer's commit and before its first submission, which a shared
    // event only the host signals holds back: the list changed since the commit, and the
    // submission's table holds the new buffer.
    let gate = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    reader.encode(&command_buffer);
    command_buffer.commit().unwrap();
    let late = placed(&heap, 4);
    reader.point(&late);
    gate.set_signaled_value(1);
    assert_eq!(
        reader.read(&command_buffer),
        words(4)[..READ],
        "a buffer placed after the commit, before the first submission"
    );
    drop((first, again, late));
}

/// A heap that buffers come and go in keeps its list of them within twice those that live, and
/// a few, pruning it only as it grows so: a placing costs the same however many the heap holds.
#[test]
fn a_heap_s_list_of_placed_buffers_stays_bounded_as_buffers_come_and_go() {
    let scratch = Scratch::new("heap-placed-bounded");
    let device = device(&scratch.0);
    let descriptor = HeapDescriptor::new(HeapType::Automatic, 64 << 10, ResourceOptions::shared());
    let heap = device.new_heap(&descriptor).unwrap();
    let kept: Vec<Buffer> = (0..4).map(|seed| placed(&heap, seed + 1)).collect();
    let mut most = 0;
    for seed in 0..200 {
        drop(placed(&heap, seed + 5));
        most = most.max(heap.placed_entries());
    }
    assert!(
        most <= 2 * (kept.len() + 1) + 8,
        "{most} entries for {} buffers that live",
        kept.len()
    );
}
