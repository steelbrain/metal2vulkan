//! A dispatch's hazard tracking does only the work its inputs since the encoder's last command
//! call for, with the answer the full work gives:
//!
//! - **An encoder's order** (`tracker::EncoderUses`): a serial compute encoder's first dispatch
//!   orders after nothing, and its order takes it in only once a later dispatch of the encoder
//!   orders after it (`WorkCounts::commands_ordered_inside`): encoders of one dispatch each take
//!   none in, and an encoder of two, whose second reads what its first wrote, takes both in and
//!   records one barrier between them, reading the first's words.
//! - **The host aliases' uses** (`host_aliases::Quiet`): a dispatch over buffers placed in a heap
//!   over host pages, beside a no-copy buffer over those pages (as the consumer places a guest's
//!   buffers in a heap over its RAM), works the uses of their host memory out from the keys the
//!   device's encoders found, with no lookup under the lock (`WorkCounts::host_alias_lookups`),
//!   while no buffer joins a memory, is placed, moves or goes; a no-copy buffer made over other
//!   pages between two encoders (placed over its bytes: `HostAliases::place`) moves that, and the
//!   next dispatch looks its keys up once again, then none. A buffer placed in the heap moves
//!   nothing: its bytes are the heap's.
//!   Each dispatch writes what it computes.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`
//! (`output[i] = input[i] * scale + i`), with a scale of 1.

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::state;
use super::encoder_holds::words_of;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState,
    Device, HeapDescriptor, HeapType, HostMemory, ResourceOptions, Size,
};
use std::alloc::Layout;
use std::ptr::NonNull;

/// The words each dispatch writes, one thread each.
const WORDS: u64 = 256;

/// The threads of a threadgroup.
const GROUP: u64 = 64;

/// The bytes apart the buffers placed in the heap lie, and the RAM's bytes before the heap's.
const CHUNK: u64 = 0x1_0000;

/// The host pages under the RAM's no-copy buffer: [`CHUNK`] before the heap's, and the heap's.
const PAGES: u64 = 8 * CHUNK;

/// The kernel's state on a device, and its scale of 1.
struct Kernel {
    device: Device,
    state: ComputePipelineState,
    scale: Buffer,
}

impl Kernel {
    fn new(scratch: &Scratch) -> Kernel {
        let device = device(&scratch.0);
        let state = state(&device);
        let scale = device
            .new_buffer_with_bytes(&1f32.to_le_bytes(), ResourceOptions::shared())
            .unwrap();
        Kernel {
            device,
            state,
            scale,
        }
    }

    /// One dispatch of [`WORDS`] threads reading `input` and writing `output`, in the encoder
    /// `encoder`.
    fn dispatch(&self, encoder: &mut ComputeCommandEncoder, input: &Buffer, output: &Buffer) {
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(input), 0, 0).unwrap();
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
        encoder
            .dispatch_threads(Size::new(WORDS, 1, 1), Size::new(GROUP, 1, 1))
            .unwrap();
    }

    /// A serial encoder of one dispatch, reading `input` and writing `output`, ended.
    fn encoder(&self, command_buffer: &CommandBuffer, input: &Buffer, output: &Buffer) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        self.dispatch(&mut encoder, input, output);
        encoder.end_encoding().unwrap();
    }
}

/// Commit `command_buffer` and wait for it: it completes with no error.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
}

/// Set `buffer`'s first [`WORDS`] floats, a host-visible buffer no command uses now, to
/// `value(i)`.
fn fill(buffer: &Buffer, value: impl Fn(u64) -> f32) {
    let contents = buffer.contents().unwrap();
    for i in 0..WORDS {
        let bytes = value(i).to_le_bytes();
        // SAFETY: inside the buffer's first `WORDS` words, which no command uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr().add(4 * i as usize), 4)
        };
    }
}

/// A `Shared` buffer of [`WORDS`] floats, word `i` `value(i)`.
fn floats(device: &Device, value: impl Fn(u64) -> f32) -> Buffer {
    let buffer = (device.new_buffer(4 * WORDS, ResourceOptions::shared())).unwrap();
    fill(&buffer, value);
    buffer
}

/// Whether `buffer`'s first [`WORDS`] floats, a host-visible buffer no command writes now, are
/// `k * i`.
fn holds(buffer: &Buffer, k: u64) -> bool {
    let words = words_of(buffer);
    (0..WORDS as usize).all(|i| f32::from_bits(words[i]) == (k * i as u64) as f32)
}

/// Encoders of one dispatch each take nothing into their orders; an encoder of two, the second
/// reading what the first wrote, takes both in, with one barrier between them, and the second
/// reads the first's words.
#[test]
fn an_encoders_order_takes_in_its_first_dispatch_only_for_a_later_one() {
    let scratch = Scratch::new("track-order");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    let queue = device.new_command_queue().unwrap();
    let chain: Vec<Buffer> = (0..4)
        .map(|k| floats(device, |i| if k == 0 { i as f32 } else { 0.0 }))
        .collect();
    // Warm: the pipeline is made, so each dispatch below records at once.
    let command_buffer = queue.command_buffer().unwrap();
    kernel.encoder(&command_buffer, &chain[0], &chain[1]);
    run(&command_buffer);

    // Three encoders of one dispatch, each reading what the one before wrote.
    fill(&chain[1], |_| 0.0);
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for k in 0..3 {
        kernel.encoder(&command_buffer, &chain[k], &chain[k + 1]);
    }
    let alone = device.work_counts().commands_ordered_inside - before.commands_ordered_inside;
    run(&command_buffer);
    let words = [2, 3, 4].map(|k| holds(&chain[k - 1], k as u64));

    // One encoder of two dispatches, the second reading what the first wrote, in a command
    // buffer of its own: the barrier between them is the encoder's own, none a group's.
    fill(&chain[1], |_| 0.0);
    fill(&chain[2], |_| 0.0);
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    kernel.dispatch(&mut encoder, &chain[0], &chain[1]);
    kernel.dispatch(&mut encoder, &chain[1], &chain[2]);
    encoder.end_encoding().unwrap();
    let after = device.work_counts();
    let two = (
        after.commands_ordered_inside - before.commands_ordered_inside,
        after.command_barriers - before.command_barriers,
    );
    run(&command_buffer);
    let read = holds(&chain[2], 3);
    println!(
        "TRACK-ORDER one-dispatch encoders took in {alone}; two-dispatch encoder took in, barriers \
         {two:?}; words right {words:?} {read}"
    );
    assert_eq!(words, [true; 3], "a chain of one-dispatch encoders' words");
    assert!(read, "the second dispatch read the first's words");
    assert_eq!(
        alone, 0,
        "one-dispatch encoders took commands into their orders"
    );
    assert_eq!(two, (2, 1), "an encoder of two dependent dispatches");
}

/// Host pages of [`PAGES`] bytes, zeroed, aligned to [`CHUNK`], freed when dropped: declared
/// before the device that uses them, so dropped after it.
struct Pages(NonNull<u8>);

impl Pages {
    fn layout() -> Layout {
        Layout::from_size_align(PAGES as usize, CHUNK as usize).unwrap()
    }

    fn new() -> Pages {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Pages::layout()) };
        Pages(NonNull::new(pointer).unwrap())
    }

    /// Its bytes from `offset` on, for a no-copy buffer or heap.
    ///
    /// # Safety
    ///
    /// They outlive every object over them, which alone use them while it lives.
    unsafe fn at(&self, offset: u64) -> HostMemory {
        // SAFETY: inside the allocation; the caller's contract.
        unsafe { HostMemory::new(NonNull::new_unchecked(self.0.as_ptr().add(offset as usize))) }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Pages::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Pages::layout()) };
    }
}

/// Dispatches over buffers placed in a heap over host pages, beside a no-copy buffer over them,
/// look the keys of their uses up under the host aliases' lock only where no encoder found them
/// while the aliases stand: none while nothing moves, one after a no-copy buffer is made over other
/// pages, none again; and each writes `2 * i`.
#[test]
fn dispatches_over_host_pages_look_their_aliases_up_once_while_they_stand() {
    let pages = Pages::new();
    let other = Pages::new();
    let scratch = Scratch::new("track-aliases");
    let kernel = Kernel::new(&scratch);
    let device = &kernel.device;
    // SAFETY: the pages outlive the device and every object over them, and only their commands
    // use them while they live.
    let ram = unsafe {
        device.new_buffer_with_bytes_no_copy(pages.at(0), PAGES, ResourceOptions::shared(), None)
    }
    .unwrap();
    let descriptor = HeapDescriptor::new(
        HeapType::Placement,
        PAGES - CHUNK,
        ResourceOptions::shared(),
    );
    // SAFETY: as above, past the RAM's first chunk.
    let heap =
        unsafe { device.new_heap_with_bytes_no_copy(pages.at(CHUNK), &descriptor, None, None) }
            .unwrap();
    let place = |n: u64| {
        let buffer = heap
            .new_buffer_at_offset(4 * WORDS, ResourceOptions::shared(), n * CHUNK)
            .unwrap();
        fill(&buffer, |i| if n == 0 { i as f32 } else { 0.0 });
        buffer
    };
    let (input, output) = (place(0), place(1));
    let queue = device.new_command_queue().unwrap();
    // The look-ups under the lock one encoder of one dispatch made as it was encoded, and
    // whether the dispatch wrote `2 * i`.
    let encode = || {
        fill(&output, |_| 0.0);
        let before = device.work_counts().host_alias_lookups;
        let command_buffer = queue.command_buffer().unwrap();
        kernel.encoder(&command_buffer, &input, &output);
        let looked = device.work_counts().host_alias_lookups - before;
        run(&command_buffer);
        (looked, holds(&output, 2))
    };
    // Warm: the pipeline is made and the keys are found.
    encode();
    let steady = encode();
    // A no-copy buffer over other pages, placed over their bytes: the aliases stand no longer.
    // SAFETY: as above.
    let _moved = unsafe {
        device.new_buffer_with_bytes_no_copy(other.at(0), CHUNK, ResourceOptions::shared(), None)
    }
    .unwrap();
    let after_move = encode();
    let again = encode();
    println!(
        "TRACK-ALIASES (look-ups, words right): steady {steady:?} after a no-copy buffer {after_move:?} \
         again {again:?}; ram {} bytes",
        ram.length()
    );
    assert_eq!(
        [steady, after_move, again],
        [(0, true), (1, true), (0, true)],
        "look-ups of a dispatch's host aliases"
    );
}
