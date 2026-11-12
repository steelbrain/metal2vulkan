//! What waits for the fill queue's fills (`device/staging.rs`), and what new memory reads, on the
//! device the cfg selects.
//!
//! - **None but those that reach them.** A compositor's command buffers on a queue of their own
//!   name only buffers made before, while an app's queue, kept busy by kernels, makes a new
//!   `Private` buffer each round (its zeros 1 MiB or more, which a device with a fill queue fills
//!   there) and reads it in its next command buffer. No batch waits on the GPU for fills its
//!   commands reach none of (`WorkCounts::fill_queue_waits_unused`); on a device with a fill
//!   queue the app's batches wait for theirs (`fill_queue_waits`); every new buffer reads zero.
//! - **Zero through every reach.** A new buffer, in memory a released buffer the GPU filled with
//!   `0xab` may have left, first used by a batch behind a long run of kernels on its queue, reads
//!   zero from another queue through its name and through a pointer loaded from memory (the
//!   owned fixture `engine/ob2_loaded_pointer.ll`), each committed while that first batch may
//!   still wait. A texture placed in a new heap, whose move a batch that names nothing of the
//!   heap takes, reads back what a blit then wrote into it.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Device, HeapDescriptor, HeapType, Origin, PipelineOptions, ResourceOptions, Size,
    TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::time::Duration;

/// The bytes of each new buffer: past the fewest the fill queue takes.
const NEW_BYTES: u64 = 2 << 20;

/// The rounds of the first test.
const ROUNDS: usize = 8;

/// The load that keeps a queue busy: dispatches over a `Private` buffer of [`LOAD_BYTES`]; a CPU
/// device runs a few over a small one.
const LOAD_DISPATCHES: usize = match cfg!(m2v_cpu_icd) {
    true => 2,
    false => 64,
};
const LOAD_BYTES: u64 = match cfg!(m2v_cpu_icd) {
    true => 1 << 20,
    false => 64 << 20,
};

/// The floats the compositor's kernel reads.
const FRAME_FLOATS: u64 = 1 << 16;

/// The state of `kernel` in the owned fixture at `path`.
fn state(device: &Device, path: &str, kernel: &str) -> ComputePipelineState {
    let function = super::encoder_holds::library(device, path)
        .new_function_with_name(kernel)
        .unwrap();
    (device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default()))
        .unwrap()
}

/// `count` floats of `scale_and_offset` from `input` into `output`, `times` over.
fn scale(
    command_buffer: &CommandBuffer,
    state: &ComputePipelineState,
    (input, output): (&Buffer, &Buffer),
    count: u64,
    times: usize,
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(input), 0, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
    for _ in 0..times {
        (encoder.dispatch_threads(Size::new(count, 1, 1), Size::new(256, 1, 1))).unwrap();
    }
    encoder.end_encoding().unwrap();
}

/// A blit of `source`, whole, into `target` at its start.
fn copy(command_buffer: &CommandBuffer, source: &Buffer, target: &Buffer) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    (blit.copy_from_buffer_to_buffer(source, 0, target, 0, source.length())).unwrap();
    blit.end_encoding().unwrap();
}

/// The bytes of `buffer`, a `Shared` one no command buffer in flight writes.
fn read(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: `length` bytes of a `Shared` buffer no command buffer in flight writes.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Wait for `command_buffer` and hold that it completed.
fn completed(name: &str, command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{name}"
    );
}

/// The device the cfg selects, its own checks' submissions done, so the command buffers of a
/// test are what takes the fills.
fn settled(scratch: &Scratch) -> Device {
    let device = device(&scratch.0);
    let name = device.name().to_owned();
    let settle = Duration::from_secs(120);
    assert!(device.settle_in_place_checks(settle), "{name}");
    assert!(device.settle_queue_switch_check(settle), "{name}");
    device
}

/// The module docs: no batch waits for fills its commands reach none of.
#[test]
fn a_batch_that_reaches_no_new_memory_waits_for_none_of_its_fills() {
    let scratch = Scratch::new("fill-waits");
    let device = settled(&scratch);
    let name = device.name().to_owned();
    let has_fill_queue = device.fill_queue_family().is_some();
    let scale_state = state(&device, "examples/runtime_examples.ll", "scale_and_offset");
    let (app, compositor) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let private = |bytes| {
        device
            .new_buffer(bytes, ResourceOptions::private())
            .unwrap()
    };
    let shared = |bytes| device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let (load_in, load_out) = (private(LOAD_BYTES), private(LOAD_BYTES));
    let (frame_in, frame_out) = (shared(FRAME_FLOATS * 4), shared(FRAME_FLOATS * 4));
    let floats: Vec<u8> = (0..FRAME_FLOATS)
        .flat_map(|at| (at as f32).to_le_bytes())
        .collect();
    let contents = frame_in.contents().unwrap();
    // SAFETY: the buffer's `FRAME_FLOATS` floats, which no command buffer uses yet.
    unsafe { std::ptr::copy_nonoverlapping(floats.as_ptr(), contents.as_ptr(), floats.len()) };
    // The buffers made so far are filled by a command buffer of their own.
    let priming = app.command_buffer().unwrap();
    scale(&priming, &scale_state, (&load_in, &load_out), 256, 1);
    priming.commit().unwrap();
    completed(&name, &priming);
    let before = device.work_counts();
    let mut flight = std::collections::VecDeque::new();
    let mut reads = Vec::new();
    for _ in 0..ROUNDS {
        // The app's new buffer, then the compositor's frame over what it made before, then the
        // app's command buffer that reads the new buffer behind its kernels.
        let new = private(NEW_BYTES);
        let frame = compositor.command_buffer().unwrap();
        scale(
            &frame,
            &scale_state,
            (&frame_in, &frame_out),
            FRAME_FLOATS,
            1,
        );
        frame.commit().unwrap();
        let read_back = shared(NEW_BYTES);
        let work = app.command_buffer().unwrap();
        scale(
            &work,
            &scale_state,
            (&load_in, &load_out),
            LOAD_BYTES / 4,
            LOAD_DISPATCHES,
        );
        copy(&work, &new, &read_back);
        work.commit().unwrap();
        // A second frame, while the app's fills may still wait behind its kernels.
        let again = compositor.command_buffer().unwrap();
        scale(
            &again,
            &scale_state,
            (&frame_in, &frame_out),
            FRAME_FLOATS,
            1,
        );
        again.commit().unwrap();
        completed(&name, &frame);
        completed(&name, &again);
        flight.push_back(work);
        if flight.len() > 2 {
            completed(&name, &flight.pop_front().unwrap());
        }
        reads.push((new, read_back));
    }
    for work in flight {
        completed(&name, &work);
    }
    let counts = device.work_counts();
    let waits = counts.fill_queue_waits - before.fill_queue_waits;
    let unused = counts.fill_queue_waits_unused - before.fill_queue_waits_unused;
    let submissions = counts.fill_queue_submissions - before.fill_queue_submissions;
    eprintln!(
        "FILL-WAITS {name}: fill queue {has_fill_queue}, {ROUNDS} rounds: fill queue submissions \
         {submissions}, batches that waited for fill-queue submissions {waits}, of them for \
         fills they reach none of {unused}"
    );
    for (at, (_, read_back)) in reads.iter().enumerate() {
        assert!(
            read(read_back).iter().all(|&byte| byte == 0),
            "{name}: round {at}'s new buffer read other bytes than zero"
        );
    }
    let out = read(&frame_out);
    for (at, word) in out.as_chunks::<4>().0.iter().enumerate().step_by(997) {
        let value = f32::from_le_bytes(*word);
        assert_eq!(value, 2.0 * at as f32, "{name}: the frame's float {at}");
    }
    assert_eq!(
        unused, 0,
        "{name}: batches waited for fills they reach none of"
    );
    if has_fill_queue {
        assert!(
            submissions >= ROUNDS as u64 && waits >= ROUNDS as u64,
            "{name}: the app's fills went to the fill queue {submissions} times, waited for \
             {waits}"
        );
    }
}

/// A new `Private` buffer of [`NEW_BYTES`], after one of its length the GPU filled with `0xab` was
/// released, so its memory may be that one's.
fn over_garbage(device: &Device, queue: &CommandQueue) -> Buffer {
    let garbage = device
        .new_buffer(NEW_BYTES, ResourceOptions::private())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&garbage, 0..NEW_BYTES, 0xab).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    drop(command_buffer);
    drop(garbage);
    device
        .new_buffer(NEW_BYTES, ResourceOptions::private())
        .unwrap()
}

/// The module docs: new memory reads zero from another queue, by name and through a pointer, and
/// a heap's texture moved by a batch that names nothing of the heap keeps what is written.
#[test]
fn new_memory_reads_zero_through_every_reach() {
    let scratch = Scratch::new("fill-reach");
    let device = settled(&scratch);
    let name = device.name().to_owned();
    let scale_state = state(&device, "examples/runtime_examples.ll", "scale_and_offset");
    let pointer_state = state(
        &device,
        "engine/ob2_loaded_pointer.ll",
        "ob2_loaded_pointer",
    );
    let [busy, named, pointed, other] =
        std::array::from_fn(|_| device.new_command_queue().unwrap());
    let shared = |bytes| device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let private = |bytes| {
        device
            .new_buffer(bytes, ResourceOptions::private())
            .unwrap()
    };
    let (load_in, load_out) = (private(LOAD_BYTES), private(LOAD_BYTES));
    let new = over_garbage(&device, &named);
    // The first batch that uses the new buffer goes behind a long run of kernels on its queue,
    // in a command buffer of their own.
    let load = busy.command_buffer().unwrap();
    scale(
        &load,
        &scale_state,
        (&load_in, &load_out),
        LOAD_BYTES / 4,
        LOAD_DISPATCHES,
    );
    load.commit().unwrap();
    let first = busy.command_buffer().unwrap();
    let first_read = shared(NEW_BYTES);
    copy(&first, &new, &first_read);
    first.commit().unwrap();
    // By its name, from another queue.
    let by_name = named.command_buffer().unwrap();
    let named_read = shared(NEW_BYTES);
    copy(&by_name, &new, &named_read);
    by_name.commit().unwrap();
    // Through a pointer loaded from memory, from a third: words 0 to 7 into `out`; the kernel
    // stores into words 8 to 15, which nothing reads.
    let through = pointed.command_buffer().unwrap();
    let out = shared(32);
    let indices: Vec<u8> = (0..16u32).flat_map(u32::to_le_bytes).collect();
    let mut encoder = through.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pointer_state).unwrap();
    encoder.set_bytes(&indices, 0).unwrap();
    encoder
        .set_bytes(&new.gpu_address().0.to_le_bytes(), 1)
        .unwrap();
    encoder.set_buffer(Some(&out), 0, 2).unwrap();
    (encoder.dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1))).unwrap();
    encoder.end_encoding().unwrap();
    through.commit().unwrap();
    for command_buffer in [&load, &first, &by_name, &through] {
        completed(&name, command_buffer);
    }
    for (what, buffer) in [("its first batch", &first_read), ("by name", &named_read)] {
        assert!(
            read(buffer).iter().all(|&byte| byte == 0),
            "{name}: the new buffer read other bytes than zero {what}"
        );
    }
    assert!(
        read(&out).iter().all(|&byte| byte == 0),
        "{name}: the new buffer read other bytes than zero through a pointer"
    );
    // A heap's texture whose move a batch that names nothing of the heap takes: what a blit then
    // writes into it reads back.
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            4 << 20,
            ResourceOptions::private(),
        ))
        .unwrap();
    const SIDE: u64 = 256;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = heap.new_texture(&descriptor).unwrap();
    let unrelated = other.command_buffer().unwrap();
    let (frame_in, frame_out) = (shared(1 << 12), shared(1 << 12));
    scale(
        &unrelated,
        &scale_state,
        (&frame_in, &frame_out),
        1 << 10,
        1,
    );
    unrelated.commit().unwrap();
    let pattern: Vec<u8> = (0..SIDE * SIDE * 4).map(|at| (at % 251) as u8).collect();
    let source = device
        .new_buffer_with_bytes(&pattern, ResourceOptions::shared())
        .unwrap();
    let back = shared(SIDE * SIDE * 4);
    let written = named.command_buffer().unwrap();
    let mut blit = written.blit_command_encoder().unwrap();
    let (size, origin) = (Size::new(SIDE, SIDE, 1), Origin::new(0, 0, 0));
    let (row, image) = (SIDE * 4, SIDE * SIDE * 4);
    let none = BlitOption::None;
    (blit.copy_from_buffer_to_texture(&source, 0, row, image, size, &texture, 0, 0, origin, none))
        .unwrap();
    (blit.copy_from_texture_to_buffer(&texture, 0, 0, origin, size, &back, 0, row, image, none))
        .unwrap();
    blit.end_encoding().unwrap();
    written.commit().unwrap();
    completed(&name, &unrelated);
    completed(&name, &written);
    assert!(
        read(&back) == pattern,
        "{name}: the heap's texture read back other texels than were written"
    );
    eprintln!("FILL-REACH {name}: new memory read zero by name and through a pointer");
}
