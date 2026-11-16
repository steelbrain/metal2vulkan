//! A predicated loop whose laying left no work apart, over a no-copy buffer that holds both the
//! loop's word and its body's input, and a rebind of that buffer.
//!
//! The loop's check (`command_buffer/rerun.rs`) reads only its unit's halt word, in the command
//! buffer's arena, and each run again submits the laying's command buffers again as they were
//! encoded, over the pages their first submission took, which the command buffer holds until it
//! completes: the commit holds none of the buffer's pages past it
//! (`Deferred::resolved_at_its_calls`). So a rebind while the loop runs again is taken, and the
//! loop goes on over the pages its submission found, as Metal's does (`rebind.rs`'s ordering
//! rule): its word ends it from those pages, and its body reads their input. A rebind before the
//! laying's submission is taken too, and that submission, whose encoders keep no log to be
//! encoded again from (`predication.rs`), runs nothing and fails `EncodedUseRebound`, announced
//! (`command_buffer/first_submission.rs`).
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The kernel is `scale_and_offset` of the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`output[i] = input[i] * scale + i`),
//! as `dispatch_pending.rs` runs it, in one threadgroup. It prints `LOOP-REBIND` lines, and
//! returns where the device imports no host memory. A leg that fails leaves its pages allocated:
//! a loop that did not end may read them still.

use super::blit_holds::{options, Scratch, WAIT};
use super::dispatch_pending::state;
use super::rebind_behind_a_wait::{Page, BYTES, COUNT, SCALE};
use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, ComputePipelineState,
    Degradation, Device, Error, EventRef, MessageSink, ResourceOptions, Size,
};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Where the loop's word lies in the buffer, and where its body's input begins.
const WORD_AT: u64 = 0;
const INPUT_AT: u64 = 256;

/// The word's values: the loop runs again while it holds [`AGAIN`] (Metal's code 0, `==`).
const AGAIN: u32 = 0;
const STOP: u32 = 1;

/// The first input word of the pages the buffer is rebound over; the first page's is 0.
const OTHER: f32 = 1000.0;

/// The degradations the device announced, in order.
type Heard = Arc<Mutex<Vec<Degradation>>>;

/// The device the cfg selects, with a scratch shader cache, and what it announces.
fn device(scratch: &Scratch) -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let hearing = Arc::clone(&heard);
    let mut options = options(&scratch.0);
    options.messages = Some(MessageSink::new(move |message| {
        if let Some(notice) = &message.degradation {
            hearing.lock().unwrap().push(notice.degradation);
        }
    }));
    (Device::system_default_with(&options).unwrap(), heard)
}

/// A page's bytes: the loop's `word` at [`WORD_AT`], and at [`INPUT_AT`] the input `first + i`
/// for each of the [`COUNT`] threads.
fn bytes(word: u32, first: f32) -> Vec<u8> {
    let mut bytes = vec![0; INPUT_AT as usize];
    bytes[WORD_AT as usize..][..4].copy_from_slice(&word.to_le_bytes());
    bytes.extend((0..COUNT).flat_map(|i| (first + i as f32).to_le_bytes()));
    bytes
}

/// What the body writes from the input whose first word is `first`.
fn expected(first: f32) -> Vec<f32> {
    (0..COUNT)
        .map(|i| (first + i as f32) * SCALE + i as f32)
        .collect()
}

/// A no-copy buffer over `page`; `None`, printed, where the device imports no host memory.
fn no_copy(device: &Device, page: &Page) -> Option<Buffer> {
    // SAFETY: the page outlives the buffer (dropped after the device, or never), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            page.memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("LOOP-REBIND: the device imports no such pages: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// The loop, encoded into `command_buffer`: a DoWhile over one dispatch of `state` that reads its
/// input from `over` at [`INPUT_AT`] and writes `output`, again while `over`'s word at
/// [`WORD_AT`] holds [`AGAIN`]. Its laying leaves no work apart: a direct dispatch, whose pipeline
/// is made.
fn encode_loop(
    command_buffer: &CommandBuffer,
    state: &ComputePipelineState,
    over: &Buffer,
    (output, scale): (&Buffer, &Buffer),
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(over), INPUT_AT, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_buffer(Some(scale), 0, 2).unwrap();
    encoder.encode_start_do_while().unwrap();
    encoder
        .dispatch_threads(Size::new(COUNT, 1, 1), Size::new(COUNT, 1, 1))
        .unwrap();
    let end = encoder.encode_end_do_while(Some(over), WORD_AT, 0, AGAIN);
    assert_eq!(end, Ok(true));
    encoder.end_encoding().unwrap();
}

/// The device's runs again of predicated loops once they reach `at_least`, or as they stand
/// [`WAIT`] after the call: the caller's assert names what it saw.
fn runs_reach(device: &Device, at_least: u64) -> u64 {
    let deadline = Instant::now() + WAIT;
    loop {
        let runs = device.work_counts().predicated_loop_runs_again;
        if runs >= at_least || Instant::now() >= deadline {
            return runs;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// `buffer`'s first [`COUNT`] floats, read once no command buffer writes them.
fn floats(buffer: &Buffer) -> Vec<f32> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `COUNT` floats, and no command buffer writes them now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), COUNT as usize) }.to_vec()
}

/// A rebind of the buffer while the loop runs again is taken, and the loop goes on over the
/// pages its submission found: the runs again go on after the rebind, the loop ends once the
/// word in those pages says so (the pages the buffer lies in then keep saying again), and the
/// body wrote what their input gives, the command buffer completing with no error.
#[test]
fn a_rebind_while_a_loop_runs_again_is_taken_and_the_loop_keeps_its_pages() {
    let pages = ManuallyDrop::new([Page::new(), Page::new()]);
    let scratch = Scratch::new("loop-rebind-running");
    let (device, heard) = device(&scratch);
    pages[0].write(&bytes(AGAIN, 0.0));
    pages[1].write(&bytes(AGAIN, OTHER));
    let Some(over) = no_copy(&device, &pages[0]) else {
        return;
    };
    let state = state(&device);
    let shared = ResourceOptions::shared();
    let output = device.new_buffer(COUNT * 4, shared).unwrap();
    let scale = device
        .new_buffer_with_bytes(&SCALE.to_le_bytes(), shared)
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    encode_loop(&command_buffer, &state, &over, (&output, &scale));
    let before = device.work_counts().predicated_loop_runs_again;
    command_buffer.commit().unwrap();
    let started = runs_reach(&device, before + 2);
    let old = over.contents().unwrap();
    // SAFETY: the second page outlives the buffer, as the first does.
    let rebound = unsafe { over.rebind_bytes_no_copy(pages[1].memory(), None, None) };
    let moved = over.contents() != Some(old);
    let at_rebind = device.work_counts().predicated_loop_runs_again;
    let after = runs_reach(&device, at_rebind + 2);
    // SAFETY: the word of the first page, which the GPU only reads; aligned, as the page is.
    unsafe { (old.as_ptr().add(WORD_AT as usize).cast::<u32>()).write_volatile(STOP) };
    let completed = command_buffer.wait_until_completed_for(WAIT);
    let runs = device.work_counts().predicated_loop_runs_again - before;
    println!(
        "LOOP-REBIND running {}: rebind {rebound:?}; runs again {} before it, {} after, {runs} \
         in all; {completed:?} {:?} {:?}; heard {:?}",
        device.name(),
        at_rebind - before,
        after - at_rebind,
        command_buffer.status(),
        command_buffer.error(),
        heard.lock().unwrap(),
    );
    assert!(
        started >= before + 2,
        "the loop ran again {} times within {WAIT:?}, not 2",
        started - before
    );
    assert_eq!(rebound, Ok(()), "the rebind while the loop runs again");
    assert!(moved, "the buffer's contents are the second page's");
    assert!(
        after >= at_rebind + 2,
        "the loop ran again {} times within {WAIT:?} after the rebind, not 2",
        after - at_rebind
    );
    assert!(
        completed.is_ok(),
        "the loop did not end within {WAIT:?} of its first page's word saying so ({runs} runs \
         again): it reads the pages the buffer lies in now"
    );
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    assert_eq!(floats(&output), expected(0.0), "the first page's input");
    drop((command_buffer, queue, over, output, scale, state));
    drop(device);
    drop(ManuallyDrop::into_inner(pages));
}

/// A rebind of the buffer before the laying's submission, the command buffer committed behind a
/// shared-event wait only the host signals, is taken: the laying's encoder cannot be encoded
/// again, so the submission runs nothing and the command buffer fails `EncodedUseRebound`,
/// announced, its output unwritten. Both pages' words stop the loop after its first pass, so a
/// laying that ran over either would have written the output and completed.
#[test]
fn a_rebind_before_a_loop_s_submission_fails_it_rebound() {
    let pages = ManuallyDrop::new([Page::new(), Page::new()]);
    let scratch = Scratch::new("loop-rebind-before");
    let (device, heard) = device(&scratch);
    pages[0].write(&bytes(STOP, 0.0));
    pages[1].write(&bytes(STOP, OTHER));
    let Some(over) = no_copy(&device, &pages[0]) else {
        return;
    };
    let state = state(&device);
    let shared = ResourceOptions::shared();
    let output = device.new_buffer(COUNT * 4, shared).unwrap();
    let scale = device
        .new_buffer_with_bytes(&SCALE.to_le_bytes(), shared)
        .unwrap();
    let gate = device.new_shared_event().unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    encode_loop(&command_buffer, &state, &over, (&output, &scale));
    command_buffer.commit().unwrap();
    // SAFETY: the second page outlives the buffer, as the first does.
    let rebound = unsafe { over.rebind_bytes_no_copy(pages[1].memory(), None, None) };
    gate.set_signaled_value(1);
    let completed = command_buffer.wait_until_completed_for(WAIT);
    // The announcement, where the command buffer failed so.
    let deadline = Instant::now() + WAIT;
    let announced =
        || (heard.lock().unwrap().iter()).any(|kind| *kind == Degradation::EncodedUseRebound);
    let rebound_failure = Some(CommandBufferError::EncodedUseRebound);
    while command_buffer.error() == rebound_failure && !announced() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    println!(
        "LOOP-REBIND before {}: rebind {rebound:?}; {completed:?} {:?} {:?}; heard {:?}",
        device.name(),
        command_buffer.status(),
        command_buffer.error(),
        heard.lock().unwrap(),
    );
    assert_eq!(rebound, Ok(()), "the rebind before the laying's submission");
    assert!(completed.is_ok(), "no completion within {WAIT:?}");
    assert_eq!(command_buffer.error(), rebound_failure);
    assert!(announced(), "encoded-use-rebound announced within {WAIT:?}");
    assert_eq!(floats(&output), vec![0.0; COUNT as usize], "nothing ran");
    drop((command_buffer, queue, over, output, scale, state, gate));
    drop(device);
    drop(ManuallyDrop::into_inner(pages));
}
