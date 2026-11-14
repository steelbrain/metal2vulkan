//! A dispatch encoded over a no-copy buffer's pages whose command buffer is committed behind a
//! shared-event wait, the buffer then detached and rebound before the host signals the event.
//!
//! The wait holds the work unstarted -- a shared event only the host signals holds the batch
//! unsubmitted, as nothing of the GPU's will reach its value (`event_timeline.rs`,
//! `scheduler::event_waits`) -- so the command buffer's first submission comes after the rebind.
//! `rebind.rs`'s rule has the rebind then encode the compute encoder that named the buffer again,
//! in its own call, before that submission (`command_buffer/reencode.rs`): the dispatch writes the
//! pages the buffer lies in then. Where the rebind cannot encode it again, the submission runs
//! nothing and the command buffer fails `CommandBufferError::EncodedUseRebound`
//! (`command_buffer/first_submission.rs`). Either is honest; what neither allows is the dispatch
//! writing the pages it was encoded over while the command buffer reports `Completed` with no
//! error, which leaves the program reading pages it no longer uses and nothing said.
//!
//! The consumer's sequence, which these follow: the buffer made over one page, detached, rebound
//! over a second (the page the dispatch is encoded over), the command buffer committed behind the
//! wait, the buffer detached again and rebound over a third, and only then the event signalled.
//! Each leg differs in which rebind is given a keep-alive ([`Keep`]), the runtime extension that
//! decides whether the import of the pages it hands over is freed in the call or on one of the
//! device's handler threads (`host_memory.rs`): the consumer's own measurement turned on it.
//!
//! The pages are the test's own ([`Page`], freed when dropped), never the keep-alive's, so every
//! leg reads both ranges back whatever it passed: no leg's quiet answer rests on a free.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`
//! (`output[i] = input[i] * scale + i`), as `dispatch_pending.rs` runs it, in one threadgroup.

use super::blit_holds::{device, Scratch, WAIT};
use super::dispatch_pending::state;
use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, ComputePipelineState, Device,
    Error, EventRef, HostMemory, KeepAlive, ResourceOptions, Size,
};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The buffer's bytes: one page of any host, as the consumer's are.
pub(super) const BYTES: usize = 16 << 10;

/// The threads the dispatch runs, in one threadgroup, and the scale it multiplies by: the kernel
/// writes `3.0` into word 1, so a written page is one whose words are not all zero.
pub(super) const COUNT: u64 = 64;
pub(super) const SCALE: f32 = 2.0;

/// One page of host memory, zeroed, freed when dropped: declared before the device that uses it,
/// so dropped after it. A leg reads it back whatever keep-alive it passed, so a quiet leg is a
/// witness and not a free.
pub(super) struct Page(NonNull<u8>);

impl Page {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, BYTES).unwrap()
    }

    pub(super) fn new() -> Page {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Page::layout()) };
        Page(NonNull::new(pointer).unwrap())
    }

    /// Whether the kernel wrote it: its words are zero until it does.
    pub(super) fn written(&self) -> bool {
        self.words().iter().any(|&word| word != 0)
    }

    /// Its words, for a leg that reads back what a shader recorded in it.
    pub(super) fn words(&self) -> &[u32] {
        // SAFETY: `BYTES` bytes, allocated in `new` aligned past a word, which no command writes
        // any more (its command buffer completed).
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u32>(), BYTES / 4) }
    }

    /// Write `bytes` at its start, while no command uses it.
    pub(super) fn write(&self, bytes: &[u8]) {
        assert!(bytes.len() <= BYTES, "the page holds {BYTES} bytes");
        // SAFETY: `BYTES` bytes, allocated in `new`, which no command uses now.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.0.as_ptr(), bytes.len()) };
    }

    /// Its pages, for a no-copy buffer or a rebind.
    ///
    /// # Safety
    ///
    /// They outlive every buffer over them, which alone uses them while it lives.
    pub(super) unsafe fn memory(&self) -> HostMemory {
        // SAFETY: the caller's contract; `BYTES` bytes, allocated in `new` page-aligned.
        unsafe { HostMemory::new(self.0) }
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Page::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Page::layout()) };
    }
}

/// Which of the two rebinds is given a keep-alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keep {
    /// Both, as the consumer's production passes one at every replace.
    Both,
    /// The one over the pages the dispatch is encoded over.
    First,
    /// The one over the pages the buffer lies in at the submission.
    Second,
    /// Neither.
    Neither,
}

impl Keep {
    /// What the first and the second rebind pass.
    fn first(self) -> Option<KeepAlive> {
        matches!(self, Keep::Both | Keep::First).then(|| Box::new(()) as KeepAlive)
    }

    fn second(self) -> Option<KeepAlive> {
        matches!(self, Keep::Both | Keep::Second).then(|| Box::new(()) as KeepAlive)
    }
}

/// The dispatch of `state` writing `output`, encoded into `command_buffer`.
pub(super) fn dispatch(
    command_buffer: &CommandBuffer,
    state: &ComputePipelineState,
    input: &Buffer,
    output: &Buffer,
    scale: &Buffer,
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(input), 0, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_buffer(Some(scale), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(COUNT, 1, 1), Size::new(COUNT, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
}

/// The leg of `keep`: the sequence above, asserting that the dispatch reached the pages the buffer
/// lies in at the submission or that the command buffer failed `EncodedUseRebound`, and never that
/// it wrote the pages it was encoded over while reporting `Completed`.
fn leg(keep: Keep) {
    // Declared before the device, so dropped after it.
    let pages = [Page::new(), Page::new(), Page::new()];
    let scratch = Scratch::new("rebind-behind-a-wait");
    let device: Device = device(&scratch.0);
    let state = state(&device);
    let queue = device.new_command_queue().unwrap();
    let bytes: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let scale = device
        .new_buffer_with_bytes(&SCALE.to_le_bytes(), ResourceOptions::shared())
        .unwrap();
    // Warm, over a buffer of the runtime's own, as the consumer's does: the state's region
    // pipeline is made, so the dispatch below records at once rather than waiting for a making.
    let warm = device
        .new_buffer(COUNT * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    dispatch(&command_buffer, &state, &input, &warm, &scale);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();

    let runs = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&runs);
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            BYTES as u64,
            ResourceOptions::shared(),
            Some(Box::new(move |_: NonNull<u8>, _: u64| {
                counted.fetch_add(1, Ordering::SeqCst);
            })),
        )
    };
    let output = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("REBIND-BEHIND-A-WAIT: the device imports no host memory: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };

    assert_eq!(
        output.detach_backing(),
        Ok(true),
        "{keep:?}: the first detach"
    );
    // SAFETY: as above.
    unsafe { output.rebind_bytes_no_copy(pages[1].memory(), None, keep.first()) }.unwrap();

    // The dispatch is encoded over the second page, and cannot start: its command buffer waits
    // for an event only the host signals, below.
    let gate = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    dispatch(&command_buffer, &state, &input, &output, &scale);
    command_buffer.commit().unwrap();

    assert_eq!(
        output.detach_backing(),
        Ok(true),
        "{keep:?}: the detach of a buffer whose only use waits for an event"
    );
    // SAFETY: as above.
    unsafe { output.rebind_bytes_no_copy(pages[2].memory(), None, keep.second()) }.unwrap();

    // Read before the submission: the rebind encodes again in its own call, so these say what it
    // did, and nothing of the submission is in them.
    let counts = device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let status = command_buffer.status();
    let error = command_buffer.error();
    let (old, new) = (pages[1].written(), pages[2].written());
    println!(
        "rebind-behind-a-wait {keep:?}: {status:?} {error:?} old {old} new {new} \
         reencoded {} refused {} again {}",
        counts.encoders_reencoded, counts.reencodes_refused, counts.segments_recorded_again,
    );
    assert!(
        !old,
        "{keep:?}: the dispatch wrote the pages it was encoded over, which the buffer left \
         before the submission ({status:?}, {error:?})"
    );
    match error {
        None => {
            assert_eq!(status, CommandBufferStatus::Completed, "{keep:?}");
            assert!(
                new,
                "{keep:?}: the command buffer completed with no error, so the dispatch ran over \
                 the pages the buffer lies in"
            );
        }
        Some(CommandBufferError::EncodedUseRebound) => {
            assert!(
                !new,
                "{keep:?}: a submission that ran nothing wrote nothing"
            );
        }
        Some(other) => panic!("{keep:?}: {other:?}"),
    }
}

#[test]
fn a_dispatch_behind_a_wait_does_not_write_rebound_pages_with_a_keep_alive_at_each_rebind() {
    leg(Keep::Both);
}

#[test]
fn a_dispatch_behind_a_wait_does_not_write_rebound_pages_with_a_keep_alive_at_the_first_rebind() {
    leg(Keep::First);
}

#[test]
fn a_dispatch_behind_a_wait_does_not_write_rebound_pages_with_a_keep_alive_at_the_second_rebind() {
    leg(Keep::Second);
}

#[test]
fn a_dispatch_behind_a_wait_does_not_write_rebound_pages_with_no_keep_alive() {
    leg(Keep::Neither);
}
