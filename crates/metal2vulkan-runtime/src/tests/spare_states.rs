//! A command buffer takes the state a released command buffer of its device left, reset by that
//! one's release, its lists of records emptied with their room kept
//! (`command_buffer/spare_states.rs`): one of no more encoders than that one grows none of its
//! lists (`WorkCounts::record_lists_grown`); one made after another's release builds no state
//! (`WorkCounts::command_buffer_states_made`); the state carries nothing of the command buffer
//! before; and the device keeps at most 64 states, none whose lists hold past 8 MiB.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `compute_parked.rs`'s.

use super::blit_holds::{until, Scratch};
use super::compute_parked::Kernel;
use super::merge_scratch::read;
use crate::{Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, CommitFailure};
use crate::{ResourceOptions, WorkCounts};
use std::time::Duration;

/// The compute encoders of each command buffer, one dispatch each.
const ENCODERS: u64 = 32;

/// How many lists of a command buffer of [`ENCODERS`] compute encoders of `queue` grew, the
/// command buffer committed, completed and dropped: read once its device kept its state
/// (`WorkCounts::record_storage_kept`), at its release.
fn grown(kernel: &Kernel, queue: &CommandQueue, output: &Buffer) -> u64 {
    let before = kernel.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for _ in 0..ENCODERS {
        kernel.dispatch(&command_buffer, (output, 0));
    }
    kernel.run(&command_buffer);
    drop(command_buffer);
    until("the command buffer's release kept its state", || {
        kernel.device.work_counts().record_storage_kept > before.record_storage_kept
    });
    kernel.device.work_counts().record_lists_grown - before.record_lists_grown
}

/// Command buffers of [`ENCODERS`] compute encoders each, one after another's release, the kernel's
/// pipelines made first (a dispatch whose pipeline is being made ends its group, so the groups
/// and the lists of a command buffer that meets one differ from the next one's): the first's lists
/// grow past the room the warming command buffer of one dispatch left (its ended encoders'
/// records and their recordings at least), and each later one starts on the room the one before
/// it left and grows none.
#[test]
fn a_command_buffer_of_as_many_encoders_as_the_last_grows_no_list() {
    let scratch = Scratch::new("record-storage");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    until(
        "the warming command buffer's release kept its state",
        || kernel.device.work_counts().record_storage_kept > 0,
    );
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output(1);
    let first = grown(&kernel, &queue, &output);
    assert!(
        first >= 2,
        "the first command buffer's lists grown: {first}"
    );
    for round in 1..3 {
        let grown = grown(&kernel, &queue, &output);
        assert_eq!(grown, 0, "round {round}: the lists grown");
    }
}

/// The longest a command buffer is waited for.
const WAIT: Duration = Duration::from_secs(30);

/// The device's states kept, and the room their lists hold, in bytes.
fn held(kernel: &Kernel) -> (usize, usize) {
    kernel.device.shared().spare_states.held()
}

/// Release `command_buffer`, committed and completed or never committed, and wait until its
/// release ended: its state kept (`WorkCounts::record_storage_kept`) or its lists counted
/// (`record_lists_grown` moves for none of a few encoders, so the kept count tells), from
/// `before`, the counts before its release.
fn release(kernel: &Kernel, command_buffer: CommandBuffer, before: &WorkCounts) {
    drop(command_buffer);
    until("the command buffer's release kept its state", || {
        kernel.device.work_counts().record_storage_kept > before.record_storage_kept
    });
}

/// A command buffer of `queue` that fills `output`'s first 4 bytes with `byte`, committed and
/// completed.
fn fill(queue: &CommandQueue, output: &Buffer, byte: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(output, 0..4, byte).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    command_buffer
}

/// Command buffers each made after the one before was released take the state it left: past the
/// first, none builds a state (`WorkCounts::command_buffer_states_made`), and each holds the very
/// state the one before held.
#[test]
fn a_command_buffer_made_after_another_s_release_builds_no_state() {
    let scratch = Scratch::new("spare-states-back-to-back");
    let kernel = Kernel::new(&scratch);
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output(1);
    let before = kernel.device.work_counts();
    let first = fill(&queue, &output, 1);
    let (mut state, _) = first.held_state();
    release(&kernel, first, &before);
    let made = kernel.device.work_counts().command_buffer_states_made;
    for round in 2..10u8 {
        let before = kernel.device.work_counts();
        let command_buffer = fill(&queue, &output, round);
        let (held, _) = command_buffer.held_state();
        assert_eq!(held, state, "round {round}: the state the one before left");
        state = held;
        release(&kernel, command_buffer, &before);
    }
    assert_eq!(
        kernel.device.work_counts().command_buffer_states_made,
        made,
        "the states built past the first command buffer"
    );
}

/// A command buffer that failed, labelled, its encoder labelled, a handler added and a call's
/// refusal kept, leaves its state on its release; the command buffer made next takes that very
/// state and finds nothing of the one before: no status, error, refusal, label or ended encoder,
/// and its own fill runs and completes.
#[test]
fn a_kept_state_carries_nothing_of_the_command_buffer_before() {
    let scratch = Scratch::new("spare-states-clean");
    let kernel = Kernel::new(&scratch);
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel
        .device
        .new_buffer(4096, ResourceOptions::shared())
        .unwrap();
    let failed = queue.command_buffer().unwrap();
    failed.set_label("failed");
    let mut blit = failed.blit_command_encoder().unwrap();
    blit.set_label("failed fill");
    blit.fill_buffer(&output, 0..4, 0xee).unwrap();
    blit.end_encoding().unwrap();
    failed.add_completed_handler(Box::new(|_| {})).unwrap();
    failed.commit_failed(CommitFailure::new(8)).unwrap();
    failed.wait_until_completed_for(WAIT).unwrap();
    failed.locked_state().refuse_call(crate::error::internal(
        "a refusal the next command buffer must not see",
    ));
    assert_eq!(failed.status(), CommandBufferStatus::Error);
    assert!(failed.error().is_some() && failed.refusal().is_some());
    let (state, _) = failed.held_state();
    let before = kernel.device.work_counts();
    release(&kernel, failed, &before);

    let next = queue.command_buffer().unwrap();
    assert_eq!(next.held_state().0, state, "the state the failed one left");
    assert_eq!(next.status(), CommandBufferStatus::NotEnqueued);
    assert!(next.error().is_none(), "{:?}", next.error());
    assert!(next.refusal().is_none(), "{:?}", next.refusal());
    assert_eq!(next.label(), None);
    assert_eq!(next.locked_state().encoders_ended(), 0);
    let mut blit = next.blit_command_encoder().unwrap();
    blit.fill_buffer(&output, 0..4, 0x3c).unwrap();
    blit.end_encoding().unwrap();
    next.commit().unwrap();
    next.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        next.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        next.error()
    );
    assert_eq!(read(&output, 4), [0x3c; 4]);
}

/// The device keeps no state whose lists of records hold past 8 MiB: a command buffer of enough
/// encoders to pass it leaves none, the device having room for more states. And it keeps at most
/// 64: 80 command buffers released at once leave 64.
#[test]
fn the_device_keeps_at_most_64_states_none_past_8_mib() {
    let scratch = Scratch::new("spare-states-bounds");
    let kernel = Kernel::new(&scratch);
    let queue = (kernel.device)
        .new_command_queue_with_max_command_buffer_count(80)
        .unwrap();
    let output = kernel.output(1);

    // A command buffer of fills enough to pass 8 MiB of list room, never committed.
    let big = queue.command_buffer().unwrap();
    let mut encoders = 0;
    while big.held_state().1 <= 8 << 20 {
        let mut blit = big.blit_command_encoder().unwrap();
        blit.fill_buffer(&output, 0..4, 1).unwrap();
        blit.end_encoding().unwrap();
        encoders += 1;
        assert!(encoders < 1 << 20, "8 MiB of list room not reached");
    }
    let (kept, bytes) = held(&kernel);
    assert!(kept < 64, "the states kept: {kept}");
    let before = kernel.device.work_counts();
    drop(big);
    assert_eq!(
        kernel.device.work_counts().record_storage_kept,
        before.record_storage_kept,
        "a state of past 8 MiB kept, of {encoders} encoders"
    );
    assert_eq!(held(&kernel), (kept, bytes));

    // Every state kept is taken by the first ones made, so 80 released keep 64.
    let made: Vec<CommandBuffer> = (0..80).map(|_| queue.command_buffer().unwrap()).collect();
    assert_eq!(
        held(&kernel).0,
        0,
        "the states kept, 80 command buffers made"
    );
    let before = kernel.device.work_counts();
    drop(made);
    until("80 command buffers released", || held(&kernel).0 == 64);
    let after = kernel.device.work_counts();
    assert_eq!(after.record_storage_kept - before.record_storage_kept, 64);
}
