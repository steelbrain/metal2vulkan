//! Where the scheduler holds a committed batch, and for how long
//! (`WorkCounts::heads_held_for_earlier_commits` and kin, `WorkCounts::batch_hold_nanos`).
//!
//! `commit` submits nothing, as Metal's returns before its command buffer is scheduled, so the
//! time between a commit and its `vkQueueSubmit` is the scheduler's: a look of its for the next
//! batch to submit either takes a head or leaves it, and these counts say which reason left it
//! and what the leaving cost. A consumer that sees a committed command buffer reach Vulkan late
//! reads them to find out where the time went.
//!
//! The hold that crosses queues is the one behind an earlier commit (`scheduler/order.rs`): Metal
//! tracks a resource across queues in the order the GPU takes the command buffers, so a batch the
//! runtime holds for its checks, or for a host step whose gate was taken, must not let a later
//! commit's conflicting batch go first, and one held for its pipelines' making gives way
//! (**Held batches give way**). Each test holds one batch by its pipeline's making (the device's
//! pool's test hold, so a making is as slow as the test makes it):
//! `a_head_that_shares_a_buffer_with_a_batch_held_for_its_making_goes_first` and
//! `a_head_over_the_same_host_bytes_through_another_buffer_goes_while_a_batch_is_held` show a
//! command buffer of another queue that writes what the held dispatch writes going first,
//! through the same buffer or through another over the same host pages, and the dispatch running
//! after it; `a_head_that_waits_for_a_fence_a_batch_held_for_its_making_updates_waits_behind_it`
//! shows the line that stops at: a fence the two share keeps commit order.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll`, as
//! `dispatch_pending.rs` runs it.

use super::blit_holds::{device, exact_waits_device, Host, Scratch, BYTES, WAIT};
use super::dispatch_pending::{dispatch, dispatch_updating, state, Buffers, Grid};
use crate::pipeline::pool::held;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, EventRef, ResourceOptions,
    Size,
};
use std::time::{Duration, Instant};

/// How long a command buffer the scheduler holds is watched not to complete.
const WATCHED: Duration = Duration::from_millis(500);

/// The threads one threadgroup of the kernel runs.
const THREADS: u64 = 64;

/// A command buffer of `queue` that fills `target`'s first bytes, waiting first for `after` where
/// it is given: no pipeline of its own, so nothing but the wait and the scheduler holds it.
fn fill(queue: &CommandQueue, target: &Buffer, after: Option<EventRef<'_>>) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    if let Some(event) = after {
        command_buffer.encode_wait_for_event(event, 1).unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..64, 7).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Wait until `done` holds, at most `within`: whether it did.
fn eventually(within: Duration, done: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + within;
    while !done() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    done()
}

/// A dispatch whose pipeline's making the pool is held from finishing gives way in the
/// scheduler's order (`scheduler/order.rs`, **Held batches give way**): a command buffer committed
/// after it **on another queue** that writes the buffer its dispatch writes goes while the making
/// holds it, as an x86 Mac's GPU may take a later commit of another queue first, and no look
/// counts it as waiting behind an earlier commit; the dispatch's own hold for its host step is
/// counted at a look of the scheduler's thread (the fill may go in a pass of the test's own
/// thread, which counts no hold), and once as it goes, with the nanoseconds it waited. Released,
/// the dispatch runs after the fill: its words are over the fill's. (A cold compute pipeline's
/// making of 3 s on an Iris Xe held a guest's compositor frames behind another process's command
/// buffer in commit order.)
#[test]
fn a_head_that_shares_a_buffer_with_a_batch_held_for_its_making_goes_first() {
    let scratch = Scratch::new("first");
    // Its recordings wait for the making until it is made: a bound passed would drop the dispatch.
    let device = exact_waits_device(&scratch.0);
    let state = state(&device);
    let buffers = Buffers::new(&device);
    let (a, b) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let hold = held::hold();
    let pending = dispatch(&a, &state, &buffers, Grid::One(Size::new(THREADS, 1, 1)));
    let before = device.work_counts();
    pending.commit().unwrap();

    let sharing = fill(&b, buffers.output(), None);
    sharing.commit().unwrap();
    sharing
        .wait_until_completed_for(WAIT)
        .expect("a batch writing the buffer the dispatch held for its making writes waited for it");
    assert_eq!(sharing.status(), CommandBufferStatus::Completed);
    assert_eq!(
        device.work_counts().heads_held_for_earlier_commits,
        before.heads_held_for_earlier_commits,
        "a batch that went before the held one was counted as waiting behind it"
    );
    // Counted at a look of the scheduler's thread: the fill may have gone in a pass of this
    // thread's, as it committed or waited (`Scheduler::submit_before_waiting`), which counts no
    // hold and leaves the held head to that thread's next look.
    assert!(
        eventually(WAIT, || {
            device.work_counts().heads_held_for_host_steps > before.heads_held_for_host_steps
        }),
        "the held dispatch's own hold for its host step was not counted"
    );
    assert_ne!(pending.status(), CommandBufferStatus::Completed);

    drop(hold);
    pending.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        pending.status(),
        CommandBufferStatus::Completed,
        "{:?} {:?}",
        pending.error(),
        pending.failed_pipeline()
    );
    assert!(
        buffers.written(THREADS),
        "the dispatch, after the fill, did not write what Metal's writes"
    );
    let after = device.work_counts();
    assert!(
        after.batches_held_past_a_look > before.batches_held_past_a_look,
        "the held dispatch was not counted as it went"
    );
    assert!(
        after.batch_hold_nanos > before.batch_hold_nanos,
        "the hold was counted as taking no time"
    );
}

/// A dispatch whose pipeline's making the pool is held from finishing updates a fence, and a
/// command buffer committed after it on another queue waits for that fence before a fill of a
/// buffer of its own: the fence keeps commit order where the held batch gives way on resources
/// (`scheduler/order.rs`, **Held batches give way**), so the fill waits behind the dispatch,
/// counted, and both complete once the making ends.
#[test]
fn a_head_that_waits_for_a_fence_a_batch_held_for_its_making_updates_waits_behind_it() {
    let scratch = Scratch::new("fence");
    let device = exact_waits_device(&scratch.0);
    let state = state(&device);
    let buffers = Buffers::new(&device);
    let apart = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let fence = device.new_fence();
    let (a, b) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let hold = held::hold();
    let grid = Grid::One(Size::new(THREADS, 1, 1));
    let pending = dispatch_updating(&a, &state, &buffers, grid, &fence);
    let before = device.work_counts();
    pending.commit().unwrap();

    let waiting = b.command_buffer().unwrap();
    let mut blit = waiting.blit_command_encoder().unwrap();
    blit.wait_for_fence(&fence).unwrap();
    blit.fill_buffer(&apart, 0..64, 7).unwrap();
    blit.end_encoding().unwrap();
    waiting.commit().unwrap();
    assert!(
        waiting.wait_until_completed_for(WATCHED).is_err(),
        "a batch waiting for a fence the held dispatch updates did not wait behind it"
    );
    assert!(
        eventually(WAIT, || {
            device.work_counts().heads_held_for_earlier_commits
                > before.heads_held_for_earlier_commits
        }),
        "the hold behind an earlier commit on another queue was not counted"
    );
    assert_ne!(pending.status(), CommandBufferStatus::Completed);

    drop(hold);
    for command_buffer in [&pending, &waiting] {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            command_buffer.status(),
            CommandBufferStatus::Completed,
            "{:?} {:?}",
            command_buffer.error(),
            command_buffer.failed_pipeline()
        );
    }
    assert!(buffers.written(THREADS), "the dispatch did not write");
}

/// A `Shared` no-copy buffer of `device` over all of `host`'s pages.
fn over(device: &Device, host: &Host) -> Buffer {
    // SAFETY: `host` is declared before the device in each leg, so its pages outlive every buffer
    // over them, and only those buffers use them while they live.
    let memory = unsafe { host.memory() };
    // SAFETY: `memory` is `BYTES` bytes of whole pages (`Host`).
    unsafe {
        device.new_buffer_with_bytes_no_copy(memory, BYTES as u64, ResourceOptions::shared(), None)
    }
    .unwrap()
}

/// A dispatch whose pipeline's making the pool is held from finishing writes host pages through
/// one no-copy buffer, and a command buffer committed after it on another queue fills the same
/// bytes through a second no-copy buffer over exactly those pages, which joins the first's memory
/// (`host_aliases.rs`), so the two meet as one resource's uses do. The dispatch gives way
/// (`scheduler/order.rs`, **Held batches give way**), so the fill goes while the dispatch's making
/// holds it, and no look counts it as waiting behind an earlier commit. It had waited for the
/// making (a guest's compositor frames behind another process's command buffer over a shared
/// surface's pages, frozen for a cold compile's 3 s). Released, the dispatch runs after the fill,
/// which went to the GPU first, so its words are what Metal's dispatch writes.
#[test]
fn a_head_over_the_same_host_bytes_through_another_buffer_goes_while_a_batch_is_held() {
    let host = Host::new();
    let scratch = Scratch::new("aliased");
    // Its recordings wait for the making until it is made: a bound passed would drop the dispatch
    // and let the fill go whatever the order says.
    let device = exact_waits_device(&scratch.0);
    let state = state(&device);
    let (first, second) = (over(&device, &host), over(&device, &host));
    let buffers = Buffers::writing(&device, first);
    let (a, b) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let hold = held::hold();
    let pending = dispatch(&a, &state, &buffers, Grid::One(Size::new(THREADS, 1, 1)));
    let before = device.work_counts();
    pending.commit().unwrap();

    let aliased = fill(&b, &second, None);
    aliased.commit().unwrap();
    aliased
        .wait_until_completed_for(WAIT)
        .expect("a batch writing the held batch's bytes through another buffer waited behind it");
    assert_eq!(aliased.status(), CommandBufferStatus::Completed);
    assert_eq!(
        device.work_counts().heads_held_for_earlier_commits,
        before.heads_held_for_earlier_commits,
        "a batch over the held one's bytes through another buffer was counted as waiting behind it"
    );
    assert_ne!(pending.status(), CommandBufferStatus::Completed);
    assert!(
        host.words()[..16].iter().all(|&word| word == 0x0707_0707),
        "the fill through the second buffer did not land in the pages"
    );

    drop(hold);
    pending.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        pending.status(),
        CommandBufferStatus::Completed,
        "{:?} {:?}",
        pending.error(),
        pending.failed_pipeline()
    );
    assert!(
        buffers.written(THREADS),
        "the dispatch, after the fill, did not write what Metal's writes"
    );
}

/// A command buffer committed behind a shared event's value only the host sets is held
/// unsubmitted (`event_timeline.rs`), counted by that reason, and the nanoseconds of its hold are
/// counted as it goes.
#[test]
fn a_batch_held_for_an_event_value_is_counted_with_the_time_it_waited() {
    let scratch = Scratch::new("event");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let event = device.new_shared_event().unwrap();
    let target = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let command_buffer = fill(&queue, &target, Some(EventRef::Shared(&event)));
    let before = device.work_counts();
    command_buffer.commit().unwrap();
    assert!(
        eventually(WAIT, || {
            device.work_counts().heads_held_for_event_values > before.heads_held_for_event_values
        }),
        "the hold for the event's value was not counted"
    );
    assert!(
        command_buffer.wait_until_completed_for(WATCHED).is_err(),
        "the wait for a value only the host sets did not hold the command buffer"
    );
    event.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let after = device.work_counts();
    assert_eq!(
        after.batches_held_past_a_look,
        before.batches_held_past_a_look + 1,
        "the one batch a look found held was not counted once as it went"
    );
    assert!(
        after.batch_hold_nanos > before.batch_hold_nanos,
        "the hold was counted as taking no time"
    );
}
