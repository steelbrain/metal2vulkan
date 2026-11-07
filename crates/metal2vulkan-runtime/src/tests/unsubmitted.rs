//! A command buffer that needs no GPU work (`command_buffer/submit.rs`, `State::needs_no_gpu`):
//! nothing it encoded recorded work, and its one batch waits for and signals no event and no
//! semaphore. It is taken with no `vkQueueSubmit` (`WorkCounts::command_buffers_unsubmitted`),
//! completes in its queue's order, behind the work submitted before it, and where its queue has
//! no work in flight the thread that took it finishes it, with no queue waiter woken
//! (`Completion::turn_owed`, `WorkCounts::completion_turns_where_taken`). One that signals an
//! event is submitted as before. Behind a batch the scheduler holds in its group
//! (`command_buffer/group.rs`), it goes in the group's submission instead, as any batch behind a
//! group held does.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::tests::SETTLE;
use crate::{
    CommandBuffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, EventRef,
    ResourceOptions,
};
use metal2vulkan::device::{CompletionPolicy, DevicePolicy};
use std::sync::{Arc, Mutex};

/// The command buffers of each kind.
const ROUNDS: u64 = 40;

/// The bytes of the fill the empty command buffer waits behind: long enough on every device
/// tested that the empty one is committed before it completes.
const FILL: u64 = 256 << 20;

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    device_with(CompletionPolicy::default())
}

/// [`device`], its policy the default's with `completion`.
fn device_with(completion: CompletionPolicy) -> Device {
    let mut policy = DevicePolicy::default();
    policy.completion = completion;
    let options = DeviceOptions {
        policy,
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// What a command buffer's handlers saw: the status in its scheduled handler, then in its
/// completed handler.
#[derive(Default)]
struct Seen {
    scheduled: Option<CommandBufferStatus>,
    completed: Option<CommandBufferStatus>,
}

/// Empty command buffers committed on an idle queue, each waited for, then a run of them never
/// waited for but the last: none is submitted, every one completes with Metal's status and
/// handlers, and its GPU start and end are one positive point at or after its kernel end, as
/// Metal's (an M3 Max, macOS 27.0.1). Each waited one is finished by the thread that took it,
/// with no queue waiter woken.
#[test]
fn an_empty_command_buffer_completes_with_no_submission() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    for _ in 0..ROUNDS {
        let command_buffer = queue.command_buffer().unwrap();
        let seen = Arc::new(Mutex::new(Seen::default()));
        let scheduled = Arc::clone(&seen);
        command_buffer
            .add_scheduled_handler(Box::new(move |buffer: &CommandBuffer| {
                scheduled.lock().unwrap().scheduled = Some(buffer.status());
            }))
            .unwrap();
        let completed = Arc::clone(&seen);
        command_buffer
            .add_completed_handler(Box::new(move |buffer: &CommandBuffer| {
                completed.lock().unwrap().completed = Some(buffer.status());
            }))
            .unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let (start, end) = (
            command_buffer.gpu_start_time().unwrap(),
            command_buffer.gpu_end_time().unwrap(),
        );
        let (kernel_start, kernel_end) = (
            command_buffer.kernel_start_time(),
            command_buffer.kernel_end_time(),
        );
        assert!(
            0.0 < kernel_start && kernel_start <= kernel_end && kernel_end <= start,
            "kernel {kernel_start}..{kernel_end}, GPU start {start}"
        );
        assert_eq!(
            start, end,
            "an empty command buffer's GPU times are one point"
        );
        // The completed handler ran before the wait returned; the scheduled one before it.
        let seen = seen.lock().unwrap();
        assert_eq!(seen.scheduled, Some(CommandBufferStatus::Scheduled));
        assert_eq!(seen.completed, Some(CommandBufferStatus::Completed));
    }
    let waited = device.work_counts();
    for _ in 0..ROUNDS {
        queue.command_buffer().unwrap().commit().unwrap();
    }
    let last = queue.command_buffer().unwrap();
    last.commit().unwrap();
    last.wait_until_completed().unwrap();
    let after = device.work_counts();
    println!(
        "unsubmitted {}, queue submissions {}, turns where taken {} of {ROUNDS} waited, waiter \
         wakes {}",
        after.command_buffers_unsubmitted - before.command_buffers_unsubmitted,
        after.queue_submissions - before.queue_submissions,
        waited.completion_turns_where_taken - before.completion_turns_where_taken,
        waited.wakes_completion_waiters - before.wakes_completion_waiters,
    );
    assert_eq!(
        after.command_buffers_unsubmitted - before.command_buffers_unsubmitted,
        2 * ROUNDS + 1,
        "every empty command buffer was taken with no submission"
    );
    assert_eq!(
        after.queue_submissions - before.queue_submissions,
        0,
        "an empty command buffer was submitted"
    );
    // A waiter whose slice ends in the moment between the taking and the turn makes the turn
    // itself (`command_buffer/completion.rs`): almost never.
    assert!(
        waited.completion_turns_where_taken - before.completion_turns_where_taken >= ROUNDS / 2,
        "the threads that took the empty command buffers did not finish them"
    );
    assert_eq!(
        waited.wakes_completion_waiters - before.wakes_completion_waiters,
        0,
        "a queue waiter was woken for an empty command buffer on an idle queue"
    );
}

/// An empty command buffer committed behind a long fill of its queue completes after it: its
/// completed handler runs after the fill's and sees the fill `Completed`, its GPU start is at or
/// after the fill's GPU end, and only the fill is submitted. Metal's completes so (an M3 Max,
/// macOS 27.0.1: its GPU times 65 to 135 us after the fill's `GPUEndTime`).
///
/// Under `CompletionPolicy::Exact`, so on every driver, the empty command buffer is taken with no
/// submission in each round. Under the default, on a device whose batches group (its check found
/// its submissions merge, `submit_check.rs`: lavapipe among them), the scheduler may hold the fill
/// in its group as it finds the empty one ready behind it, which then goes in the group's one
/// submission with the fill: either way it makes no submission of its own.
#[test]
fn an_empty_command_buffer_completes_behind_its_queue_s_earlier_work() {
    for completion in [CompletionPolicy::Exact, CompletionPolicy::Grouped] {
        completes_behind(completion);
    }
}

/// [`an_empty_command_buffer_completes_behind_its_queue_s_earlier_work`] under `completion`.
fn completes_behind(completion: CompletionPolicy) {
    let device = device_with(completion);
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    let groups = completion == CompletionPolicy::Grouped && device.submissions_merge();
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(FILL, ResourceOptions::private()).unwrap();
    // Whether the fill was still running when the empty command buffer was committed, in some
    // round: the empty one then waited behind it.
    let mut behind = false;
    for round in 0..5u8 {
        let order: Arc<Mutex<Vec<&'static str>>> = Arc::default();
        let fill = queue.command_buffer().unwrap();
        let mut blit = fill.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..FILL, round).unwrap();
        blit.end_encoding().unwrap();
        let filled = Arc::clone(&order);
        fill.add_completed_handler(Box::new(move |_: &CommandBuffer| {
            filled.lock().unwrap().push("fill");
        }))
        .unwrap();
        let empty = queue.command_buffer().unwrap();
        let emptied = Arc::clone(&order);
        let fill_status = Arc::new(Mutex::new(None));
        let status = Arc::clone(&fill_status);
        let held = fill.clone();
        empty
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                emptied.lock().unwrap().push("empty");
                *status.lock().unwrap() = Some(held.status());
            }))
            .unwrap();
        let before = device.work_counts();
        fill.commit().unwrap();
        empty.commit().unwrap();
        behind |= fill.status() != CommandBufferStatus::Completed;
        empty.wait_until_completed().unwrap();
        assert_eq!(fill.status(), CommandBufferStatus::Completed);
        fill.wait_until_completed().unwrap();
        let after = device.work_counts();
        assert_eq!(*order.lock().unwrap(), ["fill", "empty"]);
        assert_eq!(
            *fill_status.lock().unwrap(),
            Some(CommandBufferStatus::Completed),
            "the empty command buffer's completed handler ran before the fill completed"
        );
        let (fill_end, start, end) = (
            fill.gpu_end_time().unwrap(),
            empty.gpu_start_time().unwrap(),
            empty.gpu_end_time().unwrap(),
        );
        assert!(
            fill_end <= start && start == end,
            "fill end {fill_end}, empty {start}..{end}"
        );
        assert!(empty.kernel_start_time() >= fill.kernel_end_time());
        let unsubmitted = after.command_buffers_unsubmitted - before.command_buffers_unsubmitted;
        let grouped = (
            after.grouped_submissions - before.grouped_submissions,
            after.batches_grouped - before.batches_grouped,
        );
        println!(
            "{completion:?}, batches grouping {groups}, round {round}: unsubmitted {unsubmitted}, \
             grouped submissions and their batches {grouped:?}"
        );
        match (unsubmitted, grouped) {
            (1, (0, 0)) => {}
            (0, (1, 2)) if groups => {}
            _ => panic!(
                "{completion:?}: the empty command buffer was neither taken with no submission \
                 nor grouped with the fill: unsubmitted {unsubmitted}, grouped {grouped:?}"
            ),
        }
        assert!(
            after.queue_submissions - before.queue_submissions >= 1,
            "the fill was not submitted"
        );
    }
    assert!(
        behind,
        "{completion:?}: the fill always completed before the empty command buffer's commit"
    );
}

/// A command buffer whose only command signals an event is submitted: the signal is the GPU's,
/// in its queue's order.
#[test]
fn a_command_buffer_that_signals_an_event_is_submitted() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let event = device.new_shared_event().unwrap();
    let before = device.work_counts();
    for value in 1..=ROUNDS {
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_signal_event(EventRef::Shared(&event), value)
            .unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(event.signaled_value(), value);
    }
    let after = device.work_counts();
    assert_eq!(
        after.command_buffers_unsubmitted - before.command_buffers_unsubmitted,
        0
    );
    assert!(after.queue_submissions - before.queue_submissions >= ROUNDS);
}
