//! Device tests of command buffers marked to join the next under `CompletionPolicy::Exact`
//! (`CommandBuffer::set_completion_joins_next`, the runtime's `command_buffer/group.rs`), on the
//! device the cfg selects, under the validation layer and the lane's exact completion.
//!
//! Shaped like a consumer's command buffer cut into parts: [`PARTS`] command buffers committed on
//! one queue, the first waiting for an event the host raises once every one was taken in, each a
//! fill of a buffer of its own, every one but the last marked, each with a submission check where
//! `checked` (the consumer checks every command buffer). Let go together, on a device whose
//! batches merge ([`groups`]: its merge check holds, here assumed with
//! `DeviceOptions::assume_submissions_merge`, and its queues' failures are not relayed), the marked
//! ones wait in the scheduler's group and the last goes last in one of them, which then goes at
//! once (`WorkCounts::group_ends_closed`): fewer kernel submissions than command buffers,
//! `grouped-completion` announced once. With none marked, or on a device whose check finds no merge
//! (lavapipe's, unassumed), each goes alone and nothing is announced. Either way every command
//! buffer completes with its own fill, each check runs once in commit order, and the layer reports
//! nothing.
//!
//! An unmarked command buffer is never seen complete late ([`an_unmarked_command_buffer_is_seen_complete_at_its_own_end`]):
//! a marked `M`, an unmarked `U` and then `L`, [`LONG_FILLS`] large fills, let go together. `U`
//! goes last in `M`'s group, so its completed handler runs while `L` still runs on the GPU; were `U`
//! held before `L` in a group, it would be seen complete only once `L` completed. `M`, seen complete
//! with its group's last, has completed by the time `U`'s handler runs, and before `L` completes.
//! Whether `U` joins `M`'s group depends on the scheduler's pass finding `U` ready with `M`, which
//! a loaded host may not (`M` then goes alone, `WorkCounts::batches_held_alone_unkept`): the test
//! asserts the order, and that `M` either went in `U`'s submission or alone for a reason counted.

use super::degrade_notices::counts;
use super::grouped_submissions::{bytes_of, groups, taken_in, warm, GROUPED};
use super::{
    at_plan, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, EventRef, ResourceOptions, WorkCounts,
};
use std::sync::{Arc, Mutex};

/// The parts of the cut command buffer: two groups of at most four, the second closed by the last.
const PARTS: usize = 7;

/// The bytes each part fills.
const BYTES: u64 = 4096;

/// The large fills `L` runs, each of [`LONG_BYTES`]: many milliseconds of GPU work on every lane
/// device, so `U`'s completion is seen well before `L`'s end.
const LONG_FILLS: usize = 16;
const LONG_BYTES: u64 = 64 << 20;

/// What a run moved: the `vkQueueSubmit` calls that held several batches, the batches they held,
/// the kernel submissions in all, the groups an unmarked batch closed, the groups that went before
/// a batch their summed expected runs left no room for, the batches that went alone because no
/// group is kept under `Exact` or because their queue's expected run was past a group's bound, the
/// grouped-completion notices, and whether the device groups marked batches ([`groups`]).
#[derive(Debug)]
struct Moved {
    grouped: u64,
    batches: u64,
    submissions: u64,
    closed: u64,
    ends_long: u64,
    unkept: u64,
    long: u64,
    heard: usize,
    grouping: bool,
}

impl Moved {
    fn of(before: &WorkCounts, after: &WorkCounts, heard: usize, grouping: bool) -> Moved {
        Moved {
            grouped: after.grouped_submissions - before.grouped_submissions,
            batches: after.batches_grouped - before.batches_grouped,
            submissions: after.queue_submissions - before.queue_submissions,
            closed: after.group_ends_closed - before.group_ends_closed,
            ends_long: after.group_ends_long - before.group_ends_long,
            unkept: after.batches_held_alone_unkept - before.batches_held_alone_unkept,
            long: after.batches_alone_run_long - before.batches_alone_run_long,
            heard,
            grouping,
        }
    }
}

/// The module docs' parts, every one but the last marked where `marked`, each checked where
/// `checked`, on a device whose merge check is `assumed` to hold or measured.
fn parts(marked: bool, checked: bool, assumed: bool) -> Moved {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (before, after, grouping) = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.assume_submissions_merge = assumed;
        })
        .unwrap();
        let grouping = groups(&device);
        let queue = device.new_command_queue().unwrap();
        warm(&device, &queue);
        let gate = device.new_shared_event().unwrap();
        let buffers: Vec<Buffer> = (0..PARTS)
            .map(|_| device.new_buffer(BYTES, ResourceOptions::shared()).unwrap())
            .collect();
        let before = device.work_counts();
        let checks = Arc::new(Mutex::new(Vec::new()));
        let committed: Vec<CommandBuffer> = (buffers.iter().enumerate())
            .map(|(at, buffer)| {
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .set_completion_joins_next(marked && at + 1 < PARTS)
                    .unwrap();
                if checked {
                    let checks = Arc::clone(&checks);
                    command_buffer
                        .set_submission_check(Arc::new(move |_| {
                            checks.lock().unwrap().push(at);
                            None
                        }))
                        .unwrap();
                }
                if at == 0 {
                    command_buffer
                        .encode_wait_for_event(EventRef::from(&gate), 1)
                        .unwrap();
                }
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(buffer, 0..BYTES, at as u8 + 1).unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                assert!(
                    command_buffer.set_completion_joins_next(true).is_err(),
                    "part {at} marked once committed"
                );
                command_buffer
            })
            .collect();
        taken_in(&device);
        gate.set_signaled_value(1);
        for (at, command_buffer) in committed.iter().enumerate() {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "part {at}");
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        for (at, buffer) in buffers.iter().enumerate() {
            let wrong = (bytes_of(buffer).iter())
                .filter(|&&byte| byte != at as u8 + 1)
                .count();
            assert_eq!(wrong, 0, "part {at}: bytes off its fill");
        }
        let ran = checks.lock().unwrap().clone();
        let expected: Vec<usize> = (0..PARTS).filter(|_| checked).collect();
        assert_eq!(ran, expected, "the checks that ran, in order");
        (before, device.work_counts(), grouping)
    };
    let notices = counts(&messages.take_notices());
    let [heard] = GROUPED.map(|kind| notices.get(&kind).copied().unwrap_or(0));
    layer_reported_nothing_at_all(&messages);
    let moved = Moved::of(&before, &after, heard, grouping);
    eprintln!(
        "JOINED marked {marked}, checked {checked}, merge assumed {assumed}: {moved:?} for {PARTS} \
         command buffers"
    );
    moved
}

/// Marked parts group where the device's batches merge, the last unmarked one closing a group;
/// elsewhere each goes alone. A group holds batches whose queue's expected runs sum to at most
/// 16 ms (the runtime's `GROUP_RUN`): where the queue's run is longer than a quarter of that, the
/// parts group in threes or pairs, and the last finds no room in the group before it, which goes
/// first (`WorkCounts::group_ends_long`), the last then alone. On lavapipe beside the lane's tests
/// [`warm`]'s fills ran 3 to 14 ms and the parts went in pairs (job m3-max-40006).
fn grouped_where_merging(checked: bool, assumed: bool) {
    let moved = parts(true, checked, assumed);
    if moved.grouping {
        assert!(moved.grouped >= 1, "{moved:?}: no grouped call");
        assert!(moved.batches >= 3, "{moved:?}: too few batches grouped");
        assert!(moved.submissions < PARTS as u64, "{moved:?}: no call saved");
        assert!(
            moved.closed >= 1 || moved.ends_long >= 1,
            "{moved:?}: no group closed by the last, and none went for want of room"
        );
        assert_eq!(moved.heard, 1, "{moved:?}: grouped-completion notices");
    } else {
        assert_eq!(
            (moved.grouped, moved.batches, moved.closed, moved.heard),
            (0, 0, 0, 0),
            "{moved:?}"
        );
        assert_eq!(moved.submissions, PARTS as u64, "{moved:?}");
    }
}

/// The module docs' parts, marked, unchecked, the merge assumed: they group.
#[test]
fn marked_parts_go_in_fewer_kernel_submissions() {
    grouped_where_merging(false, true);
}

/// The same with a submission check on each part: the checks of the parts right behind a marked
/// head run with its own, each once, in commit order, so they are found ready together.
#[test]
fn checked_marked_parts_go_in_fewer_kernel_submissions() {
    grouped_where_merging(true, true);
}

/// The same on the device's own merge check: a no-op where it finds no merge (lavapipe).
#[test]
fn marked_parts_group_only_where_the_device_s_check_finds_the_merge() {
    grouped_where_merging(true, false);
}

/// With no part marked, under the exact completion, each goes alone: nothing grouped, nothing
/// announced, one kernel submission each.
#[test]
fn unmarked_parts_go_alone_under_the_exact_completion() {
    let moved = parts(false, true, true);
    assert_eq!(
        (
            moved.grouped,
            moved.batches,
            moved.closed,
            moved.unkept,
            moved.heard
        ),
        (0, 0, 0, 0, 0),
        "{moved:?}"
    );
    assert_eq!(moved.submissions, PARTS as u64, "{moved:?}");
}

/// The module docs' `M`, `U` and `L`: `U`'s completion is seen while `L` still runs, `M`'s by then
/// too; where the device groups, `M` and `U` went in one kernel submission that `U` closed, or
/// `M` went alone for a reason the runtime counts, and nothing else moved.
#[test]
fn an_unmarked_command_buffer_is_seen_complete_at_its_own_end() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (moved, seen) = {
        // Its work places a second command queue (`aside`): it takes its queue plan (`at_plan`).
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.assume_submissions_merge = true;
            at_plan(options);
        })
        .unwrap();
        let grouping = groups(&device);
        let queue = device.new_command_queue().unwrap();
        warm(&device, &queue);
        let gate = device.new_shared_event().unwrap();
        let small = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
        let large = device
            .new_buffer(LONG_BYTES, ResourceOptions::private())
            .unwrap();
        // Their first work (`texture/first_move.rs`) done before: a batch that takes it waits in no
        // group. On a queue of its own: a long fill on `queue` would raise its expected run past
        // the group's bound (the runtime's `GROUP_RUN`), and `M` would go alone.
        let aside = device.new_command_queue().unwrap();
        let first = aside.command_buffer().unwrap();
        let mut blit = first.blit_command_encoder().unwrap();
        blit.fill_buffer(&small, 0..BYTES, 0).unwrap();
        blit.fill_buffer(&large, 0..LONG_BYTES, 0).unwrap();
        blit.end_encoding().unwrap();
        first.commit().unwrap();
        first.wait_until_completed_for(WAIT).unwrap();
        let before = device.work_counts();
        let fill = |command_buffer: &CommandBuffer, value: u8| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&small, 0..BYTES, value).unwrap();
            blit.end_encoding().unwrap();
        };
        let marked = queue.command_buffer().unwrap();
        marked.set_completion_joins_next(true).unwrap();
        marked
            .encode_wait_for_event(EventRef::from(&gate), 1)
            .unwrap();
        fill(&marked, 1);
        let unmarked = queue.command_buffer().unwrap();
        fill(&unmarked, 2);
        let long = queue.command_buffer().unwrap();
        let mut blit = long.blit_command_encoder().unwrap();
        for at in 0..LONG_FILLS {
            blit.fill_buffer(&large, 0..LONG_BYTES, at as u8).unwrap();
        }
        blit.end_encoding().unwrap();
        // At `U`'s completed handler: whether `M` and `L` read `Completed`.
        let seen = Arc::new(Mutex::new(None));
        {
            let (seen, marked, long) = (Arc::clone(&seen), marked.clone(), long.clone());
            unmarked
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    let done = |command_buffer: &CommandBuffer| {
                        command_buffer.status() == CommandBufferStatus::Completed
                    };
                    *seen.lock().unwrap() = Some((done(&marked), done(&long)));
                }))
                .unwrap();
        }
        for command_buffer in [&marked, &unmarked, &long] {
            command_buffer.commit().unwrap();
        }
        taken_in(&device);
        gate.set_signaled_value(1);
        for command_buffer in [&marked, &unmarked, &long] {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        }
        assert!(bytes_of(&small).iter().all(|&byte| byte == 2));
        let after = device.work_counts();
        let seen = *seen.lock().unwrap();
        (Moved::of(&before, &after, 0, grouping), seen)
    };
    layer_reported_nothing_at_all(&messages);
    eprintln!("JOINED-ORDER {moved:?}; at U's completed handler (M, L) completed: {seen:?}");
    assert_eq!(
        seen,
        Some((true, false)),
        "at U's completed handler, M complete and L still running"
    );
    assert_eq!(
        moved.submissions + moved.grouped,
        3,
        "{moved:?}: M, U and L in three kernel submissions, or M and U in one"
    );
    if moved.grouping && moved.grouped == 1 {
        assert_eq!(
            (moved.batches, moved.closed),
            (2, 1),
            "{moved:?}: U closes M's group"
        );
    } else if moved.grouping {
        assert_eq!(moved.unkept + moved.long, 1, "{moved:?}: why M went alone");
    } else {
        assert_eq!((moved.grouped, moved.closed), (0, 0), "{moved:?}");
    }
}
