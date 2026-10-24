//! Device tests of the scheduler's grouped submissions (`CompletionPolicy`, the runtime's
//! `command_buffer/group.rs`), on the device the cfg selects, under the validation layer.
//!
//! [`BATCHES`] command buffers committed on one queue, each filling a buffer of its own with a
//! value of its own, the first after a wait for an event the host has not raised, all wait behind
//! that wait; once the host raises it the scheduler finds them ready together. Under the default
//! `CompletionPolicy::Grouped`, on a driver measured to merge a submission that signals nothing
//! into the next one (`DriverFact::SubmissionsWithoutSignalsMerge`), they go in `vkQueueSubmit`
//! calls of up to four batches each, only the last of each signalling
//! (`WorkCounts::grouped_submissions`,
//! `WorkCounts::batches_grouped`), announced once ([`GROUPED`]); under `CompletionPolicy::Exact`,
//! and on every other driver, each alone, announcing nothing. Either way every command buffer
//! completes without error, each buffer holds its own value, and the layer reports nothing. Each
//! group counts once by what sent it (`WorkCounts::group_ends_full` and kin): the first four go
//! full; under `Exact` none is counted.
//!
//! The same with a submission check on each command buffer (`CommandBuffer::set_submission_check`,
//! as the consumer commits every one): the scheduler checks the batches right behind a line's
//! head with it under the default, so they go together too; each check runs once, in commit order.
//!
//! Command buffers committed one at a time behind running work (`behind_running_work`): a long
//! command buffer `A` of fills, then `B1` to `B7`, each a small fill, each taken in by the
//! scheduler before the next commit. Under the default on a driver that merges, `B1`, committed
//! while `A`'s submission alone runs, goes alone; `B2` stays held while those two run
//! (`WorkCounts::groups_kept_pending`), yet reads `Scheduled` at once, its scheduled handler run
//! and its kernel end time set, with no kernel submission made for it, as Metal schedules a command
//! buffer committed behind running work at once; the later ones join it until it fills and goes.
//! The last stays held in turn, and a thread's wait for it sends it at once
//! (`WorkCounts::groups_sent_for_waits`), while `A` still runs. Under `CompletionPolicy::Exact`,
//! and on a driver that does not merge, each goes alone as it is taken in, `Scheduled` as it is
//! submitted, none kept. Either way every command buffer completes with its own fill, and the layer
//! reports nothing. The same with a submission handler and a submission check on each `B`, as the
//! consumer's queue gives every command buffer: each is held and kept alike, each check runs once,
//! and each submission handler runs as its batch is taken, held or submitted, before the command
//! buffer reads `Scheduled` and before its scheduled handler.
//!
//! `grouped_completion_delays` (a measurement, not a check) prints how late the host sees each
//! command buffer complete under each `CompletionPolicy`: [`BATCHES`] command buffers of
//! [`DELAY_FILL`] bytes of fill each, released together by an event the host raises, a round at a
//! time, on a device without the validation layer whose GPU times are the device's calibrated
//! timestamps; for each place in the round, the time from its `GPUEndTime` to its completed
//! handler's start on the host's clock, over [`DELAY_ROUNDS`] rounds after [`DELAY_WARM`].

use super::degrade_notices::counts;
use super::{loader, open_device, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{CompletionPolicy, DeviceDescription, DriverFact};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, Degradation, DeviceOptions, EventRef,
    ResourceOptions, ShaderCacheOptions, WorkCounts,
};
use std::sync::{Arc, Mutex};

/// The command buffers committed behind the event's wait, the first included.
const BATCHES: usize = 8;

/// The bytes each command buffer fills.
const BYTES: u64 = 4096;

/// The kind a grouped submission announces.
pub(super) const GROUPED: [Degradation; 1] = [Degradation::GroupedCompletion];

/// The groups that went between `before` and `after`, by what sent them: full, before a wait for
/// a held signal, before a batch apart, with first work, drained, for checks or steps left, held
/// back (`WorkCounts::group_ends_full` and kin).
fn group_ends(before: &WorkCounts, after: &WorkCounts) -> [u64; 7] {
    let ends = |counts: &WorkCounts| {
        [
            counts.group_ends_full,
            counts.group_ends_wait,
            counts.group_ends_apart,
            counts.group_ends_first_work,
            counts.group_ends_drained,
            counts.group_ends_checks,
            counts.group_ends_held,
        ]
    };
    let (before, after) = (ends(before), ends(after));
    std::array::from_fn(|at| after[at] - before[at])
}

/// Whether the device's driver merges a submission that signals nothing into the next: the
/// driver's fact answers it, never its name, and the runtime reads the same fact
/// (`DriverFact::SubmissionsWithoutSignalsMerge`, the runtime's `group.rs`), so a device whose
/// identity no lane measured takes the ungrouped path and this lane expects it.
fn merges(description: &DeviceDescription) -> bool {
    (description.driver_facts()).contains(&DriverFact::SubmissionsWithoutSignalsMerge)
}

/// Wait until the scheduler's thread has taken in every command buffer committed before the call:
/// it counts a sleep (`WorkCounts::scheduler_sleeps`) only once its inbox holds no commit, so the
/// first sleep counted after the last commit returned follows the pass that took that commit in.
/// Raising a gate after this finds every command buffer behind it in the scheduler's order, where
/// a pass that took in only the first commit could otherwise see the gate raised and submit the
/// first batch alone.
fn taken_in(device: &metal2vulkan_runtime::Device) {
    let slept = device.work_counts().scheduler_sleeps;
    let start = std::time::Instant::now();
    while device.work_counts().scheduler_sleeps == slept {
        assert!(start.elapsed() < WAIT, "the scheduler's thread never slept");
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
}

/// The bytes of a host-visible buffer.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds `length` bytes at `contents`, alive with the buffer, and no command
    // buffer uses them while they are read.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// The module docs' run under `completion`, each command buffer `checked` or not: the
/// `vkQueueSubmit` calls that held several batches, the batches they held, the kernel submissions
/// in all, the grouped-completion notices, and whether the driver merges.
fn run(completion: CompletionPolicy, checked: bool) -> (u64, u64, u64, usize, bool, [u64; 7]) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (grouped, batches, submissions, merging, ends) = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.completion = completion;
        })
        .unwrap();
        let merging = merges(&device.properties().description);
        let queue = device.new_command_queue().unwrap();
        let event = device.new_shared_event().unwrap();
        let buffers: Vec<Buffer> = (0..BATCHES)
            .map(|_| device.new_buffer(BYTES, ResourceOptions::shared()).unwrap())
            .collect();
        let before = device.work_counts();
        // The command buffers whose checks ran, in the order they ran.
        let checks = Arc::new(Mutex::new(Vec::new()));
        let committed: Vec<CommandBuffer> = (buffers.iter().enumerate())
            .map(|(at, buffer)| {
                let command_buffer = queue.command_buffer().unwrap();
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
                        .encode_wait_for_event(EventRef::from(&event), 1)
                        .unwrap();
                }
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(buffer, 0..BYTES, at as u8 + 1).unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        taken_in(&device);
        event.set_signaled_value(1);
        for (at, command_buffer) in committed.iter().enumerate() {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "command buffer {at}");
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "command buffer {at}"
            );
        }
        for (at, buffer) in buffers.iter().enumerate() {
            let bytes = bytes_of(buffer);
            let wrong = bytes.iter().filter(|&&byte| byte != at as u8 + 1).count();
            assert_eq!(wrong, 0, "buffer {at}: bytes off its command buffer's fill");
        }
        let ran = checks.lock().unwrap().clone();
        let expected: Vec<usize> = (0..BATCHES).filter(|_| checked).collect();
        assert_eq!(ran, expected, "the checks that ran, in order");
        let after = device.work_counts();
        (
            after.grouped_submissions - before.grouped_submissions,
            after.batches_grouped - before.batches_grouped,
            after.queue_submissions - before.queue_submissions,
            merging,
            group_ends(&before, &after),
        )
    };
    let notices = counts(&messages.take_notices());
    let [heard] = GROUPED.map(|kind| notices.get(&kind).copied().unwrap_or(0));
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
    eprintln!(
        "GROUPED {completion:?}, checked {checked}: {grouped} vkQueueSubmit calls held {batches} batches together, \
         of {submissions} calls in all; groups by what sent them {ends:?}; {heard} \
         grouped-completion notices; the driver merges: {merging}"
    );
    (grouped, batches, submissions, heard, merging, ends)
}

/// Under the default, on a driver that merges, the command buffers the host's event let go
/// together go in calls of up to four batches, announced once; each completes with its own fill.
/// Elsewhere each goes alone, announcing nothing.
#[test]
fn ready_batches_go_in_one_kernel_submission_where_the_driver_merges() {
    grouped_where_merging(false);
}

/// The same with a submission check on each command buffer: each check runs once, in commit
/// order, those of the batches behind a head with the head's.
#[test]
fn checked_ready_batches_go_in_one_kernel_submission_where_the_driver_merges() {
    grouped_where_merging(true);
}

/// The default's run, `checked` or not, as [`ready_batches_go_in_one_kernel_submission_where_the_driver_merges`]
/// says.
fn grouped_where_merging(checked: bool) {
    let (grouped, batches, submissions, heard, merging, ends) =
        run(CompletionPolicy::Grouped, checked);
    if merging {
        assert!(grouped >= 2, "{grouped} grouped calls");
        assert!(batches >= 5, "{batches} batches grouped");
        assert!(submissions < BATCHES as u64, "{submissions} calls");
        assert_eq!(heard, 1, "grouped-completion notices");
        assert!(ends[0] >= 1, "groups by what sent them {ends:?}: none full");
        let counted: u64 = ends.iter().sum();
        assert!(
            counted >= grouped,
            "groups by what sent them {ends:?}, {grouped} calls"
        );
    } else {
        assert_eq!((grouped, batches, heard, ends), (0, 0, 0, [0; 7]));
    }
}

/// The fills one command buffer on a first queue waits for, before it, each of [`LARGE_FILL`]
/// bytes (about 0.5 ms or more of GPU each), the last signalling the event a checked command
/// buffer on a second queue waits for.
const AHEAD: usize = 8;
const LARGE_FILL: u64 = 64 << 20;

/// A checked command buffer whose event wait only a submitted GPU signal meets holds its check
/// until that signal landed (it sees the event's value reached), and meanwhile the fills that
/// complete before the signalling one wake the scheduler's thread no more than once
/// (`WorkCounts::scheduler_wakes_completion`): a completion turn wakes it only once what a head
/// waits for is met, not at each batch that completes. The first fill waits for a gate the host
/// raises once the checked command buffer is committed, so the scheduler holds the check before
/// any fill completes; a wake at every completion while a head waits would count about one a
/// fill. Under `CompletionPolicy::Exact`, so each fill is a submission of its own; every command
/// buffer completes, and the layer reports nothing.
#[test]
fn completions_wake_the_scheduler_only_once_a_head_may_go() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (seen, wakes) = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.completion = CompletionPolicy::Exact;
        })
        .unwrap();
        let (first, second) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let (event, gate) = (
            device.new_shared_event().unwrap(),
            device.new_shared_event().unwrap(),
        );
        let large = device
            .new_buffer(LARGE_FILL, ResourceOptions::private())
            .unwrap();
        let small = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
        let before = device.work_counts();
        let mut committed: Vec<CommandBuffer> = (0..AHEAD)
            .map(|at| {
                let command_buffer = first.command_buffer().unwrap();
                if at == 0 {
                    command_buffer
                        .encode_wait_for_event(EventRef::from(&gate), 1)
                        .unwrap();
                }
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&large, 0..LARGE_FILL, at as u8).unwrap();
                blit.end_encoding().unwrap();
                if at + 1 == AHEAD {
                    command_buffer
                        .encode_signal_event(EventRef::from(&event), 1)
                        .unwrap();
                }
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        let seen = Arc::new(Mutex::new(None));
        let waiting = second.command_buffer().unwrap();
        {
            let (seen, event) = (Arc::clone(&seen), event.clone());
            waiting
                .set_submission_check(Arc::new(move |_| {
                    *seen.lock().unwrap() = Some(event.signaled_value());
                    None
                }))
                .unwrap();
        }
        waiting
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = waiting.blit_command_encoder().unwrap();
        blit.fill_buffer(&small, 0..BYTES, 0x5a).unwrap();
        blit.end_encoding().unwrap();
        waiting.commit().unwrap();
        committed.push(waiting);
        gate.set_signaled_value(1);
        for (at, command_buffer) in committed.iter().enumerate() {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "command buffer {at}");
        }
        assert!(bytes_of(&small).iter().all(|&byte| byte == 0x5a));
        let after = device.work_counts();
        let seen = *seen.lock().unwrap();
        (
            seen,
            after.scheduler_wakes_completion - before.scheduler_wakes_completion,
        )
    };
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
    eprintln!("GROUPED-COMPLETION-WAKES: the check saw {seen:?}; {wakes} wakings by completions");
    assert_eq!(seen, Some(1), "the event's value the check saw");
    assert!(wakes <= 1, "{wakes} wakings by completions");
}

/// Under `CompletionPolicy::Exact` each batch goes alone, signalling its own value, announcing
/// nothing, checked or not; each command buffer completes with its own fill.
#[test]
fn the_exact_completion_submits_each_batch_alone() {
    for checked in [false, true] {
        let (grouped, batches, submissions, heard, _, ends) = run(CompletionPolicy::Exact, checked);
        assert_eq!((grouped, batches, heard, ends), (0, 0, 0, [0; 7]));
        assert!(submissions >= BATCHES as u64, "{submissions} calls");
    }
}

/// One command buffer of an event scenario: its queue (0 or 1), the value of the shared event `E`
/// it waits for first, if any, and the value it signals `E` with last, if any; between them it
/// fills `fills` times [`LARGE_FILL`] bytes of a buffer they share, then a buffer of its own. The
/// first command buffer on queue 0 waits for a gate the host raises once every one is committed,
/// so they are found ready together. The shared buffer takes its first work (a new private
/// buffer's fill) in a command buffer of its own before them: a batch that takes first work goes
/// at once, never in a group, so the first step would otherwise go alone.
#[derive(Clone, Copy)]
struct Step {
    queue: usize,
    wait: Option<u64>,
    fills: usize,
    signal: Option<u64>,
}

/// What an event scenario counted ([`event_scenario`]).
struct Scenario {
    /// The grouped `vkQueueSubmit` calls.
    grouped: u64,
    /// The batches they held.
    batches: u64,
    /// Whether the driver merges.
    merging: bool,
    /// The groups by what sent them ([`group_ends`]).
    ends: [u64; 7],
    /// The event waits found met as the scheduler took their batches
    /// (`WorkCounts::event_waits_met`), the first step's wait for the gate among them.
    met: u64,
}

/// Run `steps` under `completion` on a validated device: every command buffer completes without
/// error and fills its buffer, `E` ends at `last`, and the layer reports nothing.
fn event_scenario(completion: CompletionPolicy, steps: &[Step], last: u64) -> Scenario {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let counted = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.completion = completion;
        })
        .unwrap();
        let merging = merges(&device.properties().description);
        let queues = [
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        ];
        let gate = device.new_shared_event().unwrap();
        let event = device.new_shared_event().unwrap();
        let large = (steps.iter().any(|step| step.fills > 0)).then(|| {
            device
                .new_buffer(LARGE_FILL, ResourceOptions::private())
                .unwrap()
        });
        let buffers: Vec<Buffer> = (steps.iter())
            .map(|_| device.new_buffer(BYTES, ResourceOptions::shared()).unwrap())
            .collect();
        if let Some(large) = &large {
            let command_buffer = queues[0].command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(large, 0..LARGE_FILL, 0).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                command_buffer.error(),
                None,
                "the shared buffer's first fill"
            );
        }
        let before = device.work_counts();
        let committed: Vec<CommandBuffer> = (steps.iter().zip(&buffers).enumerate())
            .map(|(at, (step, buffer))| {
                let command_buffer = queues[step.queue].command_buffer().unwrap();
                if at == 0 {
                    command_buffer
                        .encode_wait_for_event(EventRef::from(&gate), 1)
                        .unwrap();
                }
                if let Some(value) = step.wait {
                    command_buffer
                        .encode_wait_for_event(EventRef::from(&event), value)
                        .unwrap();
                }
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                if let Some(large) = &large {
                    for fill in 0..step.fills {
                        blit.fill_buffer(large, 0..LARGE_FILL, fill as u8).unwrap();
                    }
                }
                blit.fill_buffer(buffer, 0..BYTES, at as u8 + 1).unwrap();
                blit.end_encoding().unwrap();
                if let Some(value) = step.signal {
                    command_buffer
                        .encode_signal_event(EventRef::from(&event), value)
                        .unwrap();
                }
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        taken_in(&device);
        gate.set_signaled_value(1);
        for (at, command_buffer) in committed.iter().enumerate() {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "command buffer {at}");
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "command buffer {at}"
            );
        }
        for (at, buffer) in buffers.iter().enumerate() {
            let wrong = (bytes_of(buffer).iter())
                .filter(|&&byte| byte != at as u8 + 1)
                .count();
            assert_eq!(wrong, 0, "buffer {at}: bytes off its command buffer's fill");
        }
        assert!(
            event.wait_until_signaled_value(last, WAIT),
            "E never reached {last}: {}",
            event.signaled_value()
        );
        assert_eq!(event.signaled_value(), last);
        let after = device.work_counts();
        Scenario {
            grouped: after.grouped_submissions - before.grouped_submissions,
            batches: after.batches_grouped - before.batches_grouped,
            merging,
            ends: group_ends(&before, &after),
            met: after.event_waits_met - before.event_waits_met,
        }
    };
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
    eprintln!(
        "GROUPED-EVENTS {completion:?}: {} vkQueueSubmit calls held {} batches together; groups \
         by what sent them {:?}; event waits met as taken {}; the driver merges: {}",
        counted.grouped, counted.batches, counted.ends, counted.met, counted.merging
    );
    counted
}

/// Each scenario under both policies: grouped where the driver merges under the default, alone
/// under `Exact`; where `waits`, a group went before a batch that waits for a signal it held.
fn both_policies(steps: &[Step], last: u64, waits: bool) {
    let run = event_scenario(CompletionPolicy::Grouped, steps, last);
    let (grouped, ends) = (run.grouped, run.ends);
    if run.merging {
        assert!(grouped >= 1, "{grouped} grouped calls");
        assert!(
            !waits || ends[1] >= 1,
            "groups by what sent them {ends:?}: none before a wait"
        );
    } else {
        assert_eq!((grouped, ends), (0, [0; 7]));
    }
    let run = event_scenario(CompletionPolicy::Exact, steps, last);
    assert_eq!((run.grouped, run.batches, run.ends), (0, 0, [0; 7]));
}

/// Command buffers that each signal `E` with the next value ride in groups, `E`'s signals moved to
/// each group's end; one on the same queue that waits for a value an earlier one signals goes
/// after the group that holds that signal (else the wait, behind the group's end in one kernel
/// submission, would never be met).
#[test]
fn a_wait_on_the_same_queue_for_a_held_signal_goes_after_its_group() {
    let step = |wait, signal| Step {
        queue: 0,
        wait,
        fills: 0,
        signal: Some(signal),
    };
    let steps = [
        step(None, 1),
        step(None, 2),
        step(Some(1), 3),
        step(None, 4),
        step(None, 5),
        step(Some(5), 6),
        step(None, 7),
        step(None, 8),
    ];
    both_policies(&steps, 8, true);
}

/// The same across two queues: a command buffer of another queue that waits for a value a held
/// command buffer signals is submitted after that group.
#[test]
fn a_wait_on_another_queue_for_a_held_signal_goes_after_its_group() {
    let step = |queue, wait, signal| Step {
        queue,
        wait,
        fills: 0,
        signal,
    };
    let steps = [
        step(0, None, Some(1)),
        step(0, None, Some(2)),
        step(1, Some(2), None),
        step(0, None, Some(3)),
        step(0, None, Some(4)),
        step(1, Some(4), None),
    ];
    both_policies(&steps, 4, true);
}

/// A middle value of a group's signals, waited for from another queue: met by the group's end,
/// which signals `E`'s highest index.
#[test]
fn a_middle_value_waited_for_from_another_queue_is_met_by_the_group_s_end() {
    let mut steps: Vec<Step> = (1..=BATCHES as u64)
        .map(|value| Step {
            queue: 0,
            wait: None,
            fills: 0,
            signal: Some(value),
        })
        .collect();
    steps.push(Step {
        queue: 1,
        wait: Some(3),
        fills: 0,
        signal: None,
    });
    both_policies(&steps, BATCHES as u64, false);
}

/// The fills of [`LARGE_FILL`] bytes ahead of the signal a later command buffer waits for in
/// [`a_command_buffer_that_waits_leads_its_group`]: 2 GiB written, a few milliseconds of GPU on
/// the fastest venue, so the signal is not seen complete as the scheduler decides the waiter.
const WAITED_FILLS: usize = 32;

/// A command buffer that waits for a value only a submitted GPU signal meets leads a group, never
/// joins one: the merge gathers every batch's waits before the first one's commands, so a member's
/// wait would hold back an independent command buffer committed before it until another's work
/// landed, which Metal never does (an M3 Max probe: each command buffer is seen complete 0.04 to
/// 0.10 ms after its own GPU end, whatever its queue holds after it). On one queue: `A`
/// ([`WAITED_FILLS`] fills of [`LARGE_FILL`] bytes, then signals `E` 1); `B` (waits for `E` 1); `C`;
/// `D` (waits for `E` 1, which `A`'s fills keep from completing while the scheduler decides `D`);
/// `F`. Under the default on a driver that merges, `A` leads a group and `B`'s wait for the signal
/// it holds sends it; `B` then leads a group, as nothing is held, and `C` joins it; `D`'s wait,
/// for a value submitted but not seen complete that the group does not hold, sends that group and
/// `D` goes alone after it; `F` leads the last. So two groups end at a wait
/// (`WorkCounts::group_ends_wait`), where a `D` that joined would leave one. But where the
/// scheduler takes `D` only once it sees `A`'s work complete (its thread delayed past `A`'s fills,
/// under load), `D`'s wait is met (`WorkCounts::event_waits_met`) and `D` rightly joins, as a
/// command buffer with no wait does: each of `B` and `D` ends a group at its wait or has its wait
/// met. `A`'s wait for the gate the host raised is met too. A driver that does not merge holds no
/// group, and under `CompletionPolicy::Exact` none is held.
#[test]
fn a_command_buffer_that_waits_leads_its_group() {
    let step = |wait, fills, signal| Step {
        queue: 0,
        wait,
        fills,
        signal,
    };
    let steps = [
        step(None, WAITED_FILLS, Some(1)),
        step(Some(1), 0, None),
        step(None, 0, None),
        step(Some(1), 0, None),
        step(None, 0, None),
    ];
    let run = event_scenario(CompletionPolicy::Grouped, &steps, 1);
    let (ends, met) = (run.ends, run.met);
    assert!(met >= 1, "{met} event waits met: the gate's was not");
    match run.merging {
        // `B`'s wait, for the value `A`'s held group signals, is never met as it is taken.
        true => assert!(
            ends[1] >= 1 && ends[1] + (met - 1) >= 2,
            "groups by what sent them {ends:?}, {met} event waits met (the gate's among them): a \
             command buffer that waits for a value not seen complete joined a group"
        ),
        false => assert_eq!(ends, [0; 7]),
    }
    let run = event_scenario(CompletionPolicy::Exact, &steps, 1);
    assert_eq!((run.grouped, run.batches, run.ends), (0, 0, [0; 7]));
}

/// The bytes each command buffer of `grouped_completion_delays` fills: about 0.1 to 0.2 ms of the
/// Iris Xe's GPU, a command buffer of the consumer's compositor frames.
const DELAY_FILL: u64 = 4 << 20;
/// The rounds `grouped_completion_delays` measures, after those it runs first unmeasured.
const DELAY_ROUNDS: usize = 40;
const DELAY_WARM: usize = 5;

/// The host's clock, in seconds, as the runtime's GPU times read it (`gpu_time.rs`).
fn host_seconds() -> f64 {
    #[cfg(target_os = "macos")]
    let clock = libc::CLOCK_UPTIME_RAW;
    #[cfg(not(target_os = "macos"))]
    let clock = libc::CLOCK_MONOTONIC_RAW;
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    let read = unsafe { libc::clock_gettime(clock, &mut now) };
    assert_eq!(read, 0, "the host's clock");
    now.tv_sec as f64 + now.tv_nsec as f64 / 1e9
}

/// One round's command buffers' delays from `GPUEndTime` to their completed handlers, and their GPU
/// times, in microseconds, by place.
fn delay_round(device: &metal2vulkan_runtime::Device, buffers: &[Buffer]) -> Vec<(f64, f64)> {
    let queue = device.new_command_queue().unwrap();
    let event = device.new_shared_event().unwrap();
    let handled: Vec<Arc<Mutex<f64>>> = buffers.iter().map(|_| Arc::default()).collect();
    let committed: Vec<CommandBuffer> = (buffers.iter().zip(&handled).enumerate())
        .map(|(at, (buffer, handled))| {
            let command_buffer = queue.command_buffer().unwrap();
            if at == 0 {
                command_buffer
                    .encode_wait_for_event(EventRef::from(&event), 1)
                    .unwrap();
            }
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(buffer, 0..DELAY_FILL, at as u8 + 1)
                .unwrap();
            blit.end_encoding().unwrap();
            let handled = Arc::clone(handled);
            command_buffer
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    *handled.lock().unwrap() = host_seconds();
                }))
                .unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        })
        .collect();
    event.set_signaled_value(1);
    for command_buffer in &committed {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }
    // The handlers run on the queue's context: the last one's end follows its command buffer's.
    let deadline = std::time::Instant::now() + WAIT;
    while handled.iter().any(|at| *at.lock().unwrap() == 0.0) {
        assert!(
            std::time::Instant::now() < deadline,
            "a completed handler never ran"
        );
        std::thread::yield_now();
    }
    (committed.iter().zip(&handled))
        .map(|(command_buffer, handled)| {
            let (start, end) = (
                command_buffer.gpu_start_time().unwrap(),
                command_buffer.gpu_end_time().unwrap(),
            );
            ((*handled.lock().unwrap() - end) * 1e6, (end - start) * 1e6)
        })
        .collect()
}

/// The module docs' measurement: Exact, then Grouped, twice each in turn.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn grouped_completion_delays() {
    for completion in [CompletionPolicy::Exact, CompletionPolicy::Grouped].repeat(2) {
        let cache = ScratchCache::new().unwrap();
        let mut options = DeviceOptions::default();
        options.policy.completion = completion;
        options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
        options.loader = loader();
        let device = open_device(TEST_DEVICE, options).unwrap();
        let name = device.name().to_owned();
        let buffers: Vec<Buffer> = (0..BATCHES)
            .map(|_| {
                device
                    .new_buffer(DELAY_FILL, ResourceOptions::private())
                    .unwrap()
            })
            .collect();
        let before = device.work_counts();
        let mut rounds = Vec::new();
        for round in 0..DELAY_WARM + DELAY_ROUNDS {
            let delays = delay_round(&device, &buffers);
            if round >= DELAY_WARM {
                rounds.push(delays);
            }
        }
        let after = device.work_counts();
        let percentile = |mut values: Vec<f64>, at: f64| {
            values.sort_by(f64::total_cmp);
            values[((values.len() - 1) as f64 * at).round() as usize]
        };
        let places: Vec<String> = (0..BATCHES)
            .map(|place| {
                let delays: Vec<f64> = rounds.iter().map(|round| round[place].0).collect();
                let gpu: Vec<f64> = rounds.iter().map(|round| round[place].1).collect();
                format!(
                    "{place}: {:.0} us (p90 {:.0}; GPU {:.0})",
                    percentile(delays.clone(), 0.5),
                    percentile(delays, 0.9),
                    percentile(gpu, 0.5),
                )
            })
            .collect();
        let rounds_measured = (DELAY_WARM + DELAY_ROUNDS) as f64;
        eprintln!(
            "GROUPED-DELAY {name} {completion:?}: from GPUEndTime to the completed handler, the \
             median by place in a round of {BATCHES}: {}; {:.1} grouped vkQueueSubmit calls a \
             round holding {:.1} batches, of {:.1}",
            places.join(", "),
            (after.grouped_submissions - before.grouped_submissions) as f64 / rounds_measured,
            (after.batches_grouped - before.batches_grouped) as f64 / rounds_measured,
            (after.queue_submissions - before.queue_submissions) as f64 / rounds_measured,
        );
    }
}

/// The bytes `A`'s each fill writes in [`behind_running_work`]: a private buffer of its own.
const RUNNING_FILL: u64 = 128 << 20;

/// The command buffers committed behind `A` in [`behind_running_work`].
const BEHIND: usize = 7;

/// The most runs [`behind_while_a_runs`] makes, `A` twice as long each time.
const BEHIND_RUNS: u32 = 4;

/// The fills of [`RUNNING_FILL`] bytes that keep the device's GPU busy for at least `least`: a
/// command buffer of fills doubled until one takes `least` from its commit to its completion
/// (the GPU's time then outweighs the submission's), at most 1 << 14 of them, then thrice that.
fn running_fills(
    queue: &metal2vulkan_runtime::CommandQueue,
    large: &Buffer,
    least: std::time::Duration,
) -> usize {
    let mut fills = 4;
    loop {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for fill in 0..fills {
            blit.fill_buffer(large, 0..RUNNING_FILL, fill as u8)
                .unwrap();
        }
        blit.end_encoding().unwrap();
        let start = std::time::Instant::now();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{fills} fills");
        if start.elapsed() >= least || fills >= 1 << 14 {
            return fills * 3;
        }
        fills *= 2;
    }
}

/// What each command buffer behind `A` carries in [`behind_running_work`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carried {
    /// A scheduled handler.
    Plain,
    /// A submission handler and a submission check besides, as the consumer's queue gives every
    /// command buffer.
    Consumer,
}

/// What one command buffer behind `A` showed once the scheduler took it in
/// ([`behind_running_work`]).
#[derive(Debug, Default)]
struct Taken {
    /// Groups kept, and kernel submissions made, since the one before was taken in.
    kept: u64,
    submissions: u64,
    /// It read `Scheduled` (or later), its scheduled handler had run, and its kernel end time was
    /// set; `A` still ran.
    scheduled: bool,
    handled: bool,
    kernel_end: bool,
    a_ran: bool,
    /// Where it carried a submission handler ([`Carried::Consumer`]): whether that ran before
    /// the command buffer read `Scheduled` and before its scheduled handler; `None` where it ran
    /// not at all.
    submission_first: Option<bool>,
}

/// What [`behind_running_work`] saw under one `CompletionPolicy`.
#[derive(Debug)]
struct Behind {
    /// Whether the driver merges.
    merging: bool,
    /// Each command buffer behind `A`, in commit order.
    taken: Vec<Taken>,
    /// The groups kept for their queue's earlier work, and those a wait sent.
    kept: u64,
    sent: u64,
    /// `A` still ran as the wait for the last sent its group, where it did.
    a_ran_at_send: bool,
    /// The groups by what sent them ([`group_ends`]).
    ends: [u64; 7],
    /// The submission checks run.
    checks: u64,
}

/// The module docs' run under `completion`, `A` of `stretch` times the fills its calibration
/// takes, each command buffer behind it carrying `carried`.
fn behind_running_work(completion: CompletionPolicy, stretch: usize, carried: Carried) -> Behind {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let seen = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.completion = completion;
        })
        .unwrap();
        let merging = merges(&device.properties().description);
        let queue = device.new_command_queue().unwrap();
        let large = device
            .new_buffer(RUNNING_FILL, ResourceOptions::private())
            .unwrap();
        // Its first fill alone (first work goes at once), then enough to run 300 ms or more.
        let fills = running_fills(&queue, &large, std::time::Duration::from_millis(100)) * stretch;
        let buffers: Vec<Buffer> = (0..BEHIND)
            .map(|_| device.new_buffer(BYTES, ResourceOptions::shared()).unwrap())
            .collect();
        let checks = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let before = device.work_counts();
        let running = queue.command_buffer().unwrap();
        let mut blit = running.blit_command_encoder().unwrap();
        for fill in 0..fills {
            blit.fill_buffer(&large, 0..RUNNING_FILL, fill as u8)
                .unwrap();
        }
        blit.end_encoding().unwrap();
        running.commit().unwrap();
        taken_in(&device);
        let mut last = device.work_counts();
        let mut taken = Vec::new();
        let committed: Vec<CommandBuffer> = (buffers.iter().enumerate())
            .map(|(at, buffer)| {
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(buffer, 0..BYTES, at as u8 + 1).unwrap();
                blit.end_encoding().unwrap();
                let handled = Arc::new(std::sync::atomic::AtomicBool::new(false));
                {
                    let handled = Arc::clone(&handled);
                    command_buffer
                        .add_scheduled_handler(Box::new(move |_: &CommandBuffer| {
                            handled.store(true, std::sync::atomic::Ordering::SeqCst);
                        }))
                        .unwrap();
                }
                // 0 until its submission handler ran; then 1 where it ran first, else 2.
                let submitted = Arc::new(std::sync::atomic::AtomicU8::new(0));
                if carried == Carried::Consumer {
                    let (submitted, handled) = (Arc::clone(&submitted), Arc::clone(&handled));
                    command_buffer
                        .add_submission_handler(Box::new(move |taken: &CommandBuffer| {
                            let first = taken.status() == CommandBufferStatus::Committed
                                && !handled.load(std::sync::atomic::Ordering::SeqCst);
                            let seen = if first { 1 } else { 2 };
                            submitted.store(seen, std::sync::atomic::Ordering::SeqCst);
                        }))
                        .unwrap();
                    let checks = Arc::clone(&checks);
                    command_buffer
                        .set_submission_check(Arc::new(move |_| {
                            checks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            None
                        }))
                        .unwrap();
                }
                command_buffer.commit().unwrap();
                taken_in(&device);
                let counts = device.work_counts();
                // Its scheduled handler runs on the queue's context, at once where its transition
                // was posted as it was submitted or held.
                command_buffer.wait_until_scheduled().unwrap();
                taken.push(Taken {
                    kept: counts.groups_kept_pending - last.groups_kept_pending,
                    submissions: counts.queue_submissions - last.queue_submissions,
                    scheduled: command_buffer.status() != CommandBufferStatus::Committed,
                    handled: handled.load(std::sync::atomic::Ordering::SeqCst),
                    kernel_end: command_buffer.kernel_end_time() > 0.0,
                    a_ran: running.status() != CommandBufferStatus::Completed,
                    submission_first: match submitted.load(std::sync::atomic::Ordering::SeqCst) {
                        0 => None,
                        seen => Some(seen == 1),
                    },
                });
                last = counts;
                command_buffer
            })
            .collect();
        // The last one's wait, on a thread of its own, while this one watches for the group it
        // sends.
        let waited = committed.last().unwrap();
        let sent_before = device.work_counts().groups_sent_for_waits;
        let mut a_ran_at_send = false;
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| waited.wait_until_completed_for(WAIT));
            let deadline = std::time::Instant::now() + WAIT;
            loop {
                if device.work_counts().groups_sent_for_waits > sent_before {
                    a_ran_at_send = running.status() != CommandBufferStatus::Completed;
                    break;
                }
                if waiter.is_finished() || std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::yield_now();
            }
            waiter.join().unwrap().unwrap();
        });
        running.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(running.error(), None, "A");
        for (at, command_buffer) in committed.iter().enumerate() {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None, "B{}", at + 1);
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "B{}",
                at + 1
            );
        }
        for (at, buffer) in buffers.iter().enumerate() {
            let wrong = (bytes_of(buffer).iter())
                .filter(|&&byte| byte != at as u8 + 1)
                .count();
            assert_eq!(wrong, 0, "B{}: bytes off its fill", at + 1);
        }
        let after = device.work_counts();
        Behind {
            merging,
            taken,
            kept: after.groups_kept_pending - before.groups_kept_pending,
            sent: after.groups_sent_for_waits - before.groups_sent_for_waits,
            a_ran_at_send,
            ends: group_ends(&before, &after),
            checks: checks.load(std::sync::atomic::Ordering::SeqCst),
        }
    };
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
    eprintln!("GROUPED-BEHIND {completion:?} {carried:?}: {seen:#?}");
    seen
}

/// [`behind_running_work`] under the default until, on a driver that merges, `A` outlasted every
/// command buffer behind it and the wait that sends the last group: a loaded CPU device (lavapipe
/// beside other runs) may run `A` sooner than its calibration measured. `A` is twice as long at
/// each run, at most [`BEHIND_RUNS`] runs; the last run is returned whatever it saw. On a driver
/// that does not merge nothing is held, and one run serves.
fn behind_while_a_runs(carried: Carried) -> Behind {
    let mut stretch = 1;
    loop {
        let seen = behind_running_work(CompletionPolicy::Grouped, stretch, carried);
        let outlasted = seen.taken.iter().all(|taken| taken.a_ran) && seen.a_ran_at_send;
        if !seen.merging || outlasted || stretch >= 1 << (BEHIND_RUNS - 1) {
            return seen;
        }
        stretch *= 2;
        eprintln!("GROUPED-BEHIND: A ended before the run did: again, A {stretch} times as long");
    }
}

/// Every command buffer behind `A` read `Scheduled` as it was taken in, its scheduled handler run
/// and its kernel end time set; where it carried a submission handler and a check, the handler ran
/// before the command buffer read `Scheduled` and before its scheduled handler, and the check ran
/// once.
fn each_scheduled_as_taken_in(seen: &Behind, carried: Carried) {
    for (at, taken) in seen.taken.iter().enumerate() {
        assert!(
            taken.scheduled && taken.handled && taken.kernel_end,
            "B{}: {taken:?}",
            at + 1
        );
        let first = (carried == Carried::Consumer).then_some(true);
        assert_eq!(taken.submission_first, first, "B{}: {taken:?}", at + 1);
    }
    let checks = match carried {
        Carried::Plain => 0,
        Carried::Consumer => BEHIND as u64,
    };
    assert_eq!(seen.checks, checks, "{seen:?}");
}

/// Under the default, on a driver that merges: the first command buffer the scheduler keeps
/// held behind `A` (while `A`'s submission and one more run) reads `Scheduled` with no kernel
/// submission made for it; the group it leads fills and goes; the last one is kept in turn, and
/// the wait for it sends it while `A` still runs, `A` made longer until it outlasts the run
/// ([`behind_while_a_runs`]).
#[test]
fn a_command_buffer_committed_behind_running_work_is_scheduled_while_its_group_is_held() {
    held_behind_running_work(Carried::Plain);
}

/// The same, each command buffer behind `A` carrying a submission handler and a submission check,
/// as the consumer's queue gives every command buffer: its handler runs as its batch is held,
/// before it reads `Scheduled`, so the group is held and kept as without them.
#[test]
fn a_command_buffer_with_a_submission_handler_and_check_is_held_behind_running_work() {
    held_behind_running_work(Carried::Consumer);
}

/// The two tests above, each command buffer behind `A` carrying `carried`.
fn held_behind_running_work(carried: Carried) {
    let seen = behind_while_a_runs(carried);
    each_scheduled_as_taken_in(&seen, carried);
    if seen.merging {
        assert!(
            seen.taken.iter().all(|taken| taken.a_ran),
            "A ended before a command buffer behind it was taken in, at every run: {seen:?}"
        );
        let first = (seen.taken.iter().position(|taken| taken.kept == 1))
            .unwrap_or_else(|| panic!("no group kept: {seen:?}"));
        assert_eq!(seen.taken[first].submissions, 0, "{seen:?}");
        assert_eq!((seen.kept, seen.sent), (2, 1), "{seen:?}");
        assert!(seen.ends[0] >= 1, "no full group: {seen:?}");
        assert!(
            seen.a_ran_at_send,
            "A completed before the wait sent the last: {seen:?}"
        );
    } else {
        assert_eq!(
            (seen.kept, seen.sent, seen.ends),
            (0, 0, [0; 7]),
            "{seen:?}"
        );
    }
}

/// Under `CompletionPolicy::Exact` each goes alone as it is taken in, `Scheduled` as it is
/// submitted; no group is held or kept.
#[test]
fn the_exact_completion_keeps_no_group_behind_running_work() {
    let seen = behind_running_work(CompletionPolicy::Exact, 1, Carried::Plain);
    each_scheduled_as_taken_in(&seen, Carried::Plain);
    for (at, taken) in seen.taken.iter().enumerate() {
        assert_eq!(taken.submissions, 1, "B{}: {seen:?}", at + 1);
    }
    assert_eq!(
        (seen.kept, seen.sent, seen.ends),
        (0, 0, [0; 7]),
        "{seen:?}"
    );
}
