//! A batch the scheduler holds whose wait is met goes, whether or not the wake for it came
//! (`scheduler/mod.rs`, "Idle, it sleeps until it is woken" and "Woken by completions only when
//! they let a batch go"). Each test holds the runtime's threads at fixed points
//! (`completion::trace::hold_at`) to force the interleaving a loaded host makes now and then.
//!
//! - **A kept group's completion, published while it was kept, sends it.** Two submissions of a
//!   `VkQueue` run, their completions not yet published, and a third command buffer's batch is held
//!   in a group past its pass (`Group::keeps`). Between `keeps`' read of the completed value and the
//!   gate it leaves the completion turns, both completions are published and their turns run: they
//!   find no gate, and wake nothing. The scheduler reads the value again once it left the gate,
//!   and sends the group in that pass: no pass after a slice lets it go
//!   (`WorkCounts::scheduler_slice_rescues`).
//! - **A held batch goes at the next slice where its wake never came.** The same group, kept with
//!   its gate left and the scheduler asleep; the completion turns wake it for no met wait
//!   (`completion::trace::drop_completion_wakes`). The pass after the next sleep that ends at
//!   `WAIT_SLICE` sends it, counted as a slice's rescue.
//! - **A batch held back for an unshared queue goes at the completion published as it was looked
//!   at** (`unshared_hold.rs`). On a `VkQueue` an unshared queue holds, a queue's first command
//!   buffer, of unknown length, runs; its completion is published, its batch not yet out of the
//!   `VkQueue`'s flight ([`Point::Leave`]), as the queue's next is committed. The hold reads the
//!   published value, so the next goes in that pass, or its gate is one that completion met; it
//!   never waits for a slice's pass.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects; the kept groups' batches group by
//! `DeviceOptions::assume_submissions_merge`.

use super::plan_room::device_with_room;
use crate::command_buffer::completion::trace::{self, Point};
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    ResourceOptions,
};
use metal2vulkan::device::QueueSharingPolicy;
use std::time::{Duration, Instant};

/// How long a held thread, or a command buffer, is waited for.
const BOUND: Duration = Duration::from_secs(5);

/// How long the device's own checks are waited for before a test.
const SETTLE: Duration = Duration::from_secs(120);

/// The device the cfg selects, its batches grouping where `merge`: a CPU device under
/// `m2v_cpu_icd`, else a GPU. Its command queues stay on the `VkQueue`s they were made on
/// (`QueueSharingPolicy::PerQueue`). Its own checks have answered: the in-place check commits
/// command buffers of its own (`in_place_check.rs`), which a pass of the scheduler's takes in, and
/// a pass after a slice that takes a commit in is no rescue; and the submission checks hold its
/// first `VkQueue` for their calls (`submit_check.rs`). Its queues are the plan's
/// (`queue_limits: None`): the held batch's test makes its unshared queue while it holds two
/// others, which on the runtime tests' default limits would share a `VkQueue` with one of them
/// (`create::needs_the_queue_plan`).
fn device(merge: bool) -> Device {
    let mut options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        assume_submissions_merge: merge,
        queue_limits: None,
        ..DeviceOptions::default()
    };
    options.policy.queue_sharing = QueueSharingPolicy::PerQueue;
    let device = device_with_room(&options).unwrap();
    // False where the device runs no in-place check (its policy keeps no texture in place).
    let _ = device.settle_in_place_checks(SETTLE);
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    device
}

/// A command buffer of `queue` that fills `target` whole with `value`.
fn fill(queue: &CommandQueue, target: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..target.length(), value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// The points held, each let go as this drops, a failing test's unwinding too.
struct Held(Vec<Point>);

impl Held {
    fn at(&mut self, point: Point) {
        trace::hold_at(point);
        self.0.push(point);
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        for &point in &self.0 {
            trace::release(point);
        }
        trace::drop_completion_wakes(false);
    }
}

/// On `device`'s `queue`, a group kept past its pass: two command buffers submitted, each alone
/// (waited for as scheduled), whose completions the queue waiters hold unpublished
/// ([`Point::Publish`]); then a third, `kept`, committed, whose batch the scheduler holds in a
/// group and keeps, held at [`Point::KeptGate`]. The two submitted first.
fn kept_group(
    queue: &CommandQueue,
    target: &Buffer,
    held: &mut Held,
) -> (CommandBuffer, CommandBuffer, CommandBuffer) {
    held.at(Point::Publish);
    let first = fill(queue, target, 1);
    first.commit().unwrap();
    first.wait_until_scheduled().unwrap();
    let second = fill(queue, target, 2);
    second.commit().unwrap();
    second.wait_until_scheduled().unwrap();
    held.at(Point::KeptGate);
    let kept = fill(queue, target, 3);
    kept.commit().unwrap();
    assert!(
        trace::reached(Point::KeptGate, BOUND),
        "the third command buffer's group was not kept past its pass"
    );
    (first, second, kept)
}

/// Poll `command_buffer`'s status, never waiting for it (a waiter sends a kept group), until it
/// completed, at most [`BOUND`]; how long that took.
fn completes(command_buffer: &CommandBuffer, what: &str) -> Duration {
    let start = Instant::now();
    while command_buffer.status() != CommandBufferStatus::Completed {
        assert!(start.elapsed() < BOUND, "{what} never completed");
        std::thread::sleep(Duration::from_millis(1));
    }
    start.elapsed()
}

/// The first and the warm-up: `queue`'s first work, completed, untimed, then five fills more in
/// turn. A queue's batches wait in a group only within 16 ms of their queue's expected run
/// (`command_buffer/group.rs`, `GROUP_RUN`), which a first run takes at once and later shorter ones
/// halve toward theirs (`unshared_hold::Expected`): on lavapipe a first fill ran past 16 ms, and
/// five halvings bring a first run of up to about 500 ms under it.
fn warm(queue: &CommandQueue, target: &Buffer) {
    for _ in 0..6 {
        let first = fill(queue, target, 0);
        first.commit().unwrap();
        first.wait_until_completed().unwrap();
    }
}

#[test]
fn a_kept_group_whose_completion_was_published_as_it_was_kept_goes_in_that_pass() {
    let device = device(true);
    let queue = device.new_command_queue().unwrap();
    let target = device
        .new_buffer(1 << 16, ResourceOptions::private())
        .unwrap();
    warm(&queue, &target);
    let before = device.work_counts();
    let mut held = Held(Vec::new());
    let (first, second, kept) = kept_group(&queue, &target, &mut held);
    // Both completions published, and their turns' looks for a met gate made (each before its
    // command buffer completes), while the scheduler has yet to leave the group's gate.
    trace::release(Point::Publish);
    completes(&first, "the first command buffer");
    completes(&second, "the second command buffer");
    trace::release(Point::KeptGate);
    let took = completes(&kept, "the kept command buffer");
    let after = device.work_counts();
    let rescues = after.scheduler_slice_rescues - before.scheduler_slice_rescues;
    let kept_groups = after.groups_kept_pending - before.groups_kept_pending;
    println!(
        "HELD-WAKES kept gate: completed {took:?} after the gate was let go; {kept_groups} \
         groups kept, {rescues} slice rescues"
    );
    assert!(kept_groups > 0, "no group was kept");
    assert_eq!(
        rescues, 0,
        "the kept group went only at a slice: the completion published as it was kept woke nothing"
    );
}

#[test]
fn a_held_batch_whose_wake_never_came_goes_at_the_next_slice() {
    let device = device(true);
    let queue = device.new_command_queue().unwrap();
    let target = device
        .new_buffer(1 << 16, ResourceOptions::private())
        .unwrap();
    warm(&queue, &target);
    let mut held = Held(Vec::new());
    let (first, second, kept) = kept_group(&queue, &target, &mut held);
    let before = device.work_counts();
    // The scheduler's steps from here on (its passes, slices, wakes and takings in, the group's
    // keeping and sending), which a failure names.
    trace::start();
    // The gate left and the scheduler asleep, then the completions published with no wake for
    // it.
    trace::drop_completion_wakes(true);
    trace::release(Point::KeptGate);
    let start = Instant::now();
    while !device.shared().scheduler.asleep() {
        assert!(start.elapsed() < BOUND, "the scheduler never slept");
        std::thread::sleep(Duration::from_millis(1));
    }
    trace::release(Point::Publish);
    completes(&first, "the first command buffer");
    completes(&second, "the second command buffer");
    let took = completes(&kept, "the kept command buffer");
    let after = device.work_counts();
    let steps = trace::stop();
    let rescues = after.scheduler_slice_rescues - before.scheduler_slice_rescues;
    let slices = after.scheduler_sleep_slices - before.scheduler_sleep_slices;
    let delta = |of: fn(&crate::WorkCounts) -> u64| of(&after) - of(&before);
    println!(
        "HELD-WAKES lost wake: completed {took:?} after the completions were published; \
         {slices} slices, {rescues} slice rescues; wakes by commit {}, completion {}, host {}; \
         commits taken in {}, passes {}, by waiters {}",
        delta(|counts| counts.scheduler_wakes_commit),
        delta(|counts| counts.scheduler_wakes_completion),
        delta(|counts| counts.scheduler_wakes_host),
        delta(|counts| counts.commits_taken_in),
        delta(|counts| counts.scheduler_passes),
        delta(|counts| counts.scheduler_passes_by_waiters),
    );
    assert!(
        rescues > 0,
        "the kept group went with no slice's pass; the scheduler's steps: {steps:#?}"
    );
}

/// An unshared queue made on `queue`'s `VkQueue`, the queues made on the way kept; none where 64
/// went elsewhere (a device that gives unshared queues `VkQueue`s of their own holds nothing back
/// for them).
fn unshared_beside(device: &Device, queue: &CommandQueue) -> Option<Vec<CommandQueue>> {
    let at = queue.vk_queue_sharing().0;
    let mut made = Vec::new();
    for _ in 0..64 {
        let unshared = device.new_command_queue_unshared(64).unwrap();
        let there = unshared.vk_queue_sharing().0 == at;
        made.push(unshared);
        if there {
            return Some(made);
        }
    }
    None
}

#[test]
fn a_batch_held_for_an_unshared_queue_goes_at_the_completion_published_as_it_was_looked_at() {
    let device = device(false);
    let queue = device.new_command_queue().unwrap();
    let Some(_unshared) = unshared_beside(&device, &queue) else {
        println!("HELD-WAKES unshared: no unshared queue shares a VkQueue here; nothing is held");
        return;
    };
    assert!(device.settle_queue_switch_check(BOUND), "the check settles");
    let index = queue.vk_queue_sharing().0;
    assert!(device.holds_back_for_unshared(index));
    let slot = &device.shared().queues[index];
    let target = device
        .new_buffer(1 << 16, ResourceOptions::private())
        .unwrap();
    // Every value submitted there seen complete, so the completion held next is the first
    // command buffer's.
    let start = Instant::now();
    while slot.completed_value() < slot.submitted_value() {
        assert!(start.elapsed() < BOUND, "the VkQueue never went idle");
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut held = Held(Vec::new());
    held.at(Point::Leave);
    // The queue's first command buffer: of unknown length, so the next waits for its completion.
    let ahead = fill(&queue, &target, 1);
    ahead.commit().unwrap();
    ahead.wait_until_scheduled().unwrap();
    let value = slot.submitted_value();
    assert!(
        trace::reached(Point::Leave, BOUND),
        "the first command buffer's completion was never published"
    );
    assert_eq!(
        slot.completed_value(),
        value,
        "the completion held is the first command buffer's"
    );
    let before = device.work_counts();
    let passes = device.shared().scheduler.passes();
    let behind = fill(&queue, &target, 2);
    behind.commit().unwrap();
    // The scheduler's thread looked at it: it went, or it is held and the thread sleeps.
    let start = Instant::now();
    loop {
        let went = matches!(
            behind.status(),
            CommandBufferStatus::Scheduled | CommandBufferStatus::Completed
        );
        let held_back = device.work_counts().heads_held_for_unshared_isolation
            > before.heads_held_for_unshared_isolation;
        let slept =
            device.shared().scheduler.passes() > passes && device.shared().scheduler.asleep();
        if went || (held_back && slept) {
            break;
        }
        assert!(start.elapsed() < BOUND, "the scheduler never looked at it");
        std::thread::sleep(Duration::from_millis(1));
    }
    let went = behind.status();
    trace::release(Point::Leave);
    let took = completes(&behind, "the command buffer behind");
    completes(&ahead, "the first command buffer");
    let after = device.work_counts();
    let rescues = after.scheduler_slice_rescues - before.scheduler_slice_rescues;
    let looks = after.heads_held_for_unshared_isolation - before.heads_held_for_unshared_isolation;
    println!(
        "HELD-WAKES unshared: {went:?} as the completion was let go, completed {took:?} after; \
         {looks} looks held it, {rescues} slice rescues"
    );
    assert_eq!(
        rescues, 0,
        "the command buffer behind went only at a slice: the completion published as it was \
         looked at woke nothing"
    );
}
