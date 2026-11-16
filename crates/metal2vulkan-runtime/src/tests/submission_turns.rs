//! When a command buffer's submission handlers run (`scheduler/mod.rs`, **Submission handlers**;
//! `scheduler/inbox.rs`, "Who runs the after-work, and when"): on the thread whose pass took it,
//! once that pass let go of the scheduler's lock and before that thread passes again. A pass of
//! the scheduler's thread is held in a submission (`trace::Point::FirstWork`) while another queue
//! commits, so the thread's look after that pass takes the second commit in: the first command
//! buffer's submission handlers ran before the second's submission was called (its
//! `kernel_end_time`), where they once ran only after every pass the commits kept the thread
//! making. Under either `CompletionPolicy`: a batch the default `Grouped` holds in a group is taken
//! (its handlers taken with it) at the hold, which the first work's take is reached at too.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::command_buffer::completion::trace;
use crate::tests::SETTLE;
use crate::{
    host_time, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind,
    DeviceOptions, ResourceOptions,
};
use metal2vulkan::device::CompletionPolicy;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The bytes each command buffer fills.
const FILL: u64 = 64 << 10;

/// How long a step may take before the test fails rather than hangs.
const WAIT: Duration = Duration::from_secs(10);

/// The device the cfg selects (a CPU device under `m2v_cpu_icd`, else a GPU) under `completion`.
fn device(completion: CompletionPolicy) -> Device {
    let mut options = DeviceOptions {
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
    options.policy.completion = completion;
    let device = Device::system_default_with(&options).unwrap();
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    device
}

/// A command buffer of `queue` that fills `target` with `value`.
fn fill(queue: &CommandQueue, target: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..FILL, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// Wait for `command_buffer`, which completes.
fn completes(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The module docs' check, under `completion`.
fn handlers_run_before_the_next_pass(completion: CompletionPolicy) {
    let device = device(completion);
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let targets = [
        device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
        device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
    ];
    // Each queue's first work done, and each target's, before the measured commits.
    for (queue, target) in [(&first, &targets[0]), (&second, &targets[1])] {
        let warm = fill(queue, target, 1);
        warm.commit().unwrap();
        completes(&warm);
    }
    let ran = Arc::new(Mutex::new(None));
    let held = fill(&first, &targets[0], 2);
    let noted = Arc::clone(&ran);
    held.add_submission_handler(Box::new(move |_: &CommandBuffer| {
        *noted.lock().unwrap() = Some(host_time());
    }))
    .unwrap();
    // The scheduler's thread takes it in its pass and is held in its submission, the scheduler's
    // lock held and the thread counted as the passer.
    trace::hold_at(trace::Point::FirstWork);
    held.commit().unwrap();
    let reached = trace::reached(trace::Point::FirstWork, WAIT);
    // Posted while that pass is under way: left to the thread's look once the pass ends.
    let later = fill(&second, &targets[1], 3);
    if reached {
        later.commit().unwrap();
    }
    trace::release(trace::Point::FirstWork);
    assert!(reached, "the scheduler's pass never reached its submission");
    completes(&held);
    completes(&later);
    let ran = (*ran.lock().unwrap()).expect("its submission handler ran");
    let submitted = later.kernel_end_time();
    println!(
        "SUBMISSION-TURNS {} {completion:?}: the held pass's handler ran {:.1} us before the \
         later commit's submission was called",
        device.name(),
        (submitted - ran) * 1e6,
    );
    assert!(
        ran < submitted,
        "the held pass's submission handler ran {:.1} us after the next pass submitted the commit \
         posted meanwhile ({completion:?})",
        (ran - submitted) * 1e6,
    );
}

#[test]
fn a_pass_s_submission_handlers_run_before_the_thread_passes_again_exact() {
    handlers_run_before_the_next_pass(CompletionPolicy::Exact);
}

#[test]
fn a_pass_s_submission_handlers_run_before_the_thread_passes_again_grouped() {
    handlers_run_before_the_next_pass(CompletionPolicy::Grouped);
}
