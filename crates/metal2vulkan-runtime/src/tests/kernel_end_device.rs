//! A command buffer's kernel end is the host time its last batch's `vkQueueSubmit` was called,
//! never its return (`gpu_time.rs`, "Kernel times"), so its GPU end is never earlier, as Metal's
//! `GPUEndTime` never precedes its `kernelEndTime`. A driver may run the work before the call
//! returns, and a loaded host may hold the scheduler's thread once it returned: here the scheduler
//! pauses for [`PAUSE`] each time a submission returned (`completion::trace::pause_after_submit`,
//! a test's stand-in for that thread preempted) while the device runs a small fill, which ends
//! long before the pause does. With the kernel end taken once the call returned, it lay the pause
//! past the GPU's end.
//!
//! The queue times (`METAL2VULKAN_QUEUE_TIMES`, a default-off diagnostic): with them noted, each
//! of a queue's command buffers committed back to back became its queue's head no later than its
//! kernel start and after the one before it did, and its submission returned after its kernel end;
//! heads, kernel starts and ends and returns all come in commit order (two that waited in a group
//! return from one call); without them, each reads 0 and its kernel times as before.
//!
//! The slow submissions log (`slow_submissions.rs`), under the queue times: a batch whose
//! submission a test holds [`HELD`] in its first work's take (`trace::Point::FirstWork`, standing
//! in for a lock or a wait there) is named in one line, which gives that phase the time; one
//! submitted straight is named in none, and without the queue times the held one in none either.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::command_buffer::completion::trace;
use crate::slow_submissions::ID_NAME;
use crate::tests::SETTLE;
use crate::{
    CommandBuffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, MessageSink,
    ResourceOptions,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long the scheduler pauses once each submission returned.
const PAUSE: Duration = Duration::from_millis(200);

/// The bytes each command buffer fills: done well inside the pause on every device tested.
const FILL: u64 = 1 << 20;

/// How long a test holds a submission in its first work's take: past the log's bound
/// (`slow_submissions::SLOW`, 5 ms).
const HELD: Duration = Duration::from_millis(30);

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    device_with(None)
}

/// [`device`], its messages handed to `sink` where given.
fn device_with(sink: Option<MessageSink>) -> Device {
    let options = DeviceOptions {
        messages: sink,
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

#[test]
fn the_gpu_end_is_never_before_the_kernel_end_when_the_scheduler_pauses_after_the_call() {
    let device = device();
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(FILL, ResourceOptions::private()).unwrap();
    trace::pause_after_submit(Some(PAUSE));
    let mut times = Vec::new();
    for round in 0..4u8 {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..FILL, round).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let kernel = (
            command_buffer.kernel_start_time(),
            command_buffer.kernel_end_time(),
        );
        let gpu = (
            command_buffer.gpu_start_time(),
            command_buffer.gpu_end_time(),
        );
        times.push((kernel, gpu));
    }
    trace::pause_after_submit(None);
    for ((kernel_start, kernel_end), gpu) in times {
        let (Ok(gpu_start), Ok(gpu_end)) = gpu else {
            eprintln!(
                "KERNEL-END {}: the device answers no GPU times",
                device.name()
            );
            return;
        };
        println!(
            "KERNEL-END {}: kernel {kernel_start:.6}..{kernel_end:.6}, GPU {gpu_start:.6}..\
             {gpu_end:.6} (GPU end {:+.3} ms from the kernel end)",
            device.name(),
            (gpu_end - kernel_end) * 1e3,
        );
        assert!(
            0.0 < kernel_start && kernel_start <= kernel_end,
            "kernel {kernel_start}..{kernel_end}"
        );
        assert!(gpu_start <= gpu_end, "GPU {gpu_start}..{gpu_end}");
        assert!(
            gpu_start >= kernel_start,
            "the GPU started at {gpu_start}, before the kernel's start {kernel_start}"
        );
        assert!(
            gpu_end >= kernel_end,
            "the GPU ended at {gpu_end}, before the kernel's end {kernel_end}"
        );
    }
}

/// `count` command buffers of one fill each on one queue, committed back to back and waited for:
/// each one's kernel start and end, queue head time and submission's return.
fn queue_times(device: &Device, count: u8) -> Vec<[f64; 4]> {
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(FILL, ResourceOptions::private()).unwrap();
    let committed: Vec<CommandBuffer> = (0..count)
        .map(|round| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&target, 0..FILL, round).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        })
        .collect();
    (committed.iter())
        .map(|command_buffer| {
            command_buffer.wait_until_completed().unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            [
                command_buffer.queue_head_time(),
                command_buffer.kernel_start_time(),
                command_buffer.kernel_end_time(),
                command_buffer.submit_returned_time(),
            ]
        })
        .collect()
}

#[test]
fn queue_times_are_noted_in_the_order_of_a_queue_s_submissions_and_only_where_asked() {
    let device = device();
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    // Not asked (unless the process's environment asks for them): none noted.
    if !metal2vulkan::env_vars::queue_times() {
        for [head, start, end, returned] in queue_times(&device, 3) {
            assert!(0.0 < start && start <= end, "kernel {start}..{end}");
            assert_eq!((head, returned), (0.0, 0.0), "queue times noted unasked");
        }
    }
    device.shared().timing.note_queue_times();
    let times = queue_times(&device, 3);
    let after = crate::host_time();
    for [head, start, end, returned] in &times {
        println!(
            "QUEUE-TIMES {}: head {head:.6}, kernel {start:.6}..{end:.6}, returned {returned:.6} \
             (in the driver {:.3} ms)",
            device.name(),
            (returned - end) * 1e3,
        );
        assert!(
            0.0 < *head && head <= start && start <= end && end <= returned && *returned <= after,
            "head {head}, kernel {start}..{end}, returned {returned}, read at {after}"
        );
    }
    // In commit order each time: the head, the kernel start and end, the return. Not the next
    // one's kernel start after this one's return: batches that wait in a group for one kernel
    // submission (`command_buffer/group.rs`) begin and are held before the call that carries
    // them all returns (the 780M under RADV: the second and the third returned from one call).
    for pair in times.windows(2) {
        let ([head, start, end, returned], [next_head, next_start, next_end, next_returned]) =
            (pair[0], pair[1]);
        assert!(
            head <= next_head && start <= next_start && end <= next_end,
            "a later commit became head, began or ended its submission first: {pair:?}"
        );
        assert!(
            returned <= next_returned,
            "a later commit's submission returned first: {pair:?}"
        );
    }
}

/// Commit a fill of `target` on `queue`, held [`HELD`] in its submission's first work's take where
/// `held`, and wait for it.
fn fill(queue: &crate::CommandQueue, target: &crate::Buffer, held: bool) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..FILL, 7).unwrap();
    blit.end_encoding().unwrap();
    if held {
        trace::hold_at(trace::Point::FirstWork);
    }
    command_buffer.commit().unwrap();
    if held {
        let reached = trace::reached(trace::Point::FirstWork, Duration::from_secs(10));
        std::thread::sleep(HELD);
        trace::release(trace::Point::FirstWork);
        assert!(
            reached,
            "the submission never reached its first work's take"
        );
    }
    command_buffer.wait_until_completed().unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

#[test]
fn a_submission_held_past_the_bound_is_named_with_its_phase_where_queue_times_are_noted() {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let heard = Arc::clone(&lines);
    let device = device_with(Some(MessageSink::new(move |message| {
        if message.id_name == ID_NAME {
            heard.lock().unwrap().push(message.text.clone());
        }
    })));
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks never answered"
    );
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(FILL, ResourceOptions::private()).unwrap();
    // Not asked (unless the process's environment asks): no line, held or not.
    if !metal2vulkan::env_vars::queue_times() {
        fill(&queue, &target, true);
        assert!(lines.lock().unwrap().is_empty(), "named unasked");
    }
    device.shared().timing.note_queue_times();
    fill(&queue, &target, false);
    let straight = lines.lock().unwrap().clone();
    fill(&queue, &target, true);
    let named = lines.lock().unwrap().clone();
    // A submission straight through is named only where the device itself took the bound (a
    // loaded host): the held one is named once more, whatever came before.
    assert_eq!(named.len(), straight.len() + 1, "{named:#?}");
    let line = &named[straight.len()];
    println!("SLOW-SUBMISSION {}: {line}", device.name());
    let first = line
        .split("its first work's take ")
        .nth(1)
        .and_then(|rest| rest.split(" ms").next())
        .and_then(|ms| ms.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no first work's time in {line}"));
    assert!(
        first >= HELD.as_secs_f64() * 1e3 * 0.9,
        "the first work's take took {first} ms, held {HELD:?}: {line}"
    );
}
