//! A command buffer's kernel end is the host time its last batch's `vkQueueSubmit` was called,
//! never its return (`gpu_time.rs`, "Kernel times"), so its GPU end is never earlier, as Metal's
//! `GPUEndTime` never precedes its `kernelEndTime`. A driver may run the work before the call
//! returns, and a loaded host may hold the scheduler's thread once it returned: here the scheduler
//! pauses for [`PAUSE`] each time a submission returned (`completion::trace::pause_after_submit`,
//! a test's stand-in for that thread preempted) while the device runs a small fill, which ends
//! long before the pause does. With the kernel end taken once the call returned, it lay the pause
//! past the GPU's end.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::command_buffer::completion::trace;
use crate::tests::SETTLE;
use crate::{CommandBufferStatus, Device, DeviceKind, DeviceOptions, ResourceOptions};
use std::time::Duration;

/// How long the scheduler pauses once each submission returned.
const PAUSE: Duration = Duration::from_millis(200);

/// The bytes each command buffer fills: done well inside the pause on every device tested.
const FILL: u64 = 1 << 20;

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
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
