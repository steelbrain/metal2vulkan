//! A command buffer whose queue's timeline passed it while its own timestamps read unavailable:
//! work the driver dropped with no error the device reports (`gpu_time.rs`), as Linux's amdgpu
//! drops a job its timeout ended. It fails as Metal's victim of a GPU recovery
//! (`GpuRecoveryVictim`), its GPU times 0, counted (`WorkCounts::unrun_command_buffers`), and its
//! queue runs on, nothing counted against it. The unavailable read is injected
//! (`completion::trace::unstamp_of`): the work runs, and no GPU hangs.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md).

use super::device_recovery::{device, outcome, Scratch};
use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue, Device,
    ResourceOptions,
};

/// A command buffer of `queue` that fills `buffer`'s first 4096 bytes with `value`.
fn filled_by(queue: &CommandQueue, buffer: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..4096, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// `buffer`'s first 4096 bytes.
fn bytes(buffer: &Buffer) -> Vec<u8> {
    // SAFETY: a shared buffer of 4096 bytes, its command buffers completed.
    unsafe { std::slice::from_raw_parts(buffer.contents().unwrap().as_ptr(), 4096) }.to_vec()
}

/// The command buffers `device` failed as dropped work so far.
fn unrun(device: &Device) -> u64 {
    device.work_counts().unrun_command_buffers
}

#[test]
fn a_command_buffer_whose_timestamps_read_unavailable_fails_as_a_victim() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let timing = &device.shared().timing;
    if timing.clock().is_none() || timing.answers_from_host() {
        println!(
            "SKIPPED {}: its command buffers write no timestamps",
            device.name()
        );
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    // One whose timestamps read as written completes, nothing counted.
    let ran = filled_by(&queue, &buffer, 1);
    ran.commit().unwrap();
    assert_eq!(outcome(&ran), (CommandBufferStatus::Completed, None));
    assert!(ran.gpu_end_time().unwrap() > 0.0, "its GPU times were read");
    assert_eq!(unrun(&device), 0);
    // One whose timestamps read unavailable fails as a victim, its GPU times 0.
    let dropped = filled_by(&queue, &buffer, 2);
    crate::command_buffer::completion::trace::unstamp_of(&dropped);
    dropped.commit().unwrap();
    assert_eq!(
        outcome(&dropped),
        (
            CommandBufferStatus::Error,
            Some(CommandBufferError::GpuRecoveryVictim)
        )
    );
    let times = (
        dropped.gpu_start_time().unwrap(),
        dropped.gpu_end_time().unwrap(),
    );
    assert_eq!(times, (0.0, 0.0));
    assert_eq!(unrun(&device), 1);
    // Its queue runs on: a victim does not count against it, as Metal's does not.
    let next = filled_by(&queue, &buffer, 3);
    next.commit().unwrap();
    assert_eq!(outcome(&next), (CommandBufferStatus::Completed, None));
    assert!(
        bytes(&buffer).iter().all(|&byte| byte == 3),
        "the next fill ran"
    );
    assert_eq!(unrun(&device), 1);
}
