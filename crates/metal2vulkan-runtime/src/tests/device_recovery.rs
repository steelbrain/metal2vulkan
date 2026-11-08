//! A device's answers between the loss of its `VkDevice` and its recovery (`device/line.rs`), as a
//! failed wait finds a loss: the device is marked lost and its completion fails what is pending,
//! the oldest incomplete command buffer as the hang. Until the recovery, which the next call that
//! needs the device makes, a command buffer committed on the hang's queue runs nothing and
//! completes with no error, as Metal's on the queue of a hang in flight, and one committed on
//! another queue fails as the recovery's victim. Then the device and the hang's queue serve on.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked, and the work in flight
//! completes before the failures post (`Device::drain_queues`).

use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue, Device,
    DeviceKind, DeviceOptions, ResourceOptions, ShaderCacheOptions,
};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The bytes each copy of the busy command buffer moves: enough that it is still in flight as the
/// loss is marked, on a CPU device and on a GPU.
const BYTES: u64 = 32 << 20;

/// The copies the busy command buffer makes.
const COPIES: u64 = 24;

/// The most losses the test marks to catch the busy command buffer in flight at one.
const ATTEMPTS: usize = 6;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-device-recovery-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// An uncommitted command buffer of `queue` that fills `buffer`'s first 4096 bytes.
fn filled_by(queue: &CommandQueue, buffer: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..4096, 1).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// A committed command buffer of `queue` that copies between two private buffers of `device`
/// [`COPIES`] times, scheduled.
fn busy(device: &Device, queue: &CommandQueue) -> CommandBuffer {
    let source = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    let destination = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for _ in 0..COPIES {
        blit.copy_from_buffer_to_buffer(&source, 0, &destination, 0, BYTES)
            .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_scheduled().unwrap();
    command_buffer
}

/// The longest the test waits for a command buffer to complete: one that does not fails the test
/// at its line, its status named, where an unbounded wait would hang it.
const WAIT: Duration = Duration::from_secs(60);

/// The command buffer's final status and error, awaited for at most [`WAIT`].
#[track_caller]
fn outcome(command_buffer: &CommandBuffer) -> (CommandBufferStatus, Option<CommandBufferError>) {
    if command_buffer.wait_until_completed_for(WAIT).is_err() {
        panic!(
            "a command buffer did not complete in {WAIT:?}: {:?}, {:?}",
            command_buffer.status(),
            command_buffer.error()
        );
    }
    (command_buffer.status(), command_buffer.error())
}

/// Between a loss and the recovery, the hang's queue is held: a command buffer committed there
/// completes with no error and runs nothing, one committed on another queue fails as the victim;
/// the next call recovers the device, and the hang's queue serves on the new `VkDevice`.
#[test]
fn a_loss_holds_the_hang_s_queue_until_the_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let hung = device.new_command_queue().unwrap();
    let other = device.new_command_queue().unwrap();
    for _ in 0..ATTEMPTS {
        let filled = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        // SAFETY: a shared buffer of 4096 bytes, which nothing uses yet.
        unsafe { std::ptr::write_bytes(filled.contents().unwrap().as_ptr(), 0, 4096) };
        let (held, victim) = (filled_by(&hung, &filled), filled_by(&other, &filled));
        let in_flight = busy(&device, &hung);
        let recoveries = device.work_counts().device_recoveries;
        // The loss as a failed wait finds it (`completion.rs`).
        let lost = in_flight.device().clone();
        lost.mark_lost();
        lost.shared().completion.fail_in_flight();
        if outcome(&in_flight)
            != (
                CommandBufferStatus::Error,
                Some(CommandBufferError::GpuHang),
            )
        {
            // The copies completed before the loss: none was the hang. The next call recovers.
            drop((held, victim, in_flight, filled, lost));
            drop(hung.command_buffer().unwrap());
            continue;
        }
        held.commit().unwrap();
        victim.commit().unwrap();
        assert_eq!(outcome(&held), (CommandBufferStatus::Completed, None));
        assert_eq!(
            outcome(&victim),
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::GpuRecoveryVictim)
            )
        );
        // The held command buffer ran nothing; the victim neither, its device lost.
        // SAFETY: a shared buffer of 4096 bytes, nothing in flight writes it.
        let bytes =
            unsafe { std::slice::from_raw_parts(filled.contents().unwrap().as_ptr(), 4096) };
        assert!(bytes.iter().all(|&byte| byte == 0), "the held fill ran");
        drop((held, victim, in_flight, filled, lost));
        // The next call of the device recovers it; the hang's queue serves on it.
        let after = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let next = filled_by(&hung, &after);
        next.commit().unwrap();
        assert_eq!(outcome(&next), (CommandBufferStatus::Completed, None));
        assert_eq!(device.work_counts().device_recoveries, recoveries + 1);
        return;
    }
    panic!("no loss of {ATTEMPTS} caught the busy command buffer in flight");
}

/// A loss marked while `queue`'s busy command buffer is in flight, as a failed wait finds it:
/// whether it caught that command buffer as the hang, which else completed first. Either way the
/// next call recovers the device.
fn hang_caught(device: &Device, queue: &CommandQueue) -> bool {
    let in_flight = busy(device, queue);
    let lost = in_flight.device().clone();
    lost.mark_lost();
    lost.shared().completion.fail_in_flight();
    outcome(&in_flight)
        == (
            CommandBufferStatus::Error,
            Some(CommandBufferError::GpuHang),
        )
}

/// A queue's second hang revokes it from the hang's completion, as Metal's: a command buffer
/// committed on it between that loss and the recovery runs nothing and fails with
/// `AccessRevoked`, as one committed after the recovery does, not as the recovery's victim.
#[test]
fn a_queue_s_second_hang_revokes_it_before_the_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    assert!(
        (0..ATTEMPTS).any(|_| hang_caught(&device, &queue)),
        "no loss of {ATTEMPTS} caught the first busy command buffer in flight"
    );
    for _ in 0..ATTEMPTS {
        let filled = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        // SAFETY: a shared buffer of 4096 bytes, which nothing uses yet.
        unsafe { std::ptr::write_bytes(filled.contents().unwrap().as_ptr(), 0, 4096) };
        let revoked = filled_by(&queue, &filled);
        if !hang_caught(&device, &queue) {
            continue;
        }
        revoked.commit().unwrap();
        assert_eq!(
            outcome(&revoked),
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::AccessRevoked)
            )
        );
        // SAFETY: a shared buffer of 4096 bytes, nothing in flight writes it.
        let bytes =
            unsafe { std::slice::from_raw_parts(filled.contents().unwrap().as_ptr(), 4096) };
        assert!(bytes.iter().all(|&byte| byte == 0), "the revoked fill ran");
        return;
    }
    panic!("no loss of {ATTEMPTS} caught the second busy command buffer in flight");
}

/// A command buffer whose work completed, its queue's timeline past it, completes though the wait
/// for its fence then fails with the device's loss: a real loss's failed wait, which the test hook
/// never takes. The loss is recorded: the next call recovers the device, and the queue serves on.
#[test]
fn work_that_completed_before_a_failed_wait_completes() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let filled = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let completed = filled_by(&queue, &filled);
    crate::command_buffer::completion::trace::fail_settle_of(&completed);
    completed.commit().unwrap();
    assert_eq!(outcome(&completed), (CommandBufferStatus::Completed, None));
    // SAFETY: a shared buffer of 4096 bytes, its command buffer completed.
    let bytes = unsafe { std::slice::from_raw_parts(filled.contents().unwrap().as_ptr(), 4096) };
    assert!(bytes.iter().all(|&byte| byte == 1), "its fill ran");
    drop((completed, filled));
    let after = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let next = filled_by(&queue, &after);
    next.commit().unwrap();
    assert_eq!(outcome(&next), (CommandBufferStatus::Completed, None));
    assert_eq!(device.work_counts().device_recoveries, 1);
}
