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

/// The command buffer's final status and error, awaited.
fn outcome(command_buffer: &CommandBuffer) -> (CommandBufferStatus, Option<CommandBufferError>) {
    command_buffer.wait_until_completed().unwrap();
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
