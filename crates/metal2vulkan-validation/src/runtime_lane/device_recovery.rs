//! Device tests of a device's recovery from the loss of its `VkDevice` (`device/line.rs`), run
//! through the public test hook `Device::recover_for_test`, which takes the path a real loss takes
//! with no GPU fault, on the device the cfg selects, under the validation layer.
//!
//! Metal's device survives a GPU recovery: the hung command buffer fails with `GpuHang`, the others
//! in flight with `GpuRecoveryVictim`, a queue's second hang revokes it (`AccessRevoked`), and the
//! program goes on with the same device and queues. The runtime makes a new `VkDevice` behind the
//! device: its command queues go on, the events, `Private` buffers and buffers in fresh pages made
//! before go on with it, other objects made before are refused where they meet the new one
//! (`RuntimeGap::MadeBeforeRecovery`), and the lost `VkDevice` goes once they went. The layer
//! reporting nothing over each test is the check that no handle of the lost `VkDevice` reached the
//! new one.

use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue, Degradation,
    Device, Error, EventRef, ResourceOptions, RuntimeGap,
};
use std::time::{Duration, Instant};

/// The bytes each copy of a busy command buffer moves: enough that a few of them are still in
/// flight as the hook runs, on a CPU device and on a GPU.
const BYTES: u64 = 32 << 20;

/// The copies each busy command buffer makes.
const COPIES: u64 = 24;

/// The busy command buffers committed before each recovery.
const BUSY: usize = 3;

/// The small command buffers committed behind the busy one before a recovery
/// ([`a_recovery_fails_the_command_buffers_on_their_way_as_its_victims`]).
const TRAILING: usize = 32;

/// The most recoveries a test runs to catch work in flight at one (on a fast GPU the work may all
/// have completed before the hook ran).
const ATTEMPTS: usize = 6;

/// The kind each recovery takes
/// ([`a_recovery_fails_the_work_in_flight_as_a_hang_and_its_victims`]).
pub(super) const RECOVERED: [Degradation; 1] = [Degradation::DeviceRecovered];

/// Run `test` on a validated device with its messages, then check that once everything is dropped
/// (the device last) the layer reported nothing.
fn on_device(test: impl FnOnce(&Device, &Messages)) {
    on_timed_device(|device, messages, _| test(device, messages));
}

/// [`on_device`], `test` also given how long the device's making took.
fn on_timed_device(test: impl FnOnce(&Device, &Messages, Duration)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let began = Instant::now();
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device, &messages, began.elapsed());
    }
    layer_reported_nothing_at_all(&messages);
}

/// A private buffer of [`BYTES`] on `device`.
fn private(device: &Device) -> Buffer {
    device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap()
}

/// A committed command buffer of `queue` that copies `source` into `destination` [`COPIES`]
/// times.
fn busy(queue: &CommandQueue, source: &Buffer, destination: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for _ in 0..COPIES {
        blit.copy_from_buffer_to_buffer(source, 0, destination, 0, BYTES)
            .unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// An uncommitted command buffer of `queue` that fills `buffer`'s first 4096 bytes.
fn filled_by(queue: &CommandQueue, buffer: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..4096, 1).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// A command buffer's answer: its final status and error.
type Outcome = (CommandBufferStatus, Option<CommandBufferError>);

/// The answer of the recovery's hang.
const HANG: Outcome = (
    CommandBufferStatus::Error,
    Some(CommandBufferError::GpuHang),
);

/// The answer of the recovery's victims.
const VICTIM: Outcome = (
    CommandBufferStatus::Error,
    Some(CommandBufferError::GpuRecoveryVictim),
);

/// Whether each command buffer was `Scheduled` (or past it) as read now: its batches went to the
/// driver.
fn scheduled(command_buffers: &[CommandBuffer]) -> Vec<bool> {
    (command_buffers.iter())
        .map(|command_buffer| command_buffer.status() != CommandBufferStatus::Committed)
        .collect()
}

/// The command buffers' final statuses and errors, each awaited.
fn outcomes(command_buffers: &[CommandBuffer]) -> Vec<Outcome> {
    command_buffers
        .iter()
        .map(|command_buffer| {
            command_buffer.wait_until_completed().unwrap();
            (command_buffer.status(), command_buffer.error())
        })
        .collect()
}

/// Whether `outcomes`, in commit order on one queue, are a recovery's answers, `scheduled` read
/// just before it ([`scheduled`]), and one failed as the hang. Those that completed first
/// completed. The next, the oldest incomplete one, fails as the hang where it was scheduled; one
/// on its way to the driver may have gone there before the loss (the hang) or not (a victim, and
/// the recovery names no hang: nothing of the queue's was on the GPU). Every later one fails as a
/// victim.
fn recovery_answers(outcomes: &[Outcome], scheduled: &[bool]) -> bool {
    let completed = outcomes
        .iter()
        .take_while(|(status, _)| *status == CommandBufferStatus::Completed)
        .count();
    let Some((first, victims)) = outcomes[completed..].split_first() else {
        return false;
    };
    match scheduled[completed] {
        true => assert_eq!(
            *first, HANG,
            "the oldest incomplete command buffer, scheduled, fails as the hang: {outcomes:?}"
        ),
        false => assert!(
            *first == HANG || *first == VICTIM,
            "the oldest incomplete command buffer, on its way to the driver, fails as the hang \
             or as a victim: {outcomes:?}"
        ),
    }
    for victim in victims {
        assert_eq!(
            *victim, VICTIM,
            "the later ones fail as its victims: {outcomes:?}"
        );
    }
    *first == HANG
}

/// One recovery of `device` with [`BUSY`] busy command buffers of `queue` committed before it, the
/// first of them scheduled: whether it caught one of them in flight, its answers checked
/// ([`recovery_answers`]).
fn recover_busy(device: &Device, queue: &CommandQueue) -> bool {
    let (source, destination) = (private(device), private(device));
    let committed: Vec<CommandBuffer> = (0..BUSY)
        .map(|_| busy(queue, &source, &destination))
        .collect();
    committed[0].wait_until_scheduled().unwrap();
    let scheduled = scheduled(&committed);
    device.recover_for_test().unwrap();
    recovery_answers(&outcomes(&committed), &scheduled)
}

/// The hook fails the work in flight as a GPU recovery does, the oldest incomplete command buffer
/// as the hang and the later ones as its victims, and announces the recovery with the hang's queue
/// named; the device and its queue serve on, on a new `VkDevice`.
#[test]
fn a_recovery_fails_the_work_in_flight_as_a_hang_and_its_victims() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        queue.set_label("hung");
        let before = device.work_counts();
        let caught = (0..ATTEMPTS).find(|_| recover_busy(device, &queue));
        let attempts = caught.expect("a recovery caught work in flight") + 1;
        let after = device.work_counts();
        assert_eq!(
            after.device_recoveries - before.device_recoveries,
            attempts as u64
        );
        assert_eq!(after.vk_devices_made, 1 + attempts as u64);
        let notices: Vec<_> = (messages.take_notices().into_iter())
            .filter_map(|message| message.degradation)
            .filter(|notice| RECOVERED.contains(&notice.degradation))
            .collect();
        assert_eq!(
            notices.len(),
            attempts,
            "one notice per recovery: {notices:?}"
        );
        let last = notices.last().unwrap();
        assert!(
            last.reason.contains("GpuHang") && last.reason.contains("GpuRecoveryVictim"),
            "the notice names the hang and its victims: {last:?}"
        );
        // The queue serves on the new `VkDevice`, and so does a new one.
        for queue in [queue, device.new_command_queue().unwrap()] {
            let (source, destination) = (private(device), private(device));
            let command_buffer = busy(&queue, &source, &destination);
            command_buffer.wait_until_completed().unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
    });
}

/// Command buffers committed back to back behind a busy one before a recovery, some of them on
/// their way to the GPU as it runs (submitted to the driver, not yet watched by the lost device's
/// completion), fail with it: each that had not completed as the hang's victim, after the one
/// hang. Every one of [`ATTEMPTS`] recoveries is checked, each a chance of one on its way, each on
/// a queue of its own (a queue's second hang revokes it).
#[test]
fn a_recovery_fails_the_command_buffers_on_their_way_as_its_victims() {
    on_device(|device, _| {
        let caught = (0..ATTEMPTS).map(|_| {
            let queue = device.new_command_queue().unwrap();
            let (source, destination) = (private(device), private(device));
            let filled = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
            let mut committed = vec![busy(&queue, &source, &destination)];
            committed.extend((0..TRAILING).map(|_| {
                let command_buffer = filled_by(&queue, &filled);
                command_buffer.commit().unwrap();
                command_buffer
            }));
            committed[0].wait_until_scheduled().unwrap();
            let scheduled = scheduled(&committed);
            device.recover_for_test().unwrap();
            recovery_answers(&outcomes(&committed), &scheduled)
        });
        assert!(
            caught.filter(|&caught| caught).count() > 0,
            "a recovery caught work in flight"
        );
    });
}

/// An object made before a recovery that it does not carry (a pooled `Shared` buffer) is refused
/// where it meets the new `VkDevice`, typed; the same
/// object made again serves. The lost `VkDevice` stays while an object made on it lives, so the
/// device holds two, and goes with the last of them, once a `Device::raw` call that answered its
/// handles was followed by one that answered the new one's.
///
/// It prints how long the recovery took beside the device's first making, which the recovery
/// makes again, and bounds neither: both are the driver's work, which other processes' load on
/// the host and the GPU stretches.
#[test]
fn objects_made_before_a_recovery_are_refused_and_the_lost_vk_device_goes_with_them() {
    on_timed_device(|device, _, making| {
        let queue = device.new_command_queue().unwrap();
        let old = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        // SAFETY: the handles are read and not used.
        let _ = unsafe { device.raw() };
        assert_eq!(device.work_counts().vk_devices_live, 1);
        let began = Instant::now();
        device.recover_for_test().unwrap();
        eprintln!(
            "device-recovery: recovered in {:?} on {}; its first making took {making:?}",
            began.elapsed(),
            device.name()
        );
        assert_eq!(device.work_counts().vk_devices_live, 2);
        // The new `VkDevice`'s command buffer refuses the old buffer, and takes a new one.
        let new = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        assert_eq!(
            blit.copy_from_buffer_to_buffer(&old, 0, &new, 0, 4096),
            Err(Error::NotImplemented(RuntimeGap::MadeBeforeRecovery))
        );
        assert_eq!(
            blit.fill_buffer(&old, 0..4096, 7),
            Err(Error::NotImplemented(RuntimeGap::MadeBeforeRecovery))
        );
        blit.fill_buffer(&new, 0..4096, 7).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let contents = new.contents().unwrap();
        // SAFETY: a shared buffer of 4096 bytes, its command buffer completed.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 4096) };
        assert!(bytes.iter().all(|&byte| byte == 7));
        // The old buffer was the last object of the lost `VkDevice`; the handles `Device::raw`
        // answered before the recovery keep it until a later call answers the new one's.
        drop(old);
        assert_eq!(device.work_counts().vk_devices_live, 2);
        // SAFETY: the handles are read and not used.
        let _ = unsafe { device.raw() };
        assert_eq!(device.work_counts().vk_devices_live, 1);
        assert_eq!(device.work_counts().vk_devices_made, 2);
    });
}

/// The objects a recovery carries serve on the new `VkDevice` (`device/carry.rs`), under the
/// layer: a `Private` buffer filled and copied out, and a shared event the GPU signals, after one
/// recovery and after a second. The layer reporting nothing is the check that no handle of a lost
/// `VkDevice` reached a new one, and that each lost one's objects went before it; one `VkDevice`
/// is left.
#[test]
fn carried_objects_serve_on_the_new_vk_device() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let carried = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let event = device.new_shared_event().unwrap();
        for round in 1..=2u8 {
            device.recover_for_test().unwrap();
            let target = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&carried, 0..4096, round).unwrap();
            (blit.copy_from_buffer_to_buffer(&carried, 0, &target, 0, 4096)).unwrap();
            blit.end_encoding().unwrap();
            let value = u64::from(round);
            (command_buffer.encode_signal_event(EventRef::from(&event), value)).unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed().unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert!(event.wait_until_signaled_value(value, Duration::from_secs(60)));
            let contents = target.contents().unwrap();
            // SAFETY: a shared buffer of 4096 bytes, its command buffer completed.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 4096) };
            assert!(bytes.iter().all(|&byte| byte == round));
        }
        let began = Instant::now();
        while device.work_counts().vk_devices_live > 1 && began.elapsed() < Duration::from_secs(60)
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(device.work_counts().vk_devices_live, 1);
        assert_eq!(device.work_counts().vk_devices_made, 3);
    });
}

/// A queue's second hang revokes it, as Metal's: a command buffer committed on it after fails at
/// once with `AccessRevoked`, while another queue serves on.
#[test]
fn a_queue_s_second_recovery_hang_revokes_it() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        for hang in 1..=2 {
            let caught = (0..ATTEMPTS).any(|_| recover_busy(device, &queue));
            assert!(caught, "recovery {hang} caught work in flight");
        }
        let filled = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let revoked = filled_by(&queue, &filled);
        revoked.commit().unwrap();
        revoked.wait_until_completed().unwrap();
        assert_eq!(
            (revoked.status(), revoked.error()),
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::AccessRevoked)
            )
        );
        let other = device.new_command_queue().unwrap();
        let (source, destination) = (private(device), private(device));
        let command_buffer = busy(&other, &source, &destination);
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    });
}

/// A command buffer made before a recovery and committed after it fails as the recovery's victim:
/// its commands were recorded for the lost `VkDevice`.
#[test]
fn a_command_buffer_made_before_a_recovery_fails_as_its_victim() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let filled = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let made = filled_by(&queue, &filled);
        device.recover_for_test().unwrap();
        made.commit().unwrap();
        made.wait_until_completed().unwrap();
        assert_eq!(
            (made.status(), made.error()),
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::GpuRecoveryVictim)
            )
        );
    });
}
