//! A device's answers between the loss of its `VkDevice` and its recovery (`device/line.rs`), as a
//! failed wait finds a loss: the device is marked lost and its completion fails what is pending,
//! the oldest incomplete command buffer as the hang. Until the recovery, which the next call that
//! needs the device makes, a command buffer committed on the hang's queue runs nothing and
//! completes with no error, as Metal's on the queue of a hang in flight, and one committed on
//! another queue fails as the recovery's victim. Then the device and the hang's queue serve on.
//!
//! Each recovery announces its own notices (`Degradation::DeviceRecovered`), whatever an earlier
//! one announced: a later recovery's wait, its command buffers' slow failing and its new
//! `VkDevice`'s failure reach the sink as the first's would.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked, and the work in flight
//! completes before the failures post (`Device::drain_queues`).

use crate::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue, Degradation,
    DegradeNotice, Device, DeviceKind, DeviceOptions, Error, EventRef, MessageSink,
    ResourceOptions, ShaderCacheOptions,
};
use metal2vulkan::device::{DevicePolicy, IndirectRecordsPolicy, VkExtension};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The bytes each copy of the busy command buffer moves: enough that it is still in flight as the
/// loss is marked, on a CPU device and on a GPU.
const BYTES: u64 = 32 << 20;

/// The copies the busy command buffer makes.
const COPIES: u64 = 24;

/// The most losses the test marks to catch the busy command buffer in flight at one.
const ATTEMPTS: usize = 6;

/// A scratch directory for the device's shader cache, removed when dropped: one of its own for
/// each test, so that tests in one process (`cargo test`'s threads) never take one device. A
/// physical device has one device per options (`Device::system_default_with`), and the cache's
/// directory is among them, so tests sharing it would share the device and recover each
/// other's.
pub(super) struct Scratch(pub(super) PathBuf);

impl Scratch {
    pub(super) fn new() -> Scratch {
        static MADE: AtomicUsize = AtomicUsize::new(0);
        let made = MADE.fetch_add(1, Ordering::Relaxed);
        let name = format!("m2v-device-recovery-{}-{made}", std::process::id());
        let path = std::env::temp_dir().join(name);
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
pub(super) fn device(cache: &Path) -> Device {
    device_with(cache, None, &[], DevicePolicy::UNTRUSTED)
}

/// [`device`] whose indirect command buffers' records lie in the runtime's own pages wherever it
/// imports host pointers, whatever a held import costs its submissions
/// (`IndirectRecordsPolicy::Carried`): records a recovery's carry keeps.
pub(super) fn carried_device(cache: &Path) -> Device {
    let mut policy = DevicePolicy::UNTRUSTED;
    policy.indirect_records = IndirectRecordsPolicy::Carried;
    device_with(cache, None, &[], policy)
}

/// Whether `device` imports host pointers at all: where it does not, an indirect command buffer's
/// records lie in pooled memory under every policy and verdict (`fresh_pages.rs`).
pub(super) fn imports_pointers(device: &Device) -> bool {
    (device.shared().host_import.as_ref()).is_ok_and(|importer| importer.pointers().is_some())
}

/// [`device`] made without `VK_EXT_external_memory_host` (`DeviceOptions::withheld_extensions`),
/// as a device whose driver lists none is: the runtime's own buffers lie in pooled memory, an
/// indirect command buffer's records among them, which a recovery's carry moves.
pub(super) fn pooled_device(cache: &Path) -> Device {
    device_with(
        cache,
        None,
        &[VkExtension::ExtExternalMemoryHost],
        DevicePolicy::UNTRUSTED,
    )
}

/// The degrade notices a device's sink heard, in order.
pub(super) type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// [`device`], its degrade notices kept in `heard`.
pub(super) fn heard_device(cache: &Path, heard: &Heard) -> Device {
    heard_device_with(cache, heard, DevicePolicy::UNTRUSTED)
}

/// [`heard_device`] of `policy`.
pub(super) fn heard_device_with(cache: &Path, heard: &Heard, policy: DevicePolicy) -> Device {
    let heard = Arc::clone(heard);
    device_with(
        cache,
        Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        &[],
        policy,
    )
}

/// [`device`] of `policy`, its messages sent to `messages` where given, made without the
/// extensions `withheld`.
fn device_with(
    cache: &Path,
    messages: Option<MessageSink>,
    withheld: &[VkExtension],
    policy: DevicePolicy,
) -> Device {
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
        messages,
        withheld_extensions: withheld.iter().copied().collect(),
        policy,
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// An uncommitted command buffer of `queue` that fills `buffer`'s first 4096 bytes.
fn filled_by(queue: &CommandQueue, buffer: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    fill(&command_buffer, buffer);
    command_buffer
}

/// Encode into `command_buffer` a fill of `buffer`'s first 4096 bytes.
fn fill(command_buffer: &CommandBuffer, buffer: &Buffer) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..4096, 1).unwrap();
    blit.end_encoding().unwrap();
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
    command_buffer.set_label("busy");
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.set_label("copies");
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
pub(super) const WAIT: Duration = Duration::from_secs(60);

/// The command buffer's final status and error, awaited for at most [`WAIT`].
#[track_caller]
pub(super) fn outcome(
    command_buffer: &CommandBuffer,
) -> (CommandBufferStatus, Option<CommandBufferError>) {
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

/// A loss the driver reported before any fence could be read, as MoltenVK does under the
/// program's own `MVK_CONFIG_RESUME_LOST_DEVICE` off (it marks the device lost first): the relay
/// gives every batch in flight the loss's code (`device/relay.rs`), and the recovery's answers
/// stand, the command buffer in flight failing as the hang its notice names, not as a victim:
/// by its label and its queue's, with its encoder's kind and label.
/// The M3 Max's GPU fault under that setting read `GpuRecoveryVictim` there in 25 of 25 runs
/// (`runtime_lane::host_memory`'s `a_gpu_fault_fails_its_command_buffer_and_the_queue_runs_on`).
#[test]
fn a_relayed_loss_leaves_the_recovery_s_hang() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    queue.set_label("relayed");
    for _ in 0..ATTEMPTS {
        let from = heard.lock().unwrap().len();
        let in_flight = busy(&device, &queue);
        let lost = in_flight.device().clone();
        lost.mark_lost();
        in_flight.relay_loss();
        lost.shared().completion.fail_in_flight();
        let answer = outcome(&in_flight);
        drop((in_flight, lost));
        // The next call recovers the device; its notice says whether the loss caught the copies
        // in flight, which else completed first.
        drop(queue.command_buffer().unwrap());
        let made = recovery_notices(&heard, from, "the device");
        assert_eq!(made.len(), 1, "one recovery announced {made:?}");
        if made[0].contains("no command buffer was in flight") {
            continue;
        }
        assert!(
            made[0].contains(
                "command buffer `busy` of queue `relayed` (its 1 encoder: 0 blit `copies`) failed \
                 as the hang (`GpuHang`)"
            ),
            "{made:?}"
        );
        assert_eq!(
            answer,
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::GpuHang)
            ),
            "the recovery named the command buffer in flight the hang: {made:?}"
        );
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

/// What each command buffer's completed handler saw, in the order they ran: its queue's index, its
/// place among the queue's command buffers, and its error.
type Ran = Arc<Mutex<Vec<(usize, usize, Option<CommandBufferError>)>>>;

/// Command buffers committed before a loss and still held unsubmitted at it, on two queues (each
/// queue's first behind an event value nothing reaches, its second behind that one on its queue),
/// fail as the recovery's victims, every completed handler run, each queue's in commit order and
/// before that queue's command buffer committed after the recovery, which completes.
#[test]
fn work_held_unsubmitted_at_a_loss_fails_as_its_victims_in_queue_order() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queues = [
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    ];
    let event = device.new_shared_event().unwrap();
    let filled = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let ran = Ran::default();
    let handled = |queue: usize, place: usize, command_buffer: &CommandBuffer| {
        let ran = Arc::clone(&ran);
        (command_buffer.add_completed_handler(Box::new(move |done: &CommandBuffer| {
            ran.lock().unwrap().push((queue, place, done.error()));
        })))
        .unwrap();
    };
    let mut held = Vec::new();
    for (at, queue) in queues.iter().enumerate() {
        let waiting = queue.command_buffer().unwrap();
        (waiting.encode_wait_for_event(EventRef::Shared(&event), 1)).unwrap();
        fill(&waiting, &filled);
        let behind = filled_by(queue, &filled);
        for (place, command_buffer) in [waiting, behind].into_iter().enumerate() {
            handled(at, place, &command_buffer);
            command_buffer.commit().unwrap();
            held.push(command_buffer);
        }
    }
    device.recover_for_test().unwrap();
    // A buffer of the new generation: a pooled `Shared` buffer made before may be left there.
    let after = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    for (at, queue) in queues.iter().enumerate() {
        let later = filled_by(queue, &after);
        handled(at, 2, &later);
        later.commit().unwrap();
        assert_eq!(outcome(&later), (CommandBufferStatus::Completed, None));
    }
    for command_buffer in &held {
        assert_eq!(
            outcome(command_buffer),
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::GpuRecoveryVictim)
            )
        );
    }
    let ran = ran.lock().unwrap();
    for at in 0..queues.len() {
        let of_queue: Vec<_> = (ran.iter())
            .filter(|(queue, ..)| *queue == at)
            .map(|&(_, place, error)| (place, error))
            .collect();
        let victim = Some(CommandBufferError::GpuRecoveryVictim);
        assert_eq!(of_queue, [(0, victim), (1, victim), (2, None)], "{ran:?}");
    }
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

/// A failed wait's drain of the device's queues (`Device::drain_queues`), taken while the
/// device's queue-switch check is between the two batches' submissions of a round on one
/// `VkQueue` (`queue_switch_check.rs`), completes once the round goes on: the round holds its
/// queue's lock through its last submission, so the drain never waits under that lock while the
/// check waits for it, and the round's batches submitted complete with no further call of the
/// check's (their gate opens on the GPU).
#[test]
fn a_drain_beside_the_queue_switch_check_s_round_completes() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let switch = Arc::clone(&device.shared().queue_switch);
    if switch.verdict() != crate::queue_share::Verdict::Pending {
        println!(
            "SKIPPED {}: the device runs no queue-switch check",
            device.name()
        );
        return;
    }
    switch.between.hold();
    // The device's first command queue starts its check.
    let queue = device.new_command_queue().unwrap();
    switch.between.until_waiting(1, WAIT);
    let (drained, drain) = std::sync::mpsc::channel();
    let draining = device.clone();
    std::thread::spawn(move || {
        draining.drain_queues();
        let _ = drained.send(());
    });
    // The drain reaches the round's `VkQueue` meanwhile.
    std::thread::sleep(Duration::from_millis(200));
    switch.between.release();
    let answer = drain.recv_timeout(WAIT);
    drop(queue);
    assert!(
        answer.is_ok(),
        "the drain did not complete in {WAIT:?} once the check's round went on"
    );
}

/// The threads that call the device while it recovers ([`each_recovery_announces_its_own_wait`]).
const WAITERS: usize = 3;

/// The recovery notices `heard` holds from `from` on whose object is `object`: a recovery's own
/// is "the device", a call's wait for one "a call made during the device's recovery", its command
/// buffers' failing past the bound "the device's recovery" (`device/line.rs`).
pub(super) fn recovery_notices(heard: &Heard, from: usize, object: &str) -> Vec<String> {
    heard.lock().unwrap()[from..]
        .iter()
        .filter(|notice| notice.degradation == Degradation::DeviceRecovered)
        .filter(|notice| notice.object == object)
        .map(|notice| notice.reason.clone())
        .collect()
}

/// A recovery's wait, and its command buffers' failing past the bound, are announced at a later
/// recovery as at the first, each once per recovery however many calls wait: the kinds a device
/// announced go on across its generations, a recovery's notices do not. The bound is zero here,
/// so a call that meets the recovery under way announces its wait at once, and so does a recovery
/// whose work in flight has yet to fail. The first recovery, which no call waits for, announces
/// itself once.
#[test]
fn each_recovery_announces_its_own_wait() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    *device.line().hooks.notice_after.lock().unwrap() = Some(Duration::ZERO);
    device.recover_for_test().unwrap();
    assert_eq!(recovery_notices(&heard, 0, "the device").len(), 1);
    let queue = device.new_command_queue().unwrap();
    let (mut waited, mut failed_slowly) = (false, false);
    for _ in 0..ATTEMPTS {
        let from = heard.lock().unwrap().len();
        // Work in flight, so the recovery waits for it to fail before it makes its `VkDevice`.
        let in_flight = busy(&device, &queue);
        let recovered = AtomicBool::new(false);
        std::thread::scope(|scope| {
            for _ in 0..WAITERS {
                scope.spawn(|| {
                    while !recovered.load(Ordering::Acquire) {
                        drop(device.new_buffer(4096, ResourceOptions::shared()).unwrap());
                    }
                });
            }
            device.recover_for_test().unwrap();
            recovered.store(true, Ordering::Release);
        });
        drop(in_flight);
        let made = recovery_notices(&heard, from, "the device");
        assert_eq!(made.len(), 1, "one recovery announced {made:?}");
        let failing = recovery_notices(&heard, from, "the device's recovery");
        assert!(failing.len() <= 1, "one recovery announced {failing:?}");
        failed_slowly |= failing.len() == 1;
        let waits = recovery_notices(&heard, from, "a call made during the device's recovery");
        assert!(
            waits.len() <= 1,
            "{} calls announced their wait for one recovery: {waits:?}",
            waits.len()
        );
        waited |= waits.len() == 1;
        if waited && failed_slowly {
            return;
        }
    }
    panic!(
        "of {ATTEMPTS} recoveries after the first, none announced {}",
        match (waited, failed_slowly) {
            (false, false) => "a call's wait or its command buffers' slow failing",
            (false, true) => "a call's wait (no call of the threads met one under way)",
            _ => "its command buffers' slow failing",
        }
    );
}

/// A recovery whose new `VkDevice` cannot be made announces it after an earlier recovery as on
/// the first, and the device answers as a lost one from then on.
#[test]
fn a_later_recovery_announces_its_failure() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    device.recover_for_test().unwrap();
    assert_eq!(recovery_notices(&heard, 0, "the device").len(), 1);
    let from = heard.lock().unwrap().len();
    device
        .line()
        .hooks
        .making_fails
        .store(true, Ordering::Release);
    assert!(matches!(device.recover_for_test(), Err(Error::DeviceLost)));
    let failed = recovery_notices(&heard, from, "the device");
    assert_eq!(failed.len(), 1, "the failed recovery announced {failed:?}");
    assert!(
        failed[0].contains("its new VkDevice could not be made"),
        "{failed:?}"
    );
    assert!(matches!(
        device.new_buffer(4096, ResourceOptions::shared()),
        Err(Error::DeviceLost)
    ));
    assert_eq!(device.work_counts().device_recoveries, 1);
}

/// A command queue's lent `VkQueue` comes with the `VkDevice` it belongs to, read under the same
/// lock (`CommandQueue::with_vk_queue`): after a recovery the queue lends the new generation's
/// `VkQueue`, and the device it names with it is the new `VkDevice`, which `Device::raw` answers
/// at its next call. The handles `raw` answered before the recovery, the caller's one source of the
/// device before `with_vk_queue` named it, still name the lost one: a caller that took its device
/// from them would submit the new `VkQueue` work recorded on the lost `VkDevice`.
#[test]
fn a_lent_queue_names_the_vk_device_it_belongs_to_across_a_recovery() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    // SAFETY: the handles are read and not kept; nothing is submitted.
    let lent =
        || unsafe { queue.with_vk_queue(|vk_device, vk_queue, _| (vk_device.handle(), vk_queue)) };
    // SAFETY: as above.
    let raw = || unsafe { device.raw() }.device.handle();
    let (before, before_queue) = lent();
    let raw_before = raw();
    assert_eq!(
        before, raw_before,
        "the lent queue's device is the device's"
    );
    device.recover_for_test().unwrap();
    let (after, after_queue) = lent();
    assert_ne!(
        after, before,
        "the lent queue after the recovery names the lost VkDevice"
    );
    assert_ne!(
        after_queue, before_queue,
        "the queue lends the new VkDevice's VkQueue"
    );
    assert_eq!(
        after,
        raw(),
        "the lent queue's device is the one Device::raw answers after the recovery"
    );
    assert_ne!(
        after, raw_before,
        "the handles answered before the recovery name the new VkDevice"
    );
    assert_eq!(device.work_counts().device_recoveries, 1);
    eprintln!(
        "LENT-QUEUE-DEVICE: {}: before the recovery {before:?} (raw {raw_before:?}), after it \
         {after:?}, queue {before_queue:?} then {after_queue:?}",
        device.name()
    );
}

/// A device call whose generation is lost under it, refused for that, runs again on the generation
/// serving next (`Device::at_serving`): the loss here comes inside the call's first run, as a
/// recovery on another thread may, and the call answers with a buffer of the new generation.
#[test]
fn a_call_refused_for_a_loss_under_it_runs_again_on_the_next_generation() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let mut runs = 0;
    let buffer = device
        .at_serving(|serving| {
            runs += 1;
            if runs == 1 {
                serving.recover_for_test().unwrap();
            }
            serving.new_buffer_now(4096, ResourceOptions::shared())
        })
        .unwrap();
    assert_eq!(runs, 2, "the call ran once more, on the next generation");
    assert!(buffer.device().generation_serves());
    assert_eq!(device.work_counts().device_recoveries, 1);
}

/// A device call that took its arguments (`Device::at_serving_taking`) before its generation's
/// loss is not refused for it: it completes on that generation, as one the loss followed, and
/// its buffer is one made before the recovery.
#[test]
fn a_call_that_took_its_arguments_completes_on_a_generation_lost_after() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    // A pooled `Shared` buffer: in memory the device allocated, which a recovery leaves refused;
    // in a block of the runtime's own pages, which its maker carries to the next generation once
    // the call completed (`device/carry.rs`).
    let buffer = device
        .at_serving_taking(4096, |serving, length| {
            serving.recover_for_test().unwrap();
            serving.new_buffer_now(length, ResourceOptions::shared())
        })
        .unwrap();
    if buffer.in_page_block() {
        assert!(buffer.device().generation_serves());
    } else {
        assert!(!buffer.device().generation_serves());
        assert!(matches!(
            buffer
                .generation()
                .new_buffer_now(4096, ResourceOptions::shared()),
            Err(Error::NotImplemented(crate::RuntimeGap::MadeBeforeRecovery))
        ));
    }
    // The device itself goes on, on the new generation.
    let next = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    assert!(next.device().generation_serves());
}
