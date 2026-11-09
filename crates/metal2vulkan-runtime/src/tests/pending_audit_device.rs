//! The pending audit on a device (`pending_audit.rs`): a command pool given back to the device's
//! spares while a command buffer of it waits on the GPU is found, once, and said in a line; the
//! command buffer that then takes that pool (its taking resets it), the primary its completing
//! thread begins ahead in the pool, and the command buffer that takes that primary are each
//! checked, and none is found pending. The command buffer is held on the GPU by its wait for a
//! plain event's signal reserved for the caller (`Event::reserve_vk_signal`), which the test makes
//! from the host once the give-back was found: the contract's signal where the caller's submission
//! is not made.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{options, until, Scratch, WAIT};
use crate::pending_audit::{Counts, Freeing};
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, Event, MessageSink,
    ResourceOptions, VkEventSignal,
};
use ash::vk;
use std::sync::{Arc, Mutex};

/// The bytes each command buffer fills.
const BYTES: u64 = 4096;

/// The id name of the audit's lines.
const AUDIT: &str = "metal2vulkan-pending-audit";

/// A committed command buffer of `queue` that fills `target`, after a wait for `event` to reach 1
/// where given.
fn filling(queue: &CommandQueue, target: &Buffer, event: Option<&Event>) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    if let Some(event) = event {
        command_buffer
            .encode_wait_for_event(event.into(), 1)
            .unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..BYTES, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// The checks and findings of `device`'s audit so far.
fn counts(device: &Device) -> Counts {
    let audit = device.shared().pending_audit();
    audit.expect("the device's pending audit is on").counts()
}

#[test]
fn a_pool_given_back_while_its_command_buffer_waits_on_the_gpu_is_found() {
    let scratch = Scratch::new("pending-audit");
    let lines = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&lines);
    let mut options = options(&scratch.0);
    options.pending_audit = true;
    options.messages = Some(MessageSink::new(move |message| {
        if message.id_name == AUDIT {
            kept.lock().unwrap().push(message.text.clone());
        }
    }));
    let device = Device::system_default_with(&options).unwrap();
    let said = || lines.lock().unwrap().clone();
    assert!(
        said().iter().any(|line| line.contains("pending audit: on")),
        "the audit says it is on: {:?}",
        said()
    );
    let queue = device.new_command_queue().unwrap();
    let target = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    let event = device.new_event().unwrap();
    // SAFETY: the reserved signal is made below, from the host, once the give-back was found,
    // and waits for no earlier signal (asserted): the contract's signal where the caller's
    // submission is not made. The event lives until it landed.
    let reserved = unsafe { event.reserve_vk_signal(1) }.unwrap();
    let VkEventSignal::Counter { counter, after } = reserved else {
        panic!("a signal above the event's value is reserved: {reserved:?}");
    };
    assert_eq!(after, None, "the event's first signal waits for none");
    let held = filling(&queue, &target, Some(&event));
    until("the held command buffer submitted", || {
        held.status() == CommandBufferStatus::Scheduled
    });
    assert_eq!(counts(&device).found, [0; 4], "{:?}", said());

    // The deliberate early give-back: its command buffers wait on the GPU for the signal.
    assert!(
        held.give_pool_back_early(),
        "the held command buffer holds its pool"
    );
    let given = counts(&device);
    let mut expected = [0; 4];
    expected[Freeing::GiveBack as usize] = 1;
    assert_eq!(given.found, expected, "{:?}", said());
    assert!(
        said()
            .iter()
            .any(|line| line.contains("its pool given back to the device's spares")),
        "the finding is said: {:?}",
        said()
    );

    // The reserved signal lets the held command buffer run.
    let raw = device.shared().vk();
    // SAFETY: the event's timeline semaphore, signalled to its reserved counter once, as above.
    let semaphore = unsafe { event.vk_semaphore() };
    let signal = vk::SemaphoreSignalInfo::default()
        .semaphore(semaphore)
        .value(counter);
    // SAFETY: as above; the semaphore is below `counter`, which nothing else signals.
    unsafe { raw.signal_semaphore(&signal) }.unwrap();
    held.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(held.status(), CommandBufferStatus::Completed);

    // The pool given back, taken (and reset) by the next command buffer, which completes while
    // no thread waits for it, so its completing thread begins a primary of the pool ahead; then a
    // command buffer that takes that primary. Nothing is found pending.
    let checked = counts(&device);
    let work = device.work_counts();
    let next = filling(&queue, &target, None);
    until(
        "the next command buffer completed, no thread waiting",
        || next.status() == CommandBufferStatus::Completed,
    );
    until("its pool's primary begun ahead", || {
        device.work_counts().primaries_begun_ahead > work.primaries_begun_ahead
    });
    let last = filling(&queue, &target, None);
    last.wait_until_completed_for(WAIT).unwrap();
    let done = device.work_counts();
    assert!(
        done.spare_command_pools_reset > work.spare_command_pools_reset,
        "the next command buffer reset the pool given back"
    );
    assert!(
        done.encoder_primaries_begun_ahead > work.encoder_primaries_begun_ahead,
        "the last command buffer took the primary begun ahead"
    );
    let audited = counts(&device);
    for what in [Freeing::Begin, Freeing::PoolReset, Freeing::GiveBack] {
        assert!(
            audited.checks[what as usize] > checked.checks[what as usize],
            "{what:?} checked: {checked:?} then {audited:?}"
        );
    }
    assert_eq!(
        audited.found,
        given.found,
        "nothing more found: {:?}",
        said()
    );
}
