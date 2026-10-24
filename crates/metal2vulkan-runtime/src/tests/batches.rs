//! A command buffer's batches (`command_buffer/batches.rs`): where event waits and signals, and
//! binary semaphore waits and signals, split its encoders; which drivers' batches may wait in
//! a group (`command_buffer/group.rs`); and which drivers' queue failures are relayed
//! (`device/relay.rs`).

use crate::command_buffer::batches::{Batch, Batches};
use crate::command_buffer::merges_submissions;
use metal2vulkan::device::{DeviceDescription, DriverFact};
use metal2vulkan::vk::DriverId;

fn batch(waits: &[(char, u64)], end: usize, signals: &[(char, u64)]) -> Batch<char, u8> {
    Batch {
        waits: waits.to_vec(),
        binary_waits: Vec::new(),
        end,
        signals: signals.to_vec(),
        binary_signals: Vec::new(),
    }
}

/// A batch with binary semaphore waits and signals only.
fn binary(waits: &[u8], end: usize, signals: &[u8]) -> Batch<char, u8> {
    Batch {
        waits: Vec::new(),
        binary_waits: waits.to_vec(),
        end,
        signals: Vec::new(),
        binary_signals: signals.to_vec(),
    }
}

#[test]
fn a_command_buffer_has_one_batch_even_with_nothing_encoded() {
    let batches = Batches::<char, u8>::default();
    assert_eq!(batches.all(), [batch(&[], 0, &[])]);
}

#[test]
fn waits_lead_and_signals_close_a_batch() {
    let mut batches = Batches::<char, u8>::default();
    batches.wait('a', 1);
    batches.encoder();
    batches.encoder();
    batches.signal('b', 2);
    assert_eq!(batches.all(), [batch(&[('a', 1)], 2, &[('b', 2)])]);
}

#[test]
fn a_wait_after_encoders_starts_a_batch() {
    let mut batches = Batches::<char, u8>::default();
    batches.encoder();
    batches.wait('a', 1);
    batches.encoder();
    assert_eq!(
        batches.all(),
        [batch(&[], 1, &[]), batch(&[('a', 1)], 2, &[])]
    );
}

#[test]
fn a_wait_after_a_signal_starts_a_batch() {
    // Metal ran the work before the wait and signaled while the wait was pending (measured).
    let mut batches = Batches::<char, u8>::default();
    batches.signal('e', 1);
    batches.wait('f', 1);
    batches.encoder();
    batches.signal('g', 1);
    assert_eq!(
        batches.all(),
        [
            batch(&[], 0, &[('e', 1)]),
            batch(&[('f', 1)], 1, &[('g', 1)])
        ]
    );
}

#[test]
fn an_encoder_after_a_signal_starts_a_batch() {
    let mut batches = Batches::<char, u8>::default();
    batches.encoder();
    batches.signal('a', 1);
    batches.encoder();
    assert_eq!(
        batches.all(),
        [batch(&[], 1, &[('a', 1)]), batch(&[], 2, &[])]
    );
}

#[test]
fn waits_before_any_encoder_share_the_first_batch() {
    let mut batches = Batches::<char, u8>::default();
    batches.wait('a', 1);
    batches.wait('b', 2);
    batches.encoder();
    assert_eq!(batches.all(), [batch(&[('a', 1), ('b', 2)], 1, &[])]);
}

#[test]
fn one_event_twice_in_a_batch_is_its_largest_value() {
    let mut batches = Batches::<char, u8>::default();
    batches.wait('a', 5);
    batches.wait('a', 3);
    batches.encoder();
    batches.signal('b', 2);
    batches.signal('b', 9);
    batches.signal('b', 4);
    assert_eq!(batches.all(), [batch(&[('a', 5)], 1, &[('b', 9)])]);
}

#[test]
fn a_binary_wait_and_signal_frame_the_encoders_as_events_do() {
    // The present path: wait for the acquire, render, signal for the present.
    let mut batches = Batches::<char, u8>::default();
    batches.binary_wait(1);
    batches.encoder();
    batches.encoder();
    batches.binary_signal(2);
    assert_eq!(batches.all(), [binary(&[1], 2, &[2])]);
}

#[test]
fn a_binary_wait_after_encoders_or_a_signal_starts_a_batch() {
    let mut batches = Batches::<char, u8>::default();
    batches.encoder();
    batches.binary_wait(1);
    batches.encoder();
    batches.binary_signal(2);
    batches.binary_wait(3);
    batches.encoder();
    assert_eq!(
        batches.all(),
        [
            binary(&[], 1, &[]),
            binary(&[1], 2, &[2]),
            binary(&[3], 3, &[])
        ]
    );
}

#[test]
fn an_encoder_after_a_binary_signal_starts_a_batch() {
    let mut batches = Batches::<char, u8>::default();
    batches.encoder();
    batches.binary_signal(1);
    batches.encoder();
    assert_eq!(batches.all(), [binary(&[], 1, &[1]), binary(&[], 2, &[])]);
}

#[test]
fn one_binary_semaphore_twice_starts_a_batch_each_time() {
    // Each wait consumes a signal of its own and each signal needs a wait between: never merged.
    let mut batches = Batches::<char, u8>::default();
    batches.binary_wait(1);
    batches.binary_wait(1);
    batches.encoder();
    batches.binary_signal(2);
    batches.binary_signal(2);
    assert_eq!(
        batches.all(),
        [
            binary(&[1], 0, &[]),
            binary(&[1], 1, &[2]),
            binary(&[], 1, &[2])
        ]
    );
}

#[test]
fn event_and_binary_waits_share_a_batch_before_its_encoders() {
    let mut batches = Batches::<char, u8>::default();
    batches.wait('a', 1);
    batches.binary_wait(7);
    batches.encoder();
    batches.signal('b', 2);
    batches.binary_signal(8);
    let mut expected = batch(&[('a', 1)], 1, &[('b', 2)]);
    expected.binary_waits.push(7);
    expected.binary_signals.push(8);
    assert_eq!(batches.all(), [expected]);
}

/// The part of an encoder that executes commands the GPU may write starts a batch of its own, with
/// no wait or signal of its own.
#[test]
fn a_host_step_starts_a_batch() {
    let mut batches = Batches::<char, u8>::default();
    batches.wait('a', 1);
    batches.encoder();
    batches.host_step();
    batches.encoder();
    batches.signal('b', 2);
    assert_eq!(
        batches.all(),
        [batch(&[('a', 1)], 1, &[]), batch(&[], 2, &[('b', 2)])]
    );
}

/// Every event wait and signal call is counted, in the order encoded, whether it joins a batch's
/// wait or signal of the same event or not; binary semaphore waits and signals are not
/// (`CommandBuffer::ended_encoders`).
#[test]
fn event_waits_and_signals_are_counted_call_by_call() {
    let mut batches = Batches::<char, u8>::default();
    assert_eq!(batches.events(), (0, 0));
    batches.encoder();
    batches.wait('a', 1);
    batches.wait('a', 2);
    assert_eq!(batches.events(), (2, 0));
    batches.encoder();
    batches.signal('b', 1);
    batches.signal('b', 3);
    batches.binary_wait(7);
    batches.binary_signal(8);
    assert_eq!(batches.events(), (2, 2));
    batches.host_step();
    batches.sparse();
    batches.encoder();
    assert_eq!(batches.events(), (2, 2));
    batches.drop_signals();
    assert_eq!(batches.events(), (2, 2));
}

/// A command buffer Metal fails from an encoder on signals the events encoded before it, not
/// after: the batches that end past it lose their event signals, and keep their binary ones.
#[test]
fn signals_after_a_failing_encoder_are_dropped() {
    let mut batches = Batches::<char, u8>::default();
    batches.encoder();
    batches.signal('a', 1);
    batches.encoder();
    batches.encoder();
    batches.signal('b', 2);
    batches.binary_signal(9);
    batches.drop_signals_after(1);
    assert_eq!(
        batches.all(),
        [
            batch(&[], 1, &[('a', 1)]),
            Batch {
                binary_signals: vec![9],
                ..batch(&[], 3, &[])
            }
        ]
    );
}

/// Whether a driver's batches may wait in one kernel submission is the driver's **fact**, never
/// its name (`command_buffer/group.rs`): a driver measured to merge a `vkQueueSubmit`'s
/// `VkSubmitInfo`s that signal nothing into the next one's kernel submission
/// (`DriverFact::SubmissionsWithoutSignalsMerge`) groups its batches; every other driver takes the
/// ungrouped path, each batch its own call, which is exact alike.
///
/// The check over every committed profile: a profile groups only where its driver is one the fact
/// is measured of, so a decision that read the driver's name -- the Mesa drivers no lane measured
/// read alike there -- would answer otherwise for one of them. A `MEASURED` row written without a
/// measurement would show here as a MoltenVK, NVIDIA or SwiftShader profile that groups.
#[test]
fn only_drivers_measured_to_merge_their_submissions_group_their_batches() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut read = 0;
    let mut grouped = 0;
    for entry in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = entry.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let description =
            DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let merges = merges_submissions(&description.driver_facts());
        if merges {
            // Only a driver the fact is measured of: Mesa's RADV, ANV and llvmpipe.
            assert!(
                matches!(
                    description.driver_id,
                    Some(
                        DriverId::MESA_RADV
                            | DriverId::INTEL_OPEN_SOURCE_MESA
                            | DriverId::MESA_LLVMPIPE
                    )
                ),
                "{} groups its batches with no row of its driver",
                path.display()
            );
            grouped += 1;
        }
        read += 1;
    }
    assert!(read > 1, "profiles read: {read}");
    // The grouped form is reachable, and the fact alone chooses it.
    assert!(grouped > 0, "profiles that group: {grouped}");
    assert!(merges_submissions(
        &[DriverFact::SubmissionsWithoutSignalsMerge]
            .into_iter()
            .collect()
    ));
    assert!(!merges_submissions(&Default::default()));
}

/// Whether a device's queue failures are relayed to the command buffers that failed is the
/// driver's **fact**, never its name (`device/mod.rs`, `device/relay.rs`): a driver that reports
/// a command buffer's execution failure through the instance's messenger alone, and loses the
/// whole device for it, takes the relay
/// (`DriverFact::QueueFailuresReportedByMessenger`); every other driver takes none, and its
/// batches take a fence only where their command buffer ends at one.
///
/// It is a **defect** fact, so a driver with a row takes it at every release and on every GPU
/// (`DriverFact::assumed_where_unmeasured`): that is what makes the fact answer exactly as the
/// driver's name did, and the check is that MoltenVK's profiles take it while every other
/// committed profile -- RADV's, ANV's, NVIDIA's, lavapipe's and SwiftShader's -- does not,
/// including at a release and on a GPU no lane measured.
#[test]
fn only_drivers_that_report_their_failures_by_messenger_relay_them() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut moltenvk = 0;
    let mut other = 0;
    for entry in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = entry.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let mut description =
            DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let is_moltenvk = description.driver_id == Some(DriverId::MOLTENVK);
        for release in [description.driver_version, 0x4242] {
            description.driver_version = release;
            let relays = (description.driver_facts())
                .contains(&DriverFact::QueueFailuresReportedByMessenger);
            assert_eq!(
                relays,
                is_moltenvk,
                "{} relays {relays} at {release:#x}",
                path.display()
            );
        }
        match is_moltenvk {
            true => moltenvk += 1,
            false => other += 1,
        }
    }
    assert!(
        moltenvk > 0 && other > 1,
        "{moltenvk} MoltenVK, {other} other"
    );
}
