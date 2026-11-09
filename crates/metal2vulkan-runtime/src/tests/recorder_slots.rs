//! A device made while every one of the recorder's slots is held (`recorder::SLOTS`, each by a
//! device of the process until its teardown) records its commands uncounted and says so
//! (`WorkCounts::commands_uncounted`), its `commands_recorded` reading none of them; a device the
//! program let go, its last drop returned, holds its slot no more, so the next device made is
//! counted.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects; each test runs in a process of its own,
//! whose devices are its own.

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{device, Heard};
use crate::recorder::SLOTS;
use crate::{CommandBufferStatus, CommandsRecorded, Device, ResourceOptions};

/// A device of a message sink of its own, so a device of its own (`create.rs` groups the devices
/// it makes by their options and sink).
fn own_device(scratch: &Scratch) -> Device {
    device(&scratch.0, &[], &Heard::default())
}

/// Fill a buffer on `device` and wait for it: a command buffer's begin, a fill and an end.
fn fill(device: &Device) {
    let buffer = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 0..256, 7).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

#[test]
fn a_device_past_the_recorder_s_slots_says_its_commands_go_uncounted() {
    let scratch = Scratch::new("recorder-slots");
    let mut held: Vec<Device> = (0..SLOTS).map(|_| own_device(&scratch)).collect();
    for device in &held {
        assert_eq!(
            device.work_counts().commands_uncounted,
            0,
            "a device with a slot"
        );
    }
    let past = own_device(&scratch);
    assert_eq!(
        past.work_counts().commands_uncounted,
        1,
        "the device made while every slot was held"
    );
    fill(&past);
    assert_eq!(
        past.work_counts().commands_recorded,
        CommandsRecorded::default(),
        "the uncounted device's commands read none"
    );
    fill(&held[0]);
    let counted = held[0].work_counts().commands_recorded;
    assert!(
        counted.begin_command_buffer > 0 && counted.cmd_fill_buffer > 0,
        "a counted device's fill: {counted:?}"
    );
    // A device that recorded nothing, the program's last handle of it dropped: torn down on this
    // thread before the drop returns, its slot free.
    drop(held.pop());
    let again = own_device(&scratch);
    assert_eq!(
        again.work_counts().commands_uncounted,
        0,
        "the device made once a slot was let go"
    );
}
