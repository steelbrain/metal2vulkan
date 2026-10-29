//! An encoder records each command into the driver's `VkCommandBuffer` at the call, on the
//! program's thread (`command_buffer`'s `allocate_raw`): the device's count of the commands
//! recorded (`WorkCounts::commands_recorded`, counted as each goes through its slot, `recorder`)
//! moves at the call itself, before the encoder ends and before the commit, where a command
//! logged for a later recording would move it only after the commit.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{CommandBufferStatus, CommandsRecorded, ResourceOptions};

/// The fills recorded so far, as `vkCmdFillBuffer` or, where the runtime records fills as
/// transfers, `vkCmdUpdateBuffer` (`WorkCounts::fills_by_transfers`).
fn fills(counts: &CommandsRecorded) -> u64 {
    counts.cmd_fill_buffer + counts.cmd_update_buffer
}

/// A blit encoder's fill of a buffer's word is in the driver's command buffer as the call
/// returns: the count of fills recorded rose by one at the call, the encoder still open and the
/// command buffer uncommitted, and by no more at its end and commit. The bytes are the fill's.
#[test]
fn a_fill_is_recorded_at_its_call() {
    let scratch = Scratch::new("recorded-at-the-call");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let buffer = (device.new_buffer(16, ResourceOptions::shared())).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let before = device.work_counts().commands_recorded;
    blit.fill_buffer(&buffer, 4..8, 0x5a).unwrap();
    let at_call = device.work_counts().commands_recorded;
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let after = device.work_counts().commands_recorded;
    println!(
        "RECORDED-AT-THE-CALL {}: fills recorded at the call {}, after the commit {}",
        device.name(),
        fills(&at_call) - fills(&before),
        fills(&after) - fills(&before),
    );
    assert_eq!(
        fills(&at_call) - fills(&before),
        1,
        "the fill was recorded into the driver's command buffer at its call"
    );
    assert_eq!(
        fills(&after) - fills(&at_call),
        0,
        "nothing recorded the fill again after its call"
    );
    let bytes = buffer.contents().unwrap();
    // SAFETY: the buffer is shared, of 16 bytes, and its command buffer completed.
    let words = unsafe { std::slice::from_raw_parts(bytes.as_ptr(), 16) };
    assert_eq!(&words[4..8], &[0x5a; 4], "the fill's bytes");
}
