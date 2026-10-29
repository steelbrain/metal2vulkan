//! Device tests of the commands recorded, by kind (the runtime's `recorder`,
//! `WorkCounts::commands_replayed`), on the device the cfg selects, under the validation layer: a
//! command buffer of one blit encoder of fills and copies records each as one call of its own
//! kind, counted exactly: a fill of one word as `vkCmdFillBuffer`, or as `vkCmdUpdateBuffer` where
//! the runtime records fills as transfers (`WorkCounts::fills_by_transfers`, MoltenVK 1.4.1).

use super::host_memory::on_device;
use super::{read_buffer, WAIT};
use metal2vulkan_runtime::{CommandBufferStatus, CommandsReplayed, ResourceOptions};

/// The fills and the copies the command buffer encodes, each of [`WORD`] bytes at a word of its
/// own.
const FILLS: u64 = 5;
const COPIES: u64 = 2;
const WORD: u64 = 4;

/// The counts `after` holds past `before`, of the fills', the updates', the copies' and the
/// ends' kinds.
fn added(before: &CommandsReplayed, after: &CommandsReplayed) -> [u64; 4] {
    [
        after.cmd_fill_buffer - before.cmd_fill_buffer,
        after.cmd_update_buffer - before.cmd_update_buffer,
        after.cmd_copy_buffer - before.cmd_copy_buffer,
        after.end_command_buffer - before.end_command_buffer,
    ]
}

/// One blit encoder of [`FILLS`] fills of a buffer's words, then [`COPIES`] copies of its first
/// words into another's, committed and completed: it recorded `vkCmdFillBuffer` once a fill
/// recorded so and `vkCmdUpdateBuffer` once a fill recorded as transfers (one word: no copy),
/// `vkCmdCopyBuffer` once a copy, and ended at least its encoder's command buffer. The bytes are
/// what the fills and copies wrote. A device past the runtime's slots counts nothing, printed.
#[test]
fn each_command_recorded_counts_once_by_kind() {
    on_device(|device| {
        let name = device.name();
        let queue = device.new_command_queue().unwrap();
        let filled = (device.new_buffer(WORD * FILLS, ResourceOptions::shared())).unwrap();
        let copied = (device.new_buffer(WORD * COPIES, ResourceOptions::shared())).unwrap();
        let before = device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        for at in 0..FILLS {
            (blit.fill_buffer(&filled, WORD * at..WORD * (at + 1), at as u8 + 1)).unwrap();
        }
        for at in 0..COPIES {
            (blit.copy_from_buffer_to_buffer(&filled, WORD * at, &copied, WORD * at, WORD))
                .unwrap();
        }
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let after = device.work_counts();
        let expected: Vec<u8> = (0..FILLS as u8)
            .flat_map(|at| [at + 1; WORD as usize])
            .collect();
        assert_eq!(read_buffer(&filled, 0, WORD * FILLS).unwrap(), expected);
        assert_eq!(
            read_buffer(&copied, 0, WORD * COPIES).unwrap(),
            expected[..(WORD * COPIES) as usize]
        );
        if after.commands_replayed == before.commands_replayed {
            eprintln!("COMMANDS-RECORDED {name}: took no slot, so counts nothing");
            return;
        }
        let [fills, updates, copies, ends] =
            added(&before.commands_replayed, &after.commands_replayed);
        let recorded = after.fills_recorded - before.fills_recorded;
        let by_transfers = after.fills_by_transfers - before.fills_by_transfers;
        eprintln!(
            "COMMANDS-RECORDED {name}: {fills} vkCmdFillBuffer, {updates} vkCmdUpdateBuffer, \
             {copies} vkCmdCopyBuffer, {ends} vkEndCommandBuffer recorded; {recorded} fills \
             recorded so, {by_transfers} as transfers"
        );
        assert_eq!(recorded + by_transfers, FILLS, "the fills on {name}");
        assert_eq!(
            [fills, updates, copies],
            [recorded, by_transfers, COPIES],
            "the fills' and copies' calls recorded on {name}"
        );
        assert!(ends >= 1, "no command buffer ended on {name}");
    });
}
