//! A command buffer starts on the storage of the lists of records a released command buffer of
//! its device left, emptied with its room kept (`command_buffer/record_storage.rs`): one of no
//! more encoders than that one grows none of its lists (`WorkCounts::record_lists_grown`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `compute_parked.rs`'s.

use super::blit_holds::{until, Scratch};
use super::compute_parked::Kernel;
use crate::{Buffer, CommandQueue};

/// The compute encoders of each command buffer, one dispatch each.
const ENCODERS: u64 = 32;

/// How many lists of a command buffer of [`ENCODERS`] compute encoders of `queue` grew, the
/// command buffer committed, completed and dropped: read once its device kept its storage
/// (`WorkCounts::record_storage_kept`), at its release.
fn grown(kernel: &Kernel, queue: &CommandQueue, output: &Buffer) -> u64 {
    let before = kernel.device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    for _ in 0..ENCODERS {
        kernel.dispatch(&command_buffer, (output, 0));
    }
    kernel.run(&command_buffer);
    drop(command_buffer);
    until("the command buffer's release kept its storage", || {
        kernel.device.work_counts().record_storage_kept > before.record_storage_kept
    });
    kernel.device.work_counts().record_lists_grown - before.record_lists_grown
}

/// Command buffers of [`ENCODERS`] compute encoders each, one after another's release, the kernel's
/// pipelines made first (a dispatch whose pipeline is being made ends its group, so the groups
/// and the lists of a command buffer that meets one differ from the next one's): the first's lists
/// grow past the room the warming command buffer of one dispatch left (its ended encoders'
/// records and their recordings at least), and each later one starts on the room the one before
/// it left and grows none.
#[test]
fn a_command_buffer_of_as_many_encoders_as_the_last_grows_no_list() {
    let scratch = Scratch::new("record-storage");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    until(
        "the warming command buffer's release kept its storage",
        || kernel.device.work_counts().record_storage_kept > 0,
    );
    let queue = kernel.device.new_command_queue().unwrap();
    let output = kernel.output(1);
    let first = grown(&kernel, &queue, &output);
    assert!(
        first >= 2,
        "the first command buffer's lists grown: {first}"
    );
    for round in 1..3 {
        let grown = grown(&kernel, &queue, &output);
        assert_eq!(grown, 0, "round {round}: the lists grown");
    }
}
