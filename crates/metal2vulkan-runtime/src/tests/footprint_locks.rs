//! A batch's kept footprint (`CommandBufferShared::footprint`) is read without its command
//! buffer's state lock: the scheduler asks for the footprints of the batches it orders at each
//! pass, one at each commit and completion on every queue, while a host step's draws and a
//! commit's work take that lock, so a kept one's ask waits for neither
//! (`WorkCounts::footprint_lock_waits`). What it was built from moving builds it again.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::{Buffer, CommandBuffer, ResourceOptions};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

/// How long an ask of a kept footprint is given while the state is locked.
const ASKED: Duration = Duration::from_secs(5);

/// A blit encoder that fills `target`, ended on `command_buffer`.
fn fill(command_buffer: &CommandBuffer, target: &Buffer) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..64, 7).unwrap();
    blit.end_encoding().unwrap();
}

/// A batch's footprint, built once, is answered again while its command buffer's state is locked,
/// the same footprint, with no build and no wait; another encoder ended into the batch moves what
/// it is built from, and the next ask builds it again.
#[test]
fn a_kept_footprint_is_read_while_its_command_buffer_s_state_is_locked() {
    let scratch = Scratch::new("footprint-locks");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    fill(&command_buffer, &target);
    let built = command_buffer
        .footprint(0)
        .expect("a fill's footprint is known");
    let before = device.work_counts();
    let (answer, answered) = mpsc::channel();
    {
        let _locked = command_buffer.locked_state();
        let asking = command_buffer.clone();
        thread::spawn(move || answer.send(asking.footprint(0)).unwrap());
        let kept = answered
            .recv_timeout(ASKED)
            .expect("the ask of a kept footprint waited for the state's lock")
            .expect("a kept footprint");
        assert!(Arc::ptr_eq(&kept, &built), "a kept footprint built again");
    }
    let after = device.work_counts();
    assert_eq!(after.footprints_built, before.footprints_built);
    assert_eq!(after.footprint_lock_waits, before.footprint_lock_waits);
    fill(&command_buffer, &target);
    let again = command_buffer
        .footprint(0)
        .expect("two fills' footprint is known");
    assert!(
        !Arc::ptr_eq(&again, &built),
        "a footprint kept past an encoder ended into its batch"
    );
    assert_eq!(
        device.work_counts().footprints_built,
        after.footprints_built + 1
    );
}
