//! Another device's teardown beside a device's queue-switch check (`queue_switch_check.rs`) held
//! between the two submissions of a round, its first batch already submitted.
//!
//! The device's first command queue starts its check; the test holds the round between its two
//! submissions (`QueueSwitch::between`) and, on a thread of its own, makes a second device of the
//! process and drops it. The second device's teardown (its `vkDestroyDevice` among the rest) goes
//! while the round is held: no work of the first device's waits for a signal only the check's
//! thread can still make, so a driver call that waits for the GPU's work in flight waits for work
//! that completes. Seen without that: on an RTX 5080 Laptop GPU (NVIDIA 615.71.09) beside the
//! runtime suite, a device's `vkDestroyDevice` never returned while another device's check sat in
//! the `vkQueueSubmit` of its round's second batch, the first waiting on the GPU for a gate the
//! host was to signal once that call returned (`tests::recorder_slots`, 20 minutes).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that runs no check (one family of one `VkQueue`, no
//! calibrated timestamps) skips, saying so.

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{device, Heard};
use crate::queue_share::Verdict;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// How long the round is held while the other device is made and dropped.
const HELD: Duration = Duration::from_secs(20);

#[test]
fn a_device_dropped_while_another_s_queue_switch_round_is_held_goes() {
    let scratch = Scratch::new("queue-switch-gate");
    let checked = device(&scratch.0, &[], &Heard::default());
    let switch = Arc::clone(&checked.shared().queue_switch);
    if switch.verdict() != Verdict::Pending {
        println!(
            "SKIPPED {}: the device runs no queue-switch check",
            checked.name()
        );
        return;
    }
    switch.between.hold();
    // The device's first command queue starts its check, which stops between the two
    // submissions of its first round.
    let queue = checked.new_command_queue().unwrap();
    switch.between.until_waiting(1, WAIT);
    let (went, gone) = mpsc::channel();
    let cache = scratch.0.clone();
    std::thread::spawn(move || {
        // A device of a message sink of its own, so a device of its own (`create.rs`).
        let other = device(&cache, &[], &Heard::default());
        let dropping = Instant::now();
        drop(other);
        let _ = went.send(dropping.elapsed());
    });
    let held = gone.recv_timeout(HELD);
    switch.between.release();
    let after = match held {
        Ok(_) => None,
        Err(_) => Some(gone.recv_timeout(WAIT)),
    };
    println!(
        "QUEUE-SWITCH-GATE {}: the other device's teardown, while the round was held: {held:?}; \
         once it went on: {after:?}",
        checked.name()
    );
    assert!(
        held.is_ok(),
        "the other device's teardown did not end in {HELD:?} while the queue-switch round was \
         held between its submissions ({after:?} once it went on)"
    );
    drop(queue);
}
