//! A device's queue-switch check (`queue_switch_check.rs`) held in a round on one `VkQueue`:
//! between its two batches' submissions, its opening and first batch submitted
//! (`QueueSwitch::between`), or once its last batch was submitted (`QueueSwitch::submitted`).
//!
//! The device's first command queue starts its check; each test holds the round at one of the
//! two and waits on the host meanwhile for the timeline values the batches the round submitted
//! signal, which complete: no work of the device's waits for a signal only the check's thread can
//! still make, so a driver call that waits for the GPU's work in flight waits for work that
//! completes. Seen otherwise, when the check's batches waited on the GPU for a gate the host
//! signalled after their submissions: on an RTX 5080 Laptop GPU (NVIDIA 615.71.09) beside the
//! runtime suite, another device's `vkDestroyDevice` and the check's `vkQueueSubmit` never
//! returned (`tests::recorder_slots`, 20 minutes), and a dma-buf import's making never returned
//! while the check sat in its submissions (`tests::dma_buf_leak_probe`); on an RTX 3090 (NVIDIA
//! 620.70.06), a recovery's `vkDestroyDevice` of its old `VkDevice` and the new one's check in
//! `vkQueueSubmit` (`tests::managed_copy_device`).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that runs no check (one family of one `VkQueue`, no
//! calibrated timestamps) skips, saying so.

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{device, Heard};
use crate::freed_memory::held::Held;
use crate::queue_share::{QueueSwitch, Verdict};
use ash::vk;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long the round's batches are waited for while it is held.
const HELD: Duration = Duration::from_secs(20);

/// Where a test holds the round.
#[derive(Clone, Copy, Debug)]
enum At {
    /// Between its two batches' submissions.
    Between,
    /// Once its last batch was submitted.
    Submitted,
}

impl At {
    /// The check's hold of this place.
    fn hold(self, switch: &QueueSwitch) -> &Held {
        match self {
            At::Between => &switch.between,
            At::Submitted => &switch.submitted,
        }
    }
}

/// Hold `at` the first round on one `VkQueue` of a device's queue-switch check and wait on the
/// host meanwhile, for at most [`HELD`], for the timeline values the batches the round submitted
/// before its hold signal: each completes while the round is held.
fn held_round_s_batches(at: At) {
    let scratch = Scratch::new("queue-switch-gate-batches");
    let checked = device(&scratch.0, &[], &Heard::default());
    let switch = Arc::clone(&checked.shared().queue_switch);
    if switch.verdict() != Verdict::Pending {
        println!(
            "SKIPPED {}: the device runs no queue-switch check",
            checked.name()
        );
        return;
    }
    let hold = at.hold(&switch);
    hold.hold();
    // The device's first command queue starts its check, which stops there in its first round on
    // one `VkQueue`.
    let queue = checked.new_command_queue().unwrap();
    hold.until_waiting(1, WAIT);
    let (semaphores, values): (Vec<vk::Semaphore>, Vec<u64>) =
        crate::device::lock(&switch.signalled)
            .iter()
            .copied()
            .unzip();
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    let waiting = Instant::now();
    // SAFETY: the device is live, and the check's timelines too while its round is held.
    let waited = unsafe { (checked.shared().vk()).wait_semaphores(&info, HELD.as_nanos() as u64) };
    let took = waiting.elapsed();
    hold.release();
    println!(
        "QUEUE-SWITCH-GATE {}: the {} batches the round submitted before its hold ({at:?}), \
         waited for while it was held: {waited:?} in {took:?}",
        checked.name(),
        values.len()
    );
    assert_eq!(
        waited,
        Ok(()),
        "the batches a queue-switch round submitted did not complete in {HELD:?} while the round \
         was held ({at:?})"
    );
    // Its opening and first batch at least: a round on one `VkQueue` holds there.
    assert!(
        values.len() >= 2,
        "the round submitted {} batches before its hold ({at:?})",
        values.len()
    );
    drop(queue);
}

#[test]
fn a_queue_switch_round_s_batches_complete_while_it_is_held_between_them() {
    held_round_s_batches(At::Between);
}

#[test]
fn a_queue_switch_round_s_batches_complete_while_it_is_held_once_submitted() {
    held_round_s_batches(At::Submitted);
}
