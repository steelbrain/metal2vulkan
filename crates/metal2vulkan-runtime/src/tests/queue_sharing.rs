//! A device's `VkQueue`s are shared round robin among its command queues, and the sharing is
//! counted: the first command queues each take a `VkQueue` of their own, and the one past them
//! takes one an earlier command queue took (`WorkCounts::queues_sharing_a_vk_queue`).
//!
//! # Why this is a count
//!
//! Two command queues on one `VkQueue` submit to one Vulkan queue, which runs its submissions in
//! order, so one's command buffers wait for the other's whole command buffer. Metal's queues wait
//! for each other only inside one command: on this M3 Max a side queue's command buffer waited the
//! whole 2.50 s of another queue's single compute dispatch, but nothing at all behind the same
//! 2.50 s split into 256 dispatches of one command buffer (`queue.rs`). The runtime answers that
//! with queues, lending the round robin every queue the device's families have past the two
//! reserved for presenting, up to sixteen (`device/create.rs`); what it cannot answer is a
//! program with more command queues than the device has queues, and on this host MoltenVK's four
//! families of one queue each are four. The count is what makes that visible to a caller rather
//! than left to be inferred from a timing.
//!
//! The assertion is a count, never a time: it is a fact of the device's queue plan, so it holds on
//! any host under any load. The run asserts both sides of the step -- zero sharing up to the
//! device's queues, then one more share per command queue past them -- so a device that shared
//! from its first command queue fails it as surely as one that never counted a share.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{plan_device, Scratch};
use crate::CommandQueue;

/// The most command queues the run makes: one past the round robin's ceiling
/// (`device/create.rs`), so a device of any shape must have begun sharing by the last.
const MOST: usize = 17;

/// Command queues take a `VkQueue` of their own until the device has no more, and each one past
/// them is counted as sharing one.
#[test]
fn command_queues_past_the_devices_queues_are_counted_as_sharing_one() {
    let scratch = Scratch::new("queue-sharing");
    let device = plan_device(&scratch.0);
    let sharing = || device.work_counts().queues_sharing_a_vk_queue;
    assert_eq!(sharing(), 0, "a device with no command queue shares none");
    let mut kept: Vec<CommandQueue> = Vec::with_capacity(MOST);
    // Up to the device's own queues, each command queue takes one of its own.
    let mut own = 0;
    while kept.len() < MOST {
        kept.push(device.new_command_queue().unwrap());
        if sharing() > 0 {
            break;
        }
        own = kept.len();
    }
    assert!(
        own > 0,
        "the first command queue of a device must take a `VkQueue` of its own, not share one: \
         the device counted {} sharing with one command queue made",
        sharing()
    );
    assert_eq!(
        sharing(),
        1,
        "the first command queue past the device's {own} queues must be counted as sharing one, \
         once"
    );
    assert_eq!(
        kept.len(),
        own + 1,
        "sharing must begin at the command queue past the device's queues, and a device of any \
         shape must share by the {MOST}th: it began at {}",
        kept.len()
    );
    // Each further command queue shares too: the round robin never runs out.
    for step in 1..=3 {
        kept.push(device.new_command_queue().unwrap());
        assert_eq!(
            sharing(),
            1 + step,
            "every command queue past the device's {own} queues shares one, so the {}th sharing \
             queue is counted",
            1 + step
        );
    }
}
