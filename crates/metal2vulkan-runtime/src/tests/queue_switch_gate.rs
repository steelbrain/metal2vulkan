//! Driver calls beside a device's queue-switch check (`queue_switch_check.rs`) held between the
//! two batches' submissions of a round on one `VkQueue`, its opening and first batch submitted
//! (`QueueSwitch::between`).
//!
//! The device's first command queue starts its check; each test holds the round there and, on a
//! thread of its own, makes driver calls: another device's making and teardown, memory allocated
//! and freed on the device, and dma-buf imports on the device. Each returns while the round is
//! held: no work of the device's waits for a signal only the check's thread can still make, so a
//! driver call that waits for the GPU's work in flight waits for work that completes. Seen
//! otherwise, when the check's first batch waited on the GPU for a gate the host signalled after
//! its second batch's submission: on an RTX 5080 Laptop GPU (NVIDIA 615.71.09) beside the runtime
//! suite, another device's `vkDestroyDevice` and the check's `vkQueueSubmit` never returned
//! (`tests::recorder_slots`, 20 minutes), and a dma-buf import's `vkAllocateMemory` and the
//! check's `vkQueueSubmit` never returned (`tests::dma_buf_leak_probe`).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that runs no check (one family of one `VkQueue`, no
//! calibrated timestamps) skips, saying so.

use super::blit_holds::{Scratch, WAIT};
use super::draw_again::{device, Heard};
use crate::queue_share::Verdict;
use crate::{CommandQueue, Device};
use ash::vk;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long the round is held while the calls run.
const HELD: Duration = Duration::from_secs(20);

/// How long the calls are waited for once the round went on, where they had not returned.
const AFTER: Duration = Duration::from_secs(5);

/// What [`beside_held_round`] saw: how long the calls took while the round was held, or that
/// they had not returned in [`HELD`]; then, where they had not, whether they returned in
/// [`AFTER`] once it went on; and the command queue that started the check, kept until the test
/// asserted.
struct Beside {
    held: Result<Duration, RecvTimeoutError>,
    after: Option<Result<Duration, RecvTimeoutError>>,
    queue: CommandQueue,
}

/// Start `checked`'s queue-switch check, hold it as its first round on one `VkQueue` reaches its
/// second batch, run `calls` on a thread of its own meanwhile, and let the round go on: `None`
/// where the device runs no check.
fn beside_held_round(checked: &Device, calls: impl FnOnce() + Send + 'static) -> Option<Beside> {
    let switch = Arc::clone(&checked.shared().queue_switch);
    if switch.verdict() != Verdict::Pending {
        println!(
            "SKIPPED {}: the device runs no queue-switch check",
            checked.name()
        );
        return None;
    }
    switch.between.hold();
    // The device's first command queue starts its check, which stops between the two batches'
    // submissions of its first round on one `VkQueue`.
    let queue = checked.new_command_queue().unwrap();
    switch.between.until_waiting(1, WAIT);
    let (went, gone) = mpsc::channel();
    std::thread::spawn(move || {
        let calling = Instant::now();
        calls();
        let _ = went.send(calling.elapsed());
    });
    let held = gone.recv_timeout(HELD);
    switch.between.release();
    let after = held.is_err().then(|| gone.recv_timeout(AFTER));
    Some(Beside { held, after, queue })
}

/// Assert that the calls `what` returned while the round was held, printing what was seen.
fn returned(checked: &Device, what: &str, beside: Beside) {
    let Beside { held, after, queue } = beside;
    println!(
        "QUEUE-SWITCH-GATE {}: {what}, while the round was held: {held:?}; once it went on: \
         {after:?}",
        checked.name()
    );
    assert!(
        held.is_ok(),
        "{what} did not return in {HELD:?} while the queue-switch round was held between its \
         batches ({after:?} once it went on)"
    );
    drop(queue);
}

#[test]
fn a_device_dropped_while_another_s_queue_switch_round_is_held_goes() {
    let scratch = Scratch::new("queue-switch-gate");
    let checked = device(&scratch.0, &[], &Heard::default());
    let cache = scratch.0.clone();
    let Some(beside) = beside_held_round(&checked, move || {
        // A device of a message sink of its own, so a device of its own (`create.rs`).
        drop(device(&cache, &[], &Heard::default()));
    }) else {
        return;
    };
    returned(&checked, "another device's making and teardown", beside);
}

#[test]
fn memory_allocated_and_freed_beside_its_device_s_held_queue_switch_round_returns() {
    let scratch = Scratch::new("queue-switch-gate-memory");
    let checked = device(&scratch.0, &[], &Heard::default());
    let calling = checked.clone();
    let Some(beside) = beside_held_round(&checked, move || allocate_and_free(&calling)) else {
        return;
    };
    returned(
        &checked,
        "64 MiB of each memory type allocated and freed",
        beside,
    );
}

/// Allocate 64 MiB of each of `device`'s memory types that takes it, and free each at once.
fn allocate_and_free(device: &Device) {
    // SAFETY: the device is live; what is made is freed at once and used by nothing.
    let handles = unsafe { device.raw() };
    // SAFETY: the physical device is the instance's.
    let memory = unsafe {
        handles
            .instance
            .get_physical_device_memory_properties(handles.physical_device)
    };
    for type_index in 0..memory.memory_type_count {
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(64 << 20)
            .memory_type_index(type_index);
        // SAFETY: as above.
        if let Ok(made) = unsafe { handles.device.allocate_memory(&info, None) } {
            // SAFETY: as above.
            unsafe { handles.device.free_memory(made, None) };
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn dma_buf_imports_beside_their_device_s_held_queue_switch_round_return() {
    use super::export_after_dma_buf::{dma_buf_types, raw_import, scattered_udmabuf, PAGES};
    const NAME: &str = "QUEUE-SWITCH-GATE dma-buf";
    if !super::host_residency::udmabuf_openable(NAME) {
        return;
    }
    let scratch = Scratch::new("queue-switch-gate-dma-buf");
    let checked = super::host_residency::device(&scratch.0);
    let page = (checked.shared().host_import.as_ref()).map_or(4096, |import| import.page);
    let calling = checked.clone();
    let Some(beside) = beside_held_round(&checked, move || {
        // A udmabuf of scattered pages, imported dedicated to nothing as each type the driver
        // lists for it (615.71.09 refuses most such imports), each import freed at once.
        let udmabuf = scattered_udmabuf(page);
        let types = dma_buf_types(&calling, &udmabuf).unwrap_or(0);
        let answers: Vec<(u32, i32)> = (0..32)
            .filter(|index| types & (1 << index) != 0)
            .map(|index| (index, raw_import(&calling, &udmabuf, PAGES * page, index)))
            .collect();
        println!("{NAME}: each listed type's import and its answer: {answers:?}");
    }) else {
        return;
    };
    returned(&checked, "dma-buf imports of each listed type", beside);
}
