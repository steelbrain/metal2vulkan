//! How a batch of command buffers on one `VkQueue` begins after the batches before it there
//! (`DevicePolicy::batch_order`).
//!
//! Metal orders a command buffer after the earlier ones of its queue only where its work depends
//! on theirs: an independent fill committed behind a running fill of other bytes started while
//! the first ran, on an M3 Max under macOS 27.0.1 (probe `gpustart`). The exact form copies that:
//! a batch begins with a barrier for each access that depends on an earlier batch's of its
//! `VkQueue` (`tracker.rs`), its first encoder's GPU start written at the top of the pipe, and
//! ends with [`Barrier::to_host`], which orders only the host's reads and writes after it.
//!
//! Some drivers run no two kernel submissions of one queue side by side anyway. Measured with a
//! probe outside the runtime: batches A and B, each a command buffer of a start timestamp, one of
//! work and one of an end barrier and an end timestamp, signalling the queue's timeline, B's work
//! independent of A's; work of one workgroup of 64 invocations running a dependent multiply-add
//! chain of about 3 to 10 ms (so two run side by side in about one's time), or fills of other
//! buffers; with no end barrier, `to_host`'s or [`Barrier::to_all`]'s; in two `vkQueueSubmit`
//! calls or one; with and without the timestamps, the queries reset on the host or the GPU, and
//! with only the last submission signalling. On each of these, every shape ran B after A, in
//! about the sum of their times (20.0 ms for two 10 ms batches on the RTX 5080 Laptop GPU, 18.5
//! on the RX 7900 XTX, 20.0 on the UHD 770), where the same two dispatches recorded in one command
//! buffer ran side by side (10.0, 9.3 and 10.0 ms):
//! - ANV (Mesa 26.2.4) on the UHD 770 of RPL-S;
//! - RADV (Mesa 26.0.8) on the Radeon 780M and the RX 7900 XTX;
//! - NVIDIA's driver (615.71.09) on the RTX 5080 Laptop GPU.
//!
//! On those drivers ([`ends_order`]) the default form orders batches by their ends: every batch
//! ends with [`Barrier::to_all`], every write before every access of every later command of its
//! `VkQueue`, so the tracker plans no barrier after an access of an earlier batch there
//! ([`Tracker::by_ends`](crate::tracker::Tracker)). A batch's start timestamp, written at the top
//! of the pipe as on every driver (`gpu_time.rs`), is then written once everything before it in
//! submission order completed, which is when such a driver starts it: the earlier batch's
//! [`Barrier::to_all`] holds it, as "the second synchronization scope includes all commands that
//! occur later in submission order" (`vkCmdPipelineBarrier`), its destination stages all commands.
//! A start written at the bottom of the pipe there read late on RADV (Mesa 26.2.6) on an RX 7900
//! XTX and a Radeon 780M: in a traced compute benchmark's pass, a population of command buffers
//! read their start in the last tenth of their own kernel job, spans of 1 us on jobs of 4.5 to
//! 36.5 ms. A batch no longer needs a command buffer of barriers
//! for a dependency on the batch before it (on the UHD 770, with the consumer's frames over 520
//! host imports, 67.0 command buffers submitted a frame against 129.3 to 129.7), and runs as the
//! driver ran it before. Elsewhere, and under `BatchOrderPolicy::Exact`, the exact form.

use crate::tracker::Barrier;
use metal2vulkan::device::{BatchOrderPolicy, DevicePolicy};
use metal2vulkan::vk::DriverId;

/// Whether a device of `policy` and `driver` orders its batches by their ends (the module docs):
/// its policy asks for it and its driver is measured to run no two kernel submissions of one
/// queue side by side.
pub(crate) fn by_ends(policy: &DevicePolicy, driver: Option<DriverId>) -> bool {
    policy.batch_order == BatchOrderPolicy::Ends && ends_order(driver)
}

/// Whether the driver is measured to run the kernel submissions of one queue one after another,
/// with or without a barrier between them (the module docs).
fn ends_order(driver: Option<DriverId>) -> bool {
    matches!(
        driver,
        Some(DriverId::INTEL_OPEN_SOURCE_MESA | DriverId::MESA_RADV | DriverId::NVIDIA_PROPRIETARY)
    )
}

/// The barrier that ends each batch of a device that orders its batches `by_ends` or not.
pub(crate) fn end(by_ends: bool) -> Barrier {
    match by_ends {
        true => Barrier::to_all(),
        false => Barrier::to_host(),
    }
}
