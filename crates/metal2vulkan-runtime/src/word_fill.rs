//! A fill of a buffer's whole 32-bit words with one word: one `vkCmdFillBuffer`, as every fill the
//! runtime records goes, on every driver.
//!
//! - **On MoltenVK** a fill is a compute dispatch of a pipeline MoltenVK makes at the first fill
//!   each `VkCommandPool`'s command buffers encode, inside `vkQueueSubmit`
//!   (`MVKCommandEncodingPool::getCmdFillBufferMTLComputePipelineState`). At 1.4.2 that first fill
//!   took 0.022 to 0.038 ms (median) in `vkQueueSubmit` beside one to eight threads making compute
//!   pipelines (91 and 496 made; our probe), as a `vkCmdUpdateBuffer` did. A fill recorded as
//!   transfers instead (an update of the first words, then copies of what is filled onto the rest
//!   past barriers) costs the submitting thread an encoder at each barrier: 192 fills of 256 MiB
//!   so kept the conformance case `event-wait-behind-a-running-signal`'s command buffer 4.3 ms
//!   (median) from its commit to `Scheduled` on an M3 Max, where `vkCmdFillBuffer` takes 0.6 ms,
//!   and Metal schedules the same fills 0.04 to 0.18 ms after the commit.
//! - **Counted.** Each fill counts once (`WorkCounts::fills_recorded`); a new buffer's first zeros
//!   filled on a device without unified memory count apart (`WorkCounts::new_buffer_zero_fills`,
//!   `texture/first_move.rs`).

use crate::device::functions::Vk;
use crate::Device;
use ash::vk;
use std::ops::Range;

/// Record a fill of `words` of `buffer` with `word` on `owner`'s command buffer `command_buffer`,
/// counted: one `vkCmdFillBuffer` ([`record_command`]). The caller's barriers order it as one
/// transfer that writes `words`.
///
/// # Safety
///
/// `command_buffer` is recording, outside a render pass; `buffer` is live until it completes,
/// made with `TRANSFER_DST` usage, and `words` lies in it, its ends multiples of 4.
pub(crate) unsafe fn record(
    owner: &Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    words: Range<u64>,
    word: u32,
) {
    if words.is_empty() {
        return;
    }
    owner.shared().counts.fill_recorded();
    // SAFETY: the caller's contract.
    unsafe { record_command(owner.shared().vk(), command_buffer, buffer, words, word) };
}

/// Record a fill of `words` of `buffer` with `word` as one `vkCmdFillBuffer` with `device`'s
/// functions, uncounted: [`record`]'s command.
///
/// # Safety
///
/// `command_buffer` is recording, outside a render pass; `buffer` is live until it completes,
/// made with `TRANSFER_DST` usage, and `words` lies in it, its ends multiples of 4.
pub(crate) unsafe fn record_command(
    device: &Vk,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    words: Range<u64>,
    word: u32,
) {
    // SAFETY: the caller's contract.
    unsafe {
        device.cmd_fill_buffer(
            command_buffer,
            buffer,
            words.start,
            words.end - words.start,
            word,
        );
    }
}
