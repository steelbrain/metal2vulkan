//! A fill of a buffer's whole 32-bit words with one word, `vkCmdFillBuffer`'s, as every fill the
//! runtime records goes.
//!
//! - **Where the driver compiles its fill.** MoltenVK runs `vkCmdFillBuffer` as a compute
//!   dispatch of a pipeline it makes at the first fill each `VkCommandPool`'s command buffers
//!   encode (`MVKCommandEncodingPool::getCmdFillBufferMTLComputePipelineState`), inside
//!   `vkQueueSubmit`, through Metal's compiler service, which runs the process's compiles in
//!   order. Behind a storm of pipeline creations, a commit holding a pool's first fill waited
//!   250 to 330 ms in that submit on an M4. On a release measured so
//!   (`DriverFact::FillCompiledPerCommandPool`, MoltenVK 1.4.1), a fill goes as transfers, which
//!   MoltenVK runs as blits and compiles nothing for:
//!   `vkCmdUpdateBuffer` of the first words (at most 65536 bytes, its limit), then, past each
//!   barrier, one `vkCmdCopyBuffer` of up to [`FAN_OUT`] regions, each a copy of all that is
//!   filled so far onto the next words, so each barrier multiplies what is filled by
//!   `FAN_OUT + 1`.
//! - **Few barriers.** MoltenVK ends its blit encoder at each barrier and makes a new one for the
//!   next copy, and encodes it inside `vkQueueSubmit`: an encoder costs about 10 us of the
//!   submitting thread on an M3 Max, where one region more in an encoder costs a blit. A fill of
//!   256 MiB took 12 barriers when each copy doubled what was filled, and 192 such fills kept their
//!   command buffer's `vkQueueSubmit` 10 to 27 ms (an M3 Max, MoltenVK 1.4.2, the conformance case
//!   `event-wait-behind-a-running-signal`, where Metal schedules the same fills 0.04 to 0.18 ms
//!   after the commit); with sixteenfold steps it takes 3.
//! - **Exact.** Each copy reads only words the fill wrote before the barrier ahead of it, and
//!   writes words no earlier step of the fill wrote, nor another copy of its step, so one
//!   command's sources and destinations never overlap (as `vkCmdCopyBuffer` requires), and every
//!   word ends as the fill's.
//! - **Elsewhere** `vkCmdFillBuffer`, unchanged, on every other driver and release.
//! - **Counted.** Each fill counts as one of the two forms (`WorkCounts::fills_recorded` and
//!   `fills_by_transfers`), so a run shows which its device took; a new buffer's first zeros
//!   filled on a device without unified memory count apart (`WorkCounts::new_buffer_zero_fills`,
//!   `texture/first_move.rs`), which takes them only where the device's fills are
//!   `vkCmdFillBuffer`'s ([`by_transfers`]).

use crate::device::functions::Vk;
use crate::Device;
use ash::vk;
use metal2vulkan::device::DriverFact;
use std::ops::Range;

/// The most bytes one `vkCmdUpdateBuffer` writes.
const UPDATE_LIMIT: u64 = 65536;

/// The most regions one step's `vkCmdCopyBuffer` copies what is filled onto (the module docs):
/// each step multiplies what is filled by one more than this.
const FAN_OUT: u64 = 15;

/// One step of a fill by transfers ([`steps`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// `vkCmdUpdateBuffer` of these bytes with the word.
    Update(Range<u64>),
    /// The transfers before it have written before the ones after it read.
    Barrier,
    /// One `vkCmdCopyBuffer` of these regions, within the buffer.
    Copies(Vec<Copy>),
}

/// One region of a [`Step::Copies`]: `len` bytes at `from` to `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Copy {
    pub(crate) from: u64,
    pub(crate) to: u64,
    pub(crate) len: u64,
}

/// The steps that fill `words` (its ends multiples of 4) by transfers: the first words updated,
/// then, past each barrier, up to [`FAN_OUT`] copies of what is filled, from the filled words at
/// its start, onto the words after it.
pub(crate) fn steps(words: Range<u64>) -> Vec<Step> {
    let total = words.end.saturating_sub(words.start);
    if total == 0 {
        return Vec::new();
    }
    let first = total.min(UPDATE_LIMIT);
    let mut steps = vec![Step::Update(words.start..words.start + first)];
    let mut filled = first;
    while filled < total {
        let mut copies = Vec::new();
        let mut to = filled;
        while to < total && (copies.len() as u64) < FAN_OUT {
            let len = filled.min(total - to);
            copies.push(Copy {
                from: words.start,
                to: words.start + to,
                len,
            });
            to += len;
        }
        steps.push(Step::Barrier);
        steps.push(Step::Copies(copies));
        filled = to;
    }
    steps
}

/// Record a fill of `words` of `buffer` with `word` on `owner`'s command buffer
/// `command_buffer`: `vkCmdFillBuffer`, or the [`steps`] by transfers where the device's driver
/// has [`DriverFact::FillCompiledPerCommandPool`]. The caller's barriers order it as one transfer
/// that writes `words` (it also reads them, in the transfer stage).
///
/// # Safety
///
/// `command_buffer` is recording, outside a render pass; `buffer` is live until it completes,
/// made with `TRANSFER_DST` usage (and `TRANSFER_SRC` where `words` holds more than 65536 bytes),
/// and `words` lies in it, its ends multiples of 4.
pub(crate) unsafe fn record(
    owner: &Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    words: Range<u64>,
    word: u32,
) {
    let device = owner.shared().vk();
    let by_transfers = by_transfers(owner);
    if words.is_empty() {
        return;
    }
    owner.shared().counts.fill_recorded(by_transfers);
    if !by_transfers {
        // SAFETY: the caller's contract.
        unsafe { record_command(device, command_buffer, buffer, words, word) };
        return;
    }
    let first = (words.end - words.start).min(UPDATE_LIMIT) / 4;
    let bytes: Vec<u8> = (0..first).flat_map(|_| word.to_ne_bytes()).collect();
    for step in steps(words) {
        // SAFETY: the caller's contract; each update writes at most 65536 bytes of `bytes`, a
        // multiple of 4 at a multiple of 4, and each copy's regions lie in `words`, its sources'
        // union apart from its destinations' (the module docs).
        unsafe {
            match step {
                Step::Update(range) => device.cmd_update_buffer(
                    command_buffer,
                    buffer,
                    range.start,
                    &bytes[..(range.end - range.start) as usize],
                ),
                Step::Barrier => {
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
                Step::Copies(copies) => {
                    let regions: Vec<vk::BufferCopy> = (copies.iter())
                        .map(|copy| {
                            vk::BufferCopy::default()
                                .src_offset(copy.from)
                                .dst_offset(copy.to)
                                .size(copy.len)
                        })
                        .collect();
                    device.cmd_copy_buffer(command_buffer, buffer, buffer, &regions);
                }
            }
        }
    }
}

/// Whether `owner`'s fills go as transfers ([`DriverFact::FillCompiledPerCommandPool`]).
pub(crate) fn by_transfers(owner: &Device) -> bool {
    (owner.properties().driver_facts).contains(&DriverFact::FillCompiledPerCommandPool)
}

/// Record a fill of `words` of `buffer` with `word` as one `vkCmdFillBuffer` with `device`'s
/// functions, uncounted: [`record`]'s form on a device whose fills do not go as transfers
/// ([`by_transfers`]).
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
