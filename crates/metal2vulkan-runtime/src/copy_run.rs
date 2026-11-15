//! A blit encoder's run of buffer copies recorded as one `vkCmdCopyBuffer`
//! (`BlitCommandEncoder::copy_from_buffer_to_buffer`).
//!
//! - **Which copies join.** A copy between the same two `VkBuffer`s as the run's, which the
//!   encoder's order needs no barrier after (`EncoderCore::copy_joining`): no copy of the run
//!   wrote what it reads or reads or writes what it writes, per resource and range, untracked
//!   resources' too (a blit's uses name every byte it touches, `tracker.rs`). The run's regions
//!   then touch no byte both as a source and as a destination
//!   (`VUID-vkCmdCopyBuffer-pRegions-00117`), and no two of them write one byte, so their order
//!   inside the command is no order Metal sees. A program that copies tiles side by side, one copy
//!   per row, makes one `vkCmdCopyBuffer` of all of them.
//! - **What ends a run.** Any other command of the encoder, a barrier, a copy between other
//!   buffers, and the encoder's end: the run records first, in the `VkCommandBuffer` its first copy
//!   would have recorded into, so every command keeps its place in encoded order.
//! - **Where a run breaks.** The encoder's order keeps at most eight ranges of a resource, then
//!   merges them into one span, from the lowest byte of any to the highest
//!   (`tracker::ENTRIES_PER_KEY`). A copy whose destination lies inside that span of the earlier
//!   writes, though it meets none of them, takes a barrier there and starts a new run; so does one
//!   whose source lies inside it where both are one buffer's. Copies whose destination offsets rise
//!   from one copy to the next (a picture's rows in order, tile beside tile) join whole: sort them
//!   by destination offset to merge them all.
//! - **Counted.** Each copy that joined a run as another region
//!   (`WorkCounts::buffer_copies_merged`).

use crate::device::functions::Vk;
use crate::work_counts::Counters;
use ash::vk;

/// A run of copies from one `VkBuffer` into another, recorded as one `vkCmdCopyBuffer` (the
/// module docs).
pub(crate) struct CopyRun {
    /// The `VkCommandBuffer` its first copy took, which records it.
    raw: vk::CommandBuffer,
    source: vk::Buffer,
    destination: vk::Buffer,
    regions: Vec<vk::BufferCopy>,
}

impl CopyRun {
    /// A run of `region` of `source` into `destination`, to record into `raw`.
    pub(crate) fn new(
        raw: vk::CommandBuffer,
        (source, destination, region): (vk::Buffer, vk::Buffer, vk::BufferCopy),
    ) -> CopyRun {
        CopyRun {
            raw,
            source,
            destination,
            regions: vec![region],
        }
    }

    /// Join `copy` to the run where it copies between its two buffers and records into `raw`, the
    /// run's own: whether it joined.
    pub(crate) fn join(
        &mut self,
        raw: vk::CommandBuffer,
        (source, destination, region): (vk::Buffer, vk::Buffer, vk::BufferCopy),
    ) -> bool {
        let joins = self.raw == raw && self.source == source && self.destination == destination;
        if joins {
            self.regions.push(region);
        }
        joins
    }

    /// Record the run: one `vkCmdCopyBuffer` of its regions, each past the first counted.
    ///
    /// # Safety
    ///
    /// The run's command buffer is recording, outside a render pass, and nothing was recorded
    /// into it since the run's first copy; both buffers live until it completes, and every
    /// region lies inside both.
    pub(crate) unsafe fn record(self, device: &Vk, counts: &Counters) {
        // SAFETY: the caller's contract; the regions touch no byte as both a source and a
        // destination (the module docs).
        unsafe {
            device.cmd_copy_buffer(self.raw, self.source, self.destination, &self.regions);
        }
        counts.buffer_copies_merged(self.regions.len() as u64 - 1);
    }
}
