//! The regions of a copy from a texture to bytes: every `vkCmdCopyImageToBuffer` the runtime
//! records takes them from [`regions`], the blit's, a synchronize's, a texture's `get_bytes` and a
//! texture over a buffer's copy back alike.
//!
//! A copy whose row stride in bytes is not a multiple of M, the largest power of two up to 16 that
//! divides the bytes of its region's row, is recorded as two regions: every block column but the
//! last, then the last. Each part keeps the copy's `bufferRowLength` and `bufferImageHeight`, and
//! the last part's offset moves by the columns before it, so by Vulkan's addressing ("Buffer and
//! Image Addressing": a texel's bytes are `[texelOffset, texelOffset + blockSize)`, its row
//! `max(bufferRowLength, imageExtent.width)` blocks from the last) each part writes exactly the
//! bytes the copy wrote there and none of the other's (`VUID-vkCmdCopyImageToBuffer-pRegions-12485`).
//! The first part is an odd number of blocks wide and the last one block, so each part's M is the
//! block's own bytes, which every stride is a multiple of.
//!
//! The UHD 630's Windows driver (101.2145) writes a single region of that class with its rows
//! floor(stride / M) × M bytes apart where its offset is a multiple of M: rows of 24 bytes 16
//! apart for a region two 8-byte blocks wide. The form is decided by the copy's arguments alone,
//! the same on every device; a copy whose stride is a multiple of M, every copy of tight rows
//! among them, is one region.

use crate::texture::CopyUnit;
use ash::vk;

/// The regions of one copy from a texture to bytes: one, or the two parts of a split.
pub(crate) struct Regions {
    parts: [vk::BufferImageCopy; 2],
    count: usize,
}

impl Regions {
    /// The regions, in order.
    pub(crate) fn as_slice(&self) -> &[vk::BufferImageCopy] {
        &self.parts[..self.count]
    }
}

/// For the device tests: the copies [`regions`] split, in this process.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) static SPLIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The largest power of two up to 16 that divides `bytes`; 1 for 0.
pub(crate) fn row_power(bytes: u64) -> u64 {
    let mut power = 1;
    while power < 16 && bytes != 0 && bytes.is_multiple_of(power * 2) {
        power *= 2;
    }
    power
}

/// The regions of a copy of `region` from a texture whose texel blocks are `unit`'s to bytes, as
/// the module says. Only a colour aspect splits: a depth or stencil copy's offsets are multiples
/// of 4 (`VUID-vkCmdCopyImageToBuffer-srcImage-07978`), which the last part's need not be.
pub(crate) fn regions(unit: &CopyUnit, region: vk::BufferImageCopy) -> Regions {
    let one = Regions {
        parts: [region; 2],
        count: 1,
    };
    let [block_width, _] = unit.block;
    let bytes = u64::from(unit.bytes);
    let width = region.image_extent.width;
    let blocks = width.div_ceil(block_width);
    let row = u64::from(blocks) * bytes;
    let length = match region.buffer_row_length {
        0 => width,
        length => length,
    };
    let stride = u64::from(length.div_ceil(block_width)) * bytes;
    if unit.aspect != vk::ImageAspectFlags::COLOR
        || blocks < 2
        || stride.is_multiple_of(row_power(row))
    {
        return one;
    }
    let before = (blocks - 1) * block_width;
    let mut first = region;
    first.buffer_row_length = length;
    first.image_extent.width = before;
    let mut last = first;
    last.buffer_offset += u64::from(blocks - 1) * bytes;
    last.image_offset.x += before as i32;
    last.image_extent.width = width - before;
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    SPLIT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    Regions {
        parts: [first, last],
        count: 2,
    }
}
