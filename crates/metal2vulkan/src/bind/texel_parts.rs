//! A texture buffer wider than one texel buffer view holds, read and written through several
//! views of its bytes, its parts.
//!
//! **Where.** Metal makes a texture buffer of up to 2^28 texels on every Mac. One texel buffer
//! view holds at most the device's `maxTexelBufferElements` texels
//! (`VUID-VkBufferViewCreateInfo-range-00930`): 2^27 on ANV and lavapipe, 2^26 on MoltenVK. The
//! consumer states with a texture buffer's binding how many parts its texels take
//! (its texels over [`part_texels`], rounded up): one for every texture buffer within the limit,
//! which binds one view.
//!
//! **The views.** Part `k` views the texels from `k · P`, `P` the device's [`part_texels`] (its
//! `maxTexelBufferElements` rounded down to a multiple of its `minTexelBufferOffsetAlignment`, so
//! each part starts on that alignment where the texture buffer does): `P` texels for every part
//! but the last, which holds the rest. Each is a view of the device's own in the format one view
//! would take, so a read converts its texel as that view does, the device's conversion
//! (Uniform Texel Buffer, Storage Texel Buffer).
//!
//! **What a function reads.** A slot whose function reads, writes, updates by an atomic or asks
//! the width of the texture buffer bound there, and does nothing else with it
//! (`RuntimeFact::TexelBufferParts`), on a device that indexes arrays of its kind of texel buffer
//! by a `NonUniform` value, takes the parts variant where the bound texture buffer takes more than
//! one part (`VariantKey::texel_buffer_parts`). Under it the slot is declared as an array of its
//! parts, and texel `i` is texel `i − k · P` of part `k = min(⌊i / P⌋, parts − 1)`. A texel past
//! the last lies past the last part's end, so its read is Metal's out-of-bounds texel and its
//! write is dropped, by the device's robustness or the function's own check as through one view.
//! The width is `(parts − 1) · P` plus the last part's. Any other slot binds one view, which
//! `maxTexelBufferElements` bounds.

use crate::device::Limits;

/// The texels of each part but the last of a texture buffer of more texels than one texel buffer
/// view holds, which binds through views of its parts (part `k` from texel `k · P`, the last
/// holding the rest): the device's `maxTexelBufferElements` rounded down to a multiple of its
/// `minTexelBufferOffsetAlignment`, so that a part starts on that alignment wherever the texture
/// buffer does (a texel is at least a byte). 0 on a device whose limit is below its alignment,
/// which binds no parts.
pub fn part_texels(limits: &Limits) -> u64 {
    let max = u64::from(limits.max_texel_buffer_elements);
    let alignment = limits.min_texel_buffer_offset_alignment.max(1);
    max - max % alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max: u32, alignment: u64) -> Limits {
        Limits {
            max_texel_buffer_elements: max,
            min_texel_buffer_offset_alignment: alignment,
            ..Limits::default()
        }
    }

    /// A part holds the limit's texels: 2^27 on lavapipe's limits.
    #[test]
    fn parts_hold_the_limit_s_texels_each() {
        assert_eq!(part_texels(&limits(1 << 27, 16)), 1 << 27);
    }

    /// A limit off the offset alignment: each part holds the limit rounded down to it, so the next
    /// starts on it; a limit below the alignment holds no aligned part.
    #[test]
    fn parts_start_on_the_offset_alignment() {
        assert_eq!(part_texels(&limits(1000, 64)), 960);
        assert_eq!(part_texels(&limits(32, 64)), 0);
    }
}
