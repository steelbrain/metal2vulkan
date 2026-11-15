//! A texture buffer wider than one texel buffer view holds, read and written through several
//! views of its bytes, its parts.
//!
//! **Where.** Metal makes a texture buffer of up to 2^28 texels on every Mac. One texel buffer
//! view holds at most the device's `maxTexelBufferElements` texels
//! (`VUID-VkBufferViewCreateInfo-range-00930`): 2^27 on ANV and lavapipe, 2^26 on MoltenVK. The
//! consumer states with a texture buffer's binding how many parts its texels take
//! ([`texel_buffer_parts`], `BoundTexture::Buffer::parts`), which its shape carries
//! (`TextureShape::Buffer::parts`): one for every texture buffer within the limit, which binds one
//! view as it always has.
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
//! by a `NonUniform` value, takes the parts variant where the shape binds more than one part
//! there (`VariantKey::texel_buffer_parts`). Under it the slot is declared as an array of its
//! parts, and texel `i` is texel `i − k · P` of part `k = min(⌊i / P⌋, parts − 1)`. A texel past
//! the last lies past the last part's end, so its read is Metal's out-of-bounds texel and its
//! write is dropped, by the device's robustness or the function's own check as through one view.
//! The width is `(parts − 1) · P` plus the last part's. Any other slot binds one view, which
//! `maxTexelBufferElements` bounds (`BindError::TexelBufferTooLarge`).

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

/// The texel buffer views the texels of a texture buffer of `texels` texels
/// (`⌊length / bytes per texel⌋`) take on a device of `limits`, which a consumer states with its
/// binding (`BoundTexture::Buffer::parts`): 1 within the device's `maxTexelBufferElements`, which
/// one view holds, else `⌈texels / P⌉` parts of [`part_texels`] `P` each (saturated at
/// `u32::MAX`). A slot whose function only reads, writes, updates by an atomic or asks the width
/// of such a texture buffer binds an array of views of its parts (`VariantKey::texel_buffer_parts`).
pub fn texel_buffer_parts(limits: &Limits, texels: u64) -> u32 {
    let part = part_texels(limits);
    if texels <= u64::from(limits.max_texel_buffer_elements) || part == 0 {
        return 1;
    }
    u32::try_from(texels.div_ceil(part)).unwrap_or(u32::MAX)
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

    /// A texture buffer within the limit takes one part, at the limit too; past it, parts of the
    /// limit's texels, the last holding the rest: 2^28 texels take two parts of 2^27 on lavapipe's
    /// limits and four of 2^26 on MoltenVK's.
    #[test]
    fn parts_hold_the_limit_s_texels_each() {
        let lavapipe = limits(1 << 27, 16);
        assert_eq!(part_texels(&lavapipe), 1 << 27);
        assert_eq!(texel_buffer_parts(&lavapipe, 1), 1);
        assert_eq!(texel_buffer_parts(&lavapipe, 1 << 27), 1);
        assert_eq!(texel_buffer_parts(&lavapipe, (1 << 27) + 1), 2);
        assert_eq!(texel_buffer_parts(&lavapipe, 1 << 28), 2);
        assert_eq!(texel_buffer_parts(&limits(1 << 26, 256), 1 << 28), 4);
        assert_eq!(texel_buffer_parts(&limits(1 << 28, 16), 1 << 28), 1);
    }

    /// A limit off the offset alignment: each part holds the limit rounded down to it, so the next
    /// starts on it.
    #[test]
    fn parts_start_on_the_offset_alignment() {
        let odd = limits(1000, 64);
        assert_eq!(part_texels(&odd), 960);
        assert_eq!(texel_buffer_parts(&odd, 1000), 1);
        assert_eq!(texel_buffer_parts(&odd, 1001), 2);
        assert_eq!(texel_buffer_parts(&odd, 1920), 2);
        assert_eq!(texel_buffer_parts(&odd, 1921), 3);
        // A limit below the alignment holds no aligned part: one view, which the limit bounds.
        assert_eq!(part_texels(&limits(32, 64)), 0);
        assert_eq!(texel_buffer_parts(&limits(32, 64), 100), 1);
    }
}
