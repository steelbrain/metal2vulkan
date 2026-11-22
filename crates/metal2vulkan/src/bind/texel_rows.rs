//! A 2D texture over a buffer's rows whose image is a shadow, read and written in place through a
//! texel buffer view of the rows' bytes.
//!
//! **Where.** A texture over a buffer's bytes lies in an image over them where the device makes
//! one; where it makes none (the runtime's image refusals), the texture is a shadow, an image of
//! the runtime's own memory its commands' encoders copy the texels into and back. The buffer
//! itself is still in place. A texture in such a shadow states its rows in its facts
//! (`TextureFacts::texel_rows`), and a texture slot whose function only reads, writes or asks the
//! size of it (`RuntimeFact::TexelRows`: no sample, gather, atomic, read through a sampler or
//! level past the first) takes the slot's rows variant (`VariantKey::texel_rows`).
//!
//! **The binding.** Under the variant the slot's one descriptor is a texel buffer over the rows'
//! bytes (`WriteSource::TexelRows`), uniform where the function only reads the texture and storage
//! where it writes it, in the format a texture buffer of the texture's pixel format takes
//! (`bind::plan`'s texel formats: the format's own, an alias, or `R8_UNORM` for `A8Unorm`). Vulkan
//! states a texel buffer's texels "going through format conversion when read in a shader in the
//! same way as they are for an image" (Uniform Texel Buffer, Storage Texel Buffer), so a read
//! answers the device's conversion of the texel, as the shadow's image does. The view starts at
//! the texture's first texel, or before it where the device's texel buffer offset alignment asks:
//! the texture's size, its rows in texels and its first texel in the view are the slot's push
//! field (`PushField::TexelRows`, [`TexelRows::words`]).
//!
//! **What a function reads.** Texel `(x, y)` is the view's texel `first + y · row + x`. A read of
//! a texel outside the texture, or at a level past its one, is Metal's out-of-bounds texel, and a
//! write there is dropped, by the function's own test of the coordinate against the texture's
//! size: a coordinate past the width names a texel of the next row in the view, which the
//! device's robustness, bounding the view alone, would read. A size query answers the field's
//! size, at a level as at any texture, and the level count 1.
//!
//! **Ordering.** The slot's use is a use of the rows' bytes (`AccessClass::TexelRows`), as a
//! texture buffer's is: the consumer orders it against every use of the buffer, its shadow's
//! copies among them. A store to a texel buffer writes its texel's bytes alone (Vulkan's memory
//! model: "A memory operation must not affect memory at a memory location not within its set of
//! memory locations"), and a buffer view has no layout moves or load and store operations, so the
//! bytes between the rows keep whatever the CPU or another command writes there.

/// A 2D texture's rows in a texel buffer view of its buffer's bytes, where its image is a shadow
/// (the module docs): the words of its slot's push field (`PushField::TexelRows`). A texture states
/// one in its facts
/// (`TextureFacts::texel_rows`), and a slot whose function only reads, writes or asks the size of
/// it (`RuntimeFact::TexelRows`) takes the rows variant (`VariantKey::texel_rows`), which binds
/// the view (`WriteSource::TexelRows`) and addresses texel `(x, y)` at `first + y · row + x`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TexelRows {
    words: [u32; TexelRows::WORDS],
}

impl TexelRows {
    /// The words of the slot's push field ([`words`](Self::words)), one `uvec4`.
    pub const WORDS: usize = 4;
    /// Word: the texture's width in texels.
    pub const WIDTH: usize = 0;
    /// Word: the texture's height in texels.
    pub const HEIGHT: usize = 1;
    /// Word: the texels from one row's first to the next's in the view.
    pub const ROW: usize = 2;
    /// Word: the view's texel that is the texture's first.
    pub const FIRST: usize = 3;

    /// The words of the slot's push field.
    pub fn words(&self) -> [u32; Self::WORDS] {
        self.words
    }
}
