//! A 2D texture over a buffer's rows whose image is a shadow, its rows read and written in place
//! (`metal2vulkan::bind::texel_rows`).
//!
//! **Which.** A texture over a buffer is a shadow where the device makes no image over its bytes
//! (`shadow.rs`). Where the device refused that image for its format, features or limits, the
//! rows' pitch, or the bytes' place, or where the dma-buf import of the texture's pages refused
//! it (`in_place::Refusal::{Image, Pitch, Placement, Source, DmaBuf}`), the bytes are still the
//! buffer's own, in place: a texture of one level, slice and sample, of an uncompressed format
//! with no swizzle, whose rows start on a texel and lie a whole number of texels apart, and that
//! runs past no byte of its buffer, states its rows in its facts (`TextureFacts::texel_rows`). A
//! slot whose function only reads, writes or asks the size of it then binds a texel buffer over
//! the rows' bytes instead of the shadow (`WriteSource::TexelRows`), and its commands copy
//! nothing into or out of the shadow; every other slot binds the shadow as before. A shadow the
//! device's policy asks for (`HostPageTexturePolicy::Shadow`), and one made before or after its
//! class's check, keeps its shadow everywhere. So does one over a buffer with a read twin
//! (`read_twin.rs`): a slot that only reads it binds the texture's read twin, over the pages in
//! place, where it has one; a rows binding would read the buffer's copy instead.
//!
//! **The view.** It starts at the texture's first byte rounded down to the device's
//! `minTexelBufferOffsetAlignment` in the buffer's `VkBuffer`, the texels before the texture's
//! first the rows' `first`, and ends at the last row's last texel, at most
//! `maxTexelBufferElements` texels. A texture whose first byte lies a part of a texel past that
//! alignment, or whose view would hold more texels, keeps its shadow everywhere.
//!
//! **Ordering.** A binding's use is a use of the texture's bytes in its buffer
//! ([`Texture::rows_use`]): its shadow's entry meets it (`ShadowCopies::meets`), so a command
//! that binds the rows after one that wrote the shadow copies the shadow back first, and one that
//! writes both through the shadow and through the rows merges them, as for any other use of the
//! bytes.

use super::{Storage, Texture};
use crate::buffer_texture::Texels;
use crate::error::{internal, Error};
use crate::resource::ResourceKey;
use crate::shadow::{ShadowSource, ShadowTarget};
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use crate::tracker::{Access, Use, UseRange};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::TexelRows;
use metal2vulkan::metal::format::byte_format;
use metal2vulkan::metal::{TextureSwizzleChannels, TextureType};
use metal2vulkan::vk::Format;

/// A shadowed texture's rows as a function reads and writes them in place (the module docs).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowsInPlace {
    /// The words of a binding's push field.
    pub(super) rows: TexelRows,
    /// The bytes from the view's first to the texture's first.
    before: u64,
    /// The bytes of one texel.
    texel_bytes: u64,
}

impl RowsInPlace {
    /// The bytes the view holds.
    fn view_bytes(&self) -> u64 {
        self.rows.elements() * self.texel_bytes
    }
}

impl Device {
    /// The rows in place of a shadowed texture of `descriptor` (checked as `layout`) over
    /// `texels`, whose image the device refused for a reason that leaves its bytes in place (the
    /// module docs), or `None` where it keeps its shadow everywhere.
    pub(super) fn rows_in_place(
        &self,
        texels: &Texels,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
    ) -> Option<RowsInPlace> {
        let format = byte_format(descriptor.pixel_format)?;
        let block = format.block().filter(|b| b.texels == 1)?;
        let texel_bytes = u64::from(block.size);
        let at = &texels.layout;
        let [width, height, depth] = layout.extent;
        let plain = layout.texture_type == TextureType::Type2D
            && layout.levels == 1
            && layout.layers == 1
            && depth == 1
            && layout.samples == 1
            && descriptor.swizzle == TextureSwizzleChannels::DEFAULT
            && texels.texel_bytes() == texel_bytes
            && texels.past().is_none()
            && texels.staged().is_none()
            && texels.buffer.read_twin().is_none()
            && at.offset.is_multiple_of(texel_bytes)
            && at.row_pitch.is_multiple_of(texel_bytes)
            && at.row_pitch >= at.row_bytes;
        if !plain {
            return None;
        }
        let limits = self.compile_device().profile().limits();
        let alignment = limits.min_texel_buffer_offset_alignment.max(1);
        let (_, first_byte) = texels.buffer.at(at.offset);
        let before = first_byte % alignment;
        if !before.is_multiple_of(texel_bytes) {
            return None;
        }
        let rows = TexelRows::new(
            [width, height],
            at.row_pitch / texel_bytes,
            before / texel_bytes,
        )?;
        if rows.elements() > u64::from(limits.max_texel_buffer_elements) {
            return None;
        }
        self.shared().counts.texel_rows_texture();
        Some(RowsInPlace {
            rows,
            before,
            texel_bytes,
        })
    }
}

impl Texture {
    /// The texels in its buffer and the rows a function reads and writes in place, of a texture
    /// whose rows are (the module docs).
    fn rows_texels(&self) -> Result<(&Texels, &RowsInPlace), Error> {
        let rows = self.s().rows_in_place.as_ref();
        let texels = match self.storage() {
            Storage::Shadow { texels, copies, .. } => match &copies.source {
                ShadowSource::Bytes {
                    target: ShadowTarget::Image { .. },
                    ..
                } => Some(&**texels),
                _ => None,
            },
            _ => None,
        };
        match (texels, rows) {
            (Some(texels), Some(rows)) => Ok((texels, rows)),
            _ => Err(internal(
                "a rows binding of a texture whose rows are not in place",
            )),
        }
    }

    /// The texel buffer view in `format` (the plan's, `WriteSource::TexelRows`) of the texture's
    /// rows read and written in place, a storage texel buffer's (`storage`) or a uniform one's,
    /// which a slot under its rows variant binds. Internal for a texture whose rows are not.
    pub(crate) fn rows_view(&self, format: Format, storage: bool) -> Result<vk::BufferView, Error> {
        let (texels, rows) = self.rows_texels()?;
        let (_, first_byte) = texels.buffer.at(texels.layout.offset);
        let start = first_byte - rows.before;
        texels.view_of(format, storage, (start, rows.view_bytes()))
    }

    /// A use of the texture's bytes in its buffer, reading by `read` and writing by `write`, which
    /// a binding of its rows makes (the module docs). Internal for a texture whose rows are not
    /// in place.
    pub(crate) fn rows_use(&self, read: Access, write: Access) -> Result<Use<ResourceKey>, Error> {
        let (texels, rows) = self.rows_texels()?;
        let start = texels.layout.offset;
        let first = u64::from(rows.rows.words()[TexelRows::FIRST]);
        let end = start + (rows.rows.elements() - first) * rows.texel_bytes;
        let (identity, range) = texels.buffer.tracked(start..end);
        Ok(Use {
            key: identity.key,
            tracked: texels.tracked(),
            range: UseRange::Bytes(range),
            read,
            write,
        })
    }
}
