//! Blit command encoders: each copy recorded at once into the command buffer's current segment.
//!
//! A copy between textures is one `vkCmdCopyImage` between two barriers. The first moves each
//! texture's level and slice from its home layout to a transfer layout after every earlier
//! command, and makes every earlier command's writes visible to every later command: it pays the
//! barrier an earlier encoder owed, and the hazard sets start again. The second moves them home
//! once the copy has run, its writes visible to every later command. Nothing is copied but the
//! region Metal names.

use crate::command::{Building, CommandBuffer};
use crate::counters::Metric;
use crate::error::{Error, Gap, Refusal};
use crate::record::{memory_barrier, Reach};
use crate::texture::{place, CopyUnit, Placed, Region, Texture, TextureDescriptor};
use ash::vk;
use metal2vulkan::metal::TextureType;
use std::sync::Arc;

/// No Metal counterpart as a type: the slice, mipmap level and origin a blit names in one
/// texture, which Metal's calls take as three arguments (`sourceSlice:sourceLevel:sourceOrigin:`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TexturePlace {
    /// The slice: an array's element, or a cube's face, six a cube; 0 for a texture of one
    /// slice, a 3D texture among them.
    pub slice: u32,
    /// The mipmap level.
    pub level: u32,
    /// The first texel, `[x, y, z]`.
    pub origin: [u32; 3],
}

/// `MTLBlitCommandEncoder`: records each copy into its command buffer at once.
///
/// A copy runs after every earlier command of its command buffer, and every later command runs
/// after it, with the writes on either side visible across it: a blit needs no hazard tracking of
/// its own and leaves none for the encoders after it. Each texture's level and slice leaves its
/// home layout for the copy and comes back in the same call; no element of set 1 is rewritten.
///
/// Owned: one thread encodes at a time. Dropping it ends it, as
/// [`end_encoding`](Self::end_encoding) does.
pub struct BlitCommandEncoder {
    command_buffer: CommandBuffer,
    /// What the command buffer records, held while the encoder is open.
    building: Option<Building>,
}

impl CommandBuffer {
    /// `blitCommandEncoder`.
    ///
    /// Err: [`Refusal::Committed`]; [`Refusal::EncoderOpen`] while another encoder of the command
    /// buffer is open.
    pub fn blit_command_encoder(&self) -> Result<BlitCommandEncoder, Error> {
        let building = self.open_encoder()?;
        building.device.core().counters.count(Metric::Passes);
        Ok(BlitCommandEncoder {
            command_buffer: self.clone(),
            building: Some(building),
        })
    }
}

impl BlitCommandEncoder {
    fn building(&mut self) -> &mut Building {
        self.building
            .as_mut()
            .expect("an encoder holds its command buffer's recording until it ends")
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:`
    /// `destinationSlice:destinationLevel:destinationOrigin:`: `size` texels from `from` in
    /// `source` to `to` in `destination`, one `vkCmdCopyImage`. Both textures have one pixel
    /// format and one sample count, as Metal's validation requires; a compressed format's region
    /// is whole blocks or reaches the level's edge on both sides. A 3D texture's region one image
    /// deep copies into or out of
    /// a 2D slice. A size of 0 on an axis one of the textures lacks (depth on any but a 3D
    /// texture, height on a 1D one) copies one texel there, and a size of 0 on another axis copies
    /// nothing, as plain Metal does where its validation stops the call. The command buffer keeps
    /// both textures until it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's texture; [`Refusal::TextureRegion`] for
    /// a level, slice or region outside either texture or off its format's block grid;
    /// [`Refusal::TextureCopyMismatch`]; [`Gap::TextureCopy`]; [`Error::Vulkan`] where the
    /// command buffer cannot record.
    pub fn copy_from_texture(
        &mut self,
        source: &Texture,
        from: TexturePlace,
        size: [u32; 3],
        destination: &Texture,
        to: TexturePlace,
    ) -> Result<(), Error> {
        let building = self.building();
        let core = Arc::clone(building.device.core());
        if !source.is_of(&core) || !destination.is_of(&core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        let same = source.id() == destination.id();
        let sides = (Side::of(source), Side::of(destination));
        let Some(copy) = texture_copy(&sides.0, from, size, &sides.1, to, same)? else {
            return Ok(());
        };
        building.keep_texture(source);
        building.keep_texture(destination);
        let command = building.ordered_command()?;
        let (read, write) = (
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let earlier = [memory_barrier(Reach::Commands)];
        let away = [
            source.away(&copy.from, read, vk::AccessFlags2::TRANSFER_READ),
            destination.away(&copy.to, write, vk::AccessFlags2::TRANSFER_WRITE),
        ];
        let back = [
            source.back(&copy.from, read),
            destination.back(&copy.to, write),
        ];
        let regions = [vk::ImageCopy {
            src_subresource: copy.from.layers(copy.aspect),
            src_offset: copy.from.offset(),
            dst_subresource: copy.to.layers(copy.aspect),
            dst_offset: copy.to.offset(),
            extent: copy.from.extent(),
        }];
        let device = &core.device;
        // SAFETY: the command buffer records outside any render pass and keeps both images until
        // it retires; each subresource is in its home layout between commands, and the two are
        // different subresources where the textures are one (`texture_copy`), in their transfer
        // layouts between the barriers; the region lies in both.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&earlier)
                    .image_memory_barriers(&away),
            );
            device.cmd_copy_image(
                command,
                source.image(),
                read,
                destination.image(),
                write,
                &regions,
            );
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&back),
            );
        }
        core.counters.count(Metric::Barriers);
        core.counters.count(Metric::Barriers);
        Ok(())
    }

    /// `endEncoding`. Each copy ordered itself, so the encoder owes no barrier.
    pub fn end_encoding(mut self) {
        self.end();
    }

    fn end(&mut self) {
        if let Some(building) = self.building.take() {
            self.command_buffer.close_encoder(building);
        }
    }
}

impl Drop for BlitCommandEncoder {
    fn drop(&mut self) {
        self.end();
    }
}

/// One texture of a copy: its descriptor, its image's format and the unit its copies move.
pub(crate) struct Side<'a> {
    pub(crate) descriptor: &'a TextureDescriptor,
    pub(crate) format: vk::Format,
    pub(crate) unit: Option<CopyUnit>,
}

impl<'a> Side<'a> {
    fn of(texture: &'a Texture) -> Side<'a> {
        Side {
            descriptor: texture.descriptor(),
            format: texture.format(),
            unit: texture.unit(),
        }
    }
}

/// A copy between two textures, checked: the region in each, and the aspect it copies.
pub(crate) struct ImageCopy {
    pub(crate) from: Placed,
    pub(crate) to: Placed,
    pub(crate) aspect: vk::ImageAspectFlags,
}

/// Checks a copy of `size` texels from `from` in `source` to `to` in `destination`, `same` where
/// they are one texture. `None` where it copies nothing.
///
/// Err: [`Refusal::TextureCopyMismatch`]; [`Gap::TextureCopy`]; [`Refusal::TextureRegion`].
pub(crate) fn texture_copy(
    source: &Side<'_>,
    from: TexturePlace,
    size: [u32; 3],
    destination: &Side<'_>,
    to: TexturePlace,
    same: bool,
) -> Result<Option<ImageCopy>, Error> {
    let (s, d) = (source.descriptor, destination.descriptor);
    let formats = (s.pixel_format, d.pixel_format);
    let mismatch = || {
        Error::Refused(Refusal::TextureCopyMismatch {
            source: formats.0,
            destination: formats.1,
        })
    };
    let gap = || {
        Error::Unsupported(Gap::TextureCopy {
            source: formats.0,
            destination: formats.1,
        })
    };
    if formats.0 != formats.1 || s.sample_count != d.sample_count {
        return Err(mismatch());
    }
    // No unit: a multisample texture, or a format of depth and stencil both. One pixel format in
    // two Vulkan formats (chosen by the textures' usages) holds its texels in two layouts.
    let Some(unit) = source.unit else {
        return Err(gap());
    };
    if source.format != destination.format {
        return Err(gap());
    }
    let one_d = |d: &TextureDescriptor| {
        matches!(
            d.texture_type,
            TextureType::Type1D | TextureType::Type1DArray
        )
    };
    let three_d = |d: &TextureDescriptor| d.texture_type == TextureType::Type3D;
    // Vulkan 1.3 copies a 1D image only into a 1D image (`VUID-vkCmdCopyImage-srcImage-07743`).
    if one_d(s) != one_d(d) {
        return Err(gap());
    }
    let mut size = size;
    if size[1] == 0 && one_d(s) {
        size[1] = 1;
    }
    if size[2] == 0 && !(three_d(s) && three_d(d)) {
        size[2] = 1;
    }
    let region = |place: TexturePlace| Region {
        origin: place.origin,
        size,
    };
    let placed = (
        place(s, unit.block, region(from), from.level, from.slice)?,
        place(d, unit.block, region(to), to.level, to.slice)?,
    );
    let (Some(from), Some(to)) = placed else {
        return Ok(None);
    };
    // One subresource is in one layout at a time.
    if same && from.level == to.level && from.layer == to.layer {
        return Err(gap());
    }
    Ok(Some(ImageCopy {
        from,
        to,
        aspect: unit.aspect,
    }))
}
