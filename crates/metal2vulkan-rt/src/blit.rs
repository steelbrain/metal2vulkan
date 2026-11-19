//! Blit command encoders: each copy recorded at once into the command buffer's current segment.
//!
//! A copy between textures is one `vkCmdCopyImage` between two barriers. The first moves each
//! texture's level and slice from its home layout to a transfer layout after every earlier
//! command, and makes every earlier command's writes visible to every later command: it pays the
//! barrier an earlier encoder owed, and the hazard sets start again. The second moves them home
//! once the copy has run, and holds one global memory barrier beside the image barriers, so the
//! copy's writes are visible to every later command through any resource over the same bytes (a
//! heap's aliases), not through the image alone. Nothing is copied but the region Metal names.
//!
//! A copy between a buffer and a texture is the same: `vkCmdCopyBufferToImage` or
//! `vkCmdCopyImageToBuffer` against the bytes shaders reach (the buffer's range of its block, or a
//! Managed buffer's device copy where it has one), one region per image of
//! the region, between the same two barriers; a write into a Shared buffer is also visible to the
//! host. A mipmap chain ends on the same kind of barrier.
//!
//! A copy between buffers is one `vkCmdCopyBuffer` region between the bytes shaders reach, and a
//! fill one `vkCmdFillBuffer`, each between two barriers: the first after every earlier command,
//! as a copy's, the second making its writes visible to every later command.
//!
//! A synchronize of a Managed buffer with a device copy is one `vkCmdCopyBuffer` of the copy into
//! the consumer's bytes between the same two barriers, and one of a Managed texture with named
//! bytes a `vkCmdCopyImageToBuffer` of each level and slice into its place, between one pair of
//! them; either marks the segment so its end reaches the host. Of any other resource it records
//! nothing.
//!
//! A release of an exported block to another API, which Metal has no counterpart of, is one
//! buffer barrier after every earlier command; the barrier that orders it before later commands
//! is owed, as a render pass's is, and the command buffer's end barrier pays it where nothing
//! follows.

use crate::buffer::{Buffer, BufferObject};
use crate::command::{Building, CommandBuffer};
use crate::copy_out;
use crate::counters::Metric;
use crate::device::Core;
use crate::encoder::Resource;
use crate::error::{Error, Gap, Refusal};
use crate::fence::Fence;
use crate::hazard::Access;
use crate::memory::Own;
use crate::over;
use crate::record::{memory_barrier, Reach};
use crate::surface::Surface;
use crate::texture::{
    place, BytesPlace, CopyUnit, Placed, Region, Texture, TextureDescriptor, TextureObject,
};
use ash::vk;
use metal2vulkan::metal::format::{self as formats, FormatEmulation};
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::pipeline::{PrecisionNote, SampledType};
use std::ops::Range;
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

/// No Metal counterpart as a type: where a texture's region lies in a buffer, which Metal's
/// copies between a buffer and a texture take as three arguments
/// (`sourceOffset:sourceBytesPerRow:sourceBytesPerImage:`, or the destination's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BufferLayout {
    /// The region's first byte in the buffer.
    pub offset: u64,
    /// From one row's first byte to the next's: a row of texels, or of blocks of a compressed
    /// format. Unused for a region one row high.
    pub bytes_per_row: u64,
    /// From one image's first byte to the next's, for a region of a 3D texture more than one image
    /// deep; unused for any other.
    pub bytes_per_image: u64,
}

/// `MTLBlitCommandEncoder`: records each copy into its command buffer at once.
///
/// A copy runs after every earlier command of its command buffer, and every later command runs
/// after it, with the writes on either side visible across it: a blit needs no hazard tracking of
/// its own and leaves none for the encoders after it. Each texture's level and slice leaves its
/// home layout for the copy and comes back in the same call; no element of set 1 is rewritten. A
/// release to another API ([`release_to_external`](Self::release_to_external)) leaves the barrier
/// after it owed.
///
/// Owned: one thread encodes at a time. Dropping it ends it, as
/// [`end_encoding`](Self::end_encoding) does.
pub struct BlitCommandEncoder {
    command_buffer: CommandBuffer,
    /// The device: each call borrows it, taking no reference of its own.
    core: Arc<Core>,
    /// What the command buffer records, held while the encoder is open.
    building: Option<Building>,
}

impl CommandBuffer {
    /// `blitCommandEncoder`.
    ///
    /// Err: [`Refusal::Committed`]; [`Refusal::EncoderOpen`] while another encoder of the command
    /// buffer is open.
    pub fn blit_command_encoder(&self) -> Result<BlitCommandEncoder, Error> {
        let mut building = self.open_encoder()?;
        building.count(Metric::Passes);
        Ok(BlitCommandEncoder {
            command_buffer: self.clone(),
            core: Arc::clone(building.device.core()),
            building: Some(building),
        })
    }
}

impl BlitCommandEncoder {
    fn building(&mut self) -> &mut Building {
        self.parts().1
    }

    /// The device and the recording, borrowed apart.
    fn parts(&mut self) -> (&Arc<Core>, &mut Building) {
        let building = (self.building.as_mut())
            .expect("an encoder holds its command buffer's recording until it ends");
        (&self.core, building)
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
        let (core, building) = self.parts();
        if !source.is_of(core) || !destination.is_of(core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        let same = source.id() == destination.id();
        let sides = (Side::of(source), Side::of(destination));
        let Some(copy) = texture_copy(&sides.0, from, size, &sides.1, to, same)? else {
            return Ok(());
        };
        building.keep_texture(source);
        building.keep_texture(destination);
        building.wrote_texture_managed(destination);
        let overs = [
            (source.hazard_id(), Access::Read),
            (destination.hazard_id(), Access::ReadWrite),
        ];
        building.overs_before(&overs, false)?;
        let command = building.ordered_command()?;
        // Marked after `ordered_command`, as in `buffer_copy`: the host reads a texture over a
        // Shared buffer's bytes in the buffer.
        if destination.host_reads() {
            building.wrote_host();
        }
        building.overs_after(&overs, false);
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
        let (named_read, named_write) = (source.copy_layout(read), destination.copy_layout(write));
        // SAFETY: the command buffer records outside any render pass and keeps both images until
        // it retires; each subresource is in its home layout between commands, and the two are
        // different subresources where the textures are one (`texture_copy`), in their copy
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
                named_read,
                destination.image(),
                named_write,
                &regions,
            );
            close(core, command, &back);
        }
        building.count(Metric::Barriers);
        building.count(Metric::Barriers);
        Ok(())
    }

    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:`
    /// `toTexture:destinationSlice:destinationLevel:destinationOrigin:`: `size` texels from
    /// `source`, laid out as `layout` says, to `to` in `destination`, one
    /// `vkCmdCopyBufferToImage` a 3D region's image. The options form with no option is this
    /// call. Sizes of 0 copy as [`copy_from_texture`](Self::copy_from_texture)'s do. The command
    /// buffer keeps the texture and the buffer until it retires. A Managed buffer's bytes here are
    /// those shaders reach: its device copy, where the device keeps one.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's buffer or texture;
    /// [`Refusal::TextureRegion`]; [`Refusal::BufferCopyLayout`] for a layout Metal's validation
    /// stops; [`Gap::BufferCopyAlignment`] for one no copy command in the floor addresses;
    /// [`Gap::TextureBytes`]; [`Gap::FloatDepthUpload`]; [`Error::Vulkan`] where the command
    /// buffer cannot record.
    pub fn copy_from_buffer(
        &mut self,
        source: &Buffer,
        layout: BufferLayout,
        size: [u32; 3],
        destination: &Texture,
        to: TexturePlace,
    ) -> Result<(), Error> {
        self.buffer_copy(destination, to, size, source, layout, true)
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:`
    /// `destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`: `size` texels at
    /// `from` in `source` to `destination`, laid out as `layout` says, one
    /// `vkCmdCopyImageToBuffer` a 3D region's image. The options form with no option is this
    /// call. Bytes between the region's rows and images are kept. Sizes of 0 copy as
    /// [`copy_from_texture`](Self::copy_from_texture)'s do. The command buffer keeps the texture
    /// and the buffer until it retires.
    ///
    /// Err: as [`copy_from_buffer`](Self::copy_from_buffer)'s, but for [`Gap::FloatDepthUpload`].
    pub fn copy_from_texture_to_buffer(
        &mut self,
        source: &Texture,
        from: TexturePlace,
        size: [u32; 3],
        destination: &Buffer,
        layout: BufferLayout,
    ) -> Result<(), Error> {
        self.buffer_copy(source, from, size, destination, layout, false)
    }

    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`: `size` bytes at
    /// `source_offset` in `source` to `destination_offset` in `destination`, one
    /// `vkCmdCopyBuffer` region between the bytes shaders reach (a Managed buffer's device copy,
    /// where the device keeps one), between a fill's two barriers. A size of 0 copies nothing. The
    /// command buffer keeps both buffers until it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's buffer; [`Refusal::BufferCopyRange`] for
    /// a range past either buffer's end; [`Refusal::BufferCopyUnaligned`] for an offset or a size
    /// that is not a multiple of 4, as Metal's validation stops it on macOS;
    /// [`Refusal::BufferCopyOverlap`] for a source and a destination that share bytes;
    /// [`Error::Vulkan`] where the command buffer cannot record.
    pub fn copy_from_buffer_to_buffer(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        destination: &Buffer,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        let (core, building) = self.parts();
        if !source.is_of(core) || !destination.is_of(core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        // A Managed buffer's GPU side is its device copy, where it has one.
        let span = |buffer: &Buffer| {
            let (buffer_of, base) = buffer.device_range();
            Span {
                buffer: buffer_of,
                base,
                len: buffer.length(),
            }
        };
        let (from, to) = (span(source), span(destination));
        let Some(region) = buffer_to_buffer(from, source_offset, to, destination_offset, size)?
        else {
            return Ok(());
        };
        building.keep(source);
        building.keep(destination);
        // The textures over either device copy go back to it first; the destination's owe their
        // copies in (`crate::over::ManagedOvers`).
        building.managed_backs(over::backs(source, None, true))?;
        building.managed_backs(over::backs(destination, None, true))?;
        building.wrote_managed(destination);
        let overs = [
            (source.id(), Access::Read),
            (destination.id(), Access::Write),
        ];
        building.overs_before(&overs, false)?;
        let command = building.ordered_command()?;
        building.overs_after(&overs, false);
        // Marked after `ordered_command`, as a copy into a buffer is.
        if destination.is_shared() {
            building.wrote_host();
        }
        let earlier = [memory_barrier(Reach::Commands)];
        let written = [written()];
        let device = &core.device;
        // SAFETY: the command buffer records outside any render pass and keeps both buffers until
        // it retires; the region lies in each buffer's device range and its source and
        // destination share no byte (`buffer_to_buffer`).
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().memory_barriers(&earlier),
            );
            device.cmd_copy_buffer(command, from.buffer, to.buffer, &[region]);
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().memory_barriers(&written),
            );
        }
        building.count(Metric::Barriers);
        building.count(Metric::Barriers);
        Ok(())
    }

    /// A copy of `size` texels at `at` in `texture` from or, not `into_image`, to `buffer`.
    fn buffer_copy(
        &mut self,
        texture: &Texture,
        at: TexturePlace,
        size: [u32; 3],
        buffer: &Buffer,
        layout: BufferLayout,
        into_image: bool,
    ) -> Result<(), Error> {
        if !texture.is_of(&self.core) || !buffer.is_of(&self.core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        // A Managed buffer's GPU side is its device copy, where it has one.
        let base = buffer.device_range().1;
        let lies = Lies {
            base,
            len: buffer.length(),
            layout,
        };
        let Some(copy) = buffer_copy(&Side::of(texture), at, size, lies, into_image)? else {
            return Ok(());
        };
        self.record_buffer_copies(texture, buffer, &[copy], into_image)
    }

    /// Records `copies` between `texture` and `buffer`, checked, between two barriers: the first
    /// after every earlier command, moving each copy's level and slice to a transfer layout; the
    /// second moving them home, a write into the buffer visible to every later command.
    fn record_buffer_copies(
        &mut self,
        texture: &Texture,
        buffer: &Buffer,
        copies: &[BufferCopy],
        into_image: bool,
    ) -> Result<(), Error> {
        let (core, building) = self.parts();
        let vk_buffer = buffer.device_range().0;
        building.keep_texture(texture);
        building.keep(buffer);
        // The device copy's textures' copies back go first; a copy into the buffer owes them
        // their copies in, and one into the texture its copy back (`crate::over::ManagedOvers`).
        building.managed_backs(over::backs(buffer, None, true))?;
        match into_image {
            true => building.wrote_texture_managed(texture),
            false => building.wrote_managed(buffer),
        }
        let overs = match into_image {
            true => [
                (texture.hazard_id(), Access::ReadWrite),
                (buffer.id(), Access::Read),
            ],
            false => [
                (texture.hazard_id(), Access::Read),
                (buffer.id(), Access::ReadWrite),
            ],
        };
        building.overs_before(&overs, false)?;
        let command = building.ordered_command()?;
        building.overs_after(&overs, false);
        // Marked after `ordered_command`, which may have split the recording at a signal: the
        // mark is the copy's own segment's.
        let host = match into_image {
            true => texture.host_reads(),
            false => buffer.is_shared(),
        };
        if host {
            building.wrote_host();
        }
        let (layout, access) = match into_image {
            true => (
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            false => (
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags2::TRANSFER_READ,
            ),
        };
        let earlier = [memory_barrier(Reach::Commands)];
        // One copy, a copy call's, takes its barriers on the stack; a synchronize's of each level
        // and slice, in lists.
        let (away_one, back_one, away_each, back_each);
        let (away, back): (&[_], &[_]) = match copies {
            [copy] => {
                away_one = [texture.away(&copy.placed, layout, access)];
                back_one = [texture.back(&copy.placed, layout)];
                (&away_one, &back_one)
            }
            _ => {
                away_each = (copies.iter())
                    .map(|copy| texture.away(&copy.placed, layout, access))
                    .collect::<Vec<_>>();
                back_each = (copies.iter())
                    .map(|copy| texture.back(&copy.placed, layout))
                    .collect::<Vec<_>>();
                (&away_each, &back_each)
            }
        };
        let device = &core.device;
        let named = texture.copy_layout(layout);
        // SAFETY: the command buffer records outside any render pass and keeps the image and the
        // buffer until it retires; each copy's level and slice is in its home layout between
        // commands and in `named` between the barriers; each region lies in the image and, by
        // `buffer_copy`, in the buffer's device range, its offset aligned as the command
        // requires.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&earlier)
                    .image_memory_barriers(away),
            );
            for copy in copies {
                for region in copy.regions() {
                    match into_image {
                        true => device.cmd_copy_buffer_to_image(
                            command,
                            vk_buffer,
                            texture.image(),
                            named,
                            &[region],
                        ),
                        false => device.cmd_copy_image_to_buffer(
                            command,
                            texture.image(),
                            named,
                            vk_buffer,
                            copy_out::regions(&copy.unit, region).as_slice(),
                        ),
                    }
                }
            }
            close(core, command, back);
        }
        building.count(Metric::Barriers);
        building.count(Metric::Barriers);
        Ok(())
    }

    /// `fillBuffer:range:value:`: `value` in every byte of `range` of `buffer`, one
    /// `vkCmdFillBuffer` of the byte four times over in each word. The command buffer keeps the
    /// buffer until it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's buffer; [`Refusal::FillRange`] for an
    /// empty range or one past the buffer's end; [`Gap::FillAlignment`] for a range that does not
    /// start and end on whole 4-byte words of the bytes shaders reach; [`Error::Vulkan`] where the
    /// command buffer cannot record.
    pub fn fill_buffer(
        &mut self,
        buffer: &Buffer,
        range: Range<u64>,
        value: u8,
    ) -> Result<(), Error> {
        let (core, building) = self.parts();
        if !buffer.is_of(core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        let (vk_buffer, base) = buffer.device_range();
        let (offset, length) = fill_range(base, buffer.length(), range)?;
        building.keep(buffer);
        building.managed_backs(over::backs(buffer, None, true))?;
        building.wrote_managed(buffer);
        let overs = [(buffer.id(), Access::Write)];
        building.overs_before(&overs, false)?;
        let command = building.ordered_command()?;
        building.overs_after(&overs, false);
        // Marked after `ordered_command`, as a copy into a buffer is.
        if buffer.is_shared() {
            building.wrote_host();
        }
        let earlier = [memory_barrier(Reach::Commands)];
        let written = [written()];
        let device = &core.device;
        // SAFETY: the command buffer records outside any render pass and keeps the buffer until it
        // retires; the range lies in the buffer's device range, in whole words
        // (`fill_range`).
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().memory_barriers(&earlier),
            );
            device.cmd_fill_buffer(
                command,
                vk_buffer,
                offset,
                length,
                u32::from_ne_bytes([value; 4]),
            );
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().memory_barriers(&written),
            );
        }
        building.count(Metric::Barriers);
        building.count(Metric::Barriers);
        Ok(())
    }

    /// The static precision note of [`generate_mipmaps`](Self::generate_mipmaps): each level is
    /// the device's linear filter of the one above, where Apple states no filter.
    pub const GENERATE_MIPMAPS_NOTE: PrecisionNote = PrecisionNote::DeviceFilteredMipmaps;

    /// `generateMipmapsForTexture:`: each level of `texture` below its first made from the level
    /// above it, over each of its slices, as Apple defines the call: one `vkCmdBlitImage` a level
    /// with the linear filter, the device's ([`GENERATE_MIPMAPS_NOTE`](Self::GENERATE_MIPMAPS_NOTE)).
    /// A view's levels and slices are its own, its first level the source of the rest. A texture
    /// of one level has none to make: the call records nothing. The levels leave the texture's
    /// home layout for the chain and come back in the same call, between a copy's barriers with
    /// one between each level and the next; the command buffer keeps the texture until it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's texture; [`Refusal::MipmapFormat`] for
    /// a format Apple defines no mipmaps of; [`Gap::MipmapFilter`] where the device's blits do not
    /// filter the texture's image; [`Error::Vulkan`] where the command buffer cannot record.
    pub fn generate_mipmaps(&mut self, texture: &Texture) -> Result<(), Error> {
        let (core, building) = self.parts();
        if !texture.is_of(core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        let d = *texture.descriptor();
        let pixel_format = d.pixel_format;
        if d.mipmap_level_count < 2 {
            return Ok(());
        }
        if !mipmapped(pixel_format) {
            return Err(Error::Refused(Refusal::MipmapFormat { pixel_format }));
        }
        // The blit filters the image's own format, so a view in another format's levels, or a
        // format the image holds through a conversion, would be filtered as other values.
        use vk::FormatFeatureFlags2 as F;
        let filters = F::BLIT_SRC | F::BLIT_DST | F::SAMPLED_IMAGE_FILTER_LINEAR;
        let (_, optimal) = core.format_features(texture.image_format());
        if texture.format() != texture.image_format()
            || !filtered(texture.emulation())
            || !optimal.contains(filters)
        {
            return Err(Error::Unsupported(Gap::MipmapFilter { pixel_format }));
        }
        building.keep_texture(texture);
        building.wrote_texture_managed(texture);
        let overs = [(texture.hazard_id(), Access::ReadWrite)];
        building.overs_before(&overs, false)?;
        let command = building.ordered_command()?;
        if texture.host_reads() {
            building.wrote_host();
        }
        building.overs_after(&overs, false);
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let (read, write) = (
            texture.copy_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            texture.copy_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
        );
        let levels = d.mipmap_level_count;
        let earlier = [memory_barrier(Reach::Commands)];
        // The first level is read; every other is written whole, its contents discarded.
        let away = [
            texture
                .levels_barrier(0..1, None, Some(read))
                .src_stage_mask(S::ALL_COMMANDS)
                .src_access_mask(A::MEMORY_WRITE)
                .dst_stage_mask(S::BLIT)
                .dst_access_mask(A::TRANSFER_READ),
            texture
                .levels_barrier(1..levels, Some(vk::ImageLayout::UNDEFINED), Some(write))
                .src_stage_mask(S::ALL_COMMANDS)
                .src_access_mask(A::NONE)
                .dst_stage_mask(S::BLIT)
                .dst_access_mask(A::TRANSFER_WRITE),
        ];
        // Home: every level but the last from the layout it was read in, the last from the one
        // it was written in, its writes visible to every later command.
        let back = [
            texture
                .levels_barrier(0..levels - 1, Some(read), None)
                .src_stage_mask(S::BLIT)
                .src_access_mask(A::NONE)
                .dst_stage_mask(S::ALL_COMMANDS)
                .dst_access_mask(A::MEMORY_READ | A::MEMORY_WRITE),
            texture
                .levels_barrier(levels - 1..levels, Some(write), None)
                .src_stage_mask(S::BLIT)
                .src_access_mask(A::TRANSFER_WRITE)
                .dst_stage_mask(S::ALL_COMMANDS)
                .dst_access_mask(A::MEMORY_READ | A::MEMORY_WRITE),
        ];
        let [level, layer] = texture.base();
        let slices = match d.texture_type {
            TextureType::Cube | TextureType::CubeArray => d.array_length * 6,
            TextureType::Type3D => 1,
            _ => d.array_length,
        };
        let extent = |at: u32| [d.width, d.height, d.depth].map(|e| (e >> at).max(1));
        let corner = |e: [u32; 3]| vk::Offset3D {
            // A level's extent is a `u32` the image format query bounds far below 2^31.
            x: e[0] as i32,
            y: e[1] as i32,
            z: e[2] as i32,
        };
        let layers = |at: u32| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: level + at,
            base_array_layer: layer,
            layer_count: slices,
        };
        let device = &core.device;
        // SAFETY: the command buffer records outside any render pass and keeps the image until it
        // retires; its levels are in their home layout between commands and in the chain's
        // layouts between its barriers; each blit reads one level and writes the next, two
        // different subresources, inside both; the image has the transfer usages every image
        // takes, and its format filters linearly in blits (checked above).
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&earlier)
                    .image_memory_barriers(&away),
            );
            for at in 1..levels {
                let region = vk::ImageBlit {
                    src_subresource: layers(at - 1),
                    src_offsets: [vk::Offset3D::default(), corner(extent(at - 1))],
                    dst_subresource: layers(at),
                    dst_offsets: [vk::Offset3D::default(), corner(extent(at))],
                };
                device.cmd_blit_image(
                    command,
                    texture.image(),
                    read,
                    texture.image(),
                    write,
                    &[region],
                    vk::Filter::LINEAR,
                );
                if at + 1 == levels {
                    break;
                }
                // The level written is read by the next blit.
                let written = [texture
                    .levels_barrier(at..at + 1, Some(write), Some(read))
                    .src_stage_mask(S::BLIT)
                    .src_access_mask(A::TRANSFER_WRITE)
                    .dst_stage_mask(S::BLIT)
                    .dst_access_mask(A::TRANSFER_READ)];
                device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().image_memory_barriers(&written),
                );
            }
            close(core, command, &back);
        }
        // The chain's barriers: before it, between each level and the next, and after it.
        for _ in 0..levels {
            building.count(Metric::Barriers);
        }
        Ok(())
    }

    /// `synchronizeResource:`: the host's bytes of a Managed `resource` brought up to date with
    /// the GPU's writes to it before the call, which the host reads once the command buffer
    /// completes or it sees an event signaled after the call. Counted in
    /// [`Metrics::synchronizations`](crate::Metrics::synchronizations); it makes no submission of
    /// its own.
    ///
    /// A Managed buffer with a device copy
    /// ([`Device::new_managed_buffer`](crate::Device::new_managed_buffer) on a device whose
    /// Shared memory is not device-local) is one `vkCmdCopyBuffer` of the copy into the
    /// consumer's bytes, between a fill's two barriers. A Managed texture with named bytes
    /// ([`Device::new_managed_texture`](crate::Device::new_managed_texture)), on every device, is
    /// one `vkCmdCopyImageToBuffer` of each level and slice of its image into its place in them,
    /// between one pair of a buffer copy's barriers, as
    /// [`copy_from_texture_to_buffer`](Self::copy_from_texture_to_buffer) records. Either marks
    /// the segment so its end barrier reaches the host, and the command buffer keeps the resource
    /// until it retires. A buffer with no device copy is one the host reads where shaders write
    /// it, and a texture without named bytes has none the host reads apart from its image: the
    /// call records nothing for either.
    ///
    /// Metal's validation stops the call on a resource that is not Managed (probe
    /// `blit_synchronize_rules.m` in the conformance crate, an Apple M3 Max, macOS 27.0.1, job
    /// m3-max-43632); the runtime keeps no storage mode for a buffer without a device copy or for a
    /// texture without named bytes, and records nothing there.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's resource; [`Error::Vulkan`] where the
    /// command buffer cannot record.
    pub fn synchronize_resource(&mut self, resource: Resource<'_>) -> Result<(), Error> {
        match resource {
            Resource::Buffer(buffer) if buffer.is_of(&self.core) => self.read_back(buffer)?,
            Resource::Texture(texture) if texture.is_of(&self.core) => {
                self.read_back_texture(texture)?
            }
            _ => return Err(Error::Refused(Refusal::OtherDevice)),
        }
        self.building().count(Metric::Synchronizations);
        Ok(())
    }

    /// A Managed texture's synchronize: each level and slice of its image into its place in the
    /// named bytes, where it has them.
    fn read_back_texture(&mut self, texture: &Texture) -> Result<(), Error> {
        let Some((bytes, places)) = texture.named_bytes() else {
            return Ok(());
        };
        let d = texture.descriptor();
        let slices = places.len() as u32 / d.mipmap_level_count;
        let side = Side::of(texture);
        let base = bytes.device_range().1;
        let mut copies = Vec::with_capacity(places.len());
        for (at, &place) in (0..).zip(places) {
            let BytesPlace {
                offset,
                bytes_per_row,
                bytes_per_image,
            } = place;
            let layout = BufferLayout {
                offset,
                bytes_per_row,
                bytes_per_image,
            };
            let lies = Lies {
                base,
                len: bytes.length(),
                layout,
            };
            let level = at / slices;
            let at = TexturePlace {
                slice: at % slices,
                level,
                origin: [0; 3],
            };
            let size = [d.width, d.height, d.depth].map(|extent| (extent >> level).max(1));
            copies.extend(buffer_copy(&side, at, size, lies, false)?);
        }
        self.record_buffer_copies(texture, &bytes, &copies, false)
    }

    /// A Managed buffer's synchronize: its device copy into the consumer's bytes, where it has
    /// one.
    fn read_back(&mut self, buffer: &Buffer) -> Result<(), Error> {
        let (core, building) = self.parts();
        if let (block, Some(copy)) = buffer.held() {
            let ((from, src_offset), (to, dst_offset)) =
                (copy.range(0), block.memory.range(buffer.offset()));
            let region = vk::BufferCopy {
                src_offset,
                dst_offset,
                size: buffer.length(),
            };
            building.keep(buffer);
            // The textures over the device copy go back to it first (`crate::over`).
            building.managed_backs(over::backs(buffer, None, true))?;
            let command = building.ordered_command()?;
            // Marked after `ordered_command`, as a copy into a Shared buffer is.
            building.wrote_host();
            let earlier = [memory_barrier(Reach::Commands)];
            let written = [written()];
            let device = &core.device;
            // SAFETY: the command buffer records outside any render pass and keeps the buffer, its
            // block and its device copy until it retires; the copy's place and the buffer's range
            // of its block are each `length` bytes, and do not overlap.
            unsafe {
                device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().memory_barriers(&earlier),
                );
                device.cmd_copy_buffer(command, from, to, &[region]);
                device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().memory_barriers(&written),
                );
            }
            building.count(Metric::Barriers);
            building.count(Metric::Barriers);
        }
        Ok(())
    }

    /// No Metal counterpart: releases `buffer`'s exported block
    /// ([`Device::new_exported_block`](crate::Device::new_exported_block)) to the other API, which
    /// reads the block once this command buffer has completed: what its commands wrote (a video
    /// encoder that reads the frames a blit wrote, for one), or what the other API wrote before
    /// and the commands only read (a decoder that reads a reference picture again). The consumer
    /// calls it after the encoder's last command that uses the block, and has the other API read
    /// the block after its own wait on the command buffer's completion.
    ///
    /// The call records the whole block's release from the runtime's queue family to the other
    /// API's (`VK_QUEUE_FAMILY_EXTERNAL` for an opaque handle, `VK_QUEUE_FAMILY_FOREIGN_EXT` for
    /// a dma-buf) into this command buffer, after every command before it, whose writes it makes
    /// available: one buffer barrier, counted in
    /// [`Metrics::external_releases`](crate::Metrics::external_releases). It is no submission of
    /// its own and waits for nothing. A texture over a buffer in the block, in an image of its
    /// own, is copied back into it first, as before any command that writes the buffer. The
    /// barrier that orders the release before later commands is owed: the command buffer's next
    /// command records it, or its end barrier where none follows.
    ///
    /// The contract: from the call until the consumer's next write into the block through the
    /// runtime or the next [`Block::external_writes_done`](crate::Block::external_writes_done),
    /// the consumer encodes no read of the block (of a buffer in it, or of a texture over one),
    /// and the runtime reads none of it. That next write, once the other API's reads have
    /// completed, needs no acquire: Vulkan lets the releasing queue use the block again without
    /// one, its contents written anew ("It is valid to never call the acquire operation after a
    /// release, and instead simply start using the resource on any queue (even the releasing
    /// queue), but the contents should be reinitialized before being read"). Where the other API
    /// writes the block in between, `external_writes_done` comes before the runtime's commands
    /// use what it wrote; where it only reads the block and the runtime is to read it again,
    /// `external_writes_done` comes first too, once those reads have completed: the acquire after
    /// the release, which keeps the block's bytes.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's buffer; [`Refusal::NotExported`] for
    /// a buffer whose block is not exported; [`Error::Vulkan`] where the command buffer cannot
    /// record.
    pub fn release_to_external(&mut self, buffer: &Buffer) -> Result<(), Error> {
        let (core, building) = self.parts();
        if !buffer.is_of(core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        let (block, _) = buffer.held();
        let Some(Own::Export(handle)) = block.memory.own else {
            return Err(Error::Refused(Refusal::NotExported {
                block: block.memory.id,
            }));
        };
        let (whole, _) = block.memory.range(0);
        building.keep(buffer);
        // The textures over a buffer in the block go back to it first, as before a write of it.
        let overs = [(buffer.id(), Access::Write)];
        building.overs_before(&overs, false)?;
        let command = building.release_command()?;
        building.overs_after(&overs, false);
        let release = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .src_queue_family_index(core.family)
            .dst_queue_family_index(handle.family())
            .buffer(whole)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        let dependency = vk::DependencyInfo::default().buffer_memory_barriers(&release);
        // SAFETY: the command buffer records outside any render pass and keeps the buffer, so its
        // block, until it retires; the barrier spans the block's own allocation, its buffer's,
        // and a dma-buf's foreign family is enabled (`Device::new_exported_block`).
        unsafe { core.device.cmd_pipeline_barrier2(command, &dependency) };
        building.count(Metric::Barriers);
        building.count(Metric::ExternalReleases);
        Ok(())
    }

    /// No Metal counterpart: releases `surface`
    /// ([`Device::new_exported_surface`](crate::Device::new_exported_surface)) to the other API,
    /// which reads its planes once this command buffer has completed: what its commands wrote (a
    /// video encoder that reads the pictures a pass rendered, for one), or what the other API
    /// wrote before and the commands only read (a decoder that reads a reference picture again).
    /// The consumer calls it after the encoder's last command that uses the planes, and has the
    /// other API read the surface after its own wait on the command buffer's completion.
    ///
    /// In place, the call records each plane's image's release from the runtime's queue family to
    /// the other API's (`VK_QUEUE_FAMILY_EXTERNAL` for an opaque handle,
    /// `VK_QUEUE_FAMILY_FOREIGN_EXT` for a dma-buf), the general layout kept, and the surface's
    /// buffer's where [`Surface::buffer`] has made it, its textures over it in images of their own
    /// copied back into it first, after every command before it: one barrier of the planes and
    /// the buffer, counted in
    /// [`Metrics::external_releases`](crate::Metrics::external_releases). In the copy form it is
    /// the block's release ([`release_to_external`](Self::release_to_external)), the planes'
    /// images copied back into the block first. The contract is the block's: from the call until
    /// the consumer's next write into the surface through the runtime or the next
    /// [`Surface::external_writes_done`](crate::Surface::external_writes_done), the consumer
    /// encodes no read of its planes.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's surface; [`Error::Vulkan`] where the
    /// command buffer cannot record.
    pub fn release_surface_to_external(&mut self, surface: &Surface) -> Result<(), Error> {
        if let Some(buffer) = surface.copied() {
            return self.release_to_external(buffer);
        }
        let (core, building) = self.parts();
        if !surface.is_of(core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        for texture in surface.planes() {
            building.keep_texture(texture);
        }
        let buffer = surface.made_buffer();
        let overs: Vec<(u32, Access)> = (buffer.iter())
            .map(|buffer| (buffer.id(), Access::Write))
            .collect();
        if let Some(buffer) = &buffer {
            building.keep(buffer);
            // The textures over the buffer go back to it first, as before a write of it.
            building.overs_before(&overs, false)?;
        }
        let command = building.release_command()?;
        if buffer.is_some() {
            building.overs_after(&overs, false);
        }
        let family = surface.family();
        let buffers: Vec<vk::BufferMemoryBarrier2<'static>> = (buffer.iter())
            .map(|buffer| {
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .src_queue_family_index(core.family)
                    .dst_queue_family_index(family)
                    .buffer(buffer.device_range().0)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();
        let releases: Vec<vk::ImageMemoryBarrier2<'static>> = (surface.planes().iter())
            .map(|texture| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(core.family)
                    .dst_queue_family_index(family)
                    .image(texture.image())
                    .subresource_range(texture.whole())
            })
            .collect();
        let dependency = vk::DependencyInfo::default()
            .image_memory_barriers(&releases)
            .buffer_memory_barriers(&buffers);
        // SAFETY: the command buffer records outside any render pass and keeps the planes and the
        // buffer until it retires; each image is in its home layout, general, between commands,
        // the buffer spans the surface's memory, and a dma-buf's foreign family is enabled.
        unsafe { core.device.cmd_pipeline_barrier2(command, &dependency) };
        building.count(Metric::Barriers);
        building.count(Metric::ExternalReleases);
        Ok(())
    }

    /// `updateFence:`: encoders that wait on `fence` after this one, of any queue, see what this
    /// encoder and the ones before it wrote. It records nothing ([`Fence`]): each copy's own
    /// barrier already reaches every command submitted after it.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's fence.
    pub fn update_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        fence.check(&self.core)?;
        self.building().fences().update(fence);
        Ok(())
    }

    /// `waitForFence:`: this encoder sees what the encoders up to each update of `fence`
    /// submitted before it wrote, of any queue. It records nothing ([`Fence`]): each copy's
    /// barrier before it already orders them.
    ///
    /// Err: as [`update_fence`](Self::update_fence).
    pub fn wait_for_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        fence.check(&self.core)?;
        self.building().fences().wait(fence);
        Ok(())
    }

    /// `endEncoding`. Each copy ordered itself, so the encoder owes no barrier but a release's.
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

/// The barrier after a blit's write: the write visible to every later command.
fn written() -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
}

/// Records the barrier that ends a blit of an image: `back`, its subresources' moves home, with
/// one global memory barrier ([`written`]) in the same call. An image barrier makes the blit's
/// writes visible through that image alone; a heap texture's bytes may be another heap resource's
/// too, and Vulkan's memory aliasing asks the barrier between two aliases' uses to reach every
/// byte they share.
///
/// # Safety
///
/// `command` is recording outside any render pass, and each subresource of `back` is in the
/// layout its move starts from.
unsafe fn close(core: &Core, command: vk::CommandBuffer, back: &[vk::ImageMemoryBarrier2<'_>]) {
    let written = [written()];
    let dependency = vk::DependencyInfo::default()
        .memory_barriers(&written)
        .image_memory_barriers(back);
    // SAFETY: the caller's contract.
    unsafe { core.device.cmd_pipeline_barrier2(command, &dependency) };
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    core.counters.closed(dependency.memory_barrier_count);
}

/// One texture of a copy: its descriptor, its image's format, the unit its copies move, and
/// where it starts in its image (a view's first level and array layer).
pub(crate) struct Side<'a> {
    pub(crate) descriptor: &'a TextureDescriptor,
    pub(crate) format: vk::Format,
    pub(crate) unit: Option<CopyUnit>,
    pub(crate) base: [u32; 2],
}

impl<'a> Side<'a> {
    fn of(texture: &'a Texture) -> Side<'a> {
        Side {
            descriptor: texture.descriptor(),
            format: texture.format(),
            unit: texture.unit(),
            base: texture.base(),
        }
    }
}

/// A copy between two textures, checked: the region in each, and the aspect it copies.
pub(crate) struct ImageCopy {
    pub(crate) from: Placed,
    pub(crate) to: Placed,
    pub(crate) aspect: vk::ImageAspectFlags,
}

/// Checks a fill of `range` of a buffer at `base` in its block, `len` bytes long: its offset in
/// the block and its length.
///
/// Err: [`Refusal::FillRange`]; [`Gap::FillAlignment`].
pub(crate) fn fill_range(base: u64, len: u64, range: Range<u64>) -> Result<(u64, u64), Error> {
    let Range { start, end } = range;
    if start >= end || end > len {
        return Err(Error::Refused(Refusal::FillRange { start, end, len }));
    }
    let offset = base + start;
    let length = end - start;
    if offset % 4 != 0 || length % 4 != 0 {
        return Err(Error::Unsupported(Gap::FillAlignment { start, end }));
    }
    Ok((offset, length))
}

/// One buffer of a copy between buffers: the `VkBuffer` shaders reach it in, its offset there,
/// and its length.
#[derive(Clone, Copy)]
pub(crate) struct Span {
    pub(crate) buffer: vk::Buffer,
    pub(crate) base: u64,
    pub(crate) len: u64,
}

/// Checks a copy of `size` bytes from `source_offset` in `source` to `destination_offset` in
/// `destination`: the region of `vkCmdCopyBuffer` to record, `None` for a size of 0.
///
/// Err: [`Refusal::BufferCopyRange`]; [`Refusal::BufferCopyUnaligned`];
/// [`Refusal::BufferCopyOverlap`].
pub(crate) fn buffer_to_buffer(
    source: Span,
    source_offset: u64,
    destination: Span,
    destination_offset: u64,
    size: u64,
) -> Result<Option<vk::BufferCopy>, Error> {
    for (span, offset) in [(source, source_offset), (destination, destination_offset)] {
        if offset.checked_add(size).is_none_or(|end| end > span.len) {
            return Err(Error::Refused(Refusal::BufferCopyRange {
                offset,
                size,
                len: span.len,
            }));
        }
    }
    if source_offset % 4 != 0 || destination_offset % 4 != 0 || size % 4 != 0 {
        return Err(Error::Refused(Refusal::BufferCopyUnaligned {
            source_offset,
            destination_offset,
            size,
        }));
    }
    if size == 0 {
        return Ok(None);
    }
    let (from, to) = (
        source.base + source_offset,
        destination.base + destination_offset,
    );
    if source.buffer == destination.buffer && from < to + size && to < from + size {
        return Err(Error::Refused(Refusal::BufferCopyOverlap {
            source_offset,
            destination_offset,
            size,
        }));
    }
    Ok(Some(vk::BufferCopy {
        src_offset: from,
        dst_offset: to,
        size,
    }))
}

/// A buffer's side of a copy: its offset in its block, its length, and where the region lies
/// in it.
#[derive(Clone, Copy)]
pub(crate) struct Lies {
    pub(crate) base: u64,
    pub(crate) len: u64,
    pub(crate) layout: BufferLayout,
}

/// A copy between a buffer and a texture, checked: the texture's region, its copy unit, and where
/// each image of it lies in the `VkBuffer` shaders reach the buffer through.
pub(crate) struct BufferCopy {
    pub(crate) placed: Placed,
    pub(crate) unit: CopyUnit,
    /// The first image's first byte in the block.
    pub(crate) offset: u64,
    /// A row's stride in texels; 0 for a region one row high.
    pub(crate) row_length: u32,
    pub(crate) bytes_per_image: u64,
}

impl BufferCopy {
    /// One region an image of the texture's region, each with its own offset: a stride of
    /// images that is no whole number of rows needs no row count.
    pub(crate) fn regions(&self) -> impl Iterator<Item = vk::BufferImageCopy> + '_ {
        let [x, y, z] = self.placed.origin;
        let [width, height, depth] = self.placed.size;
        (0..depth).map(move |image| vk::BufferImageCopy {
            buffer_offset: self.offset + u64::from(image) * self.bytes_per_image,
            buffer_row_length: self.row_length,
            buffer_image_height: 0,
            image_subresource: self.placed.layers(self.unit.aspect),
            image_offset: vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: (z + image) as i32,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        })
    }
}

/// Checks a copy of `size` texels at `at` in `texture` and the bytes `lies` names in a buffer,
/// `into_image` for a copy into the texture. `None` where it copies nothing.
///
/// Metal's validation stops an offset or a row stride of a format of single texels that is no
/// whole number of its texels, rows or images closer than the region's, and a region past the
/// buffer's end ([`Refusal::BufferCopyLayout`]). A copy command in the floor further takes an
/// offset of whole texel blocks, a multiple of 4 for a depth or stencil aspect, for each image,
/// and a row stride of whole blocks ([`Gap::BufferCopyAlignment`]).
///
/// Err: [`Gap::TextureBytes`]; [`Gap::FloatDepthUpload`]; [`Refusal::TextureRegion`];
/// [`Refusal::BufferCopyLayout`]; [`Gap::BufferCopyAlignment`].
pub(crate) fn buffer_copy(
    texture: &Side<'_>,
    at: TexturePlace,
    size: [u32; 3],
    lies: Lies,
    into_image: bool,
) -> Result<Option<BufferCopy>, Error> {
    let d = texture.descriptor;
    let pixel_format = d.pixel_format;
    let Some(unit) = texture.unit else {
        return Err(Error::Unsupported(Gap::TextureBytes { pixel_format }));
    };
    if into_image && unit.float_depth() {
        return Err(Error::Unsupported(Gap::FloatDepthUpload { pixel_format }));
    }
    let mut size = size;
    if size[1] == 0
        && matches!(
            d.texture_type,
            TextureType::Type1D | TextureType::Type1DArray
        )
    {
        size[1] = 1;
    }
    if size[2] == 0 && d.texture_type != TextureType::Type3D {
        size[2] = 1;
    }
    let region = Region {
        origin: at.origin,
        size,
    };
    let Some(placed) = place(d, texture.base, unit.block, region, at.level, at.slice)? else {
        return Ok(None);
    };
    let Lies { base, len, layout } = lies;
    let BufferLayout {
        offset,
        bytes_per_row,
        bytes_per_image,
    } = layout;
    let refused = || {
        Error::Refused(Refusal::BufferCopyLayout {
            offset,
            bytes_per_row,
            bytes_per_image,
            len,
        })
    };
    let [bw, bh] = unit.block;
    let bytes = u64::from(unit.bytes);
    let row = u64::from(placed.size[0].div_ceil(bw)) * bytes;
    let rows = u64::from(placed.size[1].div_ceil(bh));
    let images = u64::from(placed.size[2]);
    let single = unit.block == [1, 1];
    if single && (offset % bytes != 0 || (rows > 1 && bytes_per_row % bytes != 0)) {
        return Err(refused());
    }
    let image_span = match rows {
        1 => Some(row),
        _ if bytes_per_row < row => return Err(refused()),
        _ => (rows - 1)
            .checked_mul(bytes_per_row)
            .and_then(|span| span.checked_add(row)),
    };
    let span = match (images, image_span) {
        (_, None) => None,
        (1, span) => span,
        (_, Some(image_span)) if bytes_per_image < image_span => return Err(refused()),
        (_, Some(image_span)) => (images - 1)
            .checked_mul(bytes_per_image)
            .and_then(|span| span.checked_add(image_span)),
    };
    if span
        .and_then(|span| span.checked_add(offset))
        .is_none_or(|end| end > len)
    {
        return Err(refused());
    }
    let align = match unit.aspect {
        vk::ImageAspectFlags::COLOR => bytes,
        _ => 4,
    };
    let start = base + offset;
    let row_length = match rows {
        1 => Some(0),
        _ => u32::try_from(bytes_per_row / bytes)
            .ok()
            .and_then(|elements| elements.checked_mul(bw)),
    };
    let aligned = start % align == 0
        && (rows == 1 || bytes_per_row % bytes == 0)
        && (images == 1 || bytes_per_image % align == 0);
    let (Some(row_length), true) = (row_length, aligned) else {
        return Err(Error::Unsupported(Gap::BufferCopyAlignment {
            pixel_format,
            offset,
            bytes_per_row,
            bytes_per_image,
        }));
    };
    Ok(Some(BufferCopy {
        placed,
        unit,
        offset: start,
        row_length,
        bytes_per_image,
    }))
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
        place(
            s,
            source.base,
            unit.block,
            region(from),
            from.level,
            from.slice,
        )?,
        place(
            d,
            destination.base,
            unit.block,
            region(to),
            to.level,
            to.slice,
        )?,
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

/// Whether Apple defines mipmaps of `pf`: a colour-renderable, filterable format, which is not a
/// depth or stencil, compressed, YCbCr or integer one ("generateMipmapsForTexture:").
pub(crate) fn mipmapped(pf: PixelFormat) -> bool {
    !formats::is_depth_stencil(pf)
        && formats::block_extent(pf).is_none()
        && !formats::ycbcr(pf)
        && formats::sampled_type(pf) == Some(SampledType::Float)
}

/// Whether an image holding its pixel format as `emulation` holds each channel as one channel of
/// the same values, which a blit filters as the pixel format's.
fn filtered(emulation: FormatEmulation) -> bool {
    matches!(
        emulation,
        FormatEmulation::Direct
            | FormatEmulation::SwizzledView
            | FormatEmulation::AlphaInRed
            | FormatEmulation::RotatedView
            | FormatEmulation::XChannel
    )
}
