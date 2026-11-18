//! Blit command encoders: each copy recorded at once into the command buffer's current segment.
//!
//! A copy between textures is one `vkCmdCopyImage` between two barriers. The first moves each
//! texture's level and slice from its home layout to a transfer layout after every earlier
//! command, and makes every earlier command's writes visible to every later command: it pays the
//! barrier an earlier encoder owed, and the hazard sets start again. The second moves them home
//! once the copy has run, its writes visible to every later command. Nothing is copied but the
//! region Metal names.
//!
//! A copy between a buffer and a texture is the same: `vkCmdCopyBufferToImage` or
//! `vkCmdCopyImageToBuffer` against the bytes shaders reach (the buffer's range of its block, or a
//! Managed buffer's device copy where it has one), one region per image of
//! the region, between the same two barriers; the second also makes a write into the buffer
//! visible to every later command, and to the host where the buffer is Shared.
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

use crate::buffer::Buffer;
use crate::command::{Building, CommandBuffer};
use crate::counters::Metric;
use crate::device::Core;
use crate::encoder::Resource;
use crate::error::{Error, Gap, Refusal};
use crate::fence::Fence;
use crate::hazard::Access;
use crate::over;
use crate::record::{memory_barrier, Reach};
use crate::texture::{place, BytesPlace, CopyUnit, Placed, Region, Texture, TextureDescriptor};
use ash::vk;
use metal2vulkan::metal::TextureType;
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
/// home layout for the copy and comes back in the same call; no element of set 1 is rewritten.
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
        let written = [written()];
        let written = match into_image {
            true => &written[..0],
            false => &written[..],
        };
        let device = &core.device;
        // SAFETY: the command buffer records outside any render pass and keeps the image and the
        // buffer until it retires; each copy's level and slice is in its home layout between
        // commands and in `layout` between the barriers; each region lies in the image and, by
        // `buffer_copy`, in the buffer's device range, its offset aligned as the command
        // requires.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&earlier)
                    .image_memory_barriers(&away),
            );
            for region in copies.iter().flat_map(BufferCopy::regions) {
                match into_image {
                    true => device.cmd_copy_buffer_to_image(
                        command,
                        vk_buffer,
                        texture.image(),
                        layout,
                        &[region],
                    ),
                    false => device.cmd_copy_image_to_buffer(
                        command,
                        texture.image(),
                        layout,
                        vk_buffer,
                        &[region],
                    ),
                }
            }
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(written)
                    .image_memory_barriers(&back),
            );
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
        self.record_buffer_copies(texture, bytes, &copies, false)
    }

    /// A Managed buffer's synchronize: its device copy into the consumer's bytes, where it has
    /// one.
    fn read_back(&mut self, buffer: &Buffer) -> Result<(), Error> {
        let (core, building) = self.parts();
        if let (block, Some(copy)) = buffer.held() {
            let (from, to) = (copy.buffer(), block.memory.buffer);
            let region = vk::BufferCopy {
                src_offset: copy.offset(),
                dst_offset: buffer.offset(),
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

    /// `updateFence:`: encoders that wait on `fence` after this one see what this encoder and the
    /// ones before it wrote. It records nothing ([`Fence`]): each copy's own barrier already
    /// reaches them.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's fence; [`Gap::FenceOtherQueue`] for a
    /// fence another queue used first.
    pub fn update_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        fence.used_on(&self.core, self.command_buffer.queue())
    }

    /// `waitForFence:`: this encoder sees what the encoders up to `fence`'s last update wrote. It
    /// records nothing ([`Fence`]): each copy's barrier before it already orders them.
    ///
    /// Err: as [`update_fence`](Self::update_fence).
    pub fn wait_for_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        fence.used_on(&self.core, self.command_buffer.queue())
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

/// The barrier after a blit's write into a buffer: the write visible to every later command.
fn written() -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
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

/// A copy between a buffer and a texture, checked: the texture's region, its aspect, and where
/// each image of it lies in the `VkBuffer` shaders reach the buffer through.
pub(crate) struct BufferCopy {
    pub(crate) placed: Placed,
    pub(crate) aspect: vk::ImageAspectFlags,
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
            image_subresource: self.placed.layers(self.aspect),
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
        aspect: unit.aspect,
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
