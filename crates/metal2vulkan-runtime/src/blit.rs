//! Blit command encoders: each copy recorded at once into the command buffer's current segment.
//!
//! A copy between textures is one `vkCmdCopyImage` between two barriers. The first moves each
//! texture's level and slice from its home layout to a transfer layout after every earlier
//! command, and makes every earlier command's writes visible to every later command: it pays the
//! barrier an earlier encoder owed, and the hazard sets start again. The second moves them home
//! once the copy has run, and holds one global memory barrier beside the image barriers, so the
//! copy's writes are visible to every later command through any resource over the same bytes (a
//! heap's aliases), not through the image alone. Nothing is copied but the region Metal names.
//! Regions whose bytes meet, of one level and slice or of two textures in place over one buffer,
//! are the one exception to one command: slabs, one `vkCmdCopyImage` each, an execution barrier
//! between two ([`Slabs`]). Memory a texture placed in a heap shares with the other side is in no
//! order a copy command keeps (`Gap::OverlappingCopy`).
//!
//! A copy between a buffer and a texture is the same: `vkCmdCopyBufferToImage` or
//! `vkCmdCopyImageToBuffer` against the bytes shaders reach (the buffer's range of its block, or a
//! Managed buffer's device copy where it has one), one region per image of
//! the region, between the same two barriers, and in slabs where the region's bytes meet those of
//! the texture in place over them; a write into a Shared buffer is also visible to the host. A
//! mipmap chain ends on the same kind of barrier.
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
use crate::indirect::{reset_regions, IndirectCommandBuffer};
use crate::memory::Own;
use crate::over::Reached;
use crate::record::{memory_barrier, Reach};
use crate::surface::Surface;
use crate::texture::{
    copy_unit, place, BytesPlace, CopyUnit, Placed, Region, Rows, Texture, TextureDescriptor,
    TextureObject,
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

/// `MTLBlitOption`, the option of a copy between a buffer and a texture: which aspect of a depth
/// and stencil format the copy moves. A copy of `None` moves the texture's one aspect. One option
/// a copy, as Metal's validation takes them; Metal's `MTLBlitOptionRowLinearPVRTC` is not offered,
/// the runtime making no PVRTC texture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlitOption {
    /// `MTLBlitOptionNone`.
    #[default]
    None,
    /// `MTLBlitOptionDepthFromDepthStencil`: the depth aspect, its bytes those of the format's
    /// depth alone (4 a texel of `Depth32Float_Stencil8`).
    DepthFromDepthStencil,
    /// `MTLBlitOptionStencilFromDepthStencil`: the stencil aspect, one byte a texel.
    StencilFromDepthStencil,
}

/// The way a copy between a buffer and a texture goes, and the option it takes.
#[derive(Clone, Copy)]
enum Way {
    /// From the buffer into the texture.
    IntoImage(BlitOption),
    /// From the texture into the buffer.
    IntoBuffer(BlitOption),
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
    /// is whole blocks or reaches the level's edge on both sides, and whole blocks may end past
    /// the edge inside the level's last block (the one 4 x 4 block of a level of 4 x 2 texels):
    /// plain Metal copies that block whole, and the command's extent stops at the edge, both
    /// sides' at the same one (where they stop at different ones, [`Gap::TextureCopy`]). A format
    /// of depth and stencil both copies its pixels whole, as Metal's copy takes no plane option: each region of the
    /// command names both aspects. A 3D texture's region one image
    /// deep copies into or out of
    /// a 2D slice. A size of 0 on an axis one of the textures lacks (depth on any but a 3D
    /// texture, height on a 1D one) copies one texel there, and a size of 0 on another axis copies
    /// nothing, as plain Metal does where its validation stops the call. Within one level and
    /// slice of one image (the texture into itself, or a texture and its view), regions apart
    /// copy with that subresource in the general layout, and the same region copies nothing and
    /// records no command, every texel already its own. Overlapping regions there, and regions of
    /// two textures in place over one buffer ([`Buffer::new_texture`]) whose bytes meet, which
    /// Metal leaves undefined, copy as if the source were read whole before any texel is written,
    /// the answer an Apple M3 Max gives: in slabs across one axis, each no thicker than the shift
    /// between the regions there (between two textures over one buffer, from each destination
    /// row to each source row on its bytes: in rows along y, in whole texels along x), so that its
    /// two regions share no byte, one `vkCmdCopyImage` a slab and the far end of the shift first,
    /// an execution barrier between two slabs. The axis is the one of fewest slabs; a shift of one
    /// texel takes one slab a texel. Two textures over one buffer whose regions are the same bytes
    /// copy nothing. The command buffer keeps both textures until it retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's texture; [`Refusal::TextureRegion`] for
    /// a level, slice or region outside either texture or off its format's block grid;
    /// [`Refusal::TextureCopyMismatch`]; [`Gap::TextureCopy`]; [`Gap::OverlappingCopy`], counted,
    /// for regions of two textures over one buffer whose bytes meet where no axis cuts them so,
    /// and for two textures whose memory meets where one is placed in a heap
    /// ([`Device::new_heap_texture`](crate::Device::new_heap_texture)); [`Error::Vulkan`] where the
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
        // A texture buffer's texels are its buffer's bytes, which no image copy reaches.
        if source.texels() || destination.texels() {
            return Err(Error::Unsupported(Gap::TextureCopy {
                source: source.descriptor().pixel_format,
                destination: destination.descriptor().pixel_format,
            }));
        }
        let same = source.id() == destination.id();
        let sides = (Side::of(source), Side::of(destination));
        let Some(copy) = texture_copy(&sides.0, from, size, &sides.1, to, same)? else {
            return Ok(());
        };
        if copy.slabs.is_none() {
            building.count(Metric::OverlappingCopiesRefused);
            let pixel_format = sides.0.descriptor.pixel_format;
            return Err(Error::Unsupported(Gap::OverlappingCopy { pixel_format }));
        }
        building.keep_texture(source);
        building.keep_texture(destination);
        building.wrote_texture_managed(destination);
        // A heap texture it writes that an alias's write left stale moves from undefined first
        // (`crate::alias`).
        if building.aliased() {
            building.places(
                [
                    (source.object().placement(), Access::Read),
                    (destination.object().placement(), Access::Write),
                ]
                .into_iter(),
            )?;
        }
        let overs = [
            (Reached::Texture(source.object()), Access::Read),
            (Reached::Texture(destination.object()), Access::ReadWrite),
        ];
        building.overs_before(overs, false)?;
        let command = building.ordered_command()?;
        // Marked after `ordered_command`, as in `buffer_copy`: the host reads a texture over a
        // Shared buffer's bytes in the buffer.
        if destination.host_reads() {
            building.wrote_host();
        }
        building.overs_after(overs, false);
        let (read, write) = match copy.shared {
            false => (
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ),
            true => (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL),
        };
        let earlier = [memory_barrier(Reach::Commands)];
        // Regions apart in one subresource: one barrier moves it to the general layout and one
        // home, the copy reading and writing it there.
        let (away_two, back_two, away_one, back_one);
        let (away, back): (&[_], &[_]) = match copy.shared {
            false => {
                away_two = [
                    source.away(&copy.from, read, vk::AccessFlags2::TRANSFER_READ),
                    destination.away(&copy.to, write, vk::AccessFlags2::TRANSFER_WRITE),
                ];
                back_two = [
                    source.back(&copy.from, read),
                    destination.back(&copy.to, write),
                ];
                (&away_two, &back_two)
            }
            true => {
                let both = vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE;
                away_one = [destination.away(&copy.to, write, both)];
                back_one = [destination.back(&copy.to, write)];
                (&away_one, &back_one)
            }
        };
        let between = [between()];
        let device = &core.device;
        let (named_read, named_write) = (source.copy_layout(read), destination.copy_layout(write));
        // SAFETY: the command buffer records outside any render pass and keeps both images until
        // it retires; each subresource is in its home layout between commands, and in its copy
        // layout between the barriers: two subresources, or one in the general layout; each
        // slab's two regions share no byte (`texture_copy`), and lie in both.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&earlier)
                    .image_memory_barriers(away),
            );
            for (n, region) in copy.regions().enumerate() {
                if n > 0 {
                    let dependency = vk::DependencyInfo::default().memory_barriers(&between);
                    device.cmd_pipeline_barrier2(command, &dependency);
                    building.count(Metric::Barriers);
                }
                device.cmd_copy_image(
                    command,
                    source.image(),
                    named_read,
                    destination.image(),
                    named_write,
                    &[region],
                );
            }
            close(core, command, back);
        }
        building.count(Metric::Barriers);
        building.count(Metric::Barriers);
        Ok(())
    }

    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:`
    /// `toTexture:destinationSlice:destinationLevel:destinationOrigin:`: `size` texels from
    /// `source`, laid out as `layout` says, to `to` in `destination`, one
    /// `vkCmdCopyBufferToImage` a 3D region's image. The options form with no option is this
    /// call. Sizes of 0 copy as [`copy_from_texture`](Self::copy_from_texture)'s do, and a
    /// compressed region takes whole blocks ending inside the level's last block as its does.
    /// Metal's validation layer stops this call's (`(destinationOrigin.y + sourceSize.height)(4)
    /// must be <= height(2)` at a level of 4 x 2) and plain Metal copies the block; a guest's
    /// Metal runs without the layer, so the runtime does what plain Metal does. The command
    /// buffer keeps the texture and the buffer until it retires. A Managed buffer's bytes here are
    /// those shaders reach: its device copy, where the device keeps one. A texture in place over
    /// those bytes ([`Buffer::new_texture`]) whose rows meet the region's, which Metal leaves
    /// undefined, copies as [`copy_from_texture`](Self::copy_from_texture) copies two textures over
    /// one buffer whose bytes meet: in slabs, as from the source before the copy; the region on
    /// its own texels copies nothing. Into a 32-bit float depth it carries
    /// [`Texture::FLOAT_DEPTH_UPLOAD_NOTE`].
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's buffer or texture;
    /// [`Refusal::TextureRegion`]; [`Refusal::BufferCopyLayout`] for a layout Metal's validation
    /// stops; [`Gap::BufferCopyAlignment`] for one no copy command in the floor addresses;
    /// [`Gap::TextureBytes`]; [`Gap::OverlappingCopy`], counted, for
    /// rows of a texture in place that meet the region's where no axis cuts them so, and for a
    /// texture placed in a heap whose memory meets the region's bytes; [`Error::Vulkan`] where the
    /// command buffer cannot record.
    pub fn copy_from_buffer(
        &mut self,
        source: &Buffer,
        layout: BufferLayout,
        size: [u32; 3],
        destination: &Texture,
        to: TexturePlace,
    ) -> Result<(), Error> {
        let option = BlitOption::None;
        self.copy_from_buffer_with_options(source, layout, size, destination, to, option)
    }

    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:`
    /// `toTexture:destinationSlice:destinationLevel:destinationOrigin:options:`: as
    /// [`copy_from_buffer`](Self::copy_from_buffer), into the aspect `option` names, one
    /// `vkCmdCopyBufferToImage` region of that aspect alone. Metal takes a plane option of a
    /// format of depth and stencil both, and of a format of the option's aspect alone. Into a
    /// 32-bit float depth it carries [`Texture::FLOAT_DEPTH_UPLOAD_NOTE`].
    ///
    /// Err: as [`copy_from_buffer`](Self::copy_from_buffer)'s; [`Gap::TextureBytes`] for a plane
    /// option of a format without its aspect, for `None` of a format of depth and stencil both,
    /// and for a multisample texture.
    pub fn copy_from_buffer_with_options(
        &mut self,
        source: &Buffer,
        layout: BufferLayout,
        size: [u32; 3],
        destination: &Texture,
        to: TexturePlace,
        option: BlitOption,
    ) -> Result<(), Error> {
        let way = Way::IntoImage(option);
        self.buffer_copy(destination, to, size, source, layout, way)
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:`
    /// `destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`: `size` texels at
    /// `from` in `source` to `destination`, laid out as `layout` says, one
    /// `vkCmdCopyImageToBuffer` a 3D region's image. The options form with no option is this
    /// call. Bytes between the region's rows and images are kept. Sizes of 0 copy as
    /// [`copy_from_texture`](Self::copy_from_texture)'s do. The command buffer keeps the texture
    /// and the buffer until it retires.
    ///
    /// Err: as [`copy_from_buffer`](Self::copy_from_buffer)'s.
    pub fn copy_from_texture_to_buffer(
        &mut self,
        source: &Texture,
        from: TexturePlace,
        size: [u32; 3],
        destination: &Buffer,
        layout: BufferLayout,
    ) -> Result<(), Error> {
        let option = BlitOption::None;
        self.copy_from_texture_to_buffer_with_options(
            source,
            from,
            size,
            destination,
            layout,
            option,
        )
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:`
    /// `destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:`: as
    /// [`copy_from_texture_to_buffer`](Self::copy_from_texture_to_buffer), out of the aspect
    /// `option` names, one `vkCmdCopyImageToBuffer` region of that aspect alone, as
    /// [`copy_from_buffer_with_options`](Self::copy_from_buffer_with_options) takes it.
    ///
    /// Err: as [`copy_from_buffer_with_options`](Self::copy_from_buffer_with_options)'s.
    pub fn copy_from_texture_to_buffer_with_options(
        &mut self,
        source: &Texture,
        from: TexturePlace,
        size: [u32; 3],
        destination: &Buffer,
        layout: BufferLayout,
        option: BlitOption,
    ) -> Result<(), Error> {
        let way = Way::IntoBuffer(option);
        self.buffer_copy(source, from, size, destination, layout, way)
    }

    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`: `size` bytes at
    /// `source_offset` in `source` to `destination_offset` in `destination`, one
    /// `vkCmdCopyBuffer` region between the bytes shaders reach (a Managed buffer's device copy,
    /// where the device keeps one), between a fill's two barriers. A size of 0 copies nothing. The
    /// command buffer keeps both buffers until it retires. Any offsets and any size copy, each byte
    /// of the range and no other, as plain Metal's copy does on macOS though its documentation
    /// asks for multiples of 4 there.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's buffer; [`Refusal::BufferCopyRange`] for
    /// a range past either buffer's end; [`Refusal::BufferCopyOverlap`] for a source and a
    /// destination that share bytes;
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
        let mut backs = Vec::new();
        building.managed_reach(source, &mut backs);
        building.managed_reach(destination, &mut backs);
        building.managed_backs(backs)?;
        building.wrote_managed(destination);
        // A write of a heap buffer leaves the textures over its bytes stale (`crate::alias`).
        if building.aliased() {
            building.places(
                [
                    (source.object().placement(), Access::Read),
                    (destination.object().placement(), Access::Write),
                ]
                .into_iter(),
            )?;
        }
        let overs = [
            (Reached::Buffer(source.span()), Access::Read),
            (Reached::Buffer(destination.span()), Access::Write),
        ];
        building.overs_before(overs, false)?;
        let command = building.ordered_command()?;
        building.overs_after(overs, false);
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

    /// A copy of `size` texels at `at` in `texture` from or to `buffer`, the way `way` says.
    fn buffer_copy(
        &mut self,
        texture: &Texture,
        at: TexturePlace,
        size: [u32; 3],
        buffer: &Buffer,
        layout: BufferLayout,
        way: Way,
    ) -> Result<(), Error> {
        let (into_image, option) = match way {
            Way::IntoImage(option) => (true, option),
            Way::IntoBuffer(option) => (false, option),
        };
        if !texture.is_of(&self.core) || !buffer.is_of(&self.core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        // A texture buffer's texels are its buffer's bytes, which no image copy reaches.
        if texture.texels() {
            return Err(Error::Unsupported(Gap::TextureBytes {
                pixel_format: texture.descriptor().pixel_format,
            }));
        }
        // A Managed buffer's GPU side is its device copy, where it has one.
        let (vk_buffer, base) = buffer.device_range();
        let lies = Lies {
            buffer: vk_buffer,
            base,
            len: buffer.length(),
            layout,
        };
        let side = Side::of(texture).with_option(option)?;
        let Some(copy) = buffer_copy(&side, at, size, lies, into_image)? else {
            return Ok(());
        };
        if copy.slabs.is_none() {
            self.building().count(Metric::OverlappingCopiesRefused);
            let pixel_format = side.descriptor.pixel_format;
            return Err(Error::Unsupported(Gap::OverlappingCopy { pixel_format }));
        }
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
        let mut backs = Vec::new();
        building.managed_reach(buffer, &mut backs);
        building.managed_backs(backs)?;
        match into_image {
            true => building.wrote_texture_managed(texture),
            false => building.wrote_managed(buffer),
        }
        // A heap texture it writes that an alias's write left stale moves from undefined first,
        // and a write of a heap buffer leaves the textures over its bytes stale (`crate::alias`).
        let (into, out) = match into_image {
            true => (Access::Write, Access::Read),
            false => (Access::Read, Access::Write),
        };
        if building.aliased() {
            building.places(
                [
                    (texture.object().placement(), into),
                    (buffer.object().placement(), out),
                ]
                .into_iter(),
            )?;
        }
        let overs = match into_image {
            true => [
                (Reached::Texture(texture.object()), Access::ReadWrite),
                (Reached::Buffer(buffer.span()), Access::Read),
            ],
            false => [
                (Reached::Texture(texture.object()), Access::Read),
                (Reached::Buffer(buffer.span()), Access::ReadWrite),
            ],
        };
        building.overs_before(overs, false)?;
        let command = building.ordered_command()?;
        building.overs_after(overs, false);
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
        let between = [between()];
        let device = &core.device;
        let named = texture.copy_layout(layout);
        // SAFETY: the command buffer records outside any render pass and keeps the image and the
        // buffer until it retires; each copy's level and slice is in its home layout between
        // commands and in `named` between the barriers; each region lies in the image and, by
        // `buffer_copy`, in the buffer's device range, its offset aligned as the command
        // requires, and each slab's bytes in the buffer share none with its texels.
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&earlier)
                    .image_memory_barriers(away),
            );
            for copy in copies {
                for (n, slab) in copy.slabs().enumerate() {
                    if n > 0 {
                        let dependency = vk::DependencyInfo::default().memory_barriers(&between);
                        device.cmd_pipeline_barrier2(command, &dependency);
                        building.count(Metric::Barriers);
                    }
                    for region in slab {
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
        let mut backs = Vec::new();
        building.managed_reach(buffer, &mut backs);
        building.managed_backs(backs)?;
        building.wrote_managed(buffer);
        // A write of a heap buffer leaves the textures over its bytes stale (`crate::alias`).
        if building.aliased() {
            building.places([(buffer.object().placement(), Access::Write)].into_iter())?;
        }
        let overs = [(Reached::Buffer(buffer.span()), Access::Write)];
        building.overs_before(overs, false)?;
        let command = building.ordered_command()?;
        building.overs_after(overs, false);
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

    /// `resetCommandsInBuffer:withRange:`: commands `range` of `buffer` reset, each drawing nothing
    /// with every slot bound to nothing, by copies from the device's template of reset commands of
    /// the buffer's layout (`crate::indirect`): one `vkCmdCopyBuffer`, a region for each run of
    /// as many commands as the template holds, between a fill's two barriers. An empty range
    /// records nothing. The command buffer keeps the buffer's records and the template until it
    /// retires.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's buffer; [`Refusal::CommandRange`] for a
    /// range past the buffer's commands; what making the template returns
    /// ([`Device::new_indirect_command_buffer`](crate::Device::new_indirect_command_buffer)'s);
    /// [`Error::Vulkan`] where the command buffer cannot record.
    pub fn reset_commands_in_buffer(
        &mut self,
        buffer: &IndirectCommandBuffer,
        range: Range<u32>,
    ) -> Result<(), Error> {
        let (core, building) = self.parts();
        if !buffer.is_of(core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        if range.start > range.end || range.end > buffer.size() {
            return Err(Error::Refused(Refusal::CommandRange));
        }
        if range.is_empty() {
            return Ok(());
        }
        let indirect = (core.indirect.as_ref())
            .expect("a device that made an indirect command buffer enabled the extension");
        let template = indirect.template(core, buffer.layout())?;
        let (source, destination, regions) =
            reset_regions(&template, buffer.layout(), buffer.records(), range);
        building.keep_block(buffer.records());
        building.keep_block(template.block());
        let command = building.ordered_command()?;
        let earlier = [memory_barrier(Reach::Commands)];
        let written = [written()];
        let device = &core.device;
        // SAFETY: the command buffer records outside any render pass and keeps both blocks until
        // it retires; each region lies in the template and in the records (`reset_regions`).
        unsafe {
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().memory_barriers(&earlier),
            );
            device.cmd_copy_buffer(command, source, destination, &regions);
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().memory_barriers(&written),
            );
        }
        building.count(Metric::Barriers);
        building.count(Metric::Barriers);
        Ok(())
    }

    /// `optimizeIndirectCommandBuffer:withRange:`: records nothing, the commands already in the
    /// form an execution reads.
    ///
    /// Err: [`Refusal::OtherDevice`] for another device's buffer; [`Refusal::CommandRange`] for a
    /// range past the buffer's commands.
    pub fn optimize_indirect_command_buffer(
        &mut self,
        buffer: &IndirectCommandBuffer,
        range: Range<u32>,
    ) -> Result<(), Error> {
        if !buffer.is_of(&self.core) {
            return Err(Error::Refused(Refusal::OtherDevice));
        }
        if range.start > range.end || range.end > buffer.size() {
            return Err(Error::Refused(Refusal::CommandRange));
        }
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
        // Its levels past the first are written: a heap texture an alias's write left stale
        // moves from undefined first (`crate::alias`).
        if building.aliased() {
            building.places([(texture.object().placement(), Access::ReadWrite)].into_iter())?;
        }
        let overs = [(Reached::Texture(texture.object()), Access::ReadWrite)];
        building.overs_before(overs, false)?;
        let command = building.ordered_command()?;
        if texture.host_reads() {
            building.wrote_host();
        }
        building.overs_after(overs, false);
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
    /// call records nothing for either, nor for an indirect command buffer, whose records the host
    /// never reads.
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
            Resource::IndirectCommandBuffer(commands) if commands.is_of(&self.core) => {}
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
        let (vk_buffer, base) = bytes.device_range();
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
                buffer: vk_buffer,
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
            let mut backs = Vec::new();
            building.managed_reach(buffer, &mut backs);
            building.managed_backs(backs)?;
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
        let Some(Own::Export { handle, .. }) = block.memory.own else {
            return Err(Error::Refused(Refusal::NotExported {
                block: block.memory.id,
            }));
        };
        let (whole, _) = block.memory.range(0);
        building.keep(buffer);
        // The textures over a buffer in the block go back to it first, as before a write of it.
        let overs = [(Reached::Buffer(buffer.span()), Access::Write)];
        building.overs_before(overs, false)?;
        let command = building.release_command()?;
        building.overs_after(overs, false);
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
    /// encodes no read of its planes. For a Direct3D 11 import (`Device::import_d3d11_surface`,
    /// on Windows) the call records nothing: each submission that uses its planes releases them
    /// itself
    /// ([`Metrics::d3d11_plane_submissions`](crate::Metrics::d3d11_plane_submissions)).
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
        // Each submission that uses a Direct3D 11 import's planes releases them itself.
        if surface.handle().is_none() {
            return Ok(());
        }
        for texture in surface.planes() {
            building.keep_texture(texture);
        }
        let buffer = surface.made_buffer();
        let overs =
            || (buffer.iter()).map(|buffer| (Reached::Buffer(buffer.span()), Access::Write));
        if let Some(buffer) = &buffer {
            building.keep(buffer);
            // The textures over the buffer go back to it first, as before a write of it.
            building.overs_before(overs(), false)?;
        }
        let command = building.release_command()?;
        if buffer.is_some() {
            building.overs_after(overs(), false);
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

/// The barrier between two slabs of a copy ([`Slabs`]): an execution dependency, every hazard
/// between them a write after a read, as no slab reads a byte an earlier one wrote.
fn between() -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
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

/// One texture of a copy: its descriptor, its image's format, the unit its copies move, where it
/// starts in its image (a view's first level and array layer), in place over a buffer where its
/// texels lie in the buffer's bytes, and where its memory lies that other resources may share
/// (`TextureObject::bound`).
pub(crate) struct Side<'a> {
    pub(crate) descriptor: &'a TextureDescriptor,
    pub(crate) format: vk::Format,
    pub(crate) unit: Option<CopyUnit>,
    pub(crate) base: [u32; 2],
    pub(crate) rows: Option<Rows>,
    pub(crate) bound: Option<Span>,
}

impl<'a> Side<'a> {
    fn of(texture: &'a Texture) -> Side<'a> {
        Side {
            descriptor: texture.descriptor(),
            format: texture.format(),
            unit: texture.unit(),
            base: texture.base(),
            rows: texture.in_place(),
            bound: texture.bound(),
        }
    }

    /// The side as a copy of `option` moves it: as it is for `None`, and for a plane option the
    /// aspect it names, where the texture's pixel format has that aspect (Metal takes the option
    /// of a format of that aspect alone too, as `Depth16Unorm` and `Stencil8`).
    ///
    /// Err: [`Gap::TextureBytes`] for a plane option of a format without its aspect.
    pub(crate) fn with_option(self, option: BlitOption) -> Result<Self, Error> {
        let aspect = match option {
            BlitOption::None => return Ok(self),
            BlitOption::DepthFromDepthStencil => vk::ImageAspectFlags::DEPTH,
            BlitOption::StencilFromDepthStencil => vk::ImageAspectFlags::STENCIL,
        };
        let pixel_format = self.descriptor.pixel_format;
        let aspects = formats::aspects(pixel_format).map(vk::ImageAspectFlags::from);
        if !aspects.is_some_and(|aspects| aspects.contains(aspect)) {
            return Err(Error::Unsupported(Gap::TextureBytes { pixel_format }));
        }
        let samples = self.descriptor.sample_count;
        let unit = copy_unit(pixel_format, self.format, aspect, 0, samples);
        Ok(Side { unit, ..self })
    }
}

/// A copy between two textures, checked: the region in each, the aspect it copies, whether the
/// two regions lie in one subresource of one image, which the copy takes in the general layout,
/// and the slabs it copies in: `None` where the two share bytes that no copy command orders
/// (`texture_copy`), which [`Gap::OverlappingCopy`] names.
pub(crate) struct ImageCopy {
    pub(crate) from: Placed,
    pub(crate) to: Placed,
    pub(crate) aspect: vk::ImageAspectFlags,
    pub(crate) shared: bool,
    pub(crate) slabs: Option<Slabs>,
}

impl ImageCopy {
    /// Its copy commands' regions, in the order they run: one a slab.
    pub(crate) fn regions(&self) -> impl Iterator<Item = vk::ImageCopy> + '_ {
        self.slabs.into_iter().flat_map(move |slabs| {
            slabs.of(self.from).map(move |(start, from)| {
                let mut to = self.to;
                to.origin[slabs.axis] += start;
                vk::ImageCopy {
                    src_subresource: from.layers(self.aspect),
                    src_offset: from.offset(),
                    dst_subresource: to.layers(self.aspect),
                    dst_offset: to.offset(),
                    extent: from.extent(),
                }
            })
        })
    }
}

/// The slabs a copy between textures runs as, one copy command each: the region cut across
/// `axis` into slabs `thickness` texels thick (the last what is left), from the far end of the
/// axis first where `reversed`. Regions that share no byte copy as one slab. Regions that do are
/// cut ([`cut`]) no thicker than the least shift between them along an axis every shift points
/// the same way along, so that each slab's two regions share no byte, on the axis of fewest
/// slabs; and the slab at the end the shifts point to goes first, so that no slab reads a byte
/// an earlier one wrote. The copy so gives what reading its source whole before writing gives,
/// as `memmove` does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slabs {
    pub(crate) axis: usize,
    pub(crate) thickness: u32,
    pub(crate) reversed: bool,
}

impl Slabs {
    /// `placed` cut into its slabs, in the order they run: each one's first texel along the axis
    /// from the region's, and its region.
    fn of(self, placed: Placed) -> impl Iterator<Item = (u32, Placed)> {
        let Slabs {
            axis,
            thickness,
            reversed,
        } = self;
        let extent = placed.size[axis];
        let count = extent.div_ceil(thickness);
        (0..count).map(move |n| {
            let start = thickness
                * match reversed {
                    true => count - 1 - n,
                    false => n,
                };
            let mut slab = placed;
            slab.origin[axis] += start;
            slab.size[axis] = thickness.min(extent - start);
            (start, slab)
        })
    }
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

/// Bytes a copy reaches: the `VkBuffer` shaders reach them in, their offset there, and their
/// length. One buffer of a copy between buffers, or a texture's memory (`TextureObject::bound`).
#[derive(Clone, Copy)]
pub(crate) struct Span {
    pub(crate) buffer: vk::Buffer,
    pub(crate) base: u64,
    pub(crate) len: u64,
}

impl Span {
    /// Whether the two share a byte.
    fn meets(self, other: Span) -> bool {
        self.buffer == other.buffer
            && self.base < other.base + other.len
            && other.base < self.base + self.len
    }
}

/// Checks a copy of `size` bytes from `source_offset` in `source` to `destination_offset` in
/// `destination`: the region of `vkCmdCopyBuffer` to record, `None` for a size of 0.
///
/// Err: [`Refusal::BufferCopyRange`]; [`Refusal::BufferCopyOverlap`].
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

/// A buffer's side of a copy: the `VkBuffer` shaders reach it through, its offset there, its
/// length, and where the region lies in it.
#[derive(Clone, Copy)]
pub(crate) struct Lies {
    pub(crate) buffer: vk::Buffer,
    pub(crate) base: u64,
    pub(crate) len: u64,
    pub(crate) layout: BufferLayout,
}

/// A copy between a buffer and a texture, checked: the texture's region, its copy unit, where
/// each image of it lies in the `VkBuffer` shaders reach the buffer through, and the slabs it
/// copies in: `None` where the buffer's bytes meet the texture's where no copy command orders
/// them (`buffer_copy`), which [`Gap::OverlappingCopy`] names.
pub(crate) struct BufferCopy {
    pub(crate) placed: Placed,
    pub(crate) unit: CopyUnit,
    /// The first image's first byte in the block.
    pub(crate) offset: u64,
    /// A row's stride in texels; 0 for a region one row high.
    pub(crate) row_length: u32,
    pub(crate) bytes_per_image: u64,
    pub(crate) slabs: Option<Slabs>,
}

impl BufferCopy {
    /// Its copy commands' regions, slab by slab in the order the slabs run: one region an image
    /// of the slab, each with its own offset (a stride of images that is no whole number of rows
    /// needs no row count). A slab's bytes start as far on as its first texel: whole texels across,
    /// rows down, as a texture in place over the buffer, of one texel an element, is cut.
    pub(crate) fn slabs(
        &self,
    ) -> impl Iterator<Item = impl Iterator<Item = vk::BufferImageCopy> + '_> + '_ {
        let texel = u64::from(self.unit.bytes);
        let step = [
            texel,
            u64::from(self.row_length) * texel,
            self.bytes_per_image,
        ];
        self.slabs.into_iter().flat_map(move |slabs| {
            slabs.of(self.placed).map(move |(start, placed)| {
                let offset = self.offset + u64::from(start) * step[slabs.axis];
                let [x, y, z] = placed.origin;
                let [width, height, depth] = placed.size;
                (0..depth).map(move |image| vk::BufferImageCopy {
                    buffer_offset: offset + u64::from(image) * self.bytes_per_image,
                    buffer_row_length: self.row_length,
                    buffer_image_height: 0,
                    image_subresource: placed.layers(self.unit.aspect),
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
            })
        })
    }
}

/// Checks a copy of `size` texels at `at` in `texture` and the bytes `lies` names in a buffer,
/// `into_image` for a copy into the texture. `None` where it copies nothing; slabs of `None`
/// where its bytes meet the texture's in no order a copy command keeps.
///
/// Metal's validation stops an offset or a row stride of a format of single texels that is no
/// whole number of its texels, rows or images closer than the region's, and a region past the
/// buffer's end ([`Refusal::BufferCopyLayout`]). A copy command in the floor further takes an
/// offset of whole texel blocks, a multiple of 4 for a depth or stencil aspect, for each image,
/// and a row stride of whole blocks ([`Gap::BufferCopyAlignment`]).
///
/// Err: [`Gap::TextureBytes`]; [`Refusal::TextureRegion`];
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
    let Lies {
        buffer,
        base,
        len,
        layout,
    } = lies;
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
    let Some(span) = span.filter(|span| span.checked_add(offset).is_some_and(|end| end <= len))
    else {
        return Err(refused());
    };
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
    // The region's bytes may be a texture's own: in place over the buffer, or placed in a heap
    // with it. No command's source and destination share a byte
    // (`VUID-vkCmdCopyBufferToImage-pRegions-00173`, `VUID-vkCmdCopyImageToBuffer-pRegions-00184`):
    // rows of a texture in place cut into slabs as two textures over one buffer do (`meeting`),
    // the region's rows `bytes_per_row` apart; bytes of an image the device lays out as it
    // chooses, none.
    let region = Span {
        buffer,
        base: start,
        len: span,
    };
    let slabs = match (texture.rows, texture.bound) {
        (Some(rows), _) if rows.buffer == buffer => {
            let image = first(rows, &placed, bytes);
            let pitch = bytes_per_row.max(row);
            let bytes_at = Rows {
                offset: start,
                pitch,
                ..rows
            };
            let (from, to) = match into_image {
                true => (bytes_at, image),
                false => (image, bytes_at),
            };
            cut(
                meeting(from, to, placed.size, bytes),
                [u128::from(bytes), 1, 1],
                placed.size,
            )
        }
        (_, Some(bound)) if bound.meets(region) => None,
        _ => cut(None, [1; 3], placed.size),
    };
    let slabs = match slabs {
        Some(None) => return Ok(None),
        slabs => slabs.flatten(),
    };
    Ok(Some(BufferCopy {
        placed,
        unit,
        offset: start,
        row_length,
        bytes_per_image,
        slabs,
    }))
}

/// Checks a copy of `size` texels from `from` in `source` to `to` in `destination`, `same` where
/// they are one texture. `None` where it copies nothing; slabs of `None` where the two regions'
/// bytes meet in no order a copy command keeps.
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
    // What one region copies: the unit's aspect, in its blocks. A format of depth and stencil
    // both has no unit, no buffer copy taking its two aspects in one region; one sample of it
    // copies both in one region of texels, as Metal copies its pixels whole
    // (`VkImageSubresourceLayers::aspectMask` selects "the color, depth and/or stencil aspects to
    // be copied"). No unit else: a multisample texture.
    let both = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
    let (aspect, block) = match source.unit {
        Some(unit) => (unit.aspect, unit.block),
        None if s.sample_count == 1
            && formats::aspects(s.pixel_format).map(vk::ImageAspectFlags::from) == Some(both) =>
        {
            (both, [1, 1])
        }
        None => return Err(gap()),
    };
    // One pixel format in two Vulkan formats (chosen by the textures' usages) holds its texels in
    // two layouts.
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
        place(s, source.base, block, region(from), from.level, from.slice)?,
        place(d, destination.base, block, region(to), to.level, to.slice)?,
    );
    let (Some(from), Some(to)) = placed else {
        return Ok(None);
    };
    // One command's one extent copies both regions, and stops at an image's edge only where
    // that region does ("Copy regions for each image must be aligned to a multiple of the texel
    // block extent in each dimension, except at the edges of the image, where region extents
    // must match the edge of the image"): whole blocks that end inside one level's last block
    // past its edge and not the other's copy with no one extent, though plain Metal copies the
    // block whole.
    if from.size != to.size {
        return Err(gap());
    }
    // One subresource of one image, the textures one or a texture and its view, copies in the
    // general layout (`VUID-vkCmdCopyImage-srcImage-09460`); two textures in place over one
    // buffer may share bytes too. No command's source and destination share a byte
    // (`-pRegions-00124`). The same texels copy nothing, every one already its own. Metal leaves
    // a copy whose regions overlap undefined (Apple's `copy(from:...to:...)` page: "unexpected
    // behavior"); an Apple M3 Max reads the source whole before it writes, which slabs give
    // (`Slabs`).
    let shared = same && from.level == to.level && from.layer == to.layer;
    let slabs = match (source.rows, destination.rows, source.unit) {
        _ if shared => {
            let apart = (0..3).any(|axis| {
                let (a, b, extent) = (from.origin[axis], to.origin[axis], from.size[axis]);
                a + extent <= b || b + extent <= a
            });
            let shift = std::array::from_fn(|axis| {
                i128::from(to.origin[axis]) - i128::from(from.origin[axis])
            });
            cut((!apart).then_some(shift), [1; 3], from.size)
        }
        // A texture in place is of one color texel an element (`Buffer::new_texture`), a unit.
        (Some(a), Some(b), Some(unit)) if a.buffer == b.buffer => {
            let texel = u64::from(unit.bytes);
            let rows = meeting(
                first(a, &from, texel),
                first(b, &to, texel),
                from.size,
                texel,
            );
            cut(rows, [u128::from(texel), 1, 1], from.size)
        }
        // Memory one of them shares with the other where the device lays its texels out as it
        // chooses (a texture placed in a heap): no order.
        _ if !same && (source.bound.zip(destination.bound)).is_some_and(|(a, b)| a.meets(b)) => {
            None
        }
        _ => cut(None, [1; 3], from.size),
    };
    let slabs = match slabs {
        Some(None) => return Ok(None),
        slabs => slabs.flatten(),
    };
    Ok(Some(ImageCopy {
        from,
        to,
        aspect,
        shared,
        slabs,
    }))
}

/// The slabs of a copy of `size` texels from `shifts`: for each destination texel, or row, on
/// bytes of a source texel, or row, how far that source lies from it in the region (the source's
/// place less the destination's), along each axis in `units`: texels, or bytes along x between
/// two textures over one buffer. Regions of no shift share no byte, and copy as one slab. Shifts
/// all of 0 make every destination texel its own source: `Some(None)`, nothing to copy. Otherwise
/// an axis that every shift moves along, all one way, is cut no thicker than the least shift
/// along it in whole texels, so that no slab's source and destination share a byte; the slab at
/// the end the shifts point to goes first, so that no slab reads a byte an earlier one wrote (an
/// earlier slab's destination bytes that are a later one's source would be a shift the other
/// way). The cut is along the axis of fewest slabs, the first of them; `None` where no axis cuts.
fn cut(
    shifts: impl IntoIterator<Item = [i128; 3]>,
    units: [u128; 3],
    size: [u32; 3],
) -> Option<Option<Slabs>> {
    // Along each axis: the least shift's size, and whether a shift points toward lower and one
    // toward higher coordinates.
    let mut along = [(u128::MAX, false, false); 3];
    let (mut any, mut own) = (false, true);
    for shift in shifts {
        any = true;
        own &= shift == [0; 3];
        for ((least, lower, higher), shift) in along.iter_mut().zip(shift) {
            *least = (*least).min(shift.unsigned_abs());
            *lower |= shift < 0;
            *higher |= shift > 0;
        }
    }
    if !any {
        return Some(Some(Slabs {
            axis: 0,
            thickness: size[0],
            reversed: false,
        }));
    }
    if own {
        return Some(None);
    }
    (0..3)
        .filter_map(|axis| {
            let (least, lower, higher) = along[axis];
            let thickness = u32::try_from(least / units[axis]).ok()?;
            (thickness > 0 && !(lower && higher)).then_some(Slabs {
                axis,
                thickness,
                reversed: higher,
            })
        })
        .min_by_key(|slabs| size[slabs.axis].div_ceil(slabs.thickness))
        .map(Some)
}

/// Of a texture whose texels of `texel` bytes lie in `rows`, where the first texel of `placed`
/// lies.
fn first(rows: Rows, placed: &Placed, texel: u64) -> Rows {
    Rows {
        offset: rows.offset
            + u64::from(placed.origin[1]) * rows.pitch
            + u64::from(placed.origin[0]) * texel,
        ..rows
    }
}

/// The shifts ([`cut`]) of a copy of `size` texels of `texel` bytes between two regions of rows on
/// one buffer (two textures in place over it, or a region of the buffer and a texture in place
/// over it), from the region whose first texel is at `from` to the one at `to`: for each
/// destination row on bytes of a source row, the destination row's first byte less the source
/// row's along x (in bytes, the source texel on a destination texel's bytes less that texel, in
/// the region), and the source row less the destination row along y. A row is no longer than its
/// pitch, so a source row meets at most the two destination rows whose starts lie nearest before
/// and after its own. Regions whose spans of bytes do not meet meet in no row.
fn meeting(from: Rows, to: Rows, size: [u32; 3], texel: u64) -> impl Iterator<Item = [i128; 3]> {
    let row = i128::from(size[0]) * i128::from(texel);
    let rows = i128::from(size[1]);
    let (first, pitch) = (i128::from(from.offset), i128::from(from.pitch));
    let (onto, onto_pitch) = (i128::from(to.offset), i128::from(to.pitch));
    let spans =
        first < onto + (rows - 1) * onto_pitch + row && onto < first + (rows - 1) * pitch + row;
    (0..if spans { rows } else { 0 }).flat_map(move |j| {
        let start = first + j * pitch;
        let k = (start - onto).div_euclid(onto_pitch);
        [k, k + 1].into_iter().filter_map(move |k| {
            let shift = onto + k * onto_pitch - start;
            ((0..rows).contains(&k) && shift.abs() < row).then_some([shift, j - k, 0])
        })
    })
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
