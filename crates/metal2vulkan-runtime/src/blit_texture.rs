//! The blit encoder's texture commands: copies between buffers and textures and between textures,
//! `generate_mipmaps`, `synchronize_texture`, and the `optimize_contents_*` hints.
//!
//! - **Rules.** Metal's checks are `copy_plan.rs`'s and `blit_plan.rs`'s (measured on an Apple M3
//!   Max and M4). A size of 0 on an axis a texture of the copy lacks (depth on all but 3D textures,
//!   height on 1D textures and texture buffers) copies one texel there, and a copy with another
//!   axis 0, or of no slices or levels, copies nothing while the encoder's other commands run, as
//!   plain Metal does (its validation stops each). A copy to or from a memoryless texture, and its
//!   mipmaps, are refused (`StorageMode`), as Metal's validation refuses them; the
//!   `optimize_contents_*` hints take one, as Metal's do (a Metal probe on an M4).
//! - **Recording.** Each copy is one command of the encoder with what it reads and writes: a
//!   texture's uses name its image's levels and layers (every aspect), so the encoder's order and
//!   the commit's barriers cover views and their textures as one resource. A buffer layout
//!   Vulkan cannot state (`blit_plan.rs`) goes through a scratch buffer of the encoder, two
//!   commands.
//! - **Depth.** A copy from a buffer into 32-bit float depth copies the bytes as Metal holds
//!   them, any `float`, on every device, as `Texture::replace_region` does: Vulkan leaves a copied
//!   value outside `[0, 1]` undefined without `VK_EXT_depth_range_unrestricted`
//!   (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), and such a copy into a depth image announces
//!   `restricted-depth-upload` where the device's driver is not measured to keep such a value
//!   (`DriverFact::DepthImageKeepsValuesOutsideUnitRange`). A texture with a depth twin takes a
//!   copy in both its images, its texels kept in colour; a copy out of it into a depth texture
//!   without a twin reads its twin, and one into it from such a texture writes its `R32_SFLOAT`
//!   image through a scratch buffer of the encoder.
//! - **Textures over buffers**. A copy that names a texture over a buffer runs on the buffer's
//!   bytes at Metal's layout, whatever else holds the texels (an aliasing image, or a shadow image
//!   the next encoder that uses it copies in from the bytes): one buffer-to-buffer copy of its
//!   texel rows, never of the bytes between them, from or into a buffer, another texture over a
//!   buffer, or a texture's image. Where a source row and a destination row of one buffer share
//!   bytes, the rows go through a scratch buffer of the encoder, as Metal moves them (the whole
//!   source read before any of it is written).
//! - **Depth and stencil with colour**. Metal copies the bytes between a depth or stencil format,
//!   or the stencil view `X32Stencil8`, and a colour format of its unit, and between stencil
//!   formats (`copy_plan.rs`). Vulkan copies a depth or stencil aspect only into the same aspect of
//!   the same format, so each image of the destination that no image of the source copies into
//!   takes the regions through a scratch buffer of the encoder, out of the image the source's
//!   texels are read from: two copies instead of one.
//! - **Overlap**. A region copied onto itself keeps its texels whatever order Metal copies them in,
//!   and nothing is recorded for it. A copy between other overlapping regions of one texture's
//!   texels (one image, through any views: the regions in the image's own levels and layers and in
//!   one aspect) goes through a scratch buffer, the whole source read before any of it is written,
//!   which is `memmove`'s answer: Metal's for regions up to a few hundred texels, where past that
//!   its copy races, some texels reading what the copy already wrote, differently from run to run
//!   (probes). Announced once (`overlapping-texture-copy-as-memmove`). A
//!   multisampled texture, which Vulkan copies into no buffer, stays refused
//!   `NotImplemented(OverlappingTextureCopies)`.
//! - **PVRTC**. A copy between a buffer and a PVRTC texture with a nonzero `bytesPerRow` moves its
//!   blocks row by row, as Metal's does (and the runtime's carrier holds them, `pvrtc.rs`), with
//!   `MTLBlitOptionRowLinearPVRTC` or without it, which plain Metal copies as it copies with it
//!   (its validation stops it). With a `bytesPerRow` of 0, with the option or without it, the
//!   blocks lie in plain Metal's block order, the region's own Morton order
//!   (`blit_plan::block_order_position`; its validation stops that too), a region off its blocks
//!   that of the blocks it is rounded to. Another format ignores the option, as plain Metal does.
//! - **Not yet:** a depth or stencil view of another format of its texture, other than the
//!   stencil view `X32Stencil8` of `Depth32FloatStencil8`, is refused `NotImplemented`.

use super::blit::{reads, reads_by, writes, writes_by, BlitCommandEncoder, TRANSFER};
use super::blit_plan::{
    copied_rows, plan_block_order_copy, plan_buffer_copy, plan_levels_copy, plan_region_copy,
    plan_rows_copy, plan_whole_copy, texel_layout, texel_rows, BufferCopyPlan, BufferImageRegion,
    BufferLayout, BufferRange, CopyPair, ImageCopyRegion, RowsCopy,
};
use super::compressed_blit::{rule_of, rules_of};
use super::copy_plan::{
    check_region, colour_unit, copy_unit, depth_bytes, level_extent, shape, BlitOption,
    CopyProblem, CopyRefusal, CopyUnit, ImageRegion, TextureRegion,
};
use super::deferred::Deferred;
use super::degradation::Degradation;
use super::error::{internal, invalid, Error, Problem, RuntimeGap};
use super::fresh_unwritten::FreshUnwritten;
use super::image_plan::copies_between;
use super::mipmap::record_blit;
use super::mipmap_plan::{leaves_as_is, not_implemented, plan_mipmaps, MipmapPlan, MipmapRefusal};
use super::nan_clear;
use super::pvrtc::{image_block_extent, image_buffer_region, image_region, is_pvrtc};
use super::replay::{BlitCall, BufferToTexture, TextureToBuffer, TextureToTexture};
use super::row_copies::RowCopies;
use super::sample_copy;
use super::sample_interleave;
use super::sample_planes;
use super::shadow::Covered;
use super::sparse::emulated::{Scrub, Written};
use super::texture::{buffer_image_copy, texture_aspects, texture_region, twin_copy, Texture};
use super::tracker::Access;
use super::two_planes;
use super::values::{Origin, Region, ResourceOptions, Size, StorageMode};
use super::Buffer;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::{aspects, is_depth_stencil};
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::{Format, ImageAspectFlags};
use std::ops::Range;
use std::sync::Arc;

fn copy_error(call: &'static str, refusal: CopyRefusal) -> Error {
    match refusal {
        CopyRefusal::Invalid(problem) => invalid(call, Problem::Copy(problem)),
        CopyRefusal::Unsupported(unsupported) => Error::Unsupported(unsupported),
    }
}

fn read() -> Access {
    Access::new(super::blit::TRANSFER, vk::AccessFlags2::TRANSFER_READ)
}

fn write() -> Access {
    Access::new(super::blit::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE)
}

/// The levels and layers of the image a region of `region`'s level and slices covers.
fn subresources(region: &ImageRegion) -> (Range<u32>, Range<u32>) {
    (
        region.level..region.level + 1,
        region.layer..region.layer + region.layers,
    )
}

/// The size a blit copies of `size` between textures of `types`, as plain Metal copies it: an axis
/// of 0 that a texture of the copy lacks, depth on all but 3D textures and height on 1D textures
/// and texture buffers, copies one texel there; `None` for a copy with another axis 0, which copies
/// nothing.
fn copy_size(size: Size, types: &[TextureType]) -> Option<Size> {
    use TextureType::{TextureBuffer, Type1D, Type1DArray, Type3D};
    let any = |lacks: fn(TextureType) -> bool| types.iter().any(|&t| lacks(t));
    let mut size = size;
    if size.height == 0 && any(|t| matches!(t, Type1D | Type1DArray | TextureBuffer)) {
        size.height = 1;
    }
    if size.depth == 0 && any(|t| !matches!(t, Type3D)) {
        size.depth = 1;
    }
    (size.width != 0 && size.height != 0 && size.depth != 0).then_some(size)
}

/// `level` and `slice`, and `origin` and `size`, as a region of a texture; refused as Metal
/// refuses a region past the texture when a value is past any texture.
fn region(
    call: &'static str,
    origin: Origin,
    size: Size,
    level: u64,
    slice: u64,
) -> Result<super::copy_plan::TextureRegion, Error> {
    texture_region(&Region { origin, size }, level, slice)
        .ok_or_else(|| invalid(call, Problem::Copy(CopyProblem::Region)))
}

/// Refuse a depth or stencil view of another format of its texture, except the stencil view of
/// a combined one (`X32Stencil8` of `Depth32FloatStencil8`, which copies its stencil, probes):
/// Metal's copies through another are not measured.
fn check_depth_view(texture: &Texture) -> Result<(), Error> {
    let pf = texture.record().pixel_format;
    // A colour texture is no depth view, whatever holds it: a texture buffer has no image.
    if !is_depth_stencil(pf) {
        return Ok(());
    }
    let of = texture.image_plan()?.pixel_format;
    let stencil_view = pf == PixelFormat::X32Stencil8 && of == PixelFormat::Depth32FloatStencil8;
    if pf != of && !stencil_view {
        return Err(Error::NotImplemented(RuntimeGap::Textures));
    }
    Ok(())
}

/// The aspect a blit between a buffer and the multisampled `texture` with `option` moves: a
/// depth or stencil format's copy unit's ([`unit`], refused as the single-sample copy refuses
/// it), a colour format's colour.
fn multisampled_aspect(
    call: &'static str,
    texture: &Texture,
    option: BlitOption,
) -> Result<ImageAspectFlags, Error> {
    match is_depth_stencil(texture.record().pixel_format) {
        true => Ok(unit(call, texture, option)?.aspect),
        false => Ok(ImageAspectFlags::COLOR),
    }
}

/// `region` of the multisampled `texture` (in its own levels and slices) in its image's: Metal's
/// checks of a region past the texture's levels, slices or the level's extent (`Invalid`, as the
/// single-sample copy's `check_region`; a multisampled texture has no blocks), then the view's
/// first level and layer added, as the per-sample copy (`sample_interleave.rs`) and the
/// texture's uses name them.
fn multisampled_region(
    call: &'static str,
    texture: &Texture,
    region: TextureRegion,
) -> Result<TextureRegion, Error> {
    let record = texture.record();
    let refused = || invalid(call, Problem::Copy(CopyProblem::Region));
    if region.level >= record.level_count || region.slice >= record.layer_count {
        return Err(refused());
    }
    let extent = level_extent(record, region.level);
    let axes = || region.origin.iter().zip(&region.size).zip(&extent);
    if axes().any(|((&o, &s), &e)| u64::from(o) + u64::from(s) > u64::from(e)) {
        return Err(refused());
    }
    Ok(TextureRegion {
        level: record.base_level + region.level,
        slice: record.base_layer + region.slice,
        ..region
    })
}

/// The box `region` covers (of a multisampled texture's image, in its levels and layers,
/// [`multisampled_region`]), every sample of it.
fn multisampled_texels(region: &TextureRegion) -> Written {
    Written {
        level: region.level,
        layers: region.slice..region.slice + 1,
        origin: region.origin,
        size: region.size,
    }
}

/// The box a multisampled copy of `region` into `texture`'s `aspect` writes, every sample of it.
fn multisampled_box(
    texture: &Texture,
    aspect: ImageAspectFlags,
    region: &TextureRegion,
) -> Result<Covered, Error> {
    let [x, y, z] = region.origin;
    let [width, height, depth] = region.size;
    Ok(Covered {
        image: texture.raw_image()?,
        subresource: vk::ImageSubresourceLayers {
            aspect_mask: aspect.into(),
            mip_level: region.level,
            base_array_layer: region.slice,
            layer_count: 1,
        },
        offset: vk::Offset3D {
            x: x as i32,
            y: y as i32,
            z: z as i32,
        },
        extent: vk::Extent3D {
            width,
            height,
            depth,
        },
    })
}

/// The copy unit of `texture` copied with `option`.
pub(super) fn unit(
    call: &'static str,
    texture: &Texture,
    option: BlitOption,
) -> Result<CopyUnit, Error> {
    // A texture over a buffer holds a colour format (`linear_plan.rs`): its copy unit is its
    // texel, whatever holds it.
    if texture.over_buffer() {
        // A depth or stencil texture over a surface's plane, or a view of one, moves the aspect
        // the plane holds; the stencil of `Depth32Float_Stencil8`, which is not in the plane
        // (that texture's with `StencilFromDepthStencil`, or its `X32_Stencil8` view's), is the
        // runtime's gap (`linear_plan.rs`).
        let pf = texture.record().pixel_format;
        if let Some(held) = texture.surface_plane_unit() {
            let own = aspects(pf).unwrap_or(ImageAspectFlags::empty());
            let stencil = !own.contains(ImageAspectFlags::DEPTH)
                || (own.contains(ImageAspectFlags::STENCIL)
                    && option == BlitOption::StencilFromDepthStencil);
            if stencil && held.aspect == ImageAspectFlags::DEPTH {
                return Err(Error::NotImplemented(RuntimeGap::SurfaceDepthStencil {
                    pixel_format: pf,
                }));
            }
            return Ok(held);
        }
        // A two-plane extended-range texture over a surface's planes: plane 0's color words,
        // which Metal's blits with a buffer move (`texture/surface_two_planes.rs`).
        if two_planes::holds_planes(texture.record().pixel_format) {
            return Ok(two_planes::UNIT);
        }
        return colour_unit(texture.record().pixel_format)
            .map_err(|refusal| copy_error(call, refusal));
    }
    check_depth_view(texture)?;
    copy_unit(
        texture.record().pixel_format,
        &texture.image_plan()?.mapping,
        option,
    )
    .map_err(|refusal| copy_error(call, refusal))
}

/// The plan of a copy between `buffer`, laid out as `layout`, and `region` of `texture` in
/// `unit`s (`into_texture` from the buffer, which reads it to its copy reach). A PVRTC texture's
/// blocks lie row by row at a nonzero `bytesPerRow`, with `MTLBlitOptionRowLinearPVRTC` or
/// without it (plain Metal copies a PVRTC texture without the option, which its validation
/// stops, as with it: our probe on an M4 under macOS 27.0.1, both directions, 4- and
/// 2-bit formats, padded rows and a region; the reims-vgpu consumer's run on an M3 Max under
/// macOS 26.7), and in plain Metal's block order at a `bytesPerRow` of 0, with the option or
/// without it (`plan_block_order_copy`; its validation stops both), a region off its blocks as the
/// blocks it is rounded to.
fn plan_copy(
    call: &'static str,
    texture: &Texture,
    region: &TextureRegion,
    unit: &CopyUnit,
    layout: &BufferLayout,
    buffer: &Buffer,
    into_texture: bool,
) -> Result<BufferCopyPlan, Error> {
    let record = texture.record();
    let buffer_length = if into_texture {
        buffer.copy_reach()
    } else {
        buffer.length()
    };
    let base = buffer.start();
    if is_pvrtc(texture.pixel_format()) && layout.bytes_per_row == 0 {
        return plan_block_order_copy(record, region, unit, layout, buffer_length, base)
            .map_err(|problem| invalid(call, Problem::Copy(problem)));
    }
    plan_buffer_copy(
        record,
        region,
        unit,
        layout,
        buffer_length,
        base,
        into_texture,
    )
    .map_err(|problem| invalid(call, Problem::Copy(problem)))
}

/// The Vulkan region of `region`.
fn buffer_image_region(region: &BufferImageRegion) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: region.buffer_offset,
        buffer_row_length: region.row_length,
        buffer_image_height: region.image_height,
        ..buffer_image_copy(&region.image)
    }
}

fn buffer_copy(range: &BufferRange) -> vk::BufferCopy {
    vk::BufferCopy {
        src_offset: range.src,
        dst_offset: range.dst,
        size: range.size,
    }
}

/// `region` as one `VkImageCopy`, whose one extent is the source's: regions of one size only
/// (others go through a buffer, `texture_copies_of_bytes`).
fn image_copy(region: &ImageCopyRegion) -> vk::ImageCopy {
    debug_assert_eq!(region.src.size, region.dst.size, "one extent, two sizes");
    let layers = |r: &ImageRegion| vk::ImageSubresourceLayers {
        aspect_mask: r.aspect.into(),
        mip_level: r.level,
        base_array_layer: r.layer,
        layer_count: r.layers,
    };
    let offset = |[x, y, z]: [u32; 3]| vk::Offset3D {
        x: x as i32,
        y: y as i32,
        z: z as i32,
    };
    let [width, height, depth] = region.src.size;
    vk::ImageCopy {
        src_subresource: layers(&region.src),
        src_offset: offset(region.src.origin),
        dst_subresource: layers(&region.dst),
        dst_offset: offset(region.dst.origin),
        extent: vk::Extent3D {
            width,
            height,
            depth,
        },
    }
}

/// Whether two regions of one texture's texels share texels: of one level, layer and texel,
/// unless one lies in a depth/stencil image's depth and the other in its stencil, which Vulkan
/// keeps apart. Any other two aspects are the same texels (a texture's colour image and its
/// depth twin's depth hold one texture's).
fn regions_meet(a: &ImageRegion, b: &ImageRegion) -> bool {
    let meet = |a0: u32, an: u32, b0: u32, bn: u32| {
        u64::from(a0) < u64::from(b0) + u64::from(bn)
            && u64::from(b0) < u64::from(a0) + u64::from(an)
    };
    let depth_stencil = ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL;
    let apart = depth_stencil.contains(a.aspect)
        && depth_stencil.contains(b.aspect)
        && !a.aspect.intersects(b.aspect);
    !apart
        && a.level == b.level
        && meet(a.layer, a.layers, b.layer, b.layers)
        && (0..3).all(|i| meet(a.origin[i], a.size[i], b.origin[i], b.size[i]))
}

impl BlitCommandEncoder {
    /// Refuse `texture` of another device; note it for its heap shadow (`BlitCommandEncoder::see`).
    fn check_texture(&mut self, call: &'static str, texture: &Texture) -> Result<(), Error> {
        self.core().check_device(call, texture.device())?;
        self.see(texture);
        Ok(())
    }

    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:
    /// destinationSlice:destinationLevel:destinationOrigin:options:`: `source_size` texels of
    /// the buffer, rows `source_bytes_per_row` apart and images `source_bytes_per_image` apart
    /// (read only for more than one image), into the texture's
    /// `destination_level` and `destination_slice` at `destination_origin`. A combined
    /// depth/stencil texture takes one aspect, which `options` names, and its depth with
    /// [`BlitOption::None`]; another texture ignores the option, as plain Metal does (its
    /// validation stops both). A compressed region off its blocks copies the blocks it starts in,
    /// as plain Metal does. A PVRTC texture's blocks lie in the buffer row by row with a nonzero
    /// `source_bytes_per_row`, with [`BlitOption::RowLinearPVRTC`] or without it, and with a
    /// `source_bytes_per_row` of 0 in the region's own Morton order, the x bit lowest, a position
    /// no block takes neither read nor written, a region off its blocks that of the blocks it is
    /// rounded to, as plain Metal lays them out (its validation stops the copy without the option
    /// and every copy at 0).
    ///
    /// An offset or `bytes_per_row` off the copy unit, or rows that overlap, read each row where
    /// plain Metal reads it, for a colour format and `Depth32Float`, and so does a
    /// `bytes_per_image` of more than one image off the copy unit or below the region's rows. A
    /// no-copy buffer is read past its length to the end of its host pages, the caller's bytes, as
    /// plain Metal reads them (its validation stops both).
    ///
    /// A size of 0 copies as the module docs say, as plain Metal does. Refused: what Metal stops
    /// (`Invalid`: with a [`CopyProblem`], a region past the level, an offset or `bytes_per_row`
    /// off the copy unit of another depth or stencil format or of a texture over a buffer, a
    /// layout that runs past the buffer but for that read); a buffer or texture of another
    /// device (`Invalid`). 32-bit float depth is copied as `Texture::replace_region` copies it,
    /// as Metal holds it.
    pub fn copy_from_buffer_to_texture(
        &mut self,
        source_buffer: &Buffer,
        source_offset: u64,
        source_bytes_per_row: u64,
        source_bytes_per_image: u64,
        source_size: Size,
        destination_texture: &Texture,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: Origin,
        options: BlitOption,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                BlitCall::CopyBufferToTexture(Box::new(BufferToTexture {
                    source: source_buffer.clone(),
                    source_offset,
                    bytes_per_row: source_bytes_per_row,
                    bytes_per_image: source_bytes_per_image,
                    size: source_size,
                    destination: destination_texture.clone(),
                    slice: destination_slice,
                    level: destination_level,
                    origin: destination_origin,
                    options,
                }))
            },
            |this| {
                const CALL: &str = "copy_from_buffer_to_texture";
                this.core().check_device(CALL, source_buffer.device())?;
                this.check_texture(CALL, destination_texture)?;
                destination_texture.check_bytes_reached(CALL)?;
                let Some(size) = copy_size(source_size, &[destination_texture.texture_type()])
                else {
                    return Ok(());
                };
                // Vulkan copies no multisampled image from a buffer, so the buffer-image path below
                // cannot take one. The empty copy is the no-op Metal makes (above); a real one into
                // a multisampled texture is emulated per sample through a wide image and a scratch
                // (`sample_interleave.rs`), or refused typed where the device cannot run that.
                let samples = destination_texture.record().samples;
                if samples > 1 {
                    let layout = BufferLayout {
                        offset: source_offset,
                        bytes_per_row: source_bytes_per_row,
                        bytes_per_image: source_bytes_per_image,
                    };
                    let region = region(
                        CALL,
                        destination_origin,
                        size,
                        destination_level,
                        destination_slice,
                    )?;
                    let region = multisampled_region(CALL, destination_texture, region)?;
                    let (raw_buffer, base) = this.core().blit_at(source_buffer, 0)?;
                    let aspect = multisampled_aspect(CALL, destination_texture, options)?;
                    let planned = sample_interleave::InterleaveCopy::plan(
                        CALL,
                        this.core().command_buffer(),
                        raw_buffer,
                        base,
                        source_buffer.copy_reach(),
                        &layout,
                        destination_texture,
                        &region,
                        aspect,
                        samples,
                        true,
                    )?;
                    return match planned {
                        Some(copy) => {
                            // Every sample of the region written: a heap shadow it covers takes no
                            // copy in.
                            let covered = [multisampled_box(destination_texture, aspect, &region)?];
                            let boxes = (&[][..], &covered[..]);
                            this.command_in_reading(
                                copy.stages,
                                &copy.uses,
                                boxes,
                                |device, raw| {
                                    // SAFETY: the encoder's command buffer is recording outside a
                                    // render pass; the buffer (held by `blit_at`), the texture and
                                    // the copy's images and objects (held by the copy; both
                                    // retained below) are live until it completes.
                                    unsafe { copy.record(device, raw) };
                                },
                            )?;
                            // A depth written as the single-sample path writes one announces as it
                            // does.
                            destination_texture.degrade_depth_upload(CALL, aspect, true, true);
                            destination_texture.degrade_depth_denormals(CALL, aspect, false, true);
                            copy.announce(this.core().device(), CALL, destination_texture);
                            let format = destination_texture.pixel_format();
                            copy.announce_staging(this.core().device(), format, true);
                            this.core().retain(copy);
                            this.core().retain(destination_texture.clone());
                            // Every sample of the region written, as the single-sample path's.
                            let written = multisampled_texels(&region);
                            this.scrub(destination_texture, vec![(aspect.into(), written)])
                        }
                        None => Err(Error::NotImplemented(RuntimeGap::MultisampledBufferCopy)),
                    };
                }
                let texture = destination_texture;
                let layout = BufferLayout {
                    offset: source_offset,
                    bytes_per_row: source_bytes_per_row,
                    bytes_per_image: source_bytes_per_image,
                };
                if two_planes::holds_planes(texture.pixel_format()) && !texture.over_buffer() {
                    let region = region(
                        CALL,
                        destination_origin,
                        size,
                        destination_level,
                        destination_slice,
                    )?;
                    return this.plane_copy(CALL, source_buffer, texture, &region, &layout, true);
                }
                let unit = unit(CALL, texture, options)?;
                let region = region(
                    CALL,
                    destination_origin,
                    size,
                    destination_level,
                    destination_slice,
                )?;
                if let Some(texels) = texture.buffer_texels() {
                    let (from, to) = plan_rows_copy(
                        texture.record(),
                        &region,
                        &unit,
                        &layout,
                        source_buffer.copy_reach(),
                        &texels.layout,
                    )
                    .map_err(|problem| invalid(CALL, Problem::Copy(problem)))?;
                    return this.rows_copy(
                        source_buffer,
                        &texels.buffer,
                        &RowsCopy::new(&from, &to),
                    );
                }
                let plan = plan_copy(CALL, texture, &region, &unit, &layout, source_buffer, true)?;
                let written: Vec<ImageRegion> = plan.regions.iter().map(|r| r.image).collect();
                this.buffer_texture_copy(source_buffer, texture, plan, true)?;
                texture.degrade_depth_upload(CALL, unit.aspect, true, true);
                texture.degrade_depth_denormals(CALL, unit.aspect, false, true);
                // Into a texture Metal keeps compressed: the texels its blit writes.
                if let Some(rule) = rule_of(texture) {
                    this.compressed_image(texture, &written, &[rule])?;
                }
                // Into a texture with a decoded twin: the twin's texels of what the copy wrote.
                this.decode_twin(texture, &written)
            },
        )
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:
    /// destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:`: the
    /// texture's region into the buffer, laid out as
    /// [`copy_from_buffer_to_texture`](Self::copy_from_buffer_to_texture) reads it. The bytes
    /// between rows and images are left as they are; images closer than the region's rows are
    /// written in turn, the later image's bytes staying, as plain Metal writes them.
    ///
    /// Refused as `copy_from_buffer_to_texture` is, except that depth is copied out on every
    /// device.
    pub fn copy_from_texture_to_buffer(
        &mut self,
        source_texture: &Texture,
        source_slice: u64,
        source_level: u64,
        source_origin: Origin,
        source_size: Size,
        destination_buffer: &Buffer,
        destination_offset: u64,
        destination_bytes_per_row: u64,
        destination_bytes_per_image: u64,
        options: BlitOption,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                BlitCall::CopyTextureToBuffer(Box::new(TextureToBuffer {
                    source: source_texture.clone(),
                    slice: source_slice,
                    level: source_level,
                    origin: source_origin,
                    size: source_size,
                    destination: destination_buffer.clone(),
                    destination_offset,
                    bytes_per_row: destination_bytes_per_row,
                    bytes_per_image: destination_bytes_per_image,
                    options,
                }))
            },
            |this| {
                const CALL: &str = "copy_from_texture_to_buffer";
                this.check_texture(CALL, source_texture)?;
                source_texture.check_bytes_reached(CALL)?;
                this.core()
                    .check_device(CALL, destination_buffer.device())?;
                let Some(size) = copy_size(source_size, &[source_texture.texture_type()]) else {
                    return Ok(());
                };
                let layout = BufferLayout {
                    offset: destination_offset,
                    bytes_per_row: destination_bytes_per_row,
                    bytes_per_image: destination_bytes_per_image,
                };
                // Vulkan copies no multisampled image into a buffer, so the buffer-image path below
                // cannot take one. The empty copy is the no-op Metal makes (above); a real one out
                // of a multisampled texture is emulated per sample through a scratch and a wide
                // image (`sample_interleave.rs`), or refused typed where the device cannot run
                // that. Rows that overlap out of a multisampled texture race on Metal (no exact
                // answer), as out of a single-sample one (`blit_plan::placed_rows`), so the
                // emulation's placement of the rows refuses them
                // (`sample_interleave::placed_rows`).
                let samples = source_texture.record().samples;
                if samples > 1 {
                    let aspect = multisampled_aspect(CALL, source_texture, options)?;
                    let region = region(CALL, source_origin, size, source_level, source_slice)?;
                    let region = multisampled_region(CALL, source_texture, region)?;
                    let (raw_buffer, base) = this.core().blit_at(destination_buffer, 0)?;
                    let planned = sample_interleave::InterleaveCopy::plan(
                        CALL,
                        this.core().command_buffer(),
                        raw_buffer,
                        base,
                        destination_buffer.length(),
                        &layout,
                        source_texture,
                        &region,
                        aspect,
                        samples,
                        false,
                    )?;
                    // The region's samples alone are read (`fresh_unwritten.rs`).
                    let boxes = [(aspect.into(), multisampled_texels(&region))];
                    let fresh_reads =
                        FreshUnwritten::reads(this.core().device(), source_texture, &boxes)?;
                    return match planned {
                        Some(copy) => {
                            this.command_in_reading(
                                copy.stages,
                                &copy.uses,
                                (&fresh_reads, &[]),
                                |device, raw| {
                                    // SAFETY: as in `copy_from_buffer_to_texture`'s emulation.
                                    unsafe { copy.record(device, raw) };
                                },
                            )?;
                            let format = source_texture.pixel_format();
                            copy.announce_staging(this.core().device(), format, false);
                            this.core().retain(copy);
                            this.core().retain(source_texture.clone());
                            Ok(())
                        }
                        None => Err(Error::NotImplemented(RuntimeGap::MultisampledBufferCopy)),
                    };
                }
                if two_planes::holds_planes(source_texture.pixel_format())
                    && !source_texture.over_buffer()
                {
                    let region = region(CALL, source_origin, size, source_level, source_slice)?;
                    return this.plane_copy(
                        CALL,
                        destination_buffer,
                        source_texture,
                        &region,
                        &layout,
                        false,
                    );
                }
                let unit = unit(CALL, source_texture, options)?;
                let region = region(CALL, source_origin, size, source_level, source_slice)?;
                if let Some(texels) = source_texture.buffer_texels() {
                    let (to, from) = plan_rows_copy(
                        source_texture.record(),
                        &region,
                        &unit,
                        &layout,
                        destination_buffer.length(),
                        &texels.layout,
                    )
                    .map_err(|problem| invalid(CALL, Problem::Copy(problem)))?;
                    let copy = RowsCopy::new(&from, &to);
                    return this.rows_copy(&texels.buffer, destination_buffer, &copy);
                }
                let plan = plan_copy(
                    CALL,
                    source_texture,
                    &region,
                    &unit,
                    &layout,
                    destination_buffer,
                    false,
                )?;
                this.buffer_texture_copy(destination_buffer, source_texture, plan, false)?;
                // Out of a texture Metal keeps compressed: the texels its blit writes, at the
                // buffer's rows the copy wrote, image after image.
                if let Some(rule) = rule_of(source_texture) {
                    let rows = copied_rows(source_texture.record(), &region, &unit, &layout, false)
                        .map_err(|problem| invalid(CALL, Problem::Copy(problem)))?;
                    this.compressed_rows(destination_buffer, &rows, &[rule])?;
                }
                Ok(())
            },
        )
    }

    /// A blit between `buffer`, laid out as `layout`, and `region` of a two-plane
    /// extended-range texture (`into_texture` from the buffer; `two_planes.rs`): Metal's checks
    /// of the copy in plane 0's color words, then their bands. Refused as other copies are, and
    /// `NotImplemented(Format)` for a layout Metal's answer was not measured for.
    fn plane_copy(
        &mut self,
        call: &'static str,
        buffer: &Buffer,
        texture: &Texture,
        region: &TextureRegion,
        layout: &BufferLayout,
        into_texture: bool,
    ) -> Result<(), Error> {
        let unit = two_planes::UNIT;
        let image = check_region(texture.record(), region, &unit)
            .map_err(|problem| invalid(call, Problem::Copy(problem)))?;
        // Metal's checks of the buffer's layout, its range among them, as of a 4-byte format.
        plan_copy(call, texture, region, &unit, layout, buffer, into_texture)?;
        let s = shape(region.size, &unit);
        if !two_planes::measured_layout(
            s.row_bytes,
            s.rows,
            s.images,
            layout.bytes_per_row,
            layout.bytes_per_image,
            layout.offset,
        ) {
            return Err(Error::Unsupported(two_planes::unmeasured(
                texture.pixel_format(),
            )));
        }
        // At a width that is not a multiple of 4 Metal's blit addresses the level's rows short of
        // where they lie, moving texels between rows: at those rows for a 2D texture or array of
        // one level (`two_planes/buffer_rows.rs`); for another (more levels, a view, 1D, 3D, a
        // surface's), announced, the texels moved as `getBytes` and `replaceRegion` move them.
        let width = level_extent(texture.record(), region.level)[0];
        if !width.is_multiple_of(4) && two_planes::rows_held(texture) {
            return self.plane_rows_copy(buffer, texture, &image, layout, into_texture);
        }
        if !width.is_multiple_of(4) {
            let pf = texture.pixel_format();
            self.core()
                .device()
                .degrade(Degradation::TwoPlaneBufferBlit, || {
                    let object = format!(
                        "`{call}` between a buffer and a {} texture level {width} texels wide",
                        pf.name()
                    );
                    let reason = "a texture of more than one level, a view, a 1D or 3D texture or a surface's: \
                                  the texels move at the rows the texture holds them in, where \
                                  Metal's blit addresses its rows floor(4w/16)*16 bytes apart, \
                                  short of them, and so moves texels between rows"
                        .to_owned();
                    (object, reason)
                });
        }
        // A two-plane 3D texture (`two_planes.rs`): Metal's blit into it writes each image of
        // the region over texel `(x, y)` of its first slice in turn, so the last image stays
        // there; its blit out of it reads texel `(0, 0, 0)` at every texel of every image.
        if texture.texture_type() == TextureType::Type3D {
            let first = ImageRegion {
                origin: [image.origin[0], image.origin[1], 0],
                size: [image.size[0], image.size[1], 1],
                ..image
            };
            if !into_texture {
                return self.plane_first_texel_copy(buffer, texture, &image, layout);
            }
            let last = u64::from(image.size[2].saturating_sub(1));
            let layout = BufferLayout {
                offset: layout.offset + last * layout.bytes_per_image,
                ..*layout
            };
            return self.plane_words_copy(buffer, texture, &first, &layout, true);
        }
        self.plane_words_copy(buffer, texture, &image, layout, into_texture)
    }

    /// Record a checked copy between `buffer` and `texture` (`into_texture`: from the buffer
    /// into the texture), through a scratch buffer when the plan stages it.
    fn buffer_texture_copy(
        &mut self,
        buffer: &Buffer,
        texture: &Texture,
        plan: BufferCopyPlan,
        into_texture: bool,
    ) -> Result<(), Error> {
        let Some(first) = plan.regions.first() else {
            unreachable!("an empty copy was refused")
        };
        let (levels, layers) = subresources(&first.image);
        let image_use = texture.image_use(
            levels,
            layers,
            if into_texture { write() } else { read() },
            into_texture,
        )?;
        // A copy out of the texture reads its regions alone (`fresh_unwritten.rs`).
        let boxes: Vec<_> = match into_texture {
            true => Vec::new(),
            false => (plan.regions.iter())
                .map(|r| (r.image.aspect.into(), Written::of(&r.image)))
                .collect(),
        };
        let fresh_reads = FreshUnwritten::reads(self.core().device(), texture, &boxes)?;
        // The plan's offsets are the `VkBuffer`'s (`plan_buffer_copy`'s base), its regions in the
        // texture's texels, which a 2-bit PVRTC texture's carrier halves across
        // (`pvrtc::image_buffer_region`). Read as a blit's (`host_residency.rs`).
        let (raw_buffer, _) = self.core().blit_at(buffer, 0)?;
        let image = texture.raw_image()?;
        let image_plan = texture.image_plan()?;
        let regions: Vec<vk::BufferImageCopy> = plan
            .regions
            .iter()
            .map(|region| buffer_image_region(&image_buffer_region(image_plan, *region)))
            .collect();
        // A copy into a texture with a depth twin writes the twin too.
        let twin = texture.twin_image().filter(|_| into_texture);
        let twin_regions: Vec<vk::BufferImageCopy> =
            regions.iter().copied().map(twin_copy).collect();
        let buffer_use = if into_texture {
            reads(buffer, plan.span.clone())
        } else {
            writes(buffer, plan.span.clone())
        };
        // A copy into the texture writes every texel of its regions.
        let covered = match into_texture {
            true => Covered::into_image(image, &regions),
            false => Vec::new(),
        };
        match plan.staging {
            None => {
                let boxes = (&fresh_reads[..], &covered[..]);
                self.command_reading(&[buffer_use, image_use], boxes, |device, raw| {
                    // SAFETY: the encoder's command buffer is recording; the buffer and the image
                    // are live and retained until it completes, the image is in the `GENERAL`
                    // layout, and the regions lie in both (checked by the plan).
                    unsafe {
                        if into_texture {
                            device.cmd_copy_buffer_to_image(
                                raw,
                                raw_buffer,
                                image,
                                vk::ImageLayout::GENERAL,
                                &regions,
                            );
                            if let Some(twin) = twin {
                                device.cmd_copy_buffer_to_image(
                                    raw,
                                    raw_buffer,
                                    twin,
                                    vk::ImageLayout::GENERAL,
                                    &twin_regions,
                                );
                            }
                        } else {
                            device.cmd_copy_image_to_buffer(
                                raw,
                                image,
                                vk::ImageLayout::GENERAL,
                                raw_buffer,
                                &regions,
                            );
                        }
                    }
                })?
            }
            Some(staging) => {
                let scratch = self
                    .core()
                    .device()
                    .own_buffer(staging.length, ResourceOptions::private())?;
                // A buffer `own_buffer` makes starts at its `VkBuffer`'s start, where the
                // plan's staging offsets are.
                let (through, _) = scratch.at(0);
                let whole = 0..staging.length;
                let rows: Vec<vk::BufferCopy> = if into_texture {
                    staging.rows.iter().map(buffer_copy).collect()
                } else {
                    // The plan's rows run buffer to staging; out of the texture they run back.
                    let back = |r: &BufferRange| BufferRange {
                        src: r.dst,
                        dst: r.src,
                        size: r.size,
                    };
                    staging.rows.iter().map(|r| buffer_copy(&back(r))).collect()
                };
                let (src, dst) = if into_texture {
                    (raw_buffer, through)
                } else {
                    (through, raw_buffer)
                };
                let device = self.core().device().clone();
                let rows = RowCopies::new(&device, src, dst, rows);
                let move_rows = |device: &Vk, raw: vk::CommandBuffer| {
                    // SAFETY: as below; the rows lie in both buffers (checked by the plan), the
                    // staging buffer is another buffer, and the rows written are disjoint (out of
                    // the texture, `later_rows_win`).
                    unsafe { rows.record(device, raw) };
                };
                let copy_image = |device: &Vk, raw: vk::CommandBuffer| {
                    // SAFETY: the encoder's command buffer is recording; the scratch buffer and
                    // the image are live and retained until it completes, the image is in the
                    // `GENERAL` layout, and the region lies in both.
                    unsafe {
                        if into_texture {
                            device.cmd_copy_buffer_to_image(
                                raw,
                                through,
                                image,
                                vk::ImageLayout::GENERAL,
                                &regions,
                            );
                            if let Some(twin) = twin {
                                device.cmd_copy_buffer_to_image(
                                    raw,
                                    through,
                                    twin,
                                    vk::ImageLayout::GENERAL,
                                    &twin_regions,
                                );
                            }
                        } else {
                            device.cmd_copy_image_to_buffer(
                                raw,
                                image,
                                vk::ImageLayout::GENERAL,
                                through,
                                &regions,
                            );
                        }
                    }
                };
                let span = plan.span.clone();
                if into_texture {
                    let uses = [
                        reads_by(buffer, span, rows.read()),
                        writes_by(&scratch, whole.clone(), rows.write()),
                    ];
                    self.command_in(rows.stages(), &uses, move_rows)?;
                    let uses = [reads(&scratch, whole), image_use];
                    self.command_reading(&uses, (&[], &covered), copy_image)?;
                } else {
                    let uses = [image_use, writes(&scratch, whole.clone())];
                    self.command_reading(&uses, (&fresh_reads, &[]), copy_image)?;
                    let uses = [
                        reads_by(&scratch, whole, rows.read()),
                        writes_by(buffer, span, rows.write()),
                    ];
                    self.command_in(rows.stages(), &uses, move_rows)?;
                }
                if let Some(sets) = rows.retained() {
                    self.core().retain(sets);
                }
                self.core().retain(scratch);
                let (offset, format) = (plan.span.start, texture.pixel_format());
                self.core()
                    .device()
                    .degrade(Degradation::StagedTextureCopy, || {
                        let object = if into_texture {
                            format!(
                                "a copy from a buffer at offset {offset} into a {format:?} texture"
                            )
                        } else {
                            format!(
                                "a copy from a {format:?} texture into a buffer at offset {offset}"
                            )
                        };
                        let reason = "Vulkan cannot state the buffer's layout: its offset in the \
                                  VkBuffer is off the texel block (or off 4 bytes for depth and \
                                  stencil), or its images lie apart by other than a multiple of \
                                  its row pitch";
                        (object, reason.to_owned())
                    });
            }
        }
        self.core().retain(buffer.clone());
        self.core().retain(texture.clone());
        if into_texture {
            let written = (plan.regions.iter())
                .map(|r| (r.image.aspect.into(), Written::of(&r.image)))
                .collect();
            self.scrub(texture, written)?;
        }
        Ok(())
    }

    /// After a command that wrote `written` of `texture` (each box with the aspects it wrote):
    /// its texels are no longer a new texture's unwritten pattern (`fresh_unwritten.rs`, by box and
    /// aspect); and where it is a sparse texture the runtime zeroes, its scrub, so what the command
    /// wrote into unmapped tiles reads zero, as Metal drops it.
    ///
    /// Measured for a **blit** and not only for a shader's read, on an Apple M3 Max under macOS
    /// 27.0.1 (26A434), three runs each in its own process, over a 256 by 256 `R32Uint` sparse
    /// texture of 64-texel tiles with eight of its sixteen tiles mapped in a checkerboard:
    /// `copyFromBuffer:toTexture:` over the whole level leaves every one of the 4,096 texels of
    /// each mapped tile holding the copy's word and every texel of each unmapped tile at **zero**,
    /// read back both by `copyFromTexture:toBuffer:` and by a kernel's `read()`, which agree tile
    /// for tile. A second such copy after a tile is mapped writes it, so the first reading is not
    /// a copy that did nothing.
    ///
    /// Where this scrub is **not** Metal: a tile mapped **after** a blit covered it while unmapped
    /// reads Metal's unwritten pattern rather than zero, because Metal dropped the write and so
    /// still holds the tile unwritten, while this call has already marked the blit's whole
    /// destination written. `docs/FAST-PATHS.md`'s known gaps carries it with the measurement.
    pub(super) fn scrub(
        &mut self,
        texture: &Texture,
        written: Vec<(vk::ImageAspectFlags, Written)>,
    ) -> Result<(), Error> {
        FreshUnwritten::wrote(self.core().device(), texture, &written)?;
        let written = || written.into_iter().map(|(_, texels)| texels).collect();
        match Scrub::of(texture, written) {
            Some(scrub) => self.core().defer(Deferred::SparseScrub(Arc::new(scrub))),
            None => Ok(()),
        }
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:
    /// destinationSlice:destinationLevel:destinationOrigin:`: a region of one texture into
    /// another at `destination_origin`.
    ///
    /// The textures have one sample count and formats of one copy unit (macOS 26.7's rule: the
    /// bytes are copied, `RGBA8Unorm` into `RGBA8Uint` or `R32Uint` among them, and
    /// `Depth32Float` into `R32Float` or back), or a compressed format and an uncompressed one
    /// whose texel is its block's bytes (BC1 and `RG32Uint`, BC7 and `RGBA32Uint`): each block
    /// copies into one texel and back, and the origins and size are the destination's texels, the
    /// source addressed as if it held the destination's format (our probe on an Apple M3 Max
    /// under macOS 27.0.1: out of BC1 an origin of (4, 4) is the block from texel (16, 16); into
    /// it an origin of (1, 2) and a size of 2 x 2 copy `RG32Uint` texel (0, 0) alone, into the
    /// block a destination origin of (4, 4) names).
    ///
    /// A size of 0 copies as the module docs say, as plain Metal does, and a compressed region
    /// off its blocks copies the blocks it starts in on each side, a whole block of a level
    /// smaller than one included, into or out of a larger level too. Refused: what Metal stops
    /// (`Invalid`: with a [`CopyProblem`], formats of different copy units, different sample
    /// counts, a region past either level); a PVRTC format with an uncompressed one, whose blocks
    /// plain Metal moves in an order of its own, and two compressed formats of one block's bytes
    /// and different extents, not measured (`NotImplemented(Format)`); a copy between overlapping regions of one
    /// multisampled texture (`NotImplemented(OverlappingTextureCopies)`), other than of a region
    /// onto itself (one texture's other overlapping regions copy as `memmove` does, the module
    /// docs); a copy through a scratch buffer of 24-bit depth, whose bytes Metal's reference GPUs
    /// give no answer for (`NotImplemented(Format)`: an Apple M3 Max and an M4 make no
    /// `Depth24Unorm_Stencil8` texture); a multisampled copy between a depth or stencil aspect and
    /// a colour format, or between two depth/stencil formats, on a device without
    /// `VK_KHR_maintenance8` or whose driver
    /// copies such images wrong, where its sample-rate draws cannot copy it either: a 16-bit
    /// depth, a source the draws cannot sample or a destination they cannot render to (`Unsupported`,
    /// `Feature::Format`, naming maintenance8 or the defect); textures of another device
    /// (`Invalid`).
    pub fn copy_from_texture_to_texture(
        &mut self,
        source_texture: &Texture,
        source_slice: u64,
        source_level: u64,
        source_origin: Origin,
        source_size: Size,
        destination_texture: &Texture,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: Origin,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                BlitCall::CopyTextureToTexture(Box::new(TextureToTexture {
                    source: source_texture.clone(),
                    source_slice,
                    source_level,
                    source_origin,
                    source_size,
                    destination: destination_texture.clone(),
                    destination_slice,
                    destination_level,
                    destination_origin,
                }))
            },
            |this| {
                const CALL: &str = "copy_from_texture_to_texture";
                this.check_pair(CALL, source_texture, destination_texture)?;
                let types = [
                    source_texture.texture_type(),
                    destination_texture.texture_type(),
                ];
                let Some(size) = copy_size(source_size, &types) else {
                    return Ok(());
                };
                let source = region(CALL, source_origin, size, source_level, source_slice)?;
                let at = region(
                    CALL,
                    destination_origin,
                    size,
                    destination_level,
                    destination_slice,
                )?;
                let copies = plan_region_copy(
                    &pair(source_texture, destination_texture)?,
                    &source,
                    at.level,
                    at.slice,
                    at.origin,
                )
                .map_err(|refusal| copy_error(CALL, refusal))?;
                this.texture_copies(CALL, source_texture, destination_texture, &copies)
            },
        )
    }

    /// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:
    /// sliceCount:levelCount:`: `slice_count` slices of `level_count` levels, whole, each level
    /// into a destination level. Levels of different sizes copy the region common to both from
    /// their origins, the rest of the larger one kept, as plain Metal copies them (an M4 under
    /// macOS 27.0.1; its validation stops them).
    ///
    /// Refused as [`copy_from_texture_to_texture`](Self::copy_from_texture_to_texture) is, and
    /// for levels or slices past either texture and compressed levels of different sizes
    /// (`Invalid`, unmeasured), and between a compressed and an uncompressed format
    /// (`NotImplemented(Format)`: which levels Metal pairs is not measured). A count of 0 copies
    /// nothing, wherever its slices or levels lie, as plain Metal does.
    pub fn copy_texture_slices(
        &mut self,
        source_texture: &Texture,
        source_slice: u64,
        source_level: u64,
        destination_texture: &Texture,
        destination_slice: u64,
        destination_level: u64,
        slice_count: u64,
        level_count: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| {
                BlitCall::CopyTextureSlices(Box::new((
                    source_texture.clone(),
                    source_slice,
                    source_level,
                    destination_texture.clone(),
                    destination_slice,
                    destination_level,
                    slice_count,
                    level_count,
                )))
            },
            |this| {
                const CALL: &str = "copy_texture_slices";
                this.check_pair(CALL, source_texture, destination_texture)?;
                if slice_count == 0 || level_count == 0 {
                    return Ok(());
                }
                let n = |value: u64| {
                    u32::try_from(value)
                        .map_err(|_| invalid(CALL, Problem::Copy(CopyProblem::Region)))
                };
                let copies = plan_levels_copy(
                    &pair(source_texture, destination_texture)?,
                    n(source_slice)?,
                    n(source_level)?,
                    n(destination_slice)?,
                    n(destination_level)?,
                    n(slice_count)?,
                    n(level_count)?,
                )
                .map_err(|refusal| copy_error(CALL, refusal))?;
                this.texture_copies(CALL, source_texture, destination_texture, &copies)
            },
        )
    }

    /// `copyFromTexture:toTexture:`: every level and slice two textures share, as plain Metal
    /// copies them: the levels paired by size from the first pair of one size (the source's first
    /// and the destination's, or one's first and the other's first level of the other's first
    /// size), and the slices both have; a 2D texture into a 2D array copies its one slice.
    ///
    /// Refused as [`copy_texture_slices`](Self::copy_texture_slices) is, and for textures with
    /// no levels of one size (`Invalid`: plain Metal aborts).
    pub fn copy_texture(
        &mut self,
        source_texture: &Texture,
        destination_texture: &Texture,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::CopyTexture(source_texture.clone(), destination_texture.clone()),
            |this| {
                const CALL: &str = "copy_texture";
                this.check_pair(CALL, source_texture, destination_texture)?;
                let copies = plan_whole_copy(&pair(source_texture, destination_texture)?)
                    .map_err(|refusal| copy_error(CALL, refusal))?;
                this.texture_copies(CALL, source_texture, destination_texture, &copies)
            },
        )
    }

    /// The checks every copy between two textures makes first.
    fn check_pair(
        &mut self,
        call: &'static str,
        source: &Texture,
        destination: &Texture,
    ) -> Result<(), Error> {
        self.check_texture(call, source)?;
        self.check_texture(call, destination)?;
        source.check_bytes_reached(call)?;
        destination.check_bytes_reached(call)?;
        check_depth_view(source)?;
        check_depth_view(destination)
    }

    /// Record checked copies between two textures. `copies` run between the images [`pair`]
    /// presents, a texture with a depth twin by its twin, each in its texture's aspect. Each image
    /// holding the destination's texels ([`held`]: both of a texture with a twin) takes them from
    /// an image of the source Vulkan copies into it (colour into colour, a depth or stencil aspect
    /// into the same aspect of the same format, regions of one size), else through a scratch
    /// buffer of the encoder out of the image the source's texels are read from. So out of a
    /// texture with a twin into a depth texture without one they read its twin, and into it from a
    /// depth texture they write its twin directly and its image through the buffer.
    pub(super) fn texture_copies(
        &mut self,
        call: &'static str,
        source: &Texture,
        destination: &Texture,
        copies: &[ImageCopyRegion],
    ) -> Result<(), Error> {
        self.texture_copies_of_bytes(call, source, destination, copies)?;
        // Into a texture with a decoded twin: the twin's texels of what the copies wrote.
        let written: Vec<ImageRegion> = copies.iter().map(|copy| copy.dst).collect();
        self.decode_twin(destination, &written)?;
        // Out of or into a texture Metal keeps compressed: the texels its blit writes, in the
        // destination's image or its buffer's rows. A region copied onto itself records nothing.
        let rules = rules_of(source, destination);
        if rules.is_empty() {
            return Ok(());
        }
        let one_storage = source.same_storage(destination);
        let copies: Vec<&ImageCopyRegion> = copies
            .iter()
            .filter(|copy| !(one_storage && copy.src == copy.dst))
            .collect();
        match destination.buffer_texels() {
            Some(texels) => {
                let unit = texels.unit();
                let rows: Vec<Range<u64>> = copies
                    .iter()
                    .flat_map(|copy| texel_rows(&texels.layout, &unit, &copy.dst))
                    .collect();
                let buffer = texels.buffer.clone();
                self.compressed_rows(&buffer, &rows, &rules)
            }
            None => {
                let regions: Vec<ImageRegion> = copies.iter().map(|copy| copy.dst).collect();
                self.compressed_image(destination, &regions, &rules)
            }
        }
    }

    /// Record checked copies between two textures, their bytes as the images hold them (the
    /// texels Metal's blit through a compressed texture writes are `texture_copies`').
    fn texture_copies_of_bytes(
        &mut self,
        call: &'static str,
        source: &Texture,
        destination: &Texture,
        copies: &[ImageCopyRegion],
    ) -> Result<(), Error> {
        if source.over_buffer() || destination.over_buffer() {
            // A two-plane extended-range texture over a surface's planes (the plan paired it with
            // another): its bytes as Metal's blit moves them, else refused.
            if two_planes::holds_planes(source.pixel_format()) {
                return match self.modelled_plane_copies(source, destination, copies)? {
                    true => Ok(()),
                    false => Err(internal("a surface's two-plane copy left unmodelled")),
                };
            }
            return self.texel_copies(call, source, destination, copies);
        }
        // Two two-plane extended-range textures: their bytes moved as Metal's blit moves them,
        // where that was measured; else (announced) both words of each texel, below, a 3D one's
        // at its first depth slice, where its texels lie (`two_planes.rs`).
        if two_planes::holds_planes(source.pixel_format())
            && self.modelled_plane_copies(source, destination, copies)?
        {
            return Ok(());
        }
        let first_slices: Vec<ImageCopyRegion>;
        let copies = match two_planes::holds_planes(source.pixel_format())
            && [source, destination]
                .iter()
                .any(|t| t.texture_type() == TextureType::Type3D)
        {
            true => {
                first_slices = copies.iter().map(two_planes::first_slices).collect();
                &first_slices[..]
            }
            false => copies,
        };
        // A region copied onto itself keeps its texels in whatever order Metal copies them
        // (probes): nothing is recorded for it. Between other overlapping regions of one texture
        // Metal's copy races past a few hundred texels across (probes 7 and 7b). Such a copy goes
        // through the scratch buffer below, every source region read before any destination region
        // is written: `memmove`'s answer, Metal's for small regions, announced once. A multisampled
        // image goes into no buffer: still refused.
        let one_storage = source.same_storage(destination);
        let overlap = one_storage
            && copies.iter().enumerate().any(|(i, a)| {
                copies
                    .iter()
                    .enumerate()
                    .any(|(j, b)| regions_meet(&a.src, &b.dst) && !(i == j && a.src == a.dst))
            });
        if overlap && source.record().samples > 1 {
            return Err(Error::NotImplemented(RuntimeGap::OverlappingTextureCopies));
        }
        let copies: Vec<ImageCopyRegion> = copies
            .iter()
            .filter(|copy| !(one_storage && copy.src == copy.dst))
            .copied()
            .collect();
        let Some(first) = copies.first() else {
            return Ok(());
        };
        let mut uses = Vec::with_capacity(2 * copies.len());
        for copy in &copies {
            let (levels, layers) = subresources(&copy.src);
            uses.push(source.image_use(levels, layers, read(), false)?);
            let (levels, layers) = subresources(&copy.dst);
            uses.push(destination.image_use(levels, layers, write(), true)?);
        }
        // The source's regions alone are read (`fresh_unwritten.rs`).
        let boxes: Vec<_> = (copies.iter())
            .map(|copy| (copy.src.aspect.into(), Written::of(&copy.src)))
            .collect();
        let fresh_reads = FreshUnwritten::reads(self.core().device(), source, &boxes)?;
        // Each image of the destination takes the regions from an image of the source that
        // Vulkan copies into it, else through a scratch buffer from the source's texels.
        let from = held(source, first.src.aspect)?;
        let into = held(destination, first.dst.aspect)?;
        let mut direct = Vec::with_capacity(into.len());
        let mut staged = Vec::new();
        // The regions in each image's texels: a 2-bit PVRTC texture's carrier halves them across
        // (`pvrtc::image_region`).
        let (source_plan, destination_plan) = (source.image_plan()?, destination.image_plan()?);
        let in_images: Vec<ImageCopyRegion> = copies
            .iter()
            .map(|copy| ImageCopyRegion {
                src: image_region(source_plan, copy.src),
                dst: image_region(destination_plan, copy.dst),
            })
            .collect();
        // A pair of regions of different sizes (a compressed region that ends at its level's edge
        // inside a block on one side only, `ImageCopyRegion`) has no one Vulkan extent: it goes
        // through the scratch buffer too, which holds the blocks both cover.
        let one_extent = in_images.iter().all(|copy| copy.src.size == copy.dst.size);
        let samples = source.record().samples;
        let mut bits = Vec::new();
        for to in &into {
            match from
                .iter()
                .find(|from| from.copies_into(to))
                .filter(|_| !overlap && one_extent)
            {
                Some(from) => {
                    let mut regions: Vec<ImageCopyRegion> =
                        in_images.iter().map(|copy| from.region(to, copy)).collect();
                    // Into a multisampled float image whose NaNs the device's copy converts:
                    // the regions at one origin in both images by the sample planes' copies,
                    // which move every sample's bits; the rest, or all where those cannot run,
                    // copied by the device and announced.
                    let device = self.core().device().clone();
                    if nan_clear::converting_copy(&device, to.format, samples) {
                        let (exact, rest): (Vec<&ImageCopyRegion>, Vec<&ImageCopyRegion>) = regions
                            .iter()
                            .partition(|copy| copy.src.origin == copy.dst.origin);
                        let pairs: Vec<(ImageRegion, ImageRegion)> =
                            exact.iter().map(|copy| (copy.src, copy.dst)).collect();
                        if let Some(copy) = sample_planes::BitsCopy::plan(
                            self.core().command_buffer(),
                            source,
                            destination,
                            to.format,
                            &pairs,
                            samples,
                        )? {
                            bits.push(copy);
                            regions = rest.into_iter().copied().collect();
                        }
                        if !regions.is_empty() {
                            nan_clear::degrade_converting_copy(&device, to.format, samples, || {
                                format!(
                                    "a blit copy into a multisampled {:?} texture",
                                    destination.pixel_format()
                                )
                            });
                        }
                    }
                    if !regions.is_empty() {
                        let regions: Vec<vk::ImageCopy> = regions.iter().map(image_copy).collect();
                        direct.push((from.image, to.image, regions));
                        // A depth that may hold a negative value or a NaN, copied as it is.
                        if to.aspect == ImageAspectFlags::DEPTH && source.may_hold_signed_depth() {
                            destination.degrade_signed_depth(call, true);
                        }
                    }
                }
                None => staged.push(*to),
            }
        }
        for copy in &bits {
            self.command_in_reading(
                sample_planes::BITS_COPY_STAGES,
                &copy.uses,
                (&fresh_reads, &[]),
                |device, raw| {
                    // SAFETY: the encoder's command buffer is recording outside a render pass; the
                    // images and the copy's planes and objects are live until it completes
                    // (retained below).
                    unsafe { copy.record(device, raw) };
                },
            )?;
        }
        if !bits.is_empty() {
            self.core().retain(bits);
        }
        // Multisampled images, which Vulkan copies into no buffer, copy through colour images
        // of their samples instead, by copies or by draws (`sample_copy.rs`). A multisampled
        // texture has no twin, so its one image takes them.
        if samples > 1 && !staged.is_empty() {
            let side = |texture, held: &Held| sample_copy::Side {
                texture,
                image: held.image,
                aspect: held.aspect,
                format: held.format,
            };
            let mut planned = Vec::with_capacity(staged.len());
            let mut stages = TRANSFER;
            for to in &staged {
                let regions: Vec<(ImageRegion, ImageRegion)> = in_images
                    .iter()
                    .map(|copy| {
                        let copy = from[0].region(to, copy);
                        (copy.src, copy.dst)
                    })
                    .collect();
                let copy = sample_copy::plan(
                    self.core().command_buffer(),
                    &regions,
                    side(source, &from[0]),
                    side(destination, to),
                    samples,
                )?;
                uses.extend(copy.uses.iter().cloned());
                stages |= copy.stages;
                destination.degrade_depth_upload(call, to.aspect, true, true);
                destination.degrade_depth_denormals(call, to.aspect, false, true);
                planned.push(copy);
            }
            self.command_in_reading(stages, &uses, (&fresh_reads, &[]), |device, raw| {
                for (from, to, regions) in &direct {
                    // SAFETY: the encoder's command buffer is recording; the images are live and
                    // retained until it completes, in the `GENERAL` layout, and the regions lie
                    // in both (checked by the plan).
                    unsafe {
                        device.cmd_copy_image(
                            raw,
                            *from,
                            vk::ImageLayout::GENERAL,
                            *to,
                            vk::ImageLayout::GENERAL,
                            regions,
                        );
                    }
                }
                for copy in &planned {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // the images and what the copies go through are live until it completes
                    // (retained below), in `GENERAL`, and `sample_copy::plan` checked the rest.
                    unsafe { copy.record(device, raw) };
                }
            })?;
            for copy in &planned {
                copy.announce(self.core().device(), call, destination);
            }
            self.core().retain(planned);
            self.core().retain(source.clone());
            if source != destination {
                self.core().retain(destination.clone());
            }
            let written = (copies.iter())
                .map(|copy| (copy.dst.aspect.into(), Written::of(&copy.dst)))
                .collect();
            return self.scrub(destination, written);
        }
        let staging = match staged.is_empty() {
            true => None,
            false => Some(
                through_buffer(
                    &in_images,
                    &from[0],
                    image_block_extent(from[0].format, source.pixel_format()).unwrap_or([1, 1]),
                    &staged,
                )
                .map_err(|refusal| copy_error(call, refusal))?,
            ),
        };
        let scratch = match &staging {
            Some(staging) => {
                let length = staging.length;
                let scratch = self
                    .core()
                    .device()
                    .own_buffer(length, ResourceOptions::private())?;
                let (through, _) = scratch.at(0);
                let mut reads_source = Vec::with_capacity(copies.len() + 1);
                for copy in &copies {
                    let (levels, layers) = subresources(&copy.src);
                    reads_source.push(source.image_use(levels, layers, read(), false)?);
                }
                reads_source.push(writes(&scratch, 0..length));
                let (image, out_of) = (from[0].image, &staging.out_of);
                self.command_reading(&reads_source, (&fresh_reads, &[]), |device, raw| {
                    // SAFETY: the encoder's command buffer is recording; the image and the
                    // scratch buffer are live and retained until it completes, the image is in
                    // the `GENERAL` layout, and the regions lie in both.
                    unsafe {
                        device.cmd_copy_image_to_buffer(
                            raw,
                            image,
                            vk::ImageLayout::GENERAL,
                            through,
                            out_of,
                        );
                    }
                })?;
                uses.push(reads(&scratch, 0..length));
                for (to, _) in &staging.into {
                    let signed = source.may_hold_signed_depth();
                    destination.degrade_depth_upload(call, to.aspect, signed, true);
                    destination.degrade_depth_denormals(call, to.aspect, false, true);
                }
                Some((scratch, through))
            }
            None => None,
        };
        // The copies write every texel of their regions.
        let mut covered = Vec::new();
        for (_, to, regions) in &direct {
            covered.extend(Covered::into_image_from_image(*to, regions));
        }
        for (to, regions) in staging.iter().flat_map(|staging| &staging.into) {
            covered.extend(Covered::into_image(to.image, regions));
        }
        self.command_reading(&uses, (&fresh_reads, &covered), |device, raw| {
            // SAFETY: the encoder's command buffer is recording; the images (and the scratch
            // buffer) are live and retained until it completes, in the `GENERAL` layout, the
            // regions lie in both (checked by the plan), and regions of one image do not overlap.
            unsafe {
                for (from, to, regions) in &direct {
                    device.cmd_copy_image(
                        raw,
                        *from,
                        vk::ImageLayout::GENERAL,
                        *to,
                        vk::ImageLayout::GENERAL,
                        regions,
                    );
                }
                if let (Some((_, through)), Some(staging)) = (&scratch, &staging) {
                    for (to, regions) in &staging.into {
                        device.cmd_copy_buffer_to_image(
                            raw,
                            *through,
                            to.image,
                            vk::ImageLayout::GENERAL,
                            regions,
                        );
                    }
                }
            }
        })?;
        if let Some((scratch, ..)) = scratch {
            self.core().retain(scratch);
        }
        self.core().retain(source.clone());
        if source != destination {
            self.core().retain(destination.clone());
        }
        if overlap {
            let texels: u64 = copies
                .iter()
                .map(|copy| {
                    let [w, h, d] = copy.src.size;
                    u64::from(w) * u64::from(h) * u64::from(d) * u64::from(copy.src.layers)
                })
                .sum();
            let format = destination.pixel_format();
            self.core()
                .device()
                .degrade(Degradation::OverlappingTextureCopyAsMemmove, || {
                    (
                        format!(
                            "{call} of {texels} texels between overlapping regions of one \
                             {format:?} texture"
                        ),
                        "Metal's copy reads as memmove's for regions up to a few hundred \
                         texels and races past that; the runtime gives memmove's answer"
                            .to_owned(),
                    )
                });
        }
        let written = (copies.iter())
            .map(|copy| (copy.dst.aspect.into(), Written::of(&copy.dst)))
            .collect();
        self.scrub(destination, written)
    }

    /// Record checked copies between two textures of which one or both are over buffers, on the
    /// buffers' bytes (the module's note): rows between two textures over buffers, else a
    /// buffer-image copy of each slice between the bytes and the other texture's image, as
    /// `copy_from_buffer_to_texture` and `copy_from_texture_to_buffer` record one.
    fn texel_copies(
        &mut self,
        call: &'static str,
        source: &Texture,
        destination: &Texture,
        copies: &[ImageCopyRegion],
    ) -> Result<(), Error> {
        let copy_problem = |problem| invalid(call, Problem::Copy(problem));
        match (source.buffer_texels(), destination.buffer_texels()) {
            (Some(from), Some(to)) => {
                let (unit, mut src, mut dst) = (from.unit(), Vec::new(), Vec::new());
                for copy in copies {
                    src.extend(texel_rows(&from.layout, &unit, &copy.src));
                    dst.extend(texel_rows(&to.layout, &unit, &copy.dst));
                }
                self.rows_copy(&from.buffer, &to.buffer, &RowsCopy::new(&src, &dst))
            }
            (Some(texels), None) | (None, Some(texels)) => {
                // Rows that overlap (a rotated texture's) are no layout a buffer-image copy
                // states.
                if texels.rows_overlap() {
                    return Err(Error::NotImplemented(RuntimeGap::RotatedTexture));
                }
                let into_texture = source.over_buffer();
                let texture = if into_texture { destination } else { source };
                let record = texture.record();
                let (unit, buffer) = (texels.unit(), &texels.buffer);
                // The image's own unit: of one size, in its aspect (a depth or stencil format's).
                let image_unit = self::unit(call, texture, BlitOption::None)?;
                for copy in copies {
                    let (bytes, image) = if into_texture {
                        (&copy.src, &copy.dst)
                    } else {
                        (&copy.dst, &copy.src)
                    };
                    for layer in 0..image.layers {
                        let region = TextureRegion {
                            level: image.level - record.base_level,
                            slice: image.layer + layer - record.base_layer,
                            origin: image.origin,
                            size: image.size,
                        };
                        let layout =
                            texel_layout(&texels.layout, &unit, bytes, bytes.layer + layer);
                        let plan = plan_buffer_copy(
                            record,
                            &region,
                            &image_unit,
                            &layout,
                            buffer.length(),
                            buffer.start(),
                            into_texture,
                        )
                        .map_err(copy_problem)?;
                        if into_texture {
                            texture.degrade_depth_upload(call, image_unit.aspect, true, true);
                            texture.degrade_depth_denormals(call, image_unit.aspect, false, true);
                        }
                        self.buffer_texture_copy(buffer, texture, plan, into_texture)?;
                    }
                }
                Ok(())
            }
            (None, None) => Err(internal("texel copies between two images")),
        }
    }

    /// Record `copy` from `source`'s bytes into `destination`'s, a copy that names a texture
    /// over a buffer (the module's note): its rows in one command, or, where a source row and a
    /// destination row of one buffer share bytes, through a scratch buffer of the encoder, the
    /// whole source read before any of it is written, as Metal moves them.
    fn rows_copy(
        &mut self,
        source: &Buffer,
        destination: &Buffer,
        copy: &RowsCopy,
    ) -> Result<(), Error> {
        // Read as a blit's (`host_residency.rs`): the command buffer holds the imports.
        let (src, src_base) = self.core().blit_at(source, 0)?;
        let (dst, dst_base) = self.core().blit_at(destination, 0)?;
        let device = self.core().device().clone();
        // One buffer's bytes, or two no-copy buffers' made over the same bytes, which lie in one
        // import (`host_aliases.rs`): one `VkBuffer` from one byte 0.
        if (src, src_base) == (dst, dst_base) && copy.meets() {
            let length = copy.length();
            let scratch = self
                .core()
                .device()
                .own_buffer(length, ResourceOptions::private())?;
            let (through, through_base) = scratch.at(0);
            let (mut into, mut out_of) = (Vec::new(), Vec::new());
            let mut packed = through_base;
            for row in &copy.rows {
                into.push(vk::BufferCopy {
                    src_offset: src_base + row.src,
                    dst_offset: packed,
                    size: row.size,
                });
                out_of.push(vk::BufferCopy {
                    src_offset: packed,
                    dst_offset: dst_base + row.dst,
                    size: row.size,
                });
                packed += row.size;
            }
            let into = RowCopies::new(&device, src, through, into);
            let uses = [
                reads_by(source, copy.src_span.clone(), into.read()),
                writes_by(&scratch, 0..length, into.write()),
            ];
            self.command_in(into.stages(), &uses, |device, raw| {
                // SAFETY: the encoder's command buffer is recording; the buffers are live and
                // retained until it completes, the rows lie inside them, and the scratch buffer
                // is another buffer.
                unsafe { into.record(device, raw) };
            })?;
            let out_of = RowCopies::new(&device, through, dst, out_of);
            let uses = [
                reads_by(&scratch, 0..length, out_of.read()),
                writes_by(destination, copy.dst_span.clone(), out_of.write()),
            ];
            self.command_in(out_of.stages(), &uses, |device, raw| {
                // SAFETY: as above.
                unsafe { out_of.record(device, raw) };
            })?;
            for sets in [into.retained(), out_of.retained()].into_iter().flatten() {
                self.core().retain(sets);
            }
            self.core().retain(scratch);
            let (from, to) = (copy.src_span.clone(), copy.dst_span.clone());
            self.core()
                .device()
                .degrade(Degradation::OverlappingTexelCopy, || {
                    let object = format!(
                        "a copy of texel rows from bytes {from:?} to bytes {to:?} of one buffer"
                    );
                    let reason = "a source row and a destination row share bytes, which Vulkan's \
                                  copy forbids";
                    (object, reason.to_owned())
                });
        } else {
            let regions: Vec<vk::BufferCopy> = copy
                .rows
                .iter()
                .map(|row| vk::BufferCopy {
                    src_offset: src_base + row.src,
                    dst_offset: dst_base + row.dst,
                    size: row.size,
                })
                .collect();
            let rows = RowCopies::new(&device, src, dst, regions);
            let uses = [
                reads_by(source, copy.src_span.clone(), rows.read()),
                writes_by(destination, copy.dst_span.clone(), rows.write()),
            ];
            self.command_in(rows.stages(), &uses, |device, raw| {
                // SAFETY: the encoder's command buffer is recording; the buffers are live and
                // retained until it completes, the rows lie inside them, the destination rows are
                // disjoint (`RowsCopy::new`), and in one buffer no source row meets a destination
                // row.
                unsafe { rows.record(device, raw) };
            })?;
            if let Some(sets) = rows.retained() {
                self.core().retain(sets);
            }
        }
        self.core().retain(source.clone());
        if source != destination {
            self.core().retain(destination.clone());
        }
        Ok(())
    }

    /// `generateMipmapsForTexture:`: each level after the texture's first from the one above it,
    /// by bilinear (in 3D, trilinear) filtering, as a chain of blits; every slice of an array or
    /// cube texture, and on a view its own levels and slices.
    ///
    /// Every colour format Metal filters but `B5G6R5Unorm` and `BGR5A1Unorm`, extended range
    /// included, is computed as Metal's filter computes it, on every device with `shaderInt64`
    /// and without a blit (`mipmap_filter.rs`): a scratch buffer of the levels, one compute
    /// dispatch per level, and copies in and out, by the rule of the device's Metal level
    /// ([`MetalLevel::mipmap_rule`](crate::MetalLevel::mipmap_rule)). Under macOS 27.0.1 every
    /// level is an Apple M3 Max's filter of the level above. Under macOS 26.7 the first computed
    /// level, and every level of a 3D texture, is an Apple M3 Max's answer; below the first level
    /// of a 2D chain, an array or a cube whose M3 Max passes are not known an Apple M4's stands in,
    /// announced once per device (`mipmaps-as-apple-m4`), as for the formats an M3 Max converts
    /// through a half (`ABGR4Unorm`, `A1BGR5Unorm`, extended range) and, in 3D, those summed as
    /// half floats.
    ///
    /// Elsewhere (those two formats, a device without `shaderInt64`, limits that do not fit the
    /// chain) Metal's answers differ between Apple GPUs, so the runtime checks this device's
    /// blits per pixel format at the format's first `generate_mipmaps` (on the device's internal
    /// submission, synchronously) and keeps the verdict ([`Device::mipmap_generation`]): an
    /// accepted format's mipmaps lie within the spread measured between an Apple M3 Max and an
    /// Apple M4, not at Metal's exact answer, a near-equivalent announced once per device
    /// (`mipmaps-within-spread`).
    ///
    /// An integer format, which Metal's API validation stops, takes plain Metal's answer exactly,
    /// on every device (an M4 under macOS 27.0.1): each level's texel `i` is the level above's
    /// texel `floor((2i + 1) W / 2w)` on each axis (`W` and `w` the two levels' sizes on it),
    /// copied as its bytes through a scratch buffer.
    ///
    /// A texture or view of one level (every texture over a buffer has one), of any format, and
    /// a depth or stencil texture of any levels, are left as they are, as Metal's plain call
    /// leaves them; Metal's API validation asserts on them.
    ///
    /// Refused: what Metal stops with no defined answer (`Invalid` with a
    /// [`MipmapProblem`](super::MipmapProblem): a compressed format); a format the device cannot
    /// filter linearly (`Unsupported`, naming the format); a format whose check failed or whose
    /// Metal answer is not measured, a 1D texture, a view of another format than its texture (of
    /// another texel size, for an integer format), an integer texture whose image takes no
    /// transfers or whose scratch buffer would pass the device's `max_buffer_length`, and a
    /// texture of an extended-range format the filter cannot compute on the device (its image
    /// holds raw codes, which no blit filters: `Unsupported`, `NotImplemented` for
    /// `MipmapGeneration`); a texture of another device (`Invalid`).
    ///
    /// [`Device::mipmap_generation`]: super::Device::mipmap_generation
    pub fn generate_mipmaps(&mut self, texture: &Texture) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::GenerateMipmaps(texture.clone()),
            |this| {
                // A view that reads its image in the image's own format generates them as a texture
                // of that format over the same levels: the same bits, filtered alike
                // (`bind::mipmap_filter`'s layouts pair the formats `metal::format::swizzle_over`
                // pairs).
                match texture.bound_twin() {
                    Some(twin) => this.mipmaps(&twin),
                    None => this.mipmaps(texture),
                }
            },
        )
    }

    /// [`generate_mipmaps`](Self::generate_mipmaps) of `texture`, logged.
    fn mipmaps(&mut self, texture: &Texture) -> Result<(), Error> {
        const CALL: &str = "generate_mipmaps";
        // One level, of any format, or a depth or stencil texture: Metal's plain call leaves the
        // texture as it is, over a buffer or not, and its API validation asserts. Every texture
        // over a buffer has one level.
        self.core().check_device(CALL, texture.device())?;
        texture.check_bytes_reached(CALL)?;
        let record = texture.record();
        let gap = || Error::Unsupported(Unsupported::not_implemented(Feature::MipmapGeneration));
        // A texture of one level, of an extended-range format too: left as it is.
        if leaves_as_is(record) {
            return Ok(());
        }
        self.check_texture(CALL, texture)?;
        let device = self.core().device().clone();
        let facts = device
            .shared()
            .textures
            .facts(device.compile_device().profile());
        let blits = match plan_mipmaps(texture.record(), texture.image_plan()?, &facts) {
            // An integer format: plain Metal picks texels, which copies give exactly
            // (`mipmap_pick.rs`).
            Ok(MipmapPlan::Picks(picks)) => {
                return self.pick_mipmaps(texture, &picks, read(), write());
            }
            Ok(MipmapPlan::Blits(blits)) => Ok(blits),
            Err(MipmapRefusal::Invalid(problem)) => {
                return Err(invalid(CALL, Problem::Mipmap(problem)));
            }
            Err(MipmapRefusal::Unsupported(unsupported)) => Err(Error::Unsupported(unsupported)),
        };
        // Metal's filter, computed (`mipmap_filter.rs`), where the device runs it; no blit, so
        // no check. An extended-range format, which no blit filters, is refused without it.
        let own_format = texture.record().pixel_format == texture.image_plan()?.pixel_format;
        if own_format && self.filter_mipmaps(texture, &facts, read(), write())? {
            return Ok(());
        }
        if not_implemented(texture.record().pixel_format) {
            return Err(gap());
        }
        let blits = blits?;
        let one_dimensional = matches!(
            texture.record().texture_type,
            TextureType::Type1D | TextureType::Type1DArray
        );
        if one_dimensional || !own_format {
            return Err(gap());
        }
        if !device.mipmaps_accepted(texture)? {
            return Err(gap());
        }
        let (image, aspects) = (texture.raw_image()?, texture_aspects(texture.image_plan()?));
        device.shared().counts.pool_compiled(
            crate::work_counts::PoolCompiled::ImageBlit,
            blits.len() as u64,
        );
        for blit in &blits {
            let layers = blit.base_layer..blit.base_layer + blit.layers;
            let uses = [
                texture.image_use(
                    blit.src_level..blit.src_level + 1,
                    layers.clone(),
                    read(),
                    false,
                )?,
                texture.image_use(blit.dst_level..blit.dst_level + 1, layers, write(), true)?,
            ];
            self.command(&uses, |device, raw| {
                record_blit(device, raw, image, aspects, blit);
            })?;
        }
        self.core().retain(texture.clone());
        let written = blits
            .iter()
            .map(|blit| {
                let written = Written {
                    level: blit.dst_level,
                    layers: blit.base_layer..blit.base_layer + blit.layers,
                    origin: [0; 3],
                    size: blit.dst_extent,
                };
                (aspects, written)
            })
            .collect();
        self.scrub(texture, written)?;
        let pixel_format = texture.record().pixel_format;
        device.degrade(Degradation::MipmapsWithinSpread, || {
            (
                format!("generate_mipmaps of a {} texture", pixel_format.name()),
                "the device's blit chain gives the format's mipmaps within Metal's measured \
                 spread, which the runtime checked at the format's first generate_mipmaps"
                    .to_owned(),
            )
        });
        Ok(())
    }

    /// Metal's check of a level and slice of `texture` named by a hint or a synchronization.
    fn check_level_slice(
        &mut self,
        call: &'static str,
        texture: &Texture,
        slice: u64,
        level: u64,
    ) -> Result<(), Error> {
        self.check_texture(call, texture)?;
        check_slice_level(call, texture, slice, level)
    }

    /// `synchronizeTexture:slice:level:`: the CPU sees the GPU's writes to a level and slice of a
    /// `Managed` texture once the command buffer completes. Managed memory behaves as shared
    /// memory here (and on Apple silicon), so nothing more is recorded. On a texture of another
    /// storage mode it records nothing either, whatever the level, as plain Metal does
    /// (`Shared` and `Private` textures, level 0 and a level past the texture;
    /// Metal's validation stops it).
    ///
    /// Refused (`Invalid`): a managed texture's slice or level past the texture; a texture of
    /// another device.
    pub fn synchronize_texture(
        &mut self,
        texture: &Texture,
        slice: u64,
        level: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::SynchronizeTexture(texture.clone(), slice, level),
            |this| {
                const CALL: &str = "synchronize_texture";
                this.check_texture(CALL, texture)?;
                if texture.storage_mode() != StorageMode::Managed {
                    return Ok(());
                }
                check_slice_level(CALL, texture, slice, level)
            },
        )
    }

    /// `optimizeContentsForGPUAccess:`: a hint; Vulkan images keep one layout for every access
    /// here, so it records nothing. Refused (`Invalid`) for a texture of another device.
    pub fn optimize_contents_for_gpu_access(&mut self, texture: &Texture) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::OptimizeForGpu(texture.clone()),
            |this| this.check_texture("optimize_contents_for_gpu_access", texture),
        )
    }

    /// `optimizeContentsForGPUAccess:slice:level:`: as
    /// [`optimize_contents_for_gpu_access`](Self::optimize_contents_for_gpu_access), and refused
    /// (`Invalid`) for a slice or level past the texture.
    pub fn optimize_contents_for_gpu_access_slice_level(
        &mut self,
        texture: &Texture,
        slice: u64,
        level: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::OptimizeForGpuSliceLevel(texture.clone(), slice, level),
            |this| {
                const CALL: &str = "optimize_contents_for_gpu_access_slice_level";
                this.check_level_slice(CALL, texture, slice, level)
            },
        )
    }

    /// `optimizeContentsForCPUAccess:`: a hint; it records nothing. Refused (`Invalid`) for a
    /// texture of another device.
    pub fn optimize_contents_for_cpu_access(&mut self, texture: &Texture) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::OptimizeForCpu(texture.clone()),
            |this| this.check_texture("optimize_contents_for_cpu_access", texture),
        )
    }

    /// `optimizeContentsForCPUAccess:slice:level:`: as
    /// [`optimize_contents_for_cpu_access`](Self::optimize_contents_for_cpu_access), and refused
    /// (`Invalid`) for a slice or level past the texture.
    pub fn optimize_contents_for_cpu_access_slice_level(
        &mut self,
        texture: &Texture,
        slice: u64,
        level: u64,
    ) -> Result<(), Error> {
        self.logged(
            |_| BlitCall::OptimizeForCpuSliceLevel(texture.clone(), slice, level),
            |this| {
                const CALL: &str = "optimize_contents_for_cpu_access_slice_level";
                this.check_level_slice(CALL, texture, slice, level)
            },
        )
    }
}

/// Refuse (`Invalid`) a slice or level `call` names past `texture`'s.
pub(super) fn check_slice_level(
    call: &'static str,
    texture: &Texture,
    slice: u64,
    level: u64,
) -> Result<(), Error> {
    let record = texture.record();
    if slice >= u64::from(record.layer_count) || level >= u64::from(record.level_count) {
        return Err(invalid(call, Problem::Copy(CopyProblem::Region)));
    }
    Ok(())
}

/// The plan's view of two textures: a texture with a depth twin by its twin, which holds its
/// texels as a depth image (`texture_copies` writes its image too); a texture buffer by none.
fn pair<'a>(source: &'a Texture, destination: &'a Texture) -> Result<CopyPair<'a>, Error> {
    let image = |texture: &'a Texture| match texture.texels() {
        Some(_) => Ok(None),
        None => texture.depth_plan().map(Some),
    };
    Ok(CopyPair {
        src: source.record(),
        src_image: image(source)?,
        dst: destination.record(),
        dst_image: image(destination)?,
    })
}

/// An image holding a texture's texels for a copy between textures: the image, the aspect the
/// texels are in, and its Vulkan format.
#[derive(Clone, Copy, Debug)]
struct Held {
    image: vk::Image,
    aspect: ImageAspectFlags,
    format: Format,
}

impl Held {
    /// Whether Vulkan copies this image's texels into `to`'s: both in colour (formats of one
    /// size, checked by the plan, both alpha formats or neither, `image_plan::copies_between`:
    /// an `A8Unorm` view's image is its `R8Unorm` texture's), or in the same depth or stencil
    /// aspects of one format.
    fn copies_into(&self, to: &Held) -> bool {
        self.aspect == to.aspect
            && match self.aspect == ImageAspectFlags::COLOR {
                true => copies_between(self.format, to.format),
                false => self.format == to.format,
            }
    }

    /// `copy` from this image to `to`, each in its aspect.
    fn region(&self, to: &Held, copy: &ImageCopyRegion) -> ImageCopyRegion {
        ImageCopyRegion {
            src: ImageRegion {
                aspect: self.aspect,
                ..copy.src
            },
            dst: ImageRegion {
                aspect: to.aspect,
                ..copy.dst
            },
        }
    }

    /// The bytes of a texel of this image in a buffer, where they are Metal's copy unit: a
    /// colour format's texel, 32-bit float or 16-bit normalized depth, a stencil byte.
    fn texel_bytes(&self) -> Option<u32> {
        if self.aspect == ImageAspectFlags::COLOR {
            self.format.block().map(|block| block.size)
        } else if self.aspect == ImageAspectFlags::DEPTH {
            depth_bytes(self.format)
        } else if self.aspect == ImageAspectFlags::STENCIL {
            Some(1)
        } else {
            None
        }
    }
}

/// The images holding `texture`'s texels, the one its texels are read from first: a texture
/// with a depth twin holds them in colour in its image and in `aspect` (depth) in its twin; any
/// other texture in `aspect` of its image.
fn held(texture: &Texture, aspect: ImageAspectFlags) -> Result<Vec<Held>, Error> {
    let image = Held {
        image: texture.raw_image()?,
        aspect,
        format: texture.image_plan()?.format,
    };
    Ok(match texture.twin_image() {
        Some(twin) => vec![
            Held {
                aspect: ImageAspectFlags::COLOR,
                ..image
            },
            Held {
                image: twin,
                aspect,
                format: texture.depth_plan()?.format,
            },
        ],
        None => vec![image],
    })
}

/// A copy between textures through a scratch buffer of the encoder: the buffer's length, the
/// regions from the source's image into it, tightly packed one after another, and from it into
/// each image that takes them.
struct Staging {
    length: u64,
    out_of: Vec<vk::BufferImageCopy>,
    into: Vec<(Held, Vec<vk::BufferImageCopy>)>,
}

/// `copies` from `from` into each of `into` through a scratch buffer: Vulkan copies a depth or
/// stencil aspect only into the same aspect of the same format, so the bytes go out of the
/// source and back into each destination image by buffer copies, which carry Metal's copy units
/// as Vulkan lays them out in a buffer; and a compressed pair of regions of different sizes, which
/// no one extent states, goes the same way, the blocks both cover out of one and into the other.
/// `block` is the texel extent of `from`'s blocks ([`image_block_extent`]; 1 x 1 uncompressed),
/// each a texel's bytes in the buffer. Single-sampled textures only: multisampled ones, which
/// Vulkan copies into no buffer, copy through colour images (`sample_copy.rs`). Refused
/// `NotImplemented(Format)` where a side's buffer texels are not the copy unit: both aspects of a
/// combined format, and 24-bit depth, whose bytes in a buffer Metal's reference GPUs give no
/// answer for (an Apple M3 Max and an M4 make no `Depth24Unorm_Stencil8` texture).
fn through_buffer(
    copies: &[ImageCopyRegion],
    from: &Held,
    block: [u32; 2],
    into: &[Held],
) -> Result<Staging, CopyRefusal> {
    let not_implemented =
        || CopyRefusal::Unsupported(Unsupported::not_implemented(Feature::Format));
    let bytes = from.texel_bytes().ok_or_else(not_implemented)?;
    if into.iter().any(|to| to.texel_bytes() != Some(bytes)) {
        return Err(not_implemented());
    }
    // Each copy starts where Vulkan takes either side's bytes: at a multiple of the texel block,
    // and of 4 for a depth or stencil aspect.
    let align = u64::from(if bytes.is_multiple_of(4) {
        bytes
    } else {
        bytes * 4
    });
    let mut length: u64 = 0;
    let mut out_of = Vec::with_capacity(copies.len());
    let mut regions = vec![Vec::with_capacity(copies.len()); into.len()];
    for copy in copies {
        length = length.next_multiple_of(align);
        let [width, height, depth] = copy.src.size;
        out_of.push(vk::BufferImageCopy {
            buffer_offset: length,
            ..buffer_image_copy(&ImageRegion {
                aspect: from.aspect,
                ..copy.src
            })
        });
        for (to, regions) in into.iter().zip(&mut regions) {
            regions.push(vk::BufferImageCopy {
                buffer_offset: length,
                ..buffer_image_copy(&ImageRegion {
                    aspect: to.aspect,
                    ..copy.dst
                })
            });
        }
        let blocks = [width.div_ceil(block[0]), height.div_ceil(block[1])];
        length += [blocks[0], blocks[1], depth, copy.src.layers]
            .iter()
            .map(|&n| u64::from(n))
            .product::<u64>()
            * u64::from(bytes);
    }
    Ok(Staging {
        length,
        out_of,
        into: into.iter().copied().zip(regions).collect(),
    })
}
