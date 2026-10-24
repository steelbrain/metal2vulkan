//! Copies of Apple's two-plane extended-range textures, `RGB10A8_2P_XR10` and its `_sRGB` twin.
//!
//! **Metal's answers** (an Apple M4 under macOS 27.0.1; a 48 x 20 texture of
//! random texels, plane 0's words with bits 30 and 31 set at random and random alpha bytes, each
//! case in its own process, plain and under Metal's validation layer, which stops none of them
//! but a copy between the two formats):
//!
//! - `getBytes` and `replaceRegion`, and (at a width that is a multiple of 4) blits from the
//!   texture into a buffer and from a buffer into it, move plane 0 alone: each texel's color
//!   word, 4 bytes, at the caller's rows and
//!   images (rows of 4, 5 and 8 bytes a texel, images twice the plane apart, a 17 x 9 or
//!   13 x 11 region at an origin, a buffer offset of 256). The alpha plane is neither read nor
//!   written, and no byte between the rows is touched.
//! - A blit between two such textures moves both planes' bytes as if their rows were
//!   `⌊4 w / 16⌋ · 16` bytes apart, where plane 0's rows lie `align(4 w, 16)` and the alpha
//!   plane's `align(w, 16)` bytes apart (an Apple M4 under macOS 27.0.1, every width from 1 to 21
//!   and 23 to 257 sampled; `bind::plane_copy` states the model): at widths that are a multiple of
//!   4 plane 0 copies exactly and the alpha bytes scatter over other rows; at other widths plane
//!   0's texels are lost or moved too. It is Metal's defect, which depends on the planes' layout in
//!   memory. A copy into the other format of the pair copies the same way.
//! - At widths that are not a multiple of 4, blits between a buffer and the texture address its
//!   rows the same way, so they move texels between rows and the bytes between them too
//!   (`two_planes/buffer_rows.rs` states the rows, measured on an Apple M3 Max under macOS
//!   27.0.1); `getBytes` and `replaceRegion` move the texels at every width.
//!
//! **Held** as one `R32G32_UINT` image (`metal::format::ExtendedRange::Planes`): each texel's
//! color word, then its alpha byte in the second word.
//!
//! - **CPU copies** move the region's texels, 8 bytes each, repacked on the host ([`narrow`],
//!   [`widen`]): `getBytes` takes each texel's first word; `replaceRegion` reads the region's
//!   texels first and writes them back with their first words replaced, so the alpha words stay.
//!   They move on the host where the image copies there (`texture/host_copy.rs`), with no
//!   submission and so no wait, as Metal's `getBytes` and `replaceRegion` of such a texture wait
//!   for no GPU work; else through a staging buffer, by a submission each waited for.
//! - **Blits between a buffer and the texture** go through a scratch buffer of the encoder in
//!   bands of rows ([`bands`]), one blit command: per band, one copy of its texels out of the
//!   image, one dispatch of `bind::plane_words_module` moving each texel's first word into the
//!   buffer's bytes or out of them, and, into the texture, one copy of the texels back.
//! - **Blits between a buffer and the texture** at a width that is not a multiple of 4: a 2D
//!   texture or array of one level is blitted at Metal's rows through a scratch buffer laid out
//!   as plane 0 (`buffer_rows.rs`), announced (`two-plane-buffer-blit`, precision) where rows meet
//!   in a blit into it and where a blit out of a texture of the device's memory reads bytes
//!   between its rows that a fill or a blit wrote; any other such texture moves the texels at the
//!   texture's rows, announced the same way.
//! - **Blits between two such textures** of one level and one slice, 2D, each level within a
//!   storage buffer, move both planes' bytes as Metal's blit does (`modelled_plane_copies`):
//!   one command of the encoder, both levels copied into a scratch buffer, the kernel
//!   `bind::plane_copy_module` over the destination's texels, and its texels copied back. A copy
//!   whose rows overlap in the destination with other bytes than each other's (Metal's blit
//!   writes them in an order that varies; the kernel keeps the later row's), or of a texture
//!   narrower than 4 texels (Metal's blit reads alpha bytes from outside the planes; the kernel
//!   reads 0), or reading bytes outside the source's texels that an earlier blit wrote there
//!   (Metal's blit keeps them and reads them back; the kernel holds the texels alone and reads
//!   0) or that a heap's texture's neighbors hold, is announced once per device
//!   (`two-plane-texture-copy`, precision). Other copies
//!   between such textures (more levels or slices, one texture's own, a level past a storage
//!   buffer) copy both words of each texel, announced the same way; open.
//! - **Blits into a heap's texture** whose planes lie in the heap's bytes write the heap's bytes as
//!   Metal's blit does, past the planes too, from a source's heap bytes or its texels
//!   (`heap_bytes.rs`).
//! - **Blits with a texture over a surface's planes** address its planes at rows of plane 0's GPU
//!   rows, as Metal's blit does: into it, the surface's bytes are written as a heap's are, past the
//!   planes too but not past the surface; out of it, its bytes are read as they lie, 0 past the
//!   surface. Such a texture is its bytes to a blit, so a copy not modelled (within one texture,
//!   between two over one buffer, past a storage buffer) is refused rather than copied as texels.
//!
//! - **A 3D texture** lies on Metal as one 2D slice's planes, every depth slice over the first's
//!   (an Apple M3 Max under macOS 27.0.1, `Shared` and `Private`, in a heap and of its own
//!   memory): a blit from a buffer writes each image of its region over texel `(x, y)` of the
//!   first slice in turn, so the last image stays; a blit into a buffer reads texel `(0, 0, 0)`'s
//!   color word at every texel of every image; `replaceRegion` writes the region's first image
//!   at the first slice and `getBytes` reads that slice into the first image, the caller's other
//!   images as they were. The runtime holds such a texture's texels in its image's first depth
//!   slice alone (`BlitCommandEncoder::plane_first_texel_copy`, `HostCopy::first_image`, and the
//!   shader's reads and writes, `lower::texture::emit::planes_volume`). A copy between it and
//!   another two-plane texture copies its texels at the first slice, announced
//!   (`two-plane-texture-copy`): Metal's copy out of it reads texel `(0, 0, 0)` at every texel,
//!   and into it at a width of 12 wrote nothing, not modelled.
//!
//! Refused `NotImplemented(Format)` naming the format, Metal's answer not measured: a layout
//! whose rows or images overlap or lie below a row or an image apart, or whose offset, rows or
//! images are off 4 bytes; a copy between such a texture and one of another pixel format.

use super::blit::{BlitCommandEncoder, TRANSFER};
use super::blit_plan::{BufferLayout, ImageCopyRegion};
use super::copy_plan::{CopyUnit, HostLayout, ImageRegion};
use super::degradation::Degradation;
use super::error::{internal, Error};
use super::owned_kernel::{OwnedModule, OwnedSets};
use super::sparse::emulated::Written;
use super::texture::{buffer_image_copy, Texture};
use super::tracker::{Access, Barrier, Use, UseRange};
use super::values::ResourceOptions;
use super::{Buffer, Device};
use ash::vk;
use metal2vulkan::bind::{PlaneCopyModule, PlaneRows, PlaneSource, PlaneWordsModule};
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::ExtendedRange;
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::ImageAspectFlags;
use std::ops::Range;

mod buffer_rows;
mod heap_bytes;

#[cfg(test)]
pub(super) use buffer_rows::later_row_stands;
pub(super) use buffer_rows::{between_rows, buffer_blit_rows, row_spans, rows_held};

use heap_bytes::{stage_planes, BytePlanes, Staged};

/// `copy` between two such textures, one of them 3D, at the first depth slice and layer of each
/// (the module docs): one image of the region, at `z` 0.
pub(super) fn first_slices(copy: &ImageCopyRegion) -> ImageCopyRegion {
    let first = |r: &ImageRegion| ImageRegion {
        layers: 1,
        origin: [r.origin[0], r.origin[1], 0],
        size: [r.size[0], r.size[1], 1],
        ..*r
    };
    ImageCopyRegion {
        src: first(&copy.src),
        dst: first(&copy.dst),
    }
}

/// Whether `pf` is a two-plane extended-range format.
pub(super) fn holds_planes(pf: PixelFormat) -> bool {
    matches!(ExtendedRange::of(pf), Some(ExtendedRange::Planes { .. }))
}

/// Metal's copy unit of a CPU copy of such a texture, or of a blit between it and a buffer:
/// plane 0's color word.
pub(super) const UNIT: CopyUnit = CopyUnit {
    bytes: 4,
    extent: [1, 1],
    aspect: ImageAspectFlags::COLOR,
};

/// Bytes of a texel of the image: the color word and the alpha word.
const TEXEL_BYTES: u64 = 8;

/// The copy unit of a blit between two such textures: the image's texel, both words.
pub(super) const TEXEL_UNIT: CopyUnit = CopyUnit {
    bytes: TEXEL_BYTES as u32,
    extent: [1, 1],
    aspect: ImageAspectFlags::COLOR,
};

/// The refusal of a copy of a texture of `pf` whose answer Metal's was not measured for.
pub(super) fn unmeasured(pf: PixelFormat) -> Unsupported {
    Unsupported::not_implemented(Feature::Format).naming_format(pf)
}

/// Whether a layout of `rows` rows of `row_bytes` and `images` images, `bytes_per_row` and
/// `bytes_per_image` apart, from `offset`, is one Metal's answer was measured for: rows and
/// images apart by 4-byte multiples at least a row and an image of the region (either ignored
/// where there is one), from a 4-byte multiple.
pub(super) fn measured_layout(
    row_bytes: u64,
    rows: u64,
    images: u64,
    bytes_per_row: u64,
    bytes_per_image: u64,
    offset: u64,
) -> bool {
    let rows_ok = rows <= 1 || (bytes_per_row >= row_bytes && bytes_per_row.is_multiple_of(4));
    // An image of more bytes than `u64::MAX` is longer than any `bytes_per_image`.
    let image =
        ((rows.max(1) - 1).checked_mul(bytes_per_row)).and_then(|rows| rows.checked_add(row_bytes));
    let images_ok = images <= 1
        || image.is_some_and(|image| bytes_per_image >= image && bytes_per_image.is_multiple_of(4));
    rows_ok && images_ok && offset.is_multiple_of(4)
}

/// Whether the CPU copy laid out as `host` is one Metal's answer was measured for
/// ([`measured_layout`]; a single row's pitch at least the row too).
pub(super) fn measured_host_layout(host: &HostLayout) -> bool {
    measured_layout(
        host.row_bytes,
        host.rows,
        host.images,
        host.bytes_per_row,
        host.bytes_per_image,
        0,
    ) && host.bytes_per_row >= host.row_bytes
}

/// Each texel's color word of `texels` (the image's, 8 bytes a texel) into `words` (4 bytes a
/// texel), as many as `words` holds.
pub(super) fn narrow(texels: &[u8], words: &mut [u8]) {
    for (word, texel) in words
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(texels.as_chunks::<8>().0.iter())
    {
        word.copy_from_slice(&texel[..4]);
    }
}

/// Each color word of `words` (4 bytes a texel) over the first word of its texel in `texels` (8
/// bytes a texel), the second words as they are.
pub(super) fn widen(words: &[u8], texels: &mut [u8]) {
    for (word, texel) in words
        .as_chunks::<4>()
        .0
        .iter()
        .zip(texels.as_chunks_mut::<8>().0.iter_mut())
    {
        texel[..4].copy_from_slice(word);
    }
}

/// One band of a blit's rows (the module docs): `rows` rows from `first_row` of image `image`
/// of the region, whose buffer bytes the words binding holds from `window.start` (a byte of the
/// buffer's `VkBuffer`, a multiple of the device's storage-buffer offset alignment), its first
/// texel at word `first_word` there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Band {
    pub(super) image: u32,
    pub(super) first_row: u32,
    pub(super) rows: u32,
    pub(super) window: Range<u64>,
    pub(super) first_word: u32,
}

/// What the bands keep to: the device's storage-buffer offset alignment and largest range, its
/// workgroups along x, and the scratch's largest band of texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BandLimits {
    pub(super) align: u64,
    pub(super) max_range: u64,
    pub(super) max_groups: u32,
    pub(super) max_scratch: u64,
}

/// The scratch's largest band of texels: 32 MiB.
const MAX_SCRATCH: u64 = 32 << 20;

/// The bands of a region of `size` texels whose buffer is laid out as `layout` from byte `base`
/// of its `VkBuffer` (a measured layout, [`measured_layout`]): each image's rows in order, as
/// many a band as keep its texels and its buffer window within `maxStorageBufferRange` and the
/// scratch's limit, and its texels within `maxComputeWorkGroupCount[0]` workgroups. `None` where
/// one row does not fit.
pub(super) fn bands(
    size: [u32; 3],
    layout: &BufferLayout,
    base: u64,
    limits: &BandLimits,
) -> Option<Vec<Band>> {
    let width = u64::from(size[0]);
    let (row_bytes, texel_row) = (4 * width, TEXEL_BYTES * width);
    let align = limits.align.max(1);
    let per_dispatch = u64::from(PlaneWordsModule::LOCAL_SIZE) * u64::from(limits.max_groups);
    let mut out = Vec::new();
    for image in 0..size[2] {
        let mut row = 0;
        while row < size[1] {
            let start = base
                .checked_add(layout.offset)?
                .checked_add(u64::from(image).checked_mul(layout.bytes_per_image)?)?
                .checked_add(u64::from(row).checked_mul(layout.bytes_per_row)?)?;
            let aligned = start / align * align;
            let lead = start - aligned;
            if lead + row_bytes > limits.max_range {
                return None;
            }
            let mut rows = u64::from(size[1] - row)
                .min(limits.max_range.min(limits.max_scratch) / texel_row)
                .min(per_dispatch / width);
            if let Some(more) =
                (limits.max_range - lead - row_bytes).checked_div(layout.bytes_per_row)
            {
                rows = rows.min(more + 1);
            }
            if rows == 0 {
                return None;
            }
            let end = start + (rows - 1) * layout.bytes_per_row + row_bytes;
            out.push(Band {
                image,
                first_row: row,
                rows: rows as u32,
                window: aligned..end,
                first_word: (lead / 4) as u32,
            });
            row += rows as u32;
        }
    }
    Some(out)
}

impl BlitCommandEncoder {
    /// Record a blit between `buffer`, laid out as `layout` (a measured layout), and `region`
    /// of the two-plane `texture` (`into_texture` from the buffer): the module docs' bands, one
    /// command of the encoder. The caller made Metal's checks of the copy.
    pub(super) fn plane_words_copy(
        &mut self,
        buffer: &Buffer,
        texture: &Texture,
        region: &ImageRegion,
        layout: &BufferLayout,
        into_texture: bool,
    ) -> Result<(), Error> {
        self.plane_words_moves(buffer, texture, region, layout, (into_texture, false))
    }

    /// Record a blit out of the two-plane 3D `texture` into `buffer`, laid out as `layout` (a
    /// measured layout), of `region`: texel `(0, 0, 0)`'s color word at every texel of every
    /// image, as Metal's blit reads such a texture (an Apple M3 Max under macOS 27.0.1: a blit
    /// of one and of two images at slices 0, 1 and 3 of a `Shared` heap's 16 x 8 x 4 texture),
    /// one command of the encoder. The caller made Metal's checks of the copy.
    pub(super) fn plane_first_texel_copy(
        &mut self,
        buffer: &Buffer,
        texture: &Texture,
        region: &ImageRegion,
        layout: &BufferLayout,
    ) -> Result<(), Error> {
        self.plane_words_moves(buffer, texture, region, layout, (false, true))
    }

    /// [`plane_words_copy`](Self::plane_words_copy), its texels out of the image texel
    /// `(0, 0, 0)` of level 0 at every one of them where `first_texel`
    /// ([`plane_first_texel_copy`](Self::plane_first_texel_copy)): that texel copied into the
    /// scratch's first, then the scratch's texels doubled until they fill its largest band.
    fn plane_words_moves(
        &mut self,
        buffer: &Buffer,
        texture: &Texture,
        region: &ImageRegion,
        layout: &BufferLayout,
        (into_texture, first_texel): (bool, bool),
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let pf = texture.pixel_format();
        let limits = &device.properties().description.limits;
        let band_limits = BandLimits {
            align: limits.min_storage_buffer_offset_alignment.max(4),
            max_range: u64::from(limits.max_storage_buffer_range),
            max_groups: limits.max_compute_work_group_count[0],
            max_scratch: MAX_SCRATCH,
        };
        // Read as a blit's (`host_residency.rs`): the command buffer holds the import.
        let (raw_buffer, base) = self.core().blit_at(buffer, 0)?;
        let bands = bands(region.size, layout, base, &band_limits)
            .ok_or(Error::Unsupported(unmeasured(pf)))?;
        let align = band_limits.align;
        let width = u64::from(region.size[0]);
        let texels = bands
            .iter()
            .map(|band| TEXEL_BYTES * width * u64::from(band.rows))
            .max()
            .unwrap_or(0)
            .next_multiple_of(align);
        let shape_bytes = u64::from(4 * PlaneWordsModule::SHAPE_WORDS);
        let slot = shape_bytes.next_multiple_of(align);
        let scratch = device.own_buffer(
            texels + slot * bands.len() as u64,
            ResourceOptions::private(),
        )?;
        let (raw_scratch, at) = scratch.at(0);
        if !at.is_multiple_of(align) {
            return Err(internal("a plane words scratch off its alignment"));
        }
        let kernel = device.owned_kernel(OwnedModule::PlaneWords)?;
        let row_words = |band: &Band| -> u32 {
            match band.rows {
                1 => 0,
                _ => (layout.bytes_per_row / 4) as u32,
            }
        };
        let mut ranges = Vec::with_capacity(bands.len());
        let mut steps = Vec::with_capacity(bands.len());
        for (i, band) in bands.iter().enumerate() {
            let count = width * u64::from(band.rows);
            let shape_at = at + texels + slot * i as u64;
            ranges.push(vec![
                (raw_scratch, at, TEXEL_BYTES * count),
                (
                    raw_buffer,
                    band.window.start,
                    band.window.end - band.window.start,
                ),
                (raw_scratch, shape_at, shape_bytes),
            ]);
            let shape = PlaneWordsModule::shape(
                region.size[0],
                count as u32,
                row_words(band),
                band.first_word,
                into_texture,
            );
            let [x, y, z] = region.origin;
            let image = ImageRegion {
                layers: 1,
                origin: [x, y + band.first_row, z + band.image],
                size: [region.size[0], band.rows, 1],
                ..*region
            };
            let copy = vk::BufferImageCopy {
                buffer_offset: at,
                ..buffer_image_copy(&image)
            };
            let groups = count.div_ceil(u64::from(PlaneWordsModule::LOCAL_SIZE)) as u32;
            steps.push((shape_at, shape, copy, groups));
        }
        let sets = OwnedSets::write(&device, &kernel, &ranges)?;
        let [_, height, depth] = region.size.map(u64::from);
        let span = layout.offset
            ..layout.offset
                + (depth - 1) * layout.bytes_per_image
                + (height - 1) * layout.bytes_per_row
                + 4 * width;
        let (identity, range) = buffer.tracked(span);
        let buffer_use = if into_texture {
            let read = Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            );
            Use::reads(identity.key, identity.tracked, UseRange::Bytes(range), read)
        } else {
            let write = Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
            Use::writes(
                identity.key,
                identity.tracked,
                UseRange::Bytes(range),
                write,
            )
        };
        let (levels, layers) = match first_texel {
            true => (0..1, 0..1),
            false => (
                region.level..region.level + 1,
                region.layer..region.layer + region.layers,
            ),
        };
        // The first texel's copy, and the doublings that spread it over the largest band.
        let first = vk::BufferImageCopy {
            buffer_offset: at,
            ..buffer_image_copy(&ImageRegion {
                level: 0,
                layer: 0,
                layers: 1,
                origin: [0; 3],
                size: [1; 3],
                ..*region
            })
        };
        let largest = texels / TEXEL_BYTES;
        let mut doublings = Vec::new();
        let mut held = 1;
        while first_texel && held < largest {
            let more = held.min(largest - held);
            doublings.push(vk::BufferCopy {
                src_offset: at,
                dst_offset: at + TEXEL_BYTES * held,
                size: TEXEL_BYTES * more,
            });
            held += more;
        }
        let image_use = if into_texture {
            let access = Access::new(
                TRANSFER,
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            );
            texture.image_use(levels, layers, access, true)?
        } else {
            let access = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
            texture.image_use(levels, layers, access, false)?
        };
        let image = texture.raw_image()?;
        let descriptor_sets = sets.sets.clone();
        let stages = TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        self.command_in(
            stages,
            &[buffer_use, image_use],
            |device, command_buffer| {
                // SAFETY: the command buffer is recording, outside a render pass; the buffer, the
                // scratch, the sets and the texture are retained until it completes; the image is in
                // the `GENERAL` layout; every copy and binding lies inside the scratch, the buffer
                // and the image (`bands`, and the caller's checks of the copy); each step reads what
                // the one before it wrote, past a barrier, and the scratch's texels are written again
                // only past a barrier after the band before is done with them.
                unsafe {
                    for (shape_at, shape, _, _) in &steps {
                        device.cmd_update_buffer(command_buffer, raw_scratch, *shape_at, shape);
                    }
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.pipeline,
                    );
                    if first_texel {
                        Barrier::everything().record(device, command_buffer);
                        device.cmd_copy_image_to_buffer(
                            command_buffer,
                            image,
                            vk::ImageLayout::GENERAL,
                            raw_scratch,
                            std::slice::from_ref(&first),
                        );
                        for doubling in &doublings {
                            Barrier::everything().record(device, command_buffer);
                            device.cmd_copy_buffer(
                                command_buffer,
                                raw_scratch,
                                raw_scratch,
                                std::slice::from_ref(doubling),
                            );
                        }
                    }
                    for (set, (_, _, copy, groups)) in descriptor_sets.iter().zip(&steps) {
                        Barrier::everything().record(device, command_buffer);
                        if !first_texel {
                            device.cmd_copy_image_to_buffer(
                                command_buffer,
                                image,
                                vk::ImageLayout::GENERAL,
                                raw_scratch,
                                std::slice::from_ref(copy),
                            );
                            Barrier::everything().record(device, command_buffer);
                        }
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.layout,
                            0,
                            std::slice::from_ref(set),
                            &[],
                        );
                        device.cmd_dispatch(command_buffer, *groups, 1, 1);
                        if into_texture {
                            Barrier::everything().record(device, command_buffer);
                            device.cmd_copy_buffer_to_image(
                                command_buffer,
                                raw_scratch,
                                image,
                                vk::ImageLayout::GENERAL,
                                std::slice::from_ref(copy),
                            );
                        }
                    }
                }
            },
        )?;
        self.core().retain(scratch);
        self.core().retain(sets);
        self.core().retain(buffer.clone());
        self.core().retain(texture.clone());
        if into_texture {
            // Plane 0 alone is written (the module docs): the texel is not, so a new texture's
            // pattern is not (`fresh_unwritten.rs`).
            self.scrub(
                texture,
                vec![(vk::ImageAspectFlags::empty(), Written::of(region))],
            )?;
        }
        Ok(())
    }

    /// Record `copies` between two such textures as Metal's blit moves their bytes (the module
    /// docs), each one command of the encoder: `false`, having recorded nothing, where Metal's
    /// answer was not measured or the level does not fit one storage buffer, after announcing
    /// that the caller copies each texel's two words (`two-plane-texture-copy`). A copy the model
    /// gives where Metal's own answer varies or reads bytes from outside the planes is
    /// announced the same way. A copy with a texture over a surface's planes, whose bytes no other
    /// copy holds, is refused instead where it is not modelled.
    pub(super) fn modelled_plane_copies(
        &mut self,
        source: &Texture,
        destination: &Texture,
        copies: &[ImageCopyRegion],
    ) -> Result<bool, Error> {
        let device = self.core().device().clone();
        let limit = u64::from(
            device
                .properties()
                .description
                .limits
                .max_storage_buffer_range,
        );
        let level = |texture: &Texture| -> Result<Option<[u32; 2]>, Error> {
            let plan = texture.image_plan()?;
            let record = texture.record();
            let [w, h, d] = plan.extent;
            let measured = record.texture_type == TextureType::Type2D
                && record.samples == 1
                && plan.levels == 1
                && plan.layers == 1
                && d == 1;
            Ok((measured && TEXEL_BYTES * u64::from(w) * u64::from(h) <= limit).then_some([w, h]))
        };
        let volume = [source, destination]
            .iter()
            .any(|t| t.texture_type() == TextureType::Type3D);
        let why_not = if source.same_storage(destination) {
            Some("a copy within one texture")
        } else if volume {
            Some(
                "a 3D texture, out of which Metal's blit reads the first texel at every texel and                  into which it wrote nothing at some widths, not modelled: the texels copy at the                  first depth slice",
            )
        } else {
            match (level(source)?, level(destination)?) {
                (Some(_), Some(_)) => None,
                _ => Some(
                    "a texture of more than one level or slice, or a level larger than a storage \
                     buffer, where Metal's blit was not measured",
                ),
            }
        };
        // A texture over a surface's planes is its bytes to a blit: no copy of its texels'
        // words stands in. Its planes at blit rows of 0 (one row narrower than 4 texels) and a
        // surface of other than whole words are not measured.
        let surface = |texture: &Texture| -> Result<bool, Error> {
            if !texture.over_buffer() {
                return Ok(false);
            }
            let refused = || Error::Unsupported(unmeasured(texture.pixel_format()));
            let level = level(texture)?.ok_or_else(refused)?;
            match BytePlanes::of(texture, level) {
                Some(planes) if planes.blit > 0 && planes.bytes.length().is_multiple_of(4) => {
                    Ok(true)
                }
                _ => Err(refused()),
            }
        };
        let surfaces = [surface(source)?, surface(destination)?];
        if let Some(why) = why_not {
            if surfaces.contains(&true) {
                return Err(Error::Unsupported(unmeasured(destination.pixel_format())));
            }
            announce_texture_copy(&device, source, destination, why);
            return Ok(false);
        }
        let (Some(from), Some(to)) = (level(source)?, level(destination)?) else {
            return Err(internal("a measured two-plane level without its extent"));
        };
        let blits = [blit_rows(source, from), blit_rows(destination, to)];
        for copy in copies {
            if let Some(why) = residue(blits, copy) {
                announce_texture_copy(&device, source, destination, why);
            }
            let [width, height, _] = copy.src.size;
            let size = [width, height];
            let [sx, sy, _] = copy.src.origin;
            let [dx, dy, _] = copy.dst.origin;
            // Into a texture whose planes lie in a heap's or a surface's bytes, the bytes Metal's
            // blit writes (`heap_bytes.rs`), read from a source's bytes as they lie.
            if let Some(planes) = BytePlanes::of(destination, to) {
                if self.plane_bytes_copy(source, destination, &planes, from, copy)? {
                    if outside_texels(from, [sx, sy], size)
                        && source.outside_written()
                        && BytePlanes::of(source, from).is_none()
                    {
                        announce_texture_copy(
                            &device,
                            source,
                            destination,
                            "a copy reading bytes outside the source's texels that an earlier \
                             blit wrote, which Metal's blit keeps; the copy reads them as 0",
                        );
                    }
                    continue;
                }
                if planes.surface {
                    return Err(Error::Unsupported(unmeasured(destination.pixel_format())));
                }
                announce_texture_copy(
                    &device,
                    source,
                    destination,
                    "a copy into a heap's texture whose bytes pass the device's storage buffer \
                     range; it copies the texels alone",
                );
            }
            // A source over a surface's planes is read as its bytes lie (`plane_texture_copy`).
            if !surfaces[0]
                && outside_texels(from, [sx, sy], size)
                && (source.outside_written() || source.heap().is_some())
            {
                announce_texture_copy(
                    &device,
                    source,
                    destination,
                    "a copy reading bytes outside the source's texels that hold other bytes than \
                     0: an earlier blit's, which Metal's blit keeps, or a heap's other \
                     resources'; the copy reads them as 0",
                );
            }
            self.plane_texture_copy(source, destination, (from, to), copy)?;
            if outside_texels(to, [dx, dy], size) {
                destination.keep_outside();
            }
        }
        Ok(true)
    }

    /// Record `copy` from `source`'s one level (`levels.0`, width and height) to `destination`'s
    /// (`levels.1`) through the plane copy kernel: the source's level (or, over a surface's
    /// planes, the bytes the copy reads, as they lie) and the destination's copied into a scratch
    /// buffer of the encoder, the kernel's dispatches over the destination's texels (in batches
    /// within `maxComputeWorkGroupCount[0]`), and the destination's texels copied back.
    fn plane_texture_copy(
        &mut self,
        source: &Texture,
        destination: &Texture,
        (from, to): ([u32; 2], [u32; 2]),
        copy: &ImageCopyRegion,
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let per_dispatch = limits.max_compute_work_group_count[0].max(1);
        let level_bytes = |[w, h]: [u32; 2]| TEXEL_BYTES * u64::from(w) * u64::from(h);
        let [sx, sy, _] = copy.src.origin;
        let [dx, dy, _] = copy.dst.origin;
        let [width, height, _] = copy.src.size;
        let whole = |[w, h]: [u32; 2]| ImageRegion {
            aspect: ImageAspectFlags::COLOR,
            level: 0,
            layer: 0,
            layers: 1,
            origin: [0, 0, 0],
            size: [w, h, 1],
        };
        // The source: its level's texels, or (over a surface's planes) the bytes the copy reads.
        let (staged, source_bytes, plane_source, source_use) = match BytePlanes::of(source, from) {
            Some(planes) if source.over_buffer() => {
                let size = [u64::from(width), u64::from(height)];
                let staged = stage_planes(&planes, [u64::from(sx), u64::from(sy)], size)?;
                if staged.bytes > u64::from(limits.max_storage_buffer_range) {
                    return Err(Error::Unsupported(unmeasured(source.pixel_format())));
                }
                (
                    staged.staged,
                    staged.bytes,
                    PlaneSource::Bytes(staged.planes),
                    staged.uses,
                )
            }
            _ => {
                let region = buffer_image_copy(&whole(from));
                let read = source.image_use(
                    0..1,
                    0..1,
                    Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ),
                    false,
                )?;
                (
                    Staged::Texels(source.raw_image()?, region),
                    level_bytes(from),
                    PlaneSource::Texels {
                        level: from,
                        from: [sx, sy],
                    },
                    vec![read],
                )
            }
        };
        let destination_bytes = level_bytes(to);
        let groups = (to[0] * to[1]).div_ceil(PlaneCopyModule::LOCAL_SIZE);
        let batches = groups.div_ceil(per_dispatch);
        let shape_bytes = u64::from(4 * PlaneCopyModule::SHAPE_WORDS);
        let slot = shape_bytes.next_multiple_of(align);
        let destination_at = source_bytes.next_multiple_of(align);
        let shapes_at = destination_at + destination_bytes.next_multiple_of(align);
        let scratch = device.own_buffer(
            shapes_at + slot * u64::from(batches),
            ResourceOptions::private(),
        )?;
        let (raw_scratch, at) = scratch.at(0);
        if !at.is_multiple_of(align) {
            return Err(internal("a plane copy scratch off its alignment"));
        }
        let kernel = device.owned_kernel(OwnedModule::PlaneCopy)?;
        let mut ranges = Vec::with_capacity(batches as usize);
        let mut steps = Vec::with_capacity(batches as usize);
        for batch in 0..batches {
            let shape_at = at + shapes_at + slot * u64::from(batch);
            ranges.push(vec![
                (raw_scratch, at, source_bytes),
                (raw_scratch, at + destination_at, destination_bytes),
                (raw_scratch, shape_at, shape_bytes),
            ]);
            let first = batch * per_dispatch;
            let shape = PlaneCopyModule::shape(
                &plane_source,
                to,
                [dx, dy],
                [width, height],
                first * PlaneCopyModule::LOCAL_SIZE,
            );
            steps.push((shape_at, shape, per_dispatch.min(groups - first)));
        }
        let sets = OwnedSets::write(&device, &kernel, &ranges)?;
        let destination_level = whole(to);
        let destination_texels = vk::BufferImageCopy {
            buffer_offset: at + destination_at,
            ..buffer_image_copy(&destination_level)
        };
        let mut uses = source_use;
        uses.push(destination.image_use(
            0..1,
            0..1,
            Access::new(
                TRANSFER,
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            ),
            true,
        )?);
        let destination_image = destination.raw_image()?;
        let descriptor_sets = sets.sets.clone();
        let stages = TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        self.command_in(stages, &uses, |device, command_buffer| {
            // SAFETY: the command buffer is recording, outside a render pass; the textures (and
            // the bytes a source's planes lie in), the scratch and the sets are retained until it
            // completes; the images are in the `GENERAL` layout; every copy and binding lies
            // inside the scratch, the source's bytes and the images' one level; each step reads
            // what the one before it wrote, past a barrier.
            unsafe {
                for (shape_at, shape, _) in &steps {
                    device.cmd_update_buffer(command_buffer, raw_scratch, *shape_at, shape);
                }
                Barrier::everything().record(device, command_buffer);
                staged.record(device, command_buffer, raw_scratch, at);
                device.cmd_copy_image_to_buffer(
                    command_buffer,
                    destination_image,
                    vk::ImageLayout::GENERAL,
                    raw_scratch,
                    std::slice::from_ref(&destination_texels),
                );
                Barrier::everything().record(device, command_buffer);
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline,
                );
                for (set, (_, _, groups)) in descriptor_sets.iter().zip(&steps) {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.layout,
                        0,
                        std::slice::from_ref(set),
                        &[],
                    );
                    device.cmd_dispatch(command_buffer, *groups, 1, 1);
                }
                Barrier::everything().record(device, command_buffer);
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    raw_scratch,
                    destination_image,
                    vk::ImageLayout::GENERAL,
                    std::slice::from_ref(&destination_texels),
                );
            }
        })?;
        self.core().retain(scratch);
        self.core().retain(sets);
        self.core().retain(source.clone());
        self.core().retain(destination.clone());
        let aspects = destination_level.aspect.into();
        self.scrub(
            destination,
            vec![(aspects, Written::of(&destination_level))],
        )
    }
}

/// The rows Metal's blit addresses both planes of `texture`'s one level of `level` texels at:
/// plane 0's GPU rows over a surface's planes, else `⌊4 w / 16⌋ · 16`.
fn blit_rows(texture: &Texture, level: [u32; 2]) -> u64 {
    match texture.buffer_texels() {
        Some(texels) if texels.alpha().is_some() => texels.layout.row_pitch,
        _ => u64::from(PlaneRows::of(level[0]).blit),
    }
}

/// Why Metal's own answer for `copy` between textures whose blit rows are `blits` (source,
/// destination) is not the model's bytes, or `None` where it is:
/// below a width of 4 a texture of its own memory's blit addresses every row at the first row's
/// bytes and reads alpha bytes from outside the planes (other bytes than 0 in some runs), and where
/// the copy's rows overlap in the destination with other bytes than each other's, Metal's blit
/// writes them in an order that varies with the width (and with the run, out of a surface's
/// planes).
fn residue(blits: [u64; 2], copy: &ImageCopyRegion) -> Option<&'static str> {
    let [s, d] = blits;
    if blits.contains(&0) {
        return Some(
            "a texture narrower than 4 texels, where Metal's blit reads alpha bytes from \
             outside the planes, which the copy reads as 0",
        );
    }
    let [width, height, _] = copy.src.size.map(u64::from);
    let [sx, sy, _] = copy.src.origin.map(u64::from);
    let [dx, dy, _] = copy.dst.origin.map(u64::from);
    for unit in [4, 1] {
        let overlap = height > 1 && unit * width > d;
        let same = s == d && sy * s + unit * sx == dy * d + unit * dx;
        if overlap && !same {
            return Some(
                "a copy whose rows overlap in the destination, where Metal's blit writes them \
                 in an order that varies; the copy keeps the later row's bytes",
            );
        }
    }
    None
}

/// Whether a copy of `size` texels at `origin` in a level of `level` texels (width, height)
/// reaches, at Metal's blit rows, bytes outside the level's texels: a row's padding or the bytes
/// past a plane. Metal's blit keeps the bytes it writes there and reads them back in a later
/// blit (padding and the bytes past the alpha plane, chained copies between
/// private and between shared textures); the kernel holds the planes' texels alone.
fn outside_texels(level: [u32; 2], origin: [u32; 2], size: [u32; 2]) -> bool {
    let rows = PlaneRows::of(level[0]);
    let blit = u64::from(rows.blit);
    [(4, rows.color), (1, rows.alpha)]
        .into_iter()
        .any(|(unit, pitch): (u64, u32)| {
            let pitch = u64::from(pitch);
            let texels = unit * u64::from(level[0]);
            let plane = pitch * u64::from(level[1]);
            let bytes = unit * u64::from(size[0]);
            let base = u64::from(origin[1]) * blit + unit * u64::from(origin[0]);
            (0..u64::from(size[1])).any(|row| {
                let start = base + row * blit;
                start + bytes > plane || (pitch > texels && start % pitch + bytes > texels)
            })
        })
}

/// Announce, once per device, a blit from `source` to `destination` whose answer is not Metal's
/// measured one, `why`.
fn announce_texture_copy(device: &Device, source: &Texture, destination: &Texture, why: &str) {
    device.degrade(Degradation::TwoPlaneTextureCopy, || {
        let object = format!(
            "a blit from a {} texture into a {} texture ({why})",
            source.pixel_format().name(),
            destination.pixel_format().name()
        );
        let reason = "where Metal's answer was measured the copy moves both planes' bytes as \
                      Metal's blit does; here it copies each texel's two words, keeps the later \
                      row where rows overlap, or reads as 0 bytes Metal's blit reads from \
                      outside the planes"
            .to_owned();
        (object, reason)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: BandLimits = BandLimits {
        align: 64,
        max_range: 1 << 27,
        max_groups: 65535,
        max_scratch: MAX_SCRATCH,
    };

    fn layout(offset: u64, bytes_per_row: u64, bytes_per_image: u64) -> BufferLayout {
        BufferLayout {
            offset,
            bytes_per_row,
            bytes_per_image,
        }
    }

    /// A guest's pitches reach the measured-layout check as they are: an image of rows past
    /// `u64::MAX` is longer than any `bytesPerImage`, and one image ignores it.
    #[test]
    fn layouts_past_u64_max_are_checked_without_wrapping() {
        let wraps = 0x5555_5555_5555_5560;
        for bpr in [1 << 63, wraps, u64::MAX - 3] {
            assert!(measured_layout(16, 4, 1, bpr, 0, 0), "{bpr}");
            for bpi in [0, 64, u64::MAX - 3, 0x30] {
                assert!(!measured_layout(16, 4, 2, bpr, bpi, 0), "{bpr} {bpi}");
            }
        }
        assert!(measured_layout(16, 4, 2, 16, 64, 0));
    }

    /// The color words move out and in alone: a texel's second word is never read into the
    /// words, nor written from them.
    #[test]
    fn reaching_outside_the_texels_is_measured_at_the_blit_rows() {
        // 48 wide: plane 0's rows are the blit's; the alpha plane's second blit row lies past a
        // plane of 4 rows.
        assert!(!outside_texels([48, 4], [0, 0], [48, 1]));
        assert!(outside_texels([48, 4], [0, 0], [48, 2]));
        // 9 wide: blit rows 32 bytes apart, plane 0's 48: the second reaches row 0's padding.
        assert!(!outside_texels([9, 7], [0, 0], [8, 1]));
        assert!(outside_texels([9, 7], [0, 0], [9, 2]));
        // 8 wide: blit row 1's alpha bytes land on alpha row 2's texels.
        assert!(!outside_texels([8, 4], [0, 1], [8, 1]));
    }

    #[test]
    fn narrow_and_widen_move_the_first_words() {
        let texels: Vec<u8> = (0..32).collect();
        let mut words = [0xee; 16];
        narrow(&texels, &mut words);
        assert_eq!(
            words,
            [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
        );
        let mut back = vec![0xaa; 32];
        widen(&words, &mut back);
        for (i, b) in back.iter().enumerate() {
            let want = if i % 8 < 4 { texels[i] } else { 0xaa };
            assert_eq!(*b, want, "byte {i}");
        }
    }

    /// The layouts measured on Metal are taken, and the rest refused.
    #[test]
    fn only_measured_layouts_are_taken() {
        // r4, r5, r8, r4i and sub: 48 x 20, and 17 x 9 at offset 256 with rows of 80 bytes.
        for (row, bpr, bpi) in [(192, 192, 0), (192, 240, 0), (192, 384, 0), (68, 80, 0)] {
            assert!(measured_layout(row, 20, 1, bpr, bpi, 256), "{bpr}");
        }
        assert!(measured_layout(192, 20, 2, 192, 2 * 192 * 20, 0));
        // Rows that overlap, off 4 bytes, or of 0; images that overlap; an offset off 4.
        assert!(!measured_layout(192, 20, 1, 188, 0, 0));
        assert!(!measured_layout(192, 20, 1, 194, 0, 0));
        assert!(!measured_layout(192, 20, 1, 0, 0, 0));
        assert!(!measured_layout(192, 20, 2, 192, 192 * 19, 0));
        assert!(!measured_layout(192, 20, 1, 192, 0, 2));
        // One row takes any pitch, one image any image pitch.
        assert!(measured_layout(192, 1, 1, 0, 0, 0));
        assert!(measured_layout(192, 20, 1, 192, 3, 0));
    }

    /// A small copy is one band per image, its window from the aligned byte below its first.
    #[test]
    fn a_small_copy_is_a_band_an_image() {
        let bands = bands([17, 9, 2], &layout(260, 80, 1000), 4, &LIMITS).expect("bands");
        assert_eq!(
            bands,
            vec![
                Band {
                    image: 0,
                    first_row: 0,
                    rows: 9,
                    window: 256..264 + 8 * 80 + 68,
                    first_word: 2,
                },
                Band {
                    image: 1,
                    first_row: 0,
                    rows: 9,
                    window: 1216..1264 + 8 * 80 + 68,
                    first_word: 12,
                },
            ]
        );
    }

    /// Bands keep their texels within the scratch's limit and the dispatch's workgroups, and
    /// their window within the storage range; together they cover every row once, in order.
    #[test]
    fn bands_keep_to_the_limits_and_cover_every_row() {
        let tight = BandLimits {
            align: 256,
            max_range: 4096,
            max_groups: 2,
            max_scratch: 1 << 20,
        };
        for (width, bpr) in [(8u32, 32u64), (8, 1000), (100, 400), (128, 512)] {
            let size = [width, 50, 3];
            let l = layout(12, bpr, bpr * 50 + 4);
            let bands = bands(size, &l, 0, &tight).expect("bands");
            let mut next = (0, 0);
            for band in &bands {
                assert_eq!((band.image, band.first_row), next, "{width} {bpr}");
                let texels = u64::from(width) * u64::from(band.rows);
                assert!(8 * texels <= tight.max_range.min(tight.max_scratch));
                assert!(texels <= 64 * u64::from(tight.max_groups));
                assert!(band.window.end - band.window.start <= tight.max_range);
                assert_eq!(band.window.start % tight.align, 0);
                let first = l.offset
                    + u64::from(band.image) * l.bytes_per_image
                    + u64::from(band.first_row) * l.bytes_per_row;
                assert_eq!(band.window.start + 4 * u64::from(band.first_word), first);
                next = match band.first_row + band.rows {
                    50 => (band.image + 1, 0),
                    row => (band.image, row),
                };
            }
            assert_eq!(next, (3, 0), "{width} {bpr}");
        }
        // A row past the storage range has no band.
        let narrow_range = BandLimits {
            max_range: 64,
            ..tight
        };
        assert_eq!(
            bands([32, 1, 1], &layout(0, 128, 0), 0, &narrow_range),
            None
        );
    }
}
