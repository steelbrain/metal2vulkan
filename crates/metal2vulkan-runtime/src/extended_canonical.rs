//! Metal's canonical encoding of `BGRA10_XR` and `BGRA10_XR_sRGB` texels, written after a blit
//! through a texture Metal keeps losslessly compressed, under `ExtendedRangeBlitPolicy::Canonical`
//! (`compressed_blit.rs` says when Metal writes it and what it is).
//!
//! - **Into an image** ([`BlitCommandEncoder::canonical_texels`]): the regions the blit wrote are
//!   copied out of the image into a scratch buffer, tightly packed, each at an offset the device
//!   binds a storage buffer at; one dispatch of `bind::extended_canonical_module` a region
//!   rewrites its texels' words in place; and the regions are copied back. Three commands the
//!   encoder's tracker orders as any others: the copies' transfer reads and writes, the dispatch's
//!   storage reads and writes of the scratch.
//! - **Into a buffer** ([`BlitCommandEncoder::canonical_words`]): one dispatch rewrites the words
//!   of the texels the blit wrote, at the layout's rows and images, in place, where the device
//!   binds them: the region's first byte, its pitches and its images' 4-byte aligned, and its
//!   bytes within `maxStorageBufferRange` of the binding's offset.
//!
//! - **Into a texture over a buffer** ([`BlitCommandEncoder::canonical_texel_words`]): as into a
//!   buffer, at the texels' own rows in the buffer's bytes.
//!
//! Each dispatch writes no word but the texels' (the module's contract), so the bytes between a
//! layout's rows and images, and past them, keep what the blit left. A region the device cannot
//! bind is not rewritten: the caller announces the note instead, as under `Kept`.

use super::blit::{reads, writes, writes_by, BlitCommandEncoder, TRANSFER};
use super::blit_plan::{texel_layout, BufferLayout, ImageCopyRegion};
use super::compressed_blit;
use super::copy_plan::ImageRegion;
use super::error::{internal, Error};
use super::owned_kernel::{OwnedBinds, OwnedModule, OwnedSet};
use super::texture::{buffer_image_copy, Texture};
use super::tracker::Access;
use super::values::HazardTrackingMode;
use super::Buffer;
use ash::vk;
use metal2vulkan::bind::ExtendedCanonicalModule;

/// The canonical encoding's storage access of the words it rewrites: a read and a write in the
/// compute shader stage.
fn storage_access() -> (Access, Access) {
    let stage = vk::PipelineStageFlags2::COMPUTE_SHADER;
    (
        Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ),
        Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE),
    )
}

/// One dispatch of the kernel: its binding's buffer, offset and range, its shape and workgroups.
struct Dispatch {
    binding: (vk::Buffer, u64, u64),
    shape: [u8; 16],
    groups: [u32; 3],
}

impl BlitCommandEncoder {
    /// Rewrite the texels of `regions` of `texture`'s image, which a blit just wrote, in Metal's
    /// canonical encoding (the module docs). `false`, with nothing recorded, where a region's
    /// texels exceed the device's storage buffer range.
    pub(super) fn canonical_texels(
        &mut self,
        texture: &Texture,
        regions: &[ImageRegion],
    ) -> Result<bool, Error> {
        let limits = &self.core().device().properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(8);
        let most = u64::from(limits.max_storage_buffer_range);
        let (most_groups_y, most_groups_z) = (
            limits.max_compute_work_group_count[1],
            limits.max_compute_work_group_count[2],
        );
        // Each region's place in the scratch buffer and its dispatch's shape.
        let mut placed = Vec::with_capacity(regions.len());
        let mut length = 0u64;
        for region in regions {
            let [width, height, depth] = region.size;
            let images = depth
                .checked_mul(region.layers)
                .ok_or_else(|| internal("a canonical region's images"))?;
            let bytes = u64::from(width) * u64::from(height) * u64::from(images) * 8;
            if bytes == 0 {
                continue;
            }
            if bytes > most || height > most_groups_y || images > most_groups_z {
                return Ok(false);
            }
            let at = length.next_multiple_of(align);
            let row_words = width
                .checked_mul(2)
                .ok_or_else(|| internal("a canonical region's rows"))?;
            let image_words = row_words
                .checked_mul(height)
                .ok_or_else(|| internal("a canonical region's images"))?;
            placed.push((
                *region,
                at,
                bytes,
                ExtendedCanonicalModule::shape(0, width, row_words, image_words),
                ExtendedCanonicalModule::groups(width, height, images),
            ));
            length = at + bytes;
        }
        if placed.is_empty() {
            return Ok(true);
        }
        // The levels and layers the regions span, of the image's own.
        let span = |of: fn(&ImageRegion) -> (u32, u32)| {
            let starts = placed.iter().map(|p| of(&p.0).0).min().unwrap_or(0);
            let ends = placed.iter().map(|p| of(&p.0).1).max().unwrap_or(0);
            starts..ends
        };
        let levels = span(|r| (r.level, r.level + 1));
        let layers = span(|r| (r.layer, r.layer + r.layers));
        let device = self.core().device().clone();
        // The copy out writes every byte the dispatches and the copy back read (the regions,
        // tightly packed), and nothing reads the alignment gaps between them, so the scratch is
        // made without zeros, which would take a copy or a fill in a submission's first work.
        let scratch = device.unzeroed_buffer(length, HazardTrackingMode::Default)?;
        let (through, base) = scratch.at(0);
        let copies: Vec<vk::BufferImageCopy> = placed
            .iter()
            .map(|(region, at, ..)| vk::BufferImageCopy {
                buffer_offset: base + at,
                buffer_row_length: 0,
                buffer_image_height: 0,
                ..buffer_image_copy(region)
            })
            .collect();
        let dispatches: Vec<Dispatch> = placed
            .iter()
            .map(|&(_, at, bytes, shape, groups)| Dispatch {
                binding: (through, base + at, bytes),
                shape,
                groups,
            })
            .collect();
        let image = texture.raw_image()?;
        let whole = 0..length;
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let uses = [
            texture.image_use(levels.clone(), layers.clone(), read, false)?,
            writes(&scratch, whole.clone()),
        ];
        self.command(&uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording outside a render pass; the image
            // and the scratch buffer are live and retained until it completes, the image is in
            // the `GENERAL` layout, and each region lies in both.
            unsafe {
                device.cmd_copy_image_to_buffer(
                    raw,
                    image,
                    vk::ImageLayout::GENERAL,
                    through,
                    &copies,
                );
            }
        })?;
        self.rewrite(&scratch, whole.clone(), &dispatches)?;
        let uses = [
            reads(&scratch, whole),
            texture.image_use(levels, layers, write, true)?,
        ];
        self.command(&uses, |device, raw| {
            // SAFETY: as the copy out above.
            unsafe {
                device.cmd_copy_buffer_to_image(
                    raw,
                    through,
                    image,
                    vk::ImageLayout::GENERAL,
                    &copies,
                );
            }
        })?;
        self.core().retain(scratch);
        self.core().retain_shared(texture.held());
        Ok(true)
    }

    /// Rewrite the words of the texels of `size` (width, height and images) that a blit just wrote
    /// into `buffer` at `layout`, in Metal's canonical encoding (the module docs). `false`, with
    /// nothing recorded, where the device cannot bind them.
    pub(super) fn canonical_words(
        &mut self,
        buffer: &Buffer,
        layout: &BufferLayout,
        [width, height, images]: [u32; 3],
    ) -> Result<bool, Error> {
        if width == 0 || height == 0 || images == 0 {
            return Ok(true);
        }
        let limits = &self.core().device().properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let most = u64::from(limits.max_storage_buffer_range);
        let most_groups = limits.max_compute_work_group_count;
        let (raw, base) = self.core().blit_at(buffer, 0)?;
        let start = base + layout.offset;
        let row = if height > 1 { layout.bytes_per_row } else { 0 };
        let image = if images > 1 {
            layout.bytes_per_image
        } else {
            0
        };
        let aligned = |v: u64| v.is_multiple_of(4);
        if !aligned(start) || !aligned(row) || !aligned(image) {
            return Ok(false);
        }
        let reach = u64::from(images - 1)
            .checked_mul(image)
            .and_then(|i| i.checked_add(u64::from(height - 1).checked_mul(row)?))
            .and_then(|r| r.checked_add(u64::from(width) * 8));
        let Some(reach) = reach else {
            return Ok(false);
        };
        let offset = start - start % align;
        let range = start - offset + reach;
        let groups = ExtendedCanonicalModule::groups(width, height, images);
        let words = |v: u64| u32::try_from(v / 4).ok();
        let (Some(first), Some(row_words), Some(image_words)) =
            (words(start - offset), words(row), words(image))
        else {
            return Ok(false);
        };
        if range > most || groups.iter().zip(most_groups).any(|(g, m)| *g > m) {
            return Ok(false);
        }
        let dispatch = Dispatch {
            binding: (raw, offset, range),
            shape: ExtendedCanonicalModule::shape(first, width, row_words, image_words),
            groups,
        };
        let written = layout.offset..layout.offset + reach;
        self.rewrite(buffer, written, &[dispatch])?;
        Ok(true)
    }

    /// Rewrite the texels `copies` wrote into `texture`, a texture over a buffer's bytes, in
    /// Metal's canonical encoding, in place at the texels' rows ([`Self::canonical_words`]).
    /// `false` where its texels are no `BGRA10_XR` texels at their own rows in a buffer (rows
    /// that share bytes, or another format), or where the device cannot bind a region's words:
    /// the regions before that one are rewritten.
    pub(super) fn canonical_texel_words(
        &mut self,
        texture: &Texture,
        copies: &[ImageCopyRegion],
    ) -> Result<bool, Error> {
        let Some(texels) = texture.buffer_texels() else {
            return Ok(false);
        };
        // Texels past the buffer's length lie in the texture's tail, which the kernel's binding
        // of the buffer's words does not reach (`buffer_texture.rs`).
        let past = texels.past().is_some();
        if !compressed_blit::extended(texture.pixel_format()) || texels.plane_rows_overlap() || past
        {
            return Ok(false);
        }
        let unit = texels.unit();
        for copy in copies {
            let region = &copy.dst;
            let [width, height, depth] = region.size;
            for layer in 0..region.layers {
                let layout = texel_layout(&texels.layout, &unit, region, region.layer + layer);
                if !self.canonical_words(&texels.buffer, &layout, [width, height, depth])? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Record `dispatches` of the canonical encoding's kernel, which read and write the bytes
    /// `range` of `buffer`, as one command.
    fn rewrite(
        &mut self,
        buffer: &Buffer,
        range: std::ops::Range<u64>,
        dispatches: &[Dispatch],
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let kernel = device.owned_kernel(OwnedModule::ExtendedCanonical)?;
        let sets: Vec<OwnedSet> = dispatches
            .iter()
            .map(|dispatch| OwnedSet {
                images: Vec::new(),
                storage_images: Vec::new(),
                buffers: vec![dispatch.binding],
            })
            .collect();
        let binds = OwnedBinds::new(&device, &kernel, &sets)?;
        let (read, write) = storage_access();
        let mut words = writes_by(buffer, range, write);
        words.read = read;
        let stages = vk::PipelineStageFlags2::COMPUTE_SHADER;
        self.command_in(stages, &[words], |functions, raw| {
            let point = vk::PipelineBindPoint::COMPUTE;
            // SAFETY: the encoder's command buffer is recording outside a render pass; the
            // kernel is the device's, the bound words are live and retained until it completes,
            // and each shape and its workgroups stay inside its binding (the callers').
            unsafe {
                functions.cmd_bind_pipeline(raw, point, kernel.pipeline);
                for (index, dispatch) in dispatches.iter().enumerate() {
                    binds.bind(functions, raw, &kernel, index);
                    functions.cmd_push_constants(
                        raw,
                        kernel.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &dispatch.shape,
                    );
                    let [x, y, z] = dispatch.groups;
                    functions.cmd_dispatch(raw, x, y, z);
                }
            }
        })?;
        if let OwnedBinds::Sets(sets) = binds {
            self.core().retain(sets);
        }
        Ok(())
    }
}
