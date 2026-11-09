//! 32-bit float depths clamped into `[0, 1]` before a copy takes them into a depth image, on a
//! device that may store one outside there clamped (`ImageFacts::depth_clamps`: no
//! `VK_EXT_depth_range_unrestricted` enabled, and no
//! `DriverFact::DepthImageKeepsValuesOutsideUnitRange`).
//!
//! - **Why.** Vulkan takes no value outside `[0, 1]` into a depth image by a copy there
//!   (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), and such a device stores one clamped. Metal's
//!   `Depth32Float` texel keeps any `float`. The clamp is `bind::clamped_depth` on a depth's bits:
//!   a NaN of either sign 1.0, `-0.0` as it is, another negative value 0, a value above 1.0 1.0,
//!   no denormal flushed. Announced `restricted-depth-upload` by the upload, as before.
//! - **Where.** Every copy of a depth from a buffer or the host into a depth plane of a
//!   `D32_SFLOAT` or `D32_SFLOAT_S8_UINT` image ([`clamps_depth`]): `replace_region`'s packed
//!   bytes on the CPU, whichever way they go on (a host copy, the staging queue, a submission;
//!   `texture.rs`), so no depth image holds a value outside `[0, 1]` that a later copy of its
//!   texels moves on; a blit's copy from a buffer into the texture, through a scratch buffer of
//!   the encoder that takes the copy's bytes, or through the staging buffer its plan copies its
//!   rows into; and a copy between textures whose texels go through the encoder's scratch buffer
//!   into a depth (a colour texture of its unit, `blit_texture.rs`). A heap's tiled texels clamp
//!   as they move into their image (`texture/heap_tiled.rs`). Not on another device: each value
//!   is copied as Metal holds it.
//! - **How**, on the GPU ([`BlitCommandEncoder::clamp_depths`]): one dispatch of
//!   `bind::depth_clamp_module` per region (or per piece of one that its binding cannot hold
//!   whole), each its own command over the bytes it rewrites in place, which the encoder's
//!   tracker orders after the copy that wrote them and before the copy that reads them on.
//! - **Counted.** `WorkCounts::depth_uploads_clamped`: one a `replace_region`'s clamp, and one a
//!   blit copy's clamp of the depths it copies into an image.

use super::blit::{reads, writes, writes_by, BlitCommandEncoder};
use super::buffer::Buffer;
use super::error::{internal, Error};
use super::owned_kernel::{OwnedBinds, OwnedModule, OwnedSet};
use super::tracker::Access;
use super::values::ResourceOptions;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{clamped_depth, DepthClampModule};
use metal2vulkan::vk::{Format, ImageAspectFlags};
use std::ops::Range;

/// Whether a copy into `aspect` of an image of `format` on `device` clamps its 32-bit float
/// depths (the module docs): the depth of a `D32_SFLOAT` or `D32_SFLOAT_S8_UINT` image, on a
/// device whose depth images may store a value outside `[0, 1]` clamped.
pub(super) fn clamps_depth(device: &Device, format: Format, aspect: ImageAspectFlags) -> bool {
    let profile = device.compile_device().profile();
    aspect == ImageAspectFlags::DEPTH
        && matches!(format, Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT)
        && device.shared().textures.facts(profile).depth_clamps()
}

/// Clamp each 32-bit float depth of `bytes`, packed 4 bytes a depth, in place (the CPU's clamp,
/// `bind::clamped_depth`).
pub(super) fn clamp_depth_bytes(bytes: &mut [u8]) {
    for word in bytes.as_chunks_mut::<4>().0 {
        *word = clamped_depth(u32::from_le_bytes(*word)).to_le_bytes();
    }
}

/// The depths a copy `region` from a buffer into a depth image reads: the depths a row, its rows
/// and images, and the rows and images apart in words.
#[derive(Clone, Copy)]
struct Depths {
    width: u32,
    rows: u32,
    images: u32,
    row_words: u32,
    image_words: u32,
}

impl Depths {
    /// `region`'s depths; `None` for a region of none.
    fn of(region: &vk::BufferImageCopy) -> Result<Option<Depths>, Error> {
        let extent = region.image_extent;
        let images = extent
            .depth
            .checked_mul(region.image_subresource.layer_count)
            .ok_or_else(|| internal("a depth copy's images past u32"))?;
        if extent.width == 0 || extent.height == 0 || images == 0 {
            return Ok(None);
        }
        let row_words = match region.buffer_row_length {
            0 => extent.width,
            length => length,
        };
        let image_rows = match region.buffer_image_height {
            0 => extent.height,
            height => height,
        };
        let image_words = row_words
            .checked_mul(image_rows)
            .ok_or_else(|| internal("a depth copy's images apart past u32"))?;
        Ok(Some(Depths {
            width: extent.width,
            rows: extent.height,
            images,
            row_words,
            image_words,
        }))
    }

    /// The bytes from the first depth to past the last of `images` images of `rows` rows.
    fn reach(&self, images: u32, rows: u32) -> u64 {
        let images = u64::from(images - 1) * u64::from(self.image_words);
        let rows = u64::from(rows - 1) * u64::from(self.row_words);
        4 * (images + rows + u64::from(self.width))
    }
}

/// One dispatch of the kernel: its binding's buffer, offset and range, its shape and workgroups,
/// and the bytes it rewrites, of the scratch buffer's own.
struct Dispatch {
    binding: (vk::Buffer, u64, u64),
    shape: [u8; 16],
    groups: [u32; 3],
    bytes: Range<u64>,
}

/// The device's bounds on a dispatch: the binding's offset alignment, its range, and the
/// workgroups along each axis.
struct Bounds {
    align: u64,
    most: u64,
    groups: [u32; 3],
}

impl Bounds {
    fn of(device: &Device) -> Bounds {
        let limits = &device.properties().description.limits;
        Bounds {
            align: limits.min_storage_buffer_offset_alignment.max(4),
            most: u64::from(limits.max_storage_buffer_range),
            groups: limits.max_compute_work_group_count,
        }
    }

    /// The dispatch over `images` images of `rows` rows of `depths` from byte `start` of `raw`
    /// (the scratch buffer's `VkBuffer`, which starts at its byte `base`); `None` where its
    /// binding or workgroups pass the device's bounds.
    fn dispatch(
        &self,
        (raw, base): (vk::Buffer, u64),
        start: u64,
        depths: &Depths,
        [rows, images]: [u32; 2],
    ) -> Option<Dispatch> {
        let reach = depths.reach(images, rows);
        let offset = start - start % self.align;
        let range = start - offset + reach;
        let groups = DepthClampModule::groups(depths.width, rows, images);
        if range > self.most || groups.iter().zip(self.groups).any(|(g, m)| *g > m) {
            return None;
        }
        let first = u32::try_from((start - offset) / 4).ok()?;
        let shape =
            DepthClampModule::shape(first, depths.width, depths.row_words, depths.image_words);
        let at = start.checked_sub(base)?;
        Some(Dispatch {
            binding: (raw, offset, range),
            shape,
            groups,
            bytes: at..at + reach,
        })
    }

    /// The dispatches over `depths` from byte `start` of `raw`: all its images at once, else an
    /// image at a time, else rows of an image at a time, as many as the device binds.
    fn dispatches(
        &self,
        raw: (vk::Buffer, u64),
        start: u64,
        depths: &Depths,
        into: &mut Vec<Dispatch>,
    ) -> Result<(), Error> {
        let past = || internal("a depth copy's row past the device's storage buffer range");
        if let Some(whole) = self.dispatch(raw, start, depths, [depths.rows, depths.images]) {
            into.push(whole);
            return Ok(());
        }
        let image_bytes = 4 * u64::from(depths.image_words);
        let row_bytes = 4 * u64::from(depths.row_words);
        for image in 0..u64::from(depths.images) {
            let at = start + image * image_bytes;
            if let Some(one) = self.dispatch(raw, at, depths, [depths.rows, 1]) {
                into.push(one);
                continue;
            }
            // Rows of the image, as many as one binding holds (a row at least).
            let slack = self.align - 4;
            let fit = (self
                .most
                .saturating_sub(slack + 4 * u64::from(depths.width))
                / row_bytes)
                .saturating_add(1)
                .min(u64::from(self.groups[1]));
            let fit = u32::try_from(fit.max(1)).map_err(|_| past())?;
            let mut row = 0;
            while row < depths.rows {
                let rows = fit.min(depths.rows - row);
                let at = at + u64::from(row) * row_bytes;
                into.push(self.dispatch(raw, at, depths, [rows, 1]).ok_or_else(past)?);
                row += rows;
            }
        }
        Ok(())
    }
}

impl BlitCommandEncoder {
    /// The copies `regions` from `buffer`'s `VkBuffer` `raw` (the regions' offsets in it), of the
    /// buffer's bytes `span`, into a depth image, with their depths clamped (the module docs):
    /// each region's bytes copied into a scratch buffer of the encoder at an offset the device
    /// binds, and clamped there. The scratch buffer, its `VkBuffer`, and the regions of the copy
    /// from it into the image; the caller records that copy and retains the scratch buffer.
    pub(super) fn clamped_depth_copies(
        &mut self,
        (buffer, raw, span): (&Buffer, vk::Buffer, Range<u64>),
        regions: &[vk::BufferImageCopy],
    ) -> Result<(Buffer, vk::Buffer, Vec<vk::BufferImageCopy>), Error> {
        let device = self.core().device().clone();
        let align = Bounds::of(&device).align;
        // Each region's bytes, from its first depth to past its last, at its place in the scratch.
        let mut placed = Vec::with_capacity(regions.len());
        let mut length = 0u64;
        for region in regions {
            let reach = match Depths::of(region)? {
                Some(depths) => depths.reach(depths.images, depths.rows),
                None => 0,
            };
            let at = length.next_multiple_of(align);
            placed.push((at, reach));
            length = at + reach;
        }
        let scratch = device.own_buffer(length.max(4), ResourceOptions::private())?;
        let (through, base) = scratch.at(0);
        let moves: Vec<vk::BufferCopy> = (regions.iter().zip(&placed))
            .filter(|(_, (_, reach))| *reach > 0)
            .map(|(region, &(at, reach))| vk::BufferCopy {
                src_offset: region.buffer_offset,
                dst_offset: base + at,
                size: reach,
            })
            .collect();
        let uses = [reads(buffer, span), writes(&scratch, 0..length.max(4))];
        self.command(&uses, |device, command_buffer| {
            // SAFETY: the encoder's command buffer is recording outside a render pass; both
            // buffers are live and retained until it completes, each move lies in both (the
            // regions' bytes, which the plan checked, and the scratch's places), and the scratch
            // is another buffer whose places do not overlap.
            unsafe { device.cmd_copy_buffer(command_buffer, raw, through, &moves) };
        })?;
        let staged: Vec<vk::BufferImageCopy> = (regions.iter().zip(&placed))
            .map(|(region, &(at, _))| vk::BufferImageCopy {
                buffer_offset: base + at,
                ..*region
            })
            .collect();
        self.clamp_depths(&scratch, &staged)?;
        Ok((scratch, through, staged))
    }

    /// Clamp the depths of `regions`, copies from `scratch` (their offsets in its `VkBuffer`) into
    /// a depth image, in place (the module docs), before the copy: one command a dispatch.
    pub(super) fn clamp_depths(
        &mut self,
        scratch: &Buffer,
        regions: &[vk::BufferImageCopy],
    ) -> Result<(), Error> {
        let device = self.core().device().clone();
        let bounds = Bounds::of(&device);
        let raw = scratch.at(0);
        let mut dispatches = Vec::new();
        for region in regions {
            if let Some(depths) = Depths::of(region)? {
                bounds.dispatches(raw, region.buffer_offset, &depths, &mut dispatches)?;
            }
        }
        if dispatches.is_empty() {
            return Ok(());
        }
        let kernel = device.owned_kernel(OwnedModule::DepthClamp)?;
        let sets: Vec<OwnedSet> = (dispatches.iter())
            .map(|dispatch| OwnedSet {
                images: Vec::new(),
                storage_images: Vec::new(),
                buffers: vec![dispatch.binding],
            })
            .collect();
        let binds = OwnedBinds::new(&device, &kernel, &sets)?;
        let stage = vk::PipelineStageFlags2::COMPUTE_SHADER;
        for (index, dispatch) in dispatches.iter().enumerate() {
            let mut words = writes_by(
                scratch,
                dispatch.bytes.clone(),
                Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE),
            );
            words.read = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ);
            let (binds, kernel) = (&binds, &kernel);
            self.command_in(stage, &[words], |functions, raw| {
                // SAFETY: the encoder's command buffer is recording outside a render pass; the
                // kernel is the device's, the bound words are the scratch's, live and retained
                // until it completes, and the shape and its workgroups stay inside the binding
                // (`Bounds::dispatch`).
                unsafe {
                    functions.cmd_bind_pipeline(
                        raw,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.pipeline,
                    );
                    binds.bind(functions, raw, kernel, index);
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
            })?;
        }
        if let OwnedBinds::Sets(sets) = binds {
            self.core().retain(sets);
        }
        device.shared().counts.depth_upload_clamped();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{clamp_depth_bytes, Bounds, Depths, Dispatch};
    use ash::vk;
    use metal2vulkan::bind::clamped_depth;
    use std::collections::BTreeSet;

    /// The bytes of the depths `dispatch` clamps, by its binding and shape: each a 4-byte word's
    /// first byte in the `VkBuffer`.
    fn clamped_words(dispatch: &Dispatch) -> Vec<u64> {
        let word = |i: usize| {
            let bytes = &dispatch.shape[4 * i..4 * i + 4];
            u64::from(u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let (first, width, row_words, image_words) = (word(0), word(1), word(2), word(3));
        let (_, offset, range) = dispatch.binding;
        let [_, rows, images] = dispatch.groups.map(u64::from);
        let mut words = Vec::new();
        for z in 0..images {
            for y in 0..rows {
                for x in 0..width {
                    let at = offset + 4 * (first + z * image_words + y * row_words + x);
                    assert!(at + 4 <= offset + range, "a word past the binding");
                    words.push(at);
                }
            }
        }
        words
    }

    /// The pieces of a region of 3 images of 5 rows of 37 depths, rows 40 words apart and images
    /// 230, from byte 1028 of the buffer, under bounds that bind it whole, an image at a time, and
    /// rows at a time: every depth of the region clamped once, no other word, each binding at the
    /// alignment and within the range: 1, 3, 9 (two rows a piece) and 15 pieces.
    #[test]
    fn pieces_clamp_each_depth_of_a_region_once() {
        let region = vk::BufferImageCopy {
            buffer_offset: 1028,
            buffer_row_length: 40,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 3,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width: 37,
                height: 5,
                depth: 1,
            },
        };
        let depths = Depths::of(&region).unwrap().unwrap();
        let want: Vec<u64> = (0..3u64)
            .flat_map(|z| (0..5u64).flat_map(move |y| (0..37u64).map(move |x| (z, y, x))))
            .map(|(z, y, x)| 1028 + 4 * (z * 200 + y * 40 + x))
            .collect();
        for (most, pieces) in [(1 << 20, 1), (1024, 3), (400, 9), (330, 15)] {
            let bounds = Bounds {
                align: 64,
                most,
                groups: [65535; 3],
            };
            let mut dispatches = Vec::new();
            let raw = (vk::Buffer::null(), 1024);
            bounds
                .dispatches(raw, region.buffer_offset, &depths, &mut dispatches)
                .unwrap();
            assert_eq!(dispatches.len(), pieces, "pieces under a range of {most}");
            let mut got: Vec<u64> = dispatches.iter().flat_map(clamped_words).collect();
            let unique: BTreeSet<u64> = got.iter().copied().collect();
            assert_eq!(
                unique.len(),
                got.len(),
                "a depth clamped twice under {most}"
            );
            got.sort_unstable();
            assert_eq!(got, want, "the depths clamped under {most}");
            for dispatch in &dispatches {
                let (_, offset, range) = dispatch.binding;
                assert_eq!(offset % 64, 0, "a binding off the alignment");
                assert!(range <= most, "a binding past the range");
                let words = clamped_words(dispatch);
                let (low, high) = (words[0] - 1024, words[words.len() - 1] + 4 - 1024);
                assert_eq!(dispatch.bytes, low..high, "the bytes its use names");
            }
        }
    }

    /// The CPU's clamp of packed depths: each word as `bind::clamped_depth` gives it.
    #[test]
    fn packed_depths_clamp_word_by_word() {
        let words = [
            1.5f32.to_bits(),
            0x7fc0_0000,
            0.25f32.to_bits(),
            0xbf80_0000,
        ];
        let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        clamp_depth_bytes(&mut bytes);
        let got: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        assert_eq!(got, words.map(clamped_depth));
        assert_eq!(got, [0x3f80_0000, 0x3f80_0000, 0.25f32.to_bits(), 0]);
    }
}
