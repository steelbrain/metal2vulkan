//! A blit into a heap's two-plane extended-range texture whose planes lie in the heap's bytes
//! (`AliasRoute::Planes`), as Metal's blit writes them there.
//!
//! **Metal's answer** (an Apple M4 under macOS 27.0.1, and an Apple M3 Max under macOS 26.7; our
//! own probe: every byte of a placement heap, `Shared` and `Private`, filled with random bytes and
//! read through a buffer over the whole heap before and after one blit between two of its textures;
//! 550 and 551, widths on 16, on 4 and off 4, whole levels, regions and single rows; each blit
//! three times on the same bytes and the list twice, every byte alike, and alike on both chips):
//! the copy writes the heap's bytes at Metal's blit rows (`bind::plane_bytes_copy_module`'s model),
//! the alpha bytes past the alpha plane among them, over whatever else the heap holds there; it
//! reads the source's bytes the same way, as they were before the copy where its writes reach them.
//! A copy whose alpha bytes pass the heap's end raised no error; the bytes past it are not the
//! heap's.
//!
//! **Over a surface's planes** (our own probe `xrs2`: an Apple M4 under macOS
//! 27.0.1, a copy between such a texture over a surface and a texture of its own, each way, and
//! into a second surface's, each twice over the same bytes, on 7 surfaces and the consumer's 384 x
//! 256 `'b3a8'` one, `Shared` and `Managed`): Metal's blit addresses both of such a texture's
//! planes at rows of plane 0's GPU rows (its `bytesPerRow` rounded down to 16, where a texture of
//! its own memory's are `⌊4 w / 16⌋ · 16`), each from its plane's GPU first byte, and it writes and
//! reads the surface's bytes there as a heap's are, the alpha bytes past plane 1's rows and past
//! the plane among them; a byte past the surface's end is not written and reads 0. Every byte of
//! every surface is that model's, each copy twice alike, but where a copy's rows overlap in a
//! destination of its own memory (Metal's order varies there, the model keeps the later row's,
//! announced as below).
//!
//! **Held**: one command of the encoder. The source's bytes the copy reads, from its planes in a
//! heap's or a surface's bytes or from its image's texels, are copied into a scratch buffer; the
//! kernel then writes the destination's bytes in place, and the destination's image is copied in
//! from them at its next use (`shadow.rs`). A copy from a surface's planes into a texture of the
//! device's memory reads the source's bytes the same way and writes the destination's texels
//! (`plane_texture_copy`). Announced as `two-plane-texture-copy`, as near-equivalents: a copy
//! whose alpha bytes pass a heap's end (Metal's blit writes past it; the copy writes the heap's
//! bytes alone) and a source whose read bytes pass its heap's end (read as 0). Refused, Metal's
//! answer not measured: a copy between two textures over the bytes of one buffer of which one is
//! over a surface's planes, and a surface's planes past a storage buffer.

use super::super::blit::{reads, BlitCommandEncoder, TRANSFER};
use super::super::blit_plan::ImageCopyRegion;
use super::super::copy_plan::ImageRegion;
use super::super::device::functions::Vk;
use super::super::error::{internal, Error};
use super::super::owned_kernel::{OwnedModule, OwnedSets};
use super::super::resource::ResourceKey;
use super::super::texture::heap_two_planes::TwoPlaneHeapLayout;
use super::super::texture::{buffer_image_copy, Texture};
use super::super::tracker::{Access, Barrier, Use, UseRange};
use super::super::values::ResourceOptions;
use super::super::Buffer;
use super::{announce_texture_copy, unmeasured, TEXEL_BYTES};
use ash::vk;
use metal2vulkan::bind::{PlaneBytes, PlaneBytesCopyModule, PlaneRows, PlaneSource};
use metal2vulkan::vk::ImageAspectFlags;
use std::ops::Range;

/// The compute shader's reads and writes of `range` of `buffer`.
fn shader_writes(buffer: &Buffer, range: Range<u64>) -> [Use<ResourceKey>; 2] {
    let (identity, range) = buffer.tracked(range);
    let stage = vk::PipelineStageFlags2::COMPUTE_SHADER;
    let read = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ);
    let write = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    let bytes = UseRange::Bytes(range);
    [
        Use::reads(identity.key, identity.tracked, bytes.clone(), read),
        Use::writes(identity.key, identity.tracked, bytes, write),
    ]
}

/// Where the source's bytes go before the kernel reads them.
pub(super) enum Staged {
    /// Ranges of the bytes its planes lie in (`VkBuffer`, offset, size), copied one after the
    /// other.
    Bytes(vk::Buffer, Vec<(u64, u64)>),
    /// Its image's one level, 8 bytes a texel.
    Texels(vk::Image, vk::BufferImageCopy),
}

impl Staged {
    /// Record the copy of the source's bytes into `scratch` from `at`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass; the source and `scratch` are live
    /// until it completes, the source's image in the `GENERAL` layout; every range lies inside
    /// its buffer, and the staged bytes inside `scratch` from `at`.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        scratch: vk::Buffer,
        at: u64,
    ) {
        match self {
            Staged::Bytes(raw, regions) => {
                let mut into = at;
                let copies: Vec<vk::BufferCopy> = regions
                    .iter()
                    .map(|&(from, size)| {
                        let copy = vk::BufferCopy {
                            src_offset: from,
                            dst_offset: into,
                            size,
                        };
                        into += size;
                        copy
                    })
                    .collect();
                // SAFETY: the caller's contract.
                unsafe { device.cmd_copy_buffer(command_buffer, *raw, scratch, &copies) };
            }
            Staged::Texels(image, region) => {
                let region = vk::BufferImageCopy {
                    buffer_offset: at,
                    ..*region
                };
                // SAFETY: the caller's contract.
                unsafe {
                    device.cmd_copy_image_to_buffer(
                        command_buffer,
                        *image,
                        vk::ImageLayout::GENERAL,
                        scratch,
                        std::slice::from_ref(&region),
                    );
                }
            }
        }
    }
}

/// Where a two-plane texture's planes lie in a buffer's bytes, as Metal's blit addresses them: a
/// heap's one-level 2D texture's in the heap's bytes or a texture's over a surface's two planes.
pub(in super::super) struct BytePlanes<'a> {
    /// The buffer whose bytes hold the planes.
    pub(in super::super) bytes: &'a Buffer,
    /// Plane 0's first byte in the buffer.
    pub(in super::super) color: u64,
    /// The alpha plane's first byte in the buffer.
    pub(in super::super) alpha: u64,
    /// The rows Metal's blit addresses both planes at.
    pub(in super::super) blit: u64,
    /// Whether the buffer is a surface's, past whose end Metal's blit writes nothing and reads 0;
    /// past a heap's it reaches other memory.
    pub(in super::super) surface: bool,
}

impl BytePlanes<'_> {
    /// The planes of `texture`, whose one level is `level` texels (width, height), where they lie
    /// in a buffer's bytes; `None` for a texture of the device's memory and a heap's texture whose
    /// heap holds no bytes.
    pub(in super::super) fn of(texture: &Texture, level: [u32; 2]) -> Option<BytePlanes<'_>> {
        if let Some((heap, at)) = texture.heap_planes() {
            let layout = TwoPlaneHeapLayout::of(level.map(u64::from));
            return Some(BytePlanes {
                bytes: heap.bytes()?,
                color: at,
                alpha: at + layout.alpha_at,
                blit: u64::from(PlaneRows::of(level[0]).blit),
                surface: false,
            });
        }
        let texels = texture.buffer_texels()?;
        let alpha = texels.alpha()?;
        Some(BytePlanes {
            bytes: &texels.buffer,
            color: texels.layout.offset,
            alpha: alpha.offset,
            blit: texels.layout.row_pitch,
            surface: true,
        })
    }

    /// The bytes a copy of `size` texels (width, height; neither 0) from `origin` reaches: its
    /// first color byte and the end of its color bytes, and the same of its alpha bytes (which
    /// may pass the buffer's end).
    fn reach(&self, [x, y]: [u64; 2], [width, height]: [u64; 2]) -> [Range<u64>; 2] {
        let color = self.color + y * self.blit + 4 * x;
        let alpha = self.alpha + y * self.blit + x;
        let last = (height - 1) * self.blit;
        [color..color + last + 4 * width, alpha..alpha + last + width]
    }

    /// Whether these planes and `other`'s lie in one buffer's bytes.
    fn shares_bytes(&self, other: &BytePlanes<'_>) -> bool {
        self.bytes.at(0).0 == other.bytes.at(0).0
    }
}

/// A source's bytes staged for a kernel: what to copy into the scratch, the bytes that takes, the
/// source's planes there, its uses, and whether the copy reads past its heap's end.
pub(super) struct StagedPlanes {
    pub(super) staged: Staged,
    pub(super) bytes: u64,
    pub(super) planes: PlaneBytes,
    pub(super) uses: Vec<Use<ResourceKey>>,
    pub(super) past_heap: bool,
}

/// Stage the bytes of `planes` a copy of `size` texels (width, height; neither 0) from `origin`
/// reads: its color bytes, then its alpha bytes to the buffer's end, from the words holding them.
pub(super) fn stage_planes(
    planes: &BytePlanes<'_>,
    origin: [u64; 2],
    size: [u64; 2],
) -> Result<StagedPlanes, Error> {
    let narrow = |v: u64| u32::try_from(v).map_err(|_| internal("a plane bytes copy past u32"));
    let [color, alpha] = planes.reach(origin, size);
    let bytes = planes.bytes;
    let end = bytes.length();
    let (raw, color_from) = bytes.at(color.start);
    if !color_from.is_multiple_of(4) {
        return Err(internal("a two-plane texture's bytes off 4 bytes"));
    }
    let color_size = (color.end - color.start).next_multiple_of(4);
    let mut regions = vec![(color_from, color_size)];
    let mut uses = vec![reads(bytes, color.clone())];
    let alpha_end = alpha.end.min(end);
    let (alpha_first, alpha_size) = if alpha.start < alpha_end {
        let (_, alpha_from) = bytes.at(alpha.start);
        let first = alpha_from / 4 * 4;
        let last = (alpha_from + (alpha_end - alpha.start)).next_multiple_of(4);
        regions.push((first, last - first));
        uses.push(reads(bytes, alpha.start..alpha_end));
        (alpha_from - first, last - first)
    } else {
        (0, 0)
    };
    let staged_bytes = color_size + alpha_size;
    Ok(StagedPlanes {
        staged: Staged::Bytes(raw, regions),
        bytes: staged_bytes,
        planes: PlaneBytes::new(
            0,
            narrow(color_size + alpha_first)?,
            narrow(planes.blit)?,
            narrow(staged_bytes)?,
        ),
        uses,
        past_heap: alpha.end > end && !planes.surface,
    })
}

impl BlitCommandEncoder {
    /// Record `copy` from `source`'s one level (`from` texels, width and height) into
    /// `destination`'s, whose planes lie in a buffer's bytes as `to_planes`, as Metal's blit
    /// writes them (the module docs). `false`, recording nothing, where a binding
    /// would pass the device's `maxStorageBufferRange` or a dispatch its workgroup counts.
    pub(in super::super) fn plane_bytes_copy(
        &mut self,
        source: &Texture,
        destination: &Texture,
        to_planes: &BytePlanes<'_>,
        from: [u32; 2],
        copy: &ImageCopyRegion,
    ) -> Result<bool, Error> {
        let device = self.core().device().clone();
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let range = u64::from(limits.max_storage_buffer_range).min(u64::from(u32::MAX));
        let per_dispatch = limits.max_compute_work_group_count[0].max(1);
        let narrow = |v: u64| u32::try_from(v).map_err(|_| internal("a plane bytes copy past u32"));
        let [width, height, _] = copy.src.size.map(u64::from);
        let [sx, sy, _] = copy.src.origin.map(u64::from);
        let [dx, dy, _] = copy.dst.origin.map(u64::from);
        if width == 0 || height == 0 {
            return Ok(true);
        }
        let source_planes = BytePlanes::of(source, from);
        // Two textures over one buffer's bytes, one over a surface's planes: not measured.
        if let Some(planes) = &source_planes {
            if (planes.surface || to_planes.surface) && planes.shares_bytes(to_planes) {
                return Err(Error::Unsupported(unmeasured(destination.pixel_format())));
            }
        }
        // The destination's bytes the copy writes, to the buffer's end at most.
        let bytes = to_planes.bytes;
        let [color, alpha] = to_planes.reach([dx, dy], [width, height]);
        let buffer_end = bytes.length();
        if alpha.end > buffer_end && !to_planes.surface {
            announce_texture_copy(
                &device,
                source,
                destination,
                "a copy whose alpha bytes pass the heap's end, where Metal's blit writes past \
                 it; the copy writes the heap's bytes alone",
            );
        }
        let end = alpha.end.min(buffer_end);
        let (raw_bytes, color_at) = bytes.at(color.start);
        if !color_at.is_multiple_of(4) {
            return Err(internal("a two-plane texture's bytes off 4 bytes"));
        }
        let bind_at = color_at / align * align;
        let bind_size = (color_at + (end - color.start) - bind_at).next_multiple_of(4);
        let target = PlaneBytes::new(
            narrow(color_at - bind_at)?,
            narrow(color_at - bind_at + (alpha.start - color.start))?,
            narrow(to_planes.blit)?,
            narrow(bind_size)?,
        );
        // The source: its planes' bytes in a heap's or a surface's bytes, or its image's texels.
        let (staged, source_bytes, plane_source, source_uses) = match &source_planes {
            Some(planes) => {
                let staged = stage_planes(planes, [sx, sy], [width, height])?;
                if staged.past_heap {
                    announce_texture_copy(
                        &device,
                        source,
                        destination,
                        "a copy reading alpha bytes past the source's heap's end, which Metal's \
                         blit reads; the copy reads them as 0",
                    );
                }
                (
                    staged.staged,
                    staged.bytes,
                    PlaneSource::Bytes(staged.planes),
                    staged.uses,
                )
            }
            None => {
                let level = ImageRegion {
                    aspect: ImageAspectFlags::COLOR,
                    level: 0,
                    layer: 0,
                    layers: 1,
                    origin: [0, 0, 0],
                    size: [from[0], from[1], 1],
                };
                let texels = TEXEL_BYTES * u64::from(from[0]) * u64::from(from[1]);
                let uses = vec![source.image_use(
                    0..1,
                    0..1,
                    Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ),
                    false,
                )?];
                let image = source.raw_image()?;
                let region = buffer_image_copy(&level);
                (
                    Staged::Texels(image, region),
                    texels,
                    PlaneSource::Texels {
                        level: from,
                        from: [narrow(sx)?, narrow(sy)?],
                    },
                    uses,
                )
            }
        };
        let invocations =
            PlaneBytesCopyModule::invocations(&target, [narrow(width)?, narrow(height)?]);
        let groups = invocations.div_ceil(u64::from(PlaneBytesCopyModule::LOCAL_SIZE));
        let batches = groups.div_ceil(u64::from(per_dispatch));
        if bind_size > range || source_bytes > range {
            return Ok(false);
        }
        let shape_bytes = u64::from(4 * PlaneBytesCopyModule::SHAPE_WORDS);
        let slot = shape_bytes.next_multiple_of(align);
        let shapes_at = source_bytes.next_multiple_of(align);
        let scratch = device.own_buffer(shapes_at + slot * batches, ResourceOptions::private())?;
        let (raw_scratch, scratch_at) = scratch.at(0);
        if !scratch_at.is_multiple_of(align) {
            return Err(internal("a plane bytes copy scratch off its alignment"));
        }
        let kernel = device.owned_kernel(OwnedModule::PlaneBytesCopy)?;
        let size = [narrow(width)?, narrow(height)?];
        let mut ranges = Vec::new();
        let mut steps = Vec::new();
        for batch in 0..batches {
            let shape_at = scratch_at + shapes_at + slot * batch;
            ranges.push([
                (raw_scratch, scratch_at, source_bytes),
                (raw_bytes, bind_at, bind_size),
                (raw_scratch, shape_at, shape_bytes),
            ]);
            let first = batch * u64::from(per_dispatch);
            let count = u64::from(per_dispatch).min(groups - first);
            let first = narrow(first * u64::from(PlaneBytesCopyModule::LOCAL_SIZE))?;
            let shape = PlaneBytesCopyModule::shape(&plane_source, &target, size, first);
            steps.push((shape_at, shape, narrow(count)?));
        }
        let sets = OwnedSets::write(&device, &kernel, &ranges)?;
        let mut uses = source_uses;
        uses.extend(shader_writes(bytes, color.start..end));
        let descriptor_sets = sets.sets.clone();
        let stages = TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        self.command_in(stages, &uses, |device, command_buffer| {
            // SAFETY: the command buffer is recording, outside a render pass; the textures (and so
            // the heaps and buffers their bytes lie in), the scratch and the sets are retained
            // until it completes; the source's image is in the `GENERAL` layout; every copy and
            // binding lies inside the scratch, the bytes and the source's one level; each step
            // reads what the one before it wrote, past a barrier.
            unsafe {
                for (shape_at, shape, _) in &steps {
                    device.cmd_update_buffer(command_buffer, raw_scratch, *shape_at, shape);
                }
                staged.record(device, command_buffer, raw_scratch, scratch_at);
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
            }
        })?;
        self.core().retain(scratch);
        self.core().retain(sets);
        self.core().retain(source.clone());
        self.core().retain(destination.clone());
        Ok(true)
    }
}
