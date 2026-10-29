//! A linear attachment's pass rendered through the image its texture keeps, merged back into the
//! surface's bytes word by word (`linear_attachment.rs`).
//!
//! A pass that renders a bound linear attachment through the image its texture keeps copies the
//! surface's box into it and renders there. A copy back of the box would write every pixel of the
//! box, those no draw wrote too, with what the copy in read: where anything else wrote the
//! surface meanwhile (the CPU over its host pages, another queue), the copy back would put the
//! old bytes back, where Metal on a Mac family GPU never writes a pixel no draw wrote (the
//! consumer's x86 guest, its alert's corners and hover fills). So the pass merges instead, with the
//! shadow merge's kernel (`bind::shadow_merge_module`, `shadow_merge.rs`):
//!
//! - **Kept.** After the copy in, the box's texels of the kept image are copied into the *kept*
//!   part of a scratch buffer, row after row (each layer's from a multiple of 256 bytes).
//! - **Merged.** After the pass, they are copied into the *changed* part, laid out alike, and the
//!   kernel binds the surface's own bytes as its current buffer, over the box's rows: a word whose
//!   kept and changed words are equal is neither read nor written; in a word the pass changed,
//!   each byte it changed becomes the pass's, and the others keep the surface's. A byte the pass
//!   changed and another writer wrote meanwhile is a race Metal leaves unordered.
//! - **Words.** The device's linear rows and their offset are whole words
//!   (`VK_NV_linear_color_attachment`'s images). (the 3090's `padded_loads` lane) A texel is 1, 2,
//!   4, 8 or 16 bytes: an `R10UnormX6` or `R8Unorm` box's row may start and end inside a word,
//!   where the form refused every pass of it (internal, "a linear attachment whose rows are not
//!   whole words"). Each row is taken from its first byte's word to its last byte's, the scratch's
//!   rows laid out alike: the copies write the box's texels at their place in those words, and the
//!   bytes before and after them are zero in both parts (the scratch is filled with zero once,
//!   before the first keep), so the merge keeps the surface's there. A word the pass changed is
//!   read and written whole, as one whose texel's channels it changed in part: another writer's
//!   write to its other bytes during the merge is a race the merge does not order.
//! - **Words across the invocations.** The kernel gives an invocation four words a workgroup
//!   apart, not four in a row, so one store's words are consecutive across the invocations that
//!   run together (`bind::shadow_merge`'s module docs). Its dispatches and the words they examine
//!   are counted (`WorkCounts::shadow_merge_dispatches_planned`,
//!   `WorkCounts::shadow_merge_words_examined`).
//! - **Windows.** The current buffer is bound from the box's first byte rounded down to 256 (the
//!   largest `minStorageBufferOffsetAlignment`), its first word pushed; each layer's rows in bands
//!   no binding of which passes `maxStorageBufferRange` and no dispatch of which passes
//!   `maxComputeWorkGroupCount[0]` workgroups, each band's kept rows starting at a multiple of
//!   256 bytes. One dispatch per band.
//! - **The scratch** is the pass's (`Device::scratch_buffer`, no zeroing: every word the kernel
//!   reads, the copies write), retained until its command buffer completes.
//!
//! - **Reloaded at a barrier.** A draw after a memory barrier inside the pass may read the
//!   surface's bytes (a sampled texture over them, a buffer): before the render pass instance that
//!   follows the barrier, the words the instances before changed are merged back, the box is
//!   copied in again from the surface, and kept again ([`LinearMerge::reload_in`]), so that draw
//!   reads what the pass drew, and the merge after the pass writes only what the pass changed
//!   since (`render.rs`'s `reload_at`, `render/pass_steps.rs`).
//!
//! It costs the scratch, the copy into the kept part, the copy into the changed part and the
//! dispatches, in place of one copy back of the box; and, at each barrier, a merge, a copy in
//! and a keep.

use super::buffer::Buffer;
use super::encoder::EncoderCore;
use super::error::{internal, Error};
use super::owned_kernel::{OwnedBinds, OwnedKernel, OwnedModule, OwnedSet};
use super::render_pass::Target;
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, Use};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{ShadowMergeModule, ShadowMergeStores};

/// What the offsets of the bindings are rounded to: a multiple of every
/// `minStorageBufferOffsetAlignment`.
const ALIGNMENT: u64 = 256;

/// One dispatch of the merge: its workgroups and push bytes.
struct Dispatch {
    groups: u32,
    shape: [u8; 16],
}

/// The merge of one linear attachment's pass (the module docs).
pub(super) struct LinearMerge {
    /// The image the pass renders, and where in it the box lies.
    backing: vk::Image,
    /// The surface's own image, and the copy of the box from it into `backing` (a reload's).
    surface: vk::Image,
    copy_in: vk::ImageCopy,
    /// The kept part's copies, then the changed part's: the same regions, `parts` bytes apart.
    regions: Vec<vk::BufferImageCopy>,
    /// The scratch's `VkBuffer`, and where its changed part starts past its kept one's.
    scratch: vk::Buffer,
    part: u64,
    /// Where the scratch's two parts lie in its `VkBuffer`, filled with zero before the first
    /// keep, where a row of the box starts or ends inside a word (the module's words).
    padding: Option<(u64, u64)>,
    _held: Buffer,
    kernel: OwnedKernel,
    /// One set 0 per dispatch, each binding its band of the kept part, of the changed part and of
    /// the surface's bytes; pushed where the kernel's set 0 is pushed (`OwnedBinds`).
    binds: OwnedBinds,
    dispatches: Vec<Dispatch>,
    /// The keep's uses: a read of the image the pass renders.
    keep_uses: Vec<Use<ResourceKey>>,
    /// The merge's uses: that read, and a read and a write of the surface's bytes.
    merge_uses: Vec<Use<ResourceKey>>,
}

impl LinearMerge {
    /// The merge of the box `[x0, y0, x1, y1]` (half-open, inside the render area) of the pass
    /// rendering `own`, a bound linear attachment's target, through `backing`, the image its
    /// texture keeps: its scratch, binds and dispatches on `device`. Refused (internal) for an
    /// attachment whose rows or their offset are not whole words, whose texel is not 1, 2, 4, 8
    /// or 16 bytes, or whose bands no binding holds, which no device's linear attachment is; and
    /// as `scratch_buffer` refuses the scratch.
    pub(super) fn new(
        device: &Device,
        backing: &Target,
        own: &Target,
        [x0, y0, x1, y1]: [u32; 4],
    ) -> Result<LinearMerge, Error> {
        let texels = own
            .texture
            .buffer_texels()
            .ok_or_else(|| internal("a linear attachment merged without its buffer's bytes"))?;
        let texel = texels.texel_bytes();
        let layout = &texels.layout;
        let (width, rows) = (u64::from(x1 - x0), u64::from(y1 - y0));
        // Each row from its first byte's word to its last byte's (the module's words): `lead`
        // bytes before the box's first texel, and the row's bytes rounded up to a word. A whole
        // word holds a whole number of texels of 1, 2 or 4 bytes, and one of 8 or 16 bytes is
        // whole words, so `lead` and `row_bytes` are whole texels.
        let lead = u64::from(x0) * texel % 4;
        let row_bytes = (lead + width * texel).next_multiple_of(4);
        if own.level != 0
            || ![1, 2, 4, 8, 16].contains(&texel)
            || !layout.row_pitch.is_multiple_of(4)
            || !layout.offset.is_multiple_of(4)
            || !layout.slice_pitch.is_multiple_of(4)
            || width == 0
            || rows == 0
        {
            return Err(internal(
                "a linear attachment whose rows are not whole words",
            ));
        }
        let layers = u64::from(own.layers.max(1));
        let layer_part = (rows * row_bytes).next_multiple_of(ALIGNMENT);
        let part = layers * layer_part;
        let held = device.scratch_buffer(2 * part)?;
        let (scratch, base) = held.at(0);
        let padding = (lead > 0 || row_bytes > width * texel).then_some((base, 2 * part));
        let limits = &device.properties().description.limits;
        let range = u64::from(limits.max_storage_buffer_range);
        let groups = u64::from(limits.max_compute_work_group_count[0]);
        let per_group = u64::from(ShadowMergeModule::LOCAL_SIZE * ShadowMergeModule::WORDS);
        let words = row_bytes / 4;
        let pitch = layout.row_pitch;
        // A band's rows: within every limit, and a whole number of steps, so the next band's kept
        // rows start at a multiple of the alignment.
        let step = ALIGNMENT / gcd(row_bytes, ALIGNMENT);
        let by_current = match range.checked_sub(ALIGNMENT + row_bytes) {
            Some(rest) => rest / pitch + 1,
            None => 0,
        };
        let by_kept = range / row_bytes;
        let by_groups = groups * per_group / words;
        let most = rows.min(by_current).min(by_kept).min(by_groups);
        let band = match most == rows {
            true => rows,
            false => most / step * step,
        };
        if band == 0 {
            return Err(internal("a linear attachment's rows no merge window holds"));
        }
        let mut regions = Vec::with_capacity(layers as usize);
        let mut sets = Vec::new();
        let mut dispatches = Vec::new();
        // The words the dispatches examine (`WorkCounts::shadow_merge_words_examined`).
        let mut examined = 0;
        let u32_of = |value: u64| {
            u32::try_from(value).map_err(|_| internal("a linear merge past u32 words"))
        };
        for layer in 0..layers {
            let kept_at = base + layer * layer_part;
            regions.push(vk::BufferImageCopy {
                buffer_offset: kept_at + lead,
                buffer_row_length: u32_of(row_bytes / texel)?,
                buffer_image_height: y1 - y0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: backing.level,
                    base_array_layer: backing.layer + layer as u32,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: x0 as i32,
                    y: y0 as i32,
                    z: 0,
                },
                image_extent: vk::Extent3D {
                    width: x1 - x0,
                    height: y1 - y0,
                    depth: 1,
                },
            });
            let slice = layout.offset + (u64::from(own.layer) + layer) * layout.slice_pitch;
            let mut start = 0;
            while start < rows {
                let count = band.min(rows - start);
                let first = slice + (u64::from(y0) + start) * pitch + u64::from(x0) * texel - lead;
                let (surface, at) = texels.buffer.at(first);
                let bound = at / ALIGNMENT * ALIGNMENT;
                let skip = at - bound;
                let kept = kept_at + start * row_bytes;
                let size = count * row_bytes;
                sets.push(OwnedSet {
                    images: Vec::new(),
                    storage_images: Vec::new(),
                    buffers: vec![
                        (scratch, kept, size),
                        (scratch, kept + part, size),
                        (surface, bound, skip + (count - 1) * pitch + row_bytes),
                    ],
                });
                let total = count * words;
                dispatches.push(Dispatch {
                    groups: ShadowMergeModule::groups(u32_of(total)?),
                    shape: ShadowMergeModule::shape(
                        u32_of(words)?,
                        u32_of(total)?,
                        u32_of(pitch / 4)?,
                        u32_of(skip / 4)?,
                    ),
                });
                examined += total;
                start += count;
            }
        }
        (device.shared().counts).shadow_merge_planned(dispatches.len(), examined);
        let kernel = device.owned_kernel(OwnedModule::ShadowMerge(ShadowMergeStores::Words))?;
        let binds = OwnedBinds::new(device, &kernel, &sets)?;
        let levels = backing.level..backing.level + 1;
        let backing_layers = backing.layer..backing.layer + backing.layers.max(1);
        let read = Access::new(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let backing_read = backing
            .texture
            .image_use(levels, backing_layers, read, false)?;
        let storage = |access| Access::new(vk::PipelineStageFlags2::COMPUTE_SHADER, access);
        let surface = own.texture.bound_use(
            storage(vk::AccessFlags2::SHADER_STORAGE_READ),
            storage(vk::AccessFlags2::SHADER_STORAGE_WRITE),
        );
        let layers_of = |target: &Target| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: target.level,
            base_array_layer: target.layer,
            layer_count: own.layers.max(1),
        };
        let offset = vk::Offset3D {
            x: x0 as i32,
            y: y0 as i32,
            z: 0,
        };
        let copy_in = vk::ImageCopy {
            src_subresource: layers_of(own),
            src_offset: offset,
            dst_subresource: layers_of(backing),
            dst_offset: offset,
            extent: vk::Extent3D {
                width: x1 - x0,
                height: y1 - y0,
                depth: 1,
            },
        };
        Ok(LinearMerge {
            backing: backing.texture.raw_image()?,
            surface: own.texture.raw_image()?,
            copy_in,
            regions,
            scratch,
            part,
            padding,
            _held: held,
            kernel,
            binds,
            dispatches,
            keep_uses: vec![backing_read.clone()],
            merge_uses: vec![backing_read, surface],
        })
    }

    /// Record the keep in `core`'s command buffer: after the copy in, before the pass, between
    /// barriers from everything before and to everything after; first, where rows start or end
    /// inside a word, the scratch's fill with zero.
    pub(super) fn record_keep(&self, core: &mut EncoderCore) {
        if let Some((offset, size)) = self.padding {
            core.command(vk::PipelineStageFlags2::COPY, &[], |device, raw| {
                // SAFETY: the encoder's command buffer is recording outside a render pass; the
                // scratch is the merge's, retained with it, a transfer destination, and the
                // range is its two parts, whole words.
                unsafe { device.cmd_fill_buffer(raw, self.scratch, offset, size, 0) }
            });
        }
        core.barrier(Barrier::everything());
        core.command(
            vk::PipelineStageFlags2::COPY,
            &self.keep_uses,
            |device, raw| {
                // SAFETY: the encoder's command buffer is recording outside a render pass, and
                // the encoder retains the merge and the textures until it completes.
                unsafe { self.keep_in(device, raw) }
            },
        );
        core.barrier(Barrier::everything());
    }

    /// Record the merge in `core`'s command buffer: after the pass, in place of its copy back,
    /// between barriers from everything before and to everything after.
    pub(super) fn record_merge(&self, core: &mut EncoderCore) {
        let stages = vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER;
        core.barrier(Barrier::everything());
        core.command(stages, &self.merge_uses, |device, raw| {
            // SAFETY: as in `record_keep`.
            unsafe { self.merge_in(device, raw) }
        });
        core.barrier(Barrier::everything());
    }

    /// Record, into `command_buffer` between two render pass instances of the pass, the merge of
    /// what the instances before changed, the box's copy in from the surface again, and its
    /// keep, each between barriers (the module docs). The pass's uses hold the keep's and the
    /// merge's: the steps of its first copy in, keep and last merge.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; the
    /// merge, the images and the scratch live until it completes, and every image rests in
    /// `GENERAL`.
    pub(super) unsafe fn reload_in(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract; the copy's region lies in both images (the box, inside
        // the render area), whose transfer usage the copies in already take.
        unsafe {
            Barrier::everything().record(device, command_buffer);
            self.merge_in(device, command_buffer);
            Barrier::everything().record(device, command_buffer);
            device.cmd_copy_image(
                command_buffer,
                self.surface,
                vk::ImageLayout::GENERAL,
                self.backing,
                vk::ImageLayout::GENERAL,
                &[self.copy_in],
            );
            Barrier::everything().record(device, command_buffer);
            self.keep_in(device, command_buffer);
            Barrier::everything().record(device, command_buffer);
        }
    }

    /// The keep's copy.
    ///
    /// # Safety
    ///
    /// As for [`LinearMerge::reload_in`].
    unsafe fn keep_in(&self, device: &ash::Device, raw: vk::CommandBuffer) {
        // SAFETY: the caller's contract; the image rests in `GENERAL` with transfer source usage
        // (the kept image's), and each region lies in it and in the scratch's kept part.
        unsafe {
            device.cmd_copy_image_to_buffer(
                raw,
                self.backing,
                vk::ImageLayout::GENERAL,
                self.scratch,
                &self.regions,
            );
        }
    }

    /// The merge's copy into the changed part and its dispatches, a barrier between.
    ///
    /// # Safety
    ///
    /// As for [`LinearMerge::reload_in`].
    unsafe fn merge_in(&self, device: &Vk, raw: vk::CommandBuffer) {
        let changed: Vec<vk::BufferImageCopy> = self
            .regions
            .iter()
            .map(|region| vk::BufferImageCopy {
                buffer_offset: region.buffer_offset + self.part,
                ..*region
            })
            .collect();
        // SAFETY: as in `keep_in`; the binds name bands inside the scratch's parts and the
        // surface's bytes, one per dispatch, and the kernel's push range holds the shape.
        unsafe {
            device.cmd_copy_image_to_buffer(
                raw,
                self.backing,
                vk::ImageLayout::GENERAL,
                self.scratch,
                &changed,
            );
            Barrier::everything().record(device, raw);
            device.cmd_bind_pipeline(raw, vk::PipelineBindPoint::COMPUTE, self.kernel.pipeline);
            for (index, dispatch) in self.dispatches.iter().enumerate() {
                self.binds.bind(device, raw, &self.kernel, index);
                device.cmd_push_constants(
                    raw,
                    self.kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &dispatch.shape,
                );
                device.cmd_dispatch(raw, dispatch.groups, 1, 1);
            }
        }
    }
}

/// The greatest common divisor of `a` and `b`.
pub(super) fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
