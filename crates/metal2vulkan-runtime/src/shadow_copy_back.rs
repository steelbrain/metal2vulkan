//! A shadowed texture's copy back as a compute dispatch, where the device runs one faster than
//! its copy (`DriverFact::ShadowCopyBackDispatchFaster`).
//!
//! A texture over a buffer whose rows no image lays as Metal does renders through a shadow image,
//! copied back into the buffer's rows after the commands that write it (`shadow.rs`). The copy is
//! `vkCmdCopyImageToBuffer` on every device. On a device measured to run that copy slower than a
//! dispatch that stores the same words, the shadow of a one-slice 2D texture whose texel is 4, 8
//! or 16 bytes, copied in place, is made `MUTABLE_FORMAT` with the unsigned-integer
//! format of its texel's width listed (`R32_UINT`, `R32G32_UINT` or `R32G32B32A32_UINT`), and
//! sampled usage ([`with_word_view`]), and its copies back are one dispatch of
//! `bind::shadow_copy_back_module` each: invocation `(x, y)` fetches texel `(x0 + x, y0 + y)`
//! through that view and stores its words at the texel's place in the buffer ([`CopyBack`]). A
//! merge's copy of such a shadow into its scratch before the commands, which stores the same
//! texels laid out the same way, is such a dispatch too, and the merge after them fetches the
//! texels the commands left through the view rather than copy them out (`shadow_merge.rs`). On a
//! device also measured to take a merged texture's copy in and its keep faster as one dispatch
//! (`DriverFact::ShadowCopyInDispatchFaster`), the view stores too, and the copy in of the
//! texels a keep keeps, right before it, is that dispatch (`shadow_copy_in.rs`); the other copies
//! in stay copies.
//!
//! **Exact.** The dispatch writes the bytes the copy writes and no other: the texels inside the
//! copy's region (every texel, or the rectangle a pass's scissors bound, `ShadowCopy::area`), each
//! at its row's place; never a byte between the rows or past the texels. Vulkan states that
//! "Views of compatible formats will have the same mapping between texel coordinates and memory
//! locations irrespective of the format, with only the interpretation of the bit pattern
//! changing" (`VkImageViewCreateInfo`): the texel's format and its unsigned-integer twin of its
//! width share a compatibility class, and the twin's components are unconverted unsigned
//! integers, so the components a fetch returns are the texel's bytes in memory order (a 4-byte
//! texel's word through `R32_UINT`, its four bytes through `R8G8B8A8_UINT`, which the kernel
//! packs into its word; a wider texel's words), which the copy writes.
//!
//! **Storage usage.** Only on a device with `DriverFact::ShadowCopyInDispatchFaster`, where it
//! stores the view's format, and but for a shadow of a single 32-bit channel's format
//! (`R32_UINT`, `R32_SINT`, `R32_SFLOAT`) whose image has no storage usage of its own; a 4-byte
//! texel's view is then `R8G8B8A8_UINT`. Storage usage beside a 32-bit single-channel format, the
//! image's own or a listed view's, may cost the image its compression where the device keeps
//! atomics apart from it (Mesa's ANV on gfx12.0 then makes the image without its aux surface, so
//! its clears are not fast). Such a shadow's copy in and keep stay two (`shadow_copy_in.rs`).
//!
//! **Synchronization.** The dispatch's uses are a sampled read of the shadow and a storage write
//! of the buffer's bytes in `COMPUTE_SHADER`, which the encoder's tracker orders as any command's
//! (after a pass's `COLOR_ATTACHMENT_OUTPUT` writes; before the buffer's next use). Before the
//! host reads the bytes, the command buffer's last barrier makes every write available to the
//! host (`tracker::Barrier::to_host`, `ALL_COMMANDS`/`MEMORY_WRITE` to `HOST`/`HOST_READ`, or
//! `Barrier::to_all`, which adds every later command of its `VkQueue`: `batch_order.rs`).
//!
//! **Levels.** A tiled texture's shadow holds every level, and its copies move its first alone
//! (`Buffer::new_tiled_texture`): the view is of that level, as the copies' one region is.
//!
//! **Where the copy stays.** A device without the fact, a texel of another width, a texture of
//! more slices, bytes copied through a staging buffer, a region past the device's
//! storage buffer range or workgroup counts, a buffer whose `VkBuffer` went, or a set the device
//! does not make: each copies back as before, the exact form every device takes.

use super::image_plan::{compatible, ImageFacts, ImagePlan};
use super::owned_kernel::{OwnedBinds, OwnedKernel, OwnedModule, OwnedSet};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{ShadowCopyBackModule, ShadowTexelView};
use metal2vulkan::device::DriverFact;
use metal2vulkan::vk::{Format, FormatFeatureFlags2, ImageCreateFlags, ImageUsageFlags};

/// A shadow's sampled unsigned-integer view of its texel's width, through which its copies back
/// dispatch: one of the shadow image's own views (`Image::own_view`), of its own `VkImage`, which
/// stays through a detach of the texture's bytes (the image presenting a stand-in lets go of its
/// other views) and goes as the image drops.
#[derive(Clone, Copy, Debug)]
pub(super) struct WordView {
    pub(super) view: vk::ImageView,
    /// Its format, as the kernels read it (its texel's words: 1, 2 or 4).
    pub(super) texel: ShadowTexelView,
    /// Whether the view is also a storage image, through which a merged texture's copy in and
    /// keep dispatch (`shadow_copy_in.rs`): the module's storage usage.
    pub(super) stores: bool,
}

/// Whether `device`'s shadows copy back by dispatch (the module docs).
pub(super) fn dispatches(device: &Device) -> bool {
    device
        .properties()
        .driver_facts
        .contains(&DriverFact::ShadowCopyBackDispatchFaster)
}

/// Whether `device` takes a merged texture's copy in and its keep as one dispatch, its shadows'
/// views storing (the module's storage usage).
pub(super) fn dispatches_in(device: &Device) -> bool {
    device
        .properties()
        .driver_facts
        .contains(&DriverFact::ShadowCopyInDispatchFaster)
}

/// The unsigned-integer format of a texel `bytes` wide that a [`WordView`] takes, and the view as
/// the kernels read it: a 4-byte texel's bytes where the view `stores`, else its word.
pub(super) fn word_format(bytes: u32, stores: bool) -> Option<(Format, ShadowTexelView)> {
    match (bytes, stores) {
        (4, true) => Some((Format::R8G8B8A8_UINT, ShadowTexelView::R8G8B8A8Uint)),
        (4, false) => Some((Format::R32_UINT, ShadowTexelView::R32Uint)),
        (8, _) => Some((Format::R32G32_UINT, ShadowTexelView::R32G32Uint)),
        (16, _) => Some((Format::R32G32B32A32_UINT, ShadowTexelView::R32G32B32A32Uint)),
        _ => None,
    }
}

/// `plan`, a shadow's image, made to take its [`WordView`]: `MUTABLE_FORMAT`, with the
/// unsigned-integer format of its texel's width listed beside its own (or none listed where it
/// takes any compatible format), sampled usage, and, where `stores_in` (the device's
/// [`dispatches_in`]), storage usage where the device stores the integer format and the image has
/// it already or its own format is not a single 32-bit channel (the module's storage usage;
/// `EXTENDED_USAGE` where its own format takes either not); with that format, the view and
/// whether it stores. `None`, `plan` unchanged, for an image of more than one slice, a texel of
/// another width, a compressed block, or a device that does not sample the integer format. An
/// image of more levels takes it, its view of the first level alone (the module's levels).
pub(super) fn with_word_view(
    mut plan: ImagePlan,
    facts: &ImageFacts<'_>,
    stores_in: bool,
) -> (ImagePlan, Option<(Format, ShadowTexelView, bool)>) {
    let one_slice = plan.layers == 1 && plan.extent[2] == 1;
    let single_channel_word = matches!(
        plan.format,
        Format::R32_UINT | Format::R32_SINT | Format::R32_SFLOAT
    );
    let storing =
        stores_in && (plan.usage.contains(ImageUsageFlags::STORAGE) || !single_channel_word);
    let word = plan
        .format
        .block()
        .and_then(|block| word_format(block.size, storing));
    let sampled = FormatFeatureFlags2::SAMPLED_IMAGE;
    let Some((format, texel)) = word.filter(|&(format, _)| {
        one_slice
            && compatible(plan.format, format, false)
            && facts.optimal(format).contains(sampled)
    }) else {
        return (plan, None);
    };
    let storage = FormatFeatureFlags2::STORAGE_IMAGE;
    let stores = storing && facts.optimal(format).contains(storage);
    let own = facts.optimal(plan.format);
    if !own.contains(sampled) || (stores && !own.contains(storage)) {
        plan.flags |= ImageCreateFlags::EXTENDED_USAGE;
    }
    plan.usage |= ImageUsageFlags::SAMPLED;
    if stores {
        plan.usage |= ImageUsageFlags::STORAGE;
    }
    let any_format =
        plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT) && plan.view_formats.is_empty();
    plan.flags |= ImageCreateFlags::MUTABLE_FORMAT;
    if !any_format {
        if plan.view_formats.is_empty() {
            plan.view_formats.push(plan.format);
        }
        if !plan.view_formats.contains(&format) {
            plan.view_formats.push(format);
        }
    }
    (plan, Some((format, texel, stores)))
}

/// One copy back's dispatch (the module docs): the kernel of its texel's width, its set 0 (the
/// shadow's view and the buffer's bytes from the region's first texel, as they lie now; pushed
/// where the device pushes, `OwnedBinds`), its shape and its workgroups. The command buffer
/// retains it until it completes.
pub(super) struct CopyBack {
    kernel: OwnedKernel,
    binds: OwnedBinds,
    shape: [u8; 16],
    groups: [u32; 2],
}

impl CopyBack {
    /// The dispatch on `device` that copies `region` of the shadow's texels (its offset in the
    /// buffer `buffer`'s `VkBuffer`, its row length, origin and extent) back through `words`;
    /// `None` where the copy takes it (the module docs).
    pub(super) fn new(
        device: &Device,
        words: WordView,
        buffer: vk::Buffer,
        region: &vk::BufferImageCopy,
    ) -> Option<CopyBack> {
        if buffer == vk::Buffer::null() {
            return None;
        }
        let limits = &device.properties().description.limits;
        let texel = 4 * u64::from(words.texel.words());
        let extent = region.image_extent;
        let (width, height) = (extent.width, extent.height);
        let row = match region.buffer_row_length {
            0 => width,
            row => row,
        };
        let [x0, y0] = [region.image_offset.x, region.image_offset.y].map(|o| o as u32);
        let first = region.buffer_offset;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let offset = first - first % align;
        let row_bytes = u64::from(row) * texel;
        let end = first + u64::from(height.max(1) - 1) * row_bytes + u64::from(width) * texel;
        let range = end - offset;
        let groups = ShadowCopyBackModule::groups([width, height]);
        let fits = width > 0
            && height > 0
            && extent.depth == 1
            && first.is_multiple_of(4)
            && x0 < 1 << 16
            && y0 < 1 << 16
            && range <= u64::from(limits.max_storage_buffer_range)
            && groups[0] <= limits.max_compute_work_group_count[0]
            && groups[1] <= limits.max_compute_work_group_count[1];
        if !fits {
            return None;
        }
        // Each below `maxStorageBufferRange`, a `u32`.
        let row_words = (row_bytes / 4) as u32;
        let first_word = ((first - offset) / 4) as u32;
        let kernel = device
            .owned_kernel(OwnedModule::ShadowCopyBack(words.texel))
            .ok()?;
        let set = OwnedSet {
            images: vec![words.view],
            storage_images: Vec::new(),
            buffers: vec![(buffer, offset, range)],
        };
        let binds = OwnedBinds::new(device, &kernel, &[set]).ok()?;
        Some(CopyBack {
            kernel,
            binds,
            shape: ShadowCopyBackModule::shape([x0, y0], width, row_words, first_word),
            groups,
        })
    }

    /// Record the dispatch into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the shadow, its view and the buffer
    /// are live until it completes, and the shadow is in the `GENERAL` layout.
    pub(super) unsafe fn record(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract; the set binds the view and the bytes the region's
        // texels lie in, and the shape and workgroups stay inside both (checked in `new`).
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            self.binds.bind(device, command_buffer, &self.kernel, 0);
            device.cmd_push_constants(
                command_buffer,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.shape,
            );
            device.cmd_dispatch(command_buffer, self.groups[0], self.groups[1], 1);
        }
    }
}
