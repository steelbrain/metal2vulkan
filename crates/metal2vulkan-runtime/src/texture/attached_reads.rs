//! A draw's slot that samples a texture its own render pass instance renders: the slot reads
//! the texture's shadow (`texture/usage_shadow.rs`), a copy, never the attachment.
//!
//! - **Vulkan's rule.** The Render Pass chapter ("Non-Overlapping Access Between Resources"):
//!   "it is invalid to access a non-attachment subresource that is simultaneously being used as
//!   an attachment where either access performs a write operation", unless a feedback loop is
//!   enabled for it. A draw's `VUID-vkCmdDraw-None-10795`: "If a color attachment is written by
//!   any prior command in this subpass or by the load, store, or resolve operations for this
//!   subpass, and feedback loop is not enabled for VK_IMAGE_ASPECT_COLOR_BIT on that
//!   attachment, it must not be accessed in any way other than as an attachment by this
//!   command"; `VUID-vkCmdDraw-None-12338` likewise where the draw writes it. A store writes, so
//!   every such read in an instance that stores the attachment is one, with a barrier before it
//!   or not. The validation layer reports none of them here: the draws record into secondary
//!   command buffers whose inheritance names no framebuffer (`render.rs`'s `begin_draws`), so
//!   it cannot tie a descriptor to an attachment.
//! - **Metal's answer** (probe `selfsample`, an M3 Max under macOS 27.0.1: a private texture
//!   and a texture over an `MTLBuffer`, 256x256 and 1440x900, a draw that samples its own
//!   color attachment's texture after 1 or 50 draws, no barrier). Metal's API validation warns
//!   ("uses Texture binding at index 0 for t\[0\] which is also a color attachment at index 0.
//!   This may be a conflict as the color attachment's storeAction == MTLStoreActionStore and
//!   writeMask != MTLColorWriteMaskNone", and "... which is currently being written to in this
//!   render pass. This may be a conflict.") and stops nothing. The draw read, at its own pixel,
//!   the texel from before the pass in every run (80 of 80; probe `rtbarrier` found 70 of 72,
//!   the other two at 500 draws over 1920x1200 and 3840x2400 reading the drawn texels); at
//!   other pixels the drawn texel or the one from before the pass, tile by tile and run by run.
//!   Metal defines no answer there: the copy is for Vulkan's validity, and taken before the
//!   instance, it holds the texels from before the pass in the first, Metal's answer at the
//!   draw's own pixel and one of its answers elsewhere.
//! - **The x86 Macs' answer** (probe `ownread`, macOS 26.7.1: a pass that loads or clears a
//!   `BGRA8Unorm` texture, private, over an IOSurface or over a private `MTLBuffer`, 512x512 to
//!   3840x2400; a draw writes a rectangle, 0 or 50 draws of another color follow, then a draw
//!   reads the texture at its own pixels over the rectangle, 3 runs each). An AMD Radeon Pro
//!   5300M read the earlier draw's texels at 98.6% of the pixels with no draws between and at
//!   every one after 50 draws or a memory barrier over the render targets; the rest read the
//!   load's or the clear's texel (a private 512x512 texture's every pixel, run after run) or, at
//!   9,600 pixels of a private 1600x1000 one from the rectangle's first, neither. An Intel UHD Graphics 630 read the earlier
//!   draw's texels at all but 0.01% of the pixels, at every one after a barrier. An M3 Max read
//!   the texels from before the pass at every pixel, in a clear pass too. The Macs disagree, so
//!   the runtime takes the cheaper answer where it can, the x86 Macs': what the load and every
//!   earlier draw of the instance wrote (`docs/API.md`'s accepted differences).
//! - **With a feedback loop too.** On a device with `VK_EXT_attachment_feedback_loop_layout`
//!   the read in place is valid usage, and a barrier inside the subpass before the reading draw
//!   orders the earlier draws' writes and the load's before it at each pixel ("A
//!   vkCmdPipelineBarrier inside a subpass can synchronize prior attachment writes in the
//!   subpass with subsequent non-attachment reads if feedback loop is enabled for the
//!   attachment", Common Render Pass Data Races): one pixel's (a barrier inside a subpass is by
//!   region, `VUID-vkCmdPipelineBarrier-dependencyFlags-07891`), the fragment's own. A draw that
//!   reads a color at its own pixel alone and writes no channel of it reads it in place so
//!   (`render/snapshot.rs`, in place), with a barrier or without: the x86 Macs' answer, and after
//!   a barrier every Mac2 GPU's. A draw that writes the color it reads would race its own writes
//!   there: its other primitives' at the pixel it reads, which nothing orders before a sampled
//!   read (the Rasterization Order section orders fragment operations and color writes alone), a
//!   data race, which Vulkan forbids ("Applications must not cause a data race", the Shader
//!   Memory Access Ordering section). Such a draw reads in place through a snapshot where the
//!   device can and the snapshot costs no more than the copy (`render/snapshot.rs`; below), with
//!   a barrier or without, seeing what the draws before it wrote; else it reads the copy, before a
//!   barrier the texels from before the pass, the M3 Max's answer: a private texture's reader
//!   through small scissors reads that, a surface's reads the x86 Macs'.
//! - **After a memory barrier.** A barrier over the render targets orders the draws after it
//!   behind those before it, which a Mac2 GPU runs (Apple GPUs do not support that scope, the
//!   Metal Feature Set Tables: `fence.rs`). Inside an instance it orders one pixel alone
//!   (`render_target::between_draws`), so a draw after it that samples the texture records into
//!   a new instance (`render.rs`, `WorkCounts::attachment_read_splits`), before which the copy
//!   is taken: it holds every texel the draws before the barrier wrote, and those of the draws
//!   between the barrier and it, which Metal does not order before it. Not where the instance
//!   drew none of the texels the draw reads (`render.rs`'s `reads_what_the_instance_drew`,
//!   `WorkCounts::attachment_read_splits_spared`): in an instance of draws alone, each color the
//!   texture presents one the instance loads and changes inside its draws' scissors alone
//!   (`render/shadows.rs`'s `drawn_alone`), no other write of the texture's truth among its
//!   draws, and no rectangle its draws drew meeting the texels the draw reads (those around its
//!   scissors where it reads the texture at its fragments' own pixels alone, below; else every
//!   one). The copy taken before the instance then holds each of them as the split's would, and
//!   the draw stays in the instance and reads it.
//! - **After a memory barrier before the pass's first draw.** Such a barrier orders the pass's
//!   load actions before the draws after it (`fence.rs`): a draw after it that samples the
//!   texture of a color the pass clears ([`Texture::cleared_in`]) reads the clear on a Mac2 GPU,
//!   where the copy taken ahead of the pass's load actions holds the texels from before the pass.
//!   Before any draw of the pass has recorded, the copy holds nothing but the clear there, so
//!   where it can be cleared to it ([`Texture::cleared_copy`]: the copy is an image of its own,
//!   the texture a single-sampled 2D one of one level and slice that one color renders whole, in
//!   place, in the image's format, inside the render area), the copy in before the pass is a clear
//!   of it to the pass's clear value (`vkCmdClearColorImage`, which Vulkan converts as the load
//!   operation, the pass's `device-color-conversion` note the same; on a driver whose clears are
//!   Metal's, `DriverFact::ColorClearIsMetals`, the clear command's texel is the load
//!   operation's, Metal's, so a clear whose note the load withholds owes none here either,
//!   `ColorTarget::withheld_clear`), and the draw stays in the pass's first
//!   instance (`WorkCounts::attachment_read_clears`): no instance of the clear alone, no store and
//!   load of the attachment between two, and no copy. Later draws of that instance read the same
//!   copy, as they would read the copy after the clear's instance. Otherwise, and once a draw has
//!   recorded (it may have written the attachment, which the copy must then hold), it records
//!   into an instance after the pass's first, which runs the pass's load actions and draws
//!   nothing where no draw came before (`render.rs`, `WorkCounts::attachment_read_splits`),
//!   before which the copy is taken, or reads through snapshots inside its instance (below). A
//!   draw that samples a color the pass loads, or none, records as with no barrier: the copy
//!   ahead of the load holds what it loads.
//! - **The copy** is the texture's shadow, an image of its own, whose truth is the texture's
//!   image or its buffer's bytes. The pass's shadow plan (`render/shadows.rs`) copies the truth
//!   in before each instance whose draws read the shadow, after the instances before it wrote
//!   the attachment; before the first, ahead of the pass's load actions. Each such draw counts
//!   (`WorkCounts::attachment_reads_through_copies`, the shadow served or not), and so does each
//!   copy (`WorkCounts::shadow_copies`). It is exact and announces nothing.
//! - **Two faces instead of the first copy** (`texture/faces.rs`). A `Private`, tracked texture
//!   of the runtime's own memory, of one level, slice and sample and no view, takes a second
//!   image of its plan at its first attached read, as the copy's image would have been. A later
//!   pass that renders the whole texture from one color loaded `Clear` or `DontCare` turns it at
//!   its open, where nothing else may name the image it presented (its command buffer the
//!   device's only uncommitted one, no batch held, no heap set read meanwhile:
//!   `render/face_swap.rs`): the pass renders the other image, and its first instance's reads
//!   take the one presented until then, which holds the texels from before the pass as the copy
//!   ahead of the load actions would, with no copy (`WorkCounts::attached_read_swaps`,
//!   `attached_read_swap_reads`). A read after a barrier copies in what the draws wrote, as
//!   above. Every copy in no turn took off is counted by why (`attached_read_unswapped_*`).
//! - **Levels apart from the attached ones.** A slot every read of which reaches only levels of
//!   the texture before the lowest one an attachment of the instance renders, or only levels past
//!   the highest, through the samplers the draw binds (`StageModule::level_reads`), reads the
//!   texture's own image in place of the copy, through a view of the levels up to the lowest
//!   attached one, or up to the highest it reads, and names the levels it reads in its uses
//!   (`render/attached_levels.rs`, `WorkCounts::attached_read_slots_narrowed`,
//!   `attached_read_slots_past`): the copy would hold the same texels there, as nothing the
//!   instance does writes them. Its view's first level is the texture's, so an explicit level of
//!   detail selects the same level through either.
//! - **Only what the instances drew.** A copy after the pass's first instance, of a buffer's
//!   bytes or of an image's one level and slice, where the instances since the last copy changed
//!   them only by their draws into a color attachment the pass loads (the encoder's load,
//!   single-sampled, resolved nowhere, rendered in place, the whole of a one-level, one-slice
//!   texture), moves only the texels inside the rectangle bounding those instances' draws'
//!   scissors: a draw changes no texel outside its scissors, and the load and store write back
//!   what they read (`render/shadows.rs`). Every other texel of the shadow holds the truth's from
//!   the last copy (`shadow.rs`, `WorkCounts::bounded_shadow_copies`). Any other write of the
//!   truth (a clear, a `DontCare` load, a resolve, a stand-in, a storage write, a tile dispatch,
//!   an execution, a rasterization-rate map), an image's several levels or slices as the truth,
//!   and the first copy of each encoder copy every texel, but where the copy stands across
//!   command buffers (below).
//! - **Across command buffers.** Where only the runtime's commands write the copy's truth (a
//!   tracked `Private` texture's own image, or the bytes of a tracked `Private` buffer the device
//!   made, a surface plane's among them), the copy may stand across command buffers
//!   (`shadow_standing.rs`): a command buffer's end that left it holding the truth leaves it to the
//!   next command buffer's first copy, which moves nothing where nothing wrote the truth or the
//!   copy since. A pass's draws over a color attachment it loads (as above, its first instance's
//!   too) change the truth inside their scissors alone, so the copy stands stale inside the
//!   rectangle bounding them alone, and that next copy moves those texels alone
//!   (`WorkCounts::shadow_texels_unskipped_stale_within`).
//! - **Only what a draw reads at its own pixels.** A draw every binding of whose texture is a
//!   fragment slot its function reads at the fragment's own pixel alone
//!   (`StageModule::own_pixel_textures`: an `OpImageFetch` at level 0 of Metal's saturating
//!   `uint2(position.xy)`, `spirv::requirements::own_pixel`), over a texture whose copy is cut to
//!   a rectangle (`ShadowCopies::cuts`), in a pass of no rasterization-rate map or views, reads
//!   the texels inside its scissors alone, and its helper invocations a pixel beside them (the
//!   quad scope: "neighboring framebuffer locations"). Its copy in moves only the texels inside
//!   the rectangle bounding its Vulkan scissors, a pixel wider each way, clamped as the read's
//!   coordinate is (`ShadowUse::loads`, `WorkCounts::own_pixel_reads`); a later copy, only those
//!   of them the instances since changed. A command buffer's first copy in of a shadow over a
//!   `Private` buffer's bytes is cut so too: its pass renders the shadow's truth apart from it, so
//!   no standing across command buffers (`shadow_standing.rs`) takes every texel. It is exact with a barrier before it or not: the copy
//!   before the pass holds, at the draw's own pixels, the texels from before the pass, Metal's
//!   answer there (probe `selfsample`), and one after a split what the draws before the barrier
//!   wrote; the rest of the copy holds what no read of the draw reaches. Every texel of a copy
//!   that a draw of the instance reads otherwise is copied as above.
//! - **Through a snapshot, after a barrier.** On a device that enabled
//!   `attachmentFeedbackLoopLayout`, `extendedDynamicState3ColorWriteMask` and
//!   `fragmentShaderPixelInterlock`, a draw after a barrier inside its instance whose function
//!   reads such a texture at the fragment's own pixel alone stays in its instance instead of the
//!   split, where every check of `render/snapshot.rs` holds: drawn twice, it first copies each
//!   fragment's texel into a snapshot of the texture inside the pixel interlock, and then reads
//!   the snapshot (`WorkCounts::reading_draws_snapshotted`). The texture's image takes feedback
//!   loop usage where that costs it nothing ([`feedback_candidate`]): an image of its own, the
//!   linear image that aliases a buffer's bytes (`Device::storage_over_buffer` tries the usage
//!   first, its memory being the buffer's, and counts the images made with it,
//!   `WorkCounts::feedback_loop_alias_images`), or the shadow image a texture over a buffer's
//!   bytes renders through (`shadow.rs`, `WorkCounts::feedback_loop_shadow_images`), as a surface
//!   over host pages that renders is (`in_place.rs`). The instance
//!   holds its attachment in the feedback loop layout, the bytes too while it renders them, when a host
//!   access of them races the pass in Metal as well. Not the attachment itself: "In that case the overlap
//!   is defined to occur at a per-pixel granularity" (the Render Pass chapter), but at its own
//!   pixel, a draw whose primitives overlap there would read what an earlier primitive of the
//!   same draw wrote unordered, as the Rasterization Order section orders only "Fragment
//!   operations" and "Blending, logic operations, and color writes" from primitive to primitive,
//!   never a shader's read of a sampled image: a data race, where Metal's read after a barrier
//!   gives what the draws before it wrote. The snapshot is taken before the draw's first color
//!   write. A read at another pixel than its own takes the split: a barrier inside a subpass is
//!   by-region (`VUID-vkCmdPipelineBarrier-dependencyFlags-07891`), so it would see no write of
//!   another pixel.
//! - **Not served** (`docs/FAST-PATHS.md`, known gaps): a read through the heap set (a handle
//!   or an argument buffer's texture, which no slot names), a storage binding, a depth or
//!   stencil attachment, a texture buffer or a buffer binding over the same buffer's bytes (a
//!   texture over them whose image aliases them has the buffer's identity, so it presents the
//!   attachment, `attached_in`, and reads its copy), and a texture whose shadow is not served (`texture/usage_shadow.rs`): those read the attachment as before, and
//!   in an instance that holds it in the feedback loop layout (`render/snapshot.rs`) through a
//!   descriptor that names `VK_IMAGE_LAYOUT_GENERAL`. A draw whose uses share memory with an
//!   attachment of its instance (`render_target::shares_attachment`, by the tracker's keys and
//!   ranges: a copy or a snapshot is another resource) announces it,
//!   `Degradation::AttachmentReadInPlace`.

use super::{Device, Owner, Storage, Texture};
use crate::binding_view::{binding_view, VkView};
use crate::command_buffer::CommandBuffer;
use crate::error::Error;
use crate::image_plan::ImagePlan;
use crate::render_pass::{PassLayout, MAX_COLOR_ATTACHMENTS};
use crate::render_values::LoadAction;
use crate::resource::ResourceKey;
use crate::texture::view_error;
use crate::texture_descriptor::TextureDescriptor;
use crate::tracker::{Access, ImagePlace, Use, UseRange};
use crate::values::ResourceOptions;
use crate::view_plan::ViewRecord;
use ash::vk;
use metal2vulkan::bind::ImageViewDesc;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::SnapshotForm;
use metal2vulkan::vk::{
    ComponentMapping, Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
    ImageViewType, SampleCountFlags,
};
use std::ops::Range;

/// The color attachments of a render pass instance as the textures a draw samples meet them, each
/// read once ([`AttachedColors::of`]), by color index: a draw asks of each texture it samples
/// whether one renders it ([`Texture::attached_among`]), and so reads each attachment's image once
/// a draw, not once a texture it samples.
pub(crate) struct AttachedColors([Option<AttachedColor>; MAX_COLOR_ATTACHMENTS]);

/// A color attachment as [`AttachedColors`] holds it.
#[derive(Clone)]
struct AttachedColor {
    /// The key of the image its texture renders, and where the image lies in that key's memory:
    /// the images of a heap share its key, and those over a buffer's bytes the buffer's, each at
    /// its own bytes (`tracker::ImagePlace`).
    image: ResourceKey,
    place: ImagePlace,
    /// The level it renders.
    level: u32,
    /// The layers it renders.
    layers: Range<u32>,
    /// Its load action clears it.
    clears: bool,
}

impl AttachedColors {
    /// The color attachments of `layout`: those whose texture has an image (not a texture
    /// buffer).
    pub(crate) fn of(layout: &PassLayout) -> AttachedColors {
        let mut colors = layout.colors.iter();
        AttachedColors(std::array::from_fn(|_| {
            let color = colors.next()?.as_ref()?;
            let target = &color.target;
            let image = target.texture.image().ok()?;
            Some(AttachedColor {
                image: image.identity.key,
                place: image.place(),
                level: target.level,
                layers: target.layer..target.layer + target.layers,
                clears: color.load == LoadAction::Clear,
            })
        }))
    }

    /// Whether the pass has a color attachment with an image: else no texture is attached in it.
    pub(crate) fn any(&self) -> bool {
        self.0.iter().any(Option::is_some)
    }
}

/// The levels of a texture that the color attachments of a render pass instance render
/// ([`Texture::attached_span`]), each counted from the texture's own first level.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttachedSpan {
    /// The lowest and the highest attached level.
    pub(crate) lowest: u32,
    pub(crate) highest: u32,
    /// The colors that render them, by index: bit `i` for color `i`.
    pub(crate) colors: u32,
    /// The texture's level count.
    pub(crate) level_count: u32,
}

/// The pixel format of a snapshot of format `format` (`SnapshotForm::format`, an unsigned
/// integer format of four channels), as the runtime's memory counts it; `None` for another.
fn snapshot_pixel_format(format: Format) -> Option<PixelFormat> {
    match format {
        Format::R8G8B8A8_UINT => Some(PixelFormat::RGBA8Uint),
        Format::R16G16B16A16_UINT => Some(PixelFormat::RGBA16Uint),
        Format::R32G32B32A32_UINT => Some(PixelFormat::RGBA32Uint),
        _ => None,
    }
}

/// `plan` with `VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT`, where a draw may read the
/// texture in place or through a snapshot while its pass renders it (the module docs): a
/// single-sampled 2D color attachment the shaders sample, not transient, on a device that enabled
/// `attachmentFeedbackLoopLayout`. `None` where the plan has it already or cannot take it. The
/// usage is added only where it costs the image nothing (`Device::with_free_usage`).
pub(crate) fn feedback_candidate(plan: &ImagePlan, device: &Device) -> Option<ImagePlan> {
    let usage = plan.usage;
    let enabled = (device.enabled().features).contains(&VkFeature::AttachmentFeedbackLoopLayout);
    let candidate = enabled
        && usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
        && usage.contains(ImageUsageFlags::SAMPLED)
        && !usage.contains(ImageUsageFlags::TRANSIENT_ATTACHMENT)
        && !usage.contains(ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT)
        && plan.samples.as_raw() == 1
        && plan.image_type == ImageType::TYPE_2D;
    candidate.then(|| ImagePlan {
        usage: usage | ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT,
        ..plan.clone()
    })
}

impl Texture {
    /// Whether the texture presents a level and layer that a color attachment of `layout`'s
    /// instance renders: the same image, and an overlapping level and layer (every layer of a 3D
    /// image, which has one, its depth planes the attachment's layers).
    pub(crate) fn attached_in(&self, layout: &PassLayout) -> bool {
        self.attached_among(&AttachedColors::of(layout))
    }

    /// Whether the texture presents a level and layer that a color attachment of `layout`'s
    /// instance whose load action clears it renders, as [`attached_in`](Self::attached_in) tells:
    /// what a draw after a memory barrier before the pass's first draw reads of it is the clear
    /// (the module docs).
    pub(crate) fn cleared_in(&self, layout: &PassLayout) -> bool {
        self.cleared_among(&AttachedColors::of(layout))
    }

    /// [`attached_in`](Self::attached_in) of the pass whose color attachments `colors` read.
    pub(crate) fn attached_among(&self, colors: &AttachedColors) -> bool {
        self.attached_levels(colors, |_| true).next().is_some()
    }

    /// The indices of the color attachments among `colors` that render the texture at a level
    /// and layer it presents, as [`attached_among`](Self::attached_among) tells.
    pub(crate) fn attached_colors<'s>(
        &'s self,
        colors: &'s AttachedColors,
    ) -> impl Iterator<Item = usize> + 's {
        self.attached_levels(colors, |_| true)
            .map(|(index, _)| index)
    }

    /// [`cleared_in`](Self::cleared_in) of the pass whose color attachments `colors` read.
    pub(crate) fn cleared_among(&self, colors: &AttachedColors) -> bool {
        (self.attached_levels(colors, |color| color.clears))
            .next()
            .is_some()
    }

    /// The copy a sampled slot binding of the texture by the draw `call` reads, and the clear it
    /// reads there, after a memory barrier before its pass's first draw and before any draw of the
    /// pass, where that copy may be cleared to it instead of the pass splitting (the module docs):
    /// the texture's shadow, made here where the draw's binding has not made it yet
    /// ([`attached_shadow_copies`](Self::attached_shadow_copies)), is an image of its own that a
    /// clear in clears (`ShadowCopies::cleared_image`), and the texture a
    /// single-sampled 2D one of one level and slice, presented whole in its image's format, that
    /// exactly one color attachment of `layout` renders in place, all of it inside the render
    /// area, in that format, loading it `Clear` with a value the device's own clear converts as
    /// its load operation does (no extended-range route; a clear whose note the load withholds,
    /// `ColorTarget::withheld_clear`, on a driver whose clear command stores the load operation's
    /// texel, `DriverFact::ColorClearIsMetals`, owes none) and no stand-in or sample planes.
    /// `None` elsewhere: the pass splits as before.
    pub(crate) fn cleared_copy(
        &self,
        layout: &PassLayout,
        call: &'static str,
    ) -> Option<(ResourceKey, vk::ClearColorValue)> {
        let image = self.image().ok()?;
        let plan = image.plan();
        let record = &self.0.record;
        let whole_texture = record.texture_type == TextureType::Type2D
            && plan.image_type == ImageType::TYPE_2D
            && plan.samples.as_raw() == 1
            && plan.levels == 1
            && plan.layers == 1
            && (record.base_level, record.level_count) == (0, 1)
            && (record.base_layer, record.layer_count) == (0, 1)
            && record.read_as.is_none()
            && record.pixel_format == plan.pixel_format;
        if !whole_texture {
            return None;
        }
        let mut rendering = (layout.colors.iter().flatten()).filter(|color| {
            (color.target.texture.image()).is_ok_and(|other| {
                other.identity.key == image.identity.key && other.place().meets(image.place())
            })
        });
        let color = rendering.next()?;
        let target = &color.target;
        let whole = (target.level, target.layer, target.layers) == (0, 0, 1)
            && target.extent == layout.extent
            && target.extent == [plan.extent[0], plan.extent[1]]
            && target.format == plan.format
            && target.texture.pixel_format() == plan.pixel_format;
        let plain = color.load == LoadAction::Clear
            && !color.memoryless
            && color.stand_in.is_none()
            && color.planes.is_none()
            && color.straight_clear.is_none();
        if rendering.next().is_some() || !whole || !plain {
            return None;
        }
        // Last, as the shadow is made where no binding made it yet: the draw's own binding makes
        // it in any case, as the texture presents an attachment of its instance.
        let copies = self.attached_shadow_copies(call)?;
        copies.cleared_image()?;
        Some((copies.shadow_key, color.clear))
    }

    /// The levels of the texture, counted from its own first level, that the color attachments of
    /// `layout` render at a layer it presents, as [`attached_in`](Self::attached_in) tells; `None`
    /// where none does.
    pub(crate) fn attached_span(&self, layout: &PassLayout) -> Option<AttachedSpan> {
        let colors = AttachedColors::of(layout);
        let mut attached = self.attached_levels(&colors, |_| true);
        let (color, level) = attached.next()?;
        let first = AttachedSpan {
            lowest: level,
            highest: level,
            colors: 1 << color,
            level_count: self.0.record.level_count,
        };
        Some(attached.fold(first, |span, (color, level)| AttachedSpan {
            lowest: span.lowest.min(level),
            highest: span.highest.max(level),
            colors: span.colors | 1 << color,
            ..span
        }))
    }

    /// Whether a sampled slot binding of the texture can read its own image's levels before an
    /// attached one in place of its copy (`render/attached_levels.rs`): its storage is an image of
    /// its own, not a sparse one, with no decoded twin, whose image has the sampled usage.
    pub(crate) fn reads_levels_in_place(&self) -> bool {
        let Storage::Image(image) = self.storage() else {
            return false;
        };
        !matches!(image.owner, Owner::Sparse(_))
            && !self.has_decoded_twin()
            && image.plan().usage.contains(ImageUsageFlags::SAMPLED)
    }

    /// A slot binding's use of the texture's levels `levels`, counted from its own first level,
    /// at every layer it presents, reading by `read` and writing by `write`: its own image's, as
    /// [`reads_levels_in_place`](Self::reads_levels_in_place) holds.
    pub(crate) fn levels_use(
        &self,
        levels: Range<u32>,
        (read, write): (Access, Access),
    ) -> Use<ResourceKey> {
        let mut used = self.bound_use(read, write);
        if let UseRange::Image { levels: range, .. } = &mut used.range {
            let first = range.start;
            *range = (first + levels.start).min(range.end)..(first + levels.end).min(range.end);
        }
        used
    }

    /// The view of a sampled slot binding `desc` of `call` over the texture's own image's levels
    /// `0..levels`, counted from its own first level, as
    /// [`reads_levels_in_place`](Self::reads_levels_in_place) holds: the view of every level but
    /// its level count, so an explicit level of detail selects the same level through either.
    pub(crate) fn levels_view(
        &self,
        call: &'static str,
        desc: &ImageViewDesc,
        levels: u32,
    ) -> Result<vk::ImageView, Error> {
        let image = self.image()?;
        let record = ViewRecord {
            level_count: levels,
            ..self.0.record.bound()
        };
        let view = binding_view(image.plan(), &record, desc, false)
            .map_err(|refusal| view_error(call, refusal))?;
        image.view(&view)
    }

    /// The color attachments of `colors` that `keep` keeps that render the texture at a layer it
    /// presents, each by its index and the level it renders, counted from the texture's own first
    /// level: of its image, at one of its levels, and at one of its layers (any layer of a 3D
    /// image's). An image placed in a heap apart from the attachment's, or over the attachment's
    /// buffer apart from it, at bytes of its own, presents none of it, though the heap's images,
    /// and the buffer's, share one key (`tracker::ImagePlace`).
    fn attached_levels<'s>(
        &'s self,
        colors: &'s AttachedColors,
        keep: impl Fn(&AttachedColor) -> bool + 's,
    ) -> impl Iterator<Item = (usize, u32)> + 's {
        let image = self.image().ok();
        let record = &self.0.record;
        let levels = record.base_level..record.base_level + record.level_count;
        let layers = record.base_layer..record.base_layer + record.layer_count;
        let volume = image.is_some_and(|image| image.plan().image_type == ImageType::TYPE_3D);
        let key = image.map(|image| image.identity.key);
        let place = image.map_or(ImagePlace::WHOLE, |image| image.place());
        let colors = key.map(|_| colors.0.iter().enumerate());
        (colors.into_iter().flatten())
            .filter_map(|(index, color)| color.as_ref().map(|color| (index, color)))
            .filter(move |(_, c)| keep(c))
            .filter_map(move |(index, color)| {
                let rendered = &color.layers;
                let presented =
                    volume || (rendered.start < layers.end && layers.start < rendered.end);
                let same = Some(color.image) == key && place.meets(color.place);
                (same && levels.contains(&color.level) && presented)
                    .then(|| (index, color.level - record.base_level))
            })
    }
}

impl Texture {
    /// The index of the color attachment of `layout` a fragment slot binding of the texture
    /// reads through a snapshot (the module docs), where it can: the texture is its own image, a
    /// linear image aliasing a buffer's bytes while they lie in their pages, or the shadow image
    /// the encoders copy a buffer's bytes into and back (`shadow.rs`), which its passes render (not
    /// a texture buffer or a view read through another), a single-sampled 2D texture whose image
    /// has feedback loop and sampled usage
    /// ([`feedback_candidate`]) and whose sampled binding takes no shadow; and the attachment
    /// renders that image in place (no stand-in or sample planes), one layer of it, at the
    /// texture's own first level and slice, so the binding's level 0 at a fragment's
    /// coordinate is the attachment's texel at that pixel.
    pub(crate) fn snapshot_color(&self, layout: &PassLayout) -> Option<usize> {
        let image = match self.storage() {
            Storage::Image(image) | Storage::Shadow { image, .. } => image,
            Storage::Alias { image, .. } if self.image_over_buffer() => image,
            _ => return None,
        };
        let plan = image.plan();
        let usage = ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT | ImageUsageFlags::SAMPLED;
        let record = &self.0.record;
        if !plan.usage.contains(usage)
            || plan.image_type != ImageType::TYPE_2D
            || plan.samples.as_raw() != 1
            || record.texture_type != TextureType::Type2D
            || record.read_as.is_some()
            || self.needs_usage_shadow(false)
        {
            return None;
        }
        layout.colors.iter().enumerate().find_map(|(index, color)| {
            let color = color.as_ref()?;
            let target = &color.target;
            let same = (target.texture.image()).is_ok_and(|other| {
                other.identity.key == image.identity.key && other.place().meets(image.place())
            });
            let own = same
                && color.stand_in.is_none()
                && color.planes.is_none()
                && target.samples == 1
                && target.layers == 1
                && target.level == record.base_level
                && target.layer == record.base_layer;
            own.then_some(index)
        })
    }

    /// The texture's snapshot in `form` (the module docs): a single-sampled 2D storage image of
    /// the form's format (`SnapshotForm::format`: 4, 8 or 16 bytes a texel) and of the image's
    /// level-0 extent, which holds every level's render area, made by `maker` at the first draw
    /// that takes it in that form and kept with the texture a view was made from, so it goes with
    /// that texture; the runtime's memory counts it as any image of its own.
    pub(crate) fn snapshot_texels(
        &self,
        maker: &CommandBuffer,
        form: SnapshotForm,
    ) -> Result<Texture, Error> {
        let mut owner = self;
        while let Some(parent) = &owner.0.parent {
            owner = parent;
        }
        let format = form.format();
        let pixel_format = snapshot_pixel_format(format)
            .ok_or_else(|| crate::error::internal("a snapshot form of no snapshot's format"))?;
        let mut held = crate::device::lock(&owner.s().snapshot);
        if let Some(made) = held.iter().find(|t| t.pixel_format() == pixel_format) {
            return Ok(made.clone());
        }
        let [width, height, _] = owner.0.record.image_extent;
        let descriptor = TextureDescriptor {
            texture_type: TextureType::Type2D,
            pixel_format,
            width: width.into(),
            height: height.into(),
            usage: TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        };
        let plan = ImagePlan {
            pixel_format,
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::TYPE_2D,
            format,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: SampleCountFlags::from_raw(1),
            usage: ImageUsageFlags::STORAGE,
            view_formats: Vec::new(),
            mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
            unavailable: TextureUsage::SHADER_ATOMIC
                | TextureUsage::RENDER_TARGET
                | TextureUsage::PIXEL_FORMAT_VIEW,
        };
        let made = Device::new_planned_texture(maker, &descriptor, plan)?;
        held.push(made.clone());
        Ok(made)
    }

    /// The storage view of a snapshot of `form` ([`snapshot_texels`](Self::snapshot_texels)),
    /// which a draw's `WriteSource::SnapshotTexels` binds in `GENERAL`.
    pub(crate) fn snapshot_view(&self, form: SnapshotForm) -> Result<vk::ImageView, Error> {
        self.image_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format: form.format(),
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        })
    }

    /// A draw's use of a snapshot: written and read by its fragment shader as a storage image.
    pub(crate) fn snapshot_use(&self) -> Use<crate::resource::ResourceKey> {
        let stage = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        self.bound_use(
            Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ),
            Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE),
        )
    }
}
