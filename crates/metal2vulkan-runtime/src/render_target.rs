//! The Vulkan render passes and framebuffer of a render encoder: the compatible pass its draws
//! record against, and the pass and framebuffer `end_encoding` begins.
//!
//! - **The draws' subpass.** Its color references are the pass's color attachments at their
//!   Metal indices, `VK_ATTACHMENT_UNUSED` for an index without a texture, up to the last index
//!   with one; then the depth/stencil attachment. A draw's pipeline has the same formats at the
//!   same indices (checked when the encoder sets it), so its color blend state has one element
//!   per reference. Resolve attachments follow (`resolve.rs`): one reference per color index,
//!   unused where it does not resolve or resolves by the exact resolve after the pass
//!   (`color_resolve.rs`, which reads the samples the last instance stores, `STORE` whatever
//!   Metal's store action), and the depth/stencil resolve with its modes; the passes
//!   are made with `vkCreateRenderPass2`, which states the latter. Where the device's framebuffer
//!   fetch reads input attachments (`framebuffer_fetch.rs`), the subpass also reads each color
//!   attachment as the input attachment of its Metal index, with rasterization-order color
//!   access, and the attachments' views are input attachments too; an attachment whose image
//!   takes no input attachment usage (a caller's, `texture/wrapped.rs`) is not read so
//!   (`VK_ATTACHMENT_UNUSED`) until a stand-in that has it takes its place
//!   (`attachment_stand_in.rs`). A color attachment the draws
//!   read and write as a storage image in the pixel interlock is left out of the subpass
//!   (`VK_ATTACHMENT_UNUSED`) and keeps its attachment description, which loads and stores
//!   nothing (`DONT_CARE`, what Vulkan does for an attachment no subpass uses in any case), so
//!   every instance of a split pass has the same attachments in the same order; its framebuffer
//!   view is of a placeholder image the device keeps, never of the image the draws write
//!   (`storage_placeholder.rs`).
//! - **The sample planes' anchor.** A multisampled subpass that would so be left with no
//!   attachment (every color attachment it has read as sample planes, `sample_planes.rs`, and no
//!   depth/stencil attachment) keeps the first of them ([`PassShape::anchor`]): a subpass without
//!   attachments takes its pipelines' rasterization samples, but MoltenVK rasterizes one sample
//!   there unless a pipeline was made against that very `VkRenderPass` (its subpass's default
//!   sample count, set only at pipeline creation), and the runtime makes its pipelines against
//!   another, compatible one (at 4x the fetching draws changed sample 0 alone). Its pipelines write
//!   nothing there (they have no color attachment, so every element of their blend state writes
//!   nothing), and in each instance it is `DONT_CARE` both ways: the planes hold its samples there,
//!   split from it before and merged back after (`sample_planes.rs`). A subpass with an attachment
//!   of the pass's samples rasterizes them on every device, and needs no
//!   `framebufferNoAttachmentsSampleCounts`. Where the draws read and write the attachments' own
//!   multisampled images instead (a device with multisampled storage images), the subpass keeps
//!   none, since its draws write what it leaves out: its pipelines' samples, which link checks
//!   against `framebufferNoAttachmentsSampleCounts` where no attachment is left.
//! - **Passes without attachments.** A pass with no attachment at all records, makes its
//!   pipelines against, and begins the pass its device keeps for its sample count
//!   (`attachmentless.rs`), so that MoltenVK rasterizes the pipelines' samples there.
//! - **Instances.** A pass whose draws change an attachment's role (`render.rs`) is split into
//!   render pass instances of one framebuffer's attachments: Metal's load actions run in the
//!   first, its store actions and resolves in the last, and every other operation loads and
//!   stores. An instance before the last resolves nothing: its resolve references are unused,
//!   which render pass compatibility ignores in a pass of one subpass, as every pass of an
//!   instance is. An attachment the draws' subpass leaves out, or keeps as the sample
//!   planes' anchor, resolves nothing in it (its resolve reference is unused too): a pass whose
//!   last draws leave out an attachment that resolves ends with an instance of no draws that
//!   includes it (`render.rs`).
//! - **Views**. An instance of an amplified draw's views has their view mask on every subpass, with
//!   a view-local dependency of the draws' subpass on itself; its framebuffer has one layer, and
//!   each attachment's view starts at the layer of the first view and holds them all. An instance
//!   of a draw a view mapping moves to one layer views that layer alone. Its load and store
//!   operations touch those layers only, so the encoder keeps a pass's clears in an instance over
//!   every layer (`render.rs`), and its resolves in a last instance over every layer. The view mask
//!   is part of the shape: passes of different masks are not compatible.
//! - **Between draws.** The draws' subpass depends on itself, by region, from the fragment
//!   stages' writes (the fragment shader's, and the attachments') to their reads and writes
//!   (`DRAWS_SELF_DEPENDENCY`): the dependency a barrier between two draws of one instance
//!   needs. Storage the pixel interlock orders (the fragment imageblock's, `fragment_imageblock.rs`,
//!   and an interlocked color's storage image, `framebuffer_fetch.rs`) takes the fragment
//!   shader's scopes alone (`STORAGE_SELF_DEPENDENCY`, `between_storage_draws`), and a memory
//!   barrier over the render targets all of them (`between_draws`, `fence.rs`). It depends on
//!   itself too from the draws' reads of their indirect arguments to every graphics stage, an
//!   execution dependency alone, not by region (`INDIRECT_SELF_DEPENDENCY`): the dependency of
//!   the barrier before a draw that writes what an earlier draw of the instance read as its
//!   arguments (`after_indirect_reads`); and from the shader writes of the stages a draw runs
//!   before its fragments to the fragment shader's reads and writes, not by region
//!   (`GEOMETRY_SELF_DEPENDENCY`): the dependency of the barrier before a draw whose fragment
//!   shader reads or writes what an earlier draw of the instance wrote in those stages
//!   (`after_geometry_writes`). Every pass of a device has all three, so every pass of a shape
//!   stays compatible with every other.
//! - **Later resolves**. A depth/stencil resolve the device makes one aspect at a time
//!   (`resolve.rs`) resolves its first aspect in the draws' subpass and its second after the pass's
//!   instances, in a render pass of its own ([`later_resolve`]) that has only the depth/stencil
//!   attachment, loaded and stored, and that resolve; the last instance stores the aspect it reads,
//!   whatever Metal's store action (one that resolves leaves the samples undefined). So every
//!   instance's pass has one subpass, whose resolve references render pass compatibility ignores,
//!   and a resolve target the encoder sets late (`render/late_resolve.rs`) changes no draw's
//!   compatible pass.
//! - **Layouts.** Every attachment is in `GENERAL` before, during and after the pass, the layout
//!   every runtime image rests in, so a pass changes no layout and its attachments need no layout
//!   tracking. Two exceptions hold an attachment in another layout for an instance's whole render
//!   pass instance, moved there by an image barrier just before it begins and back to `GENERAL`
//!   by another just after it ends ([`moved_barriers`]): a color a draw of the instance reads
//!   through a snapshot (`render/snapshot.rs`), in `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT`
//!   ([`Feedback`]), its framebuffer view of the usage that layout needs; and, on a device
//!   measured to render it faster so (`DriverFact::MultisampledDepthFasterOutOfGeneral`), a
//!   multisampled depth/stencil attachment, in `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
//!   ([`depth_layout`]). Such a depth that a render encoder holds (`texture/held_depth.rs`) rests
//!   in that layout between passes instead: it is there when the instance begins and stays there
//!   after it ends, the scheduler moving it only between encoders. Each move keeps the texels: "The
//!   contents of any portion of an image subresource that is transitioned from one layout to
//!   another are preserved" unless the old layout is `UNDEFINED` (the Synchronization chapter's
//!   Image Layout Transitions). Render pass compatibility ignores the layouts, so such an
//!   instance's draws record against the same compatible pass as any other's.
//! - **Operations.** Metal's load actions are Vulkan's: `Load`, `Clear` (with the value
//!   `clear_value.rs` converts as Metal does) and `DontCare`. `Store` stores; `DontCare` is
//!   `DONT_CARE`, a recorded near-equivalent: Apple's GPU keeps the attachment's earlier texels
//!   unless its tiler overflows, when it writes the rendered ones (measured), a choice no program
//!   sees or controls; so is `MultisampleResolve`'s, whose samples Metal leaves undefined. On
//!   MoltenVK on the Apple M4 and M3 Max it is Metal's own store where the pass's one instance
//!   hands it so ([`handed_dont_care`], `DriverFact::DontCareStoreIsMetals`). A texture over a
//!   buffer, which Metal leaves holding the buffer's bytes, renders through an image the pass
//!   copies nothing back from (its shadow, or the image it keeps: `kept_bytes.rs`): the buffer's
//!   bytes, exactly. Under `DontCareColorPolicy::Kept` (`DevicePolicy::exact_answers`) so does a
//!   texture's own image a pass stores `DontCare` and does not hand to Metal's own, through a
//!   backing its queue keeps (`kept_bytes.rs`): its earlier texels, exactly. The operations below
//!   are those of an attachment a pass renders in place. A resolve attachment is written whole
//!   over the render area (`DONT_CARE` then `STORE`), but an aspect of a depth/stencil resolve
//!   that does not resolve keeps its texels.
//!   An aspect of the attached texture's Metal format that no attachment names keeps its texels
//!   (loaded and stored); an aspect only the Vulkan format has (the depth of a `Stencil8` texture
//!   held in a combined format) is `DONT_CARE` both ways.
//! - **Where `DONT_CARE` is no answer of Metal's**. Vulkan leaves a `DONT_CARE` attachment's
//!   contents undefined, and a driver may leave a third answer, neither texel Metal keeps (NVIDIA's
//!   left a pass's clear under its draw). Metal's `DontCare` and `MultisampleResolve` are
//!   `DONT_CARE` only on a driver measured to keep one of Metal's two answers
//!   (`DriverFact::DontCareStoreIsOneOfMetals`) or to store as Metal's own
//!   (`DriverFact::DontCareStoreIsMetals`); every other device stores the attachment (`STORE`),
//!   the answer Metal gives where its tiler overflows ([`DontCare`]). A memoryless attachment,
//!   the hidden depth and stencil, and the image a texture over a buffer keeps for a pass that
//!   does not store it (`kept_bytes.rs`) keep `DONT_CARE`: no texture of the program's holds them.
//! - **The compatible pass** has the same formats and sample counts with every operation `LOAD`
//!   and `STORE`: render pass compatibility ignores the operations, so the draws' secondary
//!   command buffer and the encoder's graphics pipelines record against it while the store
//!   actions may still change.
//! - **Made and kept.** A device keeps every pass it makes (`create`, [`KeptRenderPasses`]):
//!   the compatible pass of a shape, and each pass an instance begins, by its shape, its
//!   attachments' operations and its layouts (all a pass is made of, beside the device's
//!   profile), made when an encoder first needs it and destroyed with the device, for at most
//!   [`KEPT_PASSES`] passes; one past them is made per encoder, which its command buffer holds
//!   until it completes. A render pass is never changed once made, so one serves every encoder
//!   and thread at once. The framebuffer is made per instance; its command buffer holds it.

use super::attachment_stand_in::Cause;
use super::attachmentless;
use super::binding_view::VkView;
use super::color_list::ColorList;
use super::device::lock;
use super::error::{internal, Error};
use super::exit_guard;
use super::framebuffer_fetch;
use super::image_barrier::{image_barrier, ImageDst};
use super::key_hash::KeyMap;
use super::render_pass::{
    AspectClear, AspectTarget, ColorTarget, DepthStencilTarget, PassLayout, Target,
    MAX_COLOR_ATTACHMENTS,
};
use super::render_values::{LoadAction, StoreAction};
use super::resolve::{Resolve, ResolveMode, ResolveStep};
use super::resource::ResourceKey;
use super::storage_placeholder::Placeholder;
use super::texture::TextureMemory;
use super::tracker::{Access, LegacyBarrier, Use};
use super::Device;
use crate::release;
use ash::vk;
use metal2vulkan::bind::ViewSetup;
use metal2vulkan::device::{DeviceProfile, DriverFact};
use std::sync::Arc;

/// The color attachments of a render pass instance in
/// `VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` (the module docs), by index: bit `i` for
/// color `i`. A pass of a shape is compatible with every other of it whatever its feedback colors:
/// render pass compatibility ignores "Initial and final image layout in attachment descriptions"
/// and "Image layout in attachment references".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Feedback(pub(super) u32);

impl Feedback {
    /// No color in the feedback loop layout.
    pub(super) const NONE: Feedback = Feedback(0);

    /// Whether color `index` is.
    pub(super) fn has(self, index: usize) -> bool {
        index < u32::BITS as usize && self.0 & (1 << index) != 0
    }

    /// Its layout for color `index`.
    fn layout(self, index: usize) -> vk::ImageLayout {
        match self.has(index) {
            true => vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
            false => vk::ImageLayout::GENERAL,
        }
    }
}

/// What makes two passes compatible: each color index's format (`None` without a texture), the
/// depth/stencil attachment's format, the samples of every attachment, the draws' subpass's
/// input attachments, flags, and the color attachments it leaves out, and its view mask; and
/// what makes the pass, which compatibility ignores in a pass of one subpass: the resolves (each
/// color index's resolve format, and the depth/stencil resolve's format and modes).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct PassShape {
    pub(super) colors: ColorList<Option<metal2vulkan::vk::Format>>,
    pub(super) depth_stencil: Option<metal2vulkan::vk::Format>,
    pub(super) samples: u32,
    /// As long as `colors`: the resolve attachment's format where the index resolves.
    pub(super) color_resolves: ColorList<Option<metal2vulkan::vk::Format>>,
    /// The draws' subpass's depth/stencil resolve ([`subpass_step`]): its attachment's format,
    /// and its depth and stencil resolve mode (`resolve.rs`).
    pub(super) depth_stencil_resolve: Option<(metal2vulkan::vk::Format, ResolveMode, ResolveMode)>,
    /// The draws' subpass reads each color attachment as the input attachment of its index, with
    /// rasterization-order color access where the device has it (`PassLayout::fetch_inputs`).
    pub(super) fetch_inputs: bool,
    /// The color indices, ascending, whose images take no input attachment usage (a caller's
    /// image, `texture/wrapped.rs`) where the subpass reads the others as input attachments: their
    /// input references are `VK_ATTACHMENT_UNUSED`.
    pub(super) no_inputs: ColorList<u32>,
    /// The color indices, ascending, the draws' subpass leaves out (`VK_ATTACHMENT_UNUSED`): the
    /// attachments its pipelines read and write as storage images in the pixel interlock
    /// (`framebuffer_fetch.rs`), but for the [`anchor`](PassShape::anchor).
    pub(super) storage: ColorList<u32>,
    /// Every subpass's view mask: the views of the instance's amplified draws, 0 without
    /// multiview (the module docs).
    pub(super) view_mask: u32,
    /// The color index each of the draws' subpass's color references (and resolve references)
    /// names, by fragment output location: the pass color each location stores into
    /// (`color_reformat::Routing`), `None` for no attachment. Empty where location `i` names
    /// color `i`.
    pub(super) locations: ColorList<Option<u32>>,
}

impl PassShape {
    /// The shape of `layout`.
    pub(super) fn of(layout: &PassLayout) -> Result<PassShape, Error> {
        let mut no_inputs = ColorList::new();
        if layout.fetch_inputs {
            for (index, color) in layout.colors.iter().enumerate() {
                if let Some(color) = color {
                    if !takes_input(&color.target)? {
                        no_inputs.push(index as u32)?;
                    }
                }
            }
        }
        Ok(PassShape {
            colors: ColorList::of(
                (layout.colors.iter()).map(|color| color.as_ref().map(|c| c.target.format)),
            )?,
            depth_stencil: layout.depth_stencil.as_ref().map(|ds| ds.target.format),
            samples: layout.samples,
            color_resolves: ColorList::of(
                (layout.colors.iter())
                    .map(|color| subpass_resolve(color.as_ref()).map(|r| r.attachment.format)),
            )?,
            depth_stencil_resolve: subpass_step(layout)
                .map(|step| (step.resolve.attachment.format, step.depth, step.stencil)),
            fetch_inputs: layout.fetch_inputs,
            no_inputs,
            storage: ColorList::new(),
            view_mask: 0,
            locations: ColorList::new(),
        })
    }

    /// This shape as a pipeline's key holds it (`pipeline/graphics.rs`): its resolves left out,
    /// which render pass compatibility ignores in a pass of one subpass, as every instance's pass
    /// is (the module docs). A pipeline made against a pass that resolves is then the one a draw
    /// in a pass of the same shape that resolves nothing takes, and the other way round, as it is
    /// the one a creation predicts (`pipeline/predicted.rs`), which cannot know whether its draws'
    /// passes resolve.
    pub(super) fn keyed(&self) -> PassShape {
        let mut keyed = self.clone();
        keyed.color_resolves.fill(None);
        keyed.depth_stencil_resolve = None;
        keyed
    }

    /// Whether `keyed`, a pipeline key's shape ([`keyed`](PassShape::keyed)), is this shape's.
    pub(super) fn keys_as(&self, keyed: &PassShape) -> bool {
        let PassShape {
            colors,
            depth_stencil,
            samples,
            color_resolves: _,
            depth_stencil_resolve: _,
            fetch_inputs,
            no_inputs,
            storage,
            view_mask,
            locations,
        } = self;
        *colors == keyed.colors
            && *depth_stencil == keyed.depth_stencil
            && *samples == keyed.samples
            && keyed.color_resolves.iter().all(Option::is_none)
            && keyed.depth_stencil_resolve.is_none()
            && *fetch_inputs == keyed.fetch_inputs
            && *no_inputs == keyed.no_inputs
            && *storage == keyed.storage
            && *view_mask == keyed.view_mask
            && *locations == keyed.locations
    }

    /// This shape as the record of the driver's data of a pipeline made against it names it
    /// (`GraphicsContent::record` in `pipeline/graphics.rs`), to be read as `Debug` text: each
    /// attachment's presence in place of its format (whether each color index and the
    /// depth/stencil attachment has one), and the rest of the shape whole. Its resolves are left
    /// out, as a key's shape holds none ([`keyed`](PassShape::keyed)).
    ///
    /// A stage's conversion reads no attachment format through the pass: a format the translation
    /// reads (a channel it rounds itself, a fetch in its storage form) is in the stage's own
    /// words and constants, which name its record too. So makings that differ only in an
    /// attachment's format share one record and one `VkPipelineCache`, where the driver finds the
    /// stages it compiled for the first: ANV's fragment shader library of a module of some
    /// 200,000 words made against `RGBA8Unorm`, `RGBA16Float` and `BGRA8Unorm` passes, 3.3 to
    /// 3.7 s each with a cache of its own, is found in the first's shared cache in under a
    /// millisecond (Mesa 26.2.6, Iris Xe, its creation feedback `APPLICATION_PIPELINE_CACHE_HIT`).
    /// A driver whose
    /// conversion does read a format keys that making's entry apart in the one cache: a compile,
    /// never another pipeline.
    pub(super) fn without_formats(&self) -> impl std::fmt::Debug + '_ {
        let PassShape {
            colors,
            depth_stencil,
            samples,
            color_resolves: _,
            depth_stencil_resolve: _,
            fetch_inputs,
            no_inputs,
            storage,
            view_mask,
            locations,
        } = self;
        (
            colors.iter().map(Option::is_some).collect::<Vec<bool>>(),
            depth_stencil.is_some(),
            *samples,
            *fetch_inputs,
            no_inputs,
            storage,
            *view_mask,
            locations,
        )
    }

    /// This shape with `storage` (ascending) left out of the draws' subpass; refused (an
    /// invariant) past a pass's color attachments.
    pub(super) fn with_storage(self, storage: &[u32]) -> Result<PassShape, Error> {
        let storage = ColorList::of(storage.iter().copied())?;
        Ok(PassShape { storage, ..self })
    }

    /// This shape with every subpass of `view_mask`.
    pub(super) fn with_view_mask(self, view_mask: u32) -> PassShape {
        PassShape { view_mask, ..self }
    }

    /// This shape with the draws' subpass's color references naming `locations`
    /// ([`locations`](PassShape::locations)); refused (an invariant) past a pass's color
    /// attachments.
    pub(super) fn with_locations(self, locations: &[Option<u32>]) -> Result<PassShape, Error> {
        let locations = ColorList::of(locations.iter().copied())?;
        Ok(PassShape { locations, ..self })
    }

    /// The draws' subpass's color reference count: one per fragment output location a pipeline
    /// of the pass blends.
    pub(super) fn subpass_colors(&self) -> usize {
        match self.locations.is_empty() {
            true => self.colors.len(),
            false => self.locations.len(),
        }
    }

    /// Whether the draws' subpass references an attachment at fragment output `location`.
    pub(super) fn references(&self, location: u32) -> bool {
        let index = match self.locations.is_empty() {
            true => Some(location),
            false => self.locations.get(location as usize).copied().flatten(),
        };
        index.is_some_and(|index| {
            self.colors.get(index as usize).is_some_and(Option::is_some)
                && !self.storage.contains(&index)
        })
    }

    /// The color index of `storage` the draws' subpass keeps after all, `DONT_CARE` and written
    /// by no pipeline (the module docs): the first, in a multisampled pass whose storage
    /// attachments are sample planes (in the pixel interlock on `profile`, a device without
    /// multisampled storage images) where the subpass would have no other attachment. Where the
    /// storage attachments are the attachments' own multisampled images, their draws write them, so
    /// none is kept.
    pub(super) fn anchor(&self, profile: &DeviceProfile) -> Option<u32> {
        let planes = !self.fetch_inputs && framebuffer_fetch::through_planes(profile, self.samples);
        let attached =
            self.depth_stencil.is_some()
                || self.colors.iter().enumerate().any(|(index, format)| {
                    format.is_some() && !self.storage.contains(&(index as u32))
                });
        if planes && !attached {
            self.storage.first().copied()
        } else {
            None
        }
    }
}

/// The resolve of `color` the pass's subpass makes: its resolve, where the device makes it and
/// no exact resolve after the pass does (`color_resolve.rs`).
pub(super) fn subpass_resolve(color: Option<&ColorTarget>) -> Option<&Resolve> {
    color
        .and_then(|c| c.resolve.as_ref())
        .filter(|r| r.shader.is_none())
}

/// Where an instance of a pass falls among its instances (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Position {
    /// Metal's load actions run here.
    pub(super) first: bool,
    /// Metal's store actions and the resolves run here.
    pub(super) last: bool,
}

/// The depth/stencil resolves of `layout`, in order: the draws' subpass's, then those after the
/// pass (the module docs).
pub(super) fn depth_stencil_steps(layout: &PassLayout) -> impl Iterator<Item = &ResolveStep> {
    layout
        .depth_stencil
        .iter()
        .filter_map(|ds| ds.resolve.as_ref())
        .flat_map(|r| &r.steps)
}

/// The depth/stencil resolve `layout`'s draws' subpass makes: its first.
pub(super) fn subpass_step(layout: &PassLayout) -> Option<&ResolveStep> {
    depth_stencil_steps(layout).next()
}

/// The depth/stencil resolves of `layout` past the first, each made after the pass in a pass of
/// its own ([`later_resolve`]).
pub(super) fn later_steps(layout: &PassLayout) -> impl Iterator<Item = &ResolveStep> {
    depth_stencil_steps(layout).skip(1)
}

/// One attachment's operations: of its color or depth aspect, then of its stencil aspect.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Ops {
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    stencil_load: vk::AttachmentLoadOp,
    stencil_store: vk::AttachmentStoreOp,
}

/// Loaded and stored, both aspects.
const KEEP: Ops = Ops {
    load: vk::AttachmentLoadOp::LOAD,
    store: vk::AttachmentStoreOp::STORE,
    stencil_load: vk::AttachmentLoadOp::LOAD,
    stencil_store: vk::AttachmentStoreOp::STORE,
};

/// The Vulkan load operation of Metal's `load`.
fn load_op(load: LoadAction) -> vk::AttachmentLoadOp {
    match load {
        LoadAction::Load => vk::AttachmentLoadOp::LOAD,
        LoadAction::Clear => vk::AttachmentLoadOp::CLEAR,
        LoadAction::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}

/// The Vulkan store operation of Metal's known `store`: `MultisampleResolve` leaves the samples
/// undefined, as `DontCare` does (`resolve.rs`), and both are `dont_care`'s.
fn store_op(store: StoreAction, dont_care: DontCare) -> vk::AttachmentStoreOp {
    match store {
        StoreAction::DontCare | StoreAction::MultisampleResolve => dont_care.op(),
        _ => vk::AttachmentStoreOp::STORE,
    }
}

/// How a device stores an attachment Metal stores `DontCare` or `MultisampleResolve` (the module
/// docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum DontCare {
    /// Handed to the driver as `DONT_CARE`: a driver measured to keep one of Metal's two answers
    /// there, or to store as Metal's own, or an attachment no texture of the program's holds.
    Discarded,
    /// Stored (`STORE`): the texels the pass renders, Metal's answer where its tiler overflows.
    Stored,
}

impl DontCare {
    /// How `device` stores such an attachment.
    pub(super) fn of(device: &Device) -> Self {
        let facts = &device.properties().driver_facts;
        match facts.contains(&DriverFact::DontCareStoreIsOneOfMetals)
            || facts.contains(&DriverFact::DontCareStoreIsMetals)
        {
            true => Self::Discarded,
            false => Self::Stored,
        }
    }

    /// How an attachment held by no texture of the program's (`unheld`) is stored where the
    /// device stores as `self`.
    fn unless(self, unheld: bool) -> Self {
        match unheld {
            true => Self::Discarded,
            false => self,
        }
    }

    /// The Vulkan store operation.
    fn op(self) -> vk::AttachmentStoreOp {
        match self {
            Self::Discarded => vk::AttachmentStoreOp::DONT_CARE,
            Self::Stored => vk::AttachmentStoreOp::STORE,
        }
    }
}

/// The operations of a resolve attachment's aspect resolved in `mode`: written whole by the
/// resolve, or kept where it does not resolve (`NONE`).
fn resolve_ops(mode: ResolveMode) -> (vk::AttachmentLoadOp, vk::AttachmentStoreOp) {
    if mode == ResolveMode::None {
        (vk::AttachmentLoadOp::LOAD, vk::AttachmentStoreOp::STORE)
    } else {
        (
            vk::AttachmentLoadOp::DONT_CARE,
            vk::AttachmentStoreOp::STORE,
        )
    }
}

/// The sample planes' anchor's operations (the module docs): neither loaded nor stored.
const ANCHORED: Ops = Ops {
    load: vk::AttachmentLoadOp::DONT_CARE,
    store: vk::AttachmentStoreOp::DONT_CARE,
    stencil_load: vk::AttachmentLoadOp::DONT_CARE,
    stencil_store: vk::AttachmentStoreOp::DONT_CARE,
};

/// The operations of a color the draws' subpass leaves out for the pixel interlock (the module
/// docs), which no subpass uses: neither loaded nor stored, as such an attachment is in any case
/// (`storage_placeholder.rs`).
const LEFT_OUT: Ops = Ops {
    load: vk::AttachmentLoadOp::DONT_CARE,
    store: vk::AttachmentStoreOp::DONT_CARE,
    stencil_load: vk::AttachmentLoadOp::DONT_CARE,
    stencil_store: vk::AttachmentStoreOp::DONT_CARE,
};

/// A color resolve attachment's operations: the resolve writes the render area.
const RESOLVED: Ops = Ops {
    load: vk::AttachmentLoadOp::DONT_CARE,
    store: vk::AttachmentStoreOp::STORE,
    stencil_load: vk::AttachmentLoadOp::DONT_CARE,
    stencil_store: vk::AttachmentStoreOp::DONT_CARE,
};

/// The load operation of Metal's `load` in an instance at `at`: Metal's in the first, `LOAD` in
/// every other.
fn load_at(load: LoadAction, at: Position) -> vk::AttachmentLoadOp {
    if at.first {
        load_op(load)
    } else {
        vk::AttachmentLoadOp::LOAD
    }
}

/// The store operation of Metal's known `store` in an instance at `at`: Metal's in the last
/// (its `DontCare` `dont_care`'s), `STORE` in every other.
fn store_at(store: StoreAction, at: Position, dont_care: DontCare) -> vk::AttachmentStoreOp {
    if at.last {
        store_op(store, dont_care)
    } else {
        vk::AttachmentStoreOp::STORE
    }
}

/// The operations of one aspect of the depth/stencil attachment in an instance at `at`: the
/// attachment's own (its `DontCare` `dont_care`'s), kept when the Metal format has the aspect but
/// no attachment names it, `DONT_CARE` when only the Vulkan format has it.
fn aspect_ops(
    aspect: Option<&AspectTarget>,
    keep: bool,
    at: Position,
    dont_care: DontCare,
) -> (vk::AttachmentLoadOp, vk::AttachmentStoreOp) {
    match aspect {
        Some(aspect) => (
            load_at(aspect.load, at),
            store_at(aspect.store, at, dont_care.unless(aspect.memoryless)),
        ),
        None if keep => (vk::AttachmentLoadOp::LOAD, vk::AttachmentStoreOp::STORE),
        None => (
            vk::AttachmentLoadOp::DONT_CARE,
            vk::AttachmentStoreOp::DONT_CARE,
        ),
    }
}

/// A `VkRenderPass` the runtime made, destroyed with its last holder unless its device keeps it.
pub(super) struct RenderPassObject {
    device: Device,
    pub(super) raw: vk::RenderPass,
    owner: Owner,
}

/// Who destroys a [`RenderPassObject`]'s pass.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Owner {
    /// Its last holder.
    Holders,
    /// The device, with itself (`attachmentless.rs`).
    Device,
}

impl RenderPassObject {
    /// A holder of `raw`, a pass `device` keeps.
    pub(super) fn kept(device: &Device, raw: vk::RenderPass) -> RenderPassObject {
        RenderPassObject {
            device: device.clone(),
            raw,
            owner: Owner::Device,
        }
    }

    /// The pass, which its device keeps from now on: dropping this destroys nothing.
    pub(super) fn into_kept(mut self) -> vk::RenderPass {
        self.owner = Owner::Device;
        self.raw
    }
}

impl Drop for RenderPassObject {
    fn drop(&mut self) {
        if self.owner == Owner::Device {
            return;
        }
        // Once the process is exiting, the pass is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: no pending command uses the pass: a command buffer holds the passes its
        // encoders recorded with until it completes, and a pipeline made with it does not keep it.
        unsafe { release::object(self.device.shared().vk(), self.raw) };
    }
}

/// A `VkFramebuffer` the runtime made, destroyed with its last holder, and the placeholders whose
/// views it holds (`storage_placeholder.rs`), which outlive it.
pub(super) struct FramebufferObject {
    device: Device,
    pub(super) raw: vk::Framebuffer,
    placeholders: Vec<Arc<Placeholder>>,
}

impl Drop for FramebufferObject {
    fn drop(&mut self) {
        // As for `RenderPassObject`.
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: as for `RenderPassObject`.
        unsafe { release::object(self.device.shared().vk(), self.raw) };
    }
}

/// A dependency of the draws' subpass on itself (the module docs), whose scopes a barrier
/// recorded between two draws of an instance takes (`between_storage_draws`, `between_draws`,
/// `after_geometry_writes`): `stages` are both scopes' stages, but for
/// [`GEOMETRY_SELF_DEPENDENCY`]'s, whose source stages are the device's.
pub(super) struct SelfDependency {
    pub(super) stages: vk::PipelineStageFlags,
    pub(super) src_access: vk::AccessFlags,
    pub(super) dst_access: vk::AccessFlags,
}

/// The scopes of the barrier between two draws that reach storage the pixel interlock orders (a
/// fragment imageblock's storage, or an interlocked color's storage image): within the
/// `DRAWS_SELF_DEPENDENCY`.
pub(super) const STORAGE_SELF_DEPENDENCY: SelfDependency = SelfDependency {
    stages: vk::PipelineStageFlags::FRAGMENT_SHADER,
    src_access: vk::AccessFlags::SHADER_WRITE,
    dst_access: vk::AccessFlags::from_raw(
        vk::AccessFlags::SHADER_READ.as_raw() | vk::AccessFlags::SHADER_WRITE.as_raw(),
    ),
};

/// The draws' subpass's `SelfDependency`: from the fragment stages' writes (the fragment
/// shader's, as storage, and the attachments', in the fragment tests and the color output) to
/// their reads (input attachments among them) and writes. Every stage is a framebuffer-space one,
/// so the dependency is by region, as Vulkan requires of a subpass's dependency on itself over
/// such stages.
pub(super) const DRAWS_SELF_DEPENDENCY: SelfDependency = SelfDependency {
    stages: vk::PipelineStageFlags::from_raw(
        vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw()
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw(),
    ),
    src_access: vk::AccessFlags::from_raw(
        vk::AccessFlags::SHADER_WRITE.as_raw()
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
    ),
    dst_access: vk::AccessFlags::from_raw(
        vk::AccessFlags::SHADER_READ.as_raw()
            | vk::AccessFlags::SHADER_WRITE.as_raw()
            | vk::AccessFlags::INPUT_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags::COLOR_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
    ),
};

/// The flags of the draws' subpass's dependency on itself in a pass whose subpasses have
/// `view_mask`, which a barrier inside it takes too: by region, and view by view in a subpass of
/// more than one view (`VUID-VkRenderPassCreateInfo2-pDependencies-03060`).
pub(super) fn self_dependency_flags(view_mask: u32) -> vk::DependencyFlags {
    if view_mask.count_ones() > 1 {
        vk::DependencyFlags::BY_REGION | vk::DependencyFlags::VIEW_LOCAL
    } else {
        vk::DependencyFlags::BY_REGION
    }
}

/// The draws' subpass's dependency on itself from the draws' reads of their indirect arguments,
/// in `DRAW_INDIRECT`, to every graphics stage, with no access type: an execution dependency,
/// which "Write-after-read hazards can be solved with just" (the Synchronization chapter's
/// Execution and Memory Dependencies). `DRAW_INDIRECT` is no framebuffer-space stage, so it is not
/// by region, and the stage of its source scope is logically earlier than or equal to every stage
/// of its destination scope, as a subpass's dependency on itself over other stages needs
/// (`VUID-VkSubpassDependency2-srcSubpass-03087`). A barrier before a draw that writes what an
/// earlier draw of its instance read as its arguments takes it (`after_indirect_reads`).
pub(super) const INDIRECT_SELF_DEPENDENCY: (vk::PipelineStageFlags, vk::PipelineStageFlags) = (
    vk::PipelineStageFlags::DRAW_INDIRECT,
    vk::PipelineStageFlags::ALL_GRAPHICS,
);

/// The flags of [`INDIRECT_SELF_DEPENDENCY`] and [`GEOMETRY_SELF_DEPENDENCY`] in a pass whose
/// subpasses have `view_mask`, which their barriers take too: view by view in a subpass of more
/// than one view (`VUID-VkRenderPassCreateInfo2-pDependencies-03060`), and never by region, which
/// "must not" be in the dependency of a barrier that is not
/// (`VUID-vkCmdPipelineBarrier-None-07889`).
fn stage_dependency_flags(view_mask: u32) -> vk::DependencyFlags {
    if view_mask.count_ones() > 1 {
        vk::DependencyFlags::VIEW_LOCAL
    } else {
        vk::DependencyFlags::empty()
    }
}

/// Record, into `draws`, a render pass instance's secondary command buffer (or its pass's
/// command buffer, inline) whose subpasses have `view_mask`, the barrier before a draw that
/// writes, in `stages`, memory an earlier draw of the instance read as its indirect arguments:
/// an execution dependency from the earlier draws' `DRAW_INDIRECT` to `stages`, within
/// [`INDIRECT_SELF_DEPENDENCY`] and with its flags. Metal reads a draw's arguments before a later
/// draw of the encoder writes them, behind no barrier (measured on an M3 Max and an M4); Vulkan
/// orders the read before the write only by a dependency, which a write-after-read hazard needs
/// with no memory dependency.
///
/// # Safety
///
/// `draws` is a command buffer of `raw`'s device recording inside a pass this module made, of
/// `view_mask`, and only this thread records it; `stages` are graphics stages the device has.
pub(super) unsafe fn after_indirect_reads(
    raw: &ash::Device,
    draws: vk::CommandBuffer,
    view_mask: u32,
    stages: vk::PipelineStageFlags,
) {
    // SAFETY: the caller's contract; the pass's subpass depends on itself from `DRAW_INDIRECT`
    // to every graphics stage with these flags, so the barrier is valid inside it.
    unsafe {
        raw.cmd_pipeline_barrier(
            draws,
            INDIRECT_SELF_DEPENDENCY.0,
            stages,
            stage_dependency_flags(view_mask),
            &[],
            &[],
            &[],
        );
    }
}

/// The destination scope and the access types of the source scope of the draws' subpass's
/// dependency on itself from the shader writes of the stages a draw runs before its fragments
/// ([`geometry_write_stages`], the source stages) to the fragment shader's reads and writes:
/// Metal's tile-based passes run every draw's vertex, object and mesh functions before any draw's
/// fragments, which Vulkan orders only by a dependency ("read-after-write and write-after-write
/// hazards need appropriate memory dependencies to be included between them", the
/// Synchronization chapter's Execution and Memory Dependencies). No source stage is a
/// framebuffer-space one, so it is not by region (`VUID-VkSubpassDependency2-srcSubpass-02245`
/// asks that only of a dependency whose two scopes hold one), and every source stage is
/// logically earlier than the fragment shader (`VUID-VkSubpassDependency2-srcSubpass-03087`). A
/// barrier before a draw whose fragment shader reads or writes what an earlier draw of its
/// instance wrote in those stages takes it (`after_geometry_writes`).
pub(super) const GEOMETRY_SELF_DEPENDENCY: SelfDependency = SelfDependency {
    stages: vk::PipelineStageFlags::FRAGMENT_SHADER,
    src_access: vk::AccessFlags::SHADER_WRITE,
    dst_access: vk::AccessFlags::from_raw(
        vk::AccessFlags::SHADER_READ.as_raw()
            | vk::AccessFlags::SHADER_WRITE.as_raw()
            | vk::AccessFlags::UNIFORM_READ.as_raw(),
    ),
};

/// The source stages of [`GEOMETRY_SELF_DEPENDENCY`] on `device`: the vertex shader, and the
/// tessellation, geometry, task and mesh shaders where the device enables them (a dependency names
/// no stage whose feature is off, `VUID-VkSubpassDependency2-srcStageMask-04091` and its
/// siblings). Every pass of the device names the same, so its passes stay compatible.
fn geometry_write_stages(device: &Device) -> vk::PipelineStageFlags {
    use metal2vulkan::device::VkFeature;
    let features = &device.enabled().features;
    let mut stages = vk::PipelineStageFlags::VERTEX_SHADER;
    for (feature, more) in [
        (
            VkFeature::TessellationShader,
            vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER
                | vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
        ),
        (
            VkFeature::GeometryShader,
            vk::PipelineStageFlags::GEOMETRY_SHADER,
        ),
        (
            VkFeature::TaskShader,
            vk::PipelineStageFlags::TASK_SHADER_EXT,
        ),
        (
            VkFeature::MeshShader,
            vk::PipelineStageFlags::MESH_SHADER_EXT,
        ),
    ] {
        if features.contains(&feature) {
            stages |= more;
        }
    }
    stages
}

/// Record, into `draws`, a render pass instance's secondary command buffer (or its pass's
/// command buffer, inline) whose subpasses have `view_mask`, the barrier before a draw whose
/// fragment shader reads or writes memory an earlier draw of the instance wrote before its
/// fragments: `barrier`, from the earlier writes' stages and access types to the fragment shader
/// and the later draw's access types there, within [`GEOMETRY_SELF_DEPENDENCY`] and with its
/// flags. Metal runs the earlier draw's vertex, object and mesh functions before the later
/// draw's fragments.
///
/// # Safety
///
/// `draws` is a command buffer of `raw`'s device recording inside a pass this module made, of
/// `view_mask`, and only this thread records it; `barrier`'s source stages are among
/// [`geometry_write_stages`]'s and its access types within [`GEOMETRY_SELF_DEPENDENCY`]'s, and
/// its destination stage is the fragment shader.
pub(super) unsafe fn after_geometry_writes(
    raw: &ash::Device,
    draws: vk::CommandBuffer,
    view_mask: u32,
    barrier: LegacyBarrier,
) {
    let memory = vk::MemoryBarrier::default()
        .src_access_mask(barrier.src_access)
        .dst_access_mask(barrier.dst_access);
    // SAFETY: the caller's contract; the pass's subpass depends on itself from the device's
    // stages before the fragments to the fragment shader with these access types and flags, so
    // the barrier is valid inside it.
    unsafe {
        raw.cmd_pipeline_barrier(
            draws,
            barrier.src_stages,
            barrier.dst_stages,
            stage_dependency_flags(view_mask),
            &[memory],
            &[],
            &[],
        );
    }
}

/// Record, into `draws`, a render pass instance's secondary command buffer whose subpasses have
/// `view_mask`, a barrier of the `DRAWS_SELF_DEPENDENCY`'s scopes: the draws after it see the
/// fragment stages' writes of the draws before it, pixel by pixel.
///
/// # Safety
///
/// `draws` is a secondary command buffer of `raw`'s device recording inside a pass this module
/// made, of `view_mask`, and only this thread records it.
pub(super) unsafe fn between_draws(raw: &ash::Device, draws: vk::CommandBuffer, view_mask: u32) {
    let scopes = DRAWS_SELF_DEPENDENCY;
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(scopes.src_access)
        .dst_access_mask(scopes.dst_access);
    // SAFETY: the caller's contract; the pass's subpass depends on itself with these scopes and
    // flags, so the barrier is valid inside it.
    unsafe {
        raw.cmd_pipeline_barrier(
            draws,
            scopes.stages,
            scopes.stages,
            self_dependency_flags(view_mask),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Record, into `draws`, a render pass instance's secondary command buffer whose subpasses have
/// `view_mask`, the barrier from an earlier draw's writes of storage the pixel interlock orders to
/// the next draw's reads and writes of it: a fragment imageblock's storage
/// (`fragment_imageblock.rs`) or an interlocked color's storage image (`framebuffer_fetch.rs`).
/// Its scopes are the fragment shader's (`STORAGE_SELF_DEPENDENCY`), within the draws' subpass's
/// dependency on itself and with its flags (`self_dependency_flags`: by region, and view by view
/// under multiview). RADV on an AMD 780M (Mesa 26.2.4) does not order such draws by the interlock
/// alone, as Vulkan's rasterization order promises: a later draw, of the same pipeline or
/// another, read texels as they were before the earlier one's writes. A driver measured to order
/// them takes none between draws that reach only colors (`DriverFact::InterlockOrdersDraws`); on
/// one whose such draws race under load (`DriverFact::InterlockedDrawsRaceUnderLoad`: MoltenVK on
/// Apple's GPUs, which encodes no Metal barrier for this one) the later draw records into a new
/// render pass instance instead (`render.rs`).
///
/// # Safety
///
/// `draws` is a secondary command buffer of `raw`'s device recording inside a pass this module
/// made, of `view_mask`, and only this thread records it.
pub(super) unsafe fn between_storage_draws(
    raw: &ash::Device,
    draws: vk::CommandBuffer,
    view_mask: u32,
) {
    let scopes = STORAGE_SELF_DEPENDENCY;
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(scopes.src_access)
        .dst_access_mask(scopes.dst_access);
    // SAFETY: the caller's contract; the pass's subpass depends on itself with these scopes and
    // flags, so the barrier is valid inside it.
    unsafe {
        raw.cmd_pipeline_barrier(
            draws,
            scopes.stages,
            scopes.stages,
            self_dependency_flags(view_mask),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Which barrier a draw records after the earlier draws of its instance whose accesses it meets
/// (`render.rs`'s `InterlockedStorage`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DrawsBarrier {
    /// From the fragment shader's storage writes to its reads and writes
    /// ([`between_storage_draws`]): storage the pixel interlock orders.
    Storage,
    /// The draws' subpass's whole dependency on itself ([`between_draws`]): colors read through
    /// input attachments ordered between draws alone, whose barrier orders the earlier draws' color
    /// writes before a later draw's input attachment reads, and those reads before a later draw's
    /// color writes (and the storage's scopes too, which it holds).
    Whole,
}

impl DrawsBarrier {
    /// Record the barrier into `draws`, as [`between_storage_draws`] or [`between_draws`].
    ///
    /// # Safety
    ///
    /// As [`between_storage_draws`]'s and [`between_draws`]'s.
    pub(super) unsafe fn record(self, raw: &ash::Device, draws: vk::CommandBuffer, view_mask: u32) {
        // SAFETY: the caller's contract.
        unsafe {
            match self {
                DrawsBarrier::Storage => between_storage_draws(raw, draws, view_mask),
                DrawsBarrier::Whole => between_draws(raw, draws, view_mask),
            }
        }
    }
}

/// An attachment's image, the subresource a render pass instance renders of it, and the layout
/// the instance holds it in ([`moved_barriers`]).
#[derive(Clone, Copy)]
pub(super) struct MovedImage {
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    layout: vk::ImageLayout,
}

/// The attachments of `layout` an instance of `views` holds out of `GENERAL` ([`viewed`]): its
/// `feedback` colors, in the feedback loop layout, and its depth/stencil attachment where it holds
/// it in `depth` ([`depth_layout`]). The images, subresources and layouts its barriers move.
pub(super) fn moved_images(
    layout: &PassLayout,
    views: ViewSetup,
    feedback: Feedback,
    depth: vk::ImageLayout,
) -> Result<Vec<MovedImage>, Error> {
    let mut images = Vec::new();
    for (index, color) in layout.colors.iter().enumerate() {
        let Some(color) = color.as_ref().filter(|_| feedback.has(index)) else {
            continue;
        };
        let target = viewed(&color.target, views);
        images.push(MovedImage {
            image: target.texture.raw_image()?,
            range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: target.level,
                level_count: 1,
                base_array_layer: target.layer,
                layer_count: target.layers,
            },
            layout: vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
        });
    }
    if let Some(ds) = layout.depth_stencil.as_ref() {
        if depth != vk::ImageLayout::GENERAL {
            let target = viewed(&ds.target, views);
            images.push(MovedImage {
                image: target.texture.raw_image()?,
                range: vk::ImageSubresourceRange {
                    aspect_mask: super::texture::image_aspects(target.texture.image_plan()?.format)
                        .into(),
                    base_mip_level: target.level,
                    level_count: 1,
                    base_array_layer: target.layer,
                    layer_count: target.layers,
                },
                layout: depth,
            });
        }
    }
    Ok(images)
}

/// The layout the draws' subpass of an instance of `layout` holds its depth/stencil attachment in
/// (the module docs): `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` for a multisampled one of a depth format,
/// not a transient attachment, on a device whose driver is measured to render it faster there
/// than in `GENERAL`, the moves included (`DriverFact::MultisampledDepthFasterOutOfGeneral`);
/// `GENERAL` for every other, and for a pass without one.
pub(super) fn depth_layout(device: &Device, layout: &PassLayout) -> Result<vk::ImageLayout, Error> {
    let general = vk::ImageLayout::GENERAL;
    let Some(ds) = layout.depth_stencil.as_ref() else {
        return Ok(general);
    };
    let measured = (device.properties().driver_facts)
        .contains(&DriverFact::MultisampledDepthFasterOutOfGeneral);
    if !measured || ds.target.samples < 2 {
        return Ok(general);
    }
    let plan = ds.target.texture.image_plan()?;
    let depth = plan
        .format
        .block()
        .is_some_and(|block| block.depth.is_some());
    let transient = (plan.usage).contains(metal2vulkan::vk::ImageUsageFlags::TRANSIENT_ATTACHMENT);
    Ok(match depth && !transient {
        true => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        false => general,
    })
}

/// Record, outside a render pass, the barriers that move `images` (an instance's attachments held
/// out of `GENERAL`, [`moved_images`]) from `GENERAL` into their layouts before the instance
/// begins (`into`), or back after it ends, recorded for a queue family of `family`'s flags: from
/// every earlier command's writes to every later command's accesses.
///
/// The instance's pass moves no layout itself ([`Feedback`], [`depth_layout`]): a move from a
/// description's initial
/// layout would be ordered after nothing before the pass, as "If there is no subpass dependency
/// from VK_SUBPASS_EXTERNAL to the first subpass that uses an attachment, then an implicit subpass
/// dependency exists", whose `srcStageMask` is `VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT` (`NONE`) and
/// `srcAccessMask` 0 (the Render Pass chapter); the passes the runtime makes have no external
/// dependency, and one in every pass would order every pass's begin. An image barrier's own move
/// is ordered between its scopes: "When a layout transition is specified in a memory dependency,
/// it happens-after the availability operations in the memory dependency, and happens-before the
/// visibility operations" (Image Layout Transitions). The feedback loop
/// layout is valid for the colors' images, made with the usage it needs
/// (`texture::feedback_candidate`): "This layout is valid only for image subresources of images
/// created with the VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT usage flag set, and either the
/// VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT or ... and either the VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT or
/// VK_IMAGE_USAGE_SAMPLED_BIT usage flags set", and `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` for a
/// depth/stencil attachment's, made with `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT`
/// (`VUID-VkImageMemoryBarrier-oldLayout-01210`). Between the two barriers the subresource is in
/// that layout, which every access of it there names: the instance's attachment, and a feedback
/// color's draws' sampled descriptors of it (`binding.rs`; a draw reads a depth attachment's
/// texture through a copy, `texture/attached_reads.rs`), as "the layouts of the image
/// subresources that are accessed must all match exactly the layout specified via the API
/// controlling those accesses" (Image Layout Matching Rules).
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass, only the caller records it, the images are
/// live until it completes, and each subresource is in the layout the move leaves.
pub(super) unsafe fn moved_barriers(
    raw: &ash::Device,
    command_buffer: vk::CommandBuffer,
    images: &[MovedImage],
    into: bool,
    family: vk::QueueFlags,
) {
    if images.is_empty() {
        return;
    }
    let general = vk::ImageLayout::GENERAL;
    let barriers: Vec<vk::ImageMemoryBarrier<'_>> = images
        .iter()
        .map(|i| {
            let layouts = match into {
                true => (general, i.layout),
                false => (i.layout, general),
            };
            image_barrier(
                i.image,
                i.range,
                layouts,
                vk::AccessFlags::MEMORY_WRITE,
                ImageDst::Every,
                family,
            )
        })
        .collect();
    let all = vk::PipelineStageFlags::ALL_COMMANDS;
    // SAFETY: the caller's contract; each barrier names a live image's subresource in the
    // layout it leaves, and no queue family transfer.
    unsafe {
        raw.cmd_pipeline_barrier(
            command_buffer,
            all,
            all,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
    }
}

/// The moves of a color image's `range` from `GENERAL` into `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT`
/// and back, one after the other, as [`moved_barriers`] records them around an instance whose
/// draws read the image through a snapshot: the device's check of an image in place over host
/// pages that renders (`in_place_check.rs`), which checks that those moves leave every byte bound
/// to it beside its texels alone.
///
/// # Safety
///
/// As [`moved_barriers`]': `command_buffer` is recording outside a render pass, only the caller
/// records it, the image is live until it completes, made with feedback loop usage, and its
/// `range` rests in `GENERAL`.
pub(crate) unsafe fn feedback_moves(
    raw: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    family: vk::QueueFlags,
) {
    let moved = [MovedImage {
        image,
        range,
        layout: vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
    }];
    // SAFETY: the caller's contract; the first moves the range out of `GENERAL`, where it rests,
    // and the second back from the layout the first left.
    unsafe {
        moved_barriers(raw, command_buffer, &moved, true, family);
        moved_barriers(raw, command_buffer, &moved, false, family);
    }
}

/// The pass of `shape` with every operation `LOAD` and `STORE`: compatible with every pass of
/// that shape, kept by its device ([`create`]). A shape without attachments takes its device's
/// kept pass (`attachmentless.rs`).
pub(super) fn compatible_pass(
    device: &Device,
    shape: &PassShape,
) -> Result<RenderPassObject, Error> {
    if let Some(kept) = attachmentless::kept(device, shape)? {
        return Ok(kept);
    }
    // Found without writing its operations out.
    if let Some(raw) = lock(&device.shared().render_passes).compatible(shape) {
        return Ok(RenderPassObject::kept(device, raw));
    }
    let (colors, resolves) = kept_operations(shape);
    let rest = PassRest::compatible(shape);
    create(
        device,
        shape,
        &colors,
        rest.depth_stencil,
        &resolves,
        rest.depth_stencil_resolve,
        rest.resolving,
        rest.feedback,
        rest.depth,
    )
}

/// [`compatible_pass`], shared: every pass of a shape is compatible with every other, so one
/// serves every encoder.
pub(super) fn shared_compatible_pass(
    device: &Device,
    shape: &PassShape,
) -> Result<Arc<RenderPassObject>, Error> {
    compatible_pass(device, shape).map(Arc::new)
}

/// The pass of `shape` with every operation `LOAD` and `STORE`, made now, of its own (the pass a
/// device keeps for a shape without attachments, `attachmentless.rs`).
pub(super) fn made_compatible_pass(
    device: &Device,
    shape: &PassShape,
) -> Result<RenderPassObject, Error> {
    let (colors, resolves) = kept_operations(shape);
    let rest = PassRest::compatible(shape);
    make_pass(
        device,
        shape,
        &colors,
        rest.depth_stencil,
        &resolves,
        rest.depth_stencil_resolve,
        rest.resolving,
        rest.feedback,
        rest.depth,
    )
}

/// The color and resolve operations of a compatible pass of `shape`: [`KEEP`] for each present
/// color attachment, and for each color resolve.
fn kept_operations(shape: &PassShape) -> (Vec<Ops>, Vec<Ops>) {
    let colors = vec![KEEP; shape.colors.iter().flatten().count()];
    let resolves = vec![KEEP; shape.color_resolves.iter().flatten().count()];
    (colors, resolves)
}

/// The most passes a device keeps (the module docs): far more than a program's passes take,
/// which a few shapes, each begun with a few operations and layouts, make.
const KEPT_PASSES: usize = 256;

/// The render passes a device keeps ([`create`]), by shape, each with the rest of what it was
/// made of.
#[derive(Default)]
pub(super) struct KeptRenderPasses {
    by_shape: KeyMap<PassShape, Vec<KeptPass>>,
    /// How many it keeps, of every shape.
    count: usize,
}

/// One pass a device keeps: what [`create`] made it of beside its shape, and the pass.
struct KeptPass {
    colors: Vec<Ops>,
    resolves: Vec<Ops>,
    rest: PassRest,
    raw: vk::RenderPass,
}

/// What [`create`] makes a pass of beside its shape and its color and resolve operations.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PassRest {
    depth_stencil: Option<Ops>,
    depth_stencil_resolve: Option<Ops>,
    resolving: bool,
    feedback: Feedback,
    depth: vk::ImageLayout,
}

impl PassRest {
    /// A compatible pass's ([`compatible_pass`]): every attachment loaded and stored, in
    /// `GENERAL`, resolving.
    fn compatible(shape: &PassShape) -> PassRest {
        PassRest {
            depth_stencil: shape.depth_stencil.map(|_| KEEP),
            depth_stencil_resolve: shape.depth_stencil_resolve.map(|_| KEEP),
            resolving: true,
            feedback: Feedback::NONE,
            depth: vk::ImageLayout::GENERAL,
        }
    }
}

impl KeptRenderPasses {
    /// The kept compatible pass of `shape`: its operations those of [`kept_operations`].
    fn compatible(&self, shape: &PassShape) -> Option<vk::RenderPass> {
        let rest = PassRest::compatible(shape);
        let kept = |ops: &[Ops], count: usize| ops.len() == count && ops.iter().all(|o| *o == KEEP);
        let colors = shape.colors.iter().flatten().count();
        let resolves = shape.color_resolves.iter().flatten().count();
        let mut found = self.by_shape.get(shape)?.iter();
        let found = found.find(|k| k.rest == rest && kept(&k.colors, colors));
        found.filter(|k| kept(&k.resolves, resolves)).map(|k| k.raw)
    }

    /// The kept pass of `shape` made of `colors`, `resolves` and `rest`.
    fn get(
        &self,
        shape: &PassShape,
        (colors, resolves): (&[Ops], &[Ops]),
        rest: PassRest,
    ) -> Option<vk::RenderPass> {
        let kept = self.by_shape.get(shape)?.iter();
        let mut kept = kept.filter(|k| k.rest == rest && k.colors == colors);
        kept.find(|k| k.resolves == resolves).map(|k| k.raw)
    }

    /// Destroy every kept pass; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        for kept in self.by_shape.values().flatten() {
            // SAFETY: the device is idle, so no pending command uses the pass, and a pipeline
            // made with it does not keep it.
            unsafe { release::object(raw, kept.raw) };
        }
    }
}

/// The instance of `shape` at `at` of the pass `layout` begins, with its load and store
/// operations. Every store action is known (the encoder checked). The pass's last instance
/// resolves, and stores the depth/stencil aspects a resolve after the pass reads
/// ([`later_resolve`]); one before it resolves nothing, its resolve references unused, which
/// Vulkan's render pass compatibility ignores in a pass of one subpass (the module docs), its
/// `feedback` colors in the feedback loop layout and its depth/stencil attachment in `depth`
/// ([`depth_layout`]; the caller records the moves, [`moved_images`]). A pass without attachments
/// begins its device's kept pass, which has no operation (`attachmentless.rs`).
pub(super) fn final_pass(
    device: &Device,
    layout: &PassLayout,
    shape: &PassShape,
    at: Position,
    feedback: Feedback,
    depth: vk::ImageLayout,
) -> Result<RenderPassObject, Error> {
    if let Some(kept) = attachmentless::kept(device, shape)? {
        return Ok(kept);
    }
    let anchor = shape.anchor(device.compile_device().profile());
    let dont_care = DontCare::of(device);
    let colors: Vec<Ops> = layout
        .colors
        .iter()
        .enumerate()
        .filter_map(|(index, color)| Some((index as u32, color.as_ref()?)))
        .map(|(index, color)| match shape.storage.contains(&index) {
            // A color its draws read and write as a storage image: the sample planes' anchor,
            // which the subpass keeps, or one it leaves out.
            true if anchor == Some(index) => ANCHORED,
            true => LEFT_OUT,
            false => color_ops(color, at, dont_care),
        })
        .collect();
    let depth_stencil = layout
        .depth_stencil
        .as_ref()
        .map(|ds| depth_stencil_ops(layout, ds, at, dont_care));
    let resolves = vec![
        RESOLVED;
        layout
            .colors
            .iter()
            .filter(|c| subpass_resolve(c.as_ref()).is_some())
            .count()
    ];
    create(
        device,
        shape,
        &colors,
        depth_stencil,
        &resolves,
        subpass_step(layout).map(step_ops),
        at.last,
        feedback,
        depth,
    )
}

/// The operations of color attachment `color` of the draws' subpass in an instance at `at`, its
/// `DontCare` `dont_care`'s, but the image a texture over a buffer keeps for a pass that does not
/// store it, and the backing of a texture's own image a pass stores `DontCare`, which nothing
/// reads after the pass (`kept_bytes.rs`). An attachment the exact resolve reads after the pass
/// keeps its samples for it, and one whose texels are copied as its resolve its texels.
fn color_ops(color: &ColorTarget, at: Position, dont_care: DontCare) -> Ops {
    let exact = color.resolve.as_ref().is_some_and(|r| r.shader.is_some())
        || color.copied_resolve.is_some();
    let unheld = color.memoryless
        || color
            .stand_in
            .as_ref()
            .is_some_and(|s| matches!(s.cause, Cause::KeptBytes | Cause::KeptTexels));
    Ops {
        load: load_at(color.load, at),
        store: match exact {
            true => vk::AttachmentStoreOp::STORE,
            false => store_at(color.store, at, dont_care.unless(unheld)),
        },
        stencil_load: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store: vk::AttachmentStoreOp::DONT_CARE,
    }
}

/// The operations of `layout`'s depth/stencil attachment `ds` in an instance at `at`: the
/// aspects' own (their `DontCare` `dont_care`'s, but the hidden attachment's), but that the last
/// instance stores an aspect a resolve after the pass reads ([`later_stored`]).
fn depth_stencil_ops(
    layout: &PassLayout,
    ds: &DepthStencilTarget,
    at: Position,
    dont_care: DontCare,
) -> Ops {
    let dont_care = dont_care.unless(ds.hidden);
    let later = later_stored(layout);
    let stored = |aspect, store| match at.last && later.contains(aspect) {
        true => vk::AttachmentStoreOp::STORE,
        false => store,
    };
    let (load, store) = aspect_ops(ds.depth.as_ref(), ds.keep_depth, at, dont_care);
    let (stencil_load, stencil_store) =
        aspect_ops(ds.stencil.as_ref(), ds.keep_stencil, at, dont_care);
    Ops {
        load,
        store: stored(metal2vulkan::vk::ImageAspectFlags::DEPTH, store),
        stencil_load,
        stencil_store: stored(metal2vulkan::vk::ImageAspectFlags::STENCIL, stencil_store),
    }
}

/// The aspects of `layout`'s depth/stencil attachment a resolve after the pass reads, which the
/// last instance stores: a later step's, and both of an attachment whose stencil is picked by
/// depth after the pass (`depth_stencil_resolve.rs`).
fn later_stored(layout: &PassLayout) -> metal2vulkan::vk::ImageAspectFlags {
    later_steps(layout).fold(metal2vulkan::vk::ImageAspectFlags::empty(), |all, step| {
        all | step.aspects
    }) | picked_aspects(layout)
        | patterned_aspects(layout)
}

/// Which attachments of `layout`'s pass, recorded as one render pass instance of `shape`, that
/// instance hands the driver as `DONT_CARE` stores, as [`final_pass`] makes it: Metal's
/// `DontCare` or `MultisampleResolve` of a color attachment of the subpass (not one its draws
/// read and write as storage, nor the sample planes' anchor) that no exact resolve reads after
/// the pass and whose texels are not copied as its resolve, and of a depth or stencil aspect no
/// resolve after the pass reads (`Device::degrade_pass`).
pub(super) fn handed_dont_care(
    device: &Device,
    layout: &PassLayout,
    shape: &PassShape,
) -> HandedDontCare {
    let only = Position {
        first: true,
        last: true,
    };
    let dont_care = DontCare::of(device);
    let colors = (layout.colors.iter().enumerate())
        .filter(|(index, color)| {
            let index = *index as u32;
            color.as_ref().is_some_and(|color| {
                !shape.storage.contains(&index)
                    && color_ops(color, only, dont_care).store == vk::AttachmentStoreOp::DONT_CARE
            })
        })
        .fold(0, |colors, (index, _)| colors | 1 << index);
    let (depth, stencil) = layout.depth_stencil.as_ref().map_or((false, false), |ds| {
        let ops = depth_stencil_ops(layout, ds, only, dont_care);
        let handed = |aspect: Option<&AspectTarget>, store| {
            aspect.is_some() && store == vk::AttachmentStoreOp::DONT_CARE
        };
        (
            handed(ds.depth.as_ref(), ops.store),
            handed(ds.stencil.as_ref(), ops.stencil_store),
        )
    });
    HandedDontCare {
        colors,
        depth,
        stencil,
    }
}

/// The attachments [`handed_dont_care`] finds.
pub(super) struct HandedDontCare {
    /// A bit per color index (each below `MAX_COLOR_ATTACHMENTS`), so a pass's end allocates
    /// nothing for them.
    pub(super) colors: u32,
    pub(super) depth: bool,
    pub(super) stencil: bool,
}

/// The aspects of `layout`'s depth/stencil attachment its resolve drawn after the pass reads
/// (`depth_stencil_resolve.rs`): the depth, which picks the sample, and the stencil where it is
/// drawn; none without such a resolve.
fn picked_aspects(layout: &PassLayout) -> metal2vulkan::vk::ImageAspectFlags {
    use metal2vulkan::vk::ImageAspectFlags;
    let drawn = layout
        .depth_stencil
        .as_ref()
        .and_then(|ds| ds.resolve.as_ref())
        .and_then(|r| r.picked());
    match drawn {
        Some((_, aspects)) => ImageAspectFlags::DEPTH | (aspects & ImageAspectFlags::STENCIL),
        None => ImageAspectFlags::empty(),
    }
}

/// The stencil, where `layout`'s one depth/stencil texture stores and resolves its depth beside
/// its stencil resolved alone (`resolve::layout_patterned`): the last instance stores it too, so
/// a driver that runs the pass on Metal (MoltenVK) runs Metal's valid
/// `StoreAndMultisampleResolve` of both aspects, where a stencil `DONT_CARE` is the form Metal's
/// validation stops and plain Metal answers with a layout pattern of the resolved depth
/// (`runtime_lane::resolves` on MoltenVK, an M3 Max). `MultisampleResolve` leaves the samples
/// undefined, so the stored ones are among its answers. None otherwise.
fn patterned_aspects(layout: &PassLayout) -> metal2vulkan::vk::ImageAspectFlags {
    let patterned = layout.depth_stencil.as_ref().is_some_and(|ds| {
        ds.separate.is_none()
            && ds.depth.zip(ds.stencil).is_some_and(|(depth, stencil)| {
                super::resolve::layout_patterned(depth.store, stencil.store)
            })
    });
    match patterned {
        true => metal2vulkan::vk::ImageAspectFlags::STENCIL,
        false => metal2vulkan::vk::ImageAspectFlags::empty(),
    }
}

/// The operations of `step`'s resolve attachment.
fn step_ops(step: &ResolveStep) -> Ops {
    let (load, store) = resolve_ops(step.depth);
    let (stencil_load, stencil_store) = resolve_ops(step.stencil);
    Ops {
        load,
        store,
        stencil_load,
        stencil_store,
    }
}

/// The render pass and framebuffer that make `step`, a depth/stencil resolve of `layout` past the
/// first ([`later_steps`]), after the pass's instances (the module docs): one subpass of the
/// depth/stencil attachment, loaded and stored, and the step's resolve, over every layer of the
/// pass. Begun after a barrier from the last instance, which stored the samples it reads.
pub(super) fn later_resolve(
    device: &Device,
    layout: &PassLayout,
    step: &ResolveStep,
) -> Result<(RenderPassObject, FramebufferObject), Error> {
    let ds = layout
        .depth_stencil
        .as_ref()
        .ok_or_else(|| internal("a depth/stencil resolve without its attachment"))?;
    let shape = PassShape {
        colors: ColorList::new(),
        depth_stencil: Some(ds.target.format),
        samples: layout.samples,
        color_resolves: ColorList::new(),
        depth_stencil_resolve: Some((step.resolve.attachment.format, step.depth, step.stencil)),
        fetch_inputs: false,
        no_inputs: ColorList::new(),
        storage: ColorList::new(),
        view_mask: 0,
        locations: ColorList::new(),
    };
    let pass = create(
        device,
        &shape,
        &[],
        Some(KEEP),
        &[],
        Some(step_ops(step)),
        true,
        Feedback::NONE,
        vk::ImageLayout::GENERAL,
    )?;
    let usage = metal2vulkan::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    let attachments = [
        attachment_view(&ds.target, usage)?,
        attachment_view(&step.resolve.attachment, usage)?,
    ];
    let framebuffer = made_framebuffer(device, &pass, &attachments, layout.extent, layout.layers)?;
    Ok((pass, framebuffer))
}

/// The render pass and framebuffer in which `vkCmdClearAttachments` clears `target`, a pass's
/// attachment (its depth/stencil attachment where `depth_stencil`), over `extent` from the origin
/// before its pass (`clear_reach.rs`), and the uses they make: one subpass of the attachment
/// alone, over the pass's layers, every aspect loaded and stored. Not a load operation's clear: a
/// device that renders through Metal (MoltenVK) clears a pass without a draw over the whole
/// attachment, as Metal does (lane `clear_reach` on MoltenVK).
pub(super) fn reach_pass(
    device: &Device,
    target: &Target,
    depth_stencil: bool,
    extent: [u32; 2],
) -> Result<(RenderPassObject, FramebufferObject, Vec<Use<ResourceKey>>), Error> {
    let load = vk::AttachmentLoadOp::LOAD;
    let store = vk::AttachmentStoreOp::STORE;
    let shape = PassShape {
        colors: ColorList::new(),
        depth_stencil: None,
        samples: target.samples,
        color_resolves: ColorList::new(),
        depth_stencil_resolve: None,
        fetch_inputs: false,
        no_inputs: ColorList::new(),
        storage: ColorList::new(),
        view_mask: 0,
        locations: ColorList::new(),
    };
    let (pass, usage, stages, read, write) = match depth_stencil {
        false => {
            let shape = PassShape {
                colors: ColorList::of([Some(target.format)])?,
                color_resolves: ColorList::of([None])?,
                ..shape
            };
            let ops = Ops {
                load,
                store,
                stencil_load: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store: vk::AttachmentStoreOp::DONT_CARE,
            };
            (
                create(
                    device,
                    &shape,
                    &[ops],
                    None,
                    &[],
                    None,
                    false,
                    Feedback::NONE,
                    vk::ImageLayout::GENERAL,
                )?,
                metal2vulkan::vk::ImageUsageFlags::COLOR_ATTACHMENT,
                COLOR_STAGES,
                vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
        }
        true => {
            let shape = PassShape {
                depth_stencil: Some(target.format),
                ..shape
            };
            let ops = Ops {
                load,
                store,
                stencil_load: load,
                stencil_store: store,
            };
            (
                create(
                    device,
                    &shape,
                    &[],
                    Some(ops),
                    &[],
                    None,
                    false,
                    Feedback::NONE,
                    vk::ImageLayout::GENERAL,
                )?,
                metal2vulkan::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                DEPTH_STAGES,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
        }
    };
    let framebuffer = made_framebuffer(
        device,
        &pass,
        &[attachment_view(target, usage)?],
        extent,
        target.layers,
    )?;
    let levels = target.level..target.level + 1;
    let layers = target.layer..target.layer + target.layers;
    let uses = vec![
        target.texture.image_use(
            levels.clone(),
            layers.clone(),
            Access::new(stages, read),
            false,
        )?,
        target
            .texture
            .image_use(levels, layers, Access::new(stages, write), true)?,
    ];
    Ok((pass, framebuffer, uses))
}

/// The pass of `shape` with `colors`' operations (one per present color attachment, in index
/// order), `depth_stencil`'s, and the resolve attachments' (`resolves`, one per color index that
/// resolves, then `depth_stencil_resolve`'s), every attachment in `GENERAL` but the `feedback`
/// colors, in `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` throughout, and the depth/stencil attachment,
/// in `depth` throughout: the draws' subpass (the module docs). Made with `vkCreateRenderPass2`
/// (core in Vulkan 1.2, which every runtime device has), whose subpass states the depth/stencil
/// resolve. A color attachment the draws' subpass leaves out resolves nothing there (Vulkan
/// resolves only a referenced attachment), nor does the sample planes' anchor (its samples are
/// undefined there); the pass resolves nothing unless `resolving`. Kept by the device
/// ([`KeptRenderPasses`]), at most [`KEPT_PASSES`] of them: the pass of the same shape,
/// operations and layouts is the same object, made once (its creation reads nothing else but the
/// device's profile, which is fixed); one past them is made of its own, for its holders.
fn create(
    device: &Device,
    shape: &PassShape,
    colors: &[Ops],
    depth_stencil: Option<Ops>,
    resolves: &[Ops],
    depth_stencil_resolve: Option<Ops>,
    resolving: bool,
    feedback: Feedback,
    depth: vk::ImageLayout,
) -> Result<RenderPassObject, Error> {
    let rest = PassRest {
        depth_stencil,
        depth_stencil_resolve,
        resolving,
        feedback,
        depth,
    };
    let kept = &device.shared().render_passes;
    if let Some(raw) = lock(kept).get(shape, (colors, resolves), rest) {
        return Ok(RenderPassObject::kept(device, raw));
    }
    let pass = make_pass(
        device,
        shape,
        colors,
        depth_stencil,
        resolves,
        depth_stencil_resolve,
        resolving,
        feedback,
        depth,
    )?;
    let mut passes = lock(kept);
    // Another encoder's may have come first: then this one is its holders'.
    if passes.count < KEPT_PASSES && passes.get(shape, (colors, resolves), rest).is_none() {
        let raw = pass.into_kept();
        let entry = KeptPass {
            colors: colors.to_vec(),
            resolves: resolves.to_vec(),
            rest,
            raw,
        };
        passes
            .by_shape
            .entry(shape.clone())
            .or_default()
            .push(entry);
        passes.count += 1;
        return Ok(RenderPassObject::kept(device, raw));
    }
    Ok(pass)
}

/// [`create`]'s pass, made now with `vkCreateRenderPass2`, its holders' (counted:
/// `WorkCounts::render_passes_made`).
fn make_pass(
    device: &Device,
    shape: &PassShape,
    colors: &[Ops],
    depth_stencil: Option<Ops>,
    resolves: &[Ops],
    depth_stencil_resolve: Option<Ops>,
    resolving: bool,
    feedback: Feedback,
    depth: vk::ImageLayout,
) -> Result<RenderPassObject, Error> {
    let general = vk::ImageLayout::GENERAL;
    // A feedback color, and a depth/stencil attachment held in `depth`, is in its layout before,
    // during and after the instance: the image barriers around it move it (`moved_barriers`), so
    // the pass itself moves no layout.
    let description =
        |format: metal2vulkan::vk::Format, samples: u32, ops: &Ops, layout: vk::ImageLayout| {
            vk::AttachmentDescription2::default()
                .format(format.into())
                .samples(vk::SampleCountFlags::from_raw(samples))
                .load_op(ops.load)
                .store_op(ops.store)
                .stencil_load_op(ops.stencil_load)
                .stencil_store_op(ops.stencil_store)
                .initial_layout(layout)
                .final_layout(layout)
        };
    let unused = vk::AttachmentReference2::default()
        .attachment(vk::ATTACHMENT_UNUSED)
        .layout(vk::ImageLayout::UNDEFINED);
    let mut attachments = Vec::new();
    let add = |attachments: &mut Vec<vk::AttachmentDescription2<'static>>,
               format: metal2vulkan::vk::Format,
               samples: u32,
               ops: &Ops,
               layout: vk::ImageLayout| {
        let reference = vk::AttachmentReference2::default()
            .attachment(attachments.len() as u32)
            .layout(layout);
        attachments.push(description(format, samples, ops, layout));
        reference
    };
    let anchor = shape.anchor(device.compile_device().profile());
    let mut references = Vec::with_capacity(shape.colors.len());
    let mut ops = colors.iter();
    for (index, format) in shape.colors.iter().enumerate() {
        let Some(format) = format else {
            references.push(unused);
            continue;
        };
        let ops = ops
            .next()
            .ok_or_else(|| internal("a color attachment without its operations"))?;
        let reference = add(
            &mut attachments,
            *format,
            shape.samples,
            ops,
            feedback.layout(index),
        );
        // A storage attachment keeps its description, but no subpass uses it unless it is the
        // sample planes' anchor (the module docs).
        if shape.storage.contains(&(index as u32)) && anchor != Some(index as u32) {
            references.push(unused);
        } else {
            references.push(reference);
        }
    }
    let depth_stencil = match (shape.depth_stencil, depth_stencil) {
        (Some(format), Some(ops)) => {
            Some(add(&mut attachments, format, shape.samples, &ops, depth))
        }
        (None, None) => None,
        _ => {
            return Err(internal(
                "a depth/stencil attachment without its operations",
            ))
        }
    };
    let mut resolve_references = Vec::with_capacity(shape.color_resolves.len());
    let mut ops = resolves.iter();
    for (index, format) in shape.color_resolves.iter().enumerate() {
        let Some(format) = format else {
            resolve_references.push(unused);
            continue;
        };
        let ops = ops
            .next()
            .ok_or_else(|| internal("a resolve attachment without its operations"))?;
        let reference = add(&mut attachments, *format, 1, ops, general);
        // `storage` holds the anchor too.
        if resolving && !shape.storage.contains(&(index as u32)) {
            resolve_references.push(reference);
        } else {
            resolve_references.push(unused);
        }
    }
    if shape.depth_stencil_resolve.is_some() != depth_stencil_resolve.is_some()
        || (depth_stencil.is_none() && depth_stencil_resolve.is_some())
    {
        return Err(internal("a depth/stencil resolve without its operations"));
    }
    let ds_resolve_reference = shape
        .depth_stencil_resolve
        .zip(depth_stencil_resolve)
        .map(|((format, _, _), ops)| add(&mut attachments, format, 1, &ops, general));
    let mut ds_resolve = shape
        .depth_stencil_resolve
        .zip(ds_resolve_reference.as_ref())
        .map(|((_, depth, stencil), reference)| {
            vk::SubpassDescriptionDepthStencilResolve::default()
                .depth_resolve_mode(depth.vk())
                .stencil_resolve_mode(stencil.vk())
                .depth_stencil_resolve_attachment(reference)
        });
    // Every subpass of a multiview pass has its views (`VUID-VkRenderPassCreateInfo2-viewMask-03058`).
    let graphics = || {
        vk::SubpassDescription2::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .view_mask(shape.view_mask)
    };
    // Framebuffer fetch's input attachments: color attachment `i` is input attachment `i`, read in
    // rasterization order where the device has it, else ordered between draws by barriers of the
    // dependency below (`framebuffer_fetch.rs`). The same attachment in `GENERAL` both ways; an
    // input reference names its aspect.
    let inputs: Vec<_> = references
        .iter()
        .enumerate()
        .map(|(index, r)| {
            if shape.no_inputs.contains(&(index as u32)) {
                unused
            } else {
                r.aspect_mask(vk::ImageAspectFlags::COLOR)
            }
        })
        .collect();
    // A routed subpass references at each location the color it names, and resolves it there.
    if !shape.locations.is_empty() {
        if !shape.storage.is_empty() {
            return Err(internal("a routed subpass that leaves its attachments out"));
        }
        let missing = || internal("a location naming a color the pass lacks");
        let mut colors = Vec::with_capacity(shape.locations.len());
        let mut resolves = Vec::with_capacity(shape.locations.len());
        for index in &shape.locations {
            let (color, resolve) = match index {
                None => (unused, unused),
                Some(index) => (
                    *references.get(*index as usize).ok_or_else(missing)?,
                    *resolve_references
                        .get(*index as usize)
                        .ok_or_else(missing)?,
                ),
            };
            colors.push(color);
            resolves.push(resolve);
        }
        references = colors;
        resolve_references = resolves;
    }
    let mut draws = graphics().color_attachments(&references);
    if shape.fetch_inputs {
        draws = draws.input_attachments(&inputs);
        if super::framebuffer_fetch::inputs_in_raster_order(device.compile_device().profile()) {
            draws = draws.flags(
                vk::SubpassDescriptionFlags::RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_EXT,
            );
        }
    }
    if resolve_references
        .iter()
        .any(|r| r.attachment != vk::ATTACHMENT_UNUSED)
    {
        draws = draws.resolve_attachments(&resolve_references);
    }
    if let Some(reference) = &depth_stencil {
        draws = draws.depth_stencil_attachment(reference);
    }
    if let (true, Some(ds_resolve)) = (resolving, ds_resolve.as_mut()) {
        draws = draws.push_next(ds_resolve);
    }
    let own = DRAWS_SELF_DEPENDENCY;
    let (indirect, graphics_stages) = INDIRECT_SELF_DEPENDENCY;
    let dependencies = [
        vk::SubpassDependency2::default()
            .src_subpass(0)
            .dst_subpass(0)
            .src_stage_mask(own.stages)
            .dst_stage_mask(own.stages)
            .src_access_mask(own.src_access)
            .dst_access_mask(own.dst_access)
            .dependency_flags(self_dependency_flags(shape.view_mask)),
        vk::SubpassDependency2::default()
            .src_subpass(0)
            .dst_subpass(0)
            .src_stage_mask(indirect)
            .dst_stage_mask(graphics_stages)
            .dependency_flags(stage_dependency_flags(shape.view_mask)),
        vk::SubpassDependency2::default()
            .src_subpass(0)
            .dst_subpass(0)
            .src_stage_mask(geometry_write_stages(device))
            .dst_stage_mask(GEOMETRY_SELF_DEPENDENCY.stages)
            .src_access_mask(GEOMETRY_SELF_DEPENDENCY.src_access)
            .dst_access_mask(GEOMETRY_SELF_DEPENDENCY.dst_access)
            .dependency_flags(stage_dependency_flags(shape.view_mask)),
    ];
    let subpasses = [draws];
    let info = vk::RenderPassCreateInfo2::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    // SAFETY: `info` and every array and structure it points at are live for the call; the
    // device is Vulkan 1.2 or later (the compile layer's floor).
    let raw = unsafe { crate::creation::render_pass2(device.shared().vk(), &info) }?;
    device.shared().counts.render_pass_made();
    Ok(RenderPassObject {
        device: device.clone(),
        raw,
        owner: Owner::Holders,
    })
}

/// The usage of a feedback color's framebuffer view ([`framebuffer`]).
const FEEDBACK_VIEW: metal2vulkan::vk::ImageUsageFlags =
    metal2vulkan::vk::ImageUsageFlags::from_raw(
        metal2vulkan::vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
            | metal2vulkan::vk::ImageUsageFlags::SAMPLED.as_raw()
            | metal2vulkan::vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT.as_raw(),
    );

/// The view of color attachment `target`: an input attachment too where the pass's draws read
/// it so (`fetch_inputs`, `PassLayout::fetch_inputs`) and its image takes that usage. The
/// framebuffer's, and an input attachment descriptor's.
pub(super) fn color_view(target: &Target, fetch_inputs: bool) -> Result<vk::ImageView, Error> {
    let usage = metal2vulkan::vk::ImageUsageFlags::COLOR_ATTACHMENT;
    if fetch_inputs && takes_input(target)? {
        attachment_view(
            target,
            usage | metal2vulkan::vk::ImageUsageFlags::INPUT_ATTACHMENT,
        )
    } else {
        attachment_view(target, usage)
    }
}

/// Whether `target`'s image takes input attachment usage: every color render target the runtime
/// makes where framebuffer fetch reads input attachments, and a caller's image made with it.
pub(super) fn takes_input(target: &Target) -> Result<bool, Error> {
    let usage = target.texture.image_plan()?.usage;
    Ok(usage.contains(metal2vulkan::vk::ImageUsageFlags::INPUT_ATTACHMENT))
}

/// The storage view of color attachment `target`, which a draw reads and writes in the pixel
/// interlock (`WriteSource::TileAttachment`): its level and layer in its own format. The image
/// has `STORAGE` usage (`render.rs` checked).
pub(super) fn storage_view(target: &Target) -> Result<vk::ImageView, Error> {
    storage_view_in(target, target.format)
}

/// The storage view of `target`'s level and layer in `format`, of its image's view formats: a
/// tile dispatch's scratch in its staging format (`render/tile_dispatch.rs`).
pub(super) fn storage_view_in(
    target: &Target,
    format: metal2vulkan::vk::Format,
) -> Result<vk::ImageView, Error> {
    attachment_view_in(target, format, metal2vulkan::vk::ImageUsageFlags::STORAGE)
}

/// The view of the level and layers an attachment renders: 2D, or a 2D array in a layered pass;
/// every aspect of its format, which a depth/stencil attachment's view must have.
fn attachment_view(
    target: &Target,
    usage: metal2vulkan::vk::ImageUsageFlags,
) -> Result<vk::ImageView, Error> {
    attachment_view_in(target, target.format, usage)
}

/// [`attachment_view`] in `format`.
fn attachment_view_in(
    target: &Target,
    format: metal2vulkan::vk::Format,
    usage: metal2vulkan::vk::ImageUsageFlags,
) -> Result<vk::ImageView, Error> {
    // A 1D image's attachment is a 1D view; a 3D image's planes, a 2D view of them.
    let one_d = target.texture.image_plan()?.image_type == metal2vulkan::vk::ImageType::TYPE_1D;
    let view_type = match (one_d, target.layers > 1) {
        (true, true) => metal2vulkan::vk::ImageViewType::TYPE_1D_ARRAY,
        (true, false) => metal2vulkan::vk::ImageViewType::TYPE_1D,
        (false, true) => metal2vulkan::vk::ImageViewType::TYPE_2D_ARRAY,
        (false, false) => metal2vulkan::vk::ImageViewType::TYPE_2D,
    };
    target.texture.image_view(&VkView {
        view_type,
        format,
        components: metal2vulkan::vk::ComponentMapping::IDENTITY,
        aspect: super::texture::image_aspects(format),
        base_level: target.level,
        level_count: 1,
        base_layer: target.layer,
        layer_count: target.layers,
        usage,
    })
}

/// The layers of a pass's attachments a render pass instance of `views` renders (the module
/// docs), counted from its first: from `base_layer`, as many as its views (the highest one's
/// index plus one) under multiview, or that one layer without it; the pass's every layer for the
/// default.
fn viewed_layers(views: ViewSetup, layers: u32) -> std::ops::Range<u32> {
    if views == ViewSetup::default() {
        return 0..layers;
    }
    let count = (u32::BITS - views.view_mask.leading_zeros()).max(1);
    views.base_layer..views.base_layer.saturating_add(count)
}

/// `target` as a render pass instance of `views` renders it: its layers [`viewed_layers`] says,
/// in a pass of `target.layers`. The framebuffer's attachments, and the input attachments the
/// instance's draws read.
pub(super) fn viewed(target: &Target, views: ViewSetup) -> Target {
    let layers = viewed_layers(views, target.layers);
    Target {
        layer: target.layer + layers.start,
        layers: layers.len() as u32,
        ..target.clone()
    }
}

/// Whether a render pass instance of `views` renders every layer of a pass of `layers`, so that
/// its load and store operations and resolves cover the pass's (`render.rs`).
pub(super) fn covers(views: ViewSetup, layers: u32) -> bool {
    viewed_layers(views, layers) == (0..layers)
}

/// The framebuffer of `layout` for `pass` (its final pass) of an instance of `views`: the color
/// attachments' views in index order, then the depth/stencil attachment's, then the resolve
/// attachments' in the same order, over the render area and the layers the instance renders
/// ([`viewed`]), of the pass's, which every attachment renders (`render_pass.rs` refuses what
/// would make an image of fewer in a layered pass). A multiview instance's framebuffer has one
/// layer, and its attachments' views hold its views (`VUID-VkFramebufferCreateInfo-renderPass-02531`,
/// `-04536`). A `feedback` color's view has the usage its layout needs
/// (`VUID-vkCmdBeginRenderPass-initialLayout-07000`: "either the
/// VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT or VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT usage flags
/// set, and either the VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT or VK_IMAGE_USAGE_SAMPLED_BIT usage
/// flags set", and `-07001`: "the VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT usage flag
/// set"), which its image has (`texture::feedback_candidate`). A color an instance of `shape`
/// leaves out of its subpass for the pixel interlock, but for one whose draws read and write sample
/// planes, has a view of the device's placeholder of its format and sample count, which the
/// framebuffer holds, in place of one of the image its draws write (`storage_placeholder.rs`).
pub(super) fn framebuffer(
    device: &Device,
    pass: &RenderPassObject,
    layout: &PassLayout,
    shape: &PassShape,
    views: ViewSetup,
    feedback: Feedback,
) -> Result<FramebufferObject, Error> {
    framebuffer_over(device, pass, layout, shape, views, feedback, layout.extent)
}

/// [`framebuffer`]'s framebuffer, but over `extent` from the origin, which every attachment holds
/// and which holds the render area: an instance whose `Clear` loads reach past the render area
/// renders over their reach (`render/inline_first.rs`).
pub(super) fn framebuffer_over(
    device: &Device,
    pass: &RenderPassObject,
    layout: &PassLayout,
    shape: &PassShape,
    views: ViewSetup,
    feedback: Feedback,
    extent: [u32; 2],
) -> Result<FramebufferObject, Error> {
    let colors = layout.colors.iter().flatten();
    let resolves = colors.clone().filter_map(|c| subpass_resolve(Some(c)));
    let mut targets = colors
        .clone()
        .map(|c| &c.target)
        .chain(layout.depth_stencil.as_ref().map(|ds| &ds.target))
        .chain(resolves.clone().map(|r| &r.attachment))
        .chain(depth_stencil_steps(layout).map(|step| &step.resolve.attachment));
    if targets.any(|t| t.layers != layout.layers) {
        return Err(internal("an attachment of other layers than its pass"));
    }
    if viewed_layers(views, layout.layers).end > layout.layers {
        return Err(internal("views past their pass's layers"));
    }
    let viewed = |target: &Target| viewed(target, views);
    let anchor = shape.anchor(device.compile_device().profile());
    let mut placeholders = Vec::new();
    let mut attachments = Vec::new();
    for (index, color) in layout.colors.iter().enumerate() {
        let Some(color) = color else {
            continue;
        };
        let target = viewed(&color.target);
        let left_out = shape.storage.contains(&(index as u32))
            && anchor != Some(index as u32)
            && color.planes.is_none();
        if left_out {
            let layers = target.layer - color.target.layer
                ..target.layer - color.target.layer + target.layers;
            // Each its own: no two attachments alias.
            let place = (placeholders.iter())
                .filter(|p: &&Arc<Placeholder>| p.is_of(target.format, target.samples))
                .count();
            let of = (target.format, target.samples, place);
            let (placeholder, view) = device.storage_placeholder(of, extent, layers)?;
            placeholders.push(placeholder);
            attachments.push(view);
            device.shared().counts.storage_placeholder_attachment();
            continue;
        }
        attachments.push(match feedback.has(index) {
            true => attachment_view(&target, FEEDBACK_VIEW)?,
            false => color_view(&target, layout.fetch_inputs)?,
        });
    }
    if let Some(ds) = &layout.depth_stencil {
        attachments.push(attachment_view(
            &viewed(&ds.target),
            metal2vulkan::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?);
    }
    for resolve in resolves {
        attachments.push(attachment_view(
            &viewed(&resolve.attachment),
            metal2vulkan::vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )?);
    }
    if let Some(step) = subpass_step(layout) {
        attachments.push(attachment_view(
            &viewed(&step.resolve.attachment),
            metal2vulkan::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?);
    }
    let layers = if views == ViewSetup::default() {
        layout.layers
    } else {
        1
    };
    let mut made = made_framebuffer(device, pass, &attachments, extent, layers)?;
    made.placeholders = placeholders;
    Ok(made)
}

/// The framebuffer of `pass` with `attachments`, over `extent` and `layers`: each view of one
/// level and of those layers, or of the views of a multiview pass's one layer, at least `extent`
/// large.
fn made_framebuffer(
    device: &Device,
    pass: &RenderPassObject,
    attachments: &[vk::ImageView],
    [width, height]: [u32; 2],
    layers: u32,
) -> Result<FramebufferObject, Error> {
    let info = vk::FramebufferCreateInfo::default()
        .render_pass(pass.raw)
        .attachments(attachments)
        .width(width)
        .height(height)
        .layers(layers);
    // SAFETY: `info` and its views are live for the call; each view is of one level and of the
    // layers the instance renders, inside the pass's (checked by the callers), at least `extent`
    // large; a multiview pass's framebuffer has one layer, and each view holds every view.
    let raw = unsafe { crate::creation::framebuffer(device.shared().vk(), &info) }?;
    Ok(FramebufferObject {
        device: device.clone(),
        raw,
        placeholders: Vec::new(),
    })
}

/// The clear values of a pass's attachments, in its framebuffer's order (`clear_values`): each
/// color's, then the depth/stencil attachment's, held in place rather than allocated at every
/// pass.
#[derive(Clone, Copy)]
pub(super) struct ClearValues {
    values: [vk::ClearValue; MAX_COLOR_ATTACHMENTS + 1],
    len: usize,
}

impl ClearValues {
    fn push(&mut self, value: vk::ClearValue) {
        self.values[self.len] = value;
        self.len += 1;
    }

    /// The values, in the framebuffer's order.
    pub(super) fn as_slice(&self) -> &[vk::ClearValue] {
        &self.values[..self.len]
    }
}

/// The clear values of `layout`'s attachments, in the framebuffer's order (ignored where the load
/// operation is not `CLEAR`).
pub(super) fn clear_values(layout: &PassLayout) -> ClearValues {
    let mut values = ClearValues {
        values: [vk::ClearValue::default(); MAX_COLOR_ATTACHMENTS + 1],
        len: 0,
    };
    for color in layout.colors.iter().flatten() {
        values.push(vk::ClearValue { color: color.clear });
    }
    if let Some(ds) = &layout.depth_stencil {
        let mut value = vk::ClearDepthStencilValue::default();
        for aspect in [ds.depth.as_ref(), ds.stencil.as_ref()]
            .into_iter()
            .flatten()
        {
            match aspect.clear {
                AspectClear::Depth(depth) => value.depth = depth,
                AspectClear::Stencil(stencil) => value.stencil = stencil,
            }
        }
        values.push(vk::ClearValue {
            depth_stencil: value,
        });
    }
    values
}

/// The stages a color attachment's load, writes and store run in.
const COLOR_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;

/// The stages a depth/stencil attachment's load (early tests), tests and store (late tests) run
/// in.
const DEPTH_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
);

/// What the pass of `layout` uses, for the tracker: each attachment's level and layers, read (its
/// load) and written (its draws, clear and store), in the stages its operations run in.
pub(super) fn attachment_uses(layout: &PassLayout) -> Result<Vec<Use<ResourceKey>>, Error> {
    attachment_uses_but(layout, |_| false)
}

/// The pass's uses as [`attachment_uses`] gives them, but those of the color attachments `left`
/// names by index (their resolves' kept).
pub(super) fn attachment_uses_but(
    layout: &PassLayout,
    left: impl Fn(usize) -> bool,
) -> Result<Vec<Use<ResourceKey>>, Error> {
    let kept = || {
        let colors = layout.colors.iter().enumerate();
        colors
            .filter(|&(index, _)| !left(index))
            .filter_map(|(_, color)| color.as_ref())
    };
    // Room for every use at once, a read and a write of each attachment and resolve, so the list
    // is allocated once.
    let resolves = layout
        .colors
        .iter()
        .flatten()
        .filter(|c| c.resolve.is_some());
    let targets = kept().count()
        + usize::from(layout.depth_stencil.is_some())
        + resolves.count()
        + depth_stencil_steps(layout).count();
    let mut uses = Vec::with_capacity(2 * targets);
    for color in kept() {
        add_target_uses(&mut uses, &color.target, false)?;
    }
    if let Some(ds) = &layout.depth_stencil {
        add_target_uses(&mut uses, &ds.target, true)?;
    }
    let mut add =
        |target: &Target, stages, read, write| add_uses(&mut uses, target, stages, (read, write));
    // Resolves run in the color output stage; a depth/stencil one reads its source and writes
    // its target as both kinds of attachment, whichever the device names.
    for resolve in layout
        .colors
        .iter()
        .flatten()
        .filter_map(|c| c.resolve.as_ref())
    {
        add(
            &resolve.attachment,
            COLOR_STAGES,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )?;
    }
    for step in depth_stencil_steps(layout) {
        add(
            &step.resolve.attachment,
            ATTACHMENT_STAGES,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )?;
    }
    Ok(uses)
}

/// Whether one of `uses`, what a command's functions bind or reach, reads or writes memory an
/// attachment of the pass of `layout` renders (its resolve textures included): a use of the
/// attachment other than as an attachment, inside its render pass instance, which Vulkan leaves
/// undefined (`texture/attached_reads.rs`). A use that reads a copy or a snapshot names another
/// resource. An attachment's memory is the key and range [`attachment_uses`] gives both its read
/// and its write, taken here one attachment at a time, nothing collected: a draw against a layout
/// of no version (a stand-in's) asks so, and a render encoder's draws ask its [`AttachedMemo`].
pub(super) fn shares_attachment(
    layout: &PassLayout,
    uses: &[Use<ResourceKey>],
) -> Result<bool, Error> {
    if !any_access(uses) {
        return Ok(false);
    }
    for target in attachment_targets(layout) {
        let a = target_use(target)?;
        if meets(&a, uses) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether one of `uses` reads or writes.
fn any_access(uses: &[Use<ResourceKey>]) -> bool {
    uses.iter()
        .any(|used| used.read != Access::NONE || used.write != Access::NONE)
}

/// Whether one of `uses` that reads or writes meets `attached`'s memory.
fn meets(attached: &Use<ResourceKey>, uses: &[Use<ResourceKey>]) -> bool {
    uses.iter()
        .filter(|used| used.read != Access::NONE || used.write != Access::NONE)
        .any(|used| attached.key == used.key && attached.range.overlaps(&used.range))
}

/// The targets of `layout`'s attachments, in the order [`shares_attachment`] takes them: its
/// colors, its depth and stencil, its colors' resolves and its depth and stencil resolves.
fn attachment_targets(layout: &PassLayout) -> impl Iterator<Item = &Target> {
    let colors = layout.colors.iter().flatten();
    colors
        .clone()
        .map(|color| &color.target)
        .chain(layout.depth_stencil.as_ref().map(|ds| &ds.target))
        .chain(colors.filter_map(|color| color.resolve.as_ref().map(|r| &r.attachment)))
        .chain(depth_stencil_steps(layout).map(|step| &step.resolve.attachment))
}

/// The memory `target` renders, as a use of no access.
fn target_use(target: &Target) -> Result<Use<ResourceKey>, Error> {
    let levels = target.level..target.level + 1;
    let layers = target.layer..target.layer + target.layers;
    target
        .texture
        .image_use(levels, layers, Access::NONE, false)
}

/// [`shares_attachment`]'s answers for a render encoder's layout, its attachments' memory worked
/// out once for each version of the layout (`versioned.rs`), whose targets change only with it, and
/// kept: every draw asks while the notice is pending (`EncoderCore::degrade_attachment_reads`), and
/// working each attachment's use out again (its texture's image and aspects) was about a
/// twentieth of a draw rebound's time. Checked against the layout's answer in builds with debug
/// assertions.
#[derive(Default)]
pub(super) struct AttachedMemo {
    /// The version of the layout the uses were worked out for.
    version: Option<u64>,
    /// Its targets' uses, in [`attachment_targets`]' order, up to the first whose use is
    /// refused, and that refusal.
    uses: Vec<Use<ResourceKey>>,
    refused: Option<Error>,
}

impl AttachedMemo {
    /// [`shares_attachment`] of `layout`, at `version`, and `uses`: the same answer, the same
    /// refusal where it would meet one.
    pub(super) fn shares(
        &mut self,
        (layout, version): (&PassLayout, u64),
        uses: &[Use<ResourceKey>],
    ) -> Result<bool, Error> {
        if !any_access(uses) {
            return Ok(false);
        }
        if self.version != Some(version) {
            self.uses.clear();
            self.refused = None;
            for target in attachment_targets(layout) {
                match target_use(target) {
                    Ok(used) => self.uses.push(used),
                    Err(refused) => {
                        self.refused = Some(refused);
                        break;
                    }
                }
            }
            self.version = Some(version);
        }
        let answer = match self.uses.iter().any(|attached| meets(attached, uses)) {
            true => Ok(true),
            false => match &self.refused {
                Some(refused) => Err(refused.clone()),
                None => Ok(false),
            },
        };
        // Checked in builds with debug assertions: the answer the layout gives.
        #[cfg(debug_assertions)]
        assert_eq!(
            answer,
            shares_attachment(layout, uses),
            "an attachment memo answered otherwise than its layout"
        );
        answer
    }
}

/// The keys of the buffers whose bytes the pass's attachments lie in (their resolve textures
/// included): the buffer a texture is over, or the bytes of the heap it, or its buffer, lies in.
/// A rebind or a detach of one swaps the images of the textures over it (`rebind.rs`,
/// `detach.rs`).
pub(super) fn attachment_bytes(layout: &PassLayout) -> impl Iterator<Item = ResourceKey> + '_ {
    let colors = layout.colors.iter().flatten();
    let targets = colors
        .clone()
        .map(|color| &color.target)
        .chain(layout.depth_stencil.as_ref().map(|ds| &ds.target))
        .chain(colors.filter_map(|color| color.resolve.as_ref().map(|r| &r.attachment)));
    targets.filter_map(|target| match target.texture.memory() {
        TextureMemory::Heap(heap) => heap.bytes().map(|own| own.identity().key),
        TextureMemory::Buffer(buffer) => Some(buffer.identity().key),
        TextureMemory::Own(_) => None,
    })
}

/// What a pass's attachment `target` uses, as [`attachment_uses`] counts it: a color
/// attachment's, or a depth/stencil attachment's (`depth_stencil`).
pub(super) fn target_uses(
    target: &Target,
    depth_stencil: bool,
) -> Result<Vec<Use<ResourceKey>>, Error> {
    let mut uses = Vec::with_capacity(2);
    add_target_uses(&mut uses, target, depth_stencil)?;
    Ok(uses)
}

/// Add to `uses` what [`target_uses`] gives.
fn add_target_uses(
    uses: &mut Vec<Use<ResourceKey>>,
    target: &Target,
    depth_stencil: bool,
) -> Result<(), Error> {
    match depth_stencil {
        false => add_uses(
            uses,
            target,
            COLOR_STAGES,
            (
                vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
        ),
        true => add_uses(
            uses,
            target,
            DEPTH_STAGES,
            (
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
        ),
    }
}

/// Add to `uses` `target`'s level and layers, read (`read`) and written (`write`) in `stages`.
fn add_uses(
    uses: &mut Vec<Use<ResourceKey>>,
    target: &Target,
    stages: vk::PipelineStageFlags2,
    (read, write): (vk::AccessFlags2, vk::AccessFlags2),
) -> Result<(), Error> {
    let levels = target.level..target.level + 1;
    let layers = target.layer..target.layer + target.layers;
    uses.push(target.texture.image_use(
        levels.clone(),
        layers.clone(),
        Access::new(stages, read),
        false,
    )?);
    uses.push(
        target
            .texture
            .image_use(levels, layers, Access::new(stages, write), true)?,
    );
    Ok(())
}

/// The stages of every attachment use of a pass.
pub(super) const ATTACHMENT_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::from_raw(COLOR_STAGES.as_raw() | DEPTH_STAGES.as_raw());
