//! A texture over host pages in place (`in_place.rs` has which, and the device's check): the
//! linear image its texels are, bound where they lie, made again over a rebind's new pages.
//!
//! - **Over pages imported by their pointer** (a no-copy buffer's, a no-copy heap's buffer's, or a
//!   buffer of the runtime's own in fresh pages): an image made for host allocations
//!   (`VK_EXT_external_memory_host`, as `VUID-vkBindImageMemory-memory-02728` has an image bound to
//!   imported memory made), bound in the pages' import at the texels' offset, its rows at the
//!   texture's pitch: the driver's linear image where it lays the rows out so, its subresource at
//!   offset 0; else, where the device enables `VK_EXT_image_drm_format_modifier`, a
//!   `DRM_FORMAT_MOD_LINEAR` image whose one plane the runtime lays out at those pitches, made for
//!   host allocations too (counted: `WorkCounts::pitched_host_images`); else none (the shadow). An
//!   import a device takes for such an image only as a dedicated allocation is none: the image is
//!   bound at an offset of the buffer's import.
//! - **Over pages imported as dma-bufs** (`dma_buf.rs`): the `DRM_FORMAT_MOD_LINEAR` image at the
//!   texture's pitches bound to a dedicated import of the one dma-buf the pages' source makes of
//!   its own pages ([`Device::dma_buf_image`]), as a no-copy heap's texture there takes, its first
//!   move an acquire from `VK_QUEUE_FAMILY_FOREIGN_EXT`.
//! - **Its rows.** Rows (and slices) that lie apart are held: the bytes between them are in the
//!   image's subresource, and its first move is the one operation that may touch them, which the
//!   device's check covers. Rows that share bytes (an IOSurface plane's GPU rows) are not: no image
//!   holds a texel twice. The image's memory may run past the buffer's bytes into the rest of its
//!   placement (its last page's bytes past the buffer), as Vulkan binds it within the memory: its
//!   texels never reach them, and the check's bytes after its image are such bytes.
//! - **Its usage** is the texture's declared plan's: one a shader writes (`ShaderWrite` or
//!   `ShaderAtomic`) has storage usage, through which its kernels write its texels in place, as
//!   the device's check of storage writes covers ([`Texture::store_in_place`]); a write beyond it
//!   binds the texture's usage shadow (`texture/usage_shadow.rs`), and a render pass into one made
//!   without `RenderTarget` renders through an attachment stand-in (`attachment_stand_in.rs`),
//!   since the image has no attachment usage; their copies into it touch its texels alone. One that
//!   renders also takes the feedback loop usage a draw takes that reads the texture its own pass
//!   renders through a snapshot (`texture::feedback_candidate`, `render/snapshot.rs`) where the
//!   device makes the image with it, as an aliasing image over a buffer of the runtime's does,
//!   and with it the storage usage the snapshot's fetch of the attachment in the pixel interlock
//!   reads through (`framebuffer_fetch::storage_candidate`, which may list a view in another
//!   format), where the device's check passed for a shader's writes over the pages' class too and
//!   the texel is one its kernel stores (with that usage a shader writes the texels in place, as
//!   one declared to does): the first of those plans the device makes the image in place with,
//!   else the declared plan's image.
//! - **A rebind** (`rebind.rs`) makes it again over the new pages, its texels as far from their
//!   first byte as from the old pages', as a no-copy heap's images are
//!   ([`Device::rebound_in_place_images`]); a detach makes it present a stand-in, as it does every
//!   image bound to the pages (`detach.rs`).

use super::attached_reads::feedback_candidate;
use super::{
    first_color_subresource, pages_hold, FirstMove, Image, ImageObjects, ImageSwap, KeptResolves,
    KeyMap, LinearMade, LinearRefusal, ObjectsToken, Owner, Raw, Remade, Storage, Texture,
};
use crate::binding_view::VkView;
use crate::buffer::{Buffer, InPlaceWatch, ReadApart};
use crate::buffer_texture::Texels;
use crate::device::lock;
use crate::error::{vulkan, Error, RuntimeGap};
use crate::framebuffer_fetch::storage_candidate;
use crate::image_plan::{ImageFacts, ImagePlan};
use crate::in_place::{Access, Padding, PagesClass, Refusal, RowEnds};
use crate::linear_plan::{check_linear_limits, linear_lack, BufferLayout, LinearLack};
use crate::shadow_copy_back::WordView;
use crate::shadow_copy_in::CopyIn;
use crate::texture_descriptor::TextureDescriptor;
use crate::{release, Device};
use ash::vk;
use metal2vulkan::bind::ShadowTexelView;
use metal2vulkan::error::LinearNeed;
use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// An image in place made over a texture's texels, or why none is: the refusal's kind and words.
pub(super) type InPlace = Result<(Arc<Image>, bool), (Refusal, String)>;

/// An unbound image in place, or why none is.
type Unbound = Result<(LinearMade, bool), (Refusal, String)>;

/// An image in place made and checked before any memory is bound to it
/// ([`Device::unplaced`]), which [`Device::in_place_image`] binds and
/// [`Device::in_place_image_made`] releases.
enum Unplaced {
    /// Over pages imported by their pointer: the image made for host allocations, whether it is
    /// the `DRM_FORMAT_MOD_LINEAR` one, and the buffer's memory and the offset in it where its
    /// requirements were checked to take it (`placed`).
    Host {
        made: LinearMade,
        pitched: bool,
        memory: vk::DeviceMemory,
        at: u64,
    },
    /// Over pages imported as dma-bufs: the image over them, checked against their pages, before
    /// the caller's source is asked; with the pages, their imports and memory type, and the host
    /// address of the texels' first byte.
    #[cfg(unix)]
    DmaBuf {
        unbound: crate::dma_buf::Unbound,
        pages: Arc<crate::buffer::HostPages>,
        imports: Arc<crate::dma_buf::DmaBufPages>,
        type_index: u32,
        at: u64,
    },
}

/// Why no image in place holds texels laid out as `layout`, where their rows share bytes (an
/// IOSurface plane's GPU rows): no image holds a texel twice.
fn shared_rows(layout: &BufferLayout) -> Option<(Refusal, String)> {
    let shared_bytes = layout.shared_bytes();
    (shared_bytes != 0).then(|| {
        (
            Refusal::Rows,
            format!(
                "its rows lie {} bytes apart, {} bytes of texels each, so consecutive rows share \
                 {shared_bytes} bytes: no image holds a texel twice",
                layout.row_pitch, layout.row_bytes
            ),
        )
    })
}

/// The refusal of an image in place over dma-buf pages of `kind` (`dma_buf.rs`).
#[cfg(unix)]
fn dma_buf_refusal(kind: crate::dma_buf::DmaBufRefusal) -> Refusal {
    use crate::dma_buf::DmaBufRefusal as D;
    match kind {
        D::Pitch => Refusal::Pitch,
        D::Source => Refusal::Source,
        D::Image | D::Pages | D::Import => Refusal::DmaBuf,
    }
}

impl Texture {
    /// Make the first move of the texture's image in place now, by an internal submission the
    /// call waits for (`Image::settle`): the device's check's round (`in_place_check.rs`). `None`
    /// where the texture is no image aliasing its bytes.
    pub(crate) fn settle_in_place(&self) -> Option<Result<(), Error>> {
        match self.storage() {
            Storage::Alias { image, .. } => Some(image.settle()),
            _ => None,
        }
    }

    /// Move the texture's image in place from `GENERAL` into `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT`
    /// and back, as a render pass instance whose draws read it through a snapshot holds it
    /// (`render_target::moved_barriers`), by an internal submission the call waits for, after its
    /// first move: the device's check's round of an image that renders (`in_place_check.rs`).
    /// `None` where the texture is no image aliasing its bytes, or its image has no feedback loop
    /// usage.
    pub(crate) fn settle_feedback_moves(&self) -> Option<Result<(), Error>> {
        let Storage::Alias { image, .. } = self.storage() else {
            return None;
        };
        let plan = image.plan();
        let feedback = metal2vulkan::vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        if !plan.usage.contains(feedback) {
            return None;
        }
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: plan.levels,
            base_array_layer: 0,
            layer_count: plan.layers,
        };
        let family = image.device.shared().queue_flags_of(0);
        let raw = image.raw.get();
        Some(image.submit_copy(|commands, command_buffer| {
            // SAFETY: the internal command buffer of the device's first `VkQueue` is recording,
            // outside a render pass; the image is live until the submission, waited for,
            // completed, made with feedback loop usage, and rests in `GENERAL` after its first
            // move, which the submission makes first.
            unsafe {
                crate::render_target::feedback_moves(commands, command_buffer, raw, range, family)
            };
        }))
    }

    /// Draw zero into the texture's image in place through a scissor of its last column, in a
    /// render pass that loads and stores the image (`check_draw.rs`), by an internal submission
    /// the call waits for, after its first move: the device's check's round of an image that
    /// renders (`in_place_check.rs`), which checks that a draw leaves every byte bound to the
    /// image beside its texels alone. `None` where the texture is no image aliasing its bytes, or
    /// its image has no color attachment usage.
    pub(crate) fn draw_in_place_column(
        &self,
        draw: &mut crate::check_draw::CheckDraw,
    ) -> Option<Result<(), Error>> {
        let Storage::Alias { image, .. } = self.storage() else {
            return None;
        };
        let plan = image.plan();
        let attachment = metal2vulkan::vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if !plan.usage.contains(attachment) {
            return None;
        }
        let [width, height, _] = plan.extent;
        let target = match draw.target(image.raw.get(), plan.format.into(), [width, height]) {
            Ok(target) => target,
            Err(error) => return Some(Err(error)),
        };
        Some(image.submit_copy(|commands, command_buffer| {
            // SAFETY: the internal command buffer of the device's first `VkQueue` is recording,
            // outside a render pass; the image is live until the submission, waited for,
            // completed, with color attachment usage, and rests in `GENERAL` after its first move,
            // which the submission makes first; the target's objects live until it is dropped,
            // after the wait.
            unsafe { target.record(commands, command_buffer) };
        }))
    }

    /// For the device's check of storage writes (`in_place_check.rs`): where the texture is an
    /// image in place of storage usage, a kernel's storage writes of its `width` by `height`
    /// texels, each `texel`'s words, from the words of `source` laid row after row (one dispatch
    /// of the crate-owned copy-in kernel, `shadow_copy_in.rs`, which stores them in `kept` too),
    /// by an internal submission the call waits for, which makes the image's first move first and
    /// then makes the writes visible to the host. `None` where the texture is no such image, or
    /// the device makes no such dispatch.
    pub(crate) fn store_in_place(
        &self,
        (source, kept): (&Buffer, &Buffer),
        texel: ShadowTexelView,
        (width, height): (u32, u32),
    ) -> Option<Result<(), Error>> {
        let Storage::Alias { image, .. } = self.storage() else {
            return None;
        };
        use metal2vulkan::vk::{
            ComponentMapping, ImageAspectFlags, ImageUsageFlags, ImageViewType,
        };
        let plan = image.plan();
        if !plan.usage.contains(ImageUsageFlags::STORAGE) {
            return None;
        }
        let view = VkView {
            view_type: ImageViewType::TYPE_2D,
            format: plan.format,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        };
        let view = match image.view(&view) {
            Ok(view) => view,
            Err(error) => return Some(Err(error)),
        };
        let words = WordView {
            view,
            texel,
            stores: true,
        };
        let (buffer, at) = source.at_for(0, ReadApart::Binding);
        let region = vk::BufferImageCopy {
            buffer_offset: at,
            buffer_row_length: width,
            buffer_image_height: height,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        let copy = CopyIn::new(
            &image.device,
            words,
            buffer,
            &region,
            kept.at_for(0, ReadApart::Binding),
        )?;
        Some(image.submit_copy(|commands, command_buffer| {
            let to_host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            // SAFETY: the internal command buffer of the device's first `VkQueue` is recording,
            // outside a render pass, after the image's first move to `GENERAL`, which the
            // submission makes first; the image, its view and both buffers are live until the
            // submission, waited for, completed.
            unsafe {
                copy.record(commands, command_buffer);
                commands.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[to_host],
                    &[],
                    &[],
                );
            }
        }))
    }
}

impl Device {
    /// The image in place of a texture of `plan` (its declared usage's, with the feedback loop
    /// usage where the device makes it so, and the storage usage too where `storable`: the module
    /// docs) made with `descriptor` over `texels`, host pages of `class`, with whether it is a
    /// `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch made for host allocations; why not where the
    /// device makes none, the declared plan's refusal (the module docs). Each plan is made and
    /// checked unbound ([`unplaced`](Self::unplaced)), as
    /// [`in_place_image_made`](Self::in_place_image_made) asks it, then bound
    /// ([`place`](Self::place)). Over a no-copy buffer's pages (or a no-copy heap's) it is noted
    /// for a rebind to make again. Its first move is noted, for the device's next submission that
    /// takes it (`texture/first_move.rs`).
    pub(super) fn in_place_image(
        &self,
        plan: &ImagePlan,
        descriptor: &TextureDescriptor,
        (texels, class, storable): (&Texels, PagesClass, bool),
        facts: &ImageFacts<'_>,
    ) -> Result<InPlace, Error> {
        let layout = texels.layout;
        if let Some(refused) = shared_rows(&layout) {
            return Ok(Err(refused));
        }
        let make = |plan: &ImagePlan| match self.unplaced(plan, texels, class, facts)? {
            Ok(unplaced) => self.place(unplaced, plan, descriptor, texels, class),
            Err(refused) => Ok(Err(refused)),
        };
        let mut made = None;
        for wider in &self.wider_in_place_plans(plan, storable) {
            if let Ok(image) = make(wider)? {
                made = Some(image);
                break;
            }
        }
        let (image, pitched) = match made {
            Some(made) => made,
            None => match make(plan)? {
                Ok(made) => made,
                Err(refused) => return Ok(Err(refused)),
            },
        };
        // Over a no-copy buffer's pages, noted for a rebind.
        if class != PagesClass::Fresh {
            if let Some(base) = texels.buffer.host_pages().map(|pages| pages.memory().1) {
                let (_, buffer_offset, _) = texels.buffer.memory();
                let image_any: Arc<dyn Any + Send + Sync> = image.clone();
                texels.buffer.watch_in_place(InPlaceWatch {
                    image: Arc::downgrade(&image_any),
                    from_base: buffer_offset + layout.offset - base,
                    layout,
                    padding_owned: texels.padding_owned(),
                });
            }
        }
        Ok(Ok((image, pitched)))
    }

    /// Whether [`in_place_image`](Self::in_place_image) makes an image in place of the same
    /// arguments, its declared plan's refusal where it makes none: the same steps up to the bind,
    /// each image it makes released unbound. The declared plan is asked first, so where it is made
    /// in place the answer costs its one or two images (the driver's linear image, and the
    /// `DRM_FORMAT_MOD_LINEAR` one where the driver lays the first out otherwise); the wider plans
    /// are asked only where it is refused, as the making takes the first plan made of them all.
    /// Over pages imported as dma-bufs it asks no source of the caller's: the source's answer and
    /// the driver's import of its dma-buf are known only at the bind.
    pub(crate) fn in_place_image_made(
        &self,
        plan: &ImagePlan,
        (texels, class, storable): (&Texels, PagesClass, bool),
        facts: &ImageFacts<'_>,
    ) -> Result<Result<(), (Refusal, String)>, Error> {
        if let Some(refused) = shared_rows(&texels.layout) {
            return Ok(Err(refused));
        }
        let refused = match self.unplaced(plan, texels, class, facts)? {
            Ok(unplaced) => {
                self.release_unplaced(unplaced);
                return Ok(Ok(()));
            }
            Err(refused) => refused,
        };
        for wider in &self.wider_in_place_plans(plan, storable) {
            if let Ok(unplaced) = self.unplaced(wider, texels, class, facts)? {
                self.release_unplaced(unplaced);
                return Ok(Ok(()));
            }
        }
        Ok(Err(refused))
    }

    /// The plans wider than `plan` an image in place is tried with before it, widest first: the
    /// feedback loop usage where the device makes it, and with it the storage usage where
    /// `storable` (the module docs).
    fn wider_in_place_plans(&self, plan: &ImagePlan, storable: bool) -> Vec<ImagePlan> {
        let profile = self.compile_device().profile();
        let feedback = feedback_candidate(plan, self);
        let storage = feedback
            .as_ref()
            .filter(|_| storable)
            .and_then(|feedback| storage_candidate(feedback, profile));
        storage.into_iter().chain(feedback).collect()
    }

    /// The image in place of `plan` over `texels`, host pages of `class`, made and checked before
    /// any memory is bound to it; why not. Over pages imported by their pointer, the image made
    /// for host allocations ([`in_place_unbound`](Self::in_place_unbound)) whose requirements
    /// take its place in the buffer's memory ([`placed`]); over pages imported as dma-bufs, the
    /// `DRM_FORMAT_MOD_LINEAR` image over them, checked against their pages, before the caller's
    /// source is asked (`dma_buf.rs`).
    fn unplaced(
        &self,
        plan: &ImagePlan,
        texels: &Texels,
        class: PagesClass,
        facts: &ImageFacts<'_>,
    ) -> Result<Result<Unplaced, (Refusal, String)>, Error> {
        match class {
            PagesClass::DmaBuf => self.dma_buf_unplaced(plan, texels, facts),
            PagesClass::Pointer | PagesClass::Fresh => self.host_unplaced(plan, texels, facts),
        }
    }

    /// [`unplaced`](Self::unplaced) over pages imported by their pointer.
    fn host_unplaced(
        &self,
        plan: &ImagePlan,
        texels: &Texels,
        facts: &ImageFacts<'_>,
    ) -> Result<Result<Unplaced, (Refusal, String)>, Error> {
        let layout = texels.layout;
        let (memory, buffer_offset, type_index) = texels.buffer.memory();
        let at = buffer_offset + layout.offset;
        let room = texels.buffer.memory_room().saturating_sub(layout.offset);
        let (made, pitched) = match self.in_place_unbound(plan, &layout, facts)? {
            Ok(made) => made,
            Err(refused) => return Ok(Err(refused)),
        };
        if let Err(refused) = placed(&made.requirements, at, room, type_index) {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(self.shared().vk(), made.raw) };
            return Ok(Err(refused));
        }
        Ok(Ok(Unplaced::Host {
            made,
            pitched,
            memory,
            at,
        }))
    }

    /// [`unplaced`](Self::unplaced) over pages imported as dma-bufs.
    #[cfg(unix)]
    fn dma_buf_unplaced(
        &self,
        plan: &ImagePlan,
        texels: &Texels,
        facts: &ImageFacts<'_>,
    ) -> Result<Result<Unplaced, (Refusal, String)>, Error> {
        let layout = texels.layout;
        let Some(pages) = texels.buffer.host_pages() else {
            return Ok(Err((
                Refusal::Detached,
                "the buffer's bytes lie in no pages".to_owned(),
            )));
        };
        let Some((imports, start, type_index)) = pages.dma_buf() else {
            return Ok(Err((
                Refusal::Image,
                "the buffer's pages keep no dma-buf imports".to_owned(),
            )));
        };
        let base = pages.memory().1;
        let (_, buffer_offset, _) = texels.buffer.memory();
        let at = start + (buffer_offset + layout.offset - base);
        let dma_texels = crate::dma_buf::DmaBufTexels {
            pages: &imports,
            type_index,
            at,
            row_pitch: layout.row_pitch,
            slice_pitch: layout.slice_pitch,
        };
        match self.dma_buf_image_unbound(plan, facts, &dma_texels)? {
            Ok(unbound) => Ok(Ok(Unplaced::DmaBuf {
                unbound,
                pages,
                imports,
                type_index,
                at,
            })),
            Err((kind, why)) => Ok(Err((dma_buf_refusal(kind), why))),
        }
    }

    /// No pages are imported as dma-bufs on this system.
    #[cfg(not(unix))]
    fn dma_buf_unplaced(
        &self,
        _plan: &ImagePlan,
        _texels: &Texels,
        _facts: &ImageFacts<'_>,
    ) -> Result<Result<Unplaced, (Refusal, String)>, Error> {
        Ok(Err((
            Refusal::Image,
            "the system imports no dma-bufs".to_owned(),
        )))
    }

    /// `unplaced`, an image in place of `plan` over `texels` (host pages of `class`), bound: over
    /// pages imported by their pointer, in the buffer's memory where it was checked to lie, its
    /// first move noted; over pages imported as dma-bufs, to a dedicated import of the dma-buf the
    /// caller's source makes of its pages (called with no lock of the runtime's held), which may
    /// still refuse it. With whether it is a `DRM_FORMAT_MOD_LINEAR` image made for host
    /// allocations.
    fn place(
        &self,
        unplaced: Unplaced,
        plan: &ImagePlan,
        descriptor: &TextureDescriptor,
        texels: &Texels,
        class: PagesClass,
    ) -> Result<InPlace, Error> {
        match unplaced {
            Unplaced::Host {
                made,
                pitched,
                memory,
                at,
            } => {
                let raw = made.raw;
                let raw_device = self.shared().vk();
                // SAFETY: the image is live and unbound, made for host allocations; the memory is
                // the buffer's import of host pages (or its fresh pages'), which the image holds,
                // in a type the image takes, at an offset that meets its alignment, its size inside
                // the memory (`placed`).
                if let Err(error) =
                    unsafe { crate::creation::bind_image(raw_device, raw, memory, at) }
                {
                    // SAFETY: the image is live and unbound, and no command uses it.
                    unsafe { release::object(raw_device, raw) };
                    return Err(vulkan(error));
                }
                let image = self.in_place_of(Raw::new(raw), plan.clone(), descriptor, texels);
                // Over a no-copy buffer's pages, its move notes them (a rebind may swap the image
                // out of them); fresh pages are the buffer's for good.
                let pages = match class {
                    PagesClass::Fresh => None,
                    _ => texels.buffer.host_pages(),
                };
                let first = FirstMove::new(
                    raw,
                    image.plan().format,
                    made.initial,
                    pages.as_ref().and_then(pages_hold),
                );
                *lock(&image.first_move) = Some(first);
                Ok(Ok((image.shared(), pitched)))
            }
            #[cfg(unix)]
            Unplaced::DmaBuf {
                unbound,
                pages,
                imports,
                type_index,
                at,
            } => {
                let layout = texels.layout;
                let dma_texels = crate::dma_buf::DmaBufTexels {
                    pages: &imports,
                    type_index,
                    at,
                    row_pitch: layout.row_pitch,
                    slice_pitch: layout.slice_pitch,
                };
                let made = match self.bind_dma_buf_image(unbound, &dma_texels)? {
                    Ok(made) => made,
                    Err((kind, why)) => return Ok(Err((dma_buf_refusal(kind), why))),
                };
                let owner = Owner::Buffer {
                    memory: Mutex::new(texels.buffer.memory_hold()),
                };
                let made_with = (texels.options(), descriptor.allow_gpu_optimized_contents);
                let identity = crate::resource::Identity {
                    tracked: texels.tracked(),
                    ..texels.buffer.identity()
                };
                let image = self.dma_buf_texture_image(
                    made,
                    plan.clone(),
                    owner,
                    made_with,
                    identity,
                    &pages,
                );
                Ok(Ok((image.shared(), false)))
            }
        }
    }

    /// Release `unplaced`'s image, which no command uses and no memory is bound to.
    fn release_unplaced(&self, unplaced: Unplaced) {
        match unplaced {
            // SAFETY: the image is live and unbound, and no command uses it.
            Unplaced::Host { made, .. } => unsafe { release::object(self.shared().vk(), made.raw) },
            #[cfg(unix)]
            Unplaced::DmaBuf { unbound, .. } => unbound.release(self),
        }
    }

    /// The image of `raw` over `texels`, bound to the buffer's memory, of `plan` and made with
    /// `descriptor`: its hazard identity the buffer's bytes', tracked as the texture is.
    fn in_place_of(
        &self,
        raw: Raw,
        plan: ImagePlan,
        descriptor: &TextureDescriptor,
        texels: &Texels,
    ) -> Image {
        Image {
            device: self.clone(),
            raw,
            owner: Owner::Buffer {
                memory: Mutex::new(texels.buffer.memory_hold()),
            },
            plan,
            options: texels.options(),
            allow_gpu_optimized_contents: descriptor.allow_gpu_optimized_contents,
            identity: crate::resource::Identity {
                tracked: texels.tracked(),
                ..texels.buffer.identity()
            },
            views: Mutex::new(KeyMap::default()),
            own_views: Mutex::new(KeyMap::default()),
            resolves: KeptResolves::new(),
            decoded: None,
            second: OnceLock::new(),
            form: Default::default(),
            kept_outside: AtomicBool::new(false),
            released: AtomicBool::new(false),
            host_shadow: Mutex::default(),
            between: OnceLock::new(),
            detached: Mutex::new(None),
            first_move: Mutex::new(None),
            host_copies: false,
            linear: true,
            staging_copies: false,
            objects: ObjectsToken::default(),
        }
    }

    /// The unbound image in place of `plan` for texels laid out as `layout`, made for host
    /// allocations, with whether it is the `DRM_FORMAT_MOD_LINEAR` one: the driver's linear image
    /// where it lays the rows (and slices) out at the texels' pitches from offset 0, else the
    /// `DRM_FORMAT_MOD_LINEAR` image at those pitches; why not, each attempt's refusal named,
    /// where the device makes neither.
    pub(super) fn in_place_unbound(
        &self,
        plan: &ImagePlan,
        layout: &BufferLayout,
        facts: &ImageFacts<'_>,
    ) -> Result<Unbound, Error> {
        let lies = |report: &vk::SubresourceLayout| {
            report.offset == 0
                && report.row_pitch == layout.row_pitch
                && (layout.slices == 1 || report.array_pitch == layout.slice_pitch)
        };
        let made = match linear_lack(plan, facts) {
            Some(LinearLack { format, features }) => {
                Err(LinearRefusal::Need(LinearNeed::Features {
                    format,
                    features,
                }))
            }
            None => self.host_linear_image(plan)?,
        };
        // Whether the driver made its linear image and laid it out otherwise.
        let (linear, elsewhere) = match made {
            Ok(made) if lies(&made.report) => return Ok(Ok((made, false))),
            Ok(made) => {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(self.shared().vk(), made.raw) };
                let why = format!(
                    "its linear image for host allocations lays rows {} bytes apart (slices {}) \
                     from offset {}, where the texels' lie {} (slices {}) apart from offset 0",
                    made.report.row_pitch,
                    made.report.array_pitch,
                    made.report.offset,
                    layout.row_pitch,
                    layout.slice_pitch
                );
                (why, true)
            }
            Err(refusal) => (refusal.describe(plan), false),
        };
        Ok(match self.host_pitched_image(plan, layout, facts)? {
            Ok(made) => Ok((made, true)),
            // A linear image the driver lays out otherwise is refused for its pitch; else the
            // pitched image's refusal names why.
            Err((kind, pitched)) => {
                let kind = if elsewhere { Refusal::Pitch } else { kind };
                Err((kind, format!("{linear}; and {pitched}")))
            }
        })
    }

    /// A linear image of `plan` made for host allocations, unbound, in the `UNDEFINED` layout
    /// Vulkan makes it in (`VUID-VkImageCreateInfo-pNext-01443`), with its memory requirements
    /// and its first subresource's layout; why not where the device imports no host allocation
    /// for such an image but a dedicated one, or makes none within its limits.
    fn host_linear_image(
        &self,
        plan: &ImagePlan,
    ) -> Result<Result<LinearMade, LinearRefusal>, Error> {
        let raw_device = self.shared().vk();
        let view_formats = plan.ash_view_formats();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = plan
            .create_info(self.shared().sharing(), &view_formats, &mut list)
            .tiling(vk::ImageTiling::LINEAR)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external);
        let limits = match self.host_image_limits(&info, &view_formats)? {
            Ok(limits) => limits,
            Err(refusal) => return Ok(Err(refusal)),
        };
        if let Err(need) = check_linear_limits(plan, &limits) {
            return Ok(Err(LinearRefusal::Need(need)));
        }
        // SAFETY: the device is live and `info` (with its chain) valid and within the device's
        // limits for an image of host allocations.
        let raw = unsafe { crate::creation::image(raw_device, &info) }?;
        let (requirements, dedicated) = requirements_of(raw_device, raw);
        if dedicated {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Ok(Err(LinearRefusal::NotImportable(
                vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY,
            )));
        }
        // SAFETY: `raw` is a live linear image of this device, and its first subresource has the
        // color aspect (every format a buffer backs is a color format).
        let report =
            unsafe { raw_device.get_image_subresource_layout(raw, first_color_subresource()) };
        crate::exit_guard::called(crate::exit_guard::Call::GetImageSubresourceLayout);
        Ok(Ok(LinearMade {
            raw,
            requirements,
            report,
            initial: vk::ImageLayout::UNDEFINED,
        }))
    }

    /// A `DRM_FORMAT_MOD_LINEAR` image of `plan` made for host allocations, unbound, whose one
    /// plane lies at offset 0 with rows `layout.row_pitch` apart (and slices `layout.slice_pitch`,
    /// for a plan of several layers), its requirements and its plane's layout as the driver reports
    /// them, made `UNDEFINED`. Why not where the device does not enable the extension, makes no
    /// such image for host allocations (`Device::modifier_image_checks`), refuses the layout or
    /// lays it out otherwise, or asks a dedicated allocation of it: the runtime asks the driver and
    /// takes its answer, never a rule of its own.
    #[cfg(unix)]
    fn host_pitched_image(
        &self,
        plan: &ImagePlan,
        layout: &BufferLayout,
        facts: &ImageFacts<'_>,
    ) -> Result<Result<LinearMade, (Refusal, String)>, Error> {
        use crate::dma_buf::{memory_plane_0, ModifierMemory, DRM_FORMAT_MOD_LINEAR};
        let (row_pitch, slice_pitch) = (layout.row_pitch, layout.slice_pitch);
        let raw_device = self.shared().vk();
        let enabled = (self.shared().host_import.as_ref())
            .is_ok_and(|importer| importer.makes_dma_buf_images());
        if !enabled {
            return Ok(Err((
                Refusal::Pitch,
                "the device enables no VK_EXT_image_drm_format_modifier for an image at Metal's \
                 row pitch"
                    .to_owned(),
            )));
        }
        let host = ModifierMemory::Host;
        let view_formats = match self.modifier_image_checks(plan, facts, host)? {
            Ok(view_formats) => view_formats,
            Err(why) => return Ok(Err((Refusal::Image, why))),
        };
        if let Some(why) = self.explicit_plane_rows(plan) {
            return Ok(Err((Refusal::Pitch, why)));
        }
        let array_pitch = if plan.layers > 1 { slice_pitch } else { 0 };
        let planes = [vk::SubresourceLayout {
            offset: 0,
            size: 0,
            row_pitch,
            array_pitch,
            depth_pitch: 0,
        }];
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
            .plane_layouts(&planes);
        let raw = match self.drm_linear_image(plan, &view_formats, &mut explicit, host)? {
            Ok(raw) => raw,
            Err(error) => {
                return Ok(Err((
                    Refusal::Pitch,
                    format!(
                        "the driver refuses a DRM_FORMAT_MOD_LINEAR image for host allocations \
                         with rows {row_pitch} bytes apart ({error:?})"
                    ),
                )));
            }
        };
        let (requirements, dedicated) = requirements_of(raw_device, raw);
        // SAFETY: `raw` is a live image of a DRM format modifier's tiling, whose one memory plane
        // is plane 0 (`VUID-vkGetImageSubresourceLayout-tiling-09433`).
        let report = unsafe { raw_device.get_image_subresource_layout(raw, memory_plane_0()) };
        crate::exit_guard::called(crate::exit_guard::Call::GetImageSubresourceLayout);
        let why = if dedicated {
            Some((
                Refusal::Image,
                format!(
                    "it asks a dedicated allocation of a DRM_FORMAT_MOD_LINEAR image of {:?} for \
                     host allocations, which one bound at an offset of the buffer's import is not",
                    plan.format
                ),
            ))
        } else if report.offset != 0
            || report.row_pitch != row_pitch
            || (plan.layers > 1 && report.array_pitch != array_pitch)
        {
            Some((
                Refusal::Pitch,
                format!(
                    "the driver lays a DRM_FORMAT_MOD_LINEAR plane for host allocations out at \
                     offset {}, rows {} and slices {} bytes apart, where the texels lie at offset \
                     0, rows {row_pitch} and slices {array_pitch} apart",
                    report.offset, report.row_pitch, report.array_pitch
                ),
            ))
        } else {
            None
        };
        if let Some(why) = why {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Ok(Err(why));
        }
        Ok(Ok(LinearMade {
            raw,
            requirements,
            report,
            initial: vk::ImageLayout::UNDEFINED,
        }))
    }

    /// No image is laid out by DRM format modifier on this system.
    #[cfg(not(unix))]
    fn host_pitched_image(
        &self,
        _plan: &ImagePlan,
        _layout: &BufferLayout,
        _facts: &ImageFacts<'_>,
    ) -> Result<Result<LinearMade, (Refusal, String)>, Error> {
        Ok(Err((
            Refusal::Pitch,
            "the system lays out no image by DRM format modifier".to_owned(),
        )))
    }

    /// The images in place over a no-copy buffer's (or a no-copy heap's) host pages that `watches`
    /// note, made again over a rebind's new `pages` (`rebind.rs`): each of the same plan, its
    /// texels as far from the new pages' first byte as they were from the old's; swapped in once
    /// the rebind is made, the device's next submission that names it making its first move. Over
    /// pages imported by their pointer, an image made for host allocations as
    /// [`in_place_image`](Self::in_place_image) made it, bound in their import; over pages
    /// imported as dma-bufs, the `DRM_FORMAT_MOD_LINEAR` image `remade` holds for it, made before
    /// the rebind settled ([`dma_buf_in_place_again`](Self::dma_buf_in_place_again)), which its
    /// first move acquires from `VK_QUEUE_FAMILY_FOREIGN_EXT`; over a copy of the caller's pages
    /// (`copied.rs`), the image [`in_place_over_copy`](Self::in_place_over_copy) makes. Refused
    /// (`NotImplemented(RebindTextures)`) where the device makes no such image there: the new
    /// memory's type is not one the image takes, the image's alignment misses its place, the
    /// driver lays the copy's image out at other pitches, or `remade` holds none for it.
    pub(crate) fn rebound_in_place_images(
        &self,
        watches: &[InPlaceWatch],
        pages: &Arc<crate::buffer::HostPages>,
        remade: &mut Remade,
    ) -> Result<Vec<ImageSwap>, Error> {
        let dma_buf = pages.over_dma_buf();
        let (memory, base) = pages.memory();
        let type_index = pages.type_index();
        let raw_device = self.shared().vk();
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let mut swaps: Vec<ImageSwap> = Vec::new();
        for watch in watches {
            let Some(image) = watched(watch) else {
                continue;
            };
            if swaps.iter().any(|swap| swap.is_of(&image)) {
                continue;
            }
            if dma_buf {
                let Some(objects) = remade.take(&image) else {
                    return Err(Error::NotImplemented(RuntimeGap::RebindTextures));
                };
                let family = self.foreign_acquirer();
                // Dropped on a later failure, the images made so far go.
                swaps.push(ImageSwap::acquired(image, objects, pages, family));
                continue;
            }
            if pages.copied() {
                // Dropped on a later failure, the images made so far go.
                swaps.push(self.in_place_over_copy(image, watch, pages)?);
                continue;
            }
            let at = base + watch.from_base;
            let made = self.in_place_unbound(image.plan(), &watch.layout, &facts)?;
            let Ok((made, _)) = made else {
                return Err(Error::NotImplemented(RuntimeGap::RebindTextures));
            };
            let raw = made.raw;
            if made.requirements.memory_type_bits & (1 << type_index) == 0
                || !at.is_multiple_of(made.requirements.alignment.max(1))
            {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(raw_device, raw) };
                return Err(Error::NotImplemented(RuntimeGap::RebindTextures));
            }
            // SAFETY: the image is live and unbound, made for host allocations; the memory is the
            // new pages' import, in a type the image takes, at an offset that meets its alignment,
            // where its texels lie as they lay in the old pages (a rebind keeps the bytes' offset
            // in their page, and their length).
            if let Err(error) = unsafe { crate::creation::bind_image(raw_device, raw, memory, at) }
            {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(raw_device, raw) };
                return Err(vulkan(error));
            }
            let objects = ImageObjects::new(self, raw);
            let first = FirstMove::new(raw, image.plan().format, made.initial, pages_hold(pages));
            // Dropped on a later failure, the images made so far go.
            swaps.push(ImageSwap::new(image, objects, first, pages));
        }
        Ok(swaps)
    }

    /// Whether a rebind's copy of a buffer's bytes (`copied.rs`) may lie in fresh pages the
    /// runtime imports (`Device::fresh_page_buffer`) for the images in place that `watches` note:
    /// where the device's check passed for fresh pages, each live image's accesses and how its
    /// texels lie in the device's lines of memory (`in_place::RowEnds`: a rebind keeps the bytes'
    /// offset in their page) ([`fresh_pages_take`](Self::fresh_pages_take)), as each is then made
    /// over them for host allocations. Where it did not, the copy lies in memory the device allocates, and each image
    /// is a linear image of its own over it ([`in_place_over_copy`](Self::in_place_over_copy)).
    /// On RADV the check leaves fresh pages unchecked: no new buffer lies in them where each held
    /// pointer import costs every submission (`fresh_pages.rs`), so the check finds none to try.
    pub(crate) fn fresh_pages_take_images(&self, watches: &[InPlaceWatch]) -> bool {
        watches.iter().all(|watch| {
            watched(watch)
                .is_none_or(|image| self.fresh_pages_take(&image, self.watched_rows(watch)))
        })
    }

    /// How the texels `watch` notes lie in the device's lines of memory (`in_place::RowEnds`),
    /// and whether a shader's writes may change their rows' padding (`in_place::Padding`):
    /// `from_base` past the pages' first byte, which starts a page.
    fn watched_rows(&self, watch: &InPlaceWatch) -> (RowEnds, Padding) {
        let start = watch.from_base - watch.layout.offset;
        let line = self.shared().in_place.line();
        let padding = Padding::of(start, &watch.layout, line, watch.padding_owned);
        (RowEnds::of(start, &watch.layout, line), padding)
    }

    /// Whether the device's check passed for fresh pages, each of `image`'s accesses and texels
    /// that lie in lines as `rows` says: an image of attachment usage renders, one of storage
    /// usage is written, one of neither reads. The writes of one that does not render whose
    /// rows' padding is free take the check that lets it change (`in_place::in_place`); one that
    /// renders may hold the storage usage a render target takes where the check that keeps it
    /// passed (`Device::storage_over_buffer`'s `storable`), which it asks.
    fn fresh_pages_take(&self, image: &Image, (rows, padding): (RowEnds, Padding)) -> bool {
        use metal2vulkan::vk::ImageUsageFlags;
        let usage = image.plan().usage;
        let renders = usage.contains(ImageUsageFlags::COLOR_ATTACHMENT);
        let writes = usage.contains(ImageUsageFlags::STORAGE);
        let written = match (padding, renders) {
            (Padding::Free, false) => Access::BufferWrites,
            _ => Access::Writes,
        };
        let accesses = [
            (Access::Reads, !renders && !writes),
            (Access::Renders, renders),
            (written, writes),
        ];
        accesses.into_iter().all(|(access, takes)| {
            !takes
                || self.in_place_verdict(PagesClass::Fresh, access, rows)
                    == crate::in_place::Verdict::Passed
        })
    }

    /// The image in place of `image`'s plan over a rebind's copy of the caller's pages
    /// (`copied.rs`: the caller's gate refused their import, it failed, or the bytes moved in
    /// their page), its texels where `watch` notes them, swapped in once the rebind is made. Over
    /// a copy in fresh pages the runtime imported (`fresh_pages.rs`), the image an import of the
    /// caller's pages takes, made for host allocations (the driver's linear image at the texels'
    /// pitches, else the `DRM_FORMAT_MOD_LINEAR` one), where the device's check passed for fresh
    /// pages (`in_place.rs`); over one in memory the device allocates, a linear image of its own,
    /// as a texture's over a buffer of the runtime's, which starts `PREINITIALIZED` and so keeps
    /// the copy's bytes at its first move ("the transition away from this layout will preserve
    /// that data", Image Layouts). Either lies at the texels' pitches from offset 0, in a type of
    /// the copy's memory, at an offset its alignment meets, inside the copy's memory. Refused
    /// (`NotImplemented(RebindTextures)`) where it does not.
    fn in_place_over_copy(
        &self,
        image: Arc<Image>,
        watch: &InPlaceWatch,
        pages: &Arc<crate::buffer::HostPages>,
    ) -> Result<ImageSwap, Error> {
        let (memory, base) = pages.memory();
        let layout = &watch.layout;
        let raw_device = self.shared().vk();
        let refused = || Error::NotImplemented(RuntimeGap::RebindTextures);
        // A copy in fresh pages the runtime imported takes the image an import of the caller's
        // pages takes, made for host allocations (`in_place_unbound`), where the device's check
        // passed for fresh pages and each of the image's accesses (an image of attachment usage
        // renders, one of storage usage is written; one of neither reads); one in memory the
        // device allocates, a linear image of its own.
        let made = match pages.over_own_pages() {
            true => {
                if !self.fresh_pages_take(&image, self.watched_rows(watch)) {
                    return Err(refused());
                }
                let facts = self
                    .shared()
                    .textures
                    .facts(self.compile_device().profile());
                let Ok((made, _)) = self.in_place_unbound(image.plan(), layout, &facts)? else {
                    return Err(refused());
                };
                made
            }
            false => {
                let Ok(made) = self.linear_image(image.plan())? else {
                    return Err(refused());
                };
                made
            }
        };
        let raw = made.raw;
        let at = base + watch.from_base;
        let room = pages.room().saturating_sub(watch.from_base);
        let lies = made.report.offset == 0
            && made.report.row_pitch == layout.row_pitch
            && (layout.slices == 1 || made.report.array_pitch == layout.slice_pitch);
        if !lies || placed(&made.requirements, at, room, pages.type_index()).is_err() {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Err(refused());
        }
        // SAFETY: the image is live and unbound; the memory is the copy's, which the new pages
        // hold, in a type the image takes, at an offset that meets its alignment, its size inside
        // the memory (`placed`), where its texels lie as they lay in the old pages.
        if let Err(error) = unsafe { crate::creation::bind_image(raw_device, raw, memory, at) } {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Err(vulkan(error));
        }
        self.shared().counts.copied_rebind_image_made();
        let objects = ImageObjects::new(self, raw);
        let first = FirstMove::new(raw, image.plan().format, made.initial, pages_hold(pages));
        Ok(ImageSwap::new(image, objects, first, pages))
    }

    /// The images in place over a no-copy buffer's (or a no-copy heap's) bytes that `watches`
    /// note, made for a rebind's new pages, which it imported as dma-bufs (`placement`), the bytes
    /// at `pointer`, into `remade`: for each, the `DRM_FORMAT_MOD_LINEAR` image of its plan at its
    /// texels' pitches over the one dma-buf the pages' source makes of its own pages
    /// ([`Device::dma_buf_image`]). With no lock of the runtime's held, as the source is called
    /// once for each. Nothing for pages imported by their pointer. Refused
    /// (`NotImplemented(RebindTextures)`) where the device makes no such image of one.
    #[cfg(unix)]
    pub(crate) fn dma_buf_in_place_again(
        &self,
        watches: &[InPlaceWatch],
        placement: &crate::memory::Placement,
        pointer: std::ptr::NonNull<u8>,
        remade: &mut Remade,
    ) -> Result<(), Error> {
        let Some(imports) = placement.dma_buf_pages() else {
            return Ok(());
        };
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        for watch in watches {
            let Some(image) = watched(watch) else {
                continue;
            };
            if remade.0.iter().any(|(made, _)| Arc::ptr_eq(made, &image)) {
                continue;
            }
            let texels = crate::dma_buf::DmaBufTexels {
                pages: imports,
                type_index: placement.type_index(),
                at: placement.dma_buf_address(pointer.as_ptr() as u64 + watch.from_base),
                row_pitch: watch.layout.row_pitch,
                slice_pitch: watch.layout.slice_pitch,
            };
            let made = self.dma_buf_image(image.plan(), &facts, &texels)?;
            let made = made.map_err(|_| Error::NotImplemented(RuntimeGap::RebindTextures))?;
            let objects = ImageObjects::imported(self, made.raw, made.import);
            remade.0.push((image, objects));
        }
        Ok(())
    }

    /// No pages are imported as dma-bufs on this system: nothing is made.
    #[cfg(not(unix))]
    pub(crate) fn dma_buf_in_place_again(
        &self,
        _watches: &[InPlaceWatch],
        _placement: &crate::memory::Placement,
        _pointer: std::ptr::NonNull<u8>,
        _remade: &mut Remade,
    ) -> Result<(), Error> {
        Ok(())
    }
}

/// The image a watch notes, while it lives.
fn watched(watch: &InPlaceWatch) -> Option<Arc<Image>> {
    let image: Weak<dyn Any + Send + Sync> = Weak::clone(&watch.image);
    image.upgrade()?.downcast::<Image>().ok()
}

/// Whether an image of `requirements` bound at `at` of memory of type `type_index`, with `room`
/// bytes from there to the memory's end, lies in its place: at a multiple of its alignment, its
/// size inside the memory, in a type it takes; why not.
fn placed(
    requirements: &vk::MemoryRequirements,
    at: u64,
    room: u64,
    type_index: u32,
) -> Result<(), (Refusal, String)> {
    let why = if !at.is_multiple_of(requirements.alignment.max(1)) {
        format!(
            "its image is aligned to {} bytes, where the texels lie at offset {at} of the \
             buffer's memory",
            requirements.alignment
        )
    } else if requirements.size > room {
        format!(
            "its image takes {} bytes, past the {room} bytes of the buffer's memory from its \
             texels",
            requirements.size
        )
    } else if requirements.memory_type_bits & (1 << type_index) == 0 {
        format!("its image takes none of the buffer's memory type {type_index}")
    } else {
        return Ok(());
    };
    Err((Refusal::Placement, why))
}

/// The memory requirements of `raw`, a live image of the device of `raw_device`, and whether it
/// requires a dedicated allocation.
pub(super) fn requirements_of(
    raw_device: &crate::device::functions::Vk,
    raw: vk::Image,
) -> (vk::MemoryRequirements, bool) {
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    let query = vk::ImageMemoryRequirementsInfo2::default().image(raw);
    // SAFETY: `raw` is a live image of this device; the query is core in Vulkan 1.1.
    unsafe { raw_device.get_image_memory_requirements2(&query, &mut requirements) };
    let requirements = requirements.memory_requirements;
    (
        requirements,
        dedicated.requires_dedicated_allocation != vk::FALSE,
    )
}
