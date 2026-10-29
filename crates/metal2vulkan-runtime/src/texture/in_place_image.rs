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
//! - **Its usage** is the texture's declared plan's, nothing wider: a write beyond it binds the
//!   texture's usage shadow (`texture/usage_shadow.rs`), and a render pass into it renders through
//!   an attachment stand-in (`attachment_stand_in.rs`), since the image has neither storage nor
//!   attachment usage. Their copies into it touch its texels alone.
//! - **A rebind** (`rebind.rs`) makes it again over the new pages, its texels as far from their
//!   first byte as from the old pages', as a no-copy heap's images are
//!   ([`Device::rebound_in_place_images`]); a detach makes it present a stand-in, as it does every
//!   image bound to the pages (`detach.rs`).

use super::{
    first_color_subresource, pages_hold, FirstMove, Image, ImageObjects, ImageSwap, KeptResolves,
    KeyMap, LinearMade, LinearRefusal, ObjectsToken, Owner, Raw, Remade, Storage, Texture,
};
use crate::buffer::InPlaceWatch;
use crate::buffer_texture::Texels;
use crate::device::lock;
use crate::error::{vulkan, Error, RuntimeGap};
use crate::image_plan::{ImageFacts, ImagePlan};
use crate::in_place::{PagesClass, Refusal};
use crate::linear_plan::{check_linear_limits, linear_lack, BufferLayout, LinearLack};
use crate::texture_descriptor::TextureDescriptor;
use crate::{release, Device};
use ash::vk;
use metal2vulkan::error::LinearNeed;
use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// An image in place made over a texture's texels, or why none is: the refusal's kind and words.
pub(super) type InPlace = Result<(Arc<Image>, bool), (Refusal, String)>;

/// An unbound image in place, or why none is.
type Unbound = Result<(LinearMade, bool), (Refusal, String)>;

impl Texture {
    /// Make the first move of the texture's image in place now, by an internal submission the
    /// call waits for (`Image::settle`): the device's check's round (`in_place_check.rs`). `None`
    /// where the texture is no image aliasing its bytes.
    pub(crate) fn settle_in_place(&self) -> Option<Result<(), Error>> {
        match &self.0.storage {
            Storage::Alias { image, .. } => Some(image.settle()),
            _ => None,
        }
    }
}

impl Device {
    /// The image in place of a texture of `plan` (its declared usage's, the module docs) made with
    /// `descriptor` over `texels`, host pages of `class`, with whether it is a
    /// `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch made for host allocations; why not where the
    /// device makes none (the module docs). Over a no-copy buffer's pages (or a no-copy heap's) it
    /// is noted for a rebind to make again. Its first move is noted, for the device's next
    /// submission that takes it (`texture/first_move.rs`).
    pub(super) fn in_place_image(
        &self,
        plan: &ImagePlan,
        descriptor: &TextureDescriptor,
        texels: &Texels,
        class: PagesClass,
        facts: &ImageFacts<'_>,
    ) -> Result<InPlace, Error> {
        let layout = texels.layout;
        let shared_bytes = layout.shared_bytes();
        if shared_bytes != 0 {
            return Ok(Err((
                Refusal::Rows,
                format!(
                    "its rows lie {} bytes apart, {} bytes of texels each, so consecutive rows \
                     share {shared_bytes} bytes: no image holds a texel twice",
                    layout.row_pitch, layout.row_bytes
                ),
            )));
        }
        let made = match class {
            PagesClass::DmaBuf => self.dma_buf_in_place(plan, descriptor, texels, facts)?,
            PagesClass::Pointer | PagesClass::Fresh => {
                self.host_in_place(plan, descriptor, texels, class, facts)?
            }
        };
        let (image, pitched) = match made {
            Ok(made) => made,
            Err(refused) => return Ok(Err(refused)),
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
                });
            }
        }
        Ok(Ok((image, pitched)))
    }

    /// [`in_place_image`](Self::in_place_image) over pages imported by their pointer: the image
    /// made for host allocations, checked against the texels' place in the buffer's memory, and
    /// bound there.
    fn host_in_place(
        &self,
        plan: &ImagePlan,
        descriptor: &TextureDescriptor,
        texels: &Texels,
        class: PagesClass,
        facts: &ImageFacts<'_>,
    ) -> Result<InPlace, Error> {
        let layout = texels.layout;
        let (memory, buffer_offset, type_index) = texels.buffer.memory();
        let at = buffer_offset + layout.offset;
        let room = texels.buffer.memory_room().saturating_sub(layout.offset);
        let (made, pitched) = match self.in_place_unbound(plan, &layout, facts)? {
            Ok(made) => made,
            Err(refused) => return Ok(Err(refused)),
        };
        let raw = made.raw;
        let raw_device = self.shared().vk();
        if let Err(refused) = placed(&made.requirements, at, room, type_index) {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Ok(Err(refused));
        }
        // SAFETY: the image is live and unbound, made for host allocations; the memory is the
        // buffer's import of host pages (or its fresh pages'), which the image holds, in a type the
        // image takes, at an offset that meets its alignment, its size inside the memory
        // (`placed`).
        if let Err(error) = unsafe { crate::creation::bind_image(raw_device, raw, memory, at) } {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Err(vulkan(error));
        }
        let image = self.in_place_of(Raw::new(raw), plan.clone(), descriptor, texels);
        // Over a no-copy buffer's pages, its move notes them (a rebind may swap the image out of
        // them); fresh pages are the buffer's for good.
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

    /// [`in_place_image`](Self::in_place_image) over pages imported as dma-bufs: the
    /// `DRM_FORMAT_MOD_LINEAR` image at the texture's pitches over a dma-buf of its own pages,
    /// which the pages' source makes (called with no lock of the runtime's held).
    #[cfg(unix)]
    fn dma_buf_in_place(
        &self,
        plan: &ImagePlan,
        descriptor: &TextureDescriptor,
        texels: &Texels,
        facts: &ImageFacts<'_>,
    ) -> Result<InPlace, Error> {
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
        let dma_texels = crate::dma_buf::DmaBufTexels {
            pages: &imports,
            type_index,
            at: start + (buffer_offset + layout.offset - base),
            row_pitch: layout.row_pitch,
            slice_pitch: layout.slice_pitch,
        };
        let made = match self.dma_buf_image_or_why(plan, facts, &dma_texels)? {
            Ok(made) => made,
            Err((kind, why)) => {
                use crate::dma_buf::DmaBufRefusal as D;
                let kind = match kind {
                    D::Pitch => Refusal::Pitch,
                    D::Source => Refusal::Source,
                    D::Image | D::Pages | D::Import => Refusal::DmaBuf,
                };
                return Ok(Err((kind, why)));
            }
        };
        let owner = Owner::Buffer {
            memory: Mutex::new(texels.buffer.memory_hold()),
        };
        let made_with = (texels.options(), descriptor.allow_gpu_optimized_contents);
        let identity = crate::resource::Identity {
            tracked: texels.tracked(),
            ..texels.buffer.identity()
        };
        let image =
            self.dma_buf_texture_image(made, plan.clone(), owner, made_with, identity, &pages);
        Ok(Ok((image.shared(), false)))
    }

    /// No pages are imported as dma-bufs on this system.
    #[cfg(not(unix))]
    fn dma_buf_in_place(
        &self,
        _plan: &ImagePlan,
        _descriptor: &TextureDescriptor,
        _texels: &Texels,
        _facts: &ImageFacts<'_>,
    ) -> Result<InPlace, Error> {
        Ok(Err((
            Refusal::Image,
            "the system imports no dma-bufs".to_owned(),
        )))
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
    fn in_place_unbound(
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
        // passed for fresh pages; one in memory the device allocates, a linear image of its own.
        let made = match pages.over_own_pages() {
            true => {
                if self.in_place_verdict(PagesClass::Fresh) != crate::in_place::Verdict::Passed {
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
                at: pointer.as_ptr() as u64 + watch.from_base,
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
