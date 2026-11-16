//! Texture views whose image takes no Vulkan view of their format, which hold their texels apart
//! (the coordinator's ruling, 2026-10-06: plans (a) and (b)):
//!
//! - **Another component layout on a device whose Vulkan views reinterpret no image's
//!   components**: a portability subset without `imageViewFormatReinterpretation`, or an adopted
//!   device whose consumer does not say it enabled it (`RawDevice::view_format_reinterpretation`).
//!   Metal views `RGBA8Unorm` as `R32Uint`; such a device makes no Vulkan view of the image in the
//!   view's format (`view_plan.rs`).
//! - **Another format of a texture made without `PixelFormatView` usage**: plain Metal makes the
//!   view (its API validation stops it) and renders through it the same; the texture's image was
//!   made mutable, if at all, only for its sRGB twin's formats and, where it renders, its
//!   channel-order sibling's and that sibling's twin's (`image_plan::reordered_views`), so a view
//!   of another format of its class (`R32Uint` of an `RGBA8Unorm` texture, or `BGRA8Unorm` of one
//!   that renders nothing) is not a view of that image.
//!
//! Such a view:
//!
//! - **A view in other channels of its image's bits** is the image itself to its bindings, its
//!   copies and CPU access (`Device::read_as`): a view of an image in its pixel format's direct
//!   Vulkan format, where the view's format holds its texels in the same bits and presents them in
//!   other channels (`metal::format::swizzle_over`: `A8Unorm` of an `R8Unorm` texture, through
//!   `(Zero, Zero, Zero, Red)`), whatever its usage. Its bindings present it as a texture of the
//!   image's format made with that swizzle composed on the view's own (`ViewRecord::read_as`,
//!   `ViewRecord::bound`): it samples, gathers and reads Metal's texels of the view, the same bits
//!   through the same filter, the channels moved by the view's mapping, its border texels too; a
//!   write stores as a swizzled texture's storage binding does (`swizzled_storage`, in the
//!   function), each of the image's channels from the first lane whose source it is, as Metal
//!   stores the view's (probe `a8write`). Its copies and CPU access move the image's bytes, the
//!   view's, and `generate_mipmaps` runs as the image's format (`Texture::bound_twin`), whose
//!   filter pairs it with the view's. A pass renders into it through a shadow image of the view's
//!   format (`Texture::view_render_shadow`), copied in before the pass and back after it as the
//!   next bullet's are; Vulkan views the image in no format of the view's. Each such view counts
//!   in `WorkCounts::views_through_swizzle`; the render shadow is announced as below.
//! - **A view of a texture over a buffer** (plan (a)) is a texture over the same bytes in the
//!   view's format: an aliasing image where the device lays one out as Metal does, else a shadow
//!   of the bytes, exactly as such a texture is (`shadow.rs`, `Device::storage_over_buffer`).
//! - **A view of an image** (plan (b)) is an image of the view's format and the viewed image's
//!   shape, in memory of its own, a shadow of the viewed image's texels in the levels and layers
//!   the view presents (`ShadowSource::Image`): the encoders copy them in before a command uses
//!   the view and back after it writes through it, by `vkCmdCopyImage` between formats of one
//!   copy unit, which keeps every bit (measured: `runtime_lane::view_reinterpretation::
//!   copies_between_formats_of_one_unit_keep_every_bit`). A command that writes the image both
//!   through the view and through another alias (the image, or another such view) is refused
//!   (`ShadowSync::before`): a copy back would replace the other alias's writes, and only a
//!   buffer's bytes have a merge (`shadow_merge.rs`). One that writes through one alias and reads
//!   through another reads the texels as they were before it, as a texture over a buffer does.
//!   CPU access through the view reads and writes the viewed image. The copies are announced
//!   once per device (`shadowed-texture-view`, performance), naming the cause.
//! - **A view of a shadowed view** presents the viewed image itself where that image takes the
//!   view's format, else a shadow of it of its own.
//! - **Refused**, typed: a view of a sparse texture, of a heap's texture (whose image aliases the
//!   heap's other resources), and of an image the copies cannot serve (formats whose copy units
//!   differ, an image without transfer usage): `Unsupported` of
//!   `Feature::ViewFormatReinterpretation` (not representable) where the device reinterprets no
//!   components, else of `Feature::TextureView` (not implemented).

use super::{IdUse, Image, Over, Owner, Storage, Stored, Texture};
use crate::degradation::Degradation;
use crate::device::lock;
use crate::error::{internal, Error, RuntimeGap};
use crate::image_plan::{copies_between, plan_image, views_missed, ImagePlan};
use crate::memory::Placement;
use crate::resource::{Identity, ResourceKey};
use crate::shadow::{Backing, ImageThrough, ImageTruth, ShadowCopies, ShadowSource};
use crate::shadow_merge::ScratchSpares;
use crate::shadow_standing::StandingAcross;
use crate::texture::{copy_units, image_error, texture_aspects};
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use crate::tracker::{ImagePlace, UseRange};
use crate::values::{ResourceOptions, StorageMode};
use crate::view_placement::{ViewPlacement, ViewShadowReason};
use crate::view_plan::{keeps_components, reinterpretation_refused, ViewRecord, ViewRequest};
use crate::Device;
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::{direct_format, swizzle_over};
use metal2vulkan::metal::{PixelFormat, TextureSwizzleChannels, TextureType};
use metal2vulkan::vk::ImageUsageFlags;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// What a shadowed view's image keeps beside itself (the module docs): its memory, the image whose
/// texels it copies, and the copies.
pub(crate) struct ViewShadow {
    pub(super) placement: Placement,
    pub(super) base: Arc<Image>,
    pub(super) copies: Arc<ShadowCopies>,
}

/// The plan of a shadow image of another image's texels (`Device::shadow_plan`): its layout and
/// plan, and the bytes and block of the one copy unit its copies move through a buffer, where
/// Vulkan copies no texels between the two images' formats.
pub(super) struct ShadowPlan {
    layout: TextureLayout,
    plan: ImagePlan,
    through: Option<(u32, [u32; 2])>,
}

/// The image whose texels `image` holds: the viewed image of a shadowed view's, else `image`.
pub(super) fn truth(image: &Arc<Image>) -> &Arc<Image> {
    match &image.owner {
        Owner::ViewShadow(shadow) => &shadow.base,
        _ => image,
    }
}

/// [`truth`] of a borrowed image.
fn truth_of(image: &Image) -> &Image {
    match &image.owner {
        Owner::ViewShadow(shadow) => &shadow.base,
        _ => image,
    }
}

/// How a view presenting a record of a texture holds its texels, decided before anything is made
/// ([`Texture::view_hold`]): what the view's making makes (`Texture::view`) and
/// [`Texture::view_placement`] answers.
pub(super) enum ViewHold {
    /// The texture's own storage: its image takes a Vulkan view of the view's format (a texture
    /// buffer's: its texel buffer view takes any format of its class).
    Shared,
    /// The image the view reads in the image's own format, and the format and swizzle its bindings
    /// present it as (`Device::read_as`).
    Through(Arc<Image>, (PixelFormat, TextureSwizzleChannels)),
    /// The image a shadowed view's texels are copied from, which takes a Vulkan view of the view's
    /// format.
    Viewed(Arc<Image>),
    /// A shadow of the image (its texture's, or the one a shadowed view's texels are copied from),
    /// for the reason.
    Shadow(Arc<Image>, ViewShadowReason),
    /// Storage of its own over the bytes the texture is over ([`Texture::over_bytes`]).
    Bytes,
}

impl Texture {
    /// How a view presenting `record` of this texture holds its texels (the module docs): the
    /// texture's storage where its image takes a Vulkan view of the view's format, else the image
    /// in its own format through a swizzle where the view presents other channels of its bits, else
    /// a shadow of the image, or storage of its own over the texture's bytes. Refused (`Unsupported`,
    /// as `Device::apart_refusal` names it) for a view apart of a sparse texture's or a heap's
    /// image, of a tiled texture, and of a texture buffer.
    pub(super) fn view_hold(&self, record: &ViewRecord) -> Result<ViewHold, Error> {
        let device = self.generation();
        let pf = record.pixel_format;
        let refused = |image: Option<&Image>| Error::Unsupported(device.apart_refusal(image, pf));
        let image = match self.storage()? {
            Storage::Texels(_) => return Ok(ViewHold::Shared),
            Storage::Image(image)
            | Storage::Alias { image, .. }
            | Storage::Shadow { image, .. } => image,
        };
        if device.view_apart(image, pf).is_none() {
            return Ok(ViewHold::Shared);
        }
        if let Some((image, read_as)) = self.read_through(record) {
            return Ok(ViewHold::Through(image, read_as));
        }
        match self.storage()? {
            Storage::Image(image) => {
                let base = match &image.owner {
                    Owner::ViewShadow(shadow) => &shadow.base,
                    Owner::Runtime { .. } | Owner::Caller(_) | Owner::HeapUnaliased(_) => image,
                    Owner::Buffer { .. }
                    | Owner::Sparse(_)
                    | Owner::Heap(_)
                    | Owner::HeapShadow(_) => return Err(refused(Some(image.as_ref()))),
                };
                Ok(match device.view_apart(base, pf) {
                    None => ViewHold::Viewed(Arc::clone(base)),
                    Some(reason) => ViewHold::Shadow(Arc::clone(base), reason),
                })
            }
            // A two-plane YCbCr texture's `_sRGB` view converts the same planes, and a rotated
            // texture's view holds the stored frame in its format (`texture/rotation.rs`).
            Storage::Shadow { copies, .. }
                if copies.planes_conversion().is_some() || record.presented.rotation != 0 =>
            {
                Ok(ViewHold::Bytes)
            }
            // A tiled texture's image is a shadow of the bytes already, a copy taken at use
            // (`Buffer::new_tiled_texture`): no second shadow holds its levels.
            Storage::Shadow { image, .. } if image.plan().levels > 1 => {
                Err(refused(Some(image.as_ref())))
            }
            Storage::Alias { .. } | Storage::Shadow { .. } => Ok(ViewHold::Bytes),
            // A texture buffer's views are texel buffer views of its bytes (above).
            Storage::Texels(_) => Ok(ViewHold::Shared),
        }
    }

    /// The storage of a view presenting `record` of this texture over the bytes it is over, in a
    /// format its image takes no Vulkan view of ([`ViewHold::Bytes`]), for Metal's `call`: a
    /// two-plane YCbCr texture's converted view, a rotated texture's frame, or a texture over the
    /// same bytes in the view's format. Refused as a texture of the view's format over them is.
    pub(super) fn over_bytes(
        &self,
        call: &'static str,
        record: &ViewRecord,
    ) -> Result<Storage, Error> {
        let device = self.generation();
        let refused = || Error::Unsupported(device.apart_refusal(None, record.pixel_format));
        // A two-plane YCbCr texture's view as its `_sRGB` twin converts the same planes.
        if let Some(storage) = self.converted_view(call, record) {
            return storage;
        }
        match self.storage()? {
            // A rotated texture's view holds the stored frame in its format (`texture/rotation.rs`).
            Storage::Shadow { .. } if record.presented.rotation != 0 => {
                self.rotated_view(call, record)
            }
            Storage::Alias { texels, .. } | Storage::Shadow { texels, .. } => {
                let mut descriptor = TextureDescriptor::texture_2d(
                    record.pixel_format,
                    u64::from(record.image_extent[0]),
                    u64::from(record.image_extent[1]),
                    false,
                );
                descriptor.usage = record.usage;
                descriptor.swizzle = record.swizzle;
                descriptor.resource_options = texels.options();
                let layout = descriptor
                    .check(device.properties().texture_rules())
                    .map_err(|_| refused())?;
                // A view's storage never moves in place (`texture/storage_form.rs`), and its
                // shadow is made with it: a view binds no rows.
                let over = (Arc::clone(texels), &mut None, &mut false);
                match device.storage_over_buffer(call, over, &descriptor, &layout)? {
                    Over::Storage(storage) => Ok(storage),
                    Over::Shadow(recipe) => device.made_shadow(&recipe),
                }
            }
            Storage::Image(_) | Storage::Texels(_) => {
                Err(internal("a view over bytes of a texture over none"))
            }
        }
    }

    /// What a view of the whole texture in `pixel_format` would be if made now
    /// ([`new_texture_view`](Self::new_texture_view)): a Vulkan view of the texture's image, the
    /// image through a swizzle, a view shadow and why, or storage of its own over the texture's
    /// bytes ([`ViewPlacement`]), by the same decision the view's making takes. No Metal
    /// counterpart (a runtime extension): Metal makes the view of the texture's own memory; this
    /// says how the runtime would. Refused as `new_texture_view` refuses the view (`Nil` for a
    /// memoryless texture, `Invalid` for a format Metal stops, `Unsupported` or `NotImplemented`
    /// for a view the runtime does not make, a view shadow the copies cannot serve among them),
    /// but for a refusal of storage over the texture's bytes, made only with the view
    /// ([`ViewPlacement::OverBytes`]). Makes nothing, and changes nothing a later view takes.
    pub fn view_placement(&self, pixel_format: PixelFormat) -> Result<ViewPlacement, Error> {
        let _held = self.hold();
        const CALL: &str = "new_texture_view";
        self.check_viewable(CALL)?;
        let request = ViewRequest::of_format(self.record(), pixel_format);
        let (record, _) = self.checked_view(CALL, &request)?;
        Ok(match self.view_hold(&record)? {
            ViewHold::Shared | ViewHold::Viewed(_) => ViewPlacement::Image,
            ViewHold::Through(..) => ViewPlacement::ImageThroughSwizzle,
            ViewHold::Shadow(base, reason) => {
                let device = self.generation();
                device.view_shadow_plan(CALL, &base, &self.root().0.record, &record)?;
                ViewPlacement::Shadow(reason)
            }
            ViewHold::Bytes => ViewPlacement::OverBytes,
        })
    }

    /// The image a view presenting `record` reads in its own format, where it does
    /// (`Device::read_as`): the texture's image, or the image a shadowed view's texels are copied
    /// from, with the format and swizzle the view's bindings present.
    pub(super) fn read_through(
        &self,
        record: &ViewRecord,
    ) -> Option<(Arc<Image>, (PixelFormat, TextureSwizzleChannels))> {
        let Stored::Made(Storage::Image(image)) = self.stored() else {
            return None;
        };
        let base = truth(image);
        let read_as = self.generation().read_as(base, record)?;
        Some((Arc::clone(base), read_as))
    }

    /// The pixel format the texture's bindings present (`ViewRecord::bound`): its own, or, for a
    /// texture that reads its image in another format (`ViewRecord::read_as`: a view in other
    /// channels of its image's bits, a two-plane YCbCr texture's luma view, a texture over a P3
    /// surface), that format, whose texels a slot bound to it reads.
    pub(crate) fn bound_pixel_format(&self) -> PixelFormat {
        self.0.record.bound().pixel_format
    }

    /// A texture presenting what this one's bindings present (`ViewRecord::bound`) over the same
    /// storage, where they differ: a view that reads its image in the image's own format, as a
    /// texture of that format. `None` for every other texture. Refused as the texture's storage
    /// is ([`Texture::storage`]).
    pub(crate) fn bound_twin(&self) -> Result<Option<Texture>, Error> {
        if self.0.record.read_as.is_none() {
            return Ok(None);
        }
        Ok(Some(Texture::made(
            self.generation(),
            self.storage()?.clone(),
            self.0.record.bound(),
            Some(self.clone()),
            IdUse::Runtime,
        )))
    }

    /// The texture a pass renders into for a view that its bindings present in its image's own
    /// format (`ViewRecord::read_as`): a texture of the view's own record over a shadow image of
    /// the view's format, made at the first pass that renders into the view and kept, whose
    /// copies move its texels in before each command that uses it and back after one that writes
    /// it (the module docs). Vulkan views the image in no format of the view's, and Metal renders
    /// into such a view (probe `a8write`: an `A8Unorm` view of an `R8Unorm` texture stores the
    /// fragment's alpha). Refused as the shadow's image is.
    pub(crate) fn view_render_shadow(&self, call: &'static str) -> Result<Texture, Error> {
        let mut held = lock(&self.s().render_shadow);
        if let Some(shadow) = held.as_ref() {
            return Ok(shadow.clone());
        }
        let Storage::Image(image) = self.storage()? else {
            return Err(internal("a view read through its image without one"));
        };
        let record = ViewRecord {
            read_as: None,
            ..self.0.record
        };
        let device = self.generation();
        let base = truth(image);
        let reason = device
            .view_apart(base, record.pixel_format)
            .ok_or_else(|| internal("a view read through an image that takes its format"))?;
        let texture = &self.root().0.record;
        let shadow = device.new_view_shadow(call, base, texture, &record, reason, true)?;
        let texture = Texture::made(
            device,
            Storage::Image(shadow.shared()),
            record,
            Some(self.clone()),
            IdUse::Runtime,
        );
        *held = Some(texture.clone());
        Ok(texture)
    }

    /// The texture views are made from, at the root of `parentTexture`.
    pub(super) fn root(&self) -> &Texture {
        let mut root = self;
        while let Some(parent) = &root.0.parent {
            root = parent;
        }
        root
    }
}

impl Device {
    /// Where a view presenting `record` of `image` (its texture's image, or the image a shadowed
    /// view's texels are copied from) reads and writes the image in the image's own format (the
    /// module docs): that format and the swizzle over its channels its bindings present the view
    /// as. A view of an image of the texture's own memory in its pixel format's direct Vulkan
    /// format, whose format Metal's view presents in other channels of the same bits
    /// (`metal::format::swizzle_over`), whatever its usage; not one whose alpha reads 1 or of a
    /// rotated texture. `None` elsewhere.
    pub(super) fn read_as(
        &self,
        image: &Image,
        record: &ViewRecord,
    ) -> Option<(PixelFormat, TextureSwizzleChannels)> {
        if record.alpha_reads_one || record.presented.rotation != 0 {
            return None;
        }
        if !matches!(
            image.owner,
            Owner::Runtime { .. } | Owner::Caller(_) | Owner::HeapUnaliased(_)
        ) {
            return None;
        }
        let own = image.plan().pixel_format;
        if direct_format(own) != Some(image.plan().format) {
            return None;
        }
        let profile = self.compile_device().profile();
        swizzle_over(record.pixel_format, record.swizzle, own, profile).map(|over| (own, over))
    }

    /// Why `image` takes no Vulkan view of a view of pixel format `pf`, where it takes none; `None`
    /// where it takes one: the device's views reinterpret its components or the view keeps them
    /// (`keeps_components`), else [`ViewShadowReason::Reinterpretation`], and it lists or takes the
    /// view's formats (`image_plan::views_missed`, which names why not).
    pub(super) fn view_apart(&self, image: &Image, pf: PixelFormat) -> Option<ViewShadowReason> {
        if !self.shared().view_reinterpretation && self.reinterprets(image, pf) {
            return Some(ViewShadowReason::Reinterpretation);
        }
        views_missed(image.plan(), pf, self.compile_device().profile())
    }

    /// Whether a view of `pf` reinterprets the components of `image`'s format, by their direct
    /// Vulkan representations.
    fn reinterprets(&self, image: &Image, pf: PixelFormat) -> bool {
        match (direct_format(image.plan().pixel_format), direct_format(pf)) {
            (Some(image), Some(view)) => !keeps_components(image, view),
            _ => false,
        }
    }

    /// Why a view of pixel format `pf` that `image` (where the view has one) does not take is
    /// refused, where it is: the device's views reinterpret no components and the view's would
    /// (`Feature::ViewFormatReinterpretation`, not representable), else the runtime holds no such
    /// view's texels apart (`Feature::TextureView`, not implemented).
    fn apart_refusal(&self, image: Option<&Image>, pf: PixelFormat) -> Unsupported {
        let reinterpreting = image.is_some_and(|image| {
            !self.shared().view_reinterpretation && self.reinterprets(truth_of(image), pf)
        });
        if reinterpreting {
            reinterpretation_refused()
        } else {
            Unsupported::not_implemented(Feature::TextureView)
        }
    }

    /// The plan of the image of a view presenting `view` of `base` in a format `base` takes no
    /// Vulkan view of (the module docs), `texture` the record of the texture `base` was made for:
    /// [`shadow_plan`](Self::shadow_plan)'s. Refused (`NotImplemented(VkImagePlaneCopy)`) for a
    /// plane of the caller's image, (`Unsupported`, as `apart_refusal` names it) where the copies
    /// cannot serve the pair, and as `new_texture` refuses the image: what
    /// [`new_view_shadow`](Self::new_view_shadow) refuses, and `Texture::view_placement` answers.
    pub(super) fn view_shadow_plan(
        &self,
        call: &'static str,
        base: &Arc<Image>,
        texture: &ViewRecord,
        view: &ViewRecord,
    ) -> Result<ShadowPlan, Error> {
        if base.caller_plane().is_some() {
            return Err(Error::NotImplemented(RuntimeGap::VkImagePlaneCopy));
        }
        let apart =
            || Error::Unsupported(self.apart_refusal(Some(base.as_ref()), view.pixel_format));
        self.shadow_plan(call, base, texture, view, apart)
    }

    /// The image of a view presenting `view` of `base` in a format `base` takes no Vulkan view of,
    /// for `reason` (the module docs): [`image_shadow`](Self::image_shadow)'s, announced once per
    /// device, as a pass's render shadow where `rendered`. Refused as
    /// [`view_shadow_plan`](Self::view_shadow_plan) refuses it.
    pub(super) fn new_view_shadow(
        &self,
        call: &'static str,
        base: &Arc<Image>,
        texture: &ViewRecord,
        view: &ViewRecord,
        reason: ViewShadowReason,
        rendered: bool,
    ) -> Result<Image, Error> {
        let plan = self.view_shadow_plan(call, base, texture, view)?;
        let image = self.planned_image_shadow(plan, base, texture, view)?;
        self.shared().counts.view_shadow_made();
        let formats = (base.plan().format, image.plan().format);
        self.degrade(Degradation::ShadowedTextureView, || {
            let object = format!(
                "a {:?} view of a {:?} {:?} texture{}",
                view.pixel_format,
                texture.pixel_format,
                texture.texture_type,
                if rendered { ", rendered into" } else { "" }
            );
            let reason = match reason {
                ViewShadowReason::Reinterpretation => {
                    "the device's Vulkan views reinterpret no image's components (its portability \
                     subset withholds imageViewFormatReinterpretation, or an adopted device's \
                     consumer did not say it enabled it): the view is an image of its own"
                        .to_owned()
                }
                ViewShadowReason::CompatibilityClass => format!(
                    "the image's Vulkan format {:?} and the view's {:?} are of different \
                     compatibility classes, so no view of the image takes the view's format: \
                     the view is an image of its own",
                    formats.0, formats.1
                ),
                ViewShadowReason::NotListed => {
                    "the texture has no PixelFormatView usage, so its image takes no view of another \
                     format than its own and its sRGB twin's (kept so for the driver's \
                     compression): the view is an image of its own"
                        .to_owned()
                }
            };
            (object, reason)
        });
        Ok(image)
    }

    /// An image presenting `view` of `base` in memory of its own, a shadow of `base`'s texels
    /// (the module docs): an image of the view's format and usage and the shape of `texture`, the
    /// record of the texture `base` was made for, whose copies move the levels and layers `view`
    /// presents (`ShadowSource::Image`). Its truth is a [`Backing::PrivateImage`] where only the
    /// runtime's commands write `base` (a tracked `Private` or `Memoryless` image in memory of its
    /// own), and then it may stand across command buffers until `base`'s `VkImage` is handed out
    /// (`shadow_standing.rs`). Refused `refused()` where the copies cannot serve the pair
    /// (no transfer usage, formats whose copy units differ), and as `new_texture` refuses the
    /// image.
    pub(super) fn image_shadow(
        &self,
        call: &'static str,
        base: &Arc<Image>,
        texture: &ViewRecord,
        view: &ViewRecord,
        refused: impl Fn() -> Error,
    ) -> Result<Image, Error> {
        let plan = self.shadow_plan(call, base, texture, view, refused)?;
        self.planned_image_shadow(plan, base, texture, view)
    }

    /// The plan of [`image_shadow`](Self::image_shadow)'s image, made of nothing yet: refused
    /// `refused()` where the copies cannot serve the pair (no transfer usage, formats whose copy
    /// units differ, a copy through a buffer of more than one unit), and as `new_texture` refuses
    /// the image.
    pub(super) fn shadow_plan(
        &self,
        call: &'static str,
        base: &Image,
        texture: &ViewRecord,
        view: &ViewRecord,
        refused: impl Fn() -> Error,
    ) -> Result<ShadowPlan, Error> {
        let layout = TextureLayout {
            texture_type: texture.texture_type,
            pixel_format: view.pixel_format,
            extent: texture.image_extent,
            levels: texture.level_count,
            layers: texture.layer_count,
            samples: texture.samples,
        };
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let plan = plan_image(&layout, view.usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        let transfer = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        let units = copy_units(base.plan().format, base.plan().pixel_format);
        let same_units = copy_units(plan.format, plan.pixel_format) == units && !units.is_empty();
        if !plan.usage.contains(transfer) || !base.plan().usage.contains(transfer) || !same_units {
            return Err(refused());
        }
        // Vulkan copies no texels between an alpha format and another: the copies go through a
        // buffer, which takes one copy unit.
        let through = match copies_between(base.plan().format, plan.format) {
            true => None,
            false => match units.as_slice() {
                &[(_, bytes, block)] => Some((bytes, block)),
                _ => return Err(refused()),
            },
        };
        Ok(ShadowPlan {
            layout,
            plan,
            through,
        })
    }

    /// The image of [`image_shadow`](Self::image_shadow), of `planned`
    /// ([`shadow_plan`](Self::shadow_plan)'s). Refused as `new_texture` refuses the image.
    fn planned_image_shadow(
        &self,
        planned: ShadowPlan,
        base: &Arc<Image>,
        texture: &ViewRecord,
        view: &ViewRecord,
    ) -> Result<Image, Error> {
        let ShadowPlan {
            layout,
            plan,
            through,
        } = planned;
        let aspects = texture_aspects(&plan);
        let range = |aspects| UseRange::Image {
            aspects,
            levels: view.base_level..view.base_level + view.level_count,
            layers: view.base_layer..view.base_layer + view.layer_count,
            place: ImagePlace::WHOLE,
        };
        let volume = texture.texture_type == TextureType::Type3D;
        let regions: Vec<vk::ImageCopy> = (view.base_level..view.base_level + view.level_count)
            .map(|level| {
                let [width, height, depth] =
                    texture.image_extent.map(|axis| (axis >> level).max(1));
                let layers = vk::ImageSubresourceLayers {
                    aspect_mask: aspects,
                    mip_level: level,
                    base_array_layer: view.base_layer,
                    layer_count: view.layer_count,
                };
                vk::ImageCopy {
                    src_subresource: layers,
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: layers,
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: if volume { depth } else { 1 },
                    },
                }
            })
            .collect();
        // Between an alpha format and another, through a buffer, each region packed after the
        // last at a multiple of the copy unit and of 4.
        let through = match through {
            None => None,
            Some((bytes, [block_width, block_height])) => {
                let align = u64::from(bytes) * 4;
                let mut length = 0;
                let mut placed = Vec::with_capacity(regions.len());
                for region in &regions {
                    let e = region.extent;
                    let blocks = u64::from(e.width.div_ceil(block_width))
                        * u64::from(e.height.div_ceil(block_height))
                        * u64::from(e.depth)
                        * u64::from(region.src_subresource.layer_count);
                    placed.push((length, region));
                    length = (length + blocks * u64::from(bytes)).next_multiple_of(align);
                }
                let buffer = self.own_buffer(length.max(align), ResourceOptions::private())?;
                let (_, start) = buffer.at(0);
                let regions = placed
                    .into_iter()
                    .map(|(offset, region)| vk::BufferImageCopy {
                        buffer_offset: start + offset,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: region.src_subresource,
                        image_offset: region.src_offset,
                        image_extent: region.extent,
                    })
                    .collect();
                Some(ImageThrough { buffer, regions })
            }
        };
        let identity = Identity {
            key: ResourceKey::new(),
            tracked: base.identity.tracked,
        };
        let (truth, truth_key, truth_range) = (
            Arc::clone(base) as Arc<dyn ImageTruth>,
            base.identity.key,
            range(texture_aspects(base.plan())),
        );
        // Only the runtime's commands write a tracked private image in memory of its own that no
        // other process opens: the CPU reaches none of its texels, and no other resource lies in
        // its memory.
        let private = matches!(
            base.options.storage_mode(),
            Ok(StorageMode::Private | StorageMode::Memoryless)
        );
        let own = matches!(
            base.owner,
            Owner::Runtime {
                shareable: false,
                ..
            }
        );
        let backing = match private && own && base.identity.tracked {
            true => Backing::PrivateImage,
            false => Backing::Image,
        };
        let standing =
            StandingAcross::over_image(Arc::clone(&truth), backing == Backing::PrivateImage);
        let shadow_range = range(aspects);
        let mut descriptor = TextureDescriptor::texture_2d(view.pixel_format, 1, 1, false);
        descriptor.resource_options = base.options;
        descriptor.allow_gpu_optimized_contents = base.allow_gpu_optimized_contents;
        let image =
            self.new_image_owned(plan, &layout, &descriptor, identity, |placement, raw| {
                let copies = ShadowCopies {
                    source: ShadowSource::Image {
                        image: truth,
                        key: truth_key,
                        range: truth_range,
                        shadow: raw,
                        regions,
                        through,
                    },
                    shadow_key: identity.key,
                    shadow_range,
                    tracked: identity.tracked,
                    drops_writes: false,
                    levels: None,
                    decode: Default::default(),
                    backing,
                    merge_scratch: ScratchSpares::default(),
                    stood: AtomicU64::new(0),
                    standing,
                };
                Owner::ViewShadow(Box::new(ViewShadow {
                    placement,
                    base: Arc::clone(base),
                    copies: Arc::new(copies),
                }))
            })?;
        // A view of an image presenting a stand-in, its texture's bytes detached, presents the
        // stand-in of its own plan too, keeping its own image for its copies.
        if base.presents_stand_in() {
            let stand_in = self.stand_in(image.plan())?;
            drop(image.present_stand_in(stand_in));
        }
        Ok(image)
    }
}
