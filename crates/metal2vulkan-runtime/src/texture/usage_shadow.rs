//! Textures bound beyond their declared usage where their image lacks the usage (the orchestrator's
//! ruling of 2026-10-07).
//!
//! Plain Metal writes a texture made without `ShaderWrite` and reads (or samples) one made without
//! `ShaderRead`, as the texture with the usage (its validation stops each). A texture's image takes
//! the usage its descriptor lacks where that costs it nothing (`Device::with_undeclared_usage`),
//! and an image in place over host pages where the device's check passed for what the usage lets
//! reach the pages (`texture/in_place_image.rs`); elsewhere the image keeps the declared usage, so
//! a valid program's texture keeps the driver's compression, and a slot binding that needs the
//! usage takes an image of its own instead:
//!
//! - **The shadow** is an image of the texture's format and shape with both shader usages, made
//!   at the first binding that needs it and kept with the texture object. Its truth is the
//!   texture's: the image's texels in the levels and layers the texture presents
//!   (`ShadowSource::Image`, as a shadowed view's, `texture/view_shadow.rs`), or, for a texture
//!   over a buffer (an aliasing or a shadow image), the buffer's bytes (`ShadowSource::Bytes`,
//!   a second shadow of them, `shadow.rs`). The encoders copy the truth in before a command that
//!   binds the shadow and back after it writes through it, by `shadow.rs`'s rules.
//! - **Bindings.** A slot binding (a compute dispatch's, a draw's, a tile dispatch's) whose
//!   descriptor needs the storage or sampled usage the texture's image and its usage both lack
//!   writes the shadow's view and names the shadow in its uses; every other binding of the
//!   texture names its own image. A command that writes the texture through the shadow and
//!   another way is refused as `shadow.rs` refuses it for an image's shadow.
//! - **Not served.** A texture made resident for a function reading handles (the device's heap
//!   sets bind its own image), a sparse texture, a heap's, a tiled texture of more than one level
//!   over a buffer (`Buffer::new_tiled_texture`: the bytes hold its first level alone, and its
//!   image is a shadow of them already), and a pair the copies cannot serve stay
//!   `NotImplemented(UndeclaredTextureUsage)`.
//! - **Announced** once per device as `undeclared-usage-shadow` (performance), at a shadow's first
//!   binding beyond the texture's usage.
//! - **Attachment reads.** A draw's sampled slot binding of a texture that presents a color
//!   attachment its render pass instance renders takes the shadow too, whatever its usage
//!   (`texture/attached_reads.rs`), and announces nothing: a texture whose shadow is not served
//!   binds its own image there, as before.
//! - **YCbCr writes**. A 4:4:4 two-plane YCbCr texture whose writes Metal stores into its planes
//!   (`metal::format::ycbcr_writes_stored`), whose image has both shader usages, holds its write
//!   shadow here instead (`texture/ycbcr_image.rs`, `shadow.rs`): every storage binding of it
//!   writes the shadow's view and names the shadow in its uses, and every other binding names its
//!   own image. It is exact and announces nothing.

use super::{Image, Owner, Storage, Texture};
use crate::binding_view::binding_view;
use crate::degradation::Degradation;
use crate::device::lock;
use crate::error::{internal, Error, RuntimeGap};
use crate::image_plan::plan_image;
use crate::resource::ResourceKey;
use crate::shadow::ShadowCopies;
use crate::texture::{image_error, texture_aspects, view_error};
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use crate::tracker::{Access, Use};
use crate::values::{CpuCacheMode, ResourceOptions, StorageMode};
use crate::view_plan::ViewRecord;
use ash::vk;
use metal2vulkan::bind::ImageViewDesc;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::ycbcr_writes_stored;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::vk::ImageUsageFlags;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A texture's image of its own with both shader usages, and the copies between it and the
/// texture's truth (the module docs).
pub(crate) struct UsageShadow {
    image: Arc<Image>,
    copies: Arc<ShadowCopies>,
    /// What the shadow stands for, as its notice names it.
    truth: &'static str,
    /// A binding beyond the texture's usage took it, and announced so.
    announced: AtomicBool,
}

/// What a binding that takes the texture's shadow needs it for (the module docs).
#[derive(Clone, Copy)]
enum Need {
    /// A usage the texture's image and its usage both lack.
    Usage,
    /// A sampled binding of a texture that presents a color attachment of the draw's render pass
    /// instance.
    AttachedRead,
    /// A storage binding of a YCbCr texture whose writes Metal stores into its planes.
    YCbCrWrites,
}

impl Need {
    /// Why the shadow was made, as the shadow log says it.
    fn why(self) -> &'static str {
        match self {
            Need::Usage => "a binding needs a usage that the texture's image and its usage lack",
            Need::AttachedRead => {
                "a draw samples it where it presents a color attachment of the draw's own render \
                 pass instance (an attached read)"
            }
            Need::YCbCrWrites => {
                "a storage binding writes a YCbCr texture whose writes Metal stores into its planes"
            }
        }
    }
}

/// Why a binding the shadow does not serve is refused: the runtime's gap.
fn refused() -> Error {
    Error::Unsupported(Unsupported::not_implemented(
        Feature::UndeclaredTextureUsage,
    ))
}

impl Texture {
    /// Whether a slot binding of the texture as a storage image (`storage`) or a sampled one
    /// takes the shadow: the texture's image lacks the binding's usage, and so does the
    /// texture's own (Metal's) usage, where `binding_view` refuses the image
    /// (`UndeclaredTextureUsage`).
    pub(in crate::texture) fn needs_usage_shadow(&self, storage: bool) -> bool {
        let (vk, metal) = match storage {
            true => (ImageUsageFlags::STORAGE, TextureUsage::SHADER_WRITE),
            false => (ImageUsageFlags::SAMPLED, TextureUsage::SHADER_READ),
        };
        match self.storage() {
            Storage::Image(image)
            | Storage::Alias { image, .. }
            | Storage::Shadow { image, .. } => {
                !image.plan().usage.contains(vk) && !self.0.record.usage.contains(metal)
            }
            Storage::Texels(_) => false,
        }
    }

    /// The `VkImageView` a slot binding write of `call` needs of the texture: `desc` the binding
    /// plan's view, `storage` whether the write is a storage image's, `attached` whether the
    /// texture presents a color attachment of the draw's render pass instance
    /// (`texture/attached_reads.rs`). The shadow's where the binding takes it
    /// ([`needs_usage_shadow`](Self::needs_usage_shadow), or a sampled binding `attached`), else
    /// the texture's own (`bound_view`). Refused as `binding_view` refuses, and where the shadow
    /// is not served beyond the texture's usage (the module docs).
    pub(crate) fn slot_view(
        &self,
        call: &'static str,
        desc: &ImageViewDesc,
        storage: bool,
        attached: bool,
    ) -> Result<vk::ImageView, Error> {
        // The write shadow's texels are written, never read: no alpha reads 1 there.
        if self.takes_ycbcr_writes(storage) {
            let shadow = self.usage_shadow(call, Need::YCbCrWrites)?;
            let record = ViewRecord {
                alpha_reads_one: false,
                ..self.0.record.bound()
            };
            let view = binding_view(shadow.image.plan(), &record, desc, storage)
                .map_err(|refusal| view_error(call, refusal))?;
            return shadow.image.view(&view);
        }
        let Some(shadow) = self.slot_shadow(call, storage, attached)? else {
            return self.bound_view(call, desc, storage);
        };
        let view = binding_view(shadow.image.plan(), &self.0.record.bound(), desc, storage)
            .map_err(|refusal| view_error(call, refusal))?;
        shadow.image.view(&view)
    }

    /// The shadow a slot binding of the kind `storage` takes in the texture's place, `attached`
    /// as [`slot_view`](Self::slot_view) takes it: beyond the texture's usage, the shadow,
    /// announced at its first such binding (refused where it is not served); for a sampled
    /// binding of a texture that presents an attachment of the draw's instance, the shadow where
    /// it is served (the module docs); else `None`.
    fn slot_shadow(
        &self,
        call: &'static str,
        storage: bool,
        attached: bool,
    ) -> Result<Option<Arc<UsageShadow>>, Error> {
        if self.needs_usage_shadow(storage) {
            let shadow = self.usage_shadow(call, Need::Usage)?;
            if !shadow.announced.swap(true, Ordering::Relaxed) {
                self.announce_usage_shadow(shadow.truth);
            }
            return Ok(Some(shadow));
        }
        if !attached || storage {
            return Ok(None);
        }
        match self.usage_shadow(call, Need::AttachedRead) {
            Ok(shadow) => Ok(Some(shadow)),
            Err(Error::NotImplemented(_) | Error::Unsupported(_)) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Whether a slot binding of the kind `storage` takes the texture's YCbCr write shadow
    /// (`texture/ycbcr_image.rs`) in this slot's place: a storage binding of a 4:4:4 two-plane
    /// YCbCr texture whose writes Metal stores into its planes (`ycbcr_writes_stored`), which
    /// the binding plan gives a view of the shadow's format (`StorageImageClass::YCbCrWrite`).
    fn takes_ycbcr_writes(&self, storage: bool) -> bool {
        storage && ycbcr_writes_stored(self.0.record.pixel_format)
    }

    /// A slot binding's use of everything the texture presents, reading by `read` and writing by
    /// `write`: of the shadow where a binding of the kind `storage`, `attached` as
    /// [`slot_view`](Self::slot_view) takes it, says takes it, else the texture's own
    /// (`bound_use`).
    pub(crate) fn slot_use(
        &self,
        call: &'static str,
        storage: bool,
        attached: bool,
        (read, write): (Access, Access),
    ) -> Result<Use<ResourceKey>, Error> {
        let shadow = match self.takes_ycbcr_writes(storage) {
            true => Some(self.usage_shadow(call, Need::YCbCrWrites)?),
            false => self.slot_shadow(call, storage, attached)?,
        };
        let Some(shadow) = shadow else {
            return Ok(self.bound_use(read, write));
        };
        Ok(Use {
            key: shadow.copies.shadow_key,
            tracked: shadow.copies.tracked,
            range: shadow.copies.shadow_range.clone(),
            read,
            write,
        })
    }

    /// The copies of the shadow a sampled slot binding of `call` takes in the texture's place
    /// where it presents a color attachment of the draw's render pass instance (the module docs),
    /// made now where no binding made it yet, as that binding makes it. `None` where it is not
    /// served or could not be made: the binding then binds as it would, and answers for the
    /// failure.
    pub(crate) fn attached_shadow_copies(&self, call: &'static str) -> Option<Arc<ShadowCopies>> {
        (self.usage_shadow(call, Need::AttachedRead).ok()).map(|shadow| Arc::clone(&shadow.copies))
    }

    /// The copies of the texture's shadow, where a binding made it.
    pub(crate) fn usage_shadow_copies(&self) -> Option<Arc<ShadowCopies>> {
        (self.0.usage_shadow.get()).map(|shadow| Arc::clone(&shadow.copies))
    }

    /// The texture's shadow, made at its first need (the module docs), `need` a binding's that
    /// made it, one binding at a time; a shadow made counts among its device's
    /// (`DeviceShared::usage_shadows_made`), which a kept list of resident textures' shadows is
    /// listed again at (`shadow::ResidentShadows`), and is named in the shadow log
    /// (`shadow_log.rs`).
    fn usage_shadow(&self, call: &'static str, need: Need) -> Result<Arc<UsageShadow>, Error> {
        if let Some(shadow) = self.0.usage_shadow.get() {
            return Ok(Arc::clone(shadow));
        }
        let _making = lock(&self.0.usage_shadow_making);
        if let Some(shadow) = self.0.usage_shadow.get() {
            return Ok(Arc::clone(shadow));
        }
        // A converted texture's image has both shader usages: the slot holds its write shadow.
        let shadow = match ycbcr_writes_stored(self.0.record.pixel_format) {
            true => {
                let (image, copies) = self.ycbcr_write_shadow(call)?;
                UsageShadow {
                    image: image.shared(),
                    copies: Arc::new(copies),
                    truth: "its image",
                    announced: AtomicBool::new(false),
                }
            }
            false => self.new_usage_shadow(call)?,
        };
        self.log_usage_shadow(call, need);
        let shadow = Arc::new(shadow);
        if self.0.usage_shadow.set(Arc::clone(&shadow)).is_err() {
            return Err(internal("a usage shadow made twice"));
        }
        // Counted once it can be read, so a list made after the count reads it.
        (self.device().shared().usage_shadows_made).fetch_add(1, Ordering::Release);
        Ok(shadow)
    }

    /// A new shadow of the texture (the module docs).
    fn new_usage_shadow(&self, call: &'static str) -> Result<UsageShadow, Error> {
        self.check_plane_copied()?;
        let device = self.device();
        let record = &self.0.record.bound();
        let wider = ViewRecord {
            usage: record.usage | TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            ..*record
        };
        // A rotated texture's functions read its stored frame and their writes are dropped
        // (`texture/rotation.rs`); a binding beyond its usage is not measured.
        if record.presented.rotation != 0 {
            return Err(Error::NotImplemented(RuntimeGap::RotatedTexture));
        }
        let (image, copies, truth) = match self.storage() {
            Storage::Image(image) => {
                let base = match &image.owner {
                    Owner::ViewShadow(shadow) => &shadow.base,
                    Owner::Runtime { .. } | Owner::Caller(_) | Owner::HeapUnaliased(_) => image,
                    Owner::Buffer { .. }
                    | Owner::Sparse(_)
                    | Owner::Heap(_)
                    | Owner::HeapShadow(_) => return Err(refused()),
                };
                let image =
                    device.image_shadow(call, base, &self.root().0.record, &wider, refused)?;
                let copies = match &image.owner {
                    Owner::ViewShadow(shadow) => Arc::clone(&shadow.copies),
                    _ => return Err(internal("an image shadow without its copies")),
                };
                (image.shared(), copies, "its image")
            }
            // A tiled texture's bytes hold its first level alone (the module docs).
            Storage::Shadow { image, .. } if image.plan().levels > 1 => return Err(refused()),
            Storage::Alias { texels, .. } | Storage::Shadow { texels, .. } => {
                let layout = TextureLayout {
                    texture_type: record.texture_type,
                    pixel_format: record.pixel_format,
                    extent: record.image_extent,
                    levels: record.level_count,
                    layers: record.layer_count,
                    samples: record.samples,
                };
                let facts = device
                    .shared()
                    .textures
                    .facts(device.compile_device().profile());
                let plan = plan_image(&layout, wider.usage, false, &facts)
                    .map_err(|refusal| image_error(call, refusal))?;
                let transfer = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
                if !plan.usage.contains(transfer) {
                    return Err(refused());
                }
                // In memory of its own: nothing maps it, its truth is the buffer's bytes.
                let mut descriptor =
                    TextureDescriptor::texture_2d(record.pixel_format, 1, 1, false);
                let hazard = texels.options().hazard_tracking_mode()?;
                descriptor.resource_options =
                    ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
                let image = device.new_image(plan, &layout, &descriptor, hazard)?;
                let mut copies = texels.shadow_copies(
                    image.raw.get(),
                    image.identity,
                    texture_aspects(image.plan()),
                    (None, None),
                )?;
                // The texture's own image lies in the bytes too where it aliases them.
                if matches!(self.storage(), Storage::Alias { .. }) {
                    copies.backing = copies.backing.aliased();
                }
                (
                    image.shared(),
                    Arc::new(copies),
                    "the image over its buffer's bytes",
                )
            }
            Storage::Texels(_) => return Err(refused()),
        };
        Ok(UsageShadow {
            image,
            copies,
            truth,
            announced: AtomicBool::new(false),
        })
    }

    /// Name the shadow a binding's `need` of `call` made in the shadow log (`shadow_log.rs`):
    /// its truth, and what the texture's own image is.
    fn log_usage_shadow(&self, call: &'static str, need: Need) {
        if !crate::shadow_log::enabled() {
            return;
        }
        let record = &self.0.record.bound();
        let [width, height, _] = record.image_extent;
        let texture = crate::shadow_log::Shadowed {
            pixel_format: record.pixel_format,
            extent: [width, height],
            levels: record.level_count,
            usage: record.usage,
        };
        let feedback = ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        let (texels, own) = match self.storage() {
            Storage::Alias { image, texels } => {
                let usage = match image.plan().usage.contains(feedback) {
                    true => "with the feedback loop usage",
                    false => "without the feedback loop usage",
                };
                (
                    Some(&**texels),
                    format!("an aliasing image over the bytes, {usage}"),
                )
            }
            Storage::Shadow { texels, .. } => (Some(&**texels), "a shadow of the bytes".to_owned()),
            Storage::Image(_) | Storage::Texels(_) => (None, "an image of its own".to_owned()),
        };
        let why = format!(
            "a usage shadow: {}; the texture's own image is {own}",
            need.why()
        );
        crate::shadow_log::made_of(self.device(), call, &texture, texels, &why);
    }

    /// Announce a binding beyond the texture's usage through its shadow, `truth` what the
    /// shadow stands for (the module docs).
    fn announce_usage_shadow(&self, truth: &str) {
        let record = &self.0.record.bound();
        self.device()
            .degrade(Degradation::UndeclaredUsageShadow, || {
                let object = format!(
                    "a {:?} {:?} texture of usage {:#x}",
                    record.pixel_format,
                    record.texture_type,
                    record.usage.as_raw()
                );
                let reason = format!(
                    "a function binds it as its usage does not declare, and {truth} lacks that \
                     usage (adding it would cost the image its size or the driver's compression \
                     of a texture of its declared usage): the binding takes an image of its own \
                     with both shader usages"
                );
                (object, reason)
            });
    }
}
