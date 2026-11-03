//! Shared textures and their in-process handles (`newSharedTextureWithDescriptor:`,
//! `isShareable`, `newSharedTextureHandle`, `newSharedTextureWithHandle:`).
//!
//! A handle names a texture's image within this process: opening it gives a new texture object
//! over the same image, as Metal's does. Cross-process export is not built (not scheduled). Metal's
//! answers, measured on an Apple M3 Max (macOS 26.7, plain and under API validation):
//! - only a `Private` texture is made shareable: any other storage mode answers nil, and API
//!   validation stops it ("Requested storage mode is not allowed for Shared Textures"); a texture
//!   buffer and a multisample texture are made, and a texture opened from a texture buffer's handle
//!   reads what a kernel wrote through the texture buffer (a Metal probe on the M3 Max and an M4).
//!   The storage rule was re-read on the M3 Max under macOS 27.0.1, one cell per process and eight
//!   processes a cell: `Shared`, `Managed` and `Memoryless` answered nil 8 of 8 for `RGBA8Unorm`,
//!   `R16Uint` and each of the four padded 10-bit formats, and `Private` a texture 8 of 8 for
//!   each, the padded ones among them;
//! - a texture made otherwise, a texture over a buffer, and a view of a shareable texture are
//!   not shareable, and their handle is nil;
//! - each open answers a new texture object, shareable, with the texture's properties (its
//!   label too) and no parent, and a resource ID of its own.
//!
//! The opened texture and the texture share one image, or a texture buffer's one buffer of
//! texels, and so one resource for hazard tracking: the runtime orders their accesses as one
//! texture's.

use super::super::address::RangeToken;
use super::super::buffer_texture::TexelsHome;
use super::super::error::{invalid, nil, Error, Problem};
use super::super::image_plan::plan_image;
use super::super::resource_ids::IdPlacement;
use super::super::texture_descriptor::TextureDescriptor;
use super::super::values::StorageMode;
use super::super::view_plan::ViewRecord;
use super::super::Device;
use super::{image_error, IdUse, Owner, Storage, Texture};
use metal2vulkan::metal::TextureType;
use std::sync::Arc;

/// An in-process handle of a shareable texture (`MTLSharedTextureHandle`): opened with
/// [`Device::new_shared_texture_with_handle`], it gives a texture over the same image. Cheap to
/// clone (a retain); `Send + Sync`.
#[derive(Clone)]
pub struct SharedTextureHandle(Texture);

impl std::fmt::Debug for SharedTextureHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedTextureHandle")
            .field("texture", &self.0)
            .finish()
    }
}

impl SharedTextureHandle {
    /// `device`: the device the texture was made by, the only one that opens the handle.
    pub fn device(&self) -> &Device {
        self.0.device()
    }
}

impl Device {
    /// `newSharedTextureWithDescriptor:`: a texture of `descriptor` that has handles
    /// ([`Texture::new_shared_texture_handle`]).
    ///
    /// A texture buffer is made over a buffer of its own, whose handles open texture buffers over
    /// the same texels.
    ///
    /// Refused: as [`Device::new_texture`] refuses the descriptor; storage other than `Private`
    /// (`Nil`, `StorageMode`), which plain Metal answers nil and its validation stops, before every
    /// rule of the descriptor; a format of which Metal makes no texture alone (`Nil`) before the
    /// multisample sample count's rule; a swizzle with writes under `write_swizzle_enabled` too
    /// (`Invalid`, `SwizzleWithWrites`), which Metal's validation stops here alone.
    pub fn new_shared_texture(&self, descriptor: &TextureDescriptor) -> Result<Texture, Error> {
        const CALL: &str = "new_shared_texture";
        // Plain Metal makes a shareable texture without `framebufferOnly`, whatever the
        // descriptor says, so a rotated one asserts as one without it does.
        let descriptor = &TextureDescriptor {
            framebuffer_only: false,
            ..self.texture_options(CALL, descriptor)?
        };
        self.check_texture_placement(
            CALL,
            descriptor.placement,
            descriptor.usage,
            descriptor.texture_type,
        )?;
        let options = descriptor.resource_options;
        let storage = options.storage_mode()?;
        let hazard = options.hazard_tracking_mode()?;
        options.cpu_cache_mode()?;
        // Plain Metal answers nil for storage other than `Private` before any rule of the
        // descriptor it stops (a width of 0, a side past the largest, too many levels, a swizzle
        // with writes, one sample in a multisample type: nil in `Shared`, `Managed` and
        // `Memoryless` storage, a stop in `Private`; an M3 Max under macOS 26.7 and an M4 under
        // 27.0.1).
        if storage != StorageMode::Private {
            return Err(nil(CALL, Problem::StorageMode));
        }
        let layout = descriptor
            .check_shared(self.properties().texture_rules())
            .map_err(|refused| refused.error(CALL))?;
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let id_use = IdUse::Program(descriptor.placement);
        if layout.texture_type == TextureType::TextureBuffer {
            let texture =
                self.new_texture_buffer(descriptor, &layout, id_use, TexelsHome::Shared)?;
            self.take_device_sampler();
            return Ok(texture);
        }
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let plan = plan_image(&layout, descriptor.image_usage(), false, &facts)
            .map_err(|refusal| image_error(CALL, refusal))?;
        let mut image = self.new_texture_image(plan, &layout, descriptor, hazard)?;
        if let Owner::Runtime { shareable, .. } = &mut image.owner {
            *shareable = true;
        }
        let image = self.with_decoded_twin(image, &layout, descriptor, true)?;
        let record = ViewRecord::of_descriptor(&layout, descriptor);
        let image = image.shared();
        // Its last step that may fail, before the texture takes its ID (`resource_ids.rs`).
        image.start_decoded_twin()?;
        let texture = Texture::made(self, Storage::Image(image), record, None, id_use);
        // A shared texture takes the device's sampler ID as any texture of the device's does
        // (`resource_ids.rs`); one opened from a handle needs none, the handle's texture having
        // taken it on this device.
        self.take_device_sampler();
        Ok(texture)
    }

    /// [`Device::new_shared_texture`] with `token` (a runtime extension), dropped once the
    /// runtime frees the texture's image ([`RangeToken`]): after the texture, every view of it,
    /// every texture opened from its handles, and the command buffers that hold any of them are
    /// gone; a texture buffer's once its texels' buffer is free. A refused call's drops before it
    /// returns.
    pub fn new_shared_texture_with_range_token(
        &self,
        descriptor: &TextureDescriptor,
        token: RangeToken,
    ) -> Result<Texture, Error> {
        let texture = self.new_shared_texture(descriptor)?;
        texture.keep_range_token(token)?;
        Ok(texture)
    }

    /// `newSharedTextureWithHandle:`: a new texture over the image (or a texture buffer's texels)
    /// of `handle`'s texture, with its properties, its label at the open, and no parent,
    /// shareable as it is, sharing its purgeable state; its resource ID at `placement`
    /// (`resource_ids.rs`), as `newSharedTextureWithHandle:withResourceIndex:` (Apple's private
    /// call, a runtime extension) places it.
    ///
    /// Refused: a handle of another device's texture (`Invalid`, `ForeignObject`): Metal's handle
    /// "can only be used with this device" (`MTLTexture.h`); a placement as
    /// [`Device::new_texture`] refuses one.
    pub fn new_shared_texture_with_handle(
        &self,
        handle: &SharedTextureHandle,
        placement: IdPlacement,
    ) -> Result<Texture, Error> {
        const CALL: &str = "new_shared_texture_with_handle";
        let texture = &handle.0;
        if !texture.device().same(self) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        let record = &texture.0.record;
        self.check_texture_placement(CALL, placement, record.usage, record.texture_type)?;
        let storage = match &texture.0.storage {
            Storage::Image(image) => Storage::Image(Arc::clone(image)),
            Storage::Texels(texels) => Storage::Texels(Arc::clone(texels)),
            Storage::Alias { .. } | Storage::Shadow { .. } => {
                unreachable!("only a texture with an image or texels of its own has a handle")
            }
        };
        let id_use = IdUse::Program(placement);
        let opened = Texture::made(self, storage, texture.0.record, None, id_use);
        // Metal's opened texture starts with the texture's label, and the two change apart.
        if let Some(label) = texture.label() {
            opened.set_label(&label);
        }
        Ok(opened)
    }
}

impl Texture {
    /// `isShareable`: a texture of [`Device::new_shared_texture`], or one opened from a handle;
    /// never a view.
    pub fn is_shareable(&self) -> bool {
        self.0.parent.is_none()
            && match &self.0.storage {
                Storage::Image(image) => {
                    matches!(
                        image.owner,
                        Owner::Runtime {
                            shareable: true,
                            ..
                        }
                    )
                }
                Storage::Texels(texels) => texels.home == TexelsHome::Shared,
                Storage::Alias { .. } | Storage::Shadow { .. } => false,
            }
    }

    /// `newSharedTextureHandle`: an in-process handle of this texture.
    ///
    /// Refused for a texture that is not shareable (`Nil`, `NotShareable`), for which Metal
    /// answers nil.
    pub fn new_shared_texture_handle(&self) -> Result<SharedTextureHandle, Error> {
        if !self.is_shareable() {
            return Err(nil("new_shared_texture_handle", Problem::NotShareable));
        }
        Ok(SharedTextureHandle(self.clone()))
    }
}
