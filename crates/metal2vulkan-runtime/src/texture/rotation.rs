//! Rotated and framebuffer-only textures (`TextureDescriptor::rotation`, `framebuffer_only`,
//! `is_drawable`; the copies are `rotated.rs`'s). Plain Metal's answers, measured on an M4 under
//! macOS 27.0.1 (each the same in two runs, a GPU's two or three times), are the rules:
//!
//! - **Render passes** store the pass's texel `(x, y)` where Metal turns it into the plane
//!   (`rotated.rs`), and resolve into it likewise. They load it from there at rotations 1 to 3,
//!   and from the plane's `(y, x)` at rotation 4, so a texel loaded and stored again lands
//!   mirrored, in a pass that draws; one that draws nothing leaves every texel in place, as
//!   Metal's does (`rotated::Turn`, and the tiles no draw reaches,
//!   `Degradation::RotatedLoadTiles`). They render into the texture's twin
//!   ([`Texture::pass_texture`]): an image of the pass's frame, a shadow of the plane's bytes
//!   whose copies turn the texels (`rotated.rs`), made at the first pass that needs it and kept
//!   with the texture's texels.
//! - **CPU access and blits** (`getBytes`, `replaceRegion`, blits in and out) address the plane as
//!   a 2D texture of the texture's own extent at the plane's offset and rows, turned or not. The
//!   texture's texels lie there (`rotated_rows`), their rows a row apart or further: a rotated
//!   plane's `bytesPerRow` is a multiple of 16 and covers the texture's own row
//!   (`buffer_texture::turned_plane`, `covers_its_row`), so no two of them share a byte
//!   (`tests::buffer_texture::no_rotated_plane_s_gpu_rows_share_bytes`). Where those rows run
//!   past the buffer, whose bytes Metal reads past the stored frame, the runtime has no bytes,
//!   and those uses are refused (`NotImplemented(RotatedTexture)`,
//!   [`Texture::check_rotated_reach`]).
//! - **Functions** see the stored frame unturned: `get_width()` and `get_height()` answer the
//!   extent Metal stores the texture as, `read()` at `(x, y)` (in a fragment, a kernel, or a
//!   `read_write` kernel) gives the plane's texel `(x, y)` of that extent, a position past it
//!   reading 0, and `sample()` takes normalized coordinates over it. Their writes (`write()`, a
//!   `read_write` kernel's too) reach no byte of the plane, and a later read sees the plane's
//!   bytes. The texture's image is that frame's, a shadow of the plane's bytes whose writes are
//!   dropped (`shadow.rs`, `ShadowCopies::drops_writes`).
//! - **Views** (`newTextureViewWithPixelFormat:`) of a rotated texture: plain Metal makes them
//!   and they act as the texture does, in their own format: the texture's frame extent, rotation
//!   and framebuffer-only flag; passes store, load and resolve them turned as the texture's
//!   (rotation 4's mirror too), functions read the stored frame unturned, and CPU access reads
//!   the plane's rows (alike; Metal's validation stops every view of a framebuffer-only texture,
//!   "texture view creation not allowed from frameBufferOnly textures", at rotation 0 too).
//!   Measured, and made: the whole texture, its type and no swizzle, as one 8-bit RGBA format of
//!   another ([`MEASURED_VIEWS`]), with or without `PixelFormatView` usage. A view shares the
//!   texture's plane rows, so its passes render through a twin of its format kept with them, and
//!   its functions bind the texture's image of the stored frame where that image takes the view's
//!   format, else an image of the stored frame of the view's own, a shadow of the same bytes. Every
//!   other view of a rotated texture is refused (`NotImplemented(RotatedTexture)`).

use super::super::buffer_texture::Texels;
use super::super::degradation::Degradation;
use super::super::error::{internal, Error, RuntimeGap};
use super::super::image_plan::plan_image;
use super::super::rotated;
use super::super::texture_descriptor::{texture_usage, TextureDescriptor, TextureLayout};
use super::super::values::ResourceOptions;
use super::super::view_plan::{ViewRecord, ViewRequest};
use super::super::Device;
use super::{image_error, texture_aspects, IdUse, Image, Storage, Texture};
use metal2vulkan::metal::format::is_depth_stencil;
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::ImageUsageFlags;
use std::sync::Arc;

/// The formats a rotated texture is measured to take views between (the module docs): the
/// 8-bit RGBA formats, each viewed as each.
pub(super) const MEASURED_VIEWS: [PixelFormat; 4] = [
    PixelFormat::BGRA8Unorm,
    PixelFormat::BGRA8UnormSRGB,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA8UnormSRGB,
];

/// Whether `request` is a view of a rotated texture of `source` that plain Metal is measured
/// to make (the module docs): the whole texture, of its type and swizzle, as one of
/// [`MEASURED_VIEWS`] of another.
pub(super) fn view_measured(source: &ViewRecord, request: &ViewRequest) -> bool {
    *request == ViewRequest::of_format(source, request.pixel_format)
        && MEASURED_VIEWS.contains(&source.pixel_format)
        && MEASURED_VIEWS.contains(&request.pixel_format)
}

impl Texture {
    /// `framebufferOnly`: the descriptor's
    /// ([`TextureDescriptor::framebuffer_only`](crate::TextureDescriptor::framebuffer_only)) for
    /// a texture over an IOSurface plane, and false for every other texture, which Metal makes
    /// without it.
    pub fn framebuffer_only(&self) -> bool {
        self.record().presented.framebuffer_only
    }

    /// `rotation` (SPI): the descriptor's
    /// ([`TextureDescriptor::rotation`](crate::TextureDescriptor::rotation)).
    pub fn rotation(&self) -> u64 {
        u64::from(self.record().presented.rotation)
    }

    /// `isDrawable` (SPI): the descriptor's
    /// ([`TextureDescriptor::is_drawable`](crate::TextureDescriptor::is_drawable)).
    pub fn is_drawable(&self) -> bool {
        self.record().presented.is_drawable
    }

    /// Refused (`NotImplemented(RotatedTexture)`) for a rotated texture whose rows, as its CPU
    /// access and blits address them, run past its buffer (the module docs).
    pub(crate) fn check_rotated_reach(&self) -> Result<(), Error> {
        match self.buffer_texels() {
            Some(texels) if !texels.reached() => {
                Err(Error::NotImplemented(RuntimeGap::RotatedTexture))
            }
            _ => Ok(()),
        }
    }

    /// The texture a render pass renders, or resolves, into for this one: a texture buffer's
    /// attachment twin (`Texture::attachment_twin`), a rotated texture's twin (the module docs), a
    /// view's render shadow where its bindings present it in its image's own format
    /// (`Texture::view_render_shadow`), and otherwise the texture itself. Refused as the twin is.
    pub(crate) fn pass_texture(&self) -> Result<Texture, Error> {
        let record = self.record();
        if record.texture_type == TextureType::TextureBuffer {
            return self.attachment_twin();
        }
        // A depth or stencil texture over a buffer is a surface plane's (`linear_plan.rs`),
        // which Metal's passes store in its tiled layout: they render into its tiled twin
        // (`texture/surface_depth.rs`).
        if self.over_buffer() && is_depth_stencil(record.pixel_format) {
            return self.surface_depth_twin();
        }
        // A view its bindings present in its image's own format renders into its render shadow
        // (`texture/view_shadow.rs`).
        if record.read_as.is_some() {
            return self.view_render_shadow("render_command_encoder");
        }
        let rotation = record.presented.rotation;
        if rotation == 0 {
            return Ok(self.clone());
        }
        const CALL: &str = "render_command_encoder";
        let texels = self
            .buffer_texels()
            .ok_or_else(|| internal("a rotated texture without texels"))?;
        texels.twin(record.pixel_format, || {
            let [width, height, _] = record.extent();
            let descriptor = TextureDescriptor {
                usage: record.usage,
                resource_options: texels.options(),
                ..TextureDescriptor::texture_2d(
                    record.pixel_format,
                    u64::from(width),
                    u64::from(height),
                    false,
                )
            };
            let device = self.device();
            let layout = descriptor
                .check(device.properties().texture_rules())
                .map_err(|refused| refused.error(CALL))?;
            let turned = Arc::new(texels.turned(rotation, [width, height]));
            let storage = device.rotated_storage(CALL, turned, &descriptor, &layout, rotation)?;
            let record = ViewRecord::of_descriptor(&layout, &descriptor);
            Ok(Texture::made(device, storage, record, None, IdUse::Runtime))
        })
    }

    /// The storage of a view (`record`) of this rotated texture whose image takes no view of the
    /// view's format: an image of the stored frame in that format, a shadow of the same plane's
    /// bytes whose writes are dropped, over the texture's own plane rows, so that its passes'
    /// twins are the texture's (the module docs). Refused as the texture's own image would be.
    pub(super) fn rotated_view(
        &self,
        call: &'static str,
        record: &ViewRecord,
    ) -> Result<Storage, Error> {
        let Storage::Shadow { texels, .. } = self.storage() else {
            return Err(internal("a rotated texture without its plane's rows"));
        };
        let rotation = record.presented.rotation;
        let [width, height, _] = record.image_extent;
        let stored = texels.turned(rotation, [width, height]);
        let descriptor = TextureDescriptor {
            usage: record.usage,
            swizzle: record.swizzle,
            resource_options: texels.options(),
            ..TextureDescriptor::texture_2d(
                record.pixel_format,
                u64::from(width),
                u64::from(height),
                false,
            )
        };
        let layout = descriptor
            .check(self.device().properties().texture_rules())
            .map_err(|refused| refused.error(call))?;
        self.device().rotated_frame(
            call,
            (Arc::clone(texels), &stored),
            &descriptor,
            (&layout, rotation),
        )
    }
}

impl Device {
    /// A texture of `descriptor` (checked as `layout`) of `rotation` over an IOSurface plane (the
    /// module docs), its ID as `id_use` says, its storage [`rotated_frame`](Self::rotated_frame)'s
    /// of `rows` and `stored`. Refused as that is.
    pub(crate) fn new_rotated_texture(
        &self,
        call: &'static str,
        [rows, stored]: [Texels; 2],
        descriptor: &TextureDescriptor,
        (layout, rotation): (&TextureLayout, u8),
        id_use: IdUse,
    ) -> Result<Texture, Error> {
        let storage = self.rotated_frame(
            call,
            (Arc::new(rows), &stored),
            descriptor,
            (layout, rotation),
        )?;
        let record = ViewRecord::of_descriptor(layout, descriptor);
        Ok(Texture::made(self, storage, record, None, id_use))
    }

    /// The storage of a rotated texture of `descriptor` (checked as `layout`, the frame's extent)
    /// of `rotation`, its CPU access and blits at `rows` (the plane's rows as the texture's own
    /// extent), and its functions' image an image of the stored frame, a shadow of `stored`, the
    /// stored frame's texels, whose writes are dropped (the module docs). Refused as
    /// [`copied_image`](Self::copied_image) refuses the image.
    fn rotated_frame(
        &self,
        call: &'static str,
        (rows, stored): (Arc<Texels>, &Texels),
        descriptor: &TextureDescriptor,
        (layout, rotation): (&TextureLayout, u8),
    ) -> Result<Storage, Error> {
        let [width, height, depth] = layout.extent;
        let [stored_width, stored_height] = rotated::stored_extent(rotation, [width, height]);
        let frame = TextureLayout {
            extent: [stored_width, stored_height, depth],
            ..*layout
        };
        let image = self.copied_image(call, &frame, descriptor, rows.options())?;
        let mut copies = stored.shadow_copies(
            image.raw.get(),
            image.identity,
            texture_aspects(image.plan()),
            (None, None),
        )?;
        copies.drops_writes = true;
        let reason = || {
            format!(
                "its functions read an image of the plane's {stored_width}x{stored_height} stored \
                 frame, copied in before each command that uses it, whose writes are dropped as \
                 Metal drops them"
            )
        };
        if crate::shadow_log::enabled() {
            let why = format!("a texture of rotation {rotation}: {}", reason());
            crate::shadow_log::made(self, call, (descriptor, &frame), stored, &why);
        }
        self.degrade(Degradation::ShadowedTexture, || {
            let object = format!(
                "a {width}x{height} {:?} texture of rotation {rotation} over an IOSurface plane",
                layout.pixel_format
            );
            (object, reason())
        });
        Ok(Storage::Shadow {
            image: image.shared(),
            texels: rows,
            copies: Arc::new(copies),
        })
    }

    /// The storage of a rotated texture's twin: an image of its pass's frame (`layout`, planned
    /// for `descriptor`'s usage), a shadow of `turned` whose copies turn the texels by
    /// `rotation` (`rotated.rs`), announced once as a shadowed texture. Refused as
    /// [`copied_image`](Self::copied_image) refuses the image, and as `RotatedCopies::new`
    /// refuses the copies.
    fn rotated_storage(
        &self,
        call: &'static str,
        turned: Arc<Texels>,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        rotation: u8,
    ) -> Result<Storage, Error> {
        let image = self.copied_image(call, layout, descriptor, turned.options())?;
        let [width, height, _] = layout.extent;
        let copies =
            turned.rotated_copies(image.raw.get(), image.identity, rotation, [width, height])?;
        let reason = || {
            let [stored_width, stored_height] = rotated::stored_extent(rotation, [width, height]);
            format!(
                "it renders through an image of its pass's frame, whose texels are turned into \
                 the plane's {stored_width}x{stored_height} rows after each pass that stores it \
                 and back before each that uses it"
            )
        };
        if crate::shadow_log::enabled() {
            let why = format!("a twin of a texture of rotation {rotation}: {}", reason());
            crate::shadow_log::made(self, call, (descriptor, layout), &turned, &why);
        }
        self.degrade(Degradation::ShadowedTexture, || {
            let object = format!(
                "a {width}x{height} {:?} texture of rotation {rotation} over an IOSurface plane",
                layout.pixel_format
            );
            (object, reason())
        });
        Ok(Storage::Shadow {
            image: image.shared(),
            texels: turned,
            copies: Arc::new(copies),
        })
    }

    /// A private image of `layout` for `descriptor`'s usage, of the hazard tracking of `options`,
    /// that copies take texels into and out of: a rotated texture's or its twin's. Refused as
    /// `new_texture` refuses the image, and (`NotImplemented(RotatedTexture)`) where the device
    /// cannot copy its format.
    fn copied_image(
        &self,
        call: &'static str,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        options: ResourceOptions,
    ) -> Result<Image, Error> {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let usage = texture_usage(descriptor.usage, descriptor.swizzle);
        let plan = plan_image(layout, usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        let plan = self.with_undeclared_usage(plan, layout, usage, &facts)?;
        if !plan
            .usage
            .contains(ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST)
        {
            return Err(Error::NotImplemented(RuntimeGap::RotatedTexture));
        }
        let hazard = options.hazard_tracking_mode()?;
        let private = TextureDescriptor {
            resource_options: ResourceOptions::private(),
            ..*descriptor
        };
        self.new_image(plan, layout, &private, hazard)
    }
}
