//! The route of a 2D texture (or array) over a buffer's bytes: where it lies, as its making decides
//! before it makes an image (`Device::storage_over_buffer`), and as
//! [`Buffer::texture_placement`](crate::Buffer::texture_placement) answers it. Both call
//! [`Device::route_over_buffer`], so the answer is the making's own decision: texels past the
//! buffer's length, a buffer of no host pages, a device copy, bytes detached or copied, and over
//! host pages the in-place decision (`in_place::in_place`) against the device's check's verdicts.
//! Where that takes the image in place, the query also runs the image's making up to the bind
//! (`texture/in_place_image.rs`), so a pitch, an image, a placement or rows the device refuses
//! answer the shadow the texture then is.

use super::image_error;
use crate::buffer::CallerMemory;
use crate::buffer_texture::Texels;
use crate::image_plan::{plan_image, ImageFacts, ImagePlan};
use crate::in_place::{self, Access, Padding, PagesClass, Refusal, RowEnds, ShadowCause, Verdict};
use crate::texture_descriptor::{texture_usage, TextureDescriptor, TextureLayout};
use crate::texture_placement::TexturePlacement;
use crate::{Device, Error};
use metal2vulkan::device::HostPageTexturePolicy;
use metal2vulkan::metal::TextureUsage;

/// Where a texture over a buffer's bytes goes, as its making decides before it makes an image.
pub(crate) enum Route {
    /// Its texels run past the buffer's length, which no image over the buffer's memory reaches:
    /// the shadow, its tail holding them.
    Past,
    /// The buffer's bytes are no host pages: an image aliasing the buffer's memory where the
    /// device makes one on Metal's layout, else the shadow.
    Device,
    /// Over host pages of `class`, its texels lying in the device's lines as `rows` says, and
    /// `padding` saying whether a shader's writes may change their rows' padding.
    Host {
        class: PagesClass,
        rows: RowEnds,
        padding: Padding,
        way: HostWay,
    },
}

/// How a texture over host pages goes ([`Route::Host`]).
pub(crate) enum HostWay {
    /// A device copy (`Texels::device_copy`): a shadow whose copies in a log decides.
    DeviceCopy,
    /// The image in place, with the storage usage too where `storable`, where the device makes it
    /// (`Device::in_place_image`).
    InPlace { storable: bool },
    /// The shadow, why.
    Shadow(ShadowCause),
}

impl Device {
    /// The usage a texture of `descriptor` (checked as `layout`) over a buffer's bytes is planned
    /// for, and its image's declared plan: the buffer's rows lay the texels out, whatever
    /// `allowGPUOptimizedContents` says. Refused as `new_texture` refuses the image, for Metal's
    /// `call`.
    pub(super) fn declared_plan(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        facts: &ImageFacts<'_>,
    ) -> Result<(TextureUsage, ImagePlan), Error> {
        let usage = texture_usage(descriptor.usage, descriptor.swizzle);
        let plan = plan_image(layout, usage, false, facts)
            .map_err(|refusal| image_error(call, refusal))?;
        Ok((usage, plan))
    }

    /// Where a texture of `descriptor` (checked as `layout`) over `texels` lies if made now, for
    /// Metal's `call`: the route its making takes, `past` where its texels run past the buffer's
    /// length (`Texels::runs_past`, where the making gives them their tail); and where that is the
    /// image in place, whether the device makes it, by the making's own steps up to the bind
    /// (`Device::in_place_image_made`), each image they make released unbound. Refused as the
    /// making refuses the texture's declared plan or fails those steps. A pending verdict starts
    /// the device's check, as the making's does; nothing is bound, submitted or counted.
    pub(crate) fn placement_over_buffer(
        &self,
        call: &'static str,
        texels: &Texels,
        (descriptor, layout): (&TextureDescriptor, &TextureLayout),
        past: bool,
    ) -> Result<TexturePlacement, Error> {
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let (usage, plan) = self.declared_plan(call, descriptor, layout, &facts)?;
        let route = match self.route_over_buffer(texels, usage, past) {
            Route::Host {
                class,
                rows,
                padding,
                way: HostWay::InPlace { storable },
            } => {
                let made = (texels, class, storable);
                let way = match self.in_place_image_made(&plan, made, &facts)? {
                    Ok(()) => HostWay::InPlace { storable },
                    Err((refusal, why)) => HostWay::Shadow(ShadowCause::Refused(refusal, why)),
                };
                Route::Host {
                    class,
                    rows,
                    padding,
                    way,
                }
            }
            route => route,
        };
        Ok(route.placement())
    }

    /// The route of a texture of Metal `usage` (as the image is planned for it) over `texels`,
    /// `past` where they run past the buffer's length (the module docs). Over host pages (a no-copy
    /// buffer's, or a no-copy heap's buffer's, imported by their pointer or as dma-bufs; or fresh
    /// pages the runtime imported for a long buffer of its own, `fresh_pages.rs`): a device copy
    /// where one serves; else the shadow where the bytes were detached or copied; else as the
    /// device's policy, the texture's usage and the device's check of the pages' class for each
    /// access take it (`in_place.rs`), asking the verdicts (a pending one starts the check). The
    /// device's check's own textures take the image in place whatever the policy and the
    /// verdicts.
    pub(crate) fn route_over_buffer(
        &self,
        texels: &Texels,
        usage: TextureUsage,
        past: bool,
    ) -> Route {
        if past {
            return Route::Past;
        }
        let class = match texels.buffer.caller_memory() {
            Some(CallerMemory::HostPages) if texels.buffer.over_dma_buf() => PagesClass::DmaBuf,
            Some(CallerMemory::HostPages) => PagesClass::Pointer,
            None if texels.buffer.in_fresh_pages() => PagesClass::Fresh,
            _ => return Route::Device,
        };
        // How the texels lie in the device's lines of memory, each checked apart
        // (`in_place::RowEnds`), and whether a shader's writes may change their rows' padding
        // (`in_place::Padding`).
        let line = self.shared().in_place.line();
        let (rows, padding) = (texels.rows(line), texels.padding(line));
        let host = |way| Route::Host {
            class,
            rows,
            padding,
            way,
        };
        let pages = (class, rows, texels.texel_bytes(), padding);
        let checking = in_place::checking();
        let detached = texels.buffer.detached();
        let policy = self.compile_device().policy().host_page_textures;
        // On a discrete GPU, a `Managed` texture over a buffer's bytes that only samples or reads
        // whose policy takes the image in place, over pages whose memory carries a log, is a
        // device copy instead (`Texels::device_copy`): a shadow whose copies in take only the rows
        // the log reports changed (`dirty_pages.rs`), the GPU's copy of the two a `Managed`
        // resource is on a Mac with a discrete GPU. It binds no image to the pages, so the
        // in-place check's verdict is not asked. A `Shared` texture, an IOSurface's, and one that
        // renders work on the pages in place, across the bus, as such a Mac's GPU does.
        if !checking && !detached {
            let passed = |_, _, _| Verdict::Passed;
            let copy = Access::Reads.takes(usage)
                && in_place::in_place(policy, usage, pages, passed).is_ok()
                && texels.device_copy(self);
            if copy {
                return host(HostWay::DeviceCopy);
            }
        }
        if detached {
            // Over bytes detached from their pages (`detach.rs`) no image is bound to them: the
            // texture is their shadow, which presents a stand-in until they lie in pages again.
            return host(HostWay::Shadow(ShadowCause::Refused(
                Refusal::Detached,
                "the buffer's bytes were detached from their pages (detach_backing) as the \
                 texture was made"
                    .to_owned(),
            )));
        }
        if texels.buffer.copied() {
            // A copied host buffer's bytes the GPU uses lie in memory of the device's own, no
            // import of the pages (`copied.rs`): the shadow copies them, its copies the buffer's
            // uses, which commits copy in and completions merge back.
            return host(HostWay::Shadow(ShadowCause::Refused(
                Refusal::Placement,
                "the buffer's bytes are a copy of the caller's pages in memory of the device's \
                 own, which no image over the pages lies in"
                    .to_owned(),
            )));
        }
        // The check's own textures measure the device whatever its policy.
        let policy = match checking {
            true => HostPageTexturePolicy::InPlace,
            false => policy,
        };
        let verdict = |class, access, rows| match checking {
            true => Verdict::Passed,
            false => self.in_place_verdict(class, access, rows),
        };
        match in_place::in_place(policy, usage, pages, verdict) {
            // A texture that renders takes the storage usage too where the check passed for a
            // shader's writes over the class and its texel is one the check's kernel stores: a
            // draw that reads it through a snapshot fetches it in the pixel interlock through that
            // usage (`render/snapshot.rs`), and with it a shader's writes reach the texels in
            // place, as the check's writes do.
            Ok(()) => host(HostWay::InPlace {
                storable: Access::Renders.takes(usage)
                    && !Access::Writes.takes(usage)
                    && in_place::STORED_TEXEL_BYTES.contains(&pages.2)
                    && verdict(class, Access::Writes, rows) == Verdict::Passed,
            }),
            Err(cause) => host(HostWay::Shadow(cause)),
        }
    }
}
