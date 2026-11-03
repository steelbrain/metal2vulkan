//! A depth or stencil texture over a plane of a surface with planes, in a render pass. Plain
//! Metal (an M4 under macOS 27.0.1) loads and stores such a texture in a pass in its tiled layout
//! (`heap_tiled.rs`, `TiledLayout::element`, one aspect's unit an element:
//! `copy_plan::surface_depth_unit`) from the plane's offset rounded down to 128,
//! writing the elements' bytes alone, where its reads, samples, blits and CPU access take the
//! plane linearly (`linear_plan.rs`). The stencil of `Depth32Float_Stencil8` is not in the plane:
//! a pass loads it as 0 (`render_pass::surface_stencil_loads` makes such a load a clear to 0) and
//! its store is dropped.
//!
//! A pass renders into the texture's twin ([`Texture::surface_depth_twin`]): an image of the
//! texture's format, a shadow of the plane's bytes from that offset whose copies move the
//! plane's aspect between the image and Metal's tiles (`HeapTiledCopies`), made at the first pass
//! that needs it and kept with the texture's texels; its copies back write the runs of the
//! elements' bytes alone ([`element_runs`]: a tile of partial extent leaves bytes between its
//! elements), up to the last element's byte. A view in the texture's own format (the whole of
//! it: such a texture has one level and one slice) renders into the texture's twin. The twin's
//! other aspect is never copied: it starts as zeros and a pass's store of it reaches no byte.
//! An `X32_Stencil8` view of a `Depth32Float_Stencil8` texture, whose stencil the plane does not
//! hold, renders into a private image of its own, which no byte reaches: Metal's pass on it loads
//! 0 and changes no byte, and a kernel reads 0 through it. Refused as the
//! runtime's gap (`NotImplemented(SurfaceDepthStencil)`): a device the moves do not fit, and a
//! view in another format of a plane that holds its stencil (an `X32_Stencil8` texture's view
//! as `Depth32Float_Stencil8`, unmeasured). A blit through a view takes the plane's unit
//! ([`Texture::surface_plane_unit`]; `blit_texture::unit`). A twin whose elements run past the buffer is
//! `NotImplemented(TexelsPastBufferLength)`.

use super::super::buffer_texture::Texels;
use super::super::copy_plan::{surface_depth_unit, CopyUnit};
use super::super::error::{internal, Error, RuntimeGap};
use super::super::image_plan::{plan_image, ImagePlan};
use super::super::shadow::{ShadowBuffer, ShadowCopies, ShadowSource, ShadowTarget};
use super::super::shadow_merge::ScratchSpares;
use super::super::texture_descriptor::{texture_usage, TextureDescriptor, TextureLayout};
use super::super::tracker::UseRange;
use super::super::values::ResourceOptions;
use super::super::view_plan::ViewRecord;
use super::heap::{heap_plane_layout, heap_planes};
use super::heap_tiled::{Elements, HeapTiledCopies, TiledLayout, TiledTexture};
use super::{image_error, IdUse, Image, Storage, Texture};
use ash::vk;
use metal2vulkan::metal::format::aspects;
use metal2vulkan::metal::TextureType;
use metal2vulkan::vk::{ImageAspectFlags, ImageUsageFlags};
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// Metal's passes store the tiles from the plane's offset rounded down to this.
const TILED_ALIGNMENT: u64 = 128;

impl Texture {
    /// The texture a render pass renders into for this depth or stencil texture over a plane of
    /// a surface with planes: its tiled twin (the module docs), made at the first pass that needs
    /// it. Refused as the module docs say.
    pub(super) fn surface_depth_twin(&self) -> Result<Texture, Error> {
        let record = self.record();
        let pf = record.pixel_format;
        let gap = || Error::NotImplemented(RuntimeGap::SurfaceDepthStencil { pixel_format: pf });
        let texels = self
            .buffer_texels()
            .ok_or_else(|| internal("a surface's depth texture without texels"))?;
        let root = self.root();
        if !std::ptr::eq(root, self) {
            if root.record().pixel_format == pf {
                return root.surface_depth_twin();
            }
            // A view of a stencil the plane does not hold renders into an image of its own,
            // which no byte reaches (the module docs).
            if !self.surface_stencil_unheld() {
                return Err(gap());
            }
            return texels.twin(pf, || {
                let (image, layout, descriptor) = self.pass_image(texels, |_| Ok(()))?;
                let storage = Storage::Image(image.shared());
                let record = ViewRecord::of_descriptor(&layout, &descriptor);
                Ok(Texture::made(
                    self.device(),
                    storage,
                    record,
                    None,
                    IdUse::Runtime,
                ))
            });
        }
        let unit = surface_depth_unit(pf).ok_or_else(gap)?;
        texels.twin(pf, || {
            let device = self.device();
            let [width, height, _] = record.extent();
            let [w, h] = [u64::from(width), u64::from(height)];
            let bytes = u64::from(unit.bytes);
            let tiled = TiledLayout::of(
                bytes,
                Elements::Texels,
                TextureType::Type2D,
                [w, h, 1],
                1,
                1,
            )
            .ok_or_else(gap)?;
            // Metal writes the elements' bytes alone: the copies back write their runs.
            let runs = element_runs(&tiled);
            let span = runs.last().map_or(0, |run| run.end);
            let base = texels.cpu_layout().offset / TILED_ALIGNMENT * TILED_ALIGNMENT;
            let end = base + span;
            if end > texels.buffer.length() {
                return Err(Error::NotImplemented(RuntimeGap::TexelsPastBufferLength));
            }
            let hazard = texels.options().hazard_tracking_mode()?;
            let (image, layout, descriptor) = self.pass_image(texels, |plan| {
                let both = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
                match plan.usage.contains(both) {
                    true => Ok(()),
                    false => Err(gap()),
                }
            })?;
            let plan = image.plan();
            let (format, plan_pf, extent) = (plan.format, plan.pixel_format, plan.extent);
            // The plane's aspect alone moves between the image and the tiles.
            let (planes, tight_len) = heap_plane_layout(format, plan_pf, extent, 1, 1);
            let (regions, _) = heap_planes(format, plan_pf, extent, 1, 1);
            let planes: Vec<_> = planes
                .into_iter()
                .filter(|plane| plane.aspect == unit.aspect)
                .collect();
            let aspect = unit.aspect.into();
            let regions: Vec<_> = regions
                .into_iter()
                .filter(|region| region.image_subresource.aspect_mask == aspect)
                .collect();
            // The plane's aspect alone: one part, of the planes given.
            let tiled = TiledTexture::one(tiled);
            if !HeapTiledCopies::fit(device, &tiled, &planes, tight_len) {
                return Err(gap());
            }
            let (raw, at) = texels.buffer.at(base);
            let mut staging = HeapTiledCopies::staging(device, &tiled, at, hazard)?;
            let start = staging.into.dst_offset;
            staging.into.size = span;
            staging.back = runs
                .iter()
                .map(|run| vk::BufferCopy {
                    src_offset: start + run.start,
                    dst_offset: at + run.start,
                    size: run.end - run.start,
                })
                .collect();
            let shape = (&planes[..], tight_len, 1);
            let tiles = HeapTiledCopies::new(device, &staging, tiled, shape, &regions, hazard)?;
            let (truth, tracked) = texels.buffer.tracked(base..end);
            let copies = ShadowCopies {
                source: ShadowSource::Bytes {
                    buffer: ShadowBuffer::of(&texels.buffer, raw),
                    key: truth.key,
                    bytes: tracked,
                    target: ShadowTarget::Tiled {
                        image: image.raw.get(),
                        staging,
                        tiles: Box::new(tiles),
                    },
                    alias: texels.buffer.alias().map(|alias| alias.key),
                },
                shadow_key: image.identity.key,
                shadow_range: UseRange::Image {
                    aspects: aspect,
                    levels: 0..1,
                    layers: 0..1,
                },
                tracked: texels.tracked(),
                drops_writes: false,
                backing: texels.backing(),
                merge_scratch: ScratchSpares::default(),
                touched: Default::default(),
                stood: AtomicU64::new(0),
            };
            let storage = Storage::Shadow {
                image: image.shared(),
                texels: Arc::new(texels.tiled_span(base..end, bytes)),
                copies: Arc::new(copies),
            };
            let record = ViewRecord::of_descriptor(&layout, &descriptor);
            Ok(Texture::made(device, storage, record, None, IdUse::Runtime))
        })
    }

    /// A private image for this texture's passes, over its `texels`: a 2D texture of its format,
    /// extent and usage, its plan checked by `check` first; with its layout and descriptor.
    fn pass_image(
        &self,
        texels: &Texels,
        check: impl FnOnce(&ImagePlan) -> Result<(), Error>,
    ) -> Result<(Image, TextureLayout, TextureDescriptor), Error> {
        const CALL: &str = "render_command_encoder";
        let device = self.device();
        let record = self.record();
        let [width, height, _] = record.extent();
        let options = texels.options();
        let descriptor = TextureDescriptor {
            usage: record.usage,
            resource_options: options,
            ..TextureDescriptor::texture_2d(
                record.pixel_format,
                u64::from(width),
                u64::from(height),
                false,
            )
        };
        let layout = descriptor
            .check(device.properties().texture_rules())
            .map_err(|refused| refused.error(CALL))?;
        let facts = device
            .shared()
            .textures
            .facts(device.compile_device().profile());
        let usage = texture_usage(descriptor.usage, descriptor.swizzle);
        let plan = plan_image(&layout, usage, false, &facts)
            .map_err(|refusal| image_error(CALL, refusal))?;
        let plan = device.with_undeclared_usage(plan, &layout, usage, &facts)?;
        check(&plan)?;
        let hazard = options.hazard_tracking_mode()?;
        let private = TextureDescriptor {
            resource_options: ResourceOptions::private(),
            ..descriptor
        };
        let image = device.new_image(plan, &layout, &private, hazard)?;
        Ok((image, layout, descriptor))
    }

    /// The unit a texel of the plane under this depth or stencil texture over a plane of a
    /// surface with planes, or under a view of one, holds (`copy_plan::surface_depth_unit` of the
    /// texture views are made from); `None` for another texture.
    pub(crate) fn surface_plane_unit(&self) -> Option<CopyUnit> {
        match self.over_buffer() {
            true => surface_depth_unit(self.root().record().pixel_format),
            false => None,
        }
    }

    /// Whether this texture, of a format with a stencil, is over a plane of a surface that holds
    /// its texture's other aspect: `Depth32Float_Stencil8`'s depth, whose stencil Metal's pass
    /// loads as 0 (`render_pass::surface_stencil_loads`).
    pub(crate) fn surface_stencil_unheld(&self) -> bool {
        let stencil = ImageAspectFlags::STENCIL;
        aspects(self.record().pixel_format).is_some_and(|own| own.contains(stencil))
            && self
                .surface_plane_unit()
                .is_some_and(|unit| !unit.aspect.contains(stencil))
    }
}

/// The runs of bytes the elements of `tiled`'s one level of one chain lie in, from its first
/// byte, in order and joined where they meet: a whole tile's bytes are one run, a tile of
/// partial extent leaves bytes between its elements (`TiledLayout::element`).
pub(crate) fn element_runs(tiled: &TiledLayout) -> Vec<Range<u64>> {
    let level = &tiled.levels[0];
    let [w, h] = level.extent;
    let [tw, th] = level.tile;
    let unit = tiled.unit;
    let mut runs: Vec<Range<u64>> = Vec::new();
    let mut push = |run: Range<u64>| match runs.last_mut() {
        Some(last) if last.end == run.start => last.end = run.end,
        _ => runs.push(run),
    };
    let mut offsets = Vec::new();
    for y0 in (0..h.div_ceil(th)).map(|ty| ty * th) {
        for x0 in (0..w.div_ceil(tw)).map(|tx| tx * tw) {
            let first = tiled.element(0, 0, x0, y0);
            let [vw, vh] = [(w - x0).min(tw), (h - y0).min(th)];
            if [vw, vh] == [tw, th] {
                push(first..first + tw * th * unit);
                continue;
            }
            offsets.clear();
            offsets.extend(
                (y0..y0 + vh).flat_map(|y| (x0..x0 + vw).map(move |x| tiled.element(0, 0, x, y))),
            );
            offsets.sort_unstable();
            for &at in &offsets {
                push(at..at + unit);
            }
        }
    }
    runs
}
