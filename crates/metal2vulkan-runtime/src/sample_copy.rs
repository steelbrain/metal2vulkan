//! Blit copies between multisampled textures that Vulkan does not copy image to image: between a
//! depth or stencil aspect and a colour format of its byte size, and between one aspect held in two
//! depth/stencil formats (`Stencil8` and the `X32Stencil8` view of `Depth32FloatStencil8`).
//!
//! **Metal's answer** (a probe, an Apple M3 Max under macOS 26.7 and an M4 under 27.0.1, plain
//! Metal): each sample's bytes land in the same sample of the destination, every value kept, as a
//! single-sampled copy keeps them.
//!
//! **The copies.** Vulkan copies no multisampled image into a buffer, so the single-sampled path
//! through a scratch buffer (`blit_texture.rs`) cannot take them. `VK_KHR_maintenance8` lets
//! `vkCmdCopyImage` copy between a depth or stencil aspect and a colour format of its bits
//! ("Compatible Formats for Depth-Stencil to/from Color Copies"): a colour format of that table
//! takes one copy; any other colour format of the unit, and a depth/stencil format of another
//! Vulkan format, take two, through a multisampled colour image of the aspect's bits, the
//! encoder's own ([`colour_twin`]): `R32_SFLOAT` for
//! 32-bit depth, `R16_UNORM` for 16-bit, `R8_UNORM` for stencil. The second copy reads what the
//! first wrote, past a barrier. A copy into a 32-bit float depth image announces
//! `restricted-depth-upload` where the device may clamp.
//!
//! **By draws**, on a device without maintenance8 or whose driver copies a multisampled
//! depth/stencil image to or from a colour one wrong
//! (`RuntimeDriverDefect::MultisampledDepthStencilColourCopies`: lavapipe, MoltenVK, NVIDIA's
//! GA10x), each sample moves by the sample-rate draws that already copy the separate textures'
//! aspects there (`stencil_copy.rs`, `depth_copy.rs`), through multisampled colour images of
//! the aspect's bits, the encoder's own: out of a depth or stencil aspect, a bits draw writes
//! each sample's bits (`R32_UINT` for 32-bit float depth, `R16_UINT` for a 16-bit normalized
//! one's codes, `R8_UINT` for stencil) where the region lies in the source; one `vkCmdCopyImage`
//! between colour images of one texel size moves the region to where it lies in the destination;
//! into a stencil aspect, the plane draws read it from an `R8_UINT` image, into a 32-bit float
//! depth the `FragDepth` draw from an `R32_SFLOAT` one, and into a 16-bit normalized depth the
//! code draw from the `R16_UINT` one (`depth_copy::depth_carrier`). A colour side is copied from
//! or into directly. Each phase reads what the one before wrote, past a barrier. The colour
//! images are mutable, so MoltenVK copies between them and a colour texture of another format
//! through a view, not its broken temporary buffer.
//!
//! The plane draws are exact; the bits draws read each sample as the lanes' readbacks do; the
//! depth draws are exact where the driver stores every depth unchanged, or keeps every 16-bit
//! code (`depth_copy::copies_exactly`: lavapipe's measured releases for a 32-bit float depth,
//! and for a 16-bit one its releases and MoltenVK 1.4.2 on the Apple M3 Max). A 16-bit depth
//! crosses as its codes rather than as floats, so no conversion Vulkan leaves to the device
//! stands between the two halves. Elsewhere a copy into a 32-bit float depth runs as a
//! near-equivalent, announced once (`drawn-depth-copy`, `depth_copy::drawn`): its reason names
//! the values the driver's fragment depth is measured to change (MoltenVK, NVIDIA's; the
//! orchestrator's ruling of 2026-10-07), or, on a driver not measured, the values Vulkan leaves
//! to it (outside `[0, 1]`, denormals, -0). Refused as before, `Unsupported` naming
//! `Feature::Format` and the missing maintenance8 or the defect: a copy into a 16-bit depth on a
//! driver whose codes the draw is not measured to keep, a depth or stencil source the draws
//! cannot sample, and a destination they cannot render to.
//!
//! Refused, typed, naming `Feature::Format`: a copy of a combined format's two aspects (not
//! implemented); on a device whose copies go by draws, what the draws cannot copy (above); and
//! a colour image the device does not make at the sample count.

use super::blit::TRANSFER;
use super::copy_plan::ImageRegion;
use super::depth_copy::{self, DepthDraw, Drawn};
use super::depth_stencil_scratch::{colour_copies_refused, colour_image, colour_twin};
use super::error::{internal, Error};
use super::image_plan::{sample_count, ImagePlan};
use super::render_pass::Target;
use super::resource::ResourceKey;
use super::stencil_copy::{attachment_use, sampled_use, DrawArea, StencilPlanes, DRAW_STAGES};
use super::texture::{ImageMaker, Texture};
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::ResourceOptions;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
};

/// One image of a copy: its texture, the image, the aspect its texels are in, and its Vulkan
/// format.
#[derive(Clone, Copy)]
pub(super) struct Side<'a> {
    pub(super) texture: &'a Texture,
    pub(super) image: vk::Image,
    pub(super) aspect: ImageAspectFlags,
    pub(super) format: Format,
}

/// Whether maintenance8's table lets `vkCmdCopyImage` copy between `aspect` of the
/// depth/stencil `format` and the colour format `colour`: one-channel formats of the aspect's
/// bits, `R32` for 32-bit float depth, `R16` for 16-bit normalized depth, `R8` for stencil.
pub(super) fn copies_with_colour(format: Format, aspect: ImageAspectFlags, colour: Format) -> bool {
    use Format as F;
    if aspect == ImageAspectFlags::STENCIL {
        return format.block().is_some_and(|b| b.stencil.is_some())
            && matches!(colour, F::R8_UINT | F::R8_SINT | F::R8_UNORM | F::R8_SNORM);
    }
    match format {
        F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => {
            matches!(colour, F::R32_SFLOAT | F::R32_UINT | F::R32_SINT)
        }
        F::D16_UNORM | F::D16_UNORM_S8_UINT => matches!(
            colour,
            F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT | F::R16_SFLOAT
        ),
        _ => false,
    }
}

/// One step of the copies.
enum Step {
    /// `vkCmdCopyImage` of one region from the first image into the second.
    Copy(vk::Image, vk::Image, vk::ImageCopy),
    /// A bits draw or the `FragDepth` draw of one layer (`depth_copy.rs`).
    Depth(DepthDraw),
    /// The plane draws of one layer (`stencil_copy.rs`).
    Planes(StencilPlanes),
}

/// Checked copies of regions between two multisampled images ([`plan`]), held until the command
/// buffer completes: what they go through, their steps, their uses, and the stages they run in.
pub(super) struct SampleCopy {
    colours: Vec<Texture>,
    /// The steps phase by phase: each phase reads what the one before wrote.
    phases: Vec<Vec<Step>>,
    pub(super) uses: Vec<Use<ResourceKey>>,
    pub(super) stages: vk::PipelineStageFlags2,
    /// What the `FragDepth` draw changes where it writes a 32-bit float depth inexactly (the
    /// announced `drawn-depth-copy`); `None` where every step is exact.
    inexact_depth: Option<&'static str>,
}

/// The layers of `region` in its aspect.
fn layers(region: &ImageRegion) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: region.aspect.into(),
        mip_level: region.level,
        base_array_layer: region.layer,
        layer_count: region.layers,
    }
}

fn offset([x, y, z]: [u32; 3]) -> vk::Offset3D {
    vk::Offset3D {
        x: x as i32,
        y: y as i32,
        z: z as i32,
    }
}

fn image_copy(src: &ImageRegion, dst: &ImageRegion) -> vk::ImageCopy {
    let [width, height, depth] = src.size;
    vk::ImageCopy {
        src_subresource: layers(src),
        src_offset: offset(src.origin),
        dst_subresource: layers(dst),
        dst_offset: offset(dst.origin),
        extent: vk::Extent3D {
            width,
            height,
            depth,
        },
    }
}

/// The colour twin of `aspect` of the depth/stencil `format`, checked: a colour attachment
/// format the device copies both ways. Refused, naming the format, where it is not.
fn checked_twin(
    device: &Device,
    format: Format,
    aspect: ImageAspectFlags,
) -> Result<(Format, metal2vulkan::metal::PixelFormat), Error> {
    let twin = colour_twin(format, aspect)
        .ok_or_else(|| internal("a multisampled copy of an aspect its format lacks"))?;
    let features = FormatFeatureFlags2::COLOR_ATTACHMENT
        | FormatFeatureFlags2::TRANSFER_SRC
        | FormatFeatureFlags2::TRANSFER_DST;
    let facts = device
        .shared()
        .textures
        .facts(device.compile_device().profile());
    if !facts.optimal(twin.0).contains(features) {
        return Err(Error::Unsupported(Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format: twin.0,
                usage: FormatUsage::ColorAttachment,
            },
        )));
    }
    Ok(twin)
}

/// The copies of `regions` (each in `from`'s and `to`'s aspects, one sample count `samples`)
/// from `from` into `to`, two images Vulkan does not copy into each other directly (the module
/// docs): by maintenance8's copies, or by draws where the device's cannot. Refused as the
/// module docs say.
pub(super) fn plan(
    maker: &dyn ImageMaker,
    regions: &[(ImageRegion, ImageRegion)],
    from: Side,
    to: Side,
    samples: u32,
) -> Result<SampleCopy, Error> {
    let device = maker.device();
    // A combined format copies only into its own, both aspects, which Vulkan copies directly
    // where the images share their format; two Vulkan formats of one combined format would need
    // both aspects' twins.
    let both = ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL;
    if from.aspect.contains(both) || to.aspect.contains(both) {
        return Err(Error::Unsupported(Unsupported::not_implemented(
            Feature::Format,
        )));
    }
    let colour = ImageAspectFlags::COLOR;
    if from.aspect == colour && to.aspect == colour {
        return Err(internal("a multisampled colour copy planned through twins"));
    }
    // The depth/stencil side's aspect: the other side is a colour image.
    let aspects = if from.aspect == colour {
        to.aspect
    } else {
        from.aspect
    };
    match colour_copies_refused(device, Feature::Format, aspects) {
        None => by_copies(maker, regions, from, to, samples),
        Some(refused) => by_draws(maker, regions, from, to, samples, refused),
    }
}

/// [`plan`]'s copies by maintenance8's `vkCmdCopyImage` (the module docs).
fn by_copies(
    maker: &dyn ImageMaker,
    regions: &[(ImageRegion, ImageRegion)],
    from: Side,
    to: Side,
    samples: u32,
) -> Result<SampleCopy, Error> {
    let device = maker.device();
    let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
    let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
    let colour = ImageAspectFlags::COLOR;
    // The depth/stencil side whose bits a colour image takes, where one does.
    let direct = match (from.aspect == colour, to.aspect == colour) {
        (true, false) => copies_with_colour(to.format, to.aspect, from.format),
        (false, true) => copies_with_colour(from.format, from.aspect, to.format),
        _ => false,
    };
    let twin_of = if from.aspect == colour { to } else { from };
    let mut copy = SampleCopy {
        colours: Vec::new(),
        phases: vec![Vec::new(), Vec::new()],
        uses: Vec::new(),
        stages: TRANSFER,
        inexact_depth: None,
    };
    for (src, dst) in regions {
        let (first, second) = if direct {
            ((from.image, to.image, image_copy(src, dst)), None)
        } else {
            let twin = checked_twin(device, twin_of.format, twin_of.aspect)?;
            // Two depth/stencil formats of one aspect have one twin.
            if from.aspect != colour
                && to.aspect != colour
                && colour_twin(to.format, to.aspect).map(|t| t.0) != Some(twin.0)
            {
                return Err(internal(
                    "a multisampled copy between aspects of different bits",
                ));
            }
            let [width, height, _] = src.size;
            let image = colour_image(maker, twin, samples, [width, height], src.layers)?;
            let at = ImageRegion {
                aspect: colour,
                level: 0,
                layer: 0,
                layers: src.layers,
                origin: [0; 3],
                size: src.size,
            };
            copy.uses
                .push(image.image_use(0..1, 0..src.layers, write, true)?);
            copy.uses
                .push(image.image_use(0..1, 0..src.layers, read, false)?);
            let raw = image.raw_image()?;
            copy.colours.push(image);
            (
                (from.image, raw, image_copy(src, &at)),
                Some((raw, to.image, image_copy(&at, dst))),
            )
        };
        let step = |(from, to, region)| Step::Copy(from, to, region);
        copy.phases[0].push(step(first));
        copy.phases[1].extend(second.map(step));
    }
    Ok(copy)
}

/// A multisampled colour image of the bits `format` (Metal's `pixel_format`), `samples`,
/// `extent` and `layers`, through which the draws copy: rendered, sampled and copied both ways,
/// and mutable, so a copy between it and a colour texture of another format of its texel size
/// goes through a view on MoltenVK (the module docs). Refused where the device does not make it.
pub(super) fn bits_image(
    maker: &dyn ImageMaker,
    (format, pixel_format): (Format, PixelFormat),
    samples: u32,
    [width, height]: [u32; 2],
    layers: u32,
) -> Result<Texture, Error> {
    let samples_flag = sample_count(samples)
        .ok_or_else(|| internal("a multisampled copy of a sample count Vulkan lacks"))?;
    let descriptor = TextureDescriptor {
        texture_type: if layers > 1 {
            TextureType::Type2DMultisampleArray
        } else {
            TextureType::Type2DMultisample
        },
        pixel_format,
        width: width.into(),
        height: height.into(),
        array_length: layers.into(),
        sample_count: samples.into(),
        usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format,
        flags: ImageCreateFlags::MUTABLE_FORMAT,
        image_type: ImageType::TYPE_2D,
        format,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples: samples_flag,
        usage: ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::SAMPLED
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
        unavailable: TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    Device::new_planned_texture(maker, &descriptor, plan)
}

/// `layer` of `texture`'s `level` in `format` at `samples`, a draw's target over `extent`.
fn target(
    texture: &Texture,
    format: Format,
    (level, layer): (u32, u32),
    samples: u32,
    extent: [u32; 2],
) -> Target {
    Target {
        texture: texture.clone(),
        level,
        layer,
        layers: 1,
        format,
        samples,
        extent,
    }
}

/// [`plan`]'s copies by draws (the module docs), on a device whose maintenance8 copies cannot
/// make them; `refused` is why, the refusal of what the draws cannot copy.
fn by_draws(
    maker: &dyn ImageMaker,
    regions: &[(ImageRegion, ImageRegion)],
    from: Side,
    to: Side,
    samples: u32,
    refused: Unsupported,
) -> Result<SampleCopy, Error> {
    let device = maker.device();
    let feature = Feature::Format;
    let (colour, depth, stencil) = (
        ImageAspectFlags::COLOR,
        ImageAspectFlags::DEPTH,
        ImageAspectFlags::STENCIL,
    );
    let refuse = || Error::Unsupported(refused.clone());
    let mut inexact = None;
    let usage = |side: &Side| side.texture.image_plan().map(|plan| plan.usage);
    // The bits a depth or stencil source's draws write: a stencil's value, a 32-bit float
    // depth's bits, or a 16-bit unsigned normalized depth's code, of an image the draws sample.
    let read_bits = if from.aspect == colour {
        None
    } else if !usage(&from)?.contains(ImageUsageFlags::SAMPLED) {
        return Err(refuse());
    } else if from.aspect == stencil {
        Some((Format::R8_UINT, PixelFormat::R8Uint))
    } else if from.aspect == depth && depth_copy::float_depth(from.format) {
        Some((Format::R32_UINT, PixelFormat::R32Uint))
    } else if from.aspect == depth && depth_copy::unorm16_depth(from.format) {
        Some((Format::R16_UINT, PixelFormat::R16Uint))
    } else {
        return Err(refuse());
    };
    // The bits a depth or stencil destination's draws read: a stencil's, or a depth's where the
    // `FragDepth` draw stores it exactly, into an image they render to. A 32-bit float depth is
    // carried as its floats and a 16-bit normalized one as its codes
    // (`depth_copy::depth_carrier`).
    let write_bits = if to.aspect == colour {
        None
    } else if !usage(&to)?.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
        return Err(refuse());
    } else if to.aspect == stencil {
        Some((Format::R8_UINT, PixelFormat::R8Uint))
    } else if to.aspect == depth {
        // Exact where the driver stores every depth unchanged (a 32-bit float depth) or keeps
        // every code (a 16-bit normalized one); elsewhere the announced near-equivalent, naming
        // what the driver's fragment depth changes or may change, and a 16-bit depth whose codes
        // are not measured kept is refused (the one decision, `depth_copy::drawn`).
        if let Drawn::Changes(changes) = depth_copy::drawn(device, to.format).ok_or_else(refuse)? {
            inexact = Some(changes);
        }
        let carrier = depth_copy::depth_carrier(to.format).ok_or_else(refuse)?;
        Some(match carrier {
            Format::R16_UINT => (Format::R16_UINT, PixelFormat::R16Uint),
            _ => (Format::R32_SFLOAT, PixelFormat::R32Float),
        })
    } else {
        return Err(refuse());
    };
    let transfer = |access| Access::new(TRANSFER, access);
    let attachment_output = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
    let mut copy = SampleCopy {
        colours: Vec::new(),
        phases: vec![Vec::new(), Vec::new(), Vec::new()],
        uses: Vec::new(),
        stages: TRANSFER | DRAW_STAGES | attachment_output,
        inexact_depth: inexact,
    };
    for (src, dst) in regions {
        let [width, height, _] = src.size;
        let size = [width, height];
        let layers = src.layers;
        // Where the middle copy reads: the colour source, or the bits the draws wrote where the
        // region lies in the source.
        let (read_image, read_region) = match read_bits {
            None => (from.image, *src),
            Some(bits) => {
                let area = DrawArea {
                    origin: [src.origin[0], src.origin[1]],
                    size,
                };
                let image = bits_image(maker, bits, samples, area.end(), layers)?;
                for i in 0..layers {
                    let at = (src.level, src.layer + i);
                    let source = target(from.texture, from.format, at, samples, area.end());
                    let into = target(&image, bits.0, (0, i), samples, area.end());
                    copy.uses.push(sampled_use(&source)?);
                    copy.phases[0].push(Step::Depth(DepthDraw::bits(
                        device,
                        &source,
                        from.aspect,
                        &into,
                        area,
                        &feature,
                    )?));
                }
                let written =
                    Access::new(attachment_output, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
                let read = transfer(vk::AccessFlags2::TRANSFER_READ);
                copy.uses
                    .push(image.image_use(0..1, 0..layers, written, true)?);
                copy.uses
                    .push(image.image_use(0..1, 0..layers, read, false)?);
                let raw = image.raw_image()?;
                copy.colours.push(image);
                let region = ImageRegion {
                    aspect: colour,
                    level: 0,
                    layer: 0,
                    ..*src
                };
                (raw, region)
            }
        };
        // Where it writes: the colour destination, or the bits the draws read where the region
        // lies in the destination.
        let (write_image, write_region) = match write_bits {
            None => (to.image, *dst),
            Some(bits) => {
                let area = DrawArea {
                    origin: [dst.origin[0], dst.origin[1]],
                    size,
                };
                let image = bits_image(maker, bits, samples, area.end(), layers)?;
                for i in 0..layers {
                    let at = (dst.level, dst.layer + i);
                    let source = target(&image, bits.0, (0, i), samples, area.end());
                    let into = target(to.texture, to.format, at, samples, area.end());
                    copy.uses.push(attachment_use(&into)?);
                    copy.phases[2].push(match to.aspect == stencil {
                        true => Step::Planes(StencilPlanes::from_colour(
                            device, &source, &into, area, &feature,
                        )?),
                        false => Step::Depth(DepthDraw::from_colour(
                            device, &source, &into, area, &feature,
                        )?),
                    });
                }
                let written = transfer(vk::AccessFlags2::TRANSFER_WRITE);
                let read = Access::new(
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                );
                copy.uses
                    .push(image.image_use(0..1, 0..layers, written, true)?);
                copy.uses
                    .push(image.image_use(0..1, 0..layers, read, false)?);
                let raw = image.raw_image()?;
                copy.colours.push(image);
                let region = ImageRegion {
                    aspect: colour,
                    level: 0,
                    layer: 0,
                    ..*dst
                };
                (raw, region)
            }
        };
        copy.phases[1].push(Step::Copy(
            read_image,
            write_image,
            image_copy(&read_region, &write_region),
        ));
    }
    Ok(copy)
}

impl SampleCopy {
    /// What the `FragDepth` draw changes where it writes a 32-bit float depth inexactly (the
    /// announced `drawn-depth-copy`); `None` where every step is exact.
    pub(super) fn inexact_depth(&self) -> Option<&'static str> {
        self.inexact_depth
    }

    /// Announce, once the copy is recorded, `drawn-depth-copy` for `call` into `destination`
    /// where the `FragDepth` draw writes its depth inexactly.
    pub(super) fn announce(&self, device: &Device, call: &'static str, destination: &Texture) {
        let Some(changes) = self.inexact_depth else {
            return;
        };
        depth_copy::announce_changes(device, changes, || {
            let format = destination.pixel_format();
            format!("`{call}` into a multisampled {format:?} texture")
        });
    }

    /// Record the steps phase by phase, a barrier between each phase and the next.
    ///
    /// # Safety
    ///
    /// `raw` is `device`'s command buffer, recording outside a render pass, which only the caller
    /// records; the images, and what this copy goes through, are live until it completes and
    /// rest in `GENERAL`; the regions lie in them, each copy's images have one sample count and
    /// formats Vulkan lets it copy between, and each draw's targets are as it was made for
    /// (planned above).
    pub(super) unsafe fn record(&self, device: &Vk, raw: vk::CommandBuffer) {
        let general = vk::ImageLayout::GENERAL;
        let phases = self.phases.iter().filter(|phase| !phase.is_empty());
        // SAFETY: the caller's contract above.
        unsafe {
            for (i, phase) in phases.enumerate() {
                if i > 0 {
                    Barrier::everything().record(device, raw);
                }
                for step in phase {
                    match step {
                        Step::Copy(from, to, region) => {
                            device.cmd_copy_image(raw, *from, general, *to, general, &[*region])
                        }
                        Step::Depth(draw) => draw.record(device, raw),
                        Step::Planes(planes) => planes.record(device, raw),
                    }
                }
            }
        }
    }
}
