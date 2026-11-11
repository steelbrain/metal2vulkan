//! Separate depth and stencil textures in one pass (step 6's second part).
//!
//! Metal renders a pass whose depth and stencil attachments are two textures: measured on an M4
//! (macOS 27.0.1), `Depth32Float` with `Stencil8` and `Depth16Unorm` with `Stencil8` both test and
//! store each aspect in its own texture. A Vulkan subpass has one depth/stencil attachment, so such
//! a pass renders into a scratch image of the combined format whose depth aspect holds the depth
//! texture's bits: `D32_SFLOAT_S8_UINT` for 32-bit float depth, `D16_UNORM_S8_UINT` for 16-bit,
//! `D24_UNORM_S8_UINT` for 24-bit. Stencil is 8 bits in every format.
//!
//! - **Before the pass**, each aspect loaded with `Load` is copied from its texture into the
//!   scratch image; `Clear` and `DontCare` load as the scratch's own operations. An aspect loaded
//!   `DontCare` that the pass does not store, in a pass whose draws reach a tile, takes the answer
//!   of any attachment so loaded (`clear_reach.rs`, `UnstoredDontCarePolicy`), each aspect apart:
//!   the scratch's `Clear` to the aspect's clear value, Metal's answer (an M3 Max under macOS
//!   27.0.1, `runtime_lane::dont_care_unstored`), or its `DONT_CARE`, announced
//!   `dont-care-unstored-load`. One the pass stores loads `DONT_CARE`.
//! - **After it**, each aspect stored with `Store`, `StoreAndMultisampleResolve` or
//!   `CustomSampleDepthStore` is copied back (a resolve of the scratch image into the separate
//!   resolve textures is `resolve.rs`'s split resolve). An aspect stored `DontCare` is not: its
//!   texture keeps its texels, which is what Metal left in every case measured, so this is closer
//!   to Metal than Vulkan's `DONT_CARE` on one attachment.
//! - **The copies** are bit-wise, in two steps, since Vulkan copies no image of one depth/stencil
//!   format into another of a different one. A single-sampled aspect goes through a staging
//!   buffer. A multisampled one goes through a multisampled colour image whose format matches the
//!   aspect ([`colour_twin`]), with `vkCmdCopyImage` (equal sample counts), which maintenance8's
//!   "Compatible Formats for Depth-Stencil to/from Color Copies" allows; Vulkan copies no
//!   multisampled image to or from a buffer. They cover the render area, which the scratch image
//!   is at least as large as. The render encoder's commands are unordered, so each direction's
//!   copies take a barrier from everything before and one to everything after, and one between
//!   their two steps.
//! - **Multisampled aspects without colour copies.** Where the device lacks maintenance8 or its
//!   driver runs its depth-stencil/colour copies wrong
//!   (`RuntimeDriverDefect::MultisampledDepthStencilColourCopies`), each aspect is copied by
//!   sample-rate draws instead: the stencil by `stencil_copy.rs`'s plane draws, exact, and the
//!   depth by `depth_copy.rs`'s `FragDepth` draw, where the device's fragment depth is measured
//!   to store it unchanged (`depth_copy::copies_exactly`: a 32-bit float depth, and, a 16-bit one's
//!   codes); a 32-bit float depth elsewhere, which the draw may change, as the announced
//!   near-equivalent `drawn-depth-copy` (`depth_copy::drawn`: the values MoltenVK's and NVIDIA's
//!   are measured to change, or those Vulkan leaves to a driver not measured). In,
//!   from the texture's aspect, sampled, into the scratch image's (the depth first, then the
//!   stencil); out, from the scratch image's, which is then made sampled, into the texture's. No
//!   byte crosses between texels of different sizes.
//! - The scratch image is the pass's own, made with the encoder, and its command buffer holds it
//!   until it completes.
//! - **Textures of different sizes**, which Metal's validation stops ("When both depth (8 x 8) and
//!   stencil (4 x 4) are used from separate textures, they must have identical dimensions") and
//!   plain Metal renders over the smaller (an M4 with macOS 27.0.1, one sample,
//!   without draws; Metal probe `separate_sizes` on an M4, with one): the scratch image is the
//!   smaller, the render area at most that, and its copies move the render area. Metal's clear of
//!   the larger reaches past the render area to the end of its tiles, by 32 pixels a side for
//!   `Depth32Float` and 16 for `Stencil8` (a 64x64 depth cleared at 0..31, a 64x64 stencil at
//!   0..15, an 8x8 depth whole, each beside a 4x4 texture; and 32 alike under 16x16 tiles). A
//!   texture that ends inside those tiles, cleared and stored, is cleared whole before the copies
//!   back (`cleared_whole`); one past them, of another format, or of more samples, unmeasured, is
//!   `NotImplemented(Render)` (`check_sizes`).
//! - **A layered pass** (`renderTargetArrayLength` 2 or more) renders through a scratch image of
//!   its layers, each texture's from its slice, and the copies move every layer; the copies' draws,
//!   each of one layer, run once per layer.
//!
//! - **A resolve drawn after the pass** (`depth_stencil_resolve.rs`: a `Min` or `Max` depth on a
//!   device without the mode) reads the scratch image's samples, so the image is sampled where
//!   such a resolve may run and the device samples its format (`resolve::may_draw`).
//!
//! Refused: a device without the combined format as a depth/stencil attachment that copies both
//! ways (and, where the scratch image is read back by draws, is sampled), or, for multisampled
//! aspects copied through colour images, without a matching colour format it renders to and
//! copies both ways (`Unsupported`, naming the format); and, on a device without maintenance8 (`Unsupported`,
//! `Feature::SeparateMultisampledDepthStencil`, naming the feature) or whose driver copies a
//! multisampled depth/stencil image to or from a colour one wrong (the same feature, naming
//! `RuntimeDriverDefect::MultisampledDepthStencilColourCopies`, `driver_defects.rs`), a
//! multisampled depth that is loaded or may be stored where its draw is not measured (a driver
//! the fragment-depth probe of the depth's format has not measured), and a multisampled
//! aspect loaded from a texture that is not sampled. A device that lacks what the draws need
//! refuses the copies naming what it lacks.

use super::blit::{reads, writes, TRANSFER};
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::depth_copy::{self, DepthDraw, Drawn};
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::image_plan::{sample_count, ImagePlan};
use super::render_pass::{AspectClear, AspectTarget, Target};
use super::render_values::{LoadAction, StoreAction};
use super::resource::ResourceKey;
use super::sample_positions;
use super::stencil_copy::{self, StencilPlanes};
use super::texture::{copy_units, ImageMaker, Texture};
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::ResourceOptions;
use super::Device;
use ash::vk;
use metal2vulkan::bind::CopySource;
use metal2vulkan::device::{DeviceProfile, FormatUsage, VkFeature};
use metal2vulkan::error::{Capability, Feature, RuntimeDriverDefect, Unsupported};
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
};

/// The depth and stencil textures of a pass that renders them through a scratch image: two
/// textures, or a stencil-only one alone at custom sample positions (no depth).
#[derive(Clone, Debug)]
pub(super) struct Separate {
    pub(super) depth: Option<Target>,
    pub(super) stencil: Target,
}

impl Separate {
    /// The textures the pass renders through the scratch image, the depth's first.
    pub(super) fn targets(&self) -> impl Iterator<Item = &Target> {
        self.depth.iter().chain(std::iter::once(&self.stencil))
    }
}

/// The combined formats a stencil-only attachment's scratch image may take, in the order tried:
/// each with a depth aspect, so the flag custom sample positions need, and Metal's name for it.
const STENCIL_SCRATCH: [(Format, PixelFormat); 3] = [
    (
        Format::D32_SFLOAT_S8_UINT,
        PixelFormat::Depth32FloatStencil8,
    ),
    (Format::D24_UNORM_S8_UINT, PixelFormat::Depth24UnormStencil8),
    // Metal has no such format; nothing reads the scratch texture's pixel format.
    (Format::D16_UNORM_S8_UINT, PixelFormat::Depth16Unorm),
];

/// The scratch format of a stencil-only attachment at custom sample positions (`samples`): the
/// first of [`STENCIL_SCRATCH`] the device renders and copies both ways (and samples, where
/// `sampled`), made with the custom-location flag at the count. Refused
/// `NotImplemented(SamplePositionsAtBegin)` where none is, as the pass was before.
fn stencil_scratch(
    device: &Device,
    samples: metal2vulkan::vk::SampleCountFlags,
    sampled: bool,
) -> Result<(Format, PixelFormat), Error> {
    let profile = device.compile_device().profile();
    let facts = device.shared().textures.facts(profile);
    let mut needs = FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT
        | FormatFeatureFlags2::TRANSFER_SRC
        | FormatFeatureFlags2::TRANSFER_DST;
    if sampled {
        needs |= FormatFeatureFlags2::SAMPLED_IMAGE;
    }
    STENCIL_SCRATCH
        .into_iter()
        .find(|&(format, _)| {
            facts.optimal(format).contains(needs)
                && sample_positions::compatible_depth(profile, format, samples)
                    .contains(ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT)
        })
        .ok_or(Error::NotImplemented(RuntimeGap::SamplePositionsAtBegin))
}

/// The combined format whose depth aspect holds the bits of `depth`'s depth aspect.
fn combined(depth: Format) -> Option<Format> {
    match depth {
        Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT => Some(Format::D32_SFLOAT_S8_UINT),
        Format::D16_UNORM | Format::D16_UNORM_S8_UINT => Some(Format::D16_UNORM_S8_UINT),
        Format::X8_D24_UNORM_PACK32 | Format::D24_UNORM_S8_UINT => Some(Format::D24_UNORM_S8_UINT),
        _ => None,
    }
}

/// Whether an aspect with this load and store action copies between its texture and the scratch
/// image: it is loaded with `Load`, or its store action stores the samples (an `Unknown` one may
/// still become `Store`; `MultisampleResolve` leaves them, its resolve copying from a scratch
/// image of its own, `resolve.rs`).
fn aspect_copies(load: LoadAction, store: StoreAction) -> bool {
    load == LoadAction::Load || stores_samples(store)
}

/// Whether an aspect stored with `store` is copied back from the scratch image: an action that
/// keeps the samples, or `Unknown`, which may still become one.
fn stores_samples(store: StoreAction) -> bool {
    matches!(
        store,
        StoreAction::Store
            | StoreAction::StoreAndMultisampleResolve
            | StoreAction::CustomSampleDepthStore
            | StoreAction::Unknown
    )
}

/// `plan` with `SAMPLED` usage, where the copies without colour copies read its image: a
/// multisampled depth/stencil attachment, not transient and not yet sampled, of a format the
/// device samples, on a device whose colour copies cannot move its aspects
/// ([`colour_copies_refused`]). Its separate texture's aspects are then copied by draws, and its
/// shader-side stencil's shadow (`shadow_stencil.rs`) by the split and the planes, where it
/// would be refused unsampled. `Device::with_free_usage` gives the image the usage only where it
/// costs nothing. `None` otherwise.
pub(super) fn sampled_candidate(plan: &ImagePlan, device: &Device) -> Option<ImagePlan> {
    let aspects = super::texture::image_aspects(plan.format);
    let refused =
        colour_copies_refused(device, Feature::SeparateMultisampledDepthStencil, aspects).is_some();
    sampled_candidate_on(plan, device.compile_device().profile(), refused)
}

/// [`sampled_candidate`] on a device of `profile` whose colour copies cannot move a
/// multisampled aspect where `copies_refused`.
pub(super) fn sampled_candidate_on(
    plan: &ImagePlan,
    profile: &DeviceProfile,
    copies_refused: bool,
) -> Option<ImagePlan> {
    let usage = plan.usage;
    let candidate = copies_refused
        && plan.samples.as_raw() > 1
        && usage.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        && !usage.intersects(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSIENT_ATTACHMENT)
        && profile.supports_format(plan.format, FormatUsage::SampledImage);
    candidate.then(|| ImagePlan {
        usage: usage | ImageUsageFlags::SAMPLED,
        ..plan.clone()
    })
}

/// Why maintenance8's depth-stencil/colour copies cannot move the multisampled `aspects` on
/// `device`: it lacks maintenance8, or its driver runs them wrong (every aspect's, or, the
/// stencil's alone), refused naming `feature`. `None` where they can.
pub(super) fn colour_copies_refused(
    device: &Device,
    feature: Feature,
    aspects: ImageAspectFlags,
) -> Option<Unsupported> {
    if !device.enabled().features.contains(&VkFeature::Maintenance8) {
        return Some(Unsupported::needs(
            feature,
            Capability::Feature(VkFeature::Maintenance8),
        ));
    }
    copies_defect(&device.properties().driver_defects, aspects)
        .map(|defect| Unsupported::runtime_driver_defect(feature, defect))
}

/// The defect of a driver with `defects` that copies the multisampled `aspects` wrong through
/// maintenance8's depth-stencil/colour copies, if any: one that runs every aspect's wrong, or a
/// stencil's alone where `aspects` holds the stencil.
pub(super) fn copies_defect(
    defects: &std::collections::BTreeSet<RuntimeDriverDefect>,
    aspects: ImageAspectFlags,
) -> Option<RuntimeDriverDefect> {
    let every = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
    let stencil = RuntimeDriverDefect::MultisampledStencilColourCopies;
    if defects.contains(&every) {
        Some(every)
    } else {
        (aspects.contains(ImageAspectFlags::STENCIL) && defects.contains(&stencil))
            .then_some(stencil)
    }
}

/// The tile, in pixels a side, through which Metal clears and stores a separate texture of
/// `pixel_format` past the render area of a pass of `samples` (the module docs): measured at one
/// sample for `Depth32Float`, the tile every other attachment's clear reaches in
/// (`clear_reach::TILE`), and `Stencil8`, whose separate texture's is smaller.
fn clear_tile(pixel_format: PixelFormat, samples: u32) -> Option<u32> {
    match (pixel_format, samples) {
        (PixelFormat::Depth32Float, 1) => Some(super::clear_reach::TILE),
        (PixelFormat::Stencil8, 1) => Some(16),
        _ => None,
    }
}

/// Whether `target`, a separate texture of a pass of `samples` over the render area `extent`,
/// lies past the render area but inside the tiles Metal's clear reaches (the module docs).
pub(super) fn cleared_whole(target: &Target, extent: [u32; 2], samples: u32) -> bool {
    let past = (0..2).any(|axis| target.extent[axis] > extent[axis]);
    past && clear_tile(target.texture.pixel_format(), samples).is_some_and(|tile| {
        (0..2).all(|axis| target.extent[axis] <= extent[axis].div_ceil(tile).saturating_mul(tile))
    })
}

/// `separate`'s textures of different sizes in a pass of `samples` over the render area
/// `extent`: each that lies past the render area must end inside the tiles Metal's clear reaches
/// (`cleared_whole`), else the pass is `NotImplemented(Render)` (the module docs).
pub(super) fn check_sizes(
    separate: &Separate,
    extent: [u32; 2],
    samples: u32,
) -> Result<(), Error> {
    let mut extents = separate.targets().map(|target| target.extent);
    let first = extents.next();
    if extents.all(|extent| Some(extent) == first) {
        return Ok(());
    }
    for target in separate.targets() {
        let past = (0..2).any(|axis| target.extent[axis] > extent[axis]);
        if past && !cleared_whole(target, extent, samples) {
            return Err(Error::NotImplemented(RuntimeGap::Render));
        }
    }
    Ok(())
}

/// Whether the multisampled `aspect` of a pass of `samples` is copied by draws
/// (`stencil_copy.rs`, `depth_copy.rs`) rather than through colour images: the colour copies
/// cannot move it ([`colour_copies_refused`]).
fn by_draws(device: &Device, samples: u32, aspect: ImageAspectFlags) -> bool {
    samples > 1
        && colour_copies_refused(device, Feature::SeparateMultisampledDepthStencil, aspect)
            .is_some()
}

/// The scratch image `separate`'s pass renders into, as its depth/stencil target, whose depth and
/// stencil aspects load with `loads` and store with `stores`, and whose samples a resolve drawn
/// after the pass may read where `drawn` (`resolve::may_draw`). Refused as the module docs say.
pub(super) fn scratch(
    maker: &CommandBuffer,
    separate: &Separate,
    loads: [LoadAction; 2],
    stores: [StoreAction; 2],
    drawn: bool,
) -> Result<Target, Error> {
    let device = maker.device();
    // The first texture: both have the pass's samples and layers, and the scratch image the
    // smaller extent (textures of different sizes: the module docs).
    let first = separate.depth.as_ref().unwrap_or(&separate.stencil);
    let extent = separate.targets().fold(first.extent, |least, target| {
        [
            least[0].min(target.extent[0]),
            least[1].min(target.extent[1]),
        ]
    });
    let samples = first.samples;
    let multisampled = samples > 1;
    let samples_flag = sample_count(samples)
        .ok_or_else(|| internal("an attachment of a sample count Vulkan lacks"))?;
    let (format, pixel_format) = match &separate.depth {
        Some(depth) => (
            combined(depth.format).ok_or_else(|| internal("a depth attachment without depth"))?,
            depth.texture.pixel_format(),
        ),
        // A stencil-only texture at custom positions: its stencil, copied by draws, reads the
        // scratch image back where it may store.
        None => stencil_scratch(
            device,
            samples_flag,
            by_draws(device, samples, ImageAspectFlags::STENCIL)
                && stores[1] != StoreAction::DontCare,
        )?,
    };
    // The scratch image has the pass's layers (a layered pass's: added).
    let layers = first.layers;
    let feature = Feature::SeparateMultisampledDepthStencil;
    let facts = device
        .shared()
        .textures
        .facts(device.compile_device().profile());
    let lacks = |features: FormatFeatureFlags2, usage: FormatUsage| match facts
        .optimal(format)
        .contains(features)
    {
        true => Ok(()),
        false => Err(Error::Unsupported(Unsupported::needs(
            Feature::Format,
            Capability::Format { format, usage },
        ))),
    };
    // The scratch image is an attachment the copies write, checked before a draw's objects are
    // made for its format: a render pass of a format the device does not render to is invalid
    // (lavapipe, RADV and ANV make no `D16_UNORM_S8_UINT` attachment).
    lacks(
        FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT
            | FormatFeatureFlags2::TRANSFER_SRC
            | FormatFeatureFlags2::TRANSFER_DST,
        FormatUsage::DepthStencilAttachment,
    )?;
    // Each aspect's copies through colour images where they move it, else by draws: in, from
    // its texture, which is read, where it loads; out, from the scratch image, which is then
    // read, where it may store.
    let mut read_scratch = false;
    let aspects = [
        (0, separate.depth.as_ref(), ImageAspectFlags::DEPTH),
        (1, Some(&separate.stencil), ImageAspectFlags::STENCIL),
    ];
    for (index, target, aspect) in aspects {
        let Some(target) = target else {
            continue;
        };
        if !multisampled || !aspect_copies(loads[index], stores[index]) {
            continue;
        }
        let Some(refused) =
            colour_copies_refused(device, Feature::SeparateMultisampledDepthStencil, aspect)
        else {
            usable_twin(device, target, aspect)?;
            continue;
        };
        let is_depth = aspect == ImageAspectFlags::DEPTH;
        // The depth's draw is exact only where its fragment depth is measured so; a 32-bit float
        // depth runs it elsewhere as the announced near-equivalent, a 16-bit one not
        // (`depth_copy::drawn`).
        if is_depth && depth_copy::drawn(device, target.format).is_none() {
            return Err(Error::Unsupported(refused));
        }
        let draws_run = |destination: Format| {
            if is_depth {
                depth_copy::depth_draw_runs(device, destination, samples, &feature)
            } else {
                stencil_copy::plane_draws_run(
                    device,
                    CopySource::Aspect,
                    destination,
                    samples,
                    &feature,
                )
            }
        };
        if loads[index] == LoadAction::Load {
            if !stencil_copy::sampled(target)? {
                return Err(Error::Unsupported(refused));
            }
            draws_run(format)?;
        }
        if stores_samples(stores[index]) {
            draws_run(target.format)?;
            read_scratch = true;
        }
    }
    if read_scratch {
        lacks(
            FormatFeatureFlags2::SAMPLED_IMAGE,
            FormatUsage::SampledImage,
        )?;
    }
    // A resolve drawn after the pass reads the samples too, where the device samples the format;
    // elsewhere that resolve is refused (`depth_stencil_resolve::draws_run`).
    let sampled = read_scratch
        || (drawn
            && multisampled
            && facts
                .optimal(format)
                .contains(FormatFeatureFlags2::SAMPLED_IMAGE));
    let samples = samples_flag;
    let [width, height] = extent;
    // The descriptor states the image's size and samples; its pixel format is the depth
    // texture's, which nothing reads of an image only the pass uses (Metal has no format of
    // some combined images, `D16_UNORM_S8_UINT` among them).
    let descriptor = TextureDescriptor {
        texture_type: match (first.samples > 1, layers > 1) {
            (false, false) => TextureType::Type2D,
            (false, true) => TextureType::Type2DArray,
            (true, false) => TextureType::Type2DMultisample,
            (true, true) => TextureType::Type2DMultisampleArray,
        },
        pixel_format,
        width: width.into(),
        height: height.into(),
        array_length: layers.into(),
        sample_count: first.samples.into(),
        usage: TextureUsage::RENDER_TARGET,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format: descriptor.pixel_format,
        // The pass's depth/stencil attachment, made for its custom sample positions where the
        // device takes them.
        flags: sample_positions::compatible_depth(
            device.compile_device().profile(),
            format,
            samples,
        ),
        image_type: ImageType::TYPE_2D,
        format,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples,
        usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST
            | if sampled {
                ImageUsageFlags::SAMPLED
            } else {
                ImageUsageFlags::empty()
            },
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(format, ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL),
        unavailable: TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
    Ok(Target {
        texture,
        level: 0,
        layer: 0,
        layers,
        format,
        samples: first.samples,
        extent,
    })
}

/// The colour format whose texels hold the bits of `format`'s `aspect`, and Metal's name for it,
/// through which a multisampled aspect is copied: `R32_SFLOAT` for 32-bit and 24-bit depth,
/// `R16_UNORM` for 16-bit depth, `R8_UNORM` for stencil (maintenance8's "Compatible Formats for
/// Depth-Stencil to/from Color Copies"). Float and normalized formats, so the device's colour
/// sample counts govern them, not its narrower integer ones.
pub(super) fn colour_twin(
    format: Format,
    aspect: ImageAspectFlags,
) -> Option<(Format, PixelFormat)> {
    let block = format.block()?;
    if aspect == ImageAspectFlags::STENCIL {
        return block
            .stencil
            .map(|_| (Format::R8_UNORM, PixelFormat::R8Unorm));
    }
    block.depth?;
    match format {
        Format::D32_SFLOAT
        | Format::D32_SFLOAT_S8_UINT
        | Format::X8_D24_UNORM_PACK32
        | Format::D24_UNORM_S8_UINT => Some((Format::R32_SFLOAT, PixelFormat::R32Float)),
        Format::D16_UNORM | Format::D16_UNORM_S8_UINT => {
            Some((Format::R16_UNORM, PixelFormat::R16Unorm))
        }
        _ => None,
    }
}

/// The colour twin of `target`'s `aspect` ([`colour_twin`]), checked: a colour attachment format
/// the device copies both ways. Refused, naming the format, where it is not.
fn usable_twin(
    device: &Device,
    target: &Target,
    aspect: ImageAspectFlags,
) -> Result<(Format, PixelFormat), Error> {
    let twin = colour_twin(target.format, aspect)
        .ok_or_else(|| internal("a separate attachment without its aspect"))?;
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

/// A multisampled colour image of `format` (Metal's `pixel_format`), `samples`, `extent` and
/// `layers`, which only the copies of a pass use: a colour attachment (so it may have more than
/// one sample) that copies both ways ([`usable_twin`]). Refused where the device does not make it
/// (a sample count its colour attachments of the format do not take).
pub(super) fn colour_image(
    maker: &dyn ImageMaker,
    (format, pixel_format): (Format, PixelFormat),
    samples: u32,
    [width, height]: [u32; 2],
    layers: u32,
) -> Result<Texture, Error> {
    let samples_flag = sample_count(samples)
        .ok_or_else(|| internal("an attachment of a sample count Vulkan lacks"))?;
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
        usage: TextureUsage::RENDER_TARGET,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format,
        flags: ImageCreateFlags::empty(),
        image_type: ImageType::TYPE_2D,
        format,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples: samples_flag,
        usage: ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::TRANSFER_SRC
            | ImageUsageFlags::TRANSFER_DST,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
        unavailable: TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    Device::new_planned_texture(maker, &descriptor, plan)
}

/// One step of an aspect's copy.
enum Step {
    /// From an image into the staging buffer.
    ToBuffer(vk::Image, vk::BufferImageCopy),
    /// From the staging buffer into an image.
    FromBuffer(vk::Image, vk::BufferImageCopy),
    /// From one image into another of the same samples.
    Image(vk::Image, vk::Image, vk::ImageCopy),
    /// A clear of an image's aspect, a texture larger than the render area (`cleared_whole`).
    Clear(
        vk::Image,
        vk::ClearDepthStencilValue,
        vk::ImageSubresourceRange,
    ),
}

/// One direction of a pass's copies between its separate textures and its scratch image, over
/// the render area: into the scratch image before the pass, or back after it.
pub(super) struct ScratchCopy {
    /// What the copies go through, held until the command buffer completes: a staging buffer
    /// (single-sampled aspects) or multisampled colour images.
    staging: Option<Buffer>,
    colours: Vec<Texture>,
    /// Each aspect's first step, then its second, a barrier between.
    first: Vec<Step>,
    second: Vec<Step>,
    /// A multisampled depth the colour copies cannot move: its draw of each layer, each after a
    /// barrier.
    depth: Vec<DepthDraw>,
    /// A multisampled stencil the colour copies cannot move: its plane draws of each layer, each
    /// after a barrier (and after the depth's draws).
    planes: Vec<StencilPlanes>,
    uses: Vec<Use<ResourceKey>>,
    /// The values the depth's draws change or may change, where the driver's fragment depth is
    /// not measured to keep every value (`depth_copy::Drawn::Changes`), and the depth texture's
    /// pixel format: the copy announces `drawn-depth-copy` as it is recorded.
    inexact_depth: Option<(&'static str, PixelFormat)>,
}

impl ScratchCopy {
    /// The copies of `separate`'s aspects into `scratch` (`into_scratch`, the aspects
    /// `depth` and `stencil` load with `Load`) or back (the ones they store with `Store`), over
    /// the render area `extent`; `None` when no aspect asks for one.
    pub(super) fn plan(
        maker: &CommandBuffer,
        separate: &Separate,
        scratch: &Target,
        actions: [Option<&AspectTarget>; 2],
        extent: [u32; 2],
        into_scratch: bool,
    ) -> Result<Option<ScratchCopy>, Error> {
        let device = maker.device();
        let aspects = [
            (separate.depth.as_ref(), actions[0], ImageAspectFlags::DEPTH),
            (
                Some(&separate.stencil),
                actions[1],
                ImageAspectFlags::STENCIL,
            ),
        ];
        let actions: Vec<_> = aspects
            .into_iter()
            .filter_map(|(texture, action, aspect)| Some((texture?, action?, aspect)))
            .filter(|(_, action, _)| {
                // `StoreAndMultisampleResolve` keeps the samples too, and
                // `CustomSampleDepthStore` stores a depth as `Store` does.
                if into_scratch {
                    action.load == LoadAction::Load
                } else {
                    matches!(
                        action.store,
                        StoreAction::Store
                            | StoreAction::StoreAndMultisampleResolve
                            | StoreAction::CustomSampleDepthStore
                    )
                }
            })
            .collect();
        let wanted: Vec<_> = actions
            .iter()
            .map(|&(texture, _, aspect)| (texture, aspect))
            .collect();
        if wanted.is_empty() {
            return Ok(None);
        }
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        // Every layer of the pass's (the scratch image's), each texture's from its slice.
        let count = scratch.layers;
        let at = |target: &Target| target.level..target.level + 1;
        let on = |target: &Target| target.layer..target.layer + count;
        let layers = |target: &Target, aspect: ImageAspectFlags| vk::ImageSubresourceLayers {
            aspect_mask: aspect.into(),
            mip_level: target.level,
            base_array_layer: target.layer,
            layer_count: count,
        };
        let [width, height] = extent;
        let size = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let mut uses = Vec::new();
        let (mut first, mut second) = (Vec::new(), Vec::new());
        if scratch.samples == 1 {
            let texels = u64::from(width) * u64::from(height);
            let mut length = 0u64;
            let mut offsets = Vec::new();
            for &(texture, aspect) in &wanted {
                let bytes = copy_units(texture.format, texture.texture.pixel_format())
                    .into_iter()
                    .find(|unit| unit.0 == aspect)
                    .map(|unit| u64::from(unit.1))
                    .ok_or_else(|| internal("a separate attachment without its aspect"))?;
                // A depth/stencil copy's buffer offset is a multiple of 4; its layers lie one
                // after another.
                let offset = length.next_multiple_of(4);
                length = offset + bytes * texels * u64::from(count);
                offsets.push(offset);
            }
            let staging = device.own_buffer(length, ResourceOptions::private())?;
            uses.push(writes(&staging, 0..length));
            uses.push(reads(&staging, 0..length));
            // A texture larger than the render area, cleared and stored, inside the tiles
            // Metal's clear reaches, is cleared whole before its copy back (the module docs).
            for &(texture, action, aspect) in &actions {
                if into_scratch
                    || action.load != LoadAction::Clear
                    || !cleared_whole(texture, extent, scratch.samples)
                {
                    continue;
                }
                let value = match action.clear {
                    AspectClear::Depth(depth) => vk::ClearDepthStencilValue { depth, stencil: 0 },
                    AspectClear::Stencil(stencil) => vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil,
                    },
                };
                let range = vk::ImageSubresourceRange {
                    aspect_mask: aspect.into(),
                    base_mip_level: texture.level,
                    level_count: 1,
                    base_array_layer: texture.layer,
                    layer_count: count,
                };
                first.push(Step::Clear(texture.texture.raw_image()?, value, range));
            }
            for (&(texture, aspect), offset) in wanted.iter().zip(offsets) {
                let region = |target: &Target| vk::BufferImageCopy {
                    buffer_offset: staging.at(offset).1,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: layers(target, aspect),
                    image_offset: vk::Offset3D::default(),
                    image_extent: size,
                };
                let (from, to) = if into_scratch {
                    (texture, scratch)
                } else {
                    (scratch, texture)
                };
                uses.push(from.texture.image_use(at(from), on(from), read, false)?);
                uses.push(to.texture.image_use(at(to), on(to), write, true)?);
                first.push(Step::ToBuffer(from.texture.raw_image()?, region(from)));
                second.push(Step::FromBuffer(to.texture.raw_image()?, region(to)));
            }
            return Ok(Some(ScratchCopy {
                staging: Some(staging),
                colours: Vec::new(),
                first,
                second,
                depth: Vec::new(),
                planes: Vec::new(),
                uses,
                inexact_depth: None,
            }));
        }
        let mut colours = Vec::new();
        let (mut depth, mut planes) = (Vec::new(), Vec::new());
        let mut inexact_depth = None;
        for &(texture, aspect) in &wanted {
            let by_draws = by_draws(device, scratch.samples, aspect);
            let (from, to) = if into_scratch {
                (texture, scratch)
            } else {
                (scratch, texture)
            };
            if by_draws {
                let feature = Feature::SeparateMultisampledDepthStencil;
                if aspect == ImageAspectFlags::DEPTH {
                    match depth_copy::drawn(device, texture.format) {
                        Some(Drawn::Exact) => {}
                        Some(Drawn::Changes(changes)) => {
                            inexact_depth = Some((changes, texture.texture.pixel_format()))
                        }
                        None => {
                            return Err(internal(
                                "a multisampled depth copy its draw does not copy at all",
                            ))
                        }
                    }
                }
                // The draws are of one layer: each layer of the pass's its own, from each
                // target's first.
                for i in 0..count {
                    let layer = |target: &Target| Target {
                        layer: target.layer + i,
                        layers: 1,
                        ..target.clone()
                    };
                    let (from, to) = (layer(from), layer(to));
                    uses.push(stencil_copy::sampled_use(&from)?);
                    uses.push(stencil_copy::attachment_use(&to)?);
                    if aspect == ImageAspectFlags::DEPTH {
                        depth.push(DepthDraw::new(device, &from, &to, extent, &feature)?);
                    } else {
                        planes.push(StencilPlanes::from_aspect(
                            device, &from, &to, extent, &feature,
                        )?);
                    }
                }
                continue;
            }
            let twin = usable_twin(device, texture, aspect)?;
            let colour = colour_image(maker, twin, scratch.samples, extent, count)?;
            let colour_target = Target {
                texture: colour.clone(),
                level: 0,
                layer: 0,
                layers: count,
                format: twin.0,
                samples: scratch.samples,
                extent,
            };
            let copy = |src: &Target, src_aspect, dst: &Target, dst_aspect| vk::ImageCopy {
                src_subresource: layers(src, src_aspect),
                src_offset: vk::Offset3D::default(),
                dst_subresource: layers(dst, dst_aspect),
                dst_offset: vk::Offset3D::default(),
                extent: size,
            };
            uses.push(from.texture.image_use(at(from), on(from), read, false)?);
            uses.push(colour.image_use(0..1, 0..count, write, true)?);
            uses.push(colour.image_use(0..1, 0..count, read, false)?);
            uses.push(to.texture.image_use(at(to), on(to), write, true)?);
            let colour_raw = colour.raw_image()?;
            first.push(Step::Image(
                from.texture.raw_image()?,
                colour_raw,
                copy(from, aspect, &colour_target, ImageAspectFlags::COLOR),
            ));
            second.push(Step::Image(
                colour_raw,
                to.texture.raw_image()?,
                copy(&colour_target, ImageAspectFlags::COLOR, to, aspect),
            ));
            colours.push(colour);
        }
        Ok(Some(ScratchCopy {
            staging: None,
            colours,
            first,
            second,
            depth,
            planes,
            uses,
            inexact_depth,
        }))
    }

    /// Record the copies in `core`'s command buffer, between barriers from everything before and
    /// to everything after, and hold what they go through until the command buffer completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let ScratchCopy {
            staging,
            colours,
            first,
            second,
            depth,
            planes,
            uses,
            inexact_depth,
        } = self;
        if let Some((changes, format)) = inexact_depth {
            depth_copy::announce_changes(core.device(), changes, || {
                format!(
                    "a render pass's separate multisampled {format:?} depth texture, copied to \
                     and from its scratch image"
                )
            });
        }
        let stages = if !depth.is_empty() || !planes.is_empty() {
            TRANSFER | stencil_copy::DRAW_STAGES
        } else {
            TRANSFER
        };
        // A multisampled pass's copies take no buffer.
        let buffer = staging
            .as_ref()
            .map_or(vk::Buffer::null(), |staging| staging.at(0).0);
        core.barrier(Barrier::everything());
        core.command(stages, &uses, |device, raw| {
            let general = vk::ImageLayout::GENERAL;
            let step = |step: &Step| match step {
                // SAFETY: the encoder's command buffer is recording outside a render pass; the
                // images and the staging buffer are live until it completes (the encoder retains
                // the textures, and this what the copies go through); every image rests in
                // `GENERAL`; each region lies in its images and in the buffer, and an image copy's
                // images have one sample count and formats of one size whose aspects
                // maintenance8 lets it copy (planned above).
                Step::ToBuffer(image, region) => unsafe {
                    device.cmd_copy_image_to_buffer(raw, *image, general, buffer, &[*region])
                },
                // SAFETY: as above.
                Step::FromBuffer(image, region) => unsafe {
                    device.cmd_copy_buffer_to_image(raw, buffer, *image, general, &[*region])
                },
                // SAFETY: as above.
                Step::Image(from, to, region) => unsafe {
                    device.cmd_copy_image(raw, *from, general, *to, general, &[*region])
                },
                // SAFETY: as above; the range is the texture's level and the pass's layers, and
                // its image takes transfer writes (its copy back is one).
                Step::Clear(image, value, range) => unsafe {
                    device.cmd_clear_depth_stencil_image(raw, *image, general, value, &[*range])
                },
            };
            first.iter().for_each(step);
            // SAFETY: the encoder's command buffer, of this device, is recording, and only this
            // encoder records it.
            unsafe { Barrier::everything().record(device, raw) };
            second.iter().for_each(step);
            // SAFETY: as above; the draws' sources and destinations and their objects are live
            // until the command buffer completes (the encoder retains the textures, and this the
            // draws' objects).
            unsafe {
                for depth in &depth {
                    Barrier::everything().record(device, raw);
                    depth.record(device, raw);
                }
                for planes in &planes {
                    Barrier::everything().record(device, raw);
                    planes.record(device, raw);
                }
            }
        });
        core.barrier(Barrier::everything());
        core.retain(staging);
        core.retain(colours);
        core.retain(depth);
        core.retain(planes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each depth format's combined format holds its depth bits; a format without depth has none.
    #[test]
    fn combined_formats_keep_the_depth_bits() {
        assert_eq!(
            combined(Format::D32_SFLOAT),
            Some(Format::D32_SFLOAT_S8_UINT)
        );
        assert_eq!(
            combined(Format::D32_SFLOAT_S8_UINT),
            Some(Format::D32_SFLOAT_S8_UINT)
        );
        assert_eq!(combined(Format::D16_UNORM), Some(Format::D16_UNORM_S8_UINT));
        assert_eq!(
            combined(Format::D24_UNORM_S8_UINT),
            Some(Format::D24_UNORM_S8_UINT)
        );
        assert_eq!(combined(Format::S8_UINT), None);
    }

    /// Each aspect's colour twin is Metal's format of the Vulkan one, one of the aspect's bits
    /// (maintenance8's table); an aspect the format lacks has none.
    #[test]
    fn colour_twins_hold_each_aspects_bits() {
        use metal2vulkan::metal::format::direct_format;
        let (depth, stencil) = (ImageAspectFlags::DEPTH, ImageAspectFlags::STENCIL);
        for (format, aspect, twin) in [
            (Format::D32_SFLOAT, depth, Format::R32_SFLOAT),
            (Format::D32_SFLOAT_S8_UINT, depth, Format::R32_SFLOAT),
            (Format::D32_SFLOAT_S8_UINT, stencil, Format::R8_UNORM),
            (Format::D24_UNORM_S8_UINT, depth, Format::R32_SFLOAT),
            (Format::D24_UNORM_S8_UINT, stencil, Format::R8_UNORM),
            (Format::X8_D24_UNORM_PACK32, depth, Format::R32_SFLOAT),
            (Format::D16_UNORM, depth, Format::R16_UNORM),
            (Format::D16_UNORM_S8_UINT, depth, Format::R16_UNORM),
            (Format::D16_UNORM_S8_UINT, stencil, Format::R8_UNORM),
            (Format::S8_UINT, stencil, Format::R8_UNORM),
        ] {
            let (vk, metal) = colour_twin(format, aspect).expect("a twin");
            assert_eq!(vk, twin, "{format:?} {aspect:?}");
            assert_eq!(direct_format(metal), Some(twin), "{format:?} {aspect:?}");
        }
        assert_eq!(colour_twin(Format::S8_UINT, depth), None);
        assert_eq!(colour_twin(Format::D32_SFLOAT, stencil), None);
        assert_eq!(colour_twin(Format::R32_SFLOAT, depth), None);
    }

    /// An aspect copies when it loads, or may store.
    #[test]
    fn copies_follow_the_actions() {
        use LoadAction::{Clear, DontCare as NoLoad, Load};
        use StoreAction::{DontCare, Store, Unknown};
        assert!(!aspect_copies(Clear, DontCare));
        assert!(!aspect_copies(NoLoad, DontCare));
        assert!(aspect_copies(Load, DontCare));
        assert!(aspect_copies(Clear, Store));
        assert!(aspect_copies(Clear, Unknown));
        assert!(!aspect_copies(Clear, StoreAction::MultisampleResolve));
        assert!(aspect_copies(
            Clear,
            StoreAction::StoreAndMultisampleResolve
        ));
    }
}
