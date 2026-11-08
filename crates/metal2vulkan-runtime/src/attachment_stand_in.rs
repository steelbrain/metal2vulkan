//! Images that stand in for a pass's attachment for the whole pass (step 6's fourth part).
//!
//! - **A memoryless texture's backing.** A pass split into instances (`render.rs`) keeps its
//!   attachments' contents in memory between them, and a draw that fetches an attachment in the
//!   pixel interlock reads and writes it as a storage image (`framebuffer_fetch.rs`): neither is
//!   open to a transient image, which may have no memory and has no `STORAGE` usage. Such a pass
//!   renders the memoryless attachment into an image of its own. Nothing is copied: Metal neither
//!   loads nor stores a memoryless texture, so its contents do not survive the pass either way. A
//!   color's backing is one its queue keeps for its passes of the attachment's shape
//!   ([`pooled_backing`]).
//! - **A fetched texture's scratch.** One rule for both forms of framebuffer fetch: a color
//!   texture whose image lacks the usage its fetch form needs renders, in a pass that fetches
//!   it, into a scratch image that has it, as its attachment in every instance too, but those
//!   that render the texture itself before it arrives (recorded inline, or pinned by a draw that
//!   reads it through a snapshot: `render/inline_first.rs`, `render/snapshot.rs`), after which it
//!   is copied in. In the
//!   pixel interlock that usage is `STORAGE`, which a texture the runtime makes holds only where
//!   it costs the image nothing (`framebuffer_fetch::Answers`); through input attachments it is
//!   input attachment usage, which every color render target the runtime makes holds and a
//!   caller's image may not (`texture/wrapped.rs`). A caller's image without `STORAGE` takes the
//!   scratch in the interlock too. Through sample planes (`sample_planes.rs`) the usage is the
//!   planes' copies': sampled, with a view in the unsigned-integer format of its texel size,
//!   which a multisampled color render target the runtime makes takes only where it costs the
//!   image nothing, like `STORAGE`. The texture's level and layer are copied into the scratch
//!   image before the pass where Metal loads it (`Load`; a `Clear` or `DontCare` load is the
//!   scratch image's own first operation) and back after the pass where Metal stores its texels
//!   (`Store`, `StoreAndMultisampleResolve`). A pass that stores `DontCare` copies nothing back:
//!   the texture keeps its texels, as Apple's GPU keeps them. In the pixel interlock an 8-bit sRGB
//!   texture renders, in a pass that fetches it, into a scratch made mutable to the
//!   unsigned-integer view a fetching draw reads and writes it through
//!   (`framebuffer_fetch::storage_view_format`), to which a texture's image is not made; so does
//!   a BGRA-order texture viewed in its RGBA-order twin (ANV) whose image does not take that view.
//!
//! - **A depth/stencil texture not made for the pass's custom sample positions**. Vulkan rasterizes
//!   a depth/stencil attachment at custom positions only where its image was made with
//!   `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT` (`sample_positions.rs`), which
//!   every image the runtime makes for them has and a caller's image may lack. Such a pass renders
//!   the texture's level and layer into a stand-in made with the flag: each aspect copied in before
//!   the pass where Metal loads it (or keeps it, an aspect of the texture's format no attachment
//!   names), and back after it where Metal stores or keeps it, over the reach of the pass's clear
//!   past the render area where an aspect is cleared and stored (`clear_reach.rs`), the copy in
//!   covering the same extent. The copies keep the format and the sample count, so they are plain
//!   image copies of the aspects. A caller's image is wrapped without the flag
//!   (`texture/wrapped.rs`). An image of a format the device does not copy both ways (no transfer
//!   usage) stays refused (`NotImplemented(SamplePositionsAtBegin)`).
//! - **A depth/stencil texture without `RenderTarget` usage**, whose image lacks depth/stencil
//!   attachment usage, which plain Metal renders as a render target: one
//!   texture renders through a stand-in copied as the custom positions' one is, each aspect in
//!   where Metal loads or keeps it and back where it stores or keeps it, so a `DontCare` store
//!   leaves the texture as it was, as Metal leaves it. Separate depth and stencil textures render
//!   through their combined scratch image (`depth_stencil_scratch.rs`), which copies them alike. A
//!   texture whose format the device does not copy both ways, and a memoryless one, stay refused
//!   (`NotImplemented(Render)`).
//! - **A texture over a buffer a pass does not store** (`DontCare`, or `MultisampleResolve`), whose
//!   image is linear over the buffer's bytes: Metal leaves the buffer's bytes, so the pass renders
//!   it through the image its texture keeps, copied into where the pass loads it (`Load`), and
//!   never back (`kept_bytes.rs`).
//! - **A multisampled color attachment whose samples a resolve after the pass reads**, which the
//!   pass stores nothing of (`MultisampleResolve`): Metal leaves the texture's samples as they
//!   were, and the crate-owned resolve (`color_resolve.rs`) reads the samples the pass stores, so
//!   the pass renders it through the image its texture keeps, copied into where the pass loads it
//!   (`Load`), and never back (`resolve::keep_samples`).
//! - **A pass's hidden depth and stencil**. In a pass without a depth/stencil attachment, plain
//!   Metal tests a draw whose state uses depth or the stencil against a hidden depth of `float`,
//!   1.0 at the pass's start, and a stencil of 0, kept per sample through the pass. Such a draw
//!   gives the pass an image of its own for them (`hidden`): the depth/stencil attachment of the
//!   draw's instance and every later one, cleared before the pass and never stored.
//!
//! A stand-in has the attachment's format, size and samples; its copies keep the sample count
//! both ways. A multisampled one (a memoryless attachment's backing, or a fetched texture's
//! scratch) takes the usage the device's fetch form needs of a multisampled attachment: through
//! the attachment's own multisampled storage image
//! (`framebuffer_fetch::through_multisampled_storage`), `STORAGE` where the device stores its
//! format at its sample count; through sample planes (`framebuffer_fetch::through_planes`), sampled
//! with the copies' view where the device has its formats, and no `STORAGE`, which such a device
//! has at one sample only; through input attachments, input attachment usage. A multisampled color
//! one takes input attachment usage too where the exact color resolve may read it
//! (`color_resolve.rs`). Only a pass of fetching draws needs a stand-in, and a pass of draws that
//! test the stencil in the shader (`shadow_stencil.rs`) one for its memoryless stencil. A
//! single-sampled one takes the usages either fetch form needs where the device has them. A color
//! one takes transfer writes (it may be cleared before the pass, `render.rs`), and a depth/stencil
//! one transfer reads and writes (the shadow's copies). A stand-in is the pass's own: made with the
//! draw that needs it and held by the command buffer until it completes, like the depth/stencil
//! scratch (`depth_stencil_scratch.rs`) and the resolve scratch (`resolve.rs`); but a memoryless
//! color's backing, which its queue keeps ([`pooled_backing`]). The copies cover
//! the render area and every layer of a layered pass, between barriers from everything before and
//! to everything after.

use super::blit::TRANSFER;
use super::color_resolve;
use super::command_buffer::CommandBuffer;
use super::device::lock;
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::framebuffer_fetch;
use super::image_plan::{sample_count, ImagePlan};
use super::nan_clear;
use super::render_pass::{PassLayout, Target};
use super::resource::ResourceKey;
use super::sample_positions;
use super::texture::Texture;
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::ResourceOptions;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::format::{aspects, FormatMapping};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
};

/// What one attachment's stand-in stands in for: the attachment's own level and layer, which the
/// pass's target names no longer.
#[derive(Clone, Debug)]
pub(super) struct StandIn {
    pub(super) own: Target,
    /// Why the attachment renders through it, which names its notice
    /// (`Device::degrade_instances`) and whether it is copied to and from the attachment.
    pub(super) cause: Cause,
    /// The box its copies cover, `[x0, y0, x1, y1]` (half-open) inside the render area, where a
    /// pass loads the attachment and its draws write inside the box alone
    /// (`linear_attachment::choose`): `None` copies over the render area (or the reach of the
    /// pass's clear, `clear_reach.rs`).
    pub(super) area: Option<[u32; 4]>,
}

/// Why an attachment renders through a stand-in for the pass. Each cause names its own notice:
/// before, every stand-in copied to and from its attachment was announced as a fetched texture's
/// scratch (`fetch-through-scratch`), so a texture over a buffer that rendered through the image
/// its texture keeps was announced as read "through the pixel interlock" where no draw read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Cause {
    /// A memoryless texture's backing, or the backing a texture keeps for a pass that clears it
    /// and resolves it alone into another format: nothing is copied.
    Backing,
    /// A fetched texture's scratch, with the usage the fetch's form needs
    /// (`fetch-through-scratch`): `by`, the draw that first fetched it in the pass, in words (its
    /// fragment function, and the blend state of the attachment's color).
    Fetch { by: String },
    /// A texture without render target usage (`undeclared-usage-shadow`).
    Unattachable,
    /// A depth/stencil texture's, made for the pass's custom sample positions
    /// (`positions-through-scratch`).
    Positions,
    /// A bound linear attachment's image its texture keeps (`linear_attachment.rs`, which
    /// announces it under its own kinds).
    Linear,
    /// The image a texture over a buffer keeps for a pass that does not store it, so that the
    /// buffer keeps its bytes (`kept_bytes.rs`, which announces it, `dont-care-stand-in`): copied
    /// into where the pass loads it, and never back.
    KeptBytes,
    /// A multisampled color attachment the pass stores nothing of (`MultisampleResolve`) whose
    /// samples a resolve the runtime draws after the pass reads, which the pass must store for it
    /// (`resolve::keep_samples`): it renders through the image its texture keeps for the queue,
    /// copied into where the pass loads it and never back, so the texture keeps the samples it
    /// held before the pass, as Metal's does.
    KeptSamples,
    /// The image a pass renders a texture's own image through where it stores it `DontCare`
    /// under `DontCareColorPolicy::Kept` and the device's own `DONT_CARE` would not leave the
    /// texture's earlier texels, which Metal leaves (`kept_bytes.rs`): one its queue keeps for its passes of the attachment's shape
    /// ([`pooled_backing`]), copied into where the pass loads it and never back. No notice names
    /// it: the texture keeps its texels, Metal's answer.
    KeptTexels,
}

impl StandIn {
    /// Whether it is copied to and from its attachment: all but a backing.
    pub(super) fn copies(&self) -> bool {
        self.cause != Cause::Backing
    }
}

/// A memoryless attachment's backing that its texture keeps ([`kept_backing`]): for the passes
/// of one queue, of one subresource of the texture (its level, layers, format, samples and
/// extent as the pass's target names them).
pub(super) struct KeptBacking {
    queue: u64,
    /// It was made for copies to and from its attachment (`stand_in`'s `copies`).
    copies: bool,
    level: u32,
    layer: u32,
    layers: u32,
    format: metal2vulkan::vk::Format,
    samples: u32,
    extent: [u32; 2],
    backing: Target,
}

impl KeptBacking {
    /// Whether it backs `own` for the passes of `queue`, copied to and from it where `copies`.
    fn backs(&self, own: &Target, queue: u64, copies: bool) -> bool {
        self.queue == queue
            && (self.copies || !copies)
            && self.level == own.level
            && self.layer == own.layer
            && self.layers == own.layers
            && self.format == own.format
            && self.samples == own.samples
            && self.extent == own.extent
    }
}

/// The backing of `own`, a memoryless attachment of a pass of `queue` whose image is lazily
/// allocated (`render/executions.rs`'s `Rendered::back_memoryless`), or an attachment a pass
/// without a draw call clears and resolves into another pixel format (`render.rs`), whose samples
/// Metal leaves as they were: the stand-in the attachment's texture keeps for that queue and
/// subresource, made now and kept where it keeps none. Metal neither loads nor stores a memoryless
/// texture, and the other kind of pass clears the backing first, so no pass reads what another left
/// in it, and the passes of one queue that back it are ordered by their uses of it, as by their
/// uses of the texture: one backing serves them all, and the memory a texture's backings take does
/// not grow with its passes. It goes with the texture (each command buffer that rendered it holds
/// it until it completes). Or, `copies`, a linear attachment a pass renders through a stand-in
/// (`linear_attachment.rs`), copied in where the pass loads it and back where it stores it, as a
/// texture without render target usage is: the copies, and the clears of the tiles a `DontCare`
/// load reaches (`clear_reach.rs`), give each pass all it reads of the backing first, so one serves
/// them too. So does a texture over a buffer that a pass does not store (`kept_bytes.rs`), copied
/// in where the pass loads it and never back: nothing reads what such a pass leaves there.
pub(super) fn kept_backing(
    maker: &CommandBuffer,
    own: &Target,
    queue: u64,
    copies: bool,
) -> Result<Target, Error> {
    let device = maker.device();
    let kept = own.texture.kept_backings();
    if let Some(found) = lock(kept).iter().find(|k| k.backs(own, queue, copies)) {
        // Held as one `maker` made: its first batch moves it to `GENERAL` where the command
        // buffer that made it has not yet submitted a batch.
        maker.hold_own_texture(found.backing.texture.clone());
        return Ok(found.backing.clone());
    }
    // Made with no lock held; one another thread kept meanwhile is taken, and this one dropped.
    let made = stand_in(maker, own, copies)?;
    let mut kept = lock(kept);
    if let Some(found) = kept.iter().find(|k| k.backs(own, queue, copies)) {
        maker.hold_own_texture(found.backing.texture.clone());
        return Ok(found.backing.clone());
    }
    device.shared().counts.kept_backing_made();
    kept.push(KeptBacking {
        queue,
        copies,
        level: own.level,
        layer: own.layer,
        layers: own.layers,
        format: own.format,
        samples: own.samples,
        extent: own.extent,
        backing: made.clone(),
    });
    Ok(made)
}

/// The most color backings a queue keeps ([`pooled_backing`]), and the most bytes of
/// device memory their images take together: past either, the backing passes took least recently
/// goes (but the one just made, which alone may take more). The count is above the shapes a
/// frame cycles through: a compositor's frames on the consumer's macOS guest render 11 to 14
/// extents of one memoryless `RGBA16Float` attachment (1920x1080 down to 192x64), about 40 MB of
/// backings together, and a pool that keeps fewer than a frame takes, in the same order each
/// frame, finds none of them kept.
const MOST_POOLED_BACKINGS: usize = 32;
const MOST_POOLED_BYTES: u64 = 256 << 20;

/// A color attachment's backing its queue keeps ([`pooled_backing`]): the shape it backs (the
/// attachment's pixel format, the format its pass renders it in, its samples, layers and extent),
/// whether it was made for copies from its attachment (`stand_in`'s `copies`), the bytes its image
/// takes, and the pool's count of takes at its last take.
struct PooledBacking {
    pixel_format: PixelFormat,
    format: metal2vulkan::vk::Format,
    samples: u32,
    layers: u32,
    extent: [u32; 2],
    copies: bool,
    bytes: u64,
    taken: u64,
    backing: Target,
}

impl PooledBacking {
    /// Whether it backs an attachment of `own`'s shape, copied from it where `copies`.
    fn backs(&self, own: &Target, copies: bool) -> bool {
        (self.copies || !copies)
            && self.pixel_format == own.texture.pixel_format()
            && self.format == own.format
            && self.samples == own.samples
            && self.layers == own.layers
            && self.extent == own.extent
    }
}

/// The color backings one command queue keeps for its passes ([`pooled_backing`]): at
/// most [`MOST_POOLED_BACKINGS`], over at most [`MOST_POOLED_BYTES`] but for the one just made.
/// They go with the queue; each command buffer that rendered into one holds it until it
/// completes.
#[derive(Default)]
pub(crate) struct BackingPool {
    kept: Vec<PooledBacking>,
    /// The takes so far, the clock each backing's `taken` reads.
    takes: u64,
}

impl BackingPool {
    /// Drop the backings taken least recently while the pool keeps more than its bounds, but
    /// the last one, just made.
    fn bound(&mut self) {
        loop {
            let bytes: u64 = self.kept.iter().map(|k| k.bytes).sum();
            let over = self.kept.len() > MOST_POOLED_BACKINGS || bytes > MOST_POOLED_BYTES;
            let last = self.kept.len().saturating_sub(1);
            let oldest = self.kept[..last]
                .iter()
                .enumerate()
                .min_by_key(|(_, k)| k.taken)
                .map(|(at, _)| at);
            match oldest {
                Some(at) if over => drop(self.kept.remove(at)),
                _ => return,
            }
        }
    }
}

/// The backing of `own`, a memoryless color attachment of a pass of `maker`'s queue that renders
/// into an image of its own (a framebuffer fetch through the pixel interlock reads it, or the pass
/// splits), or a color a pass stores `DontCare` that renders through one so that its texture
/// keeps its texels (`kept_bytes.rs`), made for the copy into it where `copies` (the pass loads
/// it): one the queue keeps of `own`'s pixel format, format, samples, layers and extent (made for
/// copies where `copies`) that is none of `taken` (the images the pass renders its other
/// attachments into), else one made now and kept ([`BackingPool`]). Metal neither loads nor stores
/// a memoryless texture (`render_pass::check_memoryless`), and a pass gives a memoryless color's
/// every texel a draw can read a value before its draws: its `Clear` load, or the zero its
/// `DontCare` load clears to in a pass that draws (`clear_reach::dont_care_loads`), before the pass
/// where the pass leaves it out of its subpass (`render.rs`); a `DontCare` store's backing is
/// given the same before its pass, or the texture's texels over the render area where the pass
/// loads it, and nothing reads what its pass leaves in it. A fresh image's texels were undefined
/// there as well. So no pass reads what another left in the backing, and the passes of one queue
/// that take it are ordered by their uses of it, as by any texture's: one image serves every pass
/// of the shape, across command buffers and frames (`WorkCounts::pooled_backings_made`).
pub(super) fn pooled_backing(
    maker: &CommandBuffer,
    own: &Target,
    taken: &[Texture],
    copies: bool,
) -> Result<Target, Error> {
    let queue = maker.queue_place();
    let free = |k: &PooledBacking| k.backs(own, copies) && !taken.contains(&k.backing.texture);
    {
        let mut pool = lock(queue.memoryless_backings());
        pool.takes += 1;
        let takes = pool.takes;
        if let Some(found) = pool.kept.iter_mut().find(|k| free(k)) {
            found.taken = takes;
            let backing = found.backing.clone();
            drop(pool);
            // Held as one `maker` made: its first batch moves it to `GENERAL` where the command
            // buffer that made it has not yet submitted a batch.
            maker.hold_own_texture(backing.texture.clone());
            return Ok(backing);
        }
    }
    // Made with no lock held; the pool may keep one another thread made meanwhile beside it.
    let made = stand_in(maker, own, copies)?;
    maker.device().shared().counts.pooled_backing_made();
    let mut pool = lock(queue.memoryless_backings());
    pool.takes += 1;
    let takes = pool.takes;
    pool.kept.push(PooledBacking {
        pixel_format: own.texture.pixel_format(),
        format: own.format,
        samples: own.samples,
        layers: own.layers,
        extent: own.extent,
        copies,
        bytes: made.texture.placed_bytes().unwrap_or(0),
        taken: takes,
        backing: made.clone(),
    });
    pool.bound();
    Ok(made)
}

/// The stand-in for `target`, an attachment the pass renders whole into an image of its own
/// (the module docs): `copies` where the pass copies between them (a fetched texture's scratch).
/// A single-sampled color stand-in takes `STORAGE` usage where the device stores the format of a
/// fetching draw's storage view of it (`framebuffer_fetch::storage_view_format`), which every
/// fetched format's has (the compile layer's interlock form needs it), and is made mutable to
/// that view where it is another format (an 8-bit sRGB format's, a BGRA-order one's RGBA-order
/// twin on ANV); a multisampled one where the
/// device's fetch reads the attachment's own multisampled image and stores that view at its
/// sample count. Either takes input attachment usage where the device's
/// framebuffer fetch reads input attachments, and a multisampled one the sample planes' copies'
/// usage where the fetch reads planes and the device has it. A sample count or usage the
/// device's image limits for the format lack is refused where the image is made (`Unsupported`,
/// `image_plan::check_image_limits`). A stand-in has the pass's layers (a layered pass's
/// attachment: added).
pub(super) fn stand_in(
    maker: &CommandBuffer,
    target: &Target,
    copies: bool,
) -> Result<Target, Error> {
    stand_in_as(maker, target, copies, target.format)
}

/// The stand-in for `target` in `format`, a format of the attachment's texel size: a color the
/// pass's draws render in another encoding where the attachment's image takes no view of it
/// (`render.rs`'s `Reformat`), copied between them, as [`stand_in`]'s scratch is otherwise.
/// Refused (`NotImplemented(Render)`) where the device copies no image of `format`.
pub(super) fn stand_in_as(
    maker: &CommandBuffer,
    target: &Target,
    copies: bool,
    format: metal2vulkan::vk::Format,
) -> Result<Target, Error> {
    let device = maker.device();
    // A 3D attachment's planes, and a 1D attachment, are not copied to or from a stand-in
    // (`render_pass::not_2d`).
    if super::render_pass::not_2d(target) {
        return Err(Error::NotImplemented(RuntimeGap::Render));
    }
    let profile = device.compile_device().profile();
    let layers = target.layers;
    let multisampled = target.samples != 1;
    let pixel_format = target.texture.pixel_format();
    let aspects = aspects(pixel_format).unwrap_or(ImageAspectFlags::COLOR);
    let color = aspects == ImageAspectFlags::COLOR;
    let mut usage = if color {
        ImageUsageFlags::COLOR_ATTACHMENT
    } else {
        ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
    };
    let own_image = !multisampled
        || (framebuffer_fetch::through_multisampled_storage(profile, target.samples)
            && framebuffer_fetch::stores_samples(profile, target.samples));
    // A fetching draw's storage view is in `view`: an 8-bit sRGB format's, the unsigned-integer
    // format of its texel size, a BGRA-order one's, its RGBA-order twin on a device that accesses
    // its format without one in neither direction (`framebuffer_fetch::storage_view_format`).
    let view = framebuffer_fetch::storage_view_format(profile, format);
    let stored = color && own_image && profile.supports_format(view, FormatUsage::StorageImage);
    if stored {
        usage |= ImageUsageFlags::STORAGE;
    }
    // The crate-owned color resolve reads a multisampled extended-range color attachment as an
    // input attachment (`color_resolve.rs`), as does the resolve into the attachment's format's
    // sRGB twin, whose encoding no device's average writes (`color_resolve::twin_resolved`).
    let resolved = multisampled
        && (color_resolve::shader_rules(pixel_format, target.format, target.samples).is_some()
            || color_resolve::twin_resolved(pixel_format, target.samples));
    if color && (framebuffer_fetch::through_inputs(profile) || resolved) {
        usage |= ImageUsageFlags::INPUT_ATTACHMENT;
    }
    // A color stand-in may be cleared before the pass (`render.rs`), and a scratch is copied. A
    // depth/stencil one may be copied to and from a shader-side stencil's shadow
    // (`shadow_stencil.rs`), whichever draw first made the pass need the stand-in.
    let transfer = device.shared().textures.facts(profile).optimal(format);
    if color && transfer.contains(FormatFeatureFlags2::TRANSFER_DST) {
        usage |= ImageUsageFlags::TRANSFER_DST;
    }
    let both = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
    if !color && transfer.contains(both) {
        usage |= ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    }
    // A multisampled depth/stencil stand-in (a memoryless attachment's backing) is read sampled
    // where the device's depth/stencil-colour image copies cannot move its samples: a
    // shader-side depth test's split fills the depth's stand-in from it, as the copies of a
    // separate texture read one (`depth_stencil_scratch::sampled_candidate`). It is the
    // runtime's own image, so no answer the caller sees moves.
    let feature = metal2vulkan::error::Feature::FramebufferFetch;
    if !color
        && multisampled
        && super::depth_stencil_scratch::colour_copies_refused(device, feature, aspects).is_some()
        && profile.supports_format(format, FormatUsage::SampledImage)
    {
        usage |= ImageUsageFlags::SAMPLED;
    }
    if copies {
        if !transfer.contains(both) {
            return Err(Error::NotImplemented(RuntimeGap::Render));
        }
        usage |= ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    }
    let [width, height] = target.extent;
    let descriptor = TextureDescriptor {
        texture_type: match (multisampled, layers > 1) {
            (false, false) => TextureType::Type2D,
            (false, true) => TextureType::Type2DArray,
            (true, false) => TextureType::Type2DMultisample,
            (true, true) => TextureType::Type2DMultisampleArray,
        },
        pixel_format,
        width: width.into(),
        height: height.into(),
        array_length: layers.into(),
        sample_count: target.samples.into(),
        usage: TextureUsage::RENDER_TARGET,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let samples = sample_count(target.samples)
        .ok_or_else(|| internal("an attachment of a sample count Vulkan lacks"))?;
    let mut plan = ImagePlan {
        pixel_format,
        // A depth/stencil stand-in is the pass's attachment, made for its custom sample positions
        // where the device takes them.
        flags: sample_positions::compatible_depth(profile, format, samples),
        image_type: ImageType::TYPE_2D,
        format,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples,
        usage,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(format, aspects),
        unavailable: TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    // A stand-in a fetching draw views in another format than its own (an sRGB format's, an
    // RGBA-order twin's) is mutable to that view, with extended usage: its `STORAGE` is the
    // view's, which its own format need not have (a view in its own format serves the draws that
    // do not fetch it, each view's usage restricted to what it serves, `binding_view.rs`).
    if stored && view != format {
        plan.flags |= ImageCreateFlags::MUTABLE_FORMAT | ImageCreateFlags::EXTENDED_USAGE;
        plan.view_formats = vec![format, view];
    }
    // A stand-in in another format than the attachment's, copied to or from it, is mutable to
    // the attachment's format: a device that copies between formats through a view of one image
    // in the other's format copies it so (MoltenVK, whose copy of a multisampled image between
    // formats where neither takes such a view goes through a buffer, which holds no samples).
    if copies && format != target.format {
        plan.flags |= ImageCreateFlags::MUTABLE_FORMAT;
        if plan.view_formats.is_empty() {
            plan.view_formats.push(format);
        }
        plan.view_formats.push(target.format);
    }
    // A multisampled color stand-in is read and written by the sample planes' copies where the
    // fetch reads planes and the device views it so (`sample_planes.rs` refuses the planes
    // otherwise).
    if color && framebuffer_fetch::through_planes(profile, target.samples) {
        if let Some(planes) = framebuffer_fetch::with_planes_copies(&plan, profile) {
            plan = planes;
        }
    }
    // A multisampled 8-bit color stand-in that a resolve into its format's twin reads through
    // the other member of the pair (`color_resolve::twin_read_view`: an sRGB one through the
    // UNORM twin its rules are in, a UNORM one through the sRGB twin the device decodes) is
    // mutable to that view.
    let raw = color_resolve::twin_read_view(profile, pixel_format, plan.format, target.samples)
        .filter(|_| resolved);
    if let Some(raw) = raw {
        plan.flags |= ImageCreateFlags::MUTABLE_FORMAT;
        if plan.view_formats.is_empty() {
            plan.view_formats.push(plan.format);
        }
        if !plan.view_formats.contains(&raw) {
            plan.view_formats.push(raw);
        }
    }
    let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
    Ok(Target {
        texture,
        level: 0,
        layer: 0,
        layers,
        format,
        samples: target.samples,
        extent: target.extent,
    })
}

/// The image of `layout`'s hidden depth and stencil (`render_pass::DepthStencilTarget::hidden`):
/// a private `Depth32Float_Stencil8` texture of the pass's samples and layers over its render
/// area, as a depth/stencil attachment and cleared before the pass by a transfer. Metal's hidden
/// depth is a `float` (a Metal probe: a write at 0.5 compared unequal to 0.5 plus 2^-24), and its
/// stencil 8 bits; the one image holds both, so a pass whose draws use the one and then the other
/// keeps one attachment. Refused `Unsupported` where the device lacks the format as a depth/stencil
/// attachment at the pass's samples, and `NotImplemented(HiddenDepthStencil)` where it does not
/// clear it by a transfer.
pub(super) fn hidden(maker: &CommandBuffer, layout: &PassLayout) -> Result<Target, Error> {
    let device = maker.device();
    let profile = device.compile_device().profile();
    let pixel_format = PixelFormat::Depth32FloatStencil8;
    let usages = [FormatUsage::DepthStencilAttachment].into();
    let mapping = metal2vulkan::metal::format::vk_format(pixel_format, &usages, profile).map_err(
        |error| match error {
            metal2vulkan::error::Error::Unsupported(unsupported) => Error::Unsupported(unsupported),
            other => Error::Compile(other),
        },
    )?;
    let format = mapping.format;
    let transfer = device.shared().textures.facts(profile).optimal(format);
    if !transfer.contains(FormatFeatureFlags2::TRANSFER_DST) {
        return Err(Error::NotImplemented(RuntimeGap::HiddenDepthStencil));
    }
    let samples = sample_count(layout.samples)
        .ok_or_else(|| internal("a pass of a sample count Vulkan lacks"))?;
    let (layers, multisampled) = (layout.layers, layout.samples != 1);
    let [width, height] = layout.extent;
    let descriptor = TextureDescriptor {
        texture_type: match (multisampled, layers > 1) {
            (false, false) => TextureType::Type2D,
            (false, true) => TextureType::Type2DArray,
            (true, false) => TextureType::Type2DMultisample,
            (true, true) => TextureType::Type2DMultisampleArray,
        },
        pixel_format,
        width: width.into(),
        height: height.into(),
        array_length: layers.into(),
        sample_count: layout.samples.into(),
        usage: TextureUsage::RENDER_TARGET,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format,
        // The pass's attachment, made for its custom sample positions where the device takes
        // them, as a depth/stencil stand-in is.
        flags: sample_positions::compatible_depth(profile, format, samples),
        image_type: ImageType::TYPE_2D,
        format,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples,
        usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::TRANSFER_DST,
        view_formats: Vec::new(),
        mapping,
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
        samples: layout.samples,
        extent: layout.extent,
    })
}

/// The clear of a pass's hidden depth and stencil (`hidden`) to Metal's 1.0 and 0, every layer of
/// its image, recorded before the pass: each of the pass's instances loads it.
pub(super) struct HiddenClear {
    image: vk::Image,
    layers: u32,
    uses: [Use<ResourceKey>; 1],
}

impl HiddenClear {
    /// The clear of `target`, the hidden depth and stencil's image.
    pub(super) fn plan(target: &Target) -> Result<HiddenClear, Error> {
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        Ok(HiddenClear {
            image: target.texture.raw_image()?,
            layers: target.layers,
            uses: [target
                .texture
                .image_use(0..1, 0..target.layers, write, true)?],
        })
    }

    /// Record the clear in `core`'s command buffer, between barriers from everything before and
    /// to everything after.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let HiddenClear {
            image,
            layers,
            uses,
        } = self;
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |device, raw| {
            let value = vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            };
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: layers,
            };
            // SAFETY: the encoder's command buffer is recording outside a render pass; the image
            // is live until it completes (the encoder retains the pass's targets), rests in
            // `GENERAL`, takes transfer writes (`hidden`), and has one level and `layers` layers
            // of both aspects.
            unsafe {
                device.cmd_clear_depth_stencil_image(
                    raw,
                    image,
                    vk::ImageLayout::GENERAL,
                    &value,
                    &[range],
                );
            }
        });
        core.barrier(Barrier::everything());
    }
}

/// Refuse (`NotImplemented(Render)`) a fetched texture's scratch, or a stand-in for a texture
/// without render target usage, where `target`'s image does not copy both ways: its format has no
/// transfer usage on the device.
pub(super) fn check_copies(target: &Target) -> Result<(), Error> {
    let both = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    if target.texture.image_plan()?.usage.contains(both) {
        Ok(())
    } else {
        Err(Error::NotImplemented(RuntimeGap::Render))
    }
}

/// One direction of a pass's copies between its attachments and their stand-ins, over the render
/// area: into the stand-ins before the pass, or back after it.
pub(super) struct StandInCopies {
    /// `(from, to, region)` of each copy.
    copies: Vec<(vk::Image, vk::Image, vk::ImageCopy)>,
    uses: Vec<Use<ResourceKey>>,
}

impl StandInCopies {
    /// The copies between `pairs` (each color attachment's stand-in, then its own target, then
    /// the extent from the origin copied: the render area, or the reach of the pass's clear past
    /// it, `clear_reach.rs`): into the stand-ins (`into_stand_in`) or back; `None` when there is
    /// none.
    pub(super) fn plan(
        pairs: &[(&Target, &Target, [u32; 2])],
        into_stand_in: bool,
    ) -> Result<Option<StandInCopies>, Error> {
        let color: Vec<_> = pairs
            .iter()
            .map(|&(stand_in, own, [width, height])| {
                let area = [0, 0, width, height];
                (stand_in, own, area, vk::ImageAspectFlags::COLOR)
            })
            .collect();
        StandInCopies::plan_aspects(&color, into_stand_in)
    }

    /// [`StandInCopies::plan`] of `pairs` that each name the aspects copied: a color attachment's
    /// color, or a depth/stencil attachment's depth, stencil or both (one pair per stand-in and
    /// direction, so each image has one use). Each pair names the box copied, `[x0, y0, x1, y1]`
    /// (half-open): from the origin over the render area or a clear's reach, or a stand-in's
    /// `area`.
    pub(super) fn plan_aspects(
        pairs: &[(&Target, &Target, [u32; 4], vk::ImageAspectFlags)],
        into_stand_in: bool,
    ) -> Result<Option<StandInCopies>, Error> {
        if pairs.is_empty() {
            return Ok(None);
        }
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let mut copies = Vec::with_capacity(pairs.len());
        let mut uses = Vec::with_capacity(2 * pairs.len());
        for &(stand_in, own, [x0, y0, x1, y1], aspect_mask) in pairs {
            let (from, to) = if into_stand_in {
                (own, stand_in)
            } else {
                (stand_in, own)
            };
            // Every layer of the pass's, each image's from its first.
            let count = stand_in.layers;
            let layers = |target: &Target| vk::ImageSubresourceLayers {
                aspect_mask,
                mip_level: target.level,
                base_array_layer: target.layer,
                layer_count: count,
            };
            // Inside the render area, whose extent fits an `i32`.
            let offset = vk::Offset3D {
                x: x0 as i32,
                y: y0 as i32,
                z: 0,
            };
            let region = vk::ImageCopy {
                src_subresource: layers(from),
                src_offset: offset,
                dst_subresource: layers(to),
                dst_offset: offset,
                extent: vk::Extent3D {
                    width: x1.saturating_sub(x0),
                    height: y1.saturating_sub(y0),
                    depth: 1,
                },
            };
            if aspect_mask == vk::ImageAspectFlags::COLOR {
                let direction = if into_stand_in { "into" } else { "back from" };
                nan_clear::degrade_converting_copy(
                    to.texture.device(),
                    to.format,
                    to.samples,
                    || {
                        format!(
                            "the copy of a multisampled {:?} attachment {direction} the image it \
                         renders through",
                            own.texture.pixel_format()
                        )
                    },
                );
            }
            let at = |target: &Target| target.level..target.level + 1;
            let on = |target: &Target| target.layer..target.layer + count;
            uses.push(from.texture.image_use(at(from), on(from), read, false)?);
            uses.push(to.texture.image_use(at(to), on(to), write, true)?);
            copies.push((from.texture.raw_image()?, to.texture.raw_image()?, region));
        }
        Ok(Some(StandInCopies { copies, uses }))
    }

    /// The copies' uses.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Record the copies into `command_buffer`, between barriers from everything before and to
    /// everything after, as [`StandInCopies::record`] records them in an encoder's command
    /// buffer: a step of a pass recorded later (`render/executions.rs`), whose uses the pass
    /// took.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; the
    /// images are live until it completes and rest in `GENERAL`.
    pub(super) unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let general = vk::ImageLayout::GENERAL;
        // SAFETY: the caller's contract; every image has the transfer usage its copy needs
        // (checked when the stand-in was made), and each region lies in both images.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            for (from, to, region) in &self.copies {
                raw.cmd_copy_image(command_buffer, *from, general, *to, general, &[*region]);
            }
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// Record the copies in `core`'s command buffer, between barriers from everything before and
    /// to everything after.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let StandInCopies { copies, uses } = self;
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |device, raw| {
            let general = vk::ImageLayout::GENERAL;
            // SAFETY: the encoder's command buffer is recording outside a render pass; the images
            // are live until it completes (the encoder retains the textures); every image rests in
            // `GENERAL`, has the transfer usage its copy needs (checked when the stand-in was
            // made), and each region lies in both images (the render area).
            unsafe {
                for (from, to, region) in &copies {
                    device.cmd_copy_image(raw, *from, general, *to, general, &[*region]);
                }
            }
        });
        core.barrier(Barrier::everything());
    }
}
