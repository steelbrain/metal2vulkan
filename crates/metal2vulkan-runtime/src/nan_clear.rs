//! Clears of a NaN, and copies of NaN texels.
//!
//! **Color**. Metal writes the canonical quiet NaN for a `half` NaN clear (`0x7e00` of either
//! sign), and `0x7e0` and `0x3f0` for the 11- and 10-bit channels of `RG11B10Float`, the sign
//! dropped, but where it keeps the attachment's texture losslessly compressed, its level is 16
//! texels or more a side, and the pass has no draw call ([`keeps_nan_sign`]), where it keeps a
//! `half` NaN's sign. Its resolve of such a clear in a pass with no draw call keeps the clear's
//! texel or averages the samples, each NaN counted zero, and makes `0x7fc00000` in each `float`
//! NaN channel. Vulkan converts a clear value to the attachment's format as the device likes, and
//! resolves its samples as the device likes: the pass clears and resolves by the device, announced
//! (`device-nan-encoding`, [`degrade_nan_clears`]). An integer clear converts nothing.
//!
//! **Copies that convert NaNs**. A device whose copy into a multisampled float image draws the
//! samples through that image's format (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`:
//! RADV, whose copies store every 16-bit float NaN as `0xfe00`) does not copy a NaN's bits into a
//! multisampled image. A multisampled color render target the runtime makes of such a format takes
//! a view in the unsigned-integer format of its texel size where it costs the image nothing
//! ([`view_candidate`]), as one of `PixelFormatView` usage does, and a blit copy between two images
//! that both take the unsigned view runs through it, exactly (`sample_planes::BitsCopy`). Into an
//! image that takes none (a caller's, or one the view would cost) the device copies, announced once
//! (`device-nan-encoding`, [`degrade_converting_copy`]): a blit copy between textures and an
//! attachment stand-in's.
//!
//! **Depth** (our probe `nanclear`, the M3 Max under macOS 27.0.1, plain Metal): a `Clear` load
//! whose `clearDepth` is a NaN of any sign or payload stores `0x7fc00000` in every texel and
//! sample of a `Depth32Float` or `Depth32Float_Stencil8` attachment, 1x and 4x, and a draw in that
//! pass tests against it as Metal's test orders a NaN, above every value (a draw at 0.5 passes
//! `Less` and fails `Greater`); a `Depth16Unorm` one stores 0, and its draw tests against 0.
//! Vulkan's clear value takes no NaN. So a 16-bit depth clears to 0 ([`NanDepth::Cleared`]), and
//! a 32-bit float depth is loaded instead, the word written into its texture before the pass over
//! the clear's reach ([`NanDepth::Written`], [`NanDepthWrite`]): a single-sample one by a buffer
//! filled with it and copied in, a multisampled one by an `R32_UINT` image cleared to it (an
//! integer clear converts nothing) and copied in by maintenance8's depth/colour copy, or, where
//! those copies are missing or run wrong, drawn in by the `FragDepth` draw where the driver's
//! fragment depth stores every value unchanged (`depth_copy::drawn`). The pass's depth test then
//! compares the NaN as the device's does: Metal's where its test is Metal's own, and elsewhere as
//! it is, announced (`signed-depth-test`). Written before anything the pass copies its depth out
//! with (separate depth and stencil textures' scratch, a stand-in), which loads the word with the
//! rest.
//! Where it cannot be written, the pass clears to 1.0, the value Metal's test orders a NaN
//! nearest, announced (`signed-depth-test`; on a device that may clamp a depth outside `[0, 1]`,
//! `restricted-depth-upload`): a memoryless depth; a pass that does not store it on a device that
//! hands such a store to its driver (`render_target::DontCare::Discarded`, MoltenVK among them),
//! where Metal keeps the texture's earlier texels, which the word would replace; and a
//! multisampled depth on a device without either form of the write.

use super::blit::TRANSFER;
use super::buffer::Buffer;
use super::clear_value;
use super::command_buffer::CommandBuffer;
use super::degradation::Degradation;
use super::depth_copy::{self, DepthDraw, Drawn};
use super::depth_stencil_scratch::colour_copies_refused;
use super::encoder::EncoderCore;
use super::error::{internal, Error};
use super::framebuffer_fetch::views_in;
use super::image_plan::{compatible, ImagePlan};
use super::render_pass::{PassLayout, Target};
use super::render_target::DontCare;
use super::render_values::{LoadAction, StoreAction};
use super::resource::ResourceKey;
use super::sample_copy::bits_image;
use super::stencil_copy::{self, DrawArea};
use super::texture::heap_tiled::metal_compresses;
use super::texture::Texture;
use super::tracker::{Access, Barrier, Use};
use super::Device;
use ash::vk;
use metal2vulkan::bind::PlanesCopy;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Feature, RuntimeDriverDefect};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::{
    Format, ImageAspectFlags, ImageCreateFlags, ImageUsageFlags, NumericFormat,
};

/// Announce `PrecisionNote::DeviceNanEncoding` where a color attachment of `layout`, a pass whose
/// draws reach a tile where `drew`, clears to a NaN whose bits the device's conversion decides (the
/// module docs): a `half` or small-float channel's, or a resolve's of a pass whose draws reach no
/// tile, where every sample holds the clear.
pub(super) fn degrade_nan_clears(device: &Device, layout: &PassLayout, drew: bool) {
    let held = layout.colors.iter().flatten().find_map(|color| {
        let resolves = matches!(
            color.store,
            StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
        ) && color.resolve.is_some()
            && !drew
            && color.target.samples > 1;
        let pf = color.target.texture.pixel_format();
        let narrow = (color.load == LoadAction::Clear)
            .then(|| clear_value::nan_channels(pf, &color.clear))
            .flatten()?;
        (narrow || resolves).then_some((pf, narrow))
    });
    let Some((pf, narrow)) = held else {
        return;
    };
    device.degrade(Degradation::DeviceNanEncoding, || {
        let what = match narrow {
            true => "the device converts it to the format's NaN as it likes",
            false => "the device's resolve of its samples keeps or drops the NaN as it likes",
        };
        (
            format!("a {pf:?} color attachment's NaN clear"),
            format!(
                "{what}, where Metal stores its canonical quiet NaN or the clear's sign and payload"
            ),
        )
    });
}

/// Whether Metal's NaN clear of `target`, in a pass with a draw call where `called`, keeps a
/// `half` NaN's sign (the module docs): on a device that presents an Apple silicon Mac's
/// compression (`Device::metal_compresses_textures`; on one that presents a Mac keeping none, as
/// a Mac with an Intel or AMD GPU does, never: expected rather than measured), its texture (a view's first
/// texture) is one Metal keeps losslessly compressed (`heap_tiled::metal_compresses`: not over a
/// buffer, of no usage Metal keeps uncompressed, optimized contents allowed), the attachment's
/// level is 16 texels or more a side, counted in pixels at any sample count, and the pass has no
/// draw call. A resolve texture's level of the same keeps the clear's texel through the pass's
/// resolve.
pub(super) fn keeps_nan_sign(target: &Target, called: bool) -> bool {
    if called || !target.texture.device().metal_compresses_textures() {
        return false;
    }
    let mut root = &target.texture;
    while let Some(parent) = root.parent_texture() {
        root = parent;
    }
    let level = target.extent.map(u64::from);
    !root.over_buffer()
        && metal_compresses(
            true,
            root.usage(),
            root.allow_gpu_optimized_contents(),
            level,
        )
}

/// Whether `format` is a color format of floats narrower than 32 bits (a `half`'s, a packed
/// small float's): those whose NaNs a device whose multisampled copies convert float NaNs converts
/// (`driver_defects.rs`: RADV's copy exports them through a conversion).
pub(super) fn narrow_float(format: Format) -> bool {
    format.block().is_some_and(|block| {
        matches!(
            block.color,
            Some(NumericFormat::Sfloat | NumericFormat::Ufloat)
        ) && block.component_bits.is_none_or(|bits| bits < 32)
    })
}

/// Whether `device`'s copy into a multisampled image of `format` of `samples` samples converts
/// its NaNs: the device's multisampled copies convert float NaNs
/// (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`), and the format's are among those
/// converted ([`narrow_float`]).
pub(super) fn converting_copy(device: &Device, format: Format, samples: u32) -> bool {
    samples > 1 && narrow_float(format) && converts_nans(device)
}

/// Announce `device-nan-encoding` once for a copy into a multisampled image of `format` of
/// `samples` samples, `what` naming it, where the device's copy converts its NaNs
/// ([`converting_copy`]): they land as the device converts them, where Metal's copy keeps every
/// bit.
pub(super) fn degrade_converting_copy(
    device: &Device,
    format: Format,
    samples: u32,
    what: impl FnOnce() -> String,
) {
    if converting_copy(device, format, samples) {
        device.degrade(Degradation::DeviceNanEncoding, || {
            (
                what(),
                "the device's copy into a multisampled float image stores each NaN of a channel \
                 narrower than 32 bits as it converts it \
                 (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`), where Metal's copy \
                 keeps its sign and payload"
                    .to_owned(),
            )
        });
    }
}

/// Whether `device`'s driver copies into a multisampled float image through its format,
/// converting its NaNs (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`).
fn converts_nans(device: &Device) -> bool {
    device
        .properties()
        .driver_defects
        .contains(&RuntimeDriverDefect::MultisampledCopiesConvertFloatNans)
}

/// `plan` mutable with a view in the unsigned-integer format of its texel size, through which a
/// copy of its samples keeps their bits (the module docs, `sample_planes::BitsCopy`), where the
/// device's multisampled copies convert float NaNs: a multisampled color attachment that is not
/// transient, of a format of floats narrower than 32 bits ([`narrow_float`]), whose view format
/// is of its compatibility class and one the device renders at its sample count
/// ([`renders_unsigned`]). `None` elsewhere, and where the plan takes the view already.
pub(super) fn view_candidate(
    plan: &ImagePlan,
    device: &Device,
) -> Result<Option<ImagePlan>, Error> {
    let samples = plan.samples.as_raw();
    let Some(view) = PlanesCopy::for_format(plan.format).map(PlanesCopy::format) else {
        return Ok(None);
    };
    let candidate = samples > 1
        && converts_nans(device)
        && narrow_float(plan.format)
        && compatible(plan.format, view, false)
        && plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
        && !plan.usage.contains(ImageUsageFlags::TRANSIENT_ATTACHMENT)
        && !views_in(plan, view)
        && renders_unsigned(device, view, samples)?;
    if !candidate {
        return Ok(None);
    }
    let mut candidate = plan.clone();
    candidate.flags |= ImageCreateFlags::MUTABLE_FORMAT;
    if candidate.view_formats.is_empty() {
        candidate.view_formats.push(plan.format);
    }
    candidate.view_formats.push(view);
    Ok(Some(candidate))
}

/// Whether `device` makes an optimal 2D image of `format`, an unsigned-integer format, as a color
/// attachment of `samples` samples (its image format properties): what a render pass through a
/// view in that format at that count needs. The framebuffer's integer sample counts are only
/// the counts every such format has (RADV lists 1 alone there and makes these images at 4).
fn renders_unsigned(device: &Device, format: Format, samples: u32) -> Result<bool, Error> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format.into())
        .extent(vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::from_raw(samples))
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);
    let limits = device.image_limits(&info, &[])?;
    Ok(limits
        .sample_counts
        .contains(vk::SampleCountFlags::from_raw(samples)))
}

/// The word Metal stores for a NaN clear depth of a 32-bit float depth (the module docs).
const NAN_DEPTH: u32 = 0x7fc0_0000;

/// How a depth attachment whose `Clear` load holds a NaN is cleared (the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum NanDepth {
    /// Metal's `0x7fc00000`, written into its texture before the pass ([`NanDepthWrite`]), which
    /// loads it.
    Written,
    /// The pass clears it to this value: Metal's 0 for a 16-bit depth, or the announced 1.0.
    Cleared(f64),
}

/// How the depth attachment `target`, stored with `store` and memoryless where `memoryless`,
/// clears a NaN its `Clear` load holds on `device` (the module docs), the near-equivalent
/// announced as it is decided. A format other than a 32-bit float or a 16-bit depth is refused
/// by the caller.
pub(super) fn nan_depth(
    device: &Device,
    target: &Target,
    store: StoreAction,
    memoryless: bool,
) -> NanDepth {
    let pixel_format = target.texture.pixel_format();
    if pixel_format == PixelFormat::Depth16Unorm {
        return NanDepth::Cleared(0.0);
    }
    let profile = device.compile_device().profile();
    let clamps = device.shared().textures.facts(profile).depth_clamps();
    let stored = matches!(
        store,
        StoreAction::Store
            | StoreAction::StoreAndMultisampleResolve
            | StoreAction::CustomSampleDepthStore
    );
    let why = if memoryless {
        Some("its memoryless texture keeps no texels to write it into")
    } else if clamps {
        None
    } else if !stored && DontCare::of(device) == DontCare::Discarded {
        Some(
            "the pass does not store the depth, and the device keeps the texture's earlier \
             texels there, as Metal does, which writing it would replace",
        )
    } else if nan_write(device, target).is_none() {
        Some("the device writes no such word into a depth of its samples")
    } else {
        return NanDepth::Written;
    };
    let format = pixel_format;
    match why {
        None => device.degrade(Degradation::RestrictedDepthUpload, || {
            (
                format!("a {format:?} texture"),
                "a render pass's NaN clear depth, which Metal stores as 0x7fc00000, is cleared to \
                 1.0 on a device without VK_EXT_depth_range_unrestricted"
                    .to_owned(),
            )
        }),
        Some(why) => device.degrade(Degradation::SignedDepthTest, || {
            (
                format!("a {format:?} texture"),
                format!(
                    "a render pass's NaN clear depth, which Metal stores as 0x7fc00000 and its \
                     depth test orders above every value, is cleared to 1.0, which a fragment at \
                     exactly 1.0 tests otherwise: {why}"
                ),
            )
        }),
    }
    NanDepth::Cleared(1.0)
}

/// How a NaN clear depth's word is written into `target`'s depth on `device` (the module
/// docs), `None` where it is not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanWrite {
    /// A buffer filled with it, copied in: a single-sample depth with transfer writes.
    Buffer,
    /// An `R32_UINT` image cleared to it, copied in by maintenance8's depth/colour copy.
    Colour,
    /// That image, drawn in by the `FragDepth` draw, where it stores every value unchanged.
    Draw,
}

/// [`NanWrite`] for `target` on `device`.
fn nan_write(device: &Device, target: &Target) -> Option<NanWrite> {
    let usage = target.texture.image_plan().ok()?.usage;
    if !usage.contains(ImageUsageFlags::TRANSFER_DST) {
        return None;
    }
    if target.samples <= 1 {
        return Some(NanWrite::Buffer);
    }
    let feature = Feature::SeparateMultisampledDepthStencil;
    if colour_copies_refused(device, feature, ImageAspectFlags::DEPTH).is_none() {
        return Some(NanWrite::Colour);
    }
    let shading = device
        .enabled()
        .features
        .contains(&VkFeature::SampleRateShading);
    (shading
        && usage.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        && depth_copy::drawn(device, target.format) == Some(Drawn::Exact))
    .then_some(NanWrite::Draw)
}

/// The write of a NaN clear depth's word into a depth texture before its pass (the module docs).
pub(super) struct NanDepthWrite {
    steps: Steps,
    uses: Vec<Use<ResourceKey>>,
}

/// A [`NanDepthWrite`]'s steps.
enum Steps {
    /// A buffer filled with the word, copied into the depth.
    Buffer {
        depth: vk::Image,
        buffer: vk::Buffer,
        offset: u64,
        size: u64,
        region: vk::BufferImageCopy,
        _held: Buffer,
    },
    /// An `R32_UINT` image cleared to the word, copied into the depth (`copy`) or drawn in by
    /// the draws of each layer.
    Bits {
        depth: vk::Image,
        bits: vk::Image,
        layers: u32,
        copy: Option<vk::ImageCopy>,
        draws: Vec<DepthDraw>,
        _held: Texture,
    },
}

impl NanDepthWrite {
    /// The write of the word into `own`, a depth texture's target, over `extent` from the origin
    /// in its layers, made with `maker`. Refused (internal) where the device makes none (the
    /// pass was not to load it), and as the buffer, the image and the draws are.
    pub(super) fn plan(
        maker: &CommandBuffer,
        own: &Target,
        extent: [u32; 2],
    ) -> Result<NanDepthWrite, Error> {
        let device = maker.device();
        let route = nan_write(device, own)
            .ok_or_else(|| internal("a NaN clear depth's write the device does not make"))?;
        let [width, height] = extent;
        let layers = own.layers.max(1);
        let depth = own.texture.raw_image()?;
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let mut uses = vec![own.texture.image_use(
            own.level..own.level + 1,
            own.layer..own.layer + layers,
            write,
            true,
        )?];
        let subresource =
            |aspect: ImageAspectFlags, (level, layer): (u32, u32)| vk::ImageSubresourceLayers {
                aspect_mask: aspect.into(),
                mip_level: level,
                base_array_layer: layer,
                layer_count: layers,
            };
        let size = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let steps = match route {
            NanWrite::Buffer => {
                let bytes = u64::from(width) * u64::from(height) * u64::from(layers) * 4;
                if bytes == 0 {
                    return Err(internal("a NaN clear depth of no texels"));
                }
                let held = device.scratch_buffer(bytes)?;
                let (buffer, offset) = held.at(0);
                let region = vk::BufferImageCopy {
                    buffer_offset: offset,
                    buffer_row_length: width,
                    buffer_image_height: height,
                    image_subresource: subresource(ImageAspectFlags::DEPTH, (own.level, own.layer)),
                    image_offset: vk::Offset3D::default(),
                    image_extent: size,
                };
                Steps::Buffer {
                    depth,
                    buffer,
                    offset,
                    size: bytes,
                    region,
                    _held: held,
                }
            }
            NanWrite::Colour | NanWrite::Draw => {
                let format = (Format::R32_UINT, PixelFormat::R32Uint);
                let image = bits_image(maker, format, own.samples, extent, layers)?;
                uses.push(image.image_use(0..1, 0..layers, write, true)?);
                let bits = image.raw_image()?;
                let (copy, draws) = match route {
                    NanWrite::Colour => {
                        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
                        uses.push(image.image_use(0..1, 0..layers, read, false)?);
                        let copy = vk::ImageCopy {
                            src_subresource: subresource(ImageAspectFlags::COLOR, (0, 0)),
                            src_offset: vk::Offset3D::default(),
                            dst_subresource: subresource(
                                ImageAspectFlags::DEPTH,
                                (own.level, own.layer),
                            ),
                            dst_offset: vk::Offset3D::default(),
                            extent: size,
                        };
                        (Some(copy), Vec::new())
                    }
                    _ => {
                        let area = DrawArea::whole(extent);
                        let feature = Feature::SampleRateShading;
                        let mut draws = Vec::new();
                        for i in 0..layers {
                            // The cleared bits read as the `float` they are.
                            let source = Target {
                                texture: image.clone(),
                                level: 0,
                                layer: i,
                                layers: 1,
                                format: Format::R32_SFLOAT,
                                samples: own.samples,
                                extent,
                            };
                            let into = Target {
                                layer: own.layer + i,
                                layers: 1,
                                ..own.clone()
                            };
                            uses.push(stencil_copy::sampled_use(&source)?);
                            uses.push(stencil_copy::attachment_use(&into)?);
                            draws.push(DepthDraw::from_colour(
                                device, &source, &into, area, &feature,
                            )?);
                        }
                        (None, draws)
                    }
                };
                Steps::Bits {
                    depth,
                    bits,
                    layers,
                    copy,
                    draws,
                    _held: image,
                }
            }
        };
        Ok(NanDepthWrite { steps, uses })
    }

    /// Record the write in `core`'s command buffer, between barriers from everything before and
    /// to everything after.
    pub(super) fn record(&self, core: &mut EncoderCore) {
        let stages = match &self.steps {
            Steps::Buffer { .. } => TRANSFER,
            Steps::Bits { .. } => TRANSFER | stencil_copy::DRAW_STAGES,
        };
        core.barrier(Barrier::everything());
        core.command(stages, &self.uses, |device, raw| {
            let general = vk::ImageLayout::GENERAL;
            match &self.steps {
                Steps::Buffer {
                    depth,
                    buffer,
                    offset,
                    size,
                    region,
                    ..
                } => {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // the buffer is the write's own, of `size` bytes from `offset`, a multiple of
                    // 4 at a multiple of 4; the depth rests in `GENERAL` with transfer writes
                    // (`nan_write`), the region lies in it (the clear's reach of its level and
                    // layers) and in the buffer; the encoder retains the write and the texture
                    // until the command buffer completes.
                    unsafe {
                        device.cmd_fill_buffer(raw, *buffer, *offset, *size, NAN_DEPTH);
                        Barrier::everything().record(device, raw);
                        device.cmd_copy_buffer_to_image(raw, *buffer, *depth, general, &[*region]);
                    }
                }
                Steps::Bits {
                    depth,
                    bits,
                    layers,
                    copy,
                    draws,
                    ..
                } => {
                    let value = vk::ClearColorValue {
                        uint32: [NAN_DEPTH, 0, 0, 0],
                    };
                    let range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: *layers,
                    };
                    // SAFETY: as above; the image is the write's own `R32_UINT` image of the
                    // depth's samples, in `GENERAL` with transfer writes, cleared whole; the copy
                    // is between it and the depth's aspect, which maintenance8 lets
                    // `vkCmdCopyImage` copy (`nan_write`); the draws read it and write the
                    // depth, one layer each.
                    unsafe {
                        device.cmd_clear_color_image(raw, *bits, general, &value, &[range]);
                        if let Some(copy) = copy {
                            Barrier::everything().record(device, raw);
                            device.cmd_copy_image(raw, *bits, general, *depth, general, &[*copy]);
                        }
                        for draw in draws {
                            Barrier::everything().record(device, raw);
                            draw.record(device, raw);
                        }
                    }
                }
            }
        });
        core.barrier(Barrier::everything());
    }
}
