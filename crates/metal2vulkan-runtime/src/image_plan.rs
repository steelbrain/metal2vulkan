//! A Metal texture's Vulkan image: the create info for a checked descriptor on one device, from
//! `metal::format`'s representations and the device's format facts.
//!
//! - **The format.** The image holds the representation a sampled view of the pixel format reads
//!   (`metal::format::vk_format` for `SampledImage`, as binding plans view it), or the direct
//!   format where a storage view, an attachment, or no sampled usage needs it. Every view the
//!   runtime makes of the texture (plans' sampled and storage views, its sRGB twin's, and Metal's
//!   texture views) uses a format of the image's compatibility class.
//! - **Usage.** Each Metal usage maps to the Vulkan usage it needs where the device has it for the
//!   format: `ShaderRead` to sampled, `ShaderWrite` and `ShaderAtomic` to storage (through
//!   `storage_image_choice`, so a storage alias gets its mutable format and, where the image's
//!   own format lacks storage, extended usage), `RenderTarget` to the color or depth/stencil
//!   attachment, a color one also to an input attachment where the device's framebuffer fetch
//!   reads input attachments (`framebuffer_fetch.rs`). A usage the device lacks for the format is
//!   left off: creating the texture succeeds, and a command that needs it is refused when
//!   recorded (a plan's refusal names the capability). A device that holds the format for none
//!   of the texture's uses, nor for copies, lacks the format: the texture is refused, naming it.
//!   Metal's `Unknown` usage is the usage Metal gives it
//!   (`texture_descriptor::effective_usage`). Transfer usage is always asked for where the format
//!   has it, because Metal copies every texture with blits, `replaceRegion` and `getBytes`. A
//!   single-sample depth attachment the runtime makes also takes sampled usage where its driver
//!   keeps a depth image's hierarchical depth in `GENERAL` only for a sampled image
//!   ([`hiz_candidate`], `DriverFact::SampledDepthKeepsHiZ`: ANV on the Iris Xe, measured).
//! - **`ulong` atomics.** A `RG32Uint` texture (Metal's `ulong` texture) made with `ShaderAtomic`
//!   also takes the `R64_UINT` view its `ulong` atomics update, where the device has that format's
//!   storage atomics.
//! - **Mutable format.** A texture whose views may use another format is created mutable:
//!   `PixelFormatView` usage without a format list, and otherwise exactly the
//!   formats its own views and its sRGB twin's need (`VkImageFormatListCreateInfo`), so drivers
//!   that compress only listed formats keep doing so. Metal views between a format and its sRGB
//!   twin without `PixelFormatView` (measured on an Apple M3 Max and M4).
//! - **Cubes.** Cube and cube-array textures are cube-compatible, and so is a square 2D array of
//!   at least six single-sample layers, which Metal views as cubes (Vulkan has no cube view of an
//!   image that is not square).
//! - **Memoryless.** A memoryless texture is a transient attachment (a color one an input
//!   attachment too, as above) and nothing else, of any texture type (a 3D one 2D-array
//!   compatible, as a stored 3D render target).
//! - **Sample positions.** A multisampled depth or depth/stencil render target is made
//!   `SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT` where the device takes custom sample positions at its
//!   count, so a pass with sample positions may attach it (`sample_positions::compatible_depth`).
//! - **sRGB twins**. A device that samples an 8-bit sRGB format's UNORM twin but not the format
//!   (lavapipe's `R8_SRGB`, ANV's and NVIDIA's `R8G8_SRGB`) samples its texture through the twin's
//!   view: the image holds the twin's format, the same bytes, and the shader reads the raw codes
//!   and decodes them as Metal does (the variant; [`holds_srgb_twin`]).
//! - **Exactness.** A texture is held only in an exact representation: 24-bit depth held as
//!   32-bit float depth (`FormatEmulation::DepthWidened`) is refused, naming the format it needs.
//!
//! The device's image limits for the result (`vkGetPhysicalDeviceImageFormatProperties2`) are
//! checked by [`check_image_limits`].

use super::chroma_pairs::plan_pairs;
use super::framebuffer_fetch;
use super::memory::MemoryClass;
use super::pvrtc::plan_blocks;
use super::sample_positions::compatible_depth;
use super::sharing::Sharing;
use super::texture::heap_tiled::metal_keeps_uncompressed;
use super::texture_descriptor::{effective_usage, TextureLayout};
use super::values::StorageMode;
use ash::vk;
use metal2vulkan::device::{
    DeviceProfile, DriverFact, FormatSupport, FormatUsage, VkExtension, VkFeature, VkLimit,
    VkProperty,
};
use metal2vulkan::error::{
    Capability, Error as CompileError, Feature, Needs, RuntimeDriverDefect, Unsupported,
};
use metal2vulkan::metal::format::{
    alpha_in_red, alpha_in_red_format, direct_format, image_type, is_depth_stencil, srgb_raw_twin,
    storage_image_choice, storage_view_formats, vk_format, FormatEmulation, FormatMapping,
    WordAttachment,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    ColorComponentFlags, Format, FormatBlock, FormatFeatureFlags2, ImageCreateFlags, ImageType,
    ImageUsageFlags, NumericType, SampleCountFlags,
};
use std::collections::BTreeSet;

/// The device facts an image plan reads beyond the profile's format tables: the raw format
/// properties (for transfer and blit bits), whether storage images may be multisampled, and what
/// the driver is measured to keep in a depth image.
#[derive(Clone, Copy, Debug)]
pub(super) struct ImageFacts<'a> {
    pub(super) profile: &'a DeviceProfile,
    pub(super) formats: &'a FormatSupport,
    /// `shaderStorageImageMultisample` is enabled.
    pub(super) storage_multisample: bool,
    /// What the driver is measured to keep in a `D32_SFLOAT` image.
    pub(super) depth_images: DepthImageFacts,
    /// The device enabled `VK_NV_linear_color_attachment`'s `linearColorAttachment`: a linear
    /// image of a format whose linear tiling has `LINEAR_COLOR_ATTACHMENT_NV` renders, where
    /// every color attachment of the render pass instance is linear (`linear_attachment.rs`).
    pub(super) linear_attachments: bool,
}

/// What a device's driver is measured to keep in a `D32_SFLOAT` image, of the `float`s Metal's
/// `Depth32Float` texel holds (`device/driver_facts.rs`): an unmeasured driver keeps neither, and
/// a write that may store such a value announces it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct DepthImageFacts {
    /// `DriverFact::DepthImageKeepsDenormals`.
    pub(super) denormals: bool,
    /// `DriverFact::DepthImageKeepsValuesOutsideUnitRange`: without
    /// `VK_EXT_depth_range_unrestricted`.
    pub(super) outside_unit_range: bool,
    /// `DriverFact::DepthTestKeepsDenormals`.
    pub(super) test_denormals: bool,
    /// `DriverFact::DepthTestComparesMagnitudes`: the depth test is Metal's for every value a
    /// `D32_SFLOAT` image holds (`Texture::degrade_signed_depth`).
    pub(super) test_magnitudes: bool,
}

impl DepthImageFacts {
    /// The depth-image facts among a driver's `facts`.
    pub(super) fn of(facts: &BTreeSet<DriverFact>) -> DepthImageFacts {
        DepthImageFacts {
            denormals: facts.contains(&DriverFact::DepthImageKeepsDenormals),
            outside_unit_range: facts.contains(&DriverFact::DepthImageKeepsValuesOutsideUnitRange),
            test_denormals: facts.contains(&DriverFact::DepthTestKeepsDenormals),
            test_magnitudes: facts.contains(&DriverFact::DepthTestComparesMagnitudes),
        }
    }
}

impl ImageFacts<'_> {
    /// Why a denormal a write stores in a `D32_SFLOAT` image (`on_host`: a host copy, which keeps
    /// it on every driver measured) may answer otherwise than Metal's: the driver stores one a
    /// write on the GPU gives it as a zero (no `DriverFact::DepthImageKeepsDenormals`), or its
    /// depth test compares one otherwise (no `DriverFact::DepthTestKeepsDenormals`); `None` where
    /// neither.
    pub(super) fn denormal_loss(&self, on_host: bool) -> Option<&'static str> {
        let stored = on_host || self.depth_images.denormals;
        match (stored, self.depth_images.test_denormals) {
            (true, true) => None,
            (false, true) => Some("the device's driver stores it in a depth image as a zero"),
            (true, false) => Some(
                "the device's depth test compares a stored denormal otherwise than Metal's (a \
                 positive one as zero, or a negative one below zero)",
            ),
            (false, false) => Some(
                "the device's driver stores it in a depth image as a zero, and its depth test \
                 compares a stored denormal otherwise than Metal's",
            ),
        }
    }

    /// Whether a `D32_SFLOAT` image may store a copied value outside `[0, 1]` clamped: the
    /// device lacks `VK_EXT_depth_range_unrestricted`, where Vulkan leaves such a value undefined
    /// (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), and its driver is not measured to keep one
    /// (`DriverFact::DepthImageKeepsValuesOutsideUnitRange`).
    pub(super) fn depth_clamps(&self) -> bool {
        let unrestricted = self
            .profile
            .enabled()
            .extensions
            .contains(&VkExtension::ExtDepthRangeUnrestricted);
        !unrestricted && !self.depth_images.outside_unit_range
    }

    /// The optimal-tiling features of `format`.
    pub(super) fn optimal(&self, format: Format) -> FormatFeatureFlags2 {
        self.formats
            .formats
            .get(&format)
            .map(|p| p.optimal_tiling)
            .unwrap_or_default()
    }

    /// The linear-tiling features of `format`, as the device enabled them: with the color
    /// attachment feature where it enabled `linearColorAttachment` and the tiling has
    /// `LINEAR_COLOR_ATTACHMENT_NV`, under the extension's rule, which the render passes keep
    /// ([`linear_attachment_rule`](Self::linear_attachment_rule)).
    pub(super) fn linear(&self, format: Format) -> FormatFeatureFlags2 {
        let reported = self.reported_linear(format);
        match self.linear_attachment_rule(format) {
            true => reported | FormatFeatureFlags2::COLOR_ATTACHMENT,
            false => reported,
        }
    }

    /// Whether a linear image renders as a color attachment viewed in `format` only under
    /// `VK_NV_linear_color_attachment`'s rule, that every color attachment of the render pass
    /// instance be linear: the device enabled `linearColorAttachment`, and `format`'s linear
    /// tiling has `LINEAR_COLOR_ATTACHMENT_NV` and not the color attachment feature, under which
    /// a linear image renders beside any other (`linear_attachment.rs`).
    pub(super) fn linear_attachment_rule(&self, format: Format) -> bool {
        let reported = self.reported_linear(format);
        self.linear_attachments
            && reported.contains(FormatFeatureFlags2::LINEAR_COLOR_ATTACHMENT_NV)
            && !reported.contains(FormatFeatureFlags2::COLOR_ATTACHMENT)
    }

    /// The linear-tiling features the device reports of `format`.
    fn reported_linear(&self, format: Format) -> FormatFeatureFlags2 {
        self.formats
            .formats
            .get(&format)
            .map(|p| p.linear_tiling)
            .unwrap_or_default()
    }
}

/// Why an image cannot be planned: the pixel format is no format, or the device lacks what the
/// texture needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ImageRefusal {
    /// `Invalid` or `Unspecialized`: Metal's validation stops the descriptor.
    NotAFormat,
    /// A texture buffer, which is a texel view of a buffer and has no image.
    TextureBuffer,
    /// The device cannot hold the texture.
    Unsupported(Unsupported),
}

impl ImageRefusal {
    /// The refusal, naming `pf` where it refuses the texture's format (`Feature::Format`) and
    /// names nothing yet (`Unsupported::naming_format`, diagnostics only).
    fn naming(self, pf: PixelFormat) -> ImageRefusal {
        match self {
            ImageRefusal::Unsupported(u) if u.feature == Feature::Format && u.symbol.is_none() => {
                ImageRefusal::Unsupported(u.naming_format(pf))
            }
            refusal => refusal,
        }
    }

    fn of(error: CompileError) -> ImageRefusal {
        match error {
            CompileError::Unsupported(u) => ImageRefusal::Unsupported(u),
            // `vk_format` and `storage_image_choice` refuse a placeholder format as an invalid
            // descriptor, and nothing else.
            _ => ImageRefusal::NotAFormat,
        }
    }
}

/// A texture's Vulkan image on one device.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ImagePlan {
    /// The texture's pixel format.
    pub(super) pixel_format: PixelFormat,
    pub(super) flags: ImageCreateFlags,
    pub(super) image_type: ImageType,
    pub(super) format: Format,
    pub(super) extent: [u32; 3],
    pub(super) levels: u32,
    pub(super) layers: u32,
    pub(super) samples: SampleCountFlags,
    pub(super) usage: ImageUsageFlags,
    /// The formats a mutable image is viewed as (`VkImageFormatListCreateInfo`), the image's own
    /// first; empty for an image that is not mutable or may take any compatible format.
    pub(super) view_formats: Vec<Format>,
    /// How the image presents the pixel format: its format, the component mapping that gives
    /// Metal's channels, and its aspects.
    pub(super) mapping: FormatMapping,
    /// The Metal usages the device lacks for this texture: a command that needs one is refused.
    pub(super) unavailable: TextureUsage,
}

impl ImagePlan {
    /// The `VkImageCreateInfo` of this plan, with the format list chained through `list` (caller
    /// storage) when it has one. The image is optimal-tiled, shared among the device's queue
    /// families as `sharing` says (`sharing.rs`), and starts `UNDEFINED`.
    pub(super) fn create_info<'a, 'b>(
        &self,
        sharing: Sharing<'a>,
        formats: &'b [vk::Format],
        list: &'a mut vk::ImageFormatListCreateInfo<'b>,
    ) -> vk::ImageCreateInfo<'a> {
        let [width, height, depth] = self.extent;
        let info = vk::ImageCreateInfo::default()
            .flags(self.flags.into())
            .image_type(self.image_type.into())
            .format(self.format.into())
            .extent(vk::Extent3D {
                width,
                height,
                depth,
            })
            .mip_levels(self.levels)
            .array_layers(self.layers)
            .samples(self.samples.into())
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(self.usage.into())
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let mut info = sharing.image(info);
        if !formats.is_empty() {
            *list = vk::ImageFormatListCreateInfo::default().view_formats(formats);
            info = info.push_next(list);
        }
        info
    }

    /// The format list as `ash` values, for [`create_info`](Self::create_info).
    pub(super) fn ash_view_formats(&self) -> Vec<vk::Format> {
        self.view_formats.iter().map(|&f| f.into()).collect()
    }

    /// The memory class of this plan's image for a texture of `storage` on a device whose
    /// driver has `defects`: a memoryless texture's is lazily allocated memory where the device
    /// has it ([`MemoryClass::Memoryless`]), every other texture's device-local memory. Where the
    /// driver crashes on a 2D view of a 3D image in lazily allocated memory
    /// (`RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews`: MoltenVK), a memoryless image
    /// made 2D-array compatible for its depth planes' views takes device-local memory too, and
    /// where a view of some of its levels or layers has no texture
    /// (`RuntimeDriverDefect::LazilyAllocatedSubresourceViews`: MoltenVK), one of more than one
    /// level or array layer. A pass neither loads nor stores a memoryless attachment, so its
    /// texels are no more observable there; a device without lazily allocated memory holds every
    /// memoryless texture so.
    pub(super) fn memory_class(
        &self,
        storage: StorageMode,
        defects: &BTreeSet<RuntimeDriverDefect>,
    ) -> MemoryClass {
        let plane_views = self
            .flags
            .contains(ImageCreateFlags::CREATE_2D_ARRAY_COMPATIBLE)
            && defects.contains(&RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews);
        let part_views = (self.levels > 1 || self.layers > 1)
            && defects.contains(&RuntimeDriverDefect::LazilyAllocatedSubresourceViews);
        if storage == StorageMode::Memoryless && !plane_views && !part_views {
            MemoryClass::Memoryless
        } else {
            MemoryClass::Private
        }
    }
}

/// The Vulkan sample count flag of `samples`, or `None` when Vulkan has none (not a power of two
/// up to 64).
pub(super) fn sample_count(samples: u32) -> Option<SampleCountFlags> {
    (samples.is_power_of_two() && samples <= 64).then(|| SampleCountFlags::from_raw(samples))
}

/// Metal's sRGB pairs: the linear format, then the format whose texels are the same bytes read
/// through sRGB encoding.
const SRGB_PAIRS: [(PixelFormat, PixelFormat); 31] = [
    (PixelFormat::R8Unorm, PixelFormat::R8UnormSRGB),
    (PixelFormat::RG8Unorm, PixelFormat::RG8UnormSRGB),
    (PixelFormat::RGBA8Unorm, PixelFormat::RGBA8UnormSRGB),
    (PixelFormat::BGRA8Unorm, PixelFormat::BGRA8UnormSRGB),
    (PixelFormat::BGR10XR, PixelFormat::BGR10XRSRGB),
    (PixelFormat::BGRA10XR, PixelFormat::BGRA10XRSRGB),
    (PixelFormat::BC1RGBA, PixelFormat::BC1RGBASRGB),
    (PixelFormat::BC2RGBA, PixelFormat::BC2RGBASRGB),
    (PixelFormat::BC3RGBA, PixelFormat::BC3RGBASRGB),
    (PixelFormat::BC7RGBAUnorm, PixelFormat::BC7RGBAUnormSRGB),
    (PixelFormat::PVRTCRGB2BPP, PixelFormat::PVRTCRGB2BPPSRGB),
    (PixelFormat::PVRTCRGB4BPP, PixelFormat::PVRTCRGB4BPPSRGB),
    (PixelFormat::PVRTCRGBA2BPP, PixelFormat::PVRTCRGBA2BPPSRGB),
    (PixelFormat::PVRTCRGBA4BPP, PixelFormat::PVRTCRGBA4BPPSRGB),
    (PixelFormat::EACRGBA8, PixelFormat::EACRGBA8SRGB),
    (PixelFormat::ETC2RGB8, PixelFormat::ETC2RGB8SRGB),
    (PixelFormat::ETC2RGB8A1, PixelFormat::ETC2RGB8A1SRGB),
    (PixelFormat::ASTC4x4LDR, PixelFormat::ASTC4x4SRGB),
    (PixelFormat::ASTC5x4LDR, PixelFormat::ASTC5x4SRGB),
    (PixelFormat::ASTC5x5LDR, PixelFormat::ASTC5x5SRGB),
    (PixelFormat::ASTC6x5LDR, PixelFormat::ASTC6x5SRGB),
    (PixelFormat::ASTC6x6LDR, PixelFormat::ASTC6x6SRGB),
    (PixelFormat::ASTC8x5LDR, PixelFormat::ASTC8x5SRGB),
    (PixelFormat::ASTC8x6LDR, PixelFormat::ASTC8x6SRGB),
    (PixelFormat::ASTC8x8LDR, PixelFormat::ASTC8x8SRGB),
    (PixelFormat::ASTC10x5LDR, PixelFormat::ASTC10x5SRGB),
    (PixelFormat::ASTC10x6LDR, PixelFormat::ASTC10x6SRGB),
    (PixelFormat::ASTC10x8LDR, PixelFormat::ASTC10x8SRGB),
    (PixelFormat::ASTC10x10LDR, PixelFormat::ASTC10x10SRGB),
    (PixelFormat::ASTC12x10LDR, PixelFormat::ASTC12x10SRGB),
    (PixelFormat::ASTC12x12LDR, PixelFormat::ASTC12x12SRGB),
];

/// The other member of `pf`'s sRGB pair (`RGBA8Unorm` and `RGBA8Unorm_sRGB`): Metal views a
/// texture as its twin without `PixelFormatView` usage.
pub(super) fn srgb_twin(pf: PixelFormat) -> Option<PixelFormat> {
    SRGB_PAIRS.iter().find_map(|&(linear, srgb)| {
        if pf == linear {
            Some(srgb)
        } else if pf == srgb {
            Some(linear)
        } else {
            None
        }
    })
}

/// The member of `pf`'s compressed family that names it: block-compressed formats whose texels
/// are the same blocks read as sRGB or linear, signed or unsigned, or HDR or LDR. Metal views a
/// texture as another member of its family with `PixelFormatView` usage (`BC4_RUnorm` as
/// `BC4_RSnorm`, measured on an Apple M3 Max and M4), and no other compressed format; these are
/// Vulkan's compatibility classes of the same formats. Any other format names its own.
pub(super) fn block_family(pf: PixelFormat) -> PixelFormat {
    use PixelFormat as P;
    let linear = SRGB_PAIRS
        .iter()
        .find(|&&(_, srgb)| srgb == pf)
        .map_or(pf, |&(linear, _)| linear);
    match linear {
        P::BC4RSnorm => P::BC4RUnorm,
        P::BC5RGSnorm => P::BC5RGUnorm,
        P::BC6HRGBFloat => P::BC6HRGBUfloat,
        P::EACR11Snorm => P::EACR11Unorm,
        P::EACRG11Snorm => P::EACRG11Unorm,
        P::ASTC4x4HDR => P::ASTC4x4LDR,
        P::ASTC5x4HDR => P::ASTC5x4LDR,
        P::ASTC5x5HDR => P::ASTC5x5LDR,
        P::ASTC6x5HDR => P::ASTC6x5LDR,
        P::ASTC6x6HDR => P::ASTC6x6LDR,
        P::ASTC8x5HDR => P::ASTC8x5LDR,
        P::ASTC8x6HDR => P::ASTC8x6LDR,
        P::ASTC8x8HDR => P::ASTC8x8LDR,
        P::ASTC10x5HDR => P::ASTC10x5LDR,
        P::ASTC10x6HDR => P::ASTC10x6LDR,
        P::ASTC10x8HDR => P::ASTC10x8LDR,
        P::ASTC10x10HDR => P::ASTC10x10LDR,
        P::ASTC12x10HDR => P::ASTC12x10LDR,
        P::ASTC12x12HDR => P::ASTC12x12LDR,
        other => other,
    }
}

/// Whether `plan` holds an 8-bit sRGB texture in the format of its UNORM twin
/// (`metal::format::srgb_raw_twin`): the device samples the twin but not the sRGB format, so the
/// texture's sampled views are the twin's, through which the shader reads the raw codes and decodes
/// them by the MSL transfer through the device's `Pow`; no sRGB view decodes them.
pub(super) fn holds_srgb_twin(plan: &ImagePlan) -> bool {
    srgb_raw_twin(plan.pixel_format).and_then(direct_format) == Some(plan.format)
}

/// Whether views of an image of `image` may use `view` (Vulkan's format compatibility for a
/// mutable-format image): the same format, two uncompressed color formats of one compatibility
/// class, or two block-compressed formats of one class whose Metal pixel formats are of one
/// compressed family (`family`, [`block_family`]). Depth and stencil formats are compatible only
/// with themselves. The class is the registry's (`FormatBlock::class`), not the texel size:
/// `A8_UNORM` ("8-bit alpha") and the 4-pack 10- and 12-bit formats share a size but no class
/// with any other format (the scout's VUID-VkImageCreateInfo-pNext-06722).
pub(super) fn compatible(image: Format, view: Format, family: bool) -> bool {
    if image == view {
        return true;
    }
    let (Some(a), Some(b)) = (image.block(), view.block()) else {
        return false;
    };
    let color = |f: &FormatBlock| f.color.is_some() && f.depth.is_none() && f.stencil.is_none();
    if !color(&a) || !color(&b) || a.class != b.class {
        return false;
    }
    !is_compressed(&a) || family
}

/// Whether Vulkan copies texels between images of color formats `a` and `b` of one texel block
/// size (`vkCmdCopyImage`, `VUID-vkCmdCopyImage-srcImage-01548`): "Color formats with the same
/// texel block size are considered size-compatible as long as neither or both are alpha formats
/// (e.g., VK_FORMAT_A8_UNORM)" (the Formats chapter's "Size Compatibility"). An alpha format's
/// one component is alpha. Elsewhere the copy goes through a buffer, whose copies take each
/// image's texel block size alone.
pub(super) fn copies_between(a: Format, b: Format) -> bool {
    let alpha = |format: Format| {
        format
            .block()
            .is_some_and(|block| block.color_components == ColorComponentFlags::A)
    };
    alpha(a) == alpha(b)
}

/// Whether a color format's block is a compressed block: more than one texel, or (Vulkan's
/// registry counts a PVRTC block as one texel) components that share no width and pack into no
/// word.
pub(super) fn is_compressed(block: &FormatBlock) -> bool {
    block.color.is_some()
        && (block.texels > 1 || (block.packed_bits.is_none() && block.component_bits.is_none()))
}

/// The image of a texture of `layout` with Metal `usage` on a device with `facts`; `memoryless`
/// for `MTLStorageModeMemoryless`. `layout` is not a texture buffer, which has no image.
pub(super) fn plan_image(
    layout: &TextureLayout,
    usage: TextureUsage,
    memoryless: bool,
    facts: &ImageFacts<'_>,
) -> Result<ImagePlan, ImageRefusal> {
    plan_held(layout, usage, memoryless, facts).map_err(|r| r.naming(layout.pixel_format))
}

/// The flags a 3D image rendered into needs: 2D-array compatible, for the 2D view of its depth
/// planes, but on a device whose portability subset withholds `imageView2DOn3DImage`
/// (`VUID-VkImageCreateInfo-imageView2DOn3DImage-04459`), whose pass refuses the attachment
/// (`render_pass.rs`).
fn render_3d_flags(profile: &DeviceProfile) -> ImageCreateFlags {
    if profile
        .enabled()
        .portability_withholds(VkFeature::ImageView2DOn3DImage)
    {
        ImageCreateFlags::empty()
    } else {
        ImageCreateFlags::CREATE_2D_ARRAY_COMPATIBLE
    }
}

/// [`plan_image`]'s plan.
fn plan_held(
    layout: &TextureLayout,
    usage: TextureUsage,
    memoryless: bool,
    facts: &ImageFacts<'_>,
) -> Result<ImagePlan, ImageRefusal> {
    let pf = layout.pixel_format;
    let profile = facts.profile;
    let usage = effective_usage(usage);
    let (image_type, create_flags) =
        image_type(layout.texture_type).ok_or(ImageRefusal::TextureBuffer)?;
    let mut flags = create_flags;
    // A checked descriptor of a multisample type has a count of at least 1; Vulkan has only
    // powers of two.
    let samples = sample_count(layout.samples).ok_or_else(|| {
        ImageRefusal::Unsupported(Unsupported::not_representable(Feature::TextureSampleCount))
    })?;
    // Metal makes a multisample array, memoryless ones included (a Metal probe on an M4); a device
    // whose portability subset withholds `multisampleArrayImage` makes no multisampled image of
    // more than one layer's type (`VUID-VkImageCreateInfo-multisampleArrayImage-04460`).
    if layout.texture_type == TextureType::Type2DMultisampleArray
        && profile
            .enabled()
            .portability_withholds(VkFeature::MultisampleArrayImage)
    {
        return Err(ImageRefusal::Unsupported(Unsupported::needs(
            Feature::TextureSampleCount,
            Capability::Feature(VkFeature::MultisampleArrayImage),
        )));
    }
    // A PVRTC texture the device has no format for holds its blocks in a carrier (`pvrtc.rs`).
    if !memoryless {
        if let Some(plan) = plan_blocks(layout, usage, facts) {
            return Ok(plan);
        }
        // An ASTC texture the device has no block for holds its blocks in a carrier, sampled
        // through a decoded twin (`astc.rs`).
        if let Some(plan) = super::astc::plan_blocks(layout, usage, facts) {
            return Ok(plan);
        }
        // So does an ETC2 or EAC texture on a device without the formats (`etc.rs`).
        if let Some(plan) = super::etc::plan_blocks(layout, usage, facts) {
            return Ok(plan);
        }
        // A 4:2:2 texture holds its texels in a carrier (`chroma_pairs.rs`).
        if let Some(plan) = plan_pairs(layout, usage, facts) {
            return Ok(plan);
        }
    }
    // `vk_format` with no usage names the refusal of a format without a representation (not
    // representable, not implemented, or no format at all).
    let any = vk_format(pf, &BTreeSet::new(), profile).map_err(ImageRefusal::of)?;
    // An `A8Unorm` texture on a device that holds no `A8_UNORM` image is held in `R8_UNORM`, its
    // alpha in red, for every use (`metal::format::alpha_in_red`): that is its direct format here.
    let direct = match alpha_in_red_format(pf).filter(|_| alpha_in_red(pf, profile)) {
        Some(red) => red,
        None => direct_format(pf).unwrap_or(any.format),
    };
    let depth_stencil = is_depth_stencil(pf);
    let attachment_usage = if depth_stencil {
        FormatUsage::DepthStencilAttachment
    } else {
        FormatUsage::ColorAttachment
    };
    let attachment_mapping = vk_format(pf, &[attachment_usage].into(), profile);
    // A two-plane YCbCr texture's image holds the texels its planes read, which no pass renders
    // into: its attachment format is its render shadow's or a pipeline's alone
    // (`FormatEmulation::YCbCrRendered`, `YCbCrPassRefused`).
    let attachment = attachment_mapping.as_ref().ok().copied().filter(|m| {
        m.format == direct
            && !matches!(
                m.emulation,
                FormatEmulation::YCbCrRendered | FormatEmulation::YCbCrPassRefused
            )
    });

    if memoryless {
        let Some(mapping) = attachment else {
            // A format the table does not render to at all is refused as the table refuses it
            // (`NotImplemented`, `NotRepresentable`), not as a need of the device: the guest's
            // memoryless `RGB10A8_2P_XR10` texture was refused as needing `R32G32_UINT` as a
            // color attachment, which MoltenVK has.
            if let Err(CompileError::Unsupported(u)) = &attachment_mapping {
                if matches!(u.needs, Needs::NotImplemented | Needs::NotRepresentable) {
                    return Err(ImageRefusal::Unsupported(u.clone()));
                }
            }
            return Err(ImageRefusal::Unsupported(Unsupported::needs(
                Feature::Format,
                Capability::Format {
                    format: direct,
                    usage: attachment_usage,
                },
            )));
        };
        if depth_stencil {
            flags |= compatible_depth(profile, direct, samples);
        }
        // A memoryless 3D texture renders into a depth plane, as a stored one does (below).
        if layout.texture_type == TextureType::Type3D && !depth_stencil {
            flags |= render_3d_flags(profile);
        }
        return Ok(ImagePlan {
            pixel_format: pf,
            flags,
            image_type,
            format: direct,
            extent: layout.extent,
            levels: layout.levels,
            layers: layout.layers,
            samples,
            usage: attachment_flag(depth_stencil, profile) | ImageUsageFlags::TRANSIENT_ATTACHMENT,
            view_formats: Vec::new(),
            mapping,
            // A memoryless texture is a render target only.
            unavailable: usage.difference(TextureUsage::RENDER_TARGET),
        });
    }

    let wants = |bit: TextureUsage| usage.contains(bit);
    // An 8-bit sRGB format the device cannot sample is sampled through its UNORM twin's view,
    // whose codes the shader decodes by the device's transfer ([`holds_srgb_twin`]).
    let sampled = vk_format(pf, &[FormatUsage::SampledImage].into(), profile)
        .ok()
        .or_else(|| {
            srgb_raw_twin(pf)
                .and_then(|twin| vk_format(twin, &[FormatUsage::SampledImage].into(), profile).ok())
        });
    // Apple's private `BlockWritesOnly` (0x10000) is a usage Metal writes through as through
    // `ShaderWrite`: its validation takes every write form into a texture of it, and plain Metal
    // stores them (CoreAnimation's blur textures are
    // `ShaderRead | RenderTarget | BlockWritesOnly`). Its image takes the storage usage where the
    // device stores the format; elsewhere a binding takes the usage shadow
    // (`texture/usage_shadow.rs`).
    let stored = (wants(TextureUsage::SHADER_WRITE)
        || wants(TextureUsage::SHADER_ATOMIC)
        || wants(TextureUsage::BLOCK_WRITES_ONLY))
        && (layout.samples == 1 || facts.storage_multisample);
    let storage = stored
        .then(|| storage_image_choice(pf, profile).ok())
        .flatten();
    // A format no slot both reads and writes on the device, whose written-only binding's view
    // the device stores (on ANV, which reads no storage image without a format, a format SPIR-V
    // names no image format for and no alias serves), still takes that view: a slot that only
    // writes the texture binds, and one that reads it too is refused at its binding.
    let written_only = match (stored, &storage) {
        (true, None) => storage_view_formats(pf, profile),
        _ => Vec::new(),
    };
    let wants_attachment = wants(TextureUsage::RENDER_TARGET) && attachment.is_some();
    // A format the device samples but does not render, whose word it renders, renders through a
    // view of the word (`metal::format::WordAttachment`): its image keeps the format (an 8-bit
    // sRGB format the device holds no image of, its UNORM twin), mutable, with the attachment
    // usage extended to the word's view.
    let word = match (
        wants(TextureUsage::RENDER_TARGET),
        attachment,
        depth_stencil,
    ) {
        (true, None, false) => WordAttachment::of(pf, profile).map(|w| w.word_format()),
        _ => None,
    };
    // An 8-bit sRGB texture whose format the device holds no image of at all (`R8G8_SRGB` on
    // NVIDIA) is held in its UNORM twin (`srgb_raw_twin`), which is its storage alias's own view:
    // the twin serves the storage too. Where the device holds the sRGB format, it stays the
    // image's own, and only the storage view is the alias.
    let direct_held = !facts.optimal(direct).is_empty();
    let unorm_twin = srgb_raw_twin(pf).and_then(direct_format);
    let twin_stored = |m: &FormatMapping| {
        Some(m.format) == unorm_twin
            && !direct_held
            && storage
                .as_ref()
                .is_some_and(|c| c.alias.is_some() && c.vk_format == m.format)
    };

    // The image format: the sampled representation, unless a storage view or an attachment needs
    // the direct format, or nothing samples the texture.
    let mapping = match (sampled, attachment) {
        (Some(m), _)
            if m.format == direct
                || twin_stored(&m)
                || !(storage.is_some() || !written_only.is_empty() || wants_attachment) =>
        {
            m
        }
        (_, Some(m)) => m,
        // The first representation the device exposes (the direct format where it can): a
        // storage image ignores the view's component mapping, so only the direct format serves
        // one.
        _ => any,
    };
    // 24-bit depth held as 32-bit float depth is not exact, and an ASTC, PVRTC, ETC2 or EAC
    // format's decoded twin's view holds no blocks (a memoryless texture, or a device with no
    // carrier, `astc.rs`, `pvrtc.rs`, `etc.rs`): the texture needs its own format.
    if matches!(
        mapping.emulation,
        FormatEmulation::DepthWidened | FormatEmulation::Decoded
    ) {
        return Err(ImageRefusal::Unsupported(Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format: direct,
                usage: FormatUsage::SampledImage,
            },
        )));
    }
    let format = mapping.format;
    let mut unavailable = TextureUsage::empty();
    let mut usage_flags = ImageUsageFlags::empty();
    let mut views: BTreeSet<Format> = BTreeSet::new();

    if wants(TextureUsage::SHADER_READ) {
        match sampled.filter(|m| compatible(format, m.format, false)) {
            Some(m) => {
                usage_flags |= ImageUsageFlags::SAMPLED;
                views.insert(m.format);
            }
            None => unavailable |= TextureUsage::SHADER_READ,
        }
    }
    // A storage alias views the image in the format's own Vulkan format, which the device must
    // hold (roadmap-2022 states no `A2R10G10B10` at all, whose `BGR10A2Unorm` storage its
    // `A2B10G10R10` view would otherwise serve).
    let held = !facts.optimal(format).is_empty();
    let twin = Some(format) == unorm_twin
        && storage
            .as_ref()
            .is_some_and(|c| c.alias.is_some() && c.vk_format == format);
    match storage.filter(|c| twin || (format == direct && (c.alias.is_none() || held))) {
        Some(choice) => {
            usage_flags |= ImageUsageFlags::STORAGE;
            views.insert(match choice.alias {
                Some(_) => choice.vk_format,
                None => format,
            });
            // And a written-only binding's view, where its choice differs.
            views.extend(storage_view_formats(pf, profile));
            // A `ulong` atomic updates Metal's `ulong` texture (`RG32Uint`) through an
            // `R64_UINT` view of the same bytes: a texture made for atomics takes it where the
            // device has the view's storage atomics.
            if wants(TextureUsage::SHADER_ATOMIC)
                && format == Format::R32G32_UINT
                && profile.supports_format(Format::R64_UINT, FormatUsage::StorageImageAtomic)
            {
                views.insert(Format::R64_UINT);
            }
        }
        None if format == direct
            && !written_only.is_empty()
            && (held || written_only.iter().all(|&view| view == direct)) =>
        {
            usage_flags |= ImageUsageFlags::STORAGE;
            views.extend(written_only);
        }
        None => {
            unavailable |=
                usage.intersection(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC)
        }
    }
    if wants(TextureUsage::RENDER_TARGET) {
        if wants_attachment && format == direct {
            usage_flags |= attachment_flag(depth_stencil, profile);
            views.insert(format);
            // A multisampled depth attachment is made for a pass's custom sample positions
            // where the device takes them (`sample_positions::compatible_depth`).
            if depth_stencil {
                flags |= compatible_depth(profile, format, samples);
            }
        } else if let Some(word) = word.filter(|_| format == direct || Some(format) == unorm_twin) {
            usage_flags |= attachment_flag(false, profile);
            views.insert(word);
        } else {
            unavailable |= usage.intersection(TextureUsage::RENDER_TARGET);
        }
    }
    // A color texture Metal keeps uncompressed (`metal_keeps_uncompressed`) whose image takes no
    // storage usage (a multisample one the device stores no samples of, say) is planned as one
    // of `PixelFormatView` usage, a mutable image: a driver that compresses images keeps a
    // mutable one of any view format uncompressed (MoltenVK gives its Metal texture
    // `PixelFormatView` usage, as a storage image's `ShaderWrite`), as Metal keeps the texture.
    // Such an image no shader reads, writes or renders through is sampled too where the format
    // samples: MoltenVK gives a mutable image's Metal texture `PixelFormatView` usage only where
    // the image has a view usage (sampled, storage, input or color attachment; MoltenVK 1.4.1's
    // `MVKPixelFormats::getMTLTextureUsage`); without one it made a compressed texture, larger
    // than Metal's (a `PixelFormatView` texture alone, the consumer's M3 Max).
    let any_format = usage.contains(TextureUsage::PIXEL_FORMAT_VIEW)
        || (metal_keeps_uncompressed(!depth_stencil, usage)
            && !usage_flags.contains(ImageUsageFlags::STORAGE));
    let view_usage = ImageUsageFlags::SAMPLED
        | ImageUsageFlags::STORAGE
        | ImageUsageFlags::COLOR_ATTACHMENT
        | ImageUsageFlags::INPUT_ATTACHMENT;
    if any_format && !depth_stencil && !usage_flags.intersects(view_usage) {
        if let Some(m) = sampled.filter(|m| compatible(format, m.format, false)) {
            usage_flags |= ImageUsageFlags::SAMPLED;
            views.insert(m.format);
        }
    }
    let optimal = facts.optimal(format);
    if optimal.contains(FormatFeatureFlags2::TRANSFER_SRC) {
        usage_flags |= ImageUsageFlags::TRANSFER_SRC;
    }
    if optimal.contains(FormatFeatureFlags2::TRANSFER_DST) {
        usage_flags |= ImageUsageFlags::TRANSFER_DST;
    }

    if usage_flags.is_empty() {
        // The device holds the format for none of the texture's uses, not even copies: it lacks
        // the format.
        let needed = if wants(TextureUsage::SHADER_WRITE) || wants(TextureUsage::SHADER_ATOMIC) {
            FormatUsage::StorageImage
        } else if wants(TextureUsage::RENDER_TARGET) && !wants(TextureUsage::SHADER_READ) {
            attachment_usage
        } else {
            FormatUsage::SampledImage
        };
        return Err(ImageRefusal::Unsupported(Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format,
                usage: needed,
            },
        )));
    }

    // The sRGB twin's views, which Metal makes without `PixelFormatView`; not the view of the
    // twin's word (`WordAttachment`), which would make every such render target mutable for a view
    // few textures take: an attachment view of the twin the device renders only through its word
    // is refused, as without the word.
    if let Some(twin) = srgb_twin(pf) {
        let twin_word = WordAttachment::of(twin, profile).map(|w| w.word_format());
        views.extend(
            view_formats(twin, usage_flags, profile)
                .into_iter()
                .filter(|&f| compatible(format, f, true) && Some(f) != twin_word),
        );
    }
    views.insert(format);
    // A usage the image's own format lacks, which a view format has (a storage alias, a sampled
    // twin, a word attachment), needs extended usage.
    let extended = [
        (ImageUsageFlags::SAMPLED, FormatUsage::SampledImage),
        (ImageUsageFlags::STORAGE, FormatUsage::StorageImage),
        (
            ImageUsageFlags::COLOR_ATTACHMENT,
            FormatUsage::ColorAttachment,
        ),
    ]
    .iter()
    .any(|&(bit, need)| usage_flags.contains(bit) && !profile.supports_format(format, need));
    let mut view_formats = Vec::new();
    if any_format || views.len() > 1 || extended {
        flags |= ImageCreateFlags::MUTABLE_FORMAT;
        if extended {
            flags |= ImageCreateFlags::EXTENDED_USAGE;
        }
        if !any_format {
            view_formats.push(format);
            view_formats.extend(views.iter().copied().filter(|&f| f != format));
        }
    }
    if cube_compatible(layout) {
        flags |= ImageCreateFlags::CUBE_COMPATIBLE;
    }
    // A 3D render target renders into a 2D view of its depth planes, which a 3D image takes made
    // 2D-array compatible; a device whose portability subset withholds `imageView2DOn3DImage`
    // makes no such image (`VUID-VkImageCreateInfo-imageView2DOn3DImage-04459`), and its pass
    // refuses the attachment (`render_pass.rs`).
    if layout.texture_type == TextureType::Type3D
        && usage_flags.contains(ImageUsageFlags::COLOR_ATTACHMENT)
    {
        flags |= render_3d_flags(profile);
    }
    Ok(ImagePlan {
        pixel_format: pf,
        flags,
        image_type,
        format,
        extent: layout.extent,
        levels: layout.levels,
        layers: layout.layers,
        samples,
        usage: usage_flags,
        view_formats,
        mapping,
        unavailable,
    })
}

/// The formats of the views the runtime makes of a texture of `pf` for the Vulkan usages in
/// `usage` on `profile`: its sampled representation, its storage view (the alias, or the direct
/// format), and its attachment format (the direct format, or the view of its word where the device
/// renders that and not the format, `WordAttachment`). A view the device cannot make for `pf`
/// is left out: a format list naming a format the device lacks is one some drivers make no image
/// of (ANV, for an `RG8Unorm` render target listing its twin's `R8G8_SRGB`).
pub(super) fn view_formats(
    pf: PixelFormat,
    usage: ImageUsageFlags,
    profile: &DeviceProfile,
) -> Vec<Format> {
    let mut out = Vec::new();
    if usage.contains(ImageUsageFlags::SAMPLED) {
        if let Ok(m) = vk_format(pf, &[FormatUsage::SampledImage].into(), profile) {
            out.push(m.format);
        }
    }
    // A storage binding's views, the read-and-write choice's and a written-only one's.
    if usage.contains(ImageUsageFlags::STORAGE) {
        out.extend(storage_view_formats(pf, profile));
    }
    let attachment = ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    if usage.intersects(attachment) {
        let need = if is_depth_stencil(pf) {
            FormatUsage::DepthStencilAttachment
        } else {
            FormatUsage::ColorAttachment
        };
        out.extend(direct_format(pf).filter(|&f| profile.supports_format(f, need)));
        // A format the device samples but does not render renders through its word's view.
        if usage.contains(ImageUsageFlags::COLOR_ATTACHMENT) {
            out.extend(WordAttachment::of(pf, profile).map(|w| w.word_format()));
        }
    }
    out
}

/// Whether an image of `plan` takes the views the runtime makes of a texture of `pf` on
/// `profile` (`view_formats`, for the plan's usages): each is the image's format, or the image is
/// mutable and takes any compatible format (no format list: `PixelFormatView` usage) or lists it.
/// A view of the pixel format the image was planned for, or of its sRGB twin, always is: the plan
/// lists the twin's formats the device has, and a binding's or a pass's own view of the image is
/// checked where it is made, as before.
pub(super) fn takes_views_of(plan: &ImagePlan, pf: PixelFormat, profile: &DeviceProfile) -> bool {
    if pf == plan.pixel_format || srgb_twin(plan.pixel_format) == Some(pf) {
        return true;
    }
    let mutable = plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT);
    view_formats(pf, plan.usage, profile).iter().all(|&format| {
        format == plan.format
            || (mutable
                && (plan.view_formats.is_empty() || plan.view_formats.contains(&format))
                && compatible(plan.format, format, true))
    })
}

/// The cube compatibility a sampled image of a texture of `layout` takes: a cube's, and a square
/// 2D array's of six layers or more ([`cube_compatible`]), which a cube view may present.
pub(super) fn cube_flags(layout: &TextureLayout) -> ImageCreateFlags {
    match cube_compatible(layout) {
        true => ImageCreateFlags::CUBE_COMPATIBLE,
        false => ImageCreateFlags::empty(),
    }
}

/// Whether the image of `layout` is cube-compatible: a cube or cube array, or a square 2D array
/// of at least six single-sample layers (Metal views those as cubes).
pub(super) fn cube_compatible(layout: &TextureLayout) -> bool {
    match layout.texture_type {
        TextureType::Cube | TextureType::CubeArray => true,
        TextureType::Type2DArray => {
            layout.extent[0] == layout.extent[1] && layout.layers >= 6 && layout.samples == 1
        }
        _ => false,
    }
}

/// `plan` with sampled usage, where its driver keeps a single-sample depth image's hierarchical
/// depth in `GENERAL` only for a sampled image (`keeps_hiz`: the cost fact
/// [`DriverFact::SampledDepthKeepsHiZ`]): a single-sample depth attachment with a depth aspect,
/// not transient and not yet sampled, of a format the device samples. The image holds the same
/// texels either way; every image rests in `GENERAL` (`tracker.rs`) and takes transfer-source
/// usage, which on such a driver turns the hierarchical depth off there. `Device::with_free_usage`
/// gives the image the usage wherever the device makes it, though it may enlarge the image (ANV
/// keeps the write-through form's state in the image's memory), so a heap's size answer, which
/// reads the same plan, is that image's. `None` otherwise.
pub(super) fn hiz_candidate(
    plan: &ImagePlan,
    profile: &DeviceProfile,
    keeps_hiz: bool,
) -> Option<ImagePlan> {
    let usage = plan.usage;
    let depth = plan
        .format
        .block()
        .is_some_and(|block| block.depth.is_some());
    let candidate = keeps_hiz
        && depth
        && plan.samples.as_raw() == 1
        && usage.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        && !usage.intersects(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSIENT_ATTACHMENT)
        && profile.supports_format(plan.format, FormatUsage::SampledImage);
    candidate.then(|| ImagePlan {
        usage: usage | ImageUsageFlags::SAMPLED,
        ..plan.clone()
    })
}

fn attachment_flag(depth_stencil: bool, profile: &DeviceProfile) -> ImageUsageFlags {
    if depth_stencil {
        ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
    } else if framebuffer_fetch::through_inputs(profile) {
        // Any pass may fetch the attachment, which the texture cannot foresee.
        ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::INPUT_ATTACHMENT
    } else {
        ImageUsageFlags::COLOR_ATTACHMENT
    }
}

/// Whether a plan's image fits the device's image limits for its format, type, tiling, usage,
/// and flags (`VkImageFormatProperties`, from `vkGetPhysicalDeviceImageFormatProperties2`).
///
/// Refusals: `TextureSize` naming the device's dimension limit for the image's type (or its
/// array-layer limit) with the format's own maximum as what is available, and
/// `TextureSampleCount` naming the framebuffer sample-count property of the format's class.
pub(super) fn check_image_limits(
    plan: &ImagePlan,
    texture_type: TextureType,
    limits: &vk::ImageFormatProperties,
) -> Result<(), Unsupported> {
    let dimension = match texture_type {
        TextureType::Type1D | TextureType::Type1DArray | TextureType::TextureBuffer => {
            VkLimit::MaxImageDimension1D
        }
        TextureType::Cube | TextureType::CubeArray => VkLimit::MaxImageDimensionCube,
        TextureType::Type3D => VkLimit::MaxImageDimension3D,
        _ => VkLimit::MaxImageDimension2D,
    };
    let size = |limit: VkLimit, required: u32, available: u32| {
        Unsupported::needs(
            Feature::TextureSize,
            Capability::Limit {
                limit,
                required: u64::from(required),
                available: u64::from(available),
            },
        )
    };
    let max = [
        limits.max_extent.width,
        limits.max_extent.height,
        limits.max_extent.depth,
    ];
    if let Some((&want, &have)) = plan.extent.iter().zip(max.iter()).find(|(w, h)| w > h) {
        return Err(size(dimension, want, have));
    }
    if plan.layers > limits.max_array_layers {
        return Err(size(
            VkLimit::MaxImageArrayLayers,
            plan.layers,
            limits.max_array_layers,
        ));
    }
    if plan.levels > limits.max_mip_levels {
        // A full chain of `levels` needs an extent of `2^(levels - 1)`.
        let need = |levels: u32| 1u32 << (levels.clamp(1, 32) - 1);
        return Err(size(
            dimension,
            need(plan.levels),
            need(limits.max_mip_levels),
        ));
    }
    if !SampleCountFlags::from(limits.sample_counts).contains(plan.samples) {
        let n = plan.samples.as_raw();
        let block = plan.format.block();
        let property = match block {
            Some(b) if b.depth.is_some() => VkProperty::FramebufferDepthSampleCount(n),
            Some(b) if b.stencil.is_some() => VkProperty::FramebufferStencilSampleCount(n),
            Some(b)
                if b.color.is_some_and(|c| {
                    matches!(c.numeric_type(), NumericType::Uint | NumericType::Sint)
                }) =>
            {
                VkProperty::FramebufferIntegerColorSampleCount(n)
            }
            _ => VkProperty::FramebufferColorSampleCount(n),
        };
        return Err(Unsupported::needs(
            Feature::TextureSampleCount,
            Capability::Property(property),
        ));
    }
    Ok(())
}
