//! The pixel-format rows of Apple's tables and the texture capabilities they list, as the runtime
//! gives them on a device ([`Device::pixel_format_capabilities`]).
//!
//! Each capability is answered by planning the texture that uses it exactly as
//! [`Device::new_texture`] plans it (`image_plan::plan_image`). The device's limits for that
//! image (`vkGetPhysicalDeviceImageFormatProperties2`, `image_plan::check_image_limits`) are then
//! checked. The answer is therefore the one the runtime acts on. A probe of a texture type Metal
//! makes none of in the format answers no before any plan, as `new_texture` refuses that texture
//! by the descriptor's type rules: the multisample texture of a two-plane format, extended-range
//! or YCbCr, which Metal's validation stops (measured on an M4), so neither answers MSAA or
//! Resolve. Each probe is a 16 x 16 texture of one level:
//! - **Filter**: a 2D `ShaderRead` texture whose sampled view the binder lets a linear filter
//!   read (`SampledImageFilterLinear` of the view's format), of a format whose texels are not
//!   integers: Metal filters none, a stencil view's included, nor a bfloat format's (a linear
//!   sample reads the nearest texel). A texture held as its sRGB format's UNORM twin
//!   (`image_plan::holds_srgb_twin`) is filtered by the device through the twin's view and the
//!   filtered codes decoded by the device's transfer (`PrecisionNote::SrgbTexelDecode`): its Filter
//!   is the twin's linear filter. An extended-range texture
//!   (`ExtendedRange`) is filtered by the shader from fetches of its raw texels on every device:
//!   its Filter is its `ShaderRead` texture's.
//! - **Write**: a 2D `ShaderWrite` texture with a storage view.
//! - **Color**: a 2D color `RenderTarget`.
//! - **Blend**: a color render target whose format is the one a blending pipeline's attachment
//!   takes (`vk_format` with `ColorAttachmentBlend`). A format held with its channels rotated
//!   (`ARGB8Unorm`) also needs `dualSrcBlend`, through which it blends Metal's source alpha in
//!   a color factor; the blends no Vulkan factor gives there are refused per pipeline. An
//!   extended-range format, whose raw texels no device blends, blends in the shader inside the
//!   pixel interlock where the compile layer holds that form exact, or over its input attachment
//!   where the device orders fetch through rasterization-order input attachments
//!   (`metal::format::extended_range_blend`); the blends Metal's model does not hold are refused
//!   per pipeline (`Feature::ExtendedRangeBlend`).
//! - **MSAA**: a 2D multisample render target of 4 samples (color, or depth and stencil). Every
//!   family that lists MSAA renders at 4 samples. Apple's maximum count with MSAA is 4, and 8
//!   from Apple10, which the limit row of render-pass sample counts asks.
//! - **Resolve**: MSAA, in a format Metal resolves, into a single-sample render target of the
//!   format: the attachment the pass resolves into, or its scratch image (`resolve.rs`). Integer
//!   color formats are no resolve source in Metal, measured, and no stencil view of a combined
//!   format (`X32Stencil8`) answers one, as no column of Apple's tables lists it. Depth and
//!   stencil resolve through `SAMPLE_ZERO`, which every Vulkan 1.2 device has. A private form held
//!   in raw codes that Metal makes no multisampled texture of answers the render pass's refusal of
//!   its resolve.
//! - **Atomic**: a 2D `ShaderAtomic` texture with a storage view of a format a texture atomic
//!   updates. `R32Uint` and `R32Sint` need their format's `StorageImageAtomic`. `RG32Uint` is
//!   Metal's `ulong` texture: it needs its `R64_UINT` view's 64-bit image atomics. No other format
//!   takes Metal's texture atomics.
//! - **Sparse**: a 2D `ShaderRead` texture of a sparse heap of 16-KiB pages, planned as
//!   `Heap::new_texture` plans it: a sparse resident image where the device makes one of it, else
//!   an ordinary image whose unmapped tiles the runtime keeps at zero, with the refusal it would
//!   answer otherwise; the device's own sparse support, which a profile does not state, so a
//!   profile's answer is the gap. What every sparse texture still lacks is the sparse feature rows'
//!   (`features.rs`), not the format's.
//!
//! Two answers more serve feature rows of the tables:
//! - **Volume** (the compressed volume texture row): a 3D `ShaderRead` texture of 16 x 16 x 4.
//! - **Border** (the border color row): Metal's opaque black border read by a runtime sampler
//!   through the sampled view of the Filter probe's texture: as transparent black where Metal
//!   reads that texel (`metal::sampler::opaque_black_reads_transparent`, which the binder makes
//!   the sampler with), else as the view reads it (`metal::sampler::check_border_view`). A
//!   format the runtime does not sample has no border to refuse.
//!
//! A failed query of the device's limits answers no image, so it claims nothing.
//!
//! **At most the presented GPU's**. A format a row of Apple's tables lists answers at most that
//! row's Apple9 cell, the column of the Apple GPU every device presents (an M3 Max and an M4): a
//! capability the cell does not list answers no, as on that GPU, whatever the device makes (the M3
//! Max's probe found Sparse for `GBGR422`, `BGRG422` and `X32_Stencil8` answered past it). No
//! column of any row lists more than its Apple9 cell, so no family's row changes.

use super::{Column, Facts, Refusal, RowState};
use crate::device::lock;
use crate::error::{Error, RuntimeGap};
use crate::image_plan::{check_image_limits, plan_image, ImageFacts, ImagePlan, ImageRefusal};
use crate::resolve::unmodelled_resolve;
use crate::sparse::texture::sparse_refusal;
use crate::sparse::SparsePageSize;
use crate::texture_descriptor::{type_refused, TextureDescriptor, TextureLayout};
use crate::values::ResourceOptions;
use crate::Device;
use ash::vk;
use metal2vulkan::device::{DeviceProfile, FormatUsage, VkFeature};
use metal2vulkan::error::{Capability, Error as CompileError, Feature, Unsupported};
use metal2vulkan::metal::format::{
    extended_range_blend, is_depth_stencil, rotated_channels, sampled_type, storage_image_choice,
    vk_format, ExtendedRange, FormatEmulation,
};
use metal2vulkan::metal::sampler::{check_border_view, opaque_black_reads_transparent};
use metal2vulkan::metal::{PixelFormat, TextureSwizzleChannels, TextureType, TextureUsage};
use metal2vulkan::pipeline::SampledType;
use metal2vulkan::vk::Format;
use std::collections::BTreeSet;

/// A texture capability of Apple's tables' legend.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FormatCapability {
    /// "The GPU can use atomic operations on textures with the pixel format."
    Atomic,
    /// "The GPU can filter a texture with the pixel format during sampling."
    Filter,
    /// "The GPU can write to a texture on a per-pixel basis with the pixel format."
    Write,
    /// "The GPU can use a texture with the pixel format as a color render target."
    Color,
    /// "The GPU can blend a texture with the pixel format."
    Blend,
    /// "The GPU can use a texture with the pixel format as a destination for multisample
    /// antialias (MSAA) data."
    Msaa,
    /// "The GPU supports sparse-texture allocations for textures with the pixel format."
    Sparse,
    /// "The GPU can use a texture with the pixel format as a source for multisample antialias
    /// (MSAA) resolve operations."
    Resolve,
}

impl FormatCapability {
    /// Every capability, in the legend's order.
    pub const ALL: [FormatCapability; 8] = [
        FormatCapability::Atomic,
        FormatCapability::Filter,
        FormatCapability::Write,
        FormatCapability::Color,
        FormatCapability::Blend,
        FormatCapability::Msaa,
        FormatCapability::Sparse,
        FormatCapability::Resolve,
    ];

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A set of [`FormatCapability`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FormatCapabilities(u8);

impl FormatCapabilities {
    /// No capability.
    pub const NONE: FormatCapabilities = FormatCapabilities(0);

    /// The set of `capabilities`.
    pub const fn of(capabilities: &[FormatCapability]) -> FormatCapabilities {
        let mut bits = 0;
        let mut at = 0;
        while at < capabilities.len() {
            bits |= capabilities[at].bit();
            at += 1;
        }
        FormatCapabilities(bits)
    }

    /// Whether the set holds `capability`.
    pub fn contains(self, capability: FormatCapability) -> bool {
        self.0 & capability.bit() != 0
    }

    /// The capabilities of the set, in the legend's order.
    pub fn iter(self) -> impl Iterator<Item = FormatCapability> {
        FormatCapability::ALL
            .into_iter()
            .filter(move |&c| self.contains(c))
    }

    /// The set with `capability` added.
    fn with(self, capability: FormatCapability) -> FormatCapabilities {
        FormatCapabilities(self.0 | capability.bit())
    }
}

/// Why the runtime does not give a format a capability on a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Missing {
    /// The device lacks what the texture needs, or no Vulkan mechanism states it.
    Refused(Unsupported),
    /// The runtime piece is not built.
    Gap(RuntimeGap),
}

impl Missing {
    pub(super) fn state(self) -> RowState {
        match self {
            Missing::Refused(unsupported) => RowState::Refused(Refusal::Unsupported(unsupported)),
            Missing::Gap(gap) => RowState::Gap(gap),
        }
    }
}

/// A format's answers (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FormatAnswers {
    /// Each capability's, in [`FormatCapability::ALL`]'s order.
    capabilities: [Result<(), Missing>; 8],
    /// A 3D texture's.
    volume: Result<(), Missing>,
    /// Metal's opaque black border's.
    border: Result<(), Missing>,
}

impl FormatAnswers {
    /// The answer for `capability`.
    pub(crate) fn get(&self, capability: FormatCapability) -> Result<(), Missing> {
        self.capabilities[capability as usize].clone()
    }

    /// The answer for a 3D texture.
    pub(crate) fn volume(&self) -> Result<(), Missing> {
        self.volume.clone()
    }

    /// The answer for Metal's opaque black border.
    pub(crate) fn border(&self) -> Result<(), Missing> {
        self.border.clone()
    }

    /// The capabilities given.
    pub(crate) fn capabilities(&self) -> FormatCapabilities {
        FormatCapability::ALL
            .into_iter()
            .filter(|&c| self.capabilities[c as usize].is_ok())
            .fold(FormatCapabilities::NONE, FormatCapabilities::with)
    }

    /// The answers of a format the device's Metal describes none of: no capability, no 3D
    /// texture and no border, as Metal makes no texture of one.
    fn none() -> FormatAnswers {
        let no = || {
            Err(Missing::Refused(Unsupported::not_representable(
                Feature::Format,
            )))
        };
        FormatAnswers {
            capabilities: std::array::from_fn(|_| no()),
            volume: no(),
            border: no(),
        }
    }
}

/// Whether `plan` is a texture of a stencil view of a combined depth/stencil format
/// (`X32Stencil8`): its own Vulkan format, held with no emulation, has a depth aspect the Metal
/// format does not name. Every column of Apple's tables lists such a format without Resolve.
fn stencil_view(plan: &ImagePlan) -> bool {
    let mapping = &plan.mapping;
    mapping.emulation == FormatEmulation::Direct
        && mapping.aspects == metal2vulkan::vk::ImageAspectFlags::STENCIL
        && mapping
            .format
            .block()
            .is_some_and(|block| block.depth.is_some())
}

/// The device's limits for an image of a plan, as `vkGetPhysicalDeviceImageFormatProperties2`
/// gives them (all 0 where it makes no such image).
pub(crate) type ImageLimits<'a> = &'a dyn Fn(&ImagePlan) -> vk::ImageFormatProperties;

/// The answers of `pf` on a device of `facts`, whose limits for an image `limits` gives, and
/// whose Sparse answer is `sparse` (the module docs).
pub(crate) fn answers(
    pf: PixelFormat,
    facts: &ImageFacts<'_>,
    limits: ImageLimits<'_>,
    sparse: Result<(), Missing>,
) -> FormatAnswers {
    let profile = facts.profile;
    // A planned texture of `pf` of `usage` (the probe's `bit` of it available) whose image fits
    // the device's limits.
    let probe = |texture_type: TextureType, samples: u32, usage: TextureUsage| {
        // A texture of a type Metal makes none of in the format answers no, as `new_texture`
        // refuses it: the two-plane extended-range and YCbCr formats' multisample textures.
        if type_refused(pf, texture_type) {
            return Err(Missing::Refused(Unsupported::not_representable(
                Feature::Format,
            )));
        }
        let depth = if texture_type == TextureType::Type3D {
            4
        } else {
            1
        };
        let layout = TextureLayout {
            texture_type,
            pixel_format: pf,
            extent: [16, 16, depth],
            levels: 1,
            layers: 1,
            samples,
        };
        let plan = plan_image(&layout, usage, false, facts).map_err(|refusal| match refusal {
            ImageRefusal::Unsupported(unsupported) => Missing::Refused(unsupported),
            ImageRefusal::NotAFormat | ImageRefusal::TextureBuffer => {
                Missing::Refused(Unsupported::not_representable(Feature::Format))
            }
        })?;
        if plan.unavailable.intersects(usage) {
            return Err(Missing::Refused(lacking(pf, usage, plan.format, profile)));
        }
        check_image_limits(&plan, texture_type, &limits(&plan)).map_err(Missing::Refused)?;
        Ok(plan)
    };
    let depth_stencil = is_depth_stencil(pf);
    let reads_integers = matches!(
        sampled_type(pf),
        Some(SampledType::Uint | SampledType::Sint)
    );
    let integer = reads_integers && !depth_stencil;
    let not_metal = || Missing::Refused(Unsupported::not_representable(Feature::Format));
    let single = probe(TextureType::Type2D, 1, TextureUsage::RENDER_TARGET);
    let msaa = probe(
        TextureType::Type2DMultisample,
        4,
        TextureUsage::RENDER_TARGET,
    );
    let color = if depth_stencil {
        Err(not_metal())
    } else {
        single.clone()
    };
    // A texture held as its sRGB format's UNORM twin is sampled through the twin's view, whose
    // filtered codes the function decodes (`lower::texture::srgb_filtered_texel`): its filter and
    // border are that view's.
    let sampled = probe(TextureType::Type2D, 1, TextureUsage::SHADER_READ);
    let own = TextureSwizzleChannels::DEFAULT;
    let border = match &sampled {
        // The binder makes a runtime sampler's opaque-black border transparent black where Metal
        // reads that texel (`A8Unorm`, a format without alpha), which every device reads exactly.
        Ok(_) if opaque_black_reads_transparent(pf, own) => Ok(()),
        Ok(plan) => check_border_view(true, pf, &plan.mapping, own, profile)
            .map_err(|error| Missing::Refused(unsupported(error))),
        Err(_) => Ok(()),
    };
    let extended = ExtendedRange::of(pf).is_some();
    let filter = sampled.and_then(|plan| {
        // Metal filters no bfloat texel: a linear sample reads the nearest texel.
        if ExtendedRange::of(pf).is_some_and(ExtendedRange::bfloat) {
            return Err(not_metal());
        }
        // An extended-range texture is filtered by the shader, from fetches of its raw texels, on
        // every device (`lower::sampler::extended`): its image's integer format filters nothing,
        // and needs not. A bound sampler past what the fetches model is refused per sampler
        // (`ResourcePlanner::extended_sampler`).
        if extended {
            return Ok(());
        }
        // Metal filters no integer texels, a stencil view's included, whose image's format
        // states its depth aspect's filter.
        if reads_integers {
            return Err(not_metal());
        }
        let view = plan.mapping.format;
        if profile.supports_format(view, FormatUsage::SampledImageFilterLinear) {
            Ok(())
        } else {
            Err(Missing::Refused(Unsupported::needs(
                Feature::Format,
                Capability::Format {
                    format: view,
                    usage: FormatUsage::SampledImageFilterLinear,
                },
            )))
        }
    });
    let blend = color.clone().and_then(|plan| {
        // Metal blends no bfloat attachment: its validation asserts.
        if ExtendedRange::of(pf).is_some_and(ExtendedRange::bfloat) {
            return Err(not_metal());
        }
        // No device blends an extended-range attachment's raw texels: the function blends in the
        // shader inside the pixel interlock where the compile layer's storage form holds it
        // (`extended_range_blend`), a pipeline's blend Metal's model does not hold refused per
        // pipeline (`Feature::ExtendedRangeBlend`), as a rotated format's are.
        if extended {
            return extended_range_blend(pf, profile).map_err(Missing::Refused);
        }
        let usages = BTreeSet::from([
            FormatUsage::ColorAttachment,
            FormatUsage::ColorAttachmentBlend,
        ]);
        match vk_format(pf, &usages, profile) {
            // A format held with its channels rotated blends a color factor of Metal's source
            // alpha through a second output, which needs dual-source blending.
            Ok(mapping)
                if mapping.format == plan.format
                    && rotated_channels(pf)
                    && !profile.fragment().dual_src_blend =>
            {
                Err(Missing::Refused(Unsupported::needs(
                    Feature::DualSourceBlending,
                    Capability::Feature(VkFeature::DualSrcBlend),
                )))
            }
            Ok(mapping) if mapping.format == plan.format => Ok(()),
            Ok(_) => Err(Missing::Refused(Unsupported::needs(
                Feature::Format,
                Capability::Format {
                    format: plan.format,
                    usage: FormatUsage::ColorAttachmentBlend,
                },
            ))),
            Err(error) => Err(Missing::Refused(unsupported(error))),
        }
    });
    let resolve = if integer || single.as_ref().is_ok_and(stencil_view) {
        Err(not_metal())
    } else if let Some(unsupported) = unmodelled_resolve(pf) {
        // The resolve the render pass refuses (`resolve.rs`): a private form's codes, of which
        // Metal makes no multisampled texture.
        Err(Missing::Refused(unsupported))
    } else {
        msaa.clone().and(single.clone()).map(drop)
    };
    let atomic = probe(TextureType::Type2D, 1, TextureUsage::SHADER_ATOMIC).and_then(|_| {
        match atomic_gap(pf, profile) {
            Some(missing) => Err(missing),
            None => Ok(()),
        }
    });
    let mut capabilities: [Result<(), Missing>; 8] = std::array::from_fn(|_| Ok(()));
    capabilities[FormatCapability::Atomic as usize] = atomic;
    capabilities[FormatCapability::Filter as usize] = filter;
    capabilities[FormatCapability::Write as usize] =
        probe(TextureType::Type2D, 1, TextureUsage::SHADER_WRITE).map(drop);
    capabilities[FormatCapability::Color as usize] = color.map(drop);
    capabilities[FormatCapability::Blend as usize] = blend;
    capabilities[FormatCapability::Msaa as usize] = msaa.map(drop);
    capabilities[FormatCapability::Sparse as usize] = sparse;
    capabilities[FormatCapability::Resolve as usize] = resolve;
    // No answer past the Apple GPU's the runtime presents: a format Apple's tables list takes at
    // most its cell of that GPU's column.
    if let Some(cell) = presented_cell(pf) {
        for capability in FormatCapability::ALL {
            if !cell.contains(capability) {
                capabilities[capability as usize] = Err(not_metal());
            }
        }
    }
    FormatAnswers {
        capabilities,
        volume: probe(TextureType::Type3D, 1, TextureUsage::SHADER_READ).map(drop),
        border,
    }
}

/// The column of the Apple GPU every device presents (an Apple M3 Max and an M4, both Apple9;
/// the compile layer's `device::metal::apple_gpu`). Every column's cell of every row lists at most
/// this one's capabilities, so capping an answer at it changes no family's row.
const PRESENTED: Column = Column::Apple(9);

/// The capabilities `pf`'s row of Apple's tables lists in the [`PRESENTED`] column, or `None` for
/// a format no row lists.
fn presented_cell(pf: PixelFormat) -> Option<FormatCapabilities> {
    FormatRow::ALL
        .into_iter()
        .find(|row| row.formats().contains(&pf))
        .map(|row| row.listed(PRESENTED))
}

/// What a texture atomic on a texture of `pf` lacks on `profile`, if anything (the module docs).
fn atomic_gap(pf: PixelFormat, profile: &DeviceProfile) -> Option<Missing> {
    let format = match pf {
        PixelFormat::R32Uint => Format::R32_UINT,
        PixelFormat::R32Sint => Format::R32_SINT,
        PixelFormat::RG32Uint => {
            let gaps = profile.ulong_atomic_gaps(false);
            return (!gaps.is_empty()).then(|| {
                Missing::Refused(Unsupported::needs_any_of(Feature::ImageAtomics, vec![gaps]))
            });
        }
        _ => {
            return Some(Missing::Refused(Unsupported::not_representable(
                Feature::ImageAtomics,
            )))
        }
    };
    (!profile.supports_format(format, FormatUsage::StorageImageAtomic)).then(|| {
        Missing::Refused(Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format,
                usage: FormatUsage::StorageImageAtomic,
            },
        ))
    })
}

/// The refusal of a texture of `pf` whose plan, in format `format`, left off a usage of `usage`:
/// the compile layer's refusal of the representation that usage needs.
fn lacking(
    pf: PixelFormat,
    usage: TextureUsage,
    format: Format,
    profile: &DeviceProfile,
) -> Unsupported {
    let needed = if usage.intersects(TextureUsage::SHADER_WRITE | TextureUsage::SHADER_ATOMIC) {
        match storage_image_choice(pf, profile) {
            Err(error) => return unsupported(error),
            Ok(_) => FormatUsage::StorageImage,
        }
    } else if usage.contains(TextureUsage::RENDER_TARGET) {
        if is_depth_stencil(pf) {
            FormatUsage::DepthStencilAttachment
        } else {
            FormatUsage::ColorAttachment
        }
    } else {
        FormatUsage::SampledImage
    };
    match vk_format(pf, &BTreeSet::from([needed]), profile) {
        Err(error) => unsupported(error),
        Ok(_) => Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format,
                usage: needed,
            },
        ),
    }
}

/// The compile layer's refusal `error` as an `Unsupported`.
fn unsupported(error: CompileError) -> Unsupported {
    match error {
        CompileError::Unsupported(unsupported) => unsupported,
        _ => Unsupported::not_representable(Feature::Format),
    }
}

/// A pixel-format entry of Apple's tables: one format, or a compressed family of them.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FormatRow {
    /// A8Unorm.
    A8Unorm,
    /// R8Unorm.
    R8Unorm,
    /// R8Unorm_sRGB.
    R8UnormSRGB,
    /// R8Snorm.
    R8Snorm,
    /// R8Uint.
    R8Uint,
    /// R8Sint.
    R8Sint,
    /// R16Unorm.
    R16Unorm,
    /// R16Snorm.
    R16Snorm,
    /// R16Uint.
    R16Uint,
    /// R16Sint.
    R16Sint,
    /// R16Float.
    R16Float,
    /// RG8Unorm.
    RG8Unorm,
    /// RG8Unorm_sRGB.
    RG8UnormSRGB,
    /// RG8Snorm.
    RG8Snorm,
    /// RG8Uint.
    RG8Uint,
    /// RG8Sint.
    RG8Sint,
    /// B5G6R5Unorm.
    B5G6R5Unorm,
    /// A1BGR5Unorm.
    A1BGR5Unorm,
    /// ABGR4Unorm.
    ABGR4Unorm,
    /// BGR5A1Unorm.
    BGR5A1Unorm,
    /// R32Uint.
    R32Uint,
    /// R32Sint.
    R32Sint,
    /// R32Float.
    R32Float,
    /// RG16Unorm.
    RG16Unorm,
    /// RG16Snorm.
    RG16Snorm,
    /// RG16Uint.
    RG16Uint,
    /// RG16Sint.
    RG16Sint,
    /// RG16Float.
    RG16Float,
    /// RGBA8Unorm.
    RGBA8Unorm,
    /// RGBA8Unorm_sRGB.
    RGBA8UnormSRGB,
    /// RGBA8Snorm.
    RGBA8Snorm,
    /// RGBA8Uint.
    RGBA8Uint,
    /// RGBA8Sint.
    RGBA8Sint,
    /// BGRA8Unorm.
    BGRA8Unorm,
    /// BGRA8Unorm_sRGB.
    BGRA8UnormSRGB,
    /// RGB10A2Unorm.
    RGB10A2Unorm,
    /// BGR10A2Unorm.
    BGR10A2Unorm,
    /// RGB10A2Uint.
    RGB10A2Uint,
    /// RG11B10Float.
    RG11B10Float,
    /// RGB9E5Float.
    RGB9E5Float,
    /// RG32Uint.
    RG32Uint,
    /// RG32Sint.
    RG32Sint,
    /// RG32Float.
    RG32Float,
    /// RGBA16Unorm.
    RGBA16Unorm,
    /// RGBA16Snorm.
    RGBA16Snorm,
    /// RGBA16Uint.
    RGBA16Uint,
    /// RGBA16Sint.
    RGBA16Sint,
    /// RGBA16Float.
    RGBA16Float,
    /// RGBA32Uint.
    RGBA32Uint,
    /// RGBA32Sint.
    RGBA32Sint,
    /// RGBA32Float.
    RGBA32Float,
    /// PVRTC pixel formats.
    PvrtcFormats,
    /// EAC/ETC pixel formats.
    EacEtcFormats,
    /// ASTC pixel formats.
    AstcFormats,
    /// HDR ASTC pixel formats.
    HdrAstcFormats,
    /// BC pixel formats.
    BcFormats,
    /// GBGR422.
    GBGR422,
    /// BGRG422.
    BGRG422,
    /// Depth16Unorm.
    Depth16Unorm,
    /// Depth32Float.
    Depth32Float,
    /// Stencil8.
    Stencil8,
    /// Depth24Unorm_Stencil8.
    Depth24UnormStencil8,
    /// Depth32Float_Stencil8.
    Depth32FloatStencil8,
    /// X24_Stencil8.
    X24Stencil8,
    /// X32_Stencil8.
    X32Stencil8,
    /// BGRA10_XR.
    BGRA10XR,
    /// BGRA10_XR_sRGB.
    BGRA10XRSRGB,
    /// BGR10_XR.
    BGR10XR,
    /// BGR10_XR_sRGB.
    BGR10XRSRGB,
}

impl FormatRow {
    /// Every row, in the tables' order.
    pub const ALL: [FormatRow; 69] = [
        FormatRow::A8Unorm,
        FormatRow::R8Unorm,
        FormatRow::R8UnormSRGB,
        FormatRow::R8Snorm,
        FormatRow::R8Uint,
        FormatRow::R8Sint,
        FormatRow::R16Unorm,
        FormatRow::R16Snorm,
        FormatRow::R16Uint,
        FormatRow::R16Sint,
        FormatRow::R16Float,
        FormatRow::RG8Unorm,
        FormatRow::RG8UnormSRGB,
        FormatRow::RG8Snorm,
        FormatRow::RG8Uint,
        FormatRow::RG8Sint,
        FormatRow::B5G6R5Unorm,
        FormatRow::A1BGR5Unorm,
        FormatRow::ABGR4Unorm,
        FormatRow::BGR5A1Unorm,
        FormatRow::R32Uint,
        FormatRow::R32Sint,
        FormatRow::R32Float,
        FormatRow::RG16Unorm,
        FormatRow::RG16Snorm,
        FormatRow::RG16Uint,
        FormatRow::RG16Sint,
        FormatRow::RG16Float,
        FormatRow::RGBA8Unorm,
        FormatRow::RGBA8UnormSRGB,
        FormatRow::RGBA8Snorm,
        FormatRow::RGBA8Uint,
        FormatRow::RGBA8Sint,
        FormatRow::BGRA8Unorm,
        FormatRow::BGRA8UnormSRGB,
        FormatRow::RGB10A2Unorm,
        FormatRow::BGR10A2Unorm,
        FormatRow::RGB10A2Uint,
        FormatRow::RG11B10Float,
        FormatRow::RGB9E5Float,
        FormatRow::RG32Uint,
        FormatRow::RG32Sint,
        FormatRow::RG32Float,
        FormatRow::RGBA16Unorm,
        FormatRow::RGBA16Snorm,
        FormatRow::RGBA16Uint,
        FormatRow::RGBA16Sint,
        FormatRow::RGBA16Float,
        FormatRow::RGBA32Uint,
        FormatRow::RGBA32Sint,
        FormatRow::RGBA32Float,
        FormatRow::PvrtcFormats,
        FormatRow::EacEtcFormats,
        FormatRow::AstcFormats,
        FormatRow::HdrAstcFormats,
        FormatRow::BcFormats,
        FormatRow::GBGR422,
        FormatRow::BGRG422,
        FormatRow::Depth16Unorm,
        FormatRow::Depth32Float,
        FormatRow::Stencil8,
        FormatRow::Depth24UnormStencil8,
        FormatRow::Depth32FloatStencil8,
        FormatRow::X24Stencil8,
        FormatRow::X32Stencil8,
        FormatRow::BGRA10XR,
        FormatRow::BGRA10XRSRGB,
        FormatRow::BGR10XR,
        FormatRow::BGR10XRSRGB,
    ];

    /// The row's name as Apple's tables print it.
    pub fn label(self) -> &'static str {
        match self {
            FormatRow::A8Unorm => "A8Unorm",
            FormatRow::R8Unorm => "R8Unorm",
            FormatRow::R8UnormSRGB => "R8Unorm_sRGB",
            FormatRow::R8Snorm => "R8Snorm",
            FormatRow::R8Uint => "R8Uint",
            FormatRow::R8Sint => "R8Sint",
            FormatRow::R16Unorm => "R16Unorm",
            FormatRow::R16Snorm => "R16Snorm",
            FormatRow::R16Uint => "R16Uint",
            FormatRow::R16Sint => "R16Sint",
            FormatRow::R16Float => "R16Float",
            FormatRow::RG8Unorm => "RG8Unorm",
            FormatRow::RG8UnormSRGB => "RG8Unorm_sRGB",
            FormatRow::RG8Snorm => "RG8Snorm",
            FormatRow::RG8Uint => "RG8Uint",
            FormatRow::RG8Sint => "RG8Sint",
            FormatRow::B5G6R5Unorm => "B5G6R5Unorm",
            FormatRow::A1BGR5Unorm => "A1BGR5Unorm",
            FormatRow::ABGR4Unorm => "ABGR4Unorm",
            FormatRow::BGR5A1Unorm => "BGR5A1Unorm",
            FormatRow::R32Uint => "R32Uint",
            FormatRow::R32Sint => "R32Sint",
            FormatRow::R32Float => "R32Float",
            FormatRow::RG16Unorm => "RG16Unorm",
            FormatRow::RG16Snorm => "RG16Snorm",
            FormatRow::RG16Uint => "RG16Uint",
            FormatRow::RG16Sint => "RG16Sint",
            FormatRow::RG16Float => "RG16Float",
            FormatRow::RGBA8Unorm => "RGBA8Unorm",
            FormatRow::RGBA8UnormSRGB => "RGBA8Unorm_sRGB",
            FormatRow::RGBA8Snorm => "RGBA8Snorm",
            FormatRow::RGBA8Uint => "RGBA8Uint",
            FormatRow::RGBA8Sint => "RGBA8Sint",
            FormatRow::BGRA8Unorm => "BGRA8Unorm",
            FormatRow::BGRA8UnormSRGB => "BGRA8Unorm_sRGB",
            FormatRow::RGB10A2Unorm => "RGB10A2Unorm",
            FormatRow::BGR10A2Unorm => "BGR10A2Unorm",
            FormatRow::RGB10A2Uint => "RGB10A2Uint",
            FormatRow::RG11B10Float => "RG11B10Float",
            FormatRow::RGB9E5Float => "RGB9E5Float",
            FormatRow::RG32Uint => "RG32Uint",
            FormatRow::RG32Sint => "RG32Sint",
            FormatRow::RG32Float => "RG32Float",
            FormatRow::RGBA16Unorm => "RGBA16Unorm",
            FormatRow::RGBA16Snorm => "RGBA16Snorm",
            FormatRow::RGBA16Uint => "RGBA16Uint",
            FormatRow::RGBA16Sint => "RGBA16Sint",
            FormatRow::RGBA16Float => "RGBA16Float",
            FormatRow::RGBA32Uint => "RGBA32Uint",
            FormatRow::RGBA32Sint => "RGBA32Sint",
            FormatRow::RGBA32Float => "RGBA32Float",
            FormatRow::PvrtcFormats => "PVRTC pixel formats",
            FormatRow::EacEtcFormats => "EAC/ETC pixel formats",
            FormatRow::AstcFormats => "ASTC pixel formats",
            FormatRow::HdrAstcFormats => "HDR ASTC pixel formats",
            FormatRow::BcFormats => "BC pixel formats",
            FormatRow::GBGR422 => "GBGR422",
            FormatRow::BGRG422 => "BGRG422",
            FormatRow::Depth16Unorm => "Depth16Unorm",
            FormatRow::Depth32Float => "Depth32Float",
            FormatRow::Stencil8 => "Stencil8",
            FormatRow::Depth24UnormStencil8 => "Depth24Unorm_Stencil8",
            FormatRow::Depth32FloatStencil8 => "Depth32Float_Stencil8",
            FormatRow::X24Stencil8 => "X24_Stencil8",
            FormatRow::X32Stencil8 => "X32_Stencil8",
            FormatRow::BGRA10XR => "BGRA10_XR",
            FormatRow::BGRA10XRSRGB => "BGRA10_XR_sRGB",
            FormatRow::BGR10XR => "BGR10_XR",
            FormatRow::BGR10XRSRGB => "BGR10_XR_sRGB",
        }
    }

    /// The pixel formats of the row.
    pub fn formats(self) -> &'static [PixelFormat] {
        use PixelFormat as P;
        match self {
            FormatRow::A8Unorm => &[P::A8Unorm],
            FormatRow::R8Unorm => &[P::R8Unorm],
            FormatRow::R8UnormSRGB => &[P::R8UnormSRGB],
            FormatRow::R8Snorm => &[P::R8Snorm],
            FormatRow::R8Uint => &[P::R8Uint],
            FormatRow::R8Sint => &[P::R8Sint],
            FormatRow::R16Unorm => &[P::R16Unorm],
            FormatRow::R16Snorm => &[P::R16Snorm],
            FormatRow::R16Uint => &[P::R16Uint],
            FormatRow::R16Sint => &[P::R16Sint],
            FormatRow::R16Float => &[P::R16Float],
            FormatRow::RG8Unorm => &[P::RG8Unorm],
            FormatRow::RG8UnormSRGB => &[P::RG8UnormSRGB],
            FormatRow::RG8Snorm => &[P::RG8Snorm],
            FormatRow::RG8Uint => &[P::RG8Uint],
            FormatRow::RG8Sint => &[P::RG8Sint],
            FormatRow::B5G6R5Unorm => &[P::B5G6R5Unorm],
            FormatRow::A1BGR5Unorm => &[P::A1BGR5Unorm],
            FormatRow::ABGR4Unorm => &[P::ABGR4Unorm],
            FormatRow::BGR5A1Unorm => &[P::BGR5A1Unorm],
            FormatRow::R32Uint => &[P::R32Uint],
            FormatRow::R32Sint => &[P::R32Sint],
            FormatRow::R32Float => &[P::R32Float],
            FormatRow::RG16Unorm => &[P::RG16Unorm],
            FormatRow::RG16Snorm => &[P::RG16Snorm],
            FormatRow::RG16Uint => &[P::RG16Uint],
            FormatRow::RG16Sint => &[P::RG16Sint],
            FormatRow::RG16Float => &[P::RG16Float],
            FormatRow::RGBA8Unorm => &[P::RGBA8Unorm],
            FormatRow::RGBA8UnormSRGB => &[P::RGBA8UnormSRGB],
            FormatRow::RGBA8Snorm => &[P::RGBA8Snorm],
            FormatRow::RGBA8Uint => &[P::RGBA8Uint],
            FormatRow::RGBA8Sint => &[P::RGBA8Sint],
            FormatRow::BGRA8Unorm => &[P::BGRA8Unorm],
            FormatRow::BGRA8UnormSRGB => &[P::BGRA8UnormSRGB],
            FormatRow::RGB10A2Unorm => &[P::RGB10A2Unorm],
            FormatRow::BGR10A2Unorm => &[P::BGR10A2Unorm],
            FormatRow::RGB10A2Uint => &[P::RGB10A2Uint],
            FormatRow::RG11B10Float => &[P::RG11B10Float],
            FormatRow::RGB9E5Float => &[P::RGB9E5Float],
            FormatRow::RG32Uint => &[P::RG32Uint],
            FormatRow::RG32Sint => &[P::RG32Sint],
            FormatRow::RG32Float => &[P::RG32Float],
            FormatRow::RGBA16Unorm => &[P::RGBA16Unorm],
            FormatRow::RGBA16Snorm => &[P::RGBA16Snorm],
            FormatRow::RGBA16Uint => &[P::RGBA16Uint],
            FormatRow::RGBA16Sint => &[P::RGBA16Sint],
            FormatRow::RGBA16Float => &[P::RGBA16Float],
            FormatRow::RGBA32Uint => &[P::RGBA32Uint],
            FormatRow::RGBA32Sint => &[P::RGBA32Sint],
            FormatRow::RGBA32Float => &[P::RGBA32Float],
            FormatRow::PvrtcFormats => &[
                P::PVRTCRGB2BPP,
                P::PVRTCRGB2BPPSRGB,
                P::PVRTCRGB4BPP,
                P::PVRTCRGB4BPPSRGB,
                P::PVRTCRGBA2BPP,
                P::PVRTCRGBA2BPPSRGB,
                P::PVRTCRGBA4BPP,
                P::PVRTCRGBA4BPPSRGB,
            ],
            FormatRow::EacEtcFormats => &[
                P::EACR11Unorm,
                P::EACR11Snorm,
                P::EACRG11Unorm,
                P::EACRG11Snorm,
                P::EACRGBA8,
                P::EACRGBA8SRGB,
                P::ETC2RGB8,
                P::ETC2RGB8SRGB,
                P::ETC2RGB8A1,
                P::ETC2RGB8A1SRGB,
            ],
            FormatRow::AstcFormats => &[
                P::ASTC4x4LDR,
                P::ASTC5x4LDR,
                P::ASTC5x5LDR,
                P::ASTC6x5LDR,
                P::ASTC6x6LDR,
                P::ASTC8x5LDR,
                P::ASTC8x6LDR,
                P::ASTC8x8LDR,
                P::ASTC10x5LDR,
                P::ASTC10x6LDR,
                P::ASTC10x8LDR,
                P::ASTC10x10LDR,
                P::ASTC12x10LDR,
                P::ASTC12x12LDR,
                P::ASTC4x4SRGB,
                P::ASTC5x4SRGB,
                P::ASTC5x5SRGB,
                P::ASTC6x5SRGB,
                P::ASTC6x6SRGB,
                P::ASTC8x5SRGB,
                P::ASTC8x6SRGB,
                P::ASTC8x8SRGB,
                P::ASTC10x5SRGB,
                P::ASTC10x6SRGB,
                P::ASTC10x8SRGB,
                P::ASTC10x10SRGB,
                P::ASTC12x10SRGB,
                P::ASTC12x12SRGB,
            ],
            FormatRow::HdrAstcFormats => &[
                P::ASTC4x4HDR,
                P::ASTC5x4HDR,
                P::ASTC5x5HDR,
                P::ASTC6x5HDR,
                P::ASTC6x6HDR,
                P::ASTC8x5HDR,
                P::ASTC8x6HDR,
                P::ASTC8x8HDR,
                P::ASTC10x5HDR,
                P::ASTC10x6HDR,
                P::ASTC10x8HDR,
                P::ASTC10x10HDR,
                P::ASTC12x10HDR,
                P::ASTC12x12HDR,
            ],
            FormatRow::BcFormats => &[
                P::BC1RGBA,
                P::BC1RGBASRGB,
                P::BC2RGBA,
                P::BC2RGBASRGB,
                P::BC3RGBA,
                P::BC3RGBASRGB,
                P::BC4RUnorm,
                P::BC4RSnorm,
                P::BC5RGUnorm,
                P::BC5RGSnorm,
                P::BC6HRGBFloat,
                P::BC6HRGBUfloat,
                P::BC7RGBAUnorm,
                P::BC7RGBAUnormSRGB,
            ],
            FormatRow::GBGR422 => &[P::GBGR422],
            FormatRow::BGRG422 => &[P::BGRG422],
            FormatRow::Depth16Unorm => &[P::Depth16Unorm],
            FormatRow::Depth32Float => &[P::Depth32Float],
            FormatRow::Stencil8 => &[P::Stencil8],
            FormatRow::Depth24UnormStencil8 => &[P::Depth24UnormStencil8],
            FormatRow::Depth32FloatStencil8 => &[P::Depth32FloatStencil8],
            FormatRow::X24Stencil8 => &[P::X24Stencil8],
            FormatRow::X32Stencil8 => &[P::X32Stencil8],
            FormatRow::BGRA10XR => &[P::BGRA10XR],
            FormatRow::BGRA10XRSRGB => &[P::BGRA10XRSRGB],
            FormatRow::BGR10XR => &[P::BGR10XR],
            FormatRow::BGR10XRSRGB => &[P::BGR10XRSRGB],
        }
    }

    /// The capabilities the row's cell in `column` lists (the module docs).
    pub(super) fn listed(self, column: Column) -> FormatCapabilities {
        let cells = self.cells();
        match column {
            Column::Apple(n) => cells.apple(n),
            Column::Metal3 => cells.apple(7),
            Column::Mac2 => FormatCapabilities::of(cells.mac2),
            Column::Common(_) => FormatCapabilities::NONE,
        }
    }

    /// The row's cells: the 2026 edition's Apple columns as runs of equal cells, each from the
    /// first Apple family it holds for (a family before the first run lists nothing), then the 2023
    /// edition's Mac2 cell. "All" is expanded (the module docs).
    fn cells(self) -> Cells {
        use FormatCapability::*;
        let none: &[FormatCapability] = &[];
        match self {
            // every Apple column lists capabilities
            FormatRow::A8Unorm => Cells {
                apple: &[
                    (2, &[Filter]),
                    (3, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::R8Unorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::R8UnormSRGB => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // every Apple column lists capabilities
            FormatRow::R8Snorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::R8Uint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::R8Sint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::R16Unorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa]),
                    (4, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::R16Snorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa]),
                    (4, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::R16Uint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::R16Sint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::R16Float => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RG8Unorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RG8UnormSRGB => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // every Apple column lists capabilities
            FormatRow::RG8Snorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RG8Uint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RG8Sint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::B5G6R5Unorm => Cells {
                apple: &[
                    (2, &[Filter, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // every Apple column lists capabilities
            FormatRow::A1BGR5Unorm => Cells {
                apple: &[
                    (2, &[Filter, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // every Apple column lists capabilities
            FormatRow::ABGR4Unorm => Cells {
                apple: &[
                    (2, &[Filter, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // every Apple column lists capabilities
            FormatRow::BGR5A1Unorm => Cells {
                apple: &[
                    (2, &[Filter, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // every Apple column lists capabilities
            FormatRow::R32Uint => Cells {
                apple: &[
                    (2, &[Write, Color]),
                    (6, &[Atomic, Write, Color, Sparse]),
                    (7, &[Atomic, Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Atomic, Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::R32Sint => Cells {
                apple: &[
                    (2, &[Write, Color]),
                    (6, &[Atomic, Write, Color, Sparse]),
                    (7, &[Atomic, Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Atomic, Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::R32Float => Cells {
                apple: &[
                    (2, &[Write, Color, Blend, Msaa]),
                    (6, &[Write, Color, Blend, Msaa, Sparse]),
                    (9, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RG16Unorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa]),
                    (4, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RG16Snorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa]),
                    (4, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RG16Uint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RG16Sint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RG16Float => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA8Unorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA8UnormSRGB => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA8Snorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA8Uint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA8Sint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::BGRA8Unorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::BGRA8UnormSRGB => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGB10A2Unorm => Cells {
                apple: &[
                    (2, &[Filter, Color, Blend, Msaa, Resolve]),
                    (3, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::BGR10A2Unorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGB10A2Uint => Cells {
                apple: &[
                    (2, &[Color, Msaa]),
                    (3, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RG11B10Float => Cells {
                apple: &[
                    (2, &[Filter, Color, Blend, Msaa, Resolve]),
                    (3, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGB9E5Float => Cells {
                apple: &[
                    (2, &[Filter, Color, Blend, Msaa, Resolve]),
                    (3, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter],
            },
            // every Apple column lists capabilities
            FormatRow::RG32Uint => Cells {
                apple: &[
                    (2, &[Write, Color]),
                    (6, &[Write, Color, Sparse]),
                    (7, &[Write, Color, Msaa, Sparse]),
                    (8, &[Atomic, Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RG32Sint => Cells {
                apple: &[
                    (2, &[Write, Color]),
                    (6, &[Write, Color, Sparse]),
                    (7, &[Write, Color, Msaa, Sparse]),
                    (8, &[Atomic, Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RG32Float => Cells {
                apple: &[
                    (2, &[Write, Color, Blend]),
                    (6, &[Write, Color, Blend, Sparse]),
                    (7, &[Write, Color, Blend, Msaa, Sparse]),
                    (9, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA16Unorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa]),
                    (4, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA16Snorm => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa]),
                    (4, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA16Uint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA16Sint => Cells {
                apple: &[
                    (2, &[Write, Color, Msaa]),
                    (6, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA16Float => Cells {
                apple: &[
                    (2, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA32Uint => Cells {
                apple: &[
                    (2, &[Write, Color]),
                    (6, &[Write, Color, Sparse]),
                    (7, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA32Sint => Cells {
                apple: &[
                    (2, &[Write, Color]),
                    (6, &[Write, Color, Sparse]),
                    (7, &[Write, Color, Msaa, Sparse]),
                ],
                mac2: &[Write, Color, Msaa],
            },
            // every Apple column lists capabilities
            FormatRow::RGBA32Float => Cells {
                apple: &[
                    (2, &[Write, Color, Blend]),
                    (6, &[Write, Color, Blend, Sparse]),
                    (7, &[Write, Color, Blend, Msaa, Sparse]),
                    (9, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Write, Color, Blend, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::PvrtcFormats => Cells {
                apple: &[(2, &[Filter]), (6, &[Filter, Sparse])],
                mac2: none, // Not available
            },
            // every Apple column lists capabilities
            FormatRow::EacEtcFormats => Cells {
                apple: &[(2, &[Filter]), (6, &[Filter, Sparse])],
                mac2: none, // Not available
            },
            // every Apple column lists capabilities
            FormatRow::AstcFormats => Cells {
                apple: &[(2, &[Filter]), (6, &[Filter, Sparse])],
                mac2: none, // Not available
            },
            // Apple2 to Apple5: Not available
            FormatRow::HdrAstcFormats => Cells {
                apple: &[(6, &[Filter, Sparse])],
                mac2: none, // Not available
            },
            // Apple2 to Apple6: Not available; Apple7 to Apple8: Varies
            FormatRow::BcFormats => Cells {
                apple: &[(9, &[Filter, Sparse])],
                mac2: &[Filter],
            },
            // every Apple column lists capabilities
            FormatRow::GBGR422 => Cells {
                apple: &[(2, &[Filter])],
                mac2: &[Filter],
            },
            // every Apple column lists capabilities
            FormatRow::BGRG422 => Cells {
                apple: &[(2, &[Filter])],
                mac2: &[Filter],
            },
            // every Apple column lists capabilities
            FormatRow::Depth16Unorm => Cells {
                apple: &[
                    (2, &[Filter, Msaa]),
                    (3, &[Filter, Msaa, Resolve]),
                    (8, &[Filter, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::Depth32Float => Cells {
                apple: &[
                    (2, &[Msaa]),
                    (3, &[Msaa, Resolve]),
                    (8, &[Msaa, Sparse, Resolve]),
                    (9, &[Filter, Msaa, Sparse, Resolve]),
                ],
                mac2: &[Filter, Msaa, Resolve],
            },
            // every Apple column lists capabilities
            FormatRow::Stencil8 => Cells {
                apple: &[
                    (2, &[Msaa]),
                    (3, &[Msaa, Resolve]),
                    (8, &[Msaa, Sparse, Resolve]),
                ],
                mac2: &[Msaa],
            },
            // Apple2 to Apple10: not in the 2026 edition
            FormatRow::Depth24UnormStencil8 => Cells {
                apple: &[],
                mac2: none, // Varies (footnote 5)
            },
            // every Apple column lists capabilities
            FormatRow::Depth32FloatStencil8 => Cells {
                apple: &[
                    (2, &[Msaa]),
                    (3, &[Msaa, Resolve]),
                    (9, &[Filter, Msaa, Resolve]),
                ],
                mac2: &[Filter, Msaa, Resolve],
            },
            // Apple2 to Apple10: not in the 2026 edition
            FormatRow::X24Stencil8 => Cells {
                apple: &[],
                mac2: none, // Varies (footnote 5)
            },
            // every Apple column lists capabilities
            FormatRow::X32Stencil8 => Cells {
                apple: &[(2, &[Msaa])],
                mac2: &[Msaa],
            },
            // Apple2: Not available
            FormatRow::BGRA10XR => Cells {
                apple: &[
                    (3, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // Apple2: Not available
            FormatRow::BGRA10XRSRGB => Cells {
                apple: &[
                    (3, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // Apple2: Not available
            FormatRow::BGR10XR => Cells {
                apple: &[
                    (3, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
            // Apple2: Not available
            FormatRow::BGR10XRSRGB => Cells {
                apple: &[
                    (3, &[Filter, Write, Color, Blend, Msaa, Resolve]),
                    (6, &[Filter, Write, Color, Blend, Msaa, Sparse, Resolve]),
                ],
                mac2: none, // Not available
            },
        }
    }
}

/// A format row's cells ([`FormatRow::cells`]).
struct Cells {
    /// From each Apple family on, until the next run, the column's capabilities.
    apple: &'static [(u8, &'static [FormatCapability])],
    mac2: &'static [FormatCapability],
}

impl Cells {
    /// The Apple`n` column's capabilities.
    fn apple(&self, n: u8) -> FormatCapabilities {
        self.apple
            .iter()
            .rev()
            .find(|&&(first, _)| first <= n)
            .map_or(FormatCapabilities::NONE, |&(_, caps)| {
                FormatCapabilities::of(caps)
            })
    }
}

impl FormatRow {
    /// The state of the row's `capability` under `facts`: exact where every format of the row has
    /// it, else the first format's refusal or gap.
    pub(super) fn state(self, capability: FormatCapability, facts: &Facts<'_>) -> RowState {
        self.formats()
            .iter()
            .find_map(|&pf| (facts.format)(pf).get(capability).err())
            .map_or(RowState::Exact, Missing::state)
    }
}

impl Device {
    /// The answers of `pf` on this device (the module docs), worked out once per format.
    pub(crate) fn format_answers(&self, pf: PixelFormat) -> FormatAnswers {
        // A format Metal describes none of answers nothing: Metal stops every texture of one (the
        // 24-bit depth formats among them, whatever the Vulkan driver has).
        if !crate::pixel_format_info::describes(pf) {
            return FormatAnswers::none();
        }
        let textures = &self.shared().textures;
        if let Some(answers) = lock(&textures.format_answers).get(&pf) {
            return answers.clone();
        }
        let facts = textures.facts(self.compile_device().profile());
        let limits = |plan: &ImagePlan| {
            let view_formats = plan.ash_view_formats();
            let mut list = vk::ImageFormatListCreateInfo::default();
            let info = plan.create_info(self.shared().sharing(), &view_formats, &mut list);
            self.image_limits(&info, &view_formats).unwrap_or_default()
        };
        let answers = answers(pf, &facts, &limits, self.sparse_answer(pf));
        lock(&textures.format_answers).insert(pf, answers.clone());
        answers
    }

    /// The Sparse answer of `pf` on this device (the module docs): a sparse heap's, where the
    /// device's functions read an image's own residency, then a 16 x 16 2D texture's.
    fn sparse_answer(&self, pf: PixelFormat) -> Result<(), Missing> {
        const CALL: &str = "pixel_format_capabilities";
        if self.reads_image_residency() {
            self.shared()
                .sparse
                .heaps()
                .map_err(|need| sparse_missing(sparse_refusal(need)))?;
        }
        let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        self.plan_sparse_texture(CALL, &descriptor, SparsePageSize::default())
            .map(drop)
            .map_err(sparse_missing)
    }
}

/// What a refusal of a sparse texture says the format lacks: the device's lack, the runtime's
/// gap, or, for a texture Metal's validation stops (a dual-plane format), no sparse texture of
/// the format at all. A lost device makes no texture to judge: the gap of sparse textures.
fn sparse_missing(error: Error) -> Missing {
    match error {
        Error::Unsupported(unsupported) => Missing::Refused(unsupported),
        Error::NotImplemented(gap) => Missing::Gap(gap),
        Error::Invalid(_) => Missing::Refused(Unsupported::not_representable(Feature::Format)),
        _ => Missing::Gap(RuntimeGap::SparseHeaps),
    }
}
