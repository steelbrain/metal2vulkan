//! Hand-written plain-data Vulkan structs (subsets of registry structs, no pointers).

use std::hash::{Hash, Hasher};

use super::{
    BorderColor, CompareOp, ComponentSwizzle, Filter, SamplerAddressMode, SamplerMipmapMode,
    SamplerReductionMode,
};

/// `VkSpecializationMapEntry`, with `size` narrowed to `u32` (entries are scalars of at most
/// 8 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpecializationMapEntry {
    pub constant_id: u32,
    pub offset: u32,
    pub size: u32,
}

/// A format's texel block ([`Format::block`](super::Format::block), generated from the registry's
/// `<formats>`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatBlock {
    /// Bytes of one texel block (`blockSize`).
    pub size: u32,
    /// Texels in one block (`texelsPerBlock`).
    pub texels: u32,
    /// The format's compatibility class (`class`, the spec's "Compatible Formats" table): an
    /// image view's format, and each format a mutable image lists, must share the image format's
    /// class (unless a block-texel or multi-planar view says otherwise). Formats of one block
    /// size need not share it: `A8_UNORM` is "8-bit alpha", `R8_UINT` "8-bit".
    pub class: &'static str,
    /// Bits of the packed word of a packed format (`packed`).
    pub packed_bits: Option<u32>,
    /// The bit width every component has, when they share one (none for compressed formats and
    /// mixed widths such as `D24_UNORM_S8_UINT`).
    pub component_bits: Option<u32>,
    /// The color components the format has (`R`, `G`, `B`, `A`, as the registry's `<component>`s
    /// name them), which a color write mask's bits name: a write whose mask names none of them
    /// stores nothing. Empty for a depth or stencil format.
    pub color_components: super::ColorComponentFlags,
    /// The numeric format of the color components (`R`, `G`, `B`, `A`), for a color format. The
    /// registry records an sRGB format's alpha as `UNORM`; the spec's `SRGB` numeric format covers
    /// that, so such a format is `Srgb`.
    pub color: Option<NumericFormat>,
    /// The numeric format of the depth component, for a format with depth.
    pub depth: Option<NumericFormat>,
    /// The numeric format of the stencil component, for a format with stencil.
    pub stencil: Option<NumericFormat>,
}

impl FormatBlock {
    /// The alignment in bytes Vulkan's fixed-function vertex input requires of an attribute's
    /// address in this format ("Vertex Input Address Calculation",
    /// `VUID-vkCmdDraw-format-10389`/`-10390` without `legacyVertexAttributes`): the whole packed
    /// word of a packed format, else one component. `None` when that is not a whole number of
    /// bytes (no vertex format).
    pub const fn vertex_alignment(self) -> Option<u32> {
        let bits = match (self.packed_bits, self.component_bits) {
            (Some(bits), _) | (None, Some(bits)) => bits,
            (None, None) => return None,
        };
        if bits % 8 == 0 && bits > 0 {
            Some(bits / 8)
        } else {
            None
        }
    }
}

/// Where the registry defines a format's enum value ([`Format::exposure`](super::Format::exposure)).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatExposure {
    /// The lowest core version that defines the value (`1.0` for the base formats), if any.
    pub core: Option<super::ApiVersion>,
    /// The registry names of every extension that defines the value or an alias of it (for a
    /// promoted value, the extension it was promoted from), sorted.
    pub extensions: &'static [&'static str],
}

impl FormatExposure {
    /// Whether a device of effective version `api` whose extensions include one named by
    /// `has_extension` exposes the format.
    pub fn exposed(&self, api: super::ApiVersion, has_extension: impl Fn(&str) -> bool) -> bool {
        self.core.is_some_and(|core| api >= core)
            || self.extensions.iter().any(|e| has_extension(e))
    }
}

/// How a format's components are interpreted: the registry's `numericFormat` (the Vulkan spec's
/// "Interpretation of Numeric Format" table, Vulkan-Docs `v1.4.341` `chapters/formats.adoc`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumericFormat {
    /// `UNORM`: unsigned normalized, `[0, 1]`.
    Unorm,
    /// `SNORM`: signed normalized, `[-1, 1]`.
    Snorm,
    /// `USCALED`: unsigned integers converted to floating point.
    Uscaled,
    /// `SSCALED`: signed integers converted to floating point.
    Sscaled,
    /// `UINT`: unsigned integers.
    Uint,
    /// `SINT`: signed integers.
    Sint,
    /// `UFLOAT`: unsigned floating point (packed, shared-exponent, and some compressed formats).
    Ufloat,
    /// `SFLOAT`: signed floating point.
    Sfloat,
    /// `SFIXED5`: signed fixed point with 5 fractional bits (`VK_NV_optical_flow`).
    Sfixed5,
    /// `SRGB`: sRGB-encoded unsigned normalized red, green, and blue; a plain `UNORM` alpha.
    Srgb,
    /// `BOOL`: booleans (`VK_ARM_tensors`).
    Bool,
}

/// The shader-side numeric type of a [`NumericFormat`] (the "Numeric type" column of the same
/// table): the class of `OpTypeImage`'s sampled type that reads or writes it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumericType {
    /// Floating point (`OpTypeFloat`).
    Float,
    /// Unsigned integer (`OpTypeInt`, unsigned).
    Uint,
    /// Signed integer (`OpTypeInt`, signed).
    Sint,
    /// Scaled signed integer (`OpTypeInt`; `SFIXED5`).
    ScaledSint,
    /// Boolean (`OpTypeBool`).
    Bool,
}

impl NumericFormat {
    /// The numeric type a shader reads this numeric format as.
    pub const fn numeric_type(self) -> NumericType {
        match self {
            NumericFormat::Unorm
            | NumericFormat::Snorm
            | NumericFormat::Uscaled
            | NumericFormat::Sscaled
            | NumericFormat::Ufloat
            | NumericFormat::Sfloat
            | NumericFormat::Srgb => NumericType::Float,
            NumericFormat::Uint => NumericType::Uint,
            NumericFormat::Sint => NumericType::Sint,
            NumericFormat::Sfixed5 => NumericType::ScaledSint,
            NumericFormat::Bool => NumericType::Bool,
        }
    }
}

impl super::Format {
    /// The numeric type a shader reads this format's texels as: its color components', or its
    /// only aspect's for a depth-only or stencil-only format. `None` for a combined
    /// depth/stencil format (a view selects one aspect, and each has its own numeric type),
    /// `UNDEFINED`, and values the pinned registry does not name. From the generated
    /// [`block`](super::Format::block), never from the format's name.
    pub const fn numeric_type(self) -> Option<NumericType> {
        let Some(block) = self.block() else {
            return None;
        };
        let numeric = match (block.color, block.depth, block.stencil) {
            (Some(n), None, None) | (None, Some(n), None) | (None, None, Some(n)) => n,
            _ => return None,
        };
        Some(numeric.numeric_type())
    }
}

/// `VkComponentMapping`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComponentMapping {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

impl ComponentMapping {
    /// Every component maps to itself.
    pub const IDENTITY: ComponentMapping = ComponentMapping {
        r: ComponentSwizzle::IDENTITY,
        g: ComponentSwizzle::IDENTITY,
        b: ComponentSwizzle::IDENTITY,
        a: ComponentSwizzle::IDENTITY,
    };
}

impl Default for ComponentMapping {
    fn default() -> ComponentMapping {
        ComponentMapping::IDENTITY
    }
}

/// The translation-relevant subset of `VkSamplerCreateInfo`.
///
/// Floating-point fields compare and hash by bit pattern, so a descriptor is a usable map key
/// (`-0.0` and `0.0` are distinct keys; both are valid Vulkan values).
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerDesc {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_mode: SamplerMipmapMode,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub mip_lod_bias: f32,
    /// `maxAnisotropy` when anisotropic filtering is enabled.
    pub max_anisotropy: Option<f32>,
    /// `compareOp` when depth comparison is enabled.
    pub compare: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: BorderColor,
    /// `VkSamplerReductionModeCreateInfo::reductionMode`.
    pub reduction: SamplerReductionMode,
    /// `VkSamplerBorderColorComponentMappingCreateInfoEXT::components` (`srgb` false), chained
    /// when set: the image view's component mapping, which an opaque-black border needs on a
    /// device with `borderColorSwizzle` and without `borderColorSwizzleFromImage` to read through
    /// a view of another mapping as any texel does. The sampler then reads only views of this
    /// mapping. Never set for a layout's immutable sampler.
    #[cfg_attr(feature = "serde", serde(default))]
    pub border_components: Option<ComponentMapping>,
    /// The device's sampler Y′CbCr conversion the sampler enables
    /// (`VkSamplerYcbcrConversionInfo`), a planes image's ([`SamplerDesc::ycbcr`],
    /// `ResourceKind::YCbCrPlanes`): the consumer makes the conversion of this (its format, the
    /// matrix's model, the range, midpoint chroma offsets and a linear chroma filter) and chains it,
    /// and views the image with the same conversion. Only a layout's immutable sampler sets it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub ycbcr: Option<crate::bind::YCbCrPlanesSampler>,
    /// `unnormalizedCoordinates`: the sampler reads a Metal pixel sampler's coordinate in texels
    /// (`metal::sampler::SamplerUse::unnormalized`). Vulkan's limits hold for such a sampler: equal
    /// filters, the nearest mip mode, the level clamp `[0, 0]`, clamp to the edge or a border on
    /// the `u` and `v` axes, no anisotropy and no comparison; the image views it reads have one
    /// level and one layer, of a 1D or 2D type.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unnormalized: bool,
}

impl SamplerDesc {
    /// The immutable sampler of a planes image's binding (`ResourceKind::YCbCrPlanes`): the
    /// device's conversion `ycbcr`, through a linear filter (a conversion's linear chroma filter
    /// asks it, `VUID-VkSamplerCreateInfo-minFilter-01645`), clamped to the edge on every axis, no
    /// anisotropy or comparison and the weighted average (`VUID-VkSamplerCreateInfo-addressModeU-
    /// 01646`, `VUID-VkSamplerCreateInfo-None-01647`), level 0 alone.
    pub fn ycbcr(ycbcr: crate::bind::YCbCrPlanesSampler) -> SamplerDesc {
        SamplerDesc {
            mag_filter: Filter::LINEAR,
            min_filter: Filter::LINEAR,
            mipmap_mode: SamplerMipmapMode::NEAREST,
            address_mode_u: SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: SamplerAddressMode::CLAMP_TO_EDGE,
            mip_lod_bias: 0.0,
            max_anisotropy: None,
            compare: None,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: BorderColor::FLOAT_TRANSPARENT_BLACK,
            reduction: SamplerReductionMode::WEIGHTED_AVERAGE,
            border_components: None,
            ycbcr: Some(ycbcr),
            unnormalized: false,
        }
    }
}

/// A [`SamplerDesc`] with floats as bit patterns, so equality is total.
#[derive(PartialEq, Eq, Hash)]
struct SamplerKey {
    filters: [Filter; 2],
    mipmap_mode: SamplerMipmapMode,
    address_modes: [SamplerAddressMode; 3],
    mip_lod_bias: u32,
    max_anisotropy: Option<u32>,
    compare: Option<CompareOp>,
    lods: [u32; 2],
    border_color: BorderColor,
    reduction: SamplerReductionMode,
    border_components: Option<ComponentMapping>,
    ycbcr: Option<crate::bind::YCbCrPlanesSampler>,
    unnormalized: bool,
}

impl SamplerDesc {
    fn key(&self) -> SamplerKey {
        SamplerKey {
            filters: [self.mag_filter, self.min_filter],
            mipmap_mode: self.mipmap_mode,
            address_modes: [
                self.address_mode_u,
                self.address_mode_v,
                self.address_mode_w,
            ],
            mip_lod_bias: self.mip_lod_bias.to_bits(),
            max_anisotropy: self.max_anisotropy.map(f32::to_bits),
            compare: self.compare,
            lods: [self.min_lod.to_bits(), self.max_lod.to_bits()],
            border_color: self.border_color,
            reduction: self.reduction,
            border_components: self.border_components,
            ycbcr: self.ycbcr,
            unnormalized: self.unnormalized,
        }
    }
}

impl PartialEq for SamplerDesc {
    fn eq(&self, other: &SamplerDesc) -> bool {
        self.key() == other.key()
    }
}

impl Eq for SamplerDesc {}

impl Hash for SamplerDesc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}
