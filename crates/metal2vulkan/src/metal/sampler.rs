//! Sampler mapping: `MTLSamplerDescriptor` facts → `vk::SamplerDesc`.
//!
//! The result always uses normalized coordinates; Metal pixel-coordinate samplers are handled in
//! the shader. How a sampler is **used** decides fields Metal keeps out of the descriptor: whether
//! depth comparison is enabled (a Dref sampling op, static in AIR), and, from the bound texture's
//! numeric type, whether the border color is an integer one and whether the texels are filtered at
//! all (an integer texture's never are). Whether an opaque-black border reads Metal's texel through
//! the texture view's component mapping is a property of the pair ([`check_border_swizzle`]).

use crate::bind::state::{SamplerDescriptor, SamplerShape};
use crate::device::enables::format_exposure_gap;
use crate::device::requirements::{FormatUsage, VkFeature, VkProperty};
use crate::device::DeviceProfile;
use crate::error::{Capability, DescriptorError, Error, Feature, Unsupported};
use crate::metal::format::{direct_format, FormatMapping};
use crate::metal::{
    CompareFunction, PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    SamplerMipFilter, TextureSwizzle, TextureSwizzleChannels,
};
use crate::vk;

/// The modes a texture is sampled with (the runtime's extension
/// `TextureDescriptor::address_mode_map`): for each `MTLSamplerAddressMode` a sampler states, the
/// mode used in its place. Metal has no such property; a consumer that presents a texture as
/// another format whose samplers address it otherwise states the measured rule here.
/// [`IDENTITY`](Self::IDENTITY), the default, maps every mode to itself. A runtime sampler takes it
/// at bind, and a constexpr sampler through the pipeline variant of the slot it samples
/// (`VariantKey::address_maps`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AddressModeMap {
    /// The mode used for each mode by its raw value, `ClampToEdge` (0) to `ClampToBorderColor` (5).
    to: [SamplerAddressMode; 6],
}

impl AddressModeMap {
    /// Every mode to itself.
    pub const IDENTITY: AddressModeMap = AddressModeMap {
        to: [
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::MirrorClampToEdge,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::MirrorRepeat,
            SamplerAddressMode::ClampToZero,
            SamplerAddressMode::ClampToBorderColor,
        ],
    };

    /// This map with `from` sampled as `to`. A mode past `MTLSamplerAddressMode`'s six keeps
    /// itself.
    pub fn with(mut self, from: SamplerAddressMode, to: SamplerAddressMode) -> AddressModeMap {
        if let Some(slot) = self.to.get_mut(from.as_raw() as usize) {
            *slot = to;
        }
        self
    }

    /// The mode a sampler stating `from` samples the texture with.
    pub fn mode(&self, from: SamplerAddressMode) -> SamplerAddressMode {
        self.to.get(from.as_raw() as usize).copied().unwrap_or(from)
    }

    /// Whether every mode maps to itself.
    pub fn is_identity(&self) -> bool {
        *self == AddressModeMap::IDENTITY
    }

    /// The mode a constexpr sampler addressing by `from` samples the texture with: the map's, where
    /// a constexpr sampler can state it (`ClampToEdge`, `Repeat`, `MirrorRepeat`, or `from`
    /// itself), and `None` where it cannot (`MirrorClampToEdge`, which MSL has no constexpr mode
    /// for, or a clamp to zero or to the border from another mode, whose border the state's one
    /// border color decides for every axis).
    pub fn constexpr_mode(&self, from: SamplerAddressMode) -> Option<SamplerAddressMode> {
        let to = self.mode(from);
        let stated = matches!(
            to,
            SamplerAddressMode::ClampToEdge
                | SamplerAddressMode::Repeat
                | SamplerAddressMode::MirrorRepeat
        );
        (to == from || stated).then_some(to)
    }

    /// `desc` with its three address modes mapped.
    pub fn sampler(&self, mut desc: SamplerDescriptor) -> SamplerDescriptor {
        desc.s_address_mode = self.mode(desc.s_address_mode);
        desc.t_address_mode = self.mode(desc.t_address_mode);
        desc.r_address_mode = self.mode(desc.r_address_mode);
        desc
    }
}

impl Default for AddressModeMap {
    fn default() -> AddressModeMap {
        AddressModeMap::IDENTITY
    }
}

/// How a sampler is used with a texture.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SamplerUse {
    /// The shader samples with depth comparison (`sample_compare`, `gather_compare`).
    pub compare: bool,
    /// The sampled texture has an integer format: border colors are then `INT_*`, and every
    /// filter is nearest.
    pub integer_texture: bool,
    /// The shader hands a pixel sampler's coordinate in texels to the device's unnormalized
    /// sampler (`VkSamplerCreateInfo::unnormalizedCoordinates`), at level 0 of a view of one level
    /// and one layer, without an offset or a comparison: the sampler's state is one of pixel
    /// coordinates, one minification and magnification filter, no mip filter or anisotropy, and
    /// `s` and `t` clamped to the edge, to zero or to the border.
    pub unnormalized: bool,
}

impl SamplerUse {
    /// A use with the given comparison and texture numeric type, in normalized coordinates.
    pub fn new(compare: bool, integer_texture: bool) -> SamplerUse {
        SamplerUse {
            compare,
            integer_texture,
            unnormalized: false,
        }
    }

    /// This use, reading texel coordinates through the device's unnormalized sampler where
    /// `unnormalized` holds.
    pub fn with_unnormalized(self, unnormalized: bool) -> SamplerUse {
        SamplerUse {
            unnormalized,
            ..self
        }
    }
}

/// Whether the device's unnormalized sampler (`VkSamplerCreateInfo::unnormalizedCoordinates`)
/// reads a sampler of `desc` in pixel coordinates as it is: pixel coordinates, one minification
/// and magnification filter (nearest or linear), no mip filter and no anisotropy, and on the `s`
/// and `t` axes clamping to the edge, to zero or to its border, which the device's
/// `CLAMP_TO_EDGE` and `CLAMP_TO_BORDER` are. These are Vulkan's limits on such a sampler
/// (`VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01072` to `-01076`; a comparison,
/// `-01077`, is the use's), and the Metal Shading Language's rule for a constexpr `coord::pixel`
/// sampler.
pub(crate) fn unnormalized_reads(desc: &SamplerDescriptor) -> bool {
    let clamps = |m: SamplerAddressMode| {
        matches!(
            m,
            SamplerAddressMode::ClampToEdge
                | SamplerAddressMode::ClampToZero
                | SamplerAddressMode::ClampToBorderColor
        )
    };
    !desc.normalized_coordinates
        && desc.min_filter == desc.mag_filter
        && desc.min_filter != SamplerMinMagFilter::Cubic
        && desc.mip_filter == SamplerMipFilter::NotMipmapped
        && desc.max_anisotropy <= 1
        && clamps(desc.s_address_mode)
        && clamps(desc.t_address_mode)
}

/// The `(minLod, maxLod)` of a sampler without a mip filter whose descriptor clamps the level to
/// `[lo, hi]`: the base level always (`maxLod` at most [`NOT_MIPMAPPED_MAX_LOD`]), and `λ > 0`, the
/// minification filter, exactly where Metal's level raised to `lo` and then lowered to `hi` is
/// above 0. Metal filters such a sample with its min filter where that level is above 0 and with
/// its mag filter elsewhere, whatever its texture's type or coordinates, so a sampler whose two
/// filters differ reads its min filter at level 0 where `lo` is above 0 (an Apple M3 Max on macOS
/// 26.7 and an M4 on macOS 27.0.1, byte-identical; 1D, 2D, 2D array, 3D and cube textures,
/// normalized and pixel coordinates, a kernel's sample at its default level, `level(0)`, and levels
/// and biases 0, 0.5, 1 and 2, `lodMinClamp` 0 and 1). So `minLod` is `lo` clamped into
/// `[0, maxLod]` where `hi` is above 0, and both are 0 where it is not (a descriptor's clamps are
/// ordered and not NaN, [`check_sampler`]).
pub(crate) fn not_mipmapped_lods(lo: f32, hi: f32) -> (f32, f32) {
    match hi > 0.0 {
        true => (lo.clamp(0.0, NOT_MIPMAPPED_MAX_LOD), NOT_MIPMAPPED_MAX_LOD),
        false => (0.0, 0.0),
    }
}

/// `maxLod` of a sampler without mipmapping: `NEAREST` mip selection with `maxLod = 0.25` always
/// selects the base level while still distinguishing magnification (`λ ≤ 0`) from minification,
/// as Metal's `NotMipmapped` does (the Vulkan spec's recommended emulation of non-mipmapped
/// sampling).
pub const NOT_MIPMAPPED_MAX_LOD: f32 = 0.25;

/// The last level a word's level field names: past every texture's last level (Metal's largest
/// textures have 15 levels, the last level 14), so a bound at or past it clamps nothing.
pub(crate) const LAST_WORD_LEVEL: u32 = 15;

/// Metal's level clamp bound `c` on the sixty-fourths grid of its level and weight: its value at or
/// above 0, truncated onto sixty-fourths. Metal clamps a sample's level into the sampler's clamp
/// before truncating it, so a level clamped by bounds on the grid truncates to the level clamped by
/// the bounds themselves (truncation is monotone). A bound of 64 or more is past every texture's
/// levels and stays as it is. A runtime sampler's bounds are `f32`s.
pub(crate) fn level_on_grid(c: f32) -> f32 {
    let c = c.max(0.0);
    if c >= 64.0 {
        return c;
    }
    (c * 64.0).floor() / 64.0
}

/// The level a nearest mip filter reads at level `c` clamped as Metal clamps it: with `L` its whole
/// levels and `q` its fraction truncated onto sixty-fourths, `L + 1` where `q > 1/2`, else `L`; at
/// most [`LAST_WORD_LEVEL`]. As a function of the clamp bound it is the bound's own level: Metal's
/// level clamped by the bounds is the level clamped by the bounds' levels (the rule is monotone).
pub(crate) fn nearest_level(c: f32) -> u32 {
    let c = level_on_grid(c);
    if c >= LAST_WORD_LEVEL as f32 {
        return LAST_WORD_LEVEL;
    }
    let whole = c.floor();
    let up = c - whole > 0.5;
    (whole as u32 + u32::from(up)).min(LAST_WORD_LEVEL)
}

/// The level, in sixty-fourths of a level, that Metal's `calculate_clamped_lod` answers through
/// `desc` outside a fragment function before the texture's last level bounds it: a kernel's or
/// vertex function's level of detail is 0, clamped by the sampler as a sample's level is. So it is
/// 0 without a mip filter, the level a nearest mip filter reads at `lodMinClamp`
/// ([`nearest_level`]), and `lodMinClamp` on Metal's grid ([`level_on_grid`]) under a linear one,
/// whatever the min and mag filters and the anisotropy; `lodMaxClamp`, never below `lodMinClamp`,
/// bounds nothing at 0. At most [`LAST_WORD_LEVEL`] levels, past every texture's last level.
/// Measured on the M3 Max and an M4.
pub(crate) fn query_min_level(desc: &SamplerDescriptor) -> u32 {
    const CAP: u32 = LAST_WORD_LEVEL * 64;
    match desc.mip_filter {
        SamplerMipFilter::NotMipmapped => 0,
        SamplerMipFilter::Nearest => nearest_level(desc.lod_min_clamp) * 64,
        SamplerMipFilter::Linear => {
            let c = level_on_grid(desc.lod_min_clamp);
            if c >= LAST_WORD_LEVEL as f32 {
                CAP
            } else {
                // Exact: `c` is a multiple of 1/64 below 15.
                (c * 64.0) as u32
            }
        }
    }
}

/// Whether a sampler of `shape` (the descriptor's facts, [`SamplerDescriptor::shape`]) used as
/// `use_` can be a Vulkan sampler on `profile`. The binder's plan checks the bound shape with this
/// at plan time for the binding's use; [`vk_sampler`] checks its descriptor with it, so the two
/// never disagree.
///
/// Refusals, in this order: `MirrorClampToEdge` without `samplerMirrorClampToEdge`; comparison on
/// a portability-subset device without `mutableComparisonSamplers`. NaN or inverted LOD clamps are
/// invalid descriptors. A min or max reduction is no refusal: the sampler reads the weighted
/// average ([`vk_sampler`]). Nor is anisotropy above 1 without `samplerAnisotropy`: the Vulkan
/// sampler is isotropic there, and a function samples through it with Metal's anisotropic filter
/// emulated where the emulation reaches the sample (`PrecisionNote::AnisotropicFilter`) and the
/// device's isotropic filter, recorded, elsewhere (`PrecisionNote::DeviceAnisotropicFilter`). An
/// argument buffer's anisotropic sampler, whose anisotropy the function does not know, is refused
/// there where the binding plan checks it (`Unsupported`, `SamplerAnisotropy`).
pub fn check_sampler(
    shape: &SamplerShape,
    use_: SamplerUse,
    profile: &DeviceProfile,
) -> Result<(), Error> {
    let compare = use_.compare;
    let images = &profile.fields().images;
    if shape.mirror_clamp_to_edge && !images.sampler_mirror_clamp_to_edge {
        return Err(Unsupported::needs(
            Feature::SamplerAddressMode,
            Capability::Feature(VkFeature::SamplerMirrorClampToEdge),
        )
        .into());
    }
    let mutable = VkFeature::MutableComparisonSamplers;
    if compare && profile.fields().enabled.portability_withholds(mutable) {
        return Err(
            Unsupported::needs(Feature::SamplerCompare, Capability::Feature(mutable)).into(),
        );
    }
    if !shape.valid_lod_clamps {
        return Err(Error::InvalidDescriptor(DescriptorError::SamplerLodClamp));
    }
    Ok(())
}

/// [`check_sampler`] for a sampler an argument buffer lists, which a function reads without
/// knowing its state: anisotropy above 1 without `samplerAnisotropy` is refused too, since such a
/// function neither emulates Metal's anisotropic filter through it nor records the isotropic
/// filter its Vulkan sampler takes there (`lower::sampler::anisotropic`'s reach).
pub(crate) fn check_argument_sampler(
    shape: &SamplerShape,
    use_: SamplerUse,
    profile: &DeviceProfile,
) -> Result<(), Error> {
    check_sampler(shape, use_, profile)?;
    if shape.anisotropic && !profile.fields().images.sampler_anisotropy {
        return Err(Unsupported::needs(
            Feature::SamplerAnisotropy,
            Capability::Feature(VkFeature::SamplerAnisotropy),
        )
        .into());
    }
    Ok(())
}

/// What of the Vulkan sampler [`vk_sampler`] makes for a sampler of `shape` used as `use_` decides
/// the format features it needs of a view it reads (`bind::sampler_desc`): whether some filter is
/// linear (none for an integer use, which [`vk_sampler`] makes nearest), and its reduction, the
/// weighted average ([`vk_sampler`]). A plan reads it where a runtime sampler meets its textures
/// through separate descriptors, so the textures' views are checked against the bound sampler at
/// plan time as a combined write's view is per bind.
pub(crate) fn vk_filtering(
    shape: &SamplerShape,
    use_: SamplerUse,
) -> (bool, vk::SamplerReductionMode) {
    (
        shape.linear && !use_.integer_texture,
        vk::SamplerReductionMode::WEIGHTED_AVERAGE,
    )
}

/// The Vulkan sampler for a Metal sampler descriptor used as `use_` on `profile`.
///
/// An integer texture is never filtered: Metal reads a linear filter on one as nearest (an Apple M3
/// Max returns the nearest texel of a `ushort` texture through a linear, magnifying pixel sampler,
/// authored case `subsample_quarter_luma`), and Vulkan allows a linear filter or mip mode only on a
/// format with `SAMPLED_IMAGE_FILTER_LINEAR`, which no integer format has
/// (VUID-vkCmdDispatch-magFilter-04553, VUID-vkCmdDispatch-mipmapMode-04770). So an integer use
/// takes `NEAREST` for both filters and the mip mode, and no anisotropy, whatever the descriptor
/// says.
///
/// Its reduction is the weighted average whatever `reductionMode` says: Metal reads a minimum or
/// maximum reduction so on an Apple GPU whose `supportsSamplerReductionMode` answers no (an Apple
/// M4 on macOS 27.0.1, without its validation layer, which refuses such a sampler: every channel
/// of `rgba8Unorm`, `rgba16Float` and `r32Float` linear samples through runtime samplers), and the
/// runtime answers that query no on every device
/// (`DeviceProperties::supports_sampler_reduction_mode`).
///
/// Without a mip filter it reads the base level, and its clamp keeps Metal's choice between the
/// two filters (`not_mipmapped_lods`); through Metal's own sampler with anisotropy and equal
/// filters its clamp is `[0, 0]`, where that sampler reads Metal's `NotMipmapped`. With filters
/// that differ there, no clamp reads Metal's answer (`docs/FAST-PATHS.md`, known gaps).
///
/// Its level clamp is the descriptor's, its lower bound at least 0: the device clamps its own level
/// into it and selects and blends levels at its own precision (`PrecisionNote::DeviceSampling` off
/// Metal's own sampler). Metal truncates the clamped level's fraction onto sixty-fourths and reads
/// a nearest mip filter's next level past one half; MoltenVK hands the clamp to Metal's sampler,
/// which does so itself.
///
/// A use that reads texel coordinates ([`SamplerUse::unnormalized`]) takes the device's
/// unnormalized sampler: its filters, address modes and border as above, the nearest mip mode at
/// the level clamp `[0, 0]` (which Vulkan requires; a pixel sampler reads level 0 by one filter),
/// and no anisotropy or comparison.
///
/// Refusals are [`check_sampler`]'s, for the descriptor's shape and the use; a use that reads
/// texel coordinates through any other sampler state than [`SamplerUse::unnormalized`] names, or
/// with a comparison, is an internal error.
pub fn vk_sampler(
    desc: &SamplerDescriptor,
    use_: SamplerUse,
    profile: &DeviceProfile,
) -> Result<vk::SamplerDesc, Error> {
    check_sampler(&desc.shape(), use_, profile)?;
    if use_.unnormalized && (use_.compare || !unnormalized_reads(desc)) {
        return Err(Error::Internal(crate::error::InternalError::new(
            crate::error::InternalKind::Invariant,
            "an unnormalized sampler of a state the device's cannot read so",
        )));
    }
    let limits = &profile.fields().limits;
    let modes = [
        desc.s_address_mode,
        desc.t_address_mode,
        desc.r_address_mode,
    ];
    let address = modes.map(|mode| match mode {
        SamplerAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        SamplerAddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        SamplerAddressMode::MirrorClampToEdge => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        SamplerAddressMode::ClampToZero | SamplerAddressMode::ClampToBorderColor => {
            vk::SamplerAddressMode::CLAMP_TO_BORDER
        }
    });
    let border_color = border_color(
        border_used(desc).unwrap_or(SamplerBorderColor::TransparentBlack),
        use_.integer_texture,
    );

    // Metal filters a `maxAnisotropy` rounded up to a power of two (3 as 4, 5 to 7 as 8, 9 to 15
    // as 16, a runtime and a constexpr sampler alike), so the device is handed that degree where
    // it is handed anisotropy at all (`device_anisotropy`).
    let max_anisotropy =
        (device_anisotropy(desc, use_, profile) && !use_.unnormalized).then(|| {
            let limit = f32::from_bits(limits.max_sampler_anisotropy_bits).max(1.0);
            (desc.max_anisotropy.next_power_of_two() as f32).min(limit)
        });

    let (lo, hi) = (desc.lod_min_clamp, desc.lod_max_clamp);
    let linear_mip = if use_.integer_texture {
        vk::SamplerMipmapMode::NEAREST
    } else {
        vk::SamplerMipmapMode::LINEAR
    };
    // Metal's own sampler (MoltenVK) handed anisotropy reads a sampler without a mip filter whose
    // filters agree as Metal's `NotMipmapped` at the clamp `[0, 0]`, whatever the descriptor's
    // clamps, and otherwise at none it takes: a bound of a sixty-fourth or more (`0.25` included)
    // changes Metal's anisotropic footprint (an Apple M3 Max on macOS 27.0.1, the anisotropic
    // lane's 1268 footprints at `maxAnisotropy` 2, 8 and 16, gradients scaled by 1 to 1/32,
    // linear and nearest filters, clamps from `[0, 0.25]` to `[3, 8]`: every channel Metal's).
    // Equal filters leave the device's level no filter to choose.
    let own_anisotropic = max_anisotropy.is_some()
        && desc.min_filter == desc.mag_filter
        && profile
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler);
    let (mipmap_mode, min_lod, max_lod) = match desc.mip_filter {
        _ if use_.unnormalized => (vk::SamplerMipmapMode::NEAREST, 0.0, 0.0),
        SamplerMipFilter::NotMipmapped if own_anisotropic => {
            (vk::SamplerMipmapMode::NEAREST, 0.0, 0.0)
        }
        SamplerMipFilter::NotMipmapped => {
            let (min_lod, max_lod) = not_mipmapped_lods(lo, hi);
            (vk::SamplerMipmapMode::NEAREST, min_lod, max_lod)
        }
        SamplerMipFilter::Nearest => (vk::SamplerMipmapMode::NEAREST, lo.max(0.0), hi),
        SamplerMipFilter::Linear => (linear_mip, lo.max(0.0), hi),
    };
    let filter = |f| {
        if use_.integer_texture {
            vk::Filter::NEAREST
        } else {
            filter(f)
        }
    };

    Ok(vk::SamplerDesc {
        mag_filter: filter(desc.mag_filter),
        min_filter: filter(desc.min_filter),
        mipmap_mode,
        address_mode_u: address[0],
        address_mode_v: address[1],
        address_mode_w: address[2],
        mip_lod_bias: 0.0,
        max_anisotropy,
        compare: use_.compare.then(|| compare_op(desc.compare_function)),
        min_lod,
        max_lod: max_lod.max(min_lod),
        border_color,
        reduction: vk::SamplerReductionMode::WEIGHTED_AVERAGE,
        border_components: None,
        ycbcr: None,
        unnormalized: use_.unnormalized,
    })
}

/// Whether [`vk_sampler`]'s sampler for `desc` under `use_` on `profile` filters anisotropically
/// (`anisotropyEnable`, its `max_anisotropy` set).
///
/// Anisotropy is a quality bound in both APIs; Vulkan requires it within the limit. It is a
/// filter, so an integer texture has none. A sampler with a nearest min or mag filter is handed
/// its degree too: Vulkan leaves the anisotropic scheme to the implementation (ANV's filters a
/// nearest one linearly). A sampler on a device without `samplerAnisotropy` is handed none, where
/// the emulation's taps read the isotropic sampler
/// (`VUID-VkSamplerCreateInfo-anisotropyEnable-01070`: "If the samplerAnisotropy feature is not
/// enabled, anisotropyEnable must be VK_FALSE").
pub(crate) fn device_anisotropy(
    desc: &SamplerDescriptor,
    use_: SamplerUse,
    profile: &DeviceProfile,
) -> bool {
    desc.max_anisotropy > 1 && profile.fields().images.sampler_anisotropy && !use_.integer_texture
}

/// [`device_anisotropy`] of an anisotropic sampler under `use_` on `profile`, by the filters a
/// sampler word states (`SamplerWord::MIN_LINEAR`, `SamplerWord::MAG_LINEAR`): `(linear, nearest)`,
/// its value where both filters are linear and where one is nearest. A word with a cubic filter
/// (`SamplerWord::CUBIC`) states neither: its other filter may be nearest or not.
pub(crate) fn device_anisotropy_by_filters(
    use_: SamplerUse,
    profile: &DeviceProfile,
) -> (bool, bool) {
    let of = |min_filter, mag_filter| {
        let desc = SamplerDescriptor {
            min_filter,
            mag_filter,
            max_anisotropy: 2,
            ..SamplerDescriptor::new()
        };
        device_anisotropy(&desc, use_, profile)
    };
    (
        of(SamplerMinMagFilter::Linear, SamplerMinMagFilter::Linear),
        of(SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear),
    )
}

/// The one border color a sampler replaces texels with, if any axis clamps to a border: the
/// descriptor's, under `ClampToZero` as under `ClampToBorderColor`. Metal's `ClampToZero` reads
/// the sampler's border color, not transparent black (an Apple M3 Max and an Apple M4, every probed
/// format, border color, and filter, samples, gathers, and depth comparisons alike; AIR codes both
/// modes as one).
fn border_used(desc: &SamplerDescriptor) -> Option<SamplerBorderColor> {
    [
        desc.s_address_mode,
        desc.t_address_mode,
        desc.r_address_mode,
    ]
    .into_iter()
    .any(|mode| {
        matches!(
            mode,
            SamplerAddressMode::ClampToZero | SamplerAddressMode::ClampToBorderColor
        )
    })
    .then_some(desc.border_color)
}

/// Whether a sampler of `desc` reading a texture of `pf` made with `swizzle` through its
/// `mapping` (the [`metal::format::vk_format`](crate::metal::format::vk_format) result the image
/// view is created with) returns Metal's border texels on `profile`: see [`check_border_view`], for
/// whether the border the descriptor replaces texels with is opaque black.
pub fn check_border_swizzle(
    desc: &SamplerDescriptor,
    pf: PixelFormat,
    mapping: &FormatMapping,
    swizzle: TextureSwizzleChannels,
    profile: &DeviceProfile,
) -> Result<(), Error> {
    check_border_view(
        border_used(desc) == Some(SamplerBorderColor::OpaqueBlack),
        pf,
        mapping,
        swizzle,
        profile,
    )
}

/// Whether a sampler whose replaced border is opaque black (`opaque_black`; the plan knows it
/// from the shape, `SamplerShape::opaque_black_border`), reading a texture of `pf` made with
/// `swizzle` through its `mapping`, returns Metal's border texel on `profile`.
///
/// Metal's texel is the color in the format's channels (`A8Unorm`'s alpha is the color's red), a
/// missing red, green or blue 0 and a missing alpha 1, then the swizzle (`border_texel`). Vulkan
/// replaces a border texel in the view format's channels, opaque black's alpha 1 in every format,
/// `A8_UNORM`'s included, and the view's mapping (the swizzle composed on `mapping`'s,
/// [`TextureSwizzleChannels::compose`]) applies to it as to any texel; a device that samples
/// through Metal's own sampler (`VkProperty::MetalSampler`, MoltenVK) replaces it by Metal's
/// rule. Transparent-black and opaque-white borders are defined through any mapping and read
/// Metal's texel. An **opaque-black** border is exact only when:
///
/// - the device's texel through the view is Metal's. `A8Unorm` through `A8_UNORM` reads alpha 1
///   where Metal reads 0 (measured on RADV, ANV, NVIDIA and lavapipe); there Metal's opaque-black
///   texel is its transparent-black one, which the binder makes the sampler with instead
///   ([`opaque_black_reads_transparent`]). A sampler it cannot change (a layout's immutable one)
///   refuses;
/// - read through a view of `R4G4B4A4_UNORM_PACK16` or `B4G4R4A4_UNORM_PACK16` (`ABGR4Unorm`'s
///   candidates), the device reports `rgba4OpaqueBlackSwizzled` (`VK_KHR_maintenance10`):
///   otherwise an implementation may swap alpha with red or blue for the built-in border colors,
///   as Apple GPUs do through their channel-reordered hardware format (Vulkan spec "Texel
///   Replacement"). A custom border color would not swap; the binder creates none (handed back);
/// - read through a mapping other than the identity, the device samples through Metal's sampler
///   (measured exact through six swizzles), or `borderColorSwizzle` and
///   `borderColorSwizzleFromImage` are enabled (the binder chains no
///   `VkSamplerBorderColorComponentMappingCreateInfoEXT`, whose mapping would have to match every
///   view the sampler meets). Without them Vulkan leaves the texel undefined: NVIDIA, which lacks
///   `borderColorSwizzleFromImage`, reads `(0, 0, 0, 1)` through such views.
///
/// Errors: `Unsupported{SamplerBorderColor}` needing what is missing of the two features and the
/// property; what the direct format lacks (the version or extension exposing it, then
/// sampled-image support) where only its view reads Metal's texel; `NotImplemented` where none
/// does.
pub fn check_border_view(
    opaque_black: bool,
    pf: PixelFormat,
    mapping: &FormatMapping,
    swizzle: TextureSwizzleChannels,
    profile: &DeviceProfile,
) -> Result<(), Error> {
    if !opaque_black {
        return Ok(());
    }
    border_read(pf, mapping, swizzle, profile, false).map(|_| ())
}

/// How a runtime sampler whose border is opaque black is made to read Metal's border texel
/// through a texture's view: the binder makes a sampler per use, and per view mapping where it
/// must.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OpaqueBlackBorder {
    /// As the descriptor states: the device reads Metal's texel through the view (or the
    /// sampler's border is not opaque black).
    Stated,
    /// Transparent black, whose texel Metal reads for opaque black through the texture
    /// ([`opaque_black_reads_transparent`]), and which every device reads exactly.
    Transparent,
    /// Opaque black with `VkSamplerBorderColorComponentMappingCreateInfoEXT` of this mapping, the
    /// view's (`vk::SamplerDesc::border_components`): a device with `borderColorSwizzle` and
    /// without `borderColorSwizzleFromImage` then reads the border through the view's mapping
    /// as any texel. Such a sampler reads only views of this mapping.
    Mapped(vk::ComponentMapping),
}

/// How a runtime sampler that keeps its opaque-black border reads Metal's texel through the view
/// of a texture of `pf` made with `swizzle`, through its `mapping`, on `profile`: as
/// [`check_border_view`] admits it ([`OpaqueBlackBorder::Stated`]), but with `borderColorSwizzle`
/// alone through a mapping other than the identity ([`OpaqueBlackBorder::Mapped`]), which a
/// sampler made for the view can carry. Errors as [`check_border_view`]'s, needing
/// `borderColorSwizzle` alone where it is what is missing. Where
/// [`opaque_black_reads_transparent`] holds, a sampler made with transparent black
/// ([`OpaqueBlackBorder::Transparent`]) reads Metal's texel on every device instead.
pub fn opaque_black_border(
    pf: PixelFormat,
    mapping: &FormatMapping,
    swizzle: TextureSwizzleChannels,
    profile: &DeviceProfile,
) -> Result<OpaqueBlackBorder, Error> {
    border_read(pf, mapping, swizzle, profile, true)
}

/// The opaque-black read of [`check_border_view`] (`mapped` false: a sampler the binder cannot
/// make per view, which needs `borderColorSwizzleFromImage` through a non-identity mapping) and
/// of [`opaque_black_border`] (`mapped` true: one it makes per view mapping).
fn border_read(
    pf: PixelFormat,
    mapping: &FormatMapping,
    swizzle: TextureSwizzleChannels,
    profile: &DeviceProfile,
    mapped: bool,
) -> Result<OpaqueBlackBorder, Error> {
    let f = profile.fields();
    let metal_sampler = f.enabled.properties.contains(&VkProperty::MetalSampler);
    let swapped = matches!(
        mapping.format,
        vk::Format::R4G4B4A4_UNORM_PACK16 | vk::Format::B4G4R4A4_UNORM_PACK16
    ) && !f
        .enabled
        .properties
        .contains(&VkProperty::Rgba4OpaqueBlackSwizzled);
    let rgba4 = swapped.then_some(Capability::Property(VkProperty::Rgba4OpaqueBlackSwizzled));
    let metal = black_texel(pf, swizzle, BORDER_COLORS[1]);
    let view = swizzle.compose(mapping.components);
    if device_opaque_black(mapping.format, view, metal_sampler) != metal {
        let unswizzled = swizzle.compose(vk::ComponentMapping::IDENTITY);
        let Some(direct) = direct_format(pf).filter(|&d| {
            d != mapping.format && device_opaque_black(d, unswizzled, metal_sampler) == metal
        }) else {
            return Err(Unsupported::not_implemented(Feature::SamplerBorderColor).into());
        };
        let need = format_exposure_gap(direct, f.api_version, &f.enabled.extensions)
            .into_iter()
            .chain([Capability::Format {
                format: direct,
                usage: FormatUsage::SampledImage,
            }])
            .collect();
        return Err(Unsupported::needs_any_of(Feature::SamplerBorderColor, vec![need]).into());
    }
    let has = |feature: &VkFeature| f.enabled.features.contains(feature);
    // Through another mapping Vulkan defines the texel only with the border-swizzle features:
    // the view's own (`FromImage`), or the sampler's (`Mapped`).
    let read = if is_identity(view)
        || metal_sampler
        || (has(&VkFeature::BorderColorSwizzle) && has(&VkFeature::BorderColorSwizzleFromImage))
    {
        OpaqueBlackBorder::Stated
    } else if mapped && has(&VkFeature::BorderColorSwizzle) {
        OpaqueBlackBorder::Mapped(view)
    } else {
        let needed: &[VkFeature] = if mapped {
            &[VkFeature::BorderColorSwizzle]
        } else {
            &[
                VkFeature::BorderColorSwizzle,
                VkFeature::BorderColorSwizzleFromImage,
            ]
        };
        let missing: Vec<Capability> = needed
            .iter()
            .filter(|feature| !has(feature))
            .copied()
            .map(Capability::Feature)
            .chain(rgba4)
            .collect();
        return Err(Unsupported::needs_any_of(Feature::SamplerBorderColor, vec![missing]).into());
    };
    match rgba4 {
        Some(need) => {
            Err(Unsupported::needs_any_of(Feature::SamplerBorderColor, vec![vec![need]]).into())
        }
        None => Ok(read),
    }
}

/// Whether Metal reads the same border texel of a texture of `pf` made with `swizzle` for an
/// opaque-black border as for a transparent-black one, so a sampler made with transparent black
/// reads Metal's opaque-black texel exactly on every device (Vulkan defines that border through any
/// mapping, and it reads Metal's texel). The two colors differ only in alpha, so they read alike
/// unless the format has an alpha channel of its own (not `A8Unorm`'s, which is the color's red, 0
/// in both) and the swizzle reads it.
pub fn opaque_black_reads_transparent(pf: PixelFormat, swizzle: TextureSwizzleChannels) -> bool {
    black_texel(pf, swizzle, BORDER_COLORS[1]) == black_texel(pf, swizzle, BORDER_COLORS[0])
}

/// Metal's border texel (`border_texel`) for `color`, transparent or opaque black, of a texture of
/// `pf` made with `swizzle`. Red, green and blue read 0 whether the format has them or not, so
/// only whether it has alpha matters (`out_of_bounds_alpha_one`, the same channel facts).
fn black_texel(pf: PixelFormat, swizzle: TextureSwizzleChannels, color: [bool; 4]) -> [bool; 4] {
    let alpha = crate::metal::format::out_of_bounds_alpha_one(pf) == Some(false);
    border_texel(pf, [true, true, true, alpha], swizzle, color)
}

/// The device's opaque-black border texel through a view of `format` with mapping `view`: Vulkan
/// replaces the texel in `format`'s channels with `(0, 0, 0, 1)`, a missing alpha 1 and
/// `A8_UNORM`'s alpha 1 too (measured); a device sampling through Metal's sampler (`metal_sampler`)
/// reads `A8_UNORM`'s alpha from the color's red, 0. Then the mapping.
pub(crate) fn device_opaque_black(
    format: vk::Format,
    view: vk::ComponentMapping,
    metal_sampler: bool,
) -> [bool; 4] {
    let alpha = !(metal_sampler && format == vk::Format::A8_UNORM);
    through(view, [false, false, false, alpha])
}

/// `texel` (`[r, g, b, a]`, each 0 or 1) read through mapping `m`.
fn through(m: vk::ComponentMapping, texel: [bool; 4]) -> [bool; 4] {
    use vk::ComponentSwizzle as S;
    let read = |c: vk::ComponentSwizzle, own: usize| {
        if c == S::IDENTITY {
            texel[own]
        } else if c == S::ONE {
            true
        } else {
            [S::R, S::G, S::B, S::A]
                .iter()
                .position(|&channel| channel == c)
                .is_some_and(|k| texel[k])
        }
    };
    [read(m.r, 0), read(m.g, 1), read(m.b, 2), read(m.a, 3)]
}

/// The built-in border colors as `[r, g, b, a]`, each channel 0 or 1, in the order of their bits
/// in a format's border texels (`metal::bicubic::BicubicTexels::BORDER`): transparent black,
/// opaque black, opaque white.
pub(crate) const BORDER_COLORS: [[bool; 4]; 3] =
    [[false; 4], [false, false, false, true], [true; 4]];

/// Metal's texel at the border of a texture of `pf` whose channels are `present` (red, green, blue,
/// alpha), made with `swizzle`, for the border color `color` (`[r, g, b, a]`, each 0 or 1): each
/// channel the format has is the color's, but `A8Unorm`'s alpha, which is the color's red, and a
/// format held rotated's (`ARGB8Unorm`,
/// [`rotated_channels`](crate::metal::format::rotated_channels)), whose bytes hold the color and
/// whose channels read it through the rotation: red the color's green, green its blue, blue its
/// alpha and alpha its red (opaque black reads blue 1 and alpha 0; nearest and linear); each it
/// lacks is 0 (red, green, blue) or 1 (alpha); then the swizzle, as for any texel. Measured on an
/// Apple M4 (eleven formats among them `A8Unorm`, `R8Unorm`, `RG8Unorm`, `BGRA8Unorm`, `R16Float`
/// and the integer `R32Uint`, six swizzles, the three colors and clamp-to-zero, constexpr and
/// runtime samplers, 1056 border values); the other formats take the rule by their channels, an
/// inference.
pub(crate) fn border_texel(
    pf: PixelFormat,
    present: [bool; 4],
    swizzle: TextureSwizzleChannels,
    color: [bool; 4],
) -> [bool; 4] {
    let rotated = crate::metal::format::rotated_channels(pf);
    let source = |c: usize| {
        if rotated {
            (c + 1) % 4
        } else if c == 3 && pf == PixelFormat::A8Unorm {
            0
        } else {
            c
        }
    };
    let v: [bool; 4] = std::array::from_fn(|c| if present[c] { color[source(c)] } else { c == 3 });
    let pick = |s: TextureSwizzle| match s {
        TextureSwizzle::Zero => false,
        TextureSwizzle::One => true,
        TextureSwizzle::Red => v[0],
        TextureSwizzle::Green => v[1],
        TextureSwizzle::Blue => v[2],
        TextureSwizzle::Alpha => v[3],
    };
    [
        pick(swizzle.red),
        pick(swizzle.green),
        pick(swizzle.blue),
        pick(swizzle.alpha),
    ]
}

/// The first bit of a [`constant_lanes`] word that says a lane is the constant 1: bit
/// `CONSTANT_ONES + lane`, where bit `lane` says it is a constant.
pub(crate) const CONSTANT_ONES: u32 = 4;

/// The lanes Metal reads as constants from every texel of a texture whose format holds the
/// channels `present` (red, green, blue, alpha; [`crate::metal::format::channels`]), made with
/// `swizzle`, as a word: bit `lane` (0 red to 3 alpha) where the lane's source is a channel the
/// format lacks or a `Zero` or `One` swizzle, and bit [`CONSTANT_ONES`]` + lane` where that
/// constant is 1 (a missing alpha, or `One`); a missing red, green or blue, and `Zero`, read 0.
/// Metal reads them so at a mapped texel, an unmapped one, and the border alike, and so after any
/// filter, which weighs constants alone there.
pub(crate) fn constant_lanes(present: [bool; 4], swizzle: TextureSwizzleChannels) -> u32 {
    let lanes = [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha];
    let mut word = 0;
    for (lane, source) in (0u32..).zip(lanes) {
        let constant = match source {
            TextureSwizzle::Zero => Some(false),
            TextureSwizzle::One => Some(true),
            TextureSwizzle::Red => (!present[0]).then_some(false),
            TextureSwizzle::Green => (!present[1]).then_some(false),
            TextureSwizzle::Blue => (!present[2]).then_some(false),
            TextureSwizzle::Alpha => (!present[3]).then_some(true),
        };
        if let Some(one) = constant {
            word |= 1 << lane;
            if one {
                word |= 1 << (CONSTANT_ONES + lane);
            }
        }
    }
    word
}

/// Whether every component of `m` is the identity swizzle (Vulkan's "identity mapping": each
/// component `IDENTITY` or its own channel).
pub(crate) fn is_identity(m: vk::ComponentMapping) -> bool {
    use vk::ComponentSwizzle as S;
    [(m.r, S::R), (m.g, S::G), (m.b, S::B), (m.a, S::A)]
        .into_iter()
        .all(|(c, own)| c == S::IDENTITY || c == own)
}

fn filter(f: SamplerMinMagFilter) -> vk::Filter {
    match f {
        SamplerMinMagFilter::Nearest => vk::Filter::NEAREST,
        SamplerMinMagFilter::Linear => vk::Filter::LINEAR,
        // A cubic sample is fetches under its pipeline's cubic-sampler variant
        // (`VariantKey::cubic_samplers`), as a constexpr bicubic sampler's is, so the Vulkan
        // sampler holds a nearest placeholder, which no filtered operation reads.
        SamplerMinMagFilter::Cubic => vk::Filter::NEAREST,
    }
}

/// The Vulkan border color of a Metal border color for a float or integer texture.
pub fn border_color(color: SamplerBorderColor, integer_texture: bool) -> vk::BorderColor {
    use vk::BorderColor as B;
    match (color, integer_texture) {
        (SamplerBorderColor::TransparentBlack, false) => B::FLOAT_TRANSPARENT_BLACK,
        (SamplerBorderColor::TransparentBlack, true) => B::INT_TRANSPARENT_BLACK,
        (SamplerBorderColor::OpaqueBlack, false) => B::FLOAT_OPAQUE_BLACK,
        (SamplerBorderColor::OpaqueBlack, true) => B::INT_OPAQUE_BLACK,
        (SamplerBorderColor::OpaqueWhite, false) => B::FLOAT_OPAQUE_WHITE,
        (SamplerBorderColor::OpaqueWhite, true) => B::INT_OPAQUE_WHITE,
    }
}

/// `MTLCompareFunction` → `VkCompareOp` (the same eight predicates in the same order).
pub fn compare_op(f: CompareFunction) -> vk::CompareOp {
    use vk::CompareOp as C;
    match f {
        CompareFunction::Never => C::NEVER,
        CompareFunction::Less => C::LESS,
        CompareFunction::Equal => C::EQUAL,
        CompareFunction::LessEqual => C::LESS_OR_EQUAL,
        CompareFunction::Greater => C::GREATER,
        CompareFunction::NotEqual => C::NOT_EQUAL,
        CompareFunction::GreaterEqual => C::GREATER_OR_EQUAL,
        CompareFunction::Always => C::ALWAYS,
    }
}

/// `log2` of the degree a sampler of `desc` filters with: its `maxAnisotropy` rounded up to a power
/// of two, 0 for an isotropic one; a pixel-coordinate sampler is isotropic. The runtime hands the
/// descriptor Metal samples with (`maxAnisotropy` 32 for 17 to 31), whose degree is then 5.
pub(crate) fn anisotropy_degree(desc: &SamplerDescriptor) -> u32 {
    if !desc.normalized_coordinates || desc.max_anisotropy <= 1 {
        return 0;
    }
    desc.max_anisotropy.next_power_of_two().trailing_zeros()
}

/// Metal's discrete sampler state as one `u32` word: what a shader reads of a sampler it learns
/// only at run time. Each field holds its `MTLSamplerDescriptor` value's raw Metal enumeration
/// value, so `MTLSamplerDescriptor`'s defaults are the word 0. The bits:
///
/// | bits | field |
/// |---|---|
/// | 0 | [`PIXEL_COORDINATES`](Self::PIXEL_COORDINATES): `normalizedCoordinates` is false |
/// | 1 | [`MIN_LINEAR`](Self::MIN_LINEAR): `minFilter` is `Linear` |
/// | 2 | [`MAG_LINEAR`](Self::MAG_LINEAR): `magFilter` is `Linear` |
/// | 3-4 | [`MIP_FILTER`](Self::MIP_FILTER): `mipFilter` (`MTLSamplerMipFilter`) |
/// | 5-7, 8-10, 11-13 | [`ADDRESS`](Self::ADDRESS): the `s`, `t`, `r` address modes |
/// | 14-15 | [`BORDER`](Self::BORDER): `borderColor` (`MTLSamplerBorderColor`) |
/// | 16-18 | [`COMPARE`](Self::COMPARE): `compareFunction` (`MTLCompareFunction`) |
/// | 19-20 | [`REDUCTION`](Self::REDUCTION): `reductionMode` (`MTLSamplerReductionMode`) |
/// | 21 | [`ANISOTROPIC`](Self::ANISOTROPIC): `maxAnisotropy` is above 1 |
/// | 22-25 | [`LEVEL_MIN`](Self::LEVEL_MIN): the level a nearest mip filter reads at `lodMinClamp` |
/// | 26-29 | [`LEVEL_MAX`](Self::LEVEL_MAX): 15 less the level a nearest mip filter reads at `lodMaxClamp` |
/// | 30 | [`CUBIC`](Self::CUBIC): `minFilter` or `magFilter` is `Cubic` |
///
/// An address mode is `MTLSamplerAddressMode`'s value. Every other bit is 0. The LOD clamps
/// themselves, `lodAverage`, and the anisotropy's degree are not in the word: the Vulkan sampler
/// applies them; the word holds the levels a nearest mip filter reads at the clamps, which a
/// nearest sample's snap needs to know the level it reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerWord(pub u32);

impl SamplerWord {
    /// The sampler reads pixel coordinates (`normalizedCoordinates` false).
    pub const PIXEL_COORDINATES: u32 = 1 << 0;
    /// The minification filter is linear.
    pub const MIN_LINEAR: u32 = 1 << 1;
    /// The magnification filter is linear.
    pub const MAG_LINEAR: u32 = 1 << 2;
    /// The lowest bit of the mip filter.
    pub const MIP_FILTER_SHIFT: u32 = 3;
    /// The mip filter's bits.
    pub const MIP_FILTER: u32 = 0b11 << Self::MIP_FILTER_SHIFT;
    /// The lowest bit of the `s`, `t`, and `r` axes' address modes.
    pub const ADDRESS_SHIFT: [u32; 3] = [5, 8, 11];
    /// The `s`, `t`, and `r` axes' address mode bits.
    pub const ADDRESS: [u32; 3] = [
        0b111 << Self::ADDRESS_SHIFT[0],
        0b111 << Self::ADDRESS_SHIFT[1],
        0b111 << Self::ADDRESS_SHIFT[2],
    ];
    /// The lowest bit of the border color.
    pub const BORDER_SHIFT: u32 = 14;
    /// The border color's bits.
    pub const BORDER: u32 = 0b11 << Self::BORDER_SHIFT;
    /// The lowest bit of the comparison function.
    pub const COMPARE_SHIFT: u32 = 16;
    /// The comparison function's bits.
    pub const COMPARE: u32 = 0b111 << Self::COMPARE_SHIFT;
    /// The lowest bit of the reduction mode.
    pub const REDUCTION_SHIFT: u32 = 19;
    /// The reduction mode's bits.
    pub const REDUCTION: u32 = 0b11 << Self::REDUCTION_SHIFT;
    /// `maxAnisotropy` is above 1.
    pub const ANISOTROPIC: u32 = 1 << 21;
    /// The lowest bit of the level a nearest mip filter reads at `lodMinClamp`.
    pub const LEVEL_MIN_SHIFT: u32 = 22;
    /// The level a nearest mip filter reads at `lodMinClamp`: with `L` its whole levels and `q`
    /// its fraction truncated onto sixty-fourths, `L + 1` where `q > 1/2`, else `L`; at most 15
    /// (past every texture's last level, 14).
    pub const LEVEL_MIN: u32 = 0xf << Self::LEVEL_MIN_SHIFT;
    /// The lowest bit of the levels below 15 that a nearest mip filter reads at `lodMaxClamp`.
    pub const LEVEL_MAX_SHIFT: u32 = 26;
    /// 15 less the level a nearest mip filter reads at `lodMaxClamp` (as at
    /// [`LEVEL_MIN`](Self::LEVEL_MIN)), so that the default clamp, which clamps nothing, is 0.
    pub const LEVEL_MAX: u32 = 0xf << Self::LEVEL_MAX_SHIFT;
    /// The min or the mag filter is `Cubic` (`MTLSamplerMinMagFilterCubic`, raw value 2): a
    /// function sampling through an argument buffer's sampler tests it where it reads the sampler's
    /// table entry, and fails its command buffer where it reaches one. It is the word's highest
    /// bit, so the test is `word >= CUBIC`.
    pub const CUBIC: u32 = 1 << 30;
    /// Every bit a word may set.
    pub const FIELDS: u32 = (1 << 31) - 1;
    /// The bits that state a cubic sampler's sample (a min or mag filter `Cubic`) beyond its
    /// filter, which Metal answers as a constexpr bicubic sampler of the same state: the
    /// coordinates, the mip filter, the address modes, the border color, the
    /// reduction mode, and the anisotropy. The plan's `VariantKey::cubic_samplers` holds the bound
    /// word masked by them, and the function samples through the bicubic filter of that state.
    pub const CUBIC_STATE: u32 = Self::PIXEL_COORDINATES
        | Self::MIP_FILTER
        | Self::ADDRESS[0]
        | Self::ADDRESS[1]
        | Self::ADDRESS[2]
        | Self::BORDER
        | Self::REDUCTION
        | Self::ANISOTROPIC;
    /// The bits of the filter Metal's linear rule answers: the word masked by them is
    /// [`LINEAR`](Self::LINEAR) exactly for a sampler whose minification and magnification filters
    /// are linear, without a mip filter, and isotropic. Its reduction does not decide it: a minimum
    /// or maximum one reads the weighted average, as on an Apple M4.
    pub const LINEAR_FILTER: u32 =
        Self::MIN_LINEAR | Self::MAG_LINEAR | Self::MIP_FILTER | Self::ANISOTROPIC;
    /// The measured linear filter's [`LINEAR_FILTER`](Self::LINEAR_FILTER) bits.
    pub const LINEAR: u32 = Self::MIN_LINEAR | Self::MAG_LINEAR;
    /// The fields an emulated linear sample reads of a runtime sampler: its filter, its
    /// coordinates, its address modes, and its border color; not its comparison, which a sample the
    /// emulation reaches never makes, nor its reduction, which reads the weighted average.
    pub const LINEAR_STATE: u32 = Self::LINEAR_FILTER
        | Self::PIXEL_COORDINATES
        | Self::ADDRESS[0]
        | Self::ADDRESS[1]
        | Self::ADDRESS[2]
        | Self::BORDER;
    /// The word of `desc`.
    pub fn of(desc: &SamplerDescriptor) -> SamplerWord {
        let mut w = 0;
        if !desc.normalized_coordinates {
            w |= Self::PIXEL_COORDINATES;
        }
        if desc.min_filter == SamplerMinMagFilter::Linear {
            w |= Self::MIN_LINEAR;
        }
        if desc.mag_filter == SamplerMinMagFilter::Linear {
            w |= Self::MAG_LINEAR;
        }
        if desc.min_filter == SamplerMinMagFilter::Cubic
            || desc.mag_filter == SamplerMinMagFilter::Cubic
        {
            w |= Self::CUBIC;
        }
        w |= desc.mip_filter.as_raw() << Self::MIP_FILTER_SHIFT;
        let modes = [
            desc.s_address_mode,
            desc.t_address_mode,
            desc.r_address_mode,
        ];
        for (mode, shift) in modes.into_iter().zip(Self::ADDRESS_SHIFT) {
            w |= mode.as_raw() << shift;
        }
        w |= desc.border_color.as_raw() << Self::BORDER_SHIFT;
        w |= desc.compare_function.as_raw() << Self::COMPARE_SHIFT;
        w |= desc.reduction_mode.as_raw() << Self::REDUCTION_SHIFT;
        if desc.max_anisotropy > 1 {
            w |= Self::ANISOTROPIC;
        }
        w |= nearest_level(desc.lod_min_clamp) << Self::LEVEL_MIN_SHIFT;
        w |= (LAST_WORD_LEVEL - nearest_level(desc.lod_max_clamp)) << Self::LEVEL_MAX_SHIFT;
        SamplerWord(w)
    }

    /// The bits a sample through a pixel sampler tests to read the texel its coordinate names at
    /// that texel's centre (the rule for a constexpr sampler), of a texture whose texels are
    /// `integer` or not: the word masked by them is
    /// [`PIXEL_COORDINATES`](Self::PIXEL_COORDINATES) exactly for a sampler with pixel
    /// coordinates that reads one level isotropically (no mip filter, not anisotropic) and filters
    /// nearest unless the texture is integer, whose sample is nearest whatever the filters say.
    /// Its address modes and border are the device sampler's own, at any value.
    pub fn nearest_pixel(integer: bool) -> u32 {
        let mut mask = Self::PIXEL_COORDINATES | Self::MIP_FILTER | Self::ANISOTROPIC;
        if !integer {
            mask |= Self::MIN_LINEAR | Self::MAG_LINEAR;
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::requirements::VkExtension;
    use crate::device::{Device, DevicePolicy};
    use crate::error::DriverDeviation;
    use crate::metal::format::tests::committed_descriptions;
    use crate::metal::format::FormatEmulation;
    use crate::metal::SamplerReductionMode;

    fn metal_default() -> SamplerDescriptor {
        SamplerDescriptor {
            min_filter: SamplerMinMagFilter::Nearest,
            mag_filter: SamplerMinMagFilter::Nearest,
            mip_filter: SamplerMipFilter::NotMipmapped,
            s_address_mode: SamplerAddressMode::ClampToEdge,
            t_address_mode: SamplerAddressMode::ClampToEdge,
            r_address_mode: SamplerAddressMode::ClampToEdge,
            border_color: SamplerBorderColor::TransparentBlack,
            normalized_coordinates: true,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            lod_average: false,
            max_anisotropy: 1,
            compare_function: CompareFunction::Never,
            reduction_mode: SamplerReductionMode::WeightedAverage,
        }
    }

    /// The level Metal's `calculate_clamped_lod` answers outside a fragment function before the
    /// texture's last level, for every runtime sampler our probes measured (the M3 Max's and the
    /// M4's answers are the same), in sixty-fourths.
    #[test]
    fn a_query_outside_a_fragment_function_reads_the_minimum_clamp_on_metals_grid() {
        use SamplerMinMagFilter::{Linear as L, Nearest as N};
        use SamplerMipFilter::{Linear, Nearest, NotMipmapped};
        // The mip filter, the clamps, the min and mag filters, the anisotropy, and the answer.
        type Case = (
            SamplerMipFilter,
            f32,
            f32,
            SamplerMinMagFilter,
            SamplerMinMagFilter,
            u32,
            f32,
        );
        let cases: &[Case] = &[
            (Linear, 0.0, f32::MAX, L, L, 1, 0.0),
            (Nearest, 0.0, f32::MAX, L, L, 1, 0.0),
            (NotMipmapped, 0.0, f32::MAX, L, L, 1, 0.0),
            (Linear, 2.5, 4.0, L, L, 1, 2.5),
            (Linear, 0.3, 4.0, L, L, 1, 0.296875),
            (Linear, 1.1, 1.9, L, L, 1, 1.09375),
            (Linear, 2.7, 3.2, L, L, 1, 2.6875),
            (Linear, 0.0, 0.6, L, L, 1, 0.0),
            (Linear, 5.5, 6.5, L, L, 1, 5.5),
            (Linear, 0.001, 4.0, L, L, 1, 0.0),
            (Linear, 1.0 / 3.0, 4.0, L, L, 1, 0.328125),
            (Linear, 2.5, 2.5, L, L, 1, 2.5),
            (Linear, 1e-7, 4.0, L, L, 1, 0.0),
            (Linear, 2.0 + 1.0 / 256.0, 4.0, L, L, 1, 2.0),
            (Nearest, 0.3, 4.0, L, L, 1, 0.0),
            (Nearest, 0.5, 4.0, L, L, 1, 0.0),
            (Nearest, 0.7, 4.0, L, L, 1, 1.0),
            (Nearest, 1.5, 4.0, L, L, 1, 1.0),
            (Nearest, 2.7, 3.2, L, L, 1, 3.0),
            (Nearest, 2.5, 4.0, L, L, 1, 2.0),
            (Nearest, 0.515_625, 4.0, L, L, 1, 1.0),
            (Nearest, 0.507_812_5, 4.0, L, L, 1, 0.0),
            (Nearest, 0.5, 4.0, N, L, 1, 0.0),
            (NotMipmapped, 2.5, 4.0, L, L, 1, 0.0),
            (NotMipmapped, 0.3, 0.6, L, L, 1, 0.0),
            (Linear, 0.3, 4.0, L, L, 4, 0.296875),
            (Linear, 0.3, 4.0, N, N, 1, 0.296875),
            (Linear, 0.3, 4.0, L, N, 1, 0.296875),
            (Linear, 0.3, 4.0, N, L, 1, 0.296875),
            (Linear, 13.99, 16.0, L, L, 1, 13.984375),
            (Linear, 14.9, 15.0, L, L, 1, 14.890625),
            (Nearest, 13.6, 16.0, L, L, 1, 14.0),
            (Linear, 20.0, 30.0, L, L, 1, 15.0),
            (Linear, f32::MAX, f32::MAX, L, L, 1, 15.0),
        ];
        for &(mip, lo, hi, min, mag, aniso, want) in cases {
            let mut d = metal_default();
            d.mip_filter = mip;
            d.lod_min_clamp = lo;
            d.lod_max_clamp = hi;
            d.min_filter = min;
            d.mag_filter = mag;
            d.max_anisotropy = aniso;
            assert_eq!(
                query_min_level(&d),
                (want * 64.0) as u32,
                "{mip:?} [{lo}, {hi}] {min:?}/{mag:?} x{aniso}"
            );
        }
        // Every measured answer is this level bounded by the view's last level: 14.890625 reads
        // 14 over 15 levels, 2.5 reads 2 over a view of 3, any level 0 over one level.
    }

    fn device(features: &[VkFeature], remove: &[VkFeature]) -> Device {
        let (_, mut desc) = committed_descriptions()
            .into_iter()
            .find(|(s, _)| s == "vk12-floor")
            .unwrap();
        desc.features.extend(features.iter().copied());
        for f in remove {
            desc.features.remove(f);
        }
        desc.limits.max_sampler_anisotropy_bits = 16f32.to_bits();
        Device::new(desc, DevicePolicy::default()).unwrap()
    }

    /// What a plan reads of a bound sampler's shape to check its separate textures' views
    /// (`vk_filtering`) is exactly what `vk_sampler` makes of the descriptor for the use: a linear
    /// filter or mip mode (none for an integer texture) and the reduction, the weighted average
    /// whatever the descriptor's, with or without a comparison.
    #[test]
    fn the_shape_states_the_filtering_vk_sampler_makes() {
        use SamplerMinMagFilter as F;
        let dev = device(&[], &[VkFeature::SamplerFilterMinmax]);
        let mut checked = 0;
        for min in [F::Nearest, F::Linear] {
            for mag in [F::Nearest, F::Linear] {
                for mip in [
                    SamplerMipFilter::NotMipmapped,
                    SamplerMipFilter::Nearest,
                    SamplerMipFilter::Linear,
                ] {
                    for reduction in [
                        SamplerReductionMode::WeightedAverage,
                        SamplerReductionMode::Minimum,
                        SamplerReductionMode::Maximum,
                    ] {
                        for (compare, integer) in
                            [(false, false), (true, false), (false, true), (true, true)]
                        {
                            let mut d = metal_default();
                            d.min_filter = min;
                            d.mag_filter = mag;
                            d.mip_filter = mip;
                            d.reduction_mode = reduction;
                            let use_ = SamplerUse::new(compare, integer);
                            let s = vk_sampler(&d, use_, dev.profile()).expect("a sampler");
                            assert_eq!(s.reduction, vk::SamplerReductionMode::WEIGHTED_AVERAGE);
                            let linear = s.mag_filter == vk::Filter::LINEAR
                                || s.min_filter == vk::Filter::LINEAR
                                || s.mipmap_mode == vk::SamplerMipmapMode::LINEAR;
                            assert_eq!(
                                vk_filtering(&d.shape(), use_),
                                (linear, s.reduction),
                                "{d:?} {use_:?}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 12 * 3 * 4);
    }

    #[test]
    fn defaults_map_to_a_base_level_sampler() {
        let dev = device(&[], &[]);
        let s = vk_sampler(&metal_default(), SamplerUse::default(), dev.profile()).unwrap();
        assert_eq!(s.mipmap_mode, vk::SamplerMipmapMode::NEAREST);
        assert_eq!((s.min_lod, s.max_lod), (0.0, NOT_MIPMAPPED_MAX_LOD));
        assert_eq!(s.compare, None);
        assert_eq!(s.max_anisotropy, None);
        assert_eq!(s.address_mode_u, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(s.reduction, vk::SamplerReductionMode::WEIGHTED_AVERAGE);
    }

    /// A sampler without a mip filter reads the base level, its minification filter exactly where
    /// Metal's level raised to `lodMinClamp` and lowered to `lodMaxClamp` is above 0 (the device's
    /// `λ > 0`), as Metal chooses between a min and a mag filter that differ.
    #[test]
    fn a_not_mipmapped_samplers_clamps_choose_metals_filter() {
        let dev = device(&[], &[]);
        for (lo, hi, min_lod, max_lod) in [
            (0.0, f32::MAX, 0.0, NOT_MIPMAPPED_MAX_LOD),
            (-3.0, 1000.0, 0.0, NOT_MIPMAPPED_MAX_LOD),
            (1.0, f32::MAX, NOT_MIPMAPPED_MAX_LOD, NOT_MIPMAPPED_MAX_LOD),
            (0.125, 2.0, 0.125, NOT_MIPMAPPED_MAX_LOD),
            (-1.0, 0.0, 0.0, 0.0),
            (-2.0, -1.0, 0.0, 0.0),
        ] {
            let mut d = metal_default();
            d.min_filter = SamplerMinMagFilter::Linear;
            d.lod_min_clamp = lo;
            d.lod_max_clamp = hi;
            let s = vk_sampler(&d, SamplerUse::default(), dev.profile()).unwrap();
            assert_eq!(s.mipmap_mode, vk::SamplerMipmapMode::NEAREST, "{lo} {hi}");
            assert_eq!((s.min_lod, s.max_lod), (min_lod, max_lod), "{lo} {hi}");
            // The device's level at a level `l`: minification exactly where Metal's is.
            for l in [-1.0f32, 0.0, 0.25, 0.5, 1.0, 2.0] {
                let device = l.clamp(s.min_lod, s.max_lod) > 0.0;
                let metal = l.max(lo).min(hi) > 0.0;
                assert_eq!(device, metal, "{lo} {hi} at {l}");
            }
        }
    }

    /// Through Metal's own sampler, an anisotropic sampler without a mip filter whose filters
    /// agree is clamped to `[0, 0]` whatever its clamps, where that sampler reads Metal's
    /// `NotMipmapped`; one whose filters differ, an isotropic one, and every sampler off Metal's
    /// sampler keep the clamp that chooses Metal's filter.
    #[test]
    fn metals_own_anisotropic_sampler_reads_not_mipmapped_at_level_zero() {
        use SamplerMinMagFilter::{Linear, Nearest};
        let with = |metal_sampler: bool| {
            let (_, mut desc) = committed_descriptions()
                .into_iter()
                .find(|(s, _)| s == "vk12-floor")
                .unwrap();
            desc.features.insert(VkFeature::SamplerAnisotropy);
            desc.limits.max_sampler_anisotropy_bits = 16f32.to_bits();
            Device::with_deviation(
                desc,
                DevicePolicy::default(),
                DriverDeviation::MetalSampler,
                metal_sampler,
            )
            .unwrap()
        };
        for (metal_sampler, min, mag, anisotropy, lo, hi, lods) in [
            (true, Linear, Linear, 8, 0.0, f32::MAX, (0.0, 0.0)),
            (true, Nearest, Nearest, 16, 0.0, f32::MAX, (0.0, 0.0)),
            (true, Linear, Linear, 2, 1.0, f32::MAX, (0.0, 0.0)),
            (true, Linear, Linear, 8, 0.125, 2.0, (0.0, 0.0)),
            (
                true,
                Linear,
                Nearest,
                8,
                0.0,
                f32::MAX,
                (0.0, NOT_MIPMAPPED_MAX_LOD),
            ),
            (
                true,
                Linear,
                Linear,
                1,
                0.0,
                f32::MAX,
                (0.0, NOT_MIPMAPPED_MAX_LOD),
            ),
            (
                false,
                Linear,
                Linear,
                8,
                0.0,
                f32::MAX,
                (0.0, NOT_MIPMAPPED_MAX_LOD),
            ),
        ] {
            let dev = with(metal_sampler);
            let mut d = metal_default();
            d.min_filter = min;
            d.mag_filter = mag;
            d.max_anisotropy = anisotropy;
            d.lod_min_clamp = lo;
            d.lod_max_clamp = hi;
            let s = vk_sampler(&d, SamplerUse::default(), dev.profile()).unwrap();
            let what = format!("{min:?} {mag:?} {anisotropy} [{lo}, {hi}], {metal_sampler}");
            assert_eq!(s.mipmap_mode, vk::SamplerMipmapMode::NEAREST, "{what}");
            assert_eq!((s.min_lod, s.max_lod), lods, "{what}");
        }
    }

    #[test]
    fn filters_lods_and_compare() {
        let dev = device(&[], &[]);
        let mut d = metal_default();
        d.min_filter = SamplerMinMagFilter::Linear;
        d.mip_filter = SamplerMipFilter::Linear;
        d.lod_min_clamp = 1.0;
        d.lod_max_clamp = 4.0;
        d.compare_function = CompareFunction::LessEqual;
        let s = vk_sampler(&d, SamplerUse::new(true, false), dev.profile()).unwrap();
        assert_eq!(
            (s.min_filter, s.mag_filter),
            (vk::Filter::LINEAR, vk::Filter::NEAREST)
        );
        assert_eq!(s.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
        assert_eq!((s.min_lod, s.max_lod), (1.0, 4.0));
        assert_eq!(s.compare, Some(vk::CompareOp::LESS_OR_EQUAL));
        // The same descriptor sampled without comparison has no compare op.
        let plain = vk_sampler(&d, SamplerUse::default(), dev.profile()).unwrap();
        assert_eq!(plain.compare, None);
        d.lod_min_clamp = f32::NAN;
        assert_eq!(
            vk_sampler(&d, SamplerUse::default(), dev.profile()),
            Err(Error::InvalidDescriptor(DescriptorError::SamplerLodClamp))
        );
        d.lod_min_clamp = 5.0;
        assert!(vk_sampler(&d, SamplerUse::default(), dev.profile()).is_err());
    }

    /// A portability-subset device without `mutableComparisonSamplers` takes no comparison sampler
    /// (`VUID-VkSamplerCreateInfo-mutableComparisonSamplers-04450`); a device without the subset
    /// has no such restriction.
    #[test]
    fn the_portability_subset_can_withhold_comparison_samplers() {
        let committed = |name: &str| {
            committed_descriptions()
                .into_iter()
                .find(|(s, _)| s == name)
                .unwrap()
                .1
        };
        let mut d = metal_default();
        d.compare_function = CompareFunction::LessEqual;
        let compare = SamplerUse::new(true, false);
        let mutable = VkFeature::MutableComparisonSamplers;
        let mvk = committed("moltenvk-m3");
        assert!(mvk.extensions.contains(&VkExtension::KhrPortabilitySubset));
        assert!(mvk.features.contains(&mutable));
        let dev = Device::new(mvk.clone(), DevicePolicy::default()).unwrap();
        assert!(vk_sampler(&d, compare, dev.profile()).is_ok());
        let mut withheld = mvk;
        withheld.features.remove(&mutable);
        let dev = Device::new(withheld, DevicePolicy::default()).unwrap();
        assert_eq!(
            vk_sampler(&d, compare, dev.profile()),
            Err(Unsupported::needs(Feature::SamplerCompare, Capability::Feature(mutable)).into())
        );
        assert!(vk_sampler(&d, SamplerUse::default(), dev.profile()).is_ok());
        let lavapipe = committed("lavapipe");
        assert!(!lavapipe.features.contains(&mutable));
        let dev = Device::new(lavapipe, DevicePolicy::default()).unwrap();
        assert!(vk_sampler(&d, compare, dev.profile()).is_ok());
    }

    /// Metal filters a `maxAnisotropy` rounded up to a power of two, a runtime sampler's and a
    /// constexpr one's alike (an Apple M4), so the device's sampler is handed that
    /// degree, within the device's limit.
    #[test]
    fn the_device_filters_the_anisotropy_rounded_up_to_a_power_of_two() {
        let dev = device(&[VkFeature::SamplerAnisotropy], &[]);
        for (value, handed) in [
            (1, None),
            (2, Some(2.0)),
            (3, Some(4.0)),
            (4, Some(4.0)),
            (5, Some(8.0)),
            (7, Some(8.0)),
            (8, Some(8.0)),
            (9, Some(16.0)),
            (12, Some(16.0)),
            (16, Some(16.0)),
        ] {
            let mut d = metal_default();
            d.min_filter = SamplerMinMagFilter::Linear;
            d.mag_filter = SamplerMinMagFilter::Linear;
            d.max_anisotropy = value;
            let s = vk_sampler(&d, SamplerUse::default(), dev.profile()).unwrap();
            assert_eq!(s.max_anisotropy, handed, "{value}");
        }
    }

    /// A sampler with a nearest min or mag filter is handed its degree, through Metal's sampler and
    /// natively alike, as one with linear filters is.
    #[test]
    fn a_nearest_filter_is_handed_its_anisotropy() {
        use SamplerMinMagFilter::{Linear, Nearest};
        let with = |metal_sampler: bool| {
            let (_, mut desc) = committed_descriptions()
                .into_iter()
                .find(|(s, _)| s == "vk12-floor")
                .unwrap();
            desc.features.insert(VkFeature::SamplerAnisotropy);
            desc.limits.max_sampler_anisotropy_bits = 16f32.to_bits();
            Device::with_deviation(
                desc,
                DevicePolicy::default(),
                DriverDeviation::MetalSampler,
                metal_sampler,
            )
            .unwrap()
        };
        for metal_sampler in [false, true] {
            let dev = with(metal_sampler);
            for (min, mag) in [
                (Linear, Linear),
                (Nearest, Nearest),
                (Nearest, Linear),
                (Linear, Nearest),
            ] {
                let mut d = metal_default();
                d.min_filter = min;
                d.mag_filter = mag;
                d.max_anisotropy = 5;
                let s = vk_sampler(&d, SamplerUse::default(), dev.profile()).unwrap();
                assert_eq!(
                    s.max_anisotropy,
                    Some(8.0),
                    "{min:?} {mag:?}, Metal's sampler {metal_sampler}"
                );
            }
        }
    }

    /// An integer texture is never filtered. A linear descriptor used with a float texture keeps
    /// its filters; used with an integer one it samples nearest at every level, as Metal reads it,
    /// so no Vulkan sampler asks an integer format for `SAMPLED_IMAGE_FILTER_LINEAR`.
    #[test]
    fn integer_textures_sample_nearest_through_a_linear_descriptor() {
        let mut d = metal_default();
        d.min_filter = SamplerMinMagFilter::Linear;
        d.mag_filter = SamplerMinMagFilter::Linear;
        d.mip_filter = SamplerMipFilter::Linear;
        d.max_anisotropy = 8;
        let dev = device(&[VkFeature::SamplerAnisotropy], &[]);
        let float = vk_sampler(&d, SamplerUse::new(false, false), dev.profile()).unwrap();
        assert_eq!(float.max_anisotropy, Some(8.0));
        assert_eq!(
            (float.min_filter, float.mag_filter, float.mipmap_mode),
            (
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::LINEAR
            )
        );
        let integer = vk_sampler(&d, SamplerUse::new(false, true), dev.profile()).unwrap();
        assert_eq!(
            (
                integer.min_filter,
                integer.mag_filter,
                integer.mipmap_mode,
                integer.max_anisotropy
            ),
            (
                vk::Filter::NEAREST,
                vk::Filter::NEAREST,
                vk::SamplerMipmapMode::NEAREST,
                None
            )
        );
        // Only the filters differ.
        assert_eq!(
            vk::SamplerDesc {
                min_filter: float.min_filter,
                mag_filter: float.mag_filter,
                mipmap_mode: float.mipmap_mode,
                max_anisotropy: float.max_anisotropy,
                border_color: float.border_color,
                ..integer
            },
            float
        );
    }

    #[test]
    fn compare_functions_keep_their_predicates() {
        for &f in CompareFunction::ALL {
            let op = compare_op(f);
            assert_eq!(op.as_raw() as u32, f.as_raw(), "{f:?}");
            let metal = f
                .name()
                .trim_start_matches("MTLCompareFunction")
                .to_uppercase();
            let vk = op
                .name()
                .unwrap()
                .trim_start_matches("VK_COMPARE_OP_")
                .replace('_', "");
            let vk = vk.replace("OREQUAL", "EQUAL");
            assert_eq!(metal, vk, "{f:?} -> {op:?}");
        }
    }

    /// `ClampToZero` clamps to the sampler's border color, as `ClampToBorderColor` does, so every
    /// border axis of a sampler reads its one color.
    #[test]
    fn clamp_to_zero_reads_the_sampler_s_border_color() {
        let dev = device(&[], &[]);
        let mut d = metal_default();
        d.s_address_mode = SamplerAddressMode::ClampToZero;
        let s = vk_sampler(&d, SamplerUse::new(false, true), dev.profile()).unwrap();
        assert_eq!(s.address_mode_u, vk::SamplerAddressMode::CLAMP_TO_BORDER);
        assert_eq!(s.border_color, vk::BorderColor::INT_TRANSPARENT_BLACK);
        for (color, float, int) in [
            (
                SamplerBorderColor::OpaqueBlack,
                vk::BorderColor::FLOAT_OPAQUE_BLACK,
                vk::BorderColor::INT_OPAQUE_BLACK,
            ),
            (
                SamplerBorderColor::OpaqueWhite,
                vk::BorderColor::FLOAT_OPAQUE_WHITE,
                vk::BorderColor::INT_OPAQUE_WHITE,
            ),
        ] {
            d.border_color = color;
            let s = vk_sampler(&d, SamplerUse::default(), dev.profile()).unwrap();
            assert_eq!(s.border_color, float);
            let s = vk_sampler(&d, SamplerUse::new(false, true), dev.profile()).unwrap();
            assert_eq!(s.border_color, int);
            // An axis of each border mode: one color.
            let mut mixed = d;
            mixed.t_address_mode = SamplerAddressMode::ClampToBorderColor;
            let s = vk_sampler(&mixed, SamplerUse::default(), dev.profile()).unwrap();
            assert_eq!(s.border_color, float);
            assert_eq!(s.address_mode_v, vk::SamplerAddressMode::CLAMP_TO_BORDER);
        }
        // No axis clamps to a border: the color is unused.
        d.s_address_mode = SamplerAddressMode::Repeat;
        let s = vk_sampler(&d, SamplerUse::default(), dev.profile()).unwrap();
        assert_eq!(s.border_color, vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
    }

    /// Mirror-clamp needs its feature; anisotropy needs its own only through an argument buffer
    /// (`check_argument_sampler`), the Vulkan sampler isotropic without it; a min or max
    /// reduction needs none and reads the weighted average, with or without a comparison.
    #[test]
    fn optional_sampler_features_are_required_or_refused() {
        let none = device(
            &[],
            &[
                VkFeature::SamplerMirrorClampToEdge,
                VkFeature::SamplerFilterMinmax,
                VkFeature::SamplerAnisotropy,
            ],
        );
        let all = device(
            &[
                VkFeature::SamplerMirrorClampToEdge,
                VkFeature::SamplerFilterMinmax,
                VkFeature::SamplerAnisotropy,
            ],
            &[],
        );
        let mut mirror = metal_default();
        mirror.r_address_mode = SamplerAddressMode::MirrorClampToEdge;
        let mut minmax = metal_default();
        minmax.reduction_mode = SamplerReductionMode::Maximum;
        // Linear filters: off Metal's sampler a nearest one is handed no anisotropy.
        let mut aniso = metal_default();
        aniso.min_filter = SamplerMinMagFilter::Linear;
        aniso.mag_filter = SamplerMinMagFilter::Linear;
        aniso.max_anisotropy = 32;
        let feature = |err: Error| {
            let Error::Unsupported(u) = err else {
                panic!("{err:?}")
            };
            u.needs
        };
        let needs = |f| crate::error::Needs::AnyOf(vec![vec![Capability::Feature(f)]]);
        let err = vk_sampler(&mirror, SamplerUse::default(), none.profile()).unwrap_err();
        assert_eq!(feature(err), needs(VkFeature::SamplerMirrorClampToEdge));
        assert!(vk_sampler(&mirror, SamplerUse::default(), all.profile()).is_ok());
        let s = vk_sampler(&aniso, SamplerUse::default(), none.profile()).unwrap();
        assert_eq!(s.max_anisotropy, None, "isotropic without the feature");
        let s = vk_sampler(&aniso, SamplerUse::default(), all.profile()).unwrap();
        assert_eq!(s.max_anisotropy, Some(16.0), "clamped to the device limit");
        // An argument buffer's sampler needs the feature.
        let shape = aniso.shape();
        let err = check_argument_sampler(&shape, SamplerUse::default(), none.profile());
        assert_eq!(
            feature(err.unwrap_err()),
            needs(VkFeature::SamplerAnisotropy)
        );
        assert!(check_argument_sampler(&shape, SamplerUse::default(), all.profile()).is_ok());
        for (p, compare) in [(&none, false), (&none, true), (&all, false), (&all, true)] {
            let s = vk_sampler(&minmax, SamplerUse::new(compare, false), p.profile()).unwrap();
            assert_eq!(s.reduction, vk::SamplerReductionMode::WEIGHTED_AVERAGE);
        }
    }

    /// An opaque-black border through a non-identity view mapping, the texture's swizzle composed
    /// on the format's, needs the device to swizzle borders, or Metal's sampler; a view whose
    /// border texel is not Metal's (`A8Unorm` through `A8_UNORM`) refuses, or names the direct
    /// format where its view would read Metal's; other borders, identity mappings, and samplers
    /// without a border axis need nothing.
    #[test]
    fn border_swizzle_follows_the_view_mapping() {
        use vk::ComponentSwizzle as S;
        let plain = device(&[], &[]);
        let with = |swizzle: bool, metal_sampler: bool| {
            let (_, mut desc) = committed_descriptions()
                .into_iter()
                .find(|(s, _)| s == "vk12-floor")
                .unwrap();
            if swizzle {
                desc.extensions.extend([
                    VkExtension::ExtBorderColorSwizzle,
                    VkExtension::ExtCustomBorderColor,
                ]);
                desc.features.extend([
                    VkFeature::BorderColorSwizzle,
                    VkFeature::BorderColorSwizzleFromImage,
                ]);
            }
            let policy = DevicePolicy::default();
            Device::with_deviation(desc, policy, DriverDeviation::MetalSampler, metal_sampler)
                .unwrap()
        };
        let swizzling = with(true, false);
        let metal = with(false, true);
        let mapping = |format, components, emulation| FormatMapping {
            format,
            components,
            aspects: vk::ImageAspectFlags::COLOR,
            emulation,
        };
        let direct = mapping(
            vk::Format::R8G8B8A8_UNORM,
            vk::ComponentMapping::IDENTITY,
            FormatEmulation::Direct,
        );
        let twin = mapping(
            vk::Format::B8G8R8A8_UNORM,
            vk::ComponentMapping {
                r: S::B,
                g: S::IDENTITY,
                b: S::R,
                a: S::IDENTITY,
            },
            FormatEmulation::SwizzledView,
        );
        let a8 = mapping(
            vk::Format::R8_UNORM,
            vk::ComponentMapping {
                r: S::ZERO,
                g: S::ZERO,
                b: S::ZERO,
                a: S::R,
            },
            FormatEmulation::AlphaInRed,
        );
        let mut black = metal_default();
        black.t_address_mode = SamplerAddressMode::ClampToBorderColor;
        black.border_color = SamplerBorderColor::OpaqueBlack;
        let mut white = black;
        white.border_color = SamplerBorderColor::OpaqueWhite;
        let mut zero = metal_default();
        zero.s_address_mode = SamplerAddressMode::ClampToZero;
        let mut unused = metal_default();
        unused.border_color = SamplerBorderColor::OpaqueBlack;
        let own = TextureSwizzleChannels::DEFAULT;
        let check = |d, pf, m, s, dev: &Device| check_border_swizzle(d, pf, m, s, dev.profile());
        let rgba = PixelFormat::RGBA8Unorm;
        let alpha = PixelFormat::A8Unorm;
        let a8_unorm = mapping(
            vk::Format::A8_UNORM,
            vk::ComponentMapping::IDENTITY,
            FormatEmulation::Direct,
        );
        let features = || -> Result<(), Error> {
            Err(Unsupported::needs_any_of(
                Feature::SamplerBorderColor,
                vec![vec![
                    Capability::Feature(VkFeature::BorderColorSwizzle),
                    Capability::Feature(VkFeature::BorderColorSwizzleFromImage),
                ]],
            )
            .into())
        };
        let argb = TextureSwizzleChannels::new(
            TextureSwizzle::Alpha,
            TextureSwizzle::Red,
            TextureSwizzle::Green,
            TextureSwizzle::Blue,
        );
        for dev in [&plain, &swizzling, &metal] {
            assert_eq!(check(&black, rgba, &direct, own, dev), Ok(()));
            for d in [&white, &zero, &unused] {
                for s in [own, argb] {
                    assert_eq!(check(d, rgba, &twin, s, dev), Ok(()));
                    assert_eq!(check(d, alpha, &a8, s, dev), Ok(()));
                    assert_eq!(check(d, alpha, &a8_unorm, s, dev), Ok(()));
                }
            }
        }
        // `A8_UNORM` reads an opaque-black border's alpha as 1, Metal's `A8Unorm` as the color's
        // red, 0; only Metal's sampler reads it so.
        for dev in [&plain, &swizzling] {
            assert_eq!(
                check(&black, alpha, &a8_unorm, own, dev),
                Err(Unsupported::not_implemented(Feature::SamplerBorderColor).into())
            );
        }
        assert_eq!(check(&black, alpha, &a8_unorm, own, &metal), Ok(()));
        // `R8_UNORM`'s `(0, 0, 0, R)` reads the border's red as alpha, Metal's texel, where the
        // device swizzles borders.
        assert_eq!(check(&black, alpha, &a8, own, &plain), features());
        assert_eq!(check(&black, alpha, &a8, own, &swizzling), Ok(()));
        assert_eq!(check(&black, alpha, &a8, own, &metal), Ok(()));
        // The texture's swizzle is composed on the format's mapping.
        for m in [&direct, &twin] {
            for s in [own, argb] {
                let expected = match (m == &direct, s == own) {
                    (true, true) => Ok(()),
                    _ => features(),
                };
                assert_eq!(check(&black, rgba, m, s, &plain), expected, "{m:?} {s:?}");
                assert_eq!(check(&black, rgba, m, s, &swizzling), Ok(()));
                assert_eq!(check(&black, rgba, m, s, &metal), Ok(()));
            }
        }
    }

    /// Metal reads an opaque-black border as its transparent-black one unless the format has an
    /// alpha channel of its own that the swizzle reads; `black_texel` is `border_texel` for the
    /// two colors whatever the format's red, green and blue.
    #[test]
    fn opaque_black_reads_transparent_where_metal_s_texels_agree() {
        use TextureSwizzle as T;
        let swizzles = [
            TextureSwizzleChannels::DEFAULT,
            TextureSwizzleChannels::new(T::Red, T::Red, T::Red, T::One),
            TextureSwizzleChannels::new(T::Alpha, T::Red, T::Green, T::Blue),
            TextureSwizzleChannels::new(T::Zero, T::One, T::Red, T::Green),
            TextureSwizzleChannels::new(T::Blue, T::Green, T::Red, T::Alpha),
            TextureSwizzleChannels::new(T::Green, T::Alpha, T::Zero, T::Red),
        ];
        let reads_alpha =
            |s: TextureSwizzleChannels| [s.red, s.green, s.blue, s.alpha].contains(&T::Alpha);
        let mut checked = 0;
        for &pf in PixelFormat::ALL {
            let Some(texels) = crate::metal::bicubic::texels(pf) else {
                continue;
            };
            let present: [bool; 4] = std::array::from_fn(|c| texels.has_channel(c as u32));
            for s in swizzles {
                for color in [BORDER_COLORS[0], BORDER_COLORS[1]] {
                    assert_eq!(
                        black_texel(pf, s, color),
                        border_texel(pf, present, s, color),
                        "{pf:?} {s:?}"
                    );
                }
                let own_alpha = present[3] && pf != PixelFormat::A8Unorm;
                assert_eq!(
                    opaque_black_reads_transparent(pf, s),
                    !(own_alpha && reads_alpha(s)),
                    "{pf:?} {s:?}"
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "{checked}");
        for s in swizzles {
            assert!(opaque_black_reads_transparent(PixelFormat::A8Unorm, s));
            assert!(opaque_black_reads_transparent(PixelFormat::R8Unorm, s));
        }
    }

    /// An opaque-black built-in border through an `R4G4B4A4`/`B4G4R4A4` view (`ABGR4Unorm`'s
    /// candidates) needs `rgba4OpaqueBlackSwizzled`, whatever the mapping; the twin's red/blue
    /// mapping needs the border-swizzle features as well. Other borders need nothing.
    #[test]
    fn rgba4_views_need_the_opaque_black_property() {
        use vk::ComponentSwizzle as S;
        let with = |swizzle: bool, rgba4: bool| {
            let (_, mut desc) = committed_descriptions()
                .into_iter()
                .find(|(s, _)| s == "vk12-floor")
                .unwrap();
            if swizzle {
                desc.extensions.extend([
                    VkExtension::ExtBorderColorSwizzle,
                    VkExtension::ExtCustomBorderColor,
                ]);
                desc.features.extend([
                    VkFeature::BorderColorSwizzle,
                    VkFeature::BorderColorSwizzleFromImage,
                ]);
            }
            desc.rgba4_opaque_black_swizzled = rgba4;
            Device::new(desc, DevicePolicy::default()).unwrap()
        };
        let direct = FormatMapping {
            format: vk::Format::R4G4B4A4_UNORM_PACK16,
            components: vk::ComponentMapping::IDENTITY,
            aspects: vk::ImageAspectFlags::COLOR,
            emulation: FormatEmulation::Direct,
        };
        let twin = FormatMapping {
            format: vk::Format::B4G4R4A4_UNORM_PACK16,
            components: vk::ComponentMapping {
                r: S::B,
                g: S::IDENTITY,
                b: S::R,
                a: S::IDENTITY,
            },
            aspects: vk::ImageAspectFlags::COLOR,
            emulation: FormatEmulation::SwizzledView,
        };
        let pf = PixelFormat::ABGR4Unorm;
        let needs = |caps: Vec<Capability>| -> Result<(), Error> {
            Err(Unsupported::needs_any_of(Feature::SamplerBorderColor, vec![caps]).into())
        };
        let property = Capability::Property(VkProperty::Rgba4OpaqueBlackSwizzled);
        let features = [
            Capability::Feature(VkFeature::BorderColorSwizzle),
            Capability::Feature(VkFeature::BorderColorSwizzleFromImage),
        ];
        let own = TextureSwizzleChannels::DEFAULT;
        let check = |m, dev: &Device| check_border_view(true, pf, m, own, dev.profile());
        assert_eq!(
            check(&direct, &with(false, false)),
            needs(vec![property.clone()])
        );
        assert_eq!(check(&direct, &with(false, true)), Ok(()));
        assert_eq!(
            check(&twin, &with(false, false)),
            needs(features.iter().cloned().chain([property.clone()]).collect())
        );
        assert_eq!(
            check(&twin, &with(true, false)),
            needs(vec![property.clone()])
        );
        assert_eq!(check(&twin, &with(true, true)), Ok(()));
        for m in [&direct, &twin] {
            assert_eq!(
                check_border_view(false, pf, m, own, with(false, false).profile()),
                Ok(())
            );
        }
        // The descriptor form: a transparent-black border needs nothing.
        let mut zero = metal_default();
        zero.s_address_mode = SamplerAddressMode::ClampToZero;
        let plain = with(false, false);
        assert_eq!(
            check_border_swizzle(&zero, pf, &direct, own, plain.profile()),
            Ok(())
        );
        let mut black = zero;
        black.s_address_mode = SamplerAddressMode::ClampToBorderColor;
        black.border_color = SamplerBorderColor::OpaqueBlack;
        assert_eq!(
            check_border_swizzle(&black, pf, &direct, own, plain.profile()),
            needs(vec![property])
        );
    }

    /// A runtime sampler made per view: through a mapping other than the identity, with
    /// `borderColorSwizzle` and without `borderColorSwizzleFromImage`, the device reads an
    /// opaque-black border as Metal's texel when the sampler carries the view's mapping
    /// (`Mapped`); one the binder cannot make per view needs `FromImage` there. Both features, or
    /// Metal's sampler, read the stated border through any view; without `borderColorSwizzle`
    /// the mapping is refused for it alone.
    #[test]
    fn opaque_black_maps_the_sampler_where_the_view_cannot() {
        use vk::ComponentSwizzle as S;
        let with = |features: &[VkFeature], metal_sampler: bool| {
            let (_, mut desc) = committed_descriptions()
                .into_iter()
                .find(|(s, _)| s == "vk12-floor")
                .unwrap();
            if !features.is_empty() {
                desc.extensions.extend([
                    VkExtension::ExtBorderColorSwizzle,
                    VkExtension::ExtCustomBorderColor,
                ]);
                desc.features.extend(features.iter().copied());
            }
            let policy = DevicePolicy::default();
            Device::with_deviation(desc, policy, DriverDeviation::MetalSampler, metal_sampler)
                .unwrap()
        };
        let plain = with(&[], false);
        let swizzle_only = with(&[VkFeature::BorderColorSwizzle], false);
        let both = with(
            &[
                VkFeature::BorderColorSwizzle,
                VkFeature::BorderColorSwizzleFromImage,
            ],
            false,
        );
        let metal = with(&[], true);
        let direct = FormatMapping {
            format: vk::Format::R8G8B8A8_UNORM,
            components: vk::ComponentMapping::IDENTITY,
            aspects: vk::ImageAspectFlags::COLOR,
            emulation: FormatEmulation::Direct,
        };
        let twin = FormatMapping {
            format: vk::Format::B8G8R8A8_UNORM,
            components: vk::ComponentMapping {
                r: S::B,
                g: S::IDENTITY,
                b: S::R,
                a: S::IDENTITY,
            },
            aspects: vk::ImageAspectFlags::COLOR,
            emulation: FormatEmulation::SwizzledView,
        };
        let own = TextureSwizzleChannels::DEFAULT;
        let argb = TextureSwizzleChannels::new(
            TextureSwizzle::Alpha,
            TextureSwizzle::Red,
            TextureSwizzle::Green,
            TextureSwizzle::Blue,
        );
        let rgba = PixelFormat::RGBA8Unorm;
        let needs = |features: &[VkFeature]| -> Error {
            Unsupported::needs_any_of(
                Feature::SamplerBorderColor,
                vec![features.iter().copied().map(Capability::Feature).collect()],
            )
            .into()
        };
        for dev in [&plain, &swizzle_only, &both, &metal] {
            let p = dev.profile();
            assert_eq!(
                opaque_black_border(rgba, &direct, own, p),
                Ok(OpaqueBlackBorder::Stated)
            );
            assert_eq!(check_border_view(true, rgba, &direct, own, p), Ok(()));
        }
        for dev in [&both, &metal] {
            for (m, s) in [(&direct, argb), (&twin, own), (&twin, argb)] {
                let p = dev.profile();
                assert_eq!(
                    opaque_black_border(rgba, m, s, p),
                    Ok(OpaqueBlackBorder::Stated)
                );
                assert_eq!(check_border_view(true, rgba, m, s, p), Ok(()));
            }
        }
        for (m, s) in [(&direct, argb), (&twin, own), (&twin, argb)] {
            let view = s.compose(m.components);
            assert!(!is_identity(view), "{m:?} {s:?}");
            let p = swizzle_only.profile();
            assert_eq!(
                opaque_black_border(rgba, m, s, p),
                Ok(OpaqueBlackBorder::Mapped(view)),
                "{m:?} {s:?}"
            );
            assert_eq!(
                check_border_view(true, rgba, m, s, p),
                Err(needs(&[VkFeature::BorderColorSwizzleFromImage])),
                "{m:?} {s:?}"
            );
            let p = plain.profile();
            assert_eq!(
                opaque_black_border(rgba, m, s, p),
                Err(needs(&[VkFeature::BorderColorSwizzle])),
                "{m:?} {s:?}"
            );
            assert_eq!(
                check_border_view(true, rgba, m, s, p),
                Err(needs(&[
                    VkFeature::BorderColorSwizzle,
                    VkFeature::BorderColorSwizzleFromImage
                ])),
                "{m:?} {s:?}"
            );
        }
        // The mapping the sampler carries is the composed view's: the twin's red/blue exchange,
        // then the swizzle.
        assert_eq!(
            argb.compose(twin.components),
            vk::ComponentMapping {
                r: S::A,
                g: S::B,
                b: S::G,
                a: S::R,
            }
        );
    }

    /// Vulkan's identity mapping is `IDENTITY` or the component's own channel.
    #[test]
    fn identity_mappings_keep_the_device_s_opaque_black() {
        use vk::ComponentSwizzle as S;
        let own = vk::ComponentMapping {
            r: S::IDENTITY,
            g: S::G,
            b: S::IDENTITY,
            a: S::A,
        };
        let black = BORDER_COLORS[1];
        let rgba = vk::Format::R8G8B8A8_UNORM;
        assert!(is_identity(own) && device_opaque_black(rgba, own, false) == black);
        let splat = vk::ComponentMapping {
            r: S::A,
            g: S::A,
            b: S::A,
            a: S::A,
        };
        assert!(!is_identity(splat) && device_opaque_black(rgba, splat, false) == [true; 4]);
    }

    /// A sampler's word holds each descriptor field's raw Metal value at its own bits, so
    /// `MTLSamplerDescriptor`'s defaults are 0; the fields are disjoint and make up
    /// [`SamplerWord::FIELDS`]; the LOD clamps change it by the levels a nearest mip filter reads
    /// at them alone, and `lodAverage` and the anisotropy's degree above 1 not at all.
    #[test]
    fn a_sampler_word_holds_each_field_at_its_bits() {
        type W = SamplerWord;
        let of = |f: &dyn Fn(&mut SamplerDescriptor)| {
            let mut d = metal_default();
            f(&mut d);
            W::of(&d).0
        };
        assert_eq!(W::of(&metal_default()), SamplerWord(0));
        let fields = [
            W::PIXEL_COORDINATES,
            W::MIN_LINEAR,
            W::MAG_LINEAR,
            W::MIP_FILTER,
            W::ADDRESS[0],
            W::ADDRESS[1],
            W::ADDRESS[2],
            W::BORDER,
            W::COMPARE,
            W::REDUCTION,
            W::ANISOTROPIC,
            W::LEVEL_MIN,
            W::LEVEL_MAX,
            W::CUBIC,
        ];
        let mut union = 0;
        for f in fields {
            assert_eq!(union & f, 0, "{f:#x} overlaps");
            union |= f;
        }
        assert_eq!(union, W::FIELDS);
        // `CUBIC` is the word's highest bit, which a function tests as `word >= CUBIC`.
        assert_eq!(W::FIELDS >> 30, 1);
        assert_eq!(of(&|d| d.min_filter = SamplerMinMagFilter::Cubic), W::CUBIC);
        assert_eq!(of(&|d| d.mag_filter = SamplerMinMagFilter::Cubic), W::CUBIC);
        assert_eq!(
            of(&|d| d.normalized_coordinates = false),
            W::PIXEL_COORDINATES
        );
        assert_eq!(
            of(&|d| d.min_filter = SamplerMinMagFilter::Linear),
            W::MIN_LINEAR
        );
        assert_eq!(
            of(&|d| d.mag_filter = SamplerMinMagFilter::Linear),
            W::MAG_LINEAR
        );
        for &m in SamplerMipFilter::ALL {
            let w = of(&|d| d.mip_filter = m);
            assert_eq!(w, m.as_raw() << W::MIP_FILTER_SHIFT, "{m:?}");
            assert_eq!(w & !W::MIP_FILTER, 0, "{m:?}");
        }
        for &mode in SamplerAddressMode::ALL {
            let axes = [
                of(&|d| d.s_address_mode = mode),
                of(&|d| d.t_address_mode = mode),
                of(&|d| d.r_address_mode = mode),
            ];
            for (axis, w) in axes.into_iter().enumerate() {
                assert_eq!(
                    w,
                    mode.as_raw() << W::ADDRESS_SHIFT[axis],
                    "{axis} {mode:?}"
                );
                assert_eq!(w & !W::ADDRESS[axis], 0, "{axis} {mode:?}");
            }
        }
        for &c in SamplerBorderColor::ALL {
            let w = of(&|d| d.border_color = c);
            assert_eq!(w, c.as_raw() << W::BORDER_SHIFT, "{c:?}");
            assert_eq!(w & !W::BORDER, 0, "{c:?}");
        }
        for &f in CompareFunction::ALL {
            let w = of(&|d| d.compare_function = f);
            assert_eq!(w, f.as_raw() << W::COMPARE_SHIFT, "{f:?}");
            assert_eq!(w & !W::COMPARE, 0, "{f:?}");
        }
        for &r in SamplerReductionMode::ALL {
            let w = of(&|d| d.reduction_mode = r);
            assert_eq!(w, r.as_raw() << W::REDUCTION_SHIFT, "{r:?}");
            assert_eq!(w & !W::REDUCTION, 0, "{r:?}");
        }
        for (anisotropy, want) in [(0, 0), (1, 0), (2, W::ANISOTROPIC), (16, W::ANISOTROPIC)] {
            assert_eq!(of(&|d| d.max_anisotropy = anisotropy), want, "{anisotropy}");
        }
        // The clamp's levels under a nearest mip filter.
        assert_eq!(of(&|d| d.lod_min_clamp = 2.0), 2 << W::LEVEL_MIN_SHIFT);
        assert_eq!(
            of(&|d| d.lod_max_clamp = 3.0),
            (LAST_WORD_LEVEL - 3) << W::LEVEL_MAX_SHIFT
        );
        assert_eq!(of(&|d| d.lod_average = true), 0);
    }

    /// The word answers the pixel samplers' predicates. A sampler with normalized coordinates
    /// answers neither. A pixel sampler's sample at its texel's centre
    /// ([`SamplerWord::nearest_pixel`]) needs one isotropic level (no mip filter, not
    /// anisotropic) and nearest filters unless the texture is integer, whatever its address modes
    /// and border.
    #[test]
    fn a_sampler_word_answers_the_pixel_predicates() {
        use SamplerMinMagFilter as F;
        let holds = |d: &SamplerDescriptor, mask: u32| {
            (SamplerWord::of(d).0 & mask) == SamplerWord::PIXEL_COORDINATES
        };
        let centre =
            |d: &SamplerDescriptor, integer: bool| holds(d, SamplerWord::nearest_pixel(integer));
        let normalized = metal_default();
        assert!(!holds(&normalized, SamplerWord::PIXEL_COORDINATES));
        let mut pixel = metal_default();
        pixel.normalized_coordinates = false;
        assert!(holds(&pixel, SamplerWord::PIXEL_COORDINATES));
        for integer in [false, true] {
            assert!(!centre(&normalized, integer));
            assert!(centre(&pixel, integer), "{integer}");
        }
        for (min, mag) in [
            (F::Nearest, F::Nearest),
            (F::Linear, F::Nearest),
            (F::Nearest, F::Linear),
            (F::Linear, F::Linear),
        ] {
            let mut d = pixel;
            d.min_filter = min;
            d.mag_filter = mag;
            let nearest = min == F::Nearest && mag == F::Nearest;
            assert_eq!(centre(&d, false), nearest, "{min:?} {mag:?}");
            assert!(centre(&d, true), "{min:?} {mag:?}");
        }
        for mip in [SamplerMipFilter::Nearest, SamplerMipFilter::Linear] {
            let mut d = pixel;
            d.mip_filter = mip;
            assert!(!centre(&d, true), "{mip:?}");
        }
        let mut d = pixel;
        d.max_anisotropy = 2;
        assert!(!centre(&d, true));
        for &mode in SamplerAddressMode::ALL {
            for &border in SamplerBorderColor::ALL {
                let mut d = pixel;
                (d.s_address_mode, d.t_address_mode, d.r_address_mode) = (mode, mode, mode);
                d.border_color = border;
                assert!(centre(&d, false), "{mode:?} {border:?}");
            }
        }
    }

    /// The anisotropic emulation hands its level for level 0 by the device's sampler's anisotropy
    /// (`lower::sampler::anisotropic`'s `minified_level`): a runtime sampler's by the filters its
    /// word states (`device_anisotropy_by_filters`), which must be `vk_sampler`'s own flag
    /// (`anisotropyEnable`, `max_anisotropy` set) for every sampler a word of those filters
    /// states, on every committed profile, for the use the emulation reaches (no comparison, no
    /// integer texture), whatever the sampler's other fields.
    #[test]
    fn the_word_filters_state_the_device_samplers_anisotropy() {
        use SamplerMinMagFilter as F;
        let mut checked = 0;
        for (stem, desc) in committed_descriptions() {
            let mut desc = desc;
            desc.features.insert(VkFeature::SamplerAnisotropy);
            desc.limits.max_sampler_anisotropy_bits = 16f32.to_bits();
            let dev = Device::new(desc, DevicePolicy::default()).unwrap();
            {
                let use_ = SamplerUse::new(false, false);
                let (linear, nearest) = device_anisotropy_by_filters(use_, dev.profile());
                for (min, mag) in [
                    (F::Nearest, F::Nearest),
                    (F::Nearest, F::Linear),
                    (F::Linear, F::Nearest),
                    (F::Linear, F::Linear),
                ] {
                    for mip in [
                        SamplerMipFilter::NotMipmapped,
                        SamplerMipFilter::Nearest,
                        SamplerMipFilter::Linear,
                    ] {
                        for (degree, (lo, hi)) in [2, 3, 8, 16].into_iter().zip([
                            (0.0, f32::MAX),
                            (0.25, 4.0),
                            (1.0, 2.5),
                            (0.0, 0.0),
                        ]) {
                            let d = SamplerDescriptor {
                                min_filter: min,
                                mag_filter: mag,
                                mip_filter: mip,
                                max_anisotropy: degree,
                                lod_min_clamp: lo,
                                lod_max_clamp: hi,
                                s_address_mode: SamplerAddressMode::Repeat,
                                ..metal_default()
                            };
                            let Ok(made) = vk_sampler(&d, use_, dev.profile()) else {
                                continue;
                            };
                            let word = SamplerWord::of(&d).0;
                            let both = SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR;
                            let stated = match word & both == both {
                                true => linear,
                                false => nearest,
                            };
                            assert_eq!(
                                stated,
                                made.max_anisotropy.is_some(),
                                "{stem}: {d:?} under {use_:?}"
                            );
                            assert_eq!(stated, device_anisotropy(&d, use_, dev.profile()));
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 0);
    }
}
