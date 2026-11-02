//! Constexpr samplers: their canonical state words, the symbol a combined image sampler names them
//! by, and the `VkSampler` link bakes into the layout.
//!
//! A constexpr sampler reaches a binding only through `SamplerRef::Constexpr(words)`, so its
//! identity is its decoded state, written back as the canonical words [`state_words`] produces
//! (the inverse of `SamplerState::from_words` on every decodable state). The `VkSampler` is the
//! state as an `MTLSamplerDescriptor` through the one mapping runtime samplers use
//! (`metal::sampler::vk_sampler`), plus the two fields only a constexpr sampler carries: the LOD
//! bias (`mipLodBias`, within `maxSamplerLodBias`) and Metal's bicubic filter. Metal's bicubic
//! filter is emulated by fetches in the shader (`lower::sampler::bicubic`), because what it does
//! with a texel depends on the bound pixel format, which an immutable `VK_FILTER_CUBIC_EXT` sampler
//! cannot follow (and no committed profile has `VK_EXT_filter_cubic`); a gather through a bicubic
//! sampler ignores the filter, so the `VkSampler` of a bicubic sampler has nearest filters and its
//! address modes. Pixel coordinates are a sampler property on the Vulkan side only where a binding
//! reads them through the device's unnormalized sampler (`SamplerUse::unnormalized`, a linear
//! sample's); elsewhere the shader hands a pixel coordinate over the texture's size to a normalized
//! `VkSampler` (`lower::sampler`'s pixel coordinates).

use crate::air::abi::{
    AddressMode, BorderColor, CompareFunction, Coordinates, Filter, MipFilter, Reduction,
    SamplerState,
};
use crate::bind::state::SamplerDescriptor;
use crate::device::requirements::{VkFeature, VkLimit};
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, InternalError, InternalKind, Unsupported};
use crate::metal::sampler::SamplerUse;
use crate::metal::{
    SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
    SamplerReductionMode,
};
use crate::vk;

/// The canonical AIR words of `s`: bit 63 of the first word clear, and every field in the
/// encoding `SamplerState::from_words` decodes (`from_words(state_words(s)) == Some(s)` for every
/// decoded state).
pub fn state_words(s: &SamplerState) -> [u64; 2] {
    let address = |m: AddressMode| -> u64 {
        match m {
            // Code 0 is clamp-to-zero with a transparent-black border, clamp-to-border otherwise.
            AddressMode::ClampToZero | AddressMode::ClampToBorder => 0,
            AddressMode::ClampToEdge => 1,
            AddressMode::Repeat => 2,
            AddressMode::MirroredRepeat => 3,
        }
    };
    let filter = |f: Filter| -> u64 {
        match f {
            Filter::Nearest => 0,
            Filter::Linear => 1,
            Filter::Bicubic => 2,
        }
    };
    let mip = match s.mip_filter {
        MipFilter::None => 0,
        MipFilter::Nearest => 1,
        MipFilter::Linear => 2,
    };
    let compare = match s.compare {
        CompareFunction::None => 0,
        CompareFunction::Less => 1,
        CompareFunction::LessEqual => 2,
        CompareFunction::Greater => 3,
        CompareFunction::GreaterEqual => 4,
        CompareFunction::Equal => 5,
        CompareFunction::NotEqual => 6,
        CompareFunction::Always => 7,
        CompareFunction::Never => 8,
    };
    let border = match s.border_color {
        BorderColor::TransparentBlack => 0,
        BorderColor::OpaqueBlack => 1,
        BorderColor::OpaqueWhite => 2,
    };
    let reduction = match s.reduction {
        Reduction::WeightedAverage => 0,
        Reduction::Minimum => 1,
        Reduction::Maximum => 2,
    };
    let pixel = u64::from(s.coordinates == Coordinates::Pixel);
    let aniso = u64::from(s.max_anisotropy.saturating_sub(1)) & 0xf;
    let w0 = address(s.address_s)
        | address(s.address_t) << 3
        | address(s.address_r) << 6
        | filter(s.mag_filter) << 9
        | filter(s.min_filter) << 11
        | mip << 13
        | pixel << 15
        | compare << 16
        | aniso << 20
        | u64::from(s.lod_min_clamp.0) << 24
        | u64::from(s.lod_max_clamp.0) << 40
        | border << 56
        | reduction << 58;
    [w0, u64::from(s.lod_bias.0)]
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// The state the translation lowers constexpr sampler `state` as, or its refusal:
///
/// - with pixel coordinates and a compare function (any but `never`, the word's default), it is
///   `NotRepresentable(PixelSamplerCompare)`: Apple's compiler rejects such a sampler ("invalid
///   values combination in sampler initialization"), so Metal gives it no
///   answer;
/// - a minimum or maximum reduction reads as the weighted average, as Metal reads it on an Apple
///   GPU whose `supportsSamplerReductionMode` answers no (an Apple M4 on macOS 27.0.1: every
///   channel of `rgba8Unorm`, `rgba16Float` and `r32Float` linear samples).
pub fn lowered(state: &SamplerState) -> Result<SamplerState, Error> {
    let compares = !matches!(
        state.compare,
        CompareFunction::None | CompareFunction::Never
    );
    if state.coordinates == Coordinates::Pixel && compares {
        return Err(Unsupported::not_representable(Feature::PixelSamplerCompare).into());
    }
    Ok(SamplerState {
        reduction: Reduction::WeightedAverage,
        ..*state
    })
}

/// The decoded state of canonical words `words`.
pub fn state_of(words: [u64; 2]) -> Result<SamplerState, Error> {
    SamplerState::from_words(words)
        .ok_or_else(|| invariant("a constexpr sampler symbol with undecodable words"))
}

/// The level, in sixty-fourths, a clamped LOD query outside a fragment function answers through
/// constexpr sampler `s` before the texture's last level bounds it:
/// `metal::sampler::query_min_level` of its descriptor, its `lod_clamp` minimum a `half`'s value
/// as Metal reads it (`lod_clamp(0.3, 4)` answered 19/64). A level bias of its
/// own is not measured.
pub fn query_min_level(s: &SamplerState) -> Result<u32, Error> {
    if s.lod_bias.to_f32() != 0.0 {
        return Err(Unsupported::not_implemented(Feature::ImplicitLodOutsideFragment).into());
    }
    Ok(crate::metal::sampler::query_min_level(&descriptor(s)?))
}

fn min_mag(f: Filter) -> Result<SamplerMinMagFilter, Error> {
    match f {
        Filter::Nearest => Ok(SamplerMinMagFilter::Nearest),
        Filter::Linear => Ok(SamplerMinMagFilter::Linear),
        // A sample through a bicubic sampler is emulated by fetches and a gather ignores the
        // filter: the immutable sampler's filters select nothing, so they are nearest.
        Filter::Bicubic => Ok(SamplerMinMagFilter::Nearest),
    }
}

fn address(m: AddressMode) -> SamplerAddressMode {
    match m {
        AddressMode::ClampToZero => SamplerAddressMode::ClampToZero,
        AddressMode::ClampToEdge => SamplerAddressMode::ClampToEdge,
        AddressMode::Repeat => SamplerAddressMode::Repeat,
        AddressMode::MirroredRepeat => SamplerAddressMode::MirrorRepeat,
        AddressMode::ClampToBorder => SamplerAddressMode::ClampToBorderColor,
    }
}

/// The canonical words of constexpr sampler `words` sampling a texture of address-mode map `map`:
/// each address mode the map's (`AddressModeMap::constexpr_mode`), as the binder maps a runtime
/// sampler's, so its immutable sampler and every emulation that reads its modes follow the map. A
/// mode a constexpr sampler cannot state is `NotImplemented(SamplerAddressMode)`.
pub fn mapped_words(
    words: [u64; 2],
    map: &crate::metal::sampler::AddressModeMap,
) -> Result<[u64; 2], Error> {
    let s = state_of(words)?;
    let to = |m: AddressMode| -> Result<AddressMode, Error> {
        match map.constexpr_mode(address(m)) {
            Some(mode) if mode == address(m) => Ok(m),
            Some(SamplerAddressMode::ClampToEdge) => Ok(AddressMode::ClampToEdge),
            Some(SamplerAddressMode::Repeat) => Ok(AddressMode::Repeat),
            Some(SamplerAddressMode::MirrorRepeat) => Ok(AddressMode::MirroredRepeat),
            _ => Err(Unsupported::not_implemented(Feature::SamplerAddressMode).into()),
        }
    };
    Ok(state_words(&SamplerState {
        address_s: to(s.address_s)?,
        address_t: to(s.address_t)?,
        address_r: to(s.address_r)?,
        ..s
    }))
}

/// The address modes on the `s`, `t` and `r` coordinates that constexpr sampler words `words`
/// state, as Metal names them (`address::clamp_to_zero` is `ClampToZero`, a border color's code 0
/// `ClampToBorderColor`); `None` for words that state no sampler.
pub fn address_modes(words: [u64; 2]) -> Option<[SamplerAddressMode; 3]> {
    let s = SamplerState::from_words(words)?;
    Some([
        address(s.address_s),
        address(s.address_t),
        address(s.address_r),
    ])
}

/// The `MTLSamplerDescriptor` a constexpr sampler state states (the fields Metal's runtime
/// descriptor shares with it).
fn descriptor(s: &SamplerState) -> Result<SamplerDescriptor, Error> {
    use crate::metal::CompareFunction as C;
    Ok(SamplerDescriptor {
        min_filter: min_mag(s.min_filter)?,
        mag_filter: min_mag(s.mag_filter)?,
        mip_filter: match s.mip_filter {
            MipFilter::None => SamplerMipFilter::NotMipmapped,
            MipFilter::Nearest => SamplerMipFilter::Nearest,
            MipFilter::Linear => SamplerMipFilter::Linear,
        },
        s_address_mode: address(s.address_s),
        t_address_mode: address(s.address_t),
        r_address_mode: address(s.address_r),
        border_color: match s.border_color {
            BorderColor::TransparentBlack => SamplerBorderColor::TransparentBlack,
            BorderColor::OpaqueBlack => SamplerBorderColor::OpaqueBlack,
            BorderColor::OpaqueWhite => SamplerBorderColor::OpaqueWhite,
        },
        normalized_coordinates: s.coordinates == Coordinates::Normalized,
        lod_min_clamp: s.lod_min_clamp.to_f32(),
        lod_max_clamp: s.lod_max_clamp.to_f32(),
        lod_average: false,
        max_anisotropy: s.max_anisotropy,
        // `None` is never written by the frontend; it and an unspecified comparison are `Never`.
        compare_function: match s.compare {
            CompareFunction::None | CompareFunction::Never => C::Never,
            CompareFunction::Less => C::Less,
            CompareFunction::LessEqual => C::LessEqual,
            CompareFunction::Greater => C::Greater,
            CompareFunction::GreaterEqual => C::GreaterEqual,
            CompareFunction::Equal => C::Equal,
            CompareFunction::NotEqual => C::NotEqual,
            CompareFunction::Always => C::Always,
        },
        reduction_mode: match s.reduction {
            Reduction::WeightedAverage => SamplerReductionMode::WeightedAverage,
            Reduction::Minimum => SamplerReductionMode::Minimum,
            Reduction::Maximum => SamplerReductionMode::Maximum,
        },
    })
}

/// Whether the device's unnormalized sampler reads constexpr sampler state `s` in pixel
/// coordinates as it is (`metal::sampler::unnormalized_reads` of its descriptor).
pub(crate) fn unnormalized_reads(s: &SamplerState) -> Result<bool, Error> {
    Ok(crate::metal::sampler::unnormalized_reads(&descriptor(s)?))
}

/// The immutable `VkSampler` of constexpr sampler `words` used as `use_` on `profile`. Every
/// refusal is a feature check of the constexpr state (bicubic filtering, anisotropy, border colors,
/// mirror-clamp, comparison and any LOD bias where the portability subset withholds
/// `mutableComparisonSamplers` or `samplerMipLodBias`, LOD bias beyond `maxSamplerLodBias`), so
/// `select` runs it for every binding it declares and link's call cannot fail for a module compiled
/// for the same profile.
pub fn constexpr_desc(
    words: [u64; 2],
    use_: SamplerUse,
    profile: &DeviceProfile,
) -> Result<vk::SamplerDesc, Error> {
    let s = state_of(words)?;
    let mut desc = crate::metal::sampler::vk_sampler(&descriptor(&s)?, use_, profile)?;
    // A sampler that reads texel coordinates reads level 0 by one filter: a bias selects nothing.
    let bias = match use_.unnormalized {
        true => 0.0,
        false => s.lod_bias.to_f32(),
    };
    if bias.is_nan() {
        return Err(Unsupported::not_representable(Feature::SamplerLodBias).into());
    }
    if bias != 0.0 {
        // The portability subset may withhold any nonzero bias.
        let feature = VkFeature::SamplerMipLodBias;
        if profile.fields().enabled.portability_withholds(feature) {
            return Err(
                Unsupported::needs(Feature::SamplerLodBias, Capability::Feature(feature)).into(),
            );
        }
        let limit = f32::from_bits(profile.fields().limits.max_sampler_lod_bias_bits);
        if bias.abs() > limit {
            // Float limits are compared in whole units, as the profile check reports them.
            return Err(Unsupported::needs(
                Feature::SamplerLodBias,
                Capability::Limit {
                    limit: VkLimit::MaxSamplerLodBias,
                    required: bias.abs().ceil() as u64,
                    available: if limit > 0.0 { limit.floor() as u64 } else { 0 },
                },
            )
            .into());
        }
        desc.mip_lod_bias = bias;
    }
    Ok(desc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::abi::Half;
    use crate::device::DevicePolicy;
    use crate::error::Needs;

    fn profile(stem: &str) -> DeviceProfile {
        crate::pipeline::testing::profile(stem, DevicePolicy::default()).expect("profile")
    }

    /// The words of the public fixtures' constexpr samplers (bit 63 clear).
    const FIXTURE_WORDS: [u64; 3] = [34901797601036873, 34901797601023122, 34901797601017929];

    /// `state_words` is the inverse of the decoder on every state it produces: the fixture words
    /// and every single-field variation of them.
    #[test]
    fn state_words_invert_the_decoder() {
        let mut words: Vec<[u64; 2]> = FIXTURE_WORDS.iter().map(|w| [*w, 0]).collect();
        let base = FIXTURE_WORDS[0];
        // Each field over its codes (address, filters, mip, coordinates, compare, anisotropy,
        // border, reduction), and a LOD bias.
        let fields: [(u32, u64, u64); 12] = [
            (0, 0x7, 4),
            (3, 0x7, 4),
            (6, 0x7, 4),
            (9, 0x3, 3),
            (11, 0x3, 3),
            (13, 0x3, 3),
            (15, 0x1, 2),
            (16, 0xf, 9),
            (20, 0xf, 16),
            (24, 0xffff, 1),
            (56, 0x3, 3),
            (58, 0x3, 3),
        ];
        for (shift, mask, codes) in fields {
            for code in 0..codes {
                words.push([(base & !(mask << shift)) | code << shift, 0]);
            }
        }
        words.push([base, 0x3c00]);
        let mut decoded = 0;
        for w in words {
            let Some(s) = SamplerState::from_words(w) else {
                continue;
            };
            decoded += 1;
            assert_eq!(SamplerState::from_words(state_words(&s)), Some(s), "{w:x?}");
            assert_eq!(state_words(&s), w, "canonical words of {s:?}");
        }
        assert!(decoded > 30, "only {decoded} states decoded");
    }

    /// A constexpr sampler's address modes read back per axis as Metal names them (the runtime's
    /// address-mode map compares them): code 0 is `ClampToZero` under a transparent-black border
    /// and `ClampToBorderColor` under an opaque one. A runtime sampler's slot and undecodable words
    /// state none.
    #[test]
    fn a_constexpr_samplers_modes_read_as_metal_names_them() {
        use crate::pipeline::SamplerRef;
        let base = FIXTURE_WORDS[0] & !(0x1ff | 0x3 << 56);
        for border in 0..3u64 {
            let metal = |code: u64| match code {
                0 if border == 0 => SamplerAddressMode::ClampToZero,
                0 => SamplerAddressMode::ClampToBorderColor,
                1 => SamplerAddressMode::ClampToEdge,
                2 => SamplerAddressMode::Repeat,
                _ => SamplerAddressMode::MirrorRepeat,
            };
            for (s, t, r) in [(0, 1, 2), (1, 1, 1), (2, 3, 0), (3, 0, 1), (1, 2, 3)] {
                let words = [base | s | t << 3 | r << 6 | border << 56, 0];
                assert_eq!(
                    SamplerRef::Constexpr(words).constexpr_address_modes(),
                    Some([metal(s), metal(t), metal(r)]),
                    "{words:x?}"
                );
            }
        }
        assert_eq!(SamplerRef::Slot(0).constexpr_address_modes(), None);
        assert_eq!(
            SamplerRef::Constexpr([base | 7, 0]).constexpr_address_modes(),
            None
        );
    }

    fn state(words: u64) -> SamplerState {
        SamplerState::from_words([words, 0]).expect("decodes")
    }

    /// The fixture samplers map to normalized Vulkan samplers with their filters and modes.
    #[test]
    fn a_constexpr_sampler_is_a_normalized_vk_sampler() {
        let p = profile("moltenvk-m3");
        let s = state(FIXTURE_WORDS[0]);
        let d = constexpr_desc(state_words(&s), SamplerUse::default(), &p).expect("maps");
        assert_eq!(d.mip_lod_bias, 0.0);
        assert_eq!(d.compare, None);
        assert_eq!(
            d.min_filter,
            match s.min_filter {
                Filter::Linear => vk::Filter::LINEAR,
                _ => vk::Filter::NEAREST,
            }
        );
    }

    /// The constexpr feature checks are refusals at compile time.
    #[test]
    fn constexpr_feature_checks_refuse() {
        let p = profile("vk12-floor");
        // A bicubic sampler's filters are nearest placeholders: its samples are fetches.
        let mut s = state(FIXTURE_WORDS[0]);
        s.mag_filter = Filter::Bicubic;
        s.min_filter = Filter::Bicubic;
        let d = constexpr_desc(state_words(&s), SamplerUse::default(), &p).expect("maps");
        assert_eq!(
            (d.min_filter, d.mag_filter),
            (vk::Filter::NEAREST, vk::Filter::NEAREST)
        );
        // Anisotropy maps on every device: its Vulkan sampler is isotropic without the feature.
        let mut s = state(FIXTURE_WORDS[0]);
        s.max_anisotropy = 8;
        let d = constexpr_desc(state_words(&s), SamplerUse::default(), &p).expect("maps");
        let aniso_ok = p.fields().images.sampler_anisotropy;
        assert_eq!(d.max_anisotropy.is_some(), aniso_ok, "{d:?}");
        // A bias beyond every device's limit.
        let mut s = state(FIXTURE_WORDS[0]);
        s.lod_bias = Half(0x7bff); // 65504
        match constexpr_desc(state_words(&s), SamplerUse::default(), &p) {
            Err(Error::Unsupported(Unsupported {
                feature: Feature::SamplerLodBias,
                needs: Needs::AnyOf(_),
                ..
            })) => {}
            other => panic!("{other:?}"),
        }
        // A small bias is the sampler's `mipLodBias`.
        let mut s = state(FIXTURE_WORDS[0]);
        s.lod_bias = Half(0x3c00); // 1.0
        let d = constexpr_desc(state_words(&s), SamplerUse::default(), &p).expect("maps");
        assert_eq!(d.mip_lod_bias, 1.0);
        // MoltenVK's portability subset withholds `samplerMipLodBias`, so any nonzero bias is
        // refused there (`VUID-VkSamplerCreateInfo-samplerMipLodBias-04467`); a zero one is not a
        // bias.
        let mvk = profile("moltenvk-m3");
        assert!(mvk
            .fields()
            .enabled
            .portability_withholds(VkFeature::SamplerMipLodBias));
        assert_eq!(
            constexpr_desc(state_words(&s), SamplerUse::default(), &mvk),
            Err(Unsupported::needs(
                Feature::SamplerLodBias,
                Capability::Feature(VkFeature::SamplerMipLodBias)
            )
            .into())
        );
        s.lod_bias = Half(0);
        assert!(constexpr_desc(state_words(&s), SamplerUse::default(), &mvk).is_ok());
    }

    /// Depth comparison and integer borders come from the use.
    #[test]
    fn the_use_decides_compare_and_integer_borders() {
        let p = profile("moltenvk-m3");
        let mut s = state(FIXTURE_WORDS[0]);
        s.compare = CompareFunction::Less;
        s.address_s = AddressMode::ClampToBorder;
        s.border_color = BorderColor::OpaqueWhite;
        let w = state_words(&s);
        let plain = constexpr_desc(w, SamplerUse::new(false, false), &p).expect("maps");
        assert_eq!(plain.compare, None);
        assert_eq!(plain.border_color, vk::BorderColor::FLOAT_OPAQUE_WHITE);
        let cmp = constexpr_desc(w, SamplerUse::new(true, true), &p).expect("maps");
        assert_eq!(cmp.compare, Some(vk::CompareOp::LESS));
        assert_eq!(cmp.border_color, vk::BorderColor::INT_OPAQUE_WHITE);
    }
}
