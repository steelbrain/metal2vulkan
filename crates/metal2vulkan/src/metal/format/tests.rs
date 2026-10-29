//! Pixel-format mapping tests: table totality and round trips (device-independent), selection on
//! every committed profile, and fallbacks on synthetic descriptions.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::device::description::{FormatProperties, FormatSupport};
use crate::device::requirements::{ApiVersion, VkExtension, VkFeature};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Needs;

/// Every committed profile's description (`validation/profiles/`, read-only), sorted by stem.
pub(crate) fn committed_descriptions() -> Vec<(String, DeviceDescription)> {
    crate::spirv::testing::committed_descriptions()
        .unwrap_or_else(|e| panic!("the committed profiles: {e}"))
        .to_vec()
}

/// Every committed profile's device, by file stem, in order.
pub(crate) fn committed_devices() -> Vec<(String, Device)> {
    committed_descriptions()
        .into_iter()
        .map(|(stem, desc)| {
            let device = Device::new(desc, DevicePolicy::default()).expect("device derives");
            (stem, device)
        })
        .collect()
}

/// The `vk12-floor` description with its format table replaced by `formats`.
fn floor_with(formats: &[(vk::Format, vk::FormatFeatureFlags)]) -> DeviceDescription {
    let (_, mut desc) = committed_descriptions()
        .into_iter()
        .find(|(s, _)| s == "vk12-floor")
        .expect("vk12-floor is committed");
    desc.formats = FormatSupport::default();
    for &(format, optimal) in formats {
        desc.formats.formats.insert(
            format,
            FormatProperties::from_legacy(
                vk::FormatFeatureFlags::empty(),
                optimal,
                vk::FormatFeatureFlags::empty(),
            ),
        );
    }
    desc
}

fn device(desc: DeviceDescription) -> Device {
    Device::new(desc, DevicePolicy::default()).expect("device derives")
}

fn usages(list: &[FormatUsage]) -> BTreeSet<FormatUsage> {
    list.iter().copied().collect()
}

const SAMPLED: vk::FormatFeatureFlags = vk::FormatFeatureFlags::SAMPLED_IMAGE;

/// The Metal formats without a candidate, and why. Golden: a change is deliberate.
#[test]
fn refused_formats_are_exactly_the_documented_set() {
    use PixelFormat as P;
    let mut refused: BTreeMap<&str, Vec<PixelFormat>> = BTreeMap::new();
    for &pf in PixelFormat::ALL {
        let class = match repr(pf) {
            Repr::Candidates(..) => continue,
            Repr::NotImplemented => "not_implemented",
            Repr::NotAFormat => "placeholder",
        };
        refused.entry(class).or_default().push(pf);
    }
    let count = |class: &str| refused.get(class).map_or(0, Vec::len);
    assert_eq!(refused["placeholder"], [P::Invalid, P::Unspecialized]);
    // The two-plane extended-range pair is held raw, the 4:2:2 pair through its decoded twin.
    // The 68 two-plane YCbCr formats but `YCBCR8_420_2P`, and the 3 others Metal makes no
    // texture of from a descriptor alone.
    // + ASTC's 6 formats of a 4x2 or 8x4 footprint, which no Vulkan block has.
    // - `YCBCR10_420_2P`, `RGB10_420_2P` and the `_sRGB` twins of the 4:2:0 converted pair, made
    // over a surface.
    // - `RGB8_420_2P`, made over a surface.
    // - the 4:2:2 and 4:4:4 converted formats of 8 and 10 bits and their `_sRGB` twins.
    // - the 4:2:2 and 4:4:4 code twins of 8 and 10 bits.
    // - the packed 10-bit converted formats and their `_sRGB` twins.
    // + the 80 other formats Metal names (`formats_named_not_held!` and the padded four).
    // - `R10Unorm_X6` and `RG10Unorm_X12`, held raw.
    // - their `_sRGB` twins, held raw.
    // - the eleven RGBX formats, held in their RGBA twins.
    // - the three bfloat formats, held raw.
    // - `RGB10A2Unorm_sRGB`, held raw.
    // - `BGRA4Unorm`, held in the 4-bit word of its bits.
    assert_eq!(count("not_implemented"), 45 + 3 + 6 + 60, "{refused:?}");
    assert!(refused["not_implemented"].iter().all(|&pf| {
        (ycbcr_format(pf).is_some() && !ycbcr(pf))
            || matches!(pf, P::RGBA16FloatXR | P::AnyDepth | P::AnyStencil)
            || matches!(pf, astc_without_a_vulkan_block!())
            || matches!(pf, formats_named_not_held!())
    }));
}

/// Every candidate is a named, non-`UNDEFINED` registry format, and the direct formats are
/// one-to-one: two Metal formats share one only when one is a stencil view of the other. A format
/// held only through its decoded twin's view has none.
#[test]
fn direct_formats_round_trip() {
    let mut back: BTreeMap<vk::Format, Vec<PixelFormat>> = BTreeMap::new();
    for &pf in PixelFormat::ALL {
        let Repr::Candidates(aspects, candidates) = make_repr(pf) else {
            continue;
        };
        assert!(!candidates.is_empty(), "{pf:?}");
        assert!(!aspects.is_empty(), "{pf:?}");
        for c in &candidates {
            assert!(c.format.is_known(), "{pf:?} -> {:?}", c.format);
            assert_ne!(c.format, vk::Format::UNDEFINED, "{pf:?}");
        }
        // A PVRTC RGB or 4:2:2 format, held through its decoded twin's view alone, has no
        // direct format: no Vulkan format holds its blocks as Metal reads them.
        if candidates[0].emulation == FormatEmulation::Decoded {
            assert_eq!(direct_format(pf), None, "{pf:?}");
            let pvrtc = crate::bind::PvrtcDecode::of(pf).is_some();
            assert!(
                pvrtc || crate::bind::ChromaDecode::of(pf).is_some(),
                "{pf:?}"
            );
            continue;
        }
        assert_eq!(direct_format(pf), Some(candidates[0].format));
        // `storage_image_choice` relies on it: a storage image ignores the view's mapping. The
        // first candidate is direct but for an extended-range format's raw bytes and
        // `ARGB8Unorm`'s rotated ones, whose storage choice rotates in the shader; both share their
        // format with the format of the same bytes. A YCbCr format's image holds its converted
        // values, in the format of the same values.
        match candidates[0].emulation {
            FormatEmulation::Direct => back.entry(candidates[0].format).or_default().push(pf),
            FormatEmulation::ExtendedRange => {}
            FormatEmulation::RotatedView => assert_eq!(pf, PixelFormat::ARGB8Unorm),
            FormatEmulation::YCbCr => assert!(ycbcr_format(pf).is_some(), "{pf:?}"),
            // An RGBX format's bytes are its RGBA twin's, which converts in the shader.
            FormatEmulation::XChannel => assert!(x_channel(pf), "{pf:?}"),
            // A bfloat format's codes are held raw, as an extended-range format's are.
            FormatEmulation::Bfloat => {
                assert!(
                    ExtendedRange::of(pf).is_some_and(ExtendedRange::bfloat),
                    "{pf:?}"
                )
            }
            other => panic!("{pf:?}: a first candidate of {other:?}"),
        }
    }
    for (format, metal) in back {
        match metal.as_slice() {
            [_] => {}
            [a, b] => {
                let pair = [a.name(), b.name()];
                assert!(
                    pair == [
                        "MTLPixelFormatDepth24Unorm_Stencil8",
                        "MTLPixelFormatX24_Stencil8"
                    ] || pair
                        == [
                            "MTLPixelFormatDepth32Float_Stencil8",
                            "MTLPixelFormatX32_Stencil8"
                        ],
                    "{format:?} is the direct format of {pair:?}"
                );
            }
            more => panic!("{format:?} is the direct format of {more:?}"),
        }
    }
}

/// The component letters of a Vulkan format name with R and B exchanged
/// (`R5G6B5_UNORM_PACK16` ↔ `B5G6R5_UNORM_PACK16`).
fn swap_red_blue(name: &str) -> String {
    let name = name.strip_prefix("VK_FORMAT_").expect("a format name");
    let (layout, rest) = name
        .split_once('_')
        .expect("format name has a numeric part");
    let layout: String = layout
        .chars()
        .map(|c| match c {
            'R' => 'B',
            'B' => 'R',
            c => c,
        })
        .collect();
    format!("VK_FORMAT_{layout}_{rest}")
}

/// Every swizzled twin is the direct format with red and blue exchanged in the same bit layout
/// (or `A8Unorm`'s single byte held in red), and its mapping moves the channels back.
#[test]
fn swizzled_twins_share_the_bit_layout() {
    use vk::ComponentSwizzle as S;
    for &pf in PixelFormat::ALL {
        let Repr::Candidates(_, candidates) = make_repr(pf) else {
            continue;
        };
        for c in candidates.iter().filter(|c| {
            matches!(
                c.emulation,
                FormatEmulation::SwizzledView
                    | FormatEmulation::RotatedView
                    | FormatEmulation::AlphaInRed
            )
        }) {
            // `ARGB8Unorm`'s bytes through a rotating view.
            if pf == PixelFormat::ARGB8Unorm {
                assert_eq!(c.format, vk::Format::R8G8B8A8_UNORM);
                assert_eq!(
                    (
                        c.components.r,
                        c.components.g,
                        c.components.b,
                        c.components.a
                    ),
                    (S::G, S::B, S::A, S::R)
                );
                continue;
            }
            if pf == PixelFormat::A8Unorm {
                assert_eq!(c.format, vk::Format::R8_UNORM);
                assert_eq!(
                    (
                        c.components.r,
                        c.components.g,
                        c.components.b,
                        c.components.a
                    ),
                    (S::ZERO, S::ZERO, S::ZERO, S::R)
                );
                continue;
            }
            let direct = candidates[0].format.name().unwrap();
            assert_eq!(
                c.format.name().unwrap(),
                swap_red_blue(direct),
                "{pf:?}: twin of {direct}"
            );
            assert_eq!((c.components.r, c.components.b), (S::B, S::R), "{pf:?}");
            assert_eq!(
                (c.components.g, c.components.a),
                (S::IDENTITY, S::IDENTITY),
                "{pf:?}"
            );
        }
    }
}

/// The Metal and Vulkan names agree on the numeric type of every color candidate, and packed
/// formats pair by layout (Metal names from the least significant bit, Vulkan from the most).
#[test]
fn numeric_types_and_packed_layouts_agree() {
    for &pf in PixelFormat::ALL {
        let Repr::Candidates(aspects, candidates) = make_repr(pf) else {
            continue;
        };
        let ty = sampled_type(pf).unwrap();
        if aspects != vk::ImageAspectFlags::COLOR {
            continue;
        }
        // An extended-range format's float texels are held in its bytes' integer format.
        for c in candidates.iter().filter(|c| {
            !matches!(
                c.emulation,
                FormatEmulation::ExtendedRange | FormatEmulation::Bfloat
            )
        }) {
            let name = c.format.name().unwrap();
            let expect = if name.contains("_UINT") {
                SampledType::Uint
            } else if name.contains("_SINT") {
                SampledType::Sint
            } else {
                SampledType::Float
            };
            assert_eq!(ty, expect, "{pf:?} -> {name}");
            // A YCbCr format's image holds the texels Metal reads, an `_sRGB` twin's decoded.
            // And its attachment the function's floats.
            let decoded = matches!(
                c.emulation,
                FormatEmulation::YCbCr
                    | FormatEmulation::YCbCrRendered
                    | FormatEmulation::YCbCrPassRefused
            );
            assert_eq!(
                (pf.name().contains("sRGB") || pf.name().ends_with("SRGB")) && !decoded,
                name.contains("_SRGB"),
                "{pf:?} -> {name}"
            );
        }
    }
    let direct = |pf| direct_format(pf).unwrap();
    assert_eq!(
        direct(PixelFormat::B5G6R5Unorm),
        vk::Format::R5G6B5_UNORM_PACK16
    );
    assert_eq!(
        direct(PixelFormat::BGR5A1Unorm),
        vk::Format::A1R5G5B5_UNORM_PACK16
    );
    assert_eq!(
        direct(PixelFormat::RGB10A2Unorm),
        vk::Format::A2B10G10R10_UNORM_PACK32
    );
    assert_eq!(
        direct(PixelFormat::BGR10A2Unorm),
        vk::Format::A2R10G10B10_UNORM_PACK32
    );
    assert_eq!(
        direct(PixelFormat::RG11B10Float),
        vk::Format::B10G11R11_UFLOAT_PACK32
    );
    assert_eq!(sampled_type(PixelFormat::Stencil8), Some(SampledType::Uint));
    assert_eq!(
        sampled_type(PixelFormat::X32Stencil8),
        Some(SampledType::Uint)
    );
    assert_eq!(
        sampled_type(PixelFormat::Depth32Float),
        Some(SampledType::Float)
    );
}

/// An attachment usage the format's aspects do not have is an invalid descriptor on every
/// profile, whatever its format table says: no color attachment of a depth or stencil format,
/// no depth/stencil attachment of a color format.
#[test]
fn attachments_follow_the_format_aspects() {
    use FormatUsage as U;
    for (stem, dev) in committed_devices() {
        for &pf in PixelFormat::ALL {
            let Some(aspects) = aspects(pf) else {
                continue;
            };
            let color = aspects == vk::ImageAspectFlags::COLOR;
            for (usage, allowed) in [
                (U::ColorAttachment, color),
                (U::ColorAttachmentBlend, color),
                (U::DepthStencilAttachment, !color),
            ] {
                let got = vk_format(pf, &usages(&[usage, U::SampledImage]), dev.profile());
                let refused = Err(Error::InvalidDescriptor(
                    DescriptorError::PixelFormatUsage {
                        format: pf.as_raw(),
                        usage,
                    },
                ));
                if allowed {
                    assert_ne!(got, refused, "{stem} {pf:?} {usage:?}");
                } else {
                    assert_eq!(got, refused, "{stem} {pf:?} {usage:?}");
                }
            }
        }
    }
    // Even where the table claims the usage (a synthetic device), the aspect decides.
    let all = SAMPLED
        | vk::FormatFeatureFlags::COLOR_ATTACHMENT
        | vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
    let dev = device(floor_with(&[
        (vk::Format::D32_SFLOAT, all),
        (vk::Format::R8G8B8A8_UNORM, all),
    ]));
    for (pf, usage) in [
        (PixelFormat::Depth32Float, U::ColorAttachment),
        (PixelFormat::X32Stencil8, U::ColorAttachment),
        (PixelFormat::RGBA8Unorm, U::DepthStencilAttachment),
    ] {
        assert_eq!(
            vk_format(pf, &usages(&[usage]), dev.profile()),
            Err(Error::InvalidDescriptor(
                DescriptorError::PixelFormatUsage {
                    format: pf.as_raw(),
                    usage
                }
            )),
            "{pf:?}"
        );
    }
    assert!(vk_format(
        PixelFormat::Depth32Float,
        &usages(&[U::DepthStencilAttachment]),
        dev.profile()
    )
    .is_ok());
}

/// Placeholders are descriptor errors; formats without candidates are typed refusals.
#[test]
fn placeholders_and_refusals_are_typed() {
    let (_, dev) = committed_devices().remove(0);
    let sampled = usages(&[FormatUsage::SampledImage]);
    for pf in [PixelFormat::Invalid, PixelFormat::Unspecialized] {
        assert_eq!(
            vk_format(pf, &sampled, dev.profile()),
            Err(Error::InvalidDescriptor(
                DescriptorError::PlaceholderEnumValue {
                    ty: "MTLPixelFormat",
                    value: pf.as_raw()
                }
            ))
        );
    }
    let needs = |pf| match vk_format(pf, &sampled, dev.profile()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Format);
            u.needs
        }
        other => panic!("{pf:?}: {other:?}"),
    };
    assert_eq!(needs(PixelFormat::RGBA16FloatXR), Needs::NotImplemented);
}

/// On every committed profile, a mapping is only ever returned for supported usages, a swizzled
/// view only for sampling, and a refusal lists what each admissible candidate lacks.
#[test]
fn selection_on_every_committed_profile_is_sound() {
    use FormatUsage as U;
    let sets = [
        usages(&[U::SampledImage]),
        usages(&[U::SampledImage, U::SampledImageFilterLinear]),
        usages(&[U::ColorAttachment]),
        usages(&[U::ColorAttachment, U::ColorAttachmentBlend, U::SampledImage]),
        usages(&[U::DepthStencilAttachment, U::SampledImage]),
        usages(&[U::StorageImage]),
        usages(&[U::UniformTexelBuffer]),
    ];
    let mut mapped = 0;
    for (stem, dev) in committed_devices() {
        let profile = dev.profile();
        for &pf in PixelFormat::ALL {
            let Repr::Candidates(aspects, candidates) = make_repr(pf) else {
                continue;
            };
            for set in &sets {
                if set.iter().any(|&u| !aspects_admit(aspects, u)) {
                    // `attachments_follow_the_format_aspects` covers these.
                    continue;
                }
                match vk_format(pf, set, profile) {
                    Ok(m) => {
                        mapped += 1;
                        assert!(
                            set.iter().all(|&u| profile.supports_format(m.format, u)),
                            "{stem} {pf:?} {set:?}"
                        );
                        assert!(set.iter().all(|&u| m.emulation.admits(u)));
                        assert!(candidates.iter().any(|c| c.format == m.format
                            && c.components == m.components
                            && c.emulation == m.emulation));
                    }
                    Err(Error::Unsupported(u)) => {
                        assert_eq!(u.feature, Feature::Format);
                        match u.needs {
                            Needs::AnyOf(alternatives) => {
                                assert!(!alternatives.is_empty());
                                for alt in alternatives {
                                    assert!(!alt.is_empty(), "{stem} {pf:?} {set:?}");
                                    // An unexposed format value first names what exposes it.
                                    let (gap, pairs) = match &alt[0] {
                                        Capability::Extension(_) | Capability::ApiVersion(_) => {
                                            (Some(alt[0].clone()), &alt[1..])
                                        }
                                        _ => (None, &alt[..]),
                                    };
                                    assert!(!pairs.is_empty(), "{stem} {pf:?} {set:?}");
                                    for cap in pairs {
                                        let Capability::Format { format, usage } = *cap else {
                                            panic!("{cap:?}");
                                        };
                                        assert!(!profile.supports_format(format, usage));
                                        assert_eq!(exposure_gap(format, profile), gap);
                                    }
                                }
                            }
                            Needs::NotImplemented => assert!(
                                candidates
                                    .iter()
                                    .all(|c| !set.iter().all(|&u| c.emulation.admits(u))),
                                "{stem} {pf:?} {set:?}"
                            ),
                            other => panic!("{stem} {pf:?} {set:?}: {other:?}"),
                        }
                    }
                    Err(e) => panic!("{stem} {pf:?} {set:?}: {e:?}"),
                }
            }
        }
    }
    assert!(mapped > 1000, "{mapped}");
}

/// Vulkan's mandatory format support makes these mappings total on every committed profile that
/// reports the format. Which combined depth/stencil format is an attachment is a choice the
/// Required Format Support tables leave to the device, so a requirements profile (the `vk12-*`
/// floors, `roadmap-2022`) lists neither; `device::tests::required_formats` checks the choice on
/// every device report.
#[test]
fn mandatory_formats_map_on_every_committed_profile() {
    use FormatUsage as U;
    let must = [
        (
            PixelFormat::RGBA8Unorm,
            &[U::SampledImage, U::ColorAttachment][..],
        ),
        (
            PixelFormat::BGRA8Unorm,
            &[U::SampledImage, U::ColorAttachment],
        ),
        (PixelFormat::RGBA8UnormSRGB, &[U::SampledImage]),
        (PixelFormat::R32Float, &[U::SampledImage, U::StorageImage]),
        (
            PixelFormat::RGBA16Float,
            &[U::SampledImage, U::ColorAttachment],
        ),
        (
            PixelFormat::Depth16Unorm,
            &[U::SampledImage, U::DepthStencilAttachment],
        ),
    ];
    let mut checked = 0;
    for (stem, desc) in committed_descriptions() {
        let reported = |f: vk::Format| desc.formats.formats.contains_key(&f);
        let reports_all = must
            .iter()
            .all(|&(pf, _)| reported(direct_format(pf).unwrap()));
        let dev = device(desc.clone());
        if !reports_all {
            continue;
        }
        checked += 1;
        for (pf, list) in must {
            let m = vk_format(pf, &usages(list), dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {pf:?}: {e}"));
            assert_eq!(m.emulation, FormatEmulation::Direct, "{stem} {pf:?}");
        }
        // One of the combined depth/stencil formats is a depth/stencil attachment on every
        // profile that states either.
        let combined = [
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        if combined.into_iter().any(reported) {
            let ds = usages(&[U::DepthStencilAttachment]);
            vk_format(PixelFormat::Depth32FloatStencil8, &ds, dev.profile())
                .or_else(|_| vk_format(PixelFormat::Depth24UnormStencil8, &ds, dev.profile()))
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
        }
    }
    assert!(
        checked >= 8,
        "only {checked} committed profiles report formats"
    );
}

/// `A8Unorm` is `A8_UNORM` where the profile exposes and supports it (Vulkan's alpha format reads
/// (0, 0, 0, a), as Metal does); elsewhere its byte is `R8_UNORM`'s, its alpha in red
/// (`FormatEmulation::AlphaInRed`), for sampling through `(Zero, Zero, Zero, R)`, storage, color
/// attachments and their blend; a usage neither format has names what each lacks.
#[test]
fn a8_is_direct_where_exposed_and_holds_alpha_in_red_elsewhere() {
    use FormatUsage as U;
    let attach = SAMPLED | vk::FormatFeatureFlags::COLOR_ATTACHMENT;
    let red = SAMPLED
        | vk::FormatFeatureFlags::STORAGE_IMAGE
        | vk::FormatFeatureFlags::COLOR_ATTACHMENT
        | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
    let table = [(vk::Format::R8_UNORM, red), (vk::Format::A8_UNORM, attach)];
    // The 1.2 floor without `VK_KHR_maintenance5`: the listed `A8_UNORM` is not exposed, so the
    // profile does not hold it.
    let floor = device(floor_with(&table));
    assert!(!floor
        .profile()
        .supports_format(vk::Format::A8_UNORM, U::SampledImage));
    for set in [
        &[U::SampledImage][..],
        &[U::StorageImage],
        &[U::ColorAttachment],
        &[U::ColorAttachment, U::ColorAttachmentBlend, U::SampledImage],
    ] {
        let m = vk_format(PixelFormat::A8Unorm, &usages(set), floor.profile()).unwrap();
        assert_eq!(m.format, vk::Format::R8_UNORM, "{set:?}");
        assert_eq!(m.emulation, FormatEmulation::AlphaInRed, "{set:?}");
        assert_eq!(m.components.a, vk::ComponentSwizzle::R, "{set:?}");
    }
    assert!(alpha_in_red(PixelFormat::A8Unorm, floor.profile()));
    assert!(!alpha_in_red(PixelFormat::R8Unorm, floor.profile()));
    assert_eq!(
        alpha_in_red_format(PixelFormat::A8Unorm),
        Some(vk::Format::R8_UNORM)
    );
    assert_eq!(alpha_in_red_format(PixelFormat::R8Unorm), None);
    // Neither format renders: the refusal names what each lacks.
    let sampled_red = device(floor_with(&[(vk::Format::R8_UNORM, SAMPLED)]));
    assert_eq!(
        vk_format(
            PixelFormat::A8Unorm,
            &usages(&[U::ColorAttachment]),
            sampled_red.profile()
        ),
        Err(Unsupported::needs_any_of(
            Feature::Format,
            vec![
                vec![
                    Capability::Extension(VkExtension::KhrMaintenance5),
                    Capability::Format {
                        format: vk::Format::A8_UNORM,
                        usage: U::ColorAttachment
                    }
                ],
                vec![Capability::Format {
                    format: vk::Format::R8_UNORM,
                    usage: U::ColorAttachment
                }],
            ]
        )
        .into())
    );
    // Still a device that holds `A8Unorm` in red: its attachments are what the device lacks.
    assert!(alpha_in_red(PixelFormat::A8Unorm, sampled_red.profile()));
    // With the extension (and its 1.2 requirement), the direct format.
    let mut desc = floor_with(&table);
    desc.extensions.extend([
        VkExtension::KhrMaintenance5,
        VkExtension::KhrDynamicRendering,
    ]);
    let m5 = device(desc);
    for set in [
        &[U::SampledImage][..],
        &[U::ColorAttachment, U::SampledImage],
    ] {
        let m = vk_format(PixelFormat::A8Unorm, &usages(set), m5.profile()).unwrap();
        assert_eq!(m.format, vk::Format::A8_UNORM, "{set:?}");
        assert_eq!(m.emulation, FormatEmulation::Direct);
        assert_eq!(m.components, vk::ComponentMapping::IDENTITY);
    }
    assert!(!alpha_in_red(PixelFormat::A8Unorm, m5.profile()));
    // Vulkan 1.4 exposes it in core.
    let mut desc = floor_with(&table);
    desc.api_version = ApiVersion { major: 1, minor: 4 };
    let v14 = device(desc);
    let m = vk_format(
        PixelFormat::A8Unorm,
        &usages(&[U::SampledImage]),
        v14.profile(),
    )
    .unwrap();
    assert_eq!(m.format, vk::Format::A8_UNORM);
}

/// An `A8Unorm` storage texture on a device that stores no `A8_UNORM` is its `R8_UNORM` image
/// itself, read as alpha and written from alpha by the shader (`StorageImageClass::AlphaInRed`),
/// with no alias.
#[test]
fn a8_storage_is_its_red_image_without_a8_unorm() {
    let red = SAMPLED | vk::FormatFeatureFlags::STORAGE_IMAGE;
    let table = [(vk::Format::R8_UNORM, red), (vk::Format::A8_UNORM, red)];
    let floor = device(floor_with(&table));
    let c = storage_image_choice(PixelFormat::A8Unorm, floor.profile()).unwrap();
    assert_eq!(c.vk_format, vk::Format::R8_UNORM);
    assert_eq!(c.class, StorageImageClass::AlphaInRed);
    assert_eq!(c.alias, None);
    assert_eq!(
        storage_view_formats(PixelFormat::A8Unorm, floor.profile()),
        vec![vk::Format::R8_UNORM]
    );
}

/// A `BGRA8Unorm_sRGB` storage texture is its `B8G8R8A8_UNORM` twin's encode view where the device
/// accesses the twin without a format (SPIR-V names no BGRA image format), and else an
/// `R8G8B8A8_UNORM` view that encodes and exchanges red and blue (`SrgbEncodeBgra`), which every
/// committed profile stores: ARL's and Pascal's profiles state no `VkFormatProperties3`, so their
/// twin takes neither access without a format, and the Iris Xe's ANV reads no storage image
/// without one. Its texture lists the view among its formats.
#[test]
fn bgra8_srgb_storage_takes_an_rgba8_view_where_its_twin_is_not_declarable() {
    let mut swizzled = Vec::new();
    for (name, dev) in committed_devices() {
        let p = dev.profile();
        let pf = PixelFormat::BGRA8UnormSRGB;
        let c = storage_image_choice(pf, p).unwrap_or_else(|e| panic!("{name}: {e}"));
        // The twin, which SPIR-V names no format for, is declarable without one exactly where the
        // device accesses it so, by its features or by the format's own bits.
        let twin = without_format_gaps(
            vk::Format::B8G8R8A8_UNORM,
            StorageUse::READ_WRITE_TEXTURE,
            p,
        )
        .is_empty();
        match c.class {
            StorageImageClass::Identity => {}
            StorageImageClass::SrgbEncode => {
                assert!(twin, "{name}");
                assert_eq!(c.vk_format, vk::Format::B8G8R8A8_UNORM, "{name}");
            }
            StorageImageClass::SrgbEncodeBgra => {
                assert!(!twin, "{name}");
                assert_eq!(c.vk_format, vk::Format::R8G8B8A8_UNORM, "{name}");
                assert_eq!(
                    c.alias,
                    Some(AliasView::new(vk::Format::B8G8R8A8_SRGB)),
                    "{name}"
                );
                assert!(
                    storage_view_formats(pf, p).contains(&vk::Format::R8G8B8A8_UNORM),
                    "{name}"
                );
                swizzled.push(name);
            }
            other => panic!("{name}: {other:?}"),
        }
    }
    for want in ["intel-arl-anv", "nvidia-pascal-linux", "intel-irisxe-anv"] {
        assert!(swizzled.iter().any(|n| n == want), "{want}: {swizzled:?}");
    }
}

/// On a device without `shaderStorageImageReadWithoutFormat` (the ANV profile of the Iris Xe of
/// RPL-P, read at the runtime's Vulkan 1.2), `VK_KHR_format_feature_flags2` lets a storage access
/// without a format go by each format's own bits: `A8_UNORM`, which SPIR-V names no format for and
/// which has both bits there, is then declared without one, read and written; `RGBA8Unorm` keeps
/// its typed declaration (the formatless one is preferred by the features alone); and
/// `B5G6R5_UNORM_PACK16`, which lacks the read bit there, lacks that bit. Without the extension
/// every such access lacks the feature.
#[test]
fn a_storage_access_without_a_format_goes_by_the_format_s_own_bits_with_the_extension() {
    let read = StorageUse::texture(true, false);
    let read_write = StorageUse::READ_WRITE_TEXTURE;
    for flags2 in [true, false] {
        let desc =
            crate::pipeline::testing::description_at_1_2("intel-irisxe-rplp-anv", flags2).unwrap();
        let dev = device(desc);
        let p = dev.profile();
        assert!(!p.images().storage_read_without_format);
        assert!(p.images().storage_write_without_format);
        assert_eq!(p.without_format_by_format_bits(), flags2);
        assert!(!accessed_without_format(vk::Format::A8_UNORM, read, p));
        let a8 = storage_image_choice(PixelFormat::A8Unorm, p);
        let rgba8 = storage_image_choice(PixelFormat::RGBA8Unorm, p).expect("RGBA8Unorm");
        assert_eq!(rgba8.vk_format, vk::Format::R8G8B8A8_UNORM, "typed");
        if flags2 {
            assert_eq!(without_format_gaps(vk::Format::A8_UNORM, read_write, p), []);
            let a8 = a8.expect("A8Unorm");
            assert_eq!(
                (a8.vk_format, a8.numeric, a8.class, a8.alias),
                (
                    vk::Format::UNDEFINED,
                    Some(SampledType::Float),
                    StorageImageClass::Identity,
                    None
                )
            );
            assert_eq!(
                without_format_gaps(vk::Format::B5G6R5_UNORM_PACK16, read, p),
                [Capability::Format {
                    format: vk::Format::B5G6R5_UNORM_PACK16,
                    usage: FormatUsage::StorageImageReadWithoutFormat,
                }]
            );
        } else {
            let feature = Capability::Feature(VkFeature::ShaderStorageImageReadWithoutFormat);
            assert_eq!(
                without_format_gaps(vk::Format::A8_UNORM, read_write, p),
                std::slice::from_ref(&feature)
            );
            assert!(a8.is_err(), "{a8:?}");
            assert_eq!(
                without_format_gaps(vk::Format::B5G6R5_UNORM_PACK16, read, p),
                [feature]
            );
        }
    }
}

/// An `A8Unorm` view reads an `R8Unorm` image through `(Zero, Zero, Zero, Red)` composed under the
/// view's own swizzle, where the device has `A8_UNORM` and where it does not (the probe a8view on
/// an Apple M3 Max under macOS 27.0.1: Metal's `A8Unorm` view of an `R8Unorm` texture made without
/// `PixelFormatView` samples, reads, gathers and borders as a plain `A8Unorm` texture and as an
/// `R8Unorm` view of that swizzle do, shared and private, through four swizzles). A pair whose
/// opaque-black border texel moves with the mapping (`BGRA8Unorm` over `RGBA8Unorm`: the format
/// has alpha of its own) has none, nor a format over itself, nor a format of no such candidate,
/// nor a device whose portability subset withholds `imageViewFormatSwizzle`.
#[test]
fn a_view_in_other_channels_of_the_same_bits_reads_through_a_swizzle() {
    use crate::metal::TextureSwizzle as T;
    let s = SAMPLED;
    let table = [
        (vk::Format::R8_UNORM, s),
        (vk::Format::A8_UNORM, s),
        (vk::Format::R8G8B8A8_UNORM, s),
        (vk::Format::B8G8R8A8_UNORM, s),
    ];
    let mut desc = floor_with(&table);
    desc.extensions.extend([
        VkExtension::KhrMaintenance5,
        VkExtension::KhrDynamicRendering,
    ]);
    let m5 = device(desc);
    let floor = device(floor_with(&table));
    let alpha = TextureSwizzleChannels::new(T::Zero, T::Zero, T::Zero, T::Red);
    for d in [&m5, &floor] {
        let over = |pf, swizzle, own| swizzle_over(pf, swizzle, own, d.profile());
        let default = TextureSwizzleChannels::DEFAULT;
        assert_eq!(
            over(PixelFormat::A8Unorm, default, PixelFormat::R8Unorm),
            Some(alpha)
        );
        let swizzled = TextureSwizzleChannels::new(T::Alpha, T::Red, T::One, T::Zero);
        assert_eq!(
            over(PixelFormat::A8Unorm, swizzled, PixelFormat::R8Unorm),
            Some(TextureSwizzleChannels::new(
                T::Red,
                T::Zero,
                T::One,
                T::Zero
            ))
        );
        // The first lane whose source is alpha stores the image's red (probe a8write).
        let second = TextureSwizzleChannels::new(T::Zero, T::Alpha, T::Alpha, T::Red);
        assert_eq!(
            over(PixelFormat::A8Unorm, second, PixelFormat::R8Unorm),
            Some(TextureSwizzleChannels::new(
                T::Zero,
                T::Red,
                T::Red,
                T::Zero
            ))
        );
        // No lane's source is alpha: Metal stores the alpha lane, a swizzled R8 image its red.
        let opaque = TextureSwizzleChannels::new(T::Red, T::Green, T::Blue, T::One);
        assert_eq!(
            over(PixelFormat::A8Unorm, opaque, PixelFormat::R8Unorm),
            None
        );
        assert_eq!(
            over(PixelFormat::BGRA8Unorm, default, PixelFormat::RGBA8Unorm),
            None
        );
        assert_eq!(
            over(PixelFormat::R8Unorm, default, PixelFormat::R8Unorm),
            None
        );
        assert_eq!(
            over(PixelFormat::R8Uint, default, PixelFormat::R8Unorm),
            None
        );
    }
    let mut desc = floor_with(&table);
    desc.extensions.insert(VkExtension::KhrPortabilitySubset);
    let withheld = device(desc);
    let default = TextureSwizzleChannels::DEFAULT;
    assert_eq!(
        swizzle_over(
            PixelFormat::A8Unorm,
            default,
            PixelFormat::R8Unorm,
            withheld.profile()
        ),
        None
    );
}

/// A portability subset that withholds `imageViewFormatSwizzle` makes no view with a component
/// mapping (`VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465`): `A8Unorm`'s swizzled
/// `R8_UNORM` candidate names the feature beside the direct format's needs, and is chosen once the
/// subset has it. A device without the subset has no such restriction.
#[test]
fn a_swizzled_candidate_needs_the_subset_s_view_swizzle() {
    use FormatUsage as U;
    let table = [(vk::Format::R8_UNORM, SAMPLED)];
    let sampled = usages(&[U::SampledImage]);
    let mut desc = floor_with(&table);
    desc.extensions.insert(VkExtension::KhrPortabilitySubset);
    let withheld = device(desc.clone());
    assert_eq!(
        vk_format(PixelFormat::A8Unorm, &sampled, withheld.profile()),
        Err(Unsupported::needs_any_of(
            Feature::Format,
            vec![
                vec![
                    Capability::Extension(VkExtension::KhrMaintenance5),
                    Capability::Format {
                        format: vk::Format::A8_UNORM,
                        usage: U::SampledImage
                    }
                ],
                vec![Capability::Feature(VkFeature::ImageViewFormatSwizzle)],
            ]
        )
        .into())
    );
    let swizzle = vk::ComponentMapping {
        r: vk::ComponentSwizzle::B,
        ..vk::ComponentMapping::IDENTITY
    };
    assert_eq!(
        view_mapping_gap(swizzle, withheld.profile()),
        Some(Capability::Feature(VkFeature::ImageViewFormatSwizzle))
    );
    assert_eq!(
        view_mapping_gap(vk::ComponentMapping::IDENTITY, withheld.profile()),
        None
    );
    desc.features.insert(VkFeature::ImageViewFormatSwizzle);
    let granted = device(desc);
    let m = vk_format(PixelFormat::A8Unorm, &sampled, granted.profile()).unwrap();
    assert_eq!(m.format, vk::Format::R8_UNORM);
    assert_eq!(view_mapping_gap(m.components, granted.profile()), None);
    let plain = device(floor_with(&table));
    assert_eq!(view_mapping_gap(swizzle, plain.profile()), None);
}

/// The ASTC HDR and PVRTC RGBA formats are their `SFLOAT` and PVRTC1 blocks where the profile
/// exposes them: on `moltenvk-m3` (a 1.4 device with `VK_IMG_format_pvrtc`) they map directly;
/// on the 1.2 floor listing the same blocks without the extensions, the refusal names the
/// extension, and for a PVRTC format the format of its decoded twin's codes, which that floor does
/// not list either.
#[test]
fn compressed_extension_formats_follow_exposure() {
    use FormatUsage as U;
    let sampled = usages(&[U::SampledImage]);
    let cases = [
        (
            PixelFormat::ASTC4x4HDR,
            vk::Format::ASTC_4X4_SFLOAT_BLOCK,
            VkExtension::ExtTextureCompressionAstcHdr,
        ),
        (
            PixelFormat::ASTC12x12HDR,
            vk::Format::ASTC_12X12_SFLOAT_BLOCK,
            VkExtension::ExtTextureCompressionAstcHdr,
        ),
        (
            PixelFormat::PVRTCRGBA2BPP,
            vk::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG,
            VkExtension::ImgFormatPvrtc,
        ),
        (
            PixelFormat::PVRTCRGBA4BPPSRGB,
            vk::Format::PVRTC1_4BPP_SRGB_BLOCK_IMG,
            VkExtension::ImgFormatPvrtc,
        ),
    ];
    let (_, moltenvk) = committed_devices()
        .into_iter()
        .find(|(s, _)| s == "moltenvk-m3")
        .expect("moltenvk-m3 is committed");
    let floor = device(floor_with(&cases.map(|(_, format, _)| (format, SAMPLED))));
    for (pf, format, extension) in cases {
        let m =
            vk_format(pf, &sampled, moltenvk.profile()).unwrap_or_else(|e| panic!("{pf:?}: {e}"));
        assert_eq!((m.format, m.emulation), (format, FormatEmulation::Direct));
        let mut needs = vec![vec![
            Capability::Extension(extension),
            Capability::Format {
                format,
                usage: U::SampledImage,
            },
        ]];
        // A PVRTC format's decoded twin, whose codes' format this floor does not list.
        if crate::bind::PvrtcDecode::of(pf).is_some() {
            let codes = match crate::metal::texel_format::srgb_coded(pf) {
                true => vk::Format::R8G8B8A8_SRGB,
                false => vk::Format::R8G8B8A8_UNORM,
            };
            needs.push(vec![Capability::Format {
                format: codes,
                usage: U::SampledImage,
            }]);
        }
        assert_eq!(
            vk_format(pf, &sampled, floor.profile()),
            Err(Unsupported::needs_any_of(Feature::Format, needs).into()),
            "{pf:?}"
        );
    }
    // Compressed blocks are never attachments or storage images on any committed profile.
    for u in [U::ColorAttachment, U::StorageImage] {
        assert!(vk_format(PixelFormat::ASTC4x4HDR, &usages(&[u]), moltenvk.profile()).is_err());
    }
}

/// ASTC's 4x2 and 8x4 formats take the raw values and names Metal gave them in our census, so a
/// consumer holds each, and every usage of each is refused `NotImplemented(Format)` on every
/// committed device: no Vulkan block has their footprints.
#[test]
fn astc_4x2_and_8x4_are_named_and_refused_typed() {
    use FormatUsage as U;
    let named = [
        (185, "MTLPixelFormatASTC_4x2_sRGB"),
        (191, "MTLPixelFormatASTC_8x4_sRGB"),
        (203, "MTLPixelFormatASTC_4x2_LDR"),
        (209, "MTLPixelFormatASTC_8x4_LDR"),
        (221, "MTLPixelFormatASTC_4x2_HDR"),
        (227, "MTLPixelFormatASTC_8x4_HDR"),
    ];
    for (raw, name) in named {
        let pf = PixelFormat::try_from(raw).unwrap_or_else(|e| panic!("{raw}: {e}"));
        assert_eq!(pf.name(), name);
        assert_eq!(PixelFormat::from_name(name), Some(pf));
        assert!(matches!(pf, astc_without_a_vulkan_block!()), "{name}");
        for (stem, device) in committed_devices() {
            for u in [
                U::SampledImage,
                U::SampledImageFilterLinear,
                U::ColorAttachment,
            ] {
                assert_eq!(
                    vk_format(pf, &usages(&[u]), device.profile()),
                    Err(Unsupported::not_implemented(Feature::Format).into()),
                    "{name} {u:?} on {stem}"
                );
            }
        }
    }
}

/// Every other raw value Metal names takes Metal's name (a texture descriptor's description on an
/// Apple M4 under macOS 27.0.1), so a consumer holds each, and every usage of each is refused
/// `NotImplemented(Format)` on every committed device until its reads, writes and samples are
/// measured.
#[test]
fn formats_metal_names_are_named_and_refused_typed() {
    use FormatUsage as U;
    let named = [
        (26, "MTLPixelFormatR16Bfloat"),
        (44, "MTLPixelFormatBGRA4Unorm"),
        (45, "MTLPixelFormatRGB8Unorm"),
        (46, "MTLPixelFormatRGB8Snorm"),
        (47, "MTLPixelFormatRGB8Uint"),
        (48, "MTLPixelFormatRGB8Sint"),
        (66, "MTLPixelFormatRG16Bfloat"),
        (85, "MTLPixelFormatARGB8Unorm_sRGB"),
        (95, "MTLPixelFormatRGB16Unorm"),
        (96, "MTLPixelFormatRGB16Snorm"),
        (97, "MTLPixelFormatRGB16Uint"),
        (98, "MTLPixelFormatRGB16Sint"),
        (99, "MTLPixelFormatRGB16Float"),
        (100, "MTLPixelFormatRGB16Bfloat"),
        (116, "MTLPixelFormatRGBA16Bfloat"),
        (120, "MTLPixelFormatRGB32Uint"),
        (121, "MTLPixelFormatRGB32Sint"),
        (122, "MTLPixelFormatRGB32Float"),
        (300, "MTLPixelFormatRGBX8Unorm"),
        (301, "MTLPixelFormatRGBX8Unorm_sRGB"),
        (302, "MTLPixelFormatRGBX8Snorm"),
        (303, "MTLPixelFormatRGBX8Uint"),
        (304, "MTLPixelFormatRGBX8Sint"),
        (313, "MTLPixelFormatRGBX16Uint"),
        (314, "MTLPixelFormatRGBX16Sint"),
        (315, "MTLPixelFormatRGBX16Float"),
        (323, "MTLPixelFormatRGBX32Uint"),
        (324, "MTLPixelFormatRGBX32Sint"),
        (325, "MTLPixelFormatRGBX32Float"),
        (501, "MTLPixelFormatYCBCR8_422_1P"),
        (504, "MTLPixelFormatYCBCR10_444_1P"),
        (521, "MTLPixelFormatYCBCR8_422_1P_sRGB"),
        (524, "MTLPixelFormatYCBCR10_444_1P_sRGB"),
        (560, "MTLPixelFormatYCBCRA8_444_1P"),
        (562, "MTLPixelFormatYCBCR8_422_1P_REV"),
        (563, "MTLPixelFormatYCBCR10_444_1P_PQ"),
        (576, "MTLPixelFormatR10Unorm_X6"),
        (577, "MTLPixelFormatR10Unorm_X6_sRGB"),
        (578, "MTLPixelFormatRG10Unorm_X12"),
        (579, "MTLPixelFormatRG10Unorm_X12_sRGB"),
        (586, "MTLPixelFormatRGB10A2Unorm_sRGB"),
        (587, "MTLPixelFormatRGB10A2Unorm_PQ"),
        (588, "MTLPixelFormatR10Unorm_PACKED"),
        (589, "MTLPixelFormatRG10Unorm_PACKED"),
        (590, "MTLPixelFormatYCBCR10_444_1P_XR"),
        (603, "MTLPixelFormatR12Unorm_X4"),
        (604, "MTLPixelFormatR12Unorm_X4_PQ"),
        (605, "MTLPixelFormatRG12Unorm_X8"),
        (606, "MTLPixelFormatR10Unorm_X6_PQ"),
        (607, "MTLPixelFormatYCBCR10_444_1P_HLG"),
        (621, "MTLPixelFormatR10Unorm_X6_HLG"),
        (622, "MTLPixelFormatR12Unorm_X4_HLG"),
        (623, "MTLPixelFormatRGB10A2Unorm_HLG"),
        (624, "MTLPixelFormatRGB10A8_2P"),
        (625, "MTLPixelFormatRGB10A8_2P_sRGB"),
        (626, "MTLPixelFormatRGB10A8_2P_PQ"),
        (627, "MTLPixelFormatRGB10A8_2P_HLG"),
        (628, "MTLPixelFormatBGR10A2Unorm_sRGB"),
        (629, "MTLPixelFormatBGR10A2Unorm_PQ"),
        (630, "MTLPixelFormatBGRA10Uint_PACKED"),
        (631, "MTLPixelFormatBGRA10_XR_PACKED"),
        (632, "MTLPixelFormatBGRA10_XR_PACKED_sRGB"),
        (633, "MTLPixelFormatBGRA10Unorm_PACKED"),
        (634, "MTLPixelFormatBGRA10Unorm_PACKED_sRGB"),
        (635, "MTLPixelFormatBGRA10Unorm_PACKED_HLG"),
        (636, "MTLPixelFormatBGRA10Unorm_PACKED_PQ"),
        (637, "MTLPixelFormatBGRA10Uint"),
        (638, "MTLPixelFormatBGRA10Unorm"),
        (639, "MTLPixelFormatBGRA10Unorm_sRGB"),
        (640, "MTLPixelFormatBGRA10Unorm_HLG"),
        (641, "MTLPixelFormatBGRA10Unorm_PQ"),
        (642, "MTLPixelFormatR10Uint_X6"),
        (643, "MTLPixelFormatR10Uint_PACKED"),
        (644, "MTLPixelFormatRG10Uint_X12"),
        (645, "MTLPixelFormatRG10Uint_PACKED"),
        (646, "MTLPixelFormatR12Uint_X4"),
        (647, "MTLPixelFormatR12Uint_PACKED"),
        (648, "MTLPixelFormatRG12Uint_X8"),
        (649, "MTLPixelFormatRG12Uint_PACKED"),
        (650, "MTLPixelFormatRGB10A8Uint_2P"),
    ];
    // All but `R10Unorm_X6`, `RG10Unorm_X12` and their `_sRGB` twins, held raw
    // in their words' integer formats (`padded_words`).
    // And the eleven RGBX formats, held in their RGBA twins (`x_channel`).
    // And the three bfloat formats and `RGB10A2Unorm_sRGB`, held raw (`ExtendedRange`).
    // And `BGRA4Unorm`, held in the 4-bit word of its bits (`A4R4G4B4_UNORM_PACK16`, or
    // `ABGR4Unorm`'s word through a rotating view).
    let bfloat = |pf| ExtendedRange::of(pf).is_some_and(ExtendedRange::bfloat);
    let rgb10a2_srgb = |pf| ExtendedRange::of(pf) == Some(ExtendedRange::Rgb10A2Srgb);
    assert_eq!(
        PixelFormat::ALL
            .iter()
            .filter(|&&pf| matches!(pf, formats_named_not_held!()))
            .count(),
        named.len() - 4 - 11 - 3 - 1 - 1
    );
    for (raw, name) in named {
        let pf = PixelFormat::try_from(raw).unwrap_or_else(|e| panic!("{raw}: {e}"));
        assert_eq!(pf.name(), name);
        assert_eq!(PixelFormat::from_name(name), Some(pf));
        if padded_words(pf) || x_channel(pf) || bfloat(pf) || rgb10a2_srgb(pf) {
            continue;
        }
        if pf == PixelFormat::BGRA4Unorm {
            continue;
        }
        assert!(matches!(pf, formats_named_not_held!()), "{name}");
        assert_eq!(aspects(pf), None, "{name}");
        assert_eq!(direct_format(pf), None, "{name}");
        for (stem, device) in committed_devices() {
            for u in [
                U::SampledImage,
                U::SampledImageFilterLinear,
                U::StorageImage,
                U::ColorAttachment,
            ] {
                assert_eq!(
                    vk_format(pf, &usages(&[u]), device.profile()),
                    Err(Unsupported::not_implemented(Feature::Format).into()),
                    "{name} {u:?} on {stem}"
                );
            }
        }
    }
}

/// `R10Unorm_X6` and `RG10Unorm_X12` are held raw in their words' integer formats for sampling,
/// storage and rendering (the shader decodes, filters and encodes them, `ExtendedRange::R10X6` and `Rg10X12`),
/// they and their `_sRGB` twins are exactly the formats whose codes sit in the top bits of their
/// own 16-bit words (`padded_words`), and a clear stores Metal's words: the color
/// through `float` and `half`, each to nearest even, the code of the exact product with 1023
/// rounded to nearest in the top 10 bits.
#[test]
fn padded_formats_are_held_raw() {
    use FormatUsage as U;
    use PixelFormat as P;
    let held: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| padded_words(pf))
        .collect();
    assert_eq!(
        held,
        [
            P::R10UnormX6,
            P::R10UnormX6SRGB,
            P::RG10UnormX12,
            P::RG10UnormX12SRGB
        ]
    );
    for (pf, raw, form) in [
        (P::R10UnormX6, vk::Format::R16_UINT, ExtendedRange::R10X6),
        (
            P::RG10UnormX12,
            vk::Format::R16G16_UINT,
            ExtendedRange::Rg10X12,
        ),
        (
            P::R10UnormX6SRGB,
            vk::Format::R16_UINT,
            ExtendedRange::R10X6Srgb,
        ),
        (
            P::RG10UnormX12SRGB,
            vk::Format::R16G16_UINT,
            ExtendedRange::Rg10X12Srgb,
        ),
    ] {
        assert_eq!(ExtendedRange::of(pf), Some(form));
        assert_eq!(form.raw_format(), raw);
        assert!(form.padded() && !form.srgb() && form.holds_metal_bytes());
        let twin = matches!(pf, P::R10UnormX6SRGB | P::RG10UnormX12SRGB);
        assert_eq!(form.padded_srgb(), twin, "{pf:?}");
        assert_eq!(byte_format(pf), Some(raw));
        // The shader filters a linear sample from the raw taps, as for the extended-range
        // formats: no device's linear filter is asked for. A device lacking the raw format's
        // usage refuses it by what it lacks.
        let mut held = 0;
        for (stem, device) in committed_devices() {
            for u in [U::SampledImage, U::StorageImage, U::ColorAttachment] {
                match vk_format(pf, &usages(&[u]), device.profile()) {
                    Ok(m) => {
                        held += 1;
                        assert_eq!(
                            (m.format, m.emulation),
                            (raw, FormatEmulation::ExtendedRange),
                            "{pf:?} {u:?} on {stem}"
                        );
                    }
                    Err(Error::Unsupported(e)) => {
                        assert_ne!(e.needs, Needs::NotImplemented, "{pf:?} {u:?} on {stem}");
                    }
                    Err(e) => panic!("{pf:?} {u:?} on {stem}: {e}"),
                }
                if stem.starts_with("moltenvk") || stem == "lavapipe" {
                    assert!(
                        vk_format(pf, &usages(&[u]), device.profile()).is_ok(),
                        "{pf:?} {u:?} on {stem}"
                    );
                }
            }
        }
        assert!(held > 0, "{pf:?}");
        assert_eq!(out_of_bounds_alpha_one(pf), Some(true));
    }
    let r = ExtendedRange::R10X6;
    let rg = ExtendedRange::Rg10X12;
    assert_eq!(
        rg.clear_texel([0.5, 0.25, 0.0, 1.0], ClearRoute::ThroughHalf),
        [32768, 16384, 0, 0]
    );
    assert_eq!(
        rg.clear_texel([0.3, 0.7, 0.0, 1.0], ClearRoute::ThroughHalf),
        [19648, 45824, 0, 0]
    );
    assert_eq!(
        r.clear_texel([0.3, 0.7, 0.0, 1.0], ClearRoute::ThroughHalf),
        [19648, 0, 0, 0]
    );
    assert_eq!(
        r.clear_texel([-1.0, 0.0, 0.0, 0.0], ClearRoute::ThroughHalf),
        [0, 0, 0, 0]
    );
    assert_eq!(
        r.clear_texel([2.0, 0.0, 0.0, 0.0], ClearRoute::ThroughHalf),
        [1023 << 6, 0, 0, 0]
    );
    // The `_sRGB` twins' clear of (0.3, 0.7) is Metal's (a clear-only
    // pass over a buffer stores 38208 and [4928, 7424] in every texel), through their tables.
    let rs = ExtendedRange::R10X6Srgb;
    let rgs = ExtendedRange::Rg10X12Srgb;
    assert_eq!(
        rs.clear_texel([0.3, 0.7, 0.0, 1.0], ClearRoute::ThroughHalf),
        [597 << 6, 0, 0, 0]
    );
    assert_eq!(
        rgs.clear_texel([0.3, 0.7, 0.0, 1.0], ClearRoute::ThroughHalf),
        [77 << 6, 116 << 6, 0, 0]
    );
    assert_eq!(
        rs.clear_texel([2.0, 0.0, 0.0, 0.0], ClearRoute::ThroughHalf),
        [1023 << 6, 0, 0, 0]
    );
    assert_eq!(
        rgs.clear_texel([1.0, -1.0, 0.0, 0.0], ClearRoute::ThroughHalf),
        [137 << 6, 0, 0, 0]
    );
}

/// Every swizzled sampling view reads Metal's opaque-black border texel where the device applies
/// its mapping to the border (the red/blue twins exchange two zeros; `A8Unorm` through `R8_UNORM`
/// reads the border's red, 0, as alpha, as Metal does), so `metal::sampler::check_border_view`
/// asks only for the border-swizzle features there.
#[test]
fn swizzled_views_read_metal_s_opaque_black_border() {
    let mut views = 0;
    for &pf in PixelFormat::ALL {
        let Repr::Candidates(_, candidates) = make_repr(pf) else {
            continue;
        };
        for c in candidates.iter().filter(|c| {
            matches!(
                c.emulation,
                FormatEmulation::SwizzledView
                    | FormatEmulation::RotatedView
                    | FormatEmulation::AlphaInRed
            )
        }) {
            let device = crate::metal::sampler::device_opaque_black(c.format, c.components, false);
            // Metal's: alpha 1, the color's or a missing channel's, but `A8Unorm`'s, the red,
            // and `ARGB8Unorm`'s, whose border passes through its byte rotation: blue 1
            // (measured on the M3 Max, and an M4).
            let metal = match pf {
                PixelFormat::ARGB8Unorm => [false, false, true, false],
                _ => [false, false, false, pf != PixelFormat::A8Unorm],
            };
            assert_eq!(device, metal, "{pf:?}: {c:?}");
            // The border model reads it so too, as every device through the view does.
            let swizzle = crate::metal::TextureSwizzleChannels::default();
            assert_eq!(
                crate::metal::sampler::border_texel(
                    pf,
                    channels(pf).expect("a color format's channels"),
                    swizzle,
                    crate::metal::sampler::BORDER_COLORS[1]
                ),
                metal,
                "{pf:?}"
            );
            views += 1;
        }
    }
    assert!(views >= 9, "{views}");
}

#[test]
fn channel_order_twin_is_used_only_for_sampling() {
    // A device that samples B4G4R4A4 but not R4G4B4A4.
    let dev = device(floor_with(&[(
        vk::Format::B4G4R4A4_UNORM_PACK16,
        SAMPLED | vk::FormatFeatureFlags::COLOR_ATTACHMENT,
    )]));
    let m = vk_format(
        PixelFormat::ABGR4Unorm,
        &usages(&[FormatUsage::SampledImage]),
        dev.profile(),
    )
    .unwrap();
    assert_eq!(m.format, vk::Format::B4G4R4A4_UNORM_PACK16);
    assert_eq!(m.emulation, FormatEmulation::SwizzledView);
    let err = vk_format(
        PixelFormat::ABGR4Unorm,
        &usages(&[FormatUsage::ColorAttachment]),
        dev.profile(),
    )
    .unwrap_err();
    assert_eq!(
        err,
        Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format: vk::Format::R4G4B4A4_UNORM_PACK16,
                usage: FormatUsage::ColorAttachment
            }
        )
        .into()
    );
}

#[test]
fn depth_and_stencil_fall_back_to_combined_formats() {
    let ds = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | SAMPLED;
    let dev = device(floor_with(&[(vk::Format::D32_SFLOAT_S8_UINT, ds)]));
    let attach = usages(&[FormatUsage::DepthStencilAttachment]);
    let map = |pf| vk_format(pf, &attach, dev.profile()).unwrap();
    let d24 = map(PixelFormat::Depth24UnormStencil8);
    assert_eq!(d24.format, vk::Format::D32_SFLOAT_S8_UINT);
    assert_eq!(d24.emulation, FormatEmulation::DepthWidened);
    assert_eq!(
        d24.aspects,
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    );
    let s8 = map(PixelFormat::Stencil8);
    assert_eq!(
        (s8.format, s8.emulation, s8.aspects),
        (
            vk::Format::D32_SFLOAT_S8_UINT,
            FormatEmulation::StencilInCombined,
            vk::ImageAspectFlags::STENCIL
        )
    );
    let d32 = map(PixelFormat::Depth32Float);
    assert_eq!(d32.emulation, FormatEmulation::DepthInCombined);
    // A stencil view maps to the same format as the texture it views.
    assert_eq!(map(PixelFormat::X24Stencil8).format, d24.format);
    // Every candidate is named when none is supported.
    let bare = device(floor_with(&[]));
    let err = vk_format(PixelFormat::Stencil8, &attach, bare.profile()).unwrap_err();
    let Error::Unsupported(u) = err else {
        panic!("{err:?}")
    };
    let Needs::AnyOf(alternatives) = u.needs else {
        panic!()
    };
    let formats: Vec<vk::Format> = alternatives
        .iter()
        .map(|alt| match alt[..] {
            [Capability::Format { format, .. }] => format,
            _ => panic!("{alt:?}"),
        })
        .collect();
    assert_eq!(
        formats,
        [
            vk::Format::S8_UINT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT
        ]
    );
}

/// A one- or two-channel 8-bit sRGB storage texture on a device that stores neither sRGB format
/// takes its UNORM twin's view, converting by the device's transfer (`SrgbEncode`), as the
/// four-channel ones do; without the UNORM format's storage the refusal names both routes.
#[test]
fn one_and_two_channel_srgb_storage_takes_the_unorm_alias() {
    use vk::Format as F;
    let storage = SAMPLED | vk::FormatFeatureFlags::STORAGE_IMAGE;
    let stored = device(floor_with(&[
        (F::R8_UNORM, storage),
        (F::R8G8_UNORM, storage),
        (F::R8_SRGB, SAMPLED),
        (F::R8G8_SRGB, SAMPLED),
    ]));
    for (pf, image, view) in [
        (PixelFormat::R8UnormSRGB, F::R8_SRGB, F::R8_UNORM),
        (PixelFormat::RG8UnormSRGB, F::R8G8_SRGB, F::R8G8_UNORM),
    ] {
        let mut want = StorageImageChoice::new(view, StorageImageClass::SrgbEncode);
        want.alias = Some(AliasView::new(image));
        assert_eq!(
            storage_image_choice(pf, stored.profile()),
            Ok(want),
            "{pf:?}"
        );
    }
    let unstored = device(floor_with(&[(F::R8_UNORM, SAMPLED), (F::R8_SRGB, SAMPLED)]));
    let Err(Error::Unsupported(u)) =
        storage_image_choice(PixelFormat::R8UnormSRGB, unstored.profile())
    else {
        panic!("R8Unorm_sRGB stored without R8_UNORM storage");
    };
    let named: Vec<Vec<vk::Format>> = match u.needs {
        Needs::AnyOf(routes) => routes
            .iter()
            .map(|route| {
                route
                    .iter()
                    .filter_map(|c| match c {
                        Capability::Format { format, .. } => Some(*format),
                        _ => None,
                    })
                    .collect()
            })
            .collect(),
        other => panic!("{other:?}"),
    };
    assert!(
        named[0].contains(&F::R8_SRGB) && named[1].contains(&F::R8_UNORM),
        "{named:?}"
    );
}

/// An `RGB9E5Float` storage texture takes its `R32_UINT` alias, holding Metal's word
/// (`SharedExponentBits`), on a device that does not store the format and on one that stores it
/// (whose own conversion rounds the mantissas half up, where Metal truncates them).
#[test]
fn shared_exponent_storage_takes_the_word_alias() {
    use vk::Format as F;
    let storage = SAMPLED | vk::FormatFeatureFlags::STORAGE_IMAGE;
    let unstored = device(floor_with(&[
        (F::R32_UINT, storage),
        (F::E5B9G9R9_UFLOAT_PACK32, SAMPLED),
    ]));
    let mut want = StorageImageChoice::new(F::R32_UINT, StorageImageClass::SharedExponentBits);
    want.alias = Some(AliasView::new(F::E5B9G9R9_UFLOAT_PACK32));
    assert_eq!(
        storage_image_choice(PixelFormat::RGB9E5Float, unstored.profile()),
        Ok(want)
    );
    let mut desc = floor_with(&[(F::R32_UINT, storage), (F::E5B9G9R9_UFLOAT_PACK32, storage)]);
    if let Some(e5) = desc.formats.formats.get_mut(&F::E5B9G9R9_UFLOAT_PACK32) {
        e5.optimal_tiling |= vk::FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT
            | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT;
    }
    desc.features
        .insert(VkFeature::ShaderStorageImageReadWithoutFormat);
    desc.features
        .insert(VkFeature::ShaderStorageImageWriteWithoutFormat);
    let stored = device(desc);
    assert_eq!(
        storage_image_choice(PixelFormat::RGB9E5Float, stored.profile()),
        Ok(want)
    );
}

/// The required fixture shape: on a profile where bgra8 lacks `STORAGE_IMAGE`, a bgra8 and an rgba8
/// storage texture get distinct choices (and so distinct keys).
#[test]
fn storage_choices_follow_d25() {
    let storage = SAMPLED | vk::FormatFeatureFlags::STORAGE_IMAGE;
    // The floor states neither formatless feature: its storage images are typed.
    let typed = device(floor_with(&[
        (vk::Format::R8G8B8A8_UNORM, storage),
        (vk::Format::B8G8R8A8_UNORM, SAMPLED),
        (vk::Format::R32_UINT, storage),
    ]));
    let choice = |pf, dev: &Device| storage_image_choice(pf, dev.profile());
    let rgba = choice(PixelFormat::RGBA8Unorm, &typed).unwrap();
    assert_eq!(
        rgba,
        StorageImageChoice::new(vk::Format::R8G8B8A8_UNORM, StorageImageClass::Identity)
    );
    let bgra = choice(PixelFormat::BGRA8Unorm, &typed).unwrap();
    assert_eq!(bgra.class, StorageImageClass::SwizzleBgra);
    assert_eq!(bgra.vk_format, vk::Format::R8G8B8A8_UNORM);
    assert_eq!(bgra.alias, Some(AliasView::new(vk::Format::B8G8R8A8_UNORM)));
    assert_ne!(rgba, bgra);
    let srgb = choice(PixelFormat::RGBA8UnormSRGB, &typed).unwrap();
    assert_eq!(srgb.class, StorageImageClass::SrgbEncode);
    assert_eq!(srgb.alias, Some(AliasView::new(vk::Format::R8G8B8A8_SRGB)));
    // bgra8 sRGB, whose own UNORM twin is declarable only without a format, takes both
    // conversions through rgba8.
    let mut srgb_bgra = StorageImageChoice::new(
        vk::Format::R8G8B8A8_UNORM,
        StorageImageClass::SrgbEncodeBgra,
    );
    srgb_bgra.alias = Some(AliasView::new(vk::Format::B8G8R8A8_SRGB));
    assert_eq!(choice(PixelFormat::BGRA8UnormSRGB, &typed), Ok(srgb_bgra));
    assert_eq!(
        choice(PixelFormat::Depth32Float, &typed),
        Err(Unsupported::not_representable(Feature::Format).into())
    );
    assert_eq!(
        choice(PixelFormat::RG16Float, &typed),
        Err(Unsupported::needs_any_of(
            Feature::Format,
            vec![vec![Capability::Format {
                format: vk::Format::R16G16_SFLOAT,
                usage: FormatUsage::StorageImage
            }]]
        )
        .into())
    );

    // Without `VK_KHR_maintenance5` an `A8Unorm` texture is an `R8_UNORM` image holding its alpha
    // in red, which the storage choice reads and writes as alpha (`AlphaInRed`), typed as `R8`.
    let held_in_red = StorageImageChoice::new(vk::Format::R8_UNORM, StorageImageClass::AlphaInRed);
    let typed_r8 = device(floor_with(&[(vk::Format::R8_UNORM, storage)]));
    assert_eq!(
        choice(PixelFormat::R8Unorm, &typed_r8),
        Ok(StorageImageChoice::new(
            vk::Format::R8_UNORM,
            StorageImageClass::Identity
        ))
    );
    assert_eq!(choice(PixelFormat::A8Unorm, &typed_r8), Ok(held_in_red));

    // Formatless: every storage-capable format of one numeric type shares the `Unknown` identity
    // key, which keeps the type.
    let mut desc = floor_with(&[
        (vk::Format::R8G8B8A8_UNORM, storage),
        (vk::Format::B8G8R8A8_UNORM, storage),
        (vk::Format::R8_UNORM, storage),
        (vk::Format::R8_UINT, storage),
        (vk::Format::R16G16B16A16_SINT, storage),
    ]);
    desc.features
        .insert(VkFeature::ShaderStorageImageReadWithoutFormat);
    desc.features
        .insert(VkFeature::ShaderStorageImageWriteWithoutFormat);
    let formatless = device(desc);
    let unknown = |numeric| {
        let mut c = StorageImageChoice::new(vk::Format::UNDEFINED, StorageImageClass::Identity);
        c.numeric = Some(numeric);
        c
    };
    let float = unknown(SampledType::Float);
    assert_eq!(choice(PixelFormat::RGBA8Unorm, &formatless), Ok(float));
    // `B8G8R8A8_UNORM` is not on the specification's list of formats the device features cover,
    // and this device reports no `VkFormatProperties3`: bgra8 is stored through the rgba8 alias.
    let mut swizzled =
        StorageImageChoice::new(vk::Format::R8G8B8A8_UNORM, StorageImageClass::SwizzleBgra);
    swizzled.alias = Some(AliasView::new(vk::Format::B8G8R8A8_UNORM));
    assert_eq!(choice(PixelFormat::BGRA8Unorm, &formatless), Ok(swizzled));
    assert_eq!(choice(PixelFormat::R8Unorm, &formatless), Ok(float));
    assert_eq!(
        choice(PixelFormat::R8Uint, &formatless),
        Ok(unknown(SampledType::Uint))
    );
    assert_eq!(
        choice(PixelFormat::RGBA16Sint, &formatless),
        Ok(unknown(SampledType::Sint))
    );
    assert_eq!(choice(PixelFormat::A8Unorm, &formatless), Ok(held_in_red));
    // bgra8 sRGB's UNORM twin is not accessed without a format here (no `VkFormatProperties3`):
    // both conversions through rgba8.
    assert_eq!(
        choice(PixelFormat::BGRA8UnormSRGB, &formatless),
        Ok(srgb_bgra)
    );
}

/// The RGBX formats are held in their RGBA twins: sampled through a view that reads alpha as
/// one, a storage image of the twin's choice converting the texels (`x_channel`), alpha absent
/// (1 out of bounds and at the border), no color attachment and no texture buffer yet.
/// Apple's bfloat formats are held raw in the unsigned integer format of their 16-bit words
/// ([`FormatEmulation::Bfloat`]): read as floats, sampled and stored through it, never filtered
/// by a device nor blended; a color attachment of the raw format; a multisample texture is none
/// (the runtime's descriptor rules). A fill's texel is each channel through `float` and `half`,
/// each to nearest even, then the bfloat nearest that, ties to even, a NaN `0x7fc0`.
#[test]
fn bfloat_formats_are_held_raw() {
    use vk::Format as F;
    use PixelFormat as P;
    let forms = [
        (P::R16Bfloat, ExtendedRange::R16Bfloat, F::R16_UINT, 1),
        (P::RG16Bfloat, ExtendedRange::Rg16Bfloat, F::R16G16_UINT, 2),
        (
            P::RGBA16Bfloat,
            ExtendedRange::Rgba16Bfloat,
            F::R16G16B16A16_UINT,
            4,
        ),
    ];
    let named = PixelFormat::ALL
        .iter()
        .filter(|&&pf| ExtendedRange::of(pf).is_some_and(ExtendedRange::bfloat))
        .count();
    assert_eq!(named, forms.len());
    // `RGB16Bfloat`, which no Metal level measured describes, stays unheld.
    assert!(matches!(P::RGB16Bfloat, formats_named_not_held!()));
    for (pf, form, raw, components) in forms {
        assert_eq!(ExtendedRange::of(pf), Some(form), "{pf:?}");
        assert!(form.bfloat() && !form.srgb() && !form.padded(), "{pf:?}");
        assert!(form.holds_metal_bytes(), "{pf:?}");
        assert_eq!(form.raw_format(), raw, "{pf:?}");
        assert_eq!(form.raw_components(), components, "{pf:?}");
        assert_eq!(direct_format(pf), Some(raw), "{pf:?}");
        assert_eq!(byte_format(pf), Some(raw), "{pf:?}");
        assert_eq!(sampled_type(pf), Some(SampledType::Float), "{pf:?}");
        assert_eq!(out_of_bounds_alpha_one(pf), Some(components < 4), "{pf:?}");
        let Repr::Candidates(aspects, candidates) = make_repr(pf) else {
            panic!("{pf:?} has no candidate");
        };
        assert_eq!(aspects, vk::ImageAspectFlags::COLOR);
        assert_eq!(candidates.len(), 1, "{pf:?}");
        assert_eq!(
            (candidates[0].format, candidates[0].emulation),
            (raw, FormatEmulation::Bfloat)
        );
        for (u, admitted) in [
            (FormatUsage::SampledImage, true),
            (FormatUsage::SampledImageFilterLinear, false),
            (FormatUsage::StorageImage, true),
            (FormatUsage::ColorAttachment, true),
            (FormatUsage::ColorAttachmentBlend, false),
            (FormatUsage::UniformTexelBuffer, false),
            (FormatUsage::StorageTexelBuffer, false),
        ] {
            assert_eq!(FormatEmulation::Bfloat.admits(u), admitted, "{pf:?} {u:?}");
        }
        // Every device holds a color attachment of it in the raw format, and blends none.
        for (stem, device) in committed_devices() {
            let attachment = vk_format(
                pf,
                &usages(&[FormatUsage::ColorAttachment]),
                device.profile(),
            )
            .unwrap_or_else(|e| panic!("{pf:?} on {stem}: {e:?}"));
            assert_eq!(
                (attachment.format, attachment.emulation),
                (raw, FormatEmulation::Bfloat),
                "{pf:?} on {stem}"
            );
            let blended = vk_format(
                pf,
                &usages(&[
                    FormatUsage::ColorAttachment,
                    FormatUsage::ColorAttachmentBlend,
                ]),
                device.profile(),
            );
            assert!(blended.is_err(), "{pf:?} on {stem}: {blended:?}");
        }
    }
    // Clears under `ThroughHalf`: the color through `float` and `half` to nearest even, then the
    // bfloat code nearest, ties to even (a tie of the `half` rounded down and one rounded up),
    // the overflow of the `half` its infinity, a NaN `0x7fc0`.
    let texel = |form: ExtendedRange, c: [f64; 4]| form.clear_texel(c, ClearRoute::ThroughHalf);
    let one_tie_down = 1.0 + 1.0 / 256.0;
    let one_tie_up = 1.0 + 3.0 / 256.0;
    assert_eq!(
        texel(
            ExtendedRange::Rgba16Bfloat,
            [one_tie_down, one_tie_up, -0.0, f64::NAN]
        ),
        [0x3f80, 0x3f82, 0x8000, 0x7fc0]
    );
    assert_eq!(
        texel(
            ExtendedRange::Rgba16Bfloat,
            [65520.0, -65520.0, 65504.0, 0.1]
        ),
        [0x7f80, 0xff80, 0x4780, 0x3dcd]
    );
    assert_eq!(
        texel(ExtendedRange::Rg16Bfloat, [0.25, 0.5, 0.75, 1.0]),
        [0x3e80, 0x3f00, 0, 0]
    );
    assert_eq!(
        texel(ExtendedRange::R16Bfloat, [1.5, -0.5, 7.0, 0.0]),
        [0x3fc0, 0, 0, 0]
    );
    // Clears under `Nearest`: the color through `float` to nearest even, then the bfloat code
    // nearest, ties to even, subnormals kept (the consumer's M3 Max: `-3e38`, `1e-40`, `70000`
    // and `1 + 2^-8 + 2^-23`, where `ThroughHalf` gives `0xff80`, 0, `0x7f80` and `0x3f80`), a
    // `half`'s value as under `ThroughHalf`, the infinity past the largest `float`, a NaN its
    // `float`'s sign and top payload bits, quieted.
    let nearest = |form: ExtendedRange, c: [f64; 4]| form.clear_texel(c, ClearRoute::Straight);
    let consumers = [-3e38, 1e-40, 70000.0, 1.0 + 1.0 / 256.0 + 2f64.powi(-23)];
    assert_eq!(
        nearest(ExtendedRange::Rgba16Bfloat, consumers),
        [0xff62, 0x0001, 0x4789, 0x3f81]
    );
    assert_eq!(
        texel(ExtendedRange::Rgba16Bfloat, consumers),
        [0xff80, 0x0000, 0x7f80, 0x3f80]
    );
    assert_eq!(
        nearest(
            ExtendedRange::Rgba16Bfloat,
            [-2.5, one_tie_down, one_tie_up, -0.0]
        ),
        [0xc020, 0x3f80, 0x3f82, 0x8000]
    );
    assert_eq!(
        nearest(
            ExtendedRange::Rgba16Bfloat,
            [3.402_823_466_385_288_6e38, 1e39, -1e39, f64::NAN]
        ),
        [0x7f80, 0x7f80, 0xff80, 0x7fc0]
    );
    assert_eq!(
        nearest(
            ExtendedRange::Rg16Bfloat,
            [1.401_298_464_324_817e-45, -f64::NAN, 1.0, 1.0]
        ),
        [0x0000, 0xffc0, 0, 0]
    );
    assert_eq!(
        nearest(
            ExtendedRange::Rgba16Bfloat,
            [
                f64::from_bits(0x7ff8_4000_0000_0000),
                f64::from_bits(0xfff0_0400_0000_0000),
                f64::from_bits(0x7fff_ffff_ffff_ffff),
                f64::from_bits(0x7ff4_0000_0000_0000),
            ]
        ),
        [0x7fc2, 0xffc0, 0x7fff, 0x7fe0]
    );
    // The bfloat rounding alone: ties to even, either sign, and the infinity past the largest.
    assert_eq!(bfloat_nearest_even(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(bfloat_nearest_even(f32::from_bits(0x3f81_8000)), 0x3f82);
    assert_eq!(bfloat_nearest_even(f32::from_bits(0xbf81_8001)), 0xbf82);
    assert_eq!(bfloat_nearest_even(f32::from_bits(0x7f7f_8000)), 0x7f80);
    assert_eq!(bfloat_nearest_even(f32::NEG_INFINITY), 0xff80);
    assert_eq!(bfloat_nearest_even(-f32::NAN), 0x7fc0);
}

#[test]
fn rgbx_formats_are_held_in_their_rgba_twins() {
    use vk::ComponentSwizzle as S;
    use vk::Format as F;
    use PixelFormat as P;
    let twins = [
        (P::RGBX8Unorm, P::RGBA8Unorm, F::R8G8B8A8_UNORM),
        (P::RGBX8UnormSRGB, P::RGBA8UnormSRGB, F::R8G8B8A8_SRGB),
        (P::RGBX8Snorm, P::RGBA8Snorm, F::R8G8B8A8_SNORM),
        (P::RGBX8Uint, P::RGBA8Uint, F::R8G8B8A8_UINT),
        (P::RGBX8Sint, P::RGBA8Sint, F::R8G8B8A8_SINT),
        (P::RGBX16Uint, P::RGBA16Uint, F::R16G16B16A16_UINT),
        (P::RGBX16Sint, P::RGBA16Sint, F::R16G16B16A16_SINT),
        (P::RGBX16Float, P::RGBA16Float, F::R16G16B16A16_SFLOAT),
        (P::RGBX32Uint, P::RGBA32Uint, F::R32G32B32A32_UINT),
        (P::RGBX32Sint, P::RGBA32Sint, F::R32G32B32A32_SINT),
        (P::RGBX32Float, P::RGBA32Float, F::R32G32B32A32_SFLOAT),
    ];
    assert_eq!(
        PixelFormat::ALL.iter().filter(|&&pf| x_channel(pf)).count(),
        twins.len()
    );
    for &(pf, twin, format) in &twins {
        assert!(x_channel(pf) && !x_channel(twin), "{pf:?}");
        assert_eq!(direct_format(pf), Some(format), "{pf:?}");
        assert_eq!(byte_format(pf), direct_format(twin), "{pf:?}");
        assert_eq!(sampled_type(pf), sampled_type(twin), "{pf:?}");
        assert_eq!(out_of_bounds_alpha_one(pf), Some(true), "{pf:?}");
        assert_eq!(channels(pf), Some([true, true, true, false]), "{pf:?}");
        let Repr::Candidates(aspects, candidates) = make_repr(pf) else {
            panic!("{pf:?} has no candidate");
        };
        assert_eq!(aspects, vk::ImageAspectFlags::COLOR);
        assert_eq!(candidates.len(), 1, "{pf:?}");
        let c = candidates[0];
        assert_eq!(c.emulation, FormatEmulation::XChannel, "{pf:?}");
        assert_eq!(
            [
                c.components.r,
                c.components.g,
                c.components.b,
                c.components.a
            ],
            [S::IDENTITY, S::IDENTITY, S::IDENTITY, S::ONE],
            "{pf:?}"
        );
        for (u, admitted) in [
            (FormatUsage::SampledImage, true),
            (FormatUsage::SampledImageFilterLinear, true),
            (FormatUsage::StorageImage, true),
            (FormatUsage::ColorAttachment, false),
            (FormatUsage::ColorAttachmentBlend, false),
            (FormatUsage::UniformTexelBuffer, false),
            (FormatUsage::StorageTexelBuffer, false),
        ] {
            assert_eq!(c.emulation.admits(u), admitted, "{pf:?} {u:?}");
        }
    }
    // A storage choice is the twin's, converting the texels; the sRGB format's through the
    // twin's UNORM alias where the sRGB format is no storage format.
    let storage = SAMPLED | vk::FormatFeatureFlags::STORAGE_IMAGE;
    let typed = device(floor_with(&[
        (F::R8G8B8A8_UNORM, storage),
        (F::R8G8B8A8_SRGB, SAMPLED),
        (F::R8G8B8A8_UINT, storage),
    ]));
    let choice = |pf| storage_image_choice(pf, typed.profile());
    let mut want = StorageImageChoice::new(F::R8G8B8A8_UNORM, StorageImageClass::Identity);
    want.x_channel = true;
    assert_eq!(choice(P::RGBX8Unorm), Ok(want));
    assert_eq!(
        choice(P::RGBA8Unorm),
        Ok(StorageImageChoice::new(
            F::R8G8B8A8_UNORM,
            StorageImageClass::Identity
        ))
    );
    let mut want = StorageImageChoice::new(F::R8G8B8A8_UNORM, StorageImageClass::SrgbEncode);
    want.alias = Some(AliasView::new(F::R8G8B8A8_SRGB));
    want.x_channel = true;
    assert_eq!(choice(P::RGBX8UnormSRGB), Ok(want));
    let mut want_uint = StorageImageChoice::new(F::R8G8B8A8_UINT, StorageImageClass::Identity);
    want_uint.x_channel = true;
    assert_eq!(choice(P::RGBX8Uint), Ok(want_uint));
    // Where the device stores the sRGB format itself (without a declared format: SPIR-V names
    // none for sRGB), the twin stores into it, but the RGBX format's writes keep the UNORM alias:
    // the device would store the fourth lane, green's X, linearly; a slot that only reads it
    // views the sRGB image.
    let mut desc = floor_with(&[(F::R8G8B8A8_UNORM, storage), (F::R8G8B8A8_SRGB, storage)]);
    if let Some(srgb) = desc.formats.formats.get_mut(&F::R8G8B8A8_SRGB) {
        srgb.optimal_tiling |= vk::FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT
            | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT;
    }
    desc.features
        .insert(VkFeature::ShaderStorageImageReadWithoutFormat);
    desc.features
        .insert(VkFeature::ShaderStorageImageWriteWithoutFormat);
    desc.khr_format_feature_flags2 = true;
    let srgb_storage = device(desc);
    let profile = srgb_storage.profile();
    assert_eq!(
        storage_image_choice(P::RGBA8UnormSRGB, profile).map(|c| c.class),
        Ok(StorageImageClass::Identity)
    );
    assert_eq!(storage_image_choice(P::RGBX8UnormSRGB, profile), Ok(want));
    assert_eq!(
        storage_choice(P::RGBX8UnormSRGB, StorageUse::texture(true, false), profile).map(|c| (
            c.class,
            c.alias.is_some(),
            c.x_channel
        )),
        Ok((StorageImageClass::Identity, false, true))
    );
    // A texture buffer: its view cannot read alpha as one.
    assert_eq!(
        storage_choice(
            P::RGBX8Unorm,
            StorageUse::texel_buffer(true, true),
            typed.profile()
        ),
        Err(Unsupported::not_implemented(Feature::Format).into())
    );
    // Not yet a color attachment.
    assert_eq!(
        vk_format(
            P::RGBX8Unorm,
            &usages(&[FormatUsage::ColorAttachment]),
            typed.profile()
        ),
        Err(Unsupported::not_implemented(Feature::Format).into())
    );
}

/// Every Identity choice the committed profiles produce is declarable in SPIR-V.
#[test]
fn storage_choices_on_committed_profiles_are_declarable() {
    for (stem, dev) in committed_devices() {
        for &pf in PixelFormat::ALL {
            let Ok(choice) = storage_image_choice(pf, dev.profile()) else {
                continue;
            };
            if choice.vk_format != vk::Format::UNDEFINED {
                let formatless = accessed_without_format(
                    choice.vk_format,
                    StorageUse::READ_WRITE_TEXTURE,
                    dev.profile(),
                );
                assert!(
                    formatless || spirv_image_format(choice.vk_format).is_some(),
                    "{stem} {pf:?} {choice:?}"
                );
            }
            let view = choice.vk_format;
            if view != vk::Format::UNDEFINED {
                assert!(dev
                    .profile()
                    .supports_format(view, FormatUsage::StorageImage));
            }
        }
    }
}

/// A half-float storage texture takes `Identity` on every committed profile, for every use, written
/// or read, image or texel buffer: the device converts each value and NaN (`lower::texture`'s
/// `half_storage_facts` and `half_write_nans` record what it relies on or notes). So does another
/// float format.
#[test]
fn half_float_storage_takes_its_own_format() {
    let mut seen = 0;
    for (stem, dev) in committed_devices() {
        let p = dev.profile();
        for pf in [
            PixelFormat::R16Float,
            PixelFormat::RG16Float,
            PixelFormat::RGBA16Float,
            PixelFormat::RGBX16Float,
        ] {
            for use_ in [
                StorageUse::READ_WRITE_TEXTURE,
                StorageUse::texture(false, true),
                StorageUse::texture(true, false),
                StorageUse::texel_buffer(false, true),
            ] {
                if let Ok(choice) = storage_choice(pf, use_, p) {
                    assert_eq!(
                        choice.class,
                        StorageImageClass::Identity,
                        "{stem} {pf:?} {use_:?}"
                    );
                    assert_eq!(choice.alias, None, "{stem} {pf:?} {use_:?}");
                    seen += 1;
                }
            }
        }
        if let Ok(choice) =
            storage_choice(PixelFormat::RGBA32Float, StorageUse::READ_WRITE_TEXTURE, p)
        {
            assert_eq!(
                choice.class,
                StorageImageClass::Identity,
                "{stem} RGBA32Float"
            );
        }
    }
    assert!(seen > 0, "no committed profile stores a half-float texture");
}

/// A `BGR10A2Unorm` storage texture or texture buffer is stored through its own `A2R10G10B10`
/// format where the device stores that format and the shader can declare it, else through an
/// `A2B10G10R10` view of the same bytes whose red and blue the shader exchanges (`SwizzleBgra`):
/// NVIDIA's and lavapipe's committed profiles, which store no `A2R10G10B10`, take the view,
/// RADV's, ANV's and MoltenVK's their own format.
#[test]
fn bgr10a2_storage_takes_the_rgb10a2_view_where_its_own_format_is_not_stored() {
    let own = vk::Format::A2R10G10B10_UNORM_PACK32;
    let view = vk::Format::A2B10G10R10_UNORM_PACK32;
    let mut viewed = BTreeSet::new();
    let mut direct = BTreeSet::new();
    for (stem, dev) in committed_devices() {
        let p = dev.profile();
        for use_ in [
            StorageUse::READ_WRITE_TEXTURE,
            StorageUse::texture(true, false),
            StorageUse::texel_buffer(false, true),
        ] {
            let usage = use_.usage();
            let Ok(choice) = storage_choice(PixelFormat::BGR10A2Unorm, use_, p) else {
                assert!(!p.supports_format(view, usage), "{stem} {use_:?}");
                continue;
            };
            match choice.alias {
                None => {
                    assert!(p.supports_format(own, usage), "{stem} {use_:?}");
                    assert_eq!(choice.class, StorageImageClass::Identity, "{stem} {use_:?}");
                    direct.insert(stem.clone());
                }
                Some(alias) => {
                    assert_eq!(alias, AliasView::new(own), "{stem} {use_:?}");
                    assert_eq!(choice.vk_format, view, "{stem} {use_:?}");
                    assert_eq!(
                        choice.class,
                        StorageImageClass::SwizzleBgra,
                        "{stem} {use_:?}"
                    );
                    viewed.insert(stem.clone());
                }
            }
        }
    }
    for stem in ["nvidia-rtx5080-laptop", "lavapipe", "gate-lavapipe-x86_64"] {
        assert!(viewed.contains(stem), "{stem}: {viewed:?}");
    }
    for stem in ["amd-780m-radv", "intel-irisxe-rplp-anv"] {
        assert!(direct.contains(stem), "{stem}: {direct:?}");
    }
}

/// The query list covers every candidate, every vertex format, and every neutral format the
/// binder creates, and is sorted without duplicates. Every entry is exposed by a core version or
/// a vocabulary extension; a device is asked only about the ones it exposes.
#[test]
fn queries_cover_candidates_and_neutrals() {
    let q = FormatSupport::queries();
    assert!(q.windows(2).all(|w| w[0] < w[1]));
    // Every vocabulary extension that defines a queried format.
    let every: BTreeSet<VkExtension> = q
        .iter()
        .flat_map(|f| f.exposure().map_or(&[][..], |x| x.extensions))
        .filter_map(|n| crate::device::khronos::extension_from_name(n))
        .collect();
    let newest = ApiVersion { major: 1, minor: 4 };
    let floor = ApiVersion::V1_2;
    for &f in q {
        let x = f
            .exposure()
            .unwrap_or_else(|| panic!("{f:?} is not a registry format"));
        assert!(
            crate::device::enables::format_exposed(f, newest, &BTreeSet::new())
                || crate::device::enables::format_exposed(f, floor, &every),
            "{f:?}: {x:?}"
        );
    }
    // The 1.2 floor without extensions asks about core 1.0 formats only; each gated entry is
    // asked where its extension is available.
    let plain = FormatSupport::queries_for(floor, &BTreeSet::new());
    assert!(plain
        .iter()
        .all(|f| f.exposure().unwrap().core == Some(ApiVersion { major: 1, minor: 0 })));
    let gated: Vec<vk::Format> = q.iter().copied().filter(|f| !plain.contains(f)).collect();
    assert!(gated.contains(&vk::Format::A8_UNORM), "{gated:?}");
    assert!(gated.contains(&vk::Format::ASTC_4X4_SFLOAT_BLOCK));
    assert!(gated.contains(&vk::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG));
    assert_eq!(FormatSupport::queries_for(floor, &every), q);
    let astc = BTreeSet::from([VkExtension::ExtTextureCompressionAstcHdr]);
    let with_astc = FormatSupport::queries_for(floor, &astc);
    assert!(with_astc.contains(&vk::Format::ASTC_4X4_SFLOAT_BLOCK));
    assert!(!with_astc.contains(&vk::Format::A8_UNORM));
    let set: BTreeSet<vk::Format> = q.iter().copied().collect();
    for f in crate::metal::vertex::vertex_formats_queried() {
        assert!(set.contains(&f), "vertex format {f:?} is not queried");
    }
    for &pf in PixelFormat::ALL {
        if let Repr::Candidates(_, c) = repr(pf) {
            assert!(c.iter().all(|c| set.contains(&c.format)), "{pf:?}");
        }
    }
    // The view of a `ulong` slot a `ulong` atomic updates, which no Metal format maps to.
    assert!(set.contains(&vk::Format::R64_UINT));
    for (stem, dev) in committed_devices() {
        // Neutral images of a declared storage format take it from the profile's own table;
        // the others use fixed formats the consumer must be able to ask about.
        for neutral in dev.neutral_kinds() {
            if let crate::bind::NeutralResource::Image {
                storage_format: None,
                ..
            } = neutral
            {
                let f = neutral.format().unwrap();
                assert!(set.contains(&f), "{stem}: neutral {f:?} is not queried");
            }
        }
    }
}

#[test]
fn texture_usages_and_view_types() {
    let rw = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    assert_eq!(
        texture_usages(PixelFormat::RGBA8Unorm, rw),
        usages(&[FormatUsage::SampledImage, FormatUsage::ColorAttachment])
    );
    assert_eq!(
        texture_usages(PixelFormat::Depth32Float, TextureUsage::RENDER_TARGET),
        usages(&[FormatUsage::DepthStencilAttachment])
    );
    assert!(texture_usages(PixelFormat::R32Uint, TextureUsage::SHADER_WRITE).is_empty());
    for &ty in TextureType::ALL {
        assert_eq!(
            image_view_type(ty).is_none(),
            ty == TextureType::TextureBuffer
        );
        assert_eq!(image_type(ty).is_none(), ty == TextureType::TextureBuffer);
    }
    assert_eq!(
        image_type(TextureType::CubeArray),
        Some((
            vk::ImageType::TYPE_2D,
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        ))
    );
    assert_eq!(
        image_view_type(TextureType::Type2DMultisampleArray),
        Some(vk::ImageViewType::TYPE_2D_ARRAY)
    );
}

/// Every pixel format some committed profile admits as a color attachment states the components
/// a fragment output to it needs, and no other format does: the table covers every color-renderable
/// format, by its channels.
#[test]
fn every_color_attachment_format_states_its_output_components() {
    let devices = committed_devices();
    let usages = BTreeSet::from([FormatUsage::ColorAttachment]);
    for &pf in PixelFormat::ALL {
        let renderable = devices
            .iter()
            .any(|(_, d)| vk_format(pf, &usages, d.profile()).is_ok());
        let components = color_output_components(pf);
        if renderable {
            assert!(
                components.is_some(),
                "{pf:?} is a color attachment somewhere"
            );
        }
        if components.is_some() {
            // The extended-range formats have no Vulkan representation, so no aspects.
            assert!(
                aspects(pf).is_none_or(|a| a == vk::ImageAspectFlags::COLOR),
                "{pf:?}"
            );
        }
    }
    assert_eq!(color_output_components(PixelFormat::A8Unorm), Some(4));
    assert_eq!(color_output_components(PixelFormat::RG11B10Float), Some(3));
    assert_eq!(color_output_components(PixelFormat::Depth32Float), None);
}

/// An out-of-bounds read reads alpha 1 exactly where the format has no alpha. Each entry is
/// measured on an Apple M3 Max (macOS 26.7): a 2D texture of the format read out of bounds (and a
/// texel buffer of `R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `R32Float`, `R32Uint`, `R32Sint`, and
/// `RGBA32Uint`, which agreed). A depth format's missing components are undefined in Metal (the
/// M3 leaves them unwritten), so its 1 is Vulkan's substitution, which nothing contradicts.
#[test]
fn out_of_bounds_alpha_follows_the_measured_formats() {
    use PixelFormat as P;
    let measured = [
        (P::A8Unorm, false),
        (P::B5G6R5Unorm, true),
        (P::BGR10A2Unorm, false),
        (P::BGRA8Unorm, false),
        (P::BC1RGBA, false),
        (P::BC4RUnorm, true),
        (P::R8Unorm, true),
        (P::R8Snorm, true),
        (P::R8Uint, true),
        (P::R16Float, true),
        (P::R16Uint, true),
        (P::R32Float, true),
        (P::R32Sint, true),
        (P::R32Uint, true),
        (P::RG8Unorm, true),
        (P::RG8Uint, true),
        (P::RG16Float, true),
        (P::RG16Sint, true),
        (P::RG32Uint, true),
        (P::RG11B10Float, true),
        (P::RGB9E5Float, true),
        (P::RGB10A2Unorm, false),
        (P::RGB10A2Uint, false),
        (P::RGBA8Unorm, false),
        (P::RGBA8UnormSRGB, false),
        (P::RGBA8Uint, false),
        (P::RGBA8Sint, false),
        (P::RGBA16Float, false),
        (P::RGBA32Uint, false),
        (P::RGBA32Float, false),
    ];
    for (pf, one) in measured {
        assert_eq!(out_of_bounds_alpha_one(pf), Some(one), "{pf:?}");
    }
    assert_eq!(out_of_bounds_alpha_one(P::Depth32Float), Some(true));
    // A `YCBCR8_420_2P` texture's texels read alpha 1; outside them its 1 is
    // the alpha it has none of.
    assert_eq!(out_of_bounds_alpha_one(P::YCBCR84202P), Some(true));
    // Every format answers but the placeholders, which are no format, and the formats Metal
    // makes no texture of from a descriptor alone of which this crate holds none.
    for &pf in PixelFormat::ALL {
        let placeholder = matches!(pf, P::Invalid | P::Unspecialized);
        let held_none = (ycbcr_format(pf).is_some() && !ycbcr(pf))
            || matches!(pf, P::RGBA16FloatXR | P::AnyDepth | P::AnyStencil)
            || matches!(pf, astc_without_a_vulkan_block!())
            || matches!(pf, formats_named_not_held!());
        assert_eq!(
            out_of_bounds_alpha_one(pf).is_none(),
            placeholder || held_none,
            "{pf:?}"
        );
    }
}

/// `imageblock_sample_length` answers what Metal answered for each measured list of color
/// formats on an M3 Max and an M4: texels in index order at their component's alignment, unrounded;
/// unset and `Invalid` formats take no room.
#[test]
fn imageblock_sample_lengths_are_metals() {
    use PixelFormat as P;
    let rgba32f = Some(P::RGBA32Float);
    for (formats, metal) in [
        (vec![Some(P::R8Unorm)], 1),
        (vec![Some(P::R16Float)], 2),
        (vec![Some(P::R32Float), Some(P::R8Unorm)], 5),
        (vec![Some(P::R8Unorm), Some(P::R32Float)], 8),
        (vec![Some(P::R16Float), Some(P::RGBA16Float)], 10),
        (vec![Some(P::R8Unorm), Some(P::RGBA8Unorm)], 5),
        (vec![Some(P::R32Float), Some(P::RGBA32Float)], 20),
        (
            vec![Some(P::R32Float), Some(P::R16Float), Some(P::R32Uint)],
            12,
        ),
        (vec![rgba32f; 8], 128),
        (
            vec![
                Some(P::R8Unorm),
                Some(P::R16Float),
                Some(P::R8Unorm),
                Some(P::R32Float),
            ],
            12,
        ),
        (
            vec![None, Some(P::R8Unorm), Some(P::Invalid), Some(P::R32Float)],
            8,
        ),
        (vec![], 0),
    ] {
        assert_eq!(
            imageblock_sample_length(&formats),
            Some(metal),
            "{formats:?}"
        );
    }
}

/// `color_record` places each color where Metal's render pipeline state writes it and its pass
/// reads it, on an M4: in index order at the component's alignment, an unset color taking no room;
/// its end is `imageblock_sample_length`.
#[test]
fn color_records_are_metals() {
    use PixelFormat as P;
    for (formats, metal) in [
        // `[R8Unorm, RGBA8Unorm]`'s color 1 at byte 1, which `[R8Unorm, R32Float]`'s reads
        // from byte 4 (`h_r8_rgba8_in_r8_r32f`: color 1 stored `df000000`).
        (
            vec![Some(P::R8Unorm), Some(P::RGBA8Unorm)],
            vec![Some((0, 1)), Some((1, 4))],
        ),
        (
            vec![Some(P::R8Unorm), Some(P::R32Float)],
            vec![Some((0, 1)), Some((4, 4))],
        ),
        // A state with color 1 alone writes it at byte 0, the pass's color 0
        // (`h_none_rgba8_in_rgba8_rgba8`).
        (vec![None, Some(P::RGBA8Unorm)], vec![None, Some((0, 4))]),
        (
            vec![Some(P::Invalid), Some(P::RGBA8Unorm)],
            vec![None, Some((0, 4))],
        ),
        // `[RGBA16Float, RGBA8Unorm]`'s color 1 at byte 8, past `[RGBA8Unorm, RGBA8Unorm]`'s
        // eight bytes (`h_rgba16f_rgba8_in_rgba8_rgba8`).
        (
            vec![Some(P::RGBA16Float), Some(P::RGBA8Unorm)],
            vec![Some((0, 8)), Some((8, 4))],
        ),
        (
            vec![Some(P::BGR10A2Unorm), Some(P::R16Float)],
            vec![Some((0, 4)), Some((4, 2))],
        ),
        (vec![], vec![]),
    ] {
        assert_eq!(color_record(&formats), Some(metal.clone()), "{formats:?}");
        let end = metal
            .iter()
            .flatten()
            .map(|&(o, s)| o + s)
            .max()
            .unwrap_or(0);
        assert_eq!(imageblock_sample_length(&formats), Some(end), "{formats:?}");
    }
}

/// `explicit_imageblock_sample_length` answers what Metal answered for a render pipeline whose
/// fragment function declares an explicit imageblock, on an M3 Max and an M4: the data size, then
/// the color attachments' bytes rounded up to 8 and then to the layout's alignment. Each row is a
/// layout (its data size and alignment) and color formats.
#[test]
fn explicit_imageblock_sample_lengths_are_metals() {
    use PixelFormat as P;
    let r8 = Some(P::R8Unorm);
    let rgba32f = Some(P::RGBA32Float);
    let rgba8x4 = vec![Some(P::RGBA8Unorm); 4];
    for (data_size, alignment, formats, metal) in [
        // `char`
        (1, 1, vec![], 1),
        (1, 1, vec![r8], 9),
        (1, 1, vec![rgba32f], 17),
        (1, 1, vec![r8, rgba32f], 25),
        (1, 1, rgba8x4.clone(), 17),
        // `half`
        (2, 2, vec![r8], 10),
        (2, 2, vec![r8, rgba32f], 26),
        // three `char`s
        (3, 1, vec![r8], 11),
        // three `half`s
        (6, 2, vec![rgba32f], 22),
        (6, 2, vec![r8, rgba32f], 30),
        // `float2`
        (8, 8, vec![r8], 16),
        (8, 8, vec![r8, rgba32f], 32),
        // `float`, `half` and `uint`
        (12, 4, vec![r8, rgba32f], 36),
        (12, 4, rgba8x4.clone(), 28),
        // `float4`
        (16, 16, vec![], 16),
        (16, 16, vec![r8], 32),
        (16, 16, vec![rgba32f], 32),
        (16, 16, vec![r8, rgba32f], 48),
        // `packed_float4`, and four `float`s
        (16, 4, vec![r8], 24),
        (16, 4, vec![rgba32f], 32),
        (16, 4, vec![r8, rgba32f], 40),
        // `float4` and `float`
        (32, 16, vec![], 32),
        (32, 16, vec![r8], 48),
        (32, 16, vec![r8, rgba32f], 64),
        (32, 16, rgba8x4, 48),
    ] {
        let colors = imageblock_sample_length(&formats).expect("measured formats pack");
        assert_eq!(
            explicit_imageblock_sample_length(data_size, alignment, colors),
            Some(metal),
            "{data_size} bytes aligned to {alignment}, {formats:?}"
        );
    }
}

/// [`ycbcr_format`] describes exactly the 69 two-plane YCbCr formats, each as its Metal name
/// states (`MTLPixelFormat{YCBCR|RGB}{bits}_{420|422|444}_2P[_PACKED][_{sRGB|PQ|HLG|XR}]`), and
/// this crate holds a texture of the ones of unpacked 8- and 10-bit codes read converted, through
/// the sRGB curve or as codes, and of packed 10-bit codes read converted or through the sRGB
/// curve, whose texels read three channels and alpha 1.
#[test]
fn ycbcr_formats_are_the_two_plane_family_as_named() {
    let mut described = 0;
    for &pf in PixelFormat::ALL {
        let name = pf
            .name()
            .strip_prefix("MTLPixelFormat")
            .expect("a Metal name");
        let Some(format) = ycbcr_format(pf) else {
            let two_plane = name.ends_with("_2P") || name.contains("_2P_");
            // The other two-plane formats are the color word and alpha plane ones,
            // `RGB10A8_2P_XR10` and Metal's `RGB10A8_2P` and `RGB10A8Uint_2P` families.
            assert!(!two_plane || name.starts_with("RGB10A8"), "{name}");
            continue;
        };
        described += 1;
        let (kind, rest) = name
            .strip_prefix("YCBCR")
            .map(|rest| ("YCBCR", rest))
            .or_else(|| name.strip_prefix("RGB").map(|rest| ("RGB", rest)))
            .expect("a YCBCR or RGB name");
        let mut words = rest.split('_');
        let bits: u32 = words.next().unwrap().parse().unwrap();
        let subsampling = match words.next().unwrap() {
            "420" => ChromaSubsampling::Chroma420,
            "422" => ChromaSubsampling::Chroma422,
            "444" => ChromaSubsampling::Chroma444,
            other => panic!("{name}: subsampling {other}"),
        };
        assert_eq!(words.next(), Some("2P"), "{name}");
        let mut words = words.peekable();
        let packed = words.next_if_eq(&"PACKED").is_some();
        let reads = match (kind, words.next()) {
            ("RGB", None) => YCbCrReads::Codes,
            ("YCBCR", None) => YCbCrReads::Converted,
            ("YCBCR", Some("sRGB")) => YCbCrReads::Srgb,
            ("YCBCR", Some("PQ")) => YCbCrReads::Pq,
            ("YCBCR", Some("HLG")) => YCbCrReads::Hlg,
            ("YCBCR", Some("XR")) => YCbCrReads::ExtendedRange,
            other => panic!("{name}: {other:?}"),
        };
        assert_eq!(words.next(), None, "{name}");
        assert_eq!(
            format,
            YCbCrFormat {
                bits,
                subsampling,
                packed,
                reads
            },
            "{name}"
        );
        let held = matches!(
            pf,
            PixelFormat::YCBCR84202P
                | PixelFormat::YCBCR104202P
                | PixelFormat::RGB84202P
                | PixelFormat::RGB104202P
                | PixelFormat::YCBCR84202PSRGB
                | PixelFormat::YCBCR104202PSRGB
                | PixelFormat::YCBCR84222P
                | PixelFormat::YCBCR84442P
                | PixelFormat::YCBCR104222P
                | PixelFormat::YCBCR104442P
                | PixelFormat::YCBCR84222PSRGB
                | PixelFormat::YCBCR84442PSRGB
                | PixelFormat::YCBCR104222PSRGB
                | PixelFormat::YCBCR104442PSRGB
                | PixelFormat::RGB84222P
                | PixelFormat::RGB84442P
                | PixelFormat::RGB104222P
                | PixelFormat::RGB104442P
                | PixelFormat::YCBCR104202PPACKED
                | PixelFormat::YCBCR104222PPACKED
                | PixelFormat::YCBCR104442PPACKED
                | PixelFormat::YCBCR104202PPACKEDSRGB
                | PixelFormat::YCBCR104222PPACKEDSRGB
                | PixelFormat::YCBCR104442PPACKEDSRGB
        );
        assert_eq!(ycbcr(pf), held, "{name}");
        assert_eq!(
            channels(pf),
            ycbcr(pf).then_some([true, true, true, false]),
            "{name}"
        );
    }
    assert_eq!(described, 69);
}

/// Metal drops a function's writes to a 4:2:0, 4:2:2 or packed two-plane YCbCr format and
/// stores an unpacked 4:4:4 one's into both planes: only the subsampled and packed formats'
/// writes are dropped.
#[test]
fn only_subsampled_and_packed_ycbcr_formats_drop_writes() {
    let mut four_four_four = 0;
    for &pf in PixelFormat::ALL {
        let Some(f) = ycbcr_format(pf).filter(|_| ycbcr(pf)) else {
            continue;
        };
        let stored = f.subsampling == ChromaSubsampling::Chroma444 && !f.packed;
        assert_eq!(writes_dropped(pf), !stored, "{pf:?}");
        four_four_four += usize::from(stored);
    }
    assert!(four_four_four >= 6, "{four_four_four}");
    assert!(writes_dropped(PixelFormat::GBGR422) && writes_dropped(PixelFormat::YCBCR84202P));
    assert!(writes_dropped(PixelFormat::YCBCR104442PPACKED));
    assert!(writes_dropped(PixelFormat::YCBCR104442PPACKEDSRGB));
    assert!(!writes_dropped(PixelFormat::YCBCR84442P) && !writes_dropped(PixelFormat::RGB104442P));
}

/// A pass renders exactly the YCbCr formats whose store a rule matches (the 8- and 10-bit
/// unpacked ones read converted or as codes) into `R32G32B32A32_SFLOAT`, blended where the device blends it; the
/// packed 10-bit formats held make their pipelines in the same floats, a pass into one refused;
/// their images stay the converted texels, and every other YCbCr format renders into nothing.
#[test]
fn a_pass_renders_only_the_ycbcr_formats_whose_store_is_matched() {
    use FormatUsage as U;
    use PixelFormat as P;
    let rendered: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| ycbcr_renders(pf))
        .collect();
    assert_eq!(
        rendered,
        [
            P::YCBCR84202P,
            P::YCBCR84222P,
            P::YCBCR84442P,
            P::YCBCR104202P,
            P::YCBCR104222P,
            P::YCBCR104442P,
            P::RGB84202P,
            P::RGB84222P,
            P::RGB84442P,
            P::RGB104202P,
            P::RGB104222P,
            P::RGB104442P
        ]
    );
    let refused: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| ycbcr_pass_refused(pf))
        .collect();
    assert_eq!(
        refused,
        [
            P::YCBCR104202PPACKED,
            P::YCBCR104222PPACKED,
            P::YCBCR104442PPACKED,
            P::YCBCR104202PPACKEDSRGB,
            P::YCBCR104222PPACKEDSRGB,
            P::YCBCR104442PPACKEDSRGB
        ]
    );
    for (stem, dev) in committed_devices() {
        for &pf in PixelFormat::ALL.iter().filter(|&&pf| ycbcr(pf)) {
            let attachment = vk_format(pf, &usages(&[U::ColorAttachment]), dev.profile());
            let blended = vk_format(
                pf,
                &usages(&[U::ColorAttachment, U::ColorAttachmentBlend]),
                dev.profile(),
            );
            let not_implemented: Result<FormatMapping, Error> =
                Err(Unsupported::not_implemented(Feature::Format).into());
            // A blend is the device's blend of the floats, where it blends them.
            let floats_blend = vk_format(
                P::RGBA32Float,
                &usages(&[U::ColorAttachment, U::ColorAttachmentBlend]),
                dev.profile(),
            )
            .is_ok();
            match (ycbcr_renders(pf), floats_blend) {
                (true, true) => assert_eq!(blended, attachment, "{stem} {pf:?}"),
                (true, false) => assert!(
                    matches!(&blended, Err(Error::Unsupported(u)) if u.feature == Feature::Format),
                    "{stem} {pf:?}: {blended:?}"
                ),
                (false, _) => assert_eq!(blended, not_implemented, "{stem} {pf:?}"),
            }
            let emulation = match (ycbcr_renders(pf), ycbcr_pass_refused(pf)) {
                (true, false) => FormatEmulation::YCbCrRendered,
                (false, true) => FormatEmulation::YCbCrPassRefused,
                (false, false) => {
                    assert_eq!(attachment, not_implemented, "{stem} {pf:?}");
                    continue;
                }
                (true, true) => panic!("{pf:?} both renders and is refused"),
            };
            let m = attachment.unwrap_or_else(|e| panic!("{stem} {pf:?}: {e:?}"));
            assert_eq!(m.format, vk::Format::R32G32B32A32_SFLOAT, "{stem} {pf:?}");
            assert_eq!(m.emulation, emulation, "{stem} {pf:?}");
            assert_eq!(color_output_components(pf), Some(3), "{pf:?}");
            assert_eq!(write_conversion(pf), Some(WriteConversion::Exact), "{pf:?}");
            let sampled = vk_format(pf, &usages(&[U::SampledImage]), dev.profile());
            assert_eq!(
                sampled.map(|m| m.emulation),
                Ok(FormatEmulation::YCbCr),
                "{stem} {pf:?}"
            );
        }
    }
}

/// A clear's halves are what its stored codes read back, rounded to `half`, only where each
/// channel's `half` is the `half` of its code's value: `0.1`'s (code 435 reads 51/510, whose
/// `half` is `half` 0.1's), not `0.25`'s (code 512 reads 0.25098) or `0.0012345`'s (code 385);
/// the two-plane formats' alpha byte reads `0.5` as `128 / 255`, a `BGRA10_XR` alpha code exactly;
/// `BGR10_XR` has no alpha; the padded and bfloat forms' tile memory is unmeasured.
#[test]
fn clears_read_back_where_their_halves_are_their_codes_values() {
    let planes = ExtendedRange::Planes { srgb: false };
    let bgra = ExtendedRange::Bgra10 { srgb: false };
    let bgr = ExtendedRange::Bgr10 { srgb: false };
    for form in [planes, bgra, bgr, ExtendedRange::Bgra10 { srgb: true }] {
        assert_eq!(form.clear_read_back([0.0; 4]), Some(true), "{form:?}");
        assert_eq!(form.clear_read_back([1.0; 4]), Some(true), "{form:?}");
    }
    assert_eq!(bgra.clear_read_back([0.1, 0.3, 1.0 / 3.0, 0.7]), Some(true));
    assert_eq!(
        planes.clear_read_back([0.1, 0.3, 1.0 / 3.0, 0.7]),
        Some(false)
    );
    assert_eq!(bgra.clear_read_back([0.25, 0.5, 0.0, 1.0]), Some(false));
    assert_eq!(
        bgra.clear_read_back([0.0012345, 0.0, 0.0, 1.0]),
        Some(false)
    );
    assert_eq!(bgra.clear_read_back([0.0, 0.0, 0.0, 0.5]), Some(true));
    assert_eq!(planes.clear_read_back([0.0, 0.0, 0.0, 0.5]), Some(false));
    assert_eq!(bgr.clear_read_back([0.0, 0.0, 0.0, 0.5]), Some(true));
    assert_eq!(ExtendedRange::R10X6.clear_read_back([0.0; 4]), None);
    assert_eq!(ExtendedRange::Rgba16Bfloat.clear_read_back([0.0; 4]), None);
}

/// A clear's tile holds each channel through `float` and then `half`, nearest even, unclamped:
/// `0.1` as `half` 0.1 (`0x2e66`), `1/3` as `0x3555`, `5.0` past the codes' range as 5, a value
/// halfway between two `float`s rounding twice; `BGR10_XR`'s alpha reads 1 whatever the clear's;
/// the padded and bfloat forms' tile memory is unmeasured.
#[test]
fn a_clears_tile_holds_its_colors_halves() {
    let h = crate::float16::f16_bits_to_f32;
    let bgra = ExtendedRange::Bgra10 { srgb: true };
    assert_eq!(
        bgra.clear_tile_value([0.1, 1.0 / 3.0, 5.0, -0.75]),
        Some([h(0x2e66), h(0x3555), 5.0, -0.75])
    );
    let planes = ExtendedRange::Planes { srgb: false };
    assert_eq!(
        planes.clear_tile_value([0.0, -0.0, 1.0, 0.5]),
        Some([0.0, -0.0, 1.0, 0.5])
    );
    let bgr = ExtendedRange::Bgr10 { srgb: false };
    assert_eq!(
        bgr.clear_tile_value([0.25, 0.5, 2.0, 0.0]),
        Some([0.25, 0.5, 2.0, 1.0])
    );
    // 1 + 2^-11 + 2^-40 is 1 + 2^-11 as a `float`, a tie, which `half` rounds to 1.
    let twice = 1.0 + 2f64.powi(-11) + 2f64.powi(-40);
    assert_eq!(
        bgr.clear_tile_value([twice, 0.0, 0.0, 1.0]).map(|v| v[0]),
        Some(1.0)
    );
    assert_eq!(ExtendedRange::R10X6.clear_tile_value([0.0; 4]), None);
    assert_eq!(ExtendedRange::Rgba16Bfloat.clear_tile_value([0.0; 4]), None);
}

/// The probe's clear colors: each 10-bit code's value and each tie between two
/// codes, 512 values of its generator, and the specials; color `i` is values `i` to `i + 3`.
fn straight_clear_colors() -> Vec<[f64; 4]> {
    let mut values = Vec::new();
    for k in 0..1024 {
        values.push((f64::from(k) - 384.0) / 510.0);
        values.push((f64::from(k) + 0.5 - 384.0) / 510.0);
    }
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..512 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
        values.push(-0.9 + 2.3 * u);
    }
    values.extend([
        f64::NEG_INFINITY,
        -1e30,
        -65504.0,
        -2.0,
        -0.0,
        0.0,
        1e-40,
        6e-8,
        1e-30,
        65504.0,
        1e30,
        f64::INFINITY,
        f64::NAN,
        -f64::NAN,
    ]);
    let n = values.len();
    (0..n)
        .map(|i| std::array::from_fn(|c| values[(i + c) % n]))
        .collect()
}

/// Metal's two clear routes store the texels the model gives (an Apple M4 under
/// macOS 27.0.1, plain and under Metal's validation layer alike: 2,574 colors into a 16 x 16
/// private target of plain usage in a pass with no draw call, `ClearRoute::Straight`, and into a
/// 4 x 4 one, `ClearRoute::ThroughHalf`), as an FNV-1a digest of each format's texels as Metal
/// stores them, a two-plane format's plane-0 word then its alpha byte. The two-plane formats
/// store the same on either route; `0.25` is code 511 on the straight route, 512 through `half`.
#[test]
fn clears_store_metal_s_texels_on_each_route() {
    let fnv = |bytes: &[u8]| {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    };
    let colors = straight_clear_colors();
    assert_eq!(colors.len(), 2574);
    let bytes = |form: ExtendedRange, route: ClearRoute| -> Vec<u8> {
        colors
            .iter()
            .flat_map(|&c| {
                let t = form.clear_texel(c, route);
                match form {
                    ExtendedRange::Bgr10 { .. } => t[0].to_le_bytes().to_vec(),
                    ExtendedRange::Planes { .. } => {
                        let mut word = t[0].to_le_bytes().to_vec();
                        word.push(t[1] as u8);
                        word
                    }
                    _ => t.iter().flat_map(|&f| (f as u16).to_le_bytes()).collect(),
                }
            })
            .collect()
    };
    // (form, Metal's straight digest, its digest through `half`)
    let metal = [
        (
            ExtendedRange::Bgr10 { srgb: false },
            0xbb6a_4158_d28b_a838u64,
            0x9619_2923_5cc4_c348u64,
        ),
        (
            ExtendedRange::Bgr10 { srgb: true },
            0xa62c_eca5_4ebf_3d07,
            0x1b43_711d_789f_660d,
        ),
        (
            ExtendedRange::Bgra10 { srgb: false },
            0x304d_3161_5f46_4eea,
            0xd417_f619_6f15_36c5,
        ),
        (
            ExtendedRange::Bgra10 { srgb: true },
            0x5456_e21b_8297_972b,
            0xfa8f_e6a4_b69b_bb94,
        ),
        (
            ExtendedRange::Planes { srgb: false },
            0x3051_a4b4_879c_69e9,
            0x3051_a4b4_879c_69e9,
        ),
        (
            ExtendedRange::Planes { srgb: true },
            0xf844_3bb4_8958_6eba,
            0xf844_3bb4_8958_6eba,
        ),
    ];
    for (form, straight, through_half) in metal {
        let got = fnv(&bytes(form, ClearRoute::Straight));
        assert_eq!(got, straight, "{form:?} straight: {got:#018x}");
        let got = fnv(&bytes(form, ClearRoute::ThroughHalf));
        assert_eq!(got, through_half, "{form:?} through half: {got:#018x}");
    }
    let bgr = ExtendedRange::Bgr10 { srgb: false };
    let codes = |t: [u32; 4]| {
        [
            t[0] >> 20 & 1023,
            t[0] >> 10 & 1023,
            t[0] & 1023,
            t[0] >> 30,
        ]
    };
    assert_eq!(
        codes(bgr.clear_texel([0.25, -0.25, f64::NAN, 0.0], ClearRoute::Straight)),
        [511, 257, 0, 0]
    );
    assert_eq!(
        codes(bgr.clear_texel([0.25, -0.25, f64::NAN, 0.0], ClearRoute::ThroughHalf)),
        [512, 256, 384, 3]
    );
}

/// The table [`repr`] reads holds [`make_repr`]'s entry of every format, at its raw value.
#[test]
fn the_format_table_holds_each_format_s_entry() {
    let summary = |repr: &Repr| match repr {
        Repr::Candidates(aspects, candidates) => format!(
            "{aspects:?} {:?}",
            candidates
                .iter()
                .map(|c| (c.format, c.emulation))
                .collect::<Vec<_>>()
        ),
        Repr::NotImplemented => "not implemented".to_owned(),
        Repr::NotAFormat => "not a format".to_owned(),
    };
    for &pf in PixelFormat::ALL {
        assert_eq!(summary(repr(pf)), summary(&make_repr(pf)), "{pf:?}");
    }
}

/// Every footprint [`block_extent`] states is the one Khronos's registry states for the Vulkan
/// block this crate maps the format to, and no uncompressed format answers one.
///
/// This is a cross-check and not a restatement: the width and height come from
/// [`block_extent`]'s own match, and the texels per block come from `Format::block`, which
/// `scripts/regen-vk-types` writes out of Khronos's `vk.xml` and nothing here may hand-edit. A
/// footprint mistyped in either place makes the two disagree. It holds for every BC, EAC, ETC2
/// and ASTC format of a footprint Vulkan has — 56 of this crate's 80 compressed formats.
///
/// The other 24 have no registry footprint to be checked against, and the test names both
/// reasons rather than skipping them quietly:
///
/// - **the 8 PVRTC formats**, because the pinned registry gives
///   `VK_FORMAT_PVRTC1_*_BLOCK_IMG` a `texelsPerBlock` of **1** and so states no footprint for
///   them at all; four of them this crate holds only through a decoded twin, which covers one
///   texel by construction;
/// - **ASTC's 4x2 and 8x4 footprints in all three profiles**, 6 formats, which no Vulkan ASTC
///   block has, so they are `NotImplemented` and must still answer a footprint — a texture's type
///   rules follow from it whether or not a device can sample the format.
///
/// For those the assertion runs the other way: each must answer a footprint, and each must fall
/// in one of the two named groups.
#[test]
fn every_block_extent_is_the_vulkan_blocks() {
    use PixelFormat as P;
    // The compressed Vulkan block among a format's candidates, where the registry states a
    // footprint for one: a decoded twin's candidate is the uncompressed image the runtime fills,
    // which covers one texel, and so does a PVRTC block in the pinned registry.
    let vulkan_block = |pf| match repr(pf) {
        Repr::Candidates(_, candidates) => candidates.iter().find_map(|candidate| {
            let block = candidate.format.block().expect("a named Vulkan format");
            (block.texels > 1).then_some((candidate.format, block.texels))
        }),
        _ => None,
    };
    // Whether any candidate of `pf` is a PVRTC block, whose registry entry states no footprint.
    let pvrtc = |pf| match repr(pf) {
        Repr::Candidates(_, candidates) => candidates.iter().any(|candidate| {
            candidate
                .format
                .block()
                .is_some_and(|block| block.class.starts_with("PVRTC"))
        }),
        _ => false,
    };
    let mut compressed = 0;
    let mut checked = 0;
    let mut unchecked = Vec::new();
    for &pf in PixelFormat::ALL {
        let Some([width, height]) = block_extent(pf) else {
            // An uncompressed format maps to no compressed block.
            assert!(vulkan_block(pf).is_none(), "{pf:?}");
            continue;
        };
        compressed += 1;
        match vulkan_block(pf) {
            Some((format, texels)) => {
                assert_eq!(
                    texels,
                    width * height,
                    "{pf:?}: {width}x{height} against {format:?}"
                );
                checked += 1;
            }
            None => unchecked.push(pf),
        }
    }
    // 14 BC, 6 EAC, 4 ETC2, 8 PVRTC, and ASTC's 16 footprints in 3 profiles.
    assert_eq!(compressed, 14 + 6 + 4 + 8 + 16 * 3, "compressed formats");
    assert_eq!(checked, 14 + 6 + 4 + 14 * 3, "checked against the registry");
    // The 8 PVRTC formats and ASTC's 4x2 and 8x4 in 3 profiles.
    assert_eq!(unchecked.len(), 8 + 6, "{unchecked:?}");
    for &pf in &unchecked {
        assert!(
            pvrtc(pf)
                || matches!(pf, astc_without_a_vulkan_block!())
                || matches!(
                    repr(pf),
                    Repr::Candidates(_, ref candidates)
                        if candidates
                            .iter()
                            .all(|c| c.emulation == FormatEmulation::Decoded)
                ),
            "{pf:?}: a compressed format unchecked for no named reason"
        );
    }
    assert_eq!(block_extent(P::RGBA8Unorm), None);
    assert_eq!(block_extent(P::ASTC4x4HDR), Some([4, 4]));
    assert_eq!(block_extent(P::ASTC12x10SRGB), Some([12, 10]));
    assert_eq!(block_extent(P::ASTC4x2LDR), Some([4, 2]));
}

/// [`footprint`] answers for a format this crate holds **no** image of, and for no other: a
/// footprint is arithmetic over a descriptor, which the image path owns wherever there is an
/// image, so the two tables can never answer for one format and drift apart.
#[test]
fn footprint_answers_where_no_image_does() {
    let sampled = usages(&[FormatUsage::SampledImage]);
    let mut answered = 0;
    for &pf in PixelFormat::ALL {
        let Some(shape) = footprint(pf) else {
            continue;
        };
        answered += 1;
        assert!(
            matches!(repr(pf), Repr::NotImplemented),
            "{pf:?}: {shape:?}, but the crate holds an image of it"
        );
        // Every candidate of the format is refused, so no device can hold its texture, whatever
        // the usages asked of it.
        for (stem, device) in committed_devices() {
            let refused = vk_format(pf, &sampled, device.profile());
            assert!(
                matches!(
                    refused,
                    Err(Error::Unsupported(Unsupported {
                        feature: Feature::Format,
                        needs: Needs::NotImplemented,
                        ..
                    }))
                ),
                "{pf:?} on {stem}: {refused:?}"
            );
        }
        if let Footprint::Tiled { bytes, block } = shape {
            assert!(bytes.is_power_of_two() && bytes <= 16, "{pf:?}: {bytes}");
            assert_eq!(
                block,
                block_extent(pf).unwrap_or([1, 1]),
                "{pf:?}: the footprint's block is the format's"
            );
        }
    }
    // The 32 formats of which Metal answers `heapTextureSizeAndAlignWithDescriptor:` and this
    // crate holds no image: the six ASTC footprints, `ARGB8Unorm_sRGB`, nine one-plane YCbCr
    // formats, ten padded 10- and 12-bit display formats, four two-plane ones and two packed
    // ones Metal makes no texture of.
    assert_eq!(answered, 6 + 1 + 9 + 10 + 4 + 2);
}

/// Every byte count [`footprint`] states is the bytes of a format the crate **does** hold whose
/// element is the same size: a padded 12-bit channel takes the 16-bit word a padded 10-bit one
/// takes, a transfer-function twin's is its base format's, a 4:2:2 texel is a luma byte and a chroma byte (`RG8Unorm`'s two), and an ASTC block
/// is 16 bytes at every footprint. So no number in the table is one nobody checked: each is held
/// to a Vulkan block size the mapping already answers.
#[test]
fn footprint_bytes_match_a_held_twin() {
    use PixelFormat as P;
    // (the format with no image, the format the crate holds whose element is the same size).
    const TWINS: &[(PixelFormat, PixelFormat)] = &[
        (P::ASTC4x2LDR, P::ASTC4x4LDR),
        (P::ASTC4x2SRGB, P::ASTC4x4SRGB),
        (P::ASTC4x2HDR, P::ASTC4x4HDR),
        (P::ASTC8x4LDR, P::ASTC4x4LDR),
        (P::ASTC8x4SRGB, P::ASTC4x4SRGB),
        (P::ASTC8x4HDR, P::ASTC4x4HDR),
        (P::ARGB8UnormSRGB, P::ARGB8Unorm),
        (P::R12UnormX4, P::R10UnormX6),
        (P::R12UnormX4PQ, P::R10UnormX6),
        (P::R12UnormX4HLG, P::R10UnormX6),
        (P::R10UnormX6PQ, P::R10UnormX6),
        (P::R10UnormX6HLG, P::R10UnormX6),
        (P::RG12UnormX8, P::RG10UnormX12),
        (P::RGB10A2UnormPQ, P::RGB10A2Unorm),
        (P::RGB10A2UnormHLG, P::RGB10A2Unorm),
        (P::BGR10A2UnormSRGB, P::BGR10A2Unorm),
        (P::BGR10A2UnormPQ, P::BGR10A2Unorm),
        (P::YCBCR84221P, P::RG8Unorm),
        (P::YCBCR84221PSRGB, P::RG8Unorm),
        (P::YCBCR84221PREV, P::RG8Unorm),
        (P::YCBCR104441P, P::RGB10A2Unorm),
        (P::YCBCR104441PSRGB, P::RGB10A2Unorm),
        (P::YCBCR104441PPQ, P::RGB10A2Unorm),
        (P::YCBCR104441PXR, P::RGB10A2Unorm),
        (P::YCBCR104441PHLG, P::RGB10A2Unorm),
        (P::YCBCRA84441P, P::BGRA8Unorm),
    ];
    // The bytes of one block of `pf` as the Vulkan format holding its bytes states them.
    let held = |pf: PixelFormat| {
        byte_format(pf)
            .and_then(|f| f.block())
            .map(|b| b.size)
            .unwrap_or_else(|| panic!("{pf:?} has a byte format"))
    };
    for &(pf, twin) in TWINS {
        let bytes = match footprint(pf) {
            Some(Footprint::Tiled { bytes, .. } | Footprint::OnePlane { bytes, .. }) => bytes,
            other => panic!("{pf:?}: {other:?} states no bytes"),
        };
        assert_eq!(bytes, held(twin), "{pf:?} against {twin:?}");
    }
    // Every format of the table whose footprint states bytes is in `TWINS`: a new one must name
    // the format the crate holds that confirms its number.
    let states_bytes: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| {
            matches!(
                footprint(pf),
                Some(Footprint::Tiled { .. } | Footprint::OnePlane { .. })
            )
        })
        .collect();
    assert_eq!(states_bytes.len(), TWINS.len(), "{states_bytes:?}");
    // The two-plane formats' planes are `RGB10A8_2P_XR10`'s, whose own answer the runtime holds;
    // the two packed ones state no bytes, because Metal counts none.
    for &pf in PixelFormat::ALL {
        match footprint(pf) {
            Some(Footprint::TwoPlanes) => assert!(
                matches!(
                    pf,
                    P::RGB10A82P | P::RGB10A82PSRGB | P::RGB10A82PPQ | P::RGB10A82PHLG
                ),
                "{pf:?}"
            ),
            Some(Footprint::NoTexture) => {
                assert!(
                    matches!(pf, P::R10UnormPACKED | P::RG10UnormPACKED),
                    "{pf:?}"
                )
            }
            _ => {}
        }
    }
}

/// Every packed 16-bit colour format the crate holds sits in the Vulkan word of its own bits,
/// which the two naming conventions pin exactly: Metal names such a format from the least
/// significant bit and Vulkan names a `_PACK16` one from the most significant, so the channel
/// letters and their widths read one way in the Metal name and the other way in the direct
/// candidate's. `BGRA4Unorm` (blue in bits 0 to 3, then green, red and alpha) therefore sits in
/// `A4R4G4B4_UNORM_PACK16` and not in `ABGR4Unorm`'s `R4G4B4A4_UNORM_PACK16`, whose channels are
/// one place over.
#[test]
fn packed_sixteen_bit_formats_sit_in_the_vulkan_word_of_their_bits() {
    /// The `(channel, bits)` pairs of a name like `BGRA4Unorm`, `A1BGR5Unorm` or
    /// `A4R4G4B4_UNORM_PACK16`: each run of channel letters followed by its width, in the order
    /// the name spells them.
    fn fields(name: &str) -> Vec<(char, u32)> {
        let mut out = Vec::new();
        let mut letters = Vec::new();
        for c in name.chars() {
            if c.is_ascii_digit() {
                let bits = u32::from(c as u8 - b'0');
                // A width digit closes every letter still open, so `B5G6R5` is three fields and
                // `A4R4G4B4` four; a letter run with one width (`RGBA8`) shares it.
                for l in letters.drain(..) {
                    out.push((l, bits));
                }
            } else if matches!(c, 'R' | 'G' | 'B' | 'A') {
                letters.push(c);
            } else {
                break;
            }
        }
        out
    }
    // The pairs the two conventions are read from, by the formats the crate holds a packed
    // 16-bit word of.
    const PACKED16: &[(PixelFormat, &str)] = &[
        (PixelFormat::B5G6R5Unorm, "B5G6R5"),
        (PixelFormat::A1BGR5Unorm, "A1BGR5"),
        (PixelFormat::ABGR4Unorm, "ABGR4"),
        (PixelFormat::BGR5A1Unorm, "BGR5A1"),
        (PixelFormat::BGRA4Unorm, "BGRA4"),
    ];
    for &(pf, metal) in PACKED16 {
        // The stem is the Metal name's own, so no table restates it.
        assert!(pf.name() == format!("MTLPixelFormat{metal}Unorm"), "{pf:?}");
        let direct = direct_format(pf).unwrap_or_else(|| panic!("{pf:?} has a direct format"));
        let name = direct
            .name()
            .unwrap_or_else(|| panic!("{direct:?} is named"));
        assert!(name.ends_with("_UNORM_PACK16"), "{pf:?}: {name}");
        let vulkan = fields(name.trim_start_matches("VK_FORMAT_"));
        let mut from_the_top = fields(metal);
        from_the_top.reverse();
        assert_eq!(vulkan, from_the_top, "{pf:?}: {name}");
    }
    // `BGRA4Unorm`'s word, spelled out: blue in bits 0 to 3, green 4 to 7, red 8 to 11 and alpha
    // 12 to 15, which is `A4R4G4B4_UNORM_PACK16` read from the most significant bit. Its
    // red-blue twin is the only other candidate, and the two are the Vulkan 1.3 pair
    // `VK_EXT_4444_formats` exposes below 1.3.
    let Repr::Candidates(aspects, candidates) = repr(PixelFormat::BGRA4Unorm) else {
        panic!("BGRA4Unorm has candidates");
    };
    assert_eq!(*aspects, vk::ImageAspectFlags::COLOR);
    let shape = |c: &Candidate| (c.format, c.components, c.emulation);
    assert_eq!(
        candidates.iter().map(shape).collect::<Vec<_>>(),
        [
            shape(&direct(vk::Format::A4R4G4B4_UNORM_PACK16)),
            shape(&red_blue_twin(vk::Format::A4B4G4R4_UNORM_PACK16)),
        ]
    );
    for c in candidates {
        let x = c.format.exposure().expect("a registry format");
        assert_eq!(x.core, Some(ApiVersion { major: 1, minor: 3 }), "{c:?}");
        assert_eq!(x.extensions, ["VK_EXT_4444_formats"], "{c:?}");
    }
    // So a device exposing the pair holds a sampled texture of it in the word itself, and one
    // without the enum values is refused the version or the extension that would expose them --
    // never `NotImplemented`, which would say no Vulkan format has its bits.
    let sampled = usages(&[FormatUsage::SampledImage]);
    let mut held = 0;
    for (stem, device) in committed_devices() {
        let profile = device.profile();
        match vk_format(PixelFormat::BGRA4Unorm, &sampled, profile) {
            Ok(m) => {
                held += 1;
                assert_eq!(m.format, vk::Format::A4R4G4B4_UNORM_PACK16, "{stem}");
                assert_eq!(m.components, vk::ComponentMapping::IDENTITY, "{stem}");
                assert_eq!(m.emulation, FormatEmulation::Direct, "{stem}");
                assert!(
                    profile.supports_format(
                        vk::Format::A4R4G4B4_UNORM_PACK16,
                        FormatUsage::SampledImage
                    ),
                    "{stem}"
                );
            }
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::Format, "{stem}");
                assert!(
                    !matches!(u.needs, Needs::NotImplemented),
                    "{stem}: {u:?} -- the bits have a Vulkan word"
                );
            }
            Err(e) => panic!("{stem}: {e:?}"),
        }
    }
    // The MoltenVK profile exposes the pair, so the sweep is not vacuous in either direction.
    assert!(held >= 1, "held on {held} committed profiles");
    assert!(
        vk_format(
            PixelFormat::BGRA4Unorm,
            &sampled,
            committed_devices()
                .iter()
                .find(|(s, _)| s == "vk12-floor")
                .expect("vk12-floor is committed")
                .1
                .profile(),
        )
        .is_err(),
        "the 1.2 floor exposes no 4-bit word of Metal's channel order"
    );
}

/// `ARGB8Unorm_sRGB`, the other plain straggler, stays refused `NotImplemented(Format)` and keeps
/// its own heap footprint: its bytes are `ARGB8Unorm`'s, which the crate holds, but no Vulkan
/// format applies the sRGB transfer function to the three channels Metal's does -- every Vulkan
/// `_SRGB` format encodes the word's low three and leaves the fourth linear, and Metal's alpha is
/// the low byte here. Holding it in `R8G8B8A8_SRGB` through `ARGB8Unorm`'s rotation would read
/// Metal's alpha decoded and Metal's blue linear, because format conversion runs before a view's
/// component swizzle.
#[test]
fn the_srgb_straggler_has_no_vulkan_image_and_keeps_its_footprint() {
    let pf = PixelFormat::ARGB8UnormSRGB;
    assert!(matches!(repr(pf), Repr::NotImplemented));
    assert_eq!(aspects(pf), None);
    assert_eq!(direct_format(pf), None);
    assert_eq!(
        footprint(pf),
        Some(super::footprint::Footprint::Tiled {
            bytes: 4,
            block: [1, 1],
        })
    );
    // Its sibling is held, and the two no longer answer the same way anywhere.
    assert_eq!(footprint(PixelFormat::BGRA4Unorm), None);
    // Every usage is refused `NotImplemented` on every committed profile: no candidate to pick.
    for (stem, device) in committed_devices() {
        for u in [
            FormatUsage::SampledImage,
            FormatUsage::StorageImage,
            FormatUsage::ColorAttachment,
        ] {
            let refused = vk_format(pf, &usages(&[u]), device.profile());
            assert!(
                matches!(
                    refused,
                    Err(Error::Unsupported(Unsupported {
                        feature: Feature::Format,
                        needs: Needs::NotImplemented,
                        ..
                    }))
                ),
                "{stem} {u:?}: {refused:?}"
            );
        }
    }
    // The bytes it would be held in: `ARGB8Unorm`'s image, whose three sRGB channels would be the
    // word's low bytes -- Metal's alpha, red and green -- and not Metal's red, green and blue.
    let argb = direct_format(PixelFormat::ARGB8Unorm).expect("ARGB8Unorm is held");
    assert_eq!(argb, vk::Format::R8G8B8A8_UNORM);
    assert_eq!(
        vk::Format::R8G8B8A8_SRGB.block().map(|b| b.class),
        argb.block().map(|b| b.class)
    );
    assert!(!crate::metal::texel_format::srgb_coded(pf));
}

/// `BGRA4Unorm`'s colour-attachment conversion, measured on an Apple M3 Max under macOS 27.0.1
/// (`metal2vulkan-conformance/probes/bgra4_write_and_mipmap.m`), is the packed 16-bit rule its four
/// siblings already hold: each channel's value rounded to `half` to nearest even, clamped, times
/// 15, rounded to nearest even. The probe swept 31,864 values a channel from a `float` and a
/// `half` output, plain and blended with factors one and zero, and ran `ABGR4Unorm` and
/// `B5G6R5Unorm` beside it in the same process, reproducing their recorded rules and their recorded
/// field positions.
///
/// Rounding the exact product *without* the `half` -- which is what every unpacked unorm format
/// does -- explains none of the `float` legs, and the first value where it parts company is the
/// one asserted here. That is the measurement, not a restatement of the rule: a red of
/// `0.0333404541` is half 0.033325195, whose product with 15 is 0.49988, which stores 0, while the
/// exact product 0.50011 would store 1.
#[test]
fn the_packed_four_bit_straggler_converts_writes_through_a_half() {
    use super::word::half_unorm_code;

    let rule = Some(ChannelWrite::HalfUnorm { bits: 4 });
    assert_eq!(
        write_conversion(PixelFormat::BGRA4Unorm),
        Some(WriteConversion::Rules([rule; 4])),
    );
    // Its sibling's rule, which the same run reproduced.
    assert_eq!(
        write_conversion(PixelFormat::ABGR4Unorm),
        Some(WriteConversion::Rules([rule; 4])),
    );

    // The value Metal's own answer used to reject the product rule, and the rounding either side.
    let parts = f32::from_bits(0x3d08_9000);
    assert_eq!(half_unorm_code(parts, 4), 0);
    assert_eq!((f64::from(parts) * 15.0).round_ties_even(), 1.0);

    // The probe's `graded` texel: (1, 2, 4, 8) / 15 stored 0x8124, alpha in bits 12 to 15, then
    // red, green and blue, which is this format's word read from its least significant bit.
    let graded = [1.0 / 15.0, 2.0 / 15.0, 4.0 / 15.0, 8.0 / 15.0];
    let codes = graded.map(|x| half_unorm_code(x, 4));
    assert_eq!(codes, [1, 2, 4, 8]);
    let word = (codes[3] << 12) | (codes[0] << 8) | (codes[1] << 4) | codes[2];
    assert_eq!(word, 0x8124);
    // And its single-channel texels: a channel of one fills its own nibble and no other.
    assert_eq!(half_unorm_code(1.0, 4), 15);
    // Outside `[0, 1]` and the non-finite values, over the whole sweep.
    assert_eq!(half_unorm_code(f32::NAN, 4), 0);
    assert_eq!(half_unorm_code(-1.0, 4), 0);
    assert_eq!(half_unorm_code(f32::INFINITY, 4), 15);
    assert_eq!(half_unorm_code(f32::NEG_INFINITY, 4), 0);
}
