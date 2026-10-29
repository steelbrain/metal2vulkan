//! The storage form's choice: what blocks it, with which reason, and how it declares an attachment.

use super::*;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::lower::arith::extended_range::blend::BlendMath;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::ColorAttachmentDescriptor;

/// Committed profile `stem` with `features` removed from its description.
fn profile(stem: &str, features: &[VkFeature]) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in features {
        desc.features.remove(f);
    }
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// Committed profile `stem` with `features` removed from its description, read at Vulkan 1.2 (the
/// runtime's instance) with `VK_KHR_format_feature_flags2` enabled where `flags2`
/// (`pipeline::testing::description_at_1_2`; without it, a device that lists the extension and a
/// consumer that did not enable it).
fn profile_with(stem: &str, features: &[VkFeature], flags2: bool) -> DeviceProfile {
    let mut desc = crate::pipeline::testing::description_at_1_2(stem, flags2).expect("description");
    for f in features {
        desc.features.remove(f);
    }
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The state of a pipeline of one color attachment of `format` at index 0, changed by `edit`.
fn state(
    format: PixelFormat,
    edit: impl FnOnce(&mut ColorAttachmentDescriptor),
) -> FramebufferFetchInputs {
    let mut c = ColorAttachmentDescriptor::new(format);
    edit(&mut c);
    FramebufferFetchInputs::of([0], &[Some(c)], false, false, 1)
}

const WRITES: ColorUse = ColorUse {
    writes: true,
    early_tests: true,
    draw_tested: false,
    math: BlendMath::Fast,
};
const READS: ColorUse = ColorUse {
    writes: false,
    ..WRITES
};

fn form(
    inputs: &FramebufferFetchInputs,
    multisampled: bool,
    use_: ColorUse,
    p: &DeviceProfile,
) -> Result<StorageView, Unsupported> {
    storage_form(0, inputs, multisampled, use_, p).expect("no descriptor error")
}

fn declared(
    r: Result<StorageView, Unsupported>,
) -> (vk::Format, crate::spv::ImageFormat, SampledType) {
    let v = r.expect("the storage form");
    (v.format, v.declared, v.numeric)
}

/// The image format a view of `viewed` for `use_` declares on `p`: none where the device grants
/// that access without a format, else `typed`.
fn expected_declaration(
    viewed: vk::Format,
    use_: ColorUse,
    typed: crate::spv::ImageFormat,
    p: &DeviceProfile,
) -> crate::spv::ImageFormat {
    match accessed_without_format(viewed, StorageUse::texture(true, use_.writes), p) {
        true => crate::spv::ImageFormat::Unknown,
        false => typed,
    }
}

/// A clean single-sample state takes the storage form on every device that orders fetch only
/// through the interlock, declared without a format where the device grants the format's storage
/// access without one, else with the format's SPIR-V image format.
#[test]
fn a_clean_state_takes_the_storage_form() {
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "intel-irisxe-anv",
        "nvidia-rtx50-linux",
    ] {
        let p = profile(stem, &[]);
        assert!(interlock_orders(&p), "{stem}");
        for (pf, want) in [
            (
                PixelFormat::R32Float,
                (
                    vk::Format::R32_SFLOAT,
                    crate::spv::ImageFormat::R32f,
                    SampledType::Float,
                ),
            ),
            (
                PixelFormat::RGBA8Unorm,
                (
                    vk::Format::R8G8B8A8_UNORM,
                    crate::spv::ImageFormat::Rgba8,
                    SampledType::Float,
                ),
            ),
            (
                PixelFormat::RGBA32Uint,
                (
                    vk::Format::R32G32B32A32_UINT,
                    crate::spv::ImageFormat::Rgba32ui,
                    SampledType::Uint,
                ),
            ),
            (
                PixelFormat::RG16Float,
                (
                    vk::Format::R16G16_SFLOAT,
                    crate::spv::ImageFormat::Rg16f,
                    SampledType::Float,
                ),
            ),
        ] {
            for use_ in [WRITES, READS] {
                let mut want = want;
                want.1 = expected_declaration(want.0, use_, want.1, &p);
                assert_eq!(
                    declared(form(&state(pf, |_| {}), false, use_, &p)),
                    want,
                    "{stem} {pf:?} {use_:?}"
                );
            }
        }
    }
    for stem in ["lavapipe", "vk12-floor", "swiftshader"] {
        assert!(!interlock_orders(&profile(stem, &[])), "{stem}");
    }
}

/// What the form does not model is `NotImplemented`: an amplified pipeline, and, for a written
/// attachment only, a signed normalized format under a write mask
/// that keeps some of its channels, alpha to coverage, a returned sample mask at one sample, and a
/// depth or stencil attachment without early fragment tests, but where each draw's plan checks its
/// tests. Another format's blend is the generic one ([`StorageView::generic_blend`]): a 32-bit
/// float attachment's under a blend color, which Metal's own measured arithmetic there does not
/// fit.
#[test]
fn unmodelled_states_are_not_implemented() {
    let p = profile("moltenvk-m3", &[]);
    let ni = Err(Unsupported::not_implemented(Feature::FramebufferFetch));
    let r32 = state(PixelFormat::R32Float, |_| {});
    let mut amplified = r32.clone();
    amplified.amplified = true;
    assert_eq!(form(&amplified, false, READS, &p), ni);
    // A blend color's lerp, which is no fitted form of a 32-bit float attachment at any channel
    // count: the generic blend.
    let generic = state(PixelFormat::R32Float, |c| {
        c.blending_enabled = true;
        c.source_rgb_blend_factor = crate::metal::BlendFactor::BlendColor;
        c.destination_rgb_blend_factor = crate::metal::BlendFactor::OneMinusBlendColor;
    });
    let view = form(&generic, false, WRITES, &p).expect("the generic blend");
    assert!(view.generic_blend && view.blend.is_some(), "{view:?}");
    // Only the red channel of a one-channel format needs writing.
    let red = state(PixelFormat::R32Float, |c| {
        c.write_mask = ColorWriteMask::RED
    });
    let rg = state(PixelFormat::RGBA8Snorm, |c| {
        c.write_mask = ColorWriteMask::RED.union(ColorWriteMask::GREEN)
    });
    let mut coverage = r32.clone();
    coverage.alpha_to_coverage = true;
    let mut depth = r32.clone();
    depth.depth_or_stencil = true;
    let late = ColorUse {
        early_tests: false,
        ..WRITES
    };
    for (what, inputs, use_) in [
        ("signed normalized write mask", &rg, WRITES),
        ("alpha to coverage", &coverage, WRITES),
        ("depth without early tests", &depth, late),
    ] {
        assert_eq!(form(inputs, false, use_, &p), ni, "{what}");
        let read = ColorUse {
            writes: false,
            ..use_
        };
        assert!(form(inputs, false, read, &p).is_ok(), "{what}, read only");
    }
    assert!(form(&red, false, WRITES, &p).is_ok());
    assert!(form(&depth, false, WRITES, &p).is_ok());
    // A write before the tests whose draws' plans check them.
    let tested = ColorUse {
        draw_tested: true,
        ..late
    };
    assert!(form(&depth, false, tested, &p).is_ok());
}

/// An 8-bit four-channel sRGB attachment takes the storage form through a view in `R32_UINT`,
/// declared `R32ui` (without a format where the device grants that) with `uint` texels, which the
/// function reads and writes as floats it decodes and encodes itself; a blend into one is Metal's,
/// which the function computes ([`extended_range::Blend::srgb8`]), its second source's too, and a
/// replacing one stores the source as an unblended write does.
#[test]
fn a_four_channel_srgb_attachment_takes_a_uint_view() {
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "nvidia-rtx50-linux",
    ] {
        let p = profile(stem, &[]);
        for (pf, format) in [
            (PixelFormat::RGBA8UnormSRGB, vk::Format::R8G8B8A8_SRGB),
            (PixelFormat::BGRA8UnormSRGB, vk::Format::B8G8R8A8_SRGB),
        ] {
            for use_ in [WRITES, READS] {
                let view = form(&state(pf, |_| {}), false, use_, &p).expect("the storage form");
                assert_eq!(
                    (view.format, view.declared, view.numeric, view.uint_view),
                    (
                        format,
                        expected_declaration(
                            vk::Format::R32_UINT,
                            use_,
                            crate::spv::ImageFormat::R32ui,
                            &p
                        ),
                        SampledType::Uint,
                        Some(vk::Format::R32_UINT)
                    ),
                    "{stem} {pf:?} {use_:?}"
                );
                assert_eq!(view.read_class(), SampledType::Float, "{stem} {pf:?}");
            }
            // A mask that keeps channels merges their bytes.
            let masked = state(pf, |c| c.write_mask = ColorWriteMask::RED);
            assert!(form(&masked, false, WRITES, &p).is_ok(), "{stem} {pf:?}");
            for (destination, blends) in [
                (crate::metal::BlendFactor::One, true),
                (crate::metal::BlendFactor::OneMinusSource1Color, true),
                (crate::metal::BlendFactor::Zero, false),
            ] {
                let blended = state(pf, |c| {
                    c.blending_enabled = true;
                    c.destination_rgb_blend_factor = destination;
                    c.destination_alpha_blend_factor = destination;
                });
                let view = form(&blended, false, WRITES, &p).expect("the sRGB blend");
                assert_eq!(
                    view.blend.is_some(),
                    blends,
                    "{stem} {pf:?} {destination:?}"
                );
                let read = form(&blended, false, READS, &p).expect("a read");
                assert_eq!(read.blend, None, "{stem} {pf:?} {destination:?}");
            }
        }
    }
}

/// A one- or two-channel 8-bit sRGB attachment (`R8Unorm_sRGB`, `RG8Unorm_sRGB`) on a device that
/// renders its format (MoltenVK, RADV) takes the storage form as a four-channel one does, through
/// a view in the unsigned integer format of its texel (`R8_UINT`, `R16_UINT`), read and written,
/// under a blend the function computes ([`extended_range::Blend::srgb8`]), and the two-channel
/// one's under a mask that keeps one channel.
#[test]
fn a_one_or_two_channel_srgb_attachment_takes_a_uint_view() {
    use crate::spv::ImageFormat as I;
    for stem in ["moltenvk-m3", "amd-rdna3-radv"] {
        let p = profile(stem, &[]);
        for (pf, format, uint, declared) in [
            (
                PixelFormat::R8UnormSRGB,
                vk::Format::R8_SRGB,
                vk::Format::R8_UINT,
                I::R8ui,
            ),
            (
                PixelFormat::RG8UnormSRGB,
                vk::Format::R8G8_SRGB,
                vk::Format::R16_UINT,
                I::R16ui,
            ),
        ] {
            for use_ in [WRITES, READS] {
                let view = form(&state(pf, |_| {}), false, use_, &p).expect("the storage form");
                assert_eq!(
                    (view.format, view.declared, view.numeric, view.uint_view),
                    (
                        format,
                        expected_declaration(uint, use_, declared, &p),
                        SampledType::Uint,
                        Some(uint)
                    ),
                    "{stem} {pf:?} {use_:?}"
                );
                assert_eq!(view.read_class(), SampledType::Float, "{stem} {pf:?}");
            }
            let blended = state(pf, |c| {
                c.blending_enabled = true;
                c.destination_rgb_blend_factor = crate::metal::BlendFactor::OneMinusSourceAlpha;
                c.destination_alpha_blend_factor = crate::metal::BlendFactor::OneMinusSourceAlpha;
            });
            let view = form(&blended, false, WRITES, &p).expect("the sRGB blend");
            assert!(view.blend.is_some(), "{stem} {pf:?}");
        }
        let masked = state(PixelFormat::RG8UnormSRGB, |c| {
            c.write_mask = ColorWriteMask::GREEN;
        });
        let view = form(&masked, false, WRITES, &p).expect("a masked write");
        assert_eq!(view.kept, vk::ColorComponentFlags::R, "{stem}");
    }
}

/// A BGRA-order attachment (`BGRA8Unorm`, `BGR10A2Unorm`), whose format SPIR-V names no image
/// format for, takes the storage form on a device that accesses its format without one in neither
/// direction (the ANV profiles: no `shaderStorageImageReadWithoutFormat`) through a view in its
/// RGBA-order twin, declared as the twin is (by its SPIR-V image format, or without one where the
/// device grants the twin that), with `float` texels: of one sample
/// and as sample planes, read only, written, and under a write mask that keeps some channels. So
/// does a device whose description states no per-format bits (no `VkFormatProperties3`:
/// `intel-arl-anv`), which reads without a format only the formats Vulkan lists for it, no
/// BGRA-order one among them. On a device that reads and writes the format without one it keeps
/// its own format, declared without one, as before.
#[test]
fn a_bgra_order_attachment_takes_its_rgba_twin_where_no_declaration_reaches_it() {
    use crate::spv::ImageFormat as I;
    let formats = [
        (
            PixelFormat::BGRA8Unorm,
            vk::Format::B8G8R8A8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
            I::Rgba8,
        ),
        (
            PixelFormat::BGR10A2Unorm,
            vk::Format::A2R10G10B10_UNORM_PACK32,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            I::Rgb10A2,
        ),
    ];
    let partial = ColorWriteMask::RED.union(ColorWriteMask::BLUE);
    for stem in [
        "intel-irisxe-anv",
        "intel-irisxe-rplp-anv",
        "intel-uhd770-rpls-anv",
        "intel-arl-anv",
    ] {
        let p = profile(stem, &[]);
        assert!(interlock_orders(&p), "{stem}");
        for (pf, format, twin, typed) in formats {
            assert_eq!(
                StorageAttachmentUse::view_alias_of(format, &p),
                Some(twin),
                "{stem} {pf:?}"
            );
            for multisampled in [false, true] {
                for (use_, mask) in [
                    (READS, ColorWriteMask::ALL),
                    (WRITES, ColorWriteMask::ALL),
                    (WRITES, partial),
                ] {
                    let what = format!("{stem} {pf:?} {multisampled} {use_:?} {mask:?}");
                    let view = form(&state(pf, |c| c.write_mask = mask), multisampled, use_, &p)
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    assert_eq!(
                        (
                            view.format,
                            view.rgba_view,
                            view.uint_view,
                            view.declared,
                            view.numeric
                        ),
                        (
                            format,
                            Some(twin),
                            None,
                            expected_declaration(twin, use_, typed, &p),
                            SampledType::Float
                        ),
                        "{what}"
                    );
                    let samples = match multisampled {
                        true => StorageSamples::Planes,
                        false => StorageSamples::One,
                    };
                    assert_eq!(view.samples, samples, "{what}");
                    assert_eq!(view.merges(), mask == partial, "{what}");
                }
            }
        }
    }
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "amd-780m-radv",
        "nvidia-rtx50-linux",
        "nvidia-rtx5080-laptop",
    ] {
        let p = profile(stem, &[]);
        let (pf, format, ..) = formats[0];
        assert_eq!(
            StorageAttachmentUse::view_alias_of(format, &p),
            None,
            "{stem}"
        );
        for use_ in [READS, WRITES] {
            let view = form(&state(pf, |_| {}), false, use_, &p).expect("the storage form");
            assert_eq!(
                (view.rgba_view, view.declared),
                (None, I::Unknown),
                "{stem}"
            );
        }
    }
}

/// The view a fetched attachment takes (`StorageAttachmentUse::view_alias_of`, the one decision of
/// the compile layer and the runtime) on every committed profile: an 8-bit sRGB format's
/// unsigned-integer view whatever the device; any other format the device reads and writes without
/// a format keeps its own view (as every such device had before); and a BGRA-order format it does
/// not takes its RGBA-order twin only where the device stores the twin and declares it. Some
/// committed profile takes a twin.
#[test]
fn a_view_alias_is_a_twin_only_where_no_declaration_reaches_the_format() {
    let rw = StorageUse::READ_WRITE_TEXTURE;
    let mut twins = 0;
    for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
        for &pf in PixelFormat::ALL {
            let Some(format) = crate::metal::format::direct_format(pf) else {
                continue;
            };
            let srgb = match format {
                vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB => Some(vk::Format::R32_UINT),
                vk::Format::R8G8_SRGB => Some(vk::Format::R16_UINT),
                vk::Format::R8_SRGB => Some(vk::Format::R8_UINT),
                _ => None,
            };
            let alias = StorageAttachmentUse::view_alias_of(format, &p);
            let what = format!("{stem} {pf:?}");
            if srgb.is_some() || accessed_without_format(format, rw, &p) {
                assert_eq!(alias, srgb, "{what}");
                continue;
            }
            match alias {
                Some(twin) => {
                    assert_eq!(rgba_twin(format), Some(twin), "{what}");
                    assert!(p.supports_format(twin, FormatUsage::StorageImage), "{what}");
                    assert!(
                        accessed_without_format(twin, rw, &p) || spirv_image_format(twin).is_some(),
                        "{what}"
                    );
                    twins += 1;
                }
                None => assert!(
                    rgba_twin(format).is_none_or(|t| !rgba_twin_gaps(t, &p).is_empty()),
                    "{what}"
                ),
            }
        }
    }
    assert!(
        twins > 0,
        "no committed profile views an attachment in its twin"
    );
}

/// A BGRA-order attachment whose RGBA-order twin lacks something too is refused naming both
/// routes: `BGR10A2Unorm` on the ANV profile without `shaderStorageImageExtendedFormats` (its
/// twin's `Rgb10A2` is an extended format) needs reads without a format of its own format, or the
/// extended formats for the twin. With `VK_KHR_format_feature_flags2` the first is the format's own
/// `STORAGE_READ_WITHOUT_FORMAT` bit, which `A2R10G10B10_UNORM_PACK32` lacks there; without it, the
/// feature. `BGRA8Unorm`'s twin, a base format, still serves.
#[test]
fn a_bgra_order_attachment_without_either_route_names_both() {
    use crate::error::Needs;
    let format = vk::Format::A2R10G10B10_UNORM_PACK32;
    let own_bit = Capability::Format {
        format,
        usage: FormatUsage::StorageImageReadWithoutFormat,
    };
    let feature = Capability::Feature(VkFeature::ShaderStorageImageReadWithoutFormat);
    for (flags2, reads) in [(true, own_bit), (false, feature)] {
        let p = profile_with(
            "intel-irisxe-anv",
            &[VkFeature::ShaderStorageImageExtendedFormats],
            flags2,
        );
        assert_eq!(StorageAttachmentUse::view_alias_of(format, &p), None);
        let refused = form(&state(PixelFormat::BGR10A2Unorm, |_| {}), false, WRITES, &p)
            .expect_err("no route");
        assert_eq!(
            refused.needs,
            Needs::AnyOf(vec![
                vec![reads],
                vec![Capability::Feature(
                    VkFeature::ShaderStorageImageExtendedFormats
                )],
            ]),
            "VK_KHR_format_feature_flags2 enabled: {flags2}"
        );
        let bgra8 = form(&state(PixelFormat::BGRA8Unorm, |_| {}), false, WRITES, &p);
        assert_eq!(
            bgra8.map(|v| v.rgba_view),
            Ok(Some(vk::Format::R8G8B8A8_UNORM))
        );
    }
}

/// On a device without `shaderStorageImageReadWithoutFormat` (the ANV profile), every color format
/// whose storage form is modelled takes it but those whose direct format has neither a SPIR-V
/// image format, nor an RGBA-order twin, nor its own `STORAGE_READ_WITHOUT_FORMAT` bit where
/// `VK_KHR_format_feature_flags2` is enabled, refused naming the reads without a format they would
/// need (the bit, or without the extension the feature): the 16-bit packed formats, whose exact
/// form (a view of their 16-bit words, converted by Metal's measured conversions in the shader) is
/// not built.
#[test]
fn without_formatless_reads_only_formats_no_declaration_reaches_are_refused() {
    for flags2 in [true, false] {
        formats_no_declaration_reaches_are_refused(flags2);
    }
}

fn formats_no_declaration_reaches_are_refused(flags2: bool) {
    use crate::error::Needs;
    let p = profile_with("intel-irisxe-anv", &[], flags2);
    let mut refused = Vec::new();
    for &pf in PixelFormat::ALL {
        let inputs = FramebufferFetchInputs::of(
            [0],
            &[Some(ColorAttachmentDescriptor::new(pf))],
            false,
            false,
            1,
        );
        for use_ in [READS, WRITES] {
            let Ok(Err(u)) = storage_form(0, &inputs, false, use_, &p) else {
                continue;
            };
            if matches!(u.needs, Needs::NotImplemented | Needs::NotRepresentable) {
                continue;
            }
            let reads = match flags2 {
                true => Capability::Format {
                    format: crate::metal::format::direct_format(pf).expect("a direct format"),
                    usage: FormatUsage::StorageImageReadWithoutFormat,
                },
                false => Capability::Feature(VkFeature::ShaderStorageImageReadWithoutFormat),
            };
            let needs_reads = |alt: &Vec<Capability>| alt.contains(&reads);
            let Needs::AnyOf(alternatives) = &u.needs else {
                panic!("{pf:?} {use_:?}: {u:?}");
            };
            assert!(
                alternatives.iter().all(needs_reads),
                "{pf:?} {use_:?}: {u:?}"
            );
            if !refused.contains(&pf) {
                refused.push(pf);
            }
        }
    }
    assert_eq!(
        refused,
        [
            PixelFormat::B5G6R5Unorm,
            PixelFormat::A1BGR5Unorm,
            PixelFormat::ABGR4Unorm,
            PixelFormat::BGR5A1Unorm,
        ],
        "VK_KHR_format_feature_flags2 enabled: {flags2}: {refused:?}"
    );
}

/// A native `A8_UNORM` attachment, which no SPIR-V image format names and whose compatibility class
/// is its own (no other view reaches it), takes the storage form on the ANV profile, which has no
/// `shaderStorageImageReadWithoutFormat`, through a declaration without a format where
/// `VK_KHR_format_feature_flags2` is enabled, as `A8_UNORM`'s own `STORAGE_READ_WITHOUT_FORMAT` bit
/// grants (Mesa 26.2.4 on the Iris Xe reports it); without the extension it is refused naming the
/// feature. A format that lacks the bit there (`B5G6R5Unorm`'s `R5G6B5_UNORM_PACK16`) stays refused,
/// naming the bit.
#[test]
fn an_a8_unorm_attachment_is_fetched_without_a_format_by_its_own_bit() {
    use crate::error::Needs;
    let a8 = state(PixelFormat::A8Unorm, |_| {});
    for use_ in [READS, WRITES] {
        let p = profile_with("intel-irisxe-rplp-anv", &[], true);
        assert!(!p.images().storage_read_without_format);
        assert_eq!(
            declared(form(&a8, false, use_, &p)),
            (
                vk::Format::A8_UNORM,
                crate::spv::ImageFormat::Unknown,
                SampledType::Float
            ),
            "{use_:?}"
        );
        let packed =
            form(&state(PixelFormat::B5G6R5Unorm, |_| {}), false, use_, &p).expect_err("no bit");
        assert_eq!(
            packed.needs,
            Needs::AnyOf(vec![vec![Capability::Format {
                format: vk::Format::R5G6B5_UNORM_PACK16,
                usage: FormatUsage::StorageImageReadWithoutFormat,
            }]]),
            "{use_:?}"
        );
        let without = profile_with("intel-irisxe-rplp-anv", &[], false);
        assert_eq!(
            form(&a8, false, use_, &without)
                .expect_err("no route")
                .needs,
            Needs::AnyOf(vec![vec![Capability::Feature(
                VkFeature::ShaderStorageImageReadWithoutFormat
            )]]),
            "{use_:?}"
        );
    }
}

/// A blend of one and zero, added, for color and alpha stores the source as no blend does, so a
/// written attachment under it takes the storage form unblended, but for a signed normalized format
/// (its NaN stores another code under the blend), whose blend the shader computes, the generic
/// one; one and zero for color alone is a blend, and a 32-bit float attachment's blend is the
/// generic one under every state and math: the shader computes `S*Fs op D*Fd` in `float`, as the
/// device's blend unit would. A write whose
/// mask covers none of the format's channels stores nothing: no blend, alpha to coverage, one
/// sample's returned mask, or late depth test bears on it, but an `Unspecialized` mask stays
/// refused.
#[test]
fn a_replacing_blend_and_a_write_that_stores_nothing_take_the_storage_form() {
    use crate::metal::{BlendFactor, BlendOperation};
    let p = profile("moltenvk-m3", &[]);
    let ni = Err(Unsupported::not_implemented(Feature::FramebufferFetch));
    let replacing = |pf: PixelFormat| state(pf, |c| c.blending_enabled = true);
    for pf in [
        PixelFormat::RG16Float,
        PixelFormat::R32Float,
        PixelFormat::RGBA8Unorm,
    ] {
        assert!(form(&replacing(pf), false, WRITES, &p).is_ok(), "{pf:?}");
    }
    let generic = |inputs: &FramebufferFetchInputs| {
        let view = form(inputs, false, WRITES, &p).expect("the generic blend");
        view.generic_blend && view.blend.is_some()
    };
    assert!(generic(&replacing(PixelFormat::RG8Snorm)));
    assert!(!generic(&replacing(PixelFormat::R32Float)));
    // A 32-bit float attachment's blend is the generic one, whatever its state or math.
    let alpha_adds = state(PixelFormat::R32Float, |c| {
        c.blending_enabled = true;
        c.destination_alpha_blend_factor = BlendFactor::One;
    });
    let subtracts = state(PixelFormat::R32Float, |c| {
        c.blending_enabled = true;
        c.rgb_blend_operation = BlendOperation::Subtract;
    });
    let source_by_destination = |pf| {
        state(pf, |c| {
            c.blending_enabled = true;
            c.source_rgb_blend_factor = BlendFactor::SourceColor;
            c.destination_rgb_blend_factor = BlendFactor::DestinationColor;
            c.source_alpha_blend_factor = BlendFactor::SourceColor;
            c.destination_alpha_blend_factor = BlendFactor::DestinationColor;
        })
    };
    for blended in [
        &alpha_adds,
        &subtracts,
        &source_by_destination(PixelFormat::R32Float),
        &source_by_destination(PixelFormat::RG32Float),
    ] {
        assert!(generic(blended));
    }
    let safe = ColorUse {
        math: BlendMath::Rounded,
        ..WRITES
    };
    let rounded = form(&alpha_adds, false, safe, &p).expect("the storage form");
    assert!(rounded.generic_blend && rounded.blend.is_some());
    // A write that stores nothing, under each state that refuses a store.
    let none = |edit: fn(&mut ColorAttachmentDescriptor)| {
        state(PixelFormat::RG16Float, |c| {
            c.write_mask = ColorWriteMask::NONE;
            edit(c);
        })
    };
    let blended = none(|c| {
        c.blending_enabled = true;
        c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    });
    let mut coverage = none(|_| {});
    coverage.alpha_to_coverage = true;
    let mut depth = none(|_| {});
    depth.depth_or_stencil = true;
    let late = ColorUse {
        early_tests: false,
        ..WRITES
    };
    for (what, inputs, use_) in [
        ("blending", &blended, WRITES),
        ("alpha to coverage", &coverage, WRITES),
        ("depth without early tests", &depth, late),
    ] {
        let view = form(inputs, false, use_, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert!(view.stores_nothing(), "{what}");
    }
    let unspecialized = state(PixelFormat::RG16Float, |c| {
        c.write_mask = ColorWriteMask::UNSPECIALIZED
    });
    assert_eq!(form(&unspecialized, false, WRITES, &p), ni);
    assert!(form(&unspecialized, false, READS, &p).is_ok());
}

/// A blend into a 16-bit float attachment that the write stores takes the storage form with the
/// blend the shader computes (`StorageView::blend`), under either blend math, on every device that
/// orders fetch through the pixel interlock alone: for each 16-bit float format, the macOS 27
/// window server's one over one minus the source alpha among the factors and operations Metal's
/// model holds, the blend color's difference with its one minus (under fast math Metal's form, the
/// minuend's product fused and the subtrahend's rounded), and a write mask that keeps some
/// channels. A factor of the second source takes the generic blend
/// (`StorageView::generic_blend`). A read-only attachment, a replacing blend under fast math and a write that
/// stores nothing take the form without one; a replacing blend without fast math keeps the
/// destination's product by zero, so the shader computes it.
#[test]
fn a_sixteen_bit_float_blend_is_computed_in_the_storage_form() {
    use crate::metal::{BlendFactor as F, BlendOperation as O};
    let descriptor = |pf: PixelFormat, (s, d, op): (F, F, O), mask: ColorWriteMask| {
        let mut c = ColorAttachmentDescriptor::new(pf);
        c.blending_enabled = true;
        c.source_rgb_blend_factor = s;
        c.destination_rgb_blend_factor = d;
        c.rgb_blend_operation = op;
        c.source_alpha_blend_factor = s;
        c.destination_alpha_blend_factor = d;
        c.alpha_blend_operation = op;
        c.write_mask = mask;
        c
    };
    let of =
        |c: ColorAttachmentDescriptor| FramebufferFetchInputs::of([0], &[Some(c)], false, false, 1);
    let equations = [
        (F::One, F::OneMinusSourceAlpha, O::Add),
        (F::SourceAlpha, F::OneMinusSourceAlpha, O::Subtract),
        (
            F::DestinationColor,
            F::OneMinusDestinationAlpha,
            O::ReverseSubtract,
        ),
        (F::BlendColor, F::OneMinusBlendAlpha, O::Add),
        (F::SourceAlphaSaturated, F::One, O::Add),
        (F::One, F::One, O::Max),
    ];
    let partial = ColorWriteMask::RED.union(ColorWriteMask::ALPHA);
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "nvidia-rtx50-linux",
    ] {
        let p = profile(stem, &[]);
        assert!(interlock_orders(&p), "{stem}");
        for pf in [
            PixelFormat::R16Float,
            PixelFormat::RG16Float,
            PixelFormat::RGBA16Float,
        ] {
            for math in [BlendMath::Fast, BlendMath::Rounded] {
                let writes = ColorUse { math, ..WRITES };
                let reads = ColorUse { math, ..READS };
                for eq in equations {
                    for mask in [ColorWriteMask::ALL, partial] {
                        let what = format!("{stem} {pf:?} {math:?} {eq:?} {mask:?}");
                        let c = descriptor(pf, eq, mask);
                        let view = form(&of(c), false, writes, &p)
                            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                        assert_eq!(view.blend, BlendEquation::of(&c), "{what}");
                        assert_eq!(view.extended, None, "{what}");
                        let view = form(&of(c), false, reads, &p)
                            .unwrap_or_else(|e| panic!("{what}: read: {e:?}"));
                        assert_eq!(view.blend, None, "{what}: read");
                    }
                }
                let what = format!("{stem} {pf:?} {math:?}");
                // A blend of the second source, which Metal's model does not hold: the generic
                // blend.
                let dual = descriptor(pf, (F::Source1Color, F::Zero, O::Add), ColorWriteMask::ALL);
                let view = form(&of(dual), false, writes, &p)
                    .unwrap_or_else(|e| panic!("{what}: dual: {e:?}"));
                assert!(view.generic_blend, "{what}: dual");
                let differs = descriptor(
                    pf,
                    (F::BlendColor, F::OneMinusBlendColor, O::Subtract),
                    ColorWriteMask::ALL,
                );
                // Under fast math too: Metal's form, the minuend's product fused, the
                // subtrahend's rounded.
                assert!(
                    form(&of(differs), false, writes, &p).is_ok_and(|v| v.blend.is_some()),
                    "{what}"
                );
                let replacing = descriptor(pf, (F::One, F::Zero, O::Add), ColorWriteMask::ALL);
                let view = form(&of(replacing), false, writes, &p).expect(&what);
                let kept_zero = math == BlendMath::Rounded;
                let want = BlendEquation::of(&replacing).filter(|_| kept_zero);
                assert_eq!(view.blend, want, "{what}: replacing");
                let none = descriptor(pf, equations[0], ColorWriteMask::NONE);
                let view = form(&of(none), false, writes, &p).expect(&what);
                assert!(
                    view.stores_nothing() && view.blend.is_none(),
                    "{what}: none"
                );
            }
        }
    }
}

/// A write mask short of the format's channels keeps the others: the view names them, an empty set
/// where the mask covers every channel (or the function only reads), all four where it covers none,
/// so that the write stores nothing. A signed normalized format whose mask covers none of its
/// channels stores nothing either, so nothing of it is lost. A mask that keeps some of a format's
/// channels compiles as its class: the view names the class's channels (every channel but the first
/// left out, so the first kept) and takes the pipeline's own from the draw (`kept_at_draw`),
/// whichever such mask the descriptor sets; an sRGB format's view names the mask's own and takes
/// none from the draw.
#[test]
fn a_write_mask_names_the_channels_the_write_keeps() {
    use crate::lower::fragment::kept_channels;
    use crate::vk::ColorComponentFlags as C;
    let p = profile("moltenvk-m3", &[]);
    let kept = |pf: PixelFormat, mask: ColorWriteMask, use_: ColorUse| {
        form(&state(pf, |c| c.write_mask = mask), false, use_, &p).map(|view| {
            (
                view.kept,
                view.kept_at_draw,
                view.merges(),
                view.stores_nothing(),
            )
        })
    };
    let rg = ColorWriteMask::RED.union(ColorWriteMask::GREEN);
    let ba = ColorWriteMask::BLUE.union(ColorWriteMask::ALPHA);
    let rgba8 = PixelFormat::RGBA8Unorm;
    assert_eq!(kept(rgba8, rg, WRITES), Ok((C::R, true, true, false)));
    // Another partial mask: the same class, the same view.
    assert_eq!(
        kept(rgba8, ColorWriteMask::BLUE, WRITES),
        kept(rgba8, rg, WRITES)
    );
    // The pipeline's own kept channels, which the draw names.
    assert_eq!(kept_channels(rgba8, rg, true), ba);
    assert_eq!(
        kept(rgba8, ColorWriteMask::ALL, WRITES),
        Ok((C::empty(), false, false, false))
    );
    assert_eq!(
        kept(rgba8, rg, READS),
        Ok((C::empty(), false, false, false))
    );
    let green = ColorWriteMask::GREEN;
    assert_eq!(
        kept(PixelFormat::RG16Float, green, WRITES),
        Ok((C::R, true, true, false))
    );
    // Blue and alpha are no channels of `RG16Float`: the write stores nothing.
    assert_eq!(
        kept(PixelFormat::RG16Float, ba, WRITES),
        Ok((C::known_bits(), false, false, true))
    );
    // An sRGB format's write merges its view's bytes under the mask's own channels.
    assert_eq!(
        kept(PixelFormat::BGRA8UnormSRGB, rg, WRITES),
        Ok((C::B | C::A, false, true, false))
    );
    assert_eq!(
        kept(PixelFormat::RG8Snorm, ColorWriteMask::NONE, WRITES),
        Ok((C::known_bits(), false, false, true))
    );
    assert_eq!(
        kept(PixelFormat::RG8Snorm, ColorWriteMask::ALL, WRITES),
        Ok((C::empty(), false, false, false))
    );
}

/// A multisampled pipeline takes the per-sample form: a multisampled storage image where the device
/// has `shaderStorageImageMultisample` (RADV, NVIDIA), planes, one single-sample layer per sample,
/// where it does not (ANV, MoltenVK); a pipeline of one sample takes one sample's image. The
/// `SampleId` read needs `sampleRateShading`, named where the device lacks it. Its returned sample
/// mask is modelled (the write is gated on the invocation's own sample's bit), one sample's is not.
/// Beside a depth attachment the module declares early tests, whose image writes a driver with
/// `DriverWorkaround::EarlyTestsDropMultisampledImageWrites` (MoltenVK on Apple's GPU) drops at
/// more than one sample: the writes are refused there, naming the defect, and taken elsewhere.
#[test]
fn a_multisampled_state_takes_the_per_sample_form_or_names_what_it_needs() {
    use crate::device::profile::DriverWorkaround;
    use crate::error::Needs;
    let r32 = state(PixelFormat::R32Float, |_| {});
    // Without a depth or stencil attachment the module declares no early tests.
    let writes = ColorUse {
        early_tests: false,
        ..WRITES
    };
    let mut depth = r32.clone();
    depth.depth_or_stencil = true;
    for (stem, samples) in [
        ("amd-rdna3-radv", StorageSamples::Multisampled),
        ("nvidia-rtx50-linux", StorageSamples::Multisampled),
        ("moltenvk-m3", StorageSamples::Planes),
        ("intel-arl-anv", StorageSamples::Planes),
        ("intel-irisxe-anv", StorageSamples::Planes),
    ] {
        let p = profile(stem, &[]);
        for use_ in [READS, writes] {
            let r = form(&r32, true, use_, &p);
            assert_eq!(r.clone().map(|v| v.samples), Ok(samples), "{stem} {use_:?}");
            assert_eq!(
                declared(r),
                (
                    vk::Format::R32_SFLOAT,
                    expected_declaration(
                        vk::Format::R32_SFLOAT,
                        use_,
                        crate::spv::ImageFormat::R32f,
                        &p
                    ),
                    SampledType::Float
                ),
                "{stem} {use_:?}"
            );
        }
        assert_eq!(
            form(&r32, false, READS, &p).map(|v| v.samples),
            Ok(StorageSamples::One),
            "{stem}"
        );
        let early = form(&depth, true, WRITES, &p).map(|v| v.samples);
        if stem == "moltenvk-m3" {
            assert_eq!(
                early,
                Err(Unsupported::driver_defect(
                    Feature::FramebufferFetch,
                    DriverWorkaround::EarlyTestsDropMultisampledImageWrites
                ))
            );
        } else {
            assert_eq!(early, Ok(samples), "{stem}");
        }
        assert_eq!(
            form(&depth, true, READS, &p).map(|v| v.samples),
            Ok(samples),
            "{stem}"
        );
        assert!(form(&depth, false, WRITES, &p).is_ok(), "{stem}");
    }
    let no_storage_ms = profile(
        "amd-rdna3-radv",
        &[VkFeature::ShaderStorageImageMultisample],
    );
    assert_eq!(
        form(&r32, true, READS, &no_storage_ms).map(|v| v.samples),
        Ok(StorageSamples::Planes)
    );
    for stem in ["amd-rdna3-radv", "moltenvk-m3"] {
        let bare = profile(stem, &[VkFeature::SampleRateShading]);
        let needs = form(&r32, true, READS, &bare).expect_err("blocked").needs;
        assert_eq!(
            needs,
            Needs::AnyOf(vec![vec![Capability::Feature(
                VkFeature::SampleRateShading
            )]]),
            "{stem}"
        );
        assert!(form(&r32, false, READS, &bare).is_ok(), "{stem}");
    }
}

/// What the device lacks blocks the form with that capability: the fragment-stage store for a
/// written attachment, and, for an extended format without `shaderStorageImageExtendedFormats`,
/// that feature unless the device reads and writes the format without a format, which it then
/// declares: by its features, or, without `shaderStorageImageReadWithoutFormat`, by the format's
/// own bits where `VK_KHR_format_feature_flags2` is enabled (read at the runtime's Vulkan 1.2).
#[test]
fn device_gaps_name_what_the_device_lacks() {
    use crate::error::Needs;
    let r32 = state(PixelFormat::R32Float, |_| {});
    let no_stores = profile("moltenvk-m3", &[VkFeature::FragmentStoresAndAtomics]);
    assert_eq!(
        form(&r32, false, WRITES, &no_stores)
            .expect_err("blocked")
            .needs,
        Needs::AnyOf(vec![vec![Capability::Feature(
            VkFeature::FragmentStoresAndAtomics
        )]])
    );
    assert!(form(&r32, false, READS, &no_stores).is_ok());
    let rg16 = state(PixelFormat::RG16Float, |_| {});
    let formatless = profile(
        "moltenvk-m3",
        &[VkFeature::ShaderStorageImageExtendedFormats],
    );
    assert_eq!(
        declared(form(&rg16, false, WRITES, &formatless)).1,
        crate::spv::ImageFormat::Unknown
    );
    assert!(form(&rg16, false, WRITES, &formatless)
        .expect("formatless")
        .formatless());
    let without = [
        VkFeature::ShaderStorageImageExtendedFormats,
        VkFeature::ShaderStorageImageReadWithoutFormat,
    ];
    let own_bits = profile_with("moltenvk-m3", &without, true);
    assert_eq!(
        declared(form(&rg16, false, READS, &own_bits)).1,
        crate::spv::ImageFormat::Unknown
    );
    let neither = profile_with("moltenvk-m3", &without, false);
    assert_eq!(
        form(&rg16, false, READS, &neither)
            .expect_err("blocked")
            .needs,
        Needs::AnyOf(vec![vec![Capability::Feature(
            VkFeature::ShaderStorageImageExtendedFormats
        )]])
    );
}

/// No attachment at the color read refuses the pipeline whatever the form.
#[test]
fn a_missing_attachment_is_a_descriptor_error() {
    let p = profile("moltenvk-m3", &[]);
    let none = FramebufferFetchInputs::of([0], &[None], false, false, 1);
    assert_eq!(
        storage_form(0, &none, false, READS, &p),
        Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
            index: 0
        }))
    );
}

/// An extended-range format's blend (`extended_range_blend`): taken where the device orders fetch
/// through the pixel interlock alone and the storage form holds, for every form Metal's model
/// blends and its `_sRGB` twin, and where it orders fetch through rasterization-order input
/// attachments, over which the function blends; the interlock named where it has neither; refused
/// for a padded form, whose blends are not measured, and for any other format.
#[test]
fn an_extended_range_blend_follows_the_ordered_storage_form() {
    let blended = [
        PixelFormat::BGR10XR,
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XR,
        PixelFormat::BGRA10XRSRGB,
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RGB10A82PXR10SRGB,
    ];
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "amd-780m-radv",
        "intel-arl-anv",
        "intel-irisxe-anv",
        "nvidia-rtx5080-laptop",
        "nvidia-pascal-linux",
    ] {
        let p = profile(stem, &[]);
        assert!(interlock_orders(&p), "{stem}");
        for pf in blended {
            assert_eq!(extended_range_blend(pf, &p), Ok(()), "{stem}: {pf:?}");
        }
        for pf in [PixelFormat::R10UnormX6, PixelFormat::RG10UnormX12] {
            assert!(extended_range_blend(pf, &p).is_err(), "{stem}: {pf:?}");
        }
        assert_eq!(
            extended_range_blend(PixelFormat::RGBA16Float, &p),
            Err(Unsupported::not_implemented(Feature::Format)),
            "{stem}"
        );
    }
    for stem in ["lavapipe", "gate-lavapipe-x86_64", "swiftshader"] {
        let p = profile(stem, &[]);
        assert!(!interlock_orders(&p), "{stem}");
        for pf in blended {
            assert_eq!(extended_range_blend(pf, &p), Ok(()), "{stem}: {pf:?}");
        }
        for pf in [PixelFormat::R10UnormX6, PixelFormat::RG10UnormX12] {
            assert_eq!(
                extended_range_blend(pf, &p),
                Err(Unsupported::not_implemented(Feature::Format)),
                "{stem}: {pf:?}"
            );
        }
    }
    let interlock = Unsupported::needs(
        Feature::FramebufferFetch,
        Capability::Feature(VkFeature::FragmentShaderPixelInterlock),
    );
    for stem in ["vk12-floor", "roadmap-2022"] {
        let p = profile(stem, &[]);
        assert_eq!(
            extended_range_blend(PixelFormat::BGRA10XR, &p),
            Err(interlock.clone()),
            "{stem}"
        );
    }
    // The storage form's own device needs, named where the device lacks one.
    let p = profile("moltenvk-m3", &[VkFeature::FragmentStoresAndAtomics]);
    assert!(extended_range_blend(PixelFormat::BGRA10XR, &p)
        .is_err_and(|u| u.feature == Feature::FramebufferFetch));
}

/// A packed 16-bit attachment the device samples but does not render (`A1BGR5Unorm`,
/// `ABGR4Unorm` on the RTX 5080 profiles) takes the storage form through a view of its words: the
/// view and declaration `R16_UINT`, `uint` texels the function reads as floats, and the word's
/// layout (`StorageView::word`), whose kept fields a partial write mask merges by the draw's word;
/// a replacing blend too. Another blend is the shader's (`StorageView::blend`), but one of the
/// second source, which is not built. A device that renders the format takes it directly, without
/// the word.
#[test]
fn a_word_attachment_takes_the_view_of_its_words() {
    use crate::metal::format::WordAttachment;
    let ni = Err(Unsupported::not_implemented(Feature::FramebufferFetch));
    for stem in ["nvidia-rtx5080-laptop", "nvidia-rtx50-linux"] {
        let p = profile(stem, &[]);
        assert!(interlock_orders(&p), "{stem}");
        for pf in [PixelFormat::A1BGR5Unorm, PixelFormat::ABGR4Unorm] {
            let what = format!("{stem} {pf:?}");
            let word = WordAttachment::layout(pf);
            assert_eq!(WordAttachment::of(pf, &p), word, "{what}");
            for mask in [
                ColorWriteMask::RED | ColorWriteMask::GREEN,
                ColorWriteMask::ALPHA,
                ColorWriteMask::ALL,
            ] {
                let view = form(&state(pf, |c| c.write_mask = mask), false, WRITES, &p)
                    .unwrap_or_else(|e| panic!("{what} {mask:?}: {e:?}"));
                assert_eq!(
                    (view.format, view.declared, view.numeric, view.word),
                    (
                        vk::Format::R16_UINT,
                        expected_declaration(
                            vk::Format::R16_UINT,
                            WRITES,
                            crate::spv::ImageFormat::R16ui,
                            &p
                        ),
                        SampledType::Uint,
                        word
                    ),
                    "{what} {mask:?}"
                );
                assert_eq!(view.read_class(), SampledType::Float, "{what}");
                assert_eq!(view.uint_view.or(view.rgba_view), None, "{what}");
                let partial = mask != ColorWriteMask::ALL;
                assert_eq!(
                    (view.merges(), view.kept_at_draw),
                    (partial, partial),
                    "{what}"
                );
            }
            let replacing = state(pf, |c| {
                c.blending_enabled = true;
                c.write_mask = ColorWriteMask::RED;
            });
            assert!(form(&replacing, false, WRITES, &p).is_ok(), "{what}");
            let blended = state(pf, |c| {
                c.blending_enabled = true;
                c.destination_rgb_blend_factor = crate::metal::BlendFactor::One;
            });
            let view = form(&blended, false, WRITES, &p)
                .unwrap_or_else(|e| panic!("{what} blended: {e:?}"));
            assert_eq!(
                (view.word, view.format),
                (word, vk::Format::R16_UINT),
                "{what}"
            );
            assert!(view.blend.is_some(), "{what}");
            let second = state(pf, |c| {
                c.blending_enabled = true;
                c.destination_rgb_blend_factor = crate::metal::BlendFactor::Source1Color;
            });
            assert_eq!(form(&second, false, WRITES, &p), ni, "{what}");
        }
    }
    let radv = profile("amd-rdna3-radv", &[]);
    let view = form(
        &state(PixelFormat::ABGR4Unorm, |c| {
            c.write_mask = ColorWriteMask::RED
        }),
        false,
        WRITES,
        &radv,
    );
    if let Ok(view) = view {
        assert_eq!(view.word, None);
        assert_eq!(view.format, vk::Format::R4G4B4A4_UNORM_PACK16);
    }
}
