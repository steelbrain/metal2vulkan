//! `private_properties`: Apple's private device properties, as Metal answers them and as the
//! runtime answers them on every committed device profile, read from what the runtime acts on.

use std::collections::BTreeSet;

use super::capabilities::{committed_descriptions, created};
use super::families::{all_audits, format_capabilities};
use crate::argument_table::{check_index, BUFFER_SLOTS, SAMPLER_SLOTS, TEXTURE_SLOTS};
use crate::private_properties::MEASURED_FAMILY;
use crate::texture_descriptor::{Refused, TextureDescriptor, TextureProblem};
use crate::{PrivateAnswer, PrivateCount, PrivateFlag, PrivateProperties};
use metal2vulkan::device::{Device, DevicePolicy};
use metal2vulkan::metal::TextureType;

/// The device linear texture alignment [`answers`] reads from on the committed profile `stem`
/// (`Device::device_linear_texture_alignment`, which only a device answers): MoltenVK's 16, Metal's
/// own, as the M3 Max's answers (its linear images lay rows at the packed row rounded up to 16 and
/// align their memory to 16), and 256 elsewhere, as a device whose images need more answers.
fn linear(stem: &str) -> u64 {
    match stem.starts_with("moltenvk-") {
        true => 16,
        false => 256,
    }
}

/// The answers on the committed profile `stem`, as `Device::private_properties` reads them, its
/// linear texture alignment [`linear`]'s.
fn answers(stem: &str, desc: &metal2vulkan::device::DeviceDescription) -> PrivateProperties {
    let (enabled, properties) = created(stem, desc);
    let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
    let rows = all_audits(stem, desc).remove(&MEASURED_FAMILY).unwrap();
    let formats = |pf| format_capabilities(&enabled, &compile, pf);
    PrivateProperties::of(
        &properties,
        compile.metal_properties(),
        &rows,
        (&formats, linear(stem)),
        no_copy(desc),
    )
}

/// Whether a device of the committed profile `desc` makes no-copy buffers: one that enables
/// `VK_EXT_external_memory_host` where it lists it does (a profile names no dma-buf route, which is
/// outside the vocabulary).
fn no_copy(desc: &metal2vulkan::device::DeviceDescription) -> bool {
    (desc.extensions).contains(&metal2vulkan::device::VkExtension::ExtExternalMemoryHost)
}

/// `ALL` lists each property once, in declaration order, under a selector of its own.
#[test]
fn every_property_has_one_selector() {
    for (index, p) in PrivateCount::ALL.into_iter().enumerate() {
        assert_eq!(p as usize, index, "{p:?}");
    }
    for (index, p) in PrivateFlag::ALL.into_iter().enumerate() {
        assert_eq!(p as usize, index, "{p:?}");
    }
    let selectors: BTreeSet<&str> = PrivateCount::ALL
        .iter()
        .map(|p| p.selector())
        .chain(PrivateFlag::ALL.iter().map(|p| p.selector()))
        .collect();
    assert_eq!(
        selectors.len(),
        PrivateCount::ALL.len() + PrivateFlag::ALL.len()
    );
}

/// Spot values as the probes printed them on an Apple M3 Max at macOS 27.0.1 (an Apple M4 there
/// answers `gpuCoreCount` 10 and `supportsComputeCompressedTextureWrite` yes; the M3 Max 40 and
/// no).
#[test]
fn the_answers_are_the_probes() {
    assert_eq!(PrivateCount::GpuCoreCount.metal(), 40);
    assert_eq!(PrivateCount::MaxComputeBuffers.metal(), 31);
    assert_eq!(PrivateCount::MaxComputeThreadgroupMemory.metal(), 65536);
    assert_eq!(PrivateCount::MaxVertexAmplificationCount.metal(), 2);
    assert_eq!(
        PrivateCount::MaxPredicatedNestingDepth.metal(),
        u64::from(u32::MAX)
    );
    assert!(PrivateFlag::SupportsTileShaders.metal());
    assert!(PrivateFlag::SupportsMemoryOrderAtomics.metal());
    assert!(!PrivateFlag::SupportsComputeCompressedTextureWrite.metal());
}

/// Which properties answer Metal's measured value: those no document describes.
#[test]
fn metals_answers_are_the_undocumented_properties() {
    use PrivateCount as C;
    use PrivateFlag as F;
    let metal_counts = BTreeSet::from([
        C::GpuCoreCount,
        C::MaxVertexAmplificationCount,
        C::MaxComputeThreadgroupMemory,
    ]);
    let metal_flags = BTreeSet::from([
        F::IsLargeMrtSupported,
        F::SupportsComputeCompressedTextureWrite,
        F::SupportsConditionalLoadStore,
    ]);
    for (stem, desc) in committed_descriptions() {
        let p = answers(&stem, &desc);
        for c in PrivateCount::ALL {
            let metal = c.metal();
            match p.count(c) {
                PrivateAnswer::Metal(v) => {
                    assert!(metal_counts.contains(&c), "{stem} {c:?}");
                    assert_eq!(v, metal, "{stem} {c:?}");
                    assert_eq!(p.count(c).value(), None, "{stem} {c:?}");
                    assert!(p.unknown().contains(&c.selector()), "{stem} {c:?}");
                }
                PrivateAnswer::Device { metal: m, .. } => {
                    assert!(!metal_counts.contains(&c), "{stem} {c:?}");
                    assert_eq!(m, metal, "{stem} {c:?}");
                }
            }
        }
        for f in PrivateFlag::ALL {
            let metal = f.metal();
            match p.flag(f) {
                PrivateAnswer::Metal(v) => {
                    assert!(metal_flags.contains(&f), "{stem} {f:?}");
                    assert_eq!(v, metal, "{stem} {f:?}");
                    assert_eq!(p.flag(f).value(), None, "{stem} {f:?}");
                    assert!(p.unknown().contains(&f.selector()), "{stem} {f:?}");
                }
                PrivateAnswer::Device { metal: m, .. } => {
                    assert!(!metal_flags.contains(&f), "{stem} {f:?}");
                    assert_eq!(m, metal, "{stem} {f:?}");
                }
            }
        }
    }
}

/// The per-stage table counts are what the encoders' tables take, and the 2D size and layers
/// what the descriptor takes, on every profile: the last index or size answered is taken, the
/// next refused. On MoltenVK on an M3 and an M4 every limit answer is Metal's.
#[test]
fn table_counts_and_sizes_are_what_the_runtime_takes() {
    use PrivateCount as C;
    for (stem, desc) in committed_descriptions() {
        let p = answers(&stem, &desc);
        for (counts, slots) in [
            (
                [
                    C::MaxVertexBuffers,
                    C::MaxFragmentBuffers,
                    C::MaxComputeBuffers,
                    C::MaxTileBuffers,
                ],
                BUFFER_SLOTS,
            ),
            (
                [
                    C::MaxVertexTextures,
                    C::MaxFragmentTextures,
                    C::MaxComputeTextures,
                    C::MaxTileTextures,
                ],
                TEXTURE_SLOTS,
            ),
            (
                [
                    C::MaxVertexSamplers,
                    C::MaxFragmentSamplers,
                    C::MaxComputeSamplers,
                    C::MaxTileSamplers,
                ],
                SAMPLER_SLOTS,
            ),
        ] {
            for c in counts {
                let answer = u32::try_from(p.count(c).value().unwrap()).unwrap();
                assert!(answer <= slots, "{stem} {c:?}");
                if answer == slots {
                    assert!(check_index("t", answer - 1, slots).is_ok(), "{stem} {c:?}");
                    assert!(check_index("t", answer, slots).is_err(), "{stem} {c:?}");
                }
            }
        }
        let side = p.count(C::MaxTextureWidth2D).value().unwrap();
        assert_eq!(Some(side), p.count(C::MaxTextureHeight2D).value(), "{stem}");
        let square = |n: u64| TextureDescriptor {
            texture_type: TextureType::Type2D,
            width: n,
            height: n,
            ..TextureDescriptor::default()
        };
        let rules = super::texture_descriptor::APPLE;
        assert!(square(side).check(rules).is_ok(), "{stem} {side}");
        if side == 16384 {
            assert_eq!(
                square(side + 1).check(rules),
                Err(Refused::Stops(TextureProblem::PastLargestSide {
                    max: side
                })),
                "{stem}"
            );
        }
        let layers = p.count(C::MaxTextureLayers).value().unwrap();
        let array = |n: u64| TextureDescriptor {
            texture_type: TextureType::Type2DArray,
            array_length: n,
            ..TextureDescriptor::default()
        };
        assert!(array(layers).check(rules).is_ok(), "{stem} {layers}");
        if stem.starts_with("moltenvk-") {
            for c in PrivateCount::ALL {
                if let PrivateAnswer::Device { answer, metal } = p.count(c) {
                    assert_eq!(answer, metal, "{stem} {c:?}");
                }
            }
        }
    }
}

/// The runtime draws Metal's five public primitive types and none of the private ones.
#[test]
fn the_device_draws_the_public_primitive_types() {
    let (stem, desc) = committed_descriptions().into_iter().next().unwrap();
    let (_, properties) = created(&stem, &desc);
    assert!((0..5).all(|raw| properties.supports_primitive_type(raw)));
    assert!(!(5..13).any(|raw| properties.supports_primitive_type(raw)));
    assert!(!properties.supports_primitive_type(u64::from(u32::MAX) + 1));
}

/// Command predication's properties answer Metal's on every device, with device addresses (whose
/// step module reaches memory by them) or without (whose step module binds its places).
#[test]
fn predication_answers_where_it_runs() {
    let mut seen = (0, 0);
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created(&stem, &desc);
        let compile = Device::new(enabled, DevicePolicy::UNTRUSTED).unwrap();
        match compile.profile().addressed_needs().is_empty() {
            true => seen.0 += 1,
            false => seen.1 += 1,
        }
        let p = answers(&stem, &desc);
        let jump = p.flag(PrivateFlag::SupportsCommandBufferJump).value();
        assert_eq!(jump, Some(true), "{stem}");
        let depth = p.count(PrivateCount::MaxPredicatedNestingDepth).value();
        assert_eq!(depth, Some(u64::from(u32::MAX)), "{stem}");
    }
    assert!(seen.0 > 0 && seen.1 > 0, "{seen:?}");
}

/// `gpuCoreCount` is the device's where its driver reports its cores, and none elsewhere: never
/// Apple's GPU's; `supportsNativeHardwareFP16` is the device's `shaderFloat16`.
#[test]
fn the_core_count_and_fp16_are_the_devices() {
    use crate::capabilities::DeviceProperties;
    for (stem, desc) in committed_descriptions() {
        let (enabled, properties) = created(&stem, &desc);
        let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
        let rows = all_audits(&stem, &desc).remove(&MEASURED_FAMILY).unwrap();
        let formats = |pf| format_capabilities(&enabled, &compile, pf);
        let of = |properties: &DeviceProperties| {
            PrivateProperties::of(
                properties,
                compile.metal_properties(),
                &rows,
                (&formats, linear(&stem)),
                no_copy(&desc),
            )
        };
        let unknown = of(&properties).count(PrivateCount::GpuCoreCount);
        assert_eq!(unknown.value(), None, "{stem}");
        assert_eq!(unknown.metal(), 40, "{stem}: Metal's, the M3 Max's, beside");
        let mut reported = properties.clone();
        reported.gpu_core_count = Some(84);
        assert_eq!(
            of(&reported).count(PrivateCount::GpuCoreCount).value(),
            Some(84),
            "{stem}"
        );
        assert_eq!(
            of(&properties)
                .flag(PrivateFlag::SupportsNativeHardwareFp16)
                .value(),
            Some(enabled.has(metal2vulkan::device::VkFeature::ShaderFloat16)),
            "{stem}"
        );
    }
}

/// The answers read from the device: the linear texture alignments and the IOSurface ones are the
/// device's (`Device::device_linear_texture_alignment`); ordered atomics follow the Vulkan memory
/// model at device scope; the XR10 formats and framebuffer reads follow their family rows (reads yes on
/// MoltenVK and RADV); a half-float texture is filtered
/// bicubically everywhere; and `RGB10A2Unorm_sRGB`, held as its raw word and filtered by the
/// function (`ExtendedRange::Rgb10A2Srgb`), is filtered everywhere.
#[test]
fn the_alignments_atomics_and_formats_are_the_devices() {
    use crate::families::{FamilyRow, FeatureRow, RowState};
    use metal2vulkan::device::VkFeature;
    use PrivateCount as C;
    use PrivateFlag as F;
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created(&stem, &desc);
        let p = answers(&stem, &desc);
        for c in [
            C::DeviceLinearTextureAlignmentBytes,
            C::LinearTextureAlignmentBytes,
            C::IosurfaceTextureAlignmentBytes,
            C::IosurfaceReadOnlyTextureAlignmentBytes,
        ] {
            assert_eq!(
                p.count(c),
                PrivateAnswer::Device {
                    answer: linear(&stem),
                    metal: 16
                },
                "{stem} {c:?}"
            );
        }
        let model = enabled.has(VkFeature::VulkanMemoryModel)
            && enabled.has(VkFeature::VulkanMemoryModelDeviceScope);
        let flag = |f| p.flag(f).value();
        assert_eq!(flag(F::SupportsMemoryOrderAtomics), Some(model), "{stem}");
        let rows = all_audits(&stem, &desc).remove(&MEASURED_FAMILY).unwrap();
        let xr10 = rows
            .iter()
            .find(|a| a.row == FamilyRow::Feature(FeatureRow::ExtendedRangePixelFormats))
            .map(|a| a.state == RowState::Exact);
        assert_eq!(flag(F::SupportsExtendedXr10Formats), xr10, "{stem}");
        assert_eq!(
            flag(F::SupportsFloat16BcubicFiltering),
            Some(true),
            "{stem}"
        );
        assert_eq!(flag(F::IsRgb10a2GammaSupported), Some(true), "{stem}");
        let blending = rows
            .iter()
            .find(|a| a.row == FamilyRow::Feature(FeatureRow::ProgrammableBlending))
            .map(|a| a.state == RowState::Exact);
        assert_eq!(flag(F::IsFramebufferReadSupported), blending, "{stem}");
        if stem.starts_with("moltenvk-") || stem.starts_with("amd-") {
            assert_eq!(flag(F::IsFramebufferReadSupported), Some(true), "{stem}");
        }
    }
}
