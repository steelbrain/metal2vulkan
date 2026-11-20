//! The depth/stencil copy modules: what each needs, valid on every committed profile that has it
//! (L3, `G-val`), and refused on a device that does not.

use super::*;
use crate::device::requirements::{VkExtension, VkFeature};
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::testing::assert_valid;
use std::path::PathBuf;

/// Every copy draw: each plane of each source, the export of each source, the depth, the
/// depth's and the stencil's bits into colour, a depth's magnitude as a depth and as bits, and a
/// 16-bit depth's code both ways.
fn every_copy() -> Vec<DepthStencilCopy> {
    let mut copies = Vec::new();
    for source in [CopySource::Aspect, CopySource::Shadow] {
        for bit in StencilBit::ALL {
            copies.push(DepthStencilCopy::StencilPlane { source, bit });
        }
        copies.push(DepthStencilCopy::StencilExport { source });
    }
    copies.push(DepthStencilCopy::Depth);
    copies.push(DepthStencilCopy::DepthBits);
    copies.push(DepthStencilCopy::StencilBits);
    copies.push(DepthStencilCopy::Depth16Code);
    copies.push(DepthStencilCopy::Depth16Bits);
    for form in FORMS {
        copies.push(DepthStencilCopy::DepthFromStandIn { form });
    }
    copies
}

/// Both depth stand-in forms.
const FORMS: [DepthStandInForm; 2] = [
    DepthStandInForm::Multisampled,
    DepthStandInForm::SamplePlanes,
];

/// The requirements of `m` as serialized.
fn needs(m: TypedModule) -> Requirements {
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .requirements
}

/// The bits' masks are the eight single bits, from the lowest.
#[test]
fn each_bit_masks_its_plane() {
    let masks: Vec<u32> = StencilBit::ALL.iter().map(|bit| bit.mask()).collect();
    assert_eq!(masks, [1, 2, 4, 8, 16, 32, 64, 128]);
}

/// Each module needs exactly what the module docs say: every fragment stage `sampleRateShading`,
/// a read of the shadow multisampled storage images, the export the extension, the split
/// multisampled storage images of an extended format and no sample-rate shading.
#[test]
fn each_module_needs_what_its_docs_say() {
    for copy in every_copy() {
        let r = needs(fragment_typed_module(copy).expect("fragment"));
        assert!(
            r.features.contains(&VkFeature::SampleRateShading),
            "{copy:?}: {r:?}"
        );
        let export = matches!(copy, DepthStencilCopy::StencilExport { .. });
        assert_eq!(
            r.extensions.contains(&VkExtension::ExtShaderStencilExport),
            export,
            "{copy:?}: {r:?}"
        );
        let shadow = matches!(
            copy,
            DepthStencilCopy::StencilPlane {
                source: CopySource::Shadow,
                ..
            } | DepthStencilCopy::StencilExport {
                source: CopySource::Shadow
            } | DepthStencilCopy::DepthFromStandIn {
                form: DepthStandInForm::Multisampled
            }
        );
        assert_eq!(
            r.features
                .contains(&VkFeature::ShaderStorageImageMultisample),
            shadow,
            "{copy:?}: {r:?}"
        );
    }
    let r = needs(split_typed_module().expect("split"));
    for feature in [
        VkFeature::ShaderStorageImageMultisample,
        VkFeature::ShaderStorageImageExtendedFormats,
    ] {
        assert!(r.features.contains(&feature), "{feature:?}: {r:?}");
    }
    assert!(!r.features.contains(&VkFeature::SampleRateShading), "{r:?}");
    // The depth split: multisampled storage images for a multisampled stand-in alone, and no
    // extended format (`R32_SFLOAT` is a base storage format) nor sample-rate shading.
    for form in FORMS {
        let r = needs(depth_split_typed_module(form).expect("depth split"));
        assert_eq!(
            r.features
                .contains(&VkFeature::ShaderStorageImageMultisample),
            form == DepthStandInForm::Multisampled,
            "{form:?}: {r:?}"
        );
        assert!(
            !r.features
                .contains(&VkFeature::ShaderStorageImageExtendedFormats),
            "{form:?}: {r:?}"
        );
        assert!(
            !r.features.contains(&VkFeature::SampleRateShading),
            "{form:?}: {r:?}"
        );
    }
}

/// Every committed device (`validation/profiles/*.json`), derived with the default policy, by
/// file stem.
fn committed_devices() -> Vec<(String, Device)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("profiles") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("read");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let device = Device::from_queries(desc, DevicePolicy::default()).expect("device");
        out.push((stem, device));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// On every committed device each module is valid where the device has what it needs, numbered
/// with the fixed bindings, and refused (`Unsupported`, naming the consumer's feature) where it
/// does not, never given as words it cannot run.
#[test]
fn the_modules_are_valid_where_the_device_has_what_they_need() {
    let feature = Feature::SeparateMultisampledDepthStencil;
    let (mut draws, mut splits) = (0, 0);
    for (name, device) in committed_devices() {
        let profile = device.profile();
        let fits = |m: TypedModule| profile.check(&needs(m)).is_ok();
        for copy in every_copy() {
            let fit = fits(fragment_typed_module(copy).expect("fragment"));
            match depth_stencil_copy_modules(&device, copy, feature.clone()) {
                Ok(modules) => {
                    assert!(fit, "{name} {copy:?}");
                    assert_valid(fragment_typed_module(copy).expect("fragment"), profile);
                    assert!(!modules.vertex.is_empty() && !modules.fragment.is_empty());
                    draws += 1;
                }
                Err(Error::Unsupported(u)) => {
                    assert!(!fit, "{name} {copy:?}");
                    assert_eq!(u.feature, feature, "{name} {copy:?}");
                }
                Err(other) => panic!("{name} {copy:?}: {other}"),
            }
        }
        let fit = fits(split_typed_module().expect("split"));
        match stencil_split_module(&device, feature.clone()) {
            Ok(module) => {
                assert!(fit, "{name}");
                assert_valid(split_typed_module().expect("split"), profile);
                assert!(!module.spirv.is_empty());
                splits += 1;
            }
            Err(Error::Unsupported(u)) => {
                assert!(!fit, "{name}");
                assert_eq!(u.feature, feature, "{name}");
            }
            Err(other) => panic!("{name}: {other}"),
        }
    }
    assert!(draws > 0 && splits > 0, "{draws} {splits}");
}

/// On every committed device the depth split into each stand-in form is valid where the device
/// has what it needs and refused, typed, where it does not; the planes' form runs on every
/// committed device (it needs no multisampled storage image).
#[test]
fn the_depth_split_is_valid_where_the_device_has_what_it_needs() {
    let feature = Feature::FramebufferFetch;
    for (name, device) in committed_devices() {
        let profile = device.profile();
        for form in FORMS {
            let m = || depth_split_typed_module(form).expect("depth split");
            let fit = profile.check(&needs(m())).is_ok();
            match depth_split_module(&device, form, feature.clone()) {
                Ok(module) => {
                    assert!(fit, "{name} {form:?}");
                    assert_valid(m(), profile);
                    assert!(!module.spirv.is_empty());
                }
                Err(Error::Unsupported(u)) => {
                    assert!(!fit, "{name} {form:?}");
                    assert!(form != DepthStandInForm::SamplePlanes, "{name}");
                    assert_eq!(u.feature, feature, "{name} {form:?}");
                }
                Err(other) => panic!("{name} {form:?}: {other}"),
            }
        }
    }
}

/// The 16-bit depth pair's arithmetic is exact for **every** code: of all 65536 codes, each one
/// `Depth16Code` turns into a depth and `Depth16Bits` turns back lands on itself, whether or not
/// the device contracts the multiply and add into one fused operation, and the margin the
/// rounding has left over is the module docs' number.
///
/// 65536 inputs is a small enough space to check whole, so it is checked whole rather than
/// sampled. This is the arithmetic the two shaders perform, in the same order and the same
/// precision; that a real device performs it so is a device test's half of the claim.
///
/// Its failure mode was exercised while writing it: dropping the `+ 0.5`, so the conversion to
/// unsigned truncates `d * 65535` instead of rounding it, misses 36442 of the 65536 codes.
#[test]
fn every_16_bit_code_survives_the_depth_pair_in_both_directions() {
    let mut worst = 0.0f64;
    for code in 0..=u16::MAX {
        // `Depth16Code`: ConvertUToF then one FDiv by 65535.
        let depth = f32::from(code) / CODE_SCALE;
        // `Depth16Bits`: FMul, FAdd, then ConvertFToU, which truncates. Once as the device may
        // emit it and once contracted into a fused multiply-add, which rounds only once.
        let plain = (depth * CODE_SCALE + 0.5) as u32;
        let fused = (f64::from(depth) * f64::from(CODE_SCALE) + 0.5) as u32;
        assert_eq!(plain, u32::from(code), "code {code} un-fused");
        assert_eq!(fused, u32::from(code), "code {code} fused");
        let scaled = f64::from(depth) * f64::from(CODE_SCALE);
        worst = worst.max((scaled - f64::from(code)).abs());
    }
    // The half-step budget is 0.5; the docs quote this margin, so it is held to it.
    assert!(
        worst < 2.0e-3,
        "the largest |depth * 65535 - code| is {worst}"
    );
}
