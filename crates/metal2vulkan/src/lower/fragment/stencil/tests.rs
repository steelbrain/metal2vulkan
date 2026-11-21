//! The shader-side stencil's device form: where it is declared, and what the refusal names where it
//! is not; and the push word's layout.

use super::*;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::Needs;

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
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

fn alternatives(u: Unsupported) -> Vec<Vec<Capability>> {
    assert_eq!(u.feature, Feature::StencilExport);
    match u.needs {
        Needs::AnyOf(a) => a,
        other => panic!("{other:?}"),
    }
}

/// Every NVIDIA profile (no stencil export, with the ordered interlock) declares the shadow as
/// `R8ui`. A device that also lacks the extended storage formats, and `R8_UINT` storage without a
/// format, names them; the extension is always the first alternative.
#[test]
fn nvidia_declares_the_shadow_and_a_device_without_the_interlock_names_both_ways() {
    for stem in [
        "nvidia-rtx50-linux",
        "nvidia-pascal-linux",
        "nvidia-rtx5080-laptop",
    ] {
        let p = profile(stem, &[]);
        assert!(!p.fields().fragment.stencil_export, "{stem}");
        assert_eq!(
            device_form(&p, false),
            Ok(crate::spv::ImageFormat::R8ui),
            "{stem}"
        );
        assert_eq!(
            device_form(&p, true),
            Ok(crate::spv::ImageFormat::R8ui),
            "{stem}"
        );
    }
    let ext = vec![Capability::Extension(VkExtension::ExtShaderStencilExport)];
    let p = profile(
        "nvidia-rtx50-linux",
        &[
            VkFeature::FragmentShaderPixelInterlock,
            VkFeature::FragmentStoresAndAtomics,
        ],
    );
    let a = alternatives(device_form(&p, false).expect_err("no interlock"));
    assert_eq!(a[0], ext);
    assert_eq!(
        a[1],
        [
            Capability::Feature(VkFeature::FragmentShaderPixelInterlock),
            Capability::Feature(VkFeature::FragmentStoresAndAtomics),
        ]
    );
    let stem = "runtime-floor";
    let a = alternatives(device_form(&profile(stem, &[]), false).expect_err(stem));
    assert_eq!(a.len(), 2, "{stem}");
    assert_eq!(a[0], ext, "{stem}");
    assert!(
        a[1].contains(&Capability::Feature(
            VkFeature::FragmentShaderPixelInterlock
        )),
        "{stem}: {a:?}"
    );
}

/// The word's fields sit where `PushField::ShadowStencilState` states, each masked to its width.
/// The multisampled form also needs multisampled storage images: a device without them names
/// `shaderStorageImageMultisample` beside the extension, and the single-sample form does not.
#[test]
fn the_multisampled_form_names_multisampled_storage_images() {
    let p = profile(
        "nvidia-rtx50-linux",
        &[VkFeature::ShaderStorageImageMultisample],
    );
    assert_eq!(device_form(&p, false), Ok(crate::spv::ImageFormat::R8ui));
    let a = alternatives(device_form(&p, true).expect_err("no multisampled storage"));
    assert_eq!(
        a,
        [
            vec![Capability::Extension(VkExtension::ExtShaderStencilExport)],
            vec![Capability::Feature(
                VkFeature::ShaderStorageImageMultisample
            )],
        ]
    );
}

#[test]
fn a_face_word_packs_the_state_in_its_documented_bits() {
    let w = face_word(5, 2, 7, 0x0F, 0xF0, true);
    assert_eq!(w & 7, 5);
    assert_eq!((w >> 3) & 7, 2);
    assert_eq!((w >> 6) & 7, 7);
    assert_eq!((w >> 9) & 0xFF, 0x0F);
    assert_eq!((w >> 17) & 0xFF, 0xF0);
    assert_eq!(w & ATTACHED_BIT, ATTACHED_BIT);
    // Wider masks keep their low 8 bits, as Metal's 8-bit stencil does.
    let w = face_word(7, 0, 0, u32::MAX, 0x1FF, false);
    assert_eq!((w >> 9) & 0xFF, 0xFF);
    assert_eq!((w >> 17) & 0xFF, 0xFF);
    assert_eq!(w & ATTACHED_BIT, 0);
    assert_eq!(w & 0x7E00_0000, 0, "bits 25 to 30 stay clear");
}
