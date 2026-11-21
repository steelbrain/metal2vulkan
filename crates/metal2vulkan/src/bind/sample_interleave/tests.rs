//! The interleaved per-sample copy modules: what each needs, valid on every committed profile that
//! has it (L3, `G-val`), refused on a device that does not.

use super::*;
use crate::device::requirements::VkFeature;
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::testing::assert_valid;
use std::path::PathBuf;

/// The sample counts the modules are built for (Apple's 2 and 4, and 8 a Vulkan device may have).
const SAMPLES: [u32; 3] = [2, 4, 8];

/// The requirements of `m` as serialized.
fn needs(m: TypedModule) -> Requirements {
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .requirements
}

/// Each module needs exactly what the module docs say: the split of an extended storage format
/// `shaderStorageImageExtendedFormats`, the merge's fragment stage `sampleRateShading`.
#[test]
fn each_module_needs_what_its_docs_say() {
    for copy in PlanesCopy::ALL.into_iter().filter(|c| c.moves_bits()) {
        for samples in SAMPLES {
            let r = needs(split_typed_module(copy, samples).expect("split"));
            let extended = !matches!(copy, PlanesCopy::R32 | PlanesCopy::Rgba32);
            assert_eq!(
                r.features
                    .contains(&VkFeature::ShaderStorageImageExtendedFormats),
                extended,
                "{copy:?} {samples}: {r:?}"
            );
            assert!(
                !r.features.contains(&VkFeature::SampleRateShading),
                "{copy:?} {samples}"
            );
        }
    }
    for samples in SAMPLES {
        let r = needs(merge_fragment_typed_module(samples).expect("fragment"));
        assert!(
            r.features.contains(&VkFeature::SampleRateShading),
            "{samples}: {r:?}"
        );
    }
}

/// The merge's fragment stage bakes its sample count into the wide column `x * n + s`, so each
/// count's module differs from every other's: a cache that shares one count's module with another
/// count would read the wrong wide columns (the probe failure, whose 4-sample copy
/// ran the 2-sample module made first).
#[test]
fn the_merge_fragment_module_is_one_per_sample_count() {
    let words: Vec<Vec<u32>> = SAMPLES
        .iter()
        .map(|&samples| {
            let m = merge_fragment_typed_module(samples).expect("fragment");
            crate::spirv::serialize::serialize(m)
                .expect("serialize")
                .words
        })
        .collect();
    for (i, a) in words.iter().enumerate() {
        for b in &words[i + 1..] {
            assert_ne!(a, b, "two sample counts' merge modules are the same words");
        }
    }
}

/// Every committed device (`validation/profiles/*.json`), derived with the resident-table domains,
/// by file stem.
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
        let device = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
        out.push((stem, device));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// On every committed device each module is valid where the device has what it needs, numbered
/// with the fixed bindings, and refused (`Unsupported`) where it does not, never given as words
/// it cannot run.
#[test]
fn the_modules_are_valid_where_the_device_has_what_they_need() {
    let mut built = 0;
    for (name, device) in committed_devices() {
        let profile = device.profile();
        for copy in PlanesCopy::ALL.into_iter().filter(|c| c.moves_bits()) {
            for samples in SAMPLES {
                let fits = profile
                    .check(&needs(split_typed_module(copy, samples).expect("split")))
                    .is_ok();
                match interleave_split_module(&device, copy, samples) {
                    Ok(module) => {
                        assert!(fits, "{name} {copy:?} {samples}");
                        assert_valid(split_typed_module(copy, samples).expect("split"), profile);
                        assert!(!module.spirv.is_empty());
                        built += 1;
                    }
                    Err(Error::Unsupported(_)) => assert!(!fits, "{name} {copy:?} {samples}"),
                    Err(other) => panic!("{name} {copy:?} {samples}: {other}"),
                }
            }
        }
        // The sample planes' copy of floats moves no bits: no interleave takes it.
        assert_eq!(
            interleave_split_module(&device, PlanesCopy::Alpha8, 4).err(),
            Some(Error::Unsupported(Unsupported::not_implemented(
                Feature::Format
            ))),
            "{name}"
        );
        for samples in SAMPLES {
            let fits = profile
                .check(&needs(
                    merge_fragment_typed_module(samples).expect("fragment"),
                ))
                .is_ok();
            match interleave_merge_modules(&device, samples) {
                Ok(modules) => {
                    assert!(fits, "{name} {samples}");
                    assert_valid(
                        merge_fragment_typed_module(samples).expect("fragment"),
                        profile,
                    );
                    assert!(!modules.vertex.is_empty() && !modules.fragment.is_empty());
                }
                Err(Error::Unsupported(_)) => assert!(!fits, "{name} {samples}"),
                Err(other) => panic!("{name} {samples}: {other}"),
            }
        }
    }
    assert!(built > 0);
}
