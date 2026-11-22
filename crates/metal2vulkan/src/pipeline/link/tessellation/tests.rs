//! The control plan's checks: the descriptor's factor maximum, the device's generation level,
//! multiview; the control-point index type's bindings; the fragment stage's primitive id; and the
//! control stage's push range placement.

use super::*;
use crate::device::DevicePolicy;
use crate::device::{Device, DeviceDescription};
use crate::error::Needs;
use crate::pipeline::interface::PatchDomain;

fn profile(stem: &str) -> DeviceProfile {
    crate::pipeline::testing::profile(stem, DevicePolicy::RESIDENT_TABLES).expect("profile")
}

/// Committed profile `stem` with `maxTessellationGenerationLevel` edited to `level`.
fn profile_with_level(stem: &str, level: u32) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile text");
    let edited = text.replace(
        "\"maxTessellationGenerationLevel\": 64",
        &format!("\"maxTessellationGenerationLevel\": {level}"),
    );
    assert_ne!(edited, text, "{stem} states the level");
    let desc = DeviceDescription::from_khronos_json(&edited, None).expect("description");
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
        .expect("device")
        .profile()
        .clone()
}

fn stage() -> TessellationStage {
    TessellationStage::new(PatchDomain::Triangle, 3)
}

fn descriptor(max: u32, partition: Partition) -> TessellationDescriptor {
    TessellationDescriptor {
        max_tessellation_factor: max,
        partition_mode: partition,
        ..TessellationDescriptor::default()
    }
}

/// Metal's rules for `maxTessellationFactor`: 1 to 64, a power of two for `pow2`, even for the
/// fractional partitions.
#[test]
fn the_factor_maximum_follows_metals_rules() {
    let p = profile("moltenvk-m3");
    let valid = |max, partition| control_plan(Some(&descriptor(max, partition)), stage(), 1, &p);
    for (max, partition, ok) in [
        (16, Partition::Pow2, true),
        (64, Partition::Pow2, true),
        (1, Partition::Pow2, true),
        (12, Partition::Pow2, false),
        (0, Partition::Integer, false),
        (65, Partition::Integer, false),
        (13, Partition::Integer, true),
        (13, Partition::FractionalOdd, false),
        (14, Partition::FractionalOdd, true),
        (15, Partition::FractionalEven, false),
        (64, Partition::FractionalEven, true),
    ] {
        match valid(max, partition) {
            Ok(_) => assert!(ok, "{max} {partition:?}"),
            Err(Error::InvalidDescriptor(DescriptorError::MaxTessellationFactor {
                factor,
                partition_mode,
            })) => {
                assert!(!ok, "{max} {partition:?}");
                assert_eq!((factor, partition_mode), (max, partition));
            }
            Err(e) => panic!("{max} {partition:?}: {e}"),
        }
    }
    // `None` is Metal's defaults.
    let plan = control_plan(None, stage(), 1, &p).expect("defaults");
    assert_eq!(plan.desc, TessellationDescriptor::default());
}

/// A control-point index type adds the control-point index buffer to the control stage's
/// bindings and to its control stage and state.
#[test]
fn a_control_point_index_type_binds_its_index_buffer() {
    let p = profile("moltenvk-m3");
    let mut d = TessellationDescriptor::default();
    let plan = control_plan(Some(&d), stage(), 1, &p).expect("none");
    assert_eq!(
        plan.slots(),
        [control::FACTOR_SLOT, control::PATCH_INDEX_SLOT]
    );
    for ty in [
        TessellationControlPointIndexType::UInt16,
        TessellationControlPointIndexType::UInt32,
    ] {
        d.control_point_index_type = ty;
        let plan = control_plan(Some(&d), stage(), 1, &p).expect("indexed");
        assert_eq!(
            plan.slots(),
            [
                control::FACTOR_SLOT,
                control::PATCH_INDEX_SLOT,
                control::CONTROL_POINT_INDEX_SLOT
            ]
        );
        let c = plan.control(8);
        assert_eq!((c.control_points, c.index_type), (3, ty));
        assert_eq!(plan.state(8).control_point_index_type, ty);
    }
}

/// A factor maximum above the device's generation level needs the level.
#[test]
fn the_generation_level_is_refused_typed() {
    let low = profile_with_level("moltenvk-m3", 32);
    let d = descriptor(64, Partition::Pow2);
    match control_plan(Some(&d), stage(), 1, &low) {
        Err(Error::Unsupported(u)) => assert_eq!(
            u.needs,
            Needs::AnyOf(vec![vec![Capability::Limit {
                limit: VkLimit::MaxTessellationGenerationLevel,
                required: 64,
                available: 32,
            }]])
        ),
        other => panic!("{other:?}"),
    }
    control_plan(Some(&descriptor(32, Partition::Pow2)), stage(), 1, &low).expect("32 fits");
}

/// A multiview pipeline with tessellation stages needs `multiviewTessellationShader`, which
/// MoltenVK's profile lacks and RADV's has.
#[test]
fn multiview_tessellation_needs_its_feature() {
    let m3 = profile("moltenvk-m3");
    match control_plan(None, stage(), 2, &m3) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::VertexAmplification);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::MultiviewTessellationShader
                )]])
            );
        }
        other => panic!("{other:?}"),
    }
    control_plan(None, stage(), 1, &m3).expect("one view");
    control_plan(None, stage(), 2, &profile("amd-780m-radv")).expect("radv");
}

/// A fragment stage reading `[[primitive_id]]` behind an evaluation stage is not implemented.
#[test]
fn a_fragment_primitive_id_is_refused() {
    let mut fs = crate::pipeline::link::tests::iface(
        crate::pipeline::sym::Stage::Fragment,
        Vec::new(),
        Vec::new(),
    );
    check_fragment(Some(&fs)).expect("no primitive id");
    check_fragment(None).expect("no fragment stage");
    fs.io.builtins.push(StageBuiltin::PrimitiveId);
    match check_fragment(Some(&fs)) {
        Err(Error::Unsupported(u)) => assert_eq!(u.feature, Feature::Tessellation),
        other => panic!("{other:?}"),
    }
}

/// The control stage's push range takes the first 4-aligned gap of its size no stage's range
/// covers, within `maxPushConstantsSize`: on a device of 256 bytes (`lavapipe`) and on one of 128
/// (`runtime-floor`), Vulkan's least.
#[test]
fn the_control_push_range_takes_the_first_free_gap() {
    let range = |offset, size| PushConstantRange {
        stages: vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        offset,
        size,
    };
    for (stem, bytes) in [("lavapipe", 256), ("runtime-floor", 128)] {
        let p = profile(stem);
        assert_eq!(p.limits().max_push_constants_size, bytes, "{stem}");
        let control_offset = |ranges: &[PushConstantRange]| {
            push_offset(ranges, (control::PUSH_BYTES, 4), Feature::Tessellation, &p)
        };
        assert_eq!(control_offset(&[]).expect("empty"), 0);
        assert_eq!(control_offset(&[range(0, 16)]).expect("after"), 16);
        assert_eq!(
            control_offset(&[range(0, 56), range(64, 8)]).expect("after fragment"),
            72
        );
        assert_eq!(
            control_offset(&[range(0, 40), range(64, 8)]).expect("gap"),
            40
        );
        assert_eq!(
            control_offset(&[range(0, 48), range(64, 8)]).expect("too small a gap"),
            72
        );
        // A vertex window of 64 and a fragment range to `bytes - 8` leave no 24-byte gap.
        match control_offset(&[range(0, 64), range(64, bytes - 72)]) {
            Err(Error::Unsupported(u)) => assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Limit {
                    limit: VkLimit::MaxPushConstantsSize,
                    required: u64::from(bytes) + 16,
                    available: u64::from(bytes),
                }]]),
                "{stem}"
            ),
            other => panic!("{stem}: {other:?}"),
        }
    }
}
