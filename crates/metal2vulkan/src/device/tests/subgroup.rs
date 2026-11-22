//! `subgroup::tier()` / `requirement()` and `MetalDeviceProperties`.

use super::profile::device;
use super::*;
use crate::device::requirements::VkProperty;
use crate::device::subgroup::{
    requirement, tier, OpBits, SimdClass, StageSubgroup, SubgroupModel, SubgroupRequirement,
    SubgroupSizes, Tier,
};
use crate::error::{Capability, Feature};
use crate::metal::{ArgumentBuffersTier, ReadWriteTextureTier};
use crate::pipeline::sym::Stage;

const STAGES: [Stage; 3] = [Stage::Vertex, Stage::Fragment, Stage::Kernel];

fn classes() -> Vec<(&'static str, SimdClass)> {
    vec![
        ("none", SimdClass::None),
        ("lane", SimdClass::LaneArithmetic),
        ("quad", SimdClass::Quad),
        ("ballot", SimdClass::CrossLane(OpBits::BALLOT)),
        ("shuffle", SimdClass::CrossLane(OpBits::SHUFFLE)),
        ("reduce", SimdClass::CrossLane(OpBits::ARITHMETIC)),
        ("matrix", SimdClass::Matrix),
    ]
}

fn short(t: &Result<Tier, crate::error::Unsupported>) -> String {
    match t {
        Ok(Tier::NoSubgroup) => "-".into(),
        Ok(Tier::Linear) => "L".into(),
        Ok(Tier::Direct) => "D".into(),
        Ok(Tier::Cluster32) => "C".into(),
        Ok(Tier::PartialFragment) => "P".into(),
        Ok(Tier::Emulated) => "E".into(),
        Err(u) => match u.feature {
            Feature::SimdWidth => "xW".into(),
            Feature::SubgroupOperations => "xO".into(),
            Feature::QuadOperations => "xQ".into(),
            Feature::SimdgroupMatrix => "xM".into(),
            ref other => format!("x{other:?}"),
        },
    }
}

/// The tier of every (class, stage) on every committed profile. Rows: profile; cells per stage
/// `vertex|fragment|kernel`, classes in `classes()` order. `D` Direct, `C` Cluster32, `P`
/// PartialFragment, `L` Linear, `E` Emulated, `-` no subgroup; `xW` width refusal, `xO`/`xQ`/`xM`
/// missing operations (subgroup ops, quads, matrix ops).
const GOLDEN: &str = "\
amd-780m-hawkpoint-radv: -,C,C,C,C,C,xW | -,D,D,D,D,D,D | -,L,D,D,D,D,D
amd-780m-radv: -,C,C,C,C,C,xW | -,D,D,D,D,D,D | -,L,D,D,D,D,D
amd-rdna2-apu-radv: -,C,C,C,C,C,xW | -,D,D,D,D,D,D | -,L,D,D,D,D,D
gate-lavapipe: -,xO,xO,xO,xO,xO,xM | -,P,P,P,P,P,xW | -,L,E,E,E,E,E
gate-lavapipe-x86_64: -,xO,xO,xO,xO,xO,xM | -,P,P,P,P,P,xW | -,L,E,E,E,E,E
intel-arl-anv: -,D,D,D,D,D,D | -,D,D,D,D,D,D | -,L,D,D,D,D,D
intel-irisxe-anv: -,D,D,D,D,D,D | -,D,D,D,D,D,D | -,L,D,D,D,D,D
intel-irisxe-rplp-anv: -,D,D,D,D,D,D | -,D,D,D,D,D,D | -,L,D,D,D,D,D
intel-uhd770-rpls-anv: -,D,D,D,D,D,D | -,D,D,D,D,D,D | -,L,D,D,D,D,D
lavapipe: -,xO,xO,xO,xO,xO,xM | -,P,P,P,P,P,xW | -,L,E,E,E,E,E
moltenvk-m3: -,xO,xO,xO,xO,xO,xM | -,D,D,D,D,D,D | -,L,D,D,D,D,D
nvidia-pascal-linux: -,D,D,D,D,D,D | -,D,D,D,D,D,D | -,L,D,D,D,D,D
nvidia-rtx50-linux: -,D,D,D,D,D,D | -,D,D,D,D,D,D | -,L,D,D,D,D,D
nvidia-rtx5080-laptop: -,D,D,D,D,D,D | -,D,D,D,D,D,D | -,L,D,D,D,D,D
runtime-floor: -,xO,xO,xO,xO,xO,xM | -,xO,xQ,xO,xO,xO,xM | -,L,E,E,E,E,E
";

fn table() -> String {
    let mut out = String::new();
    for (stem, _) in committed_profiles() {
        let dev = device(&stem);
        let model = &dev.profile().fields().subgroup;
        let cells: Vec<String> = STAGES
            .iter()
            .map(|&s| {
                classes()
                    .iter()
                    .map(|(_, c)| short(&tier(c, s, model)))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect();
        out.push_str(&format!("{stem}: {}\n", cells.join(" | ")));
    }
    out
}

#[test]
fn tier_golden_table() {
    let got = table();
    assert_eq!(got, GOLDEN, "tier table changed:\n{got}");
}

#[test]
fn tier_invariants_on_every_profile() {
    for (stem, _) in committed_profiles() {
        let dev = device(&stem);
        let model = &dev.profile().fields().subgroup;
        for &stage in &STAGES {
            let st = model.stage(stage);
            for (name, class) in classes() {
                let t = tier(&class, stage, model);
                // Never Cluster32 when a subgroup narrower than 32 is possible.
                if t == Ok(Tier::Cluster32) {
                    assert!(st.sizes.min >= 32, "{stem} {stage:?} {name}");
                }
                // Compute quads are tiered as cross-lane quad operations.
                if class == SimdClass::Quad && stage == Stage::Kernel {
                    assert_eq!(
                        t,
                        tier(&SimdClass::CrossLane(OpBits::QUAD), stage, model),
                        "{stem}"
                    );
                }
                // Matrices are Direct on subgroups (emulated in a kernel otherwise).
                if class == SimdClass::Matrix {
                    assert!(
                        matches!(t, Ok(Tier::Direct | Tier::Emulated) | Err(_)),
                        "{stem} {stage:?}: {t:?}"
                    );
                }
                // The emulated tier is the kernels' only, where no subgroup tier is possible,
                // and needs nothing of the subgroup.
                if t == Ok(Tier::Emulated) {
                    assert_eq!(stage, Stage::Kernel, "{stem} {name}");
                    assert_eq!(
                        requirement(Tier::Emulated, stage, model),
                        SubgroupRequirement::NONE
                    );
                }
                if stage == Stage::Kernel && class != SimdClass::None {
                    assert!(t.is_ok(), "{stem} {name}: every kernel class has a tier");
                }
                // Compute Direct/Cluster32 always require full subgroups and a multiple.
                if let Ok(t @ (Tier::Direct | Tier::Cluster32)) = t {
                    let r = requirement(t, stage, model);
                    if stage == Stage::Kernel {
                        assert!(r.full_subgroups);
                        assert!(r.local_size_multiple >= 32);
                        assert!(st.fits(r.local_size_multiple), "{stem}");
                    } else {
                        assert!(!r.full_subgroups);
                        assert_eq!(r.local_size_multiple, 1);
                    }
                    assert_eq!(r.size.is_some(), st.pinnable, "{stem} {stage:?}");
                }
            }
        }
    }
}

fn model(stage: StageSubgroup) -> SubgroupModel {
    SubgroupModel {
        vertex: stage,
        fragment: stage,
        kernel: StageSubgroup {
            full_subgroups: true,
            ..stage
        },
        quad_all_stages: true,
        maximal_reconvergence: false,
        uniform_control_flow: false,
        quad_control: false,
        max_compute_workgroup_subgroups: 64,
    }
}

fn fixed(size: u32) -> StageSubgroup {
    StageSubgroup {
        supported: true,
        ops: OpBits(0xff),
        api_size: size,
        sizes: SubgroupSizes {
            min: size,
            max: size,
        },
        pinnable: false,
        full_subgroups: false,
    }
}

#[test]
fn fixed_wave64_is_cluster32_without_a_pin() {
    let m = model(fixed(64));
    let shuffle = SimdClass::CrossLane(OpBits::SHUFFLE);
    assert_eq!(tier(&shuffle, Stage::Kernel, &m), Ok(Tier::Cluster32));
    assert_eq!(
        requirement(Tier::Cluster32, Stage::Kernel, &m),
        SubgroupRequirement {
            size: None,
            full_subgroups: true,
            local_size_multiple: 64,
        }
    );
    assert_eq!(tier(&shuffle, Stage::Fragment, &m), Ok(Tier::Cluster32));
    // Matrices need a 32-lane subgroup: emulated in a kernel, refused in fragment.
    assert_eq!(
        tier(&SimdClass::Matrix, Stage::Kernel, &m),
        Ok(Tier::Emulated)
    );
    assert!(tier(&SimdClass::Matrix, Stage::Fragment, &m).is_err());
}

#[test]
fn pinnable_ranges_prefer_32() {
    let mut st = fixed(64);
    st.pinnable = true;
    st.sizes = SubgroupSizes { min: 32, max: 64 };
    let m = model(st);
    let ballot = SimdClass::CrossLane(OpBits::BALLOT);
    assert_eq!(tier(&ballot, Stage::Kernel, &m), Ok(Tier::Direct));
    assert_eq!(
        requirement(Tier::Direct, Stage::Kernel, &m),
        SubgroupRequirement {
            size: Some(32),
            full_subgroups: true,
            local_size_multiple: 32,
        }
    );
    assert_eq!(
        requirement(Tier::Direct, Stage::Fragment, &m),
        SubgroupRequirement {
            size: Some(32),
            full_subgroups: false,
            local_size_multiple: 1,
        }
    );
    // A range above 32 pins the smallest cluster size.
    st.sizes = SubgroupSizes { min: 64, max: 128 };
    let m = model(st);
    assert_eq!(tier(&ballot, Stage::Kernel, &m), Ok(Tier::Cluster32));
    assert_eq!(
        requirement(Tier::Cluster32, Stage::Kernel, &m).size,
        Some(64)
    );
}

/// A pinned size needs `subgroupSizeControl`, the stage in `requiredSubgroupSizeStages`, and the
/// size inside the device's range; full subgroups need their stage flag; a requirement that pins
/// nothing and needs no full subgroups needs nothing. A limit already named keeps the stricter
/// value.
#[test]
fn a_subgroup_requirement_states_its_device_facts() {
    use crate::device::requirements::{Requirements, StageCreateFlag, VkLimit};
    let facts = |r: SubgroupRequirement, stage| {
        let mut out = Requirements::default();
        r.add_device_facts(stage, &mut out);
        out
    };
    let pinned = SubgroupRequirement {
        size: Some(32),
        full_subgroups: true,
        local_size_multiple: 32,
    };
    let r = facts(pinned, Stage::Kernel);
    assert_eq!(r.features, [VkFeature::SubgroupSizeControl].into());
    assert_eq!(
        r.properties,
        [VkProperty::RequiredSubgroupSizeStage(Stage::Kernel)].into()
    );
    assert_eq!(
        r.limits,
        [
            (VkLimit::MinSubgroupSize, 32),
            (VkLimit::MaxSubgroupSize, 32)
        ]
        .into()
    );
    assert_eq!(
        r.stage_flags,
        [StageCreateFlag::RequireFullSubgroups].into()
    );
    // A fragment pin names its own stage and no stage flag.
    let fragment = SubgroupRequirement {
        full_subgroups: false,
        local_size_multiple: 1,
        ..pinned
    };
    let r = facts(fragment, Stage::Fragment);
    assert_eq!(
        r.properties,
        [VkProperty::RequiredSubgroupSizeStage(Stage::Fragment)].into()
    );
    assert!(r.stage_flags.is_empty());
    // Full subgroups at the API size: the stage flag alone.
    let unpinned = SubgroupRequirement {
        size: None,
        ..pinned
    };
    let r = facts(unpinned, Stage::Kernel);
    assert_eq!(
        r.stage_flags,
        [StageCreateFlag::RequireFullSubgroups].into()
    );
    assert!(r.features.is_empty() && r.properties.is_empty() && r.limits.is_empty());
    assert_eq!(
        facts(SubgroupRequirement::NONE, Stage::Kernel),
        Requirements::default()
    );
    // Stricter values already named stay.
    let mut r = Requirements::default();
    r.limits.insert(VkLimit::MinSubgroupSize, 16);
    r.limits.insert(VkLimit::MaxSubgroupSize, 64);
    pinned.add_device_facts(Stage::Kernel, &mut r);
    assert_eq!(r.limits[&VkLimit::MinSubgroupSize], 16);
    assert_eq!(r.limits[&VkLimit::MaxSubgroupSize], 64);
}

/// `minSubgroupSize` and `maxSubgroupSize` are the device's: the range of any stage that can pin a
/// size, whichever stages can; a device that can pin none has its kernel's API size.
#[test]
fn the_device_size_range_is_any_pinnable_stages() {
    let mut pinnable = fixed(32);
    pinnable.pinnable = true;
    pinnable.sizes = SubgroupSizes { min: 8, max: 64 };
    let mut m = model(fixed(32));
    assert_eq!(m.pinnable_sizes(), SubgroupSizes { min: 32, max: 32 });
    m.fragment = pinnable;
    assert_eq!(m.pinnable_sizes(), SubgroupSizes { min: 8, max: 64 });
    m.fragment = fixed(32);
    m.kernel = pinnable;
    assert_eq!(m.pinnable_sizes(), SubgroupSizes { min: 8, max: 64 });
}

#[test]
fn narrow_subgroups_partial_in_graphics_emulated_in_compute() {
    let m = model(fixed(16));
    let vote = SimdClass::CrossLane(OpBits::VOTE);
    assert_eq!(tier(&vote, Stage::Fragment, &m), Ok(Tier::PartialFragment));
    assert_eq!(
        requirement(Tier::PartialFragment, Stage::Fragment, &m),
        SubgroupRequirement::NONE
    );
    // A kernel cannot run its 32-lane simdgroups on 16-lane subgroups: they are emulated
    // through workgroup memory, needing nothing of the subgroup.
    assert_eq!(tier(&vote, Stage::Kernel, &m), Ok(Tier::Emulated));
    assert_eq!(
        requirement(Tier::Emulated, Stage::Kernel, &m),
        SubgroupRequirement::NONE
    );
    assert_eq!(
        tier(&SimdClass::LaneArithmetic, Stage::Kernel, &m),
        Ok(Tier::Linear)
    );
    // Quads need 4 | S.
    let m = model(fixed(2));
    assert_eq!(
        tier(&SimdClass::Quad, Stage::Fragment, &m)
            .unwrap_err()
            .feature,
        Feature::QuadOperations
    );
}

#[test]
fn missing_stage_or_operations_are_named() {
    let mut st = fixed(32);
    st.ops = OpBits::BASIC.union(OpBits::VOTE);
    let mut m = model(st);
    let err = tier(&SimdClass::CrossLane(OpBits::SHUFFLE), Stage::Fragment, &m).unwrap_err();
    assert_eq!(err.feature, Feature::SubgroupOperations);
    assert_eq!(
        err.needs,
        crate::error::Needs::AnyOf(vec![vec![Capability::Property(
            VkProperty::SubgroupOperation(crate::device::requirements::SubgroupOp::Shuffle)
        )]])
    );
    m.vertex.supported = false;
    assert!(tier(&SimdClass::LaneArithmetic, Stage::Vertex, &m).is_err());
    // Vertex quads need quad operations in all stages.
    let mut m = model(fixed(32));
    m.quad_all_stages = false;
    assert_eq!(
        tier(&SimdClass::Quad, Stage::Vertex, &m)
            .unwrap_err()
            .feature,
        Feature::QuadOperations
    );
    assert_eq!(
        tier(&SimdClass::Quad, Stage::Fragment, &m),
        Ok(Tier::Direct)
    );
}

/// A class the device has (its compute stage's, which Vulkan always lists) but a stage lacks, as
/// a stage measured on a driver that does not list it lacks the classes measured nowhere there,
/// is refused naming the stage, which listing would give; a class the device lacks names the
/// class.
#[test]
fn a_class_measured_nowhere_in_a_stage_names_the_stage() {
    use crate::device::requirements::SubgroupOp;
    use crate::error::Needs;
    let mut m = model(fixed(32));
    m.vertex.ops = OpBits::BASIC.union(OpBits::ARITHMETIC);
    m.kernel.ops = m.vertex.ops.union(OpBits::SHUFFLE);
    let needs = |class: SimdClass| tier(&class, Stage::Vertex, &m).unwrap_err().needs;
    assert_eq!(
        needs(SimdClass::CrossLane(OpBits::SHUFFLE)),
        Needs::AnyOf(vec![vec![Capability::Property(VkProperty::SubgroupStage(
            Stage::Vertex
        ))]])
    );
    assert_eq!(
        needs(SimdClass::CrossLane(OpBits::SHUFFLE_RELATIVE)),
        Needs::AnyOf(vec![vec![Capability::Property(
            VkProperty::SubgroupOperation(SubgroupOp::ShuffleRelative)
        )]])
    );
    assert_eq!(
        tier(&SimdClass::CrossLane(OpBits::ARITHMETIC), Stage::Vertex, &m),
        Ok(Tier::Direct)
    );
}

/// No device answers more than an Apple GPU: each answer a vendor's limit could raise past Apple's
/// is the lower of the two. Every profile below has at least an Apple GPU's limits, and the
/// profiles of other vendors' GPUs more of some, so each answers exactly the Apple GPU's: 1024
/// threads, 32 KiB of threadgroup memory, 16384 texels a side, and no sample count above 4.
#[test]
fn no_profile_answers_more_than_an_apple_gpu() {
    let apple = vk::SampleCountFlags::COUNT_1
        | vk::SampleCountFlags::COUNT_2
        | vk::SampleCountFlags::COUNT_4;
    // (profile, its own [threads, threadgroup memory, most samples, 2D texture size])
    for (stem, own) in [
        ("nvidia-pascal-linux", [1536, 49152, 8, 32768]),
        ("nvidia-rtx50-linux", [1024, 49152, 8, 32768]),
        ("nvidia-rtx5080-laptop", [1024, 49152, 8, 32768]),
        ("amd-rdna2-apu-radv", [1024, 65536, 8, 16384]),
        ("amd-780m-radv", [1024, 65536, 8, 16384]),
        ("intel-arl-anv", [1024, 49152, 16, 16384]),
        ("intel-irisxe-anv", [1024, 49152, 16, 16384]),
        ("intel-irisxe-rplp-anv", [1024, 49152, 16, 16384]),
        ("intel-uhd770-rpls-anv", [1024, 49152, 16, 16384]),
        ("lavapipe", [1024, 32768, 8, 16384]),
        ("gate-lavapipe-x86_64", [1024, 32768, 8, 16384]),
        ("moltenvk-m3", [1024, 32768, 4, 16384]),
    ] {
        let dev = device(stem);
        let m = dev.metal_properties();
        let l = &dev.profile().fields().limits;
        let both = l.framebuffer_color_sample_counts & l.framebuffer_depth_sample_counts;
        let most = 1 << (31 - both.as_raw().leading_zeros());
        let limits = [
            l.max_compute_work_group_invocations,
            l.max_compute_shared_memory_size,
            most,
            l.max_image_dimension_2d,
        ];
        assert_eq!(limits, own, "{stem}");
        assert_eq!(m.max_total_threads_per_threadgroup, 1024, "{stem}");
        assert!(
            m.max_threads_per_threadgroup
                .iter()
                .all(|&axis| axis <= 1024),
            "{stem}"
        );
        assert_eq!(m.max_threadgroup_memory_length, 32 * 1024, "{stem}");
        assert_eq!(m.raster_sample_counts, both & apple, "{stem}");
        assert_eq!(m.max_texture_dimension_2d, 16_384, "{stem}");
    }
    let pascal = device("nvidia-pascal-linux");
    assert_eq!(
        pascal.metal_properties().max_threads_per_threadgroup,
        [1024, 1024, 1024]
    );
    let lavapipe = device("lavapipe");
    assert_eq!(
        lavapipe.metal_properties().raster_sample_counts,
        vk::SampleCountFlags::COUNT_1 | vk::SampleCountFlags::COUNT_4
    );
}

#[test]
fn metal_properties_report_d3_d4_values() {
    for (stem, _) in committed_profiles() {
        let dev = device(&stem);
        let m = dev.metal_properties();
        let l = &dev.profile().fields().limits;
        assert_eq!(m.thread_execution_width, 32, "{stem}");
        // Each the lower of the device's limit and an Apple GPU's answer.
        let threads = l.max_compute_work_group_invocations.min(1024);
        assert_eq!(m.max_total_threads_per_threadgroup, threads, "{stem}");
        assert_eq!(m.max_threads_per_threadgroup, [threads; 3], "{stem}");
        assert_eq!(
            m.max_threadgroup_memory_length,
            l.max_compute_shared_memory_size.min(32 * 1024),
            "{stem}"
        );
        assert_eq!(
            m.raster_sample_counts,
            l.framebuffer_color_sample_counts
                & l.framebuffer_depth_sample_counts
                & (vk::SampleCountFlags::COUNT_1
                    | vk::SampleCountFlags::COUNT_2
                    | vk::SampleCountFlags::COUNT_4),
            "{stem}"
        );
        assert_eq!(
            m.max_texture_dimension_2d,
            l.max_image_dimension_2d.min(16_384),
            "{stem}"
        );
        assert!(m.max_buffer_length > 0, "{stem}");
    }
    let floor = device("runtime-floor");
    let m = floor.metal_properties();
    assert_eq!(m.max_threads_per_threadgroup, [1024, 1024, 1024]);
    // Device addresses and update-after-bind sampled and storage images: tier 2.
    assert_eq!(m.argument_buffers_tier, ArgumentBuffersTier::Tier2);
    // Tier 2's 8- and 16-bit single-channel storage formats come with
    // shaderStorageImageExtendedFormats, which the floor requires.
    assert_eq!(m.read_write_texture_tier, ReadWriteTextureTier::Tier2);
    // Typed storage images are enough: intel-irisxe-anv reads storage images only with a
    // declared format and still has every tier-2 format.
    let irisxe = device("intel-irisxe-anv");
    assert!(!irisxe.profile().fields().images.storage_read_without_format);
    assert_eq!(
        irisxe.metal_properties().read_write_texture_tier,
        ReadWriteTextureTier::Tier2
    );
    // Vulkan 1.3's maxBufferSize and maxMemoryAllocationSize guarantees.
    assert_eq!(m.max_buffer_length, 1 << 30);
    // Without subgroup operations a kernel's simdgroups, matrices included, are emulated; a
    // fragment's are not.
    assert!(m.simd.kernel.matrix);
    assert!(!m.simd.fragment.matrix);
    let has = |f: &Feature| m.unsupported_features.iter().any(|u| u.feature == *f);
    // 64-bit integers are two words without `shaderInt64`, on every pipeline.
    assert!(!has(&Feature::Int64Arithmetic));
    assert!(!has(&Feature::ArgumentBufferPointers));
    assert!(has(&Feature::RasterOrderGroups));
    // Neither rasterization-order access nor the pixel interlock: framebuffer fetch reads input
    // attachments ordered between draws, a note, not a refusal.
    let fragment = floor.profile().fields().fragment;
    assert!(!fragment.rasterization_order_color_access && !fragment.pixel_interlock);
    assert!(!has(&Feature::FramebufferFetch));
    let nv = device("nvidia-rtx50-linux");
    let m = nv.metal_properties();
    assert_eq!(m.read_write_texture_tier, ReadWriteTextureTier::Tier2);
    assert_eq!(m.argument_buffers_tier, ArgumentBuffersTier::Tier2);
    assert!(m.simd.kernel.matrix);
    assert!(m.simd.kernel.cross_lane.contains(OpBits::SHUFFLE));
    assert!(!m
        .unsupported_features
        .iter()
        .any(|u| u.feature == Feature::Int64Arithmetic));
}
