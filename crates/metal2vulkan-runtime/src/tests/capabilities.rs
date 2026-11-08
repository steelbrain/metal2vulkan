//! `capabilities`: Metal's device queries over every committed device profile
//! (`validation/profiles/`, read-only), each checked against its source.

use std::path::PathBuf;

use crate::capabilities::{
    amplification_count_supported, linear_alignment_rule, rate_map_layers_supported,
    sample_count_supported, working_set_buffer_bound, DeviceIdentity, RATE_MAP_LAYERS,
};
use crate::{
    BufferTextureProblem, DeviceKind, DeviceProperties, Error, Invalid, MacClass, Problem, Query,
    Size,
};
use metal2vulkan::bind::{RasterizationRateMap, RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::device::{
    Device, DeviceDescription, DevicePolicy, FormatUsage, VkExtension, VkFeature,
};
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::heap_sampler_capacity;
use metal2vulkan::vk::{ApiVersion, Format, SampleCountFlags};
use std::collections::BTreeMap;

/// Every committed profile's description: (file stem, description), sorted by stem.
pub(super) fn committed_descriptions() -> Vec<(String, DeviceDescription)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, DeviceDescription)> = std::fs::read_dir(&dir)
        .expect("validation/profiles is readable")
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).expect("profile is readable");
            let desc = DeviceDescription::from_khronos_json(&text, None)
                .unwrap_or_else(|e| panic!("{stem}: {e}"));
            (stem, desc)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The value the profile `stem` states under the member `key`, the first in its file; `None`
/// where it states none.
fn profile_value(stem: &str, key: &str) -> Option<serde_json::Value> {
    fn find<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
        match value {
            serde_json::Value::Object(map) => map
                .get(key)
                .or_else(|| map.values().find_map(|v| find(v, key))),
            serde_json::Value::Array(list) => list.iter().find_map(|v| find(v, key)),
            _ => None,
        }
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{stem}: {e}"));
    let json: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{stem}: {e}"));
    find(&json, key).cloned()
}

/// The sample counts the profile `stem` states under the `VkPhysicalDeviceLimits` member `key`;
/// none where it states none (the roadmap profiles), as the description reads its own counts.
pub(super) fn profile_counts(stem: &str, key: &str) -> SampleCountFlags {
    let Some(list) = profile_value(stem, key) else {
        return SampleCountFlags::empty();
    };
    list.as_array()
        .unwrap_or_else(|| panic!("{stem}: {key} is no list"))
        .iter()
        .filter_map(|bit| SampleCountFlags::from_name(bit.as_str().unwrap()))
        .fold(SampleCountFlags::empty(), |all, bit| all | bit)
}

/// The identity of the device the profile `stem` states, as `device::properties::query` reads
/// it of a physical device.
fn identity(stem: &str) -> DeviceIdentity {
    DeviceIdentity {
        name: "test device".to_owned(),
        kind: DeviceKind::Discrete,
        vendor_id: 0x1002,
        device_id: 0x15bf,
        device_uuid: [7; 16],
        pci_bus: None,
        drm_render: None,
        api_version: ApiVersion::V1_3,
        max_viewports: 16,
        max_sampler_allocation_count: sampler_allocation_count(stem),
        gpu_core_count: None,
        sampled_sample_counts: profile_counts(stem, "sampledImageColorSampleCounts")
            & profile_counts(stem, "sampledImageDepthSampleCounts"),
        device_local_heap: None,
    }
}

/// The profile `stem`'s `maxSamplerAllocationCount`, or Vulkan's least, 4000, where it states none
/// (the roadmap profile).
fn sampler_allocation_count(stem: &str) -> u32 {
    profile_value(stem, "maxSamplerAllocationCount").map_or(4000, |count| {
        let count = count.as_u64().unwrap_or_else(|| panic!("{stem}: {count}"));
        u32::try_from(count).unwrap_or_else(|e| panic!("{stem}: {e}"))
    })
}

/// The properties of a device created from `desc` as the runtime creates one: the recommended
/// enables for the untrusted policy, the description restricted to them, and the compile device
/// built from that.
pub(super) fn created(
    stem: &str,
    desc: &DeviceDescription,
) -> (DeviceDescription, DeviceProperties) {
    let policy = DevicePolicy::UNTRUSTED;
    let enables = desc.recommended_enables(&policy);
    let enabled = desc
        .clone()
        .with_enabled(&enables)
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
    let compile = Device::new(enabled.clone(), policy).unwrap_or_else(|e| panic!("{stem}: {e}"));
    let properties = DeviceProperties::new(identity(stem), enabled.clone(), &compile, None);
    (enabled, properties)
}

/// `hasUnifiedMemory` is the device kind's: an integrated GPU's memory and a CPU
/// implementation's are the host's; a discrete GPU answers no, as Metal does on one, and so do a
/// virtual GPU and any other kind.
#[test]
fn unified_memory_is_the_device_kinds() {
    let (stem, desc) = &committed_descriptions()[0];
    let policy = DevicePolicy::UNTRUSTED;
    let enabled = desc
        .clone()
        .with_enabled(&desc.recommended_enables(&policy))
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
    let compile = Device::new(enabled.clone(), policy).unwrap_or_else(|e| panic!("{stem}: {e}"));
    for (kind, unified) in [
        (DeviceKind::Discrete, false),
        (DeviceKind::Integrated, true),
        (DeviceKind::Virtual, false),
        (DeviceKind::Cpu, true),
        (DeviceKind::Other, false),
    ] {
        let identity = DeviceIdentity {
            kind,
            ..identity(stem)
        };
        let p = DeviceProperties::new(identity, enabled.clone(), &compile, None);
        assert_eq!(p.has_unified_memory, unified, "{kind:?}");
    }
}

/// `lossless_compression` is the Mac class the device presents: the one the consumer declares,
/// whatever its GPU, else its GPU's, which is an Apple silicon Mac's on an Apple GPU alone. An
/// integrated GPU that is not Apple's, of unified memory as an Intel Mac's is, presents a Mac
/// keeping no texture compressed by default, as the consumer's probe of an Intel UHD 630 reads.
#[test]
fn lossless_compression_is_the_presented_mac_class() {
    let (stem, desc) = &committed_descriptions()[0];
    let policy = DevicePolicy::UNTRUSTED;
    let enabled = desc
        .clone()
        .with_enabled(&desc.recommended_enables(&policy))
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
    let compile = Device::new(enabled.clone(), policy).unwrap_or_else(|e| panic!("{stem}: {e}"));
    // Apple, Intel, AMD, NVIDIA, and Mesa's for a CPU implementation (lavapipe).
    let gpus = [
        (0x106b, DeviceKind::Integrated, MacClass::AppleSilicon),
        (0x8086, DeviceKind::Integrated, MacClass::IntelOrAmd),
        (0x1002, DeviceKind::Integrated, MacClass::IntelOrAmd),
        (0x1002, DeviceKind::Discrete, MacClass::IntelOrAmd),
        (0x10de, DeviceKind::Discrete, MacClass::IntelOrAmd),
        (0x10005, DeviceKind::Cpu, MacClass::IntelOrAmd),
    ];
    for (vendor_id, kind, own) in gpus {
        assert_eq!(MacClass::of_vendor(vendor_id), own, "{vendor_id:#x}");
        for declared in [
            None,
            Some(MacClass::AppleSilicon),
            Some(MacClass::IntelOrAmd),
        ] {
            let identity = DeviceIdentity {
                kind,
                vendor_id,
                ..identity(stem)
            };
            let p = DeviceProperties::new(identity, enabled.clone(), &compile, declared);
            let class = declared.unwrap_or(own);
            assert_eq!(
                p.lossless_compression,
                class == MacClass::AppleSilicon,
                "{vendor_id:#x} {kind:?} declared {declared:?}"
            );
            assert_eq!(p.has_unified_memory, kind.has_unified_memory(), "{kind:?}");
        }
    }
}

/// Metal's rule for `maxBufferLength`, three quarters of the working set to 16 KiB, gives what
/// Metal answered: an AMD Radeon Pro 5300M (4 GiB, macOS 26.7.1) of `recommendedMaxWorkingSetSize`
/// 4,278,190,080 and an Apple M3 Max (64 GiB, macOS 27.0.1) of 55,662,788,608. A device's answer
/// is the compile layer's bound by its largest device-local heap's (a 16 GiB heap's 12 GiB, a heap
/// of the limit's size three quarters of it), and the compile layer's where the bound lies above
/// it or no heap was read.
#[test]
fn max_buffer_length_is_three_quarters_of_the_device_local_heap_at_most() {
    assert_eq!(working_set_buffer_bound(4_278_190_080), 3_208_642_560);
    assert_eq!(working_set_buffer_bound(55_662_788_608), 41_747_087_360);
    let (stem, desc) = &committed_descriptions()[0];
    let policy = DevicePolicy::UNTRUSTED;
    let enabled = desc
        .clone()
        .with_enabled(&desc.recommended_enables(&policy))
        .unwrap_or_else(|e| panic!("{stem}: {e}"));
    let compile = Device::new(enabled.clone(), policy).unwrap_or_else(|e| panic!("{stem}: {e}"));
    let limit = compile.metal_properties().max_buffer_length;
    for heap in [
        None,
        Some(16 << 30),
        Some(64 << 30),
        Some(limit),
        Some(u64::MAX),
    ] {
        let identity = DeviceIdentity {
            device_local_heap: heap,
            ..identity(stem)
        };
        let p = DeviceProperties::new(identity, enabled.clone(), &compile, None);
        let want = heap.map_or(limit, |heap| {
            limit.min(heap / 4 * 3 / (16 << 10) * (16 << 10))
        });
        assert_eq!(p.max_buffer_length, want, "{stem}: heap {heap:?}");
        assert!(p.max_buffer_length <= limit, "{stem}: heap {heap:?}");
    }
}

/// Every answer equals its source on every committed profile: the Vulkan limit, feature or
/// format the device was created with, or the compile layer's answer for it, an Apple GPU's at
/// most.
#[test]
fn every_committed_profile_answers_from_its_sources() {
    let profiles = committed_descriptions();
    assert!(profiles.len() >= 18, "{} profiles", profiles.len());
    for (stem, desc) in &profiles {
        let (enabled, p) = created(stem, desc);
        let l = &enabled.limits;
        let has = |f: VkFeature| enabled.has(f);
        let invocations = l.max_compute_work_group_invocations.min(1024);
        let axis = |i: usize| u64::from(l.max_compute_work_group_size[i].min(invocations));
        assert_eq!(
            p.max_threads_per_threadgroup,
            Size::new(axis(0), axis(1), axis(2)),
            "{stem}"
        );
        assert_eq!(
            p.max_threadgroup_memory_length,
            u64::from(l.max_compute_shared_memory_size.min(32 * 1024)),
            "{stem}"
        );
        let addressed = has(VkFeature::BufferDeviceAddress) && has(VkFeature::ShaderInt64);
        let a = &enabled.allocation;
        let max_buffer_length = if !addressed {
            u64::from(l.max_storage_buffer_range)
        } else if a.max_buffer_size == 0 {
            a.max_memory_allocation_size
        } else {
            a.max_buffer_size.min(a.max_memory_allocation_size)
        };
        assert_eq!(p.max_buffer_length, max_buffer_length, "{stem}");
        let formats = &enabled.formats;
        assert_eq!(
            p.supports_32bit_float_filtering,
            formats.supports(Format::R32_SFLOAT, FormatUsage::SampledImageFilterLinear),
            "{stem}"
        );
        let answers = p.device_answers();
        assert_eq!(
            answers.raster_order_groups,
            has(VkFeature::FragmentShaderPixelInterlock),
            "{stem}"
        );
        assert_eq!(
            answers.texture_sample_counts,
            l.framebuffer_color_sample_counts
                & l.framebuffer_depth_sample_counts
                & profile_counts(stem, "sampledImageColorSampleCounts")
                & profile_counts(stem, "sampledImageDepthSampleCounts")
                & SampleCountFlags::from_raw(0b111),
            "{stem}"
        );
        let amplification = if has(VkFeature::Multiview) { 8 } else { 1 };
        assert_eq!(
            answers.max_vertex_amplification_count, amplification,
            "{stem}"
        );
        let located = |count| {
            let count = SampleCountFlags::from_raw(count);
            !answers.texture_sample_counts.contains(count)
                || l.sample_location_sample_counts.contains(count)
        };
        let programmable = enabled
            .extensions
            .contains(&VkExtension::ExtSampleLocations)
            && l.variable_sample_locations
            && [2, 4, 8, 16, 32, 64].into_iter().all(located);
        assert_eq!(
            p.are_programmable_sample_positions_supported(),
            Ok(programmable),
            "{stem}"
        );
        // The interpolation functions: `sampleRateShading`, and on a portability-subset device
        // its member for them.
        let portability = enabled
            .extensions
            .contains(&VkExtension::KhrPortabilitySubset);
        let functions = has(VkFeature::SampleRateShading)
            && (!portability || has(VkFeature::ShaderSampleRateInterpolationFunctions));
        assert_eq!(
            p.supports_pull_model_interpolation(),
            Ok(functions),
            "{stem}"
        );
        assert_eq!(answers.pull_model_interpolation, functions, "{stem}");
        assert_eq!(p.description, enabled, "{stem}");
        assert_eq!(p.max_viewports, 16, "{stem}");
        assert_eq!(p.name, "test device", "{stem}");
        assert_eq!(p.device_uuid, [7; 16], "{stem}");
    }
}

/// The compile layer agrees: where the pixel interlock is enabled, no fragment function's raster
/// order groups are refused up front, so answering them supported claims nothing it refuses.
#[test]
fn raster_order_groups_agree_with_the_compile_layer() {
    for (stem, desc) in &committed_descriptions() {
        let policy = DevicePolicy::UNTRUSTED;
        let enabled = desc
            .clone()
            .with_enabled(&desc.recommended_enables(&policy))
            .unwrap();
        let compile = Device::new(enabled.clone(), policy).unwrap();
        let refused = compile
            .metal_properties()
            .unsupported_features
            .iter()
            .any(|u: &Unsupported| u.feature == Feature::RasterOrderGroups);
        let p = DeviceProperties::new(identity(stem), enabled, &compile, None);
        if p.device_answers().raster_order_groups {
            assert!(!refused, "{stem}");
        }
    }
}

/// The venues' own devices, pinned from their profiles: the answers a Metal program reads on each.
#[test]
fn the_venue_devices_answer_as_pinned() {
    // (stem, threads per threadgroup, threadgroup memory, raster order groups, sample counts,
    // amplification count: Metal's 8 wherever multiview is). The threadgroup memory and sample
    // counts are an Apple GPU's at most: RADV's 65536 bytes and ANV's and NVIDIA's 49152 answer
    // 32768, and their 8 and 16 samples (lavapipe's 8) no.
    let pinned: [(&str, [u64; 3], u64, bool, &[u64], u64); 7] = [
        (
            "moltenvk-m3",
            [1024, 1024, 1024],
            32768,
            true,
            &[1, 2, 4],
            8,
        ),
        ("gate-lavapipe", [1024, 1024, 1024], 32768, true, &[1, 4], 8),
        (
            "gate-lavapipe-x86_64",
            [1024, 1024, 1024],
            32768,
            true,
            &[1, 4],
            8,
        ),
        (
            "amd-780m-radv",
            [1024, 1024, 1024],
            32768,
            true,
            &[1, 2, 4],
            8,
        ),
        (
            "intel-irisxe-rplp-anv",
            [1024, 1024, 1024],
            32768,
            true,
            &[1, 2, 4],
            8,
        ),
        (
            "intel-uhd770-rpls-anv",
            [1024, 1024, 1024],
            32768,
            true,
            &[1, 2, 4],
            8,
        ),
        (
            "nvidia-rtx5080-laptop",
            [1024, 1024, 64],
            32768,
            true,
            &[1, 2, 4],
            8,
        ),
    ];
    let profiles = committed_descriptions();
    for (stem, threads, memory, rog, counts, amplification) in pinned {
        let (_, desc) = profiles.iter().find(|(s, _)| s == stem).unwrap();
        let (_, p) = created(stem, desc);
        let [w, h, d] = threads;
        assert_eq!(p.max_threads_per_threadgroup, Size::new(w, h, d), "{stem}");
        assert_eq!(p.max_threadgroup_memory_length, memory, "{stem}");
        let answers = p.device_answers();
        assert_eq!(answers.raster_order_groups, rog, "{stem}");
        assert_eq!(
            p.is_depth24_stencil8_pixel_format_supported(),
            Ok(false),
            "{stem}"
        );
        let supported: Vec<u64> = (0..=128)
            .filter(|&n| sample_count_supported(answers.texture_sample_counts, n))
            .collect();
        assert_eq!(supported, counts, "{stem}");
        assert_eq!(
            answers.max_vertex_amplification_count, amplification,
            "{stem}"
        );
        // Every venue takes custom sample positions at every count it renders, and reads an
        // interpolant every way (as the M3 Max, a Radeon Pro 5300M and a UHD 630 answer).
        assert_eq!(
            p.are_programmable_sample_positions_supported(),
            Ok(true),
            "{stem}"
        );
        assert_eq!(p.supports_pull_model_interpolation(), Ok(true), "{stem}");
    }
}

/// `maxArgumentBufferSamplerCount` is one fewer than the smaller of a set-1 sampler array's
/// elements (`heap_sampler_capacity` of the device's profile) and the device's
/// `maxSamplerAllocationCount`, on every committed profile. Pinned: the handle table's 65536
/// entries bound it on MoltenVK (whose driver takes 500000 samplers, as Metal's answer on the M3
/// Max is 500000), RADV and ANV (65536 samplers); lavapipe's 32768 samplers and NVIDIA's 4000 bound
/// it there; the desktop baseline's 16 update-after-bind samplers a stage bound it there; and on
/// the floor, which has no descriptor indexing, no function reads a set-1 array.
#[test]
fn argument_buffer_samplers_are_one_fewer_than_the_array_and_the_sampler_limit() {
    let pinned = [
        ("moltenvk-m3", 65535),
        ("amd-780m-radv", 65535),
        ("intel-irisxe-rplp-anv", 65535),
        ("intel-uhd770-rpls-anv", 65535),
        ("lavapipe", 32767),
        ("gate-lavapipe", 32767),
        ("gate-lavapipe-x86_64", 32767),
        ("nvidia-rtx5080-laptop", 3999),
        ("lunarg-desktop-baseline-2023", 15),
        ("vk12-floor", 0),
    ];
    let profiles = committed_descriptions();
    let mut answers = BTreeMap::new();
    for (stem, desc) in &profiles {
        let (enabled, p) = created(stem, desc);
        let compile = Device::new(enabled, DevicePolicy::UNTRUSTED).unwrap();
        let elements = heap_sampler_capacity(compile.profile());
        let limit = sampler_allocation_count(stem);
        assert_eq!(
            p.max_argument_buffer_sampler_count,
            u64::from(elements.min(limit).saturating_sub(1)),
            "{stem}: {elements} elements, {limit} samplers"
        );
        answers.insert(stem.as_str(), p.max_argument_buffer_sampler_count);
    }
    for (stem, answer) in pinned {
        assert_eq!(answers.get(stem), Some(&answer), "{stem}: {answers:?}");
    }
}

/// A sample count is supported only when it is a Vulkan sample count the flags hold.
#[test]
fn sample_counts_are_the_flags_bits() {
    let flags = SampleCountFlags::from_raw(0b0111);
    for (count, supported) in [
        (0, false),
        (1, true),
        (2, true),
        (3, false),
        (4, true),
        (8, false),
        (64, false),
        (1 << 33, false),
        (u64::MAX, false),
    ] {
        assert_eq!(sample_count_supported(flags, count), supported, "{count}");
    }
}

/// An amplification count is supported from 1 to the maximum.
#[test]
fn amplification_counts_run_from_one_to_the_maximum() {
    // Metal answers yes for 0.
    assert!(amplification_count_supported(6, 0));
    assert!(amplification_count_supported(6, 1));
    assert!(amplification_count_supported(6, 6));
    assert!(!amplification_count_supported(6, 7));
    assert!(amplification_count_supported(1, 1));
    assert!(!amplification_count_supported(1, 2));
}

/// The rate-map layer counts are the ones the compile layer builds maps of: up to
/// [`RATE_MAP_LAYERS`], and no more.
#[test]
fn rate_map_layers_match_the_compile_layer() {
    let layer = || RateMapLayerDescriptor::new(vec![1.0], vec![1.0]);
    let map = |layers: u64| {
        let mut d = RateMapDescriptor::new([64, 64]);
        d.layers = (0..layers).map(|_| layer()).collect();
        RasterizationRateMap::new(&d)
    };
    for layers in 1..=RATE_MAP_LAYERS {
        assert!(rate_map_layers_supported(layers), "{layers}");
        assert!(map(layers).is_ok(), "{layers}");
    }
    assert!(!rate_map_layers_supported(0));
    assert!(!rate_map_layers_supported(RATE_MAP_LAYERS + 1));
    assert!(map(RATE_MAP_LAYERS + 1).is_err());
}

/// Every query's runtime piece is built: none has a gap, and each method answers the device's
/// value.
#[test]
fn every_query_answers_the_device() {
    let (_, desc) = committed_descriptions()
        .into_iter()
        .find(|(s, _)| s == "moltenvk-m3")
        .unwrap();
    let (_, p) = created("moltenvk-m3", &desc);
    let answer = |q: Query| match q {
        Query::RasterOrderGroups => {
            assert_eq!(p.are_raster_order_groups_supported(), Ok(true));
        }
        Query::Depth24Stencil8 => {
            assert_eq!(p.is_depth24_stencil8_pixel_format_supported(), Ok(false));
        }
        Query::ProgrammableSamplePositions => {
            assert_eq!(p.are_programmable_sample_positions_supported(), Ok(true));
        }
        Query::TextureSampleCount => {
            assert_eq!(p.supports_texture_sample_count(4), Ok(true));
            assert_eq!(p.supports_texture_sample_count(8), Ok(false));
        }
        Query::VertexAmplificationCount => {
            assert_eq!(p.supports_vertex_amplification_count(2), Ok(true));
            // Metal answers yes for 0.
            assert_eq!(p.supports_vertex_amplification_count(0), Ok(true));
            assert_eq!(p.supports_vertex_amplification_count(9), Ok(false));
        }
        Query::RasterizationRateMapLayerCount => {
            assert_eq!(
                p.supports_rasterization_rate_map_with_layer_count(2),
                Ok(true)
            );
            assert_eq!(
                p.supports_rasterization_rate_map_with_layer_count(3),
                Ok(false)
            );
        }
        Query::MinimumLinearTextureAlignment => {
            // The device's answer is a probe of its images (`texture/linear_alignment.rs`), at
            // least Metal's rule's.
            let answer = linear_alignment_rule(PixelFormat::BGRA8Unorm);
            assert_eq!(answer, Ok(16));
        }
        Query::SamplerReductionMode => {
            assert_eq!(p.supports_sampler_reduction_mode(), Ok(false));
        }
        Query::PullModelInterpolation => {
            assert_eq!(p.supports_pull_model_interpolation(), Ok(true));
        }
    };
    for q in Query::ALL {
        assert_eq!(q.gap(), None, "{q:?}");
        answer(q);
        assert!(
            q.selector().starts_with(|c: char| c.is_ascii_lowercase()),
            "{q:?}"
        );
    }
}

/// `minimumLinearTextureAlignmentForPixelFormat:` asserts on a format no buffer can back, which
/// the runtime refuses as `Invalid`, before it asks the device anything.
#[test]
fn the_linear_alignment_of_a_compressed_or_depth_format_is_refused() {
    for pf in [
        PixelFormat::BC1RGBA,
        PixelFormat::Depth32Float,
        PixelFormat::GBGR422,
    ] {
        assert_eq!(
            linear_alignment_rule(pf),
            Err(Error::Invalid(Invalid::new(
                "minimum_linear_texture_alignment_for_pixel_format",
                Problem::BufferTexture(BufferTextureProblem::NotLinear { pixel_format: pf }),
            ))),
            "{pf:?}"
        );
    }
}

/// `supportsPullModelInterpolation` answers no on a device without `sampleRateShading`, the Vulkan
/// 1.2 floor's, where the compile layer refuses every interpolation function but
/// `interpolate_at_center`; and on MoltenVK only with its portability member for them.
#[test]
fn pull_model_interpolation_needs_the_interpolation_functions() {
    let profiles = committed_descriptions();
    let (_, floor) = profiles.iter().find(|(s, _)| s == "vk12-floor").unwrap();
    let (enabled, p) = created("vk12-floor", floor);
    assert!(!enabled.has(VkFeature::SampleRateShading));
    assert_eq!(p.supports_pull_model_interpolation(), Ok(false));
    let (_, moltenvk) = profiles.iter().find(|(s, _)| s == "moltenvk-m3").unwrap();
    let member = VkFeature::ShaderSampleRateInterpolationFunctions;
    let (enabled, p) = created("moltenvk-m3", moltenvk);
    assert!(enabled.has(member));
    assert_eq!(p.supports_pull_model_interpolation(), Ok(true));
    let mut withheld = moltenvk.clone();
    withheld.features.remove(&member);
    let (enabled, p) = created("moltenvk-m3", &withheld);
    assert!(!enabled.has(member));
    assert_eq!(p.supports_pull_model_interpolation(), Ok(false));
}
