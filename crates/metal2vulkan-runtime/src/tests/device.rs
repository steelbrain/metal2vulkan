use crate::device::properties::registry_id;
use crate::device::submit::binding_seen;

#[test]
fn the_registry_id_folds_the_uuid_and_is_never_zero() {
    assert_eq!(registry_id(&[0; 16]), 1);
    assert_eq!(
        registry_id(&[7; 16]),
        1,
        "equal halves fold to 0, answered as 1"
    );
    let mut uuid = [0; 16];
    uuid[0] = 0x12;
    uuid[8] = 0x30;
    uuid[15] = 0x80;
    assert_eq!(registry_id(&uuid), 0x8000_0000_0000_0022);
}

use crate::capabilities::DeviceIdentity;
use crate::device::properties::{bus_addresses_of, PhysicalDeviceIdentity};
use crate::DeviceKind;
use metal2vulkan::vk::SampleCountFlags;

/// The identity `device::properties::query` reads of a physical device whose driver reported
/// `pci_bus` and `drm_render`, with every other field a fixed value this file does not read.
fn identity(
    pci_bus: Option<(u32, u32, u32, u32)>,
    drm_render: Option<(i64, i64)>,
) -> DeviceIdentity {
    DeviceIdentity {
        name: "test device".to_owned(),
        kind: DeviceKind::Discrete,
        vendor_id: 0x1002,
        device_id: 0x15bf,
        device_uuid: [7; 16],
        pci_bus,
        drm_render,
        api_version: ApiVersion::V1_3,
        max_viewports: 16,
        max_sampler_allocation_count: 4000,
        gpu_core_count: None,
        sampled_sample_counts: SampleCountFlags::empty(),
        device_local_heap: None,
    }
}

#[test]
fn a_reported_pci_address_and_render_node_are_read_as_the_driver_filled_them() {
    // Every field of both structures distinct, so a field read in place of another fails: a PCI
    // function that is no domain, and a minor that is no major.
    let bus = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default()
        .pci_domain(0x1_0000)
        .pci_bus(0x41)
        .pci_device(0x02)
        .pci_function(0x03);
    let nodes = vk::PhysicalDeviceDrmPropertiesEXT::default()
        .has_primary(true)
        .has_render(true)
        .primary_major(226)
        .primary_minor(0)
        .render_major(226)
        .render_minor(128);
    assert_eq!(
        bus_addresses_of(Some(&bus), Some(&nodes)),
        (Some((0x1_0000, 0x41, 0x02, 0x03)), Some((226, 128))),
        "the domain, bus, device and function, and the render node's major and minor"
    );
    // The widths and signedness are the structures' own: a domain Vulkan's `uint32_t` holds and a
    // major and minor its `int64_t` holds, negative, pass through unchanged.
    let wide = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default()
        .pci_domain(u32::MAX)
        .pci_bus(u32::MAX)
        .pci_device(u32::MAX)
        .pci_function(u32::MAX);
    let signed = vk::PhysicalDeviceDrmPropertiesEXT::default()
        .has_render(true)
        .render_major(i64::MIN)
        .render_minor(i64::MAX);
    assert_eq!(
        bus_addresses_of(Some(&wide), Some(&signed)),
        (
            Some((u32::MAX, u32::MAX, u32::MAX, u32::MAX)),
            Some((i64::MIN, i64::MAX))
        )
    );
}

#[test]
fn an_unlisted_extension_and_a_device_with_no_render_node_answer_none() {
    let nodes = vk::PhysicalDeviceDrmPropertiesEXT::default()
        .has_render(true)
        .render_major(226)
        .render_minor(128);
    // An extension the physical device does not list: no structure, so no answer. Both unlisted
    // is MoltenVK, which lists neither.
    assert_eq!(bus_addresses_of(None, None), (None, None));
    assert_eq!(
        bus_addresses_of(None, Some(&nodes)),
        (None, Some((226, 128)))
    );
    let bus = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default().pci_bus(0x41);
    assert_eq!(
        bus_addresses_of(Some(&bus), None),
        (Some((0, 0x41, 0, 0)), None)
    );
    // A driver that lists the extension and reports the device has no render node: `None`, not
    // the major and minor it left, whatever they are.
    let primary_only = vk::PhysicalDeviceDrmPropertiesEXT::default()
        .has_primary(true)
        .has_render(false)
        .primary_major(226)
        .primary_minor(0)
        .render_major(226)
        .render_minor(128);
    assert_eq!(bus_addresses_of(None, Some(&primary_only)), (None, None));
    let zeroed = vk::PhysicalDeviceDrmPropertiesEXT::default();
    assert_eq!(bus_addresses_of(None, Some(&zeroed)), (None, None));
}

#[test]
fn a_physical_device_s_identity_carries_the_bus_address_the_device_answers() {
    let read = identity(Some((0x1_0000, 0x41, 0x02, 0x03)), Some((226, 128)));
    let answered = PhysicalDeviceIdentity::of(&read);
    assert_eq!(answered.pci_bus, read.pci_bus);
    assert_eq!(answered.drm_render, read.drm_render);
    assert_eq!(answered.device_uuid, read.device_uuid);
    assert_eq!(answered.registry_id, registry_id(&read.device_uuid));
    assert_eq!(answered.name, read.name);
    assert_eq!(answered.kind, read.kind);
    assert_eq!(answered.vendor_id, read.vendor_id);
    assert_eq!(answered.device_id, read.device_id);
    assert_eq!(answered.api_version, read.api_version);
    // A device of a physical device that lists neither extension answers `None` for both, and
    // the identity read of it does too (every MoltenVK device, this host's among them).
    let neither = PhysicalDeviceIdentity::of(&identity(None, None));
    assert_eq!(neither.pci_bus, None);
    assert_eq!(neither.drm_render, None);
}

use crate::device::create::{
    default_loaders, dma_buf_image_names, dma_buf_names, drops_layers, enable_set, host_image_copy,
    instance_version, layer_settings, layers_dropped, linear_color_attachment, open_loader,
    pageable_device_local_memory, queue_plan, setting_values, timeline_difference, QueuePlan,
    LAYERS_DROPPED,
};
use crate::device::create::{extra_queue_places, extra_queues_absent, reserved_queues_absent};
use crate::enables::host_image_copy_names;
use crate::{DeviceOptions, Error, InstanceApiVersion, LayerSetting, LayerSettingValue};
use ash::vk;
use metal2vulkan::device::{ApiVersion, DeviceDescription, VkExtension, VkFeature};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};

/// Every committed profile's description at the runtime's Vulkan 1.2 instance, as `(file,
/// description)`.
fn descriptions() -> Vec<(std::path::PathBuf, DeviceDescription)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let Ok(mut description) = DeviceDescription::from_khronos_json(&text, None) else {
            continue;
        };
        description.api_version = description.api_version.min(ApiVersion::V1_2);
        out.push((path, description));
    }
    assert!(out.len() > 3, "read {} committed profiles", out.len());
    out
}

/// The dma-buf route's extensions: `VK_KHR_external_memory_fd` and
/// `VK_EXT_external_memory_dma_buf`.
const DMA_BUF: [&CStr; 2] = [
    c"VK_KHR_external_memory_fd",
    c"VK_EXT_external_memory_dma_buf",
];

/// The extensions of the images of textures over the route's pages:
/// `VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign`.
const DMA_BUF_IMAGES: [&CStr; 2] = [
    c"VK_EXT_image_drm_format_modifier",
    c"VK_EXT_queue_family_foreign",
];

/// The dma-buf route of a no-copy buffer (`dma_buf.rs`), the last resort where a device refuses
/// the pointer, takes its two extensions on every committed profile that lists both (RADV's,
/// NVIDIA's, lavapipe's and ANV's among them), on no profile that lacks either, and outside the
/// vocabulary; the images of textures over its pages take theirs
/// (`VK_EXT_image_drm_format_modifier`, `VK_EXT_queue_family_foreign`) on every profile that
/// lists those two and the route's, and on no other. Every profile's enable set is the
/// recommendation and `VK_EXT_external_memory_host` where listed, as before the route, so no
/// compile profile or fingerprint (the shader cache's key) moves with them.
#[test]
fn the_dma_buf_route_is_enabled_wherever_listed_outside_the_profile() {
    let (mut lists_both, mut lists_all) = (0, 0);
    for (path, description) in descriptions() {
        let text = std::fs::read_to_string(&path).unwrap();
        let listed: BTreeSet<CString> = DMA_BUF
            .iter()
            .chain(&DMA_BUF_IMAGES)
            .filter(|name| text.contains(&format!("\"{}\"", name.to_str().unwrap())))
            .map(|name| (*name).to_owned())
            .collect();
        let lists = DMA_BUF.iter().all(|name| listed.contains(*name));
        let lists_images = DMA_BUF_IMAGES.iter().all(|name| listed.contains(*name));
        assert_eq!(dma_buf_names(&listed).is_some(), lists, "{path:?}");
        assert_eq!(
            dma_buf_image_names(&listed).is_some(),
            lists && lists_images,
            "{path:?}"
        );
        for name in DMA_BUF.iter().chain(&DMA_BUF_IMAGES) {
            let mut one_short = listed.clone();
            one_short.remove(*name);
            assert_eq!(dma_buf_image_names(&one_short), None, "{path:?}");
            if DMA_BUF.contains(name) {
                assert_eq!(dma_buf_names(&one_short), None, "{path:?}");
            }
        }
        lists_both += usize::from(lists);
        lists_all += usize::from(lists && lists_images);
        let options = DeviceOptions::default();
        let mut before = description.recommended_enables(&options.policy);
        let host_memory = VkExtension::ExtExternalMemoryHost;
        if description.extensions.contains(&host_memory) {
            before.extensions.insert(host_memory);
        }
        assert_eq!(enable_set(&description, &options), before, "{path:?}");
    }
    assert!(
        lists_both > 1,
        "fewer than two committed profiles list both"
    );
    assert!(
        lists_all > 1,
        "fewer than two committed profiles list all four"
    );
}

/// The robustness features.
const ROBUSTNESS: [VkFeature; 4] = [
    VkFeature::RobustBufferAccess,
    VkFeature::RobustBufferAccess2,
    VkFeature::RobustImageAccess,
    VkFeature::RobustImageAccess2,
];

/// A created device enables the recommendation with `VK_EXT_external_memory_host` where listed;
/// withholding the robustness features (the executor's robustness-off leg) leaves every other
/// recommended enable, is a set the description accepts on every committed profile, and makes a
/// compile device that states no robust access.
#[test]
fn withheld_features_leave_the_rest_of_the_recommendation() {
    for (path, description) in descriptions() {
        let options = DeviceOptions::default();
        let mut recommended = description.recommended_enables(&options.policy);
        if description
            .extensions
            .contains(&VkExtension::ExtExternalMemoryHost)
        {
            recommended
                .extensions
                .insert(VkExtension::ExtExternalMemoryHost);
        }
        assert_eq!(enable_set(&description, &options), recommended, "{path:?}");
        let options = DeviceOptions {
            withheld_features: ROBUSTNESS.into_iter().collect(),
            ..DeviceOptions::default()
        };
        let set = enable_set(&description, &options);
        let mut expected = recommended.clone();
        expected
            .features
            .retain(|feature| !ROBUSTNESS.contains(feature));
        assert_eq!(set, expected, "{path:?}");
        assert_eq!(set.extensions, recommended.extensions, "{path:?}");
        let enabled = description
            .clone()
            .with_enabled(&set)
            .unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
        for feature in ROBUSTNESS {
            assert!(
                !enabled.features.contains(&feature),
                "{path:?}: {feature:?}"
            );
        }
        metal2vulkan::device::Device::new(enabled, options.policy)
            .unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
    }
}

/// Withholding a feature another enabled one requires is refused by the description's check,
/// never trimmed further: `robustBufferAccess` without `robustBufferAccess2` where both are
/// recommended.
#[test]
fn withholding_a_required_feature_is_refused() {
    let both = descriptions().into_iter().find(|(_, description)| {
        let set = description.recommended_enables(&DeviceOptions::default().policy);
        set.features.contains(&VkFeature::RobustBufferAccess)
            && set.features.contains(&VkFeature::RobustBufferAccess2)
    });
    let (path, description) = both.expect("a committed profile recommends both");
    let options = DeviceOptions {
        withheld_features: [VkFeature::RobustBufferAccess].into_iter().collect(),
        ..DeviceOptions::default()
    };
    let set = enable_set(&description, &options);
    assert!(
        description.with_enabled(&set).is_err(),
        "{path:?}: robustBufferAccess2 without robustBufferAccess was accepted"
    );
}

/// Withholding `VK_EXT_shader_stencil_export` (the shader-side stencil's lane run) leaves every
/// other enable, and where the recommendation has it, is a set the description accepts that makes a
/// compile device without stencil export.
#[test]
fn withheld_extensions_leave_the_rest_of_the_recommendation() {
    let stencil = VkExtension::ExtShaderStencilExport;
    let mut had = 0;
    for (path, description) in descriptions() {
        let recommended = enable_set(&description, &DeviceOptions::default());
        let options = DeviceOptions {
            withheld_extensions: [stencil].into_iter().collect(),
            ..DeviceOptions::default()
        };
        let set = enable_set(&description, &options);
        let mut expected = recommended.clone();
        expected.extensions.remove(&stencil);
        assert_eq!(set, expected, "{path:?}");
        if !recommended.extensions.contains(&stencil) {
            continue;
        }
        had += 1;
        let enabled = description
            .clone()
            .with_enabled(&set)
            .unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
        let compile = metal2vulkan::device::Device::new(enabled, options.policy)
            .unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
        assert!(!compile.profile().fragment().stencil_export, "{path:?}");
    }
    assert!(had > 0, "no committed profile recommends stencil export");
}

/// Withholding an extension an enabled feature needs is refused by the description's check,
/// never trimmed further: `VK_EXT_fragment_shader_interlock` where `fragmentShaderPixelInterlock`
/// is recommended.
#[test]
fn withholding_a_needed_extension_is_refused() {
    let interlock = descriptions().into_iter().find(|(_, description)| {
        enable_set(description, &DeviceOptions::default())
            .features
            .contains(&VkFeature::FragmentShaderPixelInterlock)
    });
    let (path, description) = interlock.expect("a committed profile recommends the interlock");
    let options = DeviceOptions {
        withheld_extensions: [VkExtension::ExtFragmentShaderInterlock]
            .into_iter()
            .collect(),
        ..DeviceOptions::default()
    };
    let set = enable_set(&description, &options);
    assert!(
        set.features
            .contains(&VkFeature::FragmentShaderPixelInterlock),
        "{path:?}: the feature was trimmed with its extension"
    );
    assert!(
        description.with_enabled(&set).is_err(),
        "{path:?}: fragmentShaderPixelInterlock without its extension was accepted"
    );
}

/// Each layer setting chains as `VK_EXT_layer_settings` reads it: its layer and name, a boolean
/// as one `VkBool32` of its value, and strings as one pointer each, in order.
#[test]
fn layer_settings_chain_their_values() {
    let layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
    let settings = vec![
        LayerSetting::new(
            layer.clone(),
            c"enable_message_limit".to_owned(),
            LayerSettingValue::Bool(false),
        ),
        LayerSetting::new(
            layer.clone(),
            c"validate_sync".to_owned(),
            LayerSettingValue::Bool(true),
        ),
        LayerSetting::new(
            layer.clone(),
            c"profile_file".to_owned(),
            LayerSettingValue::Strings(vec![c"a.json".to_owned(), c"b.json".to_owned()]),
        ),
    ];
    let (bools, strings) = setting_values(&settings);
    let raw = layer_settings(&settings, &bools, &strings);
    assert_eq!(raw.len(), 3);
    let text = |pointer: *const std::ffi::c_char| {
        // SAFETY: the names point into `settings`, NUL-terminated and alive.
        unsafe { CStr::from_ptr(pointer) }.to_owned()
    };
    let read_bool = |setting: &vk::LayerSettingEXT<'_>| {
        assert!(setting.ty == vk::LayerSettingTypeEXT::BOOL32);
        assert_eq!(setting.value_count, 1);
        // SAFETY: a boolean setting points at one `VkBool32` of `bools`.
        unsafe { *setting.p_values.cast::<vk::Bool32>() }
    };
    for setting in &raw {
        assert_eq!(text(setting.p_layer_name), layer);
    }
    assert_eq!(
        text(raw[0].p_setting_name),
        c"enable_message_limit".to_owned()
    );
    assert_eq!(read_bool(&raw[0]), vk::FALSE);
    assert_eq!(text(raw[1].p_setting_name), c"validate_sync".to_owned());
    assert_eq!(read_bool(&raw[1]), vk::TRUE);
    assert_eq!(text(raw[2].p_setting_name), c"profile_file".to_owned());
    assert!(raw[2].ty == vk::LayerSettingTypeEXT::STRING);
    assert_eq!(raw[2].value_count, 2);
    // SAFETY: a string setting points at its `value_count` string pointers in `strings`.
    let pointers =
        unsafe { std::slice::from_raw_parts(raw[2].p_values.cast::<*const std::ffi::c_char>(), 2) };
    assert_eq!(text(pointers[0]), c"a.json".to_owned());
    assert_eq!(text(pointers[1]), c"b.json".to_owned());
}

/// The instance API version a device's creator requests: Vulkan 1.2 unless set, without asking the
/// loader; a request at or below the loader's version as made; below 1.2 or above the loader's
/// version refused, typed, never clamped.
#[test]
fn the_instance_api_version_is_the_request_or_refused() {
    let v = |major, minor| ApiVersion { major, minor };
    let unasked = || -> Result<ApiVersion, Error> { panic!("the loader was asked") };
    assert_eq!(instance_version(None, unasked), Ok(ApiVersion::V1_2));
    assert_eq!(
        instance_version(Some(v(1, 1)), unasked),
        Err(Error::InstanceApiVersion(InstanceApiVersion::BelowFloor {
            requested: v(1, 1)
        }))
    );
    let loader = || Ok(v(1, 3));
    assert_eq!(
        instance_version(Some(ApiVersion::V1_2), loader),
        Ok(ApiVersion::V1_2)
    );
    assert_eq!(instance_version(Some(v(1, 3)), loader), Ok(v(1, 3)));
    assert_eq!(
        instance_version(Some(v(1, 4)), loader),
        Err(Error::InstanceApiVersion(InstanceApiVersion::AboveLoader {
            requested: v(1, 4),
            loader: v(1, 3)
        }))
    );
    assert_eq!(
        instance_version(Some(v(1, 3)), || Err(Error::NoDevice)),
        Err(Error::NoDevice)
    );
}

/// A device whose timeline semaphores step less than Vulkan requires of every device
/// (`maxTimelineSemaphoreValueDifference` below `2^31 - 1`, as the profiles layer presents
/// lunarg-desktop-baseline-2023, whose published profile states 0) is refused at creation, typed,
/// naming the value: the runtime's first signal, priming the device, is already a step. Vulkan's
/// minimum and above are taken.
#[test]
fn a_device_below_the_timeline_semaphore_minimum_is_refused() {
    use metal2vulkan::error::{Feature, Unsupported};
    let refused = |available| {
        Err(Error::Unsupported(Unsupported::not_representable(
            Feature::TimelineSemaphores {
                max_value_difference: available,
            },
        )))
    };
    assert_eq!(timeline_difference(0), refused(0));
    assert_eq!(timeline_difference((1 << 31) - 2), refused((1 << 31) - 2));
    assert_eq!(timeline_difference((1 << 31) - 1), Ok(()));
    assert_eq!(timeline_difference(u64::MAX), Ok(()));
}

/// A device whose queue families are all equivalent (the same flags, queue count, timestamp bits
/// and transfer granularity: MoltenVK's four families of one queue each) takes a queue of each
/// family in turn, the first family's first queue leading; a device of one family takes its
/// queues, at most four; a device whose families differ (a transfer-only family beside the
/// graphics one) takes the first graphics-and-compute family's queues alone, wherever it sits.
/// The round robin takes those queues less the two reserved for presenting queues, at least four
/// and at most sixteen, so a family with queues to spare lends them rather than leaving them
/// idle; the next queues in the same order, at most two, are reserved for presenting queues, and
/// the first queue of a family of transfers alone whose copies take any texel for staging copies;
/// a family of compute without graphics is reserved for none.
#[test]
fn equivalent_queue_families_are_all_taken_in_turn() {
    let all = vk::QueueFlags::GRAPHICS
        | vk::QueueFlags::COMPUTE
        | vk::QueueFlags::TRANSFER
        | vk::QueueFlags::SPARSE_BINDING;
    let family = |flags: vk::QueueFlags, count: u32| vk::QueueFamilyProperties {
        queue_flags: flags,
        queue_count: count,
        timestamp_valid_bits: 64,
        min_image_transfer_granularity: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
    };
    let staged = |primary, queues: &[(u32, u32)], present: &[(u32, u32)], staging| {
        Some(QueuePlan {
            primary,
            queues: queues.to_vec(),
            present: present.to_vec(),
            reserved: Vec::new(),
            staging,
        })
    };
    let plan = |primary, queues: &[(u32, u32)], present: &[(u32, u32)]| {
        staged(primary, queues, present, None)
    };
    assert_eq!(
        queue_plan(&[family(all, 1); 4], 0),
        plan(0, &[(0, 0), (1, 0), (2, 0), (3, 0)], &[])
    );
    assert_eq!(
        queue_plan(&[family(all, 3); 2], 0),
        plan(0, &[(0, 0), (1, 0), (0, 1), (1, 1)], &[(0, 2), (1, 2)])
    );
    assert_eq!(
        queue_plan(&[family(all, 6)], 0),
        plan(0, &[(0, 0), (0, 1), (0, 2), (0, 3)], &[(0, 4), (0, 5)])
    );
    assert_eq!(
        queue_plan(&[family(all, 5)], 0),
        plan(0, &[(0, 0), (0, 1), (0, 2), (0, 3)], &[(0, 4)])
    );
    assert_eq!(queue_plan(&[family(all, 1)], 0), plan(0, &[(0, 0)], &[]));
    let transfer = family(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING, 2);
    assert_eq!(
        queue_plan(&[family(all, 2), transfer], 0),
        staged(0, &[(0, 0), (0, 1)], &[], Some((1, 0)))
    );
    // NVIDIA's shape (an RTX 3090 with 615.71.09): a graphics family of 16, a
    // copy engine of 2, a compute family of 8, then video and optical flow families that also
    // transfer.
    let compute = family(all & !vk::QueueFlags::GRAPHICS, 8);
    let decode = family(transfer.queue_flags | vk::QueueFlags::VIDEO_DECODE_KHR, 1);
    let encode = family(transfer.queue_flags | vk::QueueFlags::VIDEO_ENCODE_KHR, 1);
    let flow = family(transfer.queue_flags | vk::QueueFlags::OPTICAL_FLOW_NV, 1);
    // Fourteen of the sixteen go to the round robin, two to presenting: its queues past the two
    // reserved would otherwise sit idle while logical queues shared four.
    let fourteen: Vec<(u32, u32)> = (0..14).map(|index| (0, index)).collect();
    assert_eq!(
        queue_plan(
            &[family(all, 16), transfer, compute, decode, encode, flow],
            0
        ),
        staged(0, &fourteen, &[(0, 14), (0, 15)], Some((1, 0)))
    );
    assert_eq!(
        queue_plan(&[family(all, 16), compute, decode, encode, flow], 0),
        plan(0, &fourteen, &[(0, 14), (0, 15)]),
        "no family of transfers alone: no staging queue"
    );
    // The round robin's ceiling: a family of more than eighteen leaves the rest untaken.
    let sixteen: Vec<(u32, u32)> = (0..16).map(|index| (0, index)).collect();
    assert_eq!(
        queue_plan(&[family(all, 24)], 0),
        plan(0, &sixteen, &[(0, 16), (0, 17)]),
        "the round robin takes at most sixteen, and presenting the next two"
    );
    // Its floor: a family of seven lends five, and one of six still lends four, so no device
    // takes fewer round-robin queues than it took before.
    assert_eq!(
        queue_plan(&[family(all, 7)], 0),
        plan(
            0,
            &[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
            &[(0, 5), (0, 6)]
        )
    );
    // RADV's shape (a Radeon 780M with Mesa 26.2.4): a graphics family of one, a compute family of
    // four, video decode and encode families, and a family of sparse binding alone. Its compute
    // queues stall behind a kernel on every compute unit (a readback there took up to 1054 ms), so
    // none stages: its textures copy on the host from its host-cached memory.
    let video = |flags| family(flags, 1);
    assert_eq!(
        queue_plan(
            &[
                family(all, 1),
                family(all & !vk::QueueFlags::GRAPHICS, 4),
                video(vk::QueueFlags::VIDEO_DECODE_KHR),
                video(vk::QueueFlags::VIDEO_ENCODE_KHR),
                family(vk::QueueFlags::SPARSE_BINDING, 1),
            ],
            0
        ),
        plan(0, &[(0, 0)], &[])
    );
    let mut blocks = transfer;
    blocks.min_image_transfer_granularity.width = 8;
    assert_eq!(
        queue_plan(&[family(all, 1), blocks], 0),
        plan(0, &[(0, 0)], &[]),
        "a copy engine whose copies take whole blocks of texels stages nothing"
    );
    assert_eq!(
        queue_plan(&[transfer, family(all, 1), family(all, 1)], 0),
        staged(1, &[(1, 0)], &[], Some((0, 0)))
    );
    let mut coarse = family(all, 1);
    coarse.timestamp_valid_bits = 36;
    assert_eq!(
        queue_plan(&[family(all, 1), coarse], 0),
        plan(0, &[(0, 0)], &[]),
        "families whose timestamps differ are not equivalent"
    );
    assert_eq!(queue_plan(&[transfer], 0), None);
    assert_eq!(queue_plan(&[family(all, 0)], 0), None);
}

/// The caller's extra queues (`DeviceOptions::extra_queues`): each a queue the device's own plan
/// leaves free, of the first family whose flags contain the asked ones and that has one; where
/// none is free, the last such queue the plan took (presenting's, then the round robin's but its
/// first, then staging's), which the plan gives up; none for flags no family has, empty flags, or
/// a device of one queue.
#[test]
fn extra_queues_take_a_free_queue_else_one_the_device_gives_up() {
    let all = vk::QueueFlags::GRAPHICS
        | vk::QueueFlags::COMPUTE
        | vk::QueueFlags::TRANSFER
        | vk::QueueFlags::SPARSE_BINDING;
    let family = |flags: vk::QueueFlags, count: u32| vk::QueueFamilyProperties {
        queue_flags: flags,
        queue_count: count,
        timestamp_valid_bits: 64,
        min_image_transfer_granularity: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
    };
    let (decode, encode) = (
        vk::QueueFlags::VIDEO_DECODE_KHR,
        vk::QueueFlags::VIDEO_ENCODE_KHR,
    );
    let (compute, transfer) = (vk::QueueFlags::COMPUTE, vk::QueueFlags::TRANSFER);
    // NVIDIA's shape: video families of their own, a compute family with queues to spare, a copy
    // engine whose second queue staging leaves.
    let copy = family(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING, 2);
    let nvidia = [
        family(all, 16),
        copy,
        family(all & !vk::QueueFlags::GRAPHICS, 8),
        family(copy.queue_flags | decode, 1),
        family(copy.queue_flags | encode, 1),
    ];
    let mut plan = queue_plan(&nvidia, 0).unwrap();
    let before = plan.clone();
    assert_eq!(
        extra_queue_places(
            &nvidia,
            &mut plan,
            &[decode, encode, compute, transfer, decode]
        ),
        [Some((3, 0)), Some((4, 0)), Some((2, 0)), Some((1, 1)), None],
        "the second decode queue: the family has one, and the graphics family none to give"
    );
    assert_eq!(
        plan, before,
        "every one was free: the device keeps all its own"
    );
    // MoltenVK's shape: four families of one queue each, every one the device's own. The round
    // robin gives up its last queues, never its first.
    let moltenvk = [family(all, 1); 4];
    let mut plan = queue_plan(&moltenvk, 0).unwrap();
    assert_eq!(
        extra_queue_places(&moltenvk, &mut plan, &[compute, decode, transfer]),
        [Some((3, 0)), None, Some((2, 0))]
    );
    assert_eq!(plan.queues, [(0, 0), (1, 0)]);
    let mut plan = queue_plan(&moltenvk, 0).unwrap();
    let four = [compute; 4];
    assert_eq!(
        extra_queue_places(&moltenvk, &mut plan, &four),
        [Some((3, 0)), Some((2, 0)), Some((1, 0)), None]
    );
    assert_eq!(plan.queues, [(0, 0)]);
    // A family of six: the presenting reserve goes first, from its last.
    let mut plan = queue_plan(&[family(all, 6)], 0).unwrap();
    assert_eq!(
        extra_queue_places(&[family(all, 6)], &mut plan, &[compute, compute, compute]),
        [Some((0, 5)), Some((0, 4)), Some((0, 3))]
    );
    assert_eq!((plan.queues.len(), plan.present.len()), (3, 0));
    // One queue (lavapipe's), empty flags, flags no family has: none.
    let mut plan = queue_plan(&[family(all, 1)], 0).unwrap();
    assert_eq!(
        extra_queue_places(
            &[family(all, 1)],
            &mut plan,
            &[compute, vk::QueueFlags::empty(), decode]
        ),
        [None, None, None]
    );
    assert_eq!(plan.queues, [(0, 0)]);
    // Staging's queue last of all.
    let shape = [family(all, 1), family(copy.queue_flags, 1)];
    let mut plan = queue_plan(&shape, 0).unwrap();
    assert_eq!(plan.staging, Some((1, 0)));
    assert_eq!(
        extra_queue_places(&shape, &mut plan, &[transfer]),
        [Some((1, 0))]
    );
    assert_eq!(
        (plan.queues.as_slice(), plan.staging),
        (&[(0, 0)][..], None)
    );
    let text = extra_queues_absent(&[decode, compute | transfer], "a GPU").unwrap();
    assert!(
        text.contains("VIDEO_DECODE_KHR; COMPUTE | TRANSFER")
            && text.ends_with("[metal2vulkan-extra-queues-absent]"),
        "{text}"
    );
    assert_eq!(extra_queues_absent(&[], "a GPU"), None);
}

/// The caller's reserved queues (`DeviceOptions::reserved_queues`): queues of the families the
/// round robin takes, which it gives up for them, so it shrinks by each one; the round robin keeps
/// its first four and presenting its two before the caller has any, and a family with queues past
/// the round robin's ceiling and presenting's two gives those first. A device with none to spare
/// (MoltenVK's four families of one queue, lavapipe's one queue, a family of six) reserves none,
/// and says so; no extra queue takes a reserved one.
#[test]
fn reserved_queues_leave_the_round_robin_after_presenting() {
    let all = vk::QueueFlags::GRAPHICS
        | vk::QueueFlags::COMPUTE
        | vk::QueueFlags::TRANSFER
        | vk::QueueFlags::SPARSE_BINDING;
    let family = |flags: vk::QueueFlags, count: u32| vk::QueueFamilyProperties {
        queue_flags: flags,
        queue_count: count,
        timestamp_valid_bits: 64,
        min_image_transfer_granularity: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
    };
    let of = |family: u32, indices: std::ops::Range<u32>| -> Vec<(u32, u32)> {
        indices.map(|index| (family, index)).collect()
    };
    let shape = |plan: &QueuePlan| {
        (
            plan.queues.len(),
            plan.present.clone(),
            plan.reserved.clone(),
        )
    };
    // NVIDIA's shape (an RTX 5080): a graphics family of 16, a copy engine of 2, a compute family
    // of 8, video families. One reserved queue: the round robin 13, presenting 2, the caller the
    // family's last; staging and every other family as before.
    let copy = family(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING, 2);
    let nvidia = [
        family(all, 16),
        copy,
        family(all & !vk::QueueFlags::GRAPHICS, 8),
        family(copy.queue_flags | vk::QueueFlags::VIDEO_DECODE_KHR, 1),
    ];
    let unreserved = queue_plan(&nvidia, 0).unwrap();
    assert_eq!(unreserved.queues.len(), 14);
    assert!(unreserved.reserved.is_empty());
    let one = queue_plan(&nvidia, 1).unwrap();
    assert_eq!(
        one,
        QueuePlan {
            primary: 0,
            queues: of(0, 0..13),
            present: of(0, 13..15),
            reserved: of(0, 15..16),
            staging: Some((1, 0)),
        }
    );
    // As many as the family spares: the round robin keeps four, presenting two, so ten.
    for asked in [10, 11, 64, u32::MAX] {
        assert_eq!(
            shape(&queue_plan(&nvidia, asked).unwrap()),
            (4, of(0, 4..6), of(0, 6..16)),
            "{asked} asked"
        );
    }
    // No extra queue takes a reserved one: graphics extra queues take presenting's, then the
    // round robin's last, as without a reservation.
    let mut plan = queue_plan(&nvidia, 1).unwrap();
    let graphics = vk::QueueFlags::GRAPHICS;
    assert_eq!(
        extra_queue_places(&nvidia, &mut plan, &[graphics; 3]),
        [Some((0, 14)), Some((0, 13)), Some((0, 12))]
    );
    assert_eq!(shape(&plan), (12, Vec::new(), of(0, 15..16)));
    // A family of seven spares one; of six, none (the round robin's four and presenting's two).
    assert_eq!(
        shape(&queue_plan(&[family(all, 7)], 1).unwrap()),
        (4, of(0, 4..6), of(0, 6..7))
    );
    assert_eq!(
        shape(&queue_plan(&[family(all, 6)], 1).unwrap()),
        (4, of(0, 4..6), Vec::new())
    );
    // A family of five: presenting keeps its one, the caller has none.
    assert_eq!(
        shape(&queue_plan(&[family(all, 5)], 1).unwrap()),
        (4, of(0, 4..5), Vec::new())
    );
    // Past the round robin's ceiling and presenting's two, the caller's come first: the round
    // robin of a family of 24 stays 16.
    assert_eq!(
        shape(&queue_plan(&[family(all, 24)], 2).unwrap()),
        (16, of(0, 16..18), of(0, 18..20))
    );
    // MoltenVK's shape and lavapipe's: none to spare, the round robin as without.
    let moltenvk = [family(all, 1); 4];
    assert_eq!(queue_plan(&moltenvk, 1), queue_plan(&moltenvk, 0));
    assert_eq!(
        queue_plan(&[family(all, 1)], 1),
        queue_plan(&[family(all, 1)], 0)
    );
    // Equivalent families take their reserved queues in the round robin's order, of families it
    // takes.
    assert_eq!(
        shape(&queue_plan(&[family(all, 4); 2], 1).unwrap()),
        (5, vec![(1, 2), (0, 3)], vec![(1, 3)])
    );
    let text = reserved_queues_absent(1, 0, "a GPU").unwrap();
    assert!(
        text.contains("reserved 0 of the 1 queue(s)")
            && text.ends_with("[metal2vulkan-reserved-queues-absent]"),
        "{text}"
    );
    assert!(reserved_queues_absent(11, 10, "a GPU").is_some());
    assert_eq!(reserved_queues_absent(1, 1, "a GPU"), None);
    assert_eq!(reserved_queues_absent(0, 0, "a GPU"), None);
}

/// The runtime enables `hostImageCopy` itself wherever the device lists the extensions its
/// version needs (at the runtime's Vulkan 1.2 instance, `VK_EXT_host_image_copy`,
/// `VK_KHR_copy_commands2` and `VK_KHR_format_feature_flags2`) and has the feature: a `Shared`
/// or `Managed` texture's CPU copies run on the host there. It asks for the feature only where the
/// extensions are listed.
#[test]
fn host_image_copy_is_enabled_wherever_the_device_has_it() {
    use std::cell::Cell;
    let names = host_image_copy_names(ApiVersion::V1_2);
    assert_eq!(names.len(), 3);
    let listed: BTreeSet<CString> = names.iter().map(|&name| name.to_owned()).collect();
    let short: BTreeSet<CString> = listed.iter().skip(1).cloned().collect();
    let asked = Cell::new(false);
    assert!(host_image_copy(names, &listed, || {
        asked.set(true);
        true
    }));
    assert!(asked.get());
    assert!(
        !host_image_copy(names, &listed, || false),
        "enabled without the feature"
    );
    assert!(!host_image_copy(names, &short, || panic!(
        "asked without the extensions"
    )));
}

/// The runtime enables `VK_NV_linear_color_attachment` exactly where the device lists it with
/// what it requires, Vulkan 1.3 (where `VK_KHR_format_feature_flags2` is core) or that extension
/// (the runtime's default Vulkan 1.2 instance, the consumer's probe), and has
/// `linearColorAttachment`, asked only then.
#[test]
fn linear_color_attachment_is_enabled_where_the_device_has_it_at_vulkan_1_3() {
    use std::cell::Cell;
    let listed: BTreeSet<CString> = [c"VK_NV_linear_color_attachment".to_owned()].into();
    let asked = Cell::new(false);
    assert!(linear_color_attachment(ApiVersion::V1_3, &listed, || {
        asked.set(true);
        true
    }));
    assert!(asked.get());
    assert!(
        !linear_color_attachment(ApiVersion::V1_3, &listed, || false),
        "enabled without the feature"
    );
    assert!(!linear_color_attachment(
        ApiVersion::V1_2,
        &listed,
        || panic!("asked below Vulkan 1.3 without VK_KHR_format_feature_flags2")
    ));
    let mut with_flags2 = listed.clone();
    with_flags2.insert(c"VK_KHR_format_feature_flags2".to_owned());
    assert!(linear_color_attachment(
        ApiVersion::V1_2,
        &with_flags2,
        || true
    ));
    assert!(!linear_color_attachment(
        ApiVersion::V1_3,
        &BTreeSet::new(),
        || panic!("asked without the extension")
    ));
}

/// The runtime enables `VK_EXT_pageable_device_local_memory` exactly where the device lists it
/// and `VK_EXT_memory_priority`, which it requires, and has `pageableDeviceLocalMemory`, asked
/// only then.
#[test]
fn pageable_device_local_memory_is_enabled_where_the_device_has_it() {
    use std::cell::Cell;
    let listed: BTreeSet<CString> = [
        c"VK_EXT_pageable_device_local_memory".to_owned(),
        c"VK_EXT_memory_priority".to_owned(),
    ]
    .into();
    let asked = Cell::new(false);
    assert!(pageable_device_local_memory(&listed, || {
        asked.set(true);
        true
    }));
    assert!(asked.get());
    assert!(
        !pageable_device_local_memory(&listed, || false),
        "enabled without the feature"
    );
    let alone: BTreeSet<CString> = [c"VK_EXT_pageable_device_local_memory".to_owned()].into();
    assert!(!pageable_device_local_memory(&alone, || panic!(
        "asked without VK_EXT_memory_priority"
    )));
    assert!(!pageable_device_local_memory(&BTreeSet::new(), || panic!(
        "asked without the extension"
    )));
}

#[test]
fn the_default_loaders_reach_past_the_dynamic_linker_s_paths_on_macos() {
    // The platform's own loader by name first; on macOS then the Vulkan SDK's, Homebrew's and
    // /usr/local's, which no default search path holds (a consumer's MoltenVK through Homebrew's
    // loader answered NoDevice before).
    let sdk = std::path::Path::new("/sdk/macOS");
    let found: Vec<String> = default_loaders(Some(sdk))
        .iter()
        .map(|path| path.display().to_string())
        .collect();
    if cfg!(target_os = "macos") {
        assert_eq!(
            found,
            [
                "libvulkan.dylib",
                "libvulkan.1.dylib",
                "/sdk/macOS/lib/libvulkan.1.dylib",
                "/opt/homebrew/lib/libvulkan.1.dylib",
                "/usr/local/lib/libvulkan.1.dylib",
            ]
        );
        assert_eq!(default_loaders(None).len(), 4);
    } else if cfg!(target_os = "linux") {
        assert_eq!(found, ["libvulkan.so.1"]);
    }
}

#[test]
fn a_loader_that_does_not_open_is_named_not_reported_as_no_device() {
    let missing = std::path::PathBuf::from("/nonexistent/m2v/libvulkan.so.1");
    match open_loader(Some(&missing)) {
        Err(Error::LoaderNotFound(not_found)) => {
            assert_eq!(not_found.tried, std::slice::from_ref(&missing));
            let message = Error::LoaderNotFound(not_found).to_string();
            assert!(
                message.contains("/nonexistent/m2v/libvulkan.so.1"),
                "{message}"
            );
        }
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("a loader opened from {}", missing.display()),
    }
}

#[test]
fn a_device_extension_whose_instance_extension_the_instance_lacks_is_refused() {
    use crate::device::instance_requirements::instance_requirements_met;
    use crate::InstanceExtensionRequired;
    let names = |names: &[&CStr]| {
        names
            .iter()
            .map(|&name| name.to_owned())
            .collect::<Vec<_>>()
    };
    let refused = |extension, instance_extension| {
        Err(Error::InstanceExtensionRequired(
            InstanceExtensionRequired {
                extension,
                instance_extension,
            },
        ))
    };
    let swapchain = names(&[c"VK_KHR_swapchain"]);
    assert_eq!(
        instance_requirements_met(&swapchain, &[c"VK_EXT_debug_utils"]),
        refused("VK_KHR_swapchain", "VK_KHR_surface")
    );
    assert_eq!(
        instance_requirements_met(&swapchain, &[c"VK_KHR_surface"]),
        Ok(())
    );
    // Every instance extension of the dependency, in the registry's order, and the first device
    // extension, in the options' order, that lacks one.
    let maintained = names(&[c"VK_KHR_swapchain", c"VK_EXT_swapchain_maintenance1"]);
    assert_eq!(
        instance_requirements_met(&maintained, &[c"VK_KHR_surface"]),
        refused(
            "VK_EXT_swapchain_maintenance1",
            "VK_EXT_surface_maintenance1"
        )
    );
    let surfaces = [
        c"VK_KHR_surface",
        c"VK_KHR_get_surface_capabilities2",
        c"VK_EXT_surface_maintenance1",
    ];
    assert_eq!(instance_requirements_met(&maintained, &surfaces), Ok(()));
    // The `KHR` promotion depends on the `KHR` instance extension, not its `EXT` alias.
    assert_eq!(
        instance_requirements_met(&names(&[c"VK_KHR_swapchain_maintenance1"]), &surfaces),
        refused(
            "VK_KHR_swapchain_maintenance1",
            "VK_KHR_surface_maintenance1"
        )
    );
    assert_eq!(
        instance_requirements_met(
            &names(&[c"VK_KHR_present_wait2", c"VK_KHR_swapchain"]),
            &[c"VK_KHR_surface"]
        ),
        refused("VK_KHR_present_wait2", "VK_KHR_get_surface_capabilities2")
    );
    // A device extension whose dependency names no instance extension past Vulkan 1.1, or one
    // the registry does not have, requires nothing of the instance.
    let others = names(&[
        c"VK_EXT_present_mode_fifo_latest_ready",
        c"VK_KHR_external_memory_fd",
        c"VK_EXT_metal_objects",
        c"VK_M2V_not_in_the_registry",
    ]);
    assert_eq!(instance_requirements_met(&others, &[]), Ok(()));
    let message = refused("VK_KHR_swapchain", "VK_KHR_surface")
        .unwrap_err()
        .to_string();
    assert!(message.contains("VK_KHR_surface"), "{message}");
}

/// A no-copy buffer's binding stops being waited for only once a host wait returned for its
/// value (`device/submit.rs`): a timeline's read alone orders no host operation after the
/// signal. A read below the value asks for no wait, and a failed wait leaves it waited for.
#[test]
fn a_binding_is_seen_complete_only_once_a_host_wait_returned_for_it() {
    let mut asked = Vec::new();
    let mut seen = |read: u64, returned: Result<bool, ()>| {
        binding_seen(read, 5, |value| {
            asked.push(value);
            returned
        })
    };
    assert!(
        !seen(5, Ok(false)),
        "the read shows it; the wait did not return"
    );
    assert!(seen(6, Ok(true)));
    assert!(!seen(4, Ok(true)), "the read shows it pending");
    assert!(!seen(5, Err(())), "a lost device");
    assert_eq!(asked, [5, 5, 5]);
}

/// `float-tininess-after-rounding`'s condition (`device/float_tininess.rs`): every committed
/// profile's device but MoltenVK's, whose float functions are Metal's own, takes the device's
/// `float` multiply and fused multiply-add, which round a tiny result before they flush it.
#[test]
fn the_device_s_multiply_rounds_a_tiny_product_before_its_flush_off_moltenvk() {
    use crate::device::float_tininess::rounds_before_flushing;
    let (mut rounding, mut metals) = (0, 0);
    for (path, description) in descriptions() {
        let moltenvk = path
            .file_stem()
            .is_some_and(|stem| stem.to_string_lossy().starts_with("moltenvk"));
        let options = DeviceOptions::default();
        let set = enable_set(&description, &options);
        let enabled = description
            .clone()
            .with_enabled(&set)
            .unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
        let compile = metal2vulkan::device::Device::new(enabled, options.policy)
            .unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
        let rounds = rounds_before_flushing(&compile);
        assert_eq!(rounds, !moltenvk, "{path:?}");
        rounding += usize::from(rounds);
        metals += usize::from(moltenvk);
    }
    assert!(
        rounding > 0 && metals > 0,
        "{rounding} rounding, {metals} Metal's own"
    );
}

/// Only a `vkCreateInstance` that answered `VK_ERROR_LAYER_NOT_PRESENT` with a layer asked for is
/// answered by creating the instance again without the layers; every other failure, and that
/// answer with no layer asked for, is the caller's. The creation clears the list to drop them, so
/// the second attempt cannot answer so again: two `vkCreateInstance` calls at most.
#[test]
fn only_a_missing_layer_drops_the_layers_asked_for() {
    let asked = vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
    let missing = vk::Result::ERROR_LAYER_NOT_PRESENT;
    assert!(drops_layers(missing, &asked));
    // The list cleared: the guard is false, so the attempts end at two.
    assert!(!drops_layers(missing, &[]));
    // Every other failure, which the caller answers: the fallback is this error code's alone.
    for error in [
        vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        vk::Result::ERROR_INITIALIZATION_FAILED,
        vk::Result::ERROR_EXTENSION_NOT_PRESENT,
        vk::Result::ERROR_INCOMPATIBLE_DRIVER,
        vk::Result::ERROR_DEVICE_LOST,
        vk::Result::SUCCESS,
    ] {
        assert!(!drops_layers(error, &asked), "{error:?} dropped the layers");
    }
}

/// The message a device of such an instance sends as it is made names its layers, its device and
/// the id name; a device whose instance dropped none sends nothing.
#[test]
fn the_dropped_layers_are_announced_by_name() {
    assert_eq!(layers_dropped(&[], "a device"), None);
    let dropped = vec![
        CString::new("VK_LAYER_KHRONOS_validation").unwrap(),
        CString::new("VK_LAYER_absent").unwrap(),
    ];
    let text = layers_dropped(&dropped, "an Apple M3 Max").expect("the message");
    assert!(
        text.contains("VK_LAYER_KHRONOS_validation, VK_LAYER_absent"),
        "{text}"
    );
    assert!(text.contains("an Apple M3 Max"), "{text}");
    assert!(text.contains(LAYERS_DROPPED), "{text}");
    assert_eq!(LAYERS_DROPPED, "metal2vulkan-instance-layers-dropped");
}
