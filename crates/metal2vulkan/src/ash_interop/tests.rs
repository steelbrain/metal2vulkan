//! `ash_interop` tests. `DeviceDescription::query` runs against a **fake instance table**: an
//! `ash::Instance` loaded from a test `vkGetInstanceProcAddr` whose physical-device entry points
//! answer from a [`FakeDevice`]. No loader, ICD, or device is involved.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, CStr};

use ash::vk as avk;
use ash::vk::{Handle, TaggedStructure};

use super::convert::sampler_create_info;
use super::query::{
    DepthClampControlFeatures, Maintenance10Properties, Maintenance8Features, ShaderFmaFeatures,
    ALL_FEATURES,
};
use crate::device::description::FormatProperties;
use crate::device::profile::Independence;
use crate::device::requirements::{ApiVersion, VkExtension, VkFeature};
use crate::device::{AllocationLimits, Device, DeviceDescription, DevicePolicy, FormatSupport};
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::vk;

/// What the fake physical device answers.
struct FakeDevice {
    props: avk::PhysicalDeviceProperties,
    /// The instance's `apiVersion` passed to the query (the device's version by default).
    instance_api: ApiVersion,
    extensions: Vec<&'static CStr>,
    /// Every `VkBool32` of every chained feature structure (unless overridden below).
    all_features: bool,
    v12_features: Option<avk::PhysicalDeviceVulkan12Features<'static>>,
    v11_props: avk::PhysicalDeviceVulkan11Properties<'static>,
    v12_props: avk::PhysicalDeviceVulkan12Properties<'static>,
    v13_props: avk::PhysicalDeviceVulkan13Properties<'static>,
    subgroup_size_control: avk::PhysicalDeviceSubgroupSizeControlProperties<'static>,
    push_descriptor: avk::PhysicalDevicePushDescriptorPropertiesKHR<'static>,
    divisor_khr: avk::PhysicalDeviceVertexAttributeDivisorPropertiesKHR<'static>,
    divisor_ext: avk::PhysicalDeviceVertexAttributeDivisorPropertiesEXT<'static>,
    portability: avk::PhysicalDevicePortabilitySubsetPropertiesKHR<'static>,
    maintenance10: Maintenance10Properties,
    acceleration_structure: avk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    formats: BTreeMap<i32, avk::FormatProperties>,
    /// The bits only `VkFormatProperties3` carries, per format: `(optimal, buffer)`.
    formats3: BTreeMap<i32, (avk::FormatFeatureFlags2, avk::FormatFeatureFlags2)>,
    /// The sample counts `vkGetPhysicalDeviceImageFormatProperties2` reports per format; it refuses
    /// a format not listed with `VK_ERROR_FORMAT_NOT_SUPPORTED`.
    image_sample_counts: BTreeMap<i32, avk::SampleCountFlags>,
    /// What each `vkGetPhysicalDeviceImageFormatProperties2` asked, raw: the format, type,
    /// tiling, usage, and flags.
    images_queried: RefCell<Vec<(i32, i32, i32, u32, u32)>>,
    /// The raw structure types of every feature / property chain the query passed.
    feature_chain: RefCell<Vec<i32>>,
    property_chain: RefCell<Vec<i32>>,
    formats_queried: RefCell<Vec<i32>>,
    /// The formats asked through `vkGetPhysicalDeviceFormatProperties2` with a chained
    /// `VkFormatProperties3`.
    formats_queried3: RefCell<Vec<i32>>,
}

impl FakeDevice {
    fn new(api: ApiVersion) -> FakeDevice {
        FakeDevice {
            props: avk::PhysicalDeviceProperties {
                api_version: api.packed(),
                ..Default::default()
            },
            instance_api: api,
            extensions: Vec::new(),
            all_features: false,
            v12_features: None,
            v11_props: Default::default(),
            v12_props: Default::default(),
            v13_props: Default::default(),
            subgroup_size_control: Default::default(),
            push_descriptor: Default::default(),
            divisor_khr: Default::default(),
            divisor_ext: Default::default(),
            portability: Default::default(),
            maintenance10: Default::default(),
            acceleration_structure: Default::default(),
            formats: BTreeMap::new(),
            formats3: BTreeMap::new(),
            image_sample_counts: BTreeMap::new(),
            images_queried: RefCell::new(Vec::new()),
            feature_chain: RefCell::new(Vec::new()),
            property_chain: RefCell::new(Vec::new()),
            formats_queried: RefCell::new(Vec::new()),
            formats_queried3: RefCell::new(Vec::new()),
        }
    }

    fn handle(&self) -> avk::PhysicalDevice {
        avk::PhysicalDevice::from_raw(self as *const FakeDevice as u64)
    }

    /// Runs the query through the fake instance table.
    fn query(&self) -> Result<DeviceDescription, Error> {
        let static_fn = ash::StaticFn {
            get_instance_proc_addr: fake_get_instance_proc_addr,
        };
        // SAFETY: the fake table answers every entry point the query calls; the handle is this
        // live `FakeDevice`, which outlives the call.
        unsafe {
            let instance = ash::Instance::load(&static_fn, avk::Instance::from_raw(1));
            DeviceDescription::query(&instance, self.handle(), self.instance_api)
        }
    }
}

/// # Safety
/// `pd` was made by [`FakeDevice::handle`] from a live device.
unsafe fn fake<'a>(pd: avk::PhysicalDevice) -> &'a FakeDevice {
    // SAFETY: the caller's contract.
    unsafe { &*(pd.as_raw() as *const FakeDevice) }
}

/// The size of every feature structure the query can chain (all members are `VkBool32` after
/// the 16-byte `sType`/`pNext` header).
fn feature_struct_size(ty: avk::StructureType) -> usize {
    macro_rules! sizes {
        ($($t:ty),* $(,)?) => {
            $(if ty == <$t>::STRUCTURE_TYPE {
                return std::mem::size_of::<$t>();
            })*
        };
    }
    sizes!(
        avk::PhysicalDeviceVulkan11Features<'static>,
        avk::PhysicalDeviceVulkan12Features<'static>,
        avk::PhysicalDeviceVulkan13Features<'static>,
        avk::PhysicalDeviceImageRobustnessFeatures<'static>,
        avk::PhysicalDeviceSubgroupSizeControlFeatures<'static>,
        avk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures<'static>,
        avk::PhysicalDeviceShaderTerminateInvocationFeatures<'static>,
        avk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures<'static>,
        avk::PhysicalDeviceSynchronization2Features<'static>,
        avk::PhysicalDeviceMaintenance4Features<'static>,
        avk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR<'static>,
        avk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR<'static>,
        avk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR<'static>,
        avk::PhysicalDeviceMaintenance5FeaturesKHR<'static>,
        avk::PhysicalDeviceRobustness2FeaturesEXT<'static>,
        avk::PhysicalDeviceShaderAtomicFloatFeaturesEXT<'static>,
        avk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT<'static>,
        avk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT<'static>,
        avk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR<'static>,
        avk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR<'static>,
        avk::PhysicalDeviceShaderQuadControlFeaturesKHR<'static>,
        avk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR<'static>,
        avk::PhysicalDeviceComputeShaderDerivativesFeaturesNV<'static>,
        avk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR<'static>,
        avk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT<'static>,
        avk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT<'static>,
        avk::PhysicalDeviceExtendedDynamicStateFeaturesEXT<'static>,
        avk::PhysicalDeviceExtendedDynamicState2FeaturesEXT<'static>,
        avk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static>,
        avk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT<'static>,
        Maintenance8Features,
        ShaderFmaFeatures,
        avk::PhysicalDeviceBorderColorSwizzleFeaturesEXT<'static>,
        avk::PhysicalDeviceDepthBiasControlFeaturesEXT<'static>,
        avk::PhysicalDeviceDepthClipEnableFeaturesEXT<'static>,
        DepthClampControlFeatures,
        avk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT<'static>,
        avk::PhysicalDeviceProvokingVertexFeaturesEXT<'static>,
        avk::PhysicalDeviceLineRasterizationFeaturesEXT<'static>,
        avk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
        avk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
        avk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
        avk::PhysicalDevicePortabilitySubsetFeaturesKHR<'static>,
    );
    panic!(
        "query chained an unexpected feature structure {}",
        ty.as_raw()
    );
}

/// Sets every `VkBool32` of `size` bytes at `p`, after `header` bytes, to `value`.
///
/// # Safety
/// `p` points to at least `size` writable bytes.
unsafe fn fill_bools(p: *mut u8, header: usize, size: usize, value: bool) {
    let words = (size - header) / 4;
    let bools = p.add(header).cast::<avk::Bool32>();
    for i in 0..words {
        bools.add(i).write_unaligned(value.into());
    }
}

/// Replaces the structure at `out` with `value`, keeping `out`'s `pNext`.
///
/// # Safety
/// `out` points to a `T` whose layout begins with `sType` and `pNext`.
unsafe fn copy_keeping_next<T: Copy>(out: *mut avk::BaseOutStructure<'_>, value: &T) {
    let next = (*out).p_next;
    out.cast::<T>().write(*value);
    (*out).p_next = next;
}

unsafe extern "system" fn fake_get_physical_device_properties(
    pd: avk::PhysicalDevice,
    out: *mut avk::PhysicalDeviceProperties,
) {
    out.write(fake(pd).props);
}

unsafe extern "system" fn fake_get_physical_device_features2(
    pd: avk::PhysicalDevice,
    out: *mut avk::PhysicalDeviceFeatures2<'_>,
) {
    let dev = fake(pd);
    let core = std::ptr::addr_of_mut!((*out).features).cast::<u8>();
    fill_bools(
        core,
        0,
        std::mem::size_of::<avk::PhysicalDeviceFeatures>(),
        dev.all_features,
    );
    let mut next = (*out).p_next.cast::<avk::BaseOutStructure<'_>>();
    while !next.is_null() {
        let ty = (*next).s_type;
        dev.feature_chain.borrow_mut().push(ty.as_raw());
        fill_bools(next.cast(), 16, feature_struct_size(ty), dev.all_features);
        if ty == avk::PhysicalDeviceVulkan12Features::STRUCTURE_TYPE {
            if let Some(v12) = &dev.v12_features {
                copy_keeping_next(next, v12);
            }
        }
        next = (*next).p_next;
    }
}

unsafe extern "system" fn fake_get_physical_device_properties2(
    pd: avk::PhysicalDevice,
    out: *mut avk::PhysicalDeviceProperties2<'_>,
) {
    let dev = fake(pd);
    (*out).properties = dev.props;
    let mut next = (*out).p_next.cast::<avk::BaseOutStructure<'_>>();
    while !next.is_null() {
        let ty = (*next).s_type;
        dev.property_chain.borrow_mut().push(ty.as_raw());
        if ty == avk::PhysicalDeviceVulkan11Properties::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.v11_props);
        } else if ty == avk::PhysicalDeviceVulkan12Properties::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.v12_props);
        } else if ty == avk::PhysicalDeviceVulkan13Properties::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.v13_props);
        } else if ty == avk::PhysicalDeviceSubgroupSizeControlProperties::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.subgroup_size_control);
        } else if ty == avk::PhysicalDevicePushDescriptorPropertiesKHR::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.push_descriptor);
        } else if ty == avk::PhysicalDeviceVertexAttributeDivisorPropertiesKHR::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.divisor_khr);
        } else if ty == avk::PhysicalDeviceVertexAttributeDivisorPropertiesEXT::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.divisor_ext);
        } else if ty == avk::PhysicalDevicePortabilitySubsetPropertiesKHR::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.portability);
        } else if ty == Maintenance10Properties::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.maintenance10);
        } else if ty == avk::PhysicalDeviceAccelerationStructurePropertiesKHR::STRUCTURE_TYPE {
            copy_keeping_next(next, &dev.acceleration_structure);
        }
        next = (*next).p_next;
    }
}

unsafe extern "system" fn fake_enumerate_device_extension_properties(
    pd: avk::PhysicalDevice,
    _layer: *const c_char,
    count: *mut u32,
    out: *mut avk::ExtensionProperties,
) -> avk::Result {
    let dev = fake(pd);
    let n = dev.extensions.len() as u32;
    if out.is_null() {
        *count = n;
        return avk::Result::SUCCESS;
    }
    let written = (*count).min(n);
    for (i, name) in dev.extensions.iter().take(written as usize).enumerate() {
        let mut e = avk::ExtensionProperties::default();
        for (dst, &src) in e.extension_name.iter_mut().zip(name.to_bytes_with_nul()) {
            *dst = src as c_char;
        }
        out.add(i).write(e);
    }
    *count = written;
    if written < n {
        avk::Result::INCOMPLETE
    } else {
        avk::Result::SUCCESS
    }
}

unsafe extern "system" fn fake_get_physical_device_format_properties(
    pd: avk::PhysicalDevice,
    format: avk::Format,
    out: *mut avk::FormatProperties,
) {
    let dev = fake(pd);
    dev.formats_queried.borrow_mut().push(format.as_raw());
    out.write(
        dev.formats
            .get(&format.as_raw())
            .copied()
            .unwrap_or_default(),
    );
}

unsafe extern "system" fn fake_get_physical_device_format_properties2(
    pd: avk::PhysicalDevice,
    format: avk::Format,
    out: *mut avk::FormatProperties2<'_>,
) {
    let dev = fake(pd);
    let legacy = dev
        .formats
        .get(&format.as_raw())
        .copied()
        .unwrap_or_default();
    (*out).format_properties = legacy;
    let widen =
        |f: avk::FormatFeatureFlags| avk::FormatFeatureFlags2::from_raw(u64::from(f.as_raw()));
    let mut next = (*out).p_next.cast::<avk::BaseOutStructure<'_>>();
    while !next.is_null() {
        if (*next).s_type == avk::FormatProperties3::STRUCTURE_TYPE {
            dev.formats_queried3.borrow_mut().push(format.as_raw());
            let (optimal, buffer) = dev
                .formats3
                .get(&format.as_raw())
                .copied()
                .unwrap_or_default();
            let p3 = avk::FormatProperties3 {
                linear_tiling_features: widen(legacy.linear_tiling_features),
                optimal_tiling_features: widen(legacy.optimal_tiling_features) | optimal,
                buffer_features: widen(legacy.buffer_features) | buffer,
                ..Default::default()
            };
            copy_keeping_next(next, &p3);
        }
        next = (*next).p_next;
    }
}

unsafe extern "system" fn fake_get_physical_device_image_format_properties2(
    pd: avk::PhysicalDevice,
    info: *const avk::PhysicalDeviceImageFormatInfo2<'_>,
    out: *mut avk::ImageFormatProperties2<'_>,
) -> avk::Result {
    let dev = fake(pd);
    let info = &*info;
    dev.images_queried.borrow_mut().push((
        info.format.as_raw(),
        info.ty.as_raw(),
        info.tiling.as_raw(),
        info.usage.as_raw(),
        info.flags.as_raw(),
    ));
    match dev.image_sample_counts.get(&info.format.as_raw()) {
        Some(&sample_counts) => {
            (*out).image_format_properties = avk::ImageFormatProperties {
                sample_counts,
                ..Default::default()
            };
            avk::Result::SUCCESS
        }
        None => avk::Result::ERROR_FORMAT_NOT_SUPPORTED,
    }
}

unsafe extern "system" fn fake_get_instance_proc_addr(
    _instance: avk::Instance,
    name: *const c_char,
) -> avk::PFN_vkVoidFunction {
    let name = CStr::from_ptr(name).to_bytes();
    // SAFETY (all arms): the loader casts each pointer back to the named entry point's type,
    // which is the type of the function returned.
    match name {
        b"vkGetPhysicalDeviceProperties" => Some(std::mem::transmute::<
            avk::PFN_vkGetPhysicalDeviceProperties,
            unsafe extern "system" fn(),
        >(fake_get_physical_device_properties)),
        b"vkGetPhysicalDeviceFeatures2" => Some(std::mem::transmute::<
            avk::PFN_vkGetPhysicalDeviceFeatures2,
            unsafe extern "system" fn(),
        >(fake_get_physical_device_features2)),
        b"vkGetPhysicalDeviceProperties2" => Some(std::mem::transmute::<
            avk::PFN_vkGetPhysicalDeviceProperties2,
            unsafe extern "system" fn(),
        >(fake_get_physical_device_properties2)),
        b"vkEnumerateDeviceExtensionProperties" => {
            Some(std::mem::transmute::<
                avk::PFN_vkEnumerateDeviceExtensionProperties,
                unsafe extern "system" fn(),
            >(fake_enumerate_device_extension_properties))
        }
        b"vkGetPhysicalDeviceFormatProperties" => {
            Some(std::mem::transmute::<
                avk::PFN_vkGetPhysicalDeviceFormatProperties,
                unsafe extern "system" fn(),
            >(fake_get_physical_device_format_properties))
        }
        b"vkGetPhysicalDeviceFormatProperties2" => {
            Some(std::mem::transmute::<
                avk::PFN_vkGetPhysicalDeviceFormatProperties2,
                unsafe extern "system" fn(),
            >(fake_get_physical_device_format_properties2))
        }
        b"vkGetPhysicalDeviceImageFormatProperties2" => Some(std::mem::transmute::<
            avk::PFN_vkGetPhysicalDeviceImageFormatProperties2,
            unsafe extern "system" fn(),
        >(
            fake_get_physical_device_image_format_properties2,
        )),
        _ => None,
    }
}

/// Every extension structure name the query knows.
const EVERY_EXTENSION: &[&CStr] = &[
    c"VK_EXT_image_robustness",
    c"VK_EXT_subgroup_size_control",
    c"VK_EXT_shader_demote_to_helper_invocation",
    c"VK_KHR_shader_terminate_invocation",
    c"VK_KHR_zero_initialize_workgroup_memory",
    c"VK_KHR_synchronization2",
    c"VK_KHR_maintenance4",
    c"VK_KHR_shader_subgroup_rotate",
    c"VK_KHR_dynamic_rendering_local_read",
    c"VK_KHR_vertex_attribute_divisor",
    c"VK_KHR_maintenance5",
    c"VK_EXT_robustness2",
    c"VK_EXT_shader_atomic_float",
    c"VK_EXT_shader_atomic_float2",
    c"VK_EXT_shader_image_atomic_int64",
    c"VK_KHR_workgroup_memory_explicit_layout",
    c"VK_KHR_shader_maximal_reconvergence",
    c"VK_KHR_shader_quad_control",
    c"VK_KHR_shader_subgroup_uniform_control_flow",
    c"VK_NV_compute_shader_derivatives",
    c"VK_KHR_fragment_shader_barycentric",
    c"VK_EXT_fragment_shader_interlock",
    c"VK_EXT_rasterization_order_attachment_access",
    c"VK_EXT_extended_dynamic_state",
    c"VK_EXT_extended_dynamic_state2",
    c"VK_EXT_extended_dynamic_state3",
    c"VK_EXT_attachment_feedback_loop_layout",
    c"VK_KHR_push_descriptor",
    c"VK_KHR_maintenance8",
    c"VK_KHR_shader_fma",
    c"VK_EXT_border_color_swizzle",
    c"VK_EXT_custom_border_color",
    c"VK_EXT_depth_bias_control",
    c"VK_EXT_depth_clip_enable",
    c"VK_EXT_depth_clamp_control",
    c"VK_EXT_primitive_topology_list_restart",
    c"VK_EXT_provoking_vertex",
    c"VK_EXT_line_rasterization",
    c"VK_EXT_sample_locations",
    c"VK_KHR_ray_query",
    c"VK_KHR_acceleration_structure",
    c"VK_KHR_deferred_host_operations",
    c"VK_KHR_portability_subset",
    c"VK_EXT_mesh_shader",
];

fn duplicates(chain: &[i32]) -> Vec<i32> {
    let mut seen = BTreeSet::new();
    chain
        .iter()
        .filter(|t| !seen.insert(**t))
        .copied()
        .collect()
}

/// A 1.3 device with every extension and every feature bit: every `VkFeature` is read, no
/// structure is chained twice, and no pre-1.3 structure is chained beside `Vulkan13Features`.
#[test]
fn every_feature_has_a_source() {
    let mut dev = FakeDevice::new(ApiVersion::V1_3);
    dev.extensions = EVERY_EXTENSION.to_vec();
    dev.all_features = true;
    let desc = dev.query().unwrap();
    let all: BTreeSet<VkFeature> = ALL_FEATURES.iter().copied().collect();
    assert_eq!(all.len(), ALL_FEATURES.len(), "a feature is listed twice");
    assert_eq!(desc.features, all);
    let chain = dev.feature_chain.borrow();
    assert!(duplicates(&chain).is_empty(), "{chain:?}");
    for promoted in [
        avk::PhysicalDeviceSubgroupSizeControlFeatures::STRUCTURE_TYPE,
        avk::PhysicalDeviceMaintenance4Features::STRUCTURE_TYPE,
        avk::PhysicalDeviceSynchronization2Features::STRUCTURE_TYPE,
        avk::PhysicalDeviceImageRobustnessFeatures::STRUCTURE_TYPE,
    ] {
        assert!(
            !chain.contains(&promoted.as_raw()),
            "{} beside Vulkan13Features",
            promoted.as_raw()
        );
    }
    // Every named extension the core consumes is recorded.
    for ext in [
        VkExtension::ExtRobustness2,
        VkExtension::KhrShaderMaximalReconvergence,
        VkExtension::KhrPushDescriptor,
        VkExtension::KhrComputeShaderDerivatives,
        VkExtension::KhrMaintenance8,
        VkExtension::KhrShaderFma,
        VkExtension::KhrMaintenance5,
        VkExtension::ExtBorderColorSwizzle,
        VkExtension::ExtCustomBorderColor,
        VkExtension::ExtDepthBiasControl,
        VkExtension::ExtDepthClipEnable,
        VkExtension::ExtDepthClampControl,
        VkExtension::ExtPrimitiveTopologyListRestart,
        VkExtension::ExtProvokingVertex,
        VkExtension::ExtSampleLocations,
    ] {
        assert!(desc.extensions.contains(&ext), "{ext:?}");
    }
    for structure in [
        avk::PhysicalDeviceDepthClipEnableFeaturesEXT::STRUCTURE_TYPE,
        DepthClampControlFeatures::STRUCTURE_TYPE,
        avk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT::STRUCTURE_TYPE,
        avk::PhysicalDeviceProvokingVertexFeaturesEXT::STRUCTURE_TYPE,
    ] {
        assert!(
            chain.contains(&structure.as_raw()),
            "{}",
            structure.as_raw()
        );
    }
    assert!(dev
        .property_chain
        .borrow()
        .contains(&avk::PhysicalDeviceSampleLocationsPropertiesEXT::STRUCTURE_TYPE.as_raw()));
    assert!(
        chain.contains(&avk::PhysicalDeviceDepthBiasControlFeaturesEXT::STRUCTURE_TYPE.as_raw())
    );
    assert!(chain.contains(&avk::PhysicalDeviceMaintenance5FeaturesKHR::STRUCTURE_TYPE.as_raw()));
    assert!(chain.contains(&Maintenance8Features::STRUCTURE_TYPE.as_raw()));
    assert!(chain.contains(&ShaderFmaFeatures::STRUCTURE_TYPE.as_raw()));
    assert!(
        chain.contains(&avk::PhysicalDeviceBorderColorSwizzleFeaturesEXT::STRUCTURE_TYPE.as_raw())
    );
}

/// A bare 1.2 device chains only the 1.1/1.2 core structures: nothing reaches the description
/// from a structure the device does not expose.
#[test]
fn unexposed_structures_are_never_chained() {
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.all_features = true;
    let desc = dev.query().unwrap();
    assert_eq!(
        *dev.feature_chain.borrow(),
        [
            avk::PhysicalDeviceVulkan12Features::STRUCTURE_TYPE.as_raw(),
            avk::PhysicalDeviceVulkan11Features::STRUCTURE_TYPE.as_raw(),
        ]
    );
    assert_eq!(
        *dev.property_chain.borrow(),
        [
            avk::PhysicalDeviceVulkan12Properties::STRUCTURE_TYPE.as_raw(),
            avk::PhysicalDeviceVulkan11Properties::STRUCTURE_TYPE.as_raw(),
        ]
    );
    for absent in [
        VkFeature::RobustBufferAccess2,
        VkFeature::RobustImageAccess,
        VkFeature::SubgroupSizeControl,
        VkFeature::Maintenance4,
        VkFeature::ShaderImageFloat32AtomicAdd,
        VkFeature::ExtendedDynamicState,
        VkFeature::ExtendedDynamicState2,
        VkFeature::ShaderSubgroupRotate,
        VkFeature::Maintenance8,
        VkFeature::ShaderFmaFloat16,
        VkFeature::ShaderFmaFloat32,
        VkFeature::Maintenance5,
        VkFeature::DepthBiasControl,
        VkFeature::FloatRepresentation,
        VkFeature::DepthBiasExact,
        VkFeature::DepthClipEnable,
        VkFeature::DepthClampControl,
        VkFeature::PrimitiveTopologyListRestart,
        VkFeature::PrimitiveTopologyPatchListRestart,
        VkFeature::ProvokingVertexLast,
        VkFeature::MultisampleArrayImage,
        VkFeature::ImageView2DOn3DImage,
        VkFeature::SeparateStencilMaskRef,
        VkFeature::ImageViewFormatSwizzle,
    ] {
        assert!(!desc.features.contains(&absent), "{absent:?}");
    }
    for present in [
        VkFeature::ShaderInt64,
        VkFeature::ShaderDrawParameters,
        VkFeature::BufferDeviceAddress,
        VkFeature::VulkanMemoryModel,
    ] {
        assert!(desc.features.contains(&present), "{present:?}");
    }
    assert!(desc.extensions.is_empty());
}

/// Below 1.3, the promoted extensions' own structures are read, for features and properties.
#[test]
fn pre_1_3_extension_structures_are_read() {
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.extensions = vec![
        c"VK_EXT_subgroup_size_control",
        c"VK_KHR_maintenance4",
        c"VK_EXT_extended_dynamic_state",
        c"VK_EXT_extended_dynamic_state2",
    ];
    dev.all_features = true;
    dev.subgroup_size_control.min_subgroup_size = 8;
    dev.subgroup_size_control.max_subgroup_size = 64;
    dev.subgroup_size_control.required_subgroup_size_stages = avk::ShaderStageFlags::COMPUTE;
    let desc = dev.query().unwrap();
    for f in [
        VkFeature::SubgroupSizeControl,
        VkFeature::ComputeFullSubgroups,
        VkFeature::Maintenance4,
        VkFeature::ExtendedDynamicState,
        VkFeature::ExtendedDynamicState2,
    ] {
        assert!(desc.features.contains(&f), "{f:?}");
    }
    assert!(desc
        .extensions
        .contains(&VkExtension::ExtExtendedDynamicState2));
    assert!(dev
        .feature_chain
        .borrow()
        .contains(&avk::PhysicalDeviceExtendedDynamicState2FeaturesEXT::STRUCTURE_TYPE.as_raw()));
    assert!(!desc.features.contains(&VkFeature::Synchronization2));
    assert_eq!((desc.subgroup.min_size, desc.subgroup.max_size), (8, 64));
    assert_eq!(
        desc.subgroup.required_size_stages,
        vk::ShaderStageFlags::COMPUTE
    );
    assert!(duplicates(&dev.feature_chain.borrow()).is_empty());
}

/// A 1.3 device under a 1.2 instance is a 1.2 device (the spec limits core functionality to
/// `min(instance, device)`): no `Vulkan13` structure is chained, the pre-1.3 extension
/// structures are read instead, and nothing is sourced from core 1.3.
#[test]
fn the_instance_version_caps_the_effective_version() {
    let v13_features = avk::PhysicalDeviceVulkan13Features::STRUCTURE_TYPE.as_raw();
    let v13_props = avk::PhysicalDeviceVulkan13Properties::STRUCTURE_TYPE.as_raw();
    let mut dev = FakeDevice::new(ApiVersion::V1_3);
    dev.instance_api = ApiVersion::V1_2;
    dev.extensions = vec![
        c"VK_EXT_subgroup_size_control",
        c"VK_KHR_maintenance4",
        c"VK_KHR_synchronization2",
    ];
    dev.all_features = true;
    dev.v13_props.min_subgroup_size = 4;
    dev.v13_props.max_buffer_size = 1 << 40;
    dev.subgroup_size_control.min_subgroup_size = 8;
    let desc = dev.query().unwrap();
    assert_eq!(desc.api_version, ApiVersion::V1_2);
    let features = dev.feature_chain.borrow();
    assert!(!features.contains(&v13_features), "{features:?}");
    for ext in [
        avk::PhysicalDeviceSubgroupSizeControlFeatures::STRUCTURE_TYPE,
        avk::PhysicalDeviceMaintenance4Features::STRUCTURE_TYPE,
        avk::PhysicalDeviceSynchronization2Features::STRUCTURE_TYPE,
    ] {
        assert!(features.contains(&ext.as_raw()), "{features:?}");
    }
    let props = dev.property_chain.borrow();
    assert!(!props.contains(&v13_props), "{props:?}");
    assert!(
        props.contains(&avk::PhysicalDeviceSubgroupSizeControlProperties::STRUCTURE_TYPE.as_raw())
    );
    assert_eq!(desc.subgroup.min_size, 8);
    drop((features, props));
    // Sourced from the extension structures only.
    for f in [
        VkFeature::SubgroupSizeControl,
        VkFeature::Maintenance4,
        VkFeature::Synchronization2,
    ] {
        assert!(desc.features.contains(&f), "{f:?}");
    }
    // Core in 1.3 only (no extension listed): absent at the effective 1.2.
    for f in [
        VkFeature::ExtendedDynamicState,
        VkFeature::ExtendedDynamicState2,
        VkFeature::RobustImageAccess,
        VkFeature::ShaderDemoteToHelperInvocation,
        VkFeature::ShaderTerminateInvocation,
        VkFeature::ShaderZeroInitializeWorkgroupMemory,
    ] {
        assert!(!desc.features.contains(&f), "{f:?}");
    }
    // The same device under a 1.3 instance chains `Vulkan13` and none of the 1.3 extensions.
    dev.instance_api = ApiVersion::V1_3;
    dev.feature_chain.borrow_mut().clear();
    dev.property_chain.borrow_mut().clear();
    let desc = dev.query().unwrap();
    assert_eq!(desc.api_version, ApiVersion::V1_3);
    assert!(desc.features.contains(&VkFeature::ExtendedDynamicState));
    assert!(desc.features.contains(&VkFeature::ExtendedDynamicState2));
    assert!(dev.feature_chain.borrow().contains(&v13_features));
    assert!(!dev
        .feature_chain
        .borrow()
        .contains(&avk::PhysicalDeviceMaintenance4Features::STRUCTURE_TYPE.as_raw()));
    assert_eq!(desc.subgroup.min_size, 4);
    // A 1.4 device under a 1.3 instance reads no 1.4-promoted structure it does not list.
    let mut dev = FakeDevice::new(ApiVersion { major: 1, minor: 4 });
    dev.instance_api = ApiVersion::V1_3;
    dev.all_features = true;
    let desc = dev.query().unwrap();
    assert_eq!(desc.api_version, ApiVersion::V1_3);
    for promoted in [
        avk::PhysicalDeviceShaderSubgroupRotateFeaturesKHR::STRUCTURE_TYPE,
        avk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR::STRUCTURE_TYPE,
        avk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR::STRUCTURE_TYPE,
        avk::PhysicalDeviceMaintenance5FeaturesKHR::STRUCTURE_TYPE,
    ] {
        assert!(!dev.feature_chain.borrow().contains(&promoted.as_raw()));
    }
    assert!(!desc.features.contains(&VkFeature::ShaderSubgroupRotate));
    assert!(!desc.features.contains(&VkFeature::Maintenance5));
    assert!(!dev
        .property_chain
        .borrow()
        .contains(&avk::PhysicalDevicePushDescriptorPropertiesKHR::STRUCTURE_TYPE.as_raw()));
}

/// Individual members reach the matching `VkFeature` (spot checks against a hand-set structure).
#[test]
fn members_map_to_their_features() {
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.v12_features = Some(avk::PhysicalDeviceVulkan12Features {
        shader_int8: avk::TRUE,
        scalar_block_layout: avk::TRUE,
        sampler_filter_minmax: avk::TRUE,
        ..Default::default()
    });
    let desc = dev.query().unwrap();
    assert_eq!(
        desc.features,
        [
            VkFeature::ShaderInt8,
            VkFeature::ScalarBlockLayout,
            VkFeature::SamplerFilterMinmax
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn properties_limits_and_float_controls_are_read() {
    let mut dev = FakeDevice::new(ApiVersion::V1_3);
    dev.extensions = vec![c"VK_KHR_push_descriptor"];
    dev.props.vendor_id = 0x10de;
    dev.props.device_id = 0x2204;
    dev.props.driver_version = 7;
    dev.props.device_type = avk::PhysicalDeviceType::DISCRETE_GPU;
    dev.props.limits.max_compute_work_group_invocations = 1024;
    dev.props.limits.max_compute_work_group_size = [1024, 1024, 64];
    dev.props.limits.max_image_dimension2_d = 32768;
    dev.props.limits.min_storage_buffer_offset_alignment = 16;
    dev.props.limits.max_sampler_anisotropy = 16.0;
    dev.props.limits.min_texel_offset = -8;
    dev.props.limits.framebuffer_color_sample_counts =
        avk::SampleCountFlags::TYPE_1 | avk::SampleCountFlags::TYPE_4;
    dev.v11_props.subgroup_size = 32;
    dev.v11_props.subgroup_supported_stages = avk::ShaderStageFlags::COMPUTE;
    dev.v11_props.subgroup_supported_operations =
        avk::SubgroupFeatureFlags::BASIC | avk::SubgroupFeatureFlags::BALLOT;
    dev.v11_props.max_memory_allocation_size = 1 << 32;
    dev.v12_props.driver_id = avk::DriverId::NVIDIA_PROPRIETARY;
    dev.v12_props.shader_signed_zero_inf_nan_preserve_float32 = avk::TRUE;
    dev.v12_props.denorm_behavior_independence = avk::ShaderFloatControlsIndependence::ALL;
    dev.v12_props
        .max_descriptor_set_update_after_bind_sampled_images = 1 << 20;
    dev.v12_props.filter_minmax_single_component_formats = avk::TRUE;
    dev.v13_props.min_subgroup_size = 32;
    dev.v13_props.max_subgroup_size = 32;
    dev.v13_props.max_compute_workgroup_subgroups = 32;
    dev.v13_props.max_buffer_size = 1 << 40;
    dev.push_descriptor.max_push_descriptors = 32;
    let d = dev.query().unwrap();
    assert_eq!(d.api_version, ApiVersion::V1_3);
    assert_eq!(
        (d.vendor_id, d.device_id, d.driver_version),
        (0x10de, 0x2204, 7)
    );
    assert_eq!(d.device_type, Some(vk::PhysicalDeviceType::DISCRETE_GPU));
    assert_eq!(d.driver_id, Some(vk::DriverId::NVIDIA_PROPRIETARY));
    let l = &d.limits;
    assert_eq!(l.max_compute_work_group_invocations, 1024);
    assert_eq!(l.max_compute_work_group_size, [1024, 1024, 64]);
    assert_eq!(l.max_image_dimension_2d, 32768);
    assert_eq!(l.min_storage_buffer_offset_alignment, 16);
    assert_eq!(f32::from_bits(l.max_sampler_anisotropy_bits), 16.0);
    assert_eq!(l.min_texel_offset, -8);
    assert_eq!(
        l.framebuffer_color_sample_counts,
        vk::SampleCountFlags::COUNT_1 | vk::SampleCountFlags::COUNT_4
    );
    assert_eq!(
        d.allocation,
        AllocationLimits::new(1 << 32, 1 << 40),
        "the allocation sizes are the description's, not the profile's"
    );
    assert_eq!(l.max_push_descriptors, 32);
    assert_eq!(
        l.max_descriptor_set_update_after_bind_sampled_images,
        1 << 20
    );
    assert_eq!(l.max_vertex_attrib_divisor, 0);
    assert!(!l.supports_non_zero_first_instance);
    assert_eq!(d.subgroup.size, 32);
    assert_eq!(d.subgroup.supported_stages, vk::ShaderStageFlags::COMPUTE);
    assert_eq!(
        d.subgroup.supported_operations,
        vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::BALLOT
    );
    assert_eq!((d.subgroup.min_size, d.subgroup.max_size), (32, 32));
    assert_eq!(d.subgroup.max_compute_workgroup_subgroups, 32);
    assert!(d.float_controls.f32.signed_zero_inf_nan_preserve);
    assert!(!d.float_controls.f16.signed_zero_inf_nan_preserve);
    assert_eq!(d.float_controls.denorm_independence, Independence::All);
    assert_eq!(
        d.float_controls.rounding_independence,
        Independence::Only32Bit
    );
    assert!(d.filter_minmax_single_component_formats);
}

/// `supportsNonZeroFirstInstance` is read from the KHR divisor properties; a device exposing only
/// the EXT divisor cannot report it and reads `false`.
#[test]
fn divisor_first_instance_support_is_read() {
    let khr_type = avk::PhysicalDeviceVertexAttributeDivisorPropertiesKHR::STRUCTURE_TYPE.as_raw();
    let ext_type = avk::PhysicalDeviceVertexAttributeDivisorPropertiesEXT::STRUCTURE_TYPE.as_raw();
    for supports in [false, true] {
        let mut dev = FakeDevice::new(ApiVersion::V1_2);
        dev.extensions = vec![
            c"VK_EXT_vertex_attribute_divisor",
            c"VK_KHR_vertex_attribute_divisor",
        ];
        dev.all_features = true;
        dev.divisor_khr.max_vertex_attrib_divisor = 8;
        dev.divisor_khr.supports_non_zero_first_instance = supports.into();
        dev.divisor_ext.max_vertex_attrib_divisor = 99;
        let d = dev.query().unwrap();
        assert!(d.has(VkFeature::VertexAttributeInstanceRateDivisor));
        assert_eq!(d.limits.max_vertex_attrib_divisor, 8);
        assert_eq!(d.limits.supports_non_zero_first_instance, supports);
        let chain = dev.property_chain.borrow();
        assert!(
            chain.contains(&khr_type) && !chain.contains(&ext_type),
            "{chain:?}"
        );
    }
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.extensions = vec![c"VK_EXT_vertex_attribute_divisor"];
    dev.all_features = true;
    dev.divisor_ext.max_vertex_attrib_divisor = 16;
    let d = dev.query().unwrap();
    assert_eq!(d.limits.max_vertex_attrib_divisor, 16);
    assert!(!d.limits.supports_non_zero_first_instance);
    let chain = dev.property_chain.borrow();
    assert!(
        chain.contains(&ext_type) && !chain.contains(&khr_type),
        "{chain:?}"
    );
}

/// `minVertexInputBindingStrideAlignment` is read exactly where `VK_KHR_portability_subset` is
/// exposed; other devices have no stride rule.
#[test]
fn portability_stride_alignment_is_read() {
    let ty = avk::PhysicalDevicePortabilitySubsetPropertiesKHR::STRUCTURE_TYPE.as_raw();
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.extensions = vec![c"VK_KHR_portability_subset"];
    dev.portability.min_vertex_input_binding_stride_alignment = 4;
    let d = dev.query().unwrap();
    assert_eq!(d.limits.min_vertex_input_binding_stride_alignment, Some(4));
    assert!(dev.property_chain.borrow().contains(&ty));
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.portability.min_vertex_input_binding_stride_alignment = 4;
    let d = dev.query().unwrap();
    assert_eq!(d.limits.min_vertex_input_binding_stride_alignment, None);
    assert!(!dev.property_chain.borrow().contains(&ty));
}

/// The acceleration-structure descriptor limits are read exactly where
/// `VK_KHR_acceleration_structure` is exposed; elsewhere they are 0.
#[test]
fn acceleration_structure_limits_are_read() {
    let ty = avk::PhysicalDeviceAccelerationStructurePropertiesKHR::STRUCTURE_TYPE.as_raw();
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.extensions = vec![
        c"VK_KHR_acceleration_structure",
        c"VK_KHR_deferred_host_operations",
    ];
    dev.acceleration_structure
        .max_per_stage_descriptor_acceleration_structures = 16;
    dev.acceleration_structure
        .max_descriptor_set_acceleration_structures = 32;
    let d = dev.query().unwrap();
    assert_eq!(
        d.limits.max_per_stage_descriptor_acceleration_structures,
        16
    );
    assert_eq!(d.limits.max_descriptor_set_acceleration_structures, 32);
    assert!(dev.property_chain.borrow().contains(&ty));
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.acceleration_structure
        .max_per_stage_descriptor_acceleration_structures = 16;
    let d = dev.query().unwrap();
    assert_eq!(d.limits.max_per_stage_descriptor_acceleration_structures, 0);
    assert!(!dev.property_chain.borrow().contains(&ty));
}

/// `rgba4OpaqueBlackSwizzled` is read exactly where `VK_KHR_maintenance10` is exposed (the
/// hand-declared structure `ash 0.38` lacks); elsewhere it is false.
#[test]
fn rgba4_opaque_black_swizzled_is_read() {
    let ty = Maintenance10Properties::STRUCTURE_TYPE.as_raw();
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.extensions = vec![c"VK_KHR_maintenance10"];
    dev.maintenance10.rgba4_opaque_black_swizzled = avk::TRUE;
    let d = dev.query().unwrap();
    assert!(d.rgba4_opaque_black_swizzled);
    assert!(dev.property_chain.borrow().contains(&ty));
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.maintenance10.rgba4_opaque_black_swizzled = avk::TRUE;
    let d = dev.query().unwrap();
    assert!(!d.rgba4_opaque_black_swizzled);
    assert!(!dev.property_chain.borrow().contains(&ty));
}

/// `robustBufferAccessUpdateAfterBind` is read from `VkPhysicalDeviceVulkan12Properties`, which
/// every description's version chains.
#[test]
fn robust_buffer_access_update_after_bind_is_read() {
    for reported in [false, true] {
        let mut dev = FakeDevice::new(ApiVersion::V1_2);
        dev.v12_props.robust_buffer_access_update_after_bind = reported.into();
        let d = dev.query().unwrap();
        assert_eq!(d.robust_buffer_access_update_after_bind, reported);
    }
}

/// Exactly the formats `FormatSupport::queries_for` the device's version and available
/// extensions is asked about, and formats without features are omitted.
#[test]
fn formats_are_queried_exactly() {
    let asked = |dev: &FakeDevice| -> Vec<vk::Format> {
        dev.formats_queried
            .borrow()
            .iter()
            .map(|&f| vk::Format::from_raw(f))
            .collect()
    };
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    let sampled = avk::FormatProperties {
        optimal_tiling_features: avk::FormatFeatureFlags::SAMPLED_IMAGE,
        buffer_features: avk::FormatFeatureFlags::VERTEX_BUFFER,
        ..Default::default()
    };
    dev.formats
        .insert(avk::Format::R8G8B8A8_UNORM.as_raw(), sampled);
    let d = dev.query().unwrap();
    let plain = asked(&dev);
    assert_eq!(
        plain,
        FormatSupport::queries_for(ApiVersion::V1_2, &BTreeSet::new())
    );
    assert!(!plain.contains(&vk::Format::A8_UNORM));
    assert_eq!(d.formats.formats.len(), 1);
    assert_eq!(
        d.formats.formats[&vk::Format::R8G8B8A8_UNORM],
        FormatProperties {
            linear_tiling: vk::FormatFeatureFlags2::empty(),
            optimal_tiling: vk::FormatFeatureFlags2::SAMPLED_IMAGE,
            buffer: vk::FormatFeatureFlags2::VERTEX_BUFFER,
            color_attachment_sample_counts: vk::SampleCountFlags::empty(),
        }
    );
    // An available extension adds the formats it defines; a 1.4 device has `A8_UNORM` and the
    // ASTC HDR blocks in core, PVRTC only with its extension.
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.extensions = vec![c"VK_IMG_format_pvrtc"];
    dev.query().unwrap();
    let pvrtc = asked(&dev);
    assert!(pvrtc.contains(&vk::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG));
    assert!(!pvrtc.contains(&vk::Format::ASTC_4X4_SFLOAT_BLOCK));
    let dev = FakeDevice::new(ApiVersion { major: 1, minor: 4 });
    dev.query().unwrap();
    let v14 = dev
        .formats_queried3
        .borrow()
        .iter()
        .map(|&f| vk::Format::from_raw(f))
        .collect::<Vec<_>>();
    assert!(asked(&dev).is_empty());
    assert!(v14.contains(&vk::Format::A8_UNORM));
    assert!(v14.contains(&vk::Format::ASTC_4X4_SFLOAT_BLOCK));
    assert!(!v14.contains(&vk::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG));
}

/// A format whose optimal tiling has `COLOR_ATTACHMENT` takes the sample counts
/// `vkGetPhysicalDeviceImageFormatProperties2` reports for a 2D, optimal-tiling color attachment
/// without flags, and none where the query refuses it; no other format is asked.
#[test]
fn color_attachment_sample_counts_are_read_per_color_attachment_format() {
    let color = avk::FormatProperties {
        optimal_tiling_features: avk::FormatFeatureFlags::COLOR_ATTACHMENT,
        ..Default::default()
    };
    let sampled = avk::FormatProperties {
        optimal_tiling_features: avk::FormatFeatureFlags::SAMPLED_IMAGE,
        ..Default::default()
    };
    let (integer, refused, plain) = (
        avk::Format::R32G32B32A32_UINT,
        avk::Format::R16G16B16A16_UINT,
        avk::Format::R8_UNORM,
    );
    let mut dev = FakeDevice::new(ApiVersion::V1_2);
    dev.formats.insert(integer.as_raw(), color);
    dev.formats.insert(refused.as_raw(), color);
    dev.formats.insert(plain.as_raw(), sampled);
    let counts = avk::SampleCountFlags::TYPE_1
        | avk::SampleCountFlags::TYPE_2
        | avk::SampleCountFlags::TYPE_4
        | avk::SampleCountFlags::TYPE_8;
    dev.image_sample_counts.insert(integer.as_raw(), counts);
    dev.image_sample_counts
        .insert(plain.as_raw(), avk::SampleCountFlags::TYPE_1);
    let d = dev.query().unwrap();
    let of = |format: avk::Format| {
        d.formats.formats[&vk::Format::from_raw(format.as_raw())].color_attachment_sample_counts
    };
    assert_eq!(of(integer).as_raw(), counts.as_raw());
    assert!(of(refused).is_empty());
    assert!(of(plain).is_empty());
    let asked = dev.images_queried.borrow();
    let wanted = |format: avk::Format| {
        (
            format.as_raw(),
            avk::ImageType::TYPE_2D.as_raw(),
            avk::ImageTiling::OPTIMAL.as_raw(),
            avk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw(),
            0,
        )
    };
    let mut expected = vec![wanted(integer), wanted(refused)];
    expected.sort_by_key(|q| q.0);
    let mut asked = asked.clone();
    asked.sort_by_key(|q| q.0);
    assert_eq!(asked, expected);
}

/// `VkFormatProperties3` is read where Vulkan 1.3 or `VK_KHR_format_feature_flags2` has it, so
/// the storage-without-format bits of each format reach the description; elsewhere
/// `VkFormatProperties` is, and the description says the bits are not reported.
#[test]
fn format_feature_flags2_are_read_where_available() {
    let without = avk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT;
    let storage = avk::FormatProperties {
        optimal_tiling_features: avk::FormatFeatureFlags::STORAGE_IMAGE,
        buffer_features: avk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER,
        ..Default::default()
    };
    let r32 = avk::Format::R32_UINT.as_raw();
    let with = |dev: &mut FakeDevice| {
        dev.formats.insert(r32, storage);
        dev.formats3
            .insert(r32, (without, avk::FormatFeatureFlags2::empty()));
    };
    let expected = FormatProperties {
        linear_tiling: vk::FormatFeatureFlags2::empty(),
        optimal_tiling: vk::FormatFeatureFlags2::STORAGE_IMAGE
            | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT,
        buffer: vk::FormatFeatureFlags2::STORAGE_TEXEL_BUFFER,
        color_attachment_sample_counts: vk::SampleCountFlags::empty(),
    };
    // The extension under a 1.2 instance.
    let mut dev = FakeDevice::new(ApiVersion::V1_3);
    dev.instance_api = ApiVersion::V1_2;
    dev.extensions = vec![c"VK_KHR_format_feature_flags2"];
    with(&mut dev);
    let d = dev.query().unwrap();
    assert!(d.has_format_feature_flags2());
    assert_eq!(d.formats.formats[&vk::Format::R32_UINT], expected);
    assert!(dev.formats_queried.borrow().is_empty());
    // Vulkan 1.3.
    let mut dev = FakeDevice::new(ApiVersion::V1_3);
    with(&mut dev);
    let d = dev.query().unwrap();
    assert!(d.has_format_feature_flags2());
    assert_eq!(d.formats.formats[&vk::Format::R32_UINT], expected);
    // Neither: `VkFormatProperties` alone.
    let mut dev = FakeDevice::new(ApiVersion::V1_3);
    dev.instance_api = ApiVersion::V1_2;
    with(&mut dev);
    let d = dev.query().unwrap();
    assert!(!d.has_format_feature_flags2());
    assert!(dev.formats_queried3.borrow().is_empty());
    assert_eq!(
        d.formats.formats[&vk::Format::R32_UINT],
        FormatProperties::from_legacy(
            vk::FormatFeatureFlags::empty(),
            vk::FormatFeatureFlags::STORAGE_IMAGE,
            vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER,
        )
    );
}

#[test]
fn devices_below_the_floor_are_refused() {
    let refused = Err(Unsupported::needs(
        Feature::VulkanVersion,
        Capability::ApiVersion(ApiVersion::V1_2),
    )
    .into());
    let dev = FakeDevice::new(ApiVersion { major: 1, minor: 1 });
    assert_eq!(dev.query(), refused);
    assert!(dev.feature_chain.borrow().is_empty());
    // A 1.3 device under a 1.1 instance: nothing past the 1.0 properties query is called.
    let mut dev = FakeDevice::new(ApiVersion::V1_3);
    dev.instance_api = ApiVersion { major: 1, minor: 1 };
    assert_eq!(dev.query(), refused);
    assert!(dev.feature_chain.borrow().is_empty());
    assert!(dev.property_chain.borrow().is_empty());
}

/// A queried description derives a device.
#[test]
fn a_queried_description_derives_a_device() {
    let mut dev = FakeDevice::new(ApiVersion::V1_3);
    dev.extensions = EVERY_EXTENSION.to_vec();
    dev.all_features = true;
    dev.props.limits.max_compute_work_group_invocations = 1024;
    dev.props.limits.max_compute_work_group_size = [1024, 1024, 64];
    dev.props.limits.max_compute_work_group_count = [65535; 3];
    dev.props.limits.max_push_constants_size = 256;
    dev.props.limits.max_bound_descriptor_sets = 8;
    dev.props.limits.max_storage_buffer_range = u32::MAX;
    dev.props.limits.max_uniform_buffer_range = 65536;
    dev.props.limits.min_storage_buffer_offset_alignment = 16;
    dev.props.limits.min_uniform_buffer_offset_alignment = 64;
    dev.props.limits.max_compute_shared_memory_size = 32768;
    dev.v11_props.subgroup_size = 32;
    dev.v11_props.subgroup_supported_stages = avk::ShaderStageFlags::ALL;
    dev.v11_props.subgroup_supported_operations = avk::SubgroupFeatureFlags::from_raw(0xff);
    let desc = dev.query().unwrap();
    Device::new(desc, DevicePolicy::default()).expect("queried description derives");
}

#[test]
fn values_convert_both_ways() {
    for &f in vk::Format::KNOWN {
        let a: avk::Format = f.into();
        assert_eq!(a.as_raw(), f.as_raw());
        assert_eq!(vk::Format::from(a), f);
    }
    // A value the pinned registry does not name converts unchanged.
    let unknown = vk::Format::from_raw(0x7fff_0000);
    assert_eq!(vk::Format::from(avk::Format::from(unknown)), unknown);
    for &(flags, _) in vk::AccessFlags2::KNOWN {
        let a: avk::AccessFlags2 = flags.into();
        assert_eq!(vk::AccessFlags2::from(a), flags);
    }
    for &stage in vk::ShaderStage::KNOWN {
        let a: avk::ShaderStageFlags = stage.into();
        assert_eq!(
            vk::ShaderStage::try_from(a).map_err(|f| f.as_raw()),
            Ok(stage)
        );
    }
    assert!(vk::ShaderStage::try_from(avk::ShaderStageFlags::ALL_GRAPHICS).is_err());
    let m = vk::ComponentMapping {
        r: vk::ComponentSwizzle::B,
        ..vk::ComponentMapping::IDENTITY
    };
    assert_eq!(
        vk::ComponentMapping::from(avk::ComponentMapping::from(m)),
        m
    );
    let e: avk::SpecializationMapEntry = vk::SpecializationMapEntry {
        constant_id: 3,
        offset: 8,
        size: 4,
    }
    .into();
    assert_eq!((e.constant_id, e.offset, e.size), (3, 8, 4));
}

#[test]
fn sampler_create_info_chains_reduction_only_when_needed() {
    let desc = vk::SamplerDesc {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::NEAREST,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
        mip_lod_bias: 0.0,
        max_anisotropy: Some(8.0),
        compare: Some(vk::CompareOp::LESS),
        min_lod: 0.0,
        max_lod: 4.0,
        border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
        reduction: vk::SamplerReductionMode::WEIGHTED_AVERAGE,
        border_components: None,
    };
    let mut reduction = avk::SamplerReductionModeCreateInfo::default();
    let mut border = avk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
    let info = sampler_create_info(&desc, &mut reduction, &mut border);
    assert!(info.p_next.is_null());
    assert_eq!(info.mag_filter.as_raw(), avk::Filter::LINEAR.as_raw());
    assert_eq!(
        info.address_mode_u.as_raw(),
        avk::SamplerAddressMode::CLAMP_TO_BORDER.as_raw()
    );
    assert_eq!(
        (info.anisotropy_enable, info.max_anisotropy),
        (avk::TRUE, 8.0)
    );
    assert_eq!(
        (info.compare_enable, info.compare_op.as_raw()),
        (avk::TRUE, avk::CompareOp::LESS.as_raw())
    );
    assert_eq!(info.unnormalized_coordinates, avk::FALSE);
    let min = vk::SamplerDesc {
        reduction: vk::SamplerReductionMode::MIN,
        compare: None,
        ..desc
    };
    let mut reduction = avk::SamplerReductionModeCreateInfo::default();
    let mut border = avk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
    let info = sampler_create_info(&min, &mut reduction, &mut border);
    assert!(!info.p_next.is_null());
    assert_eq!(info.compare_enable, avk::FALSE);
    let chained = info
        .p_next
        .cast::<avk::SamplerReductionModeCreateInfo<'_>>();
    // SAFETY: `p_next` points at `reduction`, alive for this scope.
    let mode = unsafe { (*chained).reduction_mode };
    assert_eq!(mode.as_raw(), avk::SamplerReductionMode::MIN.as_raw());
    // A border mapping is chained with `srgb` false.
    let argb = vk::ComponentMapping {
        r: vk::ComponentSwizzle::A,
        g: vk::ComponentSwizzle::R,
        b: vk::ComponentSwizzle::G,
        a: vk::ComponentSwizzle::B,
    };
    let mapped = vk::SamplerDesc {
        border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
        border_components: Some(argb),
        ..desc
    };
    let mut reduction = avk::SamplerReductionModeCreateInfo::default();
    let mut border = avk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
    let info = sampler_create_info(&mapped, &mut reduction, &mut border);
    let chained = info
        .p_next
        .cast::<avk::SamplerBorderColorComponentMappingCreateInfoEXT<'_>>();
    // SAFETY: `p_next` points at `border`, alive for this scope.
    let (s_type, components, srgb) =
        unsafe { ((*chained).s_type, (*chained).components, (*chained).srgb) };
    // `ash`'s types here have no `Debug`.
    assert!(s_type == avk::StructureType::SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT);
    assert!(components.r == avk::ComponentSwizzle::A);
    assert!(components.g == avk::ComponentSwizzle::R);
    assert!(components.b == avk::ComponentSwizzle::G);
    assert!(components.a == avk::ComponentSwizzle::B);
    assert_eq!(srgb, avk::FALSE);
}
