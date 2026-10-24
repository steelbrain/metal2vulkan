//! The device a pipeline is compiled for: what the consumer enabled, and its policy.
//!
//! A [`DeviceDescription`] states one Vulkan physical device as the consumer creates it: its API
//! version, the features and extensions it enables, its limits, subgroup and float-controls
//! properties, and format support. Build one field by field from [`DeviceDescription::new`], read
//! one from a Khronos profiles JSON document with [`DeviceDescription::from_khronos_json`], or,
//! with feature `ash`, query a physical device with `DeviceDescription::query`. A queried
//! description states what the device supports: enable (at least)
//! [`DeviceDescription::recommended_enables`] at `vkCreateDevice`, merged into your own list
//! within Vulkan's co-enable rules, and pass the description through
//! [`DeviceDescription::with_enabled`] (which refuses a set that breaks one) so that it states
//! what is enabled. Translation uses only what the description enables.
//!
//! [`Device::new`] derives an opaque [`DeviceProfile`] from the description and a
//! [`DevicePolicy`], the consumer's product choices (the default, [`DevicePolicy::UNTRUSTED`],
//! contains and isolates guest shaders and bounds their loops). It refuses a device below Vulkan
//! 1.2 and an inconsistent description or policy. A [`Device`] is cheap to clone, and every
//! compile and plan takes one:
//!
//! - [`Device::compile_function`], [`Device::compile_compute_pipeline`],
//!   [`Device::compile_render_pipeline`], and their keys (see [`pipeline`](crate::pipeline));
//! - [`Device::required_enables`]: everything an artifact compiled for the device can need
//!   enabled;
//! - [`Device::metal_properties`]: the Metal device properties to report to the guest;
//! - [`Device::neutral_kinds`]: the neutral resources plans bind where nothing compatible is
//!   bound, to create once up front;
//! - [`DeviceProfile::fingerprint`]: a hash over every profile field, policy included, which
//!   every cache key includes, so an artifact never serves another device.
//!
//! A consumer that creates its own `VkDevice` sets each feature of an [`EnableSet`] in the
//! structure that holds it at the device's version ([`VkFeature::exposure`], an [`Exposure`]),
//! and enables each extension with the ones it requires ([`VkExtension::requires`]);
//! [`VkFeature::ALL`] lists the vocabulary, for a table of its own over it.
//!
//! A pipeline's [`Requirements`] state what its modules need of the device, in the vocabulary of
//! [`VkFeature`], [`VkExtension`], [`VkLimit`], [`VkProperty`], [`FormatUsage`], and
//! [`SpirvCapability`]. [`DeviceProfile::check`] says whether a profile provides them, naming the
//! first missing [`Capability`](crate::error::Capability); a refusal names capabilities in the
//! same vocabulary.
//!
//! The frozen surface is the items named at this root; the submodules are crate-private.

pub(crate) mod check;
pub(crate) mod derive;
pub(crate) mod description;
pub(crate) mod driver_facts;
pub(crate) mod enables;
pub(crate) mod function_storage;
pub(crate) mod gpus;
pub(crate) mod khronos;
pub(crate) mod mechanism;
pub(crate) mod metal;
pub(crate) mod policy;
pub(crate) mod profile;
pub(crate) mod requirements;
pub(crate) mod subgroup;

pub use derive::{vk_stage, Device};
pub use description::{
    AllocationLimits, DeviceDescription, FormatProperties, FormatSupport,
    GeneratedCommandsProperties, SubgroupProperties,
};
pub use driver_facts::{DriverDiagnostic, DriverFact};
pub use enables::{EnableSet, Exposure};
// The mechanism derivation: internal; a pipeline states each mechanism in its `dependencies`.
pub(crate) use mechanism::mechanism;
pub use metal::{MetalDeviceProperties, SimdSupport, StageSimd};
pub use policy::{
    BatchOrderPolicy, CompletionPolicy, ContainmentPolicy, DecodedTexelsPolicy, DevicePolicy,
    FloatMathsPolicy, HalfBlendPolicy, HalfRoundingPolicy, HalfTexelBitsPolicy,
    HeapCompressionPolicy, NanMinMaxPolicy,
    RenderLoweringPolicy, ResolvePolicy, RoundingOrderPolicy, SamplingPolicy,
    TexelConversionPolicy, TileStorePolicy, Unorm8LinearFilterPolicy, YCbCrFilterPolicy,
};
pub(crate) use profile::whole_point_size;
pub use profile::{DeviceProfile, SpirvValidation};
// The vocabulary the device surface reaches: description and policy inputs, and the requirement
// vocabulary of `Requirements`, `Capability`, and `DeviceDescription`.
pub use profile::{
    EnabledCaps, FloatControls, FloatWidthControls, FragmentCaps, GuestIsolation, HandleDomain,
    ImageCaps, Independence, Limits, MeshLimits, PointerDomain,
};
// The profile's facts a consumer or a validation tool reads: its descriptor capabilities, its
// subgroup model and simdgroup tiers, how it contains an out-of-bounds access, and the Function
// storage a thread holds.
pub use profile::{Containment, DescriptorCaps, FunctionStorage, Robustness};
pub use subgroup::{SimdClass, StageSubgroup, SubgroupModel, SubgroupSizes, Tier};
// The driver defect a refusal names (`Needs::DriverDefect`).
pub use profile::DriverWorkaround;
// The GPU generation a driver workaround and a runtime driver defect key on.
pub use gpus::NVIDIA_AMPERE_GA10X;
pub use requirements::{
    ApiVersion, FloatControl, FloatWidth, FormatUsage, Requirements, ResolveMode, SpirvCapability,
    SpirvExtension, StageCreateFlag, SubgroupOp, VkExtension, VkFeature, VkLimit, VkProperty,
};
pub use subgroup::OpBits;

#[cfg(test)]
mod tests;
