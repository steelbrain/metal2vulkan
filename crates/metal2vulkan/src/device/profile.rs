//! `DeviceProfile` and its translation-relevant fields.
//!
//! `Device::from_queries(desc, policy)` is the only producer of a [`DeviceProfile`]. The profile is the
//! normalized projection of a device description plus the policy; lowering selects representations
//! from it and nothing else. Derivation lives in `device::derive`, the fingerprint in
//! [`DeviceProfile::fingerprint`].

use std::collections::{BTreeMap, BTreeSet};

use crate::air::container::Hash256;
use crate::device::requirements::{ApiVersion, FormatUsage, VkExtension, VkFeature, VkProperty};
use crate::device::subgroup::{SimdClass, SubgroupModel, Tier};
use crate::error::{Capability, Unsupported};
use crate::memory::contain::Check;
use crate::pipeline::sym::Stage;
use crate::vk;

/// What `spirv-val` validates a module compiled for a device under
/// ([`DeviceProfile::spirv_validation`]): run `spirv-val --target-env <env>` with the relaxations
/// for each environment.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpirvValidation {
    /// The `--target-env` values: `vulkan1.2`, every module's floor, then the device's own API
    /// environment where it is newer (`vulkan1.4` for any device from 1.4, the newest environment
    /// `spirv-val` knows).
    pub target_envs: Vec<&'static str>,
    /// The `spirv-val` arguments that relax a rule a feature the device enables lifts:
    /// `--scalar-block-layout` (`scalarBlockLayout`), `--uniform-buffer-standard-layout`
    /// (`uniformBufferStandardLayout`), `--workgroup-scalar-block-layout`
    /// (`workgroupMemoryExplicitLayoutScalarBlockLayout`), `--allow-localsizeid` (`maintenance4`)
    /// and `--allow-offset-texture-operand` (`maintenance8`), in that order.
    pub relaxations: Vec<&'static str>,
}

/// The `spirv-val` relaxations, each after the feature whose enablement lifts its rule
/// ([`SpirvValidation::relaxations`]).
const SPIRV_VAL_RELAXATIONS: [(VkFeature, &str); 5] = [
    (VkFeature::ScalarBlockLayout, "--scalar-block-layout"),
    (
        VkFeature::UniformBufferStandardLayout,
        "--uniform-buffer-standard-layout",
    ),
    (
        VkFeature::WorkgroupMemoryExplicitLayoutScalarBlockLayout,
        "--workgroup-scalar-block-layout",
    ),
    (VkFeature::Maintenance4, "--allow-localsizeid"),
    (VkFeature::Maintenance8, "--allow-offset-texture-operand"),
];

/// The normalized, translation-relevant device projection. Opaque; built only by
/// `Device::from_queries`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceProfile {
    fields: ProfileFields,
    /// [`DeviceProfile::fingerprint`], derived once here, where the fields are fixed.
    ///
    /// Deriving it renders every field of the profile -- a format table alone is tens of
    /// kilobytes -- and SHA-256s the rendering, and a stage's compile takes it twice: once for the
    /// compiled function's key and once to stamp the artifact with the profile it was compiled
    /// for. Those two are inside the lowering, which holds the profile and not the device, so
    /// neither could reach `Device::profile_fingerprint`'s memo. The profile is immutable once
    /// built, so the derivation belongs here; a clone carries the same fields and so the same
    /// fingerprint.
    fingerprint: Hash256,
}

impl DeviceProfile {
    /// The profile of `fields`, with its fingerprint derived (the only way to build one): the
    /// fingerprint is a function of the fields, so holding them together is what keeps it true of
    /// them. A test's edit goes through `edit`, which only the crate's tests build.
    pub(crate) fn of(fields: ProfileFields) -> DeviceProfile {
        let fingerprint = fingerprint_of(&fields);
        DeviceProfile {
            fields,
            fingerprint,
        }
    }

    /// Apply `change` to this profile's fields and derive its fingerprint again: the only way to
    /// move a field, so no profile can carry a fingerprint of fields it no longer holds. The
    /// crate's own tests build the profiles a venue does not ship.
    #[cfg(test)]
    pub(crate) fn edit<T>(&mut self, change: impl FnOnce(&mut ProfileFields) -> T) -> T {
        let answer = change(&mut self.fields);
        self.fingerprint = fingerprint_of(&self.fields);
        answer
    }

    /// The profile's fields: internal; the profile is opaque to consumers, who compare
    /// [`DeviceProfile::fingerprint`] and read its facts through the accessors.
    pub(crate) fn fields(&self) -> &ProfileFields {
        &self.fields
    }

    /// What the consumer enabled on the device, as the capability vocabulary
    /// [`Requirements`](crate::device::Requirements) uses: the features, extensions and
    /// properties a module compiled for the profile may rely on.
    pub fn enabled(&self) -> &EnabledCaps {
        &self.fields.enabled
    }

    /// The device limits the profile was built with (`VkPhysicalDeviceLimits` and the extension
    /// properties it reads).
    pub fn limits(&self) -> &Limits {
        &self.fields.limits
    }

    /// The fragment-stage and fixed-function capabilities the profile enables.
    pub fn fragment(&self) -> &FragmentCaps {
        &self.fields.fragment
    }

    /// The image and sampler capabilities the profile enables.
    pub fn images(&self) -> &ImageCaps {
        &self.fields.images
    }

    /// Whether the profile reads and writes a storage image or storage texel buffer through a
    /// declaration without a format by its format's own `STORAGE_READ_WITHOUT_FORMAT` /
    /// `STORAGE_WRITE_WITHOUT_FORMAT` bit, whatever `shaderStorageImage{Read,Write}WithoutFormat`
    /// say: at Vulkan 1.3, or with `VK_KHR_format_feature_flags2` enabled, each an enablement of
    /// the SPIR-V capabilities `StorageImageReadWithoutFormat` and `StorageImageWriteWithoutFormat`
    /// beside the features. The format tables then hold the bits
    /// (`DeviceDescription::has_format_feature_flags2`), which govern each format's access.
    pub(crate) fn without_format_by_format_bits(&self) -> bool {
        self.fields.api_version >= ApiVersion::V1_3
            || self
                .fields
                .enabled
                .extensions
                .contains(&VkExtension::KhrFormatFeatureFlags2)
    }

    /// Whether the profile reads storage through a declaration without a format by the formats'
    /// own bits alone: it lacks `shaderStorageImageReadWithoutFormat`, and Vulkan 1.3 or
    /// `VK_KHR_format_feature_flags2` enables the SPIR-V capability `StorageImageReadWithoutFormat`
    /// ([`DeviceProfile::without_format_by_format_bits`]), so a read of a view without a format is
    /// valid exactly where the view's format has `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT`
    /// (`VUID-vkCmdDispatch-OpTypeImage-07028`). ANV before Gfx12.5 is such a device.
    pub(crate) fn storage_reads_by_format_bits(&self) -> bool {
        !self.fields.images.storage_read_without_format && self.without_format_by_format_bits()
    }

    /// Whether the profile can read (`read`) and write (`write`) storage through a declaration
    /// without a format at all: by the device's `shaderStorageImage{Read,Write}WithoutFormat`, or
    /// by the formats' own bits ([`DeviceProfile::without_format_by_format_bits`]). Which formats
    /// it may then view is the format tables' (`metal::format::without_format_gaps`).
    pub(crate) fn accesses_without_format(&self, read: bool, write: bool) -> bool {
        let images = &self.fields.images;
        let bits = self.without_format_by_format_bits();
        (!read || images.storage_read_without_format || bits)
            && (!write || images.storage_write_without_format || bits)
    }

    /// The descriptor and addressing capabilities the profile enables: whether it reaches buffers
    /// by device address, binds descriptors after binding, and so on.
    pub fn descriptors(&self) -> &DescriptorCaps {
        &self.fields.descriptors
    }

    /// The Function storage a thread holds correctly on the device: its measured value, or the
    /// floor of a device not measured. No Vulkan property states it; Metal states none either.
    pub fn function_storage(&self) -> FunctionStorage {
        self.fields.function_storage
    }

    /// The profile's subgroup model: each Metal stage's subgroup sizes, operations, and whether it
    /// can pin a size or require full subgroups. What a simdgroup runs as follows from it
    /// ([`DeviceProfile::simd_tier`]); a compiled function states what it requires
    /// (`CompiledFunction::required_subgroup_size`, `requires_full_subgroups`).
    pub fn subgroups(&self) -> &SubgroupModel {
        &self.fields.subgroup
    }

    /// How simdgroups of usage class `class` run in `stage` on this profile: its [`Tier`], or the
    /// feature it lacks. It depends on the class, the stage and the profile, never on a threadgroup
    /// size.
    pub fn simd_tier(&self, class: &SimdClass, stage: Stage) -> Result<Tier, Unsupported> {
        crate::device::subgroup::tier(class, stage, &self.fields.subgroup)
    }

    /// The buffer robustness this profile enables, which bounds a descriptor view's access at a
    /// run-time offset where it holds ([`checks_buffers_in_shader`](Self::checks_buffers_in_shader)).
    pub fn robustness(&self) -> Robustness {
        self.fields.robustness
    }

    /// Whether a module compiled for this profile checks a buffer access in the shader: exactly
    /// when no buffer robustness is enabled. Elsewhere the descriptor range and the device's
    /// robustness bound it.
    pub fn checks_buffers_in_shader(&self) -> bool {
        crate::memory::contain::encoding(self) == Check::InShader
    }

    /// Whether this profile leaves an out-of-bounds read of an image (`texel_buffer`: of a texel
    /// buffer) to the device, whose robustness answers Metal's out-of-bounds texel; elsewhere the
    /// shader checks the read.
    pub fn device_reads_texels(&self, texel_buffer: bool) -> bool {
        crate::memory::contain::device_reads_texels(self, texel_buffer)
    }

    /// Whether this profile leaves an out-of-bounds write of an image (`texel_buffer`: of a texel
    /// buffer) to the device, whose robustness discards it as Metal does; elsewhere the shader
    /// guards the write.
    pub fn device_drops_texel_writes(&self, texel_buffer: bool) -> bool {
        crate::memory::contain::device_drops_texel_writes(self, texel_buffer)
    }

    /// What this profile lacks of `GLSL.std.450`'s interpolation functions (the
    /// `InterpolationFunction` capability: `InterpolateAtCentroid`, `InterpolateAtSample`,
    /// `InterpolateAtOffset`): `sampleRateShading`, and on a portability-subset device
    /// `shaderSampleRateInterpolationFunctions`, whichever are missing, in that order. Empty
    /// exactly when a fragment function may read an `interpolant` at its centroid, a sample or an
    /// offset (Metal's pull-model interpolation, `supportsPullModelInterpolation`).
    pub fn interpolation_function_needs(&self) -> Vec<Capability> {
        let f = &self.fields;
        let mut needs = Vec::new();
        if !f.fragment.sample_rate_shading {
            needs.push(Capability::Feature(VkFeature::SampleRateShading));
        }
        let member = VkFeature::ShaderSampleRateInterpolationFunctions;
        if f.enabled.portability_withholds(member) {
            needs.push(Capability::Feature(member));
        }
        needs
    }

    /// What this profile lacks of Profile A's device-address requirements: `bufferDeviceAddress`
    /// and `shaderInt64`, whichever are missing, in that order. Empty exactly when the profile
    /// reaches buffers by device address (Profile A). A Profile F refusal that device addresses
    /// would lift needs this one conjunct, so a device that has one of the two features is never
    /// told it needs that feature.
    pub fn addressed_needs(&self) -> Vec<Capability> {
        let f = &self.fields;
        let mut needs = Vec::new();
        if !f.descriptors.buffer_device_address {
            needs.push(Capability::Feature(VkFeature::BufferDeviceAddress));
        }
        if !f.leaf.int64 {
            needs.push(Capability::Feature(VkFeature::ShaderInt64));
        }
        needs
    }

    /// What `spirv-val` validates a module compiled for this profile under (the L3): each of the
    /// target environments with the relaxations. Every module the crate returns for the profile has
    /// requirements within it ([`DeviceProfile::check`]) and is valid under each; the crate's own
    /// tests, this workspace's validation and the command-line tool run exactly this. A consumer
    /// that wants Khronos's own check of what it was handed runs it (the library runs no external
    /// tool).
    pub fn spirv_validation(&self) -> SpirvValidation {
        let api = self.fields.api_version;
        let own = if api >= (ApiVersion { major: 1, minor: 4 }) {
            "vulkan1.4"
        } else if api >= ApiVersion::V1_3 {
            "vulkan1.3"
        } else {
            "vulkan1.2"
        };
        let mut target_envs = vec!["vulkan1.2"];
        if own != "vulkan1.2" {
            target_envs.push(own);
        }
        let enabled = &self.fields.enabled.features;
        SpirvValidation {
            target_envs,
            relaxations: SPIRV_VAL_RELAXATIONS
                .iter()
                .filter(|(feature, _)| enabled.contains(feature))
                .map(|&(_, argument)| argument)
                .collect(),
        }
    }

    /// A SHA-256 over **every** field, policy included.
    ///
    /// The encoding is the derived `Debug` rendering of the profile's fields under a versioned
    /// domain tag: it is total by construction (a new field is hashed without further code) and
    /// unambiguous for these types (no free-form strings). Formatting changes across toolchains
    /// can only change fingerprints, which costs cache misses, never a false hit.
    ///
    /// Read here, not derived: the profile derived it when it was built, because rendering a
    /// profile with a format table is tens of kilobytes of `Debug` and every stage's compile
    /// takes the fingerprint twice.
    pub fn fingerprint(&self) -> Hash256 {
        self.fingerprint
    }
}

/// Count one derivation ([`FINGERPRINTS_DERIVED`]).
#[cfg(test)]
fn note_fingerprint_derived() {
    FINGERPRINTS_DERIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Count one derivation; nothing outside the crate's own tests.
#[cfg(not(test))]
fn note_fingerprint_derived() {}

/// The profile fingerprints derived in this process, which `tests::profile`'s count reads: a
/// device's compiles must add none to what its own making derived.
#[cfg(test)]
static FINGERPRINTS_DERIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`FINGERPRINTS_DERIVED`].
#[cfg(test)]
pub(crate) fn fingerprints_derived() -> u64 {
    FINGERPRINTS_DERIVED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The fingerprint of `fields` ([`DeviceProfile::fingerprint`]): the rendering streams into the
/// hash, because a materialized rendering of a profile with a format table is tens of kilobytes
/// that no input-scaled ledger charge accounts for.
fn fingerprint_of(fields: &ProfileFields) -> Hash256 {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    struct Hashing(Sha256);
    impl std::fmt::Write for Hashing {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0.update(s.as_bytes());
            Ok(())
        }
    }
    note_fingerprint_derived();
    let mut h = Hashing(Sha256::new());
    h.0.update(b"metal2vulkan/device-profile/v1\0");
    // `Hashing` never fails, and `Debug` of these plain-data types only fails when the
    // writer does.
    let _ = write!(h, "{:?}", fields);
    h.0.finalize().into()
}

/// Every field of a profile, policy included. The fingerprint hashes all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileFields {
    /// Memory's A/F split.
    pub buffer_access: BufferAccess,
    pub pointer_domain: PointerDomain,
    pub handle_domain: HandleDomain,
    /// The buffer robustness the profile enables.
    pub robustness: Robustness,
    pub leaf: LeafCaps,
    pub subgroup: SubgroupModel,
    pub images: ImageCaps,
    pub descriptors: DescriptorCaps,
    pub fragment: FragmentCaps,
    pub float_controls: FloatControls,
    pub vulkan_memory_model: bool,
    pub draw_parameters: bool,
    pub limits: Limits,
    /// Formats usable as vertex attributes (`VertexBuffer`).
    pub vertex_formats: FormatTable,
    /// Formats usable as storage images or storage texel buffers (plain or atomic).
    pub storage_formats: FormatTable,
    /// Every other queried usage: sampled images (and their filtering), uniform texel buffers, and
    /// attachments.
    pub image_formats: FormatTable,
    /// The device's Vulkan version (the floor is 1.2; every profile emits SPIR-V 1.5).
    pub api_version: ApiVersion,
    /// Vertex-buffer strides, with the other extended dynamic states, are dynamic state: the
    /// description lists `extendedDynamicState` (Vulkan 1.3, or `VK_EXT_extended_dynamic_state`
    /// with its feature enabled). A render pipeline that makes them dynamic requires the feature.
    pub dynamic_vertex_stride: bool,
    /// The enabled capability sets `Requirements` are checked against.
    pub enabled: EnabledCaps,
    /// Whether a two-plane YCbCr texture's planes may be read in place
    /// (`bind::YCbCrInPlace`): on a device whose GPU reads the planes' memory with no bus between
    /// (`VkPhysicalDeviceType` integrated or CPU), a texture slot whose function reads such a
    /// texture only as the in-place form reads records `RuntimeFact::YCbCrInPlace`, and the
    /// consumer states the planes of each texture it holds so. A discrete, virtual or other
    /// device, or one of no stated type, keeps every such texture's image of converted texels,
    /// which its refresh reads across the bus once per encoder where the in-place form would read
    /// the planes there at every pass. Both forms read Metal's texels exactly.
    pub ycbcr_in_place: bool,
    /// The Function storage a thread holds correctly (`device::function_storage`): a stage's
    /// private memory, its call stacks included, may take this much a thread
    /// (`VkLimit::FunctionStorageBytesPerThread`), and a vertex or fragment function on a device
    /// not measured more ([`FunctionStorage::Assumed`]).
    pub function_storage: FunctionStorage,
}

/// The Function storage a thread holds correctly ([`DeviceProfile::function_storage`]), in bytes:
/// how much private memory a stage may keep a thread (its `thread` variables and arrays, its call
/// stacks, and the constants it holds per thread), which a driver backs with scratch memory and
/// past which it can answer wrongly or hang with no error. Measured per device, since no Vulkan
/// property states it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FunctionStorage {
    /// Measured on the device: a thread holds this many bytes correctly.
    Measured(u64),
    /// The device was not measured: it holds this many, the floor, with no note. A vertex or
    /// fragment function may keep more, up to the 192 KiB a thread Metal was measured to hold, and
    /// records `PrecisionNote::FunctionStorageAssumed`; any other stage past the floor is refused.
    Assumed(u64),
}

impl FunctionStorage {
    /// The bytes a thread holds, measured or assumed.
    pub fn bytes(self) -> u64 {
        match self {
            FunctionStorage::Measured(b) | FunctionStorage::Assumed(b) => b,
        }
    }
}

/// What the consumer enabled, as the capability vocabulary `Requirements` uses
/// ([`DeviceProfile::enabled`]). Selection reads the typed caps above; the finalize check
/// (`requirements ⊆ profile`) reads these.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnabledCaps {
    /// The enabled features.
    pub features: BTreeSet<VkFeature>,
    /// The enabled extensions.
    pub extensions: BTreeSet<VkExtension>,
    /// The properties the device reports that a module may rely on.
    pub properties: BTreeSet<VkProperty>,
}

impl EnabledCaps {
    /// Whether `VK_KHR_portability_subset` withholds what its feature `feature` permits: the
    /// extension is enabled (as it is wherever the device lists it) and the feature is not. A
    /// device without the extension has none of the subset's restrictions.
    pub fn portability_withholds(&self, feature: VkFeature) -> bool {
        self.extensions.contains(&VkExtension::KhrPortabilitySubset)
            && !self.features.contains(&feature)
    }
}

/// Memory profile: Profile A (`Addressed`, buffer device address available) or Profile F
/// (`Floor`, descriptors only).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    Addressed,
    Floor,
}

/// How pointer values are resolved to resources. A device takes `Identity` only with
/// [`HandleDomain::Identity`], and the other two together.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PointerDomain {
    /// The sorted resident set of bound arguments and used resources. It exists for the
    /// validation executor only
    /// ([`DevicePolicy::RESIDENT_TABLES`](crate::device::DevicePolicy::RESIDENT_TABLES)), which
    /// builds the resident root itself.
    ResidentTable,
    /// Pointers are device addresses used directly: a `device` or `constant` pointer is the
    /// 64-bit Vulkan device address of the byte it names, as Metal's `gpuAddress` is, and a
    /// buffer slot is reached at its own address, pushed, and a pointer loaded from
    /// memory or made from an integer is the address it holds, used directly. Needs
    /// `bufferDeviceAddress` and `shaderInt64`. So far it lowers a kernel that reaches memory
    /// only through its buffer slots and such addresses, and, on `Device::from_queries`' profile,
    /// a vertex or fragment function that does; any other function is refused as
    /// [`Feature::IdentityDomains`](crate::error::Feature::IdentityDomains).
    Identity,
}

/// How resource handles are resolved. A device takes `Identity` only with
/// [`PointerDomain::Identity`], and the other two together.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandleDomain {
    /// The handle table plus the set-1 heap. It exists for the validation executor only
    /// ([`DevicePolicy::RESIDENT_TABLES`](crate::device::DevicePolicy::RESIDENT_TABLES)), which
    /// builds the handle table itself.
    Table,
    /// A texture's or sampler's handle is its `gpuResourceID`: the low 32 bits index the
    /// pipeline's descriptor arrays and the high 32 bits are the word of its Metal descriptor
    /// (`docs/RESOURCE-ID.md`). A texture or sampler use with no form under it is refused as
    /// [`Feature::IdentityDomains`](crate::error::Feature::IdentityDomains).
    Identity,
}

/// Which buffer robustness the profile enables: a descriptor view's access at a run-time offset is
/// bounded by it where it holds, and by the normative in-shader check where none is enabled
/// (`memory::contain`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Robustness {
    /// Neither; descriptor views get in-shader checks.
    None,
    /// `robustBufferAccess`.
    Rba,
    /// `robustBufferAccess2`.
    Rba2,
}

/// Leaf arithmetic, storage, and atomic capabilities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeafCaps {
    pub int8: bool,
    pub int16: bool,
    pub int64: bool,
    pub float16: bool,
    pub float64: bool,
    pub storage_buffer_8bit: bool,
    pub uniform_buffer_8bit: bool,
    pub push_constant_8bit: bool,
    pub storage_buffer_16bit: bool,
    pub uniform_buffer_16bit: bool,
    pub push_constant_16bit: bool,
    pub workgroup_explicit_layout: bool,
    pub workgroup_explicit_layout_8bit: bool,
    pub workgroup_explicit_layout_16bit: bool,
    pub buffer_int64_atomics: bool,
    pub shared_int64_atomics: bool,
    pub buffer_float32_atomic_add: bool,
    pub buffer_float32_atomic_min_max: bool,
    pub shared_float32_atomic_add: bool,
    pub shared_float32_atomic_min_max: bool,
    pub image_float32_atomic_add: bool,
    pub image_float32_atomic_min_max: bool,
    pub image_int64_atomics: bool,
    pub vertex_stores_and_atomics: bool,
    pub fragment_stores_and_atomics: bool,
    pub zero_initialize_workgroup_memory: bool,
    /// `OpFmaKHR` on 32-bit floats (`shaderFmaFloat32`): a correctly rounded fused multiply-add.
    pub fma32: bool,
    /// `OpFmaKHR` on 16-bit floats (`shaderFmaFloat16`): a correctly rounded fused multiply-add.
    pub fma16: bool,
    /// 8-bit floats of the encodings `Float8E4M3EXT` and `Float8E5M2EXT` (`shaderFloat8`,
    /// `SPV_EXT_float8`): an 8-bit float converts to a wider float through the device's own type.
    pub float8: bool,
}

/// Image and sampler capabilities ([`DeviceProfile::images`]). Each field without its own note is
/// the Vulkan feature of that name (`shaderStorageImageMultisample` for `storage_multisample`,
/// ...), enabled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ImageCaps {
    pub storage_read_without_format: bool,
    pub storage_write_without_format: bool,
    pub storage_extended_formats: bool,
    pub storage_multisample: bool,
    pub cube_array: bool,
    pub gather_extended: bool,
    /// The `Offset` image operand on `OpImageSample*` (`maintenance8`).
    pub dynamic_sample_offset: bool,
    pub min_lod: bool,
    /// `robustImageAccess` or `robustImageAccess2`: an image access outside the view is
    /// discarded (a write) or reads zero (a read; under the first level alone, a format's missing
    /// alpha reads 0 or 1, and a fetch at a level past the view's is undefined).
    pub robust_image_access: bool,
    /// `robustImageAccess2`: an image read outside the view (coordinate, layer, sample, or level)
    /// returns zero with `(0, 0, 1)` substituted for the format's missing G, B, and A, which is
    /// Metal's out-of-bounds texel.
    pub robust_image_access2: bool,
    pub sampler_anisotropy: bool,
    pub sampler_filter_minmax: bool,
    pub sampler_mirror_clamp_to_edge: bool,
    pub compute_derivative_quads: bool,
    pub compute_derivative_linear: bool,
}

/// Descriptor and addressing capabilities ([`DeviceProfile::descriptors`]). Each field is the
/// Vulkan feature or extension of that name (`bufferDeviceAddress` for `buffer_device_address`,
/// `VK_KHR_push_descriptor` for `push_descriptor`, ...), enabled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DescriptorCaps {
    pub buffer_device_address: bool,
    pub scalar_block_layout: bool,
    pub uniform_buffer_standard_layout: bool,
    pub runtime_descriptor_array: bool,
    pub partially_bound: bool,
    pub variable_descriptor_count: bool,
    pub update_after_bind_sampled_image: bool,
    pub update_after_bind_storage_image: bool,
    pub update_after_bind_storage_buffer: bool,
    pub update_after_bind_uniform_texel_buffer: bool,
    pub update_after_bind_storage_texel_buffer: bool,
    pub update_unused_while_pending: bool,
    pub non_uniform_sampled_image: bool,
    pub non_uniform_storage_image: bool,
    pub non_uniform_storage_buffer: bool,
    pub non_uniform_uniform_texel_buffer: bool,
    pub non_uniform_storage_texel_buffer: bool,
    pub push_descriptor: bool,
    pub null_descriptor: bool,
}

/// Fragment-stage and fixed-function capabilities ([`DeviceProfile::fragment`]). Each field is the
/// Vulkan feature or extension of that name (`shaderStencilExport` for `stencil_export`,
/// `rasterizationOrderColorAttachmentAccess` for `rasterization_order_color_access`, ...), enabled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FragmentCaps {
    pub demote_to_helper: bool,
    pub terminate_invocation: bool,
    pub pixel_interlock: bool,
    pub sample_interlock: bool,
    pub barycentric: bool,
    pub stencil_export: bool,
    pub sample_rate_shading: bool,
    pub dual_src_blend: bool,
    pub independent_blend: bool,
    pub geometry_shader: bool,
    pub tessellation_shader: bool,
    pub output_layer: bool,
    pub output_viewport_index: bool,
    pub clip_distance: bool,
    pub cull_distance: bool,
    pub rasterization_order_color_access: bool,
    pub dynamic_rendering_local_read: bool,
    pub vertex_attribute_divisor: bool,
    pub vertex_attribute_zero_divisor: bool,
}

/// Float-controls support of one width.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FloatWidthControls {
    pub denorm_preserve: bool,
    pub denorm_flush_to_zero: bool,
    pub rounding_rte: bool,
    pub rounding_rtz: bool,
    pub signed_zero_inf_nan_preserve: bool,
}

/// `VkShaderFloatControlsIndependence`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Independence {
    Only32Bit,
    All,
    #[default]
    None,
}

impl Independence {
    /// The contracts from most constrained to fully independent: `None` (every width uses the
    /// same mode, omission included) < `Only32Bit` (32-bit may differ; 16 and 64 agree) < `All`.
    const fn rank(self) -> u8 {
        match self {
            Independence::None => 0,
            Independence::Only32Bit => 1,
            Independence::All => 2,
        }
    }

    /// Whether a device with this independence admits a module that needs `required`
    /// (RuntimeSpirv VUIDs 06289-06292, ported from `src/vulkan/float_controls.rs`).
    pub const fn admits(self, required: Independence) -> bool {
        self.rank() >= required.rank()
    }
}

/// Float-controls properties.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FloatControls {
    pub f16: FloatWidthControls,
    pub f32: FloatWidthControls,
    pub f64: FloatWidthControls,
    pub denorm_independence: Independence,
    pub rounding_independence: Independence,
}

/// The device limits the new core consumes (named after `VkPhysicalDeviceLimits` and extension
/// properties; see `device::requirements::VkLimit`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Limits {
    pub max_compute_work_group_size: [u32; 3],
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_push_constants_size: u32,
    pub max_push_descriptors: u32,
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
    pub max_per_stage_descriptor_storage_images: u32,
    pub max_per_stage_descriptor_input_attachments: u32,
    pub max_per_stage_resources: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_sampled_images: u32,
    pub max_descriptor_set_storage_images: u32,
    pub max_descriptor_set_input_attachments: u32,
    pub max_descriptor_set_update_after_bind_samplers: u32,
    pub max_descriptor_set_update_after_bind_sampled_images: u32,
    pub max_descriptor_set_update_after_bind_storage_images: u32,
    pub max_descriptor_set_update_after_bind_storage_buffers: u32,
    pub max_per_stage_descriptor_update_after_bind_samplers: u32,
    pub max_per_stage_descriptor_update_after_bind_sampled_images: u32,
    pub max_per_stage_descriptor_update_after_bind_storage_images: u32,
    pub max_per_stage_descriptor_update_after_bind_storage_buffers: u32,
    pub max_per_stage_update_after_bind_resources: u32,
    pub max_storage_buffer_range: u32,
    pub max_uniform_buffer_range: u32,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_texel_buffer_offset_alignment: u64,
    pub max_texel_buffer_elements: u32,
    /// `maxSamplerLodBias`, as the IEEE-754 bits of the `f32` limit (keeps the struct `Eq + Hash`).
    pub max_sampler_lod_bias_bits: u32,
    /// `maxSamplerAnisotropy`, as the IEEE-754 bits of the `f32` limit.
    pub max_sampler_anisotropy_bits: u32,
    pub min_texel_offset: i32,
    pub max_texel_offset: u32,
    pub min_texel_gather_offset: i32,
    pub max_texel_gather_offset: u32,
    /// `minInterpolationOffset`, as the IEEE-754 bits of the `f32` limit.
    pub min_interpolation_offset_bits: u32,
    /// `maxInterpolationOffset`, as the IEEE-754 bits of the `f32` limit.
    pub max_interpolation_offset_bits: u32,
    pub max_image_dimension_1d: u32,
    pub max_image_dimension_2d: u32,
    pub max_image_dimension_3d: u32,
    pub max_image_dimension_cube: u32,
    pub max_image_array_layers: u32,
    pub max_vertex_input_attributes: u32,
    pub max_vertex_input_bindings: u32,
    pub max_vertex_input_attribute_offset: u32,
    pub max_vertex_input_binding_stride: u32,
    pub max_vertex_output_components: u32,
    /// `maxTessellationGenerationLevel`: the largest tessellation factor (0 without
    /// `tessellationShader`).
    pub max_tessellation_generation_level: u32,
    pub max_tessellation_evaluation_output_components: u32,
    pub max_fragment_input_components: u32,
    pub max_fragment_output_attachments: u32,
    pub max_fragment_dual_src_attachments: u32,
    pub max_fragment_combined_output_resources: u32,
    pub max_color_attachments: u32,
    pub max_clip_distances: u32,
    pub max_cull_distances: u32,
    pub max_combined_clip_and_cull_distances: u32,
    pub max_vertex_attrib_divisor: u32,
    /// `maxPerStageDescriptorAccelerationStructures` (`VK_KHR_acceleration_structure`; 0 where
    /// the extension is not exposed).
    pub max_per_stage_descriptor_acceleration_structures: u32,
    /// `maxDescriptorSetAccelerationStructures` (0 where `VK_KHR_acceleration_structure` is not
    /// exposed).
    pub max_descriptor_set_acceleration_structures: u32,
    /// Whether a draw with a nonzero `firstInstance` may use a vertex binding divisor other
    /// than 1: `supportsNonZeroFirstInstance` where `VK_KHR_vertex_attribute_divisor` or
    /// Vulkan 1.4 exposes it (VUID-vkCmdDraw-pNext-09461 otherwise requires `firstInstance == 0`),
    /// and `false` otherwise, including a device exposing only `VK_EXT_vertex_attribute_divisor`
    /// (which cannot report the property; its drivers do carry the restriction).
    pub supports_non_zero_first_instance: bool,
    /// `VkPhysicalDevicePortabilitySubsetPropertiesKHR::minVertexInputBindingStrideAlignment` on a
    /// device exposing `VK_KHR_portability_subset` (which must then be enabled): every vertex
    /// binding stride must be a multiple of it and at least as large
    /// (`VUID-VkVertexInputBindingDescription-stride-04456`). `None` on other devices, which
    /// have no such rule; `Some(u32::MAX)` when a portability device does not report it (absence
    /// is "not supported").
    pub min_vertex_input_binding_stride_alignment: Option<u32>,
    pub framebuffer_color_sample_counts: vk::SampleCountFlags,
    pub framebuffer_depth_sample_counts: vk::SampleCountFlags,
    /// `maxMultiviewViewCount` (Vulkan 1.1; at least 6 where `multiview` is supported): the
    /// largest view count of a multiview render pass instance, and so of a vertex amplification.
    pub max_multiview_view_count: u32,
    /// `maxMultiviewInstanceIndex` (Vulkan 1.1; at least 2^27 - 1 where `multiview` is
    /// supported): the largest instance index of a draw in a multiview render pass instance.
    pub max_multiview_instance_index: u32,
    /// `framebufferIntegerColorSampleCounts` (Vulkan 1.2): the sample counts every integer color
    /// attachment supports (`framebufferColorSampleCounts` covers the float and fixed-point ones).
    pub framebuffer_integer_color_sample_counts: vk::SampleCountFlags,
    /// `framebufferStencilSampleCounts`: the sample counts every attachment with stencil supports.
    pub framebuffer_stencil_sample_counts: vk::SampleCountFlags,
    /// `framebufferNoAttachmentsSampleCounts`: the rasterization sample counts a subpass without
    /// attachments supports.
    pub framebuffer_no_attachments_sample_counts: vk::SampleCountFlags,
    /// `storageImageSampleCounts`: the sample counts every 2D optimal-tiling storage image
    /// supports (a multisampled shadow of the stencil).
    pub storage_image_sample_counts: vk::SampleCountFlags,
    /// `maxFramebufferWidth`: the widest framebuffer.
    pub max_framebuffer_width: u32,
    /// `maxFramebufferHeight`: the tallest framebuffer.
    pub max_framebuffer_height: u32,
    /// `maxFramebufferLayers`: the most layers a framebuffer may have.
    pub max_framebuffer_layers: u32,
    /// `maxDrawIndexedIndexValue`: the largest index an indexed draw may fetch (`2^32 - 1` with
    /// `fullDrawIndexUint32`, at least `2^24 - 1` otherwise).
    pub max_draw_indexed_index_value: u32,
    /// `maxTessellationPatchSize`: the most control points a patch may have (0 without
    /// `tessellationShader`).
    pub max_tessellation_patch_size: u32,
    /// `maxGeometryInputComponents`: the scalar components of a geometry stage's user inputs per
    /// vertex (0 without `geometryShader`).
    pub max_geometry_input_components: u32,
    /// `maxGeometryOutputComponents`: the scalar components of a geometry stage's user outputs
    /// per vertex (0 without `geometryShader`).
    pub max_geometry_output_components: u32,
    /// `maxGeometryTotalOutputComponents`: the scalar components of every vertex a geometry stage
    /// emits for one input primitive, together (0 without `geometryShader`).
    pub max_geometry_total_output_components: u32,
    /// `supportedDepthResolveModes` (`VkPhysicalDeviceDepthStencilResolveProperties`, Vulkan
    /// 1.2): the modes a depth attachment may resolve by.
    pub supported_depth_resolve_modes: vk::ResolveModeFlags,
    /// `supportedStencilResolveModes`: the modes a stencil attachment may resolve by.
    pub supported_stencil_resolve_modes: vk::ResolveModeFlags,
    /// `independentResolveNone`: one aspect may resolve while the other does not.
    pub independent_resolve_none: bool,
    /// `independentResolve`: the two aspects may resolve by different modes.
    pub independent_resolve: bool,
    /// `sampleLocationSampleCounts` (`VkPhysicalDeviceSampleLocationsPropertiesEXT`): the sample
    /// counts that take custom sample locations; empty where `VK_EXT_sample_locations` is not
    /// exposed.
    pub sample_location_sample_counts: vk::SampleCountFlags,
    /// `variableSampleLocations`: the sample locations may differ between pipelines of one
    /// subpass; `false` where `VK_EXT_sample_locations` is not exposed.
    pub variable_sample_locations: bool,
    /// `provokingVertexModePerPipeline` (`VkPhysicalDeviceProvokingVertexPropertiesEXT`): the
    /// pipelines of one render pass instance may differ in provoking vertex mode; `false` where
    /// `VK_EXT_provoking_vertex` is not exposed.
    pub provoking_vertex_mode_per_pipeline: bool,
    /// `VkPhysicalDeviceMeshShaderPropertiesEXT`: every limit 0 where `VK_EXT_mesh_shader` is not
    /// exposed.
    pub mesh: MeshLimits,
    /// The upper bound of `pointSizeRange`, in whole pixels, rounded down: the device clamps a
    /// point's size to at most the bound, so a size past it draws smaller than written. Whole
    /// pixels, because a description's JSON states the bound to two decimals (2047.94 for
    /// NVIDIA's 2047.9375, 8191.88 for RADV's 8191.875), where the device's query gives the
    /// `f32` itself: both round down to the same number, and Metal's largest point (511 pixels)
    /// is whole. 1 where a description omits the range: Vulkan's bound without `largePoints`,
    /// `pointSizeRange` (1.0, 1.0).
    pub max_point_size: u32,
    /// `lineWidthRange`, as the IEEE-754 bits of its two `f32` bounds: the widths a line drawn
    /// with `wideLines` takes, others clamped into it. `(1.0, 1.0)` where a description omits it,
    /// Vulkan's bound without `wideLines`.
    pub line_width_range_bits: [u32; 2],
    /// `lineWidthGranularity`, as the IEEE-754 bits of the `f32` limit: the step between the
    /// widths the device draws, a width between two drawn as "the nearest supported width" (the
    /// Vulkan specification's line segments).
    pub line_width_granularity_bits: u32,
    /// `strictLines`: the device draws a wide line as the rectangle its width spans across it,
    /// where a non-strict device may draw it otherwise (below).
    pub strict_lines: bool,
    /// `nonStrictWideLinesUseParallelogram` (`VkPhysicalDeviceMaintenance5Properties`): a
    /// non-strict wide line is the parallelogram whose minor-axis edges pass through its
    /// endpoints, Metal's (`setLineWidth:`); `false` where `VK_KHR_maintenance5` is not exposed.
    pub non_strict_wide_lines_use_parallelogram: bool,
    /// `earlyFragmentMultisampleCoverageAfterSampleCounting`
    /// (`VkPhysicalDeviceMaintenance5Properties`): under a fragment shader's `EarlyFragmentTests`,
    /// its fragment shading and multisample coverage operations (its `SampleMask` output among
    /// them) come after sample counting, so an occlusion query counts every sample the early tests
    /// pass, as Metal counts under its private masks. Where `false`, Vulkan lets them come before
    /// (the specification's fragment operations); `false` too where `VK_KHR_maintenance5` is not
    /// exposed.
    pub early_fragment_multisample_coverage_after_sample_counting: bool,
    /// `subPixelPrecisionBits`: the bits of a framebuffer coordinate's fraction the rasterizer
    /// keeps (4 at least).
    pub sub_pixel_precision_bits: u32,
}

/// The limits of `VK_EXT_mesh_shader` (`VkPhysicalDeviceMeshShaderPropertiesEXT`) the task and mesh
/// stages are held to; the preference hints and the output granularities (which only the output
/// memory size, not held yet, would read) are not read. Each field is the structure's member of the
/// same name (`max_mesh_output_vertices` is `maxMeshOutputVertices`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshLimits {
    pub max_task_work_group_total_count: u32,
    pub max_task_work_group_count: [u32; 3],
    pub max_task_work_group_invocations: u32,
    pub max_task_work_group_size: [u32; 3],
    pub max_task_payload_size: u32,
    pub max_task_shared_memory_size: u32,
    pub max_task_payload_and_shared_memory_size: u32,
    pub max_mesh_work_group_total_count: u32,
    pub max_mesh_work_group_count: [u32; 3],
    pub max_mesh_work_group_invocations: u32,
    pub max_mesh_work_group_size: [u32; 3],
    pub max_mesh_shared_memory_size: u32,
    pub max_mesh_payload_and_shared_memory_size: u32,
    pub max_mesh_output_memory_size: u32,
    pub max_mesh_payload_and_output_memory_size: u32,
    pub max_mesh_output_components: u32,
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    pub max_mesh_output_layers: u32,
    pub max_mesh_multiview_view_count: u32,
}

impl Limits {
    /// Whether a vertex binding may have `stride` under this device's portability subset: at
    /// least `min_vertex_input_binding_stride_alignment` and a multiple of it
    /// (`VUID-VkVertexInputBindingDescription-stride-04456`); always true on a device without
    /// the rule.
    pub fn portable_vertex_stride(&self, stride: u32) -> bool {
        portable_vertex_stride(self.min_vertex_input_binding_stride_alignment, stride)
    }
}

/// `Limits::max_point_size` of a `pointSizeRange` whose upper bound is `high`: the bound rounded
/// down to whole pixels, 0 for a bound that is not a number or not above zero, and saturating.
pub(crate) fn whole_point_size(high: f32) -> u32 {
    // `as` rounds toward zero, maps NaN to 0 and saturates.
    high.max(0.0) as u32
}

/// Whether a vertex binding may have `stride` under the portability subset's
/// `minVertexInputBindingStrideAlignment` `alignment`: a multiple of it and at least as large
/// (`VUID-VkVertexInputBindingDescription-stride-04456`); always on a device without the rule
/// (`None`).
pub fn portable_vertex_stride(alignment: Option<u32>, stride: u32) -> bool {
    alignment.is_none_or(|a| stride >= a && stride.is_multiple_of(a))
}

/// Per-format usage support for the formats the translator queries.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FormatTable {
    pub formats: BTreeMap<vk::Format, BTreeSet<FormatUsage>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::requirements::Requirements;
    use crate::pipeline::facts::{Mechanism, RuntimeFact};
    use crate::spirv::ids::{BlockId, DemoteTy, Value, VarId};

    /// Vocabulary shared across threads by the consumer's worker pool.
    #[test]
    fn vocabulary_is_send_and_sync() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<DeviceProfile>();
        send_sync::<ProfileFields>();
        send_sync::<Requirements>();
        send_sync::<RuntimeFact>();
        send_sync::<Mechanism>();
        send_sync::<DemoteTy>();
        send_sync::<Value>();
        send_sync::<VarId>();
        send_sync::<BlockId>();
        send_sync::<crate::air::ir::value::ValueId>();
        send_sync::<crate::bind::status::ExecutionStatus>();
    }
}
