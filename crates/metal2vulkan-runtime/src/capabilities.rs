//! Metal's device queries, answered from the Vulkan device.
//!
//! [`DeviceProperties`] holds the device's identity and the public `MTLDevice` properties and
//! queries, each answered from an exact source:
//! - a Vulkan property, limit, feature or format the device was created with (its enabled
//!   [`DeviceDescription`]);
//! - the compile layer's answer for the device
//!   ([`MetalDeviceProperties`](metal2vulkan::device::MetalDeviceProperties), which the translation
//!   itself obeys);
//! - or a value measured on Apple's GPU that the compile layer reproduces.
//!
//! Apple-only features answer missing unless the compile layer emulates them exactly. Metal's GPU
//! families (`supportsFamily:`) are the device's own query, judged row by row against Apple's
//! feature set tables ([`Device::supports_family`](super::Device::supports_family)).
//!
//! Apple's private (SPI) device properties are
//! [`Device::private_properties`](super::Device::private_properties)'s, but for
//! `supportsTextureWriteRoundingMode:`, which a pipeline's translation answers, and
//! `supportsPrimitiveType:`, both here.
//!
//! A query whose runtime piece is not built yet answers the [`RuntimeGap`] it waits on
//! ([`Query::gap`]). The step that builds the piece deletes the gap, and the query then answers the
//! device's value, which this type already holds.
//!
//! The fields are fixed when the device is made. An answer that changes while the device lives is
//! a method of its object's step on `Device` instead: step 5's per-format mipmap verdict, settled
//! by the first mipmap generation of each format.

use super::device::argument_buffers::APPLE;
use super::device::DeviceKind;
use super::driver_defects;
use super::error::{invalid, Error, Problem, RuntimeGap};
use super::linear_plan::linear_alignment;
use super::texture_descriptor::TextureRules;
use super::values::Size;
use metal2vulkan::device::{DeviceDescription, DriverFact, VkFeature};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::{ArgumentBuffersTier, PixelFormat, ReadWriteTextureTier};
use metal2vulkan::vk::{ApiVersion, SampleCountFlags};
use std::collections::BTreeSet;

/// Rasterization-rate maps of up to this many layers compile: the most Apple's compile was
/// measured with, which is Apple M3's `supportsRasterizationRateMapWithLayerCount:`, and the most
/// Metal's API validation takes on an M3 Max and an M4.
/// [`bind::RasterizationRateMap::new`](metal2vulkan::bind::RasterizationRateMap::new) refuses more, and
/// so does [`Device::new_rasterization_rate_map`](super::Device::new_rasterization_rate_map).
pub(super) const RATE_MAP_LAYERS: u64 = 2;

/// The largest vertex amplification count Metal supports: `supportsVertexAmplificationCount:`
/// answers yes from 1 to 8 and no from 9 to 16 (a Metal probe: the M3 Max with macOS 26.7 and an M4
/// with macOS 27.0.1, alike), and yes for 0 (alike). A device with `multiview` draws every such
/// count: as views where multiview draws them as Metal does, else one amplification at a time.
pub(super) const MAX_VERTEX_AMPLIFICATION: u64 = 8;

/// A device's identity and Metal's public device queries, as the runtime answers them on the
/// Vulkan device.
///
/// The fields hold final answers. The methods answer the queries whose runtime piece may not be
/// built yet ([`Query`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceProperties {
    /// `name`: `VkPhysicalDeviceProperties::deviceName`.
    pub name: String,
    /// The kind of physical device (`VkPhysicalDeviceProperties::deviceType`).
    pub kind: DeviceKind,
    /// `hasUnifiedMemory`: the device's memory is the host's, from its kind: an integrated GPU's
    /// or a CPU implementation's. A discrete GPU answers no, as Metal answers on one, and so do a
    /// virtual GPU and any other kind, whose memory Vulkan does not place. `Managed` storage
    /// follows Metal's contract for the answer (`managed.rs`), and `Shared` storage lies where a
    /// Mac's does (docs/API.md "The runtime's memory model").
    pub has_unified_memory: bool,
    /// Whether the Mac the device presents keeps textures losslessly compressed, as an Apple
    /// silicon Mac's GPU does, a Mac with an Intel or AMD GPU's does not: true where the device
    /// presents [`MacClass::AppleSilicon`], the class
    /// [`DeviceOptions::mac_class`](crate::DeviceOptions::mac_class) declares, else the one of
    /// the device's GPU ([`MacClass::of_vendor`]). Metal has no property for it: it shows in what
    /// a new `Private` texture reads before anything writes it (Metal's fixed pattern or zero,
    /// against zero or a released texture's texels), in the encoding a GPU blit through such a
    /// texture writes, and in a `half` NaN clear's sign (docs/API.md, the accepted differences).
    pub lossless_compression: bool,
    /// `VkPhysicalDeviceProperties::vendorID`.
    pub vendor_id: u32,
    /// `VkPhysicalDeviceProperties::deviceID`.
    pub device_id: u32,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`, which `registryID` is folded from
    /// (`Device::registry_id`).
    pub device_uuid: [u8; 16],
    /// The device's PCI bus address, as its driver reports it:
    /// `VkPhysicalDevicePCIBusInfoPropertiesEXT`'s `pciDomain`, `pciBus`, `pciDevice` and
    /// `pciFunction`, in that order, read where the physical device lists
    /// `VK_EXT_pci_bus_info`. No Metal counterpart: Metal names no bus, and macOS has no such
    /// Vulkan extension, so a MoltenVK device answers `None`. A consumer that must drive another
    /// API (a video decoder, a compute runtime) on the same GPU this device draws on addresses it
    /// by this. `None` where the physical device does not list the extension, which is the only
    /// thing the answer then says: the device still has a bus address, and the runtime does not
    /// know it.
    pub pci_bus: Option<(u32, u32, u32, u32)>,
    /// The device's DRM render node, as its driver reports it:
    /// `VkPhysicalDeviceDrmPropertiesEXT`'s `renderMajor` and `renderMinor`, in that order, read
    /// where the physical device lists `VK_EXT_physical_device_drm`. No Metal counterpart: DRM is
    /// Linux's, and macOS has no such Vulkan extension, so a MoltenVK device answers `None`. A
    /// consumer that must open the same GPU through another Linux interface (VA-API, V4L2) opens
    /// this node. `None` where the physical device does not list the extension, and `None` where
    /// it lists it and reports the device has no render node (`hasRender` false), whose major and
    /// minor numbers say nothing; never a pair of zeroes standing for either.
    pub drm_render: Option<(i64, i64)>,
    /// The Vulkan version the device runs at: the lower of the instance's and the device's.
    pub api_version: ApiVersion,
    /// `maxBufferLength`: the longest buffer a shader can bind whole, the compile layer's answer:
    /// `maxStorageBufferRange` where buffers are reached through descriptors only, else the lower
    /// of `maxBufferSize` and `maxMemoryAllocationSize` (the device's allocation sizes, which
    /// follow its machine's memory on integrated GPUs, as Metal's answer does).
    pub max_buffer_length: u64,
    /// `maxThreadsPerThreadgroup`: per axis, `maxComputeWorkGroupSize` capped by
    /// `maxComputeWorkGroupInvocations` and by an Apple GPU's 1024.
    pub max_threads_per_threadgroup: Size,
    /// `maxThreadgroupMemoryLength`: `maxComputeSharedMemorySize`, since a kernel's threadgroup
    /// arenas are exactly its Metal allocations, capped by an Apple GPU's 32 KiB.
    pub max_threadgroup_memory_length: u64,
    /// `argumentBuffersSupport`: `Tier2` where buffers are reached by device address and texture
    /// and sampler handles through the descriptor heap (descriptor indexing), else `Tier1`.
    pub argument_buffers_support: ArgumentBuffersTier,
    /// `readWriteTextureSupport`: the highest tier whose every format has a storage-image form
    /// on the device.
    pub read_write_texture_support: ReadWriteTextureTier,
    /// `supports32BitFloatFiltering`: `R32_SFLOAT` supports linear filtering.
    pub supports_32bit_float_filtering: bool,
    /// `VkPhysicalDeviceLimits::maxViewports`, a documented fact for the consumer (the viewports
    /// a draw may select, with the description's `multiViewport`). Metal has no public property
    /// for it.
    pub max_viewports: u32,
    /// `gpuCoreCount` (Apple SPI): the GPU's cores as its driver reports them, read from the
    /// device through the one extension its vendor names them by, where the driver lists it:
    /// `VK_NV_shader_sm_builtins`' `shaderSMCount` (its streaming multiprocessors),
    /// `VK_AMD_shader_core_properties2`' `activeComputeUnitCount` (its active compute units; else
    /// `VK_AMD_shader_core_properties`' engines times arrays per engine times compute units per
    /// array), or `VK_ARM_shader_core_builtins`' `shaderCoreCount`. `None` where the driver lists
    /// none of them (Intel's drivers, MoltenVK and lavapipe among them): the runtime knows no
    /// count of the device's, and gives none. Never a number of Apple's GPUs.
    pub gpu_core_count: Option<u32>,
    /// The description the device was created with: what it enables of Vulkan, which is all the
    /// translation and the runtime use. The facts a consumer computes Apple's private properties
    /// from.
    pub description: DeviceDescription,
    /// The defects of the device's driver in commands the runtime records for itself, from its
    /// identity and release (`driver_defects.rs`): what one breaks is taken another way where an
    /// exact one exists, else refused on the device, typed
    /// ([`Needs::RuntimeDriverDefect`](metal2vulkan::error::Needs::RuntimeDriverDefect)).
    pub driver_defects: BTreeSet<RuntimeDriverDefect>,
    /// What the device's driver is measured to answer where Vulkan leaves the result undefined,
    /// from its identity and GPU ([`DeviceDescription::driver_facts`]): the runtime relies on one
    /// only where the driver has it, and elsewhere takes the exact path that needs none.
    pub driver_facts: BTreeSet<DriverFact>,
    /// The pixel interlock is enabled, so every fragment function's raster order groups are
    /// ordered.
    raster_order_groups: bool,
    /// The sample counts every color and depth target supports, as an attachment and as a
    /// sampled image, of an Apple GPU's 1, 2 and 4
    /// ([`DeviceProperties::supports_texture_sample_count`]).
    texture_sample_counts: SampleCountFlags,
    /// The largest vertex amplification count a render pipeline may declare: Metal's
    /// (`MAX_VERTEX_AMPLIFICATION`) where `multiview` is enabled, else 1.
    max_vertex_amplification_count: u64,
    /// Custom sample positions are taken at every texture sample count above 1
    /// (`sample_positions.rs`).
    programmable_sample_positions: bool,
    /// The compile layer lowers an `interpolant` read at its centroid, a sample or an offset
    /// ([`DeviceProperties::supports_pull_model_interpolation`]).
    pull_model_interpolation: bool,
}

/// What [`DeviceProperties::new`] reads of the physical device itself, beside the description
/// it was created with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceIdentity {
    /// `VkPhysicalDeviceProperties::deviceName`.
    pub name: String,
    /// `VkPhysicalDeviceProperties::deviceType`.
    pub kind: DeviceKind,
    /// `VkPhysicalDeviceProperties::vendorID`.
    pub vendor_id: u32,
    /// `VkPhysicalDeviceProperties::deviceID`.
    pub device_id: u32,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`.
    pub device_uuid: [u8; 16],
    /// The device's PCI bus address ([`DeviceProperties::pci_bus`]).
    pub pci_bus: Option<(u32, u32, u32, u32)>,
    /// The device's DRM render node ([`DeviceProperties::drm_render`]).
    pub drm_render: Option<(i64, i64)>,
    /// The lower of the instance's and the device's Vulkan version.
    pub api_version: ApiVersion,
    /// `VkPhysicalDeviceLimits::maxViewports`.
    pub max_viewports: u32,
    /// The GPU's cores as its driver reports them ([`DeviceProperties::gpu_core_count`]).
    pub gpu_core_count: Option<u32>,
    /// The sample counts every color and depth image can be sampled at:
    /// `sampledImageColorSampleCounts & sampledImageDepthSampleCounts`.
    pub sampled_sample_counts: SampleCountFlags,
}

/// The class of Mac a device presents, whose GPU decides answers Metal gives without a property
/// of its own: what a new `Private` texture reads before anything writes it, the encoding a GPU
/// blit through a compressed texture writes, and a `half` NaN clear's sign (docs/API.md, the
/// accepted differences). Every Mac's Metal answers one class's way, by the GPU it has (Apple
/// silicon, or an Intel or AMD GPU); the runtime answers the class the consumer declares
/// ([`DeviceOptions::mac_class`](crate::DeviceOptions::mac_class)), else the one of the device's
/// GPU ([`MacClass::of_vendor`]), and states it as
/// [`DeviceProperties::lossless_compression`]. No Metal counterpart: a Mac's GPU is what it is.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MacClass {
    /// An Apple silicon Mac, whose GPU keeps a texture losslessly compressed where Metal's rule
    /// holds (`allowGPUOptimizedContents`, a level 0 of 16 texels or more on its shorter side, a
    /// colour format of neither `ShaderWrite` nor `PixelFormatView` usage): a new `Private` such
    /// texture reads Metal's fixed pattern over its first levels and zero elsewhere, blits
    /// through it write three formats' canonical encodings, and a `half` NaN clear keeps its
    /// sign. The runtime writes the pattern and the zeros (a pattern no clear or copy writes
    /// exactly reads zero, announced as `fresh-pattern-zeroed`).
    AppleSilicon,
    /// A Mac with an Intel or AMD GPU, which keeps no texture compressed: a new `Private`
    /// texture holds what its memory held (zero, or a released texture's texels: an Intel UHD
    /// 630 and an AMD Radeon Pro 5300M), blits keep the bytes, and a `half` NaN clear goes
    /// through `half`. On a GPU that keeps no texture compressed the runtime writes nothing into
    /// a new `Private` texture (`WorkCounts::private_textures_unfilled`); on an Apple GPU, whose
    /// own compressed texture would read Apple's pattern, it fills it with zeros
    /// (`WorkCounts::private_textures_zero_filled`).
    IntelOrAmd,
}

impl MacClass {
    /// The class of the Mac whose GPU is of PCI vendor `vendor_id`
    /// (`VkPhysicalDeviceProperties::vendorID`): an Apple GPU's (`0x106b`, MoltenVK on Apple
    /// silicon) is [`AppleSilicon`](Self::AppleSilicon); every other GPU's (Intel, AMD, NVIDIA, a
    /// CPU implementation's) is [`IntelOrAmd`](Self::IntelOrAmd), whose GPUs keep no texture
    /// compressed, an integrated GPU of unified memory such as Intel's included. The class a
    /// device presents where the consumer declares none. No Metal counterpart.
    pub fn of_vendor(vendor_id: u32) -> MacClass {
        match vendor_id == APPLE {
            true => MacClass::AppleSilicon,
            false => MacClass::IntelOrAmd,
        }
    }
}

/// Whether the Mac a device of `identity`, its class declared as `declared`, presents keeps
/// textures losslessly compressed ([`DeviceProperties::lossless_compression`]): the one place the
/// runtime decides it. Every answer that differs between an Apple silicon Mac and a Mac with an
/// Intel or AMD GPU because the first compresses (a new `Private` texture's contents, a blit's
/// canonical encodings, a `half` NaN clear's sign) reads that field, and nothing else decides it.
fn presents_lossless_compression(identity: &DeviceIdentity, declared: Option<MacClass>) -> bool {
    declared.unwrap_or_else(|| MacClass::of_vendor(identity.vendor_id)) == MacClass::AppleSilicon
}

impl DeviceProperties {
    /// The properties of the device `identity` names, created with `description` (what it
    /// enables), whose compile-layer device is `compile`, presenting the Mac class `mac_class`
    /// declares (`None`: its GPU's).
    pub(crate) fn new(
        identity: DeviceIdentity,
        description: DeviceDescription,
        compile: &metal2vulkan::device::Device,
        mac_class: Option<MacClass>,
    ) -> DeviceProperties {
        let lossless_compression = presents_lossless_compression(&identity, mac_class);
        let metal = compile.metal_properties();
        let [width, height, depth] = metal.max_threads_per_threadgroup.map(u64::from);
        let texture_sample_counts = metal.raster_sample_counts & identity.sampled_sample_counts;
        let max_vertex_amplification_count = if description.has(VkFeature::Multiview) {
            MAX_VERTEX_AMPLIFICATION
        } else {
            1
        };
        DeviceProperties {
            name: identity.name,
            kind: identity.kind,
            has_unified_memory: identity.kind.has_unified_memory(),
            lossless_compression,
            vendor_id: identity.vendor_id,
            device_id: identity.device_id,
            device_uuid: identity.device_uuid,
            pci_bus: identity.pci_bus,
            drm_render: identity.drm_render,
            api_version: identity.api_version,
            max_buffer_length: metal.max_buffer_length,
            max_threads_per_threadgroup: Size::new(width, height, depth),
            max_threadgroup_memory_length: u64::from(metal.max_threadgroup_memory_length),
            argument_buffers_support: metal.argument_buffers_tier,
            read_write_texture_support: metal.read_write_texture_tier,
            supports_32bit_float_filtering: metal.supports_32bit_float_filtering,
            max_viewports: identity.max_viewports,
            gpu_core_count: identity.gpu_core_count,
            raster_order_groups: description.has(VkFeature::FragmentShaderPixelInterlock),
            texture_sample_counts,
            programmable_sample_positions: super::sample_positions::programmable(
                compile.profile(),
                texture_sample_counts,
            ),
            max_vertex_amplification_count,
            pull_model_interpolation: compile.profile().interpolation_function_needs().is_empty(),
            driver_defects: driver_defects::of(&description),
            driver_facts: description.driver_facts(),
            description,
        }
    }

    /// `areRasterOrderGroupsSupported`: the pixel interlock is enabled
    /// (`fragmentShaderPixelInterlock`), so the compile layer orders the raster order groups of
    /// every fragment function the device draws as Metal does: its body runs in the pixel's
    /// ordered critical section (`PixelInterlockOrderedEXT`). Without it the answer is no: a device
    /// with only the sample interlock orders the groups of a function that runs per sample alone,
    /// and on any device a group holding only color outputs the function does not read compiles
    /// without an interlock, ordered by blending's own rasterization order; neither is every group
    /// Metal's answer promises.
    pub fn are_raster_order_groups_supported(&self) -> Result<bool, RuntimeGap> {
        Query::RasterOrderGroups.answer(self.raster_order_groups)
    }

    /// `isDepth24Stencil8PixelFormatSupported`: no, on every device, as Metal answers on an
    /// Apple GPU (an M3 Max under macOS 26.7 and an M4 under 27.0.1), whose Metal stops a
    /// texture and an attachment of `Depth24Unorm_Stencil8` and `X24_Stencil8` as of no format
    /// (it describes neither: [`Device::pixel_format_info`](crate::Device::pixel_format_info)). Every device answers as Apple's GPU, whatever its Vulkan
    /// driver's `D24_UNORM_S8_UINT`, so one guest gets one answer on every vendor.
    pub fn is_depth24_stencil8_pixel_format_supported(&self) -> Result<bool, RuntimeGap> {
        Query::Depth24Stencil8.answer(false)
    }

    /// `areProgrammableSamplePositionsSupported`: the device enables `VK_EXT_sample_locations`,
    /// has `variableSampleLocations`, and takes custom sample locations
    /// (`sampleLocationSampleCounts`) at every sample count above 1 it renders textures at
    /// ([`DeviceProperties::supports_texture_sample_count`]), as Metal takes custom positions at
    /// those (measured on an M4: 2 and 4, not 1; `sample_positions.rs`).
    pub fn are_programmable_sample_positions_supported(&self) -> Result<bool, RuntimeGap> {
        Query::ProgrammableSamplePositions.answer(self.programmable_sample_positions)
    }

    /// `supportsTextureSampleCount:`: a texture of `count` samples can be a color and a depth
    /// render target and be read in a shader as `texture2d_ms` / `depth2d_ms`, which is what
    /// Metal's answer promises. `count` is in every one of `framebufferColorSampleCounts`,
    /// `framebufferDepthSampleCounts`, `sampledImageColorSampleCounts` and
    /// `sampledImageDepthSampleCounts`, and is one an Apple GPU takes: 1, 2 or 4 (Apple9's
    /// "Maximum sample count in render passes with MSAA" is 4), so a device that renders 8 or 16
    /// samples answers no to them as Metal does, and stops such a texture. A count that is no
    /// Vulkan sample count (0, 3, above 64) is not supported.
    ///
    /// Integer formats are left out of the answer (`framebufferIntegerColorSampleCounts`,
    /// `sampledImageIntegerSampleCounts`), as stencil-only formats are: a device that guarantees
    /// every integer format one sample only would otherwise answer no to every count above 1,
    /// though some of its integer formats take more. Creating an integer texture at a count its
    /// format does not take is refused, `Unsupported` with `Feature::TextureSampleCount` naming
    /// the device property it lacks ([`Device::new_texture`](super::Device::new_texture)).
    pub fn supports_texture_sample_count(&self, count: u64) -> Result<bool, RuntimeGap> {
        Query::TextureSampleCount.answer(sample_count_supported(self.texture_sample_counts, count))
    }

    /// What a texture's descriptor takes of the device (`texture_descriptor`): the sample counts
    /// [`supports_texture_sample_count`](Self::supports_texture_sample_count) answers yes for,
    /// which a multisample texture takes.
    pub(super) fn texture_rules(&self) -> TextureRules {
        TextureRules {
            sample_counts: self.texture_sample_counts,
        }
    }

    /// `supportsPullModelInterpolation`: a fragment function may read an `interpolant` through
    /// every interpolation function, `interpolate_at_center`, `_centroid`, `_sample` and
    /// `_offset`, which the compile layer lowers to `GLSL.std.450`'s interpolation functions: the
    /// device has `sampleRateShading`, and on a portability-subset device (MoltenVK)
    /// `shaderSampleRateInterpolationFunctions`
    /// ([`DeviceProfile::interpolation_function_needs`](metal2vulkan::device::DeviceProfile::interpolation_function_needs)
    /// is empty). Without them only `interpolate_at_center` compiles, and a function that reads
    /// otherwise is refused, `Unsupported(PullModelInterpolation)`. What differs from Metal: an
    /// offset is read on the device's grid and within its range
    /// (`PrecisionNote::DeviceInterpolationOffset`), a centroid where Vulkan puts it
    /// (`PrecisionNote::MultisamplePosition`), and a draw at custom sample positions through a
    /// function that reads at a sample is refused
    /// ([`RuntimeGap::InterpolateAtSampleWithPositions`]).
    pub fn supports_pull_model_interpolation(&self) -> Result<bool, RuntimeGap> {
        Query::PullModelInterpolation.answer(self.pull_model_interpolation)
    }

    /// `supportsSamplerReductionMode`: no, on every device, as Metal answers on an Apple M4
    /// (macOS 27.0.1). There a sampler's minimum or maximum reduction reads the weighted average
    /// (Metal's validation layer refuses one), and so does every sampler the runtime makes and
    /// every constexpr sampler the compile layer lowers, whatever the device's
    /// `samplerFilterMinmax`.
    pub fn supports_sampler_reduction_mode(&self) -> Result<bool, RuntimeGap> {
        Query::SamplerReductionMode.answer(false)
    }

    /// `supportsVertexAmplificationCount:`: `count` is at most Metal's 8 with `multiview` enabled
    /// (1 without it), which render encoders draw as views where multiview draws them as Metal
    /// does, else one amplification at a time. A count of 0 is supported, as Metal answers: a draw
    /// at it draws nothing, as plain Metal's (`render.rs`).
    pub fn supports_vertex_amplification_count(&self, count: u64) -> Result<bool, RuntimeGap> {
        Query::VertexAmplificationCount.answer(amplification_count_supported(
            self.max_vertex_amplification_count,
            count,
        ))
    }

    /// `supportsRasterizationRateMapWithLayerCount:`: 1 or 2 layers, the layer counts Apple's
    /// map compile was measured with and the compile layer reproduces on every device.
    pub fn supports_rasterization_rate_map_with_layer_count(
        &self,
        count: u64,
    ) -> Result<bool, RuntimeGap> {
        Query::RasterizationRateMapLayerCount.answer(rate_map_layers_supported(count))
    }

    /// `supportsTextureWriteRoundingMode:` (Apple SPI): yes for Metal's raw 0 to 2, every mode
    /// a pipeline descriptor's `texture_write_rounding_mode` takes on every device
    /// ([`TextureWriteRoundingMode::from_raw`](metal2vulkan::metal::TextureWriteRoundingMode::from_raw)),
    /// as an M4 answers (macOS 27.0.1); no for 3 and above.
    pub fn supports_texture_write_rounding_mode(&self, mode: u64) -> bool {
        metal2vulkan::metal::TextureWriteRoundingMode::from_raw(mode).is_ok()
    }

    /// Whether this device keeps the GPU's writes out of read-only memory as Metal does
    /// everywhere Metal does: no-copy memory made with
    /// [`ResourceOptions::PRIVATE_BIT`](crate::ResourceOptions::PRIVATE_BIT), Metal's private
    /// option bit 24, which is what marks a range read-only (the companion of Apple's
    /// `replaceBackingWithRanges:readOnly:`). Metal has no query of its own for this: it is one of
    /// the runtime's, so that a consumer can tell what it may stop guarding.
    ///
    /// This answers yes only where the memory itself takes the GPU's writes nowhere, as Metal's
    /// does, so that every shape Metal answers holds: a shader's stores through a buffer slot or
    /// through an address loaded out of an argument buffer or a pointer, a blit fill and a
    /// visibility result all complete and leave the pages as they were, and a colour attachment with
    /// [`StoreAction::Store`](crate::StoreAction::Store), a resolve target or a depth or stencil
    /// attachment over such memory ends its command buffer in error. On macOS, where the device
    /// imports the pages through a read-only view of them, that is yes; elsewhere the import is of
    /// the caller's own pages and this answers no, with the difference announced
    /// ([`Degradation::PrivateResourceOption`](crate::Degradation::PrivateResourceOption)).
    pub fn drops_gpu_writes_to_read_only_memory(&self) -> bool {
        crate::read_only_view::VIEWS_DROP_WRITES
    }

    /// `minimumLinearTextureAlignmentForPixelFormat:`: the alignment of a texture's offset and
    /// rows in a buffer, 16 bytes for every format a buffer can back, as Metal answers on an M3 Max
    /// and an M4. Refused, `Invalid` with
    /// [`BufferTextureProblem::NotLinear`](crate::BufferTextureProblem::NotLinear), for a
    /// compressed, depth or stencil format, on which Metal's query asserts, and for a format Metal
    /// describes none of ([`Device::pixel_format_info`](crate::Device::pixel_format_info) answers
    /// [`PixelFormatInfo::INVALID`](crate::PixelFormatInfo::INVALID)), on which it asserts too.
    pub fn minimum_linear_texture_alignment_for_pixel_format(
        &self,
        format: PixelFormat,
    ) -> Result<u64, Error> {
        linear_alignment(format).map_err(|problem| {
            invalid(
                "minimum_linear_texture_alignment_for_pixel_format",
                Problem::BufferTexture(problem),
            )
        })
    }
}

/// Whether `count` samples is one of the Vulkan sample counts `flags` holds: a power of two whose
/// bit is set (`VkSampleCountFlagBits` are the counts themselves).
pub(super) fn sample_count_supported(flags: SampleCountFlags, count: u64) -> bool {
    count.is_power_of_two()
        && u32::try_from(count).is_ok_and(|bit| flags.contains(SampleCountFlags::from_raw(bit)))
}

/// Whether a vertex amplification count of `count` is in `[0, max]`, as Metal answers (0 too).
pub(super) fn amplification_count_supported(max: u64, count: u64) -> bool {
    count <= max
}

/// `supportsRasterizationRateMapWithLayerCount:`: 1 to [`RATE_MAP_LAYERS`], as Metal answers
/// (a map of no layers is made all the same, as Metal makes it).
pub(super) fn rate_map_layers_supported(count: u64) -> bool {
    (1..=RATE_MAP_LAYERS).contains(&count)
}

/// Metal's public device queries whose runtime piece may not be built yet.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Query {
    /// `areRasterOrderGroupsSupported`.
    RasterOrderGroups,
    /// `isDepth24Stencil8PixelFormatSupported`.
    Depth24Stencil8,
    /// `areProgrammableSamplePositionsSupported`.
    ProgrammableSamplePositions,
    /// `supportsTextureSampleCount:`.
    TextureSampleCount,
    /// `supportsVertexAmplificationCount:`.
    VertexAmplificationCount,
    /// `supportsRasterizationRateMapWithLayerCount:`.
    RasterizationRateMapLayerCount,
    /// `minimumLinearTextureAlignmentForPixelFormat:`.
    MinimumLinearTextureAlignment,
    /// `supportsSamplerReductionMode`.
    SamplerReductionMode,
    /// `supportsPullModelInterpolation`.
    PullModelInterpolation,
}

impl Query {
    /// Every query, in declaration order.
    pub const ALL: [Query; 9] = [
        Query::RasterOrderGroups,
        Query::Depth24Stencil8,
        Query::ProgrammableSamplePositions,
        Query::TextureSampleCount,
        Query::VertexAmplificationCount,
        Query::RasterizationRateMapLayerCount,
        Query::MinimumLinearTextureAlignment,
        Query::SamplerReductionMode,
        Query::PullModelInterpolation,
    ];

    /// Metal's selector.
    pub fn selector(self) -> &'static str {
        match self {
            Query::RasterOrderGroups => "areRasterOrderGroupsSupported",
            Query::Depth24Stencil8 => "isDepth24Stencil8PixelFormatSupported",
            Query::ProgrammableSamplePositions => "areProgrammableSamplePositionsSupported",
            Query::TextureSampleCount => "supportsTextureSampleCount:",
            Query::VertexAmplificationCount => "supportsVertexAmplificationCount:",
            Query::RasterizationRateMapLayerCount => "supportsRasterizationRateMapWithLayerCount:",
            Query::MinimumLinearTextureAlignment => "minimumLinearTextureAlignmentForPixelFormat:",
            Query::SamplerReductionMode => "supportsSamplerReductionMode",
            Query::PullModelInterpolation => "supportsPullModelInterpolation",
        }
    }

    /// The runtime piece the query's answer waits on, while it is not built; `None` once it is, as
    /// every query's is.
    pub fn gap(self) -> Option<RuntimeGap> {
        match self {
            Query::ProgrammableSamplePositions
            | Query::RasterOrderGroups
            | Query::VertexAmplificationCount
            | Query::RasterizationRateMapLayerCount
            | Query::Depth24Stencil8
            | Query::TextureSampleCount
            | Query::MinimumLinearTextureAlignment
            | Query::SamplerReductionMode
            | Query::PullModelInterpolation => None,
        }
    }

    /// The device's answer `device`, or the gap the query waits on.
    fn answer<T>(self, device: T) -> Result<T, RuntimeGap> {
        match self.gap() {
            Some(gap) => Err(gap),
            None => Ok(device),
        }
    }
}

#[cfg(test)]
impl DeviceProperties {
    /// The device's answers of the gated queries, gaps aside, for tests.
    pub(super) fn device_answers(&self) -> DeviceAnswers {
        DeviceAnswers {
            raster_order_groups: self.raster_order_groups,
            texture_sample_counts: self.texture_sample_counts,
            max_vertex_amplification_count: self.max_vertex_amplification_count,
            pull_model_interpolation: self.pull_model_interpolation,
        }
    }
}

/// The device's answers of the gated queries (tests only).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DeviceAnswers {
    pub raster_order_groups: bool,
    pub texture_sample_counts: SampleCountFlags,
    pub max_vertex_amplification_count: u64,
    pub pull_model_interpolation: bool,
}
