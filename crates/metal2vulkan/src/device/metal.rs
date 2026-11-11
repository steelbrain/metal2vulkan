//! `MetalDeviceProperties`: what the consumer reports to the guest, from the same profile that
//! drives translation.

use crate::device::description::AllocationLimits;
use crate::device::profile::{BufferAccess, DeviceProfile};
use crate::device::requirements::{FormatUsage, VkFeature};
use crate::device::subgroup::{tier, OpBits, SimdClass};
use crate::error::{Capability, Feature, Unsupported};
use crate::metal::format::storage_image_choice;
use crate::metal::{ArgumentBuffersTier, PixelFormat, ReadWriteTextureTier};
use crate::pipeline::sym::Stage;
use crate::vk;

/// Metal device properties reported to the guest.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDeviceProperties {
    /// Always 32: the consumer answers 32 on both reply paths (info-encoder `0x1c2` and
    /// `CmdGetComputeInfo` key 3).
    pub thread_execution_width: u32,
    /// Per axis: `maxComputeWorkGroupSize[i]` capped by `maxComputeWorkGroupInvocations`, and by an
    /// Apple GPU's 1024.
    pub max_threads_per_threadgroup: [u32; 3],
    /// `maxComputeWorkGroupInvocations` (pipelines report a flattened-tier cap of their own),
    /// capped by an Apple GPU's 1024.
    pub max_total_threads_per_threadgroup: u32,
    /// `maxComputeSharedMemorySize`: threadgroup arenas are exactly the Metal allocations. A kernel
    /// whose simdgroups are emulated takes scratch beside them, which its pipeline's
    /// `max_total_threads_per_threadgroup` accounts for. Capped by an Apple GPU's 32 KiB.
    pub max_threadgroup_memory_length: u32,
    /// `Tier2` when buffer pointers (Profile A) and texture/sampler handles (descriptor indexing)
    /// are both available; `Tier1` (inline constants and statically rooted buffers) otherwise.
    pub argument_buffers_tier: ArgumentBuffersTier,
    /// `Tier1` when every Metal read-write tier-1 format (`R32Float`, `R32Uint`, `R32Sint`) has a
    /// storage-image choice, `Tier2` when every tier-2 format has one as well (on Vulkan 1.2 the
    /// 8- and 16-bit single-channel ones need `shaderStorageImageExtendedFormats`, or a device
    /// that reports their storage images anyway; formatless access alone is not enough), else
    /// `None`.
    pub read_write_texture_tier: ReadWriteTextureTier,
    /// The largest buffer a shader can address: `maxStorageBufferRange` in Profile F (larger
    /// windows are refused at plan time); the allocation limit in Profile A.
    pub max_buffer_length: u64,
    /// Sample counts usable for both color and depth attachments, of an Apple GPU's 1, 2 and 4.
    pub raster_sample_counts: vk::SampleCountFlags,
    /// Per-stage simd feasibility. A kernel's simdgroups that no subgroup tier runs are emulated,
    /// for operations in workgroup-uniform control flow.
    pub simd: SimdSupport,
    /// `R32_SFLOAT` supports linear filtering.
    pub supports_32bit_float_filtering: bool,
    /// `D24_UNORM_S8_UINT` is a depth/stencil attachment format.
    pub supports_depth24_stencil8: bool,
    /// `maxImageDimension2D`, capped by an Apple GPU's 16384.
    pub max_texture_dimension_2d: u32,
    /// Families every pipeline using them will refuse on this device, known up front. A summary
    /// for the guest-visible feature set, not an exhaustive list of refusals.
    pub unsupported_features: Vec<Unsupported>,
}

/// The answers of the Apple GPU every device presents to its guest, where a Vulkan device's own
/// limit can be larger: an Apple M3 Max under macOS 26.7 and an M4 under 27.0.1, the GPUs whose
/// Metal the runtime answers as, both of the Apple9 family. Each is that family's cell of Apple's
/// Metal feature set tables, which those GPUs answer. A device reports the lower of each and its
/// own limit, so no vendor's driver offers a guest more than an Apple GPU does, as
/// `isDepth24Stencil8PixelFormatSupported` answers Apple's no everywhere.
pub(crate) mod apple_gpu {
    use crate::vk::SampleCountFlags;

    /// `maxThreadsPerThreadgroup` on each axis and `maxTotalThreadsPerThreadgroup`: "Maximum
    /// threads per threadgroup", 1024 from Apple4.
    pub(crate) const MAX_THREADS_PER_THREADGROUP: u32 = 1024;
    /// `maxThreadgroupMemoryLength`: "Maximum total threadgroup memory allocation", 32 KiB from
    /// Apple4.
    pub(crate) const MAX_THREADGROUP_MEMORY_LENGTH: u32 = 32 * 1024;
    /// `supportsTextureSampleCount:` answers yes for 1, 2 and 4: "Maximum sample count in render
    /// passes with MSAA" is 4 to Apple9 (8 from Apple10).
    pub(crate) const TEXTURE_SAMPLE_COUNTS: SampleCountFlags = SampleCountFlags::COUNT_1
        .union(SampleCountFlags::COUNT_2)
        .union(SampleCountFlags::COUNT_4);
    /// "Maximum 2D texture width and height": 16384 from Apple3 (32768 from Apple10).
    pub(crate) const MAX_TEXTURE_DIMENSION_2D: u32 = 16_384;
}

/// Metal's read-write tier-1 formats.
pub const READ_WRITE_TIER1_FORMATS: &[PixelFormat] = &[
    PixelFormat::R32Float,
    PixelFormat::R32Uint,
    PixelFormat::R32Sint,
];

/// The formats Metal's read-write tier 2 adds to tier 1.
pub const READ_WRITE_TIER2_FORMATS: &[PixelFormat] = &[
    PixelFormat::RGBA32Float,
    PixelFormat::RGBA32Uint,
    PixelFormat::RGBA32Sint,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA16Uint,
    PixelFormat::RGBA16Sint,
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA8Uint,
    PixelFormat::RGBA8Sint,
    PixelFormat::R16Float,
    PixelFormat::R16Uint,
    PixelFormat::R16Sint,
    PixelFormat::R8Unorm,
    PixelFormat::R8Uint,
    PixelFormat::R8Sint,
];

/// Which simd usage classes each stage can tier.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SimdSupport {
    pub vertex: StageSimd,
    pub fragment: StageSimd,
    pub kernel: StageSimd,
}

/// Simd feasibility in one stage.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StageSimd {
    /// The operation classes whose cross-lane use tiers in this stage (each class alone).
    pub cross_lane: OpBits,
    /// Quad operations tier in this stage.
    pub quad: bool,
    /// Simdgroup matrices tier in this stage.
    pub matrix: bool,
}

impl MetalDeviceProperties {
    /// Derives the reported properties from a profile and the device's allocation sizes (which
    /// are no profile field).
    pub(crate) fn derive(
        profile: &DeviceProfile,
        allocation: &AllocationLimits,
    ) -> MetalDeviceProperties {
        let f = profile.fields();
        let l = &f.limits;
        let threads = l
            .max_compute_work_group_invocations
            .min(apple_gpu::MAX_THREADS_PER_THREADGROUP);
        let per_axis = l.max_compute_work_group_size.map(|axis| axis.min(threads));
        let d = &f.descriptors;
        let argument_buffers_tier = if f.buffer_access == BufferAccess::Addressed
            && d.runtime_descriptor_array
            && d.partially_bound
            && d.update_after_bind_sampled_image
            && d.update_after_bind_storage_image
        {
            ArgumentBuffersTier::Tier2
        } else {
            ArgumentBuffersTier::Tier1
        };
        // A tier holds when each of its formats can be a storage image on this profile, by the
        // same choice lowering and the binder make.
        let read_write = |formats: &[PixelFormat]| {
            formats
                .iter()
                .all(|&pf| storage_image_choice(pf, profile).is_ok())
        };
        let read_write_texture_tier = if !read_write(READ_WRITE_TIER1_FORMATS) {
            ReadWriteTextureTier::None
        } else if !read_write(READ_WRITE_TIER2_FORMATS) {
            ReadWriteTextureTier::Tier1
        } else {
            ReadWriteTextureTier::Tier2
        };
        let max_buffer_length = match f.buffer_access {
            BufferAccess::Floor => u64::from(l.max_storage_buffer_range),
            _ => match allocation.max_buffer_size {
                0 => allocation.max_memory_allocation_size,
                cap => cap.min(allocation.max_memory_allocation_size),
            },
        };
        let stage = |s: Stage| {
            let mut cross_lane = OpBits::NONE;
            for op in crate::device::derive::SUBGROUP_OPS {
                let bit = OpBits::of(op);
                if tier(&SimdClass::CrossLane(bit), s, &f.subgroup).is_ok() {
                    cross_lane = cross_lane.union(bit);
                }
            }
            StageSimd {
                cross_lane,
                quad: tier(&SimdClass::Quad, s, &f.subgroup).is_ok(),
                matrix: tier(&SimdClass::Matrix, s, &f.subgroup).is_ok(),
            }
        };
        MetalDeviceProperties {
            thread_execution_width: 32,
            max_threads_per_threadgroup: per_axis,
            max_total_threads_per_threadgroup: threads,
            max_threadgroup_memory_length: l
                .max_compute_shared_memory_size
                .min(apple_gpu::MAX_THREADGROUP_MEMORY_LENGTH),
            argument_buffers_tier,
            read_write_texture_tier,
            max_buffer_length,
            raster_sample_counts: l.framebuffer_color_sample_counts
                & l.framebuffer_depth_sample_counts
                & apple_gpu::TEXTURE_SAMPLE_COUNTS,
            simd: SimdSupport {
                vertex: stage(Stage::Vertex),
                fragment: stage(Stage::Fragment),
                kernel: stage(Stage::Kernel),
            },
            supports_32bit_float_filtering: profile.supports_format(
                vk::Format::R32_SFLOAT,
                FormatUsage::SampledImageFilterLinear,
            ),
            supports_depth24_stencil8: profile.supports_format(
                vk::Format::D24_UNORM_S8_UINT,
                FormatUsage::DepthStencilAttachment,
            ),
            max_texture_dimension_2d: l
                .max_image_dimension_2d
                .min(apple_gpu::MAX_TEXTURE_DIMENSION_2D),
            unsupported_features: unsupported_features(profile),
        }
    }
}

/// The feature families that refuse on every pipeline of this profile.
fn unsupported_features(profile: &DeviceProfile) -> Vec<Unsupported> {
    let f = profile.fields();
    let feature = Capability::Feature;
    let mut out = Vec::new();
    // 64-bit integer arithmetic needs no `shaderInt64`: it is held as two words.
    if !f.leaf.float64 {
        out.push(Unsupported::needs(
            Feature::Float64Arithmetic,
            feature(VkFeature::ShaderFloat64),
        ));
    }
    if f.buffer_access == BufferAccess::Floor {
        out.push(Unsupported::needs_any_of(
            Feature::ArgumentBufferPointers,
            vec![profile.addressed_needs()],
        ));
    }
    if !f.fragment.pixel_interlock && !f.fragment.sample_interlock {
        out.push(Unsupported::needs_any_of(
            Feature::RasterOrderGroups,
            vec![
                vec![feature(VkFeature::FragmentShaderPixelInterlock)],
                vec![feature(VkFeature::FragmentShaderSampleInterlock)],
            ],
        ));
    }
    // Framebuffer fetch refuses on no device: ordered by rasterization-order access, by the pixel
    // interlock's storage form, or, with neither, through input attachments ordered between draws
    // (`PrecisionNote::UnorderedFetchWithinDraw`).
    // `[[barycentric_coord]]`: the device's builtins, or varyings a generated geometry stage
    // writes (`pipeline::link::geometry`).
    if !f.fragment.barycentric && !f.fragment.geometry_shader {
        out.push(Unsupported::needs_any_of(
            Feature::Barycentrics,
            vec![
                vec![feature(VkFeature::FragmentShaderBarycentric)],
                vec![feature(VkFeature::GeometryShader)],
            ],
        ));
    }
    for stage in [Stage::Vertex, Stage::Fragment, Stage::Kernel] {
        let cross_lane = SimdClass::CrossLane(OpBits::BASIC);
        if let Err(refusal) = tier(&cross_lane, stage, &f.subgroup) {
            if !out.contains(&refusal) {
                out.push(refusal);
            }
        }
    }
    out
}
