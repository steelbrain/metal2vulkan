//! `mechanism()` totality over every (fact, committed profile).

use super::profile::device;
use crate::device::mechanism::mechanism;
use crate::pipeline::facts::{Mechanism, RuntimeFact};
use crate::pipeline::sym::{
    ImageAccess, ImageDim, ImageHandleClass, SampledType, SlotClass, SlotRef, Stage,
    StorageFormatClass,
};

/// One value of every `RuntimeFact` variant.
fn every_fact() -> Vec<RuntimeFact> {
    let slot = SlotRef {
        stage: Stage::Fragment,
        class: SlotClass::Texture,
        index: 3,
    };
    vec![
        RuntimeFact::ThreadsPerThreadgroup,
        RuntimeFact::ThreadgroupMemoryLength { index: 0 },
        RuntimeFact::LocalSizeConstraint { multiple: 32 },
        RuntimeFact::LocalSizeLimits,
        RuntimeFact::DispatchGrid,
        RuntimeFact::IndirectGroupCount,
        RuntimeFact::ResourcePresence { slot },
        RuntimeFact::SampleCount,
        RuntimeFact::StorageTextureFormat { slot },
        RuntimeFact::SamplerCoordinates { slot },
        RuntimeFact::SamplerReduction { slot },
        RuntimeFact::ImageViewShape { slot },
        RuntimeFact::VertexFetch { attr: 1 },
        RuntimeFact::VertexBufferStride { layout: 0 },
        RuntimeFact::BaseVertexInstance,
        RuntimeFact::DeviceAddresses { slot },
        RuntimeFact::BufferRangeLimit { slot },
        RuntimeFact::TexelBufferRange { slot },
        RuntimeFact::FramebufferFetch { color: 0 },
        RuntimeFact::ExecutionStatus,
        RuntimeFact::HeapNumericType {
            class: ImageHandleClass {
                dim: ImageDim::D2,
                arrayed: false,
                multisampled: false,
                sampled_type: SampledType::Float,
                depth_compare: false,
                access: ImageAccess::Sampled,
                format: StorageFormatClass::NotStorage,
            },
        },
        RuntimeFact::SimdgroupEmulation {
            bytes_per_simdgroup: 128,
            arena: None,
        },
        RuntimeFact::AmplificationCount,
        RuntimeFact::EmulatedSimdgroups,
        RuntimeFact::ImageblockSize { cell_bytes: 16 },
        RuntimeFact::RenderTargetArrayLength {
            layer_dropped: false,
        },
        RuntimeFact::RenderTargetArrayLength {
            layer_dropped: true,
        },
        RuntimeFact::StageInRegion,
        RuntimeFact::StageInStride { layout: 2 },
        RuntimeFact::TileFunction,
        RuntimeFact::TileAttachmentFormat {
            color: 1,
            format: crate::metal::PixelFormat::R32Float,
        },
        RuntimeFact::StorageTextureNumericType { slot },
        RuntimeFact::OutOfBoundsAlpha { slot },
        RuntimeFact::BicubicTexels { slot },
        RuntimeFact::BorderTexels { slot },
        RuntimeFact::StorageTextureLevels { slot },
        RuntimeFact::StorageTextureLevelViews { slot },
        RuntimeFact::SparseTexture {
            slot,
            sparse_arm: true,
            tile_map: false,
        },
        RuntimeFact::SparseTexture {
            slot,
            sparse_arm: false,
            tile_map: true,
        },
        RuntimeFact::SparseTexture {
            slot,
            sparse_arm: false,
            tile_map: false,
        },
        RuntimeFact::AccelerationStructureInstanceCount { slot },
        RuntimeFact::PredicationVote,
        RuntimeFact::SrgbTexture { slot },
        RuntimeFact::ViewportCount,
        RuntimeFact::AliasedImageblock { data_size: 24 },
        RuntimeFact::CallStackDepth { depth: 8 },
        RuntimeFact::OpenClMode,
        RuntimeFact::OpenClLineTexture { slot },
        RuntimeFact::RowLockstep,
        RuntimeFact::TexelBufferParts { slot },
        RuntimeFact::AddressAlignment { slot, align: 16 },
        RuntimeFact::FlatGrid,
        RuntimeFact::CallThreadgroups,
        RuntimeFact::SparseTextureWrites {
            slot,
            atomic: false,
        },
        RuntimeFact::SparseTextureWrites { slot, atomic: true },
        RuntimeFact::TileThreadgroupMemory { index: 1 },
        RuntimeFact::TextureSwizzle { slot },
        RuntimeFact::TileImageblock {
            data_size: 8,
            alignment: 4,
        },
        RuntimeFact::AmplificationDraws {
            layered: true,
            viewports: false,
        },
        RuntimeFact::FoldedGroups,
        RuntimeFact::ReplayBindings,
        RuntimeFact::MergedDraws,
        RuntimeFact::PartialObjectThreadgroups,
        RuntimeFact::QuadPrimitiveIds,
        RuntimeFact::RotatedWeights,
        RuntimeFact::PrivateList,
        RuntimeFact::RegionViewports,
        RuntimeFact::PartialThreadgroups,
        RuntimeFact::GridTiles,
        RuntimeFact::ZeroedColor {
            color: 1,
            format: crate::metal::PixelFormat::RGBA8Unorm,
        },
        RuntimeFact::DroppedColor { color: 1 },
        RuntimeFact::DepthClampRange,
        RuntimeFact::ConstantLanes { slot },
        RuntimeFact::FunctionTableRange { addressed: false },
        RuntimeFact::TileThreadgroup { tile_sized: true },
        RuntimeFact::ConstexprSampled {
            slot,
            modes: [crate::metal::SamplerAddressMode::Repeat; 3],
        },
        RuntimeFact::CubicSampler { slot },
        RuntimeFact::CubicArgumentSampler {
            stage: Stage::Kernel,
        },
        RuntimeFact::StorageTextureIntegerChannels { slot },
        RuntimeFact::HeapRawSamples {
            stage: Stage::Fragment,
            sampler: Some(1),
        },
        RuntimeFact::CrossClassReads { slot },
        RuntimeFact::StorageTextureWriteRounding { slot, halves: true },
        RuntimeFact::HeapWriteRounding {
            stage: Stage::Fragment,
            halves: false,
        },
        RuntimeFact::HeapCrossClassNeutral {
            stage: Stage::Kernel,
            class: crate::pipeline::sym::ImageHandleClass {
                dim: crate::pipeline::sym::ImageDim::D2,
                arrayed: false,
                multisampled: false,
                sampled_type: crate::pipeline::sym::SampledType::Float,
                depth_compare: false,
                access: crate::pipeline::sym::ImageAccess::Sampled,
                format: crate::pipeline::sym::StorageFormatClass::NotStorage,
            },
            every_member: true,
        },
        RuntimeFact::HeapResidency {
            stage: Stage::Fragment,
        },
        RuntimeFact::HeapStorageReads {
            stage: Stage::Kernel,
            class: crate::pipeline::sym::ImageHandleClass {
                dim: crate::pipeline::sym::ImageDim::D2,
                arrayed: false,
                multisampled: false,
                sampled_type: crate::pipeline::sym::SampledType::Float,
                depth_compare: false,
                access: crate::pipeline::sym::ImageAccess::Storage,
                format: crate::pipeline::sym::StorageFormatClass::Unknown,
            },
        },
        RuntimeFact::AnisotropicSampler {
            slot: SlotRef {
                stage: Stage::Fragment,
                class: SlotClass::Sampler,
                index: 0,
            },
        },
        RuntimeFact::ObjectThreadgroupArena {
            static_bytes: 16,
            payload: Some(64),
        },
        RuntimeFact::ObjectThreadgroupArena {
            static_bytes: 0,
            payload: None,
        },
        RuntimeFact::PrivateArena { slot_bytes: 28_704 },
        RuntimeFact::StencilBitDraws,
    ]
}

/// The variant index of a fact. Exhaustive without a wildcard, so a new variant fails to compile
/// here until `every_fact` and the table cover it.
fn variant(f: &RuntimeFact) -> usize {
    match f {
        RuntimeFact::ThreadsPerThreadgroup => 0,
        RuntimeFact::ThreadgroupMemoryLength { .. } => 1,
        RuntimeFact::LocalSizeConstraint { .. } => 2,
        RuntimeFact::LocalSizeLimits => 3,
        RuntimeFact::DispatchGrid => 4,
        RuntimeFact::IndirectGroupCount => 5,
        RuntimeFact::ResourcePresence { .. } => 6,
        RuntimeFact::SampleCount => 7,
        RuntimeFact::StorageTextureFormat { .. } => 8,
        RuntimeFact::SamplerCoordinates { .. } => 9,
        RuntimeFact::SamplerReduction { .. } => 10,
        RuntimeFact::ImageViewShape { .. } => 11,
        RuntimeFact::VertexFetch { .. } => 12,
        RuntimeFact::VertexBufferStride { .. } => 13,
        RuntimeFact::BaseVertexInstance => 14,
        RuntimeFact::DeviceAddresses { .. } => 15,
        RuntimeFact::BufferRangeLimit { .. } => 16,
        RuntimeFact::TexelBufferRange { .. } => 17,
        RuntimeFact::FramebufferFetch { .. } => 18,
        RuntimeFact::ExecutionStatus => 19,
        RuntimeFact::HeapNumericType { .. } => 20,
        RuntimeFact::SimdgroupEmulation { .. } => 21,
        RuntimeFact::AmplificationCount => 22,
        RuntimeFact::EmulatedSimdgroups => 23,
        RuntimeFact::ImageblockSize { .. } => 24,
        RuntimeFact::RenderTargetArrayLength { .. } => 25,
        RuntimeFact::StageInRegion => 26,
        RuntimeFact::StageInStride { .. } => 27,
        RuntimeFact::RegionViewports => 28,
        RuntimeFact::TileFunction => 29,
        RuntimeFact::TileAttachmentFormat { .. } => 30,
        RuntimeFact::StorageTextureNumericType { .. } => 31,
        RuntimeFact::OutOfBoundsAlpha { .. } => 32,
        RuntimeFact::BicubicTexels { .. } => 33,
        RuntimeFact::StorageTextureLevels { .. } => 34,
        RuntimeFact::StorageTextureLevelViews { .. } => 35,
        RuntimeFact::SparseTexture { .. } => 36,
        RuntimeFact::AccelerationStructureInstanceCount { .. } => 37,
        RuntimeFact::PredicationVote => 38,
        RuntimeFact::HeapStorageReads { .. } => 39,
        RuntimeFact::SrgbTexture { .. } => 40,
        RuntimeFact::ViewportCount => 41,
        RuntimeFact::AliasedImageblock { .. } => 42,
        RuntimeFact::CallStackDepth { .. } => 43,
        RuntimeFact::FlatGrid => 44,
        RuntimeFact::SparseTextureWrites { .. } => 45,
        RuntimeFact::TileThreadgroupMemory { .. } => 46,
        RuntimeFact::TextureSwizzle { .. } => 47,
        RuntimeFact::TileImageblock { .. } => 48,
        RuntimeFact::AmplificationDraws { .. } => 49,
        RuntimeFact::FoldedGroups => 51,
        RuntimeFact::ZeroedColor { .. } => 53,
        RuntimeFact::DroppedColor { .. } => 54,
        RuntimeFact::DepthClampRange => 55,
        RuntimeFact::ConstantLanes { .. } => 56,
        RuntimeFact::FunctionTableRange { .. } => 57,
        RuntimeFact::TileThreadgroup { .. } => 59,
        RuntimeFact::ConstexprSampled { .. } => 60,
        RuntimeFact::CubicSampler { .. } => 61,
        RuntimeFact::CubicArgumentSampler { .. } => 62,
        RuntimeFact::HeapRawSamples { .. } => 66,
        RuntimeFact::CrossClassReads { .. } => 69,
        RuntimeFact::StorageTextureWriteRounding { .. } => 70,
        RuntimeFact::HeapWriteRounding { .. } => 71,
        RuntimeFact::HeapCrossClassNeutral { .. } => 72,
        RuntimeFact::AnisotropicSampler { .. } => 73,
        RuntimeFact::StorageTextureIntegerChannels { .. } => 74,
        RuntimeFact::ObjectThreadgroupArena { .. } => 77,
        RuntimeFact::ReplayBindings => 80,
        RuntimeFact::MergedDraws => 65,
        RuntimeFact::QuadPrimitiveIds => 52,
        RuntimeFact::RotatedWeights => 75,
        RuntimeFact::PrivateList => 78,
        RuntimeFact::PartialObjectThreadgroups => 81,
        RuntimeFact::OpenClMode => 83,
        RuntimeFact::RowLockstep => 84,
        RuntimeFact::BorderTexels { .. } => 58,
        RuntimeFact::HeapResidency { .. } => 76,
        RuntimeFact::CallThreadgroups => 82,
        RuntimeFact::PartialThreadgroups => 79,
        RuntimeFact::TexelBufferParts { .. } => 63,
        RuntimeFact::OpenClLineTexture { .. } => 67,
        RuntimeFact::AddressAlignment { .. } => 64,
        RuntimeFact::GridTiles => 50,
        RuntimeFact::PrivateArena { .. } => 68,
        RuntimeFact::StencilBitDraws => 85,
    }
}

#[test]
fn mechanism_is_total_over_every_fact_and_profile() {
    let facts = every_fact();
    let covered: std::collections::BTreeSet<usize> = facts.iter().map(variant).collect();
    assert_eq!(covered, (0..86).collect(), "every_fact misses a variant");
    for (stem, _) in super::committed_profiles() {
        let dev = device(&stem);
        let p = dev.profile();
        let f = p.fields();
        for fact in &facts {
            let m = mechanism(fact, p);
            // Deterministic: one mechanism per (fact, profile).
            assert_eq!(m, mechanism(fact, p));
            let expected = match fact {
                RuntimeFact::ThreadsPerThreadgroup
                | RuntimeFact::ThreadgroupMemoryLength { .. }
                | RuntimeFact::ResourcePresence { .. }
                | RuntimeFact::SampleCount
                | RuntimeFact::SamplerCoordinates { .. }
                | RuntimeFact::AmplificationCount
                | RuntimeFact::ImageblockSize { .. }
                | RuntimeFact::TileFunction
                | RuntimeFact::OutOfBoundsAlpha { .. }
                | RuntimeFact::BicubicTexels { .. }
                | RuntimeFact::BorderTexels { .. }
                | RuntimeFact::SparseTexture {
                    sparse_arm: true, ..
                }
                | RuntimeFact::SparseTexture { tile_map: true, .. }
                | RuntimeFact::SparseTextureWrites { atomic: false, .. }
                | RuntimeFact::TileThreadgroupMemory { .. }
                | RuntimeFact::TextureSwizzle { .. }
                | RuntimeFact::ConstantLanes { .. } => Mechanism::Specialization,
                RuntimeFact::LocalSizeConstraint { .. }
                | RuntimeFact::LocalSizeLimits
                | RuntimeFact::SamplerReduction { .. }
                | RuntimeFact::DeviceAddresses { .. }
                | RuntimeFact::TexelBufferRange { .. }
                | RuntimeFact::SimdgroupEmulation { .. }
                | RuntimeFact::PredicationVote
                | RuntimeFact::SparseTexture {
                    sparse_arm: false,
                    tile_map: false,
                    ..
                }
                | RuntimeFact::SparseTextureWrites { atomic: true, .. }
                | RuntimeFact::CubicArgumentSampler { .. }
                | RuntimeFact::HeapRawSamples { .. }
                | RuntimeFact::StorageTextureWriteRounding { .. }
                | RuntimeFact::HeapWriteRounding { .. }
                | RuntimeFact::HeapCrossClassNeutral { .. }
                | RuntimeFact::HeapResidency { .. }
                | RuntimeFact::HeapStorageReads { .. }
                | RuntimeFact::ObjectThreadgroupArena { .. }
                | RuntimeFact::AddressAlignment { .. }
                | RuntimeFact::RowLockstep => Mechanism::PlanTimeCheck,
                RuntimeFact::DispatchGrid
                | RuntimeFact::TileThreadgroup { .. }
                | RuntimeFact::PartialObjectThreadgroups
                | RuntimeFact::GridTiles => Mechanism::RegionSelected,
                RuntimeFact::IndirectGroupCount => Mechanism::None,
                RuntimeFact::StorageTextureFormat { .. }
                | RuntimeFact::StorageTextureNumericType { .. }
                | RuntimeFact::EmulatedSimdgroups
                | RuntimeFact::RenderTargetArrayLength { .. }
                | RuntimeFact::SrgbTexture { .. }
                | RuntimeFact::CrossClassReads { .. }
                | RuntimeFact::FlatGrid
                | RuntimeFact::CallThreadgroups
                | RuntimeFact::PartialThreadgroups
                | RuntimeFact::AmplificationDraws { .. }
                | RuntimeFact::FoldedGroups
                | RuntimeFact::ReplayBindings
                | RuntimeFact::MergedDraws
                | RuntimeFact::QuadPrimitiveIds
                | RuntimeFact::PrivateList
                | RuntimeFact::RegionViewports
                | RuntimeFact::TexelBufferParts { .. }
                | RuntimeFact::OpenClLineTexture { .. }
                | RuntimeFact::ZeroedColor { .. }
                | RuntimeFact::DroppedColor { .. }
                | RuntimeFact::DepthClampRange
                | RuntimeFact::RotatedWeights
                | RuntimeFact::ConstexprSampled { .. }
                | RuntimeFact::CubicSampler { .. }
                | RuntimeFact::AnisotropicSampler { .. } => Mechanism::Variant,
                RuntimeFact::ImageViewShape { .. }
                | RuntimeFact::StorageTextureLevelViews { .. } => Mechanism::BinderView,
                RuntimeFact::VertexFetch { .. } | RuntimeFact::TileAttachmentFormat { .. } => {
                    Mechanism::KeyProjection
                }
                RuntimeFact::VertexBufferStride { .. } => {
                    if f.dynamic_vertex_stride {
                        Mechanism::DynamicStateOrPushConstant
                    } else {
                        Mechanism::PipelineStateOrPushConstant
                    }
                }
                RuntimeFact::BaseVertexInstance => {
                    if f.draw_parameters {
                        Mechanism::Builtin
                    } else {
                        Mechanism::PushConstant
                    }
                }
                RuntimeFact::BufferRangeLimit { .. } | RuntimeFact::FunctionTableRange { .. } => {
                    match f.buffer_access {
                        crate::device::profile::BufferAccess::Addressed => Mechanism::Variant,
                        _ => Mechanism::PlanTimeCheck,
                    }
                }
                RuntimeFact::FramebufferFetch { .. } | RuntimeFact::TileImageblock { .. } => {
                    Mechanism::RenderPassContract
                }
                RuntimeFact::ExecutionStatus => match f.buffer_access {
                    crate::device::profile::BufferAccess::Addressed => Mechanism::PushConstant,
                    _ => Mechanism::SyntheticBinding,
                },
                RuntimeFact::HeapNumericType { .. } => Mechanism::BinderView,
                RuntimeFact::StageInRegion
                | RuntimeFact::StageInStride { .. }
                | RuntimeFact::StorageTextureLevels { .. }
                | RuntimeFact::StorageTextureIntegerChannels { .. }
                | RuntimeFact::AccelerationStructureInstanceCount { .. }
                | RuntimeFact::ViewportCount
                | RuntimeFact::PrivateArena { .. }
                | RuntimeFact::StencilBitDraws => Mechanism::PushConstant,
                RuntimeFact::AliasedImageblock { .. } => Mechanism::None,
                RuntimeFact::CallStackDepth { .. } | RuntimeFact::OpenClMode => Mechanism::None,
            };
            assert_eq!(m, expected, "{stem}: {fact:?}");
        }
    }
}

#[test]
fn profile_dependent_rows_differ_where_the_design_says() {
    let range = RuntimeFact::BufferRangeLimit {
        slot: SlotRef {
            stage: Stage::Kernel,
            class: SlotClass::Buffer,
            index: 0,
        },
    };
    // Below the runtime's floor: `runtime-floor` as a Vulkan 1.2 context reads it, without device
    // addresses, 64-bit integers or shader draw parameters.
    use crate::device::requirements::VkFeature;
    let mut below = crate::pipeline::testing::description_at_1_2("runtime-floor", false).unwrap();
    for f in [
        VkFeature::BufferDeviceAddress,
        VkFeature::ShaderInt64,
        VkFeature::ShaderDrawParameters,
    ] {
        below.features.remove(&f);
    }
    let floor =
        crate::device::Device::from_queries(below, crate::device::DevicePolicy::RESIDENT_TABLES)
            .unwrap();
    let addressed = device("nvidia-rtx50-linux");
    assert_eq!(mechanism(&range, floor.profile()), Mechanism::PlanTimeCheck);
    assert_eq!(mechanism(&range, addressed.profile()), Mechanism::Variant);
    let stride = RuntimeFact::VertexBufferStride { layout: 0 };
    assert_eq!(
        mechanism(&stride, floor.profile()),
        Mechanism::PipelineStateOrPushConstant
    );
    assert_eq!(
        mechanism(&stride, addressed.profile()),
        Mechanism::DynamicStateOrPushConstant
    );
    // The device reads GPU-written group counts where Metal wrote them.
    assert_eq!(
        mechanism(&RuntimeFact::IndirectGroupCount, floor.profile()),
        Mechanism::None
    );
    assert_eq!(
        mechanism(&RuntimeFact::BaseVertexInstance, floor.profile()),
        Mechanism::PushConstant
    );
}
