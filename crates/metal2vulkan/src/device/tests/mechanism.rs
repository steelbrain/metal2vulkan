//! `mechanism()` totality over every (fact, committed profile).

use super::profile::device;
use crate::device::mechanism::mechanism;
use crate::pipeline::facts::{Mechanism, RuntimeFact, UniformViewChoice};
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
        RuntimeFact::UniformViewRange {
            slot,
            choice: UniformViewChoice::Viewed,
        },
        RuntimeFact::UniformViewRange {
            slot,
            choice: UniformViewChoice::Spare,
        },
        RuntimeFact::UniformViewRange {
            slot,
            choice: UniformViewChoice::Kept,
        },
        RuntimeFact::UniformViewAliasing { slot },
        RuntimeFact::FragmentImageblock {
            words: 4,
            writes: true,
            master: [7; 32],
            data_size: 16,
            alignment: 16,
        },
        RuntimeFact::TileFunction,
        RuntimeFact::TileAttachmentFormat {
            color: 1,
            format: crate::metal::PixelFormat::R32Float,
        },
        RuntimeFact::StorageTextureNumericType { slot },
        RuntimeFact::OutOfBoundsAlpha { slot },
        RuntimeFact::BicubicTexels { slot },
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
        RuntimeFact::InterlockedFramebufferFetch {
            color: 0,
            format: crate::vk::Format::R32G32B32A32_SFLOAT,
            writes: true,
            early_tests: true,
            kept: crate::vk::ColorComponentFlags::empty(),
            blend: None,
            rounded: false,
        },
        RuntimeFact::ShadowStencil {
            formatless: false,
            multisampled: false,
        },
        RuntimeFact::PredicationVote,
        RuntimeFact::TexelFormat { slot },
        RuntimeFact::SrgbTexture { slot },
        RuntimeFact::HalfBitsTexture { slot },
        RuntimeFact::UlongAtomicView { slot },
        RuntimeFact::ViewportCount,
        RuntimeFact::AliasedImageblock { data_size: 24 },
        RuntimeFact::CallStackDepth { depth: 8 },
        RuntimeFact::FlatGrid,
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
        RuntimeFact::ViewportCompensation {
            depth: true,
            point_guard: true,
            mirrors: true,
            vertex_clamp: true,
        },
        RuntimeFact::FoldedGroups,
        RuntimeFact::ReplayBindings,
        RuntimeFact::MergedDraws,
        RuntimeFact::QuadPrimitiveIds,
        RuntimeFact::RotatedWeights,
        RuntimeFact::PrivateList,
        RuntimeFact::MeshLastFirst,
        RuntimeFact::AbsentArgumentRead {
            stage: Stage::Fragment,
        },
        RuntimeFact::SrgbDeviceDecode { slot },
        RuntimeFact::ZeroedColor {
            color: 1,
            format: crate::metal::PixelFormat::RGBA8Unorm,
        },
        RuntimeFact::DroppedColor { color: 1 },
        RuntimeFact::DepthClampRange,
        RuntimeFact::ConstantLanes { slot },
        RuntimeFact::FunctionTableRange { addressed: false },
        RuntimeFact::TestedStorageWrites {
            early_tests: false,
            early_variant: true,
        },
        RuntimeFact::YCbCrTexture { slot },
        RuntimeFact::TileThreadgroup { tile_sized: true },
        RuntimeFact::ConstexprSampled {
            slot,
            modes: [crate::metal::SamplerAddressMode::Repeat; 3],
        },
        RuntimeFact::CubicSampler { slot },
        RuntimeFact::CubicArgumentSampler {
            stage: Stage::Kernel,
        },
        RuntimeFact::ShaderTests {
            formatless: false,
            multisampled: true,
            planes: true,
        },
        RuntimeFact::StorageTextureHalfChannels { slot },
        RuntimeFact::StorageTextureIntegerChannels { slot },
        RuntimeFact::CapturedPositions {
            form: crate::pipeline::CaptureForm::Indexed(crate::metal::IndexType::UInt16),
        },
        RuntimeFact::HeapRawSamples {
            stage: Stage::Fragment,
            sampler: Some(1),
        },
        RuntimeFact::ShaderStencilTest { formatless: true },
        RuntimeFact::ExtendedRangeInputBlend {
            color: 0,
            blend: {
                let mut c = crate::pipeline::ColorAttachmentDescriptor::new(
                    crate::metal::PixelFormat::BGRA10XR,
                );
                c.blending_enabled = true;
                crate::pipeline::lower_inputs::BlendEquation::of(&c).expect("a blend")
            },
            kept: crate::vk::ColorComponentFlags::empty(),
        },
        RuntimeFact::WordInputMerge {
            color: 1,
            kept: crate::vk::ColorComponentFlags::G,
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
        RuntimeFact::AnisotropicSampler {
            slot: SlotRef {
                stage: Stage::Fragment,
                class: SlotClass::Sampler,
                index: 0,
            },
        },
        RuntimeFact::HeapHalfTexelWrites {
            stage: Stage::Kernel,
            packed: true,
        },
        RuntimeFact::MovableStorageWrites {
            early_variant: true,
        },
        RuntimeFact::Depth16Codes { biased: true },
        RuntimeFact::Depth16Codes { biased: false },
        RuntimeFact::Depth16Pass { depth16: true },
        RuntimeFact::Depth16Biased,
        RuntimeFact::Depth16BiasSide { raises: true },
        RuntimeFact::Depth16BiasSide { raises: false },
        RuntimeFact::SynthesizedGeometry {
            weights: true,
            layer: false,
            viewport: false,
        },
        RuntimeFact::SynthesizedGeometry {
            weights: false,
            layer: true,
            viewport: true,
        },
        RuntimeFact::ObjectThreadgroupArena {
            static_bytes: 16,
            payload: Some(64),
        },
        RuntimeFact::ObjectThreadgroupArena {
            static_bytes: 0,
            payload: None,
        },
        RuntimeFact::ExactFormsOverCompilerLimit {
            sampling: true,
            float_maths: true,
            texel_conversion: false,
            half_rounding: false,
            nan_and_min_max: true,
            rounding_order: false,
            half_blend: false,
        },
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
        RuntimeFact::UniformViewRange { .. } => 28,
        RuntimeFact::UniformViewAliasing { .. } => 29,
        RuntimeFact::FragmentImageblock { .. } => 30,
        RuntimeFact::TileFunction => 31,
        RuntimeFact::TileAttachmentFormat { .. } => 32,
        RuntimeFact::StorageTextureNumericType { .. } => 33,
        RuntimeFact::OutOfBoundsAlpha { .. } => 34,
        RuntimeFact::BicubicTexels { .. } => 35,
        RuntimeFact::StorageTextureLevels { .. } => 36,
        RuntimeFact::StorageTextureLevelViews { .. } => 37,
        RuntimeFact::SparseTexture { .. } => 38,
        RuntimeFact::AccelerationStructureInstanceCount { .. } => 39,
        RuntimeFact::InterlockedFramebufferFetch { .. } => 40,
        RuntimeFact::ShadowStencil { .. } => 41,
        RuntimeFact::PredicationVote => 42,
        RuntimeFact::TexelFormat { .. } => 43,
        RuntimeFact::SrgbTexture { .. } => 44,
        RuntimeFact::HalfBitsTexture { .. } => 98,
        RuntimeFact::UlongAtomicView { .. } => 45,
        RuntimeFact::ViewportCount => 46,
        RuntimeFact::AliasedImageblock { .. } => 47,
        RuntimeFact::CallStackDepth { .. } => 48,
        RuntimeFact::FlatGrid => 49,
        RuntimeFact::SparseTextureWrites { .. } => 50,
        RuntimeFact::TileThreadgroupMemory { .. } => 51,
        RuntimeFact::TextureSwizzle { .. } => 52,
        RuntimeFact::TileImageblock { .. } => 53,
        RuntimeFact::AmplificationDraws { .. } => 54,
        RuntimeFact::ViewportCompensation { .. } => 55,
        RuntimeFact::FoldedGroups => 56,
        RuntimeFact::AbsentArgumentRead { .. } => 57,
        RuntimeFact::SrgbDeviceDecode { .. } => 58,
        RuntimeFact::ZeroedColor { .. } => 59,
        RuntimeFact::DroppedColor { .. } => 60,
        RuntimeFact::DepthClampRange => 61,
        RuntimeFact::ConstantLanes { .. } => 62,
        RuntimeFact::FunctionTableRange { .. } => 63,
        RuntimeFact::TestedStorageWrites { .. } => 64,
        RuntimeFact::YCbCrTexture { .. } => 65,
        RuntimeFact::TileThreadgroup { .. } => 66,
        RuntimeFact::ConstexprSampled { .. } => 67,
        RuntimeFact::CubicSampler { .. } => 68,
        RuntimeFact::CubicArgumentSampler { .. } => 69,
        RuntimeFact::ShaderTests { .. } => 70,
        RuntimeFact::StorageTextureHalfChannels { .. } => 71,
        RuntimeFact::CapturedPositions { .. } => 72,
        RuntimeFact::HeapRawSamples { .. } => 73,
        RuntimeFact::ShaderStencilTest { .. } => 74,
        RuntimeFact::ExtendedRangeInputBlend { .. } => 75,
        RuntimeFact::CrossClassReads { .. } => 76,
        RuntimeFact::StorageTextureWriteRounding { .. } => 77,
        RuntimeFact::HeapWriteRounding { .. } => 78,
        RuntimeFact::HeapCrossClassNeutral { .. } => 79,
        RuntimeFact::AnisotropicSampler { .. } => 80,
        RuntimeFact::StorageTextureIntegerChannels { .. } => 81,
        RuntimeFact::HeapHalfTexelWrites { .. } => 82,
        RuntimeFact::MovableStorageWrites { .. } => 83,
        RuntimeFact::Depth16Codes { .. } => 84,
        RuntimeFact::Depth16Pass { .. } => 85,
        RuntimeFact::Depth16Biased => 86,
        RuntimeFact::Depth16BiasSide { .. } => 87,
        RuntimeFact::SynthesizedGeometry { .. } => 88,
        RuntimeFact::ObjectThreadgroupArena { .. } => 89,
        RuntimeFact::ExactFormsOverCompilerLimit { .. } => 90,
        RuntimeFact::WordInputMerge { .. } => 91,
        RuntimeFact::ReplayBindings => 92,
        RuntimeFact::MergedDraws => 93,
        RuntimeFact::QuadPrimitiveIds => 94,
        RuntimeFact::RotatedWeights => 95,
        RuntimeFact::PrivateList => 96,
        RuntimeFact::MeshLastFirst => 97,
    }
}

#[test]
fn mechanism_is_total_over_every_fact_and_profile() {
    let facts = every_fact();
    let covered: std::collections::BTreeSet<usize> = facts.iter().map(variant).collect();
    assert_eq!(covered, (0..99).collect(), "every_fact misses a variant");
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
                | RuntimeFact::UniformViewAliasing { .. }
                | RuntimeFact::SparseTexture {
                    sparse_arm: false,
                    tile_map: false,
                    ..
                }
                | RuntimeFact::SparseTextureWrites { atomic: true, .. }
                | RuntimeFact::SrgbDeviceDecode { .. }
                | RuntimeFact::CubicArgumentSampler { .. }
                | RuntimeFact::HeapRawSamples { .. }
                | RuntimeFact::StorageTextureWriteRounding { .. }
                | RuntimeFact::HeapWriteRounding { .. }
                | RuntimeFact::HeapCrossClassNeutral { .. }
                | RuntimeFact::ObjectThreadgroupArena { .. } => Mechanism::PlanTimeCheck,
                RuntimeFact::DispatchGrid | RuntimeFact::TileThreadgroup { .. } => {
                    Mechanism::RegionSelected
                }
                RuntimeFact::IndirectGroupCount => {
                    if f.limits.max_compute_work_group_count == [u32::MAX; 3] {
                        Mechanism::None
                    } else {
                        Mechanism::IndirectClamp
                    }
                }
                RuntimeFact::StorageTextureFormat { .. }
                | RuntimeFact::StorageTextureNumericType { .. }
                | RuntimeFact::EmulatedSimdgroups
                | RuntimeFact::RenderTargetArrayLength { .. }
                | RuntimeFact::UniformViewRange { .. }
                | RuntimeFact::SrgbTexture { .. }
                | RuntimeFact::CrossClassReads { .. }
                | RuntimeFact::FlatGrid
                | RuntimeFact::AmplificationDraws { .. }
                | RuntimeFact::ViewportCompensation { .. }
                | RuntimeFact::FoldedGroups
                | RuntimeFact::ReplayBindings
                | RuntimeFact::MergedDraws
                | RuntimeFact::QuadPrimitiveIds
                | RuntimeFact::PrivateList
                | RuntimeFact::ZeroedColor { .. }
                | RuntimeFact::DroppedColor { .. }
                | RuntimeFact::DepthClampRange
                | RuntimeFact::RotatedWeights
                | RuntimeFact::MeshLastFirst
                | RuntimeFact::Depth16Pass { .. }
                | RuntimeFact::Depth16Biased
                | RuntimeFact::Depth16BiasSide { .. }
                | RuntimeFact::TestedStorageWrites { .. }
                | RuntimeFact::MovableStorageWrites { .. }
                | RuntimeFact::YCbCrTexture { .. }
                | RuntimeFact::ConstexprSampled { .. }
                | RuntimeFact::CubicSampler { .. }
                | RuntimeFact::AnisotropicSampler { .. }
                | RuntimeFact::CapturedPositions { .. } => Mechanism::Variant,
                RuntimeFact::ImageViewShape { .. }
                | RuntimeFact::StorageTextureLevelViews { .. }
                | RuntimeFact::HalfBitsTexture { .. }
                | RuntimeFact::UlongAtomicView { .. } => Mechanism::BinderView,
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
                RuntimeFact::FramebufferFetch { .. }
                | RuntimeFact::FragmentImageblock { .. }
                | RuntimeFact::InterlockedFramebufferFetch { .. }
                | RuntimeFact::ShadowStencil { .. }
                | RuntimeFact::ShaderTests { .. }
                | RuntimeFact::ShaderStencilTest { .. }
                | RuntimeFact::TileImageblock { .. }
                | RuntimeFact::ExtendedRangeInputBlend { .. }
                | RuntimeFact::WordInputMerge { .. } => Mechanism::RenderPassContract,
                RuntimeFact::ExecutionStatus => match f.buffer_access {
                    crate::device::profile::BufferAccess::Addressed => Mechanism::PushConstant,
                    _ => Mechanism::SyntheticBinding,
                },
                RuntimeFact::HeapNumericType { .. }
                | RuntimeFact::HeapHalfTexelWrites { .. } => Mechanism::BinderView,
                RuntimeFact::StageInRegion
                | RuntimeFact::StageInStride { .. }
                | RuntimeFact::StorageTextureLevels { .. }
                | RuntimeFact::StorageTextureHalfChannels { .. }
                | RuntimeFact::StorageTextureIntegerChannels { .. }
                | RuntimeFact::AccelerationStructureInstanceCount { .. }
                | RuntimeFact::ViewportCount
                | RuntimeFact::Depth16Codes { biased: true }
                // The bound texture's format class, which the binder pushes.
                | RuntimeFact::TexelFormat { .. } => Mechanism::PushConstant,
                RuntimeFact::AliasedImageblock { .. } => Mechanism::None,
                RuntimeFact::CallStackDepth { .. }
                | RuntimeFact::AbsentArgumentRead { .. }
                | RuntimeFact::SynthesizedGeometry { .. }
                | RuntimeFact::ExactFormsOverCompilerLimit { .. }
                | RuntimeFact::Depth16Codes { biased: false } => Mechanism::None,
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
    let floor = device("vk12-floor");
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
    // Vulkan 1.2 minimum group counts are 65535 per axis: indirect dispatches are clamped.
    assert_eq!(
        mechanism(&RuntimeFact::IndirectGroupCount, floor.profile()),
        Mechanism::IndirectClamp
    );
    assert_eq!(
        mechanism(&RuntimeFact::BaseVertexInstance, floor.profile()),
        Mechanism::PushConstant
    );
}
