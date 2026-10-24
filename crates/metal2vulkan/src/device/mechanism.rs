//! `mechanism(fact, profile)`: how a profile honors each runtime fact.
//!
//! Total by construction: the match below names every [`RuntimeFact`] variant (no wildcard), and
//! each arm returns exactly one [`Mechanism`] from the fact and the profile alone.

use crate::device::profile::{BufferAccess, DeviceProfile};
use crate::pipeline::facts::{Mechanism, RuntimeFact};

/// The one mechanism by which `profile` honors `fact`.
pub fn mechanism(fact: &RuntimeFact, profile: &DeviceProfile) -> Mechanism {
    let f = profile.fields();
    match *fact {
        RuntimeFact::ThreadsPerThreadgroup => Mechanism::Specialization,
        // Arena layout is the exact sum of the Metal allocations, no sink.
        RuntimeFact::ThreadgroupMemoryLength { .. } => Mechanism::Specialization,
        RuntimeFact::LocalSizeConstraint { .. } => Mechanism::PlanTimeCheck,
        RuntimeFact::LocalSizeLimits => Mechanism::PlanTimeCheck,
        // NumWorkgroups for single-region dispatches, push fields for multi-region ones.
        RuntimeFact::DispatchGrid => Mechanism::RegionSelected,
        // GPU-written group counts need the clamp prepass only when some axis limit is below
        // what an indirect argument can hold.
        RuntimeFact::IndirectGroupCount => {
            if f.limits
                .max_compute_work_group_count
                .iter()
                .any(|&axis| axis < u32::MAX)
            {
                Mechanism::IndirectClamp
            } else {
                Mechanism::None
            }
        }
        RuntimeFact::ResourcePresence { .. } => Mechanism::Specialization,
        RuntimeFact::SampleCount => Mechanism::Specialization,
        // Keyed by a per-slot StorageImageChoice, a separate per-function compile.
        RuntimeFact::StorageTextureFormat { .. }
        | RuntimeFact::StorageTextureNumericType { .. } => Mechanism::Variant,
        // Keyed by the slots bound in an sRGB format whose raw code the function reads.
        RuntimeFact::SrgbTexture { .. } => Mechanism::Variant,
        // The binder views a bound 16-bit float texture in its unsigned integer alias at the
        // slot's second binding, and binds the unsigned integer neutral there for any other.
        RuntimeFact::HalfBitsTexture { .. } => Mechanism::BinderView,
        // Keyed by the slots bound in a YCbCr format whose filter the function emulates on a
        // profile that takes the variant (`ProfileFields::ycbcr_variant`).
        RuntimeFact::YCbCrTexture { .. } => Mechanism::Variant,
        RuntimeFact::SamplerCoordinates { .. } => Mechanism::Specialization,
        RuntimeFact::SamplerReduction { .. } => Mechanism::PlanTimeCheck,
        RuntimeFact::ImageViewShape { .. } => Mechanism::BinderView,
        // The binder views the slot's `RG32Uint` texture as `R64_UINT`.
        RuntimeFact::UlongAtomicView { .. } => Mechanism::BinderView,
        RuntimeFact::VertexFetch { .. } => Mechanism::KeyProjection,
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
        RuntimeFact::DeviceAddresses { .. } => Mechanism::PlanTimeCheck,
        // F: DescriptorRangeExceeded at plan time; A: a variant that makes the slot PSB.
        RuntimeFact::BufferRangeLimit { .. } => match f.buffer_access {
            BufferAccess::Addressed => Mechanism::Variant,
            BufferAccess::Floor => Mechanism::PlanTimeCheck,
        },
        RuntimeFact::TexelBufferRange { .. } => Mechanism::PlanTimeCheck,
        RuntimeFact::FramebufferFetch { .. } => Mechanism::RenderPassContract,
        // A function that writes the status buffer reaches it at the device address the binder
        // writes into its push field (`PushField::ExecutionStatusAddress`), or, on a device that
        // does not address buffers, through the stage's binding of it
        // (`SyntheticBinding::ExecutionStatus`).
        RuntimeFact::ExecutionStatus => match f.buffer_access {
            BufferAccess::Addressed => Mechanism::PushConstant,
            BufferAccess::Floor => Mechanism::SyntheticBinding,
        },
        // The binder writes each heap element from the texture's facts and reports a texture
        // the class cannot hold.
        RuntimeFact::HeapNumericType { .. } => Mechanism::BinderView,
        // The scratch's length is a specialization of the local size; the planner checks each
        // region's total against the limit.
        RuntimeFact::SimdgroupEmulation { .. } => Mechanism::PlanTimeCheck,
        // The count is fixed with the view mask a Vulkan pipeline is created for.
        RuntimeFact::AmplificationCount => Mechanism::Specialization,
        // A separate compile of the kernel, which a dispatch with a partial simdgroup asks for.
        RuntimeFact::EmulatedSimdgroups => Mechanism::Variant,
        // The imageblock's extent sizes its Workgroup region; the planner checks the total.
        RuntimeFact::ImageblockSize { .. } => Mechanism::Specialization,
        // A separate compile of the vertex function, which a pass of one layer asks for.
        RuntimeFact::RenderTargetArrayLength { .. } => Mechanism::Variant,
        // The binder writes the region's origin and size per dispatch and checks the axes whose
        // origin must be 0.
        RuntimeFact::StageInRegion => Mechanism::PushConstant,
        // The binder writes the bound stride per dispatch.
        RuntimeFact::StageInStride { .. } => Mechanism::PushConstant,
        // A shape that does not prove a uniform view's range fits asks for the variant that keeps
        // the slot a storage view.
        RuntimeFact::UniformViewRange { .. } => Mechanism::Variant,
        // Instantiation compares the bound Metal addresses.
        RuntimeFact::UniformViewAliasing { .. } => Mechanism::PlanTimeCheck,
        // The pass's imageblock storage image, bound to every draw of the pass.
        RuntimeFact::FragmentImageblock { .. } => Mechanism::RenderPassContract,
        // The render area is specialized; the planner refuses a layered pass.
        RuntimeFact::TileFunction => Mechanism::Specialization,
        // Compiled in under the function's key; the tile pipeline checks its descriptor.
        RuntimeFact::TileAttachmentFormat { .. } => Mechanism::KeyProjection,
        // The binder specializes the alpha from the bound texture's pixel format.
        RuntimeFact::OutOfBoundsAlpha { .. } => Mechanism::Specialization,
        // The bound format's treatment is specialized; the planner refuses an unmeasured
        // format.
        RuntimeFact::BicubicTexels { .. } => Mechanism::Specialization,
        // The binder writes the bound texture's level count per draw or dispatch.
        RuntimeFact::StorageTextureLevels { .. } => Mechanism::PushConstant,
        // The binder writes whether the bound texture's channels are 16-bit floats.
        RuntimeFact::StorageTextureHalfChannels { .. } => Mechanism::PushConstant,
        // The binder writes the bound texture's integer channel widths.
        RuntimeFact::StorageTextureIntegerChannels { .. } => Mechanism::PushConstant,
        // The binder views each level of the bound texture at its element.
        RuntimeFact::StorageTextureLevelViews { .. } => Mechanism::BinderView,
        // The bound texture's sparseness selects the sparse form or the tile map where the module
        // has one; without either the planner refuses a sparse texture.
        RuntimeFact::SparseTexture {
            sparse_arm: true, ..
        }
        | RuntimeFact::SparseTexture { tile_map: true, .. } => Mechanism::Specialization,
        RuntimeFact::SparseTexture {
            sparse_arm: false,
            tile_map: false,
            ..
        } => Mechanism::PlanTimeCheck,
        // The binder writes the bound structure's instance count per dispatch or draw.
        RuntimeFact::AccelerationStructureInstanceCount { .. } => Mechanism::PushConstant,
        // The consumer binds the attachment as a storage image outside the subpass's color
        // attachments (`GraphicsState::storage_attachments`).
        RuntimeFact::InterlockedFramebufferFetch { .. } => Mechanism::RenderPassContract,
        // The consumer keeps the stencil's shadow and binds it, with the encoder's stencil state
        // as push data (`GraphicsState::shadow_stencil`).
        RuntimeFact::ShadowStencil { .. } => Mechanism::RenderPassContract,
        // The consumer keeps the depth attachment's stand-in and binds it, with the encoder's
        // depth state as push data (`GraphicsState::shader_tests`).
        RuntimeFact::ShaderTests { .. } => Mechanism::RenderPassContract,
        // The consumer keeps the stencil attachment's stand-in and binds it, with the encoder's
        // stencil state and references as push data (`ShaderTestsUse::stencil`).
        RuntimeFact::ShaderStencilTest { .. } => Mechanism::RenderPassContract,
        // The vote word is fixed in the module; the planner counts it against the limit.
        RuntimeFact::PredicationVote => Mechanism::PlanTimeCheck,
        // The binder pushes the bound format's class, which selects its conversion.
        RuntimeFact::TexelFormat { .. } => Mechanism::PushConstant,
        // The binder writes the encoder's viewport count per draw.
        RuntimeFact::ViewportCount => Mechanism::PushConstant,
        // A Metal query's answer: nothing on the device reads it.
        RuntimeFact::AliasedImageblock { .. } => Mechanism::None,
        // The depth a kernel's call stack was sized for: the compile layer checks it against the
        // pipeline's; nothing on the device reads it.
        RuntimeFact::CallStackDepth { .. } => Mechanism::None,
        // A pipeline compiled under the flat-grid variant; its plans state it.
        RuntimeFact::FlatGrid => Mechanism::Variant,
        // A sparse texture's writes are dropped under the slot's `SparseTexture` specialization;
        // at a slot updated atomically the planner refuses one.
        RuntimeFact::SparseTextureWrites { atomic: false, .. } => Mechanism::Specialization,
        RuntimeFact::SparseTextureWrites { atomic: true, .. } => Mechanism::PlanTimeCheck,
        // The offsets and the tile memory are specialized; the pass binds its tile memory.
        RuntimeFact::TileThreadgroupMemory { .. } => Mechanism::Specialization,
        // The binder checks the object function's arena with the encoder's lengths, which it
        // specializes, against the object stage's workgroup memory limit.
        RuntimeFact::ObjectThreadgroupArena { .. } => Mechanism::PlanTimeCheck,
        // The binder specializes the bound texture buffer's swizzle, which the function applies
        // to its reads.
        RuntimeFact::TextureSwizzle { .. } => Mechanism::Specialization,
        // The consumer keeps the pass's imageblock storage and binds it to the dispatch.
        RuntimeFact::TileImageblock { .. } => Mechanism::RenderPassContract,
        // A separate compile of the pipeline, which an amplification multiview does not express
        // asks for; the binder writes each amplification's words before its draw.
        RuntimeFact::AmplificationDraws { .. } => Mechanism::Variant,
        // A separate compile of the pipeline, which a viewport Vulkan's cannot be asks for; the
        // binder writes the draw's compensation.
        RuntimeFact::ViewportCompensation { .. } => Mechanism::Variant,
        // A pipeline compiled under the folded-groups variant; its plans state it.
        RuntimeFact::FoldedGroups => Mechanism::Variant,
        // A render pipeline compiled under the replay-bindings variant; its plans state it.
        RuntimeFact::ReplayBindings => Mechanism::Variant,
        // A render pipeline compiled under the merged-draws variant; its plans state it.
        RuntimeFact::MergedDraws => Mechanism::Variant,
        RuntimeFact::QuadPrimitiveIds => Mechanism::Variant,
        RuntimeFact::PrivateList => Mechanism::Variant,
        // The function marks the `ExecutionStatus` buffer its `ExecutionStatus` fact binds; the
        // consumer reads the bit after the command buffer.
        RuntimeFact::AbsentArgumentRead { .. } => Mechanism::None,
        // The planner refuses an sRGB texture the device cannot sample at a slot the function
        // decodes through the device.
        RuntimeFact::SrgbDeviceDecode { .. } => Mechanism::PlanTimeCheck,
        // A separate compile of the pipeline, which a pass with colors the pipeline lacks asks
        // for.
        RuntimeFact::ZeroedColor { .. } | RuntimeFact::DroppedColor { .. } => Mechanism::Variant,
        // A separate compile of the fragment function, which a draw in depth clamp mode through
        // a range above 0 asks for.
        RuntimeFact::DepthClampRange => Mechanism::Variant,
        RuntimeFact::RotatedWeights => Mechanism::Variant,
        // A separate link of the mesh pipeline's synthesized vertex stage.
        RuntimeFact::MeshLastFirst => Mechanism::Variant,
        // The binder specializes the lanes from the bound texture's pixel format and swizzle.
        RuntimeFact::ConstantLanes { .. } => Mechanism::Specialization,
        // A separate compile that reads the tables' words through their device addresses, which
        // a shape whose tables may pass `maxStorageBufferRange` asks for; without device
        // addresses the binder refuses such a table where it binds it.
        RuntimeFact::FunctionTableRange { .. } => match f.buffer_access {
            BufferAccess::Addressed => Mechanism::Variant,
            BufferAccess::Floor => Mechanism::PlanTimeCheck,
        },
        // A separate compile that declares early tests, which a draw whose tests drop fragments
        // but write and count nothing asks for; the planner refuses a draw whose tests may write
        // or count.
        RuntimeFact::TestedStorageWrites { .. } => Mechanism::Variant,
        // The tests a draw that moves the function's color into the interlocked storage form
        // selects: that variant's.
        RuntimeFact::MovableStorageWrites { .. } => Mechanism::Variant,
        // The binder's regions carry the tile as Metal's threadgroup; the tile pipeline checks its
        // descriptor's flag.
        RuntimeFact::TileThreadgroup { .. } => Mechanism::RegionSelected,
        // Keyed by the slots a constexpr sampler samples whose bound texture's address-mode map
        // is not the identity.
        RuntimeFact::ConstexprSampled { .. } => Mechanism::Variant,
        // Keyed by the slots bound to a cubic sampler, with its state.
        RuntimeFact::CubicSampler { .. } => Mechanism::Variant,
        // The planner admits a listed cubic sampler beside the stage's sampler table; the
        // function marks the `ExecutionStatus` buffer where it reaches one.
        RuntimeFact::CubicArgumentSampler { .. } => Mechanism::PlanTimeCheck,
        // A pipeline compiled in the capture form, which a consumer asks for; its plans state
        // it, and the binder writes each draw's capture layout.
        RuntimeFact::CapturedPositions { .. } => Mechanism::Variant,
        // The planner refuses a texture a heap holds by its raw texels alone that a plan names
        // beside the stage; a consumer's own heap announces one.
        RuntimeFact::HeapRawSamples { .. } => Mechanism::PlanTimeCheck,
        // Link gives the attachment no Vulkan blend and its input attachment, which the consumer
        // binds as `GraphicsState::input_attachments` states.
        RuntimeFact::ExtendedRangeInputBlend { .. } => Mechanism::RenderPassContract,
        // Link writes the whole word and gives the attachment its input attachment, which the
        // consumer binds as `GraphicsState::input_attachments` states.
        RuntimeFact::WordInputMerge { .. } => Mechanism::RenderPassContract,
        // Keyed by the slots bound in a format of another numeric class than the declared one,
        // whose texels the function reads in that class.
        RuntimeFact::CrossClassReads { .. } => Mechanism::Variant,
        // The planner refuses a half-float or packed small-float texture bound to the slot, or
        // reached in a heap's storage class by the stage, that a mode-2 write the translation has
        // not built would write.
        RuntimeFact::StorageTextureWriteRounding { .. } | RuntimeFact::HeapWriteRounding { .. } => {
            Mechanism::PlanTimeCheck
        }
        // The planner reports a member of another numeric class the stage reaches in the class.
        RuntimeFact::HeapCrossClassNeutral { .. } => Mechanism::PlanTimeCheck,
        // Keyed by the slots bound to a sampler whose word states anisotropy, where the
        // function's samples take Metal's anisotropic filter, emulated.
        RuntimeFact::AnisotropicSampler { .. } => Mechanism::Variant,
        // The binder holds each half-float member the twin takes there; the consumer announces
        // a member it does not take where its heap holds one.
        RuntimeFact::HeapHalfTexelWrites { .. } => Mechanism::BinderView,
        // The binder pushes the draw's depth bias; the consumer records its Vulkan bias as zero.
        RuntimeFact::Depth16Codes { biased: true } => Mechanism::PushConstant,
        // The function's returned depth, which Metal does not bias: nothing to meet.
        RuntimeFact::Depth16Codes { biased: false } => Mechanism::None,
        // A separate compile of the pipeline, which a pass whose depth is or is not
        // `Depth16Unorm` where the pipeline names otherwise asks for.
        RuntimeFact::Depth16Pass { .. } => Mechanism::Variant,
        // A separate compile of the pipeline, which a draw that biases a `Depth16Unorm` depth
        // asks for where the device stores its biased depths otherwise than Metal.
        RuntimeFact::Depth16Biased => Mechanism::Variant,
        // A separate compile of the pipeline, which a draw whose depth bias has a proven side
        // asks for where the fragment stage adds the bias itself.
        RuntimeFact::Depth16BiasSide { .. } => Mechanism::Variant,
        // The pipeline's own stages, which every draw of its class runs; the consumer announces
        // them.
        RuntimeFact::SynthesizedGeometry { .. } => Mechanism::None,
        // The pipeline's exact forms were dropped to fit the device's compiler word bound; the
        // consumer announces it, and the reduced stages carry their own notes.
        RuntimeFact::ExactFormsOverCompilerLimit { .. } => Mechanism::None,
    }
}
