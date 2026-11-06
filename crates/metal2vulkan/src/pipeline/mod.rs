//! Pipelines: Metal-shaped descriptors in, Vulkan-ready artifacts out.
//!
//! **Descriptors.** [`ComputePipelineDescriptor`] and [`RenderPipelineDescriptor`] mirror Metal's,
//! with [`VertexDescriptor`], [`ColorAttachmentDescriptor`], and the rest. Each stage is a
//! [`StageSource`]: a [`FunctionDescriptor`] to compile (a library function, its constant values,
//! and its [`LinkedFunctions`]), or a [`CompiledFunction`](crate::library::CompiledFunction) to
//! link as it is. Construct a descriptor with `new`, which holds Metal's defaults, and assign the
//! fields that differ: the descriptors are `#[non_exhaustive]`, so struct literals do not compile
//! outside the crate.
//!
//! **Artifacts.** [`Device::compile_compute_pipeline`](crate::device::Device::compile_compute_pipeline)
//! and [`Device::compile_render_pipeline`](crate::device::Device::compile_render_pipeline) return a
//! [`ComputePipeline`] or a [`RenderPipeline`]: immutable plain data, cheap to clone (the SPIR-V
//! sits behind an `Arc`), holding
//!
//! - per stage a [`StageModule`]: SPIR-V 1.5 words valid for the device's profile, the entry
//!   point, the [`SpecializationLayout`], a required subgroup size and stage flags, and the
//!   stage's [`PrecisionNote`]s;
//! - one [`PipelineLayout`]: dense descriptor set layouts and push-constant ranges, to create as
//!   they are (no renumbering);
//! - a render pipeline's [`VertexInputState`] and [`GraphicsState`] (with a
//!   [`TessellationState`] when it tessellates, and a mesh render pipeline's answers to Metal's
//!   mesh queries, [`MeshPipelineMetalInfo`]), or a compute pipeline's answers to Metal's
//!   per-pipeline queries ([`ComputePipelineMetalInfo`]);
//! - the [`PipelineInterface`]: every statically used Metal argument slot ([`SlotRef`]), how the
//!   shader accesses it, and each buffer's proven byte window;
//! - its [`RuntimeDependency`]s: each piece of bind-time state it depends on ([`RuntimeFact`])
//!   and how this profile honors it ([`Mechanism`]);
//! - its [`Requirements`](crate::device::Requirements): what its modules need of the device.
//!
//! A pipeline's `plan` turns what the consumer binds into descriptor writes and push bytes (see
//! [`bind`](crate::bind)).
//!
//! **Keys and variants.** `Device::compute_pipeline_key` computes a pipeline's key without
//! decoding; `Device::render_pipeline_key` decodes (but does not lower) the vertex function to
//! learn the attributes it reads, and returns [`RenderPipelineKeys`]: the pipeline's key, each
//! stage's function key, and the [`LoweringInputs`] each function stage compiles under (see
//! [`cache`](crate::cache)). A binding plan can ask for another variant of the pipeline, named by
//! a [`VariantKey`](crate::cache::VariantKey) (`BindingPlan::variant`,
//! `BindError::NeedsVariant`). A render pipeline compiles its variant with
//! `Device::compile_render_variant`; a compute pipeline compiles its kernel with
//! `Device::compile_function` under `VariantKey::lowering_inputs(Stage::Kernel)` and links it as
//! `StageSource::Compiled`.
//!
//! **Stage inputs and tessellation.** A kernel's `[[stage_in]]` attributes are fetched in the
//! shader as the compute descriptor's `stage_input` ([`StageInputDescriptor`]) describes; a
//! kernel compiled on its own for such a pipeline compiles under the same descriptor
//! (`LoweringInputs::stage_input`). A dispatch of a kernel that steps an attribute by thread
//! position, or reads `[[grid_origin]]` or `[[grid_size]]`, needs the encoder's stage-in region
//! (`BindState::stage_in_region`). A render pipeline whose vertex function is a post-tessellation
//! vertex function is a tessellation pipeline, under the descriptor's `tessellation`
//! ([`TessellationDescriptor`], or Metal's defaults): a synthesized vertex and tessellation
//! control stage, the function's evaluation stage, and the fragment stage when rasterization is
//! enabled. It draws patches (`DrawCall::Patches`).
//!
//! **Tile pipelines.** `Device::compile_tile_pipeline` compiles a [`TileRenderPipelineDescriptor`]
//! into a [`ComputePipeline`]: its tile function compiled under the descriptor's color attachment
//! formats (`LoweringInputs::tile_attachments`), which the consumer dispatches over the render
//! area's tiles in place of `dispatchThreadsPerTile:`, with the pass's tile size
//! (`BindState::imageblock_size`) and render area (`BindState::render_area`), and the pass's
//! attachments bound as storage images in the plan's formats (`WriteSource::TileAttachment`: a
//! float attachment through the unsigned-integer format of its channels, so its bits move).
//!
//! **Mesh pipelines.** `Device::compile_mesh_pipeline` compiles a [`MeshPipelineDescriptor`]
//! into a [`RenderPipeline`] whose stages are its object (`TASK_EXT`), mesh (`MESH_EXT`) and
//! fragment stages, with no vertex input; its fragment function reads the mesh function's primitive
//! outputs per primitive (`LoweringInputs::primitive_varyings`). Its plan draws mesh draws
//! (`DrawCall::MeshThreadgroups` and its siblings), each with the specialization of its mesh
//! threadgroup's size (`DrawCommands::specialization`).
//!
//! **Not implemented yet.** The plan of a mesh pipeline with an object function, an object
//! function's payload, and a patch draw with a patch index buffer are refused, each as
//! `Unsupported` with `Needs::NotImplemented`. Additions are expected.
//!
//! The rest of this module is the vocabulary those types share: slots and stages ([`SlotRef`],
//! [`Stage`], [`SlotClass`]), specialization constants ([`SpecKey`], [`SpecConstant`]), push
//! fields ([`PushField`], [`PushPlacement`]), descriptor sources ([`BindingSource`],
//! [`ResourceSlot`]), handle-heap classes ([`HandleClass`]), and varyings ([`VaryingKey`],
//! [`AirUserKey`]).
//!
//! The frozen surface is the items named at this root; the submodules are crate-private.

pub(crate) mod artifact;
pub(crate) mod color_writes;
pub(crate) mod compile;
pub(crate) mod descriptor;
pub(crate) mod facts;
pub(crate) mod graphics;
pub(crate) mod interface;
pub(crate) mod layout;
pub(crate) mod link;
pub(crate) mod lower_inputs;
pub(crate) mod private_state;
pub(crate) mod relocate;
pub(crate) mod rotated_blend;
#[cfg(feature = "serde")]
pub(crate) mod serde_pairs;
pub(crate) mod stage_io;
pub(crate) mod sym;
pub(crate) mod table;
// Test support for the crate's tests.
#[cfg(test)]
pub(crate) mod testing;
pub(crate) mod tile_map;
pub(crate) mod vertex;

pub use artifact::{
    ComputePipeline, ComputePipelineMetalInfo, EntryMath, MeshPipelineMetalInfo, ModuleStats,
    PrecisionNote, Reconvergence, RenderPipeline, RuntimeDependency, SpecClause, SpecConstant,
    SpecTest, SpecializationLayout, SpecializedNote, StageModule, StrategyCensus, StrategyUse,
};
pub use descriptor::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, DynamicLibraries, FunctionDescriptor,
    LinkedFunction, LinkedFunctions, LinkedList, MeshPipelineDescriptor, PrivateRenderState,
    RenderPipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource, TessellationDescriptor, TileRenderPipelineDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
pub use graphics::{
    BandPrimitive, BlendState, ColorAttachmentState, EmulatedTessellation, FragmentImageblock,
    GraphicsState, ImageblockMechanism, InputAttachmentUse, LineBandState, ShaderTestsUse,
    ShadowStencilUse, StandInUse, StandInView, StorageAttachmentUse, StorageAttachmentView,
    TessellationState,
};
pub use interface::{
    AccessSet, AttributeFetch, BufferForm, BufferKind, BufferSlot, CommandWrites, DescriptorView,
    MeshRecords, NumberedSampler, PatchDomain, PipelineInterface, PlacedField, PushLoc,
    PushPlacement, RecordField, RecordValue, SamplerSlot, SlotInterface, TextureAccess,
    TextureSlot, ThreadgroupArgument, Window,
};
pub use layout::{
    BindingSource, DescriptorBinding, DescriptorSetLayout, PipelineLayout, PushConstantRange,
};
pub use link::{heap_sampler_capacity, heap_unused_while_pending};
pub use lower_inputs::{
    AliasView, AmplificationDraws, BlendEquation, ChannelWrite, ColorGroup, ExtendedRangeColor,
    FactorLane, FetchedAttachment, FramebufferFetchInputs, LoweringInputs, RotatedBlendForm,
    StorageImageChoice, StorageImageClass, VertexFetch, VertexFetchProjection,
    ViewportCompensation,
};
pub use table::FunctionHandle;
// The vocabulary the pipeline surface reaches: render-pipeline keys, runtime facts and their
// mechanisms, and the link symbols artifacts, layouts, and plans name.
pub use compile::api::{CompileReport, RenderPipelineKeys};
// Translating one entry of a bare module.
pub use compile::{
    EntryDescriptor, EntryPlan, EntryRef, LinkedSource, StageEntries, StandaloneModule,
};
pub use facts::{Mechanism, RuntimeFact, UniformViewChoice};
pub use sym::{
    AccelerationStructureLevel, AirUserKey, Axis, CombinedSampler, HandleClass, ImageAccess,
    ImageDim, ImageHandleClass, PushField, ResourceKind, ResourceSlot, SampledType, SamplerRef,
    SlotClass, SlotRef, SnapshotForm, SpecKey, Stage, StorageFormatClass, Symbol, SyntheticBinding,
    VaryingKey,
};
pub use vertex::{PulledAttribute, VertexAttribute, VertexBinding, VertexInputState, VertexStride};

#[cfg(test)]
mod api_tests;
