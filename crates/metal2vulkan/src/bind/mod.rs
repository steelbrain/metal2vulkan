//! The binder: from what the guest bound to what the consumer records, as plain data.
//!
//! The binder never calls Vulkan. The consumer describes what is bound for one draw or dispatch,
//! keyed by Metal slot, in a [`BindState`]: per [`Stage`](crate::pipeline::Stage) a [`StageTable`]
//! of [`BoundBuffer`]s, [`BoundTexture`]s, [`BoundSampler`]s, and threadgroup memory lengths, the
//! resources made resident with `useResource` or `useHeap` ([`UsedResource`]), the live sampler
//! states an argument buffer can name ([`ArgumentSampler`]), and encoder state
//! such as the compute encoder's stage-in region ([`StageInRegion`]) and the render encoder's
//! tessellation factor buffer ([`TessellationFactors`]). Binding then runs in two levels:
//!
//! 1. **Plan, per shape.** [`BindState::shape`] keeps what can change a plan ([`BindShape`]:
//!    which slots are bound with what kind of resource, formats, sampler modes, alignment
//!    classes, and the like). `ComputePipeline::plan` and `RenderPipeline::plan` turn a pipeline
//!    and a shape into a [`BindingPlan`]: one [`DescriptorWrite`] per layout binding (the resource
//!    bound at a slot, a buffer the binder synthesizes, or a [`NeutralResource`] where nothing
//!    compatible is bound), the vertex buffers, the [`ResourceUse`]s that drive staging,
//!    writeback, and barriers, and where push fields go. A plan is a pure function of the
//!    pipeline, the device, and the shape: memoize it on `(PipelineKey, BindShape)`.
//! 2. **Instantiate, per command.** [`BindingPlan::instantiate_dispatch`] of a [`Dispatch`] gives
//!    [`DispatchCommands`], and [`BindingPlan::instantiate_draw`] of a [`DrawCall`] gives
//!    [`DrawCommands`]: the range to bind each buffer descriptor over ([`DescriptorRange`]:
//!    exactly that range, never more), the bytes to upload for argument and handle tables
//!    ([`DescriptorBytes`]) and for the resident root table ([`RootTable`]), the push-constant
//!    bytes with the device addresses the consumer fills in ([`PushHole`]), and the Vulkan
//!    dispatches or draw to record. A Metal dispatch can become several Vulkan dispatches
//!    ([`DispatchRegion`]), each with its complete specialization constants, so the consumer
//!    creates one `VkPipeline` per distinct specialization.
//!
//! A plan whose [`variant`](BindingPlan::variant) is not empty asks for another variant of the
//! pipeline: compile that variant (see [`pipeline`](crate::pipeline)) and plan it for the same
//! shape. Refusals are [`BindError`](crate::error::BindError)s.
//!
//! Per command buffer, bind one [`ExecutionStatus`] buffer: initialize it with [`ExecutionStatus::initial`], and read it
//! back with [`ExecutionStatus::decode`], which reports resident-set misses and the other faults
//! [`StatusReport`] names. A command reaches it by its device address where the device addresses buffers, and
//! through the binding [`WriteSource::ExecutionStatus`] names where it does not. An indirect
//! dispatch reads its group counts from Metal's buffer itself ([`IndirectArgs::Buffer`]).
//! A pass's visibility results are its queries' results copied to their
//! offsets, a `Boolean` one the device's own nonzero value where a sample passed.
//! [`clamped_depth`] is a 32-bit float depth clamped into `[0, 1]`, as a copy into a depth image
//! of a device without `VK_EXT_depth_range_unrestricted` needs it. An ASTC texture
//! on a device without its block is read through a twin of Metal's decode of its blocks
//! ([`AstcDecode`], in the texels [`AstcTexels`] names); a PVRTC texture on a device without its
//! format through a twin of its texels' codes ([`PvrtcDecode`]); an ETC2 or EAC texture on a
//! device without the formats through a twin of its codes or an 11-bit format's values
//! ([`EtcDecode`], in the texels [`EtcTexels`] names).
//! [`sampler_desc`] gives the Vulkan sampler a plan's sampler write needs, and
//! [`FunctionTableSnapshot`] the one buffer of a command's visible-function tables' words. A
//! plan writes an acceleration structure as the top-level structure the consumer builds for it.
//! A [`RasterizationRateMap`] is the consumer's `MTLRasterizationRateMap`: it compiles a
//! [`RateMapDescriptor`], answers the map's queries, and gives the parameter data a
//! `rasterization_rate_map_decoder` reads. An [`ArgumentBufferLayout`] says where an argument
//! encoder (`MTLArgumentEncoder`)
//! writes each argument of an argument buffer: a function's (`Function::argument_buffer_layout`),
//! or one of [`ArgumentDescriptor`]s ([`ArgumentBufferLayout::from_descriptors`]).
//!
//! The frozen surface is the items named at this root; the submodules are crate-private.

pub(crate) mod argument_buffer;
pub(crate) mod astc_decode;
pub(crate) mod depth_clamp;
pub(crate) mod dispatch;
pub(crate) mod etc_decode;
pub(crate) mod graphics;
pub(crate) mod heap;
pub mod indirect_commands;
pub(crate) mod instantiate;
pub(crate) mod mesh_arguments;
pub(crate) mod neutral;
pub(crate) mod plan;
pub(crate) mod pvrtc_decode;
pub(crate) mod rate_map;
pub(crate) mod replay;
pub(crate) mod resident;
pub(crate) mod state;
pub(crate) mod status;
pub(crate) mod table;
pub(crate) mod texel_parts;
pub(crate) mod texel_rows;
pub(crate) mod ycbcr_convert;
pub(crate) mod ycbcr_in_place;

pub use argument_buffer::{
    ArgumentBufferLayout, ArgumentDescriptor, ArgumentLayoutError, ArgumentMember, ArgumentType,
    ARGUMENT_HANDLE_BYTES,
};
pub use astc_decode::{AstcDecode, AstcTexels};
pub use depth_clamp::clamped_depth;
pub use dispatch::{Dispatch, DispatchCommands, DispatchRegion, IndirectArgs};
pub use etc_decode::{EtcDecode, EtcTexels};
pub use graphics::{alpha_test_code, DrawCall, DrawCommands, IndexOverrun};
#[cfg(test)]
pub(crate) use heap::heap_classes;
pub use heap::{handle_entry, HeapSource, HANDLE_ENTRY_BYTES};
pub use instantiate::{
    BufferExtent, BufferRange, DescriptorBytes, DescriptorRange, IndexBufferRef, IndirectRef,
    PushConstantWrite, PushHole, ResidentRange, RootTable,
};
pub use mesh_arguments::{
    fold_object_grid, metal_object_grid, MeshArguments, MeshArgumentsForm, MeshGridLimits,
    ObjectFold, METAL_MESH_GRID_AXIS, METAL_MESH_GRID_TOTAL, METAL_OBJECT_GRID_TOTAL,
};
pub use neutral::NeutralResource;
pub use plan::resources::sampler_desc;
pub use plan::{
    AccessClass, BindingPlan, ByteRange, ComputeTemplate, DescriptorWrite, FieldLocation,
    GraphicsTemplate, ImageSource, ImageViewDesc, PlanDiagnostic, PushFieldPlacement, PushTemplate,
    ResourceUse, SamplerSource, UseTarget, VertexBufferBind, ViewSetup, WriteSource,
};
pub use rate_map::{
    RasterizationRateMap, RateMapAxis, RateMapDescriptor, RateMapError, RateMapLayerDescriptor,
    UnpinnedCompile, MAX_SCREEN_EXTENT,
};
pub use replay::DRAW_ARGUMENT_BYTES;
pub(crate) use replay::{merged_fragment_table, merged_vertex, MergedVertex};
pub use replay::{DRAW_INDEX_ENTRY_BYTES, MERGED_REST_FIRST};
pub use state::{
    ArgumentFunctionTable, ArgumentSampler, ArgumentSamplersShape, BindShape, BindState,
    BoundAccelerationStructure, BoundBuffer, BoundSampler, BoundSamplerShape, BoundTexture,
    BufferRanges, BufferShape, FunctionTablesShape, HeapPlacement, ImageblockSize, ResourceClass,
    SamplerDescriptor, SamplerShape, StageInRegion, StageShape, StageTable,
    TessellationFactorBuffer, TessellationFactors, TextureFacts, TextureShape,
    TileThreadgroupArgument, TileThreadgroupMemory, UsedClass, UsedResource, UsedShape,
    VertexAmplification, ViewMapping,
};
// How `BindState::shape` classes a buffer: the crate reads it from `state`, and a consumer takes
// shapes from `BindState::shape`; only the crate's tests and validation reach it here.
pub use pvrtc_decode::PvrtcDecode;
pub use status::{ExecutionStatus, ReplayFaults, StatusBits, StatusReport, StatusWord};
pub use table::FunctionTableSnapshot;
#[cfg(test)]
pub(crate) use table::FunctionTableWords;
pub use texel_parts::part_texels;
pub use texel_rows::TexelRows;
#[cfg(test)]
pub(crate) use ycbcr_convert::YCbCrPlanes;
pub use ycbcr_convert::{YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrRange};
pub use ycbcr_in_place::{YCbCrInPlace, YCbCrPlanesSampler};
