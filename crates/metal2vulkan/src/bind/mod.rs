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
//! An indexed indirect draw whose index fetch
//! nothing bounds (`DrawCommands::index_bounds_pass`) can bound it on the GPU instead of reading
//! its arguments: an [`IndexBoundsPass`] of [`index_bounds_plan_module`] and
//! [`index_bounds_fill_module`] before the render pass instance, and two indirect draws (a run of
//! such draws over one arguments buffer and one index buffer, one [`IndexBoundsRun`] plan of
//! [`index_bounds_plan_run_module`] and one fill of [`index_bounds_fill_run_module`] for all of
//! them instead). An
//! indexed draw whose primitives Vulkan's input assembly does not form as Metal does (a strip
//! restarting at another index than all-ones) can draw them as a list instead: an
//! [`IndexRewritePass`] of three [`index_rewrite_module`] dispatches ([`RewritePhase`]) before the
//! render pass instance, and one indirect draw (a run of direct draws of one walk group each,
//! over one index buffer, one [`IndexRewriteRun`] dispatch of [`index_rewrite_run_module`] for
//! all of them instead). A pass's visibility results are its queries' results copied to their
//! offsets, a `Boolean` one the device's own nonzero value where a sample passed. A pass whose
//! fetching draws
//! read a multisampled attachment's sample planes (`StorageAttachmentView::SamplePlanes`) splits
//! the attachment into them with [`planes_split_module`] and merges them back with
//! [`planes_merge_modules`], through the [`PlanesCopy`] view of its texel size. A multisampled
//! depth or stencil aspect moves between images without colour copies through the draws of
//! [`depth_stencil_copy_modules`] and the kernel of [`stencil_split_module`]. Compute commands
//! a kernel encodes into an indirect command buffer replay on the device where it has device
//! addresses: the [`replay_preprocess`] of an execution ([`ReplayExecution`]) over registered
//! [`ReplayVariant`]s, recorded with [`replay_preprocess_module`], writes each command's arguments
//! and slot table for the variants' dispatches ([`BindingPlan::instantiate_replay`]);
//! [`ReplayFaults`] reports the commands it refused. Where the device states
//! `VK_EXT_device_generated_commands` with compute pipeline binding,
//! [`replay_preprocess_generated`] over each variant's [`GeneratedVariant`] writes a token stream
//! for one execution per segment instead ([`GeneratedLayout`]).
//! [`clamped_depth`] is a 32-bit float depth clamped into `[0, 1]`, as a copy into a depth image
//! of a device without `VK_EXT_depth_range_unrestricted` needs it. An ASTC texture
//! on a device without its block is read through a twin of Metal's decode of its blocks
//! ([`AstcDecode`], in the texels [`AstcTexels`] names); a PVRTC texture on a device without its
//! format through a twin of its texels' codes ([`PvrtcDecode`]); an ETC2 or EAC texture on a
//! device without the formats through a twin of its codes or an 11-bit format's values
//! ([`EtcDecode`], in the texels [`EtcTexels`] names); a 4:2:2 texture through a twin of its
//! texels' codes ([`ChromaDecode`]).
//! [`sampler_desc`] gives the Vulkan sampler a plan's sampler write needs, [`heap_classes`] the
//! classes of the handle heap's arrays on a profile, [`FunctionTableWords`] the words of a
//! visible-function table's buffer, and [`FunctionTableSnapshot`] the one buffer of a command's
//! tables' words on a device without the function-table array's features. A plan writes an
//! acceleration structure as the top-level structure the consumer builds for it:
//! [`instance_record`] gives each Metal instance's record, [`primitive_wrap`] the one instance over
//! a bound primitive structure, and [`geometry_flags`] and [`geometry_table_offset`] a primitive
//! structure's geometries. A [`RasterizationRateMap`] is the consumer's `MTLRasterizationRateMap`:
//! it compiles a [`RateMapDescriptor`], answers the map's queries, and gives the parameter data a
//! `rasterization_rate_map_decoder` reads, and plans a pass rendered through one region by region
//! ([`RateMapRegions`]). A [`SparseTileMap`] mirrors a sparse texture's tile mappings where the
//! device has no shader residency: a plan binds its bytes where a function reads the texture's
//! residency. An [`ArgumentBufferLayout`] says where an argument encoder (`MTLArgumentEncoder`)
//! writes each argument of an argument buffer: a function's (`Function::argument_buffer_layout`),
//! or one of [`ArgumentDescriptor`]s ([`ArgumentBufferLayout::from_descriptors`]).
//!
//! The frozen surface is the items named at this root; the submodules are crate-private.

pub(crate) mod acceleration;
pub(crate) mod argument_buffer;
pub(crate) mod astc_decode;
pub(crate) mod chroma_decode;
pub(crate) mod depth_clamp;
pub(crate) mod depth_stencil_copy;
pub(crate) mod dispatch;
pub(crate) mod etc_decode;
pub mod fan_out;
pub(crate) mod graphics;
pub(crate) mod heap;
pub(crate) mod index_bounds;
pub(crate) mod index_rewrite;
pub mod indirect_commands;
pub(crate) mod instantiate;
pub(crate) mod mesh_arguments;
pub(crate) mod neutral;
pub(crate) mod plan;
pub(crate) mod pvrtc_decode;
pub(crate) mod rate_map;
pub(crate) mod replay;
pub(crate) mod resident;
pub(crate) mod sample_interleave;
pub(crate) mod sample_planes;
pub(crate) mod state;
pub(crate) mod status;
pub(crate) mod table;
pub(crate) mod texel_parts;
pub(crate) mod texel_rows;
pub(crate) mod tile_map;
pub(crate) mod ycbcr_convert;
pub(crate) mod ycbcr_in_place;
pub(crate) mod ycbcr_write;

pub use acceleration::{
    geometry_flags, geometry_table_offset, instance_record, primitive_wrap, InstanceRecord,
    MetalInstance, GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION, GEOMETRY_OPAQUE,
};
pub use argument_buffer::{
    ArgumentBufferLayout, ArgumentDescriptor, ArgumentLayoutError, ArgumentMember, ArgumentPlace,
    ArgumentType, ARGUMENT_HANDLE_BYTES,
};
pub use astc_decode::{AstcDecode, AstcTexels};
pub use chroma_decode::ChromaDecode;
pub use depth_clamp::clamped_depth;
pub use depth_stencil_copy::{
    depth_split_module, depth_stencil_copy_modules, stencil_split_module, CopySource,
    DepthSplitModule, DepthStandInForm, DepthStencilCopy, DepthStencilCopyModules, StencilBit,
    StencilSplitModule,
};
pub use dispatch::{Dispatch, DispatchCommands, DispatchRegion, IndirectArgs};
pub use etc_decode::{EtcDecode, EtcTexels};
pub use graphics::{alpha_test_code, DrawCall, DrawCommands, IndexOverrun};
pub use heap::{
    handle_entry, handle_entry_flags, heap_classes, heap_element, heap_raw_form, HeapElement,
    HeapElements, HeapSource, HANDLE_ENTRY_BYTES, HANDLE_TABLE_CAPACITY, NEUTRAL_HEAP_ELEMENT,
};
pub use index_bounds::{
    index_bounds_fill_module, index_bounds_fill_run_module, index_bounds_plan_module,
    index_bounds_plan_run_module, BoundsRunDraw, IndexBoundsFillModule, IndexBoundsPass,
    IndexBoundsPlanModule, IndexBoundsRun,
};
pub use index_rewrite::{
    index_rewrite_module, index_rewrite_run_module, IndexRewriteModule, IndexRewritePass,
    IndexRewriteRun, RewriteOrder, RewriteOutput, RewritePhase, RewriteProgram, RewriteRunDraw,
    RewriteSource,
};
pub use instantiate::{
    BufferExtent, BufferRange, DescriptorBytes, DescriptorRange, IndexBufferRef, IndirectRef,
    PushConstantWrite, PushHole, ResidentRange, ResidentSet, RootTable,
};
pub use mesh_arguments::{
    fold_object_grid, mesh_arguments_module, metal_object_grid, MeshArguments, MeshArgumentsForm,
    MeshArgumentsModule, MeshGridLimits, MeshGridRule, ObjectFold, METAL_MESH_GRID_AXIS,
    METAL_MESH_GRID_TOTAL, METAL_OBJECT_GRID_TOTAL,
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
    RateMapRegion, RateMapRegions, RegionError, RegionLimits, UnpinnedCompile, MAX_SCREEN_EXTENT,
};
pub use replay::{
    draw_merge_module, DrawMergeModule, MergePhase, MergedLayout, DRAW_INDEX_ENTRY_BYTES,
    MERGED_CHUNK, MERGED_HEADER_BYTES, MERGED_REST_FIRST,
};
pub use replay::{
    draw_replay_draw, draw_replay_indexed, draw_replay_merges, draw_replay_primitive_type,
    draw_replay_slots, draw_replay_state, draw_replay_vertex_layouts, DrawReplayRule,
    DRAW_REPLAY_WRITES,
};
pub use replay::{
    draw_replay_module, draw_replay_preprocess, DrawReplay, DrawReplayLayout, DrawReplayModule,
    DrawReplayPreprocess, InheritedBytes, DRAW_ARGUMENT_BYTES, MAX_DRAW_REPLAY_TYPES,
};
pub(crate) use replay::{merged_fragment_table, merged_vertex, MergedVertex};
pub use replay::{
    replay_preprocess, replay_preprocess_generated, replay_preprocess_module, GeneratedLayout,
    GeneratedRefusal, GeneratedReplay, GeneratedVariant, ReplayExecution, ReplayLayout,
    ReplayPreprocess, ReplayPreprocessModule, ReplayRefusal, ReplayVariant, MAX_REPLAY_DISPATCHES,
    MAX_REPLAY_REGIONS, MAX_REPLAY_SLOTS, MAX_REPLAY_VARIANTS,
};
pub use state::{
    ArgumentFunctionTable, ArgumentSampler, ArgumentSamplersShape, BindShape, BindState,
    BoundAccelerationStructure, BoundBuffer, BoundSampler, BoundSamplerShape, BoundTexture,
    BufferRanges, BufferShape, FragmentTests, FunctionTablesShape, HeapPlacement, ImageblockSize,
    ResourceClass, SamplerDescriptor, SamplerShape, StageInRegion, StageShape, StageTable,
    TessellationFactorBuffer, TessellationFactors, TextureFacts, TextureShape,
    TileThreadgroupArgument, TileThreadgroupMemory, UsedClass, UsedResource, UsedShape,
    VertexAmplification, ViewMapping,
};
// How `BindState::shape` classes a buffer: the crate reads it from `state`, and a consumer takes
// shapes from `BindState::shape`; only the crate's tests and validation reach it here.
pub use pvrtc_decode::PvrtcDecode;
pub use sample_interleave::{
    interleave_merge_modules, interleave_split_module, InterleaveMergeModules,
    InterleaveSplitModule,
};
pub use sample_planes::{
    planes_merge_modules, planes_split_module, PlanesCopy, PlanesMergeModules, PlanesSplitModule,
};
pub use status::{ExecutionStatus, ReplayFaults, StatusBits, StatusReport, StatusWord};
pub use table::{FunctionTableSnapshot, FunctionTableWords};
pub use texel_parts::{part_texels, texel_buffer_parts};
pub use texel_rows::TexelRows;
pub use tile_map::{SparseTileMap, TileMapError, TileMapping};
pub use ycbcr_convert::{
    ycbcr_read, ycbcr_texel, YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrPlanes, YCbCrRange,
};
pub use ycbcr_in_place::{YCbCrInPlace, YCbCrPlanesSampler};
pub use ycbcr_write::{ycbcr_store, ycbcr_write, ycbcr_write_module, YCbCrWriteModule};
