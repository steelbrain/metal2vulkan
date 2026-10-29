//! Render planning and draws: [`RenderPipeline::plan`] and [`BindingPlan::instantiate_draw`].
//!
//! A render plan is the compute plan's algorithm over the pipeline's two Metal stages:
//!
//! - **Descriptors.** Every set-0 binding of the layout gets one write: a buffer slot of the
//!   vertex or fragment stage its `Buffer`, `InlineBytes`, or `ZeroBuffer` neutral (a pulled
//!   vertex buffer is a read-only, `Whole`-window storage slot of the vertex stage); each stage's
//!   `ArgumentTable` and `ConstantBlob`; textures, samplers, combined image samplers, texel
//!   buffers, and input attachments through the shared resource planner (`plan::resources`); each
//!   stage's handle table and the set-1 heap arrays through the heap planner (`bind::heap`): the
//!   heap's members are shared by both stages, and each stage's table enters the members that reach
//!   it (a used texture's render stages, a bound texture's own stage). A fragment function's
//!   imageblock storage binding is `WriteSource::Imageblock`; the pass's imageblock is one plane of
//!   cells, so a pass of more than one layer or an amplified draw is not implemented. An attachment
//!   the fragment function reads in the interlocked storage form
//!   (`GraphicsState::storage_attachments`) is `WriteSource::TileAttachment` in the attachment's
//!   own format, likewise one 2D view, so such a pass or draw is not implemented either. A fragment
//!   function's threadgroup arguments (`RuntimeFact::TileThreadgroupMemory`) are the pass's tile
//!   memory, `WriteSource::TileMemory` as a tile function's (`plan::compute`): the shape's
//!   `tile_threadgroup` places each argument, and its `render_area` and `imageblock_size` (the
//!   pass's tile) the tiles; a layered pass or an amplified draw is not implemented.
//! - **Vertex buffers.** One [`VertexBufferBind`] per fixed-function binding of the pipeline's
//!   `VertexInputState`; the slot must be bound (`UnboundVertexBuffer`), and a `PerBind` stride
//!   is the bind's `attributeStride`: 0 for a buffer or inline bytes bound without one, as plain
//!   Metal's encoder reads them (its debug layer stops the draw); inline bytes
//!   bound at `MTLAttributeStrideStatic`, which an M4 read as element 0 and zeros past it, are
//!   `MissingVertexStride`. A binding the shape binds where fixed-function input cannot fetch it (a
//!   bind offset whose alignment class misses an attribute's Vulkan alignment; a per-bind stride
//!   past `maxVertexInputBindingStride`, not a multiple of the alignment, breaking the portability
//!   subset's stride rule, or as dynamic state below an element's attributes) names its layout in
//!   the plan's pulled-vertex-buffer variant, which pulls the layout's attributes; a pipeline
//!   compiled under such a variant keeps naming its layouts, so the key is a fixed point. Per draw,
//!   a binding's extent is the bytes its vertex or instance range fetches, and with dynamic strides
//!   every binding's stride is listed.
//! - **Push fields** per stage window: the root-table and status addresses are holes the consumer
//!   fills (one root table per stage that reads one: its `Psb` entries are that stage's slots),
//!   `BaseVertexInstance` is the draw's Vulkan base vertex and first instance,
//!   `SlotBindOffset`/`SlotGuestBase` as in compute, and `VertexBufferStride` a pulled layout's
//!   static stride, or for a per-bind one the bind's `attributeStride` (0 without one, inline
//!   bytes' as a buffer's, as for fixed function).
//! - **Specialization.** One key over both stages' constants (link numbers `SpecId`s over the
//!   pipeline, so a key's entries are disjoint per id): `Presence`, `SamplerState` (the bound
//!   sampler's word masked by the fields the function reads), `BicubicTexels` (the bound format's
//!   treatment under Metal's bicubic filter), `TexelFormat` (the bound format's conversion to
//!   `half`), `SampleCount` (the pipeline's rasterization samples), `AmplificationCount` (the
//!   shape's vertex amplification count), and a fragment function's tile threadgroup memory's
//!   (`TgOffset`, `TileThreadgroupLength`, `TileGridWidth`, and the pass's tile as
//!   `ImageblockWidth` and `ImageblockHeight`).
//! - **Views**. A count of 0, above 1 for a pipeline whose `maxVertexAmplificationCount` is 1, or
//!   with fewer view mappings than the count is `InvalidVertexAmplification`. A count above a
//!   larger maximum draws that many amplifications, as plain Metal draws them where its validation
//!   stops the count (a Metal probe: 3 and 4 on a maximum of 2, 4 on 3, 8 on 2 and on 4, an M4).
//!   Amplification `a` renders into the layer of the pass its render target offset `rto[a]` (plus
//!   the slice the vertex function chooses, if it does) names where that is below the pass's
//!   `renderTargetArrayLength`, and into the first layer where it is not, every one in a pass of at
//!   most one layer (`nil` mappings are offsets of 0; a Metal probe on an M4, the rule for an
//!   index), and a pipeline of `maxVertexAmplificationCount` 1 takes no mapping (its draws render
//!   the first layer through viewport 0; Metal probes); through the viewport its viewport offset
//!   `vpo[a]` plus the function's index names; and Metal renders every primitive of every instance
//!   of one amplification before the next's (a Metal probe on an M3 Max and an M4, with and without
//!   API validation). The plan's `views` are Vulkan multiview, amplification `a` view `a` in
//!   attachment layer `a`, where that is exact: the mappings add no viewport offset, those layers
//!   are consecutive (`l[a] = l[0] + a`, `l[0]` the views' base layer), the count is within the
//!   device's `maxMultiviewViewCount`, the shape does not ask for one draw per amplification
//!   (`VertexAmplification::one_at_a_time`, under a visibility query), and no stage declares
//!   `Layer` and the vertex function chooses its slice in no variant (a multiview pipeline cannot
//!   declare it) unless the count is 1 into the first layer; a count of 1 draws without multiview
//!   into layer `l[0]`. Every other amplification draws one amplification at a time: the plan's
//!   variant is `amplification_draws` (`layered` in a pass of more than one layer, `viewports`
//!   where an offset is not 0), whose stages read the amplification's id and offsets from push
//!   fields (`PushField::AmplificationId`, `AmplificationLayerOffset`,
//!   `AmplificationViewportOffset`), and `DrawCommands::amplifications` writes them before each
//!   draw. A tessellation pipeline drawn so is not implemented.
//! - **Render target slice**. A vertex function that writes `[[render_target_array_index]]`
//!   (`RuntimeFact::RenderTargetArrayLength`) selects a slice only in a pass of at least two
//!   layers. In a pass of at most one (`BindShape::render_target_array_length` 0, Metal's default,
//!   or 1) every index renders slice 0, and the plan's variant drops the vertex `Layer` output
//!   (`VariantKey::dropped_vertex_layer`): exact, and a one-layer framebuffer is never written an
//!   out-of-range `Layer`, which Vulkan leaves undefined. In a pass of more the pipeline writes
//!   `Layer` as 0 for an index at or past its `PushField::RenderTargetArrayLength`, the pass's
//!   length (`BindState::render_target_array_length`), which each draw writes: Metal renders such
//!   an index into the pass's first layer. A draw of a plan whose pipeline is the other variant is
//!   `BindError::NeedsVariant`.
//! - **Viewport count**. A vertex function that writes `[[viewport_array_index]]`
//!   (`RuntimeFact::ViewportCount`) stores `ViewportIndex` as 0 for an index at or past its
//!   `PushField::ViewportCount`, the encoder's viewport count (`BindState::viewport_count`), which
//!   each draw writes: Metal draws such an index through viewport 0.
//! - **Viewport compensation**. A draw through a viewport Vulkan's cannot be (a negative width, or
//!   a depth range past `[0, 1]` the device does not take; `BindShape::viewport_compensation`) runs
//!   the variant whose vertex function compensates its position
//!   (`VariantKey::viewport_compensation`), from the draw's `PushField::ViewportCompensation`
//!   (`BindState::viewport_compensation`); the plan of the pipeline compiled under another
//!   compensation is `BindError::NeedsVariant`. A mesh pipeline has no vertex function to
//!   compensate: not implemented.
//! - **Zeroed colors**. A draw into a pass with color attachments its pipeline has no pixel format
//!   for (`BindShape::zeroed_colors`), where plain Metal stores zero bytes at every sample the draw
//!   covers, runs the variant whose pipeline has them, written whole and unblended, and whose
//!   fragment function writes zero to them (`VariantKey::zeroed_colors`); and its colors the pass
//!   has no attachment for (`BindShape::dropped_colors`), whose writes plain Metal drops: the
//!   variant whose fragment function keeps no output there (`VariantKey::dropped_colors`). A mesh
//!   pipeline's is not implemented.
//! - **Interlocked colors**. A draw into a render pass instance that leaves color attachments out
//!   of its subpass for other draws' interlocked storage form (`BindShape::interlocked_colors`)
//!   runs the variant whose fragment function writes those it writes through that form too
//!   (`VariantKey::interlocked_colors`), so the draw records in the instance. That variant stores
//!   them before the depth and stencil tests (tested storage writes, below), and so does the
//!   variant of a draw that holds the `half` shadow of an extended-range color the function writes
//!   (`BindShape::extended_shadows`): the plan of the pipeline as compiled selects such a moved
//!   writer's tests from `RuntimeFact::MovableStorageWrites`. A mesh pipeline's is not
//!   implemented, nor a multisampled, layered or amplified draw's.
//! - **Snapshot reads**. A draw the consumer reads a texture of its own subpass for through a
//!   snapshot (`BindShape::snapshot_textures`) runs the variant whose fragment function reads it so
//!   (`VariantKey::snapshot_textures`, `lower::fragment`'s `snapshot`), for each slot of the
//!   fragment stage's `StageInterface::snapshot_textures`, where the pipeline's two draws are
//!   Metal's one: a device with `fragmentShaderPixelInterlock`, `fragmentStoresAndAtomics`,
//!   `attachmentFeedbackLoopLayout` and `extendedDynamicState3ColorWriteMask` (the variant's
//!   pipeline requires the last two: its feedback loop flag and dynamic write masks), one
//!   sample, no depth or stencil attachment, no rasterizer discard, no memory the vertex or
//!   fragment function writes, no attachment in the storage form or input attachment, no
//!   imageblock, tile memory, shader-side stencil or tests, no interlocked or shadowed colors, no
//!   tessellation or mesh stages, and no amplified or layered draw. Every other
//!   slot the shape names is left out, and the plan's `variant.snapshot_textures` says which it
//!   kept. Its plan binds each slot's snapshot (`WriteSource::SnapshotTexels`) and leaves which of
//!   the two draws runs to the consumer (`PushHole::SnapshotPhase`, zero in the plan's bytes);
//!   `GraphicsState::snapshot_reads` states the consumer's part.
//! - **Depth clamp range**. A draw in depth clamp mode through a range above 0 whose Vulkan depth
//!   clamp holds 0 (`BindShape::depth_clamp_range`) runs the variant whose fragment function, where
//!   it writes its depth, clamps that depth to the draw's range itself
//!   (`VariantKey::depth_clamp_range`), from the draw's `PushField::DepthClampRange`
//!   (`BindState::depth_clamp_range`), as Metal stores a negative NaN as 0 past the range.
//! - **Rotated weights**. A draw of triangles drawn with Metal's last vertex first
//!   (`BindShape::rotated_weights`) of a pipeline whose fragment function reads
//!   `[[barycentric_coord]]` runs the variant that takes Metal's weights back from the drawn
//!   triangle's (`VariantKey::rotated_weights`, `LoweringInputs::rotated_weights`). A mesh draw
//!   drawn from each primitive's last index first (`BindShape::mesh_last_first`) of an emulated
//!   mesh function's pipeline of lines or triangles runs the variant whose synthesized vertex
//!   stage reads its corners so (`VariantKey::mesh_last_first`).
//! - **Tested storage writes**. A fragment function that stores a color through the interlocked
//!   storage form and declares no early tests only because it discards
//!   (`RuntimeFact::TestedStorageWrites`), in a pipeline with a depth or stencil attachment, stores
//!   before Vulkan's tests where Metal stores after its own: a draw whose tests drop no fragment
//!   (`BindShape::fragment_tests`, `FragmentTests::Pass`) runs it as compiled; one whose tests may
//!   drop fragments but write neither attachment and count nothing (`Drop`) runs the variant whose
//!   function declares early tests (`VariantKey::read_only_tests`); any other is
//!   `NotImplemented(FramebufferFetch)`. A function that also writes other memory, or returns
//!   depth, stencil, or a sample mask, states the fact without that variant (`early_variant`): a
//!   draw whose tests drop no fragment runs it as compiled, and any other is
//!   `NotImplemented(FramebufferFetch)` at the depth attachment (the stencil one where the pipeline
//!   has no depth attachment). A draw that moves a color the function writes as an ordinary output
//!   into the storage form (interlocked colors, above) takes its tests alike from
//!   `RuntimeFact::MovableStorageWrites`, which states the `early_variant` the moved writer's
//!   variant states, where the pipeline as compiled states no `TestedStorageWrites`; a moved write
//!   whose mask names none of the attachment's components stores nothing, and takes no tests. (A
//!   draw whose depth test alone acts, `Depth`, runs the variant whose function runs it itself,
//!   `VariantKey::shader_tests`, unless it writes nothing and the early-tests variant serves it;
//!   one whose stencil test acts and whose tests write, `Stencil`, or a `Drop` draw of a function
//!   without the early-tests variant, runs the variant whose function runs both,
//!   `VariantKey::shader_stencil`, in a pass with a depth and a stencil attachment; a pass with a
//!   stencil attachment alone is `NotImplemented(FramebufferFetch)` at it.)
//! - **Draws.** A Metal draw maps to the Vulkan draw of the same arguments; an indirect draw is
//!   GPU-side unless a stage reads `BaseVertexInstance` from push constants or the plan is
//!   multiview, in which case `indirect_needs_arguments` asks for the CPU read.
//!   `VerticesPerPrimitive` (a vertex function that writes the primitive ID,
//!   `LoweringInputs::primitive_id_varying`) is the draw's primitive type's vertices per
//!   primitive (`BindState::primitive_type`; `MissingPrimitiveType` without one), a direct or
//!   indirect draw's alike. An indexed draw of such a pipeline, whose vertices primitives share,
//!   runs its de-indexed variant of the draw's index type (`BindShape::primitive_id_indices`,
//!   `VariantKey::primitive_id_indices`): `index count` vertices from 0 with its instances,
//!   layout `[0, index count, base instance, base vertex]` (`PushField::CaptureLayout`), leaving
//!   `PushHole::CaptureIndexAddress` to the consumer (its first index's address), and reading its
//!   indices in the vertex stage as storage (`DrawCommands::uses`); any other draw of that variant is
//!   `NeedsVariant` without it. A strip, an indirect indexed draw, indices past their buffer and
//!   an indexed draw of a plan without the form are `NotImplemented(PrimitiveId)`. A draw of one
//!   of Metal's private primitive types in the private list form (`BindShape::private_list`,
//!   `VariantKey::private_list`) is the consumer's list, which it draws without indices from the
//!   list's first index as first vertex whatever the call it gave: its vertices per primitive are
//!   its list's, twice that where the list holds Metal's quads as two triangles a quad
//!   (`BindState::quad_primitive_ids`), and it leaves `PushHole::CaptureIndexAddress` to the
//!   consumer (its list's address, whose word before its first index holds Metal's base
//!   vertex). A direct
//!   draw of a multiview plan whose last instance index exceeds `maxMultiviewInstanceIndex` is
//!   `Unsupported{VertexAmplification; Limit}`. `DrawCommands.uses` lists the draw's index buffer
//!   and indirect arguments.
//! - **Index buffer bounds**. A direct indexed draw whose indices run past where Metal's fetch
//!   stops reading its index buffer (`IndexBufferRef::fetch_end`: the length rounded up to 4 bytes;
//!   Metal's API validation rejects the draw, and Apple's GPUs read index 0 past there) is drawn
//!   unchanged where the device enables `robustBufferAccess2`, which bounds the index fetch, and
//!   otherwise clamped to the indices wholly before there, since Vulkan then requires every index
//!   inside the bound range (`VUID-vkCmdDrawIndexed-robustBufferAccess2-08798`);
//!   `DrawCommands::index_overrun` records which. Vulkan names no index for a fetch past the bound
//!   range even under `robustBufferAccess2`, so the runtime draws such a draw from a copy of its
//!   indices padded with zeros, which leaves no overrun here. The check is on the CPU: no shader
//!   reads anything for it. An indexed indirect draw's counts are in GPU memory, so where the
//!   device lacks `robustBufferAccess2` (or the index offset is at or past the buffer's length) it
//!   asks for its arguments on the CPU (`indirect_needs_arguments`), and the direct draw is bounded
//!   as above. Where that is the only reason, `index_bounds_pass` says the GPU index bounds pass
//!   may bound it instead (`bind::IndexBoundsPass`).
//! - **Tessellation**. A tessellation pipeline's Metal stages are its evaluation stage (the vertex
//!   function, whose push range, bindings, and uses name `TESSELLATION_EVALUATION`) and its
//!   fragment stage; its synthesized vertex and control stages bind nothing of Metal's. The control
//!   stage's factor buffer is the encoder's (`BindState::tessellation_factors`,
//!   `WriteSource::TessellationFactors`, read in `TESSELLATION_CONTROL_SHADER`), or the zero
//!   buffer, whose patches are all culled, when the encoder set none. Its draws draw patches
//!   (`DrawCall::Patches`, `DrawCall::PatchesIndirect`): one vertex per patch, from the patch
//!   start, with the draw's instances; each pushes the control stage's range (the factor buffer's
//!   residual bind offset, instance stride rounded down to a multiple of 4, and scale rounded to
//!   `half`) and binds the factors it reads (`patch_draw`). A draw's patch index buffer binds the
//!   control stage's `WriteSource::PatchIndices` over the indices it reads (read in
//!   `TESSELLATION_CONTROL_SHADER`), and its push range says the draw has one; a draw without one
//!   leaves the zero buffer there. A pipeline with a control-point index type draws indexed patches
//!   only, each binding its control-point index buffer (`WriteSource::ControlPointIndices`, read in
//!   `TESSELLATION_CONTROL_SHADER`) over the indices it reads, whose residual bind offset its push
//!   range holds; one without draws patches without such a buffer only. A draw of primitives with
//!   it, a draw of patches with any other pipeline, another control-point count than the
//!   function's, a patch draw whose indexing differs from the pipeline's, misaligned or overlong
//!   factors, patch indices or control-point indices, and a scale that is not a positive normal
//!   `half` (which Metal forbids, and which keeps every float the control stage computes finite)
//!   are `BindError`s. On a device without `drawIndirectFirstInstance` an indirect draw asks for
//!   its arguments on the CPU (`DrawCommands::indirect_needs_arguments`).
//!
//!   On a device without `tessellationShader` the pipeline is emulated
//!   (`GraphicsState::emulated_tessellation`): its Metal stages are its vertex function, as a
//!   `VERTEX` stage, and its fragment stage, and its plan binds the draw's work buffer and records
//!   (`WriteSource::TessellationWork`, `TessellationRecords`) and the index buffers in the vertex
//!   stage, which reads them (`VERTEX_SHADER`) where its function reads what they index. A patch
//!   draw, checked as above, is a tessellation scan and chunk draws
//!   (`DrawCommands::tessellation_dispatch`, [`TessellationDispatch`]): the scan reads the factors
//!   and an indirect draw's arguments, which the consumer copies into its work buffer
//!   (`COMPUTE_SHADER` uses, the scan's), so no draw asks for its arguments on the CPU, and `draw`
//!   is chunk 0's `vkCmdDrawIndirect`. A direct draw whose records pass
//!   `maxStorageBufferRange` is refused (`Unsupported`, `Feature::Tessellation`).
//! - **Mesh draws**. A mesh render pipeline's Metal stages are its object function, when it has one
//!   (`TASK_EXT`), its mesh function (`MESH_EXT`) and its fragment function. It draws only mesh
//!   draws (`DrawCall::MeshThreadgroups`, `MeshThreads`, `MeshThreadgroupsIndirect`), and only it
//!   does (`BindError::InvalidMeshDraw`). Each stage's threadgroup size is a specialization each
//!   draw decides, as a kernel's region's is, the object function's under keys of its own
//!   (`SpecKey::ObjectLocalSize`): the plan's `specialization` leaves them out, and each draw's
//!   `DrawCommands::specialization` is the complete key. A direct draw is `vkCmdDrawMeshTasksEXT`
//!   of the first stage's threadgroups (without an object function, `drawMeshThreads`' threads
//!   rounded down to whole mesh threadgroups, Metal's answer; with one, its object threadgroups, a
//!   partial one not implemented) and each stage reads its grid from `NumWorkgroups` (`MultiRegion`
//!   0, each `DispatchRegion` written all the same); a threadgroup of no thread or of more than
//!   Metal's 1024, a grid of none, and indirect arguments at an offset that is not a multiple of 4
//!   launch nothing, as plain Metal draws nothing. A threadgroup or a grid past the device's limits
//!   for its stage is `Unsupported{MeshShading}` naming the limit (an object grid only where the
//!   object function is not folded, below). An object function's threadgroup
//!   memory arguments are the encoder's lengths (`setObjectThreadgroupMemoryLength:atIndex:`, the
//!   object stage's `threadgroup_lengths`, 0 where unset), each rounded up to 16 as a kernel's
//!   are; its arena with them past the object stage's workgroup memory limit is
//!   `Unsupported{ThreadgroupMemorySize}` naming it (`RuntimeFact::ObjectThreadgroupArena`). A
//!   mesh function's arguments, which no encoder sets, are not implemented. A mesh grid Metal
//!   draws nothing for (an axis past 65,535, or more than 1,048,575 threadgroups:
//!   `bind::mesh_arguments`) launches nothing. An indirect draw reads the counts an arguments
//!   pass writes (`DrawCommands::mesh_arguments`, `MeshArguments`): Metal's where Metal draws
//!   them and the device's first stage launches them, else none. A folded object function
//!   (`lower::mesh::folds_object_grids`) runs any object grid in pieces within the task stage's
//!   limits instead, reading its grid from its region record: a direct draw's fold is the
//!   consumer's upload (`DrawCommands::object_fold`), an indirect one's its arguments pass's. A
//!   direct object grid is Metal's (`metal_object_grid`: its threads per grid wrap at 2^32, and it
//!   runs the threadgroups they fill); one of more threadgroups than Metal runs
//!   (`METAL_OBJECT_GRID_TOTAL`) launches nothing and fails its command buffer as Metal's GPU
//!   discards it (`DrawCommands::discards_command_buffer`), on every device.
//!   A mesh function writes `Layer` in every variant: a pass of more layers than
//!   `maxMeshOutputLayers`, into which a mesh function choosing its slice could render, is
//!   `Unsupported{MeshShading}` naming it. A view mapping with a viewport offset, or with a slice
//!   offset where the mesh function chooses its slice, is not implemented: the mesh function does
//!   not add it. A mesh pipeline draws with an amplification count of 1 (its
//!   `maxVertexAmplificationCount`).
//!
//!   An emulated mesh pipeline (`GraphicsState::mesh_records`) is different in these ways:
//!   - Its object and mesh functions are `COMPUTE` stages, the object function's first, and
//!     their threadgroups are held to the compute limits, as is the object function's grid.
//!   - Each draw's compute work is `DrawCommands::mesh_dispatch` (`MeshDispatch` states the
//!     consumer contract), over a work buffer of the draw's own. A direct draw of `G` mesh
//!     threadgroups without an object function is the host's: its dispatch is `G` folded into
//!     rows of `maxComputeWorkGroupCount[0]`, its records hold `G · group_words` words (at
//!     least 16 bytes; past `maxStorageBufferRange` refused, naming it), and its draw is a
//!     `Draw` of `G · NP · k` vertices. A direct draw with an object function, and an indirect
//!     draw without one, are scanned on the GPU (`MeshScanModule`): the mesh dispatch and the
//!     draw are indirect from the work buffer, and the records lie in the command buffer's
//!     records pool, where a draw that does not fit draws nothing and fails the command buffer
//!     (`StatusReport::mesh_records_overflow`). Neither asks for its arguments on the CPU.
//!   - An indirect draw of a pipeline with an object function, whose object threadgroups only the
//!     GPU knows, runs an object arguments pass first (`mesh_object_arguments_module`,
//!     `MeshDispatch::object_arguments`): it reads Metal's arguments, reserves the object
//!     threadgroups' slots and offsets in the records pool (`lower::mesh::work::SLOTS`), and
//!     writes the object dispatch, which reads its counts from the work buffer; then the scan as
//!     above. Every threadgroup of a grid whose threads per grid wrap runs, reporting the wrapped
//!     grid as Metal's (`lower::dispatch`'s object grid). A grid Metal's GPU discards, one past the
//!     compute limits, and one whose slots the pool cannot hold draw nothing and say so in the
//!     command buffer's `ExecutionStatus`.

use std::collections::BTreeMap;

use crate::bind::heap::{planner, HeapPlanner};
use crate::bind::instantiate::{
    bind_buffers, descriptor_range, put, slot_field, BufferExtent, BufferRange, DescriptorBytes,
    DescriptorRange, FieldBytes, IndexBufferRef, IndirectRef, PushConstantWrite, PushHole,
    RangeError, RootTable,
};
use crate::bind::mesh_arguments::{
    fold_object_grid, metal_mesh_grid_draws_nothing, metal_object_grid, MeshArguments,
    MeshArgumentsForm, MeshGridLimits, ObjectFold, METAL_OBJECT_GRID_TOTAL,
};
use crate::bind::neutral::{zero_buffer_bytes, NeutralResource};
use crate::bind::plan::compute::{internal, not_implemented, spec_key, tile_memory};
use crate::bind::plan::resources::{separate_samplers, ResourcePlanner};
use crate::bind::plan::slots::{
    address_map_variant, anisotropic_variant, bindings, bound_value, buffer_use, check_slots,
    check_sparse_textures, cross_variant, cubic_variant, large_view_candidates, mark_disjoint,
    plan_buffer, psb_variant, range_variant, record_specialized, snapshot_forms, srgb_variant,
    storage_view_variant, used_uses, ycbcr_in_place_variant, ycbcr_planes_sampled_variant,
};
use crate::bind::plan::{
    AccessClass, BindingPlan, BoundKind, ByteRange, DescriptorWrite, FactorSite, FactorTemplate,
    FieldLocation, GraphicsTemplate, GroupLimits, GroupStage, MeshTemplate, PlannedVertexBuffer,
    PushFieldPlacement, PushTemplate, RecordsTemplate, ResourceUse, ShaderTestsTemplate,
    ShadowTemplate, StageTemplate, StrideRules, UseTarget, VertexBufferBind, ViewSetup,
    WriteSource,
};
use crate::bind::resident::{resident_entries, stage_root_table};
use crate::bind::state::{
    offset_alignment, BindShape, BindState, BoundBuffer, BufferShape, FragmentTests,
    VertexAmplification,
};
use crate::cache::{SpecializationKey, VariantKey};
use crate::device::profile::portable_vertex_stride;
use crate::device::requirements::{VkFeature, VkLimit};
use crate::device::{vk_stage, Device};
use crate::error::{Attachment, BindError, Capability, Feature, LowerFamily, Unsupported};
use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::lower::dispatch::region_bytes;
use crate::metal;
use crate::pipeline::artifact::RenderPipeline;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{
    AccessSet, BufferForm, BufferSlot, PushLoc, SlotInterface, Window,
};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{
    Axis, DispatchKeys, PushField, SlotClass, SlotRef, SpecKey, Stage, SyntheticBinding,
};
use crate::pipeline::vertex::{VertexAttribute, VertexStride};
use crate::pipeline::{AmplificationDraws, ViewportCompensation};
use crate::vk;

/// A Metal draw call.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DrawCall {
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`.
    Primitives {
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
        base_instance: u32,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:…`. Indices
    /// past `index_buffer`'s length are bounded as [`DrawCommands::index_overrun`] states.
    Indexed {
        index_type: vk::IndexType,
        index_buffer: IndexBufferRef,
        index_count: u32,
        index_offset: u64,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// `drawPrimitives:indirectBuffer:…`, or the indexed form when `indexed` is set (its index
    /// type, buffer, and index offset). Where nothing bounds the indexed form's index fetch it
    /// asks for its arguments on the CPU ([`DrawCommands::indirect_needs_arguments`]).
    Indirect {
        indirect: IndirectRef,
        indexed: Option<(vk::IndexType, IndexBufferRef, u64)>,
    },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:` of a tessellation pipeline. `control_points` is `numberOfPatchControlPoints`;
    /// `patch_index` the patch index buffer and its offset, a multiple of 4 (`None` for `nil`): the
    /// `uint` at the offset plus `4 · (patchStart + i)` is patch `i`'s index, whose control points
    /// and per-patch attributes it reads, while its factors stay at `patchStart + i`.
    /// `control_point_index` is `drawIndexedPatches:...`'s control-point index buffer and its
    /// offset, a multiple of the pipeline's index size, for a pipeline with a control-point index
    /// type (`None` for `drawPatches`): control point `k` of the patch whose index is `p` is the
    /// index at position `p · numberOfPatchControlPoints + k` from the offset.
    Patches {
        control_points: u32,
        patch_start: u32,
        patch_count: u32,
        patch_index: Option<(IndexBufferRef, u64)>,
        instance_count: u32,
        base_instance: u32,
        control_point_index: Option<(IndexBufferRef, u64)>,
    },
    /// `drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:
    /// indirectBufferOffset:`, whose `MTLDrawPatchIndirectArguments` (`patchCount`,
    /// `instanceCount`, `patchStart`, `baseInstance`) are a `VkDrawIndirectCommand`, or with
    /// `control_point_index`, `drawIndexedPatches:...indirectBuffer:indirectBufferOffset:`.
    PatchesIndirect {
        control_points: u32,
        patch_index: Option<(IndexBufferRef, u64)>,
        indirect: IndirectRef,
        control_point_index: Option<(IndexBufferRef, u64)>,
    },
    /// `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:` of a mesh
    /// render pipeline: `groups` object threadgroups of `threads_per_object_group` threads, each
    /// launching the mesh grid it sets, or, without an object function, `groups` mesh
    /// threadgroups of `threads_per_mesh_group` threads.
    MeshThreadgroups {
        groups: [u32; 3],
        threads_per_object_group: [u32; 3],
        threads_per_mesh_group: [u32; 3],
    },
    /// `drawMeshThreads:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`: without an object
    /// function, `threads / threads_per_mesh_group` mesh threadgroups along each axis, rounded
    /// down (Metal's measured answer).
    MeshThreads {
        threads: [u32; 3],
        threads_per_object_group: [u32; 3],
        threads_per_mesh_group: [u32; 3],
    },
    /// `drawMeshThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerObjectThreadgroup:
    /// threadsPerMeshThreadgroup:`, whose `MTLDispatchThreadgroupsIndirectArguments` are the
    /// direct form's `groups` (GPU-written).
    MeshThreadgroupsIndirect {
        indirect: IndirectRef,
        threads_per_object_group: [u32; 3],
        threads_per_mesh_group: [u32; 3],
    },
}

/// What the consumer records for one draw: upload `descriptor_bytes` and each stage's root table,
/// bind every plan descriptor (buffer and texel-buffer descriptors over exactly
/// `descriptor_ranges`), bind the plan's vertex buffers at `buffer_offsets` (with
/// `vertex_strides` as dynamic strides), fill the `push_holes`, push `push_constants`, and record
/// `draw`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DrawCommands {
    /// One write per stage with push fields, covering the stage's whole push range.
    pub push_constants: Vec<PushConstantWrite>,
    /// Each stage's `ArgumentTable` contents and handle table.
    pub descriptor_bytes: Vec<DescriptorBytes>,
    /// Vertex-buffer offsets by Vulkan vertex binding (`vkCmdBindVertexBuffers`).
    pub buffer_offsets: Vec<(u32, u64)>,
    pub draw: vk::DrawCommand,
    /// The command is indirect but the profile emulates base vertex/instance through push fields
    /// (`BaseVertexInstance` without `shaderDrawParameters`), the plan's draws are multiview, whose
    /// instances must stay within `maxMultiviewInstanceIndex`, or the device lacks
    /// `drawIndirectFirstInstance`, without which an indirect command's `firstInstance` must be 0
    /// while Metal's indirect arguments may carry a base instance
    /// (`VUID-VkDrawIndirectCommand-firstInstance-00501`), or nothing bounds an indexed draw's
    /// index fetch (the device lacks `robustBufferAccess2`, without which every index must lie
    /// inside the bound range, or the index offset is at or past the buffer's length): the consumer
    /// must read the indirect arguments on the CPU and instantiate a direct draw instead. An
    /// indexed draw's direct `index_offset` is the indirect one plus `indexStart` indices.
    pub indirect_needs_arguments: bool,
    /// Per bound buffer slot and vertex buffer, the window this draw can touch (a fixed-function
    /// vertex buffer's is what the draw's vertex or instance range fetches, and `Whole` when those
    /// are GPU-side).
    pub extents: Vec<BufferExtent>,
    pub push_holes: Vec<PushHole>,
    /// The range of every buffer and texel-buffer descriptor the plan writes.
    pub descriptor_ranges: Vec<DescriptorRange>,
    /// The resident root table of each stage that reads one; the stage's
    /// `PushHole::RootTableAddress` takes its address.
    pub root_tables: Vec<(vk::ShaderStageFlags, RootTable)>,
    /// Vertex strides by Vulkan vertex binding, in binding order. When the pipeline's strides are
    /// dynamic state (`GraphicsState.dynamic_states` holds `VERTEX_INPUT_BINDING_STRIDE`), every
    /// binding's stride, static ones included: `vkCmdBindVertexBuffers2` takes one per binding in
    /// `pStrides`. Otherwise each `VertexStride::PerBind` binding's, which a profile without
    /// dynamic strides makes static pipeline state.
    pub vertex_strides: Vec<(u32, u32)>,
    /// The draw's own resources beyond the plan's `uses`: its index buffer and indirect
    /// arguments. A de-indexed draw reads its indices in the vertex stage as storage
    /// (`VERTEX_SHADER`, from the index offset over its indices), by address, and binds no index
    /// buffer.
    pub uses: Vec<ResourceUse>,
    /// A direct indexed draw that reads indices past its index buffer: how `draw`
    /// treats those index positions. `None` for every other draw.
    pub index_overrun: Option<IndexOverrun>,
    /// An indexed indirect draw asks for its arguments on the CPU (`indirect_needs_arguments`)
    /// only because nothing bounds its index fetch: the consumer may record the GPU index bounds
    /// pass instead of the CPU read (`IndexBoundsPass`), which draws each position past the buffer
    /// at index 0 plus the base vertex.
    pub index_bounds_pass: bool,
    /// A plan that draws each vertex amplification as a draw of its own (`BindingPlan::variant`'s
    /// `amplification_draws`): one entry per amplification, in order, the push writes that make the
    /// next draw that amplification's (its id and its view mapping's offsets, in each stage that
    /// reads them). The consumer records `draw` once per entry, each after `push_constants` and the
    /// entry's writes, all in one render pass instance: Metal renders every primitive of every
    /// instance of one amplification before the next amplification's (a Metal probe on the M3 Max
    /// and an M4), which these draws reproduce. An indirect draw is recorded from the same
    /// arguments each time. Empty for every other plan, whose draw is recorded once.
    pub amplifications: Vec<Vec<PushConstantWrite>>,
    /// A mesh draw's complete specialization, which replaces the plan's: the plan's constants,
    /// the mesh threadgroup's size (`SpecKey::LocalSize`, and its product where the stage's local
    /// size is flattened), and `SpecKey::MultiRegion` 0, and an object function's likewise
    /// (`SpecKey::ObjectLocalSize`, `ObjectFlatLocalSize`, `ObjectMultiRegion`). The consumer creates the draw's
    /// `VkPipeline` with it. `None` for every other draw, whose pipeline takes the plan's.
    pub specialization: Option<SpecializationKey>,
    /// A mesh draw of an emulated mesh function's pipeline (`GraphicsState::mesh_records`): the
    /// compute dispatch that runs its mesh threadgroups before `draw` draws their primitives
    /// ([`MeshDispatch`] states the consumer contract). `None` for every other draw.
    pub mesh_dispatch: Option<MeshDispatch>,
    /// A patch draw of an emulated tessellation pipeline
    /// (`GraphicsState::emulated_tessellation`): the scan that runs before the draw and the
    /// chunk draws that replace `draw` ([`TessellationDispatch`] states the consumer contract).
    /// `None` for every other draw.
    pub tessellation_dispatch: Option<TessellationDispatch>,
    /// An indirect mesh draw on a device with `VK_EXT_mesh_shader`, whose first stage's group
    /// counts `vkCmdDrawMeshTasksIndirectEXT` requires within the device's limits: the form of the
    /// arguments pass the consumer records before the draw ([`MeshArguments`], which it makes with
    /// the device's `minStorageBufferOffsetAlignment`), whose scratch the draw then reads in place
    /// of Metal's arguments (`draw` keeps Metal's offset; the consumer substitutes the scratch's
    /// buffer and offset). Its read of Metal's 12 bytes is among `uses`, in the compute stage.
    /// `None` for every other draw.
    ///
    /// The folded object form's (`MeshArgumentsForm::FoldedObject`, a folded object function's
    /// draw) `draw` is `DrawMeshTasksIndirectCount`, both of whose offsets are into the scratch
    /// (the consumer adds the scratch's offset), and the object stage's
    /// `PushHole::ReplayRegionAddress` takes the scratch's device address (plus
    /// [`MeshArguments::OBJECT_RECORD`](crate::bind::MeshArguments::OBJECT_RECORD)); its scratch
    /// needs `SHADER_DEVICE_ADDRESS` usage and a 16-aligned address.
    pub mesh_arguments: Option<MeshArgumentsForm>,
    /// A direct mesh draw of a folded object function's pipeline: the fold of its object grid
    /// ([`ObjectFold`]), whose [`bytes`](crate::bind::ObjectFold::bytes) the consumer uploads at a
    /// 16-aligned device address of a buffer of `SHADER_DEVICE_ADDRESS` and `INDIRECT_BUFFER`
    /// usage, where the host's write is visible to the draw. The object stage's
    /// `PushHole::ReplayRegionAddress` takes that address, and a `draw` of `DrawMeshTasksIndirect`
    /// (its pieces) reads its commands from the upload, its offset into it. A fold `past` the
    /// device's pieces draws nothing (`draw` launches no threadgroup), and the consumer fails the
    /// command buffer as for `ExecutionStatus`'s `MESH_GRID_PAST_LIMITS`. `None` for every other
    /// draw.
    pub object_fold: Option<crate::bind::ObjectFold>,
    /// A direct mesh draw whose object grid Metal's GPU discards: more threadgroups than
    /// [`METAL_OBJECT_GRID_TOTAL`], past what Metal runs within its watchdog's 5 seconds
    /// (`bind::mesh_arguments`). `draw` launches no threadgroup, and the consumer fails the command
    /// buffer from the draw's encoder on as Metal does: the encoder and every one after it run
    /// nothing, and the command buffer completes with `MTLCommandBufferErrorInternal` (1,
    /// "Discarded (victim of GPU error/recovery)").
    pub discards_command_buffer: bool,
}

/// No draw's commands: empty vectors and a draw of no vertices, which
/// [`BindingPlan::instantiate_draw_into`] writes over.
impl Default for DrawCommands {
    fn default() -> DrawCommands {
        DrawCommands {
            push_constants: Vec::new(),
            descriptor_bytes: Vec::new(),
            buffer_offsets: Vec::new(),
            draw: vk::DrawCommand::Draw {
                vertex_count: 0,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            },
            indirect_needs_arguments: false,
            extents: Vec::new(),
            push_holes: Vec::new(),
            descriptor_ranges: Vec::new(),
            root_tables: Vec::new(),
            vertex_strides: Vec::new(),
            uses: Vec::new(),
            index_overrun: None,
            index_bounds_pass: false,
            mesh_arguments: None,
            object_fold: None,
            discards_command_buffer: false,
            amplifications: Vec::new(),
            specialization: None,
            mesh_dispatch: None,
            tessellation_dispatch: None,
        }
    }
}

/// The compute work of a mesh draw whose mesh function is emulated: its object threadgroups, where
/// the pipeline has an object function, set the mesh grids they launch; its mesh threadgroups write
/// their vertices and primitives to records, from which the pipeline's synthesized vertex stage
/// draws them.
///
/// The draw's **work buffer** (`SyntheticBinding::MeshWork`) is the region of
/// [`work_bytes`](Self::work_bytes) that every step of the draw binds whole: its first words are
/// [`work`](Self::work), the rest zero. Its header holds the draw's object threadgroups and its
/// mesh threadgroups, where its records begin, the mesh dispatch's arguments
/// ([`DISPATCH_OFFSET`](Self::DISPATCH_OFFSET)) and the draw's ([`DRAW_OFFSET`](Self::DRAW_OFFSET));
/// then each object threadgroup's slot (the grid it set and its payload) and each one's first
/// mesh threadgroup.
///
/// The consumer, outside the render pass instance the draw records into, and after every command
/// of the pass that writes what the object or mesh function reads or reads what they write:
///
/// - makes the work buffer and the records ([`records`](Self::records)), and writes them to the
///   plan's `WriteSource::MeshWork` and `WriteSource::MeshRecords` bindings of the draw's set 0;
///   for records in the command buffer's pool it writes the pool's words at word
///   [`POOL_WORDS`](Self::POOL_WORDS) of the work buffer, and, for any draw, its number for the
///   draw at word [`DRAW_ID`](Self::DRAW_ID), which a records fault reports
///   (`StatusReport::mesh_records_overflow`);
/// - creates a compute pipeline of each of the pipeline's `COMPUTE` modules (the object
///   function's first, where it has one), under the pipeline's layout and the draw's
///   `DrawCommands::specialization`;
/// - for an indirect draw with an object function ([`object_arguments`](Self::object_arguments)),
///   dispatches [`mesh_object_arguments_module`](crate::bind::mesh_object_arguments_module) (one
///   workgroup) under the scan's bindings first: it reads Metal's arguments, reserves the draw's
///   slots and offsets in the records pool, and writes the object dispatch's
///   `VkDispatchIndirectCommand` at [`OBJECT_DISPATCH_OFFSET`](Self::OBJECT_DISPATCH_OFFSET);
/// - with an object function, binds its pipeline and the draw's sets at the compute bind point,
///   records the push writes whose stages are `COMPUTE`, and `vkCmdDispatch` of
///   [`object_groups`](Self::object_groups), or, for an indirect draw, `vkCmdDispatchIndirect`
///   from the work buffer at [`OBJECT_DISPATCH_OFFSET`](Self::OBJECT_DISPATCH_OFFSET);
/// - for [`MeshGroups::Scanned`], dispatches [`MeshScanModule`](crate::bind::MeshScanModule)
///   (one workgroup) over the work buffer, the command buffer's `ExecutionStatus` buffer, the
///   arguments the variant names and the records;
/// - binds the mesh function's pipeline and the draw's sets, records the push writes whose
///   stages are `COMPUTE`, and dispatches its threadgroups: `vkCmdDispatch` of
///   [`MeshGroups::Direct`]'s counts (nothing where one is 0), or `vkCmdDispatchIndirect` from
///   the work buffer at [`DISPATCH_OFFSET`](Self::DISPATCH_OFFSET);
/// - between the steps, orders each one's writes before the next one's reads (a barrier from
///   compute shader writes to compute shader and indirect command reads), and after the last,
///   before the draw's vertex stage and indirect command reads; then records `DrawCommands::draw`
///   in the render pass instance as any draw, its indirect buffer (`DrawIndirect`, for
///   [`MeshGroups::Scanned`]) the work buffer.
///
/// The object and mesh functions' resource uses are compute-stage uses (`ResourceUse`'s
/// stages).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshDispatch {
    /// The work buffer's first words; its other words are zero.
    pub work: Vec<u32>,
    /// The work buffer's bytes, at least 16.
    pub work_bytes: u64,
    /// The object function's threadgroups, the object dispatch's group counts; `None` for a
    /// pipeline without an object function, and for an indirect draw with one.
    pub object_groups: Option<[u32; 3]>,
    /// An indirect draw with an object function: the range of the Metal indirect buffer its
    /// object arguments pass binds at `MeshScanModule::ARGUMENTS_BINDING`, from the arguments'
    /// offset aligned down to `minStorageBufferOffsetAlignment` through their 12 bytes (the work
    /// buffer's `OBJECT_ARGUMENTS` word names where they begin). `None` for any other draw.
    pub object_arguments: Option<BufferRange>,
    /// The mesh threadgroups.
    pub mesh: MeshGroups,
    /// The records.
    pub records: MeshRecordsRegion,
}

impl MeshDispatch {
    /// The work buffer's word where the consumer writes its records pool's words, for
    /// [`MeshRecordsRegion::Pool`].
    pub const POOL_WORDS: usize = crate::lower::mesh::work::POOL_WORDS as usize;
    /// The work buffer's word where the consumer writes its number for the draw.
    pub const DRAW_ID: usize = crate::lower::mesh::work::DRAW_ID as usize;
    /// The byte of the work buffer where the mesh dispatch's `VkDispatchIndirectCommand` lies.
    pub const DISPATCH_OFFSET: u64 = 4 * crate::lower::mesh::work::DISPATCH as u64;
    /// The byte of the work buffer where the draw's `VkDrawIndirectCommand` lies.
    pub const DRAW_OFFSET: u64 = 4 * crate::lower::mesh::work::DRAW as u64;
    /// The byte of the work buffer where the object dispatch's `VkDispatchIndirectCommand` lies,
    /// which the object arguments pass writes ([`object_arguments`](Self::object_arguments)).
    pub const OBJECT_DISPATCH_OFFSET: u64 = 4 * crate::lower::mesh::work::OBJECT_DISPATCH as u64;
}

/// Where an emulated mesh draw's mesh threadgroups come from ([`MeshDispatch`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MeshGroups {
    /// The host knows them (a direct draw without an object function): the dispatch's group
    /// counts, the draw's threadgroups folded into rows of `maxComputeWorkGroupCount[0]`. The
    /// work buffer is complete, and the draw is a `Draw`.
    Direct([u32; 3]),
    /// The scan writes them (a draw with an object function, or an indirect draw): the mesh
    /// dispatch is indirect, and the draw a `DrawIndirect`, both from the work buffer.
    Scanned {
        /// The range of the Metal indirect buffer the scan binds at
        /// `MeshScanModule::ARGUMENTS_BINDING`, for an indirect draw: from the arguments' offset
        /// aligned down to `minStorageBufferOffsetAlignment` through their 12 bytes. `None` for
        /// any other draw, whose scan reads no argument (any 4 bytes bind there).
        arguments: Option<BufferRange>,
    },
}

/// The records of an emulated mesh draw ([`MeshDispatch`]), which its mesh dispatch and its draw
/// bind whole.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MeshRecordsRegion {
    /// A region of the draw's own of this many bytes (at least 16), for
    /// [`MeshGroups::Direct`].
    Exact(u64),
    /// The command buffer's records pool, for [`MeshGroups::Scanned`]: storage of the
    /// consumer's size (its budget for the command buffer's emulated mesh draws, within
    /// `maxStorageBufferRange`), which the scans reserve from front to back
    /// (`StatusWord::MeshRecordsCursor`). Its words need no initial value.
    Pool,
}

/// The work of a patch draw whose pipeline's post-tessellation vertex function runs as a plain
/// vertex function (`GraphicsState::emulated_tessellation`, on a device without
/// `tessellationShader`): the tessellation scan
/// ([`tessellation_scan_module`](crate::bind::tessellation_scan_module) of
/// [`kernel`](Self::kernel)) processes each patch's factors into the draw's records, one per
/// instance and patch, and writes the draw's chunks' `VkDrawIndirectCommand`s; each chunk's draw
/// then draws up to `2^30` of the draw's triangles, three vertices each, which the vertex function
/// finds in the records (`lower::tessellation`'s module docs).
///
/// The consumer, outside the render pass instance the draw records into, and after every command
/// of the pass that writes the factor buffer or Metal's indirect arguments:
///
/// - takes `K` = [`chunks`](Self::chunks) of the records pool's bytes (for
///   [`TessellationRecords::Pool`]; any value for `Exact`), and makes the draw's work buffer of
///   [`work`](Self::work)`(K)`'s words, in which it writes its number for the draw at word
///   [`DRAW_ID`](Self::DRAW_ID), which a records fault reports
///   (`StatusReport::tessellation_records_overflow`), and for records in the pool the pool's words
///   at word [`POOL_WORDS`](Self::POOL_WORDS);
/// - makes the records ([`records`](Self::records)), and writes the work buffer and the records to
///   the plan's `WriteSource::TessellationWork` and `WriteSource::TessellationRecords` bindings of
///   the draw's set 0, each bound whole;
/// - for an indirect draw, copies the 16 bytes of Metal's arguments at
///   [`arguments`](Self::arguments) into the work buffer's first 16 bytes, after the arguments'
///   writes, and orders the copy before the scan's reads;
/// - dispatches the scan (one workgroup) with its set over the work buffer, the records, the
///   command buffer's `ExecutionStatus` buffer and the factor buffer's [`factors`](Self::factors)
///   range (the zero buffer, at least 16 bytes, where it is `None`), each scan after the command
///   buffer's earlier scans, mesh or tessellation (each reads and writes the pool's cursor);
/// - orders the scan's writes before the draws' indirect command reads (`DRAW_INDIRECT`) and
///   vertex shader reads (`VERTEX_SHADER`), and the factors' writes before the scan's reads
///   (`COMPUTE_SHADER`);
/// - then in the render pass instance records, in place of `DrawCommands::draw`, each chunk
///   `k < K` in order: the push writes [`draw`](Self::draw)`(k)` gives, then its
///   `vkCmdDrawIndirect` from the work buffer. A chunk past the draw's triangles draws no vertex.
///
/// The scan's reads of the factors and of the arguments (through their copy, which the consumer
/// records with the scan) are `COMPUTE_SHADER` uses of the draw (`DrawCommands::uses`), and the
/// vertex stage's of the patch and control-point index buffers `VERTEX_SHADER` ones.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationDispatch {
    /// The scan's pipeline state.
    pub kernel: crate::bind::TessellationScanKernel,
    /// The work buffer's header (`lower::tessellation::work::HEADER` words): the draw's
    /// arguments (a direct draw's; zero for an indirect one, whose copy fills them), and the
    /// factor buffer's residual bind offset, instance stride and scale.
    pub header: Vec<u32>,
    /// The records.
    pub records: TessellationRecords,
    /// The range of the factor buffer the scan binds, from its bind offset aligned down to
    /// `minStorageBufferOffsetAlignment` over the factors the draw reads; `None` where the
    /// encoder set no factor buffer, whose zero factors cull every patch.
    pub factors: Option<BufferRange>,
    /// For an indirect draw, the offset of Metal's arguments (`MTLDrawPatchIndirectArguments`,
    /// 16 bytes) in the indirect buffer's `VkBuffer`, a multiple of 4, which the consumer copies
    /// into the work buffer's first 16 bytes before the scan. `None` for a direct draw.
    pub arguments: Option<u64>,
    /// Where each stage that reads `PushField::TessellationChunk` holds it, in its push window.
    pub(crate) chunk_fields: Vec<(vk::ShaderStageFlags, u32)>,
    /// The most triangles one patch of the pipeline draws.
    pub(crate) patch_triangles: u32,
}

impl TessellationDispatch {
    /// The work buffer's word where the consumer writes its records pool's words, for
    /// [`TessellationRecords::Pool`].
    pub const POOL_WORDS: usize = crate::lower::tessellation::work::POOL_WORDS as usize;
    /// The work buffer's word where the consumer writes its number for the draw.
    pub const DRAW_ID: usize = crate::lower::tessellation::work::DRAW_ID as usize;

    /// `K`, the draw's chunks: enough for every triangle its records can draw, at most the
    /// pipeline's most per patch for each record. For [`TessellationRecords::Pool`] that is every
    /// record a pool of `pool_bytes` bytes holds; `pool_bytes` is not read otherwise.
    pub fn chunks(&self, pool_bytes: u64) -> u32 {
        use crate::lower::tessellation::work;
        let records = match self.records {
            TessellationRecords::Exact { records } => u64::from(records),
            TessellationRecords::Pool => pool_bytes / (4 * u64::from(work::RECORD_WORDS)),
        };
        let triangles = records.saturating_mul(u64::from(self.patch_triangles));
        u32::try_from(work::chunks(triangles)).unwrap_or(u32::MAX)
    }

    /// The work buffer's words for `chunks` chunks: the header with `K` in it, then each
    /// chunk's slot, zero. Its bytes are 4 per word, which the consumer binds whole.
    pub fn work(&self, chunks: u32) -> Vec<u32> {
        use crate::lower::tessellation::work;
        let words = usize::try_from(work::words(chunks)).unwrap_or(usize::MAX);
        let mut out = Vec::with_capacity(words);
        out.extend_from_slice(&self.header);
        out.resize(words, 0);
        out[work::CHUNKS as usize] = chunks;
        out
    }

    /// Chunk `k`'s draw: the push writes that make the draw chunk `k`'s
    /// (`PushField::TessellationChunk`, in each stage that reads it), and its `vkCmdDrawIndirect`
    /// of one command from the work buffer, whose offset is the chunk slot's.
    pub fn draw(&self, k: u32) -> (Vec<PushConstantWrite>, vk::DrawCommand) {
        let writes = self
            .chunk_fields
            .iter()
            .map(|&(stages, offset)| PushConstantWrite {
                stages,
                offset,
                bytes: k.to_le_bytes().to_vec(),
            })
            .collect();
        let command = vk::DrawCommand::DrawIndirect {
            offset: crate::lower::tessellation::work::draw_offset(k),
            draw_count: 1,
            stride: 16,
        };
        (writes, command)
    }
}

/// The records of an emulated tessellation draw ([`TessellationDispatch`]), `32` bytes per
/// instance and patch, which its scan and its draws bind whole.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TessellationRecords {
    /// A direct draw's `records` (`patchCount · instanceCount`), in a region of the draw's own
    /// of [`bytes`](Self::bytes) bytes.
    Exact { records: u32 },
    /// An indirect draw's, in the command buffer's records pool (the emulated mesh draws' pool,
    /// `MeshRecordsRegion::Pool`), which its scan reserves them from: a draw that does not fit
    /// draws nothing and fails the command buffer
    /// (`StatusReport::tessellation_records_overflow`).
    Pool,
}

impl TessellationRecords {
    /// The bytes of an `Exact` region, at least 16 (the smallest binding); 0 for the pool.
    pub fn bytes(&self) -> u64 {
        match *self {
            TessellationRecords::Exact { records } => {
                (u64::from(records) * 4 * u64::from(crate::lower::tessellation::work::RECORD_WORDS))
                    .max(16)
            }
            TessellationRecords::Pool => 0,
        }
    }
}

/// How a direct indexed draw treats its index positions past the index buffer: the positions whose
/// bytes do not lie wholly before where Metal's fetch stops reading it
/// ([`IndexBufferRef::fetch_end`], the length rounded up to 4 bytes). Metal's API validation
/// rejects such a draw (`indexBufferOffset + indexCount × size` must be at most the buffer's
/// length), and Apple documents no result for it; on an Apple M3 Max (macOS 26.7) and an M4
/// (macOS 27.0.1) each such position reads index 0, to which the base vertex is added.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndexOverrun {
    /// Drawn: the device enables `robustBufferAccess2`, under which Vulkan bounds the index fetch
    /// by the bound index buffer range, so `draw` is the Metal draw's. A position in the
    /// buffer's padding past the bound range of an unsized binding reads the padding's bytes. One
    /// past the `VkBuffer` is taken to read zero, to which Vulkan adds the vertex offset: an
    /// inference from `VK_KHR_robustness2`'s description of out-of-bounds reads, since the
    /// specification names no index value for the fetch.
    Robust {
        /// The index positions past the buffer.
        past_end: u32,
    },
    /// Not drawn: without `robustBufferAccess2` every index a Vulkan draw reads must lie inside
    /// the bound index buffer range (`VUID-vkCmdDrawIndexed-robustBufferAccess2-08798`), so
    /// `draw` takes only the indices wholly before [`IndexBufferRef::fetch_end`]. A draw with
    /// no index before there is clamped with the feature too, since its index offset may lie past the
    /// `VkBuffer`, where it cannot be bound: it draws no index, bound at offset 0.
    Clamped {
        /// The index positions past the buffer, which `draw` leaves out.
        past_end: u32,
    },
}

impl RenderPipeline {
    /// The shape-level binding plan of this pipeline for `shape`: a pure function of the pipeline,
    /// `device` (the device that compiled it), and the shape. Per-draw bytes come from
    /// [`BindingPlan::instantiate_draw`].
    pub fn plan(&self, device: &Device, shape: &BindShape) -> Result<BindingPlan, BindError> {
        plan(self, device, shape)
    }
}

/// The Metal stage of each of a render pipeline's modules `stages` (in pipeline order) that runs
/// a Metal function, with its Vulkan stage: a tessellation pipeline's (`tessellated`) synthesized
/// vertex and control stages run none, and its post-tessellation vertex function is its
/// evaluation stage; a mesh render pipeline's object and mesh functions are its task and mesh
/// stages, and an emulated pipeline's (`emulated`) are its compute stages, the object function's
/// first where it has one, and its synthesized vertex stage runs none; the geometry stages link
/// generates (`link::geometry`: barycentric weights read as varyings, a selected slice or
/// viewport) run none and bind nothing.
fn metal_stages(
    stages: impl ExactSizeIterator<Item = vk::ShaderStage>,
    tessellated: bool,
    emulated: bool,
) -> Result<Vec<(Stage, vk::ShaderStage)>, BindError> {
    let stages: Vec<vk::ShaderStage> = stages.collect();
    let mut out = Vec::with_capacity(stages.len());
    for (i, &s) in stages.iter().enumerate() {
        let compute = s == vk::ShaderStage::COMPUTE && emulated;
        let object_first = compute && stages.get(i + 1) == Some(&vk::ShaderStage::COMPUTE);
        let stage = if s == vk::ShaderStage::FRAGMENT {
            Stage::Fragment
        } else if object_first {
            Stage::Object
        } else if s == vk::ShaderStage::MESH_EXT || compute {
            Stage::Mesh
        } else if s == vk::ShaderStage::TASK_EXT {
            Stage::Object
        } else if s == vk::ShaderStage::VERTEX && emulated {
            continue;
        } else if s == vk::ShaderStage::VERTEX && !tessellated
            || s == vk::ShaderStage::TESSELLATION_EVALUATION && tessellated
        {
            Stage::Vertex
        } else if tessellated
            && (s == vk::ShaderStage::VERTEX || s == vk::ShaderStage::TESSELLATION_CONTROL)
            || s == vk::ShaderStage::GEOMETRY && !emulated
        {
            continue;
        } else {
            return Err(internal(format!("a render pipeline stage {s:?}")));
        };
        out.push((stage, s));
    }
    Ok(out)
}

/// Whether a module of mesh render pipeline `p` other than its object function's declares
/// threadgroup memory arguments (`SpecKey::TgLength`): a mesh function's, whose lengths no encoder
/// setter sets. The object module is the task stage, or, `emulated`, the first of the two compute
/// stages, as [`metal_stages`] names them.
fn other_stage_lengths(p: &RenderPipeline, emulated: bool) -> bool {
    p.stages.iter().enumerate().any(|(i, m)| {
        let object = m.stage == vk::ShaderStage::TASK_EXT
            || emulated
                && m.stage == vk::ShaderStage::COMPUTE
                && p.stages.get(i + 1).map(|n| n.stage) == Some(vk::ShaderStage::COMPUTE);
        !object
            && m.specialization
                .constants
                .iter()
                .any(|c| matches!(c.key, SpecKey::TgLength(_)))
    })
}

/// The vertex-stage buffer slot of Metal buffer index `index`.
fn vertex_buffer(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Vertex,
        class: SlotClass::Buffer,
        index,
    }
}

fn pipeline_stage(stage: Stage) -> vk::PipelineStageFlags2 {
    match stage {
        Stage::Vertex => vk::PipelineStageFlags2::VERTEX_SHADER,
        Stage::Fragment => vk::PipelineStageFlags2::FRAGMENT_SHADER,
        Stage::Kernel => vk::PipelineStageFlags2::COMPUTE_SHADER,
        Stage::Object => vk::PipelineStageFlags2::TASK_SHADER_EXT,
        Stage::Mesh => vk::PipelineStageFlags2::MESH_SHADER_EXT,
    }
}

/// [`pipeline_stage`] of a tessellation pipeline, whose vertex function is its evaluation stage.
fn tessellated_pipeline_stage(stage: Stage) -> vk::PipelineStageFlags2 {
    match stage {
        Stage::Vertex => vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
        other => pipeline_stage(other),
    }
}

/// [`pipeline_stage`] of an emulated mesh function's pipeline, whose object and mesh functions
/// run in the draw's compute dispatches.
fn emulated_mesh_pipeline_stage(stage: Stage) -> vk::PipelineStageFlags2 {
    match stage {
        Stage::Object | Stage::Mesh => vk::PipelineStageFlags2::COMPUTE_SHADER,
        other => pipeline_stage(other),
    }
}

/// How a plan draws the shape's vertex amplification.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Route {
    /// As Vulkan multiview views (at a count of 1, into one layer without multiview).
    Views(ViewSetup),
    /// One draw per amplification, in amplification order: the `(render target slice offset,
    /// viewport offset)` of each, in a pass of more than one layer when `layered`.
    Draws {
        offsets: Vec<[u32; 2]>,
        layered: bool,
    },
}

/// The route of draws under vertex amplification `amp` with a pipeline whose
/// `maxVertexAmplificationCount` is `max` and in which a stage chooses or reads the render
/// target slice when `declares_layer`, in a pass whose `renderTargetArrayLength` is `length`,
/// on a device of `max_views` multiview views (see the module docs).
fn route(
    amp: &VertexAmplification,
    max: u32,
    declares_layer: bool,
    length: u32,
    max_views: u32,
) -> Result<Route, BindError> {
    let count = amp.count;
    let given = u32::try_from(amp.view_mappings.len()).unwrap_or(u32::MAX);
    if count == 0 || (count > 1 && max == 1) || (given != 0 && given < count) {
        return Err(BindError::InvalidVertexAmplification {
            count,
            max,
            view_mappings: given,
        });
    }
    // A pipeline of `maxVertexAmplificationCount` 1 takes no mapping; the layer amplification `a`
    // renders into is its offset below the pass's layers, the first layer at or past them (a Metal
    // probe on an M4).
    let mapping = |a: u32| amp.mapping(a, max);
    let layer = |a: u32| amp.layer(a, length, max);
    let base_layer = layer(0);
    let consecutive = (0..count).all(|a| {
        mapping(a).viewport_array_index_offset == 0 && Some(layer(a)) == base_layer.checked_add(a)
    });
    let as_views = consecutive
        && (count == 1 || (count <= max_views && !amp.one_at_a_time))
        && !(declares_layer && (count > 1 || base_layer != 0));
    if !as_views {
        // Metal renders amplification-major, every primitive of every instance of one amplification
        // before the next's (a Metal probe on the M3 Max and an M4): exactly a draw of each in
        // turn.
        return Ok(Route::Draws {
            offsets: (0..count)
                .map(|a| {
                    let m = mapping(a);
                    [
                        m.render_target_array_index_offset,
                        m.viewport_array_index_offset,
                    ]
                })
                .collect(),
            layered: length > 1,
        });
    }
    Ok(Route::Views(ViewSetup {
        view_mask: if count == 1 {
            0
        } else {
            1u32.checked_shl(count).map_or(u32::MAX, |bit| bit - 1)
        },
        base_layer,
    }))
}

/// Whether the vertex function of `p` chooses its render target slice, and if so whether it was
/// compiled without its `Layer` output (`RuntimeFact::RenderTargetArrayLength`).
fn vertex_layer(p: &RenderPipeline) -> Option<bool> {
    p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::RenderTargetArrayLength { layer_dropped } => Some(layer_dropped),
        _ => None,
    })
}

/// Whether a draw's write to the ordinary color attachment of state `a` stores something: its
/// mask names some of the format's color components (Vulkan writes nothing to a component the
/// mask leaves out, nor to one the format lacks), as the storage form's write stores something
/// where its mask covers some of the format's channels (`StorageView::stores_nothing`).
fn stores(a: crate::pipeline::graphics::ColorAttachmentState) -> bool {
    a.format
        .block()
        .is_some_and(|b| a.write_mask.intersects(b.color_components))
}

fn plan(p: &RenderPipeline, device: &Device, shape: &BindShape) -> Result<BindingPlan, BindError> {
    check_slots(shape)?;
    let records = p.graphics.mesh_records.as_ref();
    // An emulated tessellation pipeline: its post-tessellation vertex function is its vertex
    // stage, and its draws scan their patches first.
    let emulated_tessellation = p.graphics.emulated_tessellation;
    let tessellation = p.graphics.tessellation.is_some() || emulated_tessellation.is_some();
    let modules = metal_stages(
        p.stages.iter().map(|m| m.stage),
        p.graphics.tessellation.is_some(),
        records.is_some(),
    )?;
    let stages: Vec<Stage> = modules.iter().map(|(stage, _)| *stage).collect();
    // A mesh render pipeline (`Device::compile_mesh_pipeline`) is drawn by mesh draws.
    let mesh = stages.contains(&Stage::Mesh);
    // A vertex function that chooses its slice is never drawn amplified as views, whether or
    // not this variant writes `Layer`: its amplified draws are draws of one amplification each.
    let vertex_layer = vertex_layer(p);
    // A fragment function compiled to read its slice as the amplification's offset declares `Layer`
    // in the pipeline this variant is of: the route stays that pipeline's.
    let reads_layer_offset = p.interface.push.iter().any(|pl| {
        pl.stage == Stage::Fragment
            && pl
                .fields
                .iter()
                .any(|f| f.field == PushField::AmplificationLayerOffset)
    });
    // A pipeline whose vertex function writes the primitive ID's varying runs an indexed draw
    // in its de-indexed form, of the draw's index type.
    let writes_primitive_ids = p.interface.push.iter().any(|pl| {
        pl.stage == Stage::Vertex
            && pl
                .fields
                .iter()
                .any(|f| f.field == PushField::VerticesPerPrimitive)
    });
    let primitive_id_indices = shape.primitive_id_indices.filter(|_| writes_primitive_ids);
    // A draw of Metal's private quads, listed as two triangles a quad, whose fragment function
    // reads Vulkan's `PrimitiveId` (not the varying a vertex function writes): the variant that
    // reads the quad's.
    let quad_primitive_ids =
        shape.quad_primitive_ids && p.graphics.reads_primitive_id && !writes_primitive_ids;
    let compiled_quads = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::QuadPrimitiveIds);
    // A draw of one of Metal's private primitive types the consumer draws in the pulled list
    // form: the variant that runs it so.
    let private_list = shape.private_list;
    let compiled_list = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::PrivateList);
    let route = route(
        &shape.vertex_amplification,
        p.graphics.max_vertex_amplification_count,
        p.graphics.declares_layer || vertex_layer.is_some() || reads_layer_offset,
        shape.render_target_array_length,
        device.profile().fields().limits.max_multiview_view_count,
    )?;
    let (views, amplification_draws, amplifications) = match route {
        Route::Views(views) => (views, None, Vec::new()),
        // A tessellation pipeline's patches are not drawn one amplification at a time.
        Route::Draws { .. } if tessellation => {
            return Err(not_implemented(Feature::VertexAmplification))
        }
        // A mesh function does not add a view mapping's offsets yet.
        Route::Draws { .. } if mesh => return Err(not_implemented(Feature::MeshShading)),
        Route::Draws { offsets, layered } => {
            let viewports = offsets.iter().any(|&[_, viewport]| viewport != 0);
            (
                ViewSetup::default(),
                Some(AmplificationDraws::new(layered, viewports)),
                offsets,
            )
        }
    };
    // The variant the pipeline was compiled under: its vertex function states it.
    let compiled_draws = p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::AmplificationDraws { layered, viewports } => {
            Some(AmplificationDraws::new(layered, viewports))
        }
        _ => None,
    });
    // The fragment imageblock is one plane per render pass instance. What Metal gives each layer of
    // a layered pass, or each view of an amplified draw, is not read yet: such a pass is not
    // implemented.
    if p.graphics.fragment_imageblock.is_some()
        && (shape.render_target_array_length > 1 || shape.vertex_amplification.count > 1)
    {
        return Err(not_implemented(Feature::Imageblocks));
    }
    // An attachment read in the interlocked storage form is one 2D view: what a layered pass, or an
    // amplified draw, reads and writes of each layer or view is not modelled.
    if !p.graphics.storage_attachments.is_empty()
        && (shape.render_target_array_length > 1 || shape.vertex_amplification.count > 1)
    {
        return Err(not_implemented(Feature::FramebufferFetch));
    }
    // The shader-side stencil's shadow is one 2D image: the same holds.
    let shadow = shadow_template(p);
    if shadow.is_some()
        && (shape.render_target_array_length > 1 || shape.vertex_amplification.count > 1)
    {
        return Err(not_implemented(Feature::StencilExport));
    }
    // A fragment function's tile threadgroup memory is the pass's, every tile's in one buffer:
    // what Metal gives each layer of a layered pass, or each view of an amplified draw, is not
    // read yet, as for a tile dispatch.
    let tile_memory = tile_memory(&p.dependencies, device, shape)?;
    if tile_memory.is_some()
        && (shape.render_target_array_length > 1 || shape.vertex_amplification.count > 1)
    {
        return Err(not_implemented(Feature::TileShading));
    }
    // In a pass of at most one layer every index renders slice 0: the variant drops the vertex
    // `Layer` output there, and only there. A mesh function writes `Layer` in every variant.
    let dropped_vertex_layer =
        vertex_layer.is_some() && !mesh && shape.render_target_array_length <= 1;
    // A viewport Vulkan's cannot be: the vertex function compensates its position, and the
    // pipeline compiled under another compensation (or none) is the other variant.
    let viewport_compensation = shape.viewport_compensation;
    let compiled_compensation = p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::ViewportCompensation {
            depth,
            point_guard,
            mirrors,
            vertex_clamp,
        } => Some(
            ViewportCompensation::new(depth)
                .with_point_guard(point_guard)
                .with_mirrors(mirrors)
                .with_vertex_clamp(vertex_clamp),
        ),
        _ => None,
    });
    let profile = device.profile();
    let limits = &profile.fields().limits;
    // A mesh pipeline has no vertex function to compensate.
    if viewport_compensation.is_some() && !stages.contains(&Stage::Vertex) {
        return Err(not_implemented(Feature::MeshShading));
    }
    // The pass's colors the pipeline lacks, which plain Metal zeroes where the draw covers them:
    // the variant whose pipeline has them and whose fragment function writes the zero. A mesh
    // pipeline's is not built; a pipeline without a fragment function rasterizes into no color,
    // and what Metal stores there is unmeasured, so a consumer zeroes none for it.
    let zeroed_colors = shape.zeroed_colors.clone();
    if !zeroed_colors.is_empty() && mesh {
        return Err(not_implemented(Feature::MeshShading));
    }
    if !zeroed_colors.is_empty() && !stages.contains(&Stage::Fragment) {
        return Err(internal(
            "colors zeroed for a pipeline without a fragment function",
        ));
    }
    let compiled_zeroed: std::collections::BTreeMap<u32, crate::metal::PixelFormat> = p
        .dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::ZeroedColor { color, format } => Some((color, format)),
            _ => None,
        })
        .collect();
    // The pipeline's colors the pass lacks, whose writes plain Metal drops: the variant whose
    // fragment function keeps no output there, so the pipeline writes no location without an
    // attachment. A mesh pipeline's is not built.
    let dropped_colors = shape.dropped_colors.clone();
    if !dropped_colors.is_empty() && mesh {
        return Err(not_implemented(Feature::MeshShading));
    }
    if !dropped_colors.is_empty() && !stages.contains(&Stage::Fragment) {
        return Err(internal(
            "colors dropped for a pipeline without a fragment function",
        ));
    }
    let compiled_dropped: std::collections::BTreeSet<u32> = p
        .dependencies
        .iter()
        .filter_map(|d| match d.fact {
            RuntimeFact::DroppedColor { color } => Some(color),
            _ => None,
        })
        .collect();
    // The colors the draw's instance keeps out of its subpass that the fragment function writes
    // through its first output: the variant that writes them through the interlocked storage
    // form, as a fetching function does. A pipeline that writes them so is that variant. A mesh
    // pipeline's is not built.
    let interlocked_colors: std::collections::BTreeSet<u32> = shape
        .interlocked_colors
        .iter()
        .copied()
        .filter(|&c| {
            1u32.checked_shl(c)
                .is_some_and(|bit| p.graphics.color_outputs & bit != 0)
        })
        .collect();
    if !interlocked_colors.is_empty() && mesh {
        return Err(not_implemented(Feature::MeshShading));
    }
    // Nor a multisampled draw's, whose storage form reads and writes the attachment per sample,
    // which runs the function per sample where a plain writer runs it per pixel; nor a layered
    // pass's or an amplified draw's, whose storage form is one 2D view (above).
    if !interlocked_colors.is_empty()
        && (p.graphics.rasterization_samples.as_raw() > 1
            || shape.render_target_array_length > 1
            || shape.vertex_amplification.count > 1)
    {
        return Err(not_implemented(Feature::FramebufferFetch));
    }
    let compiled_interlocked: std::collections::BTreeSet<u32> = p
        .graphics
        .storage_attachments
        .iter()
        .filter(|a| a.writes && shape.interlocked_colors.contains(&a.color_index))
        .map(|a| a.color_index)
        .collect();
    // The extended-range colors whose `half` shadow the draw's instance holds that the fragment
    // function writes or reads in the storage form: the variant that reads and writes the
    // shadows beside the codes. Every such color is single-sampled in a pass of one layer (the
    // encoder holds no other shadow), as the interlocked colors above are; a pipeline compiled
    // with shadows binds them.
    let extended_shadows: std::collections::BTreeSet<u32> = shape
        .extended_shadows
        .iter()
        .copied()
        .filter(|&c| {
            1u32.checked_shl(c)
                .is_some_and(|bit| p.graphics.color_outputs & bit != 0)
                || p.graphics
                    .storage_attachments
                    .iter()
                    .any(|a| a.color_index == c)
        })
        .collect();
    if !extended_shadows.is_empty()
        && (mesh
            || p.graphics.rasterization_samples.as_raw() > 1
            || shape.render_target_array_length > 1
            || shape.vertex_amplification.count > 1)
    {
        return Err(not_implemented(Feature::FramebufferFetch));
    }
    let compiled_shadows: std::collections::BTreeSet<u32> = p
        .layout
        .sets
        .iter()
        .flat_map(|set| &set.bindings)
        .filter_map(|b| match b.source {
            BindingSource::ExtendedShadow(color) => Some(color),
            _ => None,
        })
        .collect();
    // The fragment texture slots the draw reads through a snapshot: those the shape names that
    // the fragment function can read so, where the pipeline's two draws are Metal's one. The two
    // run the same fragments (the pipeline differs only in its dynamic write masks), the first
    // with every mask 0 running none of the function past its copy, so the pipeline may hold
    // nothing else either draw writes, tests, or orders: one sample, no depth or stencil, no
    // memory either function writes, no attachment, imageblock, tile memory, stencil shadow or
    // stand-in of the storage form or an input attachment, and no other interlock. A tessellation
    // or mesh pipeline's, and an amplified or layered draw's are not built.
    // The device makes the variant's pipeline: its feedback loop flag and dynamic write masks
    // need `attachmentFeedbackLoopLayout` and `extendedDynamicState3ColorWriteMask`.
    // Every other slot the shape names is left out, and the consumer reads it as it would without
    // a snapshot.
    let enabled = &profile.fields().enabled.features;
    let snapshot_able = profile.fields().fragment.pixel_interlock
        && enabled.contains(&VkFeature::FragmentStoresAndAtomics)
        && enabled.contains(&VkFeature::AttachmentFeedbackLoopLayout)
        && enabled.contains(&VkFeature::ExtendedDynamicState3ColorWriteMask)
        && p.graphics.rasterization_samples.as_raw() <= 1
        && p.graphics.depth_format.is_none()
        && p.graphics.stencil_format.is_none()
        && !p.graphics.rasterizer_discard
        && !p.graphics.vertex_writes_memory
        && !p.graphics.fragment_writes_memory
        && p.graphics.storage_attachments.is_empty()
        && p.graphics.input_attachments.is_empty()
        && p.graphics.fragment_imageblock.is_none()
        && p.graphics.shader_tests.is_none()
        && p.graphics.tessellation.is_none()
        && shadow.is_none()
        && tile_memory.is_none()
        && records.is_none()
        && !mesh
        && interlocked_colors.is_empty()
        && extended_shadows.is_empty()
        && shape.vertex_amplification.count <= 1
        && shape.render_target_array_length <= 1;
    let snapshot_textures: std::collections::BTreeSet<SlotRef> = match snapshot_able {
        true => p
            .stages
            .iter()
            .filter(|m| m.stage == vk::ShaderStage::FRAGMENT)
            .flat_map(|m| &m.snapshot_textures)
            .filter(|index| shape.snapshot_textures.contains(index))
            .map(|&index| SlotRef {
                stage: Stage::Fragment,
                class: SlotClass::Texture,
                index,
            })
            .collect(),
        false => std::collections::BTreeSet::new(),
    };
    let compiled_snapshots: std::collections::BTreeSet<SlotRef> = p
        .layout
        .sets
        .iter()
        .flat_map(|set| &set.bindings)
        .filter_map(|b| match b.source {
            BindingSource::SnapshotTexels(index) => Some(SlotRef {
                stage: Stage::Fragment,
                class: SlotClass::Texture,
                index,
            }),
            _ => None,
        })
        .collect();
    // A draw in depth clamp mode through a range above 0 whose Vulkan depth clamp holds 0: the
    // variant whose fragment function clamps the depth it writes to the draw's range. A pipeline
    // whose fragment function writes no depth has none to clamp.
    let depth_clamp_range = shape.depth_clamp_range && p.graphics.writes_depth;
    let compiled_range = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::DepthClampRange);
    // A draw of triangles with Metal's last vertex first: the variant whose fragment function
    // takes Metal's barycentric weights back from the drawn triangle's. A pipeline whose fragment
    // function reads no weights has none to take.
    let rotated_weights = shape.rotated_weights && p.graphics.reads_barycentric_coord;
    let compiled_rotated = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::RotatedWeights);
    // A mesh draw drawn from each primitive's last index first: the variant whose emulated
    // mesh pipeline's synthesized vertex stage reads its corners so. A pipeline of points, or
    // one whose mesh function is not emulated, has no order to change.
    let mesh_last_first = shape.mesh_last_first
        && p.graphics
            .mesh_records
            .as_ref()
            .is_some_and(|r| r.corners > 1);
    let compiled_last_first = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::MeshLastFirst);
    // A fragment function that stores a color through the interlocked storage form before the
    // depth and stencil tests, since it discards (`RuntimeFact::TestedStorageWrites`), in a
    // pipeline with a depth or stencil attachment: Metal's store, as compiled, where the draw's
    // tests drop no fragment; where they drop some but write and count nothing, the variant whose
    // function declares early tests, which a pipeline compiled under it states; refused
    // otherwise. A draw that moves a color the function writes as an ordinary output into that
    // form (an interlocked color, a shadowed extended-range one) asks for the variant that stores
    // it so: where the pipeline as compiled stores no color so, the function states that variant's
    // tests (`RuntimeFact::MovableStorageWrites`), so this plan selects them, and the variant it
    // names plans itself. A moved write whose mask names none of its attachment's components
    // stores nothing, so no test bears on it: its variant states none, and this plan selects none
    // for it. (A color the pipeline as compiled already writes through the storage form has no
    // ordinary attachment state; its stores state their own tests.)
    let moves = interlocked_colors
        .iter()
        .chain(&extended_shadows)
        .any(|&c| {
            1u32.checked_shl(c)
                .is_some_and(|bit| p.graphics.color_outputs & bit != 0)
                && p.graphics
                    .color_attachments
                    .get(c as usize)
                    .copied()
                    .flatten()
                    .is_some_and(stores)
        });
    let tested = p
        .dependencies
        .iter()
        .find_map(|d| match d.fact {
            RuntimeFact::TestedStorageWrites {
                early_tests,
                early_variant,
            } => Some((early_tests, early_variant)),
            _ => None,
        })
        .or_else(|| {
            let movable = p.dependencies.iter().find_map(|d| match d.fact {
                RuntimeFact::MovableStorageWrites { early_variant } => Some((false, early_variant)),
                _ => None,
            });
            movable.filter(|_| moves)
        });
    let tested_attachments =
        p.graphics.depth_format.is_some() || p.graphics.stencil_format.is_some();
    // What stores before the tests: the storage form's colors, or else the imageblock.
    let tested_feature = match p.graphics.fragment_imageblock {
        Some(_) if p.graphics.storage_attachments.is_empty() => Feature::Imageblocks,
        _ => Feature::FramebufferFetch,
    };
    // Where only the depth test can drop a fragment, the stencil test does nothing and nothing
    // is counted, the variant whose function runs the depth test itself (`shader_tests`), where
    // the device ships it at the pipeline's samples; a draw whose tests write nothing keeps the
    // early-tests variant where the function has one.
    let samples = p.graphics.rasterization_samples.as_raw().max(1);
    // The variant whose function runs the depth test itself, and the stencil test too where
    // `stencil`, or the refusal of the device or the attachment that lacks it.
    let in_shader = |stencil: bool| -> Result<(bool, bool, bool, bool), BindError> {
        // The stand-in holds 32-bit float depth (link refuses Metal's other depth formats).
        let float = matches!(
            p.graphics.depth_format,
            Some(vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT)
        );
        if !float {
            return Err(BindError::Unsupported(
                Unsupported::not_implemented(tested_feature.clone())
                    .at_attachment(Attachment::Depth),
            ));
        }
        // A function that returns its depth tests that, at every sample alike.
        let form =
            crate::lower::fragment::depth_test_form(profile, samples > 1, p.graphics.writes_depth)
                .map_err(|refused| {
                    BindError::Unsupported(refused.at_attachment(Attachment::Depth))
                })?;
        if stencil {
            crate::lower::fragment::stencil_test_form(profile, samples > 1).map_err(|refused| {
                BindError::Unsupported(refused.at_attachment(Attachment::Stencil))
            })?;
        }
        // Each sample's depth computed from the vertex function's window depth.
        Ok((false, true, form.computed, stencil))
    };
    let depth = p.graphics.depth_format.is_some();
    let stencil = p.graphics.stencil_format.is_some();
    let (read_only_tests, shader_tests, window_depth_varying, shader_stencil) =
        match (tested, shape.fragment_tests) {
            (Some((_, true)), FragmentTests::Drop | FragmentTests::Depth { writes: false })
                if tested_attachments =>
            {
                (true, false, false, false)
            }
            (Some(_), FragmentTests::Depth { .. }) if depth => in_shader(false)?,
            // The stencil test acts (or, for a function without the early-tests variant, may):
            // the function runs both tests, in a pass with both attachments. A pass with a
            // stencil attachment alone has no depth for the form's attachment: not implemented,
            // refused at the stencil attachment.
            (Some(_), FragmentTests::Drop | FragmentTests::Stencil) if depth && stencil => {
                in_shader(true)?
            }
            (Some(_), FragmentTests::Drop | FragmentTests::Stencil) if stencil => {
                return Err(BindError::Unsupported(
                    Unsupported::not_implemented(tested_feature.clone())
                        .at_attachment(Attachment::Stencil),
                ));
            }
            // A function that writes other memory or returns depth, stencil, or a sample mask: its
            // stores are Metal's only where no test can drop a fragment. The refusal is placed at
            // the tested attachment (the depth one where the pipeline has one, else the stencil).
            (
                Some((_, false)),
                FragmentTests::Drop
                | FragmentTests::Depth { .. }
                | FragmentTests::Any
                | FragmentTests::Stencil,
            ) if tested_attachments => {
                let at = match p.graphics.depth_format {
                    Some(_) => Attachment::Depth,
                    None => Attachment::Stencil,
                };
                return Err(BindError::Unsupported(
                    Unsupported::not_implemented(tested_feature.clone()).at_attachment(at),
                ));
            }
            (Some(_), FragmentTests::Any | FragmentTests::Stencil) if tested_attachments => {
                return Err(not_implemented(tested_feature.clone()));
            }
            _ => (false, false, false, false),
        };
    let compiled_tests = tested.is_some_and(|(early, _)| early);
    let compiled_shader_tests = p
        .dependencies
        .iter()
        .any(|d| matches!(d.fact, RuntimeFact::ShaderTests { .. }));
    let compiled_shader_stencil = p
        .dependencies
        .iter()
        .any(|d| matches!(d.fact, RuntimeFact::ShaderStencilTest { .. }));
    // A mesh entry's `Layer` at or past `maxMeshOutputLayers` is undefined: a pass of more layers
    // than that, which a mesh function choosing its slice could render into, is past the device. An
    // emulated mesh function's slice is its synthesized vertex stage's `Layer`, as a vertex
    // function's is.
    let mesh_layers = limits.mesh.max_mesh_output_layers;
    if mesh
        && records.is_none()
        && vertex_layer.is_some()
        && shape.render_target_array_length > mesh_layers
    {
        return Err(BindError::Unsupported(Unsupported::needs(
            Feature::MeshShading,
            Capability::Limit {
                limit: VkLimit::MaxMeshOutputLayers,
                required: u64::from(shape.render_target_array_length),
                available: u64::from(mesh_layers),
            },
        )));
    }
    // The Vulkan stage of a Metal stage's module.
    let vk_of = |stage: Stage| {
        modules
            .iter()
            .find(|(s, _)| *s == stage)
            .map_or_else(|| vk_stage(stage), |(_, m)| *m)
    };
    let stage_flags: fn(Stage) -> vk::PipelineStageFlags2 =
        match (&p.graphics.tessellation, records) {
            (Some(_), _) => tessellated_pipeline_stage,
            (None, Some(_)) => emulated_mesh_pipeline_stage,
            (None, None) => pipeline_stage,
        };
    let bindings = bindings(&p.layout);
    let stage_shape = |s: Stage| shape.stages.get(&s);

    // Buffer slots: the interface's. Every pulled source buffer is one of them, in the form the
    // access family chose (a storage view or a device address, shared with a buffer argument of
    // its index or its own).
    let mut slots: BTreeMap<SlotRef, BufferSlot> = BTreeMap::new();
    for (&slot, iface) in &p.interface.slots {
        match iface {
            SlotInterface::Buffer(b) if stages.contains(&slot.stage) => {
                slots.insert(slot, *b);
            }
            SlotInterface::Buffer(_) => {
                return Err(internal(format!("a render pipeline uses {slot:?}")))
            }
            SlotInterface::Texture(_) | SlotInterface::Sampler(_) => {}
        }
    }
    if let Some(a) = p
        .vertex_input
        .pulled
        .iter()
        .find(|a| !slots.contains_key(&vertex_buffer(a.buffer_index)))
    {
        return Err(internal(format!(
            "pulled vertex buffer {} without an interface slot",
            a.buffer_index
        )));
    }
    let mut buffers = Vec::with_capacity(slots.len());
    for (slot, b) in &slots {
        let bound = stage_shape(slot.stage).and_then(|s| s.buffers.get(&slot.index));
        buffers.push(plan_buffer(*slot, b, bound, limits, &bindings)?);
    }
    mark_disjoint(&mut buffers, &p.dependencies);
    let planned = |slot: SlotRef| buffers.iter().find(|b| b.slot == slot);
    // The layout stride of each pulled layout (one per Metal buffer index).
    let pulled_stride = |layout: u32| {
        p.vertex_input
            .pulled
            .iter()
            .find(|a| a.buffer_index == layout)
            .map(|a| a.stride)
    };
    let mut pulled_strides: Vec<(u32, u32)> = p
        .vertex_input
        .pulled
        .iter()
        .filter_map(|a| match a.stride {
            VertexStride::Static(s) => Some((a.buffer_index, s)),
            VertexStride::PerBind => None,
        })
        .collect();
    pulled_strides.sort_unstable();
    pulled_strides.dedup();

    // Push placement, per stage.
    let mut push = PushTemplate::default();
    let mut templates = Vec::new();
    for &stage in &stages {
        let placement = p.interface.push.iter().find(|pl| pl.stage == stage);
        let fields = placement.map(|pl| pl.fields.clone()).unwrap_or_default();
        let mut reads_root_table = false;
        for field in &fields {
            let hole = matches!(
                field.field,
                PushField::RootTableAddress
                    | PushField::ExecutionStatusAddress
                    | PushField::CaptureIndexAddress
                    | PushField::MergedDrawAddress
                    | PushField::SnapshotPhase
                    | PushField::ReplayRegionAddress
            );
            match field.field {
                PushField::RootTableAddress => reads_root_table = true,
                PushField::ExecutionStatusAddress => {}
                // A de-indexed or private list draw's indices, and its draw's layout: the address
                // is the consumer's hole, the layout instantiation writes from the draw.
                PushField::CaptureIndexAddress | PushField::CaptureLayout
                    if stage == Stage::Vertex => {}
                PushField::BaseVertexInstance if stage == Stage::Vertex => {}
                // A merged draw replay's header: the consumer's hole.
                PushField::MergedDrawAddress if stage == Stage::Vertex => {}
                // The draw's vertices per primitive, which instantiation writes from its
                // primitive type.
                PushField::VerticesPerPrimitive if stage == Stage::Vertex => {}
                // Which of a snapshot variant's draws runs: the consumer's hole.
                PushField::SnapshotPhase if stage == Stage::Fragment => {}
                PushField::SlotBindOffset(slot)
                    if planned(slot).is_some_and(|b| b.binding.is_some()) => {}
                // The bound buffer's Metal address; a `set*Bytes` binding has none.
                PushField::SlotGuestBase(slot)
                    if planned(slot).is_some_and(|b| b.bound != BoundKind::Bytes) => {}
                PushField::SlotGuestBase(slot) if planned(slot).is_some() => {
                    return Err(BindError::MissingGuestAddress { slot })
                }
                PushField::SlotBindOffset(SlotRef {
                    class: SlotClass::Texture,
                    ..
                }) => return Err(not_implemented(Feature::TexelBuffer)),
                // A storage texture's level count and integer channel widths, which the resource
                // plan records; and a value the bound resources decide, which the plan records too
                // (`slots::record_specialized`).
                PushField::TextureLevels(_)
                | PushField::TextureIntegerChannels(_)
                | PushField::Specialized(_) => {}
                // A structure's instance count, which instantiation reads from the state.
                PushField::InstanceCount(_) => {}
                // An in-place YCbCr texture's words, which instantiation reads from the state.
                PushField::YCbCrInPlace(_) => {}
                // The encoder's stencil state, which instantiation reads from the state.
                PushField::ShadowStencilState if stage == Stage::Fragment => {}
                // The encoder's depth state, which instantiation reads from the state, and its
                // stencil state and references.
                PushField::ShaderTestsState | PushField::ShaderStencilState
                    if stage == Stage::Fragment => {}
                // The encoder's viewport count, which instantiation reads from the state, in the
                // stage that chooses a viewport (a mesh function per primitive).
                PushField::ViewportCount if matches!(stage, Stage::Vertex | Stage::Mesh) => {}
                // The pass's array length, which instantiation reads from the state.
                PushField::RenderTargetArrayLength
                    if matches!(stage, Stage::Vertex | Stage::Mesh) => {}
                // A mesh draw's grid of each stage's threadgroups, which instantiation writes from
                // the draw.
                PushField::DispatchRegion if matches!(stage, Stage::Object | Stage::Mesh) => {}
                // A folded object function's region record (`bind::mesh_arguments`): the
                // consumer's hole.
                PushField::ReplayRegionAddress if stage == Stage::Object => {}
                // The draw's viewport compensation, which instantiation reads from the state.
                PushField::ViewportCompensation if stage == Stage::Vertex => {}
                // The draw's point guard band, which instantiation reads from the state.
                PushField::PointGuard if stage == Stage::Vertex => {}
                // The draw's mirrored viewports, which instantiation reads from the state.
                PushField::ViewportMirrors if stage == Stage::Vertex => {}
                // The draw's clamp of each vertex's depth, which instantiation reads from the
                // state.
                PushField::VertexDepthClamp if stage == Stage::Vertex => {}
                // The draw's depth clamp range, which instantiation reads from the state.
                PushField::DepthClampRange if stage == Stage::Fragment => {}
                // The encoder's alpha test reference, which instantiation reads from the state.
                PushField::AlphaTestReference if stage == Stage::Fragment => {}
                // The encoder's blend color, which instantiation reads from the state.
                PushField::HalfBlendColor if stage == Stage::Fragment => {}
                // The storage form's kept channels, which instantiation reads from the template.
                PushField::StorageKeptChannels if stage == Stage::Fragment => {}
                // A draw of one amplification's id and offsets, which instantiation writes before
                // each draw; the placement order puts them right after the addresses, so the window
                // holds them.
                PushField::AmplificationId
                | PushField::AmplificationLayerOffset
                | PushField::AmplificationViewportOffset
                    if matches!(field.loc, PushLoc::Push { .. }) => {}
                // An emulated tessellation draw's chunk, which instantiation writes before each
                // chunk's draw, in the window as the amplification fields are; and its index
                // buffers' fields, which it writes from the draw.
                PushField::TessellationChunk
                    if stage == Stage::Vertex
                        && emulated_tessellation.is_some()
                        && matches!(field.loc, PushLoc::Push { .. }) => {}
                PushField::PatchIndexOffset
                | PushField::PatchIndexBound
                | PushField::ControlPointIndexOffset
                    if stage == Stage::Vertex && emulated_tessellation.is_some() => {}
                PushField::VertexBufferStride(layout) if stage == Stage::Vertex => {
                    match pulled_stride(layout) {
                        Some(VertexStride::Static(_)) => {}
                        Some(VertexStride::PerBind) => {
                            // A buffer's or inline bytes' stride is its bind's, 0 without one;
                            // inline bytes at `MTLAttributeStrideStatic` are not emulated.
                            match stage_shape(Stage::Vertex).and_then(|s| s.buffers.get(&layout)) {
                                Some(BufferShape::Buffer { .. }) => {}
                                Some(&BufferShape::Bytes { attribute_stride }) => {
                                    bytes_stride(attribute_stride, vertex_buffer(layout))?;
                                }
                                _ => {
                                    return Err(internal(format!(
                                        "the stride of vertex buffer {layout} bound by nothing"
                                    )))
                                }
                            }
                        }
                        None => {
                            return Err(internal(format!(
                                "the stride of vertex buffer {layout} without a pulled layout"
                            )))
                        }
                    }
                }
                other => {
                    return Err(internal(format!(
                        "a {stage:?} push field {other:?} in a render pipeline"
                    )))
                }
            }
            if hole && !matches!(field.loc, PushLoc::Push { .. }) {
                // The placement order puts the addresses first, so a stage window always holds
                // them.
                return Err(not_implemented(Feature::Lowering(LowerFamily::StageIo)));
            }
            push.fields.push(PushFieldPlacement {
                stage,
                field: field.field,
                location: match field.loc {
                    PushLoc::Push { offset } => FieldLocation::PushWindow { offset },
                    PushLoc::ArgumentTable { offset } => FieldLocation::ArgumentTable { offset },
                },
                size: field.size,
            });
        }
        let root_first = reads_root_table.then(|| {
            let top = buffers
                .iter()
                .filter(|b| b.slot.stage == stage && b.form == BufferForm::Psb)
                .map(|b| b.slot.index + 1)
                .max()
                .unwrap_or(0);
            2 + top
        });
        templates.push(StageTemplate {
            stage,
            flags: vk_of(stage).flags(),
            fields,
            push_range: placement.and_then(|pl| pl.push_range),
            table: None,
            table_size: placement.map_or(0, |pl| pl.argument_table_size),
            root_first,
        });
    }

    // Descriptors.
    let mut resources = ResourcePlanner {
        profile,
        shape: &shape.stages,
        slots: &p.interface.slots,
        dependencies: &p.dependencies,
        stage_flags,
        variant: Default::default(),
        diagnostics: Vec::new(),
        uses: Vec::new(),
        template: Default::default(),
        alpha_red_texels: Default::default(),
    };
    let heap_reach = |s: metal::RenderStages| {
        let mut out = Vec::new();
        for (flag, stage) in [
            (metal::RenderStages::VERTEX, Stage::Vertex),
            (metal::RenderStages::FRAGMENT, Stage::Fragment),
            (metal::RenderStages::OBJECT, Stage::Object),
            (metal::RenderStages::MESH, Stage::Mesh),
        ] {
            if s.contains(flag) && stages.contains(&stage) {
                out.push(stage);
            }
        }
        out
    };
    let mut heap = HeapPlanner::new(profile, &p.layout, &p.dependencies, shape, heap_reach)?;
    let mut descriptors = Vec::new();
    let separate = separate_samplers(&p.layout);
    let mut factor_binding = None;
    let mut patch_index_binding = None;
    let mut control_point_index_binding = None;
    for set in &p.layout.sets {
        for b in &set.bindings {
            if let BindingSource::Heap(_) = b.source {
                descriptors.extend(planner(&mut heap)?.array(set.set, b)?);
                continue;
            }
            // A separate sampler: a constexpr one is the layout's immutable sampler, no write; a
            // runtime one the sampler bound at its slot, made for the use.
            if let BindingSource::SeparateSampler { stage, sampler } = b.source {
                let source =
                    resources.separate_sampler(stage, sampler, b.immutable_sampler.as_ref())?;
                if let Some(source) = source {
                    descriptors.push(DescriptorWrite {
                        set: set.set,
                        binding: b.binding,
                        array_element: 0,
                        ty: b.ty,
                        source,
                    });
                }
                continue;
            }
            if let BindingSource::HeapSampler { stage, sampler } = b.source {
                // A constexpr heap sampler is the layout's immutable sampler: no write. A runtime
                // one is the sampler bound at its slot, made for the use.
                let runtime =
                    planner(&mut heap)?.sampler(stage, sampler, b.immutable_sampler.as_ref())?;
                if let Some(slot) = runtime {
                    descriptors.push(DescriptorWrite {
                        set: set.set,
                        binding: b.binding,
                        array_element: 0,
                        ty: b.ty,
                        source: resources.heap_sampler(slot, sampler)?,
                    });
                }
                continue;
            }
            // A storage slot's level views.
            if let (0, BindingSource::Texture(slot)) = (set.set, b.source) {
                if b.count > 1 {
                    let views = resources.level_views(set.set, b.binding, b.ty, slot, b.count)?;
                    descriptors.extend(views.into_iter().zip(0..).map(|(source, i)| {
                        DescriptorWrite {
                            set: set.set,
                            binding: b.binding,
                            array_element: i,
                            ty: b.ty,
                            source,
                        }
                    }));
                    continue;
                }
            }
            if set.set != 0 || b.count != 1 {
                return Err(internal(format!(
                    "a {:?} binding of {} descriptors in set {}",
                    b.source, b.count, set.set
                )));
            }
            let source = match b.source {
                BindingSource::Buffer(slot) => {
                    let pb = planned(slot).ok_or_else(|| {
                        internal(format!("binding of {slot:?} without an interface slot"))
                    })?;
                    match pb.bound {
                        BoundKind::Buffer { .. } => WriteSource::Buffer { slot },
                        BoundKind::Bytes => WriteSource::InlineBytes { slot },
                        BoundKind::Unbound => WriteSource::Neutral(NeutralResource::ZeroBuffer {
                            min_bytes: zero_buffer_bytes(profile),
                        }),
                    }
                }
                BindingSource::Texture(slot) => {
                    resources.texture(set.set, b.binding, b.ty, slot)?
                }
                BindingSource::Sampler(slot) => resources.sampler(slot)?,
                BindingSource::CombinedTextureSampler { texture, sampler } => {
                    resources.combined(texture, sampler, b.immutable_sampler.as_ref())?
                }
                BindingSource::SeparateTexture { texture, sampler } => {
                    let immutable = separate.get(&(texture.stage, sampler)).ok_or_else(|| {
                        internal(format!(
                            "separate texture {texture:?} without its sampler {sampler:?}"
                        ))
                    })?;
                    resources.separate_texture(texture, sampler, *immutable)?
                }
                BindingSource::InputAttachment(color_index) => {
                    WriteSource::Attachment { color_index }
                }
                BindingSource::Synthetic { stage, binding } => match binding {
                    SyntheticBinding::ArgumentTable => {
                        let t = templates
                            .iter_mut()
                            .find(|t| t.stage == stage)
                            .ok_or_else(|| internal(format!("an ArgumentTable of {stage:?}")))?;
                        if t.table_size == 0 {
                            return Err(internal("an ArgumentTable binding without fields"));
                        }
                        t.table = Some((set.set, b.binding));
                        WriteSource::ArgumentTable {
                            stage,
                            bytes_len: t.table_size,
                        }
                    }
                    // Each stage binds its own blob.
                    SyntheticBinding::ConstantBlob
                        if p.stages
                            .iter()
                            .any(|m| m.stage == vk_of(stage) && m.constant_blob.is_some()) =>
                    {
                        WriteSource::ConstantBlob { stage }
                    }
                    // The control stage's factor buffer, or the zero buffer when the encoder
                    // set none: every patch reads zero factors and is culled.
                    SyntheticBinding::TessellationFactors if p.graphics.tessellation.is_some() => {
                        if shape.tessellation_factor_buffer {
                            factor_binding = Some((set.set, b.binding));
                            WriteSource::TessellationFactors
                        } else {
                            WriteSource::Neutral(NeutralResource::ZeroBuffer {
                                min_bytes: zero_buffer_bytes(profile),
                            })
                        }
                    }
                    // The draw's patch index buffer, or the zero buffer for a draw without one.
                    SyntheticBinding::PatchIndices if tessellation => {
                        patch_index_binding = Some((set.set, b.binding));
                        WriteSource::PatchIndices {
                            min_bytes: zero_buffer_bytes(profile),
                        }
                    }
                    // The draw's control-point index buffer, of a pipeline with a control-point
                    // index type.
                    SyntheticBinding::ControlPointIndices if tessellation => {
                        control_point_index_binding = Some((set.set, b.binding));
                        WriteSource::ControlPointIndices
                    }
                    // An emulated mesh function's records and work buffer, each draw's own.
                    SyntheticBinding::MeshRecords if records.is_some() => WriteSource::MeshRecords,
                    SyntheticBinding::MeshWork if records.is_some() => WriteSource::MeshWork,
                    // An emulated tessellation draw's work buffer and records, each its own.
                    SyntheticBinding::TessellationWork if emulated_tessellation.is_some() => {
                        WriteSource::TessellationWork
                    }
                    SyntheticBinding::TessellationRecords if emulated_tessellation.is_some() => {
                        WriteSource::TessellationRecords
                    }
                    SyntheticBinding::HandleTable => {
                        planner(&mut heap)?.table(stage, set.set, b.binding)?
                    }
                    SyntheticBinding::SamplerTable => {
                        planner(&mut heap)?.sampler_table(stage, set.set, b.binding)?
                    }
                    SyntheticBinding::FunctionTableDirectory => {
                        planner(&mut heap)?.directory(stage, set.set, b.binding)?
                    }
                    // The tables' words on a device without the array's features.
                    SyntheticBinding::FunctionTableSnapshot => {
                        planner(&mut heap)?.snapshot(stage, set.set, b.binding)?
                    }
                    // The pass's imageblock storage.
                    SyntheticBinding::Imageblock
                        if stage == Stage::Fragment && p.graphics.fragment_imageblock.is_some() =>
                    {
                        WriteSource::Imageblock
                    }
                    // The pass's tile threadgroup memory.
                    SyntheticBinding::TileMemory if stage == Stage::Fragment => {
                        match &tile_memory {
                            Some(memory) => WriteSource::TileMemory {
                                bytes: memory.bytes(),
                            },
                            None => {
                                return Err(internal("a tile memory binding without its facts"))
                            }
                        }
                    }
                    // The stencil's shadow, or, in a pass without a stencil attachment, a neutral
                    // the function never reads (its state says nothing is attached), of the
                    // pipeline's samples as the function's declaration is.
                    SyntheticBinding::ShadowStencil if stage == Stage::Fragment => {
                        match &p.graphics.shadow_stencil {
                            Some(_) => WriteSource::ShadowStencil,
                            None => WriteSource::Neutral(NeutralResource::Image {
                                view_type: vk::ImageViewType::TYPE_2D,
                                sampled_type: crate::pipeline::sym::SampledType::Uint,
                                depth_compare: false,
                                samples: p.graphics.rasterization_samples.as_raw().max(1),
                                storage_format: Some(crate::lower::fragment::SHADOW_FORMAT),
                            }),
                        }
                    }
                    // The depth attachment's stand-in: a plan asks for the variant only where the
                    // pipeline has a depth attachment, whose pipeline states its stand-in.
                    SyntheticBinding::DepthStandIn if stage == Stage::Fragment => {
                        match &p.graphics.shader_tests {
                            Some(_) => WriteSource::DepthStandIn,
                            None => {
                                return Err(internal("a depth stand-in without a depth attachment"))
                            }
                        }
                    }
                    // The stencil attachment's stand-in: a plan asks for the variant only where
                    // the pipeline has a depth and a stencil attachment.
                    SyntheticBinding::StencilStandIn if stage == Stage::Fragment => {
                        match p.graphics.shader_tests.and_then(|t| t.stencil) {
                            Some(_) => WriteSource::StencilStandIn,
                            None => {
                                return Err(internal(
                                    "a stencil stand-in without a stencil attachment",
                                ))
                            }
                        }
                    }
                    // The command buffer's status, on a device that does not address buffers.
                    SyntheticBinding::ExecutionStatus => WriteSource::ExecutionStatus,
                    other => return Err(internal(format!("a render binding {other:?}"))),
                },
                BindingSource::Heap(_) | BindingSource::HeapSampler { .. } => {
                    return Err(internal("a heap binding outside the heap planner"))
                }
                BindingSource::SeparateSampler { .. } => {
                    return Err(internal("a separate sampler outside its planner"))
                }
                // The consumer builds the structures (`bind::acceleration`).
                BindingSource::AccelerationStructure { slot, level } => {
                    WriteSource::AccelerationStructure { slot, level }
                }
                // An attachment the fragment function reads in the interlocked storage form, a
                // storage view of the attachment, or of its sample planes, in its own format.
                BindingSource::TileAttachment(color_index) => {
                    let a = p
                        .graphics
                        .storage_attachments
                        .iter()
                        .find(|a| a.color_index == color_index)
                        .ok_or_else(|| {
                            internal(format!(
                                "a storage binding of color attachment {color_index}, which the \
                                 pipeline does not read in the storage form"
                            ))
                        })?;
                    WriteSource::TileAttachment {
                        color_index,
                        format: a.format,
                        view: a.view,
                        view_alias: a.view_alias,
                    }
                }
                // A sparse texture's tile map.
                BindingSource::TileMap(slot) => resources.tile_map(slot),
                // The `half` shadow of an extended-range attachment the draw's instance holds.
                BindingSource::ExtendedShadow(color_index) => {
                    WriteSource::ExtendedShadow { color_index }
                }
                // A texture's snapshot, which the draw reads the texture through.
                BindingSource::SnapshotTexels(slot) => WriteSource::SnapshotTexels { slot },
                // A two-plane YCbCr texture's planes image, through the device's conversion.
                BindingSource::YCbCrPlanes(slot) => resources.planes_image_write(slot)?,
            };
            descriptors.push(DescriptorWrite {
                set: set.set,
                binding: b.binding,
                array_element: 0,
                ty: b.ty,
                source,
            });
        }
    }

    // Vertex buffers (fixed function), and the pulled-vertex-buffer variant: a layout the shape
    // binds where fixed-function input cannot fetch it is pulled in the variant, and a pipeline
    // compiled under such a variant keeps pulling its layouts.
    let vertex_strides = StrideRules {
        max: limits.max_vertex_input_binding_stride,
        portability_alignment: limits.min_vertex_input_binding_stride_alignment,
        dynamic: p
            .graphics
            .dynamic_states
            .contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE),
        within_stride: profile
            .fields()
            .enabled
            .portability_withholds(VkFeature::VertexAttributeAccessBeyondStride),
    };
    let vs = stage_shape(Stage::Vertex);
    let mut vertex_buffers = Vec::new();
    let mut planned_vertex = Vec::new();
    let mut pulled_vertex_buffers = p.vertex_input.variant_pulled.clone();
    for vb in &p.vertex_input.bindings {
        let slot = vertex_buffer(vb.buffer_index);
        let (alignment, reads) = attribute_reads(&p.vertex_input.attributes, vb.binding)?;
        if let VertexStride::Static(s) = vb.stride {
            // Link pulls what a static stride under dynamic state does not cover (06209).
            if vertex_strides.dynamic && s != 0 && s < reads.1 {
                return Err(internal(format!(
                    "static stride {s} of vertex binding {} under dynamic strides",
                    vb.binding
                )));
            }
        }
        let planned = PlannedVertexBuffer {
            binding: vb.binding,
            slot,
            stride: vb.stride,
            input_rate: vb.input_rate,
            divisor: vb.divisor,
            alignment,
            reads,
        };
        let fetchable = match (vs.and_then(|s| s.buffers.get(&vb.buffer_index)), vb.stride) {
            (None, _) => return Err(BindError::UnboundVertexBuffer { slot }),
            // The consumer places inline bytes at an offset of its own (aligned); a per-bind
            // layout reads them at their bind's stride, 0 without one, as a buffer's.
            (Some(&BufferShape::Bytes { attribute_stride }), stride) => {
                let per_bind = match stride {
                    VertexStride::PerBind => Some(bytes_stride(attribute_stride, slot)?),
                    VertexStride::Static(_) => None,
                };
                fetchable(
                    &planned,
                    &vertex_strides,
                    metal::vertex::MAX_VERTEX_ALIGNMENT,
                    per_bind,
                )
            }
            (
                Some(&BufferShape::Buffer {
                    attribute_stride,
                    offset_alignment,
                    ..
                }),
                stride,
            ) => {
                // A buffer bound without a stride reads a per-bind layout at stride 0.
                let per_bind = match stride {
                    VertexStride::PerBind => Some(attribute_stride.unwrap_or(0)),
                    VertexStride::Static(_) => None,
                };
                fetchable(&planned, &vertex_strides, offset_alignment, per_bind)
            }
        };
        if !fetchable {
            pulled_vertex_buffers.insert(vb.buffer_index);
        }
        vertex_buffers.push(VertexBufferBind {
            binding: vb.binding,
            buffer_index: vb.buffer_index,
            per_bind_stride: vb.stride == VertexStride::PerBind,
        });
        planned_vertex.push(planned);
    }

    // Specialization the shape decides, over both stages.
    let mut layout: Vec<crate::pipeline::artifact::SpecConstant> = p
        .stages
        .iter()
        .flat_map(|m| m.specialization.constants.iter().copied())
        .collect();
    layout.sort_by_key(|c| c.id);
    layout.dedup_by_key(|c| c.id);
    let samples = u64::from(p.graphics.rasterization_samples.as_raw().max(1));
    check_sparse_textures(&p.dependencies, profile, shape)?;
    // An object function's threadgroup memory arguments, whose lengths the encoder sets
    // (`setObjectThreadgroupMemoryLength:atIndex:`): its arena's facts, and the total of its
    // specialized lengths.
    let object_arena = p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::ObjectThreadgroupArena {
            static_bytes,
            payload,
        } => Some((static_bytes, payload)),
        _ => None,
    });
    if mesh && other_stage_lengths(p, records.is_some()) {
        // A mesh function's arguments, whose lengths no Metal setter sets.
        return Err(not_implemented(Feature::MeshShading));
    }
    let mut object_regions: Option<u64> = None;
    let mut fixed = Vec::new();
    for c in &layout {
        if let Some(v) = bound_value(c.key, shape, &resources)? {
            fixed.push((c.key, v));
            continue;
        }
        let v = match c.key {
            SpecKey::SampleCount => samples,
            SpecKey::AmplificationCount => u64::from(shape.vertex_amplification.count),
            // A mesh draw's threadgroup sizes and grid forms, which each draw decides
            // (`DrawCommands::specialization`).
            SpecKey::LocalSize(_)
            | SpecKey::FlatLocalSize
            | SpecKey::MultiRegion
            | SpecKey::ObjectLocalSize(_)
            | SpecKey::ObjectFlatLocalSize
            | SpecKey::ObjectMultiRegion
                if mesh =>
            {
                continue
            }
            // An object function's threadgroup memory arguments: the encoder's lengths, each
            // rounded up to 16 as a kernel's are (an unset one is 0).
            SpecKey::TgLength(n) if mesh && object_arena.is_some() => {
                let len = stage_shape(Stage::Object)
                    .and_then(|s| s.threadgroup_lengths.get(&n))
                    .map_or(0, |&len| u64::from(len))
                    .next_multiple_of(crate::memory::arena::REGION_ALIGN);
                object_regions = Some(object_regions.unwrap_or(0).saturating_add(len));
                len
            }
            SpecKey::TgLength(_) if mesh => return Err(not_implemented(Feature::MeshShading)),
            // A fragment function's tile threadgroup memory and the pass's tile.
            SpecKey::TgOffset(_)
            | SpecKey::TileThreadgroupLength
            | SpecKey::TileGridWidth
            | SpecKey::ImageblockWidth
            | SpecKey::ImageblockHeight => tile_memory
                .as_ref()
                .and_then(|m| m.spec(c.key))
                .ok_or_else(|| internal(format!("{:?} without its facts", c.key)))?,
            other => {
                return Err(internal(format!(
                    "a render specialization constant {other:?}"
                )))
            }
        };
        fixed.push((c.key, v));
    }
    // The object function's arena with those lengths, within its stage's workgroup memory limit.
    if let (Some((static_bytes, payload)), Some(regions)) = (object_arena, object_regions) {
        crate::lower::mesh::stage_workgroup_limit(
            Stage::Object,
            profile,
            payload.map(u64::from),
            0,
        )
        .check(u64::from(static_bytes).saturating_add(regions))
        .map_err(|e| match e {
            crate::error::Error::Unsupported(u) => BindError::Unsupported(u),
            other => internal(format!("an object function's threadgroup memory: {other}")),
        })?;
    }
    record_specialized(&push, shape, &mut resources)?;
    let specialization = spec_key(&layout, |k| {
        fixed.iter().find(|(key, _)| *key == k).map(|(_, v)| *v)
    });
    let m = &limits.mesh;
    // An emulated mesh function's threadgroups are the draw's compute dispatch's.
    let mesh_limits = match records {
        Some(_) => GroupLimits {
            size: limits.max_compute_work_group_size,
            invocations: limits.max_compute_work_group_invocations,
            count: limits.max_compute_work_group_count,
            total: None,
            stage: GroupStage::Compute,
        },
        None => GroupLimits {
            size: m.max_mesh_work_group_size,
            invocations: m.max_mesh_work_group_invocations,
            count: m.max_mesh_work_group_count,
            total: Some(m.max_mesh_work_group_total_count),
            stage: GroupStage::Mesh,
        },
    };
    let records = records.map(|r| RecordsTemplate {
        group_words: r.group_words(),
        group_vertices: r.primitives.saturating_mul(r.corners),
        max_range: limits.max_storage_buffer_range,
        slot_words: r.slot_words,
        row: limits.max_compute_work_group_count[0],
        rows: limits.max_compute_work_group_count[1],
        alignment: limits.min_storage_buffer_offset_alignment,
    });
    let mesh = mesh.then(|| MeshTemplate {
        multiples: p
            .dependencies
            .iter()
            .filter_map(|d| match d.fact {
                RuntimeFact::LocalSizeConstraint { multiple } => Some(multiple),
                _ => None,
            })
            .collect(),
        spec: layout,
        fixed,
        mesh: mesh_limits,
        // An emulated object function's threadgroups are a compute dispatch.
        object: stages.contains(&Stage::Object).then_some(match records {
            Some(_) => GroupLimits {
                size: limits.max_compute_work_group_size,
                invocations: limits.max_compute_work_group_invocations,
                count: limits.max_compute_work_group_count,
                total: None,
                stage: GroupStage::Compute,
            },
            None => GroupLimits {
                size: m.max_task_work_group_size,
                invocations: m.max_task_work_group_invocations,
                count: m.max_task_work_group_count,
                total: Some(m.max_task_work_group_total_count),
                stage: GroupStage::Task,
            },
        }),
        folded: p
            .dependencies
            .iter()
            .any(|d| d.fact == RuntimeFact::FoldedGroups),
        records,
    });

    // Uses: bound buffer slots, vertex buffers, textures, attachments, used resources.
    let mut uses: Vec<ResourceUse> = buffers
        .iter()
        .filter(|b| matches!(b.bound, BoundKind::Buffer { .. }))
        .map(|b| buffer_use(b, stage_flags(b.slot.stage)))
        .collect();
    // The factors: the control stage's, or an emulated pipeline's scan's.
    if factor_binding.is_some()
        || emulated_tessellation.is_some() && shape.tessellation_factor_buffer
    {
        uses.push(ResourceUse {
            resource: UseTarget::TessellationFactors,
            access: AccessSet::READ,
            class: AccessClass::Storage,
            stages: match emulated_tessellation {
                Some(_) => vk::PipelineStageFlags2::COMPUTE_SHADER,
                None => vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            },
            range: None,
        });
    }
    for vb in &planned_vertex {
        uses.push(ResourceUse {
            resource: UseTarget::Slot(vb.slot),
            access: AccessSet::READ,
            class: AccessClass::VertexInput,
            stages: vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
            range: None,
        });
    }
    uses.append(&mut resources.uses);
    let reach = |s: metal::RenderStages| {
        let mut f = vk::PipelineStageFlags2::empty();
        for (flag, stage) in [
            (metal::RenderStages::VERTEX, Stage::Vertex),
            (metal::RenderStages::FRAGMENT, Stage::Fragment),
            (metal::RenderStages::OBJECT, Stage::Object),
            (metal::RenderStages::MESH, Stage::Mesh),
        ] {
            if s.contains(flag) && stages.contains(&stage) {
                f |= stage_flags(stage);
            }
        }
        f
    };
    // The stages whose function may write memory: a vertex or fragment function the compile
    // found storing nothing writes no used resource (`GraphicsState::vertex_writes_memory`,
    // `fragment_writes_memory`); an object or mesh function has no such fact.
    let writers = stages
        .iter()
        .filter(|stage| match stage {
            Stage::Vertex => p.graphics.vertex_writes_memory,
            Stage::Fragment => p.graphics.fragment_writes_memory,
            _ => true,
        })
        .fold(vk::PipelineStageFlags2::empty(), |all, &stage| {
            all | stage_flags(stage)
        });
    uses.extend(used_uses(
        &shape.used,
        p.interface.needs_pointer_translation,
        p.interface.needs_handle_translation,
        reach,
        writers,
    ));

    let mut diagnostics = resources.diagnostics;
    let mut template = resources.template;
    if let Some(h) = heap {
        h.finish(
            &mut descriptors,
            &mut uses,
            stage_flags,
            &mut template,
            &mut diagnostics,
        );
    }
    let mut variant = resources.variant;
    variant.psb_slots = psb_variant(&p.dependencies, &buffers, profile);
    variant.storage_view_slots = storage_view_variant(&p.dependencies, &buffers, profile);
    variant.cross_class = cross_variant(&p.dependencies, &p.interface.slots, shape);
    variant.srgb_textures = srgb_variant(profile, &p.dependencies, shape, &variant.cross_class);
    variant.ycbcr_in_place = ycbcr_in_place_variant(&p.dependencies, shape);
    variant.ycbcr_planes_sampled =
        ycbcr_planes_sampled_variant(&p.dependencies, shape, &variant.ycbcr_in_place);
    variant.address_maps = address_map_variant(&p.dependencies, shape);
    variant.cubic_samplers = cubic_variant(&p.dependencies, shape)?;
    variant.anisotropic_samplers = anisotropic_variant(&p.dependencies, shape);
    // A layout the variant newly pulls is read through a `Whole` storage slot, which declares
    // `BufferRangeLimit`: the variant's pipeline would ask for its device address when the bound
    // buffer is large, so the key names it now and stays a fixed point.
    for &layout in &pulled_vertex_buffers {
        let slot = vertex_buffer(layout);
        let bound = BoundKind::of(vs.and_then(|s| s.buffers.get(&layout)));
        if !p.vertex_input.variant_pulled.contains(&layout) && range_variant(slot, bound, profile) {
            variant.psb_slots.insert(slot);
        }
    }
    // A layout the variant newly pulls adds a storage buffer to the vertex stage, whose fallback
    // may then view any of its candidates as a uniform buffer: the key keeps every large one a
    // storage view now, so the variant's plan states the same key.
    if pulled_vertex_buffers
        .iter()
        .any(|l| !p.vertex_input.variant_pulled.contains(l))
    {
        variant.storage_view_slots.extend(large_view_candidates(
            &p.dependencies,
            &buffers,
            profile,
            Some(Stage::Vertex),
        ));
    }
    variant.pulled_vertex_buffers = pulled_vertex_buffers;
    variant.dropped_vertex_layer = dropped_vertex_layer;
    variant.amplification_draws = amplification_draws;
    variant.viewport_compensation = viewport_compensation;
    // A pipeline compiled for a draw replay's bindings states its variant.
    variant.replay_bindings = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::ReplayBindings);
    variant.merged_draws = p
        .dependencies
        .iter()
        .any(|d| d.fact == RuntimeFact::MergedDraws);
    // The pipeline compiled under other zeroed or dropped colors (or none) is the other variant.
    let other_colors = compiled_zeroed != zeroed_colors
        || compiled_dropped != dropped_colors
        || compiled_interlocked != interlocked_colors
        || compiled_shadows != extended_shadows
        || compiled_snapshots != snapshot_textures;
    variant.zeroed_colors = zeroed_colors;
    variant.dropped_colors = dropped_colors;
    variant.interlocked_colors = interlocked_colors;
    variant.extended_shadows = extended_shadows;
    variant.snapshot_textures = snapshot_forms(
        profile,
        &snapshot_textures,
        &p.interface.slots,
        shape,
        &variant.cross_class,
    );
    variant.depth_clamp_range = depth_clamp_range;
    variant.rotated_weights = rotated_weights;
    variant.mesh_last_first = mesh_last_first;
    variant.read_only_tests = read_only_tests;
    variant.shader_tests = shader_tests;
    variant.shader_stencil = shader_stencil;
    variant.window_depth_varying = window_depth_varying;
    variant.primitive_id_indices = primitive_id_indices;
    variant.quad_primitive_ids = quad_primitive_ids;
    variant.private_list = private_list;
    // Tables that may pass `maxStorageBufferRange`: the variant that reads them at their
    // addresses (and a pipeline compiled so states it).
    variant.addressed_function_tables = crate::bind::heap::addressed_function_tables(
        profile,
        &p.dependencies,
        &shape.function_tables,
    );
    Ok(BindingPlan {
        variant,
        specialization,
        descriptors,
        vertex_buffers,
        uses,
        push,
        diagnostics,
        compute: None,
        graphics: Some(GraphicsTemplate {
            buffers,
            stages: templates,
            vertex_buffers: planned_vertex,
            pulled_strides,
            resources: template,
            vertex_strides,
            multiview_instance_limit: (views.view_mask != 0)
                .then_some(limits.max_multiview_instance_index),
            other_layer_variant: vertex_layer
                .is_some_and(|dropped| dropped != dropped_vertex_layer)
                || compiled_draws != amplification_draws
                || compiled_compensation != viewport_compensation
                || other_colors
                || compiled_range != depth_clamp_range
                || compiled_rotated != rotated_weights
                || compiled_last_first != mesh_last_first
                || compiled_tests != read_only_tests
                || compiled_shader_tests != shader_tests
                || compiled_shader_stencil != shader_stencil
                || compiled_quads != quad_primitive_ids
                || compiled_list != private_list,
            amplifications,
            indirect_first_instance_missing: !profile
                .fields()
                .enabled
                .features
                .contains(&VkFeature::DrawIndirectFirstInstance),
            robust_index_fetch: profile
                .fields()
                .enabled
                .features
                .contains(&VkFeature::RobustBufferAccess2),
            tessellation: factor_template(
                p,
                (
                    factor_binding,
                    patch_index_binding,
                    control_point_index_binding,
                ),
                (
                    limits.min_storage_buffer_offset_alignment,
                    u64::from(limits.max_storage_buffer_range),
                ),
                shape.tessellation_factor_buffer,
            )?,
            shadow_stencil: shadow,
            mesh,
            shader_tests: p
                .graphics
                .shader_tests
                .is_some()
                .then_some(ShaderTestsTemplate),
            deindexed: primitive_id_indices,
            private_list,
            storage_kept: storage_kept(p)?,
        }),
        views,
    })
}

/// `PushField::StorageKeptChannels` of `p`: each storage-form color attachment's kept channels
/// (`StorageAttachmentUse::kept`, Vulkan's `R`, `G`, `B` and `A` bits) at bits `4c` to `4c + 3`
/// for color `c`.
fn storage_kept(p: &RenderPipeline) -> Result<u32, BindError> {
    p.graphics
        .storage_attachments
        .iter()
        .try_fold(0u32, |word, s| match s.color_index {
            c @ 0..8 => Ok(word | ((s.kept.as_raw() & 0xF) << (4 * c))),
            _ => Err(internal("a storage-form color attachment past the eighth")),
        })
}

/// The shader-side stencil of `p`'s fragment function, when it has one.
fn shadow_template(p: &RenderPipeline) -> Option<ShadowTemplate> {
    let has = p.interface.push.iter().any(|pl| {
        pl.fields
            .iter()
            .any(|f| f.field == PushField::ShadowStencilState)
    });
    has.then(|| ShadowTemplate {
        attached: p.graphics.shadow_stencil.is_some(),
        depth_attachment: p.graphics.depth_format.is_some(),
    })
}

/// The `PushField::ShadowStencilState` words of a draw under encoder state `state`: the front
/// face's, then the back face's (`lower::fragment::face_word`). A state whose stencil write could
/// depend on the depth test is refused as needing stencil export: the pass has a depth attachment,
/// the depth comparison is not `Always`, and a face's depth-failure operation differs from its pass
/// operation (the depth test runs after the function).
fn shadow_stencil_words(t: &ShadowTemplate, state: &BindState) -> Result<FieldBytes, BindError> {
    let ds = &state.depth_stencil;
    let faces = [&ds.front_face_stencil, &ds.back_face_stencil];
    if t.attached
        && t.depth_attachment
        && ds.depth_compare_function != metal::CompareFunction::Always
        && faces
            .iter()
            .any(|f| f.depth_failure_operation != f.depth_stencil_pass_operation)
    {
        return Err(BindError::Unsupported(Unsupported::needs(
            Feature::StencilExport,
            Capability::Extension(crate::device::requirements::VkExtension::ExtShaderStencilExport),
        )));
    }
    Ok(FieldBytes::words(faces.map(|f| {
        crate::lower::fragment::face_word(
            f.stencil_compare_function.as_raw(),
            f.stencil_failure_operation.as_raw(),
            f.depth_stencil_pass_operation.as_raw(),
            f.read_mask,
            f.write_mask,
            t.attached,
        )
    })))
}

/// The `PushField::ShaderTestsState` words of a draw under encoder state `state`: the depth compare
/// function, write enable and clip mode, and viewport 0's depth range.
/// The 8-bit unorm code Metal's private alpha test compares the reference `reference` as
/// (`PushField::AlphaTestReference`, probes on an Apple M3 Max): NaN is 0, the value is clamped
/// to `[0, 1]`, and the code is the exact product with 255 rounded to nearest, as the fragment
/// function converts its alpha (`lower::arith::color_write::unorm8_code`). The product of an
/// `f32` and 255 is exact as an `f64`, and its only tie, at 0.5, rounds to 128 either way.
pub(crate) fn alpha_test_code(reference: f32) -> u32 {
    let clamped = match reference.is_nan() {
        true => 0.0,
        false => f64::from(reference.clamp(0.0, 1.0)),
    };
    (clamped * 255.0).round_ties_even() as u32
}

fn shader_tests_words(state: &BindState) -> FieldBytes {
    let ds = &state.depth_stencil;
    let clamp = state.depth_clip_mode_clamp;
    let word = crate::lower::fragment::depth_test_word(
        ds.depth_compare_function.as_raw(),
        ds.depth_write_enabled,
        clamp,
    );
    let [near, far] = state.depth_range;
    FieldBytes::words([word, near.to_bits(), far.to_bits(), 0])
}

/// The `PushField::ShaderStencilState` words of a draw under encoder state `state`: each face's
/// stencil word (`lower::fragment::stencil_test_word`), the front then the back, the references'
/// low 8 bits (front in bits 0 to 7, back in bits 8 to 15), and a zero word.
fn shader_stencil_words(state: &BindState) -> FieldBytes {
    let ds = &state.depth_stencil;
    let [front_face, back_face] = [&ds.front_face_stencil, &ds.back_face_stencil].map(|f| {
        crate::lower::fragment::stencil_test_word(
            f.stencil_compare_function.as_raw(),
            f.stencil_failure_operation.as_raw(),
            f.depth_failure_operation.as_raw(),
            f.depth_stencil_pass_operation.as_raw(),
            f.read_mask,
            f.write_mask,
        )
    });
    let [front, back] = state.stencil_references;
    let references = (front & 0xFF) | (back & 0xFF) << 8;
    FieldBytes::words([front_face, back_face, references, 0])
}

/// The address alignment and the element bytes the fixed-function attributes of `binding` read
/// ([`PlannedVertexBuffer::alignment`] and `reads`): each attribute's format block gives its
/// Vulkan alignment (the component size, or the whole word of a packed format) and size.
fn attribute_reads(
    attributes: &[VertexAttribute],
    binding: u32,
) -> Result<(u32, (u32, u32)), BindError> {
    let mut alignment = 1;
    let mut reads: Option<(u32, u32)> = None;
    for a in attributes.iter().filter(|a| a.binding == binding) {
        let block = a
            .format
            .block()
            .ok_or_else(|| internal(format!("a vertex attribute of {:?}", a.format)))?;
        let align = block
            .vertex_alignment()
            .ok_or_else(|| internal(format!("a vertex attribute of {:?}", a.format)))?;
        alignment = alignment.max(align);
        let end = a
            .offset
            .checked_add(block.size)
            .ok_or_else(|| internal("a vertex attribute past 2^32 bytes"))?;
        reads = Some(match reads {
            Some((lo, hi)) => (lo.min(a.offset), hi.max(end)),
            None => (a.offset, end),
        });
    }
    let reads =
        reads.ok_or_else(|| internal(format!("vertex binding {binding} without attributes")))?;
    Ok((alignment, reads))
}

impl BindingPlan {
    /// The commands of one draw of `state`. `state` must have the shape the plan was made for; the
    /// result depends on its offsets, lengths, strides, and addresses and on the draw, never on the
    /// contents of an indirect buffer.
    pub fn instantiate_draw(
        &self,
        state: &BindState,
        draw: &DrawCall,
    ) -> Result<DrawCommands, BindError> {
        let mut commands = DrawCommands::default();
        self.instantiate_draw_into(state, draw, &mut commands)?;
        Ok(commands)
    }

    /// [`instantiate_draw`](Self::instantiate_draw)'s commands, written over `commands`: every
    /// field is set as `instantiate_draw` sets it, and the vectors (each push write's bytes and
    /// each table's among them) keep the storage the commands written there before left, so a
    /// consumer that keeps one `DrawCommands` from draw to draw allocates none of it again once
    /// it holds as much. On an error `commands` holds no draw's commands.
    pub fn instantiate_draw_into(
        &self,
        state: &BindState,
        draw: &DrawCall,
        commands: &mut DrawCommands,
    ) -> Result<(), BindError> {
        let t = self
            .graphics
            .as_ref()
            .ok_or_else(|| internal("instantiate_draw of a plan that is not a render plan"))?;
        if t.other_layer_variant {
            return Err(BindError::NeedsVariant {
                variant: Box::new(self.variant.clone()),
            });
        }
        // An indexed draw of a pipeline that writes the primitive ID runs on its de-indexed
        // variant of the draw's index type, and every other draw on the variant without it.
        if t.deindexed.is_some() {
            let asked = draw_index_type(draw);
            if asked != t.deindexed {
                return Err(BindError::NeedsVariant {
                    variant: Box::new(VariantKey {
                        primitive_id_indices: asked,
                        ..self.variant.clone()
                    }),
                });
            }
        }
        instantiate(t, state, draw, commands)
    }
}

/// The Vulkan draw of a Metal draw, and the base vertex and first instance the
/// `BaseVertexInstance` field reports (Vulkan's `BaseVertex` and `BaseInstance`). A direct
/// indexed draw whose indices run past its index buffer ([`index_overrun`]) draws as `overrun`
/// states: a `Clamped` one draws only its indices inside the buffer. A draw of no index binds the
/// buffer at offset 0 where its own offset may lie past the `VkBuffer`, which cannot be bound
/// there (`VUID-vkCmdBindIndexBuffer-offset-08782`): when it is a `Clamped` draw of none, or its
/// offset is at or past the buffer's length (a draw of no index has no overrun to record).
fn vk_draw(draw: &DrawCall, overrun: Option<IndexOverrun>) -> (vk::DrawCommand, [u32; 2]) {
    match *draw {
        DrawCall::Primitives {
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => (
            vk::DrawCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex: vertex_start,
                first_instance: base_instance,
            },
            [vertex_start, base_instance],
        ),
        DrawCall::Indexed {
            index_type,
            index_buffer,
            index_count,
            index_offset,
            instance_count,
            base_vertex,
            base_instance,
        } => {
            let clamped = matches!(overrun, Some(IndexOverrun::Clamped { .. }));
            let index_count = match overrun {
                Some(IndexOverrun::Clamped { past_end }) => index_count.saturating_sub(past_end),
                Some(IndexOverrun::Robust { .. }) | None => index_count,
            };
            let unbindable = clamped || index_offset >= index_buffer.length;
            let index_buffer_offset = if index_count == 0 && unbindable {
                0
            } else {
                index_offset
            };
            (
                vk::DrawCommand::DrawIndexed {
                    index_type,
                    index_buffer_offset,
                    index_count,
                    instance_count,
                    first_index: 0,
                    vertex_offset: base_vertex,
                    first_instance: base_instance,
                },
                [base_vertex as u32, base_instance],
            )
        }
        DrawCall::Indirect {
            indirect,
            indexed: None,
        }
        | DrawCall::PatchesIndirect { indirect, .. } => (
            vk::DrawCommand::DrawIndirect {
                offset: indirect.offset,
                draw_count: 1,
                stride: 16,
            },
            [0, 0],
        ),
        // One vertex per patch: the synthesized vertex stage's `VertexIndex` is Metal's patch
        // index.
        DrawCall::Patches {
            patch_start,
            patch_count,
            instance_count,
            base_instance,
            ..
        } => (
            vk::DrawCommand::Draw {
                vertex_count: patch_count,
                instance_count,
                first_vertex: patch_start,
                first_instance: base_instance,
            },
            [patch_start, base_instance],
        ),
        DrawCall::Indirect {
            indirect,
            indexed: Some((index_type, _, index_offset)),
        } => (
            vk::DrawCommand::DrawIndexedIndirect {
                index_type,
                index_buffer_offset: index_offset,
                offset: indirect.offset,
                draw_count: 1,
                stride: 20,
            },
            [0, 0],
        ),
        DrawCall::MeshThreadgroups {
            groups,
            threads_per_mesh_group,
            ..
        } => (mesh_tasks(groups, threads_per_mesh_group), [0, 0]),
        // Metal launches the whole mesh threadgroups the threads fill.
        DrawCall::MeshThreads {
            threads,
            threads_per_mesh_group: l,
            ..
        } => (
            mesh_tasks(
                std::array::from_fn(|i| threads[i].checked_div(l[i]).unwrap_or(0)),
                l,
            ),
            [0, 0],
        ),
        // Plain Metal draws nothing from arguments at an offset that is not a multiple of 4, where
        // Vulkan's offset must be one too.
        DrawCall::MeshThreadgroupsIndirect {
            indirect,
            threads_per_mesh_group,
            ..
        } if indirect.offset.is_multiple_of(4) && !mesh_draws_nothing(threads_per_mesh_group) => (
            vk::DrawCommand::DrawMeshTasksIndirect {
                offset: indirect.offset,
                draw_count: 1,
                stride: 12,
            },
            [0, 0],
        ),
        DrawCall::MeshThreadgroupsIndirect { .. } => (mesh_tasks([0; 3], [0; 3]), [0, 0]),
    }
}

/// Metal's most threads of a mesh threadgroup (`maxTotalThreadsPerMeshThreadgroup`).
const METAL_MESH_THREADS: u64 = 1024;

/// Whether Metal draws nothing for a mesh threadgroup of `l` threads: one of no thread, or of
/// more than [`METAL_MESH_THREADS`], past which plain Metal draws nothing (its debug layer stops
/// the draw).
fn mesh_draws_nothing(l: [u32; 3]) -> bool {
    let n: u64 = l.iter().map(|&x| u64::from(x)).product();
    n == 0 || n > METAL_MESH_THREADS
}

/// `vkCmdDrawMeshTasksEXT` of `groups` mesh threadgroups of `l` threads: none where Metal draws
/// nothing.
fn mesh_tasks(groups: [u32; 3], l: [u32; 3]) -> vk::DrawCommand {
    let [x, y, z] = if mesh_draws_nothing(l) {
        [0; 3]
    } else {
        groups
    };
    vk::DrawCommand::DrawMeshTasks {
        group_count_x: x,
        group_count_y: y,
        group_count_z: z,
    }
}

/// A mesh draw's threads per object threadgroup and per mesh threadgroup; `None` for every other
/// draw.
fn mesh_threads(draw: &DrawCall) -> Option<[[u32; 3]; 2]> {
    match *draw {
        DrawCall::MeshThreadgroups {
            threads_per_object_group,
            threads_per_mesh_group,
            ..
        }
        | DrawCall::MeshThreadgroupsIndirect {
            threads_per_object_group,
            threads_per_mesh_group,
            ..
        }
        | DrawCall::MeshThreads {
            threads_per_object_group,
            threads_per_mesh_group,
            ..
        } => Some([threads_per_object_group, threads_per_mesh_group]),
        DrawCall::Primitives { .. }
        | DrawCall::Indexed { .. }
        | DrawCall::Indirect { .. }
        | DrawCall::Patches { .. }
        | DrawCall::PatchesIndirect { .. } => None,
    }
}

/// What a mesh draw adds to its commands: the Vulkan draw, its complete specialization, and each
/// stage's `DispatchRegion` bytes.
struct MeshDraw {
    command: vk::DrawCommand,
    specialization: SpecializationKey,
    /// The object stage's region, of a pipeline with an object function.
    object_region: Option<[u8; 32]>,
    mesh_region: [u8; 32],
    /// An emulated mesh function's dispatch.
    dispatch: Option<MeshDispatch>,
    /// A direct draw's fold of its object grid, of a folded object function
    /// (`DrawCommands::object_fold`).
    fold: Option<ObjectFold>,
    /// A direct draw whose object grid Metal's GPU discards
    /// (`DrawCommands::discards_command_buffer`).
    discards: bool,
}

impl MeshDraw {
    /// The `DispatchRegion` bytes of `stage`.
    fn region(&self, stage: Stage) -> Option<[u8; 32]> {
        match stage {
            Stage::Object => self.object_region,
            Stage::Mesh => Some(self.mesh_region),
            _ => None,
        }
    }
}

/// A refusal of a mesh draw past the device's `limit`.
fn mesh_limit(limit: VkLimit, required: u64, available: u32) -> BindError {
    BindError::Unsupported(Unsupported::needs(
        Feature::MeshShading,
        Capability::Limit {
            limit,
            required,
            available: u64::from(available),
        },
    ))
}

/// The limits that name a stage's threadgroup and grid: the task stage's for an object
/// function, the mesh stage's for a mesh function, and the compute stage's for an emulated mesh
/// function, whose grid has no total limit.
struct GroupLimitNames {
    size: fn(u8) -> VkLimit,
    invocations: VkLimit,
    count: fn(u8) -> VkLimit,
    total: Option<VkLimit>,
}

impl GroupLimitNames {
    fn of(stage: GroupStage) -> GroupLimitNames {
        match stage {
            GroupStage::Task => GroupLimitNames {
                size: VkLimit::MaxTaskWorkGroupSize,
                invocations: VkLimit::MaxTaskWorkGroupInvocations,
                count: VkLimit::MaxTaskWorkGroupCount,
                total: Some(VkLimit::MaxTaskWorkGroupTotalCount),
            },
            GroupStage::Mesh => GroupLimitNames {
                size: VkLimit::MaxMeshWorkGroupSize,
                invocations: VkLimit::MaxMeshWorkGroupInvocations,
                count: VkLimit::MaxMeshWorkGroupCount,
                total: Some(VkLimit::MaxMeshWorkGroupTotalCount),
            },
            GroupStage::Compute => GroupLimitNames {
                size: VkLimit::MaxComputeWorkGroupSize,
                invocations: VkLimit::MaxComputeWorkGroupInvocations,
                count: VkLimit::MaxComputeWorkGroupCount,
                total: None,
            },
        }
    }
}

/// The check of a threadgroup of `l` threads of a stage of `limits` (named for the object stage
/// where `object`): each axis and the product (as one axis where the stage's local size is
/// `flat`), and whole subgroups of each of `multiples` (a stage has no emulated-simdgroups
/// variant: `NotImplemented`).
fn check_group(
    limits: &GroupLimits,
    l: [u32; 3],
    flat: bool,
    multiples: &[u32],
) -> Result<(), BindError> {
    let names = GroupLimitNames::of(limits.stage);
    let n: u64 = l.iter().map(|&x| u64::from(x)).product();
    if flat {
        if n > u64::from(limits.size[0]) {
            return Err(mesh_limit((names.size)(0), n, limits.size[0]));
        }
    } else if let Some(i) = (0..3).find(|&i| l[i] > limits.size[i]) {
        return Err(mesh_limit(
            (names.size)(i as u8),
            u64::from(l[i]),
            limits.size[i],
        ));
    }
    if n > u64::from(limits.invocations) {
        return Err(mesh_limit(names.invocations, n, limits.invocations));
    }
    if multiples
        .iter()
        .any(|&m| m > 0 && !n.is_multiple_of(u64::from(m)))
    {
        return Err(not_implemented(Feature::MeshShading));
    }
    Ok(())
}

/// The check of a direct draw's grid of `g` threadgroups of a stage of `limits`: each axis and
/// the product.
fn check_grid(limits: &GroupLimits, g: [u32; 3]) -> Result<(), BindError> {
    let names = GroupLimitNames::of(limits.stage);
    if let Some(i) = (0..3).find(|&i| g[i] > limits.count[i]) {
        return Err(mesh_limit(
            (names.count)(i as u8),
            u64::from(g[i]),
            limits.count[i],
        ));
    }
    let total: u64 = g.iter().map(|&x| u64::from(x)).product();
    if let (Some(available), Some(limit)) = (limits.total, names.total) {
        if total > u64::from(available) {
            return Err(mesh_limit(limit, total, available));
        }
    }
    Ok(())
}

/// The value a mesh draw specializes `key` of a stage of `keys` with, for its threadgroup of `l`
/// threads: an axis of `l`, its product, and `MultiRegion` 0. `None` for any other key.
fn size_value(keys: &DispatchKeys, key: SpecKey, l: [u32; 3]) -> Option<u64> {
    if let Some(i) = keys.axis(key) {
        return Some(u64::from(l[i]));
    }
    if key == keys.flat {
        return Some(l.iter().map(|&x| u64::from(x)).product());
    }
    (key == keys.multi_region).then_some(0)
}

/// The mesh draw of `draw` under `t`, whose plain Vulkan draw is `command` (`vk_draw`'s, which a
/// pipeline without an object function draws).
///
/// Without an object function a draw launches mesh threadgroups: `drawMeshThreads` launches the
/// whole threadgroups its threads fill, whose threads are the grid Metal's builtins report
/// (measured, a probe: 10 by 3 threads in threadgroups of 4 by 2 report a grid of 8 by 2), as
/// `NumWorkgroups` gives it. With one it launches object threadgroups, each launching the mesh grid
/// it sets: `drawMeshThreads` launches the threadgroups its threads fill, and one that leaves a
/// partial object threadgroup is not implemented (Metal runs that threadgroup with the threads
/// left, probe 4, and splitting the draw by the regions would reorder its primitives in two
/// dimensions); a draw whose object threadgroup Metal draws nothing for (no thread, or past 1024)
/// launches nothing. A direct draw's object grid is Metal's ([`metal_object_grid`], measured): each
/// axis's threads per grid wrap at 2^32, the draw runs the threadgroups the wrapped threads fill,
/// and the builtins report the wrapped threads, from the object stage's region (`MultiRegion` 1)
/// where they leave a partial threadgroup.
///
/// A draw that launches threadgroups has each stage's threadgroup checked against the device's
/// limits for it ([`check_group`]: the mesh stage's `maxMeshWorkGroup*`, the task stage's
/// `maxTaskWorkGroup*`), and a direct one its grid of the first stage's threadgroups
/// ([`check_grid`]). A grid the object function sets is bounded in the shader
/// (`lower::mesh`). Each stage takes its grid from `NumWorkgroups` (`MultiRegion` 0), as does an
/// indirect draw, whose counts its arguments pass writes within the device's limits
/// (`DrawCommands::mesh_arguments`). A draw that launches nothing is made with the smallest
/// threadgroups its stages can be made with.
///
/// A folded object function (`MeshTemplate::folded`) reads its grid from its region record
/// instead, and its draws run their object grids in pieces within the task stage's limits
/// (`bind::mesh_arguments`): a direct one's fold ([`fold_object_grid`], `DrawCommands::object_fold`)
/// is `vkCmdDrawMeshTasksEXT` of its one piece, or `vkCmdDrawMeshTasksIndirectEXT` of its pieces'
/// commands in the uploaded fold, and an indirect one is `vkCmdDrawMeshTasksIndirectCountEXT` of
/// its arguments pass's pieces. Its grid is held to no device limit.
fn mesh_draw(
    t: &MeshTemplate,
    draw: &DrawCall,
    command: vk::DrawCommand,
) -> Result<MeshDraw, BindError> {
    let Some([lo, lm]) = mesh_threads(draw) else {
        return Err(internal(format!("a mesh draw of {draw:?}")));
    };
    let command = match (&t.object, *draw) {
        // A mesh grid Metal draws nothing for (`bind::mesh_arguments`).
        (None, _)
            if matches!(command, vk::DrawCommand::DrawMeshTasks {
            group_count_x: x,
            group_count_y: y,
            group_count_z: z,
        } if metal_mesh_grid_draws_nothing([x, y, z])) =>
        {
            mesh_tasks([0; 3], [0; 3])
        }
        (None, _) => command,
        (Some(_), _) if mesh_draws_nothing(lo) => mesh_tasks([0; 3], [0; 3]),
        (Some(_), DrawCall::MeshThreads { threads, .. }) => {
            if (0..3).any(|i| !threads[i].is_multiple_of(lo[i])) {
                return Err(not_implemented(Feature::MeshShading));
            }
            mesh_tasks(std::array::from_fn(|i| threads[i] / lo[i]), lm)
        }
        (Some(_), _) => command,
    };
    let folded = t.folded && t.object.is_some();
    // An unfolded object stage's direct grid as Metal runs it (`metal_object_grid`): each axis's
    // threads per grid wrap at 2^32, the draw runs the threadgroups the wrapped threads fill, and
    // the builtins report the wrapped threads; past what Metal runs, Metal's GPU discards it.
    let (command, wrapped, discards) = match command {
        vk::DrawCommand::DrawMeshTasks {
            group_count_x: x,
            group_count_y: y,
            group_count_z: z,
        } if t.object.is_some() && !folded => {
            let (runs, threads) = metal_object_grid([x, y, z], lo, true).unwrap_or_default();
            let n: u64 = runs.iter().map(|&c| u64::from(c)).product();
            let discards = n > u64::from(METAL_OBJECT_GRID_TOTAL);
            let runs = if discards || runs.contains(&0) {
                [0; 3]
            } else {
                runs
            };
            (mesh_tasks(runs, lo), Some((threads, runs)), discards)
        }
        other => (other, None, false),
    };
    let groups = match command {
        vk::DrawCommand::DrawMeshTasks {
            group_count_x,
            group_count_y,
            group_count_z,
        } => Some([group_count_x, group_count_y, group_count_z]),
        vk::DrawCommand::DrawMeshTasksIndirect { .. } => None,
        other => return Err(internal(format!("a mesh draw of {other:?}"))),
    };
    let launches = groups.is_none_or(|g| !g.contains(&0));
    // Where the wrapped threads leave a partial threadgroup, `NumWorkgroups · L` passes them: the
    // object stage reads its grid from its region (`MultiRegion` 1).
    let partial = launches
        && wrapped.is_some_and(|(threads, runs)| {
            (0..3).any(|i| u64::from(runs[i]) * u64::from(lo[i]) != u64::from(threads[i]))
        });
    // The multiples are subgroup sizes, powers of two, so their largest is a multiple of each.
    let smallest = t.multiples.iter().copied().max().unwrap_or(1).max(1);
    let size = |l: [u32; 3]| if launches { l } else { [smallest, 1, 1] };
    let (lo, lm) = (size(lo), size(lm));
    let mesh_keys = SpecKey::dispatch_keys(Stage::Mesh);
    let object_keys = SpecKey::dispatch_keys(Stage::Object);
    let flat = |keys: &DispatchKeys| t.spec.iter().any(|c| c.key == keys.flat);
    if launches {
        check_group(&t.mesh, lm, flat(&mesh_keys), &t.multiples)?;
        if let Some(o) = &t.object {
            check_group(o, lo, flat(&object_keys), &t.multiples)?;
        }
        // An emulated mesh function's threadgroups are folded into rows, so its direct grid is
        // held to no axis limit.
        match (groups, t.object.as_ref(), &t.records) {
            (Some(_), Some(_), _) if folded => {}
            (Some(g), Some(o), _) => check_grid(o, g)?,
            (Some(g), None, None) => check_grid(&t.mesh, g)?,
            (Some(_), None, Some(_)) | (None, _, _) => {}
        }
    }
    // An emulated mesh function's threadgroups are a compute dispatch, and their primitives a
    // draw of `NP · k` vertices each.
    let (command, dispatch) = match &t.records {
        None => (command, None),
        Some(r) => {
            let (command, dispatch) = emulated_draw(r, t.object.is_some(), draw, groups, launches)?;
            (command, Some(dispatch))
        }
    };
    let object = t.object.is_some();
    // A folded object function reads every grid builtin from its region record, and an unfolded
    // one whose wrapped grid leaves a partial threadgroup from its region (`MultiRegion` 1).
    let specialization = spec_key(&t.spec, |k| {
        size_value(&mesh_keys, k, lm)
            .or_else(|| ((folded || partial) && k == object_keys.multi_region).then_some(1))
            .or_else(|| object.then(|| size_value(&object_keys, k, lo)).flatten())
            .or_else(|| t.fixed.iter().find(|(key, _)| *key == k).map(|f| f.1))
    });
    // A folded object function's draw runs its pieces (`bind::mesh_arguments`).
    let (command, fold) = match (folded, groups, &t.object) {
        (true, Some(g), Some(o)) => {
            let task = MeshGridLimits {
                axes: o.count,
                total: o.total.unwrap_or(u32::MAX),
            };
            let g = if launches { g } else { [0; 3] };
            let fold = fold_object_grid(g, lo, task, true)
                .ok_or_else(|| internal(format!("a fold of threadgroups of {lo:?} threads")))?;
            let command = match fold.pieces {
                0 | 1 => vk::DrawCommand::DrawMeshTasks {
                    group_count_x: fold.dispatch[0],
                    group_count_y: fold.dispatch[1],
                    group_count_z: fold.dispatch[2],
                },
                pieces => vk::DrawCommand::DrawMeshTasksIndirect {
                    offset: u64::from(MeshArguments::OBJECT_COMMANDS),
                    draw_count: pieces,
                    stride: 12,
                },
            };
            (command, Some(fold))
        }
        (true, None, _) => (
            vk::DrawCommand::DrawMeshTasksIndirectCount {
                offset: u64::from(MeshArguments::OBJECT_COMMANDS),
                count_offset: u64::from(MeshArguments::OBJECT_COUNT),
                max_draw_count: MeshArguments::OBJECT_PIECES,
                stride: 12,
            },
            None,
        ),
        _ => (command, None),
    };
    // The grid Metal reports of the first stage's threadgroups (`MultiRegion` 0 reads
    // `NumWorkgroups` instead): 0 for an indirect draw, whose counts only the GPU knows, and for
    // the mesh stage under an object function, whose grid the object function sets; an unfolded
    // object function's wrapped threads; a folded one reads its record instead.
    let first = if object { lo } else { lm };
    let mut grid = [0u32; 3];
    let counted = groups.filter(|_| !folded).unwrap_or_default();
    for (i, g) in counted.into_iter().enumerate() {
        grid[i] = match wrapped {
            Some((threads, _)) => threads[i],
            None => g.checked_mul(first[i]).ok_or_else(|| {
                BindError::Unsupported(Unsupported::not_representable(Feature::GroupCount {
                    axis: Some([Axis::X, Axis::Y, Axis::Z][i]),
                }))
            })?,
        };
    }
    let region = |grid, l: [u32; 3]| {
        region_bytes([0; 3], grid, l)
            .ok_or_else(|| internal(format!("a mesh draw's threadgroup of {l:?} threads")))
    };
    let (object_region, mesh_region) = if object {
        (Some(region(grid, lo)?), region([0; 3], lm)?)
    } else {
        (None, region(grid, lm)?)
    };
    let discards = discards || fold.is_some_and(|f| f.discarded);
    Ok(MeshDraw {
        command,
        specialization,
        object_region,
        mesh_region,
        dispatch,
        fold,
        discards,
    })
}

/// The draw and the compute work of an emulated mesh draw `draw` under `r` ([`MeshDispatch`]),
/// of a pipeline with an object function where `object`: `groups` are the first stage's
/// threadgroups of a direct draw (`None` for an indirect one), `lo` the threads of an object
/// threadgroup, and a draw that `launches` nothing runs nothing and draws nothing.
///
/// - A direct draw without an object function is the host's: `G` mesh threadgroups (refused as
///   a group count where they pass a `u32`), their records exactly (`maxStorageBufferRange`,
///   else `Unsupported{MeshShading}` naming it with the records' bytes), folded into rows of
///   `maxComputeWorkGroupCount[0]` (the rows within `maxComputeWorkGroupCount[1]`, else
///   refused naming it), and a draw of `G · NP · k` vertices (refused as a group count past a
///   `u32`).
/// - A direct draw with an object function dispatches its object threadgroups (`O`, whose
///   work buffer must be bindable whole, else refused naming `maxStorageBufferRange` with its
///   bytes), and an indirect draw without one reads Metal's arguments in the scan: both are
///   scanned, their records in the command buffer's pool.
/// - An indirect draw with an object function, of object threadgroups of `lo` threads, runs the
///   object arguments pass first (`MeshDispatch::object_arguments`), which reads Metal's
///   arguments, reserves the slots and offsets of its object threadgroups in the pool, and writes
///   the object dispatch, which reads its counts from the work buffer; then the scan, as above.
fn emulated_draw(
    r: &RecordsTemplate,
    object: bool,
    draw: &DrawCall,
    groups: Option<[u32; 3]>,
    launches: bool,
) -> Result<(vk::DrawCommand, MeshDispatch), BindError> {
    use crate::lower::mesh::work;
    let group_vertices = r.group_vertices;
    let over_range = |bytes: u64| mesh_limit(VkLimit::MaxStorageBufferRange, bytes, r.max_range);
    let count_past = || {
        BindError::Unsupported(Unsupported::not_representable(Feature::GroupCount {
            axis: None,
        }))
    };
    let header = |objects: u32| {
        let mut w = vec![0u32; work::HEADER as usize];
        w[work::OBJECT_GROUPS as usize] = objects;
        w[work::SLOT_WORDS as usize] = r.slot_words;
        // Within a `u32`: link sizes a threadgroup's records so (`lower::mesh::layout`).
        w[work::GROUP_WORDS as usize] = r.group_words as u32;
        w[work::GROUP_VERTICES as usize] = group_vertices;
        w[work::ROW as usize] = r.row.max(1);
        w
    };
    let work_bytes = |objects: u64| -> Result<u64, BindError> {
        let bytes = work::words(objects, r.slot_words).saturating_mul(4);
        if bytes > u64::from(r.max_range) {
            return Err(over_range(bytes));
        }
        Ok(bytes)
    };
    // The range of Metal's arguments at `offset` a pass binds, from the offset aligned down to
    // `minStorageBufferOffsetAlignment`, and the arguments' byte in it.
    let arguments_range = |offset: u64| {
        let a = r.alignment.max(4);
        let start = offset - offset % a;
        let residual = offset - start;
        let range = BufferRange {
            offset: start,
            range: residual + 12,
        };
        (range, residual)
    };
    let scanned = |objects: u32, object_groups, arguments| -> Result<_, BindError> {
        let command = vk::DrawCommand::DrawIndirect {
            offset: MeshDispatch::DRAW_OFFSET,
            draw_count: 1,
            stride: 16,
        };
        let dispatch = MeshDispatch {
            work: header(objects),
            work_bytes: work_bytes(u64::from(objects))?,
            object_groups,
            object_arguments: None,
            mesh: MeshGroups::Scanned { arguments },
            records: MeshRecordsRegion::Pool,
        };
        Ok((command, dispatch))
    };
    match (object, groups.filter(|_| launches)) {
        // The host's draw of `G` mesh threadgroups.
        (false, Some(g)) => {
            let total: u64 = g.iter().map(|&x| u64::from(x)).product();
            let t = u32::try_from(total).map_err(|_| count_past())?;
            let bytes = total
                .saturating_mul(r.group_words)
                .saturating_mul(4)
                .max(MIN_RECORDS_BYTES);
            if bytes > u64::from(r.max_range) {
                return Err(over_range(bytes));
            }
            let vertices = t.checked_mul(group_vertices).ok_or_else(count_past)?;
            let row = r.row.max(1);
            let rows = t.div_ceil(row);
            if rows > r.rows {
                return Err(mesh_limit(
                    VkLimit::MaxComputeWorkGroupCount(1),
                    u64::from(rows),
                    r.rows,
                ));
            }
            let folded = [t.min(row), rows, 1];
            let mut w = header(1);
            w[work::MESH_GROUPS as usize] = t;
            w[work::DISPATCH as usize..][..3].copy_from_slice(&folded);
            w[work::DRAW as usize..][..4].copy_from_slice(&[vertices, 1, 0, 0]);
            // Slot 0: the grid, then a zero payload; then the offsets 0 and `T`.
            w.extend(g);
            w.extend(std::iter::repeat_n(
                0,
                r.slot_words.saturating_sub(3) as usize,
            ));
            w.extend([0, t]);
            let command = vk::DrawCommand::Draw {
                vertex_count: vertices,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            };
            let dispatch = MeshDispatch {
                work: w,
                work_bytes: work_bytes(1)?,
                object_groups: None,
                object_arguments: None,
                mesh: MeshGroups::Direct(folded),
                records: MeshRecordsRegion::Exact(bytes),
            };
            Ok((command, dispatch))
        }
        // The object threadgroups set the mesh grids: scanned.
        (true, Some(g)) => {
            let objects: u64 = g.iter().map(|&x| u64::from(x)).product();
            let objects = u32::try_from(objects).map_err(|_| count_past())?;
            scanned(objects, Some(g), None)
        }
        // Metal's arguments are the mesh grid: scanned from them.
        (false, None) if launches => {
            let DrawCall::MeshThreadgroupsIndirect { indirect, .. } = *draw else {
                return Err(internal(format!("an indirect mesh draw of {draw:?}")));
            };
            let (range, residual) = arguments_range(indirect.offset);
            let mut out = scanned(1, None, Some(range))?;
            let word = u32::try_from(residual / 4 + 1).map_err(|_| count_past())?;
            out.1.work[work::ARGUMENTS as usize] = word;
            Ok(out)
        }
        // Metal's arguments are the object grid: the object arguments pass reads them, reserves
        // the slots in the pool and writes the object dispatch; then the scan.
        (true, None) if launches => {
            let DrawCall::MeshThreadgroupsIndirect { indirect, .. } = *draw else {
                return Err(internal(format!("an indirect mesh draw of {draw:?}")));
            };
            let range = arguments_range(indirect.offset);
            let mut out = scanned(0, None, None)?;
            let word = u32::try_from(range.1 / 4 + 1).map_err(|_| count_past())?;
            out.1.work[work::OBJECT_ARGUMENTS as usize] = word;
            out.1.object_arguments = Some(range.0);
            Ok(out)
        }
        // A draw that launches nothing: no step runs, and nothing is drawn.
        (_, _) => {
            let command = vk::DrawCommand::Draw {
                vertex_count: 0,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            };
            let mut w = header(1);
            w.extend(std::iter::repeat_n(0, r.slot_words as usize + 2));
            let dispatch = MeshDispatch {
                work: w,
                work_bytes: work_bytes(1)?,
                object_groups: None,
                object_arguments: None,
                mesh: MeshGroups::Direct([0; 3]),
                records: MeshRecordsRegion::Exact(MIN_RECORDS_BYTES),
            };
            Ok((command, dispatch))
        }
    }
}

/// The fewest bytes of an emulated mesh function's records: a draw of no threadgroup still binds
/// them (the smallest binding).
const MIN_RECORDS_BYTES: u64 = 16;

/// What a patch draw of `p` instantiates ([`FactorTemplate`]), from the layout's factor, patch
/// index and control-point index bindings, the device's `(minStorageBufferOffsetAlignment,
/// maxStorageBufferRange)`, and whether the shape sets a factor buffer: `None` for a pipeline
/// that draws no patches. A native pipeline's control stage binds the patch index buffer, and the
/// control-point index buffer exactly when the pipeline has a control-point index type; an
/// emulated pipeline's vertex stage binds each only where it reads it.
fn factor_template(
    p: &RenderPipeline,
    (factors, patch_indices, control_point_indices): (
        Option<(u32, u32)>,
        Option<(u32, u32)>,
        Option<(u32, u32)>,
    ),
    (alignment, max_range): (u64, u64),
    factor_buffer: bool,
) -> Result<Option<FactorTemplate>, BindError> {
    use metal::TessellationControlPointIndexType as T;
    let (domain, step, control_points, ty, site) =
        match (p.graphics.tessellation, p.graphics.emulated_tessellation) {
            (Some(t), None) => {
                if patch_indices.is_none() {
                    return Err(internal(
                        "a tessellation pipeline's layout without its patch index buffer",
                    ));
                }
                let site = FactorSite::Control {
                    push_offset: t.push_offset,
                    binding: factors,
                };
                let ty = t.control_point_index_type;
                (t.domain, t.factor_step_function, t.control_points, ty, site)
            }
            (None, Some(e)) => {
                let site = FactorSite::Scan {
                    kernel: e.scan_kernel(),
                    factors: factor_buffer,
                };
                let d = e.descriptor;
                let ty = d.control_point_index_type;
                (e.domain, d.factor_step_function, e.control_points, ty, site)
            }
            (None, None) => return Ok(None),
            (Some(_), Some(_)) => {
                return Err(internal("a tessellation pipeline both native and emulated"))
            }
        };
    let read = matches!(site, FactorSite::Scan { .. }) || control_point_indices.is_some();
    let control_point_indices = match (ty, read) {
        (T::None, _) if control_point_indices.is_none() => None,
        (T::UInt16, true) => Some((control_point_indices, 2)),
        (T::UInt32, true) => Some((control_point_indices, 4)),
        _ => {
            return Err(internal(format!(
                "a tessellation pipeline of control-point index type {ty:?} with binding \
                 {control_point_indices:?}"
            )))
        }
    };
    Ok(Some(FactorTemplate {
        step,
        patch_bytes: crate::lower::tessellation::factors::factor_bytes(domain),
        alignment,
        max_range,
        control_points,
        patch_indices,
        control_point_indices,
        site,
    }))
}

/// `+∞` as a `half`'s bits.
const HALF_INFINITY: u16 = 0x7c00;

/// The scale the control stage is pushed for a scale that is infinite as a `half`, which
/// multiplies every positive factor past the maximum level, as Metal's does (70000 and `+∞` drew
/// what the maximum factor draws). 2^32 does so finitely, as the control stage requires (its module
/// docs): its product with the least positive `half` factor, 2^-24, is 2^8, past every level (at
/// most 64), and with the largest finite one below 2^48.
const INFINITE_SCALE: f32 = 4_294_967_296.0;

/// What a patch draw binds and writes ([`patch_binding`]): the factor buffer's range, residual bind
/// offset, instance stride and scale, and the patch and control-point index buffers' ranges and
/// fields.
struct PatchBinding {
    /// The factor buffer's range, where the encoder set one.
    factors: Option<BufferRange>,
    /// The factor buffer's residual bind offset.
    residual: u32,
    /// Its instance stride, rounded down to a multiple of 4.
    stride: u32,
    /// The scale, rounded to `half` as Metal rounds it, an infinite one as `INFINITE_SCALE`.
    scale: f32,
    /// The patch index buffer's descriptor range, where a stage binds it and the draw has one.
    patch_indices: Option<DescriptorRange>,
    /// Its residual bind offset.
    index_residual: u32,
    /// Whether the draw has a patch index buffer (1 or 0).
    indexed: u32,
    /// The control-point index buffer's descriptor range, where a stage binds it.
    control_point_indices: Option<DescriptorRange>,
    /// Its residual bind offset (0 without one).
    point_residual: u32,
}

/// The control stage's writes of a patch draw of a native tessellation pipeline (`factors` is
/// the plan's, `None` for any other pipeline): [`patch_binding`]'s, as [`control_writes`] pushes
/// and binds them.
fn patch_draw(
    factors: Option<&FactorTemplate>,
    state: &BindState,
    draw: &DrawCall,
) -> Result<Option<(PushConstantWrite, Vec<DescriptorRange>)>, BindError> {
    let Some(b) = patch_binding(factors, state, draw)? else {
        return Ok(None);
    };
    match factors.map(|t| t.site) {
        Some(FactorSite::Control {
            push_offset,
            binding,
        }) => control_writes(b, push_offset, binding).map(Some),
        _ => Err(internal("a native patch draw of an emulated pipeline")),
    }
}

/// The bindings of a patch draw of a tessellation pipeline (`factors` is the plan's, `None` for
/// any other pipeline): the draw draws patches exactly when the pipeline is one, with the control
/// points the function states, indexed exactly when the pipeline has a control-point index type.
/// The factor buffer is bound over the factors the draw reads ([`factor_window`]), from any even
/// byte, at an instance stride rounded down to a multiple of 4, its scale rounded to `half` as
/// Metal rounds it (an infinite one as `INFINITE_SCALE`); the patch index buffer over the indices
/// it reads (a direct draw's from its patch start, an indirect one's from the offset to the
/// buffer's end), and the control-point index buffer over the indices it reads
/// ([`control_point_window`]), each where a stage binds it.
fn patch_binding(
    factors: Option<&FactorTemplate>,
    state: &BindState,
    draw: &DrawCall,
) -> Result<Option<PatchBinding>, BindError> {
    let (control_points, patch_index, control_point_index) = match *draw {
        DrawCall::Patches {
            control_points,
            patch_index,
            control_point_index,
            ..
        }
        | DrawCall::PatchesIndirect {
            control_points,
            patch_index,
            control_point_index,
            ..
        } => (control_points, patch_index, control_point_index),
        _ if factors.is_some() => return Err(BindError::InvalidDrawKind { patches: false }),
        _ => return Ok(None),
    };
    let Some(t) = factors else {
        return Err(BindError::InvalidDrawKind { patches: true });
    };
    if t.control_points != 0 && control_points != t.control_points {
        return Err(BindError::PatchControlPoints {
            given: control_points,
            expected: t.control_points,
        });
    }
    if control_point_index.is_some() != t.control_point_indices.is_some() {
        return Err(BindError::ControlPointIndexing {
            indexed: control_point_index.is_some(),
        });
    }
    let f = state.tessellation_factors;
    // Metal takes the scale as a `half`, rounded to nearest even, and multiplies the factors by a
    // positive one, subnormal and infinite ones included; one that is
    // 0, negative or NaN as a `half` is refused.
    let scale = f32_to_f16_bits(f.scale);
    if !(0x0001..=HALF_INFINITY).contains(&scale) {
        return Err(BindError::InvalidTessellationFactorScale {
            bits: f.scale.to_bits(),
        });
    }
    let factors_bound = match t.site {
        FactorSite::Control { binding, .. } => binding.is_some(),
        FactorSite::Scan { factors, .. } => factors,
    };
    let (factor_range, residual, stride) =
        match (factors_bound, f.buffer) {
            (false, None) => (None, 0, 0),
            (true, Some(b)) => {
                // The factors are read from any even byte: an odd offset from the byte below, and
                // each instance's at the stride rounded down to a multiple of 4, an odd one too, as
                // plain Metal reads them.
                let offset = b.offset - b.offset % 2;
                let instance_stride = b.instance_stride - b.instance_stride % 4;
                let window = factor_window(t, instance_stride, draw);
                let (range, residual) =
                    descriptor_range(offset, b.length, window, t.alignment, t.max_range).map_err(
                        |e| match e {
                            RangeError::TooLarge { needed } => BindError::TessellationFactorRange {
                                needed,
                                limit: t.max_range,
                            },
                            RangeError::Empty => BindError::Unsupported(
                                Unsupported::not_representable(Feature::BufferRange),
                            ),
                        },
                    )?;
                // The reading stage adds `instance · stride` in `u32`: a direct draw's factors lie
                // within the bound range, whose end is below 2^32; an indirect draw's beyond it are
                // past the buffer's end, a read Metal leaves undefined.
                let stride = (instance_stride & u64::from(u32::MAX)) as u32;
                (Some(range), residual, stride)
            }
            _ => return Err(internal(
                "the bind state's shape differs from the plan's at the tessellation factor buffer",
            )),
        };
    // The patch index buffer: its words from the offset (a multiple of 4, as Metal requires),
    // over the indices the draw reads.
    let (patch_indices, index_residual, indexed) = match patch_index {
        None => (None, 0, 0),
        Some((buffer, offset)) => {
            if !offset.is_multiple_of(4) {
                return Err(BindError::InvalidPatchIndexBuffer { offset });
            }
            match t.patch_indices {
                None => (None, 0, 1u32),
                Some((set, binding)) => {
                    let window = match *draw {
                        DrawCall::Patches {
                            patch_start,
                            patch_count,
                            ..
                        } => Window::Bounded {
                            start: 4 * i64::from(patch_start),
                            len: 4 * u64::from(patch_count),
                        },
                        _ => Window::Whole,
                    };
                    let (range, residual) =
                        descriptor_range(offset, buffer.length, window, t.alignment, t.max_range)
                            .map_err(|e| match e {
                            RangeError::TooLarge { needed } => BindError::PatchIndexRange {
                                needed,
                                limit: t.max_range,
                            },
                            RangeError::Empty => BindError::Unsupported(
                                Unsupported::not_representable(Feature::BufferRange),
                            ),
                        })?;
                    let range = DescriptorRange {
                        set,
                        binding,
                        range,
                    };
                    (Some(range), residual, 1u32)
                }
            }
        }
    };
    // The control-point index buffer: its indices from the offset (a multiple of the index
    // size, as Metal requires), over the indices the draw reads.
    let (control_point_indices, point_residual) =
        match (control_point_index, t.control_point_indices) {
            (Some((buffer, offset)), Some((binding, size))) => {
                if !offset.is_multiple_of(size) {
                    return Err(BindError::InvalidControlPointIndexBuffer { offset });
                }
                match binding {
                    None => (None, 0),
                    Some((set, binding)) => {
                        let window = control_point_window(size, draw);
                        let (range, residual) = descriptor_range(
                            offset,
                            buffer.length,
                            window,
                            t.alignment,
                            t.max_range,
                        )
                        .map_err(|e| match e {
                            RangeError::TooLarge { needed } => BindError::ControlPointIndexRange {
                                needed,
                                limit: t.max_range,
                            },
                            RangeError::Empty => BindError::Unsupported(
                                Unsupported::not_representable(Feature::BufferRange),
                            ),
                        })?;
                        let range = DescriptorRange {
                            set,
                            binding,
                            range,
                        };
                        (Some(range), residual)
                    }
                }
            }
            _ => (None, 0),
        };
    let scale = match scale {
        HALF_INFINITY => INFINITE_SCALE,
        finite => f16_bits_to_f32(finite),
    };
    Ok(Some(PatchBinding {
        factors: factor_range,
        residual,
        stride,
        scale,
        patch_indices,
        index_residual,
        indexed,
        control_point_indices,
        point_residual,
    }))
}

/// A native tessellation pipeline's control stage's writes for a patch draw `b`
/// ([`patch_binding`]), whose control stage's push range is at `push_offset` and whose factor
/// buffer binds at `binding`: its push range holds the factor buffer's residual bind offset,
/// instance stride and scale, then the patch index buffer's residual bind offset and whether the
/// draw has one, then the control-point index buffer's residual bind offset (0 without one); and
/// the factor, patch index and control-point index buffers' descriptor ranges.
fn control_writes(
    b: PatchBinding,
    push_offset: u32,
    binding: Option<(u32, u32)>,
) -> Result<(PushConstantWrite, Vec<DescriptorRange>), BindError> {
    let mut ranges = Vec::with_capacity(3);
    match (b.factors, binding) {
        (Some(range), Some((set, binding))) => ranges.push(DescriptorRange {
            set,
            binding,
            range,
        }),
        (None, None) => {}
        _ => {
            return Err(internal(
                "a factor range without its control stage's binding",
            ))
        }
    }
    ranges.extend(b.patch_indices);
    ranges.extend(b.control_point_indices);
    let mut bytes = Vec::with_capacity(24);
    bytes.extend_from_slice(&b.residual.to_le_bytes());
    bytes.extend_from_slice(&b.stride.to_le_bytes());
    bytes.extend_from_slice(&b.scale.to_bits().to_le_bytes());
    bytes.extend_from_slice(&b.index_residual.to_le_bytes());
    bytes.extend_from_slice(&b.indexed.to_le_bytes());
    bytes.extend_from_slice(&b.point_residual.to_le_bytes());
    Ok((
        PushConstantWrite {
            stages: vk::ShaderStageFlags::TESSELLATION_CONTROL,
            offset: push_offset,
            bytes,
        },
        ranges,
    ))
}

/// The scan of a patch draw `draw` of an emulated tessellation pipeline of `kernel`
/// ([`TessellationDispatch`]), whose patch and factor buffers `b` binds ([`patch_binding`]) and
/// whose vertex stage reads the chunk at `chunk_fields`, on a device of `max_range`
/// (`maxStorageBufferRange`). A direct draw's records are the host's to size: `patchCount ·
/// instanceCount` of them, refused past `maxStorageBufferRange` (`Unsupported{Tessellation}`
/// naming it with their bytes). An indirect draw's lie in the command buffer's records pool, and
/// the consumer copies Metal's arguments from their offset, a multiple of 4, into the header.
fn tessellation_dispatch(
    kernel: crate::bind::TessellationScanKernel,
    b: &PatchBinding,
    draw: &DrawCall,
    chunk_fields: Vec<(vk::ShaderStageFlags, u32)>,
    max_range: u64,
) -> Result<TessellationDispatch, BindError> {
    use crate::lower::tessellation::work;
    let mut header = vec![0u32; work::HEADER as usize];
    header[work::FACTOR_OFFSET as usize] = b.residual;
    header[work::FACTOR_STRIDE as usize] = b.stride;
    header[work::SCALE as usize] = b.scale.to_bits();
    let (records, arguments) = match *draw {
        DrawCall::Patches {
            patch_start,
            patch_count,
            instance_count,
            base_instance,
            ..
        } => {
            for (word, value) in [
                (work::PATCH_COUNT, patch_count),
                (work::INSTANCE_COUNT, instance_count),
                (work::PATCH_START, patch_start),
                (work::BASE_INSTANCE, base_instance),
            ] {
                header[word as usize] = value;
            }
            let total = u64::from(patch_count) * u64::from(instance_count);
            let bytes = total * 4 * u64::from(work::RECORD_WORDS);
            let records = u32::try_from(total)
                .ok()
                .filter(|_| bytes <= max_range)
                .ok_or_else(|| {
                    BindError::Unsupported(Unsupported::needs(
                        Feature::Tessellation,
                        Capability::Limit {
                            limit: VkLimit::MaxStorageBufferRange,
                            required: bytes,
                            available: max_range,
                        },
                    ))
                })?;
            (TessellationRecords::Exact { records }, None)
        }
        DrawCall::PatchesIndirect { indirect, .. } => {
            if !indirect.offset.is_multiple_of(4) {
                return Err(internal(format!(
                    "an indirect patch draw's arguments at offset {}, not a multiple of 4",
                    indirect.offset
                )));
            }
            (TessellationRecords::Pool, Some(indirect.offset))
        }
        _ => return Err(internal(format!("a tessellation scan of {draw:?}"))),
    };
    Ok(TessellationDispatch {
        kernel,
        header,
        records,
        factors: b.factors,
        arguments,
        chunk_fields,
        patch_triangles: kernel.max_patch_triangles(),
    })
}

/// The bytes of the control-point index buffer an indexed patch draw reads, relative to its
/// offset, indices `size` bytes each: the control points of the patches `[patchStart, patchStart +
/// patchCount)` of a direct draw without a patch index buffer (Metal's validation holds the buffer
/// to the first `(patchStart + patchCount) · N` indices); `Whole` for an indirect draw, a draw with
/// a patch index buffer (whose patch indices Metal does not check against the buffer), or a window
/// past `i64`.
fn control_point_window(size: u64, draw: &DrawCall) -> Window {
    let DrawCall::Patches {
        control_points,
        patch_start,
        patch_count,
        patch_index: None,
        ..
    } = *draw
    else {
        return Window::Whole;
    };
    let per_patch = u128::from(control_points) * u128::from(size);
    let start = u128::from(patch_start) * per_patch;
    let len = u128::from(patch_count) * per_patch;
    match (
        i64::try_from(start),
        u64::try_from(len),
        i64::try_from(start + len),
    ) {
        (Ok(start), Ok(len), Ok(_)) => Window::Bounded { start, len },
        _ => Window::Whole,
    }
}

/// The bytes of the factor buffer a patch draw reads, relative to its bind offset: the patches
/// `[patchStart, patchStart + patchCount)` of a per-patch step and the instances `[baseInstance,
/// baseInstance + instanceCount)` of a per-instance one (Metal's instance index includes the base
/// instance), each `patch_bytes` long; `Whole` for an indirect draw or a window past `i64`, and
/// empty for a draw of no patches or instances.
fn factor_window(t: &FactorTemplate, instance_stride: u64, draw: &DrawCall) -> Window {
    use metal::TessellationFactorStepFunction as S;
    let DrawCall::Patches {
        patch_start,
        patch_count,
        instance_count,
        base_instance,
        ..
    } = *draw
    else {
        return Window::Whole;
    };
    if patch_count == 0 || instance_count == 0 {
        return Window::Bounded { start: 0, len: 0 };
    }
    let size = u128::from(t.patch_bytes);
    let per_patch = matches!(t.step, S::PerPatch | S::PerPatchAndPerInstance);
    let per_instance = matches!(t.step, S::PerInstance | S::PerPatchAndPerInstance);
    let span = |on: bool, first: u32, count: u32, stride: u128| {
        if on {
            let first = u128::from(first);
            (first * stride, (first + u128::from(count) - 1) * stride)
        } else {
            (0, 0)
        }
    };
    let (p0, p1) = span(per_patch, patch_start, patch_count, size);
    let (i0, i1) = span(
        per_instance,
        base_instance,
        instance_count,
        u128::from(instance_stride),
    );
    let (start, end) = (p0 + i0, p1 + i1 + size);
    match (i64::try_from(start), u64::try_from(end - start)) {
        (Ok(start), Ok(len)) if i64::try_from(end).is_ok() => Window::Bounded { start, len },
        _ => Window::Whole,
    }
}

/// A direct draw of a multiview plan (`limit` is the device's `maxMultiviewInstanceIndex`) must
/// keep its last instance index within the limit
/// (`VUID-vkCmdDraw-maxMultiviewInstanceIndex-02688`); an indirect one asks for the CPU read of its
/// arguments.
fn multiview_instances(limit: Option<u32>, draw: &DrawCall) -> Result<(), BindError> {
    let Some(available) = limit else {
        return Ok(());
    };
    let (instance_count, base_instance) = match *draw {
        DrawCall::Primitives {
            instance_count,
            base_instance,
            ..
        }
        | DrawCall::Indexed {
            instance_count,
            base_instance,
            ..
        }
        | DrawCall::Patches {
            instance_count,
            base_instance,
            ..
        } => (instance_count, base_instance),
        // A mesh pipeline's draws are never multiview.
        DrawCall::Indirect { .. }
        | DrawCall::PatchesIndirect { .. }
        | DrawCall::MeshThreadgroups { .. }
        | DrawCall::MeshThreads { .. }
        | DrawCall::MeshThreadgroupsIndirect { .. } => return Ok(()),
    };
    let last = u64::from(base_instance) + u64::from(instance_count);
    if instance_count == 0 || last - 1 <= u64::from(available) {
        return Ok(());
    }
    Err(BindError::Unsupported(Unsupported::needs(
        Feature::VertexAmplification,
        Capability::Limit {
            limit: VkLimit::MaxMultiviewInstanceIndex,
            required: last - 1,
            available: u64::from(available),
        },
    )))
}

/// The index type of `draw`, a direct indexed draw of one of Metal's index types; `None` for any
/// other draw.
fn draw_index_type(draw: &DrawCall) -> Option<metal::IndexType> {
    match *draw {
        DrawCall::Indexed { index_type, .. } if index_type == vk::IndexType::UINT16 => {
            Some(metal::IndexType::UInt16)
        }
        DrawCall::Indexed { index_type, .. } if index_type == vk::IndexType::UINT32 => {
            Some(metal::IndexType::UInt32)
        }
        _ => None,
    }
}

/// How an indexed draw runs on the primitive ID's de-indexed variant
/// ([`deindexed_draw`]): `PushField::CaptureLayout`'s words, the Vulkan draw it runs as and its
/// index read.
struct DeindexedDraw {
    layout: [u32; 4],
    /// Without indices, `index count` vertices from 0, with the draw's instances.
    command: vk::DrawCommand,
    /// Its read of its indices, from the index offset, as storage in the vertex stage.
    indices: ResourceUse,
}

/// `PushField::VerticesPerPrimitive` for `draw` of primitive type `primitive`
/// (`MissingPrimitiveType` without one): 3 for a triangle list, 2 for a line list and 1 for a
/// strip or points. The vertex function reads the draw's first vertex as its base vertex, an
/// indirect draw's from its arguments as the GPU draws it. An indexed draw shares a vertex among
/// primitives, whose flat inputs read it alike: it runs `from_zero`, its indices drawn as
/// vertices from 0 (the de-indexed form, [`deindexed_draw`]), and is
/// otherwise `NotImplemented(PrimitiveId)`. A draw in the private list form (`listed`) is the
/// consumer's list, run without indices whatever its call: twice that where it lists Metal's
/// quads or quad strip as two triangles a quad (`quads`), whose primitive Metal counts once.
fn vertices_per_primitive(
    draw: &DrawCall,
    primitive: Option<metal::PrimitiveType>,
    (from_zero, listed, quads): (bool, bool, bool),
) -> Result<u32, BindError> {
    let per_primitive = match primitive.ok_or(BindError::MissingPrimitiveType)? {
        metal::PrimitiveType::Triangle => 3,
        metal::PrimitiveType::Line => 2,
        metal::PrimitiveType::Point
        | metal::PrimitiveType::LineStrip
        | metal::PrimitiveType::TriangleStrip => 1,
    };
    match draw {
        _ if listed && quads => Ok(2 * per_primitive),
        _ if listed => Ok(per_primitive),
        DrawCall::Primitives { .. } | DrawCall::Indirect { indexed: None, .. } => Ok(per_primitive),
        DrawCall::Indexed { .. } if from_zero => Ok(per_primitive),
        _ => Err(not_implemented(Feature::PrimitiveId)),
    }
}

/// How an indexed draw runs on the primitive ID's de-indexed variant of index type `ty`
/// (`VariantKey::primitive_id_indices`): `index count` vertices from 0 without indices, with the
/// draw's instances, its layout `[0, index count, base instance, base vertex]`, its indices read
/// in the vertex stage. A line or triangle strip, which Metal restarts at the all-ones index, an
/// indexed draw whose indices run past its buffer or whose index offset is not a multiple of the
/// index size, and any other draw are `NotImplemented(PrimitiveId)` (`MissingPrimitiveType`
/// without a primitive type). A draw of another index type never reaches here
/// (`instantiate_draw` asks for its variant).
fn deindexed_draw(
    ty: metal::IndexType,
    draw: &DrawCall,
    overrun: Option<IndexOverrun>,
    primitive: Option<metal::PrimitiveType>,
) -> Result<DeindexedDraw, BindError> {
    let listed = match primitive.ok_or(BindError::MissingPrimitiveType)? {
        metal::PrimitiveType::Point
        | metal::PrimitiveType::Line
        | metal::PrimitiveType::Triangle => true,
        metal::PrimitiveType::LineStrip | metal::PrimitiveType::TriangleStrip => false,
    };
    let DrawCall::Indexed {
        index_type,
        index_count,
        index_offset,
        instance_count,
        base_vertex,
        base_instance,
        ..
    } = *draw
    else {
        return Err(not_implemented(Feature::PrimitiveId));
    };
    if !listed {
        return Err(not_implemented(Feature::PrimitiveId));
    }
    if draw_index_type(draw) != Some(ty) {
        return Err(internal(
            "a de-indexed draw of another index type than its plan's",
        ));
    }
    let size = index_bytes(index_type)
        .ok_or_else(|| internal("a de-indexed draw without an index size"))?;
    if overrun.is_some() || !index_offset.is_multiple_of(size) {
        return Err(not_implemented(Feature::PrimitiveId));
    }
    Ok(DeindexedDraw {
        // The base vertex's two's-complement bits: the function adds it wrapping.
        layout: [0, index_count, base_instance, base_vertex as u32],
        command: vk::DrawCommand::Draw {
            vertex_count: index_count,
            instance_count,
            first_vertex: 0,
            first_instance: base_instance,
        },
        indices: ResourceUse {
            resource: UseTarget::IndexBuffer,
            access: AccessSet::READ,
            class: AccessClass::Storage,
            stages: vk::PipelineStageFlags2::VERTEX_SHADER,
            range: Some(ByteRange {
                offset: 0,
                len: u64::from(index_count) * size,
            }),
        },
    })
}

/// The commands of `draw` of `state`, written over `out` (`BindingPlan::instantiate_draw_into`):
/// each of its fields set, its vectors cleared or their values overwritten in place, never
/// replaced.
fn instantiate(
    t: &GraphicsTemplate,
    state: &BindState,
    draw: &DrawCall,
    out: &mut DrawCommands,
) -> Result<(), BindError> {
    let DrawCommands {
        push_constants,
        descriptor_bytes,
        buffer_offsets,
        draw: draw_command,
        indirect_needs_arguments,
        extents,
        push_holes,
        descriptor_ranges,
        root_tables,
        vertex_strides,
        uses,
        index_overrun: overrun_out,
        index_bounds_pass,
        amplifications: amplifications_out,
        specialization: specialization_out,
        mesh_dispatch: mesh_dispatch_out,
        tessellation_dispatch: tessellation_dispatch_out,
        mesh_arguments: mesh_arguments_out,
        object_fold: object_fold_out,
        discards_command_buffer: discards_out,
    } = out;
    let overrun = index_overrun(t.robust_index_fetch, draw);
    let (command, base) = vk_draw(draw, overrun);
    let deindexed = match t.deindexed {
        Some(ty) => Some(deindexed_draw(ty, draw, overrun, state.primitive_type)?),
        None => None,
    };
    let command = deindexed.as_ref().map_or(command, |d| d.command);
    // A mesh render pipeline makes mesh draws, and only it does.
    let mesh = match (&t.mesh, mesh_threads(draw)) {
        (Some(m), Some(_)) => Some(mesh_draw(m, draw, command)?),
        (None, None) => None,
        (_, given) => {
            return Err(BindError::InvalidMeshDraw {
                mesh: given.is_some(),
            })
        }
    };
    let command = mesh.as_ref().map_or(command, |m| m.command);
    descriptor_ranges.clear();
    extents.clear();
    let bound = bind_buffers(&t.buffers, state, descriptor_ranges, extents)?;
    descriptor_ranges.extend(crate::bind::instantiate::texel_ranges(&t.resources, state)?);
    let tables = crate::bind::heap::table_bytes(&t.resources, state)?;
    // A patch draw: a native pipeline's control stage's writes, or an emulated one's scan.
    let (control, scan) = match t.tessellation.map(|t| t.site) {
        Some(FactorSite::Scan { kernel, .. }) => {
            let b = patch_binding(t.tessellation.as_ref(), state, draw)?;
            (None, b.map(|b| (b, kernel)))
        }
        _ => (patch_draw(t.tessellation.as_ref(), state, draw)?, None),
    };
    // An emulated draw's instances are records, drawn as one instance.
    if scan.is_none() {
        multiview_instances(t.multiview_instance_limit, draw)?;
    }
    let resident = if t.stages.iter().any(|s| s.root_first.is_some()) {
        Some(resident_entries(state))
    } else {
        None
    };
    let vertex = state.stages.get(&Stage::Vertex);

    push_holes.clear();
    root_tables.clear();
    // The push writes and stage tables written so far, each over the one a draw before left at
    // its place where there is one.
    let (mut pushes, mut tables_written) = (0, 0);
    let mut reads_base = false;
    for st in &t.stages {
        let (push_start, push_size) = st.push_range.unwrap_or((0, 0));
        // Each stage's push bytes and table, zeroed; empty where it has none.
        let (mut no_push, mut no_table) = (Vec::new(), Vec::new());
        let push = match push_size {
            0 => &mut no_push,
            size => {
                let write = reused(push_constants, pushes, || PushConstantWrite {
                    stages: st.flags,
                    offset: push_start,
                    bytes: Vec::new(),
                });
                pushes += 1;
                write.stages = st.flags;
                write.offset = push_start;
                zeroed(&mut write.bytes, size as usize)
            }
        };
        let table = match st.table {
            Some((set, binding)) => {
                let written = reused(descriptor_bytes, tables_written, || DescriptorBytes {
                    set,
                    binding,
                    bytes: Vec::new(),
                });
                tables_written += 1;
                written.set = set;
                written.binding = binding;
                zeroed(&mut written.bytes, st.table_size as usize)
            }
            None => &mut no_table,
        };
        for f in &st.fields {
            let bytes: FieldBytes = match f.field {
                PushField::RootTableAddress
                | PushField::ExecutionStatusAddress
                | PushField::CaptureIndexAddress
                | PushField::MergedDrawAddress
                | PushField::SnapshotPhase
                | PushField::ReplayRegionAddress => {
                    let PushLoc::Push { offset } = f.loc else {
                        return Err(not_implemented(Feature::Lowering(LowerFamily::StageIo)));
                    };
                    let stages = st.flags;
                    push_holes.push(match f.field {
                        PushField::RootTableAddress => {
                            PushHole::RootTableAddress { stages, offset }
                        }
                        PushField::ExecutionStatusAddress => {
                            PushHole::ExecutionStatusAddress { stages, offset }
                        }
                        PushField::SnapshotPhase => PushHole::SnapshotPhase { stages, offset },
                        PushField::MergedDrawAddress => {
                            PushHole::MergedDrawAddress { stages, offset }
                        }
                        PushField::ReplayRegionAddress => {
                            PushHole::ReplayRegionAddress { stages, offset }
                        }
                        _ => PushHole::CaptureIndexAddress { stages, offset },
                    });
                    continue;
                }
                PushField::CaptureLayout => match &deindexed {
                    Some(d) => FieldBytes::words(d.layout.iter().copied()),
                    None => return Err(internal("a draw layout without its de-indexed form")),
                },
                PushField::ShadowStencilState => {
                    let Some(shadow) = &t.shadow_stencil else {
                        return Err(internal("a stencil state without its shader-side stencil"));
                    };
                    shadow_stencil_words(shadow, state)?
                }
                PushField::ShaderTestsState => {
                    if t.shader_tests.is_none() {
                        return Err(internal("a depth state without its in-shader depth test"));
                    }
                    shader_tests_words(state)
                }
                PushField::ShaderStencilState => {
                    if t.shader_tests.is_none() {
                        return Err(internal("a stencil state without its in-shader tests"));
                    }
                    shader_stencil_words(state)
                }
                PushField::DispatchRegion => match mesh.as_ref().and_then(|m| m.region(st.stage)) {
                    Some(region) => FieldBytes::of(&region),
                    None => return Err(internal("a dispatch region outside a mesh draw")),
                },
                PushField::ViewportCount => FieldBytes::of(&state.viewport_count.to_le_bytes()),
                PushField::RenderTargetArrayLength => {
                    FieldBytes::of(&state.render_target_array_length.to_le_bytes())
                }
                // The identity where the state holds none: a plan states its variant, so
                // instantiation under another shape asks for the other one first.
                PushField::ViewportCompensation => state
                    .viewport_compensation
                    .map_or([1.0, 0.0, 1.0, 0.0], |c| c.values)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                PushField::PointGuard => state
                    .viewport_compensation
                    .map_or([1.0, 1.0], |c| c.guard)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                // No viewport mirrored where the state holds none.
                PushField::ViewportMirrors => FieldBytes::of(
                    &state
                        .viewport_compensation
                        .map_or(0, |c| c.mirrors)
                        .to_le_bytes(),
                ),
                // No vertex's depth clamped where the state holds none.
                PushField::VertexDepthClamp => state
                    .viewport_compensation
                    .map_or([f32::NEG_INFINITY, f32::INFINITY], |c| c.vertex_clamp)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                // Metal's own `[0, 1]` where the state holds none.
                PushField::DepthClampRange => state
                    .depth_clamp_range
                    .unwrap_or([0.0, 1.0])
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                PushField::StorageKeptChannels => FieldBytes::of(&t.storage_kept.to_le_bytes()),
                // The alpha test reference's code, Metal's 1 where the encoder never set it.
                PushField::AlphaTestReference => FieldBytes::of(
                    &alpha_test_code(state.alpha_test_reference.unwrap_or(1.0)).to_le_bytes(),
                ),
                // The blend color as the encoder set it, which a blend the shader computes reads.
                PushField::HalfBlendColor => state
                    .blend_color
                    .iter()
                    .flat_map(|c| c.to_bits().to_le_bytes())
                    .collect(),
                // The first amplification's; `amplifications` writes each before its draw.
                PushField::AmplificationId
                | PushField::AmplificationLayerOffset
                | PushField::AmplificationViewportOffset => FieldBytes::of(
                    &amplification_word(f.field, 0, &t.amplifications)?.to_le_bytes(),
                ),
                PushField::BaseVertexInstance => {
                    reads_base = true;
                    FieldBytes::words(base)
                }
                PushField::VerticesPerPrimitive => {
                    let from_zero = deindexed.is_some();
                    let form = (from_zero, t.private_list, state.quad_primitive_ids);
                    FieldBytes::of(
                        &vertices_per_primitive(draw, state.primitive_type, form)?.to_le_bytes(),
                    )
                }
                // An emulated tessellation draw's first chunk (`TessellationDispatch::draw`
                // writes each), and its index buffers' fields.
                PushField::TessellationChunk
                | PushField::PatchIndexOffset
                | PushField::PatchIndexBound
                | PushField::ControlPointIndexOffset => {
                    let Some((b, _)) = &scan else {
                        return Err(internal(format!(
                            "{:?} outside an emulated patch draw",
                            f.field
                        )));
                    };
                    let word = match f.field {
                        PushField::PatchIndexOffset => b.index_residual,
                        PushField::PatchIndexBound => b.indexed,
                        PushField::ControlPointIndexOffset => b.point_residual,
                        _ => 0,
                    };
                    FieldBytes::of(&word.to_le_bytes())
                }
                PushField::VertexBufferStride(layout) => {
                    let fixed = t.pulled_strides.iter().find(|(l, _)| *l == layout);
                    let stride = match (fixed, vertex.and_then(|v| v.buffers.get(&layout))) {
                        (Some(&(_, s)), _) => s,
                        (
                            None,
                            Some(&BoundBuffer::Buffer {
                                attribute_stride, ..
                            }),
                        ) => attribute_stride.unwrap_or(0),
                        (
                            None,
                            Some(&BoundBuffer::Bytes {
                                attribute_stride, ..
                            }),
                        ) => bytes_stride(attribute_stride, vertex_buffer(layout))?,
                        _ => {
                            return Err(internal(format!(
                                "the bind state's shape differs from the plan's at vertex \
                                 buffer {layout}"
                            )))
                        }
                    };
                    FieldBytes::of(&stride.to_le_bytes())
                }
                field => slot_field(field, &bound.residuals, &t.resources, state)
                    .ok_or_else(|| internal(format!("a render push field {field:?}")))?,
            };
            put(&mut push[..], push_start, &mut table[..], f.loc, &bytes)?;
        }
        if let (Some(first), Some(resident)) = (st.root_first, resident.as_ref()) {
            let statics: Vec<_> = bound
                .statics
                .iter()
                .filter(|(slot, _)| slot.stage == st.stage)
                .map(|(slot, e)| (slot.index, *e))
                .collect();
            root_tables.push((st.flags, stage_root_table(&statics, first, resident)));
        }
    }
    push_constants.truncate(pushes);
    descriptor_bytes.truncate(tables_written);

    // Vertex buffers: offsets, per-bind strides, the address rules, and the bytes the draw's
    // vertex and instance ranges fetch.
    buffer_offsets.clear();
    vertex_strides.clear();
    for vb in &t.vertex_buffers {
        match vertex.and_then(|v| v.buffers.get(&vb.slot.index)) {
            Some(&BoundBuffer::Buffer {
                offset,
                attribute_stride,
                ..
            }) => {
                let (stride, per_bind) = match vb.stride {
                    VertexStride::Static(s) => (s, None),
                    VertexStride::PerBind => {
                        let s = attribute_stride.unwrap_or(0);
                        (s, Some(s))
                    }
                };
                // The plan put a layout fixed function cannot fetch as bound into its variant,
                // whose pipeline pulls it; this plan's pipeline is not that one.
                if !fetchable(vb, &t.vertex_strides, offset_alignment(offset, 0), per_bind) {
                    return Err(internal(format!(
                        "a draw of {:?} as the plan's pulled-vertex-buffer variant binds it; \
                         compile and plan the variant",
                        vb.slot
                    )));
                }
                // Dynamic strides take every binding's (`pStrides`).
                if per_bind.is_some() || t.vertex_strides.dynamic {
                    vertex_strides.push((vb.binding, stride));
                }
                buffer_offsets.push((vb.binding, offset));
                let window = fetch_window(vb, stride, draw);
                match extents.iter_mut().find(|e| e.slot == vb.slot) {
                    // A buffer argument at the same index: both uses' bytes.
                    Some(e) => e.window = hull(e.window, window),
                    None => extents.push(BufferExtent {
                        slot: vb.slot,
                        window,
                    }),
                }
            }
            Some(&BoundBuffer::Bytes {
                attribute_stride, ..
            }) => {
                // Inline bytes, at their bind's stride where the layout's is per bind (0 without
                // one, as a buffer's), at an aligned offset of the consumer's.
                let (stride, per_bind) = match vb.stride {
                    VertexStride::Static(s) => (s, None),
                    VertexStride::PerBind => {
                        let s = bytes_stride(attribute_stride, vb.slot)?;
                        (s, Some(s))
                    }
                };
                let aligned = metal::vertex::MAX_VERTEX_ALIGNMENT;
                if !fetchable(vb, &t.vertex_strides, aligned, per_bind) {
                    return Err(internal(format!(
                        "a draw of {:?} as the plan's pulled-vertex-buffer variant binds it; \
                         compile and plan the variant",
                        vb.slot
                    )));
                }
                if per_bind.is_some() || t.vertex_strides.dynamic {
                    vertex_strides.push((vb.binding, stride));
                }
                buffer_offsets.push((vb.binding, 0));
            }
            None => return Err(BindError::UnboundVertexBuffer { slot: vb.slot }),
        }
    }
    descriptor_bytes.extend(tables);
    if let Some((push, ranges)) = control {
        push_constants.push(push);
        descriptor_ranges.extend(ranges);
    }
    let tessellation_dispatch = match scan {
        Some((b, kernel)) => {
            let chunk_fields = t
                .stages
                .iter()
                .flat_map(|st| {
                    st.fields.iter().filter_map(|f| match (f.field, f.loc) {
                        (PushField::TessellationChunk, PushLoc::Push { offset }) => {
                            Some((st.flags, offset))
                        }
                        _ => None,
                    })
                })
                .collect();
            let max_range = t.tessellation.map(|f| f.max_range).unwrap_or_default();
            let dispatch = tessellation_dispatch(kernel, &b, draw, chunk_fields, max_range)?;
            descriptor_ranges.extend(b.patch_indices);
            descriptor_ranges.extend(b.control_point_indices);
            Some(dispatch)
        }
        None => None,
    };
    // An emulated tessellation draw draws its first chunk in place of the Metal draw's command.
    let command = tessellation_dispatch
        .as_ref()
        .map_or(command, |d| d.draw(0).1);

    // The CPU reads the arguments of an indirect draw whose base vertex and instance the profile
    // emulates, of a multiview plan, or without `drawIndirectFirstInstance`, and of an indexed one
    // whose index fetch nothing bounds, which the GPU pass may bound instead.
    // An emulated tessellation draw's scan reads its arguments (through their copy), and its
    // chunk draws are one instance each, from 0.
    let arguments_elsewhere =
        (reads_base || t.multiview_instance_limit.is_some() || t.indirect_first_instance_missing)
            && matches!(
                draw,
                DrawCall::Indirect { .. } | DrawCall::PatchesIndirect { .. }
            )
            && tessellation_dispatch.is_none();
    // An indirect mesh draw's GPU-written group counts must lie within the device's limits
    // (`VUID-VkDrawMeshTasksIndirectCommandEXT-TaskEXT-07326` and its siblings): the arguments
    // pass writes the counts the draw reads.
    // A folded object function's direct draw reads its uploaded fold instead.
    let object_fold = mesh.as_ref().and_then(|m| m.fold);
    let discards = mesh.as_ref().is_some_and(|m| m.discards);
    let mesh_arguments = match (command, &t.mesh) {
        (vk::DrawCommand::DrawMeshTasksIndirectCount { .. }, Some(_)) => {
            Some(MeshArgumentsForm::FoldedObject)
        }
        (vk::DrawCommand::DrawMeshTasksIndirect { .. }, Some(m)) if object_fold.is_none() => {
            Some(match m.object {
                Some(_) => MeshArgumentsForm::Object,
                None => MeshArgumentsForm::Mesh,
            })
        }
        _ => None,
    };
    let index_unbounded = indirect_indices_unbounded(t.robust_index_fetch, draw);
    let amplifications = amplification_writes(t)?;
    let (mesh_dispatch, specialization) = match mesh {
        Some(m) => (m.dispatch, Some(m.specialization)),
        None => (None, None),
    };
    // The de-indexed form reads its indices in the vertex stage, by address, and binds no index
    // buffer.
    uses.clear();
    match deindexed.as_ref().map(|d| d.indices) {
        Some(indices) => uses.push(indices),
        None => draw_uses(uses, draw, overrun, t.tessellation.as_ref()),
    }
    // An indirect mesh draw's arguments pass, or an emulated one's scan or object arguments pass,
    // reads Metal's 12 bytes in the compute stage.
    let scanned = mesh_dispatch.as_ref().is_some_and(|d| {
        matches!(d.mesh, MeshGroups::Scanned { arguments: Some(_) }) || d.object_arguments.is_some()
    });
    if scanned || mesh_arguments.is_some() {
        uses.push(ResourceUse {
            resource: UseTarget::Indirect,
            access: AccessSet::READ,
            class: AccessClass::Storage,
            stages: vk::PipelineStageFlags2::COMPUTE_SHADER,
            range: Some(ByteRange { offset: 0, len: 12 }),
        });
    }
    *indirect_needs_arguments = arguments_elsewhere || index_unbounded;
    *index_bounds_pass = index_unbounded && !arguments_elsewhere;
    *mesh_arguments_out = mesh_arguments;
    *object_fold_out = object_fold;
    *discards_out = discards;
    *draw_command = command;
    *overrun_out = overrun;
    *amplifications_out = amplifications;
    *mesh_dispatch_out = mesh_dispatch;
    *specialization_out = specialization;
    *tessellation_dispatch_out = tessellation_dispatch;
    Ok(())
}

/// The `at`th value of `list`, which holds at least `at`: the one a draw before left there, else
/// `made`, pushed.
fn reused<T>(list: &mut Vec<T>, at: usize, made: impl FnOnce() -> T) -> &mut T {
    if list.len() == at {
        list.push(made());
    }
    &mut list[at]
}

/// `bytes`, made `len` zero bytes in the storage it holds.
fn zeroed(bytes: &mut Vec<u8>, len: usize) -> &mut Vec<u8> {
    bytes.clear();
    bytes.resize(len, 0);
    bytes
}

/// The `u32` of amplification push field `field` for amplification `a` of a plan drawing
/// `offsets`: the amplification itself, or its render target slice or viewport offset.
fn amplification_word(field: PushField, a: usize, offsets: &[[u32; 2]]) -> Result<u32, BindError> {
    let [layer, viewport] = match offsets.get(a) {
        Some(&o) => o,
        None if a == 0 => [0, 0],
        None => return Err(internal(format!("amplification {a} of {}", offsets.len()))),
    };
    Ok(match field {
        PushField::AmplificationId => {
            u32::try_from(a).map_err(|_| internal("an amplification past 2^32"))?
        }
        PushField::AmplificationLayerOffset => layer,
        PushField::AmplificationViewportOffset => viewport,
        other => return Err(internal(format!("{other:?} is no amplification field"))),
    })
}

/// The push writes that make a draw of a plan drawing one amplification each that amplification's
/// (`DrawCommands::amplifications`): per amplification, in order, each of its fields in each stage
/// that reads one, in that stage's push window, where the placement order puts them.
fn amplification_writes(t: &GraphicsTemplate) -> Result<Vec<Vec<PushConstantWrite>>, BindError> {
    let mut out = Vec::with_capacity(t.amplifications.len());
    for a in 0..t.amplifications.len() {
        let mut writes = Vec::new();
        for st in &t.stages {
            for f in &st.fields {
                if !matches!(
                    f.field,
                    PushField::AmplificationId
                        | PushField::AmplificationLayerOffset
                        | PushField::AmplificationViewportOffset
                ) {
                    continue;
                }
                let PushLoc::Push { offset } = f.loc else {
                    return Err(internal("an amplification field outside the push window"));
                };
                let word = amplification_word(f.field, a, &t.amplifications)?;
                writes.push(PushConstantWrite {
                    stages: st.flags,
                    offset,
                    bytes: word.to_le_bytes().to_vec(),
                });
            }
        }
        out.push(writes);
    }
    Ok(out)
}

/// The stride a per-bind layout reads inline bytes bound at `attribute_stride` at, for the
/// vertex buffer `slot`: the bind's, 0 without one, as plain Metal reads them.
/// `MTLAttributeStrideStatic` (`u32::MAX`), which an M4 read as element 0 and
/// zeros past it, is not emulated: `MissingVertexStride`.
fn bytes_stride(attribute_stride: Option<u32>, slot: SlotRef) -> Result<u32, BindError> {
    match attribute_stride {
        Some(u32::MAX) => Err(BindError::MissingVertexStride { slot }),
        stride => Ok(stride.unwrap_or(0)),
    }
}

/// Whether fixed-function vertex input can fetch vertex buffer `vb` bound at an offset of
/// alignment class `offset_alignment` ([`offset_alignment`]) with the per-bind stride
/// `per_bind` (`None` for a static stride, which link already made fetchable) under `rules`:
///
/// - the bind offset is a multiple of every attribute's alignment (the attribute address is the
///   bind offset plus multiples of component-aligned offsets and strides, "Vertex Input Address
///   Calculation", `VUID-vkCmdDraw-format-10389`/`-10390`);
/// - a per-bind stride is at most `maxVertexInputBindingStride`
///   (`VUID-vkCmdBindVertexBuffers2-pStrides-03362`, `VUID-VkVertexInputBindingDescription-stride-00619`),
///   a multiple of every attribute's alignment, a multiple of and at least the portability
///   subset's `minVertexInputBindingStrideAlignment` (`VUID-VkVertexInputBindingDescription-stride-04456`
///   for the static stride a per-bind one stands for; MoltenVK passes it to Metal either way),
///   as dynamic state either 0 or covering an element's attributes
///   (`VUID-vkCmdBindVertexBuffers2-pStrides-06209`), and covering them, 0 not excepted, where
///   the portability subset withholds `vertexAttributeAccessBeyondStride`
///   (`VUID-VkVertexInputAttributeDescription-vertexAttributeAccessBeyondStride-04457`).
///
/// Otherwise the pulled fetch reads it (any offset, any `u32` stride): the plan asks for the
/// pulled-vertex-buffer variant.
fn fetchable(
    vb: &PlannedVertexBuffer,
    rules: &StrideRules,
    offset_alignment: u32,
    per_bind: Option<u32>,
) -> bool {
    let offset = offset_alignment.is_multiple_of(vb.alignment);
    let stride = per_bind.is_none_or(|s| {
        s <= rules.max
            && s.is_multiple_of(vb.alignment)
            && portable_vertex_stride(rules.portability_alignment, s)
            && (!rules.dynamic || s == 0 || s >= vb.reads.1)
            && (!rules.within_stride || s >= vb.reads.1)
    });
    offset && stride
}

/// The bytes of a fixed-function vertex buffer a draw can fetch, relative to its bind offset: the
/// elements its vertex range (a per-vertex binding) or instance range (a per-instance one, Vulkan's
/// element `firstInstance + i / divisor`) reaches, each read at `reads` of the element. `Whole`
/// when the elements are GPU-side (an indirect draw, or the vertices of an indexed draw) or the
/// window's end is past `i64`; empty for a draw of no vertices or instances.
fn fetch_window(vb: &PlannedVertexBuffer, stride: u32, draw: &DrawCall) -> Window {
    let empty = Window::Bounded { start: 0, len: 0 };
    let per_instance = vb.input_rate == vk::VertexInputRate::INSTANCE;
    let (first, count) = match *draw {
        DrawCall::Primitives {
            vertex_count: 0, ..
        }
        | DrawCall::Primitives {
            instance_count: 0, ..
        }
        | DrawCall::Indexed { index_count: 0, .. }
        | DrawCall::Indexed {
            instance_count: 0, ..
        } => return empty,
        DrawCall::Primitives {
            vertex_start,
            vertex_count,
            ..
        } if !per_instance => (vertex_start, vertex_count),
        DrawCall::Primitives {
            base_instance,
            instance_count,
            ..
        }
        | DrawCall::Indexed {
            base_instance,
            instance_count,
            ..
        } if per_instance => {
            let elements = match vb.divisor {
                0 => 1,
                d => instance_count.div_ceil(d),
            };
            (base_instance, elements)
        }
        // A tessellation pipeline fetches no fixed-function vertex buffer, nor does a mesh
        // pipeline.
        DrawCall::Primitives { .. }
        | DrawCall::Indexed { .. }
        | DrawCall::Indirect { .. }
        | DrawCall::Patches { .. }
        | DrawCall::PatchesIndirect { .. }
        | DrawCall::MeshThreadgroups { .. }
        | DrawCall::MeshThreads { .. }
        | DrawCall::MeshThreadgroupsIndirect { .. } => return Window::Whole,
    };
    let (lo, hi) = vb.reads;
    let stride = i128::from(stride);
    let start = i128::from(first) * stride + i128::from(lo);
    let end = (i128::from(first) + i128::from(count) - 1) * stride + i128::from(hi);
    // Both ends in `i64` (`0 <= start < end`), so `start + len` is representable too.
    match (i64::try_from(start), i64::try_from(end)) {
        (Ok(s), Ok(e)) => Window::Bounded {
            start: s,
            len: e.abs_diff(s),
        },
        _ => Window::Whole,
    }
}

/// The smallest window holding both `a` and `b` (an empty bounded window holds nothing).
fn hull(a: Window, b: Window) -> Window {
    match (a, b) {
        (Window::Bounded { len: 0, .. }, w) | (w, Window::Bounded { len: 0, .. }) => w,
        (Window::Bounded { start: s, len: l }, Window::Bounded { start: t, len: m }) => {
            let lo = i128::from(s.min(t));
            let hi = (i128::from(s) + i128::from(l)).max(i128::from(t) + i128::from(m));
            match u64::try_from(hi - lo) {
                Ok(len) => Window::Bounded {
                    start: lo as i64,
                    len,
                },
                Err(_) => Window::Whole,
            }
        }
        _ => Window::Whole,
    }
}

/// The index positions of a direct indexed draw that lie past its index buffer: those whose bytes
/// do not lie wholly before [`IndexBufferRef::fetch_end`]. `None` when there are none, for any
/// other draw, and for an index type without a fixed size. They are drawn (`Robust`) where the
/// device enables `robustBufferAccess2` (`robust`) and the first index lies before there, so the
/// draw's index offset can be bound; otherwise the draw is `Clamped` to the indices before there.
fn index_overrun(robust: bool, draw: &DrawCall) -> Option<IndexOverrun> {
    let DrawCall::Indexed {
        index_type,
        index_buffer,
        index_count,
        index_offset,
        ..
    } = *draw
    else {
        return None;
    };
    let size = index_bytes(index_type)?;
    let inside = index_buffer
        .fetch_end(index_offset)
        .checked_sub(index_offset)
        .map_or(0, |bytes| bytes / size);
    let inside = u32::try_from(inside).unwrap_or(u32::MAX).min(index_count);
    let past_end = index_count - inside;
    match (past_end, robust && inside > 0) {
        (0, _) => None,
        (past_end, true) => Some(IndexOverrun::Robust { past_end }),
        (past_end, false) => Some(IndexOverrun::Clamped { past_end }),
    }
}

/// Whether nothing bounds the index fetch of an indexed indirect draw, whose counts are in GPU
/// memory: without `robustBufferAccess2` (`robust`) Vulkan requires every index inside the bound
/// range (`VUID-VkDrawIndexedIndirectCommand-robustBufferAccess2-08798`), which the binder cannot
/// check, and an index offset at or past the buffer's length may lie past the `VkBuffer`, where it
/// cannot be bound (`VUID-vkCmdBindIndexBuffer-offset-08782`). Its arguments are then read on the
/// CPU, and the direct draw is bounded by [`index_overrun`].
fn indirect_indices_unbounded(robust: bool, draw: &DrawCall) -> bool {
    match *draw {
        DrawCall::Indirect {
            indexed: Some((_, index_buffer, index_offset)),
            ..
        } => !robust || index_offset >= index_buffer.length,
        _ => false,
    }
}

/// The read of a draw's `len` bytes of indirect arguments (`DRAW_INDIRECT`), from its argument
/// offset.
fn indirect_use(len: u64) -> ResourceUse {
    ResourceUse {
        resource: UseTarget::Indirect,
        access: AccessSet::READ,
        class: AccessClass::Indirect,
        stages: vk::PipelineStageFlags2::DRAW_INDIRECT,
        range: Some(ByteRange { offset: 0, len }),
    }
}

/// The draw's own resources, pushed onto `uses`: its index buffer (`INDEX_INPUT`; the indices of a
/// direct draw from the
/// index offset, the rest of the buffer for an indirect one), a patch draw's patch index buffer
/// (`TESSELLATION_CONTROL_SHADER`, an emulated tessellation pipeline's `VERTEX_SHADER`, as
/// storage; the indices of a direct draw's patches, the rest of the buffer for an indirect one)
/// and control-point index buffer (likewise; the indices [`control_point_window`] bounds, of the
/// size `factors` states), and its indirect arguments (`DRAW_INDIRECT`; 16 bytes, 20 indexed; an
/// emulated tessellation pipeline's scan reads a patch draw's 16 in `COMPUTE_SHADER`, through their
/// copy into its work buffer; a mesh draw's are its pipeline's to decide, `instantiate`), all read.
/// Ranges are relative to the draw's index offset, patch index offset, control-point index offset,
/// and argument offset. A direct draw whose indices run past the buffer (`overrun`) reads the bytes
/// from the index offset to [`IndexBufferRef::fetch_end`] when it is `Robust`, and only the indices
/// it draws when it is `Clamped` (none, from offset 0, when it draws none).
fn draw_uses(
    uses: &mut Vec<ResourceUse>,
    draw: &DrawCall,
    overrun: Option<IndexOverrun>,
    factors: Option<&FactorTemplate>,
) {
    let index = |range| ResourceUse {
        resource: UseTarget::IndexBuffer,
        access: AccessSet::READ,
        class: AccessClass::Index,
        stages: vk::PipelineStageFlags2::INDEX_INPUT,
        range,
    };
    let scanned = factors.is_some_and(|t| matches!(t.site, FactorSite::Scan { .. }));
    // A patch draw's index buffers, read by the control stage, or an emulated pipeline's vertex
    // stage.
    let control = |resource, range| ResourceUse {
        resource,
        access: AccessSet::READ,
        class: AccessClass::Storage,
        stages: match scanned {
            true => vk::PipelineStageFlags2::VERTEX_SHADER,
            false => vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
        },
        range,
    };
    // Where a stage reads it: an emulated function may not.
    let reads_patch_indices = factors.is_some_and(|t| t.patch_indices.is_some());
    if let (
        DrawCall::Patches {
            patch_index: Some(_),
            ..
        }
        | DrawCall::PatchesIndirect {
            patch_index: Some(_),
            ..
        },
        true,
    ) = (*draw, reads_patch_indices)
    {
        // A direct draw's indices from its patch start, an indirect one's from the offset on.
        let range = match *draw {
            DrawCall::Patches {
                patch_start,
                patch_count,
                ..
            } => Some(ByteRange {
                offset: 4 * u64::from(patch_start),
                len: 4 * u64::from(patch_count),
            }),
            _ => None,
        };
        uses.push(control(UseTarget::PatchIndices, range));
    }
    if let (
        DrawCall::Patches {
            control_point_index: Some(_),
            ..
        }
        | DrawCall::PatchesIndirect {
            control_point_index: Some(_),
            ..
        },
        Some(&FactorTemplate {
            control_point_indices: Some((Some(_), size)),
            ..
        }),
    ) = (*draw, factors)
    {
        let range = match control_point_window(size, draw) {
            Window::Bounded { start, len } => u64::try_from(start)
                .ok()
                .map(|offset| ByteRange { offset, len }),
            _ => None,
        };
        uses.push(control(UseTarget::ControlPointIndices, range));
    }
    match *draw {
        // A mesh draw's arguments are its pipeline's to decide (`instantiate`).
        DrawCall::Primitives { .. }
        | DrawCall::Patches { .. }
        | DrawCall::MeshThreadgroups { .. }
        | DrawCall::MeshThreads { .. }
        | DrawCall::MeshThreadgroupsIndirect { .. } => {}
        DrawCall::Indexed {
            index_type,
            index_buffer,
            index_count,
            index_offset,
            ..
        } => uses.push(index(index_bytes(index_type).map(|size| {
            ByteRange {
                offset: 0,
                len: match overrun {
                    None => u64::from(index_count) * size,
                    Some(IndexOverrun::Robust { .. }) => index_buffer
                        .fetch_end(index_offset)
                        .saturating_sub(index_offset),
                    Some(IndexOverrun::Clamped { past_end }) => {
                        u64::from(index_count.saturating_sub(past_end)) * size
                    }
                },
            }
        }))),
        // An emulated tessellation pipeline's scan reads a patch draw's arguments, through the
        // copy into its work buffer the consumer records with it.
        DrawCall::PatchesIndirect { .. } if scanned => uses.push(ResourceUse {
            resource: UseTarget::Indirect,
            access: AccessSet::READ,
            class: AccessClass::Storage,
            stages: vk::PipelineStageFlags2::COMPUTE_SHADER,
            range: Some(ByteRange { offset: 0, len: 16 }),
        }),
        DrawCall::Indirect { indexed: None, .. } | DrawCall::PatchesIndirect { .. } => {
            uses.push(indirect_use(16))
        }
        DrawCall::Indirect {
            indexed: Some(_), ..
        } => uses.extend([index(None), indirect_use(20)]),
    }
}

/// Bytes of one index of `ty`, when it is a fixed-size index type.
fn index_bytes(ty: vk::IndexType) -> Option<u64> {
    if ty == vk::IndexType::UINT16 {
        Some(2)
    } else if ty == vk::IndexType::UINT32 {
        Some(4)
    } else if ty == vk::IndexType::UINT8 {
        Some(1)
    } else {
        None
    }
}

#[cfg(test)]
mod tests;
